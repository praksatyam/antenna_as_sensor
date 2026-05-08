import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.ndimage import uniform_filter1d

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import F0, SX, SY, EPSILON, Z_F0, SIGMA_N2


f0 = F0          # operating frequency [Hz] = 28 GHz
c       = 3e8           # speed of light [m/s]
lam     = c / f0        # wavelength [m] ≈ 10.7 mm

# Array configuration (from ScreenAnt paper)
Sx      = SX             # elements along x-axis
Sy      = SY             # elements along y-axis
S       = Sx * Sy       # total elements = 49
de      = 0.5 * lam     # half-wavelength element spacing

# Wavenumber
k       = 2 * np.pi / lam

# Finger / scatterer parameters
sigma_f = 1.0           # radar cross section of finger (relative units)
z_f0    = Z_F0          # default finger height above screen [m] = 10 mm

# Noise
sigma_n2 = SIGMA_N2         # noise variance (tune for SNR analysis)

# Mutual coupling
epsilon  = EPSILON          # coupling strength: 0 = ideal, 1 = full physical

# Time parameters
dt      = 5e-3          # sampling interval [s] = 5 ms
T       = 1.0           # total duration [s]
N       = int(T / dt)   # number of time steps = 200

def build_array(Sx, Sy, de):
    """
    Build the (Sx x Sy) planar array element positions in the XY plane.
    Returns positions array of shape (S, 3) with z=0 for all elements.
    From ScreenAnt paper equation (1).
    """
    positions = []
    for s in range(1, Sx * Sy + 1):
        sx_idx = ((s - 1) % Sx)          # 0-indexed column
        sy_idx = ((s - 1) // Sx)          # 0-indexed row
        x = de * (-(Sx - 1) / 2 + sx_idx)
        y = de * ( (Sy - 1) / 2 - sy_idx)
        positions.append([x, y, 0.0])
    return np.array(positions)   # shape (S, 3)


# ─────────────────────────────────────────────
# 3. MUTUAL COUPLING MATRIX
# ─────────────────────────────────────────────

def build_coupling_matrix(array_pos, k, epsilon):
    """
    Build the mutual coupling matrix C ∈ C^(S×S).

    Physical basis:
      When element s' transmits, it induces a current in element s
      proportional to the Green's function of free space between them.
      For two isotropic radiators separated by distance d_ss', the
      mutual admittance scales as exp(-j*k*d) / (k*d) — the same
      form as a spherical wave, decaying with distance and accumulating
      phase proportional to electrical length k*d_ss'.

    Matrix structure:
      C[s, s] = 1                                     (diagonal: self-coupling)
      C[s, s'] = epsilon * exp(-j*k*d_ss') / (k*d_ss')  (off-diagonal)

      epsilon ∈ [0, 1] controls overall coupling strength:
        epsilon = 0  →  identity matrix  →  ideal isolated elements
        epsilon = 1  →  full coupling    →  maximum inter-element interference

    Connection to ScreenAnt paper:
      d_ss' is the Euclidean distance from eq. (2) of the ScreenAnt paper.
      The spatial correlation matrix R_c in eq. (7) uses sinc(2*d/lambda),
      which is the real-valued far-field approximation of the same physics.
      Our coupling matrix C is the complex near-field version, capturing
      both amplitude decay AND phase shift between element pairs.

    Args:
        array_pos : (S, 3) element positions
        k         : float  wavenumber = 2*pi/lambda
        epsilon   : float  coupling strength in [0, 1]

    Returns:
        C : (S, S) complex coupling matrix
    """
    S = len(array_pos)
    C = np.eye(S, dtype=complex)    # start with identity (self-coupling = 1)

    for s in range(S):
        for sp in range(S):
            if s == sp:
                continue            # diagonal already set to 1

            # Distance between elements s and s' — ScreenAnt eq. (2)
            d = np.linalg.norm(array_pos[s] - array_pos[sp])

            # Electrical length: how many radians of phase over distance d
            kd = k * d

            # Coupling term: spherical wave Green's function
            # Phase accumulates as e^{-jkd}, amplitude decays as 1/(kd)
            C[s, sp] = epsilon * np.exp(-1j * kd) / kd

    return C   # shape (S, S)


def inspect_coupling_matrix(C, array_pos, Sx, Sy):
    """
    Print diagnostics about the coupling matrix and return a figure
    showing its magnitude and phase structure.
    """
    S = len(array_pos)
    mag   = np.abs(C)
    phase = np.angle(C)

    # Off-diagonal stats
    off_diag_mask = ~np.eye(S, dtype=bool)
    print(f"  Coupling matrix C  ({S}×{S} complex)")
    print(f"  Off-diagonal magnitude — "
          f"mean: {mag[off_diag_mask].mean():.4f}, "
          f"max: {mag[off_diag_mask].max():.4f}, "
          f"min: {mag[off_diag_mask].min():.4f}")

    # Nearest-neighbour coupling (adjacent elements, distance = de)
    # Element 0 and element 1 are horizontal neighbours
    nn_mag = mag[0, 1]
    print(f"  Nearest-neighbour coupling magnitude : {nn_mag:.4f}")
    print(f"  Nearest-neighbour coupling phase     : "
          f"{np.angle(C[0,1]):.3f} rad")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = axes[0].imshow(mag, cmap='viridis', vmin=0)
    axes[0].set_title('|C| — Coupling Magnitude')
    axes[0].set_xlabel('Element s\'')
    axes[0].set_ylabel('Element s')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(phase, cmap='twilight', vmin=-np.pi, vmax=np.pi)
    axes[1].set_title('∠C — Coupling Phase [rad]')
    axes[1].set_xlabel('Element s\'')
    axes[1].set_ylabel('Element s')
    plt.colorbar(im1, ax=axes[1])

    plt.suptitle(f'Mutual Coupling Matrix  (ε={epsilon})', fontsize=13)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 3b. DISTANCE FROM EACH ELEMENT TO FINGER
# ─────────────────────────────────────────────

def compute_distances(array_pos, finger_pos):
    """
    Compute Euclidean distance from each antenna element to the finger.

    Args:
        array_pos  : (S, 3) array of element positions
        finger_pos : (3,)   finger position [x_f, y_f, z_f]

    Returns:
        r : (S,) array of distances
    """
    diff = array_pos - finger_pos   # (S, 3)
    r    = np.sqrt(np.sum(diff**2, axis=1))  # (S,)
    return r



# ─────────────────────────────────────────────
# 4. BACKSCATTERED SIGNAL MODEL
# ─────────────────────────────────────────────

def backscattered_signal(array_pos, finger_pos, sigma_f, k, sigma_n2):
    """
    Compute the backscattered signal at each array element from a point
    scatterer (finger) at finger_pos.

    Model:
        y_s = sigma_f * exp(-j * 2 * k * r_s) / r_s^2 + n_s

    The factor of 2 in the exponent accounts for the round-trip path.
    The r_s^2 denominator is the round-trip path loss.

    Args:
        array_pos  : (S, 3) element positions
        finger_pos : (3,)   finger position
        sigma_f    : float  finger radar cross section
        k          : float  wavenumber
        sigma_n2   : float  noise variance

    Returns:
        y : (S,) complex array of received signals
    """
    r = compute_distances(array_pos, finger_pos)  # (S,)

    # Noiseless backscattered signal
    signal = sigma_f * np.exp(-1j * 2 * k * r) / (r ** 2)

    # Complex Gaussian noise: real and imag each ~ N(0, sigma_n2/2)
    noise  = (np.random.randn(len(r)) + 1j * np.random.randn(len(r))) \
             * np.sqrt(sigma_n2 / 2)

    return signal + noise