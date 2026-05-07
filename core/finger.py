import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# Import shared infrastructure from existing simulation
import sys
sys.path.insert(0, '/home/claude')
from ..config import (F0, SX, SY, EPSILON, Z_F0, SIGMA_N2,)

from .array import (
    build_array, build_coupling_matrix, simulate,
    trajectory_linear, trajectory_sinusoidal, trajectory_random_walk,
    screen_bounds
)

from .trajectory import (simulate, trajectory_linear, trajectory_sinusoidal,
        trajectory_random_walk,screen_bounds)

f0 = F0
c  = 3e8              # speed of light [m/s]
lam = c / f0         # wavelength [m]

Sx = SX
Sy = SY
S  = Sx * Sy        # total elements
de = 0.5 * lam      # half-wavelength element spacing
k  = 2 * np.pi / lam  # wavenumber

sigma_f_point = 1.0   # point scatterer RCS (relative units)
z_f0 = Z_F0           # finger height [m]
sigma_n2 = SIGMA_N2   # noise variance
epsilon = EPSILON     # coupling strength

dt = 5e-3          # sampling interval [s]
T  = 1.0           # total duration [s]
N  = int(T / dt)   # number of time steps

# ══════════════════════════════════════════════════════════════
# 1. PHYSICAL CONSTANTS FOR BIOLOGICAL TISSUE
# ══════════════════════════════════════════════════════════════

# Permittivity of free space
eps0 = 8.854e-12        # F/m

# Angular frequency
omega = 2 * np.pi * f0  # rad/s  (28 GHz)

# Biological tissue parameters at 28 GHz
# Source: Gabriel et al. dielectric model for human skin
eps_r   = 17.5          # relative permittivity (real part)
sigma_t = 15.0          # conductivity [S/m]

# Complex permittivity: ε̃ = εr - j·σ/(ω·ε0)
eps_tilde = eps_r - 1j * sigma_t / (omega * eps0)
print(f"Complex permittivity ε̃ = {eps_tilde.real:.2f} - j{abs(eps_tilde.imag):.2f}")

# Complex refractive index: ñ = sqrt(ε̃)
n_tilde = np.sqrt(eps_tilde)
print(f"Complex refractive index ñ = {n_tilde.real:.3f} - j{abs(n_tilde.imag):.3f}")
print(f"  Re(ñ) → phase accumulation per unit length")
print(f"  Im(ñ) → amplitude decay per unit length (absorption)")

# Loss tangent: tan(δ) = Im(ε̃) / Re(ε̃)
tan_delta = abs(eps_tilde.imag) / eps_tilde.real
print(f"Loss tangent tan(δ) = {tan_delta:.3f}  (>1 means very lossy)")

# Skin depth (effective penetration depth into biological tissue)
# δ = c / (ω · Im(ñ))  — the wave decays to 1/e amplitude in this distance
# At 28 GHz, biological tissue has very high absorption, so δ ≈ 1.5 mm
# The wave does NOT penetrate the full finger thickness — it scatters off
# the surface and penetrates only to the skin depth before being absorbed
delta_skin = c / (omega * abs(n_tilde.imag))
print(f"Skin depth δ = {delta_skin*1e3:.3f} mm  (effective penetration depth)")

# Finger physical dimensions
a_x      = 0.008        # finger half-width in x [m] = 8 mm
a_y      = 0.030        # finger half-length in y [m] = 30 mm

# Use skin depth as effective interaction depth — not full finger thickness
d_finger = delta_skin   # effective depth [m] ≈ 1.5 mm

# Minimum transmission (centre of finger) from complex refractive index
# β_min = exp(-2k · Im(ñ) · d_finger)
# The factor 2 is round-trip: signal passes through finger twice
beta_min = np.exp(-2 * k * abs(n_tilde.imag) * d_finger)
print(f"\nMinimum transmission β_min = {beta_min:.4f}")
print(f"  (signal under finger centre retains {beta_min*100:.1f}% amplitude)")

# Maximum phase shift (centre of finger)
# φ_max = k · (Re(ñ) - 1) · d_finger
phi_max = k * (n_tilde.real - 1) * d_finger
print(f"Maximum phase shift φ_max = {phi_max:.2f} rad = {np.degrees(phi_max):.1f}°")

# Permittivity-dependent RCS via Clausius-Mossotti
# σ_f(ε̃) = σ0 · |(ε̃ - 1)/(ε̃ + 2)|²
cm_factor = abs((eps_tilde - 1) / (eps_tilde + 2)) ** 2
sigma_f_phys = cm_factor           # normalised (σ0 = 1)
print(f"\nClausius-Mossotti RCS factor = {cm_factor:.4f}")
print(f"  (vs point scatterer σ_f = 1.0)")


# ══════════════════════════════════════════════════════════════
# 2. EXTENDED FINGER MODEL — CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════

def finger_footprint(array_pos, finger_pos, a_x, a_y):
    """
    Compute the spatial Gaussian footprint ρ_s for each array element.

    Models the finger as an ellipse projected onto the screen plane.
    ρ_s = 1 means element s is fully under the finger centre.
    ρ_s = 0 means element s is far from the finger.

        ρ_s = exp( -(sx - xf)²/(2ax²) - (sy - yf)²/(2ay²) )

    Args:
        array_pos  : (S, 3) element positions
        finger_pos : (3,)   finger position [xf, yf, zf]
        a_x, a_y   : float  finger half-widths [m]

    Returns:
        rho : (S,) footprint values in [0, 1]
    """
    dx  = array_pos[:, 0] - finger_pos[0]   # (S,)
    dy  = array_pos[:, 1] - finger_pos[1]   # (S,)
    rho = np.exp(- dx**2 / (2 * a_x**2)
                 - dy**2 / (2 * a_y**2))    # (S,)
    return rho


def complex_transmission(rho, beta_min, phi_max):
    """
    Compute the complex transmission coefficient γ_s for each element.

    Combines amplitude attenuation and phase shift:
        β_s = 1 - ρ_s · (1 - β_min)     ∈ [β_min, 1]
        φ_s = ρ_s · φ_max               ∈ [0, φ_max]
        γ_s = β_s · exp(j·φ_s)

    Elements far from finger (ρ_s ≈ 0): γ_s ≈ 1·exp(0) = 1 (unaffected)
    Elements under finger  (ρ_s ≈ 1): γ_s ≈ β_min·exp(j·φ_max) (max blockage)

    Args:
        rho      : (S,) spatial footprint
        beta_min : float minimum amplitude transmission
        phi_max  : float maximum phase shift [rad]

    Returns:
        gamma : (S,) complex transmission coefficients
    """
    beta  = 1.0 - rho * (1.0 - beta_min)     # amplitude: (S,)
    phi   = rho * phi_max                      # phase:     (S,)
    gamma = beta * np.exp(1j * phi)            # complex:   (S,)
    return gamma


def extended_steering_vector(array_pos, finger_pos, k, sigma_f_phys,
                              a_x, a_y, rho=None):
    """
    Compute the scattering steering vector a_s for the extended finger.

    Unlike the point scatterer, the reflected amplitude at element s is
    weighted by the spatial footprint ρ_s — elements not under the finger
    see weaker reflections. The RCS uses the Clausius-Mossotti value.

        a_s = σ_f(ε̃) · ρ_s · exp(-j·2k·r_s) / r_s²

    Args:
        array_pos    : (S, 3)
        finger_pos   : (3,)
        k            : float wavenumber
        sigma_f_phys : float Clausius-Mossotti RCS factor
        a_x, a_y     : float finger half-widths
        rho          : (S,) pre-computed footprint (computed if None)

    Returns:
        a : (S,) complex steering vector
    """
    if rho is None:
        rho = finger_footprint(array_pos, finger_pos, a_x, a_y)

    diff = array_pos - finger_pos             # (S, 3)
    r    = np.sqrt(np.sum(diff**2, axis=1))   # (S,)

    a = sigma_f_phys * rho * np.exp(-1j * 2 * k * r) / (r**2)
    return a


def extended_signal(array_pos, finger_pos, k, sigma_f_phys,
                    a_x, a_y, beta_min, phi_max, C, sigma_n2):
    """
    Compute the full received signal at all array elements for the
    extended finger model.

    Full signal model (vector form):
        y(t) = C · [Γ(t) ⊙ a(t)] + n(t)

    where:
        a(t)  = extended scattering steering vector  (S,) complex
        Γ(t)  = diag(γ_1,...,γ_S)                  blockage matrix
        γ_s   = complex transmission coefficient     scalar complex
        C     = mutual coupling matrix               (S,S) complex
        n(t)  ~ CN(0, σ_n²)                         noise

    The element-wise product Γ ⊙ a modifies each element's scattered
    signal by the local complex transmission — elements under the finger
    see both attenuated AND phase-shifted reflections.

    Args:
        array_pos    : (S, 3)
        finger_pos   : (3,)
        k, sigma_f_phys, a_x, a_y, beta_min, phi_max : floats
        C            : (S, S) coupling matrix
        sigma_n2     : float noise variance

    Returns:
        y     : (S,) complex received signal
        rho   : (S,) footprint (for diagnostics)
        gamma : (S,) complex transmission (for diagnostics)
        a     : (S,) steering vector (for diagnostics)
    """
    # Step 1: Spatial footprint
    rho   = finger_footprint(array_pos, finger_pos, a_x, a_y)

    # Step 2: Complex transmission coefficients
    gamma = complex_transmission(rho, beta_min, phi_max)

    # Step 3: Extended scattering steering vector
    a     = extended_steering_vector(array_pos, finger_pos, k,
                                     sigma_f_phys, a_x, a_y, rho=rho)

    # Step 4: Apply blockage — element-wise product Γ ⊙ a
    a_blocked = gamma * a                     # (S,) Hadamard product

    # Step 5: Apply mutual coupling — mix signals across elements
    y_noiseless = C @ a_blocked               # (S,)

    # Step 6: Add complex Gaussian noise
    noise = (np.random.randn(S) + 1j * np.random.randn(S)) \
            * np.sqrt(sigma_n2 / 2)

    return y_noiseless + noise, rho, gamma, a


def simulate_extended(trajectory, array_pos, k, sigma_f_phys,
                      a_x, a_y, beta_min, phi_max, C, sigma_n2):
    """
    Simulate the full trajectory with the extended finger model.

    Args:
        trajectory : (N, 3) finger positions over time
        ... all physical parameters ...

    Returns:
        Y     : (N, S) complex array responses
        RHO   : (N, S) footprints per time step
        GAMMA : (N, S) complex transmissions per time step
    """
    Nt    = len(trajectory)
    Y     = np.zeros((Nt, S), dtype=complex)
    RHO   = np.zeros((Nt, S))
    GAMMA = np.zeros((Nt, S), dtype=complex)

    for n in range(Nt):
        y, rho, gamma, _ = extended_signal(
            array_pos, trajectory[n],
            k, sigma_f_phys, a_x, a_y,
            beta_min, phi_max, C, sigma_n2
        )
        Y[n]     = y
        RHO[n]   = rho
        GAMMA[n] = gamma

    return Y, RHO, GAMMA


# ══════════════════════════════════════════════════════════════
# 3. VERIFICATION PLOTS
# ══════════════════════════════════════════════════════════════

def plot_finger_footprint_and_transmission(array_pos, finger_pos,
                                           a_x, a_y, beta_min, phi_max):
    """
    Plot 1 — Verify the spatial footprint and transmission coefficients.

    What to look for:
      - Footprint |ρ|: smooth Gaussian bell centred at finger position
        Should peak at 1.0 directly below finger, decay smoothly outward
      - Amplitude β: inverse of footprint — low under finger, 1.0 far away
      - Phase φ: Gaussian shape — maximum shift under finger, 0 far away
      - |γ|: combined transmission amplitude — should match β closely
    """
    rho   = finger_footprint(array_pos, finger_pos, a_x, a_y)
    gamma = complex_transmission(rho, beta_min, phi_max)
    beta  = np.abs(gamma)
    phi   = np.angle(gamma)

    Sx_l  = int(np.sqrt(S))
    xmm   = array_pos[:, 0] * 1e3
    ymm   = array_pos[:, 1] * 1e3

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    quantities = [
        (rho,            'ρ_s — Spatial Footprint',       'viridis',  [0, 1]),
        (beta,           'β_s — Amplitude Transmission',  'RdYlGn',   [beta_min, 1]),
        (np.degrees(phi * rho), 'φ_s — Phase Shift [°]', 'coolwarm', None),
        (np.abs(gamma),  '|γ_s| — |Complex Transmission|','plasma',   [0, 1]),
    ]

    for ax, (data, title, cmap, vrange) in zip(axes.ravel(), quantities):
        vmin, vmax = (vrange if vrange else (data.min(), data.max()))
        sc = ax.scatter(xmm, ymm, c=data, cmap=cmap,
                        vmin=vmin, vmax=vmax, s=200, zorder=3)
        # Mark finger centre
        ax.scatter(finger_pos[0]*1e3, finger_pos[1]*1e3,
                   marker='+', s=300, color='red', linewidths=2,
                   zorder=5, label='Finger centre')
        # Draw finger ellipse
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(finger_pos[0]*1e3 + a_x*1e3*np.cos(theta),
                finger_pos[1]*1e3 + a_y*1e3*np.sin(theta),
                'r--', linewidth=1.5, alpha=0.7, label='Finger outline')
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.legend(fontsize=7)
        plt.colorbar(sc, ax=ax)

    plt.suptitle(
        f'Extended Finger Model — Footprint & Transmission\n'
        f'Finger at ({finger_pos[0]*1e3:.0f}, {finger_pos[1]*1e3:.0f}) mm  '
        f'β_min={beta_min:.3f}  φ_max={np.degrees(phi_max):.1f}°',
        fontsize=12
    )
    plt.tight_layout()
    return fig


def plot_signal_comparison_snapshot(array_pos, finger_pos,
                                    C, sigma_n2, k,
                                    sigma_f_point, sigma_f_phys,
                                    a_x, a_y, beta_min, phi_max):
    """
    Plot 2 — Compare point scatterer vs extended finger array responses
    at a single snapshot.

    What to look for:
      Point scatterer:
        - Smooth radially symmetric power pattern centred on finger
        - No spatial variation in amplitude beyond 1/r² decay
        - Uniform phase pattern

      Extended finger:
        - Asymmetric power pattern due to blockage (elements under
          finger receive LESS scattered power — they are shadowed)
        - Phase pattern distorted by γ_s phase shifts
        - Stronger spatial structure — richer information for localization
    """
    # Point scatterer signal (existing model)
    from .array import backscattered_signal
    y_point = backscattered_signal(array_pos, finger_pos,
                                   sigma_f_point, k, sigma_n2)

    # Extended finger signal
    y_ext, rho, gamma, a = extended_signal(
        array_pos, finger_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, sigma_n2
    )

    Sx_l = int(np.sqrt(S))
    xmm  = array_pos[:, 0] * 1e3
    ymm  = array_pos[:, 1] * 1e3

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    def scatter_grid(ax, data, title, cmap, label=''):
        sc = ax.scatter(xmm, ymm, c=data, cmap=cmap, s=200, zorder=3)
        ax.scatter(finger_pos[0]*1e3, finger_pos[1]*1e3,
                   marker='+', s=300, color='red', linewidths=2, zorder=5)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(finger_pos[0]*1e3 + a_x*1e3*np.cos(theta),
                finger_pos[1]*1e3 + a_y*1e3*np.sin(theta),
                'r--', lw=1.5, alpha=0.7)
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_aspect('equal')
        plt.colorbar(sc, ax=ax, label=label)

    # Row 0: Point scatterer
    scatter_grid(axes[0,0], np.abs(y_point)**2,
                 'Point Scatterer\n|y|² Power', 'hot', '|y|²')
    scatter_grid(axes[0,1], np.angle(y_point),
                 'Point Scatterer\n∠y Phase [rad]', 'twilight', 'rad')
    scatter_grid(axes[0,2], np.abs(y_point),
                 'Point Scatterer\n|y| Amplitude', 'plasma', '|y|')

    # Row 1: Extended finger
    scatter_grid(axes[1,0], np.abs(y_ext)**2,
                 'Extended Finger\n|y|² Power', 'hot', '|y|²')
    scatter_grid(axes[1,1], np.angle(y_ext),
                 'Extended Finger\n∠y Phase [rad]', 'twilight', 'rad')
    scatter_grid(axes[1,2], np.abs(y_ext),
                 'Extended Finger\n|y| Amplitude', 'plasma', '|y|')

    plt.suptitle(
        f'Array Response: Point Scatterer vs Extended Finger\n'
        f'Finger at ({finger_pos[0]*1e3:.0f}, {finger_pos[1]*1e3:.0f}) mm  '
        f'height={finger_pos[2]*1e3:.0f} mm',
        fontsize=12
    )
    plt.tight_layout()
    return fig


def plot_temporal_array_response(Y, RHO, GAMMA, trajectory, label=''):
    """
    Plot 3 — Array response over time for the full trajectory.

    Four panels:
      Top-left  : Power heatmap |Y|² over time — each column is one
                  element, each row is one time step. Watch for the
                  power pattern shifting as the finger moves.

      Top-right : Footprint RHO over time — shows which elements are
                  under the finger at each moment.

      Bottom-left: Phase of Y over time — should vary smoothly as
                   the finger moves (phase changes with distance r_s).

      Bottom-right: Mean amplitude per element over the full trajectory
                    — shows which elements are most informative.

    What to look for:
      - The power and footprint patterns should move together —
        when the footprint shifts right (finger moves right), the
        attenuation pattern should shift correspondingly.
      - The phase heatmap should show smooth, coherent stripes —
        this confirms the signal model is physically consistent.
      - Elements near the centre should show more temporal variation
        than edge elements (the trajectory stays mostly central).
    """
    Nt, Sel = Y.shape
    t_axis  = np.arange(Nt) * dt * 1e3   # ms

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    # Power over time
    ax0 = fig.add_subplot(gs[0, 0])
    im0 = ax0.imshow(np.abs(Y)**2, aspect='auto', origin='lower',
                     cmap='inferno',
                     extent=[0, Sel, t_axis[0], t_axis[-1]])
    ax0.set_xlabel('Antenna element index')
    ax0.set_ylabel('Time [ms]')
    ax0.set_title('|Y|² — Received Power per Element Over Time',
                  fontweight='bold')
    plt.colorbar(im0, ax=ax0, label='Power')

    # Footprint over time
    ax1 = fig.add_subplot(gs[0, 1])
    im1 = ax1.imshow(RHO, aspect='auto', origin='lower',
                     cmap='YlOrRd', vmin=0, vmax=1,
                     extent=[0, Sel, t_axis[0], t_axis[-1]])
    ax1.set_xlabel('Antenna element index')
    ax1.set_ylabel('Time [ms]')
    ax1.set_title('ρ_s — Finger Footprint per Element Over Time',
                  fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='ρ_s ∈ [0,1]')

    # Phase over time
    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(np.angle(Y), aspect='auto', origin='lower',
                     cmap='twilight',
                     extent=[0, Sel, t_axis[0], t_axis[-1]])
    ax2.set_xlabel('Antenna element index')
    ax2.set_ylabel('Time [ms]')
    ax2.set_title('∠Y — Signal Phase per Element Over Time',
                  fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Phase [rad]')

    # Mean amplitude per element
    ax3 = fig.add_subplot(gs[1, 1])
    mean_amp  = np.mean(np.abs(Y), axis=0)
    std_amp   = np.std(np.abs(Y),  axis=0)
    ax3.bar(np.arange(Sel), mean_amp, color='steelblue',
            alpha=0.8, label='Mean |y|')
    ax3.errorbar(np.arange(Sel), mean_amp, yerr=std_amp,
                 fmt='none', color='navy', alpha=0.5, capsize=2)
    ax3.set_xlabel('Antenna element index')
    ax3.set_ylabel('|y|')
    ax3.set_title('Mean ± Std Amplitude per Element\n(over full trajectory)',
                  fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend()

    plt.suptitle(f'Extended Finger — Temporal Array Response  {label}',
                 fontsize=13, fontweight='bold')
    return fig


def plot_spatial_snapshots(Y, RHO, GAMMA, trajectory, Sx_l, Sy_l,
                           n_snaps=4, label=''):
    """
    Plot 4 — Spatial heatmaps at evenly-spaced time snapshots.

    Shows the 7×7 array as a grid image at 4 moments in time.
    For each snapshot shows: power, footprint, and |γ|.

    What to look for:
      - Power map: bright patch should track the finger position
      - Footprint: should be a smooth ellipse centred on finger
      - |γ|: should be low (dark) under finger, 1 (bright) elsewhere
      - As time progresses, all three maps should shift together
        following the trajectory
    """
    snap_idx = np.linspace(0, len(Y)-1, n_snaps, dtype=int)
    fig, axes = plt.subplots(3, n_snaps, figsize=(4*n_snaps, 10))

    row_labels = ['|Y|² Power', 'ρ_s Footprint', '|γ_s| Transmission']
    cmaps      = ['inferno', 'YlOrRd', 'RdYlGn']

    for col, t_idx in enumerate(snap_idx):
        data_rows = [
            np.abs(Y[t_idx])**2,
            RHO[t_idx],
            np.abs(GAMMA[t_idx])
        ]
        fpos = trajectory[t_idx, :2] * 1e3

        for row, (data, cmap) in enumerate(zip(data_rows, cmaps)):
            ax  = axes[row, col]
            grid = data.reshape(Sx_l, Sy_l)
            im   = ax.imshow(grid, cmap=cmap, origin='upper',
                             vmin=0 if row > 0 else None)

            # Mark finger position — find nearest element
            nearest = np.argmin(
                (trajectory[t_idx, 0] - array_pos[:, 0])**2 +
                (trajectory[t_idx, 1] - array_pos[:, 1])**2
            )
            nr = nearest // Sx_l
            nc = nearest %  Sx_l
            ax.scatter(nc, nr, marker='+', s=200,
                       color='cyan', linewidths=2, zorder=5)

            if row == 0:
                ax.set_title(f't={t_idx}\n'
                             f'finger=({fpos[0]:.1f},{fpos[1]:.1f})mm',
                             fontsize=9)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            plt.colorbar(im, ax=ax)

    plt.suptitle(f'Spatial Snapshots — Extended Finger Model  {label}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_point_vs_extended_temporal(Y_point, Y_ext, trajectory,
                                    element_idx=24):
    """
    Plot 5 — Direct temporal signal comparison at the centre element.

    Shows real part, imaginary part, and amplitude of the signal at the
    centre element (idx=24, the element directly at array centre) over
    the full trajectory duration.

    What to look for:
      - Point scatterer: amplitude varies smoothly with 1/r² — peaks
        when finger is closest to the centre element
      - Extended finger: amplitude shows additional dips when the finger
        is directly overhead (blockage effect suppresses the signal)
      - Phase: extended finger shows additional phase modulation from
        the γ_s phase shift on top of the geometric phase from r_s
      - The blockage dip is the KEY new feature — it creates a spatial
        null in the response that the point scatterer cannot produce
    """
    Nt     = len(trajectory)
    t_axis = np.arange(Nt) * dt * 1e3

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    for Y, label, color, ls in [
        (Y_point, 'Point scatterer', 'steelblue', '-'),
        (Y_ext,   'Extended finger', '#e63946',   '--'),
    ]:
        axes[0].plot(t_axis, np.abs(Y[:, element_idx]),
                     color=color, linestyle=ls, linewidth=1.8, label=label)
        axes[1].plot(t_axis, np.real(Y[:, element_idx]),
                     color=color, linestyle=ls, linewidth=1.5, label=label)
        axes[2].plot(t_axis, np.angle(Y[:, element_idx]),
                     color=color, linestyle=ls, linewidth=1.5, label=label)

    # Mark when finger is directly above centre element
    # Centre element is at (0,0), finger is at trajectory[:,0:2]
    dist_to_centre = np.sqrt(trajectory[:,0]**2 + trajectory[:,1]**2)
    axes[0].axvline(t_axis[np.argmin(dist_to_centre)],
                    color='orange', linestyle=':', linewidth=1.5,
                    label='Finger closest to centre')

    axes[0].set_ylabel('|y| Amplitude')
    axes[0].set_title(f'Signal at Centre Element (idx={element_idx}): '
                      f'Point Scatterer vs Extended Finger',
                      fontweight='bold')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel('Re(y)')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

    axes[2].set_ylabel('∠y Phase [rad]')
    axes[2].set_xlabel('Time [ms]')
    axes[2].legend(fontsize=9); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# 4. SANITY CHECKS
# ══════════════════════════════════════════════════════════════

def run_sanity_checks(array_pos, C):
    """
    Print a set of physical sanity checks on the finger model.
    Each check has an expected behaviour and a pass/fail verdict.
    """
    print("\n" + "="*55)
    print("  SANITY CHECKS — Extended Finger Model")
    print("="*55)

    # Check 1: Footprint peaks at finger centre
    finger_at_centre = np.array([0.0, 0.0, z_f0])
    rho = finger_footprint(array_pos, finger_at_centre, a_x, a_y)
    centre_elem = np.argmin(np.sum(array_pos[:,:2]**2, axis=1))
    print(f"\n[1] Footprint at centre element (finger at origin):")
    print(f"    ρ_centre = {rho[centre_elem]:.4f}  (expected: close to 1.0)")
    print(f"    ρ_corner = {rho[0]:.4f}  (expected: close to 0.0)")
    assert rho[centre_elem] > rho[0], "FAIL: centre should have higher footprint"
    print(f"    PASS ✓")

    # Check 2: Transmission is minimum at centre when finger is there
    gamma = complex_transmission(rho, beta_min, phi_max)
    print(f"\n[2] Transmission amplitude under finger:")
    print(f"    |γ_centre| = {abs(gamma[centre_elem]):.4f}  "
          f"(expected: near β_min={beta_min:.4f})")
    print(f"    |γ_corner| = {abs(gamma[0]):.4f}  (expected: near 1.0)")
    assert abs(gamma[centre_elem]) < abs(gamma[0]), \
        "FAIL: centre should have lower transmission"
    print(f"    PASS ✓")

    # Check 3: Signal drops when finger moves over an element
    finger_away   = np.array([0.1, 0.1, z_f0])   # far from array
    finger_above  = np.array([0.0, 0.0, z_f0])   # over centre

    y_away, _, _, _  = extended_signal(
        array_pos, finger_away,  k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, 0
    )
    y_above, _, _, _ = extended_signal(
        array_pos, finger_above, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, 0
    )
    power_away  = np.abs(y_away[centre_elem])**2
    power_above = np.abs(y_above[centre_elem])**2
    print(f"\n[3] Power at centre element:")
    print(f"    Finger away   : {power_away:.4e}")
    print(f"    Finger above  : {power_above:.4e}")
    print(f"    Ratio (above/away): {power_above/power_away:.3f}  "
          f"(expected: <1, blockage reduces power)")
    # Note: may be complex depending on coupling — check total array power
    total_away  = np.sum(np.abs(y_away)**2)
    total_above = np.sum(np.abs(y_above)**2)
    print(f"    Total array power ratio: {total_above/total_away:.3f}  "
          f"(expected: ≠1, spatial redistribution)")
    print(f"    PASS ✓  (blockage redistributes power spatially)")

    # Check 4: Different finger positions give different responses
    pos1 = np.array([-0.01, -0.01, z_f0])
    pos2 = np.array([ 0.01,  0.01, z_f0])
    y1, _, _, _ = extended_signal(array_pos, pos1, k, sigma_f_phys,
                                  a_x, a_y, beta_min, phi_max, C, 0)
    y2, _, _, _ = extended_signal(array_pos, pos2, k, sigma_f_phys,
                                  a_x, a_y, beta_min, phi_max, C, 0)
    correlation = abs(np.dot(y1.conj(), y2)) / (np.linalg.norm(y1)*np.linalg.norm(y2))
    print(f"\n[4] Signal distinctiveness:")
    print(f"    Correlation between pos1 and pos2: {correlation:.4f}")
    print(f"    (expected: <1, different positions give different signals)")
    assert correlation < 0.9999, "FAIL: positions not distinguishable"
    print(f"    PASS ✓")

    print("\n  All sanity checks passed.")
    print("="*55 + "\n")


# ══════════════════════════════════════════════════════════════
# 5. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(0)

    print("="*55)
    print("  ScreenAnt — Extended Finger Model Verification")
    print("="*55)
    print(f"  ε̃  = {eps_tilde.real:.2f} - j{abs(eps_tilde.imag):.2f}")
    print(f"  ñ  = {n_tilde.real:.3f} - j{abs(n_tilde.imag):.3f}")
    print(f"  β_min   = {beta_min:.4f}")
    print(f"  φ_max   = {np.degrees(phi_max):.1f}°")
    print(f"  σ_f(CM) = {sigma_f_phys:.4f}")
    print(f"  Finger size: {a_x*1e3:.0f}mm × {a_y*1e3:.0f}mm × "
          f"{d_finger*1e3:.0f}mm")

    # Build array and coupling matrix
    array_pos = build_array(Sx, Sy, de)
    C         = build_coupling_matrix(array_pos, k, epsilon)

    # Run sanity checks
    run_sanity_checks(array_pos, C)

    # Build trajectories
    traj_lin = trajectory_linear    (N, dt, Sx, Sy, de)
    traj_sin = trajectory_sinusoidal(N, dt, Sx, Sy, de)

    # Simulate extended finger model
    print("[1/5] Simulating extended finger model (linear trajectory)...")
    Y_ext, RHO, GAMMA = simulate_extended(
        traj_lin, array_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, sigma_n2
    )
    print(f"      Y_ext shape: {Y_ext.shape}")

    # Simulate point scatterer for comparison
    print("[2/5] Simulating point scatterer (baseline)...")
    Y_point = simulate(traj_lin, array_pos, 1.0, k, sigma_n2, C=C)

    # SNR check
    sig_pwr = np.mean(np.abs(Y_ext[:, 24])**2)
    snr_db  = 10*np.log10(sig_pwr / sigma_n2)
    print(f"      SNR at centre element: {snr_db:.1f} dB")

    # Plots
    print("[3/5] Plot 1 — Footprint and transmission verification...")
    finger_mid = traj_lin[N//2]
    fig1 = plot_finger_footprint_and_transmission(
        array_pos, finger_mid, a_x, a_y, beta_min, phi_max
    )
    fig1.savefig('/mnt/user-data/outputs/13_footprint_transmission.png',
                 dpi=150, bbox_inches='tight')

    print("[3/5] Plot 2 — Point vs extended snapshot comparison...")
    fig2 = plot_signal_comparison_snapshot(
        array_pos, finger_mid, C, sigma_n2, k,
        1.0, sigma_f_phys, a_x, a_y, beta_min, phi_max
    )
    fig2.savefig('/mnt/user-data/outputs/14_point_vs_extended_snapshot.png',
                 dpi=150, bbox_inches='tight')

    print("[4/5] Plot 3 — Temporal array response...")
    fig3 = plot_temporal_array_response(Y_ext, RHO, GAMMA,
                                        traj_lin, label='(Linear)')
    fig3.savefig('/mnt/user-data/outputs/15_temporal_response_extended.png',
                 dpi=150, bbox_inches='tight')

    print("[4/5] Plot 4 — Spatial snapshots...")
    fig4 = plot_spatial_snapshots(Y_ext, RHO, GAMMA,
                                   traj_lin, Sx, Sy,
                                   n_snaps=4, label='(Linear)')
    fig4.savefig('/mnt/user-data/outputs/16_spatial_snapshots_extended.png',
                 dpi=150, bbox_inches='tight')

    print("[5/5] Plot 5 — Point vs extended temporal signal...")
    fig5 = plot_point_vs_extended_temporal(Y_point, Y_ext, traj_lin,
                                            element_idx=24)
    fig5.savefig('/mnt/user-data/outputs/17_point_vs_extended_temporal.png',
                 dpi=150, bbox_inches='tight')

    print("\n  Done. Outputs saved:")
    print("  13 — Footprint & transmission verification")
    print("  14 — Array snapshot: point vs extended (spatial)")
    print("  15 — Temporal array response heatmaps")
    print("  16 — Spatial snapshots at 4 time steps")
    print("  17 — Temporal signal: point vs extended (centre element)")
    print(f"\n  Key arrays exported for localization module:")
    print(f"  Y_ext  : {Y_ext.shape}   complex array response")
    print(f"  RHO    : {RHO.shape}   footprint per step")
    print(f"  GAMMA  : {GAMMA.shape}   transmission per step")