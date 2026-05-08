import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import F0, SX, SY, EPSILON, Z_F0, SIGMA_N2
from core.finger import *
from core.array import *
from core.trajectory import *

def extended_template(array_pos, candidate_pos, k, sigma_f_phys,
                      a_x, a_y, beta_min, phi_max, C):
    """
    Compute the noiseless extended finger template at a candidate position.

    This is the forward model prediction:
        t(q) = C · [Γ(q) ⊙ a(q)]

    Used by Matched Filter and PINN as the physics forward model.

    Unlike the point scatterer where the template was just C·a(q),
    the extended template includes the blockage matrix Γ(q) which
    encodes spatial attenuation AND phase shift — making templates
    at different positions much more distinct from each other.

    Args:
        candidate_pos : (3,) candidate finger position [x, y, z]
        ... all physical parameters ...

    Returns:
        t : (S,) complex template vector
    """
    rho   = finger_footprint(array_pos, candidate_pos, a_x, a_y)
    gamma = complex_transmission(rho, beta_min, phi_max)
    a     = extended_steering_vector(array_pos, candidate_pos, k,
                                     sigma_f_phys, a_x, a_y, rho=rho)
    return C @ (gamma * a)


def rmse_mm(pred, true):
    """RMSE between predicted and true 2D positions [mm]."""
    err = pred - true
    return np.sqrt(np.mean(np.sum(err**2, axis=1))) * 1e3


def build_search_grid(Sx, Sy, de, z_f, grid_points=35):
    """Build 2D search grid over screen area."""
    x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)
    gx = np.linspace(x_min, x_max, grid_points)
    gy = np.linspace(y_min, y_max, grid_points)
    XX, YY = np.meshgrid(gx, gy)
    grid   = np.stack([XX.ravel(), YY.ravel(),
                       np.full(XX.size, z_f)], axis=1)
    return grid, gx, gy


def prepare_features(Y):
    """
    Convert complex Y (N,S) → real feature matrix (N, 2S).
    Stack [Re(Y), Im(Y)] and normalise to O(1).
    """
    X     = np.concatenate([np.real(Y), np.imag(Y)], axis=1)
    scale = np.mean(np.abs(X)) + 1e-12
    return X / scale


def normalize_pos(pos, bounds):
    p = pos.copy()
    x_min, x_max, y_min, y_max = bounds
    p[:, 0] = 2*(pos[:, 0] - x_min)/(x_max - x_min) - 1
    p[:, 1] = 2*(pos[:, 1] - y_min)/(y_max - y_min) - 1
    return p


def denormalize_pos(p_norm, bounds):
    p = p_norm.copy()
    x_min, x_max, y_min, y_max = bounds
    p[:, 0] = (p_norm[:, 0] + 1)/2*(x_max - x_min) + x_min
    p[:, 1] = (p_norm[:, 1] + 1)/2*(y_max - y_min) + y_min
    return p



class ExtendedMUSIC:
    """
    MUSIC applied to extended finger data.

    The covariance matrix R now reflects the extended signal structure
    (blockage + phase shifts + coupling), giving a richer signal
    subspace. The noise subspace is still orthogonal to the signal
    subspace, so MUSIC still applies — but the steering vectors used
    in the pseudo-spectrum must also be extended templates t(q),
    not simple point-scatterer vectors a(q).

        R_hat = (1/W) Σ_w y(t_w) y(t_w)^H

        P_MUSIC(q) = 1 / ||U_n^H · t_norm(q)||²

    where t_norm(q) = t(q)/||t(q)|| is the normalised extended template.
    """

    def __init__(self, array_pos, k, sigma_f_phys,
                 a_x, a_y, beta_min, phi_max, C, z_f,
                 n_sources=1, window=15, grid_points=35):
        self.array_pos = array_pos
        self.n_sources = n_sources
        self.window    = window

        print(f"  [ExtMUSIC] Precomputing {grid_points}² extended templates "
              f"(W={window})...")
        self.grid, self.gx, self.gy = build_search_grid(
            Sx, Sy, de, z_f, grid_points
        )
        G = len(self.grid)
        T = np.zeros((G, S), dtype=complex)
        for g, q in enumerate(self.grid):
            T[g] = extended_template(array_pos, q, k, sigma_f_phys,
                                     a_x, a_y, beta_min, phi_max, C)
        norms    = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        self.T_n = T / norms
        print(f"  [ExtMUSIC] Ready.")

    def _noise_subspace(self, Y_win):
        W   = len(Y_win)
        R   = (Y_win.conj().T @ Y_win) / W
        _, U = np.linalg.eigh(R)
        # eigh returns ascending — reverse, take noise subspace
        U_n = U[:, ::-1][:, self.n_sources:]
        return U_n

    def locate_snapshot(self, Y_win):
        U_n  = self._noise_subspace(Y_win)
        proj = self.T_n @ U_n                           # (G, S-1)
        denom = np.sum(np.abs(proj)**2, axis=1) + 1e-12
        P    = 1.0 / denom
        best = np.argmax(P)
        P_map = P.reshape(len(self.gx), len(self.gy))
        return self.grid[best, :2], P_map

    def locate_trajectory(self, Y):
        Nt, W = len(Y), self.window
        pred  = np.zeros((Nt, 2))
        for n in range(Nt):
            w0 = max(0, n - W//2)
            w1 = min(Nt, w0 + W)
            w0 = max(0, w1 - W)
            pred[n], _ = self.locate_snapshot(Y[w0:w1])
        return pred

    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.p, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.p = pickle.load(f)