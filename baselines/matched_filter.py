
from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *

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

class ExtendedMatchedFilter:
    """
    Matched filter using the extended finger forward model as template.

    Template at candidate position q:
        t(q) = C · [Γ(q) ⊙ a(q)]

    MF score:
        P_MF(q) = |t(q)^H · y(t)|²  /  ||t(q)||²

    Because Γ(q) encodes the blockage pattern (which elements are
    under the finger), templates at different positions are much more
    distinct than point-scatterer templates. This improves localization
    resolution especially when the finger is close to the array.

    Precomputes all templates once (offline) for fast online inference.
    """

    def __init__(self, array_pos, k, sigma_f_phys,
                 a_x, a_y, beta_min, phi_max, C, z_f,
                 grid_points=35):
        self.array_pos = array_pos
        self.C         = C
        self.grid_points = grid_points

        print(f"  [ExtMF] Precomputing {grid_points}²={grid_points**2} "
              f"extended templates...")
        self.grid, self.gx, self.gy = build_search_grid(
            Sx, Sy, de, z_f, grid_points
        )
        G = len(self.grid)
        T = np.zeros((G, S), dtype=complex)

        for g, q in enumerate(self.grid):
            T[g] = extended_template(array_pos, q, k, sigma_f_phys,
                                     a_x, a_y, beta_min, phi_max, C)

        # Normalise each template to unit norm
        norms    = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        self.T_n = T / norms          # (G, S) normalised templates
        print(f"  [ExtMF] Ready.")

    def locate_snapshot(self, y):
        scores   = np.abs(self.T_n.conj() @ y) ** 2   # (G,)
        best     = np.argmax(scores)
        P_map    = scores.reshape(self.grid_points, self.grid_points)
        return self.grid[best, :2], P_map

    def locate_trajectory(self, Y):
        pred = np.zeros((len(Y), 2))
        for n in range(len(Y)):
            pred[n], _ = self.locate_snapshot(Y[n])
        return pred
