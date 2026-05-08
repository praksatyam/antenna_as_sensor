import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import F0, SX, SY, EPSILON, Z_F0, SIGMA_N2
from core.finger import *
from core.array import *
from core.trajectory import *
from localisation.base import *
from baselines.base import *
from core.finger import *
from core.array import *
from core.trajectory import *



class SignalBasedModel:
    """
    Self-supervised: no position labels used during training.

    Loss:
        L = L_signal + μ · L_smooth

        L_signal = (1/N) Σ ||t(p̂(t)) - y_obs(t)||² / ȳ
            Physics consistency: predicted position must produce
            an array response matching the observed signal.

        L_smooth = (1/N) Σ ||p̂(t) - p̂(t-1)||²
            Temporal regulariser: consecutive predictions must be
            close (finger cannot teleport). This prevents the
            network from fitting noise by jumping to implausible
            positions.

    Initialisation:
        Cold-starting from random weights produces poor predictions
        from which the physics gradient is uninformative (pointing
        away from the true position). Instead, we warm-start using
        a fast matched-filter estimate on the training data:
        for each snapshot, find the grid point minimising L_signal,
        then run 20 epochs of pure signal loss from that init.
        This places the network in the basin of attraction.

    Note on position labels:
        Position labels are used ONLY for evaluation (RMSE), not
        for training. The model is truly self-supervised.
    """

    def __init__(self, array_pos, C, lr=2e-3, seed=0, mu=0.5,
                 grid_points=20):
        self.bounds    = get_bounds()
        self.array_pos = array_pos
        self.C         = C
        self.mu        = mu
        self.gp        = grid_points
        self.p         = init_fc(2*S, seed=seed)
        self.opt       = Adam(self.p, lr=lr)
        self._build_grid()

    def _build_grid(self):
        """Precompute coarse MF grid for warm-start initialisation."""
        x_min,x_max,y_min,y_max = self.bounds
        gx = np.linspace(x_min, x_max, self.gp)
        gy = np.linspace(y_min, y_max, self.gp)
        XX, YY = np.meshgrid(gx, gy)
        grid3d = np.stack([XX.ravel(), YY.ravel(),
                           np.full(XX.size, z_f0)], axis=1)
        T = batch_forward(grid3d, self.array_pos, self.C)
        norms = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
        self.grid_T  = T / norms        # (G, S) normalised templates
        self.grid_xy = grid3d[:, :2]    # (G, 2)

    def _mf_init_pred(self, Y):
        """
        Fast matched-filter warm-start: for each snapshot find the
        grid point with highest template correlation.
        Returns (N, 2) initial position estimates.
        """
        scores = np.abs(self.grid_T.conj() @ Y.T) ** 2  # (G, N)
        best   = np.argmax(scores, axis=0)               # (N,)
        return self.grid_xy[best]                         # (N, 2)

    def _signal_loss_and_grad(self, pred_norm, Y_obs):
        """Physics consistency loss (same as physics_loss_and_grad)."""
        return physics_loss_and_grad(
            pred_norm, Y_obs, self.array_pos, self.C, self.bounds
        )

    def _smooth_loss_and_grad(self, pred_norm):
        """
        Temporal smoothness loss and gradient.

        L_smooth = (1/N) Σ_t ||p̂(t) - p̂(t-1)||²

        Gradient at position t:
            ∂L/∂p̂(t) = 2·(p̂(t) - p̂(t-1)) + 2·(p̂(t) - p̂(t+1))
        (boundary terms use only one neighbour)
        """
        N_ = len(pred_norm)
        diff_fwd = np.zeros_like(pred_norm)
        diff_bwd = np.zeros_like(pred_norm)

        # Forward differences p̂(t) - p̂(t+1)
        diff_fwd[:-1] = pred_norm[:-1] - pred_norm[1:]
        # Backward differences p̂(t) - p̂(t-1)
        diff_bwd[1:]  = pred_norm[1:]  - pred_norm[:-1]

        loss  = np.mean(np.sum(diff_bwd**2, axis=1))
        grad  = 2*(diff_fwd + diff_bwd) / N_
        return loss, grad

    def warmstart(self, Y_tr, epochs_ws=25, batch=64, verbose=True):
        """
        Warm-start: fit network to matched-filter initial estimates
        using position loss only (treating MF estimates as pseudo-labels).
        Runs for a small number of epochs — just enough to get the
        network into the right basin of attraction.
        """
        if verbose:
            print("    [Warm-start] Computing MF pseudo-labels...")
        mf_pred   = self._mf_init_pred(Y_tr)
        pseudo_norm = norm_pos(mf_pred, self.bounds)
        X    = prepare_flat(Y_tr)

        for ep in range(epochs_ws):
            idx = np.random.permutation(len(X))
            for s in range(0, len(X), batch):
                ib   = idx[s:s+batch]
                pred, cache = fc_forward(X[ib], self.p)
                diff  = pred - pseudo_norm[ib]
                grads = fc_backward(2*diff/len(ib), cache, self.p)
                self.opt.step(self.p, grads)

        if verbose:
            print(f"    [Warm-start] Done ({epochs_ws} epochs).")

    def train(self, Y_tr, epochs=250, batch=64, verbose=True,
              warmstart=True):
        """
        Self-supervised training. Y_tr only — no position labels.

        If warmstart=True, runs MF pseudo-label initialisation first.
        """
        if warmstart:
            self.warmstart(Y_tr, verbose=verbose)

        X    = prepare_flat(Y_tr)
        hist = {'total': [], 'signal': [], 'smooth': []}

        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_tot = ep_sig = ep_smo = 0.0; nb = 0

            for s in range(0, len(X), batch):
                ib   = idx[s:s+batch]
                Xb, Yb = X[ib], Y_tr[ib]

                pred, cache = fc_forward(Xb, self.p)

                # Signal consistency loss (physics)
                sig_loss, g_sig = self._signal_loss_and_grad(pred, Yb)

                # Temporal smoothness loss
                smo_loss, g_smo = self._smooth_loss_and_grad(pred)

                g_total = g_sig + self.mu * g_smo
                grads   = fc_backward(g_total, cache, self.p)
                self.opt.step(self.p, grads)

                ep_tot += sig_loss + self.mu*smo_loss
                ep_sig += sig_loss
                ep_smo += smo_loss
                nb += 1

            hist['total'].append(ep_tot/nb)
            hist['signal'].append(ep_sig/nb)
            hist['smooth'].append(ep_smo/nb)

            if verbose and (ep % 50 == 0 or ep == epochs-1):
                print(f"    [{ep:4d}/{epochs}]  "
                      f"L_sig={hist['signal'][-1]:.5f}  "
                      f"L_smo={hist['smooth'][-1]:.5f}")
        return hist

    def predict(self, Y):
        pred, _ = fc_forward(prepare_flat(Y), self.p)
        return denorm_pos(pred, self.bounds)
    
    def save(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.p, f)

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.p = pickle.load(f)
