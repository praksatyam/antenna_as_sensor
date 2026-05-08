import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import F0, SX, SY, EPSILON, Z_F0, SIGMA_N2
from core.finger import *
from core.array import *
from core.trajectory import *
from localisation.base import *
from baselines.base import *


class CurriculumPINN:
    """
    Supervised + physics with staged training.

    Phase 1 (epochs 0 .. T_warm):
        L = L_pos             (position loss only)
        λ(t) = 0

    Phase 2 (epochs T_warm .. end):
        L = L_pos + λ · L_phys
        λ(t) linearly ramps from 0 to λ_max over T_ramp epochs,
        then stays at λ_max.
        This lets the network reach the basin of attraction first,
        then the physics loss refines the prediction from within
        the trough — where its gradient is most reliable.
    """

    def __init__(self, array_pos, C, lr=3e-3, seed=0,
                 T_warm=80, T_ramp=60, lam_max=0.005):
        self.bounds    = get_bounds()
        self.array_pos = array_pos
        self.C         = C
        self.T_warm    = T_warm
        self.T_ramp    = T_ramp
        self.lam_max   = lam_max
        self.p         = init_fc(2*S, seed=seed)
        self.opt       = Adam(self.p, lr=lr)

    def _lam(self, ep):
        """Lambda schedule: 0 during warmup, then linear ramp."""
        if ep < self.T_warm:
            return 0.0
        ramp = min(1.0, (ep - self.T_warm) / max(self.T_ramp, 1))
        return self.lam_max * ramp

    def train(self, Y_tr, T_tr, epochs=300, batch=64, verbose=True):
        X      = prepare_flat(Y_tr)
        p_true = norm_pos(T_tr[:, :2], self.bounds)
        hist   = {'total': [], 'pos': [], 'phys': [], 'lam': []}

        for ep in range(epochs):
            lam = self._lam(ep)
            idx = np.random.permutation(len(X))
            ep_tot = ep_pos = ep_phys = 0.0; nb = 0

            for s in range(0, len(X), batch):
                ib   = idx[s:s+batch]
                Xb, pb, Yb = X[ib], p_true[ib], Y_tr[ib]

                pred, cache = fc_forward(Xb, self.p)
                diff     = pred - pb
                pos_loss = np.mean(np.sum(diff**2, axis=1))
                g_pos    = 2*diff / len(ib)

                if lam > 0:
                    phys_loss, g_phys = physics_loss_and_grad(
                        pred, Yb, self.array_pos, self.C, self.bounds
                    )
                    g_total = g_pos + lam * g_phys
                else:
                    phys_loss = 0.0
                    g_total   = g_pos

                grads = fc_backward(g_total, cache, self.p)
                self.opt.step(self.p, grads)

                ep_tot  += pos_loss + lam*phys_loss
                ep_pos  += pos_loss
                ep_phys += phys_loss
                nb += 1

            hist['total'].append(ep_tot/nb)
            hist['pos'].append(ep_pos/nb)
            hist['phys'].append(ep_phys/nb)
            hist['lam'].append(lam)

            if verbose and (ep % 60 == 0 or ep == epochs-1):
                print(f"    [{ep:4d}/{epochs}]  λ={lam:.4f}  "
                      f"L_pos={hist['pos'][-1]:.5f}  "
                      f"L_phys={hist['phys'][-1]:.4f}")
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