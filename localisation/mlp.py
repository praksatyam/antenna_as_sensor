from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *
from .base import *
from ..baselines.base import *

class MLP:
    """
    Fully supervised MLP.
    Input:  [Re(y), Im(y)] / scale  ∈ R^{2S}
    Loss:   L_pos = (1/N) Σ ||p̂ - p_true||²
    """

    def __init__(self, lr=3e-3, seed=0):
        self.bounds = get_bounds()
        self.p      = init_fc(2*S, seed=seed)
        self.opt    = Adam(self.p, lr=lr)

    def forward(self, X):
        return fc_forward(X, self.p)

    def train(self, Y_tr, T_tr, epochs=200, batch=64, verbose=True):
        X      = prepare_flat(Y_tr)
        p_true = norm_pos(T_tr[:, :2], self.bounds)
        hist   = []

        for ep in range(epochs):
            idx = np.random.permutation(len(X))
            ep_loss = 0.0; nb = 0
            for s in range(0, len(X), batch):
                ib = idx[s:s+batch]
                pred, cache = self.forward(X[ib])
                diff = pred - p_true[ib]
                loss = np.mean(np.sum(diff**2, axis=1))
                grads = fc_backward(2*diff/len(ib), cache, self.p)
                self.opt.step(self.p, grads)
                ep_loss += loss; nb += 1
            hist.append(ep_loss/nb)
            if verbose and (ep % 50 == 0 or ep == epochs-1):
                print(f"    [{ep:4d}/{epochs}]  L_pos={hist[-1]:.5f}")
        return hist

    def predict(self, Y):
        pred, _ = self.forward(prepare_flat(Y))
        return denorm_pos(pred, self.bounds)

class MLPLocalizer:
    """
    Plain MLP trained on extended finger data.

    Loss: L = (1/N) Σ ||p_hat(t) - p_true(t)||²

    The network sees the richer spatial patterns from the extended model
    (blockage dips, phase distortions) but has no explicit physics
    knowledge — it learns to interpret these patterns purely from data.
    """

    def __init__(self, lr=3e-3, seed=0):
        self.net    = MLP(input_dim=2*S, lr=lr, seed=seed)
        self.bounds = screen_bounds(Sx, Sy, de)

    def train(self, Y, traj, epochs=200, batch=64, verbose=True):
        X      = prepare_features(Y)
        p_true = normalize_pos(traj[:, :2], self.bounds)
        N_     = len(X)
        hist   = []

        for ep in range(epochs):
            idx  = np.random.permutation(N_)
            ep_l = 0.0; nb = 0
            for s in range(0, N_, batch):
                Xb = X[idx[s:s+batch]];  pb = p_true[idx[s:s+batch]]
                pred, cache = self.net.forward(Xb)
                diff = pred - pb
                loss = np.mean(np.sum(diff**2, axis=1))
                grads = self.net.backward(2*diff/len(Xb), cache)
                self.net.adam(grads)
                ep_l += loss; nb += 1
            hist.append(ep_l/nb)
            if verbose and (ep % 50 == 0 or ep == epochs-1):
                print(f"    Epoch {ep:4d}/{epochs}  loss={hist[-1]:.6f}")

        return hist

    def predict(self, Y):
        X = prepare_features(Y)
        pred, _ = self.net.forward(X)
        return denormalize_pos(pred, self.bounds)