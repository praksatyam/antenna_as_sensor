from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *
from .base import *
from ..baselines.base import *

class CNNPINN:
    """
    Supervised + physics + spatial CNN encoder.

    Architecture:
      Conv(2→16,3×3) → LReLU
      Conv(16→32,3×3) → LReLU
      Flatten(1568)
      FC(128) → LReLU → FC(64) → LReLU → FC(2)

    The CNN processes the 7×7 spatial structure of Re(y) and Im(y)
    before the FC head, learning position-equivariant spatial features
    (footprint shape, phase gradient) rather than permutation-sensitive
    correlations.

    Training: same curriculum schedule as CurriculumPINN.
    """

    CNN_FLAT = 32 * Sx * Sy   # 32×7×7 = 1568

    def __init__(self, array_pos, C, lr=3e-3, seed=0,
                 T_warm=80, T_ramp=60, lam_max=0.005):
        self.bounds    = get_bounds()
        self.array_pos = array_pos
        self.C         = C
        self.T_warm    = T_warm
        self.T_ramp    = T_ramp
        self.lam_max   = lam_max
        self.cp        = init_cnn(seed=seed)
        self.fp        = init_fc(self.CNN_FLAT, seed=seed)
        # Separate optimisers for CNN and FC weights
        self.opt_cnn   = Adam(self.cp, lr=lr)
        self.opt_fc    = Adam(self.fp, lr=lr)

    def _lam(self, ep):
        if ep < self.T_warm:
            return 0.0
        return self.lam_max * min(1.0, (ep-self.T_warm)/max(self.T_ramp,1))

    def _forward(self, X_spatial):
        """Full forward pass: CNN encoder → FC head."""
        flat, cnn_cache = cnn_forward(X_spatial, self.cp)
        out,  fc_cache  = fc_forward(flat, self.fp)
        return out, cnn_cache, fc_cache

    def train(self, Y_tr, T_tr, epochs=300, batch=32, verbose=True):
        """
        Smaller default batch (32) because the CNN is slower per sample
        than the flat FC network.
        """
        X_sp   = prepare_spatial(Y_tr)     # (N, 2, 7, 7)
        p_true = norm_pos(T_tr[:, :2], self.bounds)
        hist   = {'total': [], 'pos': [], 'phys': [], 'lam': []}

        for ep in range(epochs):
            lam = self._lam(ep)
            idx = np.random.permutation(len(X_sp))
            ep_tot = ep_pos = ep_phys = 0.0; nb = 0

            for s in range(0, len(X_sp), batch):
                ib   = idx[s:s+batch]
                Xb   = X_sp[ib]
                pb   = p_true[ib]
                Yb   = Y_tr[ib]

                pred, cnn_cache, fc_cache = self._forward(Xb)
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

                # Backprop through FC head
                g_fc   = fc_backward(g_total, fc_cache, self.fp)

                # Gradient w.r.t. CNN flat output
                # Full chain: g_total → W3 → LReLU(z2) → W2 → LReLU(z1) → flat
                d_a2   = g_total @ self.fp['W3'].T                    # (B, h2)
                d_z2   = d_a2 * lrelu_d(fc_cache['z2'])               # (B, h2)
                d_a1   = d_z2 @ self.fp['W2'].T                       # (B, h1)
                d_z1   = d_a1 * lrelu_d(fc_cache['z1'])               # (B, h1)
                d_flat = d_z1 @ self.fp['W1'].T                       # (B, CNN_FLAT)

                g_cnn  = cnn_backward(d_flat, cnn_cache, self.cp)

                self.opt_fc.step(self.fp, g_fc)
                self.opt_cnn.step(self.cp, g_cnn)

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
        out, _, _ = self._forward(prepare_spatial(Y))
        return denorm_pos(out, self.bounds)




class PINNLocalizer:
    """
    Physics-Informed Neural Network using the EXTENDED forward model.

    Total loss:
        L = L_pos + λ · L_phys

    L_pos  = (1/N) Σ ||p_hat - p_true||²         (position MSE)

    L_phys = (1/N) Σ ||C·[Γ(p_hat)⊙a(p_hat)] - y_obs||² / ȳ
                                                   (extended physics)

    Key upgrade from the point-scatterer PINN:
      The physics loss now checks whether the FULL extended model
      (blockage + phase + coupling) is consistent with the observation.
      A wrong position gives:
        (a) wrong distance r_s  → wrong phase in a(q)
        (b) wrong footprint ρ_s → wrong Γ(q) pattern
        (c) wrong attenuation   → wrong amplitude pattern
      All three mismatches contribute to the physics gradient,
      making it a much stronger and more informative signal than before.
    """

    def __init__(self, array_pos, k, sigma_f_phys,
                 a_x, a_y, beta_min, phi_max, C,
                 lambda_phys=0.001, lr=3e-3, seed=0):
        self.array_pos   = array_pos
        self.k           = k
        self.sigma_f_phys= sigma_f_phys
        self.a_x         = a_x
        self.a_y         = a_y
        self.beta_min    = beta_min
        self.phi_max     = phi_max
        self.C           = C
        self.lambda_phys = lambda_phys
        self.net         = MLP(input_dim=2*S, lr=lr, seed=seed)
        self.bounds      = screen_bounds(Sx, Sy, de)

    def _batch_templates(self, positions):
        """
        Vectorized extended template computation for a batch.

        For each position p_hat_b:
            t_b = C · [Γ(p_hat_b) ⊙ a(p_hat_b)]

        Computed for all batch samples simultaneously using broadcasting.

        positions : (B, 3)
        returns   : (B, S) complex templates
        """
        B = len(positions)
        # Distances: diff[b,s,:] = array_pos[s] - pos[b]
        diff = (self.array_pos[np.newaxis,:,:]
                - positions[:,np.newaxis,:])          # (B,S,3)
        r    = np.sqrt(np.sum(diff**2, axis=2))       # (B,S)

        # Scattering steering vector
        A = self.sigma_f_phys * np.exp(-1j*2*self.k*r) / (r**2)  # (B,S)

        # Footprint: broadcast finger positions vs array positions
        dx  = self.array_pos[np.newaxis,:,0] - positions[:,np.newaxis,0]
        dy  = self.array_pos[np.newaxis,:,1] - positions[:,np.newaxis,1]
        rho = np.exp(-dx**2/(2*self.a_x**2)
                     -dy**2/(2*self.a_y**2))           # (B,S)

        # Weight steering vector by footprint
        A_weighted = self.sigma_f_phys * rho * np.exp(-1j*2*self.k*r) / (r**2)

        # Complex transmission: γ = β · exp(jφ)
        beta  = 1.0 - rho*(1.0 - self.beta_min)       # (B,S)
        phi   = rho * self.phi_max                     # (B,S)
        gamma = beta * np.exp(1j*phi)                  # (B,S)

        # Apply blockage to steering vector: Γ ⊙ a
        A_blocked = gamma * A_weighted                 # (B,S)

        # Apply coupling: each row through C
        # (B,S) @ C^T gives C·row for each row
        T = A_blocked @ self.C.T                       # (B,S)
        return T

    def _physics_loss_and_grad(self, pred_norm, Y_obs):
        """
        Vectorized physics loss and gradient using central finite differences.

        L_phys = mean_b ||t(p_hat_b) - y_obs_b||² / ȳ

        Gradient via central differences (more accurate than forward diff):
            ∂L/∂x ≈ [L(p+h·ex) - L(p-h·ex)] / (2h)
        """
        h     = 1e-5
        B     = len(pred_norm)
        p_hat = denormalize_pos(pred_norm, self.bounds)  # (B,2)
        pos3d = np.concatenate([p_hat,
                                 np.full((B,1), z_f0)], axis=1)  # (B,3)

        y_scale = np.mean(np.abs(Y_obs)) + 1e-12

        # Centre
        T0  = self._batch_templates(pos3d)
        R0  = (T0 - Y_obs) / y_scale
        L0  = np.real(np.sum(R0 * R0.conj(), axis=1))   # (B,)

        # x finite differences
        px = pos3d.copy(); px[:,0] += h
        mx = pos3d.copy(); mx[:,0] -= h
        Rp = (self._batch_templates(px) - Y_obs)/y_scale
        Rm = (self._batch_templates(mx) - Y_obs)/y_scale
        dLdx = (np.real(np.sum(Rp*Rp.conj(),axis=1))
               -np.real(np.sum(Rm*Rm.conj(),axis=1))) / (2*h)

        # y finite differences
        py = pos3d.copy(); py[:,1] += h
        my = pos3d.copy(); my[:,1] -= h
        Rp = (self._batch_templates(py) - Y_obs)/y_scale
        Rm = (self._batch_templates(my) - Y_obs)/y_scale
        dLdy = (np.real(np.sum(Rp*Rp.conj(),axis=1))
               -np.real(np.sum(Rm*Rm.conj(),axis=1))) / (2*h)

        phys_loss = L0.mean()

        # Convert metre-gradient to normalised coordinate gradient
        x_min,x_max,y_min,y_max = self.bounds
        grad = np.zeros((B, 2))
        grad[:,0] = dLdx * (x_max-x_min)/2 / B
        grad[:,1] = dLdy * (y_max-y_min)/2 / B
        return phys_loss, grad

    def train(self, Y, traj, epochs=200, batch=64, verbose=True):
        X      = prepare_features(Y)
        p_true = normalize_pos(traj[:,:2], self.bounds)
        N_     = len(X)
        hist_total = []; hist_pos = []; hist_phys = []

        for ep in range(epochs):
            idx  = np.random.permutation(N_)
            el = ep_p = ep_ph = 0.0; nb = 0

            for s in range(0, N_, batch):
                ib  = idx[s:s+batch]
                Xb  = X[ib]; pb  = p_true[ib]; Yb  = Y[ib]

                pred, cache = self.net.forward(Xb)

                # Position loss
                diff     = pred - pb
                pos_loss = np.mean(np.sum(diff**2, axis=1))
                g_pos    = 2*diff/len(Xb)

                # Physics loss (extended forward model)
                phys_loss, g_phys = self._physics_loss_and_grad(pred, Yb)

                total = pos_loss + self.lambda_phys * phys_loss
                g_total = g_pos + self.lambda_phys * g_phys

                grads = self.net.backward(g_total, cache)
                self.net.adam(grads)

                el   += total; ep_p += pos_loss
                ep_ph+= phys_loss; nb += 1

            hist_total.append(el/nb)
            hist_pos.append(ep_p/nb)
            hist_phys.append(ep_ph/nb)

            if verbose and (ep % 50 == 0 or ep == epochs-1):
                print(f"    Epoch {ep:4d}/{epochs}  "
                      f"total={hist_total[-1]:.5f}  "
                      f"pos={hist_pos[-1]:.5f}  "
                      f"phys={hist_phys[-1]:.4f}")

        return hist_total, hist_pos, hist_phys

    def predict(self, Y):
        X = prepare_features(Y)
        pred, _ = self.net.forward(X)
        return denormalize_pos(pred, self.bounds)
