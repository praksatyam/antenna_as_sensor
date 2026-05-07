import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, pickle, time
sys.path.insert(0, '/home/claude')   # adjust to your local path

# ── Core physics ── 
from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *
# ══════════════════════════════════════════════════════════════
# 1. PER-ARRAY-SIZE BUILDERS
#    Everything that depends on S must be rebuilt here.
# ══════════════════════════════════════════════════════════════

def build_array_size(Sx):
    """
    Build all array-size-dependent quantities for an Sx × Sx array.

    Returns a config dict containing:
        Sx, Sy, S          : dimensions
        array_pos          : (S, 3) element positions
        C                  : (S, S) coupling matrix
        aperture_mm        : physical aperture [mm]
        x_min/max, y_min/max : screen bounds [m]
    """
    Sy  = Sx
    S   = Sx * Sy

    # Element positions — same formula as ScreenAnt eq.(1)
    positions = []
    for s in range(1, S + 1):
        sx_idx = (s - 1) % Sx
        sy_idx = (s - 1) // Sx
        x = de * (-(Sx - 1) / 2 + sx_idx)
        y = de * ( (Sy - 1) / 2 - sy_idx)
        positions.append([x, y, 0.0])
    array_pos = np.array(positions)

    # Coupling matrix
    C = np.eye(S, dtype=complex)
    for s in range(S):
        for sp in range(S):
            if s == sp:
                continue
            d   = np.linalg.norm(array_pos[s] - array_pos[sp])
            kd  = k * d
            C[s, sp] = epsilon * np.exp(-1j * kd) / kd

    x_min, x_max = -(Sx-1)/2*de, (Sx-1)/2*de
    y_min, y_max = -(Sy-1)/2*de, (Sy-1)/2*de

    return {
        'Sx': Sx, 'Sy': Sy, 'S': S,
        'array_pos': array_pos, 'C': C,
        'aperture_mm': (Sx - 1) * de * 1e3,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
    }


def make_trajectories_for_cfg(cfg, n, sn2, seed):
    """
    Generate n trajectories for a given array config and noise level.
    Trajectories are scaled to fit within the array's screen bounds.
    """
    rng    = np.random.default_rng(seed)
    types  = ['linear', 'sinusoidal', 'random']
    Sx, Sy = cfg['Sx'], cfg['Sy']
    S_loc  = cfg['S']
    xmn, xmx = cfg['x_min'], cfg['x_max']
    ymn, ymx = cfg['y_min'], cfg['y_max']
    Yl, Tl = [], []

    for i in range(n):
        t  = types[i % 3]
        si = int(rng.integers(0, 10000))

        if t == 'linear':
            tvals = np.linspace(0, 1, N)
            x0, x1 = xmn * 0.8, xmx * 0.8
            y0, y1 = ymn * 0.8, ymx * 0.8
            off    = rng.uniform(-de, de, 2)
            xf = np.clip(x0 + (x1-x0)*tvals + off[0], xmn, xmx)
            yf = np.clip(y0 + (y1-y0)*tvals + off[1], ymn, ymx)
            zf = np.full(N, z_f0)
            traj = np.stack([xf, yf, zf], axis=1)

        elif t == 'sinusoidal':
            tvals = np.linspace(0, 2*np.pi, N)
            xf = xmx * 0.8 * np.sin(tvals)
            yf = ymx * 0.8 * np.sin(2 * tvals)
            zf = np.full(N, z_f0)
            traj = np.stack([xf, yf, zf], axis=1)

        else:
            from scipy.ndimage import uniform_filter1d
            step = de * 0.3
            rng2 = np.random.default_rng(si)
            dx = uniform_filter1d(rng2.normal(0, step, N), size=10)
            dy = uniform_filter1d(rng2.normal(0, step, N), size=10)
            xf = np.cumsum(dx); yf = np.cumsum(dy)
            xf -= xf.mean(); yf -= yf.mean()
            sc = min((xmx*0.8)/(np.abs(xf).max()+1e-9),
                     (ymx*0.8)/(np.abs(yf).max()+1e-9))
            xf = np.clip(xf*sc, xmn, xmx)
            yf = np.clip(yf*sc, ymn, ymx)
            zf = np.full(N, z_f0)
            traj = np.stack([xf, yf, zf], axis=1)

        # Simulate using local array and noise sized for S_loc elements
        Y = _simulate_extended_local(
            traj, cfg['array_pos'], cfg['C'], S_loc, sn2
        )
        Yl.append(Y); Tl.append(traj)

    return Yl, Tl


def _simulate_extended_local(trajectory, array_pos, C, S_loc, sn2):
    """
    Simulate extended finger model for an arbitrary-sized array.
    Self-contained so noise vector is sized correctly for S_loc.
    """
    Nt = len(trajectory)
    Y  = np.zeros((Nt, S_loc), dtype=complex)

    for n in range(Nt):
        fp  = trajectory[n]
        dx  = array_pos[:, 0] - fp[0]
        dy  = array_pos[:, 1] - fp[1]
        rho = np.exp(-dx**2/(2*a_x**2) - dy**2/(2*a_y**2))

        diff = array_pos - fp
        r    = np.sqrt(np.sum(diff**2, axis=1))
        a_sv = sigma_f_phys * rho * np.exp(-1j*2*k*r) / (r**2)

        beta  = 1.0 - rho*(1.0 - beta_min)
        phi   = rho * phi_max
        gamma = beta * np.exp(1j*phi)

        y_nl  = C @ (gamma * a_sv)
        noise = (np.random.randn(S_loc) + 1j*np.random.randn(S_loc)) \
                * np.sqrt(sn2 / 2)
        Y[n]  = y_nl + noise

    return Y


# ══════════════════════════════════════════════════════════════
# 2. LOCALISATION METHODS (rebuilt per array size)
# ══════════════════════════════════════════════════════════════

def build_search_grid_cfg(cfg, grid_points):
    """Build 2D search grid for a given array config."""
    gx = np.linspace(cfg['x_min'], cfg['x_max'], grid_points)
    gy = np.linspace(cfg['y_min'], cfg['y_max'], grid_points)
    XX, YY = np.meshgrid(gx, gy)
    grid   = np.stack([XX.ravel(), YY.ravel(),
                       np.full(XX.size, z_f0)], axis=1)
    return grid, gx, gy


def extended_template_cfg(cfg, candidate_pos):
    """Compute extended finger template at candidate_pos for given config."""
    array_pos = cfg['array_pos']
    C         = cfg['C']

    # Footprint
    dx  = array_pos[:, 0] - candidate_pos[0]
    dy  = array_pos[:, 1] - candidate_pos[1]
    rho = np.exp(-dx**2/(2*a_x**2) - dy**2/(2*a_y**2))

    # Steering vector
    diff = array_pos - candidate_pos
    r    = np.sqrt(np.sum(diff**2, axis=1))
    a    = sigma_f_phys * rho * np.exp(-1j*2*k*r) / (r**2)

    # Complex transmission
    beta  = 1.0 - rho*(1.0 - beta_min)
    phi   = rho * phi_max
    gamma = beta * np.exp(1j*phi)

    return C @ (gamma * a)


def build_mf(cfg, grid_points=30):
    """Build matched filter for a given array config."""
    grid, gx, gy = build_search_grid_cfg(cfg, grid_points)
    G = len(grid)
    S = cfg['S']
    T = np.zeros((G, S), dtype=complex)
    for g, q in enumerate(grid):
        T[g] = extended_template_cfg(cfg, q)
    norms = np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    T_n   = T / norms
    return T_n, grid, gx, gy


def mf_locate_trajectory(T_n, grid, Y):
    """Locate finger at every timestep using matched filter."""
    pred = np.zeros((len(Y), 2))
    for n in range(len(Y)):
        scores  = np.abs(T_n.conj() @ Y[n]) ** 2
        best    = np.argmax(scores)
        pred[n] = grid[best, :2]
    return pred


def build_music(cfg, grid_points=30):
    """Build MUSIC for a given array config (same templates as MF)."""
    return build_mf(cfg, grid_points)   # templates are shared


def music_locate_trajectory(T_n, grid, Y, n_sources=1, window=12):
    """Locate finger using MUSIC with sliding window."""
    Nt   = len(Y)
    S    = Y.shape[1]
    pred = np.zeros((Nt, 2))

    for n in range(Nt):
        w0 = max(0, n - window//2)
        w1 = min(Nt, w0 + window)
        w0 = max(0, w1 - window)
        Yw = Y[w0:w1]

        R   = (Yw.conj().T @ Yw) / len(Yw)
        _, U = np.linalg.eigh(R)
        U_n  = U[:, ::-1][:, n_sources:]

        proj  = T_n @ U_n
        denom = np.sum(np.abs(proj)**2, axis=1) + 1e-12
        best  = np.argmax(1.0 / denom)
        pred[n] = grid[best, :2]

    return pred


# ══════════════════════════════════════════════════════════════
# 3. NEURAL NETWORK (pure NumPy, rebuilt per array size)
# ══════════════════════════════════════════════════════════════

def leaky_relu(x, a=0.01):
    return np.where(x > 0, x, a*x)

def leaky_relu_grad(x, a=0.01):
    return np.where(x > 0, 1.0, a)


class MLP_cfg:
    """
    MLP sized for a specific array (input_dim = 2*S).
    Architecture: FC(128) → LReLU → FC(64) → LReLU → FC(2)
    Kept constant across array sizes so results reflect the
    information content of the array, not the network capacity.
    """
    def __init__(self, S, lr=3e-3, seed=0):
        rng = np.random.default_rng(seed)
        d   = 2 * S
        self.lr = lr
        self.W1 = rng.normal(0, np.sqrt(2/d),   (d,   128))
        self.b1 = np.zeros(128)
        self.W2 = rng.normal(0, np.sqrt(2/128), (128,  64))
        self.b2 = np.zeros(64)
        self.W3 = rng.normal(0, np.sqrt(2/64),  (64,    2))
        self.b3 = np.zeros(2)
        self.m  = {k: np.zeros_like(v) for k,v in self._p().items()}
        self.v  = {k: np.zeros_like(v) for k,v in self._p().items()}
        self.t  = 0

    def _p(self):
        return dict(W1=self.W1,b1=self.b1,W2=self.W2,
                    b2=self.b2,W3=self.W3,b3=self.b3)

    def forward(self, X):
        z1=X@self.W1+self.b1; a1=leaky_relu(z1)
        z2=a1@self.W2+self.b2; a2=leaky_relu(z2)
        out=a2@self.W3+self.b3
        return out, {'X':X,'z1':z1,'a1':a1,'z2':z2,'a2':a2}

    def backward(self, g, cache):
        B=g.shape[0]
        dW3=cache['a2'].T@g/B; db3=g.mean(0); da2=g@self.W3.T
        dz2=da2*leaky_relu_grad(cache['z2'])
        dW2=cache['a1'].T@dz2/B; db2=dz2.mean(0); da1=dz2@self.W2.T
        dz1=da1*leaky_relu_grad(cache['z1'])
        dW1=cache['X'].T@dz1/B; db1=dz1.mean(0)
        return dict(W1=dW1,b1=db1,W2=dW2,b2=db2,W3=dW3,b3=db3)

    def adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for k in self._p():
            self.m[k]=b1*self.m[k]+(1-b1)*grads[k]
            self.v[k]=b2*self.v[k]+(1-b2)*grads[k]**2
            mh=self.m[k]/(1-b1**self.t)
            vh=self.v[k]/(1-b2**self.t)
            self._p()[k] -= self.lr*mh/(np.sqrt(vh)+eps)


def prepare_features_cfg(Y):
    """Normalize and stack real/imaginary parts."""
    X = np.concatenate([np.real(Y), np.imag(Y)], axis=1)
    return X / (np.mean(np.abs(X)) + 1e-12)


def normalize_pos_cfg(pos, cfg):
    p = pos.copy()
    p[:,0] = 2*(pos[:,0]-cfg['x_min'])/(cfg['x_max']-cfg['x_min']) - 1
    p[:,1] = 2*(pos[:,1]-cfg['y_min'])/(cfg['y_max']-cfg['y_min']) - 1
    return p


def denormalize_pos_cfg(p, cfg):
    out = p.copy()
    out[:,0] = (p[:,0]+1)/2*(cfg['x_max']-cfg['x_min']) + cfg['x_min']
    out[:,1] = (p[:,1]+1)/2*(cfg['y_max']-cfg['y_min']) + cfg['y_min']
    return out


def train_mlp_cfg(cfg, Y_tr, T_tr, epochs=200, batch=64):
    """Train MLP for a given array config."""
    net    = MLP_cfg(cfg['S'], lr=3e-3, seed=0)
    X      = prepare_features_cfg(Y_tr)
    p_true = normalize_pos_cfg(T_tr[:,:2], cfg)
    N_     = len(X)

    for ep in range(epochs):
        idx = np.random.permutation(N_)
        for s in range(0, N_, batch):
            ib   = idx[s:s+batch]
            Xb, pb = X[ib], p_true[ib]
            pred, cache = net.forward(Xb)
            diff = pred - pb
            grads = net.backward(2*diff/len(Xb), cache)
            net.adam(grads)

    return net


def predict_mlp_cfg(net, cfg, Y):
    X = prepare_features_cfg(Y)
    pred, _ = net.forward(X)
    return denormalize_pos_cfg(pred, cfg)


def train_pinn_cfg(cfg, Y_tr, T_tr, epochs=200, batch=64,
                   lambda_phys=0.001):
    """Train PINN for a given array config."""
    net    = MLP_cfg(cfg['S'], lr=3e-3, seed=0)
    X      = prepare_features_cfg(Y_tr)
    p_true = normalize_pos_cfg(T_tr[:,:2], cfg)
    N_     = len(X)
    array_pos = cfg['array_pos']
    C         = cfg['C']
    h         = 1e-5

    def batch_templates(positions):
        """Vectorized extended template for a batch of positions."""
        B    = len(positions)
        diff = array_pos[np.newaxis,:,:] - positions[:,np.newaxis,:]
        r    = np.sqrt(np.sum(diff**2, axis=2))
        dx   = array_pos[np.newaxis,:,0] - positions[:,np.newaxis,0]
        dy   = array_pos[np.newaxis,:,1] - positions[:,np.newaxis,1]
        rho  = np.exp(-dx**2/(2*a_x**2) - dy**2/(2*a_y**2))
        A    = sigma_f_phys * rho * np.exp(-1j*2*k*r) / (r**2)
        beta = 1.0 - rho*(1.0-beta_min)
        phi  = rho * phi_max
        gamma= beta * np.exp(1j*phi)
        return (gamma*A) @ C.T

    def physics_loss_grad(pred_norm, Y_obs):
        B     = len(pred_norm)
        p_hat = denormalize_pos_cfg(pred_norm, cfg)
        pos3d = np.concatenate([p_hat, np.full((B,1),z_f0)], axis=1)
        ys    = np.mean(np.abs(Y_obs)) + 1e-12

        T0  = batch_templates(pos3d)
        R0  = (T0 - Y_obs)/ys
        L0  = np.real(np.sum(R0*R0.conj(), axis=1))

        px=pos3d.copy(); px[:,0]+=h
        mx=pos3d.copy(); mx[:,0]-=h
        Rp=(batch_templates(px)-Y_obs)/ys
        Rm=(batch_templates(mx)-Y_obs)/ys
        dLdx=(np.real(np.sum(Rp*Rp.conj(),axis=1))
             -np.real(np.sum(Rm*Rm.conj(),axis=1)))/(2*h)

        py=pos3d.copy(); py[:,1]+=h
        my=pos3d.copy(); my[:,1]-=h
        Rp=(batch_templates(py)-Y_obs)/ys
        Rm=(batch_templates(my)-Y_obs)/ys
        dLdy=(np.real(np.sum(Rp*Rp.conj(),axis=1))
             -np.real(np.sum(Rm*Rm.conj(),axis=1)))/(2*h)

        grad=np.zeros((B,2))
        grad[:,0]=dLdx*(cfg['x_max']-cfg['x_min'])/2/B
        grad[:,1]=dLdy*(cfg['y_max']-cfg['y_min'])/2/B
        return L0.mean(), grad

    for ep in range(epochs):
        idx = np.random.permutation(N_)
        for s in range(0, N_, batch):
            ib   = idx[s:s+batch]
            Xb, pb, Yb = X[ib], p_true[ib], Y_tr[ib]
            pred, cache = net.forward(Xb)
            diff     = pred - pb
            pos_loss = np.mean(np.sum(diff**2,axis=1))
            g_pos    = 2*diff/len(Xb)
            phys_loss, g_phys = physics_loss_grad(pred, Yb)
            g_total = g_pos + lambda_phys*g_phys
            grads   = net.backward(g_total, cache)
            net.adam(grads)

    return net


# ══════════════════════════════════════════════════════════════
# 4. RMSE UTILITY
# ══════════════════════════════════════════════════════════════

def rmse_mm(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred-true)**2,axis=1))))*1e3


# ══════════════════════════════════════════════════════════════
# 5. FULL ARRAY SIZE SWEEP
# ══════════════════════════════════════════════════════════════

def run_array_sweep(array_sizes=None,
                    snr_db=25.0,
                    n_train=25,
                    n_test=8,
                    epochs=200,
                    grid_points=30,
                    snr_mode='centre'):
    """
    Sweep array size and evaluate all four methods.

    For each Sx × Sx array:
      1. Build array geometry and coupling matrix
      2. Generate training and test trajectories
      3. Build MF and MUSIC (physics methods)
      4. Train MLP and PINN
      5. Evaluate all methods and record RMSE

    Key quantities also recorded:
      - Physical aperture [mm]
      - Number of elements S
      - Array processing gain 10*log10(S) [dB]

    Args:
        array_sizes : list of Sx values (Sy = Sx always)
        snr_db      : fixed SNR for all experiments [dB]
        n_train     : training trajectories per array size
        n_test      : test trajectories per array size
        epochs      : training epochs for neural networks
        grid_points : search grid resolution for MF/MUSIC
        snr_mode    : 'mean'   — SNR relative to mean power across all elements
                      'centre' — SNR relative to centre element power only
                      'mean' is what we used before (gives higher noise at
                      large arrays because edge elements dilute mean power).
                      'centre' is fairer — same effective SNR at the most
                      informative element regardless of array size.

    Returns:
        results : dict of RMSE arrays per method
        sizes   : array size configs (for plotting)
    """
    if array_sizes is None:
        array_sizes = [3, 5, 7, 9, 11]

    print("\n" + "="*65)
    print("  ARRAY SIZE SWEEP — Extended Finger Model")
    print("="*65)
    print(f"  Array sizes  : {array_sizes}")
    print(f"  SNR          : {snr_db} dB  (fixed, mode='{snr_mode}')")
    print(f"  Train traj   : {n_train}  Test traj: {n_test}")
    print(f"  Epochs       : {epochs}")

    results = {'Matched Filter': [], 'MUSIC': [],
               'MLP': [], 'PINN-MLP': []}
    meta    = []

    for Sx in array_sizes:
        print(f"\n{'─'*65}")
        print(f"  Array: {Sx}×{Sx} = {Sx**2} elements  "
              f"aperture={(Sx-1)*de*1e3:.1f} mm")
        t_total = time.time()

        cfg = build_array_size(Sx)
        S   = cfg['S']

        # ── Compute noise variance ──
        # Run two noiseless trajectories to estimate signal power
        traj0 = make_trajectories_for_cfg(cfg, 2, 0.0, seed=999)
        Y0_list = [Y for Y, _ in zip(*traj0)]

        if snr_mode == 'centre':
            # Centre element: closest array element to origin (0,0)
            # This is the element that always receives the strongest
            # signal regardless of array size, giving a fair comparison.
            centre_idx = int(np.argmin(
                np.sum(cfg['array_pos'][:, :2]**2, axis=1)
            ))
            P_sig = float(np.mean([
                np.mean(np.abs(Y[:, centre_idx])**2) for Y in Y0_list
            ]))
            print(f"  Centre element idx={centre_idx}  "
                  f"pos=({cfg['array_pos'][centre_idx,0]*1e3:.1f},"
                  f"{cfg['array_pos'][centre_idx,1]*1e3:.1f}) mm")
        else:
            # Mean power across all elements — what we used before.
            # Diluted by edge elements at large arrays.
            P_sig = float(np.mean([
                np.mean(np.abs(Y)**2) for Y in Y0_list
            ]))

        sn2 = P_sig / (10 ** (snr_db / 10))
        print(f"  P_signal={P_sig:.3e}  σ_n²={sn2:.3e}  "
              f"SNR≈{10*np.log10(P_sig/sn2):.1f} dB  "
              f"[mode={snr_mode}]")

        # ── Adaptive grid: use Rayleigh resolution as grid spacing ──
        #
        # For near-field localisation at height z_f above an array
        # with aperture L, the Rayleigh resolution (minimum resolvable
        # separation) is:
        #
        #   d_Rayleigh = z_f * lambda / (2 * L)
        #
        # This is the physically correct grid spacing — finer grids
        # add no new information since the array cannot resolve below
        # this limit. Using it ensures fair comparison across array
        # sizes: each array's grid matches its actual resolving power.
        #
        # Note: lambda/2 element spacing sets the spatial frequency
        # bandwidth, NOT the position resolution. Near-field geometry
        # gives sub-lambda/2 position resolution.
        #
        # Cap at 80×80 points to keep runtime manageable.
        L_aperture = cfg['aperture_mm'] * 1e-3   # [m]
        if L_aperture > 0:
            d_rayleigh = z_f0 * lam / (2 * L_aperture)   # [m]
        else:
            d_rayleigh = lam / 2
        d_rayleigh = max(d_rayleigh, 0.1e-3)   # floor at 0.1mm

        gp = min(80, max(15,
                         int(cfg['aperture_mm'] * 1e-3 / d_rayleigh) + 1))
        actual_spacing_mm = (cfg['aperture_mm'] /
                             (gp - 1)) if gp > 1 else cfg['aperture_mm']
        print(f"  Rayleigh={d_rayleigh*1e3:.2f} mm  "
              f"Grid: {gp}×{gp}={gp**2} pts  "
              f"actual spacing={actual_spacing_mm:.2f} mm")

        meta.append({
            'Sx': Sx, 'S': S,
            'aperture_mm': cfg['aperture_mm'],
            'proc_gain_db': 10*np.log10(S),
            'grid_points': gp,
            'grid_spacing_mm': actual_spacing_mm,
        })

        # ── Generate data ──
        print(f"  Generating {n_train} train + {n_test} test trajectories...")
        Yl_tr, Tl_tr = make_trajectories_for_cfg(cfg, n_train, sn2, seed=0)
        Yl_te, Tl_te = make_trajectories_for_cfg(cfg, n_test,  sn2, seed=77)
        Y_tr = np.concatenate(Yl_tr); T_tr = np.concatenate(Tl_tr)
        print(f"  Train: {Y_tr.shape}")

        # ── Method 1: Matched Filter ──
        print(f"  Building MF ({gp}²={gp**2} templates)...",
              end=' ', flush=True)
        t0 = time.time()
        T_n, grid, gx, gy = build_mf(cfg, gp)
        rmse_mf = np.mean([
            rmse_mm(mf_locate_trajectory(T_n, grid, Y), T[:,:2])
            for Y, T in zip(Yl_te, Tl_te)
        ])
        print(f"RMSE={rmse_mf:.3f} mm  ({time.time()-t0:.1f}s)")

        # ── Method 2: MUSIC ──
        print(f"  Building MUSIC...", end=' ', flush=True)
        t0 = time.time()
        rmse_mu = np.mean([
            rmse_mm(music_locate_trajectory(T_n, grid, Y), T[:,:2])
            for Y, T in zip(Yl_te, Tl_te)
        ])
        print(f"RMSE={rmse_mu:.3f} mm  ({time.time()-t0:.1f}s)")

        # ── Method 3: MLP ──
        print(f"  Training MLP ({epochs} epochs)...", end=' ', flush=True)
        t0 = time.time()
        net_mlp  = train_mlp_cfg(cfg, Y_tr, T_tr, epochs=epochs)
        rmse_mlp = np.mean([
            rmse_mm(predict_mlp_cfg(net_mlp, cfg, Y), T[:,:2])
            for Y, T in zip(Yl_te, Tl_te)
        ])
        print(f"RMSE={rmse_mlp:.3f} mm  ({time.time()-t0:.1f}s)")

        # ── Method 4: PINN ──
        print(f"  Training PINN ({epochs} epochs)...", end=' ', flush=True)
        t0 = time.time()
        net_pinn  = train_pinn_cfg(cfg, Y_tr, T_tr,
                                    epochs=epochs, lambda_phys=0.001)
        rmse_pinn = np.mean([
            rmse_mm(predict_mlp_cfg(net_pinn, cfg, Y), T[:,:2])
            for Y, T in zip(Yl_te, Tl_te)
        ])
        print(f"RMSE={rmse_pinn:.3f} mm  ({time.time()-t0:.1f}s)")

        results['Matched Filter'].append(rmse_mf)
        results['MUSIC'].append(rmse_mu)
        results['MLP'].append(rmse_mlp)
        results['PINN-MLP'].append(rmse_pinn)

        print(f"  Total time: {time.time()-t_total:.1f}s")

    for k2 in results:
        results[k2] = np.array(results[k2])

    return results, meta


# ══════════════════════════════════════════════════════════════
# 6. VISUALISATION
# ══════════════════════════════════════════════════════════════

COLORS  = {'Matched Filter':'#457b9d','MUSIC':'#2a9d8f',
           'MLP':'#f4a261','PINN-MLP':'#6a4c93'}
MARKERS = {'Matched Filter':'o','MUSIC':'s','MLP':'^','PINN-MLP':'D'}


def plot_array_sweep(results, meta, snr_db=25.0,
                     snr_mode='centre', out=None):
    """
    Five-panel array size sweep plot.

    Panel 1 — RMSE vs number of elements S
    Panel 2 — RMSE vs physical aperture [mm]
    Panel 3 — RMSE vs processing gain [dB]
    Panel 4 — PINN advantage vs array size
    Panel 5 — Relative improvement from 3×3 baseline
    """
    S_vals       = np.array([m['S'] for m in meta])
    aperture_mm  = np.array([m['aperture_mm'] for m in meta])
    proc_gain_db = np.array([m['proc_gain_db'] for m in meta])
    sx_labels    = [f"{m['Sx']}×{m['Sx']}" for m in meta]

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 3, figure=fig,
                            hspace=0.4, wspace=0.35)

    # ── Panel 1: RMSE vs S ──
    ax = fig.add_subplot(gs[0, 0])
    for n, r in results.items():
        ax.plot(S_vals, r, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=7, label=n)
    # Theoretical 1/sqrt(S) reference line for physics methods
    mf0 = results['Matched Filter'][0]
    ax.plot(S_vals, mf0*np.sqrt(S_vals[0]/S_vals),
            'k--', lw=1, alpha=0.5, label='1/√S reference')
    ax.set_xticks(S_vals)
    ax.set_xticklabels(sx_labels, rotation=30, fontsize=8)
    ax.set_xlabel('Array size', fontsize=11)
    ax.set_ylabel('RMSE [mm]', fontsize=11)
    ax.set_title('RMSE vs Array Size', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 2: RMSE vs aperture ──
    ax = fig.add_subplot(gs[0, 1])
    for n, r in results.items():
        ax.plot(aperture_mm, r, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=7, label=n)
    ax.set_xlabel('Physical aperture [mm]', fontsize=11)
    ax.set_ylabel('RMSE [mm]', fontsize=11)
    ax.set_title('RMSE vs Physical Aperture\n'
                 '(device design perspective)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 3: RMSE vs processing gain ──
    ax = fig.add_subplot(gs[0, 2])
    for n, r in results.items():
        ax.plot(proc_gain_db, r, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=7, label=n)
    ax.set_xlabel('Array processing gain 10·log₁₀(S) [dB]', fontsize=11)
    ax.set_ylabel('RMSE [mm]', fontsize=11)
    ax.set_title('RMSE vs Processing Gain\n'
                 '(linearises physics-method curves)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 4: PINN advantage ──
    ax = fig.add_subplot(gs[1, 0])
    adv = results['MLP'] - results['PINN-MLP']
    ax.plot(S_vals, adv, color='#6a4c93', marker='D', lw=2.5, ms=7)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    ax.fill_between(S_vals, adv, 0, where=(adv > 0),
                    alpha=0.25, color='#6a4c93', label='PINN better')
    ax.fill_between(S_vals, adv, 0, where=(adv < 0),
                    alpha=0.25, color='#f4a261', label='MLP better')
    ax.set_xticks(S_vals)
    ax.set_xticklabels(sx_labels, rotation=30, fontsize=8)
    ax.set_xlabel('Array size', fontsize=11)
    ax.set_ylabel('RMSE_MLP − RMSE_PINN [mm]', fontsize=10)
    ax.set_title('PINN Advantage Over MLP\nvs Array Size', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # ── Panel 5: Relative improvement from 3×3 ──
    ax = fig.add_subplot(gs[1, 1:])
    for n, r in results.items():
        rel_improvement = (r[0] - r) / r[0] * 100
        ax.plot(S_vals, rel_improvement,
                color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=7, label=n)
    ax.set_xticks(S_vals)
    ax.set_xticklabels(sx_labels, rotation=30, fontsize=8)
    ax.set_xlabel('Array size', fontsize=11)
    ax.set_ylabel('Improvement over 3×3 baseline [%]', fontsize=11)
    ax.set_title('Relative RMSE Improvement From 3×3 Baseline\n'
                 '(diminishing returns curve)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)

    plt.suptitle(
        f'Array Size Sweep — Extended Finger Model\n'
        f'(SNR={snr_db:.0f} dB [{snr_mode}-element normalisation], '
        f'spacing={de*1e3:.1f} mm, f₀=28 GHz)',
        fontsize=13, fontweight='bold'
    )
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {out}")
    return fig


def print_table(results, meta):
    """Print results table."""
    print("\n" + "="*75)
    print("  ARRAY SIZE SWEEP — RMSE TABLE [mm]")
    print("="*75)
    hdr = (f"{'Array':>8}  {'S':>4}  {'Ap[mm]':>7}"
           + "".join(f"{n:>16}" for n in results))
    print(hdr); print("-"*75)
    for i, m in enumerate(meta):
        row = (f"{m['Sx']:>5}×{m['Sx']:<2}  {m['S']:>4}  "
               f"{m['aperture_mm']:>7.1f}"
               + "".join(f"{results[n][i]:>16.3f}" for n in results))
        print(row)
    print("="*75)

    # Highlight best method at each size
    print("\n  Best method at each array size:")
    for i, m in enumerate(meta):
        best = min(results, key=lambda n: results[n][i])
        print(f"    {m['Sx']}×{m['Sx']}: {best}  "
              f"({results[best][i]:.3f} mm)")


# ══════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(0)

    print("="*65)
    print("  ScreenAnt — Array Size Sweep (SNR Normalisation Comparison)")
    print("="*65)

    ARRAY_SIZES = [3, 5, 7, 9, 11]
    COMMON_ARGS = dict(
        array_sizes = ARRAY_SIZES,
        snr_db      = 25.0,
        n_train     = 25,
        n_test      = 8,
        epochs      = 200,
        grid_points = 30,
    )

    # ── Run 1: centre-element SNR (fair comparison) ──
    print("\n" + "█"*65)
    print("  RUN 1 — Centre-element SNR normalisation (FAIR)")
    print("█"*65)
    results_centre, meta = run_array_sweep(
        **COMMON_ARGS, snr_mode='centre'
    )
    print_table(results_centre, meta)
    plot_array_sweep(results_centre, meta, snr_db=25.0,
                     snr_mode='centre',
                     out='./result_array_sweep/24_array_sweep_centre_snr.png')

    # ── Run 2: mean SNR (original, for comparison) ──
    print("\n" + "█"*65)
    print("  RUN 2 — Mean SNR normalisation (shows edge dilution effect)")
    print("█"*65)
    results_mean, _ = run_array_sweep(
        **COMMON_ARGS, snr_mode='mean'
    )
    print_table(results_mean, meta)
    plot_array_sweep(results_mean, meta, snr_db=25.0,
                     snr_mode='mean',
                     out='./result_array_sweep/25_array_sweep_mean_snr.png')

    # ── Comparison plot: both modes side by side ──
    print("\nGenerating comparison plot...")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    S_vals    = np.array([m['S'] for m in meta])
    sx_labels = [f"{m['Sx']}×{m['Sx']}" for m in meta]

    COLORS  = {'Matched Filter':'#457b9d','MUSIC':'#2a9d8f',
               'MLP':'#f4a261','PINN-MLP':'#6a4c93'}
    MARKERS = {'Matched Filter':'o','MUSIC':'s','MLP':'^','PINN-MLP':'D'}

    for row, (results, mode, label) in enumerate([
        (results_centre, 'centre', 'Centre-element SNR (fair)'),
        (results_mean,   'mean',   'Mean SNR (diluted at large arrays)'),
    ]):
        # RMSE vs S
        ax = axes[row, 0]
        for n, r in results.items():
            ax.plot(S_vals, r, color=COLORS[n], marker=MARKERS[n],
                    lw=2, ms=6, label=n)
        ax.set_xticks(S_vals)
        ax.set_xticklabels(sx_labels, rotation=30, fontsize=8)
        ax.set_xlabel('Array size'); ax.set_ylabel('RMSE [mm]')
        ax.set_title(f'RMSE vs Array Size\n{label}', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Relative improvement
        ax = axes[row, 1]
        for n, r in results.items():
            rel = (r[0] - r) / r[0] * 100
            ax.plot(S_vals, rel, color=COLORS[n], marker=MARKERS[n],
                    lw=2, ms=6, label=n)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(S_vals)
        ax.set_xticklabels(sx_labels, rotation=30, fontsize=8)
        ax.set_xlabel('Array size')
        ax.set_ylabel('Improvement over 3×3 [%]')
        ax.set_title(f'Relative Improvement From 3×3\n{label}',
                     fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.suptitle(
        'SNR Normalisation Comparison\n'
        'Centre-element vs Mean — Array Size Sweep',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout()
    fig.savefig('./result_array_sweep/26_snr_mode_comparison.png',
                dpi=150, bbox_inches='tight')
    print("  Saved: ./result_array_sweep/26_snr_mode_comparison.png")

    # Save all results
    with open('/tmp/array_sweep_both.pkl', 'wb') as f:
        pickle.dump({'centre': results_centre,
                     'mean':   results_mean,
                     'meta':   meta}, f)
    print("  Results saved to /tmp/array_sweep_both.pkl")

    print("\n  Done.")
    print("  24 — Array sweep: centre-element SNR (fair)")
    print("  25 — Array sweep: mean SNR (shows edge dilution)")
    print("  26 — Side-by-side comparison of both normalisation modes")