# data/generate.py
"""
Run this once to generate and save all datasets.
Re-run only if you change physical parameters in config.py.

Usage:
    python data/generate.py
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import *
from core.finger import *
from core.array import *
from core.trajectory import *
from localisation.base import *
from baselines.matched_filter import ExtendedMatchedFilter
from baselines.music import ExtendedMUSIC
from localisation.mlp import MLP, MLPLocalizer
from localisation.curriculum_pinn import CurriculumPINN
from localisation.cnn_pinn import CNNPINN, PINNLocalizer
from localisation.signal_based import SignalBasedModel

# array_pos = build_array(SX, SY)
# C         = build_coupling_matrix(array_pos, EPSILON)

# # Standard dataset — used by all baseline experiments
# print("Generating standard dataset...")
# Y_list, T_list = make_dataset(
#     array_pos, C, n_traj=N_TRAJ_TRAIN+N_TRAJ_TEST,
#     snr_db=TRAIN_SNR_DB, seed=RANDOM_SEED
# )
# with open('standard.pkl', 'wb') as f:
#     pickle.dump({'Y': Y_list, 'T': T_list,
#                  'n_train': N_TRAJ_TRAIN, 'n_test': N_TRAJ_TEST}, f)
# print(f"  Saved standard.pkl")

# # Noise sweep datasets — one per SNR level
# print("Generating noise sweep datasets...")
# noise_data = {}
# for snr in SNR_RANGE_DB:
#     Yl, Tl = make_dataset(array_pos, C, n_traj=10,
#                            snr_db=snr, seed=RANDOM_SEED+1)
#     noise_data[snr] = {'Y': Yl, 'T': Tl}
#     print(f"  SNR={snr} dB done")
# with open('noise_sweep.pkl', 'wb') as f:
#     pickle.dump(noise_data, f)

# print("All datasets saved.")




# ══════════════════════════════════════════════════════════════
# DATASET GENERATION
# ══════════════════════════════════════════════════════════════

def generate_dataset(array_pos, C, n_traj=40, seed=0):
    """Generate labelled dataset of (Y, trajectory) pairs."""
    rng   = np.random.default_rng(seed)
    types = ['linear', 'sinusoidal', 'random']
    Yl, Tl = [], []
    x_min,x_max,y_min,y_max = get_bounds()

    for i in range(n_traj):
        t  = types[i % 3]
        si = int(rng.integers(0, 10000))

        if t == 'linear':
            traj = trajectory_linear(N, dt, Sx, Sy, de)
            off  = rng.uniform(-de, de, 2)
            traj[:,0] = np.clip(traj[:,0]+off[0], x_min, x_max)
            traj[:,1] = np.clip(traj[:,1]+off[1], y_min, y_max)
        elif t == 'sinusoidal':
            traj = trajectory_sinusoidal(N, dt, Sx, Sy, de)
        else:
            traj = trajectory_random_walk(N, dt, Sx, Sy, de, seed=si)

        Y, _, _ = simulate_extended(
            traj, array_pos, k, sigma_f_phys,
            a_x, a_y, beta_min, phi_max, C, sigma_n2
        )
        Yl.append(Y); Tl.append(traj)

    return Yl, Tl


def split(Yl, Tl, frac=0.75):
    n    = len(Yl); nt = int(n*frac)
    Y_tr = np.concatenate(Yl[:nt]);  T_tr = np.concatenate(Tl[:nt])
    Y_te = np.concatenate(Yl[nt:]);  T_te = np.concatenate(Tl[nt:])
    return Y_tr, T_tr, Y_te, T_te


# ══════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════

COLORS = {
    'MLP':            '#457b9d',
    'Curriculum PINN':'#2a9d8f',
    'CNN-PINN':       '#f4a261',
    'Signal-Based':   '#6a4c93',
}


def plot_trajectory_comparison(T_vis, results, title=''):
    """Predicted vs true trajectories for all four methods."""
    x_min,x_max,y_min,y_max = get_bounds()
    n = len(results)
    fig, axes = plt.subplots(1, n+1, figsize=(4*(n+1), 4.5))

    ax0 = axes[0]
    ax0.plot(T_vis[:,0]*1e3, T_vis[:,1]*1e3, 'k-', lw=2)
    ax0.scatter(*T_vis[0,:2]*1e3,  color='green', s=80, zorder=5)
    ax0.scatter(*T_vis[-1,:2]*1e3, color='red',   s=80, marker='X', zorder=5)
    ax0.set_xlim(x_min*1.1e3, x_max*1.1e3)
    ax0.set_ylim(y_min*1.1e3, y_max*1.1e3)
    ax0.set_title('Ground Truth', fontweight='bold')
    ax0.set_xlabel('x [mm]'); ax0.set_ylabel('y [mm]')
    ax0.set_aspect('equal'); ax0.grid(True, alpha=0.3)

    for ax, (name, pred) in zip(axes[1:], results.items()):
        r = rmse_mm(pred, T_vis[:, :2])
        ax.plot(T_vis[:,0]*1e3, T_vis[:,1]*1e3, 'k--', alpha=0.3, lw=1)
        ax.plot(pred[:,0]*1e3, pred[:,1]*1e3,
                color=COLORS[name], lw=1.8, label='Predicted')
        ax.scatter(*T_vis[0,:2]*1e3,  color='green', s=50, zorder=5)
        ax.scatter(*T_vis[-1,:2]*1e3, color='red',   s=50, marker='X', zorder=5)
        ax.set_xlim(x_min*1.1e3, x_max*1.1e3)
        ax.set_ylim(y_min*1.1e3, y_max*1.1e3)
        ax.set_title(f'{name}\nRMSE={r:.2f} mm', fontweight='bold')
        ax.set_xlabel('x [mm]')
        ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    return fig


def plot_rmse_bar(rmse_dict, title='RMSE — Four Methods'):
    names = list(rmse_dict.keys())
    vals  = list(rmse_dict.values())
    cols  = [COLORS[n] for n in names]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(names, vals, color=cols, edgecolor='white', lw=1.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.02,
                f'{v:.3f} mm', ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('RMSE [mm]', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, max(vals)*1.3)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    return fig


def plot_training_curves(histories):
    """
    Training loss curves for all four methods.
    Each method may have different loss components so we plot them
    on separate subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.ravel()

    for ax, (name, hist) in zip(axes, histories.items()):
        color = COLORS[name]
        if isinstance(hist, dict):
            # Multiple loss components
            if 'pos' in hist:
                ax.semilogy(hist['pos'],  label='L_pos',
                            color=color, lw=2)
            if 'phys' in hist and any(v > 0 for v in hist['phys']):
                ax.semilogy(hist['phys'], label='L_phys',
                            color=color, lw=1.5, linestyle='--')
            if 'lam' in hist:
                ax2 = ax.twinx()
                ax2.plot(hist['lam'], color='gray', lw=1,
                         linestyle=':', alpha=0.7, label='λ(t)')
                ax2.set_ylabel('λ', fontsize=9, color='gray')
                ax2.legend(fontsize=8, loc='lower right')
            if 'signal' in hist:
                ax.semilogy(hist['signal'], label='L_signal',
                            color=color, lw=2)
                ax.semilogy(hist['smooth'], label='L_smooth',
                            color=color, lw=1.5, linestyle='--')
        else:
            ax.semilogy(hist, label='L_pos', color=color, lw=2)

        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss (log)')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves — All Four Methods',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_error_over_time(T_vis, results):
    """Per-timestep position error [mm] for all methods."""
    t_axis = np.arange(len(T_vis)) * dt * 1e3
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for name, pred in results.items():
        err = np.sqrt(np.sum((pred - T_vis[:,:2])**2, axis=1)) * 1e3
        ax.plot(t_axis, err, color=COLORS[name], lw=1.5, label=name, alpha=0.85)
    ax.set_xlabel('Time [ms]', fontsize=11)
    ax.set_ylabel('Position error [mm]', fontsize=11)
    ax.set_title('Per-Timestep Position Error — All Four Methods',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def run_all_four(n_traj=40, epochs=300, batch_fc=64, batch_cnn=32,
                 verbose=True, save_prefix='./results_four_models/'):
    """
    Train and evaluate all four methods. Returns results dict.
    """
    np.random.seed(0)

    print("="*65)
    print("  ScreenAnt — Four Localisation Models")
    print("  MLP | Curriculum PINN | CNN-PINN | Signal-Based")
    print("="*65)

    # Setup
    array_pos = build_array(Sx, Sy, de)
    C         = build_coupling_matrix(array_pos, k, epsilon)

    # Dataset
    print(f"\n[Data] Generating {n_traj} trajectories...")
    Yl, Tl  = generate_dataset(array_pos, C, n_traj=n_traj, seed=0)
    Y_tr, T_tr, Y_te, T_te = split(Yl, Tl)
    Y_vis = Yl[-1]; T_vis = Tl[-1]
    print(f"       Train: {Y_tr.shape}  Test: {Y_te.shape}")

    results  = {}
    histories = {}
    rmse_dict = {}
    timings   = {}

    # ── Method 1: MLP ──────────────────────────────────────
    print(f"\n[1/4] MLP ({epochs} epochs)...")
    m1 = MLP(lr=3e-3)
    t0 = time.time()
    h1 = m1.train(Y_tr, T_tr, epochs=epochs, batch=batch_fc, verbose=verbose)
    timings['MLP'] = time.time() - t0
    pred1 = m1.predict(Y_vis)
    rmse1 = rmse_mm(pred1, T_vis[:,:2])
    results['MLP']   = pred1
    histories['MLP'] = h1
    rmse_dict['MLP'] = rmse1
    print(f"       RMSE: {rmse1:.3f} mm  ({timings['MLP']:.1f}s)")

    # ── Method 2: Curriculum PINN ───────────────────────────
    print(f"\n[2/4] Curriculum PINN ({epochs} epochs)...")
    m2 = CurriculumPINN(array_pos, C, lr=3e-3, T_warm=80, T_ramp=60,
                         lam_max=0.005)
    t0 = time.time()
    h2 = m2.train(Y_tr, T_tr, epochs=epochs, batch=batch_fc, verbose=verbose)
    timings['Curriculum PINN'] = time.time() - t0
    pred2 = m2.predict(Y_vis)
    rmse2 = rmse_mm(pred2, T_vis[:,:2])
    results['Curriculum PINN']   = pred2
    histories['Curriculum PINN'] = h2
    rmse_dict['Curriculum PINN'] = rmse2
    print(f"       RMSE: {rmse2:.3f} mm  ({timings['Curriculum PINN']:.1f}s)")

    # ── Method 3: CNN-PINN ──────────────────────────────────
    print(f"\n[3/4] CNN-PINN ({epochs} epochs)...")
    m3 = CNNPINN(array_pos, C, lr=3e-3, T_warm=80, T_ramp=60, lam_max=0.005)
    t0 = time.time()
    h3 = m3.train(Y_tr, T_tr, epochs=epochs, batch=batch_cnn, verbose=verbose)
    timings['CNN-PINN'] = time.time() - t0
    pred3 = m3.predict(Y_vis)
    rmse3 = rmse_mm(pred3, T_vis[:,:2])
    results['CNN-PINN']   = pred3
    histories['CNN-PINN'] = h3
    rmse_dict['CNN-PINN'] = rmse3
    print(f"       RMSE: {rmse3:.3f} mm  ({timings['CNN-PINN']:.1f}s)")

    # ── Method 4: Signal-Based ──────────────────────────────
    print(f"\n[4/4] Signal-Based (self-supervised, {epochs} epochs)...")
    m4 = SignalBasedModel(array_pos, C, lr=2e-3, mu=0.5)
    t0 = time.time()
    # Signal-based uses Y_tr only — no position labels
    h4 = m4.train(Y_tr, epochs=epochs, batch=batch_fc, verbose=verbose,
                   warmstart=True)
    timings['Signal-Based'] = time.time() - t0
    pred4 = m4.predict(Y_vis)
    rmse4 = rmse_mm(pred4, T_vis[:,:2])
    results['Signal-Based']   = pred4
    histories['Signal-Based'] = h4
    rmse_dict['Signal-Based'] = rmse4
    print(f"       RMSE: {rmse4:.3f} mm  ({timings['Signal-Based']:.1f}s)")

    # Summary
    print("\n" + "="*65)
    print("  RESULTS")
    print("="*65)
    for name, rmse in rmse_dict.items():
        sup = "(self-supervised)" if name == "Signal-Based" else "(supervised)"
        print(f"  {name:20s}: {rmse:.3f} mm  {sup}")

    # Plots
    print("\nGenerating plots...")
    os.makedirs(save_prefix, exist_ok=True)

    fig1 = plot_trajectory_comparison(
        T_vis, results,
        title='Four Localisation Methods — Extended Finger Model'
    )
    fig1.savefig(f'{save_prefix}/four_models_trajectories.png',
                 dpi=150, bbox_inches='tight')

    fig2 = plot_rmse_bar(rmse_dict)
    fig2.savefig(f'{save_prefix}/four_models_rmse.png',
                 dpi=150, bbox_inches='tight')

    fig3 = plot_training_curves(histories)
    fig3.savefig(f'{save_prefix}/four_models_training.png',
                 dpi=150, bbox_inches='tight')

    fig4 = plot_error_over_time(T_vis, results)
    fig4.savefig(f'{save_prefix}/four_models_error_time.png',
                 dpi=150, bbox_inches='tight')

    print("  Saved: four_models_trajectories.png")
    print("  Saved: four_models_rmse.png")
    print("  Saved: four_models_training.png")
    print("  Saved: four_models_error_time.png")

    return {
        'results': results, 'rmse': rmse_dict,
        'histories': histories, 'timings': timings,
        'models': {'mlp': m1, 'cpinn': m2, 'cnn': m3, 'sig': m4}
    }


"""
data/generate.py
================
Step 1 of the experiment pipeline.

Generates the standard dataset using the extended finger model,
saves it to data/standard.pkl for use by all downstream scripts,
and produces a comprehensive set of visualisations documenting
the spatial-temporal properties of the array responses.

Figures saved (suitable for report / poster):
  01_trajectories.png          — the three trajectory types on the array
  02_array_response_heatmap.png — power and phase over time (all elements)
  03_spatial_snapshots.png      — 7×7 spatial grids at 4 time steps
  04_footprint_transmission.png — blockage footprint and γ_s maps
  05_signal_statistics.png      — amplitude distributions, SNR, correlations
  06_point_vs_extended.png      — effect of extended model vs point scatterer
  07_temporal_signal.png        — Re/Im/|y|/phase at centre element over time

Usage:
    python data/generate.py          # from project root
    python generate.py               # from data/ folder
"""

import sys, os, pickle, time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import uniform_filter1d

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Output directories ───────────────────────────────────────
DATA_DIR = _HERE
FIG_DIR  = os.path.join(_ROOT, 'results', 'figures', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

np.random.seed(RANDOM_SEED)

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130,
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

TRAJ_COLORS = {'Linear': '#E63946', 'Sinusoidal': '#2A9D8F',
               'Random Walk': '#E9C46A'}


# ════════════════════════════════════════════════════════════
# 1. BUILD ARRAY
# ════════════════════════════════════════════════════════════

print("=" * 60)
print("  ScreenAnt — Dataset Generation")
print("=" * 60)

array_pos = build_array(Sx, Sy, de)
C         = build_coupling_matrix(array_pos, k, EPSILON)
x_min, x_max, y_min, y_max = screen_bounds(Sx, Sy, de)

print(f"  Array: {Sx}×{Sy} = {S} elements")
print(f"  Aperture: {(x_max-x_min)*1e3:.1f} × {(y_max-y_min)*1e3:.1f} mm")
print(f"  Wavelength: {lam*1e3:.2f} mm  |  de: {de*1e3:.2f} mm")


# ════════════════════════════════════════════════════════════
# 2. GENERATE TRAJECTORIES AND SIGNALS
# ════════════════════════════════════════════════════════════

def make_one_traj(traj_type, seed, rng):
    """Return (traj, Y, RHO, GAMMA) for one trajectory."""
    if traj_type == 'linear':
        traj = trajectory_linear(N, dt, Sx, Sy, de)
        off  = rng.uniform(-de, de, 2)
        traj[:, 0] = np.clip(traj[:, 0] + off[0], x_min, x_max)
        traj[:, 1] = np.clip(traj[:, 1] + off[1], y_min, y_max)
    elif traj_type == 'sinusoidal':
        traj = trajectory_sinusoidal(N, dt, Sx, Sy, de)
    else:
        traj = trajectory_random_walk(N, dt, Sx, Sy, de, seed=seed)

    Y, RHO, GAMMA = simulate_extended(
        traj, array_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, sigma_n2
    )
    return traj, Y, RHO, GAMMA


N_TOTAL  = N_TRAJ_TRAIN + N_TRAJ_TEST
types    = ['linear', 'sinusoidal', 'random']
rng      = np.random.default_rng(RANDOM_SEED)

print(f"\n[1/3] Generating {N_TOTAL} trajectories...")
t0 = time.time()

Y_list     = []
T_list     = []
RHO_list   = []
GAMMA_list = []
type_list  = []   # track which type each trajectory is

for i in range(N_TOTAL):
    ttype  = types[i % 3]
    seed_i = int(rng.integers(0, 100000))
    traj, Y, RHO, GAMMA = make_one_traj(ttype, seed_i, rng)
    Y_list.append(Y)
    T_list.append(traj)
    RHO_list.append(RHO)
    GAMMA_list.append(GAMMA)
    type_list.append(ttype)
    if (i+1) % 10 == 0:
        print(f"    {i+1}/{N_TOTAL} trajectories generated...")

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Each trajectory: Y={Y_list[0].shape}  traj={T_list[0].shape}")

# Compute SNR statistics
sig_power  = np.mean([np.mean(np.abs(Y)**2) for Y in Y_list])
snr_actual = 10 * np.log10(sig_power / sigma_n2)
print(f"  Mean signal power: {sig_power:.3e}")
print(f"  Actual SNR: {snr_actual:.1f} dB")


# ════════════════════════════════════════════════════════════
# 3. SAVE DATASET
# ════════════════════════════════════════════════════════════

print("\n[2/3] Saving dataset...")

dataset = {
    # Core data
    'Y':           Y_list,        # list of (N, S) complex arrays
    'T':           T_list,        # list of (N, 3) trajectories
    'RHO':         RHO_list,      # list of (N, S) footprints
    'GAMMA':       GAMMA_list,    # list of (N, S) transmissions
    'type':        type_list,     # list of str: 'linear'/'sinusoidal'/'random'

    # Split indices
    'n_train':     N_TRAJ_TRAIN,
    'n_test':      N_TRAJ_TEST,

    # Array geometry (save so downstream scripts don't need to rebuild)
    'array_pos':   array_pos,
    'C':           C,

    # Metadata
    'meta': {
        'n_total':      N_TOTAL,
        'n_train':      N_TRAJ_TRAIN,
        'n_test':       N_TRAJ_TEST,
        'Sx': Sx, 'Sy': Sy, 'S': S,
        'de_mm':        de * 1e3,
        'lam_mm':       lam * 1e3,
        'f0_GHz':       F0 / 1e9,
        'sigma_n2':     sigma_n2,
        'snr_db':       snr_actual,
        'N_steps':      N,
        'dt_ms':        dt * 1e3,
        'beta_min':     beta_min,
        'phi_max_deg':  np.degrees(phi_max),
        'sigma_f_phys': sigma_f_phys,
        'random_seed':  RANDOM_SEED,
    }
}

save_path = os.path.join(DATA_DIR, 'standard.pkl')
with open(save_path, 'wb') as f:
    pickle.dump(dataset, f)
print(f"  Saved: {save_path}")
print(f"  File size: {os.path.getsize(save_path)/1e6:.1f} MB")


# ════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ════════════════════════════════════════════════════════════

print("\n[3/3] Generating visualisations...")

# Reference trajectories: one of each type from training set
ref_lin = next(i for i,t in enumerate(type_list) if t=='linear')
ref_sin = next(i for i,t in enumerate(type_list) if t=='sinusoidal')
ref_rnd = next(i for i,t in enumerate(type_list) if t=='random')
refs = {'Linear': ref_lin, 'Sinusoidal': ref_sin, 'Random Walk': ref_rnd}

# ── Figure 1: Trajectories ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax, (label, idx) in zip(axes, refs.items()):
    traj = T_list[idx]
    t_ax = np.arange(N) * dt * 1e3

    # Screen boundary
    rect = plt.Rectangle(
        (x_min*1e3, y_min*1e3),
        (x_max-x_min)*1e3, (y_max-y_min)*1e3,
        lw=1.5, edgecolor='#888', facecolor='#f5f5f5', zorder=0
    )
    ax.add_patch(rect)

    # Array elements
    ax.scatter(array_pos[:,0]*1e3, array_pos[:,1]*1e3,
               c='#2E86AB', s=25, zorder=2, alpha=0.5)

    # Trajectory coloured by time
    sc = ax.scatter(traj[:,0]*1e3, traj[:,1]*1e3,
                    c=t_ax, cmap='plasma', s=6, zorder=3)
    ax.scatter(*traj[0,:2]*1e3,  s=80, color='green',
               zorder=5, label='Start')
    ax.scatter(*traj[-1,:2]*1e3, s=80, color='red',
               marker='X', zorder=5, label='End')

    ax.set_xlim(x_min*1e3*1.15, x_max*1e3*1.15)
    ax.set_ylim(y_min*1e3*1.15, y_max*1e3*1.15)
    ax.set_title(label, fontweight='bold', color=TRAJ_COLORS[label])
    ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Time [ms]')

plt.suptitle('Synthetic Finger Trajectories on 7×7 ScreenAnt Array',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '01_trajectories.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_trajectories.png")


# ── Figure 2: Array response heatmap (power + phase over time) ─
fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

for col, (label, idx) in enumerate(refs.items()):
    Y    = Y_list[idx]
    t_ax = np.linspace(0, N*dt*1e3, N)

    ax_pow = fig.add_subplot(gs[0, col])
    im = ax_pow.imshow(np.abs(Y).T**2, aspect='auto', origin='lower',
                       cmap='inferno',
                       extent=[t_ax[0], t_ax[-1], 0, S])
    ax_pow.set_xlabel('Time [ms]')
    ax_pow.set_ylabel('Element index')
    ax_pow.set_title(f'{label}\n|y|² power', fontweight='bold',
                     color=TRAJ_COLORS[label])
    plt.colorbar(im, ax=ax_pow, label='Power')

    ax_pha = fig.add_subplot(gs[1, col])
    im2 = ax_pha.imshow(np.angle(Y).T, aspect='auto', origin='lower',
                        cmap='twilight',
                        extent=[t_ax[0], t_ax[-1], 0, S])
    ax_pha.set_xlabel('Time [ms]')
    ax_pha.set_ylabel('Element index')
    ax_pha.set_title(f'{label}\n∠y phase', fontweight='bold',
                     color=TRAJ_COLORS[label])
    plt.colorbar(im2, ax=ax_pha, label='Phase [rad]')

plt.suptitle('Spatial-Temporal Array Response\n'
             '(rows = antenna elements, columns = time)',
             fontsize=13, fontweight='bold')
fig.savefig(os.path.join(FIG_DIR, '02_array_response_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_array_response_heatmap.png")


# ── Figure 3: Spatial snapshots at 4 time steps ─────────────
snap_idx = [0, N//4, N//2, 3*N//4]
fig, axes = plt.subplots(3, 4, figsize=(15, 10))

idx = ref_lin   # use linear trajectory for spatial snapshots
Y   = Y_list[idx]; T = T_list[idx]
RHO = RHO_list[idx]; GAM = GAMMA_list[idx]

row_labels = ['|y|² Power', 'ρ_s Footprint', '|γ_s| Transmission']
row_cmaps  = ['inferno', 'YlOrRd', 'RdYlGn']

for col, si in enumerate(snap_idx):
    fp = T[si, :2] * 1e3   # finger position in mm

    data_rows = [
        np.abs(Y[si])**2,
        RHO[si],
        np.abs(GAM[si])
    ]

    for row, (data, cmap) in enumerate(zip(data_rows, row_cmaps)):
        ax   = axes[row, col]
        grid = data.reshape(Sx, Sy)
        im   = ax.imshow(grid, cmap=cmap, origin='upper',
                         vmin=0 if row > 0 else None)

        # Mark finger position
        near = np.argmin(np.sum((array_pos[:,:2]*1e3 -
                                  fp[np.newaxis,:])**2, axis=1))
        nr, nc = near//Sx, near%Sx
        ax.scatter(nc, nr, marker='+', s=200, color='cyan',
                   linewidths=2.5, zorder=5)

        if row == 0:
            ax.set_title(f't = {si*dt*1e3:.0f} ms\n'
                         f'({fp[0]:.1f}, {fp[1]:.1f}) mm',
                         fontsize=9, fontweight='bold')
        if col == 0:
            ax.set_ylabel(row_labels[row], fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle('Spatial Array Response at Four Time Snapshots\n'
             '(cyan + = finger position  |  Linear trajectory)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '03_spatial_snapshots.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_spatial_snapshots.png")


# ── Figure 4: Footprint and transmission maps ────────────────
mid_t = N // 2
finger_pos = np.array([*T_list[ref_lin][mid_t, :2], z_f0])
rho_map    = finger_footprint(array_pos, finger_pos, a_x, a_y)
gam_map    = complex_transmission(rho_map, beta_min, phi_max)

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
panels = [
    (rho_map,            'ρ_s  Spatial Footprint',        'viridis', [0,1]),
    (np.abs(gam_map),    '|γ_s|  Amplitude Transmission', 'RdYlGn', [beta_min,1]),
    (np.degrees(rho_map*phi_max),
                         'φ_s  Phase Shift [°]',          'coolwarm', None),
    (np.abs(Y_list[ref_lin][mid_t]).reshape(Sx,Sy),
                         '|y|  Received Amplitude',        'hot', None),
]

for ax, (data, title, cmap, vrange) in zip(axes.ravel(), panels):
    grid = data.reshape(Sx, Sy) if data.shape == (S,) else data
    vmin, vmax = vrange if vrange else (data.min(), data.max())
    im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax,
                   origin='upper', aspect='equal')

    # Finger centre
    near = np.argmin(np.sum((array_pos[:,:2] -
                              finger_pos[:2][np.newaxis,:])**2, axis=1))
    ax.scatter(near%Sx, near//Sx, marker='+', s=250, color='red',
               linewidths=2.5, zorder=5, label='Finger centre')

    # Finger ellipse outline
    theta = np.linspace(0, 2*np.pi, 100)
    cx = (finger_pos[0] - x_min) / (x_max-x_min) * (Sy-1)
    cy = (y_max - finger_pos[1]) / (y_max-y_min) * (Sx-1)
    ex = a_x / (x_max-x_min) * (Sy-1)
    ey = a_y / (y_max-y_min) * (Sx-1)
    ax.plot(cx + ex*np.cos(theta), cy + ey*np.sin(theta),
            'r--', lw=1.5, alpha=0.7, label='Finger outline')

    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=8)
    plt.colorbar(im, ax=ax)

plt.suptitle(f'Extended Finger Model — Spatial Maps\n'
             f'(finger at ({finger_pos[0]*1e3:.1f}, '
             f'{finger_pos[1]*1e3:.1f}) mm, '
             f'β_min={beta_min:.3f}, φ_max={np.degrees(phi_max):.1f}°)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '04_footprint_transmission.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_footprint_transmission.png")


# ── Figure 5: Signal statistics ──────────────────────────────
Y_all = np.concatenate(Y_list, axis=0)   # (N_total*N, S)

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# 5a: Amplitude distribution across all elements and snapshots
ax = fig.add_subplot(gs[0, 0])
amps = np.abs(Y_all).ravel()
ax.hist(amps, bins=80, color='#2E86AB', alpha=0.8, density=True)
ax.axvline(amps.mean(), color='red', lw=2,
           label=f'Mean={amps.mean():.0f}')
ax.set_xlabel('|y| Amplitude'); ax.set_ylabel('Density')
ax.set_title('Amplitude Distribution\n(all elements, all snapshots)',
             fontweight='bold')
ax.legend(fontsize=9)

# 5b: Per-element mean amplitude (shows spatial non-uniformity)
ax = fig.add_subplot(gs[0, 1])
mean_amp = np.mean(np.abs(Y_all), axis=0)
im = ax.imshow(mean_amp.reshape(Sx, Sy), cmap='hot',
               origin='upper', aspect='equal')
ax.set_title('Mean |y| per Element\n(averaged over all trajectories)',
             fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, label='Mean |y|')

# 5c: Per-element std amplitude (shows spatial variation)
ax = fig.add_subplot(gs[0, 2])
std_amp = np.std(np.abs(Y_all), axis=0)
im = ax.imshow(std_amp.reshape(Sx, Sy), cmap='viridis',
               origin='upper', aspect='equal')
ax.set_title('Std |y| per Element\n(variability driven by finger motion)',
             fontweight='bold')
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im, ax=ax, label='Std |y|')

# 5d: SNR distribution
ax = fig.add_subplot(gs[1, 0])
snr_per_snap = 10*np.log10(np.mean(np.abs(Y_all)**2, axis=1) / sigma_n2)
ax.hist(snr_per_snap, bins=50, color='#2A9D8F', alpha=0.8, density=True)
ax.axvline(snr_per_snap.mean(), color='red', lw=2,
           label=f'Mean={snr_per_snap.mean():.1f} dB')
ax.set_xlabel('SNR [dB]'); ax.set_ylabel('Density')
ax.set_title('Per-Snapshot SNR Distribution', fontweight='bold')
ax.legend(fontsize=9)

# 5e: Inter-element correlation matrix (mean over snapshots)
ax = fig.add_subplot(gs[1, 1])
Y_norm = Y_all / (np.abs(Y_all).mean(axis=1, keepdims=True) + 1e-12)
corr   = np.abs(Y_norm.conj().T @ Y_norm) / len(Y_all)
im = ax.imshow(np.abs(corr), cmap='RdYlBu_r', vmin=0, vmax=1)
ax.set_title('Inter-Element Correlation |R|\n'
             '(structure → position discriminable)',
             fontweight='bold')
ax.set_xlabel('Element index'); ax.set_ylabel('Element index')
plt.colorbar(im, ax=ax)

# 5f: Phase distribution
ax = fig.add_subplot(gs[1, 2])
phases = np.angle(Y_all).ravel()
ax.hist(phases, bins=80, color='#E9C46A', alpha=0.8, density=True)
ax.set_xlabel('Phase [rad]'); ax.set_ylabel('Density')
ax.set_title('Phase Distribution\n(near-uniform → rich spatial info)',
             fontweight='bold')
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax.set_xticklabels(['-π', '-π/2', '0', 'π/2', 'π'])

plt.suptitle('Dataset Signal Statistics', fontsize=13, fontweight='bold')
fig.savefig(os.path.join(FIG_DIR, '05_signal_statistics.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_signal_statistics.png")


# ── Figure 6: Point scatterer vs extended finger ─────────────
Y_pt, _, _ = simulate_extended(
        T_list[ref_lin], array_pos, k, sigma_f_phys,
        a_x, a_y, 1.0, 0.0, C, sigma_n2   # beta_min=1, phi_max=0
    )

Y_ext = Y_list[ref_lin]
mid   = N // 2

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
rows = [
    (Y_pt,  'Point Scatterer'),
    (Y_ext, 'Extended Finger'),
]
for row, (Y_row, label) in enumerate(rows):
    y = Y_row[mid]

    im0 = axes[row,0].imshow(np.abs(y).reshape(Sx,Sy)**2,
                              cmap='hot', origin='upper', aspect='equal')
    axes[row,0].set_title(f'{label}\n|y|² Power', fontweight='bold')
    axes[row,0].set_xticks([]); axes[row,0].set_yticks([])
    plt.colorbar(im0, ax=axes[row,0])

    im1 = axes[row,1].imshow(np.real(y).reshape(Sx,Sy),
                              cmap='RdBu_r', origin='upper', aspect='equal')
    axes[row,1].set_title(f'{label}\nRe(y)', fontweight='bold')
    axes[row,1].set_xticks([]); axes[row,1].set_yticks([])
    plt.colorbar(im1, ax=axes[row,1])

    im2 = axes[row,2].imshow(np.angle(y).reshape(Sx,Sy),
                              cmap='twilight', origin='upper', aspect='equal')
    axes[row,2].set_title(f'{label}\n∠y Phase', fontweight='bold')
    axes[row,2].set_xticks([]); axes[row,2].set_yticks([])
    plt.colorbar(im2, ax=axes[row,2])

fp_mid = T_list[ref_lin][mid, :2] * 1e3
plt.suptitle(f'Point Scatterer vs Extended Finger Model\n'
             f'Snapshot at t={mid*dt*1e3:.0f} ms  '
             f'finger=({fp_mid[0]:.1f},{fp_mid[1]:.1f}) mm',
             fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '06_point_vs_extended.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_point_vs_extended.png")


# ── Figure 7: Temporal signal at centre element ──────────────
centre_elem = S // 2   # element 24 at (0,0)
t_axis = np.arange(N) * dt * 1e3

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
Y_cen = Y_ext[:, centre_elem]

axes[0].plot(t_axis, np.abs(Y_cen), color='#2E86AB', lw=1.5)
axes[0].set_ylabel('|y| Amplitude')
axes[0].set_title(f'Signal at Centre Element (idx={centre_elem}) '
                  f'— Linear Trajectory', fontweight='bold')

axes[1].plot(t_axis, np.real(Y_cen), color='#E63946', lw=1.2)
axes[1].set_ylabel('Re(y)')

axes[2].plot(t_axis, np.imag(Y_cen), color='#2A9D8F', lw=1.2)
axes[2].set_ylabel('Im(y)')

axes[3].plot(t_axis, np.angle(Y_cen), color='#6A4C93', lw=1.2)
axes[3].set_ylabel('Phase [rad]')
axes[3].set_xlabel('Time [ms]')
axes[3].set_yticks([-np.pi, 0, np.pi])
axes[3].set_yticklabels(['-π', '0', 'π'])

# Overlay finger x-position for reference
ax_r = axes[0].twinx()
ax_r.plot(t_axis, T_list[ref_lin][:,0]*1e3,
          'k--', lw=1, alpha=0.4, label='x_f(t)')
ax_r.set_ylabel('x_f [mm]', fontsize=9)
ax_r.legend(fontsize=8, loc='upper right')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '07_temporal_signal.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_temporal_signal.png")


# ════════════════════════════════════════════════════════════
# 5. PRINT DATASET SUMMARY
# ════════════════════════════════════════════════════════════

meta = dataset['meta']
print("\n" + "="*60)
print("  DATASET SUMMARY")
print("="*60)
print(f"  Trajectories: {N_TOTAL} total "
      f"({N_TRAJ_TRAIN} train, {N_TRAJ_TEST} test)")
print(f"  Types: {N_TOTAL//3} linear, {N_TOTAL//3} sinusoidal, "
      f"{N_TOTAL//3} random walk")
print(f"  Snapshots per traj: {N}  (T={N*dt:.1f}s, dt={dt*1e3:.0f}ms)")
print(f"  Array: {Sx}×{Sy}={S} elements  |  "
      f"Aperture: {(x_max-x_min)*1e3:.1f}mm")
print(f"  SNR: {meta['snr_db']:.1f} dB")
print(f"  β_min: {meta['beta_min']:.4f}  |  "
      f"φ_max: {meta['phi_max_deg']:.1f}°")
print(f"  Saved to: {save_path}")
print(f"  Figures:  {FIG_DIR}")
print("="*60)