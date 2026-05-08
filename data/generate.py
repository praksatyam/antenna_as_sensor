# data/generate.py
"""
Run this once to generate and save all datasets.
Re-run only if you change physical parameters in config.py.

Usage:
    python data/generate.py
"""
import pickle, sys
sys.path.insert(0, '..')
from config import *
from core.array import *
from core.finger import *
from core.trajectory import *

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import F0, SX, SY, EPSILON, Z_F0, SIGMA_N2
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
