import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, pickle, time
sys.path.insert(0, '/home/claude')   # adjust to your local path


from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *
from ..localisation import *
from localisation.curriculum_pinn import CurriculumPINN as cpinn
from localisation.cnn_pinn import CNNPINN as cnn
from localisation.signal_based import SignalBasedModel as sig
from baselines.matched_filter import ExtendedMatchedFilter
from baselines.music import ExtendedMUSIC
from ..baselines.base import *


# ══════════════════════════════════════════════════════════════
# 1. SNR UTILITIES
# ══════════════════════════════════════════════════════════════

def compute_signal_power(array_pos, C, seed=99):
    """
    Estimate mean noiseless signal power P_signal.
    Used to set σ_n² = P_signal / 10^(SNR/10).
    """
    powers = []
    for traj_fn in [trajectory_linear, trajectory_sinusoidal]:
        traj = traj_fn(N, dt, Sx, Sy, de)
        Y, _, _ = simulate_extended(
            traj, array_pos, k, sigma_f_phys,
            a_x, a_y, beta_min, phi_max, C, sigma_n2=0.0
        )
        powers.append(np.mean(np.abs(Y)**2))
    traj = trajectory_random_walk(N, dt, Sx, Sy, de, seed=42)
    Y, _, _ = simulate_extended(
        traj, array_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, sigma_n2=0.0
    )
    powers.append(np.mean(np.abs(Y)**2))
    return float(np.mean(powers))


def snr_to_noise_var(snr_db, P_signal):
    """σ_n² = P_signal / 10^(SNR/10)"""
    return P_signal / (10 ** (snr_db / 10))


def noise_var_to_snr(sn2, P_signal):
    """SNR [dB] = 10 log10(P_signal / σ_n²)"""
    return 10 * np.log10(P_signal / (sn2 + 1e-300))


# ══════════════════════════════════════════════════════════════
# 2. DATASET HELPERS
# ══════════════════════════════════════════════════════════════

def make_trajectories(n, array_pos, C, sn2, seed):
    """Generate n test trajectories at a given noise level."""
    rng   = np.random.default_rng(seed)
    types = ['linear', 'sinusoidal', 'random']
    Yl, Tl = [], []
    xmn, xmx, ymn, ymx = screen_bounds(Sx, Sy, de)
    for i in range(n):
        t = types[i % 3]
        si = int(rng.integers(0, 10000))
        if t == 'linear':
            traj = trajectory_linear(N, dt, Sx, Sy, de)
            off  = rng.uniform(-de, de, 2)
            traj[:, 0] = np.clip(traj[:, 0]+off[0], xmn, xmx)
            traj[:, 1] = np.clip(traj[:, 1]+off[1], ymn, ymx)
        elif t == 'sinusoidal':
            traj = trajectory_sinusoidal(N, dt, Sx, Sy, de)
        else:
            traj = trajectory_random_walk(N, dt, Sx, Sy, de, seed=si)
        Y, _, _ = simulate_extended(
            traj, array_pos, k, sigma_f_phys,
            a_x, a_y, beta_min, phi_max, C, sn2
        )
        Yl.append(Y); Tl.append(traj)
    return Yl, Tl


def mean_rmse(method, Yl, Tl):
    """Mean RMSE [mm] of a method over a list of trajectories."""
    fn = getattr(method, 'locate_trajectory', None) or method.predict
    return float(np.mean([
        rmse_mm(fn(Y), T[:, :2]) for Y, T in zip(Yl, Tl)
    ]))


# ══════════════════════════════════════════════════════════════
# 3. TRAIN NEURAL NETWORKS
# ══════════════════════════════════════════════════════════════

def train_networks(array_pos, C, P_signal,
                   train_snr_db=25.0, n_traj=30, epochs=200):
    """
    Train MLP and PINN at a reference SNR.

    Both networks are trained at the same SNR. The sweep then shows
    how each degrades when test SNR departs from training SNR.
    This is the realistic deployment scenario: you train once on
    collected data, then deploy across varying conditions.
    """
    sn2 = snr_to_noise_var(train_snr_db, P_signal)
    print(f"\n  Training at SNR={train_snr_db} dB  (σ_n²={sn2:.3e})")
    Yl, Tl = make_trajectories(n_traj, array_pos, C, sn2, seed=0)
    Y_tr = np.concatenate(Yl); T_tr = np.concatenate(Tl)
    print(f"  Training data: {Y_tr.shape}")

    print("  Training MLP...")
    mlp = MLPLocalizer(lr=3e-3, seed=0)
    mlp.train(Y_tr, T_tr, epochs=epochs, batch=64, verbose=False)

    print("  Training PINN...")
    pinn = PINNLocalizer(array_pos, k, sigma_f_phys,
                         a_x, a_y, beta_min, phi_max, C,
                         lambda_phys=0.001, lr=3e-3, seed=0)
    pinn.train(Y_tr, T_tr, epochs=epochs, batch=64, verbose=False)

    print("  Networks ready.")
    return mlp, pinn


# ══════════════════════════════════════════════════════════════
# 4. SWEEP
# ══════════════════════════════════════════════════════════════

def run_sweep(array_pos, C, P_signal,
              snr_range=None, n_test=8, grid_points=35,
              train_snr=25.0, n_train=30, epochs=200):
    """
    Full noise sweep: builds methods, trains networks, evaluates at
    every SNR point.

    Returns results dict and SNR array.
    """
    if snr_range is None:
        snr_range = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    snr_db = np.array(snr_range, dtype=float)

    print("\n" + "="*60)
    print("  NOISE SWEEP")
    print("="*60)
    print(f"  P_signal  = {P_signal:.4e}")
    print(f"  SNR range : {snr_range[0]} → {snr_range[-1]} dB")
    print(f"  Test traj : {n_test} per SNR point")
    print(f"  Train SNR : {train_snr} dB  ({n_train} trajectories)")

    # Physics methods — built once, SNR-agnostic
    print("\n  Building physics methods...")
    mf = ExtendedMatchedFilter(
        array_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, z_f0, grid_points=grid_points
    )
    mu = ExtendedMUSIC(
        array_pos, k, sigma_f_phys,
        a_x, a_y, beta_min, phi_max, C, z_f0,
        window=12, grid_points=grid_points
    )

    # Neural networks — trained once at reference SNR
    mlp, pinn = train_networks(
        array_pos, C, P_signal,
        train_snr_db=train_snr, n_traj=n_train, epochs=epochs
    )

    results = {'Matched Filter': [], 'MUSIC': [],
               'MLP': [], 'PINN-MLP': []}

    print(f"\n  {'SNR':>6}  {'MF':>8}  {'MUSIC':>8}  "
          f"{'MLP':>8}  {'PINN':>8}  {'time':>6}")
    print("  " + "-"*55)

    for snr in snr_range:
        sn2  = snr_to_noise_var(snr, P_signal)
        Yl, Tl = make_trajectories(n_test, array_pos, C, sn2, seed=77)
        t0 = time.time()

        r_mf   = mean_rmse(mf,   Yl, Tl)
        r_mu   = mean_rmse(mu,   Yl, Tl)
        r_mlp  = mean_rmse(mlp,  Yl, Tl)
        r_pinn = mean_rmse(pinn, Yl, Tl)

        results['Matched Filter'].append(r_mf)
        results['MUSIC'].append(r_mu)
        results['MLP'].append(r_mlp)
        results['PINN-MLP'].append(r_pinn)

        print(f"  {snr:>6.1f}  {r_mf:>8.3f}  {r_mu:>8.3f}  "
              f"{r_mlp:>8.3f}  {r_pinn:>8.3f}  "
              f"{time.time()-t0:>5.1f}s")

    for k2 in results:
        results[k2] = np.array(results[k2])

    return results, snr_db


# ══════════════════════════════════════════════════════════════
# 5. PLOTS
# ══════════════════════════════════════════════════════════════

COLORS  = {'Matched Filter':'#457b9d','MUSIC':'#2a9d8f',
           'MLP':'#f4a261','PINN-MLP':'#6a4c93'}
MARKERS = {'Matched Filter':'o','MUSIC':'s','MLP':'^','PINN-MLP':'D'}


def plot_sweep(results, snr_db, train_snr=25.0, out=None):
    """Four-panel noise sweep plot."""
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

    # Panel 1 — linear RMSE
    ax = fig.add_subplot(gs[0, 0])
    for n, r in results.items():
        ax.plot(snr_db, r, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=6, label=n)
    ax.axvline(train_snr, color='gray', ls=':', lw=1.5,
               label=f'Train SNR ({train_snr:.0f} dB)')
    ax.set_xlabel('SNR [dB]'); ax.set_ylabel('RMSE [mm]')
    ax.set_title('RMSE vs SNR — All Methods', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2 — log RMSE
    ax = fig.add_subplot(gs[0, 1])
    for n, r in results.items():
        ax.semilogy(snr_db, r, color=COLORS[n], marker=MARKERS[n],
                    lw=2, ms=6, label=n)
    ax.axvline(train_snr, color='gray', ls=':', lw=1.5)
    ax.set_xlabel('SNR [dB]'); ax.set_ylabel('RMSE [mm] (log)')
    ax.set_title('RMSE vs SNR — Log Scale\n'
                 '(reveals low-SNR behaviour)', fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which='both')

    # Panel 3 — degradation from high-SNR ceiling
    ax = fig.add_subplot(gs[1, 0])
    for n, r in results.items():
        deg = r - r[-1]
        ax.plot(snr_db, deg, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=6, label=n)
    ax.axhline(0, color='black', lw=0.8)
    ax.axvline(train_snr, color='gray', ls=':', lw=1.5)
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('ΔRMSE = RMSE(SNR) − RMSE(40 dB) [mm]')
    ax.set_title('Degradation From High-SNR Ceiling\n'
                 '(how much each method suffers from noise)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 4 — PINN advantage over MLP
    ax = fig.add_subplot(gs[1, 1])
    adv = results['MLP'] - results['PINN-MLP']
    ax.plot(snr_db, adv, color='#6a4c93', marker='D', lw=2.5, ms=7)
    ax.axhline(0, color='black', lw=1.2, ls='--')
    ax.fill_between(snr_db, adv, 0, where=(adv > 0),
                    alpha=0.25, color='#6a4c93', label='PINN better')
    ax.fill_between(snr_db, adv, 0, where=(adv < 0),
                    alpha=0.25, color='#f4a261', label='MLP better')
    ax.axvline(train_snr, color='gray', ls=':', lw=1.5,
               label=f'Train SNR')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('RMSE_MLP − RMSE_PINN [mm]')
    ax.set_title('PINN Advantage Over MLP\n'
                 '(+ve = PINN better, −ve = MLP better)',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle('Noise Sweep — Extended Finger Model\n'
                 f'(train SNR={train_snr:.0f} dB, 7×7 array, 28 GHz)',
                 fontsize=13, fontweight='bold')
    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
    return fig


def plot_snr_threshold(results, snr_db, target_rmse=2.0, out=None):
    """Minimum SNR to achieve target RMSE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, r in results.items():
        ax.plot(snr_db, r, color=COLORS[n], marker=MARKERS[n],
                lw=2, ms=6, label=n)
        below = snr_db[r < target_rmse]
        if len(below):
            ax.axvline(below[0], color=COLORS[n], ls='--', alpha=0.5, lw=1)

    ax.axhline(target_rmse, color='red', ls='-.', lw=1.5,
               label=f'Target = {target_rmse} mm')
    ax.set_xlabel('SNR [dB]', fontsize=11)
    ax.set_ylabel('RMSE [mm]', fontsize=11)
    ax.set_title(f'Minimum SNR for RMSE < {target_rmse} mm',
                 fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"\n  SNR thresholds for RMSE < {target_rmse} mm:")
    for n, r in results.items():
        below = snr_db[r < target_rmse]
        if len(below):
            print(f"    {n:18s}: {below[0]:.1f} dB")
        else:
            print(f"    {n:18s}: does not reach target in sweep range")

    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        print(f"  Saved: {out}")
    return fig


def print_table(results, snr_db):
    print("\n" + "="*70)
    print("  NOISE SWEEP — RMSE TABLE [mm]")
    print("="*70)
    hdr = f"{'SNR':>8}" + "".join(f"{n:>16}" for n in results)
    print(hdr); print("-"*70)
    for i, snr in enumerate(snr_db):
        row = f"{snr:>8.1f}" + "".join(
            f"{results[n][i]:>16.3f}" for n in results)
        print(row)
    print("="*70)


# ══════════════════════════════════════════════════════════════
# 6. MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    np.random.seed(0)

    print("="*60)
    print("  ScreenAnt — Noise Sweep Analysis")
    print("="*60)

    array_pos = build_array(Sx, Sy, de)
    C         = build_coupling_matrix(array_pos, k, epsilon)

    print("\nEstimating signal power...")
    P_signal = compute_signal_power(array_pos, C)
    print(f"  P_signal = {P_signal:.4e}")
    print(f"  Default σ_n²={sigma_n2:.1e} → "
          f"SNR≈{noise_var_to_snr(sigma_n2, P_signal):.1f} dB")

    results, snr_db = run_sweep(
        array_pos, C, P_signal,
        snr_range   = [0, 5, 10, 15, 20, 25, 30, 35, 40],
        n_test      = 8,      # test trajectories per SNR point
        grid_points = 35,     # MF/MUSIC grid resolution
        train_snr   = 25.0,   # reference training SNR
        n_train     = 30,     # training trajectories
        epochs      = 200,    # training epochs
    )

    print_table(results, snr_db)

    # Save
    with open('/tmp/noise_sweep.pkl', 'wb') as f:
        pickle.dump({'results': results, 'snr_db': snr_db,
                     'P_signal': P_signal}, f)
    print("\n  Results saved to /tmp/noise_sweep.pkl")

    # Plots
    plot_sweep(results, snr_db, train_snr=25.0,
               out='./results/22_noise_sweep.png')
    plot_snr_threshold(results, snr_db, target_rmse=2.0,
                       out='./results/23_snr_threshold.png')

    print("\n  Done.")
    print("  22 — Full noise sweep (4 panels)")
    print("  23 — SNR threshold for target RMSE")