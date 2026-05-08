"""
experiments/03_compare.py
==========================
Step 3 of the experiment pipeline.

Loads all trained models from checkpoints/ and the Signal-Based
model (trained here since it needs no position labels), evaluates
all four on the same test set, and produces the complete comparison.

Prerequisites:
    python data/generate.py            (generates data/standard.pkl)
    python experiments/02_train_models.py  (trains and saves MLP, CPINN, CNN)

Outputs saved to results/figures/comparison/ and results/tables/:
    01_rmse_bar.png           — RMSE bar chart: all 4 methods
    02_trajectories.png       — predicted vs true paths: all 4 methods
    03_error_over_time.png    — per-timestep error over one trajectory
    04_error_heatmap.png      — spatial error heatmap (where does each
                                method struggle on the screen?)
    05_rmse_per_traj_type.png — RMSE broken down by trajectory type
    comparison_results.csv    — all RMSE numbers for the paper table
"""

import sys, os, pickle, time, csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    F0, LR_SIGNAL, RANDOM_SEED, EPOCHS_SIGNAL, BATCH_FC, Z_F0
)
from data.load import load_standard
from localisation.mlp import MLP
from localisation.curriculum_pinn import CurriculumPINN
from localisation.cnn_pinn import CNNPINN
from localisation.signal_based import SignalBasedModel
from baselines.matched_filter import ExtendedMatchedFilter

# ── Directories ──────────────────────────────────────────────
CKPT_DIR  = os.path.join(_ROOT, 'checkpoints')
FIG_DIR   = os.path.join(_ROOT, 'results', 'figures', 'comparison')
TABLE_DIR = os.path.join(_ROOT, 'results', 'tables')
os.makedirs(FIG_DIR,   exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

COLORS = {
    'Matched Filter':   '#1d3557',
    'MLP':              '#457b9d',
    'Curriculum PINN':  '#2a9d8f',
    'CNN-PINN':         '#f4a261',
    'Signal-Based':     '#6a4c93',
}
MARKERS = {
    'Matched Filter':  'o',
    'MLP':             's',
    'Curriculum PINN': '^',
    'CNN-PINN':        'D',
    'Signal-Based':    'P',
}


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def rmse_mm(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true)**2, axis=1)))) * 1e3


def per_timestep_error(pred, true):
    return np.sqrt(np.sum((pred - true)**2, axis=1)) * 1e3


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════

print("=" * 65)
print("  ScreenAnt — Four-Method Comparison")
print("=" * 65)

print("\n[Loading dataset]")
(Y_tr, T_tr, Y_te, T_te,
 array_pos, C,
 Y_list, T_list, type_list,
 n_train) = load_standard(return_all=True)

print(f"  Train: {Y_tr.shape}  Test: {Y_te.shape}")

# One test trajectory of each type for visualisation
te_types  = type_list[n_train:]   # types for test trajectories
N_per     = len(Y_list[0])        # snapshots per trajectory

# Individual test trajectories (not stacked)
n_test_traj = len(Y_list) - n_train
Y_te_list   = Y_list[n_train:]
T_te_list   = T_list[n_train:]
type_te     = te_types

# Pick one of each type for per-trajectory plots
vis_idx = {}
for t in ['linear', 'sinusoidal', 'random']:
    try:
        vis_idx[t] = next(i for i,tp in enumerate(type_te) if tp == t)
    except StopIteration:
        vis_idx[t] = 0


# ════════════════════════════════════════════════════════════
# LOAD TRAINED MODELS
# ════════════════════════════════════════════════════════════

print("\n[Loading trained models]")

# ── Matched Filter (physics-based, no training) ──────────────
print("  Building Matched Filter...")
# mf = ExtendedMatchedFilter(array_pos, C, grid_points=35)
mf = ExtendedMatchedFilter(array_pos=array_pos, C=C,
                           k=2*np.pi*F0/3e8, sigma_f_phys=1.0,
                           a_x=0.5, a_y=0.5, beta_min=0.1, phi_max=np.pi/4, z_f=Z_F0)
print("  Matched Filter ready.")

# ── MLP ──────────────────────────────────────────────────────
mlp = MLP(lr=LR_SIGNAL, seed=RANDOM_SEED)
mlp_ckpt = os.path.join(CKPT_DIR, 'mlp.pkl')
assert os.path.exists(mlp_ckpt), \
    f"MLP checkpoint not found: {mlp_ckpt}\nRun 02_train_models.py first."
mlp.load(mlp_ckpt)
print("  MLP loaded.")

# ── Curriculum PINN ──────────────────────────────────────────
from config import T_WARM, T_RAMP, LAM_MAX, LR_DEFAULT
cpinn = CurriculumPINN(array_pos=array_pos, C=C,
                        lr=LR_DEFAULT, seed=RANDOM_SEED,
                        T_warm=T_WARM, T_ramp=T_RAMP, lam_max=LAM_MAX)
cp_ckpt = os.path.join(CKPT_DIR, 'cpinn.pkl')
assert os.path.exists(cp_ckpt), \
    f"CPINN checkpoint not found: {cp_ckpt}\nRun 02_train_models.py first."
cpinn.load(cp_ckpt)
print("  Curriculum PINN loaded.")

# ── CNN-PINN ─────────────────────────────────────────────────
cnn = CNNPINN(array_pos=array_pos, C=C,
               lr=LR_DEFAULT, seed=RANDOM_SEED,
               T_warm=T_WARM, T_ramp=T_RAMP, lam_max=LAM_MAX)
cnn_ckpt = os.path.join(CKPT_DIR, 'cnn_pinn.pkl')
assert os.path.exists(cnn_ckpt), \
    f"CNN-PINN checkpoint not found: {cnn_ckpt}\nRun 02_train_models.py first."
cnn.load(cnn_ckpt)
print("  CNN-PINN loaded.")

# ── Signal-Based (self-supervised, train here) ───────────────
sig_ckpt = os.path.join(CKPT_DIR, 'signal_based.pkl')
sig = SignalBasedModel(array_pos=array_pos, C=C,
                        lr=LR_SIGNAL, seed=RANDOM_SEED)
if os.path.exists(sig_ckpt):
    sig.load(sig_ckpt)
    print("  Signal-Based loaded from checkpoint.")
else:
    print("  Signal-Based: training now (self-supervised, no labels)...")
    sig.train(Y_tr, epochs=EPOCHS_SIGNAL,
               batch=BATCH_FC, verbose=True, warmstart=True)
    sig.save(sig_ckpt)
    print(f"  Signal-Based saved → {sig_ckpt}")


# ════════════════════════════════════════════════════════════
# EVALUATE ALL METHODS ON TEST SET
# ════════════════════════════════════════════════════════════

print("\n[Evaluating on test set]")

models = {
    'Matched Filter':  mf,
    'MLP':             mlp,
    'Curriculum PINN': cpinn,
    'CNN-PINN':        cnn
}

# Overall RMSE on full stacked test set
overall_rmse = {}
for name, model in models.items():
    pred = model.predict(Y_te) if not isinstance(model, ExtendedMatchedFilter) \
           else model.locate_trajectory(Y_te)
    r    = rmse_mm(pred, T_te[:, :2])
    overall_rmse[name] = r
    print(f"  {name:22s}: {r:.3f} mm")

# RMSE per trajectory type
rmse_by_type = {name: {} for name in models}
for tname, model in models.items():
    for ttype in ['linear', 'sinusoidal', 'random']:
        idx_list = [i for i, t in enumerate(type_te) if t == ttype]
        if not idx_list:
            continue
        Yc = np.concatenate([Y_te_list[i] for i in idx_list])
        Tc = np.concatenate([T_te_list[i] for i in idx_list])
        pred = (model.predict(Yc)
                if not isinstance(model, ExtendedMatchedFilter)
                else model.locate_trajectory(Yc))
        rmse_by_type[tname][ttype] = rmse_mm(pred, Tc[:, :2])


# ════════════════════════════════════════════════════════════
# SAVE RESULTS TABLE
# ════════════════════════════════════════════════════════════

csv_path = os.path.join(TABLE_DIR, 'comparison_results.csv')
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'Overall_RMSE_mm',
                     'Linear_mm', 'Sinusoidal_mm', 'Random_mm',
                     'Supervised'])
    supervised = {'Matched Filter': 'No (physics)',
                  'MLP': 'Yes', 'Curriculum PINN': 'Yes',
                  'CNN-PINN': 'Yes', 'Signal-Based': 'No (self-sup)'}
    for name in models:
        writer.writerow([
            name,
            f'{overall_rmse[name]:.4f}',
            f'{rmse_by_type[name].get("linear", float("nan")):.4f}',
            f'{rmse_by_type[name].get("sinusoidal", float("nan")):.4f}',
            f'{rmse_by_type[name].get("random", float("nan")):.4f}',
            supervised[name],
        ])
print(f"\n  Table saved: {csv_path}")


# ════════════════════════════════════════════════════════════
# FIGURE 1 — RMSE BAR CHART
# ════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(10, 5))
names = list(overall_rmse.keys())
vals  = list(overall_rmse.values())
cols  = [COLORS[n] for n in names]
bars  = ax.bar(names, vals, color=cols, edgecolor='white', lw=1.5)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.01,
            f'{v:.3f}', ha='center', va='bottom',
            fontsize=10, fontweight='bold')

# Supervision annotation
y_top = max(vals) * 1.25
for bar, name in zip(bars, names):
    label = '(physics)' if name == 'Matched Filter' \
            else '(self-sup)' if name == 'Signal-Based' \
            else '(supervised)'
    ax.text(bar.get_x()+bar.get_width()/2, y_top * 0.02,
            label, ha='center', va='bottom',
            fontsize=8, color='#555', rotation=0)

ax.set_ylabel('RMSE [mm]', fontsize=12)
ax.set_title('Localisation RMSE — All Methods\n'
             '(Extended Finger Model, 7×7 Array, 25 dB SNR)',
             fontsize=13, fontweight='bold')
ax.set_ylim(0, y_top)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '01_rmse_bar.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_rmse_bar.png")


# ════════════════════════════════════════════════════════════
# FIGURE 2 — TRAJECTORY COMPARISON
# ════════════════════════════════════════════════════════════

from core.trajectory import screen_bounds as sb
from config import SX, SY
x_min, x_max, y_min, y_max = sb(SX, SY, array_pos[1,0]-array_pos[0,0]
                                  if array_pos[1,0] != array_pos[0,0]
                                  else array_pos[SX,1]-array_pos[0,1])

# Use first test trajectory for visualisation
Y_vis = Y_te_list[vis_idx.get('linear', 0)]
T_vis = T_te_list[vis_idx.get('linear', 0)]

n_methods = len(models)
fig, axes = plt.subplots(1, n_methods+1, figsize=(4*(n_methods+1), 4.5))

# Ground truth
ax0 = axes[0]
ax0.plot(T_vis[:,0]*1e3, T_vis[:,1]*1e3, 'k-', lw=2, label='True')
ax0.scatter(*T_vis[0,:2]*1e3,  color='green', s=80, zorder=5)
ax0.scatter(*T_vis[-1,:2]*1e3, color='red',   s=80, marker='X', zorder=5)
ax0.set_title('Ground Truth', fontweight='bold')
ax0.set_xlabel('x [mm]'); ax0.set_ylabel('y [mm]')
ax0.set_aspect('equal'); ax0.grid(True, alpha=0.3)

for ax, (name, model) in zip(axes[1:], models.items()):
    pred = (model.predict(Y_vis)
            if not isinstance(model, ExtendedMatchedFilter)
            else model.locate_trajectory(Y_vis))
    r    = rmse_mm(pred, T_vis[:, :2])
    ax.plot(T_vis[:,0]*1e3, T_vis[:,1]*1e3,
            'k--', alpha=0.3, lw=1)
    ax.plot(pred[:,0]*1e3, pred[:,1]*1e3,
            color=COLORS[name], lw=1.8)
    ax.scatter(*T_vis[0,:2]*1e3,  color='green', s=50, zorder=5)
    ax.scatter(*T_vis[-1,:2]*1e3, color='red',   s=50,
               marker='X', zorder=5)
    ax.set_title(f'{name}\nRMSE={r:.2f} mm', fontweight='bold',
                 color=COLORS[name], fontsize=9)
    ax.set_xlabel('x [mm]')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3)

plt.suptitle('Predicted vs True Trajectory — All Methods\n'
             '(Linear test trajectory)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '02_trajectories.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_trajectories.png")


# ════════════════════════════════════════════════════════════
# FIGURE 3 — PER-TIMESTEP ERROR OVER TIME
# ════════════════════════════════════════════════════════════

from core.finger import N as N_steps, dt as dt_val
t_axis = np.arange(len(Y_vis)) * dt_val * 1e3

fig, ax = plt.subplots(figsize=(13, 4.5))
for name, model in models.items():
    pred = (model.predict(Y_vis)
            if not isinstance(model, ExtendedMatchedFilter)
            else model.locate_trajectory(Y_vis))
    err  = per_timestep_error(pred, T_vis[:, :2])
    ax.plot(t_axis, err, color=COLORS[name], marker=MARKERS[name],
            markevery=20, lw=1.5, markersize=5, label=name, alpha=0.85)

ax.set_xlabel('Time [ms]', fontsize=11)
ax.set_ylabel('Position Error [mm]', fontsize=11)
ax.set_title('Per-Timestep Localisation Error — All Methods',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '03_error_over_time.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_error_over_time.png")


# ════════════════════════════════════════════════════════════
# FIGURE 4 — SPATIAL ERROR HEATMAP
# ════════════════════════════════════════════════════════════
# For each method: divide the screen into a grid, compute mean
# error in each cell across all test snapshots.
# Shows WHERE on the screen each method struggles.

GRID = 15   # 15×15 spatial bins

def spatial_error_map(model, Y_all, T_all, x_min, x_max, y_min, y_max):
    """Mean error in spatial bins across all test snapshots."""
    pred   = (model.predict(Y_all)
              if not isinstance(model, ExtendedMatchedFilter)
              else model.locate_trajectory(Y_all))
    err    = per_timestep_error(pred, T_all[:, :2])  # (N,)
    true_x = T_all[:, 0]; true_y = T_all[:, 1]

    xbins = np.linspace(x_min, x_max, GRID+1)
    ybins = np.linspace(y_min, y_max, GRID+1)
    hmap  = np.full((GRID, GRID), np.nan)
    counts= np.zeros((GRID, GRID))

    for i in range(GRID):
        for j in range(GRID):
            mask = ((true_x >= xbins[i]) & (true_x < xbins[i+1]) &
                    (true_y >= ybins[j]) & (true_y < ybins[j+1]))
            if mask.sum() > 0:
                hmap[j, i]   = err[mask].mean()
                counts[j, i] = mask.sum()

    return hmap, xbins, ybins

# Stack all test data
Y_te_cat = np.concatenate(Y_te_list)
T_te_cat = np.concatenate(T_te_list)

# Use core de for screen bounds
from core.finger import de as de_val
xmn, xmx, ymn, ymx = sb(SX, SY, de_val)

fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4.5))
for ax, (name, model) in zip(axes, models.items()):
    hmap, xb, yb = spatial_error_map(
        model, Y_te_cat, T_te_cat, xmn, xmx, ymn, ymx
    )
    vmax = np.nanpercentile(hmap, 95)
    im = ax.imshow(hmap, cmap='RdYlGn_r', origin='lower',
                   extent=[xmn*1e3, xmx*1e3, ymn*1e3, ymx*1e3],
                   aspect='equal', vmin=0, vmax=vmax)
    ax.set_title(f'{name}\n(mean {overall_rmse[name]:.2f} mm)',
                 fontweight='bold', fontsize=9, color=COLORS[name])
    ax.set_xlabel('x [mm]')
    if ax == axes[0]:
        ax.set_ylabel('y [mm]')
    plt.colorbar(im, ax=ax, label='Error [mm]', fraction=0.046)

plt.suptitle('Spatial Error Map — Where Does Each Method Struggle?\n'
             '(mean position error per screen region)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '04_error_heatmap.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_error_heatmap.png")


# ════════════════════════════════════════════════════════════
# FIGURE 5 — RMSE BY TRAJECTORY TYPE
# ════════════════════════════════════════════════════════════

traj_types = ['linear', 'sinusoidal', 'random']
x_pos  = np.arange(len(traj_types))
width  = 0.15
n_m    = len(models)

fig, ax = plt.subplots(figsize=(11, 5))
for i, (name, model) in enumerate(models.items()):
    vals_by_type = [rmse_by_type[name].get(t, 0) for t in traj_types]
    offset = (i - n_m/2 + 0.5) * width
    bars = ax.bar(x_pos + offset, vals_by_type,
                  width=width*0.9, color=COLORS[name],
                  label=name, alpha=0.85)

ax.set_xticks(x_pos)
ax.set_xticklabels(['Linear', 'Sinusoidal', 'Random Walk'], fontsize=11)
ax.set_ylabel('RMSE [mm]', fontsize=11)
ax.set_title('RMSE by Trajectory Type — All Methods',
             fontsize=13, fontweight='bold')
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, '05_rmse_per_traj_type.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_rmse_per_traj_type.png")


# ════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════

print("\n" + "="*65)
print("  COMPARISON RESULTS")
print("="*65)
for name, rmse in overall_rmse.items():
    sup = "(physics-based)" if name == 'Matched Filter' \
          else "(self-supervised)" if name == 'Signal-Based' \
          else "(supervised)"
    print(f"  {name:22s}: {rmse:.3f} mm  {sup}")

print(f"\n  Table: {csv_path}")
print(f"  Figures: {FIG_DIR}")
print("="*65)