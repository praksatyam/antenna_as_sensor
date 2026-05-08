
import sys, os, pickle, time
import numpy as np
import matplotlib.pyplot as plt

# ── Path setup ───────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import (
    EPOCHS_MLP, EPOCHS_CPINN, EPOCHS_CNN,
    BATCH_FC, BATCH_CNN,
    LR_DEFAULT, T_WARM, T_RAMP, LAM_MAX,
    RANDOM_SEED
)
from data.load import load_standard
from localisation.mlp import MLP
from localisation.curriculum_pinn import CurriculumPINN
from localisation.cnn_pinn import CNNPINN

# ── Directories ──────────────────────────────────────────────
CKPT_DIR = os.path.join(_ROOT, 'checkpoints')
FIG_DIR  = os.path.join(_ROOT, 'results', 'figures', 'training')
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

np.random.seed(RANDOM_SEED)

plt.rcParams.update({
    'figure.dpi': 130, 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def load_or_train(name, model_fn, train_fn, ckpt_path):
    model = model_fn()

    if os.path.exists(ckpt_path):
        model.load(ckpt_path)
        print(f"  [LOADED]  {name} ← {ckpt_path}")
        return model, None, True

    print(f"  [TRAIN]   {name}...")
    t0      = time.time()
    history = train_fn(model)
    elapsed = time.time() - t0
    model.save(ckpt_path)           # ← uses your class's save() method
    print(f"  [SAVED]   {name} → {ckpt_path}  ({elapsed:.1f}s)")
    return model, history, False


def plot_training_curves(histories, save_path):
    """
    Plot training loss curves for all three models.
    Each model gets its own subplot so loss scales don't interfere.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    configs = [
        ('MLP',            '#457b9d', histories.get('MLP')),
        ('Curriculum PINN','#2a9d8f', histories.get('Curriculum PINN')),
        ('CNN-PINN',        '#f4a261', histories.get('CNN-PINN')),
    ]

    for ax, (name, color, hist) in zip(axes, configs):
        if hist is None:
            ax.text(0.5, 0.5, 'Loaded from checkpoint\n(no training history)',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=11, color='#888')
            ax.set_title(name, fontweight='bold')
            continue

        if isinstance(hist, dict):
            if 'pos' in hist:
                ax.semilogy(hist['pos'],  color=color, lw=2,
                            label='L_pos')
            if 'phys' in hist and any(v > 0 for v in hist['phys']):
                ax.semilogy(hist['phys'], color=color, lw=1.5,
                            linestyle='--', alpha=0.7, label='L_phys')
            if 'lam' in hist:
                ax2 = ax.twinx()
                ax2.plot(hist['lam'], color='#888', lw=1,
                         linestyle=':', alpha=0.6, label='λ(t)')
                ax2.set_ylabel('λ', fontsize=9, color='#888')
                ax2.legend(fontsize=8, loc='lower right')
        else:
            ax.semilogy(hist, color=color, lw=2, label='L_pos')

        ax.set_title(name, fontweight='bold', color=color)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (log scale)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Loss Curves', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ════════════════════════════════════════════════════════════
# LOAD DATASET
# ════════════════════════════════════════════════════════════

print("=" * 60)
print("  ScreenAnt — Model Training")
print("=" * 60)

print("\n[Loading dataset]")
Y_tr, T_tr, Y_te, T_te, array_pos, C = load_standard(return_array=True)
print(f"  Train: {Y_tr.shape}  Test: {Y_te.shape}")
print(f"  Array elements: {Y_tr.shape[1]}")


# ════════════════════════════════════════════════════════════
# TRAIN / LOAD MODELS
# ════════════════════════════════════════════════════════════

print("\n[Training models — delete checkpoint to force retrain]")

histories = {}

# ── MLP ──────────────────────────────────────────────────────
mlp, h_mlp, loaded = load_or_train(
    name      = 'MLP',
    model_fn  = lambda: MLP(lr=LR_DEFAULT, seed=RANDOM_SEED),
    train_fn  = lambda m: m.train(
                    Y_tr, T_tr,
                    epochs=EPOCHS_MLP,
                    batch=BATCH_FC,
                    verbose=True),
    ckpt_path = os.path.join(CKPT_DIR, 'mlp.pkl'),
)
if not loaded:
    histories['MLP'] = h_mlp

# ── Curriculum PINN ──────────────────────────────────────────
cpinn, h_cpinn, loaded = load_or_train(
    name      = 'Curriculum PINN',
    model_fn  = lambda: CurriculumPINN(
                    array_pos=array_pos, C=C,
                    lr=LR_DEFAULT, seed=RANDOM_SEED,
                    T_warm=T_WARM, T_ramp=T_RAMP,
                    lam_max=LAM_MAX),
    train_fn  = lambda m: m.train(
                    Y_tr, T_tr,
                    epochs=EPOCHS_CPINN,
                    batch=BATCH_FC,
                    verbose=True),
    ckpt_path = os.path.join(CKPT_DIR, 'cpinn.pkl'),
)
if not loaded:
    histories['Curriculum PINN'] = h_cpinn


cnn, h_cnn, loaded = load_or_train(
    name      = 'CNN-PINN',
    model_fn  = lambda: CNNPINN(
                    array_pos=array_pos, C=C,
                    lr=LR_DEFAULT, seed=RANDOM_SEED,
                    T_warm=T_WARM, T_ramp=T_RAMP,
                    lam_max=LAM_MAX),
    train_fn  = lambda m: m.train(
                    Y_tr, T_tr,
                    epochs=EPOCHS_CNN,
                    batch=BATCH_CNN,
                    verbose=True),
    ckpt_path = os.path.join(CKPT_DIR, 'cnn_pinn.pkl'),
)

if not loaded:
    histories['CNN-PINN'] = h_cnn


# ════════════════════════════════════════════════════════════
# TRAINING CURVES
# ════════════════════════════════════════════════════════════

if histories:
    plot_training_curves(
        histories,
        save_path=os.path.join(FIG_DIR, 'training_curves.png')
    )
else:
    print("  All models loaded from checkpoints — no training curves to plot.")


# ════════════════════════════════════════════════════════════
# QUICK VALIDATION ON TEST SET
# ════════════════════════════════════════════════════════════

print("\n[Quick validation on test set]")

def rmse_mm(pred, true):
    return float(np.sqrt(np.mean(np.sum((pred - true)**2, axis=1)))) * 1e3

# Use the last test trajectory for visualisation
Y_vis = Y_te[:200]   # first 200 snapshots of the stacked test set
T_vis = T_te[:200]

for name, model in [('MLP', mlp), ('Curriculum PINN', cpinn),
                     ('CNN-PINN', cnn)]:
    pred = model.predict(Y_vis)
    rmse = rmse_mm(pred, T_vis[:, :2])
    print(f"  {name:20s}: {rmse:.3f} mm  (on first 200 test snapshots)")

print("\nDone. Models saved to checkpoints/")
print("Next: run experiments/03_compare.py")