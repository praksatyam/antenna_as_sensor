# config.py — all hyperparameters in one place

# ── Physical system ──────────────────────────────────────────
F0          = 28e9          # carrier frequency [Hz]
SX, SY      = 7, 7          # array dimensions
EPSILON     = 0.3           # coupling strength
Z_F0        = 0.010         # finger height [m]
SIGMA_N2    = 1e-6          # noise variance

# ── Dataset ──────────────────────────────────────────────────
N_TRAJ_TRAIN    = 30        # training trajectories
N_TRAJ_TEST     = 10        # test trajectories
TRAIN_SNR_DB    = 25.0      # reference SNR for training data
RANDOM_SEED     = 42        # global seed for reproducibility

# ── Training ─────────────────────────────────────────────────
EPOCHS_MLP      = 300
EPOCHS_CPINN    = 300
EPOCHS_CNN      = 300
EPOCHS_SIGNAL   = 300
BATCH_FC        = 64
BATCH_CNN       = 32
LR_DEFAULT      = 3e-3
LR_SIGNAL       = 2e-3

# ── Curriculum schedule ──────────────────────────────────────
T_WARM          = 80        # warmup epochs before physics activates
T_RAMP          = 60        # ramp epochs for λ
LAM_MAX         = 0.005     # maximum physics loss weight

# ── Sweep ranges ─────────────────────────────────────────────
SNR_RANGE_DB    = [0, 5, 10, 15, 20, 25, 30, 35, 40]
ARRAY_SIZES     = [3, 5, 7, 9, 11]
DATA_SCARCITY_N = [100, 300, 600, 1000, 2000, 4000, 6000]