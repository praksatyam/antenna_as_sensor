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

array_pos = build_array(SX, SY)
C         = build_coupling_matrix(array_pos, EPSILON)

# Standard dataset — used by all baseline experiments
print("Generating standard dataset...")
Y_list, T_list = make_dataset(
    array_pos, C, n_traj=N_TRAJ_TRAIN+N_TRAJ_TEST,
    snr_db=TRAIN_SNR_DB, seed=RANDOM_SEED
)
with open('standard.pkl', 'wb') as f:
    pickle.dump({'Y': Y_list, 'T': T_list,
                 'n_train': N_TRAJ_TRAIN, 'n_test': N_TRAJ_TEST}, f)
print(f"  Saved standard.pkl")

# Noise sweep datasets — one per SNR level
print("Generating noise sweep datasets...")
noise_data = {}
for snr in SNR_RANGE_DB:
    Yl, Tl = make_dataset(array_pos, C, n_traj=10,
                           snr_db=snr, seed=RANDOM_SEED+1)
    noise_data[snr] = {'Y': Yl, 'T': Tl}
    print(f"  SNR={snr} dB done")
with open('noise_sweep.pkl', 'wb') as f:
    pickle.dump(noise_data, f)

print("All datasets saved.")