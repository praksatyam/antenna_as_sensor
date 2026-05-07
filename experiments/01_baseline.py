# experiments/01_baseline.py
"""
Experiment 1: Baseline comparison of all four methods.
Loads pre-generated dataset, trains all methods, saves results.
"""
import sys, pickle, csv
sys.path.insert(0, '..')

from config import *
from data.load import load_standard          # thin wrapper around pickle.load
from localisation.mlp import MLP
from localisation.curriculum_pinn import CurriculumPINN as cpinn
from localisation.cnn_pinn import CNNPINN as cnn
from localisation.signal_based import SignalBasedModel as sig
from baselines.matched_filter import ExtendedMatchedFilter as mf
from core.array import build_array, build_coupling_matrix

# Load data (not regenerate)
array_pos = build_array(SX, SY)
C         = build_coupling_matrix(array_pos, EPSILON)
Y_tr, T_tr, Y_te, T_te = load_standard()

# Train
mlp   = MLP(lr=LR_DEFAULT)
mlp.train(Y_tr, T_tr, epochs=EPOCHS_MLP)
mlp.save('../checkpoints/mlp.pkl')          # save weights

# ... same for other methods ...

# Evaluate and save
results = {
    'MLP':            mlp.rmse(Y_te, T_te),
    'Curriculum PINN': cpinn.rmse(Y_te, T_te),
    'CNN-PINN':        cnn.rmse(Y_te, T_te),
    'Signal-Based':    sig.rmse(Y_te, T_te),
}

# Save to CSV for the paper
with open('../results/tables/baseline.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'RMSE_mm'])
    for name, rmse in results.items():
        writer.writerow([name, f'{rmse:.4f}'])