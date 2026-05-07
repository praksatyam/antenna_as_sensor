# data/load.py
import pickle, numpy as np

def load_standard():
    with open('data/standard.pkl', 'rb') as f:
        d = pickle.load(f)
    Y, T = d['Y'], d['T']
    nt = d['n_train']
    Y_tr = np.concatenate(Y[:nt]);  T_tr = np.concatenate(T[:nt])
    Y_te = np.concatenate(Y[nt:]);  T_te = np.concatenate(T[nt:])
    return Y_tr, T_tr, Y_te, T_te

def load_noise_sweep():
    with open('data/noise_sweep.pkl', 'rb') as f:
        return pickle.load(f)

def load_array_sweep():
    with open('data/array_sweep.pkl', 'rb') as f:
        return pickle.load(f)