"""
data/load.py
============
Thin wrapper around pickle.load so experiment scripts never
deal with file paths or pickle directly.

All functions load from data/standard.pkl relative to the
project root (one level above this file).
"""

import os, pickle
import numpy as np

_HERE     = os.path.dirname(os.path.abspath(__file__))
_STD_PATH = os.path.join(_HERE, 'standard.pkl')


def _load_raw():
    assert os.path.exists(_STD_PATH), (
        f"Dataset not found: {_STD_PATH}\n"
        f"Run  python data/generate.py  first."
    )
    with open(_STD_PATH, 'rb') as f:
        return pickle.load(f)


def load_standard(return_array=False, return_all=False):
    """
    Load the standard dataset.

    Parameters
    ----------
    return_array : bool
        If True, also return (array_pos, C).
    return_all : bool
        If True, return everything including per-trajectory lists
        and trajectory type labels. Implies return_array=True.

    Returns
    -------
    Default (return_array=False, return_all=False):
        Y_tr, T_tr, Y_te, T_te

    With return_array=True:
        Y_tr, T_tr, Y_te, T_te, array_pos, C

    With return_all=True:
        Y_tr, T_tr, Y_te, T_te,
        array_pos, C,
        Y_list, T_list, type_list,
        n_train
    """
    d      = _load_raw()
    Y_list = d['Y']
    T_list = d['T']
    nt     = d['n_train']

    Y_tr = np.concatenate(Y_list[:nt])
    T_tr = np.concatenate(T_list[:nt])
    Y_te = np.concatenate(Y_list[nt:])
    T_te = np.concatenate(T_list[nt:])

    if return_all:
        return (Y_tr, T_tr, Y_te, T_te,
                d['array_pos'], d['C'],
                Y_list, T_list, d['type'],
                nt)
    elif return_array:
        return Y_tr, T_tr, Y_te, T_te, d['array_pos'], d['C']
    else:
        return Y_tr, T_tr, Y_te, T_te


def load_metadata():
    """Return the metadata dict from the dataset."""
    return _load_raw()['meta']


def load_noise_sweep():
    path = os.path.join(_HERE, 'noise_sweep.pkl')
    assert os.path.exists(path), (
        f"Noise sweep dataset not found: {path}\n"
        f"Run  python data/generate.py  first."
    )
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_array_sweep():
    path = os.path.join(_HERE, 'array_sweep.pkl')
    assert os.path.exists(path), (
        f"Array sweep dataset not found: {path}\n"
        f"Run  python data/generate.py  first."
    )
    with open(path, 'rb') as f:
        return pickle.load(f)