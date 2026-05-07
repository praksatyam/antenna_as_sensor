import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, time, pickle

from ..core.finger import *
from ..core.array import *
from ..core.trajectory import *

def rmse_mm(pred, true):
    """Root mean squared 2D position error [mm]."""
    return float(np.sqrt(np.mean(np.sum((pred - true)**2, axis=1)))) * 1e3


def get_bounds():
    return screen_bounds(Sx, Sy, de)   # (x_min, x_max, y_min, y_max)


def norm_pos(pos, bounds):
    """Normalise positions to [-1, 1]."""
    p = pos.copy()
    p[:, 0] = 2*(pos[:, 0]-bounds[0])/(bounds[1]-bounds[0]) - 1
    p[:, 1] = 2*(pos[:, 1]-bounds[2])/(bounds[3]-bounds[2]) - 1
    return p


def denorm_pos(p, bounds):
    """Inverse of norm_pos."""
    out = p.copy()
    out[:, 0] = (p[:, 0]+1)/2*(bounds[1]-bounds[0]) + bounds[0]
    out[:, 1] = (p[:, 1]+1)/2*(bounds[3]-bounds[2]) + bounds[2]
    return out


def prepare_flat(Y):
    """
    Flat feature vector: [Re(y), Im(y)] / scale  shape (N, 2S).
    Used by MLP, Curriculum PINN, Signal-Based.
    """
    X = np.concatenate([np.real(Y), np.imag(Y)], axis=1)
    return X / (np.mean(np.abs(X)) + 1e-12)


def prepare_spatial(Y):
    """
    Spatial feature tensor: shape (N, 2, Sx, Sy).
    Channel 0 = Re(y) reshaped to 7×7.
    Channel 1 = Im(y) reshaped to 7×7.
    Used by CNN-PINN.
    Each channel normalised independently.
    """
    N_ = len(Y)
    X  = np.zeros((N_, 2, Sx, Sy), dtype=np.float64)
    Re = np.real(Y)  # (N, S)
    Im = np.imag(Y)
    sc_re = np.mean(np.abs(Re)) + 1e-12
    sc_im = np.mean(np.abs(Im)) + 1e-12
    X[:, 0] = Re.reshape(N_, Sx, Sy) / sc_re
    X[:, 1] = Im.reshape(N_, Sx, Sy) / sc_im
    return X


# ── Forward model (vectorised over batch) ────────────────────

def batch_forward(positions, array_pos, C):
    """
    Extended forward model C·[Γ(p)⊙a(p)] for a batch of positions.

    positions : (B, 3) — finger positions [x, y, z]
    returns   : (B, S) complex template vectors
    """
    diff = array_pos[np.newaxis, :, :] - positions[:, np.newaxis, :]  # (B,S,3)
    r    = np.sqrt(np.sum(diff**2, axis=2))                             # (B,S)
    dx   = array_pos[np.newaxis, :, 0] - positions[:, np.newaxis, 0]
    dy   = array_pos[np.newaxis, :, 1] - positions[:, np.newaxis, 1]
    rho  = np.exp(-dx**2/(2*a_x**2) - dy**2/(2*a_y**2))               # (B,S)
    A    = sigma_f_phys * rho * np.exp(-1j*2*k*r) / (r**2)             # (B,S)
    beta = 1.0 - rho*(1.0 - beta_min)
    phi  = rho * phi_max
    gamma = beta * np.exp(1j*phi)
    return (gamma * A) @ C.T                                            # (B,S)


def physics_loss_and_grad(pred_norm, Y_obs, array_pos, C, bounds, h=1e-5):
    """
    Physics consistency loss and gradient, vectorised over the batch.

    L_phys = mean_b ||t(p̂_b) - y_obs_b||² / ȳ

    Gradient via central finite differences in (x, y).
    Returns (scalar loss, (B,2) gradient in normalised coordinates).
    """
    B     = len(pred_norm)
    p_hat = denorm_pos(pred_norm, bounds)
    pos3d = np.concatenate([p_hat, np.full((B, 1), z_f0)], axis=1)
    ys    = np.mean(np.abs(Y_obs)) + 1e-12

    T0 = batch_forward(pos3d, array_pos, C)
    R0 = (T0 - Y_obs) / ys
    L0 = np.real(np.sum(R0 * R0.conj(), axis=1))           # (B,)

    def _shifted(dim, sign):
        p = pos3d.copy(); p[:, dim] += sign * h
        R = (batch_forward(p, array_pos, C) - Y_obs) / ys
        return np.real(np.sum(R * R.conj(), axis=1))

    dLdx = (_shifted(0, +1) - _shifted(0, -1)) / (2*h)
    dLdy = (_shifted(1, +1) - _shifted(1, -1)) / (2*h)

    phys_loss = L0.mean()

    grad = np.zeros((B, 2))
    grad[:, 0] = dLdx * (bounds[1]-bounds[0])/2 / B
    grad[:, 1] = dLdy * (bounds[3]-bounds[2])/2 / B
    return phys_loss, grad


# ── Adam optimiser state ─────────────────────────────────────

class Adam:
    """Minimal Adam optimiser for numpy parameter dicts."""
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr  = lr
        self.b1  = b1; self.b2 = b2; self.eps = eps
        self.t   = 0
        self.m   = {k: np.zeros_like(v) for k, v in params.items()}
        self.v   = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for key in params:
            self.m[key] = self.b1*self.m[key] + (1-self.b1)*grads[key]
            self.v[key] = self.b2*self.v[key] + (1-self.b2)*grads[key]**2
            mh = self.m[key] / (1 - self.b1**self.t)
            vh = self.v[key] / (1 - self.b2**self.t)
            params[key] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ── Leaky ReLU ───────────────────────────────────────────────

def lrelu(x, a=0.01):
    return np.where(x > 0, x, a*x)

def lrelu_d(x, a=0.01):
    return np.where(x > 0, 1.0, a)


# ══════════════════════════════════════════════════════════════
# SHARED FC HEAD  (used by all four models after encoding)
# ══════════════════════════════════════════════════════════════

def init_fc(d_in, h1=128, h2=64, seed=0):
    """
    Initialise FC(d_in→h1→h2→2) weights.
    He initialisation for LeakyReLU.
    Returns a parameter dict.
    """
    rng = np.random.default_rng(seed)
    return {
        'W1': rng.normal(0, np.sqrt(2/d_in),  (d_in, h1)),  'b1': np.zeros(h1),
        'W2': rng.normal(0, np.sqrt(2/h1),    (h1,   h2)),  'b2': np.zeros(h2),
        'W3': rng.normal(0, np.sqrt(2/h2),    (h2,    2)),  'b3': np.zeros(2),
    }


def fc_forward(X, p):
    """Forward pass through FC head. Returns (out, cache)."""
    z1 = X  @ p['W1'] + p['b1'];  a1 = lrelu(z1)
    z2 = a1 @ p['W2'] + p['b2'];  a2 = lrelu(z2)
    out= a2 @ p['W3'] + p['b3']
    return out, {'X': X, 'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}


def fc_backward(g, cache, p):
    """Backprop through FC head. Returns grad dict."""
    B  = g.shape[0]
    dW3 = cache['a2'].T @ g / B;  db3 = g.mean(0)
    da2 = g @ p['W3'].T
    dz2 = da2 * lrelu_d(cache['z2'])
    dW2 = cache['a1'].T @ dz2 / B; db2 = dz2.mean(0)
    da1 = dz2 @ p['W2'].T
    dz1 = da1 * lrelu_d(cache['z1'])
    dW1 = cache['X'].T @ dz1 / B;  db1 = dz1.mean(0)
    return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2,
            'W3': dW3, 'b3': db3}


# ══════════════════════════════════════════════════════════════
# CNN ENCODER   (used by Method 3 — CNN-PINN)
# ══════════════════════════════════════════════════════════════

def init_cnn(seed=0):
    """
    Two-layer CNN encoder for 7×7×2 input.

    Architecture:
      Conv(2→16, 3×3, pad=1) → LReLU    output: (16, 7, 7)
      Conv(16→32, 3×3, pad=1) → LReLU   output: (32, 7, 7)
      Flatten                            output: 32×7×7 = 1568

    The 3×3 kernels capture local spatial gradients in the Re/Im
    channels — the dominant structure from the Gaussian footprint
    and radial phase pattern.
    Padding=1 preserves spatial dimensions so the 7×7 grid is
    not reduced (important for a small 7-element grid).
    """
    rng  = np.random.default_rng(seed)
    # Kernel shape: (out_channels, in_channels, kH, kW)
    k1   = rng.normal(0, np.sqrt(2/(2*3*3)),   (16, 2,  3, 3))
    b1   = np.zeros(16)
    k2   = rng.normal(0, np.sqrt(2/(16*3*3)),  (32, 16, 3, 3))
    b2   = np.zeros(32)
    return {'k1': k1, 'b1': b1, 'k2': k2, 'b2': b2}


def conv2d_pad1(X, K, b):
    """
    2-D convolution with same-padding (pad=1) and stride=1.
    Pure NumPy implementation.

    X : (N, C_in, H, W)
    K : (C_out, C_in, kH, kW)   kH=kW=3
    b : (C_out,)
    returns (N, C_out, H, W)
    """
    N_, Cin, H, W = X.shape
    Cout, _, kH, kW = K.shape
    Xp = np.pad(X, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    out = np.zeros((N_, Cout, H, W))
    for i in range(H):
        for j in range(W):
            patch = Xp[:, :, i:i+kH, j:j+kW]       # (N, Cin, 3, 3)
            # patch reshaped: (N, Cin*9), K reshaped: (Cout, Cin*9)
            out[:, :, i, j] = patch.reshape(N_, -1) @ K.reshape(Cout, -1).T
    out += b[np.newaxis, :, np.newaxis, np.newaxis]
    return out


def conv2d_pad1_backward(dout, X, K, cache_Xp):
    """
    Backprop through conv2d_pad1.
    Returns (dX, dK, db).
    """
    N_, Cin, H, W   = X.shape
    Cout, _, kH, kW = K.shape
    Xp   = cache_Xp
    dXp  = np.zeros_like(Xp)
    dK   = np.zeros_like(K)
    db   = dout.sum(axis=(0, 2, 3))

    for i in range(H):
        for j in range(W):
            patch = Xp[:, :, i:i+kH, j:j+kW].reshape(N_, -1)  # (N, Cin*9)
            g     = dout[:, :, i, j]                             # (N, Cout)
            dK   += (g.T @ patch).reshape(Cout, Cin, kH, kW)
            dpatch = (g @ K.reshape(Cout, -1)).reshape(N_, Cin, kH, kW)
            dXp[:, :, i:i+kH, j:j+kW] += dpatch

    dX = dXp[:, :, 1:-1, 1:-1]   # remove padding
    return dX, dK, db


def cnn_forward(X_spatial, cp):
    """
    CNN encoder forward pass.
    X_spatial : (N, 2, 7, 7)
    Returns (flat_features (N,1568), cache)
    """
    Xp1  = np.pad(X_spatial, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    c1   = conv2d_pad1(X_spatial, cp['k1'], cp['b1'])    # (N,16,7,7)
    a1   = lrelu(c1)
    Xp2  = np.pad(a1, ((0,0),(0,0),(1,1),(1,1)), mode='constant')
    c2   = conv2d_pad1(a1,       cp['k2'], cp['b2'])     # (N,32,7,7)
    a2   = lrelu(c2)
    flat = a2.reshape(len(X_spatial), -1)                # (N,1568)
    cache = {'X': X_spatial, 'Xp1': Xp1, 'c1': c1, 'a1': a1,
             'Xp2': Xp2,     'c2': c2, 'a2': a2}
    return flat, cache


def cnn_backward(d_flat, cache, cp):
    """CNN encoder backward pass. Returns grad dict for cp."""
    N_ = len(d_flat)
    da2  = d_flat.reshape(cache['a2'].shape)
    dc2  = da2 * lrelu_d(cache['c2'])
    da1, dk2, db2 = conv2d_pad1_backward(dc2, cache['a1'], cp['k2'], cache['Xp2'])
    dc1  = da1 * lrelu_d(cache['c1'])
    _, dk1, db1 = conv2d_pad1_backward(dc1, cache['X'], cp['k1'], cache['Xp1'])
    return {'k1': dk1, 'b1': db1, 'k2': dk2, 'b2': db2}

