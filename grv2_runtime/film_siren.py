"""grv2_runtime/film_siren.py -- psi_neural: S^{k-1} -> C1(R3, R).

Ported verbatim (minus the __main__ smoke test and the /home/claude path
hack) from /home/jg/film_siren.py, part of the same "Semantic Crystal
Engine" backend verified this session (26/26 of its own tests pass once
scikit-learn/hnswlib are installed).

A FiLM-conditioned SIREN deformation field over a semantic-vector-selected
base primitive, plus an EDT eikonal correction that guarantees a genuine
zero-crossing signed-distance field for *any* z -- deterministically seeded
so the same z always produces the same field (no unseeded randomness).

grv2_runtime.wiring.WiringBank uses this as its last-resort tier: any word
that isn't in GCODE_LIBRARY and doesn't match anything in atlas_csg.ATLAS
gets a real, per-word-deterministic sculpted shape from here instead of
silently collapsing into the generic "default" blob.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple

from . import atlas_csg

SIREN_OMEGA = 30.0
HIDDEN_DIM = 32
N_LAYERS = 4
DEFORM_ALPHA = 0.30


# ══════════════════════════════════════════════════════════════
# 1. BASE PRIMITIVE SELECTOR -- deterministic from z, no randomness
# ══════════════════════════════════════════════════════════════

def select_base_primitive(z: np.ndarray) -> dict:
    k = len(z)
    family = int(np.argmax(np.abs(z[:4])))
    r = 0.4 + 0.35 * abs(float(z[0]))

    if family == 0:
        a = r * (1 + 1.5 * max(0.0, float(z[1])))
        b = r * (1 + 1.5 * max(0.0, float(-z[1])))
        return {'kind': 'ellipsoid', 'c': [0, 0, 0], 'abc': [a, r, b]}
    elif family == 1:
        h = r * (1 + 1.5 * abs(float(z[2])))
        return {'kind': 'capsule', 'a': [0, -h, 0], 'b': [0, h, 0], 'r': r}
    elif family == 2:
        a = r * (1 + 0.8 * float(z[3]))
        b = r * (1 + 0.8 * float(z[4 % k]))
        c = r * (1 + 0.8 * float(z[5 % k]))
        return {'kind': 'ellipsoid', 'c': [0, 0, 0], 'abc': [abs(a), abs(b), abs(c)]}
    else:
        R = 0.5 + 0.2 * abs(float(z[3 % k]))
        rt = 0.15 + 0.1 * abs(float(z[4 % k]))
        return {'kind': 'torus', 'c': [0, 0, 0], 'R': R, 'r': rt}


def eval_base_primitive(pts: np.ndarray, prim: dict) -> np.ndarray:
    kind = prim['kind']
    p = prim
    if kind == 'sphere':
        return atlas_csg.sdf_sphere(pts, np.array(p['c']), p['r'])
    elif kind == 'ellipsoid':
        return atlas_csg.sdf_ellipsoid(pts, np.array(p['c']), np.array(p['abc']))
    elif kind == 'capsule':
        return atlas_csg.sdf_capsule(pts, np.array(p['a']), np.array(p['b']), p['r'])
    elif kind == 'torus':
        return atlas_csg.sdf_torus(pts, np.array(p['c']), p['R'], p['r'])
    raise ValueError(f"Unknown: {kind}")


# ══════════════════════════════════════════════════════════════
# 2. FILM-SIREN DEFORMATION NETWORK -- deterministically seeded from z
# ══════════════════════════════════════════════════════════════

class FiLMSIREN:
    def __init__(self, z: np.ndarray, n_layers: int = N_LAYERS, hidden_dim: int = HIDDEN_DIM):
        self.k = len(z)
        self.L = n_layers
        self.H = hidden_dim
        self._init_weights(z)

    def _init_weights(self, z: np.ndarray):
        seed = int(np.floor(np.linalg.norm(z, ord=1) * 1e6)) % (2 ** 31)
        rng = np.random.RandomState(seed)
        scale = 1.0 / np.sqrt(self.k)

        self.W = []
        self.b = []
        in_dim = 3
        for i in range(self.L):
            if i == 0:
                W_i = rng.normal(0, 1.0 / in_dim, (self.H, in_dim))
            else:
                W_i = rng.normal(0, np.sqrt(6.0 / in_dim) / SIREN_OMEGA, (self.H, in_dim))
            b_i = rng.normal(0, scale, (self.H,))
            self.W.append(W_i)
            self.b.append(b_i)
            in_dim = self.H

        self.W_out = rng.normal(0, np.sqrt(6.0 / self.H) / SIREN_OMEGA, (1, self.H))
        self.b_out = np.zeros(1)

        self.gamma = []
        self.beta = []
        for i in range(self.L):
            W_g = rng.normal(0, scale, (self.H, self.k))
            W_b = rng.normal(0, scale, (self.H, self.k))
            self.gamma.append(1.0 + W_g @ z)
            self.beta.append(W_b @ z)

    def forward(self, pts: np.ndarray) -> np.ndarray:
        h = pts.copy()
        h = np.sin(SIREN_OMEGA * (h @ self.W[0].T + self.b[0]))
        h = self.gamma[0] * h + self.beta[0]

        for i in range(1, self.L):
            h = np.sin(h @ self.W[i].T + self.b[i])
            h = self.gamma[i] * h + self.beta[i]

        out = (h @ self.W_out.T + self.b_out).squeeze(-1)
        return out

    def __call__(self, pts: np.ndarray) -> np.ndarray:
        return self.forward(pts)


# ══════════════════════════════════════════════════════════════
# 3. FULL psi(z) -- BASE + DEFORMATION
# ══════════════════════════════════════════════════════════════

def psi_raw(z: np.ndarray, pts: np.ndarray) -> np.ndarray:
    prim = select_base_primitive(z)
    f_base = eval_base_primitive(pts, prim)

    deformer = FiLMSIREN(z)
    delta_f = deformer.forward(pts)

    alpha = 0.25 + 0.15 * abs(float(z[-1]))
    return f_base + alpha * np.tanh(delta_f)


# ══════════════════════════════════════════════════════════════
# 4. EIKONAL CORRECTION  psi*(z) = EDT(~B) - EDT(B)  -- guarantees |grad psi*| = 1
# ══════════════════════════════════════════════════════════════

def psi_star(z: np.ndarray, pts: np.ndarray = None,
             resolution: int = 64) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from scipy.ndimage import distance_transform_edt

    N = resolution
    lin = np.linspace(-1, 1, N)
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing='ij')
    grid_pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    f_raw = psi_raw(z, grid_pts).reshape(N, N, N).astype(np.float32)

    interior_mask = (f_raw < 0)
    if not interior_mask.any():
        interior_mask[N // 2, N // 2, N // 2] = True
    if interior_mask.all():
        interior_mask[0, 0, 0] = False

    dist_ext = distance_transform_edt(~interior_mask).astype(np.float32)
    dist_int = distance_transform_edt(interior_mask).astype(np.float32)
    f_edt = (dist_ext - dist_int)

    voxel_size = 2.0 / N
    f_edt = f_edt * voxel_size

    return f_raw, f_edt, grid_pts
