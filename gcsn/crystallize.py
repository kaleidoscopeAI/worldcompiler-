"""gcsn/crystallize.py -- post-process a GCSN-generated point cloud through
grv2_runtime.wiring's existing voxelize + simulated-annealing
crystallization, the same step every other tier in the wiring dictionary
(GCODE_LIBRARY, the LLM tier) already goes through. Reused, not
reimplemented: GCSN's raw output is 72 sparse points strung along 6 path
segments (linear/arc blocks sampled at ARC_SAMPLES each) -- crystallizing
snaps that into the same dense, binary, voxel-grid geometry every other
entry in the dictionary has, so a GCSN-sourced shape is structurally
indistinguishable from a hand-authored or LLM-sourced one once it lands in
wiring_store.

Deliberately NOT used inside the training loop -- voxelize/anneal_crystal
are discrete, non-differentiable numpy operations, exactly the kind of
thing gcsn_v3_e2e.differentiable_path was built to route around so the
cycle loss stays end-to-end differentiable. Crystallization only ever runs
at generation/export time, after training, on a detached point cloud.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grv2_runtime.wiring import anneal_crystal, normalize_pts, voxelize  # noqa: E402


def crystallize(points: torch.Tensor, resolution: int = 32, scale: float = 30.0) -> np.ndarray:
    """points: (N, 3) tensor, one example's generated cloud (detached --
    crystallization never needs or produces a gradient). Returns an (M, 3)
    numpy point cloud via the exact same normalize_pts -> voxelize ->
    anneal_crystal pipeline every other wiring.py tier uses, so the result
    drops straight into a grv2_runtime.wiring.WiringEntry unchanged."""
    pts = points.detach().cpu().numpy().astype(np.float32)
    norm = normalize_pts(pts, scale)
    vox = voxelize(norm, resolution)
    vox = anneal_crystal(vox, norm, K=20)
    active = np.argwhere(vox).astype(np.float32)
    if len(active) == 0:
        active = np.zeros((10, 3), dtype=np.float32)
        return active
    return (active / (resolution - 1) - 0.5) * scale
