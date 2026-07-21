"""grv2_runtime/thermal.py -- tau_3: SDF volume -> thermal cost field.

Ported verbatim (minus the __main__ smoke test and the /home/claude path
hack) from /home/jg/thermal.py, part of the same "Semantic Crystal Engine"
backend verified this session (26/26 of its own tests pass).

Solves a layer-by-layer heat PDE over an SDF volume:
  phi[k+1] += kappa*(phi[k-1] - 2*phi[k] + phi[k+1])   (diffusion)
  phi[k]   *= exp(-lambda_phi)                          (cooling)
  phi[k]   += occupancy[k] * 0.01                       (heat deposited)

D_k = exp(-lambda_phi * phi[k]) in [0.30, 1.00] -- the "cost of thought made
visible": D_k near 1.0 means a concept is cheap to materialize, D_k near
0.30 means it is at the edge of what the world can hold stably.

grv2_runtime.wiring.WiringBank computes a thermal_cost scalar (1 -
thermal_summary(...)['D_mean']) for every WiringEntry it builds, using the
real SDF each tier already has on hand (atlas_csg.eval_parts / film_siren's
eikonal-corrected f_edt / an EDT reconstruction of the gcode tier's voxel
grid). grv2_runtime.runtime.Runtime feeds the mean cost of any newly
materialized entities into MIRA.evaluate() as a real signal for
duality_risk, instead of duality_risk being driven purely by narrative
text heuristics.
"""
from __future__ import annotations

import numpy as np

KAPPA = 0.10
LAMBDA = 0.15
D_MIN = 0.30


def compute_thermal_volume(sdf_volume: np.ndarray, resolution: int = None) -> np.ndarray:
    if resolution is None:
        resolution = sdf_volume.shape[0]
    N = resolution

    layer_occupancy = np.zeros(N, dtype=np.float32)
    for k in range(N):
        layer_slice = sdf_volume[:, k, :]
        layer_occupancy[k] = float((layer_slice < 0).sum())

    phi = np.zeros(N, dtype=np.float32)
    thermal_layers = np.zeros(N, dtype=np.float32)

    for k in range(N):
        if 0 < k < N - 1:
            phi[k] += KAPPA * (phi[k - 1] - 2 * phi[k] + phi[k + 1])
        phi[k] *= np.exp(-LAMBDA)
        phi[k] += layer_occupancy[k] * 0.01
        thermal_layers[k] = phi[k]

    t_max = thermal_layers.max()
    if t_max > 0:
        thermal_layers /= t_max

    thermal_volume = np.zeros((N, N, N), dtype=np.float32)
    for k in range(N):
        thermal_volume[:, k, :] = thermal_layers[k]

    thermal_volume += _curvature_proxy(sdf_volume) * 0.4
    thermal_volume = np.clip(thermal_volume, 0.0, 1.0)
    return thermal_volume.astype(np.float32)


def _curvature_proxy(sdf: np.ndarray) -> np.ndarray:
    gx = np.gradient(sdf, axis=0)
    gy = np.gradient(sdf, axis=1)
    gz = np.gradient(sdf, axis=2)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-6

    gxx = np.gradient(gx, axis=0)
    gyy = np.gradient(gy, axis=1)
    gzz = np.gradient(gz, axis=2)
    hessian_diag_norm = np.sqrt(gxx ** 2 + gyy ** 2 + gzz ** 2)

    curvature = hessian_diag_norm / grad_mag
    near_surface = np.exp(-np.abs(sdf) * 8.0)
    result = curvature * near_surface

    r_max = result.max()
    if r_max > 0:
        result = result / r_max
    return result.astype(np.float32)


def thermal_correction_factor(thermal_volume: np.ndarray) -> np.ndarray:
    D = np.exp(-LAMBDA * thermal_volume)
    return np.clip(D, D_MIN, 1.0).astype(np.float32)


def thermal_summary(thermal_volume: np.ndarray) -> dict:
    D = thermal_correction_factor(thermal_volume)
    return {
        'mean_cost': float(1.0 - D.mean()),
        'max_cost': float(1.0 - D.min()),
        'critical_ratio': float((D < 0.4).mean()),
        'stable_ratio': float((D > 0.85).mean()),
        'D_min': float(D.min()),
        'D_mean': float(D.mean()),
    }
