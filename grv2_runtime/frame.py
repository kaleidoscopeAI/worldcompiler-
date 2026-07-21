"""grv2_runtime/frame.py — replaces GRV2's renderer stub.

GRV2's `gen_frame_()` was an explicit placeholder ("<- YOUR NEURAL RENDERER
HERE") producing hash noise. This is not a neural renderer either -- it's the
real deterministic terrain, sampled top-down exactly the way
wc-substrate/render_scene.py and render_separated.py already do (height as
grayscale relief, coherence as a warm overlay, object positions as colored
markers), extended with a foe-mode tint and per-entity marker color driven by
the entity's measured RBNode bridge strength instead of a fixed amber dot.

This is a few-hundred-ms grid sample, not a per-frame raymarch -- consistent
with the runtime being turn-based, not framerate-based (see runtime.py).
"""
from __future__ import annotations

import io
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .sgr import Entity

_N = 160          # grid resolution (render_scene.py uses 240; smaller here for turn latency)
_SPAN_M = 350.0   # half-width of the rendered area, metres


def render_frame(substrate, entities: Sequence[Entity],
                 bridge_by_id: Optional[Dict[str, float]] = None,
                 center: Tuple[float, float] = (0.0, 0.0),
                 span: float = _SPAN_M, n: int = _N, foe_mode: bool = False) -> bytes:
    bridge_by_id = bridge_by_id or {}
    cx, cz = center
    xs = np.linspace(cx - span, cx + span, n)
    zs = np.linspace(cz - span, cz + span, n)
    height = np.zeros((n, n), dtype=np.float32)
    coh = np.zeros((n, n), dtype=np.float32)
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            height[j, i] = substrate.height(float(x), float(z))
            coh[j, i] = substrate.coherence(float(x), float(z))

    hn = (height - height.min()) / (np.ptp(height) + 1e-6)
    gz, gx = np.gradient(hn)
    relief = np.clip(0.5 + 4.0 * (gx * 0.7 + gz * 0.7), 0, 1)
    base = 0.35 + 0.5 * hn * relief
    img = np.stack([base, base, base], axis=-1)

    if foe_mode:
        # hostile: coherence reads as a warm, alarming red instead of the
        # normally reassuring teal-green -- same signal, opposite affect.
        img[..., 0] = img[..., 0] * (1 - coh * 0.2) + coh * 0.55
        img[..., 1] = img[..., 1] * (1 - coh * 0.4)
        img[..., 2] = img[..., 2] * (1 - coh * 0.4)
    else:
        img[..., 0] = img[..., 0] * (1 - coh * 0.5)
        img[..., 1] = img[..., 1] * (1 - coh * 0.2) + coh * 0.45
        img[..., 2] = img[..., 2] * (1 - coh * 0.3) + coh * 0.40

    for e in entities:
        ex, _ey, ez = e.position
        i = int((ex - (cx - span)) / (2 * span) * n)
        j = int((ez - (cz - span)) / (2 * span) * n)
        if not (0 <= i < n and 0 <= j < n):
            continue
        if e.type == "player":
            color = (1.0, 1.0, 1.0)
        else:
            b = float(np.clip(bridge_by_id.get(e.id, 0.3), 0.0, 1.0))
            color = (0.2 + 0.8 * b, 0.85 - 0.5 * b, 0.2)
        for dj in range(-2, 3):
            for di in range(-2, 3):
                jj, ii = j + dj, i + di
                if 0 <= jj < n and 0 <= ii < n:
                    img[jj, ii] = color

    img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img8).save(buf, format="PNG")
    return buf.getvalue()
