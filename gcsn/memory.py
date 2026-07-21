"""gcsn/memory.py -- memorize-and-utilize for GCSN, same pattern as
grv2_runtime/wiring_store.py: every text query the system ever resolves
gets saved once, forever, and reused exactly on exact-text match; only a
genuinely new query pays the cost (here: a network forward pass instead of
an LLM call) of actually generating something.

Why this exists: GCSN's decoder is a smooth function of a continuous,
frozen MiniLM embedding with no per-example lookup table (see the
"why isn't it memorizing" discussion) -- structurally, it CANNOT memorize
individual training examples, and empirically it hasn't even finished
fitting them (train field-accuracy 0.748, not ~1.0, at epoch 90). But we
already know the exact right answer for every training sentence -- it's
how the quantized targets (tgt) were generated -- so there's no reason to
ask the imperfect network to re-derive something we already have losslessly.

Two provenances, both real:
  "ground_truth" -- dequantized straight from dataset_v4.pt's tgt tensor,
                     the literal supervision signal the network was trained
                     to hit, bypassing the network's approximation error
                     entirely for the 3000 training sentences.
  "generated"     -- the network's own output on a query it had never seen
                      before, cached after first generation so a repeated
                      query is instant and doesn't silently redo the forward
                      pass (deterministic given fixed weights, so this is a
                      cache, not a source of inconsistency).
"""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from gcsn_v3_e2e import X_BINS, Y_BINS, Z_BINS, IJ_BINS, ARC_SAMPLES, differentiable_path


def dequantize_to_blocks(tgt_row: torch.Tensor) -> List[dict]:
    """tgt_row: (K, 7) long tensor [is_arc, cw, x_bin, y_bin, z_bin, i_bin, j_bin]
    -> the K block dicts, at bin precision -- this is the exact supervision
    signal structured_ce_loss trains the decoder against, not an
    approximation of it."""
    blocks = []
    for row in tgt_row.tolist():
        is_arc, cw, xb, yb, zb, ib, jb = row
        blocks.append(dict(
            is_arc=bool(is_arc), clockwise=bool(cw),
            x=float(X_BINS[xb]), y=float(Y_BINS[yb]), z=float(Z_BINS[zb]),
            i=float(IJ_BINS[ib]), j=float(IJ_BINS[jb]),
        ))
    return blocks


def exact_point_cloud_from_blocks(blocks: List[dict], arc_samples: int = ARC_SAMPLES) -> np.ndarray:
    """Non-differentiable, exact re-implementation of differentiable_path's
    per-block geometry (same arc-sweep-direction convention), operating on
    hard block values instead of soft/quantized network outputs. This is
    what makes the memorized ground truth genuinely exact rather than
    'exact up to what the network learned to approximate'."""
    pts = []
    cur = np.zeros(3)
    t = np.linspace(1.0 / arc_samples, 1.0, arc_samples)
    for b in blocks:
        target = np.array([b["x"], b["y"], b["z"]])
        if b["is_arc"]:
            cx, cy = cur[0] + b["i"], cur[1] + b["j"]
            r = math.hypot(cur[0] - cx, cur[1] - cy)
            a0 = math.atan2(cur[1] - cy, cur[0] - cx)
            a1 = math.atan2(target[1] - cy, target[0] - cx)
            if b["clockwise"] and a1 > a0:
                a1 -= 2 * math.pi
            elif not b["clockwise"] and a1 < a0:
                a1 += 2 * math.pi
            angles = a0 + t * (a1 - a0)
            zs = cur[2] + t * (target[2] - cur[2])
            block_pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles), zs], axis=1)
        else:
            block_pts = cur + t[:, None] * (target - cur)
        pts.append(block_pts)
        cur = target
    return np.concatenate(pts, axis=0).astype(np.float32)


class GCSNMemory:
    """text -> {"cloud": (N,3) float32 ndarray, "source": "ground_truth"|"generated"}.
    Same disk-persistence discipline as wiring_store.py: atomic writes
    (temp file + os.replace), never a partial/corrupt file on disk."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self.store: Dict[str, dict] = {}
        if path and os.path.exists(path):
            self.store = torch.load(path, weights_only=False)

    @staticmethod
    def _key(text: str) -> str:
        return text.strip().lower()

    def remember(self, text: str, cloud: np.ndarray, source: str) -> None:
        self.store[self._key(text)] = {"cloud": cloud, "source": source}

    def recall(self, text: str) -> Optional[dict]:
        return self.store.get(self._key(text))

    def save(self) -> None:
        if not self.path:
            return
        tmp = self.path + ".tmp"
        torch.save(self.store, tmp)
        os.replace(tmp, self.path)

    def seed_ground_truth(self, dataset_path: str, split: str = "train") -> int:
        """Memorize every sentence in a dataset split at its exact,
        lossless (bin-precision) supervision target -- no network involved.
        Returns the count of entries seeded."""
        data = torch.load(dataset_path, weights_only=False)
        texts, tgt = data[split]["texts"], data[split]["tgt"]
        n = 0
        for text, row in zip(texts, tgt):
            blocks = dequantize_to_blocks(row)
            cloud = exact_point_cloud_from_blocks(blocks)
            self.remember(text, cloud, source="ground_truth")
            n += 1
        return n

    def stats(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for v in self.store.values():
            out[v["source"]] = out.get(v["source"], 0) + 1
        return out


def generate(text: str, memory: GCSNMemory, decoder, lm, device: str = "cpu"
            ) -> Tuple[np.ndarray, str]:
    """Memorize-and-utilize entry point: exact recall on any known query
    (ground-truth or previously-generated), network forward pass -- then
    caching -- only for genuinely new text."""
    hit = memory.recall(text)
    if hit is not None:
        return hit["cloud"], hit["source"]

    sem = torch.tensor(lm.encode([text], show_progress_bar=False), dtype=torch.float32)
    with torch.no_grad():
        out = decoder(sem)
        cloud = differentiable_path(out, device, tau=0.3)[0].cpu().numpy()
    memory.remember(text, cloud, source="generated")
    memory.save()
    return cloud, "generated"
