"""grv2_runtime/wiring_store.py -- the dictionary itself: cache_key ->
serialized WiringEntry, persisted to disk so it survives process restarts.

This is what makes the LLM tier's work permanent instead of thrown away
when the server exits, and what the semantic-retrieval tier learns from:
every word any tier (gcode/atlas/llm/neural) ever resolves gets saved here,
once, forever. A growing, real, on-disk dictionary -- built by use, not by
a bulk pre-generation pass.

Format: one .npz archive (point clouds -- the only large arrays) plus one
.json sidecar (word -> {source, node_count, thermal_cost}, small enough to
read/write whole on every update). Rewriting the whole store on each new
word is O(dictionary size), which is fine at the realistic scale here
(hundreds to low thousands of distinct concepts, not millions).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np


class WiringStore:
    def __init__(self, path: str):
        """`path` is a directory; created on first save if it doesn't exist."""
        self.path = path
        self._npz_path = os.path.join(path, "points.npz")
        self._meta_path = os.path.join(path, "meta.json")

    def load(self) -> Dict[str, dict]:
        """Returns {cache_key: {"word": str, "points": np.ndarray[N,3],
        "node_count": int, "source": str, "thermal_cost": float}}. Empty
        dict if nothing saved yet, or if the store is corrupt/unreadable
        (never raises -- a missing/broken cache degrades to "nothing
        cached", not a crash)."""
        if not (os.path.exists(self._npz_path) and os.path.exists(self._meta_path)):
            return {}
        try:
            with open(self._meta_path, "r") as f:
                meta = json.load(f)
            npz = np.load(self._npz_path)
            out = {}
            for cache_key, m in meta.items():
                key = _npz_key(cache_key)
                if key not in npz:
                    continue
                out[cache_key] = {
                    "word": m["word"],
                    "points": npz[key].astype(np.float32),
                    "node_count": int(m["node_count"]),
                    "source": m["source"],
                    "thermal_cost": float(m.get("thermal_cost", 0.0)),
                }
            return out
        except Exception:
            return {}

    def save(self, entries: Dict[str, dict]) -> None:
        """Overwrites the store with exactly `entries` (cache_key -> the
        same dict shape `load()` returns). Never raises -- a failed save
        just means this session's new words aren't persisted, not a crash."""
        try:
            os.makedirs(self.path, exist_ok=True)
            arrays = {_npz_key(k): e["points"] for k, e in entries.items()}
            meta = {
                k: {"word": e["word"], "node_count": e["node_count"],
                   "source": e["source"], "thermal_cost": e["thermal_cost"]}
                for k, e in entries.items()
            }
            np.savez_compressed(self._npz_path, **arrays)
            with open(self._meta_path, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass


def _npz_key(word: str) -> str:
    # npz array names can't contain arbitrary characters safely across
    # platforms; encode rather than restrict what words can be dictionary
    # keys (cache keys here already look like "atlas:crystal", "llm:castle").
    return word.encode("utf-8").hex()
