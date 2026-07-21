"""grv2_runtime/texture.py — the skin, applied over the wiring.

Ports LexForge (/home/jg/world compiler/engine.py)'s real image-search and
color-extraction logic verbatim: a real DuckDuckGo scrape, a real downloaded
photo, real k-means over its pixels. This is deliberately re-run on every
invocation rather than cached -- re-querying is exactly what gives the same
wiring a different real-world color each time it recurs, which is the whole
point (grounding *and* variety, from one mechanism). Never touches shape.

Graceful, honest fallback: no network / no results / no PIL success all fall
through to a deterministic hash-derived palette, so the system still runs
(with less variety, not a crash) offline.
"""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class TextureEntry:
    color: Tuple[float, float, float]   # RGB in [0, 1] -- the dominant/primary tint
    source: str                          # "image" or "hash_fallback"
    image_url: str = ""
    # The rest of the k-means palette (or its offline equivalent), color
    # first. A real photo's palette carries real color *variation*, not
    # just a color -- collapsing it to a single flat tint (the old
    # behavior) threw that variation away. Downstream (gltf_builder's
    # point-cloud path) samples per-vertex from this instead of painting
    # every point of an entity the same flat color.
    palette: Tuple[Tuple[float, float, float], ...] = ()

    def __post_init__(self) -> None:
        if not self.palette:
            self.palette = (self.color,)


def search_images(query: str, max_results: int = 8) -> List[dict]:
    """Real DuckDuckGo image scrape. Returns [] on any failure (no network,
    package missing, no results) rather than raising -- callers fall back to
    the deterministic palette. `duckduckgo_search` (the original dependency
    this was written against) was renamed upstream to `ddgs`; the old name
    now installs a shim that returns nothing, so `ddgs` is tried first."""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.images(query, max_results=max_results):
                results.append({
                    "url": r.get("image", ""),
                    "title": r.get("title", ""),
                    "width": r.get("width", 0),
                    "height": r.get("height", 0),
                    "source": r.get("url", ""),
                })
        return [r for r in results if r["url"]]
    except Exception:
        return []


def analyze_image_colors(image_url: str) -> Optional[List[Tuple[float, float, float]]]:
    """Download image, extract a 5-color k-means palette. None on any failure."""
    try:
        import urllib.request
        from PIL import Image

        req = urllib.request.Request(image_url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; grv2_runtime/1.0)"})
        data = urllib.request.urlopen(req, timeout=5).read()
        img = Image.open(io.BytesIO(data)).convert("RGB").resize((64, 64))
        arr = np.array(img).reshape(-1, 3).astype(float) / 255.0

        rng = np.random.default_rng(42)
        centers = arr[rng.choice(len(arr), 5, replace=False)]
        for _ in range(20):
            dists = np.linalg.norm(arr[:, None] - centers[None], axis=2)
            labels = np.argmin(dists, axis=1)
            new_centers = np.array([
                arr[labels == k].mean(axis=0) if (labels == k).any() else centers[k]
                for k in range(5)])
            if np.allclose(centers, new_centers, atol=1e-4):
                break
            centers = new_centers

        sizes = [(labels == k).sum() for k in range(5)]
        order = np.argsort(sizes)[::-1]
        return [tuple(centers[k].tolist()) for k in order]
    except Exception:
        return None


def color_from_hash(word: str) -> Tuple[float, float, float]:
    """Deterministic, vivid-ish fallback palette from the word's own hash --
    used whenever real image grounding isn't available."""
    h = hashlib.sha256(word.encode()).digest()
    r, g, b = h[0] / 255.0, h[1] / 255.0, h[2] / 255.0
    mx, mn = max(r, g, b), min(r, g, b)
    if mx > 0:
        r = mn + (r - mn) / mx * 0.8 + 0.1
        g = mn + (g - mn) / mx * 0.8 + 0.1
        b = mn + (b - mn) / mx * 0.8 + 0.1
    return (min(1.0, r), min(1.0, g), min(1.0, b))


def palette_from_hash(word: str, n: int = 4) -> Tuple[Tuple[float, float, float], ...]:
    """Deterministic multi-color palette for when no real photo is available
    -- same idea as color_from_hash, but a family of n related tones (base
    hue rotated by a per-index hash offset) instead of one flat color, so
    the offline path still has real per-point variation to sample from,
    not just a solid-color fallback of a solid-color fallback."""
    base = color_from_hash(word)
    out = [base]
    for i in range(1, n):
        h = hashlib.sha256(f"{word}:{i}".encode()).digest()
        jitter = np.array([h[0], h[1], h[2]], dtype=np.float64) / 255.0 - 0.5
        shifted = np.clip(np.array(base) + jitter * 0.35, 0.0, 1.0)
        out.append(tuple(shifted.tolist()))
    return tuple(out)


def texture_for(word: str, context: str = "") -> TextureEntry:
    """The skin for one invocation of `word`. `context` (e.g. surrounding
    conversation) shapes the search query so the same word can pull a
    different, more relevant photo depending on what's being discussed."""
    query = f"{word} {context}".strip()
    for img in search_images(query, max_results=3):
        colors = analyze_image_colors(img["url"])
        if colors:
            return TextureEntry(color=colors[0], source="image", image_url=img["url"],
                                palette=tuple(colors))
    return TextureEntry(color=color_from_hash(word), source="hash_fallback",
                        palette=palette_from_hash(word))
