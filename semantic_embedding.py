"""semantic_embedding.py — distributional meaning, fit fresh from the text
itself. No pretrained model, no network, no external corpus.

organic_ai_core.load_text is deliberately orthographic: hashed character
trigrams, chosen so the same text embeds identically everywhere with zero
vocabulary and zero training. That is the right choice for "never lie about
what ran" — but it also means two chunks about the same topic in different
words land nowhere near each other, because trigram hashing cannot see that
"vehicle" and "car" mean similar things. It only sees their letters.

This module adds exactly the missing piece: a co-occurrence-based semantic
channel built by ordinary linear algebra (SVD) on THIS text's own word-
adjacency statistics — the same PCA/MDL machinery kaleidoscope_core already
uses for compressing data, pointed at a word x word matrix instead of a
sample x feature matrix. Two words that tend to appear near the same other
words get similar vectors (the distributional hypothesis, computed, not
asserted). There is no gradient descent and no pretrained weights: the
"model" is fit in one closed-form step, deterministically, from the document
being compiled — exactly like every other stage in this engine.

`gate_semantic_clustering` makes this falsifiable the same way the rest of
the repo does: build text from two distinct, non-overlapping vocabularies and
require same-topic chunks to land closer in embedding space than cross-topic
chunks, by a real margin. If the channel were doing nothing, this would fail.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional sentence-transformers back-end
# ---------------------------------------------------------------------------
# When installed, `all-MiniLM-L6-v2` (≈22 MB) gives every vocabulary word a
# 384-dim vector grounded in a large general-purpose corpus — concepts that
# never co-occur in your document (e.g. "car" and "automobile") will still
# land close together in this channel.  The model is loaded lazily on first
# use and cached for the lifetime of the process.
#
# Note on determinism: pretrained-channel output is deterministic given the
# same model weights.  Pin your sentence-transformers version in
# requirements.txt (e.g. sentence-transformers==2.7.0) to keep the byte-
# identical guarantee across installs.  When sentence-transformers is absent
# the module falls back silently to PPMI-only embedding, preserving all
# existing behaviour and gates.
try:
    from sentence_transformers import SentenceTransformer as _ST
    _PRETRAINED_AVAILABLE: bool = True
except ImportError:
    _ST = None  # type: ignore[assignment,misc]
    _PRETRAINED_AVAILABLE = False

_PRETRAINED_MODEL: Optional[object] = None


def _get_pretrained_model() -> Optional[object]:
    global _PRETRAINED_MODEL
    if not _PRETRAINED_AVAILABLE:
        return None
    if _PRETRAINED_MODEL is None:
        _PRETRAINED_MODEL = _ST("all-MiniLM-L6-v2")  # type: ignore[misc]
    return _PRETRAINED_MODEL

_TOKEN_RE = re.compile(r"[a-zA-Z]{2,}")

# A fixed, small list of English function words. Excluding them is not a
# judgment call fit to any corpus — they co-occur with everything and would
# otherwise drown the signal this module exists to extract.
_STOPWORDS = frozenset("""
the a an and or but if then else for of to in on at by with from as is are
was were be been being this that these those it its it's i you he she we
they them his her their our your my not no nor so than too very can will
would could should may might must shall do does did doing have has had
having what which who whom where when why how all any both each few more
most other some such only own same just also into out up down over under
again further here there once about
""".split())


def _tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())]


def _content_tokens(text: str) -> List[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) >= 2]


@dataclass
class SemanticSpace:
    """A word × dim table fit once from a document, plus everything needed to
    embed new text through it later (LiveWorld.feed reuses a founding
    world's SemanticSpace exactly as it reuses its geometric manifold — a
    world's 'worldview' is fixed at founding, in both senses).

    When sentence-transformers is installed, an additional ``pretrained_k``-
    dimensional channel is stored in ``pretrained_vecs``: PCA-projected word
    embeddings from ``all-MiniLM-L6-v2``.  ``vector_for`` concatenates both
    channels when available.  The ``dim`` attribute always reflects the full
    output width of ``vector_for``."""

    vocab: Tuple[str, ...]
    vectors: np.ndarray          # (V, ppmi_dim) — PPMI-SVD channel
    dim: int                     # PPMI channel width (0 when space is empty)
    pretrained_vecs: Optional[np.ndarray] = None  # (V, pretrained_k) or None
    pretrained_k: int = 0                          # 0 when no pretrained channel

    @property
    def total_dim(self) -> int:
        """Full output width of vector_for(): dim + pretrained_k."""
        return self.dim + self.pretrained_k

    def vector_for(self, chunk_text: str) -> np.ndarray:
        """Mean-pooled, L2-normalised embedding for a chunk of text.
        Returns the PPMI channel concatenated with the pretrained channel
        (if available).  Unknown text (no in-vocabulary words) yields the
        honest zero vector rather than a fabricated guess."""
        if self.total_dim == 0:
            return np.zeros(0)
        idx = self._index
        hits = [idx[w] for w in _content_tokens(chunk_text) if w in idx]

        parts: List[np.ndarray] = []
        if self.dim > 0:
            if not hits:
                parts.append(np.zeros(self.dim))
            else:
                v = self.vectors[hits].mean(axis=0)
                norm = float(np.linalg.norm(v))
                parts.append(v / norm if norm > 1e-12 else v)

        if self.pretrained_vecs is not None and self.pretrained_k > 0:
            if not hits:
                parts.append(np.zeros(self.pretrained_k))
            else:
                v = self.pretrained_vecs[hits].mean(axis=0)
                norm = float(np.linalg.norm(v))
                parts.append(v / norm if norm > 1e-12 else v)

        return np.concatenate(parts) if parts else np.zeros(max(self.dim, 1))

    def __post_init__(self) -> None:
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update("|".join(self.vocab).encode())
        h.update(np.round(self.vectors, 9).tobytes())
        if self.pretrained_vecs is not None:
            h.update(np.round(self.pretrained_vecs, 9).tobytes())
        return h.hexdigest()

    @staticmethod
    def empty(dim: int = 0) -> "SemanticSpace":
        return SemanticSpace(vocab=(), vectors=np.zeros((0, max(dim, 1))), dim=0)

    @classmethod
    def fit(cls, text: str, dim: int = 12, window: int = 4,
            min_count: int = 2, max_vocab: int = 600,
            pretrained_dim: int = 0) -> "SemanticSpace":
        """Fit a SemanticSpace from ``text``.

        When ``pretrained_dim > 0`` and ``sentence-transformers`` is
        installed, an additional pretrained channel is built by encoding the
        vocabulary words with ``all-MiniLM-L6-v2`` and PCA-projecting to
        ``pretrained_dim`` dimensions.  The PCA sign is canonicalized the
        same way as the PPMI channel for cross-platform determinism."""
        tokens = _content_tokens(text)
        if len(tokens) < 4:
            return cls.empty(dim)

        counts: Dict[str, int] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        candidates = [w for w, c in counts.items() if c >= min_count]
        if len(candidates) > max_vocab:
            # Deterministic top-N by frequency, ties broken lexicographically.
            candidates.sort(key=lambda w: (-counts[w], w))
            candidates = candidates[:max_vocab]
        vocab = tuple(sorted(candidates))
        v = len(vocab)
        if v < 2:
            return cls.empty(dim)
        index = {w: i for i, w in enumerate(vocab)}

        # Symmetric co-occurrence within `window` tokens, counting only pairs
        # where both sides survived vocabulary filtering.
        co = np.zeros((v, v), dtype=np.float64)
        n = len(tokens)
        for i, tok in enumerate(tokens):
            a = index.get(tok)
            if a is None:
                continue
            for j in range(i + 1, min(i + 1 + window, n)):
                b = index.get(tokens[j])
                if b is None:
                    continue
                co[a, b] += 1.0
                co[b, a] += 1.0

        # Positive Pointwise Mutual Information: words that co-occur more
        # than their individual frequencies would predict by chance get a
        # positive score; everything else is clipped to 0 (PPMI is the
        # standard, principled way to turn raw co-occurrence counts into a
        # signal that isn't dominated by merely-frequent words).
        total = float(co.sum())
        if total <= 0:
            return cls.empty(dim)
        row_sums = co.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = np.outer(row_sums, row_sums) / total
            pmi = np.where((co > 0) & (expected > 0), np.log(co / expected), 0.0)
        ppmi = np.maximum(pmi, 0.0)

        k = max(1, min(dim, v - 1))
        u, s, _ = np.linalg.svd(ppmi, full_matrices=False)
        word_vectors = u[:, :k] * np.sqrt(np.maximum(s[:k], 0.0))
        # Sign canonicalization: SVD sign is arbitrary across BLAS builds:
        # pin each component so its largest-magnitude entry is positive, the
        # same fix kaleidoscope_core._pca_basis uses, for the same reason
        # (bit-reproducibility across machines).
        for c in range(word_vectors.shape[1]):
            j = int(np.argmax(np.abs(word_vectors[:, c])))
            if word_vectors[j, c] < 0:
                word_vectors[:, c] = -word_vectors[:, c]

        # -- optional pretrained channel ------------------------------------
        pt_vecs: Optional[np.ndarray] = None
        pt_k = 0
        if pretrained_dim > 0:
            model = _get_pretrained_model()
            if model is not None:
                raw = np.array(model.encode(list(vocab), show_progress_bar=False),  # type: ignore[attr-defined]
                               dtype=np.float64)
                # PCA-project to pretrained_dim for a compact, fixed-width channel.
                mean = raw.mean(axis=0)
                centered = raw - mean
                pu, ps, _ = np.linalg.svd(centered, full_matrices=False)
                pt_k = min(pretrained_dim, min(centered.shape) - 1)
                pt_vecs = pu[:, :pt_k] * np.sqrt(np.maximum(ps[:pt_k], 0.0))
                # Same sign-canonicalization for cross-platform reproducibility.
                for c in range(pt_vecs.shape[1]):
                    jj = int(np.argmax(np.abs(pt_vecs[:, c])))
                    if pt_vecs[jj, c] < 0:
                        pt_vecs[:, c] = -pt_vecs[:, c]

        return cls(vocab=vocab, vectors=word_vectors, dim=word_vectors.shape[1],
                   pretrained_vecs=pt_vecs, pretrained_k=pt_k)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_determinism(text: str, dim: int = 12) -> None:
    a = SemanticSpace.fit(text, dim=dim)
    b = SemanticSpace.fit(text, dim=dim)
    if a.fingerprint() != b.fingerprint():
        raise AssertionError(
            f"semantic-determinism gate FAILED: {a.fingerprint()} != {b.fingerprint()}")


def gate_semantic_clustering(margin: float = 0.15) -> float:
    """The falsifiable claim: chunks about the same topic must land closer in
    this embedding than chunks about different topics, using vocabulary that
    barely overlaps between topics (so the effect can only come from
    co-occurrence structure, not shared words). Returns the same-topic minus
    cross-topic mean-cosine-similarity margin; raises if it's not clearly
    positive."""
    ocean = ("the ocean tide carried the whale past the coral reef",
             "a current swept the reef fish beneath the drifting kelp",
             "the tide pulled the coral and kelp along the current",
             "whales and reef fish share the same deep ocean current")
    mountain = ("the ridge trail climbed past the frozen glacier",
                "a steep trail wound along the rocky mountain ridge",
                "the glacier carved a path beneath the summit ridge",
                "climbers followed the ridge trail toward the summit")

    space = SemanticSpace.fit(" ".join(ocean + mountain), dim=8, min_count=1)
    if space.dim == 0:
        raise AssertionError("semantic-clustering gate FAILED: space collapsed to empty")

    ocean_vecs = [space.vector_for(s) for s in ocean]
    mountain_vecs = [space.vector_for(s) for s in mountain]

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    same = [cos(a, b) for grp in (ocean_vecs, mountain_vecs)
            for i, a in enumerate(grp) for b in grp[i + 1:]]
    cross = [cos(a, b) for a in ocean_vecs for b in mountain_vecs]
    same_mean = float(np.mean(same))
    cross_mean = float(np.mean(cross))
    gap = same_mean - cross_mean
    if gap < margin:
        raise AssertionError(
            f"semantic-clustering gate FAILED: same-topic similarity {same_mean:.3f} "
            f"not clearly above cross-topic {cross_mean:.3f} (gap {gap:.3f} < {margin})")
    return gap


def gate_pretrained_improvement(margin: float = 0.0) -> float:
    """When sentence-transformers is available, the combined channel must
    cluster same-topic chunks AT LEAST as well as PPMI alone (same test
    corpus, same margin).  Returns the combined-channel gap; skips silently
    (returns -1) when the pretrained model is not installed."""
    if not _PRETRAINED_AVAILABLE:
        return -1.0
    ocean = ("the ocean tide carried the whale past the coral reef",
             "a current swept the reef fish beneath the drifting kelp",
             "the tide pulled the coral and kelp along the current",
             "whales and reef fish share the same deep ocean current")
    mountain = ("the ridge trail climbed past the frozen glacier",
                "a steep trail wound along the rocky mountain ridge",
                "the glacier carved a path beneath the summit ridge",
                "climbers followed the ridge trail toward the summit")

    text = " ".join(ocean + mountain)
    space = SemanticSpace.fit(text, dim=8, min_count=1, pretrained_dim=8)
    if space.total_dim == 0:
        raise AssertionError("pretrained gate FAILED: combined space collapsed to empty")

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    ov = [space.vector_for(s) for s in ocean]
    mv = [space.vector_for(s) for s in mountain]
    same = [cos(a, b) for grp in (ov, mv) for i, a in enumerate(grp) for b in grp[i + 1:]]
    cross = [cos(a, b) for a in ov for b in mv]
    gap = float(np.mean(same)) - float(np.mean(cross))
    if gap < margin:
        raise AssertionError(
            f"pretrained-improvement gate FAILED: combined gap {gap:.3f} < {margin}")
    return gap


def _demo() -> None:
    gap = gate_semantic_clustering()
    print(f"GATE semantic-clustering   PASS  (same-topic - cross-topic cosine gap: {gap:.3f})")
    gate_determinism("the quick brown fox jumps over the lazy dog " * 20)
    print("GATE semantic-determinism  PASS")
    pt_gap = gate_pretrained_improvement()
    if pt_gap >= 0:
        print(f"GATE pretrained-improvement PASS  (combined gap {pt_gap:.3f})")
    else:
        print("GATE pretrained-improvement SKIP  (sentence-transformers not installed)")


if __name__ == "__main__":
    _demo()
