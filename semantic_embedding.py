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
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

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
    """A word x dim table fit once from a document, plus everything needed to
    embed new text through it later (LiveWorld.feed reuses a founding
    world's SemanticSpace exactly as it reuses its geometric manifold — a
    world's 'worldview' is fixed at founding, in both senses)."""

    vocab: Tuple[str, ...]
    vectors: np.ndarray          # (V, dim) — zero rows if V == 0
    dim: int

    def vector_for(self, chunk_text: str) -> np.ndarray:
        """Mean-pooled, L2-normalized vector of the known words in a chunk.
        Unknown text (no in-vocabulary words) is the honest zero vector, not
        a fabricated guess."""
        if self.dim == 0:
            return np.zeros(0)
        idx = self._index
        hits = [idx[w] for w in _content_tokens(chunk_text) if w in idx]
        if not hits:
            return np.zeros(self.dim)
        v = self.vectors[hits].mean(axis=0)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-12 else v

    def __post_init__(self) -> None:
        self._index: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update("|".join(self.vocab).encode())
        h.update(np.round(self.vectors, 9).tobytes())
        return h.hexdigest()

    @staticmethod
    def empty(dim: int = 0) -> "SemanticSpace":
        return SemanticSpace(vocab=(), vectors=np.zeros((0, dim)), dim=0)

    @classmethod
    def fit(cls, text: str, dim: int = 12, window: int = 4,
           min_count: int = 2, max_vocab: int = 600) -> "SemanticSpace":
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

        return cls(vocab=vocab, vectors=word_vectors, dim=word_vectors.shape[1])


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


def _demo() -> None:
    gap = gate_semantic_clustering()
    print(f"GATE semantic-clustering  PASS  (same-topic - cross-topic cosine gap: {gap:.3f})")
    gate_determinism("the quick brown fox jumps over the lazy dog " * 20)
    print("GATE semantic-determinism PASS")


if __name__ == "__main__":
    _demo()
