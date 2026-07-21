"""grv2_runtime/semantic_embed.py -- phi: word -> unit vector, for the
dictionary-retrieval wiring tier.

Ported (trimmed of the HNSW index and disk-caching, which wiring_store.py
now owns) from /home/jg/semantic_embed.py's SemanticEmbed. TF-IDF + SVD,
L2-normalized -- real semantic structure (via corpus co-occurrence), not
the hash-seeded pseudo-random vectors used elsewhere in this package.

Soft dependency on scikit-learn: if it isn't installed, `SemanticEmbed`
raises ImportError from its constructor and the caller (WiringBank) simply
skips the retrieval tier, same graceful-degradation pattern as texture.py's
image search and llm_gcode's missing API key.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np


class SemanticEmbed:
    """phi(q): Sigma* -> S^{k-1}. Fits on a corpus of words/short phrases
    (here: this repo's own known vocabulary, growing as the dictionary
    grows) and projects any query string to a unit vector."""

    def __init__(self, corpus: List[str], k: int = 32):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        if len(corpus) < 2:
            raise ValueError("need at least 2 corpus entries to fit an embedding space")

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 4), min_df=1, max_features=4096,
        )
        X = self.vectorizer.fit_transform(corpus)
        n_components = max(1, min(k, X.shape[0] - 1, X.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.svd.fit(X)
        self.k = n_components

    def embed(self, text: str) -> np.ndarray:
        x = self.vectorizer.transform([text])
        z = self.svd.transform(x)[0]
        norm = np.linalg.norm(z)
        return (z / norm if norm > 1e-12 else z).astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        X = self.vectorizer.transform(texts)
        Z = self.svd.transform(X)
        norms = np.linalg.norm(Z, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0
        return (Z / norms).astype(np.float32)

    def nearest(self, query: str, candidates: List[str], top_k: int = 1
               ) -> List[tuple]:
        """Return up to top_k (candidate, cosine_similarity) pairs, sorted
        by descending similarity."""
        if not candidates:
            return []
        qz = self.embed(query)
        cz = self.embed_batch(candidates)
        sims = cz @ qz
        order = np.argsort(-sims)[:top_k]
        return [(candidates[i], float(sims[i])) for i in order]
