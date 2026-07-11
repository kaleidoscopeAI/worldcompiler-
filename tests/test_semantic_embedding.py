"""Tests for semantic_embedding.py — wraps both public gate functions."""
import pytest

import semantic_embedding as sem


def test_semantic_clustering():
    """Same-topic chunks must cluster closer than cross-topic chunks."""
    gap = sem.gate_semantic_clustering(margin=0.15)
    assert gap > 0.15, f"semantic gap too small: {gap:.3f}"


def test_semantic_determinism():
    """Fitting the same text twice must produce byte-identical spaces."""
    text = "the quick brown fox jumps over the lazy dog " * 20
    sem.gate_determinism(text, dim=12)


def test_pretrained_clustering_when_available():
    """When sentence-transformers is installed the combined channel must
    cluster at least as well as PPMI alone (margin=0.0).  Skipped when
    sentence-transformers is absent."""
    pytest.importorskip("sentence_transformers")
    gap = sem.gate_pretrained_improvement(margin=0.0)
    assert gap >= 0.0, f"pretrained combined gap unexpectedly negative: {gap:.3f}"
