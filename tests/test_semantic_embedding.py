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
