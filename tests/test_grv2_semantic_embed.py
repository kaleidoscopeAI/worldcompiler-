"""Tests for grv2_runtime/semantic_embed.py -- the retrieval tier's similarity metric."""
import numpy as np
import pytest

sklearn = pytest.importorskip("sklearn", reason="semantic_embed needs scikit-learn (soft dependency)")

from grv2_runtime.semantic_embed import SemanticEmbed


_CORPUS = ["bear", "tree", "mountain", "crystal", "fire", "water", "sphere", "human"]


def test_embed_returns_unit_vector():
    e = SemanticEmbed(_CORPUS)
    z = e.embed("bear")
    assert abs(np.linalg.norm(z) - 1.0) < 1e-4


def test_nearest_finds_exact_match_first():
    e = SemanticEmbed(_CORPUS)
    matches = e.nearest("bear", _CORPUS, top_k=3)
    assert matches[0][0] == "bear"
    assert matches[0][1] == pytest.approx(1.0, abs=1e-4)


def test_nearest_prefers_lexically_similar_words():
    """Char n-gram similarity should put 'mountains' closer to 'mountain'
    than to an unrelated word like 'fire'."""
    e = SemanticEmbed(_CORPUS)
    sim_mountain = e.nearest("mountains", ["mountain"], top_k=1)[0][1]
    sim_fire = e.nearest("mountains", ["fire"], top_k=1)[0][1]
    assert sim_mountain > sim_fire


def test_too_small_corpus_raises_value_error():
    with pytest.raises(ValueError):
        SemanticEmbed(["only_one_word"])
