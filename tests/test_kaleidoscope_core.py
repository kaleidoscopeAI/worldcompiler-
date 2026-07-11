"""Tests for kaleidoscope_core.py — wraps all six gate functions."""
import numpy as np
import pytest

import organic_ai_core as core
import kaleidoscope_core as kal


@pytest.fixture(scope="module")
def cfg():
    return kal.CompressionConfig()


@pytest.fixture(scope="module")
def structured_report(cfg):
    rng = core.derive_rng("kal-test", cfg.seed)
    factors = rng.normal(size=(300, 4))
    loading = rng.normal(size=(4, 12))
    data = factors @ loading + 0.02 * rng.normal(size=(300, 12))
    return kal.CompressionOrganism(cfg).compress_array(data)


@pytest.fixture(scope="module")
def structured_data(cfg):
    rng = core.derive_rng("kal-det-test", cfg.seed)
    factors = rng.normal(size=(300, 4))
    loading = rng.normal(size=(4, 12))
    return factors @ loading + 0.02 * rng.normal(size=(300, 12))


def test_mdl_selects_true_rank(cfg):
    """MDL must recover the true rank of structured data within ±1."""
    rec_rank, true_rank = kal.gate_mdl_selects_true_rank(cfg, true_rank=3)
    assert abs(rec_rank - true_rank) <= 1


def test_two_way_map(cfg):
    """encode → reconstruct round-trip error must be low on rank-3 data."""
    rel = kal.gate_two_way_map(cfg)
    assert rel <= 0.15


def test_honest_on_noise(cfg):
    """On pure noise the organism must NOT invent a low-rank manifold."""
    rank_fraction = kal.gate_honest_on_noise(cfg)
    assert rank_fraction >= 0.7


def test_determinism(structured_data, cfg):
    """Two independent compressions of the same data must be identical."""
    kal.gate_determinism(structured_data, cfg)


def test_beats_baseline(structured_report):
    """Compression MSE must beat the trivial mean-predictor baseline."""
    kal.gate_beats_baseline(structured_report, margin=0.5)


def test_contraction(structured_report):
    """The correction loop must be monotone (a contraction)."""
    kal.gate_contraction(structured_report)
