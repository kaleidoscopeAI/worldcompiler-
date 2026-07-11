"""Tests for refinement_loop.py — wraps all four gate functions."""
import numpy as np
import pytest

import organic_ai_core as core
import kaleidoscope_group as kgroup
import refinement_loop as refine


@pytest.fixture(scope="module")
def cfg():
    return refine.RefinementConfig()


@pytest.fixture(scope="module")
def structured_data(cfg):
    rng = core.derive_rng("refine-test", cfg.seed)
    base = rng.normal(size=(5, 4))
    grp = kgroup.KaleidoscopeGroup(dim=4, max_elements=cfg.group_max_elements)
    rows = []
    for _ in range(200):
        shape = base[rng.integers(5)]
        g = grp.elements[rng.integers(grp.elements.shape[0])]
        rows.append(g @ shape)
    latent = np.stack(rows)
    loading = rng.normal(size=(4, 12))
    return latent @ loading + 0.05 * rng.normal(size=(200, 12))


@pytest.fixture(scope="module")
def refinement_report(cfg, structured_data):
    eng = refine.RefinementEngine(cfg)
    eng.load(structured_data.copy())
    return eng.run()


def test_determinism(structured_data, cfg):
    """Two refinement runs on the same data must produce the same fingerprint."""
    refine.gate_determinism(structured_data, cfg)


def test_contraction(refinement_report):
    """DNA drift must be monotone non-increasing (a contraction)."""
    refine.gate_contraction(refinement_report)


def test_reaches_purity(refinement_report):
    """After refinement, codes must converge to their canonical view."""
    refine.gate_reaches_purity(refinement_report)


def test_supernodes_only_final(cfg, structured_data):
    """Supernodes must crystallize from purified DNA, not raw codes."""
    refine.gate_supernodes_only_final(cfg, structured_data.copy())
