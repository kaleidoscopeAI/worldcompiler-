"""Tests for genetic_manifold.py — wraps all four gate functions."""
import numpy as np
import pytest

import organic_ai_core as core
import genetic_manifold as gmod


@pytest.fixture(scope="module")
def cfg():
    return gmod.GeneticConfig(generations=20)


@pytest.fixture(scope="module")
def strain_report(cfg):
    rng = core.derive_rng("gm-test", cfg.seed)
    factors = rng.normal(size=(200, 4))
    loading = rng.normal(size=(4, 12))
    data = factors @ loading + 0.03 * rng.normal(size=(200, 12))
    gm = gmod.GeneticManifold(cfg)
    gm.seed_from_data(data)
    return gm.evolve()


@pytest.fixture(scope="module")
def structured_data(cfg):
    rng = core.derive_rng("gm-det-test", cfg.seed)
    factors = rng.normal(size=(200, 4))
    loading = rng.normal(size=(4, 12))
    return factors @ loading + 0.03 * rng.normal(size=(200, 12))


def test_strains_separate_regimes(cfg):
    """Strains from distinct generative regimes must separate correctly."""
    purity = gmod.gate_strains_separate_regimes(cfg, n_regimes=3, per_regime=80)
    assert purity > 1.0 / 3


def test_energy_conservation(strain_report):
    """Total energy must equal the founding budget — no drift allowed."""
    gmod.gate_energy_conservation(strain_report)


def test_no_collapse(strain_report):
    """The population must not collapse all DNA to a single point."""
    gmod.gate_no_collapse(strain_report)


def test_determinism(structured_data, cfg):
    """Two identical evolutions must produce the same fingerprint."""
    gmod.gate_determinism(structured_data, cfg)
