"""Tests for kaleidoscope_group.py — wraps all three gate functions."""
import numpy as np
import pytest

import organic_ai_core as core
import kaleidoscope_group as kgroup


@pytest.fixture(scope="module")
def group():
    return kgroup.KaleidoscopeGroup(dim=6, fold=6)


@pytest.fixture(scope="module")
def codes(group):
    rng = core.derive_rng("kg-test", 0)
    base = rng.normal(size=(4, group.dim))
    rows = []
    for _ in range(100):
        shape = base[rng.integers(4)]
        g = group.elements[rng.integers(group.elements.shape[0])]
        rows.append(g @ shape + 0.05 * rng.normal(size=group.dim))
    return np.stack(rows)


def test_orbit_invariance(group):
    """Every member of an orbit must share one invariant signature."""
    kgroup.gate_orbit_invariance(group, trials=20, seed=0)


def test_orbit_beats_distance():
    """A mirrored-far view must be unified into the same strain."""
    same, far = kgroup.gate_orbit_beats_distance(seed=0, dim=6)
    assert same
    assert far


def test_determinism(codes, group):
    """Two identical form_strains calls must produce identical labels."""
    kgroup.gate_determinism(codes, group)
