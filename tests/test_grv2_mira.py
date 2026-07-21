"""Tests for grv2_runtime/mira.py -- the duality/co-author reward model.

Narrow, focused on the one thing changed this session: measured_thermal_cost
as a real (not text-heuristic) input to duality_risk. The rest of MIRA is a
near-verbatim port with no new behavior to pin down here.
"""
from grv2_runtime.mira import MIRA, MIRAConfig, RewardMetrics


def test_zero_thermal_cost_is_the_default_and_a_no_op():
    """Omitting measured_thermal_cost (or passing 0.0) must behave exactly
    like the pre-thermal-cost signature -- this is a pure extension."""
    m1 = MIRA(MIRAConfig())
    m2 = MIRA(MIRAConfig())
    cur = RewardMetrics()
    r1 = m1.evaluate("look around", cur, measured_coherence=0.8, measured_surprise=0.2)
    r2 = m2.evaluate("look around", cur, measured_coherence=0.8, measured_surprise=0.2,
                     measured_thermal_cost=0.0)
    assert r1.new_metrics.duality_risk == r2.new_metrics.duality_risk
    assert r1.reward == r2.reward


def test_positive_thermal_cost_increases_duality_risk():
    """Materializing something thermally expensive this turn should make
    duality_risk strictly higher than materializing nothing / something
    cheap, all else equal."""
    cur = RewardMetrics(duality_risk=0.10, unresolved=0.30)

    m_cheap = MIRA(MIRAConfig())
    cheap = m_cheap.evaluate("look around", cur, measured_coherence=0.8, measured_surprise=0.2,
                             measured_thermal_cost=0.0)

    m_costly = MIRA(MIRAConfig())
    costly = m_costly.evaluate("look around", cur, measured_coherence=0.8, measured_surprise=0.2,
                               measured_thermal_cost=0.6)

    assert costly.new_metrics.duality_risk > cheap.new_metrics.duality_risk


def test_thermal_cost_is_clamped_to_one():
    cur = RewardMetrics(duality_risk=0.95, unresolved=0.30)
    m = MIRA(MIRAConfig())
    res = m.evaluate("look around", cur, measured_coherence=0.8, measured_surprise=0.2,
                     measured_thermal_cost=5.0)
    assert res.new_metrics.duality_risk <= 1.0
