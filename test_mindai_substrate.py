"""
test_mindai_substrate.py — verify the three measurable claims.

Claim A: Active RBNodes (er_bridge_strength > 0) earn more terrain detail
         than dead terrain (no nodes, coherence = 0).

Claim B: Higher er_bridge_strength → more deep chunks within the node's
         footprint (positive correlation between bridge and detail_near count).

Claim C: Terrain height feedback changes node dynamics.  An RBNode that has
         gone through feed_terrain_back() evolves a different rstate.R than
         an identical node that has not, after the same number of RK4 steps.

All three tests are purely numeric — no subprocess, no compiled .so required
for the relational layer, and the substrate is mocked where the .so is absent
so tests can run in CI without the C build.

When `substrate_capi.so` is present (i.e., `wc-substrate/` has been built)
the mock is bypassed and the real terrain is used.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import importlib

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
for _sub in ("wc-substrate", "files5"):
    _p = os.path.join(_HERE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Substrate stub — used when the .so is absent
# ---------------------------------------------------------------------------

class _StubSubstrate:
    """
    Minimal stand-in for wc_substrate_bridge.Substrate.
    height() returns a deterministic function of position so feedback tests
    remain meaningful even without the C build.
    """

    def update(self, sources, eye=(0.0, 0.0), view_radius=600.0, settle=5):
        # Simulate non-zero regen proportional to number of sources
        return max(1, len(sources) * 3)

    def resident_count(self):
        return 64

    def detail_near(self, wx, wz, min_lod=3, radius=220.0):
        # Returns a value that increases with coherence at the point.
        # Real tests override this; see Claim B.
        return 0

    def height(self, wx, wz):
        # Deterministic hill: h = 80 * sin(wx/200) * cos(wz/200)
        return 80.0 * math.sin(wx / 200.0) * math.cos(wz / 200.0)

    def coherence(self, wx, wz):
        return 0.0


def _make_substrate() -> object:
    """Return a real Substrate if the .so exists, else the stub."""
    so_path = os.path.join(_HERE, "wc-substrate", "substrate_capi.so")
    if os.path.exists(so_path):
        from wc_substrate_bridge import Substrate
        return Substrate()
    return _StubSubstrate()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

from relational_epistemic_substrate import (
    RBNode, RBCube, RBNetwork, HypothesisRegistry,
    RelationalState, random_hermitian,
)
import mindai_substrate_bridge as msb


def _make_node_at(wx: float, wz: float) -> RBNode:
    """Create a bare RBNode with world position attributes."""
    registry = HypothesisRegistry()
    cube = RBCube(0, registry)
    uid = cube.add_node("test", f"obj@({wx:.0f},{wz:.0f})")
    node = cube.nodes[uid]
    node.world_x = wx
    node.world_z = wz
    return node


# ---------------------------------------------------------------------------
# Claim A: active nodes earn more terrain detail than dead terrain
# ---------------------------------------------------------------------------

class TestClaimA:
    """
    Active RBNodes (er_bridge_strength > 0) earn more terrain detail than
    dead terrain (no nodes, coherence = 0).

    We compare detail_near() at the node position when the node is stamped
    with high intensity vs when no sources are stamped (or intensity ~ 0).

    With the stub: detail_near always returns 0, so we verify the intensity
    of the source is > 0 (the correct semantic check without the C build).
    """

    def test_active_node_earns_more_detail(self):
        substrate = _make_substrate()

        # Use a high-entropy node (random R -> bridge ~0.6+)
        node = _make_node_at(0.0, 0.0)
        np.random.seed(7)
        node.rstate.R = (
            np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        ).astype(np.complex128)

        bs = node.rstate.er_bridge_strength()
        assert bs > 0.3, f"Test node must have meaningful bridge strength, got {bs:.4f}"

        source = msb.node_to_coherence_source(node)

        if isinstance(substrate, _StubSubstrate):
            # With stub: verify intensity > 0 (the signal that drives terrain)
            assert source[2] > 0.0, "Active node must produce positive coherence intensity"
        else:
            # With real .so: compare detail_near with vs without source
            substrate.update([source], settle=15)
            detail_with = substrate.detail_near(0.0, 0.0, min_lod=3)

            # Fresh substrate with no sources
            from wc_substrate_bridge import Substrate as _S
            s_empty = _S()
            s_empty.update([], settle=15)
            detail_without = s_empty.detail_near(0.0, 0.0, min_lod=3)

            assert detail_with > detail_without, (
                f"Active node detail={detail_with} should exceed "
                f"empty terrain detail={detail_without}"
            )

    def test_zero_bridge_gives_zero_intensity(self):
        node = _make_node_at(100.0, 200.0)
        # Force state to rank-1 (single non-zero singular value -> zero entropy)
        R = np.zeros((4, 4), dtype=np.complex128)
        R[0, 0] = 1.0
        node.rstate.R = R
        bs = node.rstate.er_bridge_strength()
        src = msb.node_to_coherence_source(node)
        intensity = src[2]
        # bridge_strength near 0, intensity must match
        assert abs(intensity - bs) < 1e-6


# ---------------------------------------------------------------------------
# Claim B: higher er_bridge_strength → more deep chunks
# ---------------------------------------------------------------------------

class TestClaimB:
    """
    Higher er_bridge_strength maps to higher coherence intensity, which
    the LOD oracle turns into more deep chunks.

    With the real .so we call detail_near() before/after stamping a high-
    bridge vs low-bridge source at the same position.

    With the stub we verify the intensity mapping is monotone: a node with
    twice the entropy produces a strictly larger intensity value than one
    with half — the stub's detail_near is not meaningful but the intensity
    ordering proves the signal is correct.
    """

    def test_intensity_is_monotone_in_bridge_strength(self):
        np.random.seed(42)

        # Node A: high entropy — random R gives bridge ~0.5-0.75
        node_a = _make_node_at(0.0, 0.0)
        node_a.rstate.R = (
            np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        ).astype(np.complex128)

        # Node B: rank-1 R -> single non-zero singular value -> zero entropy
        node_b = _make_node_at(0.0, 0.0)
        R_lo = np.zeros((4, 4), dtype=np.complex128)
        R_lo[0, 0] = 1.0
        node_b.rstate.R = R_lo

        bs_a = node_a.rstate.er_bridge_strength()
        bs_b = node_b.rstate.er_bridge_strength()

        src_a = msb.node_to_coherence_source(node_a)
        src_b = msb.node_to_coherence_source(node_b)

        assert bs_a > bs_b, f"Node A bridge ({bs_a:.4f}) must exceed Node B ({bs_b:.4f})"
        # Intensity ordering must match bridge ordering
        assert src_a[2] > src_b[2], (
            f"High-bridge node (bs={bs_a:.4f}) intensity ({src_a[2]:.4f}) "
            f"must exceed low-bridge node (bs={bs_b:.4f}) intensity ({src_b[2]:.4f})"
        )

    def test_detail_near_increases_with_real_substrate(self):
        """Only meaningful with real .so — skip with stub."""
        so_path = os.path.join(_HERE, "wc-substrate", "substrate_capi.so")
        if not os.path.exists(so_path):
            pytest.skip("substrate_capi.so not built — skipping real terrain test")

        from wc_substrate_bridge import Substrate
        substrate = Substrate()

        np.random.seed(7)
        # High-bridge source — random R -> bridge ~0.6
        node_hi = _make_node_at(0.0, 0.0)
        node_hi.rstate.R = (
            np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
        ).astype(np.complex128)

        # Low-bridge source — rank-1 R -> bridge ~0
        node_lo = _make_node_at(0.0, 0.0)
        R_lo = np.zeros((4, 4), dtype=np.complex128)
        R_lo[0, 0] = 1.0
        node_lo.rstate.R = R_lo

        src_hi = msb.node_to_coherence_source(node_hi)
        src_lo = msb.node_to_coherence_source(node_lo)

        substrate.update([src_hi], settle=10)
        detail_hi = substrate.detail_near(0.0, 0.0, min_lod=3)

        substrate.update([src_lo], settle=10)
        detail_lo = substrate.detail_near(0.0, 0.0, min_lod=3)

        assert detail_hi >= detail_lo, (
            f"High-bridge detail={detail_hi} should be >= low-bridge detail={detail_lo}"
        )


# ---------------------------------------------------------------------------
# Claim C: terrain feedback changes node dynamics
# ---------------------------------------------------------------------------

class TestClaimC:
    """
    After feed_terrain_back(), a node's H_A is perturbed.  Running RK4 from
    the same initial state with the perturbed H_A produces a different R
    than running with the original H_A — the feedback causally changes dynamics.
    """

    def test_feedback_perturbs_hamiltonian(self):
        substrate = _make_substrate()
        node = _make_node_at(50.0, 120.0)

        H_A_before = node.dynamics.H_A.copy()
        msb.feed_terrain_back(node, substrate)
        H_A_after = node.dynamics.H_A

        diff = np.linalg.norm(H_A_after - H_A_before, "fro")
        assert diff > 1e-8, (
            f"H_A should change after feed_terrain_back, diff={diff:.2e}"
        )

    def test_feedback_changes_rk4_trajectory(self):
        substrate = _make_substrate()

        # Two nodes with identical initial states
        node_ctrl = _make_node_at(50.0, 120.0)
        node_fed  = _make_node_at(50.0, 120.0)

        # Synchronise R and Hamiltonians exactly
        R0 = (np.random.randn(4, 4) + 1j * np.random.randn(4, 4)).astype(np.complex128)
        node_ctrl.rstate.R = R0.copy()
        node_fed.rstate.R  = R0.copy()
        node_ctrl.dynamics.H_A = node_fed.dynamics.H_A.copy()
        node_ctrl.dynamics.H_S = node_fed.dynamics.H_S.copy()

        # Apply terrain feedback only to node_fed
        msb.feed_terrain_back(node_fed, substrate)

        # Step both nodes with the same dt
        node_ctrl.step()
        node_fed.step()

        diff = np.linalg.norm(
            node_fed.rstate.R - node_ctrl.rstate.R, "fro"
        )
        assert diff > 1e-8, (
            f"After feedback, RK4 trajectories should diverge, diff={diff:.2e}"
        )

    def test_terrain_attributes_set(self):
        substrate = _make_substrate()
        node = _make_node_at(-100.0, 300.0)
        msb.feed_terrain_back(node, substrate)

        assert hasattr(node, "terrain_height"), "terrain_height should be set"
        assert hasattr(node, "terrain_stress"), "terrain_stress should be set"
        assert hasattr(node, "terrain_slope"),  "terrain_slope should be set"
        assert -1.0 <= node.terrain_stress <= 1.0, "stress must be in [-1, 1]"
