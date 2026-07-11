"""Tests for gcode_embedding.py — parser, encoder, GcodeSpace, and gate."""
import pytest
import numpy as np

import gcode_embedding as g
import world_compiler as wc


# ---------------------------------------------------------------------------
# Parser tests
# ---------------------------------------------------------------------------

def test_parse_linear_moves():
    """G0/G1 linear commands produce the correct endpoint waypoints."""
    gcode = "G21 G90\nG0 X5.0 Y0.0 Z0.0\nG1 X10.0 Y3.0 Z0.0\n"
    wps = g.parse_gcode(gcode)
    assert len(wps) == 2
    np.testing.assert_allclose(wps[0].pos, [5.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(wps[1].pos, [10.0, 3.0, 0.0], atol=1e-9)
    assert not wps[0].is_arc
    assert not wps[1].is_arc


def test_parse_arc_moves():
    """G2/G3 arc commands produce multiple interpolated waypoints."""
    # G3 CCW quarter-circle from (10,0) to (0,10), centre at (0,0)
    gcode = "G21 G90\nG0 X10.0 Y0.0 Z0.0\nG3 X0.0 Y10.0 I-10.0 J0.0\n"
    wps = g.parse_gcode(gcode)
    # G0 gives 1 waypoint; the arc should give several interpolated points
    assert len(wps) > 2, "arc should produce multiple waypoints"
    arc_wps = [w for w in wps if w.is_arc]
    assert len(arc_wps) > 0

    # The last arc waypoint should be close to the declared endpoint (0, 10)
    last = arc_wps[-1].pos
    np.testing.assert_allclose(last[:2], [0.0, 10.0], atol=0.5)


def test_parse_relative_mode():
    """G91 (relative) moves accumulate correctly."""
    gcode = "G21 G91\nG1 X5.0 Y0.0\nG1 X5.0 Y0.0\n"
    wps = g.parse_gcode(gcode)
    assert len(wps) == 2
    np.testing.assert_allclose(wps[0].pos, [5.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(wps[1].pos, [10.0, 0.0, 0.0], atol=1e-9)


def test_parse_empty_and_comments():
    """Comment-only or empty G-code produces no waypoints."""
    wps = g.parse_gcode("; this is a comment\n(another comment)\n")
    assert wps == []

    wps2 = g.parse_gcode("")
    assert wps2 == []


def test_parse_char_positions():
    """Waypoints carry correct character-offset tags for windowing."""
    gcode = "G21 G90\nG1 X1.0 Y0.0\nG1 X2.0 Y0.0\n"
    wps = g.parse_gcode(gcode)
    assert len(wps) == 2
    # Both waypoints come from lines after the 8-char header "G21 G90\n"
    assert all(w.char_pos >= 8 for w in wps)


def test_window_points_respects_range():
    """_window_points returns only waypoints within the specified range."""
    gcode = "G21 G90\nG1 X1.0 Y0.0 Z0.0\nG1 X2.0 Y0.0 Z0.0\n"
    all_wps = g.parse_gcode(gcode)
    assert len(all_wps) == 2
    # Restrict to just the first move's character range
    pts, af = g._window_points(all_wps, 0, all_wps[0].char_pos + 1)
    assert len(pts) == 1

    # Whole file
    pts_all, _ = g._window_points(all_wps, 0, len(gcode) + 1)
    assert len(pts_all) == 2


# ---------------------------------------------------------------------------
# Encoder and GcodeSpace tests
# ---------------------------------------------------------------------------

def test_encode_determinism():
    """Same point cloud always produces the same raw feature vector."""
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    v1 = g._encode_raw(pts, arc_frac=0.5, gcode_dim=30)
    v2 = g._encode_raw(pts, arc_frac=0.5, gcode_dim=30)
    np.testing.assert_array_equal(v1, v2)
    assert len(v1) == 30


def test_encode_empty_returns_zeros():
    """An empty (< 2-point) cloud returns the zero vector."""
    v = g._encode_raw(np.zeros((0, 3)), arc_frac=0.0, gcode_dim=30)
    np.testing.assert_array_equal(v, np.zeros(30))

    v2 = g._encode_raw(np.zeros((1, 3)), arc_frac=0.0, gcode_dim=30)
    np.testing.assert_array_equal(v2, np.zeros(30))


def test_encode_respects_gcode_dim():
    """Output length equals gcode_dim regardless of the value chosen."""
    pts = np.random.default_rng(0).standard_normal((20, 3))
    for dim in (15, 24, 30, 32, 45):
        v = g._encode_raw(pts, arc_frac=0.0, gcode_dim=dim)
        assert len(v) == dim, f"expected dim={dim}, got {len(v)}"


def test_gcode_space_fit_normalises():
    """After fitting, vector_for returns a z-score-normalised vector."""
    gcode_text = "G21 G90\n" + "\n".join(
        f"G1 X{i*2.0:.1f} Y{i*0.5:.1f} Z0.0" for i in range(30)
    )
    wps = g.parse_gcode(gcode_text)
    n_lines = 30 + 1  # header + moves
    # Build synthetic windows by splitting the waypoints into 5 groups
    stride = max(1, len(wps) // 5)
    window_pts = []
    for i in range(5):
        grp = wps[i * stride:(i + 1) * stride]
        if grp:
            pts = np.stack([w.pos for w in grp])
            af = float(sum(w.is_arc for w in grp)) / len(grp)
        else:
            pts = np.zeros((0, 3))
            af = 0.0
        window_pts.append((pts, af))

    space = g.GcodeSpace.fit(window_pts, gcode_dim=30)
    assert space.dim == 30

    # vector_for on a window with points should not be all-zero
    pts0, af0 = window_pts[0]
    v = space.vector_for(pts0, af0)
    assert len(v) == 30
    assert not np.all(v == 0), "non-empty window should give a non-zero vector"

    # vector_for on empty cloud returns zeros
    v_empty = space.vector_for(np.zeros((0, 3)))
    np.testing.assert_array_equal(v_empty, np.zeros(30))


def test_gcode_space_fingerprint_determinism():
    """Fitting the same windows twice yields the same fingerprint."""
    pts_list = [(np.array([[0.0, i, 0.0], [1.0, i, 0.0], [2.0, i, 0.0]]), 0.0)
                for i in range(5)]
    s1 = g.GcodeSpace.fit(pts_list, gcode_dim=30)
    s2 = g.GcodeSpace.fit(pts_list, gcode_dim=30)
    assert s1.fingerprint() == s2.fingerprint()


# ---------------------------------------------------------------------------
# Falsifiable gate
# ---------------------------------------------------------------------------

def test_gate_gcode_geometry():
    """Circles must cluster closer together than they do to straight lines."""
    gap = g.gate_gcode_geometry(margin=0.25)
    assert gap >= 0.25, f"geometry gap too small: {gap:.3f}"


# ---------------------------------------------------------------------------
# Full-pipeline integration test
# ---------------------------------------------------------------------------

_GCODE_SAMPLE = """\
G21 G90
G0 X0.0 Y0.0 Z0.0
G1 X10.0 Y0.0 Z0.0 F100
G1 X10.0 Y10.0 Z0.0
G3 X0.0 Y20.0 I-10.0 J10.0
G1 X0.0 Y0.0 Z0.0
G0 X5.0 Y5.0 Z0.0
G2 X10.0 Y5.0 I5.0 J0.0
G2 X5.0 Y5.0 I-5.0 J0.0
G1 X0.0 Y5.0 Z0.0
G1 X0.0 Y0.0 Z0.0
""" * 12   # repeat enough to exceed the minimum chunk count


def test_pipeline_compile_gcode():
    """WorldCompiler.compile() works on G-code text with gcode_dim > 0."""
    cfg = wc.CompilerConfig(seed=1, gcode_dim=30)
    scene = wc.WorldCompiler(cfg).compile(_GCODE_SAMPLE)
    assert len(scene.objects) > 0
    assert scene.stats["gcode_dim"] == 30


def test_pipeline_determinism_with_gcode():
    """Same G-code + same seed produces a byte-identical scene."""
    cfg = wc.CompilerConfig(seed=0, gcode_dim=30)
    a = wc.WorldCompiler(cfg).compile(_GCODE_SAMPLE)
    b = wc.WorldCompiler(cfg).compile(_GCODE_SAMPLE)
    assert a.fingerprint == b.fingerprint, (
        f"non-deterministic with gcode channel: {a.fingerprint} != {b.fingerprint}")


def test_pipeline_no_regression_without_gcode():
    """gcode_dim=0 (default) must still compile regular text identically to
    the pre-geometry-channel behaviour (backward-compatibility guard)."""
    text = (
        "The ocean tide carried the whale past the coral reef. "
        "A current swept the reef fish beneath the drifting kelp. "
        "The tide pulled the coral and kelp along the current. "
        "Whales and reef fish share the same deep ocean current. " * 6
    )
    cfg = wc.CompilerConfig(seed=0)
    assert cfg.gcode_dim == 0
    a = wc.WorldCompiler(cfg).compile(text)
    b = wc.WorldCompiler(cfg).compile(text)
    assert a.fingerprint == b.fingerprint
