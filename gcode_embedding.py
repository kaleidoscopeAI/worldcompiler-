"""gcode_embedding.py — G-code point-cloud geometry channel.

Parses G-code toolpath commands (G0/G1 linear moves, G2/G3 arc moves) into
sequences of 3-D waypoints, then encodes each window of waypoints as a
fixed-width feature vector using only linear algebra — no neural network, no
trained model, no gradient descent.

The geometry channel is designed to sit alongside the existing orthographic
(trigram) and semantic (PPMI-SVD) channels:

  DNA = [ trigram_features | semantic_features | geometry_features ]

Two G-code windows that trace similar shapes (e.g. two circles of roughly the
same radius) will land close together in the geometry sub-space even if they
appear in different regions of the file and share almost no text tokens.

`gate_gcode_geometry` makes this falsifiable: a tight-circle window and a
long-straight-line window must embed further apart than two windows from the
same arc family, by a measurable margin.  If the channel encoded nothing, the
test would fail.

Feature vector layout (15 base + 3*n_bins histogram, zero-padded to gcode_dim):
  [0:3]   centroid xyz (raw coordinates — z-scored by GcodeSpace)
  [3:6]   bounding-box extents
  [6:9]   per-axis standard deviation of point cloud
  [9:12]  PCA eigenvalue ratios (λi / Σλ for top-3 components, sums to ≤1)
  [12]    total path length (raw scalar)
  [13]    mean direction-change / π  (curvature proxy, ∈ [0, 1])
  [14]    arc fraction (fraction of moves that are G2/G3, ∈ [0, 1])
  [15:]   spatial histogram along each of the 3 PCA axes (n_bins bins each)

GcodeSpace.fit() computes per-dimension mean/std from all windows in a
founding document and applies z-score normalization thereafter — exactly the
same 'worldview fixed at founding' discipline as SemanticSpace.

Runtime: numpy + stdlib only.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# G-code parser internals
# ---------------------------------------------------------------------------

_AXIS_RE = re.compile(r'([XYZIJKR])(-?[\d.]+(?:e[+-]?\d+)?)', re.IGNORECASE)
_CMD_RE = re.compile(r'G(\d+(?:\.\d+)?)', re.IGNORECASE)
_COMMENT_RE = re.compile(r';.*$|[(][^)]*[)]', re.MULTILINE)

# G-code command codes that produce motion
_LINEAR_MOVES = frozenset({0.0, 1.0})
_ARC_MOVES = frozenset({2.0, 3.0})


@dataclass
class _MachineState:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    absolute: bool = True   # G90 = absolute (default), G91 = relative
    unit_scale: float = 1.0  # 1.0 for mm (G21), 25.4 for inch (G20)


@dataclass
class _AnnotatedWaypoint:
    """A 3-D point on the toolpath, tagged with its source character offset."""
    pos: np.ndarray   # shape (3,), coordinates in working units
    is_arc: bool      # True if this point came from a G2/G3 arc command
    char_pos: int     # byte offset into the source text that produced this point


def _parse_line(line: str, state: _MachineState,
                char_pos: int) -> Tuple[List[_AnnotatedWaypoint], _MachineState]:
    """Parse one G-code line; return new waypoints and updated machine state.

    Arc moves (G2/G3) are interpolated into a sequence of waypoints so the
    resulting point cloud reflects the actual curved path, not just endpoints.
    The sampling resolution is approximately 1 working-unit per sample.
    """
    # Strip inline comments
    line = _COMMENT_RE.sub('', line).strip().upper()
    if not line:
        return [], state

    # Parse all G-codes and axis words on this line
    cmds = [float(m.group(1)) for m in _CMD_RE.finditer(line)]
    axes: dict = {m.group(1): float(m.group(2)) for m in _AXIS_RE.finditer(line)}

    # Copy state so we can mutate without aliasing
    st = _MachineState(state.x, state.y, state.z, state.absolute, state.unit_scale)

    # Modal (non-motion) commands
    for cmd in cmds:
        if cmd == 90.0:
            st.absolute = True
        elif cmd == 91.0:
            st.absolute = False
        elif cmd == 20.0:
            st.unit_scale = 25.4
        elif cmd == 21.0:
            st.unit_scale = 1.0

    # Motion command (first G0/G1/G2/G3 found, if any)
    move_cmd = next((c for c in cmds if c in _LINEAR_MOVES | _ARC_MOVES), None)
    if move_cmd is None or not axes:
        return [], st

    s = st.unit_scale
    if st.absolute:
        # Absent axes keep the current machine position
        nx = float(axes.get('X', st.x / s)) * s
        ny = float(axes.get('Y', st.y / s)) * s
        nz = float(axes.get('Z', st.z / s)) * s
    else:
        nx = st.x + float(axes.get('X', 0.0)) * s
        ny = st.y + float(axes.get('Y', 0.0)) * s
        nz = st.z + float(axes.get('Z', 0.0)) * s

    waypoints: List[_AnnotatedWaypoint] = []

    if move_cmd in _LINEAR_MOVES:
        st.x, st.y, st.z = nx, ny, nz
        waypoints.append(_AnnotatedWaypoint(
            pos=np.array([nx, ny, nz]), is_arc=False, char_pos=char_pos))

    else:  # G2 (CW) or G3 (CCW) arc
        I = float(axes.get('I', 0.0)) * s
        J = float(axes.get('J', 0.0)) * s
        cx_arc = st.x + I
        cy_arc = st.y + J
        r = math.sqrt((st.x - cx_arc) ** 2 + (st.y - cy_arc) ** 2)

        if r < 1e-9:
            # Degenerate (zero-radius) arc — treat as a linear move
            st.x, st.y, st.z = nx, ny, nz
            waypoints.append(_AnnotatedWaypoint(
                pos=np.array([nx, ny, nz]), is_arc=True, char_pos=char_pos))
        else:
            start_angle = math.atan2(st.y - cy_arc, st.x - cx_arc)
            end_angle = math.atan2(ny - cy_arc, nx - cx_arc)
            if move_cmd == 2.0:  # CW: angle decreases
                if end_angle >= start_angle:
                    end_angle -= 2.0 * math.pi
            else:               # CCW: angle increases
                if end_angle <= start_angle:
                    end_angle += 2.0 * math.pi

            sweep = abs(end_angle - start_angle)
            # ~1 working-unit per sample; at least 4 samples for very small arcs
            n_samples = max(4, int(sweep * r) + 1)
            for k in range(1, n_samples + 1):
                t = k / n_samples
                angle = start_angle + t * (end_angle - start_angle)
                px = cx_arc + r * math.cos(angle)
                py = cy_arc + r * math.sin(angle)
                pz = st.z + t * (nz - st.z)
                waypoints.append(_AnnotatedWaypoint(
                    pos=np.array([px, py, pz]), is_arc=True, char_pos=char_pos))
            st.x, st.y, st.z = nx, ny, nz

    return waypoints, st


def parse_gcode(text: str) -> List[_AnnotatedWaypoint]:
    """Parse all motion commands in *text* and return annotated waypoints.

    The ``char_pos`` attribute of each waypoint records the byte offset of the
    source line in ``text``, which lets ``_window_points`` correctly assign
    waypoints to character-based text windows.
    """
    state = _MachineState()
    all_waypoints: List[_AnnotatedWaypoint] = []
    offset = 0
    for line in text.splitlines():
        wps, state = _parse_line(line, state, char_pos=offset)
        all_waypoints.extend(wps)
        offset += len(line) + 1  # +1 for the newline character
    return all_waypoints


def _window_points(waypoints: List[_AnnotatedWaypoint],
                   start: int, end: int) -> Tuple[np.ndarray, float]:
    """Collect waypoints whose source lines fall within the character range
    ``[start, end)``.  Returns ``(pts, arc_fraction)`` where *pts* is an
    ``(N, 3)`` array (empty array when no waypoints match).
    """
    pts = [w.pos for w in waypoints if start <= w.char_pos < end]
    arcs = [w.is_arc for w in waypoints if start <= w.char_pos < end]
    if not pts:
        return np.zeros((0, 3)), 0.0
    arc_frac = float(sum(arcs)) / len(arcs)
    return np.stack(pts), arc_frac


# ---------------------------------------------------------------------------
# Point-cloud encoder
# ---------------------------------------------------------------------------

# Detect lines with G0/G1/G2/G3 motion commands
_GCODE_MOVE_RE = re.compile(r'\bG[0-3]\b', re.IGNORECASE)


def has_gcode(text: str) -> bool:
    """Return True if *text* contains at least one G0/G1/G2/G3 command."""
    return bool(_GCODE_MOVE_RE.search(text))


def _encode_raw(pts: np.ndarray, arc_frac: float, gcode_dim: int) -> np.ndarray:
    """Encode a ``(N, 3)`` point cloud into a raw feature vector of length
    *gcode_dim*.  Returns a zero vector when *pts* has fewer than 2 points.

    Features:
      base (15 scalars): centroid, bbox extents, per-axis std,
                          PCA eigenvalue ratios, path length,
                          mean curvature, arc fraction
      histogram (3 * n_bins): density histograms along PCA axes
    The raw vector is zero-padded to *gcode_dim* if it falls short.
    """
    if len(pts) < 2:
        return np.zeros(gcode_dim)

    centroid = pts.mean(axis=0)                                    # (3,)
    bbox_extents = pts.max(axis=0) - pts.min(axis=0)              # (3,)
    std_per_axis = pts.std(axis=0)                                 # (3,)
    diffs = np.diff(pts, axis=0)                                   # (N-1, 3)
    path_len = float(np.sum(np.linalg.norm(diffs, axis=1)))

    # PCA eigenvalue ratios from SVD of the centered cloud
    centered = pts - centroid
    subsample = centered[:500] if len(centered) > 500 else centered
    try:
        _, s_vals, Vt = np.linalg.svd(subsample, full_matrices=False)
    except np.linalg.LinAlgError:
        s_vals = np.array([1.0, 0.0, 0.0])
        Vt = np.eye(3)
    eigvals = np.maximum(s_vals[:3] ** 2, 0.0)
    eigsum = eigvals.sum()
    if eigsum > 1e-15:
        pca_ratios = eigvals / eigsum
    else:
        pca_ratios = np.array([1.0, 0.0, 0.0])
    # Ensure exactly 3 values (SVD may return fewer on tiny inputs)
    if len(pca_ratios) < 3:
        pca_ratios = np.pad(pca_ratios, (0, 3 - len(pca_ratios)))

    # Mean curvature: average angle between consecutive direction vectors
    seg_norms = np.linalg.norm(diffs, axis=1)
    valid_mask = seg_norms > 1e-12
    if valid_mask.sum() >= 2:
        valid_dirs = (diffs[valid_mask] /
                      seg_norms[valid_mask, np.newaxis])
        dots = np.clip(
            np.sum(valid_dirs[:-1] * valid_dirs[1:], axis=1), -1.0, 1.0)
        mean_curve = float(np.mean(np.arccos(dots)))
    else:
        mean_curve = 0.0

    # Base feature vector (15 scalars)
    n_bins = max(0, (gcode_dim - 15) // 3)
    base = np.concatenate([
        centroid,                                    # [0:3]
        bbox_extents,                                # [3:6]
        std_per_axis,                                # [6:9]
        pca_ratios,                                  # [9:12]
        [path_len, mean_curve / math.pi, arc_frac],  # [12:15]
    ])

    # Spatial histogram along each PCA axis
    if n_bins > 0 and len(pts) >= 2:
        axes_mat = Vt[:min(3, Vt.shape[0])]          # (k, 3)
        projected = centered @ axes_mat.T             # (N, k)
        hist_parts: List[np.ndarray] = []
        for ax_idx in range(3):
            if ax_idx < projected.shape[1]:
                proj = projected[:, ax_idx]
                mn, mx = proj.min(), proj.max()
                if mx - mn > 1e-12:
                    bins = np.linspace(mn, mx, n_bins + 1)
                    hist, _ = np.histogram(proj, bins=bins)
                    hist_parts.append(hist.astype(float) / max(len(pts), 1))
                else:
                    hist_parts.append(np.zeros(n_bins))
            else:
                hist_parts.append(np.zeros(n_bins))
        hist_vec: np.ndarray = np.concatenate(hist_parts)
    else:
        hist_vec = np.zeros(n_bins * 3)

    raw = np.concatenate([base, hist_vec])
    # Zero-pad to gcode_dim if shorter; truncate if longer
    if len(raw) < gcode_dim:
        raw = np.concatenate([raw, np.zeros(gcode_dim - len(raw))])
    return raw[:gcode_dim]


# ---------------------------------------------------------------------------
# GcodeSpace — fit-once normalizer, mirrors SemanticSpace discipline
# ---------------------------------------------------------------------------


@dataclass
class GcodeSpace:
    """Per-dimension mean/std normalizer for geometry feature vectors.

    Fit once from a founding G-code document.  Reused unchanged for every
    subsequent ``feed_gcode()`` call — a world's geometric 'worldview' (the
    coordinate system in which shapes are compared) is fixed at founding,
    exactly as its semantic worldview is.
    """
    dim: int
    mean_: np.ndarray   # (dim,) per-feature mean across founding windows
    std_: np.ndarray    # (dim,) per-feature std  (1.0 wherever std ≈ 0)

    def vector_for(self, pts: np.ndarray, arc_frac: float = 0.0) -> np.ndarray:
        """Encode and z-score-normalise *pts* using the founding statistics.

        Returns the zero vector for empty point clouds (text windows that
        contain no G-code motion commands), preserving the semantic-space
        convention of returning zeros rather than a fabricated guess.
        """
        if len(pts) < 2:
            return np.zeros(self.dim)
        raw = _encode_raw(pts, arc_frac, self.dim)
        return (raw - self.mean_) / self.std_

    def zero_vector(self) -> np.ndarray:
        """Canonical 'no geometry information' vector."""
        return np.zeros(self.dim)

    def fingerprint(self) -> str:
        h = hashlib.blake2b(digest_size=12)
        h.update(self.mean_.tobytes())
        h.update(self.std_.tobytes())
        return h.hexdigest()

    @classmethod
    def fit(cls, window_pts: List[Tuple[np.ndarray, float]],
            gcode_dim: int = 30) -> "GcodeSpace":
        """Fit a ``GcodeSpace`` from a list of ``(pts, arc_frac)`` per window.

        Windows whose point clouds have fewer than 2 points are excluded from
        fitting (they contribute zero vectors during inference instead).  If no
        window has enough points (e.g. a pure-text document with no G-code),
        the returned space has zero mean and unit std, so ``vector_for``
        always returns the zero vector.
        """
        raws = [_encode_raw(pts, af, gcode_dim)
                for pts, af in window_pts
                if len(pts) >= 2]
        if not raws:
            return cls(dim=gcode_dim,
                       mean_=np.zeros(gcode_dim),
                       std_=np.ones(gcode_dim))
        mat = np.stack(raws)
        mean_ = mat.mean(axis=0)
        std_ = mat.std(axis=0)
        std_ = np.where(std_ < 1e-9, 1.0, std_)   # no division-by-zero
        return cls(dim=gcode_dim, mean_=mean_, std_=std_)


# ---------------------------------------------------------------------------
# Falsifiable gate
# ---------------------------------------------------------------------------


def gate_gcode_geometry(margin: float = 0.25) -> float:
    """The falsifiable geometry claim: two windows of circular arc toolpath
    must embed closer to each other than either does to a window of straight-
    line toolpath, by at least *margin* in cosine similarity.

    Vectors are compared after GcodeSpace normalisation (the same z-scoring
    that the live engine uses), so the dominant scale-invariant features —
    PCA eigenvalue ratios, mean curvature, and arc fraction — drive the
    comparison rather than absolute coordinate magnitudes.

    Returns the (same-shape − cross-shape) cosine-similarity gap; raises if
    it is below *margin*.  If the geometry channel encoded nothing, all
    vectors would collapse to the zero vector and no meaningful gap would
    exist.
    """
    # --- CCW arc circles of two different radii ---
    circle_a = """G21 G90
G0 X10.0 Y0.0 Z0.0
G3 X0.0 Y10.0 I-10.0 J0.0
G3 X-10.0 Y0.0 I0.0 J-10.0
G3 X0.0 Y-10.0 I10.0 J0.0
G3 X10.0 Y0.0 I0.0 J10.0
"""
    circle_b = """G21 G90
G0 X8.0 Y0.0 Z0.0
G3 X0.0 Y8.0 I-8.0 J0.0
G3 X-8.0 Y0.0 I0.0 J-8.0
G3 X0.0 Y-8.0 I8.0 J0.0
G3 X8.0 Y0.0 I0.0 J8.0
"""
    # --- long straight linear moves ---
    line_a = """G21 G90
G0 X0.0 Y0.0 Z0.0
G1 X100.0 Y0.0 Z0.0
G1 X100.0 Y0.5 Z0.0
G1 X0.0 Y0.5 Z0.0
"""

    def _pts(gcode: str) -> Tuple[np.ndarray, float]:
        wps = parse_gcode(gcode)
        if not wps:
            return np.zeros((0, 3)), 0.0
        pts = np.stack([w.pos for w in wps])
        arc_frac = float(sum(w.is_arc for w in wps)) / len(wps)
        return pts, arc_frac

    # Fit a shared normaliser from all three windows so the z-scoring is
    # grounded in the same statistics for all comparisons.
    win_pts = [_pts(circle_a), _pts(circle_b), _pts(line_a)]
    space = GcodeSpace.fit(win_pts, gcode_dim=30)

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    vc_a = space.vector_for(*win_pts[0])
    vc_b = space.vector_for(*win_pts[1])
    vl_a = space.vector_for(*win_pts[2])

    same_sim = _cos(vc_a, vc_b)
    cross_sim = (_cos(vc_a, vl_a) + _cos(vc_b, vl_a)) / 2.0
    gap = same_sim - cross_sim

    if gap < margin:
        raise AssertionError(
            f"gcode-geometry gate FAILED: circle-circle similarity {same_sim:.3f} "
            f"not clearly above circle-line {cross_sim:.3f} (gap {gap:.3f} < {margin})")
    return gap


# ---------------------------------------------------------------------------
# Module self-test
# ---------------------------------------------------------------------------


def _demo() -> None:
    gap = gate_gcode_geometry()
    print(f"GATE gcode-geometry   PASS  (circle-circle vs circle-line cosine gap: {gap:.3f})")
    # Quick parse smoke-test
    sample = "G21 G90\nG1 X10 Y0 Z0\nG1 X10 Y10 Z0\nG2 X0 Y10 I-10 J0\n"
    wps = parse_gcode(sample)
    print(f"  parse smoke-test: {len(wps)} waypoints from 4-line G-code sample")
    print("gcode_embedding demo complete.")


if __name__ == "__main__":
    _demo()
