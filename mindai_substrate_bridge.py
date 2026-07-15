"""
mindai_substrate_bridge.py — close the loop.

THE CLOSED LOOP
---------------

  sentence / text
       │  WorldCompiler.compile()
       ▼
  WorldScene (WorldObjects at positions)
       │  sentence_to_rbnodes()
       ▼
  RBCube with one RBNode per WorldObject, seeded at the object's world centroid
       │  tick()  (per frame)
       ▼
  node.rstate.er_bridge_strength()  →  CoherenceSource.intensity
       │  Substrate.update()
       ▼
  GCL field  →  terrain LOD  (more detail under active, high-entropy nodes)
       │  Substrate.height() at node world positions
       ▼
  terrain height  →  node stress_level  (feed_terrain_back)
       │  changes H_A_eff  →  RK4 dynamics
       ▼
  altered entanglement  →  altered bridge_strength
       └── loop closes ──────────────────────────────┘

THREE MEASURABLE CLAIMS
-----------------------
  A: Active RBNodes (er_bridge_strength > 0) earn more terrain detail than
     dead terrain (no nodes, coherence = 0).
  B: Higher er_bridge_strength → more deep chunks within the node's footprint.
  C: Terrain height feedback changes node dynamics (rstate.R evolves
     differently when stress is applied than without it).

All three are verified by test_mindai_substrate.py.

DESIGN NOTES
------------
• This module does NOT import substrate_capi.so directly.  It imports the
  already-verified Substrate wrapper from wc_substrate_bridge.  That wrapper
  carries the ctypes bindings and the two compiler bug-fixes.

• WorldCompiler is used from world_compiler.py (the one in the repo root,
  not the wc-substrate/ copy), so the existing tests remain green.

• RBNetwork / RBCube / RBNode are imported from files5/.  The bridge adds
  typed nodes (one per WorldObject) on top of cube 0's existing lattice —
  the same pattern as RBNetwork.seed().

• OBJECT_TO_WORLD_SCALE matches wc_substrate_bridge.OBJECT_TO_WORLD_SCALE
  (250.0 m / object-space unit) so positions are consistent.
"""

import sys
import os
import math
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make both wc-substrate/ and files5/ importable
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

for _subdir in ("wc-substrate", "files5"):
    _p = os.path.join(_HERE, _subdir)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-export the Substrate wrapper (ctypes + session lifecycle)
from wc_substrate_bridge import Substrate, OBJECT_TO_WORLD_SCALE  # noqa: E402

# Relational substrate — real quantum-flavored lattice dynamics
from relational_epistemic_substrate import (    # noqa: E402
    RBCube, RBNode, RBNetwork, HypothesisRegistry,
    RelationalState, random_hermitian, clamp,
)

# World compiler — text → WorldScene
from world_compiler import WorldCompiler, WorldScene, WorldObject, CompilerConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

# Radius of a node's coherence footprint on the terrain (meters).
# Matches the default in wc_substrate_bridge.scene_to_coherence_sources.
_DEFAULT_RADIUS_M = 150.0

# Terrain height is in meters; map it to [0, 1] for stress via this scale.
# sub_height() returns ~[-200, +200] m for the default fBm seed; 400 m range.
_HEIGHT_SCALE_M = 400.0

# Slope-proxy scale for energy decay modifier.
# Estimated from sub_height() differences across 1-meter steps.
_SLOPE_SCALE = 0.05

# Stress is injected into H_A_eff as a real diagonal shift; this scale
# controls how strongly terrain height perturbs node dynamics.
_STRESS_HAMILTONIAN_SCALE = 0.3


# ---------------------------------------------------------------------------
# sentence_to_rbnodes
# ---------------------------------------------------------------------------

def sentence_to_rbnodes(
    sentence: str,
    cube: RBCube,
    compiler: WorldCompiler | None = None,
    scene_origin: tuple[float, float] = (0.0, 0.0),
) -> tuple[WorldScene, list[int]]:
    """
    Compile *sentence* into a WorldScene, then seed one RBNode into *cube*
    per WorldObject at the object's world-space (x, z) centroid.

    Returns
    -------
    scene : WorldScene
        The compiled scene (for querying object positions, labels, etc.)
    uids  : list[int]
        UIDs of the newly added nodes, parallel to scene.objects.
    """
    if compiler is None:
        compiler = WorldCompiler()

    scene: WorldScene = compiler.compile(sentence)

    uids: list[int] = []
    ox, oz = scene_origin

    for obj in scene.objects:
        # Map WorldObject.position (object-space [-1,1]) to world meters.
        # WorldObject.position is (x, y, z); z in object space → z in world.
        wx = float(obj.position[0]) * OBJECT_TO_WORLD_SCALE + ox
        wz = float(obj.position[2]) * OBJECT_TO_WORLD_SCALE + oz

        uid = cube.add_node("world_object", obj.label[:64])
        node = cube.nodes[uid]

        # Store world position on the node for per-tick terrain queries.
        node.world_x = wx
        node.world_z = wz
        # Store the WorldObject reference for diagnostics.
        node.world_obj = obj

        uids.append(uid)

    return scene, uids


# ---------------------------------------------------------------------------
# node_to_coherence_source
# ---------------------------------------------------------------------------

def node_to_coherence_source(
    node: RBNode,
    scale: float = _DEFAULT_RADIUS_M,
) -> tuple[float, float, float, float]:
    """
    Convert a single RBNode to a (world_x, world_z, intensity, radius_m)
    coherence source tuple ready for Substrate.update().

    intensity = er_bridge_strength()  in [0, 1]
    radius_m  = scale (constant; object footprint on terrain)

    The node must have .world_x and .world_z attributes (set by
    sentence_to_rbnodes).  Nodes without these attributes are skipped by
    tick() — this function will raise AttributeError if called directly on
    a bare lattice node that never passed through sentence_to_rbnodes.
    """
    intensity = clamp(node.rstate.er_bridge_strength(), 0.0, 1.0)
    return (node.world_x, node.world_z, intensity, scale)


# ---------------------------------------------------------------------------
# feed_terrain_back
# ---------------------------------------------------------------------------

def feed_terrain_back(
    node: RBNode,
    substrate: Substrate,
    *,
    height_scale: float = _HEIGHT_SCALE_M,
    stress_h_scale: float = _STRESS_HAMILTONIAN_SCALE,
    slope_scale: float = _SLOPE_SCALE,
) -> float:
    """
    Sample terrain height at the node's world position and feed it back as
    stress into the node's Hamiltonian (H_A) and energy-decay modifier.

    The feedback path:
      height h → normalized stress s = h / height_scale (clamped to [-1, 1])
      s → diagonal perturbation of node.dynamics.H_A
      slope proxy → multiplicative energy-decay tag on the node

    Returns
    -------
    stress : float
        The normalized stress value applied this call, in [-1, 1].

    Implementation note
    -------------------
    We do NOT store a per-node energy field (that belongs to the World
    Compiler's closed energy economy, not here).  Instead we perturb H_A
    in-place; this changes the RK4 trajectory on the *next* node.step() call,
    which is the causal direction: terrain at t → dynamics at t+1.
    """
    wx = node.world_x
    wz = node.world_z

    h = substrate.height(wx, wz)
    stress = clamp(h / height_scale, -1.0, 1.0)

    # Slope proxy from finite differences (1 m step)
    h_dx = substrate.height(wx + 1.0, wz)
    h_dz = substrate.height(wx, wz + 1.0)
    slope = math.sqrt((h_dx - h) ** 2 + (h_dz - h) ** 2) * (1.0 / slope_scale)

    # Perturb H_A: add a real diagonal shift proportional to stress.
    # This breaks the Hermitian symmetry slightly but is numerically safe
    # because the shift is small (|stress| ≤ 1, scale ≤ 0.3 → shift ≤ 0.3).
    dim = node.dynamics.H_A.shape[0]
    shift = stress * stress_h_scale * np.eye(dim, dtype=np.complex128)
    node.dynamics.H_A = node.dynamics.H_A + shift

    # Tag the slope on the node for external inspection / tests.
    node.terrain_slope = float(slope)
    node.terrain_height = float(h)
    node.terrain_stress = float(stress)

    return stress


# ---------------------------------------------------------------------------
# tick  — the per-frame closed loop
# ---------------------------------------------------------------------------

def tick(
    network: RBNetwork,
    substrate: Substrate,
    eye: tuple[float, float] = (0.0, 0.0),
    *,
    view_radius: float = 600.0,
    node_radius: float = _DEFAULT_RADIUS_M,
) -> dict:
    """
    Advance one frame of the closed loop:

      1. Collect CoherenceSources from all world-object nodes across all cubes.
      2. Stamp them onto the substrate and stream (Substrate.update).
      3. Read terrain height back at each node's position (feed_terrain_back).
      4. Step all cubes (RK4 dynamics with the updated H_A from step 3).
      5. Step the hypothesis registry.

    Returns a diagnostics dict with keys:
      n_sources    : int   — number of coherence sources stamped
      regen        : int   — terrain chunks regenerated this frame
      resident     : int   — total resident terrain chunks
      mean_bridge  : float — mean er_bridge_strength across world-object nodes
      mean_stress  : float — mean terrain stress applied to world-object nodes
    """
    # --- collect sources from world-object nodes ---
    sources: list[tuple[float, float, float, float]] = []
    world_nodes: list[RBNode] = []

    for cube in network.cubes:
        for node in cube.nodes.values():
            if not (hasattr(node, "world_x") and hasattr(node, "world_z")):
                continue
            sources.append(node_to_coherence_source(node, node_radius))
            world_nodes.append(node)

    # --- stamp + stream ---
    regen = substrate.update(sources, eye=eye, view_radius=view_radius)
    resident = substrate.resident_count()

    # --- feed terrain back into node Hamiltonians ---
    stresses: list[float] = []
    for node in world_nodes:
        s = feed_terrain_back(node, substrate)
        stresses.append(s)

    # --- advance relational dynamics (uses updated H_A) ---
    t = network.cubes[0].tick if network.cubes else 0
    network.step(t)

    bridges = [n.rstate.er_bridge_strength() for n in world_nodes]

    return {
        "n_sources":   len(sources),
        "regen":       regen,
        "resident":    resident,
        "mean_bridge": float(np.mean(bridges)) if bridges else 0.0,
        "mean_stress": float(np.mean(stresses)) if stresses else 0.0,
    }


# ---------------------------------------------------------------------------
# High-level convenience: compile + run N frames
# ---------------------------------------------------------------------------

class WorldSubstrateSession:
    """
    All-in-one session: compile a sentence, wire it into the substrate,
    and run the closed loop for any number of frames.

    Usage
    -----
        sess = WorldSubstrateSession("the wolf sits on the hill")
        for _ in range(10):
            diag = sess.step()
            print(diag)
    """

    def __init__(
        self,
        sentence: str,
        *,
        substrate: Substrate | None = None,
        compiler: WorldCompiler | None = None,
        scene_origin: tuple[float, float] = (0.0, 0.0),
    ):
        self.substrate = substrate or Substrate()
        self.compiler  = compiler  or WorldCompiler()
        self.network   = RBNetwork()
        self.scene_origin = scene_origin

        # Seed the lattice with objects from the compiled sentence.
        self.scene, self.object_uids = sentence_to_rbnodes(
            sentence,
            self.network.cubes[0],
            compiler=self.compiler,
            scene_origin=scene_origin,
        )

    def step(self, eye: tuple[float, float] | None = None) -> dict:
        """Run one frame of the closed loop. Returns diagnostics dict."""
        _eye = eye if eye is not None else self.scene_origin
        return tick(self.network, self.substrate, _eye)

    def object_diagnostics(self) -> list[dict]:
        """
        Return per-object diagnostics for all world-object nodes.
        Useful for debugging and visualization.
        """
        out = []
        for cube in self.network.cubes:
            for node in cube.nodes.values():
                if not hasattr(node, "world_x"):
                    continue
                out.append({
                    "label":          node.label,
                    "world_x":        node.world_x,
                    "world_z":        node.world_z,
                    "er_bridge":      node.rstate.er_bridge_strength(),
                    "activity":       node.activity(),
                    "confidence":     node.confidence(),
                    "age":            node.age,
                    "terrain_height": getattr(node, "terrain_height", None),
                    "terrain_stress": getattr(node, "terrain_stress", None),
                    "terrain_slope":  getattr(node, "terrain_slope",  None),
                })
        return out


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Closed-loop smoke test")
    parser.add_argument("sentence", nargs="?",
                        default="the wolf sits on the hill near the river")
    parser.add_argument("--frames", type=int, default=5)
    args = parser.parse_args()

    print(f"Compiling: {args.sentence!r}")
    sess = WorldSubstrateSession(args.sentence)

    n_obj = len(sess.scene.objects)
    print(f"Scene objects: {n_obj}  |  Object UIDs in cube: {sess.object_uids}")

    for frame in range(args.frames):
        diag = sess.step()
        print(
            f"  frame {frame:3d} | sources={diag['n_sources']:3d} "
            f"regen={diag['regen']:4d} resident={diag['resident']:4d} "
            f"mean_bridge={diag['mean_bridge']:.4f} "
            f"mean_stress={diag['mean_stress']:+.4f}"
        )

    print("\nPer-object diagnostics (after last frame):")
    for d in sess.object_diagnostics():
        print(
            f"  [{d['label'][:30]:30s}] "
            f"er={d['er_bridge']:.4f} "
            f"h={d['terrain_height']:+7.1f}m "
            f"stress={d['terrain_stress']:+.3f}"
        )
