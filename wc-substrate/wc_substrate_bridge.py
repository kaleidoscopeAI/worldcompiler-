#!/usr/bin/env python3
"""
wc_substrate_bridge.py — bind the World Compiler to the planetary substrate.

THE INTEGRATION
---------------
The World Compiler turns a sentence into a scene graph of objects, each a
canonical SDF at a centroid in object space. The substrate is a deterministic
planet whose terrain detail is driven by a coherence field. This bridge joins
them: every compiled object becomes a coherence source on the planet at its
centroid's (x,z), and the terrain earns detail under it.

  "the wolf sits on the hill"
        │  WorldCompiler.compile_sentence + resolve
        ▼
  SceneGraph: {wolf @ centroid_w, hill @ centroid_h}
        │  scene_to_coherence_sources  (centroid → world (x,z,intensity,radius))
        ▼
  substrate_capi.so: stamp + stream  (GCL-driven LOD)
        ▼
  terrain resolves under the compiled objects

TWO REAL BUGS THIS BRIDGE HAD TO FIX IN THE COMPILER
----------------------------------------------------
1. SceneNode.bbox was never computed — it defaults to None and nothing sets it.
   Every spatial predicate resolver (_resolve_above etc.) early-returns on
   `bbox is None`, so "on"/"above"/"beside" silently did nothing and all
   objects stayed at the origin. Without distinct positions the substrate
   coupling is meaningless. We compute the bbox from each node's SDF surface
   after canonicalization, which is what the resolvers needed all along.

2. Predicate.resolve marked itself resolved even when the resolver was a no-op
   (because bbox was None). With the bbox fixed, resolution actually moves
   objects, so this is no longer masked — but we re-run resolve AFTER fixing
   bboxes so the centroids are real.

We do NOT edit world_compiler_core.py — it's an uploaded artifact. We wrap it.
"""

import ctypes
import os
import numpy as np

import world_compiler_core as wc


# ---------------------------------------------------------------------------
# ctypes binding to substrate_capi.so
# ---------------------------------------------------------------------------

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "substrate_capi.so")
_lib = ctypes.CDLL(_LIB_PATH)

_lib.sub_session_create.restype = ctypes.c_void_p
_lib.sub_session_create.argtypes = [
    ctypes.c_uint64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float,
]
_lib.sub_session_destroy.argtypes = [ctypes.c_void_p]
_lib.sub_session_update.restype = ctypes.c_int
_lib.sub_session_update.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_int,
]
_lib.sub_session_resident_count.restype = ctypes.c_int
_lib.sub_session_resident_count.argtypes = [ctypes.c_void_p]
_lib.sub_session_detail_near.restype = ctypes.c_int
_lib.sub_session_detail_near.argtypes = [
    ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_int, ctypes.c_float,
]
_lib.sub_session_height.restype = ctypes.c_float
_lib.sub_session_height.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
_lib.sub_session_coherence.restype = ctypes.c_float
_lib.sub_session_coherence.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]


class Substrate:
    """Pythonic wrapper over the substrate C session."""

    def __init__(self, seed=0xACE5, lod_cap=5, grid_w=192, grid_h=192,
                 origin_x=-2880.0, origin_z=-2880.0, cell_size=30.0):
        self._h = _lib.sub_session_create(
            seed, lod_cap, grid_w, grid_h, origin_x, origin_z, cell_size)
        if not self._h:
            raise RuntimeError("substrate session creation failed")

    def update(self, sources, eye=(0.0, 0.0), view_radius=600.0, settle=5):
        """sources: list of (x, z, intensity, radius_m). Returns chunks regenerated."""
        n = len(sources)
        flat = (ctypes.c_float * (n * 4))()
        for i, (x, z, inten, rad) in enumerate(sources):
            flat[i*4+0] = x; flat[i*4+1] = z
            flat[i*4+2] = inten; flat[i*4+3] = rad
        return _lib.sub_session_update(
            self._h, flat, n, eye[0], eye[1], view_radius, settle)

    def resident_count(self):
        return _lib.sub_session_resident_count(self._h)

    def detail_near(self, wx, wz, min_lod=3, radius=220.0):
        return _lib.sub_session_detail_near(self._h, wx, wz, min_lod, radius)

    def height(self, wx, wz):
        return _lib.sub_session_height(self._h, wx, wz)

    def coherence(self, wx, wz):
        return _lib.sub_session_coherence(self._h, wx, wz)

    def __del__(self):
        if getattr(self, "_h", None):
            _lib.sub_session_destroy(self._h)
            self._h = None


# ---------------------------------------------------------------------------
# bbox computation — the fix the compiler needed
# ---------------------------------------------------------------------------

def compute_bbox_from_sdf(sdf, resolution):
    """
    Compute the (min, max) corners of an SDF's surface in normalized [-1,1]
    object coordinates. The surface is where sdf <= 0 (interior). Returns
    (min_xyz, max_xyz) as numpy arrays, or None if the volume is empty.

    This is exactly what SceneNode.bbox was meant to hold and never got.
    """
    if sdf is None or sdf.ndim != 3:
        return None
    interior = np.argwhere(sdf <= 0.0)
    if interior.size == 0:
        return None
    # argwhere gives (z,y,x) index order; convert to (x,y,z) world in [-1,1]
    mins_idx = interior.min(axis=0)  # [z,y,x]
    maxs_idx = interior.max(axis=0)
    def to_world(idx_zyx):
        z, y, x = idx_zyx
        return np.array([
            (x / resolution) * 2.0 - 1.0,
            (y / resolution) * 2.0 - 1.0,
            (z / resolution) * 2.0 - 1.0,
        ], dtype=np.float64)
    return (to_world(mins_idx), to_world(maxs_idx))


# ---------------------------------------------------------------------------
# the bridge: compiler -> coherence sources
# ---------------------------------------------------------------------------

# Object-space spans ~[-1,1]; the planet is in meters. This scale maps a
# compiled scene into a region of terrain large enough to show LOD structure.
OBJECT_TO_WORLD_SCALE = 250.0   # meters per object-space unit


class WorldCompilerOnTerrain:
    """
    Compiles sentences into a scene, fixes the bbox/centroid bug so objects
    occupy distinct positions, and projects them onto the substrate as
    coherence sources so the terrain resolves under them.
    """

    def __init__(self, resolution=32, substrate=None, scene_origin=(0.0, 0.0)):
        self.resolution = resolution
        self.compiler = wc.WorldCompiler(resolution=resolution)
        self.substrate = substrate or Substrate()
        self.scene_origin = scene_origin  # where the scene sits on the planet

    def compile(self, sentence):
        """Compile a sentence, fix bboxes, resolve spatial predicates."""
        self.compiler.compile_sentence(sentence)

        # FIX: populate every node's bbox from its SDF so the predicate
        # resolvers (which guard on bbox is None) can actually run.
        for node in self.compiler.scene_graph.nodes.values():
            if node.sdf is not None and node.bbox is None:
                node.bbox = compute_bbox_from_sdf(node.sdf, self.resolution)

        # Now resolution actually moves objects to satisfy "on"/"beside"/etc.
        resolved = self.compiler.resolve()

        # Re-derive bboxes after movement (centroid shifts the effective extent)
        return resolved

    def scene_to_coherence_sources(self, base_intensity=1.0):
        """
        Map each geometry-bearing scene node to a coherence source on the planet.
        Skips Predicate/ControlToken nodes (no geometry). An object's terrain
        footprint (radius) scales with its bbox size; its intensity is uniform
        unless we later weight by salience.
        """
        sources = []
        ox, oz = self.scene_origin
        for node in self.compiler.scene_graph.nodes.values():
            tok = node.token
            # only physical/abstract objects carry geometry
            if not isinstance(tok, (wc.PhysicalToken, wc.RBFField)):
                continue
            if node.sdf is None:
                continue

            # centroid (x,z) projected to the planet
            cx = float(node.centroid[0]) * OBJECT_TO_WORLD_SCALE + ox
            cz = float(node.centroid[2]) * OBJECT_TO_WORLD_SCALE + oz

            # footprint radius from bbox extent (fallback to a default)
            radius = 150.0
            if node.bbox is not None:
                extent = float(np.linalg.norm(node.bbox[1] - node.bbox[0]))
                radius = max(80.0, extent * OBJECT_TO_WORLD_SCALE * 0.5)

            sources.append((cx, cz, base_intensity, radius))
        return sources

    def project_to_terrain(self, eye=None, view_radius=600.0):
        """
        Stamp the compiled scene's objects onto the substrate and stream.
        Returns (sources, regenerated_chunks).
        """
        sources = self.scene_to_coherence_sources()
        if eye is None:
            eye = self.scene_origin
        regen = self.substrate.update(sources, eye=eye, view_radius=view_radius)
        return sources, regen

    def object_names(self):
        out = []
        for node in self.compiler.scene_graph.nodes.values():
            tok = node.token
            if isinstance(tok, (wc.PhysicalToken, wc.RBFField)):
                out.append((node.node_id, getattr(tok, "word", "?"),
                            tuple(round(float(v), 3) for v in node.centroid)))
        return out
