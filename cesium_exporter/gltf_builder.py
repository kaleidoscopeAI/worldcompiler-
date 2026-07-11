"""gltf_builder.py — build a minimal GLB (binary glTF 2.0) from entity list.

Generates geometry for each entity and packs everything into a single
self-contained ``.glb`` file.  Requires only ``numpy`` and ``struct`` from
stdlib — no additional dependencies.

Coordinate convention
---------------------
The GLB uses a **y-up, right-handed** coordinate system (glTF default).
Cesium applies a y-up → z-up axis correction when loading glTF tiles, so::

    glTF (+X, +Y, +Z)  →  Cesium (East, Up, -North)

Entity ENU offsets (east_m, north_m, up_m) are therefore stored in the GLB
as (east_m, up_m, -north_m).  The tile ``transform`` in ``tileset.json``
converts the tile's local ENU frame to ECEF.

Geometry
--------
Each shape type maps to a procedural mesh:

* ``cube`` / ``building`` / ``structure`` — rectangular box
* ``cylinder`` / ``column`` / ``road``     — 12-sided prism
* ``cone`` / ``tower``                      — 12-sided cone
* ``sphere`` / ``landmark``                 — 16×8 UV sphere
* ``tetra``                                 — tetrahedron
* ``icosa``                                 — icosahedron (20-face)
* ``octa``                                  — octahedron
* (default)                                 — box

All meshes are centred at the origin and have a unit bounding-box before
scaling.  The node ``scale`` in the glTF JSON then applies entity-level
scaling.
"""
from __future__ import annotations

import json
import math
import struct
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# glTF / GLB constants
# ---------------------------------------------------------------------------
_GLB_MAGIC = 0x46546C67   # "glTF"
_GLB_VERSION = 2
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN  = 0x004E4942  # "BIN\0"

_COMPONENT_FLOAT          = 5126
_COMPONENT_UNSIGNED_SHORT = 5123
_COMPONENT_UNSIGNED_INT   = 5125

_TARGET_ARRAY_BUFFER         = 34962
_TARGET_ELEMENT_ARRAY_BUFFER = 34963


# ---------------------------------------------------------------------------
# Procedural geometry generators
# Each returns (positions: float32 (N,3), normals: float32 (N,3), indices: uint16/uint32 (M,))
# ---------------------------------------------------------------------------

def _box(sx: float = 1.0, sy: float = 1.0, sz: float = 1.0
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Axis-aligned box centred at origin with per-face normals (24 verts)."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    faces = [
        # (normal_xyz,  quad vertices CW-when-viewed-from-outside)
        ((0, 0, -1), [(-hx,-hy,-hz),(hx,-hy,-hz),(hx,hy,-hz),(-hx,hy,-hz)]),
        ((0, 0,  1), [( hx,-hy, hz),(-hx,-hy, hz),(-hx,hy, hz),( hx,hy, hz)]),
        ((-1, 0, 0), [(-hx,-hy, hz),(-hx,-hy,-hz),(-hx,hy,-hz),(-hx,hy, hz)]),
        (( 1, 0, 0), [( hx,-hy,-hz),( hx,-hy, hz),( hx,hy, hz),( hx,hy,-hz)]),
        ((0, -1, 0), [(-hx,-hy, hz),( hx,-hy, hz),( hx,-hy,-hz),(-hx,-hy,-hz)]),
        ((0,  1, 0), [(-hx, hy,-hz),( hx, hy,-hz),( hx, hy, hz),(-hx, hy, hz)]),
    ]
    pos_list, nrm_list, idx_list = [], [], []
    base = 0
    for (nx, ny, nz), quad in faces:
        for vx, vy, vz in quad:
            pos_list.append((vx, vy, vz))
            nrm_list.append((nx, ny, nz))
        idx_list += [base, base+1, base+2, base, base+2, base+3]
        base += 4
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _uv_sphere(radius: float = 0.5, stacks: int = 8, slices: int = 16
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV sphere centred at origin."""
    pos_list, nrm_list, idx_list = [], [], []
    for i in range(stacks + 1):
        phi = math.pi * i / stacks
        for j in range(slices + 1):
            theta = 2 * math.pi * j / slices
            x = math.sin(phi) * math.cos(theta)
            y = math.cos(phi)
            z = math.sin(phi) * math.sin(theta)
            pos_list.append((x * radius, y * radius, z * radius))
            nrm_list.append((x, y, z))
    for i in range(stacks):
        for j in range(slices):
            a = i * (slices + 1) + j
            b = a + slices + 1
            idx_list += [a, b, a+1, b, b+1, a+1]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _prism(sides: int = 12, radius: float = 0.5, height: float = 1.0
           ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upright prism (cylinder-like) centred at origin."""
    pos_list, nrm_list, idx_list = [], [], []
    hy = height * 0.5
    # Side faces
    for i in range(sides):
        a0 = 2 * math.pi * i / sides
        a1 = 2 * math.pi * (i + 1) / sides
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        # Average normal
        mx = (x0 + x1) * 0.5
        mz = (z0 + z1) * 0.5
        ln = math.sqrt(mx*mx + mz*mz) or 1.0
        nx, nz = mx/ln, mz/ln
        base = len(pos_list)
        for p, q in [(x0, z0), (x1, z1), (x1, z1), (x0, z0)]:
            y = hy if len(pos_list) - base < 2 else -hy
            pos_list.append((p, y, q))
            nrm_list.append((nx, 0, nz))
        idx_list += [base, base+1, base+2, base, base+2, base+3]
    # Cap faces (top +hy, bottom -hy)
    for sign, ny in ((1, 1.0), (-1, -1.0)):
        cx = len(pos_list)
        pos_list.append((0, sign * hy, 0))
        nrm_list.append((0, ny, 0))
        for i in range(sides):
            a = 2 * math.pi * i / sides
            pos_list.append((math.cos(a)*radius, sign*hy, math.sin(a)*radius))
            nrm_list.append((0, ny, 0))
        for i in range(sides):
            v0 = cx + 1 + i
            v1 = cx + 1 + (i + 1) % sides
            if sign > 0:
                idx_list += [cx, v0, v1]
            else:
                idx_list += [cx, v1, v0]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _cone(sides: int = 12, radius: float = 0.5, height: float = 1.0
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upright cone, base centred at origin, apex at +height."""
    pos_list, nrm_list, idx_list = [], [], []
    # Side faces (tip → base edge)
    slope = radius / height   # tan of half-angle
    ny_side = slope / math.sqrt(1.0 + slope * slope)
    for i in range(sides):
        a0 = 2 * math.pi * i / sides
        a1 = 2 * math.pi * (i + 1) / sides
        x0, z0 = math.cos(a0) * radius, math.sin(a0) * radius
        x1, z1 = math.cos(a1) * radius, math.sin(a1) * radius
        base = len(pos_list)
        pos_list += [(0, height, 0), (x0, 0, z0), (x1, 0, z1)]
        am = (a0 + a1) * 0.5
        nx, nz = math.cos(am), math.sin(am)
        ln = math.sqrt(nx*nx + ny_side*ny_side + nz*nz)
        nrm_list += [(nx/ln, ny_side/ln, nz/ln)] * 3
        idx_list += [base, base+1, base+2]
    # Base disk
    cx = len(pos_list)
    pos_list.append((0, 0, 0))
    nrm_list.append((0, -1, 0))
    for i in range(sides):
        a = 2 * math.pi * i / sides
        pos_list.append((math.cos(a)*radius, 0, math.sin(a)*radius))
        nrm_list.append((0, -1, 0))
    for i in range(sides):
        v0 = cx + 1 + i
        v1 = cx + 1 + (i + 1) % sides
        idx_list += [cx, v1, v0]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _tetrahedron(r: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular tetrahedron inscribed in a sphere of radius r."""
    v = [
        (0, r, 0),
        (r * math.sqrt(8/9), -r/3, 0),
        (-r * math.sqrt(2/9), -r/3,  r * math.sqrt(2/3)),
        (-r * math.sqrt(2/9), -r/3, -r * math.sqrt(2/3)),
    ]
    faces = [(0,1,2), (0,2,3), (0,3,1), (1,3,2)]
    pos_list, nrm_list, idx_list = [], [], []
    for f in faces:
        base = len(pos_list)
        pts = [np.array(v[i]) for i in f]
        n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        n = n / (np.linalg.norm(n) or 1.0)
        for p in pts:
            pos_list.append(tuple(p.tolist()))
            nrm_list.append(tuple(n.tolist()))
        idx_list += [base, base+1, base+2]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _icosahedron(r: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Icosahedron with 20 faces."""
    t = (1.0 + math.sqrt(5.0)) / 2.0
    raw_verts = [
        (-1,  t, 0), ( 1,  t, 0), (-1, -t, 0), ( 1, -t, 0),
        (0, -1,  t), (0,  1,  t), (0, -1, -t), (0,  1, -t),
        ( t, 0, -1), ( t, 0,  1), (-t, 0, -1), (-t, 0,  1),
    ]
    scale = r / math.sqrt(1.0 + t * t)
    raw_verts = [(x*scale, y*scale, z*scale) for x, y, z in raw_verts]
    raw_faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    pos_list, nrm_list, idx_list = [], [], []
    for f in raw_faces:
        base = len(pos_list)
        pts = [np.array(raw_verts[i]) for i in f]
        n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        n = n / (np.linalg.norm(n) or 1.0)
        for p in pts:
            pos_list.append(tuple(p.tolist()))
            nrm_list.append(tuple(n.tolist()))
        idx_list += [base, base+1, base+2]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _octahedron(r: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regular octahedron."""
    raw_verts = [
        (r,0,0),(-r,0,0),(0,r,0),(0,-r,0),(0,0,r),(0,0,-r)
    ]
    raw_faces = [
        (0,2,4),(0,4,3),(0,3,5),(0,5,2),
        (1,4,2),(1,3,4),(1,5,3),(1,2,5),
    ]
    pos_list, nrm_list, idx_list = [], [], []
    for f in raw_faces:
        base = len(pos_list)
        pts = [np.array(raw_verts[i]) for i in f]
        n = np.cross(pts[1]-pts[0], pts[2]-pts[0])
        n = n / (np.linalg.norm(n) or 1.0)
        for p in pts:
            pos_list.append(tuple(p.tolist()))
            nrm_list.append(tuple(n.tolist()))
        idx_list += [base, base+1, base+2]
    return (
        np.array(pos_list, dtype=np.float32),
        np.array(nrm_list, dtype=np.float32),
        np.array(idx_list, dtype=np.uint32),
    )


def _unit_geometry(shape: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (positions, normals, indices) for a unit-scale mesh matching *shape*.

    All returned meshes fit within a unit bounding box centred at the origin
    (before entity-level scaling is applied).
    """
    shape = shape.lower()
    if shape in ("cube", "building", "structure", "default"):
        return _box(1.0, 1.0, 1.0)
    if shape in ("cylinder", "column", "road", "pipe"):
        return _prism(12, 0.5, 1.0)
    if shape in ("cone", "tower"):
        return _cone(12, 0.5, 1.0)
    if shape in ("sphere", "landmark"):
        return _uv_sphere(0.5)
    if shape in ("tetra",):
        return _tetrahedron(0.5)
    if shape in ("icosa",):
        return _icosahedron(0.5)
    if shape in ("octa",):
        return _octahedron(0.5)
    # fallback
    return _box(1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# Binary buffer helpers
# ---------------------------------------------------------------------------

def _pad4(data: bytes) -> bytes:
    """Pad *data* to a 4-byte boundary with zero bytes."""
    rem = len(data) % 4
    return data if rem == 0 else data + b"\x00" * (4 - rem)


class _BinBuffer:
    """Incrementally accumulates binary data chunks and their byte offsets."""

    def __init__(self) -> None:
        self._chunks: List[bytes] = []
        self._offset: int = 0

    def append(self, data: np.ndarray) -> Tuple[int, int]:
        """Append *data* (aligned to 4 bytes) and return (byteOffset, byteLength)."""
        raw = data.tobytes()
        padded = _pad4(raw)
        byte_offset = self._offset
        self._chunks.append(padded)
        self._offset += len(padded)
        return byte_offset, len(raw)

    def bytes(self) -> bytes:
        return b"".join(self._chunks)

    @property
    def total_length(self) -> int:
        return self._offset


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_glb(entities: List[Dict[str, Any]]) -> bytes:
    """Build a binary glTF 2.0 file containing all scene entities.

    Parameters
    ----------
    entities:
        List of entity dicts as produced by
        :func:`~cesium_exporter.scene_to_geo.scene_to_entities`.
        Each entry must include:

        * ``enu_offset`` — ``{"east": float, "north": float, "up": float}``
        * ``entity_height_m`` — float, visual height of the entity (m)
        * ``scale_m`` — float, horizontal scale in metres
        * ``color`` — ``[r, g, b]`` each in [0, 1]
        * ``shape`` — shape name string

    Returns
    -------
    bytes
        Raw GLB binary data ready to write to a ``.glb`` file.

    Notes
    -----
    The GLB positions are in the tile's **local ENU frame** using the
    y-up glTF convention: x = east, y = up, z = −north.  The caller is
    responsible for supplying a matching ``tile.transform`` in the
    ``tileset.json`` (see :func:`~cesium_exporter.geo_anchor.enu_to_ecef_transform`).
    """
    if not entities:
        # Return a minimal valid empty GLB
        return _build_empty_glb()

    buf = _BinBuffer()
    buffer_views: List[Dict] = []
    accessors: List[Dict] = []
    materials: List[Dict] = []
    meshes: List[Dict] = []
    nodes: List[Dict] = []

    def _add_bv(data: np.ndarray, target: int) -> int:
        byte_offset, byte_length = buf.append(data)
        buffer_views.append({
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": byte_length,
            "target": target,
        })
        return len(buffer_views) - 1

    def _add_accessor(bv_idx: int, comp_type: int, count: int,
                      acc_type: str, **kwargs) -> int:
        acc: Dict[str, Any] = {
            "bufferView": bv_idx,
            "byteOffset": 0,
            "componentType": comp_type,
            "count": count,
            "type": acc_type,
        }
        acc.update(kwargs)
        accessors.append(acc)
        return len(accessors) - 1

    for ent in entities:
        enu = ent.get("enu_offset", {})
        east_m  = float(enu.get("east",  0.0))
        north_m = float(enu.get("north", 0.0))
        up_m    = float(enu.get("up",    0.0))
        scale_h = float(ent.get("entity_height_m", 20.0))
        scale_w = float(ent.get("scale_m",         20.0))
        color   = ent.get("color", [0.7, 0.7, 0.7])
        shape   = ent.get("shape", "cube")

        # Retrieve unit-scale geometry
        pos, nrm, idx = _unit_geometry(shape)
        # Scale geometry: width/depth = scale_w, height = scale_h
        pos = pos.copy()
        pos[:, 0] *= scale_w   # x = east dimension
        pos[:, 1] *= scale_h   # y = up  dimension (y-up)
        pos[:, 2] *= scale_w   # z = -north dimension

        # Move to entity location (y-up: x=east, y=up, z=-north)
        # Place base of object at ground level (up_m above origin)
        pos[:, 0] += east_m
        pos[:, 1] += up_m + scale_h * 0.5    # centre of bounding box
        pos[:, 2] += -north_m

        # POSITION accessor
        pos_min = pos.min(axis=0).tolist()
        pos_max = pos.max(axis=0).tolist()
        pos_bv = _add_bv(pos, _TARGET_ARRAY_BUFFER)
        pos_acc = _add_accessor(pos_bv, _COMPONENT_FLOAT, len(pos), "VEC3",
                                min=pos_min, max=pos_max)

        # NORMAL accessor
        nrm_bv = _add_bv(nrm, _TARGET_ARRAY_BUFFER)
        nrm_acc = _add_accessor(nrm_bv, _COMPONENT_FLOAT, len(nrm), "VEC3")

        # INDEX accessor (use uint32 to handle large meshes safely)
        idx32 = idx.astype(np.uint32)
        idx_bv = _add_bv(idx32, _TARGET_ELEMENT_ARRAY_BUFFER)
        idx_acc = _add_accessor(idx_bv, _COMPONENT_UNSIGNED_INT, len(idx32), "SCALAR",
                                min=[int(idx32.min())], max=[int(idx32.max())])

        # Material
        r, g, b = float(color[0]), float(color[1]), float(color[2])
        mat_idx = len(materials)
        materials.append({
            "pbrMetallicRoughness": {
                "baseColorFactor": [r, g, b, 1.0],
                "metallicFactor": 0.05,
                "roughnessFactor": 0.75,
            },
            "doubleSided": False,
        })

        # Mesh
        mesh_idx = len(meshes)
        meshes.append({
            "primitives": [{
                "attributes": {
                    "POSITION": pos_acc,
                    "NORMAL":   nrm_acc,
                },
                "indices":  idx_acc,
                "material": mat_idx,
                "mode":     4,   # TRIANGLES
            }]
        })

        nodes.append({"mesh": mesh_idx})

    # Assemble glTF JSON
    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0", "generator": "WorldCompiler/cesium_exporter"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": buf.total_length}],
        "materials": materials,
    }

    return _pack_glb(gltf, buf.bytes())


def _build_empty_glb() -> bytes:
    """Return a minimal valid empty GLB (no geometry)."""
    gltf: Dict[str, Any] = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": []}],
    }
    return _pack_glb(gltf, b"")


def _pack_glb(gltf_dict: Dict[str, Any], bin_data: bytes) -> bytes:
    """Pack a glTF JSON dict and binary buffer into GLB format."""
    json_raw = json.dumps(gltf_dict, separators=(",", ":")).encode("utf-8")
    json_padded = _pad4(json_raw)
    # Pad JSON with spaces to keep valid UTF-8
    json_padded = json_raw + b" " * (len(json_padded) - len(json_raw))
    bin_padded = _pad4(bin_data)

    # GLB header (12 bytes)
    total = 12
    total += 8 + len(json_padded)   # chunk 0
    if bin_padded:
        total += 8 + len(bin_padded)  # chunk 1

    header = struct.pack("<III", _GLB_MAGIC, _GLB_VERSION, total)
    json_chunk = struct.pack("<II", len(json_padded), _CHUNK_JSON) + json_padded

    parts = [header, json_chunk]
    if bin_padded:
        bin_chunk = struct.pack("<II", len(bin_padded), _CHUNK_BIN) + bin_padded
        parts.append(bin_chunk)

    return b"".join(parts)
