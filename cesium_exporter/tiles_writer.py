"""tiles_writer.py — write a 3D Tiles 1.1-compatible output directory.

Output structure
----------------
::

    <output_dir>/
      entities.json      canonical intermediate format (WorldScene → geospatial)
      scene.glb          glTF 2.0 binary with all entity geometries
      tileset.json       3D Tiles manifest (Cesium3DTileset entry point)

``tileset.json`` format
-----------------------
Follows the `OGC 3D Tiles 1.1 specification
<https://docs.ogc.org/cs/22-025r4/22-025r4.html>`_.  The tile's
``transform`` places the GLB's local ENU frame at the correct ECEF position
so Cesium can overlay the generated content on the world globe.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict

from .geo_anchor import ecef_bounding_sphere, enu_to_ecef_transform


def _scene_radius(entities: list, spread_m: float) -> float:
    """Approximate bounding-sphere radius for the scene (metres)."""
    if not entities:
        return max(spread_m, 10.0)
    max_d = 0.0
    for ent in entities:
        enu = ent.get("enu_offset", {})
        e = float(enu.get("east",  0.0))
        n = float(enu.get("north", 0.0))
        u = float(enu.get("up",    0.0)) + float(ent.get("entity_height_m", 0.0))
        d = math.sqrt(e*e + n*n + u*u) + float(ent.get("scale_m", 20.0))
        max_d = max(max_d, d)
    return max(max_d * 1.2, 10.0)


def write_tileset(
    output_dir: str,
    entities_dict: Dict[str, Any],
    glb_bytes: bytes,
    glb_filename: str = "scene.glb",
) -> None:
    """Write all output artefacts to *output_dir*.

    Parameters
    ----------
    output_dir:
        Destination directory (created if absent).
    entities_dict:
        The canonical intermediate dict as returned by
        :func:`~cesium_exporter.scene_to_geo.scene_to_entities`.
    glb_bytes:
        Raw GLB binary as returned by
        :func:`~cesium_exporter.gltf_builder.build_glb`.
    glb_filename:
        File name for the GLB asset inside *output_dir*.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- entities.json -------------------------------------------------------
    entities_path = out / "entities.json"
    entities_path.write_text(
        json.dumps(entities_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- scene.glb -----------------------------------------------------------
    glb_path = out / glb_filename
    glb_path.write_bytes(glb_bytes)

    # --- tileset.json --------------------------------------------------------
    origin = entities_dict.get("origin", {})
    lat    = float(origin.get("lat",    0.0))
    lon    = float(origin.get("lon",    0.0))
    h      = float(origin.get("height", 0.0))
    spread = float(entities_dict.get("spread_m", 300.0))

    entities = entities_dict.get("entities", [])
    radius = _scene_radius(entities, spread)

    # The tile transform maps the GLB's local ENU frame (x=east, y=up, z=-north
    # after glTF y-up correction) to ECEF.  For Cesium, the convention in 3D
    # Tiles is that the transform maps the tile's local frame to ECEF using the
    # standard ENU orientation.
    transform = enu_to_ecef_transform(lat, lon, h)

    # Bounding volume expressed in ECEF (sphere)
    bounding_sphere = ecef_bounding_sphere(lat, lon, h + radius * 0.5, radius)

    # Geometric error: rough angular measure appropriate for city-scale content.
    # A lower value = switch to this tile sooner (higher priority).
    geometric_error = max(radius * 0.1, 5.0)

    tileset: Dict[str, Any] = {
        "asset": {
            "version": "1.1",
            "tilesetVersion": "1.0.0",
            "extras": {
                "generator": "WorldCompiler/cesium_exporter",
                "title":     entities_dict.get("title", ""),
                "fingerprint": entities_dict.get("fingerprint", ""),
            },
        },
        "geometricError": geometric_error,
        "root": {
            "transform": transform,
            "boundingVolume": {
                "sphere": bounding_sphere,
            },
            "geometricError": geometric_error,
            "refine": "ADD",
            "content": {
                "uri": glb_filename,
            },
        },
    }

    tileset_path = out / "tileset.json"
    tileset_path.write_text(
        json.dumps(tileset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[cesium_exporter] wrote {len(entities)} entities to {out}/")
    print(f"  {entities_path.name}  ({entities_path.stat().st_size:,} B)")
    print(f"  {glb_path.name}  ({glb_path.stat().st_size:,} B)")
    print(f"  {tileset_path.name}  ({tileset_path.stat().st_size:,} B)")
    print(f"  bounding sphere radius: {radius:.0f} m  origin: {lat:.4f}°, {lon:.4f}°")
