"""scene_to_geo.py — map a compiled WorldScene to geospatial entities.

This module defines the **canonical intermediate format** (a plain Python
``dict`` / JSON-serialisable structure) that sits between the
:class:`~world_compiler.WorldScene` compiler output and the Cesium export
pipeline.

Intermediate format (``entities.json``)
-----------------------------------------
.. code-block:: json

    {
      "title": "My World",
      "fingerprint": "abc123",
      "origin": {"lat": 51.5, "lon": -0.1, "height": 0.0},
      "spread_m": 300.0,
      "stats": {...},
      "entities": [
        {
          "id": "motif-0",
          "type": "building",
          "label": "city place architecture",
          "position": {"lat": 51.501, "lon": -0.098, "height": 45.0},
          "enu_offset": {"east": 30.5, "north": 12.1, "up": 22.5},
          "scale_m": 28.4,
          "color": [0.82, 0.46, 0.25],
          "shape": "cube",
          "mass": 0.14,
          "members": 7
        }
      ]
    }

``type`` is inferred from ``shape`` and semantic keywords in ``label``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from .geo_anchor import enu_to_lla

# How much of the WorldScene's normalised [-3, 3] position range maps to
# the requested spread.  The scene spread is ~3 units on each side, so
# dividing spread_m by 3 gives metres-per-unit.
_SCENE_HALF_RANGE: float = 3.0

# Maximum building height in metres (at maximum scale).
_MAX_HEIGHT_M: float = 120.0
_MIN_HEIGHT_M: float = 8.0

# Metres-per-unit for the height dimension.
_HEIGHT_SCALE: float = 30.0


def _entity_type(shape: str, label: str) -> str:
    """Infer a semantic entity type from the object's shape and label text."""
    label_lower = label.lower()
    building_kw = {"building", "city", "town", "house", "room", "structure",
                   "architecture", "place", "domain", "location", "region"}
    road_kw = {"road", "path", "street", "track", "route", "channel",
               "pipeline", "stream", "flow"}
    terrain_kw = {"ground", "terrain", "land", "earth", "floor", "surface",
                  "hill", "mountain", "valley", "plain"}
    prop_kw = {"object", "prop", "thing", "item", "artefact", "artifact",
               "element", "symbol", "marker"}
    tokens = set(label_lower.split())
    if tokens & building_kw or shape in ("cube",):
        return "building"
    if tokens & road_kw or shape in ("cylinder",):
        return "road"
    if tokens & terrain_kw:
        return "terrain"
    if tokens & prop_kw or shape in ("tetra", "octa", "icosa"):
        return "prop"
    if shape in ("sphere",):
        return "landmark"
    if shape in ("cone",):
        return "tower"
    return "structure"


def scene_to_entities(
    scene: Any,
    origin_lat: float,
    origin_lon: float,
    origin_height: float = 0.0,
    spread_m: float = 300.0,
) -> Dict[str, Any]:
    """Convert a :class:`~world_compiler.WorldScene` to geospatial entity dict.

    Parameters
    ----------
    scene:
        Compiled :class:`~world_compiler.WorldScene`.
    origin_lat, origin_lon:
        WGS-84 geodetic origin of the scene (degrees).
    origin_height:
        Ellipsoidal height of the scene origin (metres).
    spread_m:
        Determines how far apart entities are placed (metres).  The
        outermost objects (scene position ±3.0) will be placed
        ``spread_m`` metres from the origin.

    Returns
    -------
    dict
        JSON-serialisable dictionary matching the canonical intermediate
        format documented in the module docstring.
    """
    metres_per_unit = spread_m / _SCENE_HALF_RANGE
    entities: List[Dict[str, Any]] = []

    for obj in scene.objects:
        px, py, pz = obj.position          # normalised scene coords [-3, 3]
        east_m  = px * metres_per_unit
        north_m = py * metres_per_unit
        # Height: clamp negative values to 0 (underground = ground level)
        raw_h = pz * _HEIGHT_SCALE
        height_m = max(0.0, raw_h)
        # Entity height (visual bounding box height) from scale + mass
        entity_h = _MIN_HEIGHT_M + (_MAX_HEIGHT_M - _MIN_HEIGHT_M) * math.sqrt(obj.mass)
        entity_h = max(entity_h, _MIN_HEIGHT_M)

        # Geodetic position of entity base
        lat, lon, h = enu_to_lla(east_m, north_m, origin_height + height_m,
                                  origin_lat, origin_lon, origin_height)

        entity_type = _entity_type(obj.shape, obj.label)
        scale_m = _MIN_HEIGHT_M + (obj.scale - 0.35) / 1.65 * (_MAX_HEIGHT_M - _MIN_HEIGHT_M)
        scale_m = max(_MIN_HEIGHT_M * 0.5, scale_m)

        entities.append({
            "id": obj.id,
            "type": entity_type,
            "label": obj.label,
            "position": {
                "lat": round(lat, 8),
                "lon": round(lon, 8),
                "height": round(h, 2),
            },
            "enu_offset": {
                "east":  round(east_m, 3),
                "north": round(north_m, 3),
                "up":    round(height_m, 3),
            },
            "entity_height_m": round(entity_h, 2),
            "scale_m": round(scale_m, 2),
            "color": [round(c, 4) for c in obj.color],
            "shape": obj.shape,
            "mass": round(obj.mass, 6),
            "members": obj.members,
        })

    return {
        "title": scene.title,
        "fingerprint": scene.fingerprint,
        "origin": {
            "lat": origin_lat,
            "lon": origin_lon,
            "height": origin_height,
        },
        "spread_m": spread_m,
        "stats": scene.stats,
        "entities": entities,
    }
