"""grv2_runtime/geo_export.py — Runtime state -> the real, free 3D globe.

cesium_exporter/ (geo_anchor, gltf_builder, tiles_writer) is reused entirely
as built -- real WGS-84 georeferencing, real glTF/3D-Tiles export, the free
CesiumJS+OSM viewer already in viewer/. This module's only job is
translating grv2_runtime.Runtime state (SGR entities + their wiring points +
current texture color + resonance) into the entities-dict schema
tiles_writer.write_tileset already expects. The only change made to
cesium_exporter itself is gltf_builder's new point-cloud path (see
wiring.py's module docstring for why: real wiring is a point cloud, not one
of the primitive shapes cesium_exporter originally only knew how to emit).
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cesium_exporter.geo_anchor import enu_to_lla       # noqa: E402
from cesium_exporter.gltf_builder import build_glb      # noqa: E402
from cesium_exporter.tiles_writer import write_tileset   # noqa: E402

from . import texture as texture_mod

# London -- matches cesium_exporter's own CLI/documented default.
DEFAULT_ORIGIN_LAT = 51.5074
DEFAULT_ORIGIN_LON = -0.1278
DEFAULT_ORIGIN_HEIGHT = 0.0


def build_entities_dict(runtime, origin_lat: float = DEFAULT_ORIGIN_LAT,
                        origin_lon: float = DEFAULT_ORIGIN_LON,
                        origin_height: float = DEFAULT_ORIGIN_HEIGHT) -> Dict[str, Any]:
    """Translate the runtime's current entities into the entities-dict schema
    cesium_exporter.tiles_writer/gltf_builder already expect, with real
    wiring point clouds instead of primitive shapes."""
    entities = []
    for ent in runtime.kernel.get_all_entities():
        word = ent.properties.get("label", ent.type)
        wiring = runtime.wiring_bank.recall(word)
        tex = runtime._texture_by_entity.get(ent.id)
        if tex is None:
            color = texture_mod.color_from_hash(word)
            palette = texture_mod.palette_from_hash(word)
        else:
            color, palette = tex.color, tex.palette

        east_m, _up_ignored, north_m = ent.position
        up_m = float(max(0.0, wiring.points[:, 1].max() * 0.5)) if wiring.node_count else 0.0
        lat, lon, height = enu_to_lla(east_m, north_m, up_m, origin_lat, origin_lon, origin_height)

        resonance = runtime._resonance_by_entity.get(ent.id, 0.0)
        try:
            mass = float(ent.properties.get("mass", "0") or 0.0)
        except ValueError:
            mass = 0.0

        entities.append({
            "id": ent.id,
            "type": ent.type,
            "label": word,
            "position": {"lat": round(lat, 8), "lon": round(lon, 8), "height": round(height, 2)},
            "enu_offset": {"east": round(east_m, 3), "north": round(north_m, 3), "up": round(up_m, 3)},
            "points": wiring.points.tolist(),
            "color": [round(float(c), 4) for c in color],
            "palette": [[round(float(c), 4) for c in tone] for tone in palette],
            "resonance": round(float(resonance), 4),
            "mass": round(mass, 6),
            "members": wiring.node_count,
            "thermal_cost": round(float(wiring.thermal_cost), 4),
            "wiring_source": wiring.source,
        })

    return {
        "title": "grv2_runtime -- live mind state",
        "fingerprint": runtime.kernel.get_merkle_root(),
        "origin": {"lat": origin_lat, "lon": origin_lon, "height": origin_height},
        "spread_m": 700.0,
        "stats": {"tick": runtime.tick, "entity_count": len(entities)},
        "entities": entities,
    }


def export_scene(runtime, output_dir: str = "cesium_output",
                 origin_lat: float = DEFAULT_ORIGIN_LAT,
                 origin_lon: float = DEFAULT_ORIGIN_LON,
                 origin_height: float = DEFAULT_ORIGIN_HEIGHT) -> Dict[str, Any]:
    """Regenerate <output_dir>/{entities.json,scene.glb,tileset.json} from the
    runtime's current state. Returns the entities dict written."""
    entities_dict = build_entities_dict(runtime, origin_lat, origin_lon, origin_height)
    glb_bytes = build_glb(entities_dict["entities"])
    write_tileset(output_dir=output_dir, entities_dict=entities_dict, glb_bytes=glb_bytes)
    return entities_dict
