"""cesium_exporter — World Compiler → CesiumJS globe pipeline.

Transforms a compiled :class:`~world_compiler.WorldScene` into geospatial
assets that can be previewed in the bundled CesiumJS viewer (``viewer/``).

Typical usage (programmatic)::

    from world_compiler import WorldCompiler
    from cesium_exporter import export_scene

    scene = WorldCompiler().compile(text)
    export_scene(scene, output_dir="cesium_output", origin_lat=51.5, origin_lon=-0.09)

Or via the CLI::

    python -m cesium_exporter input.txt -o cesium_output --lat 51.5 --lon -0.09

Then serve and view::

    python serve.py
    # open http://localhost:8080/viewer/
"""
from __future__ import annotations

from .scene_to_geo import scene_to_entities
from .tiles_writer import write_tileset

__all__ = ["scene_to_entities", "write_tileset", "export_scene"]


def export_scene(
    scene,
    output_dir: str = "cesium_output",
    origin_lat: float = 51.5074,
    origin_lon: float = -0.1278,
    origin_height: float = 0.0,
    spread_m: float = 300.0,
) -> None:
    """Full pipeline: WorldScene → geospatial JSON + GLB + tileset.json.

    Parameters
    ----------
    scene:
        A :class:`~world_compiler.WorldScene` returned by
        :meth:`~world_compiler.WorldCompiler.compile`.
    output_dir:
        Directory to write output files into (created if absent).
    origin_lat, origin_lon:
        WGS-84 geodetic coordinates of the scene origin (degrees).
    origin_height:
        Ellipsoidal height of the origin in metres.
    spread_m:
        Half-width of the scene in metres.  Entity positions (which are
        normalised to [−3, 3] in the compiled scene) are scaled so the
        outermost objects are roughly ``spread_m`` metres from the origin.
    """
    from .gltf_builder import build_glb

    entities_dict = scene_to_entities(
        scene,
        origin_lat=origin_lat,
        origin_lon=origin_lon,
        origin_height=origin_height,
        spread_m=spread_m,
    )
    glb_bytes = build_glb(entities_dict["entities"])
    write_tileset(
        output_dir=output_dir,
        entities_dict=entities_dict,
        glb_bytes=glb_bytes,
    )
