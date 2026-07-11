"""cli.py — command-line entrypoint for the cesium_exporter pipeline.

Usage
-----
Via module execution::

    python -m cesium_exporter input.txt -o cesium_output --lat 51.5 --lon -0.09

Via script reference::

    python cesium_exporter/cli.py input.txt ...

Or from a semantic JSON file (pre-compiled scene)::

    python -m cesium_exporter scene.json -o cesium_output --from-json

Arguments
---------
positional:
    ``input``           path to a text file  (or ``-`` for stdin)
                        or a JSON file when ``--from-json`` is set

optional:
    ``-o / --output``   output directory (default: ``cesium_output``)
    ``--lat``           origin latitude  (default: 51.5074 — London)
    ``--lon``           origin longitude (default: -0.1278 — London)
    ``--height``        origin ellipsoidal height in m (default: 0)
    ``--spread``        scene spread in metres (default: 300)
    ``--seed``          compiler RNG seed (default: 0)
    ``--from-json``     treat *input* as a pre-compiled ``entities.json``
                        and skip the compile step (only re-exports GLB +
                        tileset.json)
    ``--no-pretrained`` disable pretrained semantic channel even if
                        sentence-transformers is installed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cesium_exporter",
        description=(
            "Compile text into a geospatial 3D world and export "
            "CesiumJS-compatible assets (glTF + 3D Tiles)."
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        metavar="INPUT",
        help="path to a text file (default: stdin); or entities.json with --from-json",
    )
    parser.add_argument(
        "-o", "--output",
        default="cesium_output",
        metavar="DIR",
        help="output directory (default: cesium_output)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=51.5074,
        metavar="LAT",
        help="origin latitude in degrees WGS-84 (default: 51.5074 — London)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=-0.1278,
        metavar="LON",
        help="origin longitude in degrees WGS-84 (default: -0.1278 — London)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=0.0,
        metavar="M",
        help="origin ellipsoidal height in metres (default: 0)",
    )
    parser.add_argument(
        "--spread",
        type=float,
        default=300.0,
        metavar="M",
        help="scene spread in metres — half-width of the generated scene (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="compiler RNG seed for deterministic output (default: 0)",
    )
    parser.add_argument(
        "--from-json",
        action="store_true",
        dest="from_json",
        help="INPUT is a pre-compiled entities.json — skip text compilation",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        dest="no_pretrained",
        help="disable pretrained semantic channel (sentence-transformers)",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:  # noqa: C901
    """Entry point; returns exit code."""
    args = _parse_args(argv)

    # ------------------------------------------------------------------ input
    if args.input and args.input != "-":
        src = Path(args.input).read_text(encoding="utf-8")
    else:
        src = sys.stdin.read()

    if args.from_json:
        # Re-export from existing entities.json (no compilation)
        entities_dict = json.loads(src)
        print(
            f"[cesium_exporter] re-exporting {len(entities_dict.get('entities', []))} "
            "entities from JSON …"
        )
    else:
        # ----------------------------------------------------------- compile
        # Import here to keep startup fast when called with --help
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).parent.parent))
        from world_compiler import WorldCompiler, CompilerConfig

        print(f"[cesium_exporter] compiling text ({len(src)} chars) …")
        cfg = CompilerConfig(
            seed=args.seed,
            pretrained_dim=0 if args.no_pretrained else 8,
        )
        try:
            scene = WorldCompiler(cfg).compile(src)
        except Exception as exc:
            print(f"[cesium_exporter] compile error: {exc}", file=sys.stderr)
            return 1

        print(
            f"[cesium_exporter] compiled {scene.stats.get('chunks', '?')} chunks "
            f"→ {len(scene.objects)} motifs  (fingerprint: {scene.fingerprint})"
        )

        # ---------------------------------------- scene → geospatial entities
        from cesium_exporter.scene_to_geo import scene_to_entities

        entities_dict = scene_to_entities(
            scene,
            origin_lat=args.lat,
            origin_lon=args.lon,
            origin_height=args.height,
            spread_m=args.spread,
        )

    # ----------------------------------------------------------- build assets
    from cesium_exporter.gltf_builder import build_glb
    from cesium_exporter.tiles_writer import write_tileset

    glb_bytes = build_glb(entities_dict["entities"])
    write_tileset(
        output_dir=args.output,
        entities_dict=entities_dict,
        glb_bytes=glb_bytes,
    )

    print()
    print(f"[cesium_exporter] done → {Path(args.output).resolve()}/")
    print()
    print("  Next steps:")
    print("    1.  python serve.py")
    print("    2.  Open http://localhost:8080/viewer/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
