#!/usr/bin/env python3
"""serve.py — minimal local development server for the World Compiler globe viewer.

Serves the entire repository root as static files with permissive CORS headers
so the CesiumJS viewer can fetch generated artifacts from ``cesium_output/``
(or any custom output directory) without cross-origin errors.

Usage
-----
::

    # Basic (serves from repo root on port 8080)
    python serve.py

    # Custom port
    python serve.py --port 9000

    # Custom output directory (if you used -o when compiling)
    python serve.py --output my_output

After starting the server, open:

    http://localhost:8080/viewer/

Quick end-to-end workflow
-------------------------
::

    # 1. Compile a text file to geospatial assets
    python -m cesium_exporter my_world.txt -o cesium_output --lat 51.5 --lon -0.09

    # 2. Start the local server
    python serve.py

    # 3. Open the viewer
    #    http://localhost:8080/viewer/
"""
from __future__ import annotations

import argparse
import http.server
import os
import socketserver
import sys
from pathlib import Path

_MIME_EXTRAS = {
    ".glb": "model/gltf-binary",
    ".gltf": "model/gltf+json",
    ".json": "application/json",
    ".js":   "application/javascript",
    ".html": "text/html",
    ".css":  "text/css",
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".ico":  "image/x-icon",
}


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with CORS and extra MIME types."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()

    def guess_type(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext in _MIME_EXTRAS:
            return _MIME_EXTRAS[ext]
        return super().guess_type(path)  # type: ignore[arg-type]

    def log_message(self, fmt: str, *args) -> None:
        # Suppress noisy 200 OK lines; only show errors and warnings
        if args and str(args[1]) not in ("200", "304"):
            super().log_message(fmt, *args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Static file server for the World Compiler globe viewer."
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="port to listen on (default: 8080)",
    )
    parser.add_argument(
        "--output",
        default="cesium_output",
        help="generated output directory name (for display only; default: cesium_output)",
    )
    args = parser.parse_args()

    # Serve from the repository root so all paths are consistent
    repo_root = Path(__file__).parent.resolve()
    os.chdir(repo_root)

    output_dir = repo_root / args.output
    viewer_dir = repo_root / "viewer"

    print()
    print("  World Compiler — Globe Viewer")
    print("  " + "─" * 42)
    print(f"  Serving:    {repo_root}/")
    print(f"  Viewer:     {viewer_dir}/")
    print(f"  Output dir: {output_dir}/")
    if not output_dir.exists():
        print()
        print("  ⚠  Output directory not found.")
        print("  Run the pipeline first:")
        print(f"     python -m cesium_exporter <input.txt> -o {args.output}")
    print()
    print(f"  → Open:  http://localhost:{args.port}/viewer/")
    print()
    print("  Press Ctrl-C to stop.")
    print()

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", args.port), _CORSHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")
            sys.exit(0)


if __name__ == "__main__":
    main()
