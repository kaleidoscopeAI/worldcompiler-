"""grv2_runtime/server.py — serve a Runtime: seed it, act on it, watch its mind.

    python -m grv2_runtime.server --seed "the wolf sits on the hill"

A stdlib-only HTTP server (no new dependencies), same pattern as the
repo's existing world_server.py. After every POST /action, the runtime's
current state is exported through geo_export.export_scene() into
<repo_root>/cesium_output/ -- the exact directory the repo's existing
serve.py + viewer/ (free CesiumJS + OSM, no token) already expect. So the
actual way to *watch* this is two processes:

    python -m grv2_runtime.server --seed "..."   # drives the mind
    python serve.py                              # serves the repo + viewer/
    open http://localhost:8080/viewer/           # see it live

This module owns none of the rendering -- it only ever writes the same
entities.json/scene.glb/tileset.json cesium_exporter already knows how to
produce, now carrying real wiring point clouds (see geo_export.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from . import geo_export                # noqa: E402
from .runtime import Runtime            # noqa: E402

_DEFAULT_OUTPUT_DIR = os.path.join(_REPO_ROOT, "cesium_output")


class _Handler(BaseHTTPRequestHandler):
    runtime: Optional[Runtime] = None
    output_dir: str = _DEFAULT_OUTPUT_DIR

    def log_message(self, fmt: str, *args) -> None:
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_OPTIONS(self) -> None:
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send(200, {"ok": True, "seeded": self.runtime is not None})
        elif self.path == "/state":
            if self.runtime is None:
                self._send(400, {"error": "not seeded yet -- POST /seed first"})
                return
            self._send(200, self.runtime.state_dict())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"invalid JSON: {e}"})
            return

        if self.path == "/seed":
            text = str(body.get("text", "")).strip()
            if not text:
                self._send(400, {"error": "text required"})
                return
            type(self).runtime = Runtime(text)
            entities_dict = geo_export.export_scene(type(self).runtime, output_dir=self.output_dir)
            self._send(200, {"seeded": True, **type(self).runtime.state_dict(),
                             "entity_count": len(entities_dict["entities"])})

        elif self.path == "/action":
            if self.runtime is None:
                self._send(400, {"error": "not seeded yet -- POST /seed first"})
                return
            action = str(body.get("action", "")).strip()
            if not action:
                self._send(400, {"error": "action required"})
                return
            turn = self.runtime.step(action)
            entities_dict = geo_export.export_scene(self.runtime, output_dir=self.output_dir)
            self._send(200, {
                "narrative": turn.narrative,
                "reward": turn.reward,
                "foe_mode": turn.foe_mode,
                "metrics": {"agency": turn.metrics.agency, "surprise": turn.metrics.surprise,
                           "coherence": turn.metrics.coherence, "unresolved": turn.metrics.unresolved,
                           "duality_risk": turn.metrics.duality_risk},
                "sgr_root": turn.sgr_root,
                "entity_count": len(entities_dict["entities"]),
            })

        else:
            self._send(404, {"error": "not found"})


def serve(host: str = "127.0.0.1", port: int = 8421,
         seed_text: Optional[str] = None, output_dir: str = _DEFAULT_OUTPUT_DIR) -> None:
    handler = type("BoundHandler", (_Handler,), {"output_dir": output_dir})
    if seed_text:
        handler.runtime = Runtime(seed_text)
        geo_export.export_scene(handler.runtime, output_dir=output_dir)

    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"grv2_runtime — action API at http://{host}:{port}")
    print("  POST /seed   {text}     found the mind (once)")
    print("  POST /action {action}   one turn -- also regenerates cesium_output/")
    print("  GET  /state              current SGR snapshot")
    print(f"  writing 3D output to: {output_dir}")
    print("  to watch it: `python serve.py` in another terminal, then open "
         "http://localhost:8080/viewer/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a grv2_runtime Runtime.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8421)
    parser.add_argument("--seed", default=None, help="sentence to found the mind with immediately")
    parser.add_argument("--output-dir", default=_DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    serve(host=args.host, port=args.port, seed_text=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
