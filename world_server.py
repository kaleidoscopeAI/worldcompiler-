"""world_server.py — serve a LiveWorld: feed it text, watch it live.

    python3 world_server.py --seed-file organic_ai_core.py

A stdlib-only HTTP server (no new dependencies) around world_live.LiveWorld.
A background Heartbeat thread keeps the world evolving on its own; the
browser polls GET /state a few times a second and animates births, deaths,
and drift client-side. POST /feed injects new text into the SAME running
population at any time — nothing resets.

Local loopback only. Not the deterministic engine gaining a network
dependency: the engine is still numpy + stdlib and knows nothing about HTTP;
this is an application layer sitting in front of it, same as any local dev
server.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

import world_compiler as wc
import world_live as wl
import world_render


class _Handler(BaseHTTPRequestHandler):
    world: wl.LiveWorld
    page_html: str

    def log_message(self, fmt: str, *args) -> None:
        pass  # keep stdout to the world's own status line

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        if self.path == "/":
            body = self.page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            scene = self.world.scene()
            self._send(200, scene.to_json_dict())
        elif self.path == "/health":
            self._send(200, {"ok": True, "seeded": self.world.seeded})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        try:
            body = self._read_json()
        except json.JSONDecodeError as e:
            self._send(400, {"error": f"invalid JSON: {e}"})
            return
        text = str(body.get("text", "")).strip()

        if self.path == "/seed":
            if not text:
                self._send(400, {"error": "text required"})
                return
            try:
                if self.world.seeded:
                    self.world.__init__(self.world.cfg)  # fresh world, same config
                self.world.seed(text)
            except wc.WorldCompilerError as e:
                self._send(422, {"error": str(e)})
                return
            self._send(200, self.world.scene().to_json_dict())
        elif self.path == "/feed":
            if not text:
                self._send(400, {"error": "text required"})
                return
            try:
                n = self.world.feed(text)
            except wc.WorldCompilerError as e:
                self._send(422, {"error": str(e)})
                return
            self._send(200, {"chunks_added": n, **self.world.scene().to_json_dict()})
        else:
            self._send(404, {"error": "not found"})


def serve(world: wl.LiveWorld, host: str = "127.0.0.1", port: int = 8420,
         heartbeat_s: float = 1.5) -> None:
    handler = type("BoundHandler", (_Handler,), {
        "world": world, "page_html": world_render.build_live_html(),
    })
    heartbeat = wl.Heartbeat(world, interval_s=heartbeat_s)
    heartbeat.start()
    httpd = ThreadingHTTPServer((host, port), handler)
    print(f"World Compiler — live server at http://{host}:{port}")
    print("  POST /seed {text}   found the world (once)")
    print("  POST /feed {text}   feed more text into the running population")
    print("  GET  /state         current scene JSON")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        heartbeat.stop()
        httpd.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a live, feedable World Compiler.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--seed", type=int, default=0, help="engine seed")
    parser.add_argument("--seed-file", default=None,
                        help="text file to found the world with immediately")
    parser.add_argument("--heartbeat", type=float, default=1.5,
                        help="seconds between automatic ticks")
    args = parser.parse_args()

    cfg = wc.CompilerConfig(seed=args.seed)
    world = wl.LiveWorld(cfg)
    if args.seed_file:
        world.seed(Path(args.seed_file).read_text(encoding="utf-8"))
        print(f"seeded from {args.seed_file}: {world._stats()}")

    serve(world, host=args.host, port=args.port, heartbeat_s=args.heartbeat)


if __name__ == "__main__":
    main()
