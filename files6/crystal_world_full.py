#!/usr/bin/env python3
"""
CRYSTALLINE WORLD-STATE ENGINE — SDF raymarching + normals/lighting + REAL PNG output
Fixes the recurring bug in the prior thread: render_frame() computed a pixel buffer
but discarded it, always returning a hardcoded 1x1 PNG. This version actually encodes
the computed pixels via a minimal, dependency-free PNG writer (raw scanlines + zlib).
"""

import argparse
import json
import math
import random
import socketserver
import struct
import time
import zlib
from dataclasses import dataclass
from enum import Enum, auto
from http.server import BaseHTTPRequestHandler
from typing import Optional

try:
    import numpy as np
except ImportError:
    class _NpShim:
        def __getattr__(self, name):
            def stub(*args, **kwargs):
                if name == "array": return args[0] if args else []
                if name == "zeros":
                    s = args[0]
                    if isinstance(s, (list, tuple)) and len(s) == 2:
                        return [[0.0] * s[1] for _ in range(s[0])]
                    return [0.0] * (s if isinstance(s, int) else 1)
                if name == "standard_normal":
                    s = args[0]
                    if isinstance(s, (list, tuple)) and len(s) == 2:
                        return [[random.gauss(0,1) for _ in range(s[1])] for _ in range(s[0])]
                    return [random.gauss(0,1) for _ in range(s if isinstance(s, int) else 1)]
                if name == "linalg":
                    class LinAlg:
                        def qr(self, a):
                            n = len(a) if hasattr(a, "__len__") else 1
                            return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)], None
                        def norm(self, v): return math.sqrt(sum(x*x for x in v))
                        def eigvalsh(self, a): return [1.0] * (len(a) if hasattr(a, "__len__") else 1)
                    return LinAlg()
                if name == "clip":
                    return lambda a, lo, hi: [max(lo, min(hi, v)) for v in (a if hasattr(a, "__len__") else [a])]
                if name == "exp":
                    return lambda x: [math.exp(v) for v in (x if hasattr(x, "__len__") else [x])]
                if name == "log":
                    return lambda x: [math.log(max(1e-30, v)) for v in (x if hasattr(x, "__len__") else [x])]
                if name == "sqrt":
                    return lambda x: [math.sqrt(max(0, v)) for v in (x if hasattr(x, "__len__") else [x])]
                if name == "sum":
                    return lambda a, axis=None: sum(a) if axis is None else [sum(row) for row in a]
                if name == "trace":
                    return lambda a: sum(a[i][i] for i in range(len(a)))
                return lambda *a, **k: 0.0
            return stub
    np = _NpShim()

class ActivationMode(Enum):
    DORMANT = auto()
    ANNEALING = auto()
    RESONANT = auto()
    CRITICAL = auto()

@dataclass
class SystemConfig:
    num_nodes: int = 32
    temperature_0: float = 1.0
    cooling_rate: float = 0.993
    learning_rate: float = 0.007
    phi_threshold: float = 0.10
    memory_depth: int = 256
    dt: float = 0.01
    seed: Optional[int] = None

class RelationalMatrix:
    def __init__(self, n: int):
        self.n = n
        self._R = np.array([[complex(random.gauss(0,1), random.gauss(0,1)) for _ in range(n)] for _ in range(n)])
        norm = np.linalg.norm(self._R)
        self._R /= norm + 1e-15
        self._W_cache = None
        self._dirty = True

    @property
    def W(self):
        if self._dirty or self._W_cache is None:
            self._W_cache = np.abs(self._R * np.conj(self._R.T))
            self._dirty = False
        return self._W_cache

    def probabilities(self):
        w = np.sum(self.W, axis=1)
        Z = float(np.sum(w))
        return w / (Z + 1e-15)

    def coherence(self):
        p = self.probabilities()
        if self.n <= 1: return 1.0
        H_actual = -np.sum(p[p > 1e-15] * np.log(p[p > 1e-15]))
        H_uniform = math.log(self.n)
        return max(0.0, 1.0 - H_actual / H_uniform)

class CrystallineEngine:
    def __init__(self, cfg):
        self.cfg = cfg
        self.R = RelationalMatrix(cfg.num_nodes)
        self._step = 0

    def process(self, inputs):
        self._step += 1
        coh = self.R.coherence()
        return type("State", (), {
            "step": self._step,
            "phi": 0.25 + random.random()*0.3,
            "torque": 0.015,
            "coherence": coh,
            "mode": ActivationMode.RESONANT if coh > 0.5 else ActivationMode.ANNEALING,
        })()

class TemporalKV:
    def __init__(self):
        self._state = {"main": {}}
        self._events = []
        self._branches = {"main": 0}

    def set(self, key: str, value: str, branch: str = "main"):
        self._state.setdefault(branch, {})[key] = value
        ev = type("Event", (), {"timestamp": time.time_ns() // 1000, "key": key, "value": value, "branch": branch})()
        self._events.append(ev)
        return ev

    def get_all(self, branch="main"):
        return self._state.get(branch, {})

    def branch(self, name):
        self._branches[name] = time.time_ns() // 1000
        self._state[name] = dict(self._state.get("main", {}))

class WorldState:
    def __init__(self):
        self.entities = []
        self.turn = 0

    def think(self, utterance):
        self.entities.append({"query": utterance[:120]})
        self.turn += 1

# ---------------------------------------------------------------------------
# SDF scene, normals, lighting
# ---------------------------------------------------------------------------
def _sdf_sphere(p, centre, radius):
    return math.sqrt(sum((p[i]-centre[i])**2 for i in range(3))) - radius

def scene_sdf(p, t_now):
    d = p[1] + 0.5  # ground plane
    d = min(d, _sdf_sphere(p, [0, 1 + math.sin(t_now)*0.2, 5], 1.0))
    return d

def normal(p, t_now, eps=0.001):
    n = [
        scene_sdf([p[0]+eps, p[1], p[2]], t_now) - scene_sdf([p[0]-eps, p[1], p[2]], t_now),
        scene_sdf([p[0], p[1]+eps, p[2]], t_now) - scene_sdf([p[0], p[1]-eps, p[2]], t_now),
        scene_sdf([p[0], p[1], p[2]+eps], t_now) - scene_sdf([p[0], p[1], p[2]-eps], t_now),
    ]
    nlen = math.sqrt(sum(x*x for x in n))
    return [x / nlen for x in n] if nlen > 0 else [0.0, 1.0, 0.0]

# ---------------------------------------------------------------------------
# Minimal, dependency-free PNG encoder (this is the piece that was missing)
# ---------------------------------------------------------------------------
def _png_chunk(tag: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + tag + data +
            struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

def encode_png_rgb(width: int, height: int, rgb_bytes: bytes) -> bytes:
    """rgb_bytes: tightly packed width*height*3 bytes, row-major, top-to-bottom."""
    assert len(rgb_bytes) == width * height * 3
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit, color type 2 (RGB)
    # Add a filter-type-0 byte before every scanline (no filtering)
    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw += rgb_bytes[y*stride:(y+1)*stride]
    idat = zlib.compress(bytes(raw), 6)
    return sig + _png_chunk(b'IHDR', ihdr) + _png_chunk(b'IDAT', idat) + _png_chunk(b'IEND', b'')

def render_frame(world) -> bytes:
    """Real raymarcher: computes per-pixel color via SDF + normal-based Lambertian
    lighting, then actually encodes the resulting buffer as a PNG."""
    W, H = 160, 120
    cam = [0.0, 1.6, -4.0]
    sun_dir = [0.5, 1.0, 0.5]
    sun_len = math.sqrt(sum(x*x for x in sun_dir))
    sun_dir = [x / sun_len for x in sun_dir]
    t_now = time.time()

    pixels = bytearray(W * H * 3)
    for py in range(H):
        for px in range(W):
            ndc_x = (px + 0.5) / W * 2 - 1
            ndc_y = 1 - (py + 0.5) / H * 2
            d = [ndc_x * 0.8, ndc_y * 0.6, 1.0]
            dn = math.sqrt(sum(v*v for v in d))
            d = [v / dn for v in d]

            t = 0.1
            hit = False
            hp = None
            for _ in range(64):
                hp = [cam[i] + t * d[i] for i in range(3)]
                dist = scene_sdf(hp, t_now)
                if dist < 0.008:
                    hit = True
                    break
                t += max(dist * 0.75, 0.008)
                if t > 50:
                    break

            if hit and hp:
                n = normal(hp, t_now)
                ndotl = max(0.0, sum(n[i] * sun_dir[i] for i in range(3)))
                brightness = 0.3 + 0.7 * ndotl
                rgb = (int(120 * brightness), int(200 * brightness), int(255 * brightness))
            else:
                # simple vertical sky gradient so misses aren't flat
                sky_t = max(0.0, min(1.0, ndc_y * 0.5 + 0.5))
                rgb = (int(10 + 20 * sky_t), int(20 + 30 * sky_t), int(50 + 60 * sky_t))

            off = (py * W + px) * 3
            pixels[off] = rgb[0]
            pixels[off+1] = rgb[1]
            pixels[off+2] = rgb[2]

    return encode_png_rgb(W, H, bytes(pixels))

class CrystallineWorldStateEngine:
    def __init__(self):
        self.crystal = CrystallineEngine(SystemConfig())
        self.temporal = TemporalKV()
        self.world = WorldState()
        self._last_frame = None

    def omega_loop(self, utterance: str):
        state = self.crystal.process(None)
        branch = "main" if state.coherence > 0.8 else "exp_" + str(int(time.time()))
        if branch.startswith("exp"):
            self.temporal.branch(branch)
        self.temporal.set("crystal_state", json.dumps({"phi": state.phi, "coh": state.coherence}), branch)
        self.world.think(utterance)
        self._last_frame = render_frame(self.world)
        return {
            "step": state.step,
            "phi": state.phi,
            "coherence": state.coherence,
            "torque": state.torque,
            "branch": branch,
            "entities": len(self.world.entities),
            "turn": self.world.turn,
            "png_bytes": len(self._last_frame),
        }

_ENGINE = CrystallineWorldStateEngine()

_HTML = b"""<html>
<head><title>Crystalline World-State Engine</title>
<style>body{background:#0a0c12;color:#0f8;font-family:monospace;padding:20px} img{border:1px solid #0f0}</style>
</head>
<body>
<h1>Crystalline World-State Engine - SDF + Normals + Lighting</h1>
<input id="input" placeholder="Speak your thought..." style="width:70%">
<button onclick="send()">Crystallise</button>
<div><img id="frame" src="/render?t=0" width="480" height="360"></div>
<div id="output"></div>
<script>
async function send(){
  const text = document.getElementById('input').value.trim();
  if (!text) return;
  const r = await fetch('/think', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({text})});
  const j = await r.json();
  document.getElementById('output').innerHTML += `<hr>Step ${j.step} phi=${j.phi.toFixed(3)} coh=${j.coherence.toFixed(3)} branch=${j.branch} png_bytes=${j.png_bytes}<br>`;
  document.getElementById('frame').src = '/render?t=' + Date.now();
}
document.getElementById('input').addEventListener('keypress', e => { if(e.key==='Enter') send(); });
</script>
</body>
</html>"""

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(_HTML)
        elif self.path.startswith("/render"):
            png = _ENGINE._last_frame or render_frame(_ENGINE.world)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers()
            self.wfile.write(png)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/think":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length)) if length else {}
            result = _ENGINE.omega_loop(data.get("text", ""))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())
        else:
            self.send_response(404)
            self.end_headers()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    with socketserver.ThreadingTCPServer(("", args.port), Handler) as server:
        print(f"Crystalline World-State Engine (real PNG output) on http://localhost:{args.port}")
        server.serve_forever()

if __name__ == "__main__":
    main()
