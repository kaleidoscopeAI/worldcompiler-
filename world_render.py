"""world_render.py — render a WorldScene to a self-contained interactive HTML page.

Two render paths are available:

* ``build_html`` (default) — pure client-side canvas 3D, no WebGL, no CDN,
  no external assets.  The projection is a straightforward pinhole camera;
  faces are flat-shaded and depth-sorted (painter's algorithm) per frame.
  Appropriate for a scene with a few dozen convex solids and guaranteed to
  work offline.

* ``build_html_threejs`` — opt-in WebGL path that loads Three.js from
  jsDelivr CDN.  Uses real per-vertex normals, Phong shading, smooth
  SphereGeometry and CylinderGeometry, and OrbitControls with damping.
  Requires an internet connection the first time; select with
  ``world_compiler.py --threejs``.

Both paths consume the same ``WorldScene.to_json_dict()`` payload and
display the same objects — same positions, shapes, colors, labels.
"""

from __future__ import annotations

import json
import math


# ---------------------------------------------------------------------------
# Procedural geometry builders (stdlib math only; called once at import time)
# ---------------------------------------------------------------------------

def _make_sphere() -> dict:
    """Subdivide the icosahedron once to get a 42-vert, 80-face sphere
    approximation.  All vertices are projected onto the unit sphere so the
    centroid-as-normal trick the renderer uses is exact."""
    t = 1.618033988749895
    base = [
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ]
    base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ]

    def sph(v: list) -> list:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        return [c / n for c in v]

    verts = [sph(v) for v in base]
    cache: dict = {}

    def mid(i: int, j: int) -> int:
        key = (min(i, j), max(i, j))
        if key in cache:
            return cache[key]
        m = [(verts[i][k] + verts[j][k]) / 2 for k in range(3)]
        idx = len(verts)
        verts.append(sph(m))
        cache[key] = idx
        return idx

    faces = []
    for f in base_faces:
        a, b, c = f
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
    return {"verts": verts, "faces": faces}


def _make_cylinder(sides: int = 8) -> dict:
    """8-sided prism: two rings + two cap centers = 18 verts, 24 faces."""
    verts: list = []
    for i in range(sides):
        a = 2 * math.pi * i / sides
        verts.append([math.cos(a), -1.0, math.sin(a)])   # bottom ring
    for i in range(sides):
        a = 2 * math.pi * i / sides
        verts.append([math.cos(a), 1.0, math.sin(a)])    # top ring
    bc, tc = 2 * sides, 2 * sides + 1
    verts.append([0.0, -1.0, 0.0])                        # bottom centre
    verts.append([0.0, 1.0, 0.0])                         # top centre

    s = sides
    faces = []
    for i in range(s):
        j = (i + 1) % s
        faces.append([i, j, j + s, i + s])   # side quad
    for i in range(s):
        j = (i + 1) % s
        faces.append([bc, j, i])              # bottom cap
    for i in range(s):
        j = (i + 1) % s
        faces.append([tc, i + s, j + s])      # top cap
    return {"verts": verts, "faces": faces}


def _make_cone(sides: int = 8) -> dict:
    """8-sided cone: base ring + apex + base centre = 10 verts, 16 faces."""
    verts: list = []
    for i in range(sides):
        a = 2 * math.pi * i / sides
        verts.append([math.cos(a), -1.0, math.sin(a)])
    verts.append([0.0, 1.0, 0.0])   # apex
    verts.append([0.0, -1.0, 0.0])  # base centre
    apex, bc = sides, sides + 1

    s = sides
    faces = []
    for i in range(s):
        j = (i + 1) % s
        faces.append([i, j, apex])   # side triangle
    for i in range(s):
        j = (i + 1) % s
        faces.append([bc, j, i])     # base cap triangle
    return {"verts": verts, "faces": faces}


# ---------------------------------------------------------------------------
# Geometry registry — canvas renderer looks shapes up here by name
# ---------------------------------------------------------------------------

# Convex, origin-centred solids.  For the platonic shapes the outward normal
# equals the direction from the origin to the face centroid (centroid-normal
# trick), which is what the canvas renderer uses.  The procedural shapes
# (sphere, cylinder, cone) approximate this — shading is slightly imperfect
# on caps but visually fine at the scale these objects appear.
_POLYHEDRA = {
    "tetra": {
        "verts": [
            [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        ],
        "faces": [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]],
    },
    "cube": {
        "verts": [
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ],
        "faces": [
            [0, 1, 2, 3], [7, 6, 5, 4], [0, 3, 7, 4],
            [1, 5, 6, 2], [0, 4, 5, 1], [3, 2, 6, 7],
        ],
    },
    "octa": {
        "verts": [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
        ],
        "faces": [
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4],
            [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5],
        ],
    },
    "icosa": {
        "verts": (lambda t: [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ])(1.618033988749895),
        "faces": [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
    },
    # --- Semantically-assigned shapes (new) ---------------------------------
    # sphere  → persons / agents / living things
    # cylinder → processes / flows / sequences
    # cone    → actions / events / directional change
    "sphere":   _make_sphere(),
    "cylinder": _make_cylinder(),
    "cone":     _make_cone(),
}


def _page(scene_json: str, geometry_json: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; background: #05060a; }}
  body {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  canvas {{ display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}

  .hud {{ position: fixed; inset: 0; pointer-events: none; color: #e8e6f0; }}
  .panel {{
    position: absolute; pointer-events: auto;
    background: rgba(10, 10, 18, 0.62); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px; backdrop-filter: blur(6px); padding: 12px 14px;
  }}
  .title-panel {{ top: 16px; left: 16px; max-width: min(46vw, 520px); }}
  .title-panel h1 {{
    font-family: Georgia, "Times New Roman", serif; font-weight: 400; font-style: italic;
    font-size: 16px; line-height: 1.4; color: #fff; margin-bottom: 6px;
  }}
  .title-panel .sub {{ font-size: 10px; letter-spacing: 0.06em; color: #9a97ad; text-transform: uppercase; }}

  .stats-panel {{ top: 16px; right: 16px; font-size: 10.5px; color: #b9b6c9; min-width: 190px; font-variant-numeric: tabular-nums; }}
  .stats-panel .row {{ display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; }}
  .stats-panel .row b {{ color: #e8e6f0; font-weight: 500; }}
  .stats-panel .fp {{ margin-top: 6px; font-size: 9px; color: #6f6c80; word-break: break-all; }}

  @media (prefers-reduced-motion: reduce) {{
    .label-panel {{ transition: none; }}
  }}

  .label-panel {{
    bottom: 16px; left: 16px; max-width: min(70vw, 560px);
    font-size: 12px; line-height: 1.55; color: #d8d6e6; opacity: 0; transform: translateY(6px);
    transition: opacity 0.25s ease, transform 0.25s ease;
  }}
  .label-panel.visible {{ opacity: 1; transform: translateY(0); }}
  .label-panel .tag {{ font-size: 9px; letter-spacing: 0.08em; color: #8b88a0; text-transform: uppercase; margin-bottom: 5px; }}
  .label-panel .swatch {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}

  .hint {{
    position: absolute; bottom: 16px; right: 16px;
    font-size: 9.5px; letter-spacing: 0.04em; color: #6f6c80; text-transform: uppercase;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="hud">
  <div class="panel title-panel">
    <h1 id="titleText"></h1>
    <div class="sub">World Compiler &middot; text compiled to a symmetry-evolved 3D world</div>
  </div>
  <div class="panel stats-panel" id="statsPanel"></div>
  <div class="panel label-panel" id="labelPanel">
    <div class="tag">motif</div>
    <div id="labelText"></div>
  </div>
  <div class="hint">drag to orbit &middot; scroll to zoom &middot; click a shape</div>
</div>
<script>
const SCENE = {scene_json};
const GEOMETRY = {geometry_json};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, DPR;
function resize() {{
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.width = Math.floor(innerWidth * DPR);
  H = canvas.height = Math.floor(innerHeight * DPR);
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
}}
addEventListener('resize', resize);
resize();

document.getElementById('titleText').textContent = '“' + SCENE.title + '…”';
const s = SCENE.stats;
document.getElementById('statsPanel').innerHTML = `
  <div class="row"><span>source chars</span><b>${{s.source_chars}}</b></div>
  <div class="row"><span>chunks</span><b>${{s.chunks}}</b></div>
  <div class="row"><span>manifold rank</span><b>${{s.manifold_rank}}</b></div>
  <div class="row"><span>symmetry |G|</span><b>${{s.group_order}}</b></div>
  <div class="row"><span>generations</span><b>${{s.generations}}</b></div>
  <div class="row"><span>survivors</span><b>${{s.survivors}}</b></div>
  <div class="row"><span>motifs</span><b>${{s.motifs}}</b></div>
  <div class="row"><span>refine passes</span><b>${{s.refine_passes}} (${{s.refine_converged ? 'converged' : 'max'}})</b></div>
  <div class="fp">fp ${{SCENE.fingerprint}}</div>
`;

const bg = SCENE.background.map(v => Math.max(0, Math.min(1, v)));
document.documentElement.style.setProperty('--bg', `rgb(${{bg.map(v=>Math.round(v*255)).join(',')}})`);

// -- camera -------------------------------------------------------------
const cam = {{ yaw: 0.6, pitch: -0.35, dist: 10.5, focal: 780 }};
let dragging = false, lastX = 0, lastY = 0, autoRotate = true;

canvas.addEventListener('pointerdown', e => {{
  dragging = true; autoRotate = false; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener('pointerup', () => {{ dragging = false; }});
canvas.addEventListener('pointermove', e => {{
  if (!dragging) return;
  cam.yaw += (e.clientX - lastX) * 0.0055;
  cam.pitch += (e.clientY - lastY) * 0.0055;
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
  lastX = e.clientX; lastY = e.clientY;
}});
canvas.addEventListener('wheel', e => {{
  cam.dist *= e.deltaY > 0 ? 1.08 : 0.92;
  cam.dist = Math.max(4, Math.min(28, cam.dist));
  e.preventDefault();
}}, {{ passive: false }});

function project(x, y, z) {{
  const cosY = Math.cos(cam.yaw), sinY = Math.sin(cam.yaw);
  const cosP = Math.cos(cam.pitch), sinP = Math.sin(cam.pitch);
  const x1 = cosY * x - sinY * z;
  const z1 = sinY * x + cosY * z;
  const y1 = cosP * y - sinP * z1;
  const z2 = sinP * y + cosP * z1 + cam.dist;
  if (z2 <= 0.15) return null;
  const scale = (cam.focal * DPR) / z2;
  return {{
    sx: W / 2 + x1 * scale, sy: H / 2 - y1 * scale, depth: z2, scale,
  }};
}}

function rotY(p, a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
}}

const LIGHT = (() => {{
  const l = [0.45, 0.75, -0.5];
  const n = Math.hypot(...l);
  return l.map(v => v / n);
}})();

function shade(rgb, ndotl) {{
  const amb = 0.32, k = amb + (1 - amb) * Math.max(0, ndotl);
  return `rgb(${{Math.round(rgb[0]*255*k)}},${{Math.round(rgb[1]*255*k)}},${{Math.round(rgb[2]*255*k)}})`;
}}

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
let t = 0;
const screenCenters = [];

function frame() {{
  requestAnimationFrame(frame);
  t += REDUCED_MOTION ? 0 : 0.016;
  if (autoRotate && !REDUCED_MOTION) cam.yaw += 0.0011;

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.75);
  grad.addColorStop(0, `rgb(${{bg.map(v=>Math.round(Math.min(1,v*1.8)*255)).join(',')}})`);
  grad.addColorStop(1, `rgb(${{bg.map(v=>Math.round(v*60)).join(',')}})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // edges (drawn first, behind all solids)
  screenCenters.length = 0;
  for (const o of SCENE.objects) {{
    screenCenters.push(project(o.pos[0], o.pos[1], o.pos[2]));
  }}
  ctx.lineWidth = 1 * DPR;
  for (const e of SCENE.edges) {{
    const a = screenCenters[e.a], b = screenCenters[e.b];
    if (!a || !b) continue;
    ctx.strokeStyle = `rgba(180,190,255,${{0.05 + 0.18 * e.s}})`;
    ctx.beginPath();
    ctx.moveTo(a.sx, a.sy);
    ctx.lineTo(b.sx, b.sy);
    ctx.stroke();
  }}

  // faces of every solid, depth-sorted together
  const drawList = [];
  SCENE.objects.forEach((o, oi) => {{
    const geo = GEOMETRY[o.shape];
    const spin = t * 0.15 + o.spin;
    const world = geo.verts.map(v => {{
      const r = rotY(v, spin);
      const local = [r[0] * o.scale, r[1] * o.scale, r[2] * o.scale];
      return [local[0] + o.pos[0], local[1] + o.pos[1], local[2] + o.pos[2]];
    }});
    const proj = world.map(p => project(p[0], p[1], p[2]));
    geo.faces.forEach(face => {{
      if (face.some(i => !proj[i])) return;
      const pts = face.map(i => proj[i]);
      const depth = pts.reduce((a, p) => a + p.depth, 0) / pts.length;
      const cx = face.reduce((a, i) => a + geo.verts[i][0], 0) / face.length;
      const cy = face.reduce((a, i) => a + geo.verts[i][1], 0) / face.length;
      const cz = face.reduce((a, i) => a + geo.verts[i][2], 0) / face.length;
      const nLocal = rotY([cx, cy, cz], spin);
      const nlen = Math.hypot(...nLocal) || 1;
      const n = [nLocal[0]/nlen, nLocal[1]/nlen, nLocal[2]/nlen];
      const ndotl = n[0]*LIGHT[0] + n[1]*LIGHT[1] + n[2]*LIGHT[2];
      drawList.push({{ depth, pts, color: o.color, ndotl, oi }});
    }});
  }});
  drawList.sort((a, b) => b.depth - a.depth);
  for (const f of drawList) {{
    ctx.beginPath();
    ctx.moveTo(f.pts[0].sx, f.pts[0].sy);
    for (let i = 1; i < f.pts.length; i++) ctx.lineTo(f.pts[i].sx, f.pts[i].sy);
    ctx.closePath();
    ctx.fillStyle = shade(f.color, f.ndotl);
    ctx.fill();
  }}
}}
requestAnimationFrame(frame);

// -- click to inspect -----------------------------------------------------
const labelPanel = document.getElementById('labelPanel');
const labelText = document.getElementById('labelText');
let downX = 0, downY = 0;
canvas.addEventListener('pointerdown', e => {{ downX = e.clientX; downY = e.clientY; }});
canvas.addEventListener('pointerup', e => {{
  if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return; // was a drag
  const mx = e.clientX * DPR, my = e.clientY * DPR;
  let best = -1, bestD = Infinity;
  screenCenters.forEach((p, i) => {{
    if (!p) return;
    const d = Math.hypot(p.sx - mx, p.sy - my);
    if (d < bestD) {{ bestD = d; best = i; }}
  }});
  if (best === -1 || bestD > 90 * DPR) {{ labelPanel.classList.remove('visible'); return; }}
  const o = SCENE.objects[best];
  const col = o.color.map(v => Math.round(v * 255));
  labelText.innerHTML =
    `<span class="swatch" style="background:rgb(${{col.join(',')}})"></span>` +
    `<b>${{o.id}}</b> &middot; ${{o.shape}} &middot; mass ${{(o.mass*100).toFixed(1)}}% &middot; ${{o.members}} members` +
    `<br><span style="color:#a8a5ba">&ldquo;${{o.label}}&rdquo;</span>`;
  labelPanel.classList.add('visible');
}});
</script>
</body>
</html>
"""


def build_html(scene_dict: dict) -> str:
    scene_json = json.dumps(scene_dict, separators=(",", ":"))
    geometry_json = json.dumps(_POLYHEDRA, separators=(",", ":"))
    title = f"World Compiler — {scene_dict.get('title', 'untitled')}"
    return _page(scene_json, geometry_json, title)


def build_html_fragment(scene_dict: dict) -> str:
    """Same page, with the outer <!doctype>/<html>/<head>/<body> wrapper
    stripped — for embedding inside a host that supplies its own document
    skeleton (e.g. the Artifact viewer)."""
    full = build_html(scene_dict)
    start = full.index('<meta charset="utf-8">')
    end = full.rindex("</script>") + len("</script>")
    return full[start:end]


def build_live_html() -> str:
    """The living-world page served by world_server.py. Unlike build_html
    (one scene, rendered once), this page starts empty, polls GET /state a
    few times a second, and animates the DIFFERENCE between snapshots:
    unmatched new objects fade in (births), unmatched old objects fade out
    (deaths), matched ones drift toward their new position/color/scale. The
    correspondence between snapshots is a lightweight nearest-position
    match, done client-side — a rendering convenience, not a claim that the
    server tracks strain identity across ticks (it doesn't; strains are
    reformed from scratch each snapshot, same as the batch compiler)."""
    geometry_json = json.dumps(_POLYHEDRA, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Compiler — Live</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; background: #05060a; }}
  body {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  canvas {{ display: block; width: 100vw; height: 100vh; touch-action: none; cursor: grab; }}
  canvas:active {{ cursor: grabbing; }}

  .hud {{ position: fixed; inset: 0; pointer-events: none; color: #e8e6f0; }}
  .panel {{
    position: absolute; pointer-events: auto;
    background: rgba(10, 10, 18, 0.68); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px; backdrop-filter: blur(6px); padding: 12px 14px;
  }}

  .title-panel {{ top: 16px; left: 16px; max-width: min(40vw, 420px); }}
  .title-panel h1 {{
    font-family: Georgia, "Times New Roman", serif; font-weight: 400; font-style: italic;
    font-size: 16px; line-height: 1.4; color: #fff; margin-bottom: 4px; display: flex; align-items: center; gap: 8px;
  }}
  .title-panel .sub {{ font-size: 10px; letter-spacing: 0.06em; color: #9a97ad; text-transform: uppercase; }}
  .pulse {{ width: 7px; height: 7px; border-radius: 50%; background: #6ee7a8; flex: none;
    box-shadow: 0 0 8px #6ee7a8; animation: pulse 1.6s ease-in-out infinite; }}
  @keyframes pulse {{ 0%,100% {{ opacity: 0.4; transform: scale(0.85); }} 50% {{ opacity: 1; transform: scale(1.15); }} }}
  @media (prefers-reduced-motion: reduce) {{ .pulse {{ animation: none; opacity: 0.9; }} }}

  .stats-panel {{ top: 16px; right: 16px; font-size: 10.5px; color: #b9b6c9; min-width: 175px; font-variant-numeric: tabular-nums; }}
  .stats-panel .row {{ display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; }}
  .stats-panel .row b {{ color: #e8e6f0; font-weight: 500; }}

  .feed-panel {{ bottom: 16px; left: 16px; width: min(90vw, 420px); }}
  .feed-panel .tag {{ font-size: 9px; letter-spacing: 0.08em; color: #8b88a0; text-transform: uppercase; margin-bottom: 8px; }}
  .feed-panel textarea {{
    width: 100%; resize: vertical; min-height: 64px; max-height: 30vh;
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.14);
    border-radius: 6px; color: #e8e6f0; font: inherit; font-size: 11.5px; padding: 8px;
    outline: none;
  }}
  .feed-panel textarea:focus {{ border-color: #6ee7a8; }}
  .feed-panel .row {{ display: flex; align-items: center; gap: 10px; margin-top: 8px; }}
  .feed-panel button {{
    background: #e8e6f0; color: #0a0a12; border: none; border-radius: 6px;
    font: inherit; font-size: 11px; font-weight: 600; letter-spacing: 0.02em;
    padding: 8px 16px; cursor: pointer;
  }}
  .feed-panel button:disabled {{ opacity: 0.4; cursor: default; }}
  .feed-panel button:not(:disabled):hover {{ background: #6ee7a8; }}
  .feed-panel button:focus-visible {{ outline: 2px solid #6ee7a8; outline-offset: 2px; }}
  .feed-panel .msg {{ font-size: 10px; color: #8b88a0; }}
  .feed-panel .msg.err {{ color: #ff8a8a; }}

  .label-panel {{
    top: 16px; left: 50%; transform: translate(-50%, -6px); max-width: min(70vw, 520px);
    font-size: 12px; line-height: 1.55; color: #d8d6e6; opacity: 0;
    transition: opacity 0.25s ease, transform 0.25s ease; text-align: center;
  }}
  .label-panel.visible {{ opacity: 1; transform: translate(-50%, 0); }}
  .label-panel .swatch {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
  @media (prefers-reduced-motion: reduce) {{ .label-panel {{ transition: none; }} }}

  .hint {{
    position: absolute; bottom: 16px; right: 16px;
    font-size: 9.5px; letter-spacing: 0.04em; color: #6f6c80; text-transform: uppercase; text-align: right;
  }}
</style>
</head>
<body>
<canvas id="c"></canvas>
<div class="hud">
  <div class="panel title-panel">
    <h1><span class="pulse" id="pulseDot"></span><span id="titleText">a living world</span></h1>
    <div class="sub">World Compiler &middot; LIVE &middot; feed it text at any time</div>
  </div>
  <div class="panel stats-panel" id="statsPanel"></div>
  <div class="panel label-panel" id="labelPanel"></div>
  <div class="panel feed-panel">
    <div class="tag" id="feedTag">seed the world</div>
    <textarea id="feedInput" placeholder="paste text — at least a couple hundred characters — then Seed. After that, feed it more text any time and watch it compete for a place in the world."></textarea>
    <div class="row">
      <button id="feedBtn">Seed World</button>
      <span class="msg" id="feedMsg"></span>
    </div>
  </div>
  <div class="hint">drag to orbit &middot; scroll to zoom &middot; click a shape</div>
</div>
<script>
const GEOMETRY = {geometry_json};

const canvas = document.getElementById('c');
const ctx = canvas.getContext('2d');
let W, H, DPR;
function resize() {{
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = canvas.width = Math.floor(innerWidth * DPR);
  H = canvas.height = Math.floor(innerHeight * DPR);
  canvas.style.width = innerWidth + 'px';
  canvas.style.height = innerHeight + 'px';
}}
addEventListener('resize', resize);
resize();

const cam = {{ yaw: 0.6, pitch: -0.35, dist: 10.5, focal: 780 }};
let dragging = false, lastX = 0, lastY = 0, autoRotate = true;
canvas.addEventListener('pointerdown', e => {{
  dragging = true; autoRotate = false; lastX = e.clientX; lastY = e.clientY;
  canvas.setPointerCapture(e.pointerId);
}});
canvas.addEventListener('pointerup', () => {{ dragging = false; }});
canvas.addEventListener('pointermove', e => {{
  if (!dragging) return;
  cam.yaw += (e.clientX - lastX) * 0.0055;
  cam.pitch += (e.clientY - lastY) * 0.0055;
  cam.pitch = Math.max(-1.45, Math.min(1.45, cam.pitch));
  lastX = e.clientX; lastY = e.clientY;
}});
canvas.addEventListener('wheel', e => {{
  cam.dist *= e.deltaY > 0 ? 1.08 : 0.92;
  cam.dist = Math.max(4, Math.min(28, cam.dist));
  e.preventDefault();
}}, {{ passive: false }});

function project(x, y, z) {{
  const cosY = Math.cos(cam.yaw), sinY = Math.sin(cam.yaw);
  const cosP = Math.cos(cam.pitch), sinP = Math.sin(cam.pitch);
  const x1 = cosY * x - sinY * z;
  const z1 = sinY * x + cosY * z;
  const y1 = cosP * y - sinP * z1;
  const z2 = sinP * y + cosP * z1 + cam.dist;
  if (z2 <= 0.15) return null;
  const scale = (cam.focal * DPR) / z2;
  return {{ sx: W / 2 + x1 * scale, sy: H / 2 - y1 * scale, depth: z2, scale }};
}}
function rotY(p, a) {{
  const c = Math.cos(a), s = Math.sin(a);
  return [c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2]];
}}
const LIGHT = (() => {{ const l = [0.45, 0.75, -0.5]; const n = Math.hypot(...l); return l.map(v => v / n); }})();
function shade(rgb, ndotl) {{
  const amb = 0.32, k = amb + (1 - amb) * Math.max(0, ndotl);
  return `rgb(${{Math.round(rgb[0]*255*k)}},${{Math.round(rgb[1]*255*k)}},${{Math.round(rgb[2]*255*k)}})`;
}}

// -- live snapshot -> interpolated display objects -----------------------
const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)').matches;
let displayed = [];      // {{shape,pos,color,scale,spin,label,mass,members,target}}
let currentEdges = [];   // {{a,b,s}} indices into `displayed`
let background = [0.02, 0.02, 0.05];
let seeded = false;

function dist3(a, b) {{ return Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]); }}
function lerp(a, b, t) {{ return a + (b - a) * t; }}
function lerp3(a, b, t) {{ return [lerp(a[0],b[0],t), lerp(a[1],b[1],t), lerp(a[2],b[2],t)]; }}

function applySnapshot(data) {{
  const wasSeeded = seeded;
  seeded = !!(data.stats && data.stats.seeded);
  updateStatsUI(data.stats || {{}});
  document.getElementById('titleText').textContent = data.title || 'a living world';
  background = data.background || background;
  if (seeded && !wasSeeded) {{
    document.getElementById('feedTag').textContent = 'feed the world more';
    document.getElementById('feedBtn').textContent = 'Feed World';
  }}

  displayed = displayed.filter(d => !(d.dying && d.scale < 0.04));
  // Match against a FIXED snapshot of the pre-existing array. Pushing new
  // entries below extends `displayed`, but those new entries must never be
  // visible as match candidates for later objects in this same batch — that
  // self-interference was the original bug here (a later object could steal
  // an earlier object's brand-new entry mid-loop, corrupting both).
  const prevCount = displayed.length;
  const matched = new Set();
  const newIndexToDisplayed = [];
  (data.objects || []).forEach(o => {{
    let best = -1, bestD = Infinity;
    for (let i = 0; i < prevCount; i++) {{
      const d = displayed[i];
      if (matched.has(i) || d.dying || d.shape !== o.shape) continue;
      const dd = dist3(d.pos, o.pos);
      if (dd < bestD) {{ bestD = dd; best = i; }}
    }}
    if (best !== -1 && bestD < 3.5) {{
      matched.add(best);
      displayed[best].target = o;
      newIndexToDisplayed.push(best);
    }} else {{
      displayed.push({{
        shape: o.shape, pos: o.pos.slice(), color: o.color.slice(), scale: 0.001,
        spin: o.spin, label: o.label, mass: o.mass, members: o.members, id: o.id,
        dying: false, target: o,
      }});
      newIndexToDisplayed.push(displayed.length - 1);
    }}
  }});
  for (let i = 0; i < prevCount; i++) {{
    if (!matched.has(i) && !displayed[i].dying) {{
      displayed[i].dying = true;
      displayed[i].target = {{
        pos: displayed[i].pos, color: displayed[i].color, scale: 0.001,
        shape: displayed[i].shape, spin: displayed[i].spin,
      }};
    }}
  }}
  currentEdges = (data.edges || [])
    .map(e => ({{ a: newIndexToDisplayed[e.a], b: newIndexToDisplayed[e.b], s: e.s }}))
    .filter(e => e.a !== undefined && e.b !== undefined);
}}

function updateStatsUI(s) {{
  const panel = document.getElementById('statsPanel');
  if (!s.seeded) {{ panel.innerHTML = '<div class="row"><span>status</span><b>unseeded</b></div>'; return; }}
  panel.innerHTML = `
    <div class="row"><span>tick</span><b>${{s.tick}}</b></div>
    <div class="row"><span>population</span><b>${{s.population}}</b></div>
    <div class="row"><span>energy pool</span><b>${{s.pool}}</b></div>
    <div class="row"><span>total energy</span><b>${{s.total_energy}}</b></div>
    <div class="row"><span>fed chars</span><b>${{s.fed_chars}}</b></div>
    <div class="row"><span>feeds</span><b>${{s.feeds}}</b></div>
    <div class="row"><span>vocab</span><b>${{s.semantic_vocab}}</b></div>
    <div class="row"><span>manifold rank</span><b>${{s.manifold_rank}}</b></div>
    <div class="row"><span>motifs</span><b>${{s.motifs}}</b></div>
  `;
}}

async function poll() {{
  try {{
    const res = await fetch('/state');
    if (res.ok) applySnapshot(await res.json());
  }} catch (e) {{ /* server hiccup; try again next poll */ }}
  setTimeout(poll, 650);
}}
poll();

// -- render loop ------------------------------------------------------------
let t = 0;
const LERP_RATE = REDUCED_MOTION ? 1.0 : 0.10;
function frame() {{
  requestAnimationFrame(frame);
  t += REDUCED_MOTION ? 0 : 0.016;
  if (autoRotate && !REDUCED_MOTION) cam.yaw += 0.0009;

  displayed.forEach(d => {{
    d.pos = lerp3(d.pos, d.target.pos, LERP_RATE);
    d.color = lerp3(d.color, d.target.color, LERP_RATE);
    d.scale = lerp(d.scale, d.target.scale, LERP_RATE);
  }});

  ctx.setTransform(1, 0, 0, 1, 0, 0);
  const bg = background.map(v => Math.max(0, Math.min(1, v)));
  const grad = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.75);
  grad.addColorStop(0, `rgb(${{bg.map(v=>Math.round(Math.min(1,v*1.8)*255)).join(',')}})`);
  grad.addColorStop(1, `rgb(${{bg.map(v=>Math.round(v*60)).join(',')}})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  if (!seeded) {{ requestAnimationFrame_noop(); return; }}

  const screenCenters = displayed.map(d => project(d.pos[0], d.pos[1], d.pos[2]));
  ctx.lineWidth = 1 * DPR;
  for (const e of currentEdges) {{
    const a = screenCenters[e.a], b = screenCenters[e.b];
    if (!a || !b) continue;
    ctx.strokeStyle = `rgba(180,190,255,${{0.05 + 0.18 * e.s}})`;
    ctx.beginPath(); ctx.moveTo(a.sx, a.sy); ctx.lineTo(b.sx, b.sy); ctx.stroke();
  }}

  const drawList = [];
  displayed.forEach((d) => {{
    if (d.scale < 0.01) return;
    const geo = GEOMETRY[d.shape];
    const spin = t * 0.15 + d.spin;
    const world = geo.verts.map(v => {{
      const r = rotY(v, spin);
      return [r[0]*d.scale + d.pos[0], r[1]*d.scale + d.pos[1], r[2]*d.scale + d.pos[2]];
    }});
    const proj = world.map(p => project(p[0], p[1], p[2]));
    geo.faces.forEach(face => {{
      if (face.some(i => !proj[i])) return;
      const pts = face.map(i => proj[i]);
      const depth = pts.reduce((a, p) => a + p.depth, 0) / pts.length;
      const cx = face.reduce((a, i) => a + geo.verts[i][0], 0) / face.length;
      const cy = face.reduce((a, i) => a + geo.verts[i][1], 0) / face.length;
      const cz = face.reduce((a, i) => a + geo.verts[i][2], 0) / face.length;
      const nLocal = rotY([cx, cy, cz], spin);
      const nlen = Math.hypot(...nLocal) || 1;
      const n = [nLocal[0]/nlen, nLocal[1]/nlen, nLocal[2]/nlen];
      const ndotl = n[0]*LIGHT[0] + n[1]*LIGHT[1] + n[2]*LIGHT[2];
      drawList.push({{ depth, pts, color: d.color, ndotl }});
    }});
  }});
  drawList.sort((a, b) => b.depth - a.depth);
  for (const f of drawList) {{
    ctx.beginPath();
    ctx.moveTo(f.pts[0].sx, f.pts[0].sy);
    for (let i = 1; i < f.pts.length; i++) ctx.lineTo(f.pts[i].sx, f.pts[i].sy);
    ctx.closePath();
    ctx.fillStyle = shade(f.color, f.ndotl);
    ctx.fill();
  }}

  window._screenCenters = screenCenters; // for click-to-inspect below
}}
function requestAnimationFrame_noop() {{}}
requestAnimationFrame(frame);

// -- click to inspect -------------------------------------------------------
const labelPanel = document.getElementById('labelPanel');
let downX = 0, downY = 0;
canvas.addEventListener('pointerdown', e => {{ downX = e.clientX; downY = e.clientY; }});
canvas.addEventListener('pointerup', e => {{
  if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
  const centers = window._screenCenters || [];
  const mx = e.clientX * DPR, my = e.clientY * DPR;
  let best = -1, bestD = Infinity;
  centers.forEach((p, i) => {{
    if (!p || displayed[i].scale < 0.01) return;
    const d = Math.hypot(p.sx - mx, p.sy - my);
    if (d < bestD) {{ bestD = d; best = i; }}
  }});
  if (best === -1 || bestD > 90 * DPR) {{ labelPanel.classList.remove('visible'); return; }}
  const d = displayed[best];
  const col = d.color.map(v => Math.round(v * 255));
  labelPanel.innerHTML =
    `<span class="swatch" style="background:rgb(${{col.join(',')}})"></span>` +
    `<b>${{d.shape}}</b> &middot; mass ${{((d.mass||0)*100).toFixed(1)}}% &middot; ${{d.members||0}} members` +
    `<br><span style="color:#a8a5ba">&ldquo;${{(d.label||'').slice(0,140)}}&rdquo;</span>`;
  labelPanel.classList.add('visible');
}});

// -- feed / seed UI -----------------------------------------------------
const feedTag = document.getElementById('feedTag');
const feedBtn = document.getElementById('feedBtn');
const feedInput = document.getElementById('feedInput');
const feedMsg = document.getElementById('feedMsg');

async function submitFeed() {{
  const text = feedInput.value.trim();
  if (!text) return;
  feedBtn.disabled = true;
  feedMsg.textContent = 'compiling…';
  feedMsg.className = 'msg';
  try {{
    const endpoint = seeded ? '/feed' : '/seed';
    const res = await fetch(endpoint, {{
      method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ text }}),
    }});
    const data = await res.json();
    if (!res.ok) {{
      feedMsg.textContent = data.error || 'failed';
      feedMsg.className = 'msg err';
    }} else {{
      applySnapshot(data);
      feedInput.value = '';
      feedTag.textContent = 'feed the world more';
      feedBtn.textContent = 'Feed World';
      feedMsg.textContent = data.chunks_added !== undefined
        ? `+${{data.chunks_added}} chunks` : 'seeded';
      feedMsg.className = 'msg';
    }}
  }} catch (e) {{
    feedMsg.textContent = 'server unreachable';
    feedMsg.className = 'msg err';
  }}
  feedBtn.disabled = false;
}}
feedBtn.addEventListener('click', submitFeed);
feedInput.addEventListener('keydown', e => {{
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submitFeed();
}});
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Three.js WebGL renderer  (opt-in; requires CDN on first load)
# ---------------------------------------------------------------------------

def build_html_threejs(scene_dict: dict) -> str:
    """WebGL render path using Three.js (loaded from jsDelivr CDN).
    Produces real per-vertex normals, Phong shading, smooth sphere/cylinder/
    cone geometry, and OrbitControls with damping.  Requires an internet
    connection; use ``build_html`` for fully offline output.

    The SCENE payload format is identical to the canvas path so both renderers
    are drop-in replacements for each other."""
    scene_json = json.dumps(scene_dict, separators=(",", ":"))
    title_str = scene_dict.get("title", "untitled")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>World Compiler (WebGL) \u2014 {title_str}</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; overflow: hidden; background: #05060a; }}
  body {{ font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }}
  #c {{ position: fixed; inset: 0; }}

  .hud {{ position: fixed; inset: 0; pointer-events: none; color: #e8e6f0; }}
  .panel {{
    position: absolute; pointer-events: auto;
    background: rgba(10,10,18,0.62); border: 1px solid rgba(255,255,255,0.10);
    border-radius: 10px; backdrop-filter: blur(6px); padding: 12px 14px;
  }}
  .title-panel {{ top: 16px; left: 16px; max-width: min(46vw, 520px); }}
  .title-panel h1 {{
    font-family: Georgia, "Times New Roman", serif; font-weight: 400;
    font-style: italic; font-size: 16px; line-height: 1.4; color: #fff; margin-bottom: 6px;
  }}
  .title-panel .sub {{ font-size: 10px; letter-spacing: 0.06em; color: #9a97ad; text-transform: uppercase; }}

  .stats-panel {{ top: 16px; right: 16px; font-size: 10.5px; color: #b9b6c9; min-width: 190px; font-variant-numeric: tabular-nums; }}
  .stats-panel .row {{ display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; }}
  .stats-panel .row b {{ color: #e8e6f0; font-weight: 500; }}
  .stats-panel .fp {{ margin-top: 6px; font-size: 9px; color: #6f6c80; word-break: break-all; }}

  .label-panel {{
    bottom: 16px; left: 16px; max-width: min(70vw, 560px);
    font-size: 12px; line-height: 1.55; color: #d8d6e6; opacity: 0;
    transform: translateY(6px); transition: opacity 0.25s ease, transform 0.25s ease;
  }}
  .label-panel.visible {{ opacity: 1; transform: translateY(0); }}
  .label-panel .tag {{ font-size: 9px; letter-spacing: 0.08em; color: #8b88a0; text-transform: uppercase; margin-bottom: 5px; }}
  .label-panel .swatch {{ display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; vertical-align: middle; }}
  @media (prefers-reduced-motion: reduce) {{ .label-panel {{ transition: none; }} }}

  .hint {{
    position: absolute; bottom: 16px; right: 16px;
    font-size: 9.5px; letter-spacing: 0.04em; color: #6f6c80; text-transform: uppercase;
  }}
</style>
</head>
<body>
<div id="c"></div>
<div class="hud">
  <div class="panel title-panel">
    <h1 id="titleText"></h1>
    <div class="sub">World Compiler &middot; WebGL &middot; text compiled to a symmetry-evolved 3D world</div>
  </div>
  <div class="panel stats-panel" id="statsPanel"></div>
  <div class="panel label-panel" id="labelPanel">
    <div class="tag">motif</div>
    <div id="labelText"></div>
  </div>
  <div class="hint">drag to orbit &middot; scroll to zoom &middot; click a shape</div>
</div>
<script type="importmap">
{{"imports":{{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js","three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}}}
</script>
<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

const SCENE = {scene_json};

// -- renderer + scene -------------------------------------------------------
const container = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
container.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const bg = SCENE.background;
scene.background = new THREE.Color(bg[0], bg[1], bg[2]);
scene.fog = new THREE.FogExp2(new THREE.Color(bg[0] * 0.6, bg[1] * 0.6, bg[2] * 0.6), 0.035);

const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.1, 100);
camera.position.set(0, 2, 13);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.5;
controls.addEventListener('start', () => {{ controls.autoRotate = false; }});

// -- lighting ---------------------------------------------------------------
scene.add(new THREE.AmbientLight(0xffffff, 0.45));
const sun = new THREE.DirectionalLight(0xffffff, 1.3);
sun.position.set(5, 10, 7);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8888ff, 0.35);
fill.position.set(-5, -3, -7);
scene.add(fill);

// -- geometry factory -------------------------------------------------------
const GEO = {{
  tetra:    () => new THREE.TetrahedronGeometry(1),
  cube:     () => new THREE.BoxGeometry(1.4, 1.4, 1.4),
  octa:     () => new THREE.OctahedronGeometry(1),
  icosa:    () => new THREE.IcosahedronGeometry(1, 1),
  sphere:   () => new THREE.SphereGeometry(1, 20, 14),
  cylinder: () => new THREE.CylinderGeometry(0.65, 0.65, 1.8, 12),
  cone:     () => new THREE.ConeGeometry(0.85, 2.0, 12),
}};

// -- build objects ----------------------------------------------------------
const meshes = [];
SCENE.objects.forEach(o => {{
  const geoFn = GEO[o.shape] || GEO.icosa;
  const col = new THREE.Color(o.color[0], o.color[1], o.color[2]);
  const mat = new THREE.MeshPhongMaterial({{
    color: col, shininess: 55,
    specular: new THREE.Color(0.28, 0.28, 0.38),
    emissive: col, emissiveIntensity: 0.07,
  }});
  const mesh = new THREE.Mesh(geoFn(), mat);
  mesh.position.set(...o.pos);
  mesh.scale.setScalar(o.scale);
  mesh.userData = o;
  scene.add(mesh);
  meshes.push(mesh);
}});

// -- build edges ------------------------------------------------------------
SCENE.edges.forEach(e => {{
  const a = SCENE.objects[e.a], b = SCENE.objects[e.b];
  const geo = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(...a.pos), new THREE.Vector3(...b.pos),
  ]);
  scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({{
    color: 0xaabbff, transparent: true, opacity: 0.05 + 0.18 * e.s,
  }})));
}});

// -- HUD text ---------------------------------------------------------------
document.getElementById('titleText').textContent = '"' + SCENE.title + '\u2026"';
const s = SCENE.stats;
document.getElementById('statsPanel').innerHTML = `
  <div class="row"><span>source chars</span><b>${{s.source_chars}}</b></div>
  <div class="row"><span>chunks</span><b>${{s.chunks}}</b></div>
  <div class="row"><span>manifold rank</span><b>${{s.manifold_rank}}</b></div>
  <div class="row"><span>symmetry |G|</span><b>${{s.group_order}}</b></div>
  <div class="row"><span>generations</span><b>${{s.generations}}</b></div>
  <div class="row"><span>survivors</span><b>${{s.survivors}}</b></div>
  <div class="row"><span>motifs</span><b>${{s.motifs}}</b></div>
  <div class="row"><span>refine passes</span><b>${{s.refine_passes}} (${{s.refine_converged ? 'converged' : 'max'}})</b></div>
  <div class="fp">fp ${{SCENE.fingerprint}}</div>
`;

// -- animate ----------------------------------------------------------------
const REDUCED = matchMedia('(prefers-reduced-motion: reduce)').matches;
let spinAngles = SCENE.objects.map(o => o.spin || 0);
const clock = new THREE.Clock();
(function frame() {{
  requestAnimationFrame(frame);
  const dt = REDUCED ? 0 : clock.getDelta();
  meshes.forEach((m, i) => {{
    spinAngles[i] += dt * 0.15;
    m.rotation.y = spinAngles[i];
  }});
  controls.update();
  renderer.render(scene, camera);
}})();

// -- resize -----------------------------------------------------------------
window.addEventListener('resize', () => {{
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}});

// -- click to inspect -------------------------------------------------------
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const labelPanel = document.getElementById('labelPanel');
const labelText = document.getElementById('labelText');
let downX = 0, downY = 0;
renderer.domElement.addEventListener('pointerdown', e => {{ downX = e.clientX; downY = e.clientY; }});
renderer.domElement.addEventListener('pointerup', e => {{
  if (Math.hypot(e.clientX - downX, e.clientY - downY) > 6) return;
  pointer.set((e.clientX / innerWidth) * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hits = raycaster.intersectObjects(meshes);
  if (!hits.length) {{ labelPanel.classList.remove('visible'); return; }}
  const o = hits[0].object.userData;
  const col = o.color.map(v => Math.round(v * 255));
  labelText.innerHTML =
    `<span class="swatch" style="background:rgb(${{col.join(',')}})"></span>` +
    `<b>${{o.id}}</b> &middot; ${{o.shape}} &middot; mass ${{(o.mass * 100).toFixed(1)}}% &middot; ${{o.members}} members` +
    `<br><span style="color:#a8a5ba">&ldquo;${{o.label}}&rdquo;</span>`;
  labelPanel.classList.add('visible');
}});
</script>
</body>
</html>
"""
