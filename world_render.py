"""world_render.py — render a WorldScene to a self-contained interactive HTML page.

Pure client-side canvas 3D: no WebGL, no CDN, no external assets. Every
object's position, facet count, color, and size were already decided
deterministically by world_compiler.py from the compiled text; this module's
only job is projecting that scene onto a screen.

The projection is a straightforward pinhole camera. Faces are flat-shaded
with a fixed light direction and depth-sorted (painter's algorithm) per
frame — appropriate for a scene with a few dozen convex solids, not a
general-purpose renderer.
"""

from __future__ import annotations

import json

# Convex, origin-centered platonic solids. Because they are convex and
# centered at the origin, a face's outward normal direction is simply the
# direction from the origin to that face's centroid — the renderer uses this
# instead of a cross-product normal, which sidesteps winding-order bugs
# entirely for this closed family of shapes.
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

  .stats-panel {{ top: 16px; right: 16px; font-size: 10.5px; color: #b9b6c9; min-width: 190px; }}
  .stats-panel .row {{ display: flex; justify-content: space-between; gap: 14px; padding: 2px 0; }}
  .stats-panel .row b {{ color: #e8e6f0; font-weight: 500; }}
  .stats-panel .fp {{ margin-top: 6px; font-size: 9px; color: #6f6c80; word-break: break-all; }}

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

let t = 0;
const screenCenters = [];

function frame() {{
  requestAnimationFrame(frame);
  t += 0.016;
  if (autoRotate) cam.yaw += 0.0011;

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
