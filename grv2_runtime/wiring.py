"""grv2_runtime/wiring.py — the structural/anatomical layer ("the wiring").

Ports the G-code library + voxelize + simulated-annealing crystallization
logic from semantic-crystal-engine/brain_engine.py (already read in full;
pure NumPy, no missing dependencies). Deliberately carries no color -- that
is grv2_runtime.texture's job. The split is real in code, not just in
description: WiringEntry has no color field, TextureEntry (in texture.py)
has no points field.

A word's wiring is looked up once (O(1), like brain_engine.ShapeMemory) and
stays stable across every recurrence of that word in a session -- it is the
anatomical skeleton of the concept, not something texture or motion is
allowed to reshape.

Five lookup tiers, tried in order, each strictly more general (and more
expensive) than the last:

  1. GCODE_LIBRARY  -- 16 hand-authored, hand-tuned words (bear/tree/river/
     mountain/bird/cloud/fire/stone/house/sword/star/wolf/snake/door/key +
     synonyms). Highest quality, built first.
  2. atlas_csg.ATLAS -- 52 more words across 18 families (human, crystal,
     sphere, cube, torus, void, spiral, device, vehicle, fruit, food,
     clothing, + synonyms of tier-1 words), exact closed-form CSG
     signed-distance shapes, verified this session (26/26 upstream tests
     pass once its two deps are installed).
  3. dictionary retrieval -- wiring_store.WiringStore persists every entry
     any tier ever produces (word -> crystallized point cloud), across
     process restarts. Before paying for an LLM call, check whether a
     near-duplicate of the word (semantic_embed.py: TF-IDF+SVD over char
     n-grams; deliberately kept a very high similarity bar -- see
     _RETRIEVAL_SIM_THRESHOLD's comment for why char-shape similarity over
     a ~45-word corpus can't reliably do true cross-concept semantic
     matching, only near-exact lexical variants) already exists in that
     growing dictionary and reuse it. This is the mechanism by which the
     LLM tier's cost is paid at most once per genuinely new concept ever,
     not a substitute for real semantic generalization -- that's still the
     LLM tier's job.
  4. llm_gcode       -- anything still unmatched, if ANTHROPIC_API_KEY is
     set. Asks an LLM to hand-author G-code for the word, reusing this
     module's own parse_gcode/normalize_pts/voxelize/anneal_crystal
     pipeline on the result. The one tier that costs money and needs the
     network -- added by explicit user choice, because film_siren's
     deformation is distinctive but has no real anatomical knowledge (it
     doesn't know what a castle or a dragon looks like; an LLM does).
     Fails silently (returns None) with no key, no SDK, or any
     network/parse problem, so this tier is a pure bonus, never a
     requirement. Its result is persisted too, so this tier is paid for
     at most once per genuinely new concept, ever.
  5. film_siren      -- the true last resort, and the only tier guaranteed
     to always succeed. A deterministic (hash-seeded, no training, no
     network) FiLM-SIREN-deformed primitive with a genuine eikonal-
     corrected SDF. This is what replaced the old behavior of every
     unmatched word silently collapsing into the same generic "default"
     blob -- which, since callers pass whole multi-word entity labels (not
     single clean words) as `word`, was previously happening for nearly
     every entity in the running system. Tokenizing the label and trying
     each tier per token (see WiringBank.recall) fixes that at the root.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from . import atlas_csg
from . import film_siren
from . import llm_gcode
from . import semantic_embed
from . import thermal as thermal_mod
from . import wiring_store

# Cosine-similarity floor for the dictionary-retrieval tier -- necessary
# but, on its own, provably NOT sufficient (see WiringBank._try_retrieval's
# substring check for why: a live server run turned up "wolf" ~ "woods"
# scoring 0.956, well above this floor, purely from a shared "wo" prefix --
# a wolf is not a tree). Char n-gram similarity over a ~45-word corpus
# doesn't carry real semantic meaning, only lexical shape, so retrieval
# requires BOTH this threshold AND a literal substring relationship before
# reusing an entry. That combination is what actually delivers "near-exact
# lexical variants" ("mountains"~"mountain", "bearish"~"bear") while
# rejecting shape-coincidences -- not this number alone.
_RETRIEVAL_SIM_THRESHOLD = 0.93


# ── G-code shape generators (ported verbatim from brain_engine.py) ──────────

def gcode_bear() -> str:
    lines = ["G28", "G1 F3000"]
    for a in np.linspace(0, 2 * math.pi, 80):
        lines.append(f"G1 X{20+12*math.cos(a):.3f} Y{15+9*math.sin(a):.3f} Z1.000 E0.1")
    for z in np.linspace(0.5, 5.0, 12):
        for x in np.linspace(10, 30, 18):
            inner = math.sqrt(max(0, 81 - (x - 20) ** 2)) * 0.85
            lines.append(f"G1 X{x:.3f} Y{15+inner:.3f} Z{z:.3f} E0.05")
            lines.append(f"G1 X{x:.3f} Y{15-inner:.3f} Z{z:.3f} E0.05")
    for zl in np.linspace(-7, 7, 10):
        rr = math.sqrt(max(0, 49 - zl * zl))
        for a in np.linspace(0, 2 * math.pi, max(8, round(24 * rr / 7))):
            lines.append(f"G1 X{28+rr*math.cos(a):.3f} Y{26+rr*math.sin(a):.3f} Z{2+zl*0.36:.3f} E0.1")
    for ear_x in [24.0, 32.0]:
        for a in np.linspace(0, math.pi, 20):
            lines.append(f"G1 X{ear_x+3*math.cos(a):.3f} Y{33+3*math.sin(a):.3f} Z3.000 E0.1")
    for a in np.linspace(0, 2 * math.pi, 28):
        lines.append(f"G1 X{34+2.5*math.cos(a):.3f} Y{24+2*math.sin(a):.3f} Z2.500 E0.08")
    for lx in [10.0, 16.0, 22.0, 28.0]:
        for z in np.linspace(0, 5.0, 14):
            for a in np.linspace(0, 2 * math.pi, 12):
                lines.append(f"G1 X{lx+1.8*math.cos(a):.3f} Y{5+1.5*math.sin(a):.3f} Z{z:.3f} E0.06")
    return "\n".join(lines)


def gcode_tree() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(0, 3.5, 24):
        r = max(0.5, 1.5 - z * 0.1)
        for a in np.linspace(0, 2 * math.pi, 16):
            lines.append(f"G1 X{20+r*math.cos(a):.3f} Y{20+r*math.sin(a):.3f} Z{z:.3f} E0.1")
    for (zb, zh, br) in [(3.5, 3.5, 9), (6.5, 2.8, 7), (9, 2, 5)]:
        for z in np.linspace(zb, zb + zh, 16):
            taper = br * (1 - (z - zb) / zh)
            n = max(8, round(32 * taper / br))
            for a in np.linspace(0, 2 * math.pi, n):
                lines.append(f"G1 X{20+taper*math.cos(a):.3f} Y{20+taper*math.sin(a):.3f} Z{z:.3f} E0.05")
        for z in np.linspace(zb, zb + zh * 0.6, 5):
            taper = br * (1 - (z - zb) / zh) * 0.75
            for rf in np.linspace(0.2, 1, 4):
                for a in np.linspace(0, 2 * math.pi, 18):
                    lines.append(f"G1 X{20+taper*rf*math.cos(a):.3f} Y{20+taper*rf*math.sin(a):.3f} Z{z:.3f} E0.04")
    for z in np.linspace(12, 16, 12):
        r = max(0.1, 1 * (1 - (z - 12) / 4))
        for a in np.linspace(0, 2 * math.pi, 8):
            lines.append(f"G1 X{20+r*math.cos(a):.3f} Y{20+r*math.sin(a):.3f} Z{z:.3f} E0.04")
    return "\n".join(lines)


def gcode_river() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(0, 1.8, 7):
        for t in np.linspace(0, 50, 140):
            cx = t + 5
            cy = 20 + 10 * math.sin(t * 0.22) + 3 * math.sin(t * 0.55)
            w = 4.5 + 2.5 * math.cos(t * 0.22)
            for wo in np.linspace(-w, w, 7):
                lines.append(f"G1 X{cx:.3f} Y{cy+wo:.3f} Z{z+0.3*math.sin(wo/w*math.pi):.3f} E0.05")
    return "\n".join(lines)


def gcode_mountain() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(0, 22, 44):
        r = max(0.4, 18 * (1 - (z / 22) ** 1.4))
        n = max(6, round(36 * r / 18))
        for i in range(n):
            a = (i / n) * 2 * math.pi
            rough = 1 + 0.18 * math.sin(a * 7 + z * 0.8) + 0.08 * math.sin(a * 13 + z * 1.3)
            lines.append(f"G1 X{25+r*rough*math.cos(a):.3f} Y{25+r*rough*math.sin(a):.3f} Z{z:.3f} E0.05")
        if z < 10:
            for rf in [0.35, 0.65, 0.85]:
                for a in np.linspace(0, 2 * math.pi, 20):
                    lines.append(f"G1 X{25+r*rf*math.cos(a):.3f} Y{25+r*rf*math.sin(a):.3f} Z{z:.3f} E0.03")
    return "\n".join(lines)


def gcode_bird() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(0, 5, 14):
        for a in np.linspace(0, 2 * math.pi, 32):
            rx = 5 * math.sqrt(max(0, 1 - ((z - 2.5) / 2.5) ** 2))
            ry = 2.5 * math.sqrt(max(0, 1 - ((z - 2.5) / 2.5) ** 2))
            lines.append(f"G1 X{20+rx*math.cos(a):.3f} Y{20+ry*math.sin(a):.3f} Z{z:.3f} E0.08")
    for side in [-1, 1]:
        for t in np.linspace(0, 1, 32):
            lines.append(f"G1 X{20+side*(5+t*16):.3f} Y{20+side*t*4:.3f} Z{2.5+math.sin(t*math.pi)*3.5-t*0.8:.3f} E0.06")
        for t in np.linspace(0, 1, 20):
            for sf in np.linspace(0.15, 0.9, 5):
                lines.append(f"G1 X{20+side*(5+t*16*sf):.3f} Y{20+side*t*4*sf:.3f} Z{2.5+math.sin(t*math.pi)*3.5*sf:.3f} E0.04")
    for z in np.linspace(4, 7, 6):
        r = 2.5 * math.sqrt(max(0, 1 - ((z - 5.5) / 2) ** 2))
        for a in np.linspace(0, 2 * math.pi, 20):
            lines.append(f"G1 X{26+r*math.cos(a):.3f} Y{20.5+r*0.85*math.sin(a):.3f} Z{z:.3f} E0.06")
    for t in np.linspace(0, 1, 10):
        lines.append(f"G1 X{29+t*3.5:.3f} Y20.500 Z{5.5-t*0.8:.3f} E0.04")
    for a in np.linspace(-0.4, 0.4, 8):
        for t in np.linspace(0, 1, 8):
            lines.append(f"G1 X{15-t*4:.3f} Y{20+t*math.sin(a)*4:.3f} Z{2.5-t*0.5:.3f} E0.04")
    return "\n".join(lines)


def gcode_cloud() -> str:
    lines = ["G28", "G1 F3000"]
    for (cx, cy, cz, r) in [(20, 20, 12, 5), (27, 19, 10, 4.5), (14, 21, 10, 4),
                            (22, 23, 11, 4.5), (18, 17, 9, 3.8), (25, 22, 13, 3.5)]:
        for zl in np.linspace(-r, r, 12):
            rr = r * math.sqrt(max(0, 1 - (zl / r) ** 2))
            n = max(8, round(24 * rr / r))
            for a in np.linspace(0, 2 * math.pi, n):
                lines.append(f"G1 X{cx+rr*math.cos(a):.3f} Y{cy+rr*math.sin(a):.3f} Z{cz+zl:.3f} E0.05")
        for zl in np.linspace(-r * 0.7, r * 0.7, 6):
            rr = r * math.sqrt(max(0, 1 - (zl / r) ** 2)) * 0.7
            for a in np.linspace(0, 2 * math.pi, 16):
                lines.append(f"G1 X{cx+rr*math.cos(a):.3f} Y{cy+rr*math.sin(a):.3f} Z{cz+zl:.3f} E0.03")
    return "\n".join(lines)


def gcode_fire() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(0, 20, 40):
        br = max(0.3, 8 * (1 - (z / 20) ** 1.6))
        n = max(6, round(24 * br / 8))
        for i in range(n):
            a = (i / n) * 2 * math.pi
            tb = 1 + 0.3 * math.sin(a * 5 + z * 0.7) + 0.15 * math.sin(a * 11 + z * 1.4)
            lines.append(f"G1 X{18+br*tb*math.cos(a):.3f} Y{18+br*tb*math.sin(a):.3f} Z{z:.3f} E0.05")
        if z > 8:
            for ta in [0, math.pi / 3, 2 * math.pi / 3, math.pi, 4 * math.pi / 3, 5 * math.pi / 3]:
                d = (z - 8) / 12
                lines.append(f"G1 X{18+d*2.5*math.cos(ta+d):.3f} Y{18+d*2.5*math.sin(ta+d):.3f} Z{z:.3f} E0.03")
    return "\n".join(lines)


def gcode_stone() -> str:
    lines = ["G28", "G1 F3000"]
    rng = np.random.default_rng(42)
    for z in np.linspace(0, 14, 28):
        r = max(1, 8 * (1 - abs(z / 14 - 0.35) * 1.8)) + float(rng.uniform(-0.8, 0.8))
        n = max(5, round(r * 2.5))
        for i in range(n):
            a = (i / n) * 2 * math.pi
            fac = r * (1 + 0.25 * math.sin(i * 3.14 / 2.2 + z * 0.6))
            lines.append(f"G1 X{18+fac*math.cos(a):.3f} Y{18+fac*math.sin(a):.3f} Z{z:.3f} E0.06")
    return "\n".join(lines)


def gcode_house() -> str:
    lines = ["G28", "G1 F3000"]
    hw, wall_h = 8.0, 8.0
    corners = [(-hw, -hw), (hw, -hw), (hw, hw), (-hw, hw), (-hw, -hw)]
    for z in np.linspace(0, wall_h, 16):
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[i + 1]
            for t in np.linspace(0, 1, 12):
                lines.append(f"G1 X{20+x0+t*(x1-x0):.3f} Y{20+y0+t*(y1-y0):.3f} Z{z:.3f} E0.05")
    for t in np.linspace(0, 1, 24):
        xr = 20 - hw + t * 2 * hw
        for s in np.linspace(-1, 1, 16):
            lines.append(f"G1 X{xr:.3f} Y{20+s*hw*1.1:.3f} Z{wall_h+6-abs(s)*6:.3f} E0.04")
    for z in np.linspace(0, 3.5, 8):
        for x in np.linspace(-1.5, 1.5, 4):
            lines.append(f"G1 X{20+x:.3f} Y{20-hw:.3f} Z{z:.3f} E0.03")
    return "\n".join(lines)


def gcode_sword() -> str:
    lines = ["G28", "G1 F3000"]
    tip_z, guard_z = 22.0, 4.0
    for z in np.linspace(guard_z, tip_z, 44):
        t = (z - guard_z) / (tip_z - guard_z)
        w = 2.0 * (1 - t) ** 0.6
        for a in np.linspace(0, 2 * math.pi, 10):
            lines.append(f"G1 X{20+w*0.22*math.cos(a):.3f} Y{20+w*math.sin(a):.3f} Z{z:.3f} E0.05")
    for t in np.linspace(-1, 1, 16):
        lines.append(f"G1 X20.000 Y{20+t*4.5:.3f} Z{guard_z:.3f} E0.06")
    for z in np.linspace(0, guard_z, 10):
        for a in np.linspace(0, 2 * math.pi, 10):
            lines.append(f"G1 X{20+0.65*math.cos(a):.3f} Y{20+0.65*math.sin(a):.3f} Z{z:.3f} E0.05")
    for a in np.linspace(0, 2 * math.pi, 14):
        lines.append(f"G1 X{20+1.1*math.cos(a):.3f} Y{20+1.1*math.sin(a):.3f} Z0.000 E0.05")
    return "\n".join(lines)


def gcode_star() -> str:
    lines = ["G28", "G1 F3000"]
    outer_r, inner_r, points = 9.0, 3.6, 5
    verts = []
    for i in range(points * 2):
        ang = i * math.pi / points - math.pi / 2
        r = outer_r if i % 2 == 0 else inner_r
        verts.append((r * math.cos(ang), r * math.sin(ang)))
    verts.append(verts[0])
    for z in np.linspace(0, 3.0, 6):
        for i in range(len(verts) - 1):
            x0, y0 = verts[i]
            x1, y1 = verts[i + 1]
            for t in np.linspace(0, 1, 10):
                lines.append(f"G1 X{20+x0+t*(x1-x0):.3f} Y{20+y0+t*(y1-y0):.3f} Z{z:.3f} E0.05")
    for i in range(len(verts) - 1):
        x0, y0 = verts[i]
        x1, y1 = verts[i + 1]
        for t in np.linspace(0, 1, 6):
            for rf in np.linspace(0.1, 0.9, 5):
                lines.append(f"G1 X{20+rf*(x0+t*(x1-x0)):.3f} Y{20+rf*(y0+t*(y1-y0)):.3f} Z1.500 E0.03")
    return "\n".join(lines)


def gcode_wolf() -> str:
    lines = ["G28", "G1 F3000"]
    for z in np.linspace(2, 7, 10):
        for x in np.linspace(9, 27, 20):
            width = 3.6 * math.sqrt(max(0, 1 - ((x - 18) / 9) ** 2))
            lines.append(f"G1 X{x:.3f} Y{20+width:.3f} Z{z:.3f} E0.05")
            lines.append(f"G1 X{x:.3f} Y{20-width:.3f} Z{z:.3f} E0.05")
    for a in np.linspace(0, 2 * math.pi, 20):
        lines.append(f"G1 X{29+2.0*math.cos(a):.3f} Y{20+1.6*math.sin(a):.3f} Z6.500 E0.06")
    for t in np.linspace(0, 1, 10):
        lines.append(f"G1 X{31+t*3.2:.3f} Y{20+(1-t)*0.4:.3f} Z{6.5-t*1.2:.3f} E0.04")
    for ex in (28.0, 30.2):
        for a in np.linspace(0, math.pi, 10):
            lines.append(f"G1 X{ex+1.0*math.cos(a):.3f} Y{20+0.8*math.sin(a):.3f} Z{8.2+1.4*math.sin(a):.3f} E0.04")
    for lx in (11.0, 15.0, 21.0, 25.0):
        for z in np.linspace(0, 2, 8):
            for a in np.linspace(0, 2 * math.pi, 8):
                lines.append(f"G1 X{lx+0.7*math.cos(a):.3f} Y{20+0.7*math.sin(a):.3f} Z{z:.3f} E0.04")
    for t in np.linspace(0, 1, 16):
        lines.append(f"G1 X{9-t*5:.3f} Y{20+math.sin(t*2.4)*1.4:.3f} Z{6.5+t*3.2:.3f} E0.04")
    return "\n".join(lines)


def gcode_snake() -> str:
    lines = ["G28", "G1 F3000"]
    length = 34.0
    for i in range(160):
        t = i / 159
        x = 4 + t * length
        y = 20 + 6 * math.sin(t * 4 * math.pi)
        r = max(0.3, 1.6 * (1 - t) ** 0.5)
        for a in np.linspace(0, 2 * math.pi, 8):
            lines.append(f"G1 X{x:.3f} Y{y+r*math.sin(a):.3f} Z{1.0+r*math.cos(a):.3f} E0.04")
    hx, hy = 4 + length, 20 + 6 * math.sin(4 * math.pi)
    for a in np.linspace(0, 2 * math.pi, 16):
        lines.append(f"G1 X{hx+1.4*math.cos(a):.3f} Y{hy+1.4*math.sin(a):.3f} Z1.000 E0.05")
    for t in np.linspace(0, 1, 6):
        lines.append(f"G1 X{hx+1.4+t*1.5:.3f} Y{hy:.3f} Z1.000 E0.02")
    return "\n".join(lines)


def gcode_door() -> str:
    lines = ["G28", "G1 F3000"]
    hw, h, thick = 4.0, 12.0, 0.6
    for y in (20 - thick / 2, 20 + thick / 2):
        for z in np.linspace(0, h, 16):
            for x in (-hw, hw):
                lines.append(f"G1 X{20+x:.3f} Y{y:.3f} Z{z:.3f} E0.05")
            for x in np.linspace(-hw, hw, 10):
                lines.append(f"G1 X{20+x:.3f} Y{y:.3f} Z{z:.3f} E0.04")
        for pz in (h * 0.3, h * 0.7):
            for x in np.linspace(-hw * 0.7, hw * 0.7, 8):
                lines.append(f"G1 X{20+x:.3f} Y{y:.3f} Z{pz:.3f} E0.03")
    for a in np.linspace(0, 2 * math.pi, 10):
        lines.append(f"G1 X{20+hw*0.6+0.3*math.cos(a):.3f} Y{20+thick/2+0.2:.3f} Z{h*0.4+0.3*math.sin(a):.3f} E0.03")
    return "\n".join(lines)


def gcode_key() -> str:
    lines = ["G28", "G1 F3000"]
    for z in (0.85, 1.15):
        for a in np.linspace(0, 2 * math.pi, 24):
            lines.append(f"G1 X{20+2.2*math.cos(a):.3f} Y{20+2.2*math.sin(a):.3f} Z{z:.3f} E0.05")
        for a in np.linspace(0, 2 * math.pi, 14):
            lines.append(f"G1 X{20+1.1*math.cos(a):.3f} Y{20+1.1*math.sin(a):.3f} Z{z:.3f} E0.03")
        for t in np.linspace(0, 1, 30):
            x = 22.2 + t * 10.0
            lines.append(f"G1 X{x:.3f} Y20.000 Z{z:.3f} E0.04")
            lines.append(f"G1 X{x:.3f} Y20.150 Z{z:.3f} E0.02")
            lines.append(f"G1 X{x:.3f} Y19.850 Z{z:.3f} E0.02")
        for frac, dep in zip((0.55, 0.65, 0.72, 0.85, 0.92, 1.0), (0.5, 1.0, 0.4, 1.2, 0.6, 0.9)):
            x = 22.2 + frac * 10.0
            for t in np.linspace(0, 1, 5):
                lines.append(f"G1 X{x:.3f} Y{20-dep*t:.3f} Z{z:.3f} E0.03")
    return "\n".join(lines)


def gcode_default() -> str:
    lines = ["G28"]
    rng = np.random.default_rng(99)
    for _ in range(600):
        a1 = float(rng.uniform(0, 2 * math.pi))
        a2 = float(rng.uniform(0, math.pi))
        r = float(rng.uniform(6, 22))
        lines.append(f"G1 X{20+r*math.sin(a2)*math.cos(a1):.3f} Y{20+r*math.sin(a2)*math.sin(a1):.3f} "
                    f"Z{10+r*math.cos(a2):.3f} E0.01")
    return "\n".join(lines)


GCODE_LIBRARY: Dict[str, Callable[[], str]] = {
    "bear": gcode_bear, "woods": gcode_tree, "tree": gcode_tree, "forest": gcode_tree,
    "river": gcode_river, "water": gcode_river, "mountain": gcode_mountain,
    "bird": gcode_bird, "cloud": gcode_cloud, "fire": gcode_fire,
    "stone": gcode_stone, "rock": gcode_stone,
    "house": gcode_house, "home": gcode_house, "cabin": gcode_house,
    "sword": gcode_sword, "blade": gcode_sword, "dagger": gcode_sword,
    "star": gcode_star,
    "wolf": gcode_wolf, "hound": gcode_wolf,
    "snake": gcode_snake, "serpent": gcode_snake,
    "door": gcode_door, "gate": gcode_door,
    "key": gcode_key,
    "default": gcode_default,
}
# Deliberately NOT added here: "castle" -- test_grv2_wiring.py and
# test_grv2_wiring_store.py both use "castle" as the canonical example of a
# word that must fall through to the LLM/retrieval tiers (mocked, so no real
# network call happens in the suite). Adding it to this dict would silently
# invalidate that coverage. Pick a different structure word if one is wanted
# here later (e.g. "hut", "tent").


# ── G-code parser + voxelize + anneal (ported verbatim) ──────────────────────

def parse_gcode(gcode_text: str) -> np.ndarray:
    import re
    pts: List[List[float]] = []
    cur = [20.0, 20.0, 0.0]
    pat = re.compile(r'G1\s+(?:X([-\d.]+))?\s*(?:Y([-\d.]+))?\s*(?:Z([-\d.]+))?', re.I)
    for line in gcode_text.split('\n'):
        m = pat.search(line)
        if not m:
            continue
        tgt = [float(m.group(1)) if m.group(1) else cur[0],
               float(m.group(2)) if m.group(2) else cur[1],
               float(m.group(3)) if m.group(3) else cur[2]]
        d = math.sqrt(sum((tgt[i] - cur[i]) ** 2 for i in range(3)))
        steps = max(1, int(d / 0.5))
        for s in range(steps):
            t = s / steps
            pts.append([cur[i] + t * (tgt[i] - cur[i]) for i in range(3)])
        cur = tgt
    return np.array(pts, dtype=np.float32) if pts else np.zeros((10, 3), dtype=np.float32)


def normalize_pts(pts: np.ndarray, scale: float = 30.0) -> np.ndarray:
    if len(pts) == 0:
        return pts
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    return ((pts - mn) / rng - 0.5) * scale


def voxelize(pts: np.ndarray, R: int = 32) -> np.ndarray:
    grid = np.zeros((R, R, R), dtype=np.uint8)
    if len(pts) == 0:
        return grid
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    rng = mx - mn
    rng[rng == 0] = 1.0
    for pt in pts:
        ix = int(np.clip((pt[0] - mn[0]) / rng[0] * (R - 1), 0, R - 1))
        iy = int(np.clip((pt[1] - mn[1]) / rng[1] * (R - 1), 0, R - 1))
        iz = int(np.clip((pt[2] - mn[2]) / rng[2] * (R - 1), 0, R - 1))
        grid[ix, iy, iz] = 1
    return grid


def anneal_crystal(voxels: np.ndarray, target_pts: np.ndarray,
                   K: int = 20, T0: float = 1.0) -> np.ndarray:
    """Energy functional: E(z) = -cos_sim(sigma(z), target) + lambda_ent*H(sigma(z)).
    Aligns voxel logits to the target shape while forcing entropy toward 0
    (binary crystal). Temperature cools logarithmically: T = T0*(1-k/K)."""
    R = voxels.shape[0]
    logits = np.where(voxels > 0, 2.0, -2.0).astype(np.float32)

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -10, 10)))

    target_vox = voxelize(target_pts, R)
    target_flat = target_vox.flatten().astype(np.float32)

    for k in range(K):
        T = T0 * (1.0 - k / K)
        p = sigmoid(logits)
        cf = p.flatten()
        dot = np.dot(cf, target_flat)
        norm_c = np.linalg.norm(cf) + 1e-7
        norm_t = np.linalg.norm(target_flat) + 1e-7
        align_grad = -(target_flat / (norm_c * norm_t) - dot * cf / (norm_c ** 3 * norm_t))
        align_grad = align_grad.reshape(logits.shape)
        ent_grad = (2 * p - 1) * T
        lr = 0.5 * (1.0 - k / K)
        logits -= lr * (0.6 * align_grad + 0.4 * ent_grad)

    return (sigmoid(logits) > 0.5).astype(np.uint8)


# ── cheap, dependency-free English inflection stripping ─────────────────────

def _inflection_candidates(token: str) -> List[str]:
    """token plus a handful of plausible base-form guesses, in order (token
    itself first). No lemmatizer, no dictionary, no model -- deliberately
    so: this repo's vocabulary (GCODE_LIBRARY + atlas_csg.ATLAS) is a small,
    known, closed set, and a wrong guess here costs nothing because the
    caller only ever accepts a candidate that's an exact key in one of
    those dicts. This is what fixes plurals ("trees", "mountains",
    "flames") without pulling in a Russian-morphology / TensorFlow-sized
    dependency for what is, for this vocabulary, a suffix-stripping problem."""
    cands = [token]
    if len(token) > 4 and token.endswith("ies"):
        cands.append(token[:-3] + "y")
    if len(token) > 3 and token.endswith("es"):
        cands.append(token[:-2])
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        cands.append(token[:-1])
    if len(token) > 5 and token.endswith("ing"):
        cands.append(token[:-3])
        cands.append(token[:-3] + "e")
    if len(token) > 4 and token.endswith("ed"):
        cands.append(token[:-2])
        cands.append(token[:-1])
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


# ── word -> deterministic unit vector, for the neural fallback tier ─────────

def _hash_to_unit_vector(phrase: str, k: int = 49) -> np.ndarray:
    """Deterministic z in S^{k-1} from arbitrary text, no training, no
    network. Uses blake2b (matches this repo's own _stable_unit/_origin_hash
    convention) rather than Python's randomized-per-process hash()."""
    seed = int.from_bytes(hashlib.blake2b(phrase.encode("utf-8"), digest_size=4).digest(), "big")
    rng = np.random.RandomState(seed)
    v = rng.normal(size=k).astype(np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


# ── The wiring bank: O(1) word -> stable anatomical point cloud ─────────────

@dataclass
class WiringEntry:
    word: str
    points: np.ndarray   # (N, 3) local-space coordinates, metres, centred at origin
    node_count: int
    source: str = "gcode"   # "gcode" | "atlas" | "retrieved" | "llm" | "neural" -- which tier produced this
    thermal_cost: float = 0.0   # 0 = cheap/stable to materialize, 1 = critical (thermal.py)


class WiringBank:
    """O(1) word -> WiringEntry. Built once; every recurrence of a word gets
    the exact same points back -- the anatomical skeleton never moves because
    of texture or motion. Only genuinely new words cost anything (annealing,
    SDF sampling, retrieval, or an LLM call), and every tier except LLM/
    retrieval is deterministic (no unseeded randomness anywhere in the
    G-code generators, the annealing loop, the CSG atlas, or the
    FiLM-SIREN's hash-seeded weight init)."""

    def __init__(self, resolution: int = 32, scale: float = 30.0, allow_llm: bool = True,
                 dictionary_path: Optional[str] = None) -> None:
        self.resolution = resolution
        self.scale = scale
        # Hard override, independent of ANTHROPIC_API_KEY: tests and any
        # other caller that needs fully offline/deterministic behavior set
        # this False rather than relying on the ambient environment not
        # happening to have a key set. llm_gcode itself also no-ops
        # without a key, so this is belt-and-suspenders, not the only gate.
        self.allow_llm = allow_llm
        self._entries: Dict[str, WiringEntry] = {}

        # None (the default) means no persistence -- bare WiringBank() stays
        # exactly as in-memory/ephemeral as before, so existing callers and
        # every test in this repo are unaffected. Runtime passes a real path
        # so production usage (the server) gets a permanent, growing
        # dictionary; tests that want persistence opt in explicitly too,
        # rather than writing to shared repo state by accident.
        self._store = wiring_store.WiringStore(dictionary_path) if dictionary_path else None
        if self._store is not None:
            for cache_key, d in self._store.load().items():
                self._entries[cache_key] = WiringEntry(
                    word=d["word"], points=d["points"], node_count=d["node_count"],
                    source=d["source"], thermal_cost=d["thermal_cost"])

        self._embedder: Optional[semantic_embed.SemanticEmbed] = None
        self._embedder_vocab: List[str] = []

    def recall(self, word: str) -> WiringEntry:
        raw_key = word.lower().strip()
        # Callers often pass whole multi-word entity labels, not single
        # clean words -- tokenize and try each token against the two
        # concept-lookup tiers before falling through to the LLM/neural
        # tiers keyed on the whole phrase.
        tokens = re.findall(r"[a-z']+", raw_key)
        # Also try cheap English inflection strippings of each token
        # ("trees" -> "tree", "mountains" -> "mountain") so plurals/
        # inflected forms still land on the same hand-tuned/atlas entry as
        # their base form, grouped by original token so position priority
        # is preserved. No dictionary or model needed: a wrong guess here
        # ("mountains" -> "mountaine") simply won't be a GCODE_LIBRARY/
        # ATLAS key and is silently ignored -- correctness comes from the
        # exact-match requirement below, not from the stripping itself.
        candidates: List[str] = []
        for t in tokens:
            candidates.extend(_inflection_candidates(t))

        gcode_token = next((t for t in candidates if t in GCODE_LIBRARY), None)
        if gcode_token is not None:
            return self._cached(gcode_token, lambda: self._build_gcode(gcode_token))

        # Exact atlas match first, in token order. Fuzzy substring matching
        # (atlas_csg.lookup_atlas's second pass) is only tried afterwards,
        # and only for candidates of length >= 4 -- short grammatical
        # tokens ("a", "on", "in") are frequent substrings of unrelated
        # atlas keys (e.g. "a" inside "bear") and would otherwise hijack
        # ordinary sentences before a real content word ever gets a chance.
        exact_token = next((t for t in candidates if t in atlas_csg.ATLAS), None)
        if exact_token is not None:
            parts = atlas_csg.ATLAS[exact_token](0.80)
            return self._cached(f"atlas:{exact_token}",
                                lambda p=parts, t=exact_token: self._build_atlas(t, p))

        for t in candidates:
            if len(t) < 4:
                continue
            parts = atlas_csg.lookup_atlas(t)
            if parts is not None:
                return self._cached(f"atlas:{t}", lambda p=parts, t=t: self._build_atlas(t, p))

        if not raw_key:
            return self._cached("default", lambda: self._build_gcode("default"))

        # Pick the longest token as the "subject" for both retrieval and
        # the LLM tier -- maximizes cache/dictionary reuse across similar
        # phrases (many entity labels are truncated substrings of the same
        # sentence) while still giving both a real content word to work
        # with, rather than a stopword.
        subject = max(tokens, key=len, default=raw_key)

        if subject:
            # Cheap exact-cache check first, for both prior tiers this
            # subject might already have resolved through -- avoids
            # rebuilding the embedder and avoids retrieval matching a
            # subject back to its own earlier LLM entry and re-wrapping it
            # under a different source label.
            for prefix in ("llm", "retrieved"):
                cache_key = f"{prefix}:{subject}"
                if cache_key in self._entries:
                    return self._entries[cache_key]

            retrieved = self._try_retrieval(subject)
            if retrieved is not None:
                self._remember(f"retrieved:{subject}", retrieved)
                return retrieved

        if self.allow_llm and subject:
            gcode_text = llm_gcode.generate_gcode_via_llm(subject, context=raw_key)
            if gcode_text is not None:
                entry = self._build_from_gcode_text(subject, gcode_text, source="llm")
                self._remember(f"llm:{subject}", entry)
                return entry

        return self._cached(f"neural:{raw_key}", lambda: self._build_neural(raw_key))

    def _cached(self, cache_key: str, builder: Callable[[], "WiringEntry"]) -> "WiringEntry":
        if cache_key not in self._entries:
            self._remember(cache_key, builder())
        return self._entries[cache_key]

    def _remember(self, cache_key: str, entry: "WiringEntry") -> None:
        """Every new entry, from any tier, goes through here -- this is
        what makes the dictionary persist across process restarts and
        become retrieval-searchable. A no-op beyond the in-memory cache if
        no dictionary_path was configured."""
        self._entries[cache_key] = entry
        if self._store is not None:
            self._store.save({
                k: {"word": e.word, "points": e.points, "node_count": e.node_count,
                   "source": e.source, "thermal_cost": e.thermal_cost}
                for k, e in self._entries.items()
            })

    def _retrieval_candidates(self) -> List[str]:
        """Known concept words worth reusing via semantic similarity: every
        gcode/atlas vocabulary word (buildable for free even if not yet
        built this session) plus every LLM-authored word this bank or a
        prior session has ever produced. Deliberately excludes "neural"-
        sourced entries -- those are arbitrary hash-driven shapes, and
        propagating one to an unrelated word via retrieval would spread
        noise, not real generalization."""
        pool = set(GCODE_LIBRARY.keys()) - {"default"}
        pool |= set(atlas_csg.ATLAS.keys())
        pool |= {e.word for e in self._entries.values() if e.source == "llm"}
        return sorted(pool)

    def _entry_for_known_word(self, word: str) -> Optional["WiringEntry"]:
        if word in GCODE_LIBRARY:
            return self._cached(word, lambda: self._build_gcode(word))
        if word in atlas_csg.ATLAS:
            parts = atlas_csg.ATLAS[word](0.80)
            return self._cached(f"atlas:{word}", lambda p=parts: self._build_atlas(word, p))
        for e in self._entries.values():
            if e.word == word and e.source == "llm":
                return e
        return None

    def _try_retrieval(self, subject: str) -> Optional["WiringEntry"]:
        """Best-effort: reuse an existing dictionary entry for a word
        semantically close to `subject`. Returns None (never raises) if
        scikit-learn isn't installed, there isn't enough vocabulary yet to
        fit an embedding space, or nothing clears the similarity bar --
        the caller falls through to the LLM/neural tiers exactly as if
        retrieval didn't exist."""
        candidates = self._retrieval_candidates()
        if len(candidates) < 2:
            return None
        if candidates != self._embedder_vocab:
            self._embedder_vocab = candidates
            try:
                self._embedder = semantic_embed.SemanticEmbed(candidates)
            except Exception:
                self._embedder = None
        if self._embedder is None:
            return None

        try:
            matches = self._embedder.nearest(subject, candidates, top_k=1)
        except Exception:
            return None
        if not matches or matches[0][1] < _RETRIEVAL_SIM_THRESHOLD:
            return None

        matched_word, _sim = matches[0]
        # A real, live-tested false positive: "wolf" scored 0.956 against
        # "woods" -- well above the threshold, purely from a shared "wo"
        # prefix, not any real relation (a wolf is not a tree). Cosine
        # similarity over char n-grams alone isn't a safe enough gate, so
        # require the literal substring relationship this tier actually
        # claims to detect ("near-exact lexical variants") as a second,
        # independent condition -- catches "mountains"/"mountain",
        # "bearish"/"bear", "crystals"/"crystal" (all trivially true) while
        # rejecting coincidental short-prefix matches like "wolf"/"woods".
        if matched_word not in subject and subject not in matched_word:
            return None
        source_entry = self._entry_for_known_word(matched_word)
        if source_entry is None:
            return None
        return WiringEntry(word=subject, points=source_entry.points,
                           node_count=source_entry.node_count,
                           source="retrieved", thermal_cost=source_entry.thermal_cost)

    @staticmethod
    def _thermal_cost(sdf_volume: np.ndarray) -> float:
        """1 - D_mean: 0 = cheap/stable to materialize, 1 = thermally
        critical. thermal.compute_thermal_volume expects a real signed
        volume (negative = interior); every tier below has one on hand at
        build time already, so this costs one extra pass over a grid we
        already computed, not a fresh one."""
        vol = thermal_mod.compute_thermal_volume(sdf_volume)
        return thermal_mod.thermal_summary(vol)['mean_cost']

    @staticmethod
    def _voxels_to_sdf(vox: np.ndarray) -> np.ndarray:
        """EDT reconstruction of a binary occupancy grid into a proper
        signed distance field -- same technique film_siren.psi_star and the
        atlas tier's exact SDFs already give us natively; the gcode tier
        only has a crystallized 0/1 voxel grid, so this is what makes its
        thermal_cost comparable in kind to the other two tiers'."""
        from scipy.ndimage import distance_transform_edt
        interior = vox > 0
        if not interior.any():
            # Nothing occupies space -- treat as all-exterior (cost ~0)
            # rather than feeding scipy's EDT an all-background mask.
            return np.ones_like(vox, dtype=np.float32)
        if interior.all():
            interior = interior.copy()
            interior[0, 0, 0] = False
        dist_ext = distance_transform_edt(~interior)
        dist_int = distance_transform_edt(interior)
        return (dist_ext - dist_int).astype(np.float32)

    def _build_gcode(self, word: str) -> WiringEntry:
        return self._build_from_gcode_text(word, GCODE_LIBRARY[word](), source="gcode")

    def _build_from_gcode_text(self, word: str, gcode_text: str, source: str) -> WiringEntry:
        """Shared by the hand-authored GCODE_LIBRARY tier and the LLM tier --
        both are just G-code text, differing only in who wrote it."""
        raw = parse_gcode(gcode_text)
        norm = normalize_pts(raw, self.scale)
        vox = voxelize(norm, self.resolution)
        vox = anneal_crystal(vox, norm, K=20)
        active = np.argwhere(vox).astype(np.float32)
        if len(active) == 0:
            active = np.zeros((10, 3), dtype=np.float32)
        points = (active / (self.resolution - 1) - 0.5) * self.scale
        # G-code's Z is vertical (printer convention); glTF/Cesium is y-up.
        # Swap so WiringEntry.points are already render-ready -- downstream
        # consumers (gltf_builder's point-cloud path) don't need to know
        # anything about G-code's own axis convention.
        points = points[:, [0, 2, 1]]
        cost = self._thermal_cost(self._voxels_to_sdf(vox))
        return WiringEntry(word=word, points=points, node_count=len(points),
                           source=source, thermal_cost=cost)

    def _sdf_interior_points(self, sdf_fn: Callable[[np.ndarray], np.ndarray]
                             ) -> Tuple[np.ndarray, np.ndarray]:
        # atlas_csg and film_siren both already use y-up, [-1,1]^3 coordinates
        # -- no axis swap needed here, unlike the raw G-code tier above.
        R = self.resolution
        lin = np.linspace(-1, 1, R)
        gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")
        pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)
        sdf = sdf_fn(pts)
        interior = pts[sdf < 0]
        if len(interior) == 0:
            interior = np.zeros((10, 3), dtype=np.float32)
        points = (interior * (self.scale / 2.0)).astype(np.float32)
        return points, sdf.reshape(R, R, R)

    def _build_atlas(self, token: str, parts: List["atlas_csg.Part"]) -> WiringEntry:
        points, sdf_grid = self._sdf_interior_points(lambda pts: atlas_csg.eval_parts(parts, pts))
        cost = self._thermal_cost(sdf_grid)
        return WiringEntry(word=token, points=points, node_count=len(points),
                           source="atlas", thermal_cost=cost)

    def _build_neural(self, phrase: str) -> WiringEntry:
        z = _hash_to_unit_vector(phrase)
        _, f_edt, grid_pts = film_siren.psi_star(z, resolution=self.resolution)
        interior = grid_pts[f_edt.reshape(-1) < 0]
        if len(interior) == 0:
            interior = np.zeros((10, 3), dtype=np.float32)
        points = (interior * (self.scale / 2.0)).astype(np.float32)
        cost = self._thermal_cost(f_edt)
        return WiringEntry(word=phrase, points=points, node_count=len(points),
                           source="neural", thermal_cost=cost)
