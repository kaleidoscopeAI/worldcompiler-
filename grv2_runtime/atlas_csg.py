"""grv2_runtime/atlas_csg.py -- psi_atlas: known words -> exact SDF via CSG part graphs.

Ported verbatim (minus the __main__ smoke test and the /home/claude path hack)
from the "Semantic Crystal Engine" backend at /home/jg/atlas_csg.py, which was
read in full and verified this session: installing its two missing deps
(scikit-learn, hnswlib) and running its own tests/test_all.py brought it from
9/26 to 26/26 passing. This module itself has zero third-party dependencies
beyond numpy.

Deterministic. No training. No stochasticity. No paid APIs. Pure closed-form
geometry (Inigo Quilez's exact SDF formulas) derived from real-world
proportions.

grv2_runtime.wiring.WiringBank uses this as its second lookup tier: a word
not in GCODE_LIBRARY (hand-authored) is tried here next (52 words across 18
families) before falling through to film_siren's neural fallback.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from enum import Enum


# ══════════════════════════════════════════════════════════════
# 1. EXACT SDF PRIMITIVES  (Inigo Quilez formulas)
# ══════════════════════════════════════════════════════════════

def sdf_sphere(p: np.ndarray, c: np.ndarray, r: float) -> np.ndarray:
    return np.linalg.norm(p - c, axis=1) - r

def sdf_ellipsoid(p: np.ndarray, c: np.ndarray, abc: np.ndarray) -> np.ndarray:
    q = (p - c) / abc
    k1 = np.linalg.norm(q, axis=1)
    s = float(np.min(abc))
    return (k1 - 1.0) * s

def sdf_capsule(p: np.ndarray, a: np.ndarray, b: np.ndarray, r: float) -> np.ndarray:
    ab = b - a
    l2 = np.dot(ab, ab) + 1e-12
    pa = p - a
    t = np.clip((pa @ ab) / l2, 0.0, 1.0)
    closest = a + t[:, None] * ab
    return np.linalg.norm(p - closest, axis=1) - r

def sdf_box(p: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.abs(p - c) - h
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return outside + inside

def sdf_torus(p: np.ndarray, c: np.ndarray, R: float, r: float) -> np.ndarray:
    d = p - c
    q = np.stack([np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2) - R, d[:, 1]], axis=1)
    return np.linalg.norm(q, axis=1) - r

def sdf_cylinder(p: np.ndarray, c: np.ndarray, r: float, h: float) -> np.ndarray:
    d = p - c
    q = np.abs(np.stack([np.sqrt(d[:, 0] ** 2 + d[:, 2] ** 2), np.abs(d[:, 1])], axis=1)) \
        - np.array([r, h])
    outside = np.linalg.norm(np.maximum(q, 0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0)
    return outside + inside


# ── CSG boolean operations ────────────────────────────────────────

def csg_union(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.minimum(a, b)

def csg_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, -b)

def csg_intersect(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(a, b)

def csg_smooth_union(a: np.ndarray, b: np.ndarray, k: float = 0.1) -> np.ndarray:
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    blend = a * (1 - h) + b * h - k * h * (1 - h)
    return np.minimum(blend, np.minimum(a, b))


# ══════════════════════════════════════════════════════════════
# 2. PART NODE
# ══════════════════════════════════════════════════════════════

class Op(Enum):
    UNION = 'union'
    SUBTRACT = 'subtract'
    INTERSECT = 'intersect'
    SMOOTH = 'smooth'

@dataclass
class Part:
    name: str
    kind: str
    params: Dict
    op: Op = Op.SMOOTH
    blend_k: float = 0.05

    def sdf(self, pts: np.ndarray) -> np.ndarray:
        p = self.params
        if self.kind == 'sphere':
            return sdf_sphere(pts, np.array(p['c']), p['r'])
        elif self.kind == 'ellipsoid':
            return sdf_ellipsoid(pts, np.array(p['c']), np.array(p['abc']))
        elif self.kind == 'capsule':
            return sdf_capsule(pts, np.array(p['a']), np.array(p['b']), p['r'])
        elif self.kind == 'box':
            return sdf_box(pts, np.array(p['c']), np.array(p['h']))
        elif self.kind == 'torus':
            return sdf_torus(pts, np.array(p['c']), p['R'], p['r'])
        elif self.kind == 'cylinder':
            return sdf_cylinder(pts, np.array(p['c']), p['r'], p['h'])
        else:
            raise ValueError(f"Unknown primitive: {self.kind}")


def eval_parts(parts: List[Part], pts: np.ndarray) -> np.ndarray:
    if not parts:
        return np.ones(len(pts)) * 999.0
    result = parts[0].sdf(pts)
    for part in parts[1:]:
        d = part.sdf(pts)
        if part.op == Op.UNION:
            result = csg_union(result, d)
        elif part.op == Op.SUBTRACT:
            result = csg_subtract(result, d)
        elif part.op == Op.INTERSECT:
            result = csg_intersect(result, d)
        elif part.op == Op.SMOOTH:
            result = csg_smooth_union(result, d, part.blend_k)
    return result


# ══════════════════════════════════════════════════════════════
# 3. SEMANTIC ATLAS -- real-world proportions, every measurement documented
# ══════════════════════════════════════════════════════════════

def atlas_bear(S: float = 0.82) -> List[Part]:
    k = 0.06
    return [
        Part('torso', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .42, S * .32, S * .55]}, Op.UNION),
        Part('hump', 'ellipsoid', {'c': [0, S * .28, -S * .10], 'abc': [S * .28, S * .18, S * .26]}, Op.SMOOTH, k),
        Part('head', 'ellipsoid', {'c': [0, S * .22, S * .60], 'abc': [S * .22, S * .20, S * .21]}, Op.SMOOTH, k),
        Part('snout', 'ellipsoid', {'c': [0, S * .08, S * .80], 'abc': [S * .13, S * .09, S * .14]}, Op.SMOOTH, k * .7),
        Part('brow', 'ellipsoid', {'c': [0, S * .34, S * .60], 'abc': [S * .20, S * .06, S * .10]}, Op.SMOOTH, k * .5),
        Part('ear_L', 'sphere', {'c': [-S * .14, S * .40, S * .56], 'r': S * .07}, Op.SMOOTH, k * .6),
        Part('ear_R', 'sphere', {'c': [S * .14, S * .40, S * .56], 'r': S * .07}, Op.SMOOTH, k * .6),
        Part('neck', 'capsule', {'a': [0, S * .06, S * .36], 'b': [0, S * .20, S * .54], 'r': S * .13}, Op.SMOOTH, k),
        Part('fl_leg', 'capsule', {'a': [-S * .20, S * .00, S * .32], 'b': [-S * .22, -S * .52, S * .28], 'r': S * .09}, Op.SMOOTH, k),
        Part('fr_leg', 'capsule', {'a': [S * .20, S * .00, S * .32], 'b': [S * .22, -S * .52, S * .28], 'r': S * .09}, Op.SMOOTH, k),
        Part('rl_leg', 'capsule', {'a': [-S * .22, S * .00, -S * .36], 'b': [-S * .24, -S * .52, -S * .32], 'r': S * .10}, Op.SMOOTH, k),
        Part('rr_leg', 'capsule', {'a': [S * .22, S * .00, -S * .36], 'b': [S * .24, -S * .52, -S * .32], 'r': S * .10}, Op.SMOOTH, k),
        Part('fl_paw', 'ellipsoid', {'c': [-S * .22, -S * .56, S * .34], 'abc': [S * .10, S * .04, S * .14]}, Op.SMOOTH, k * .5),
        Part('fr_paw', 'ellipsoid', {'c': [S * .22, -S * .56, S * .34], 'abc': [S * .10, S * .04, S * .14]}, Op.SMOOTH, k * .5),
        Part('rl_paw', 'ellipsoid', {'c': [-S * .24, -S * .56, -S * .28], 'abc': [S * .11, S * .04, S * .15]}, Op.SMOOTH, k * .5),
        Part('rr_paw', 'ellipsoid', {'c': [S * .24, -S * .56, -S * .28], 'abc': [S * .11, S * .04, S * .15]}, Op.SMOOTH, k * .5),
        Part('tail', 'sphere', {'c': [0, S * .10, -S * .58], 'r': S * .06}, Op.SMOOTH, k * .4),
    ]

def atlas_tree(S: float = 0.82) -> List[Part]:
    k = 0.05
    return [
        Part('trunk', 'capsule', {'a': [0, -S * .85, 0], 'b': [0, S * .10, 0], 'r': S * .06}, Op.UNION),
        Part('roots', 'ellipsoid', {'c': [0, -S * .80, 0], 'abc': [S * .12, S * .06, S * .12]}, Op.SMOOTH, k),
        Part('foliage_1', 'ellipsoid', {'c': [0, -S * .30, 0], 'abc': [S * .48, S * .22, S * .48]}, Op.SMOOTH, k * .8),
        Part('foliage_2', 'ellipsoid', {'c': [0, S * .10, 0], 'abc': [S * .38, S * .20, S * .38]}, Op.SMOOTH, k * .8),
        Part('foliage_3', 'ellipsoid', {'c': [0, S * .42, 0], 'abc': [S * .26, S * .18, S * .26]}, Op.SMOOTH, k * .8),
        Part('foliage_4', 'ellipsoid', {'c': [0, S * .66, 0], 'abc': [S * .14, S * .16, S * .14]}, Op.SMOOTH, k * .8),
        Part('tip', 'capsule', {'a': [0, S * .72, 0], 'b': [0, S * .90, 0], 'r': S * .03}, Op.SMOOTH, k * .5),
    ]

def atlas_human(S: float = 0.80) -> List[Part]:
    k = 0.04
    return [
        Part('torso', 'ellipsoid', {'c': [0, S * .10, 0], 'abc': [S * .24, S * .32, S * .14]}, Op.UNION),
        Part('pelvis', 'ellipsoid', {'c': [0, -S * .22, 0], 'abc': [S * .20, S * .14, S * .12]}, Op.SMOOTH, k),
        Part('head', 'sphere', {'c': [0, S * .52, 0], 'r': S * .18}, Op.SMOOTH, k),
        Part('neck', 'capsule', {'a': [0, S * .32, 0], 'b': [0, S * .42, 0], 'r': S * .07}, Op.SMOOTH, k),
        Part('l_shoulder', 'sphere', {'c': [-S * .28, S * .28, 0], 'r': S * .10}, Op.SMOOTH, k),
        Part('r_shoulder', 'sphere', {'c': [S * .28, S * .28, 0], 'r': S * .10}, Op.SMOOTH, k),
        Part('l_uarm', 'capsule', {'a': [-S * .28, S * .28, 0], 'b': [-S * .32, S * .06, 0], 'r': S * .07}, Op.SMOOTH, k),
        Part('r_uarm', 'capsule', {'a': [S * .28, S * .28, 0], 'b': [S * .32, S * .06, 0], 'r': S * .07}, Op.SMOOTH, k),
        Part('l_farm', 'capsule', {'a': [-S * .32, S * .06, 0], 'b': [-S * .34, -S * .18, 0], 'r': S * .06}, Op.SMOOTH, k),
        Part('r_farm', 'capsule', {'a': [S * .32, S * .06, 0], 'b': [S * .34, -S * .18, 0], 'r': S * .06}, Op.SMOOTH, k),
        Part('l_thigh', 'capsule', {'a': [-S * .12, -S * .22, 0], 'b': [-S * .12, -S * .54, 0], 'r': S * .09}, Op.SMOOTH, k),
        Part('r_thigh', 'capsule', {'a': [S * .12, -S * .22, 0], 'b': [S * .12, -S * .54, 0], 'r': S * .09}, Op.SMOOTH, k),
        Part('l_shin', 'capsule', {'a': [-S * .12, -S * .54, 0], 'b': [-S * .12, -S * .82, 0], 'r': S * .07}, Op.SMOOTH, k),
        Part('r_shin', 'capsule', {'a': [S * .12, -S * .54, 0], 'b': [S * .12, -S * .82, 0], 'r': S * .07}, Op.SMOOTH, k),
        Part('l_foot', 'ellipsoid', {'c': [-S * .12, -S * .86, S * .08], 'abc': [S * .08, S * .06, S * .16]}, Op.SMOOTH, k * .5),
        Part('r_foot', 'ellipsoid', {'c': [S * .12, -S * .86, S * .08], 'abc': [S * .08, S * .06, S * .16]}, Op.SMOOTH, k * .5),
    ]

def atlas_mountain(S: float = 0.85) -> List[Part]:
    k = 0.12
    return [
        Part('base', 'ellipsoid', {'c': [0, -S * .20, 0], 'abc': [S * .90, S * .60, S * .85]}, Op.UNION),
        Part('peak', 'ellipsoid', {'c': [0, S * .55, 0], 'abc': [S * .28, S * .55, S * .28]}, Op.SMOOTH, k),
        Part('ridge_L', 'ellipsoid', {'c': [-S * .40, S * .10, 0], 'abc': [S * .25, S * .40, S * .20]}, Op.SMOOTH, k),
        Part('ridge_R', 'ellipsoid', {'c': [S * .40, S * .10, 0], 'abc': [S * .25, S * .40, S * .20]}, Op.SMOOTH, k),
        Part('snow', 'sphere', {'c': [0, S * .80, 0], 'r': S * .20}, Op.SMOOTH, k * .4),
        Part('crevasse', 'capsule', {'a': [S * .10, S * .40, 0], 'b': [S * .25, S * .80, 0], 'r': S * .04}, Op.SUBTRACT, 0),
    ]

def atlas_crystal(S: float = 0.80) -> List[Part]:
    k = 0.02
    return [
        Part('core', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .20, S * .60, S * .20]}, Op.UNION),
        Part('spike_1', 'capsule', {'a': [0, 0, 0], 'b': [S * .45, S * .70, 0], 'r': S * .08}, Op.SMOOTH, k),
        Part('spike_2', 'capsule', {'a': [0, 0, 0], 'b': [-S * .45, S * .70, 0], 'r': S * .08}, Op.SMOOTH, k),
        Part('spike_3', 'capsule', {'a': [0, 0, 0], 'b': [0, S * .70, S * .45], 'r': S * .08}, Op.SMOOTH, k),
        Part('spike_4', 'capsule', {'a': [0, 0, 0], 'b': [0, S * .70, -S * .45], 'r': S * .08}, Op.SMOOTH, k),
        Part('base_1', 'capsule', {'a': [0, 0, 0], 'b': [S * .30, -S * .50, S * .20], 'r': S * .10}, Op.SMOOTH, k),
        Part('base_2', 'capsule', {'a': [0, 0, 0], 'b': [-S * .30, -S * .50, -S * .20], 'r': S * .10}, Op.SMOOTH, k),
    ]

def atlas_fire(S: float = 0.80) -> List[Part]:
    k = 0.10
    return [
        Part('base', 'ellipsoid', {'c': [0, -S * .40, 0], 'abc': [S * .40, S * .30, S * .35]}, Op.UNION),
        Part('body', 'ellipsoid', {'c': [0, S * .00, 0], 'abc': [S * .28, S * .55, S * .25]}, Op.SMOOTH, k),
        Part('tongue_1', 'capsule', {'a': [S * .08, S * .20, 0], 'b': [S * .15, S * .75, 0], 'r': S * .12}, Op.SMOOTH, k),
        Part('tongue_2', 'capsule', {'a': [-S * .08, S * .10, 0], 'b': [-S * .18, S * .70, 0], 'r': S * .10}, Op.SMOOTH, k),
        Part('tongue_3', 'capsule', {'a': [0, S * .25, S * .06], 'b': [0, S * .85, S * .05], 'r': S * .09}, Op.SMOOTH, k),
        Part('tip', 'sphere', {'c': [0, S * .90, 0], 'r': S * .07}, Op.SMOOTH, k * .6),
    ]

def atlas_water(S: float = 0.82) -> List[Part]:
    k = 0.08
    return [
        Part('body', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .80, S * .12, S * .90]}, Op.UNION),
        Part('wave_1', 'torus', {'c': [0, S * .08, 0], 'R': S * .45, 'r': S * .08}, Op.SMOOTH, k),
        Part('wave_2', 'torus', {'c': [0, S * .04, 0], 'R': S * .25, 'r': S * .06}, Op.SMOOTH, k),
        Part('crest', 'ellipsoid', {'c': [S * .30, S * .18, 0], 'abc': [S * .25, S * .12, S * .20]}, Op.SMOOTH, k),
    ]

def atlas_sphere(S: float = 0.80) -> List[Part]:
    return [Part('sphere', 'sphere', {'c': [0, 0, 0], 'r': S * .75}, Op.UNION)]

def atlas_cube(S: float = 0.80) -> List[Part]:
    return [Part('cube', 'box', {'c': [0, 0, 0], 'h': [S * .65, S * .65, S * .65]}, Op.UNION)]

def atlas_torus(S: float = 0.80) -> List[Part]:
    return [Part('torus', 'torus', {'c': [0, 0, 0], 'R': S * .55, 'r': S * .20}, Op.UNION)]

def atlas_void(S: float = 0.80) -> List[Part]:
    k = 0.02
    return [
        Part('outer', 'sphere', {'c': [0, 0, 0], 'r': S * .80}, Op.UNION),
        Part('inner', 'sphere', {'c': [0, 0, 0], 'r': S * .70}, Op.SUBTRACT),
    ]

def atlas_device(S: float = 0.78) -> List[Part]:
    """Generic mechanical device/tool/instrument -- a body, a barrel, and a
    handle. Standing in for anything whose WordNet hypernym chain bottoms
    out at 'device'/'instrumentality'/'instrument'/'machine' (telescope,
    hammer, wrench, carburetor, engine, ...) rather than reaching a more
    specific hand-authored or CSG shape -- added so grv2_runtime.
    definition_compiler has a landing point for this WordNet-common a
    category, not because any single one of those words looks like this."""
    k = 0.05
    return [
        Part('body', 'box', {'c': [0, 0, 0], 'h': [S * .22, S * .16, S * .30]}, Op.UNION),
        Part('barrel', 'cylinder', {'c': [0, S * .05, S * .45], 'r': S * .10, 'h': S * .22}, Op.SMOOTH, k),
        Part('handle', 'capsule', {'a': [0, -S * .16, -S * .20], 'b': [0, -S * .55, -S * .30], 'r': S * .07}, Op.SMOOTH, k),
        Part('knob', 'sphere', {'c': [S * .18, S * .10, -S * .10], 'r': S * .07}, Op.SMOOTH, k * .6),
    ]

def atlas_vehicle(S: float = 0.80) -> List[Part]:
    """Generic wheeled vehicle -- a body on four wheels. Standing in for
    anything whose hypernym chain reaches 'vehicle'/'wheeled_vehicle'
    (bicycle, car, cart, wagon, ...)."""
    k = 0.05
    return [
        Part('body', 'box', {'c': [0, S * .10, 0], 'h': [S * .30, S * .14, S * .55]}, Op.UNION),
        Part('cabin', 'box', {'c': [0, S * .30, -S * .05], 'h': [S * .24, S * .12, S * .28]}, Op.SMOOTH, k),
        Part('wheel_fl', 'torus', {'c': [-S * .32, -S * .10, S * .35], 'R': S * .16, 'r': S * .06}, Op.SMOOTH, k),
        Part('wheel_fr', 'torus', {'c': [S * .32, -S * .10, S * .35], 'R': S * .16, 'r': S * .06}, Op.SMOOTH, k),
        Part('wheel_rl', 'torus', {'c': [-S * .32, -S * .10, -S * .35], 'R': S * .16, 'r': S * .06}, Op.SMOOTH, k),
        Part('wheel_rr', 'torus', {'c': [S * .32, -S * .10, -S * .35], 'R': S * .16, 'r': S * .06}, Op.SMOOTH, k),
    ]

def atlas_fruit(S: float = 0.78) -> List[Part]:
    """Generic round fruit -- a body, a stem, a blossom-end dimple. Standing
    in for anything whose hypernym chain reaches 'fruit'/'edible_fruit',
    which is where a lot of common produce words (apple, ...) actually land
    via their #1 WordNet sense (the food, not the tree that grows it --
    that sense is already covered by GCODE_LIBRARY's tree/atlas_tree)."""
    k = 0.04
    return [
        Part('body', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .48, S * .42, S * .48]}, Op.UNION),
        Part('stem', 'capsule', {'a': [0, S * .40, 0], 'b': [0, S * .60, 0], 'r': S * .03}, Op.SMOOTH, k),
        Part('dimple', 'sphere', {'c': [0, -S * .40, 0], 'r': S * .10}, Op.SUBTRACT, 0),
    ]

def atlas_food(S: float = 0.78) -> List[Part]:
    """Generic foodstuff -- a rounded loaf. Standing in for anything whose
    hypernym chain reaches 'food'/'foodstuff' (bread, cheese, ...)."""
    k = 0.06
    return [
        Part('loaf', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .55, S * .30, S * .38]}, Op.UNION),
        Part('crust_ridge', 'capsule', {'a': [-S * .30, S * .20, 0], 'b': [S * .30, S * .20, 0], 'r': S * .10}, Op.SMOOTH, k),
    ]

def atlas_clothing(S: float = 0.80) -> List[Part]:
    """Generic worn covering -- a hollow shell with an opening. Standing in
    for anything whose hypernym chain reaches 'clothing'/'covering'/
    'garment' (boot, helmet, glove, ...)."""
    return [
        Part('outer', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .40, S * .48, S * .40]}, Op.UNION),
        Part('inner', 'ellipsoid', {'c': [0, S * .05, 0], 'abc': [S * .32, S * .42, S * .32]}, Op.SUBTRACT, 0),
        Part('opening', 'cylinder', {'c': [0, -S * .44, 0], 'r': S * .28, 'h': S * .10}, Op.SUBTRACT, 0),
    ]

def atlas_cylinder(S: float = 0.80) -> List[Part]:
    return [Part('cylinder', 'cylinder', {'c': [0, 0, 0], 'r': S * .45, 'h': S * .70}, Op.UNION)]

def atlas_material(S: float = 0.80) -> List[Part]:
    """Generic raw-material chunk -- an irregular block. Standing in for
    anything whose hypernym chain reaches 'material'/'substance'/'matter'
    -- notably where common tree-species words (oak, pine, ...) actually
    land via their #1 WordNet sense, which is the wood/lumber, not the
    living tree (that's GCODE_LIBRARY's tree/atlas_tree, a different
    sense of the same word)."""
    k = 0.15
    return [
        Part('block', 'box', {'c': [0, 0, 0], 'h': [S * .45, S * .32, S * .38]}, Op.UNION),
        Part('chip_1', 'ellipsoid', {'c': [S * .30, S * .20, S * .10], 'abc': [S * .18, S * .14, S * .16]}, Op.SMOOTH, k),
        Part('chip_2', 'ellipsoid', {'c': [-S * .28, -S * .18, -S * .12], 'abc': [S * .16, S * .12, S * .14]}, Op.SMOOTH, k),
    ]

def atlas_plant(S: float = 0.76) -> List[Part]:
    """Generic bushy/flowering plant -- a short stem with a cluster of
    foliage, structurally distinct from atlas_tree's tall trunk + layered
    canopy. Standing in for anything whose hypernym chain reaches
    'plant'/'shrub'/'vascular_plant' (rose, cactus, fern, ...) without a
    more specific hand-authored or CSG match."""
    k = 0.05
    return [
        Part('stem', 'capsule', {'a': [0, -S * .60, 0], 'b': [0, -S * .10, 0], 'r': S * .05}, Op.UNION),
        Part('leaf_1', 'ellipsoid', {'c': [S * .18, S * .05, 0], 'abc': [S * .22, S * .14, S * .16]}, Op.SMOOTH, k),
        Part('leaf_2', 'ellipsoid', {'c': [-S * .18, S * .00, S * .05], 'abc': [S * .20, S * .13, S * .15]}, Op.SMOOTH, k),
        Part('leaf_3', 'ellipsoid', {'c': [0, S * .18, -S * .10], 'abc': [S * .16, S * .15, S * .14]}, Op.SMOOTH, k),
    ]

def atlas_storm(S: float = 0.85) -> List[Part]:
    """Generic turbulent atmospheric mass -- larger and more overlapping
    than GCODE_LIBRARY's gcode_cloud (a calm/decorative cloud). Standing in
    for anything whose hypernym chain reaches 'storm'/'atmospheric_
    phenomenon' (thunderstorm, ...)."""
    k = 0.10
    return [
        Part('core', 'ellipsoid', {'c': [0, 0, 0], 'abc': [S * .55, S * .30, S * .50]}, Op.UNION),
        Part('mass_1', 'ellipsoid', {'c': [S * .35, S * .10, S * .10], 'abc': [S * .30, S * .22, S * .28]}, Op.SMOOTH, k),
        Part('mass_2', 'ellipsoid', {'c': [-S * .32, S * .05, -S * .15], 'abc': [S * .28, S * .20, S * .26]}, Op.SMOOTH, k),
        Part('trail', 'capsule', {'a': [0, -S * .20, 0], 'b': [0, -S * .70, 0], 'r': S * .05}, Op.SMOOTH, k * .4),
    ]

def atlas_structure(S: float = 0.85) -> List[Part]:
    """Generic built structure -- a stacked architectural block. Standing
    in for anything whose hypernym chain reaches 'structure' or 'tower'
    (bridge, lighthouse, ...)."""
    k = 0.08
    return [
        Part('base', 'box', {'c': [0, -S * .30, 0], 'h': [S * .40, S * .12, S * .40]}, Op.UNION),
        Part('tower', 'box', {'c': [0, S * .10, 0], 'h': [S * .18, S * .40, S * .18]}, Op.SMOOTH, k),
        Part('cap', 'box', {'c': [0, S * .54, 0], 'h': [S * .22, S * .06, S * .22]}, Op.SMOOTH, k),
    ]

def atlas_chair(S: float = 0.82) -> List[Part]:
    """Chair -- seat, backrest, four legs, two armrests. Sharp-edged (small
    k) unlike the organic families above, since furniture reads as blocky."""
    k = 0.02
    return [
        Part('seat', 'box', {'c': [0, 0, 0], 'h': [S * .38, S * .04, S * .36]}, Op.UNION),
        Part('back', 'box', {'c': [0, S * .38, -S * .34], 'h': [S * .38, S * .40, S * .03]}, Op.SMOOTH, k),
        Part('leg_fl', 'box', {'c': [-S * .32, -S * .38, S * .30], 'h': [S * .03, S * .38, S * .03]}, Op.SMOOTH, k),
        Part('leg_fr', 'box', {'c': [S * .32, -S * .38, S * .30], 'h': [S * .03, S * .38, S * .03]}, Op.SMOOTH, k),
        Part('leg_rl', 'box', {'c': [-S * .32, -S * .38, -S * .30], 'h': [S * .03, S * .38, S * .03]}, Op.SMOOTH, k),
        Part('leg_rr', 'box', {'c': [S * .32, -S * .38, -S * .30], 'h': [S * .03, S * .38, S * .03]}, Op.SMOOTH, k),
        Part('arm_L', 'box', {'c': [-S * .40, S * .20, -S * .04], 'h': [S * .03, S * .04, S * .34]}, Op.SMOOTH, k),
        Part('arm_R', 'box', {'c': [S * .40, S * .20, -S * .04], 'h': [S * .03, S * .04, S * .34]}, Op.SMOOTH, k),
    ]

def atlas_skull(S: float = 0.84) -> List[Part]:
    """Skull -- cranium, face plate, cheekbones, jaw, with eye sockets and
    a nasal cavity carved out via Op.SUBTRACT (this atlas's first user of
    subtraction, not just union/smooth-union)."""
    k = 0.06
    return [
        Part('cranium', 'ellipsoid', {'c': [0, S * .18, 0], 'abc': [S * .40, S * .46, S * .44]}, Op.UNION),
        Part('face', 'ellipsoid', {'c': [0, -S * .10, S * .34], 'abc': [S * .32, S * .34, S * .18]}, Op.SMOOTH, k),
        Part('cheek_L', 'ellipsoid', {'c': [-S * .26, -S * .06, S * .26], 'abc': [S * .12, S * .10, S * .10]}, Op.SMOOTH, k * .7),
        Part('cheek_R', 'ellipsoid', {'c': [S * .26, -S * .06, S * .26], 'abc': [S * .12, S * .10, S * .10]}, Op.SMOOTH, k * .7),
        Part('eye_L', 'ellipsoid', {'c': [-S * .16, S * .06, S * .44], 'abc': [S * .10, S * .09, S * .06]}, Op.SUBTRACT),
        Part('eye_R', 'ellipsoid', {'c': [S * .16, S * .06, S * .44], 'abc': [S * .10, S * .09, S * .06]}, Op.SUBTRACT),
        Part('nasal', 'ellipsoid', {'c': [0, -S * .06, S * .50], 'abc': [S * .06, S * .08, S * .06]}, Op.SUBTRACT),
        Part('jaw', 'ellipsoid', {'c': [0, -S * .36, S * .12], 'abc': [S * .30, S * .14, S * .22]}, Op.SMOOTH, k),
        Part('teeth', 'box', {'c': [0, -S * .22, S * .36], 'h': [S * .24, S * .03, S * .04]}, Op.SMOOTH, k * .3),
    ]

def atlas_rocket(S: float = 0.87) -> List[Part]:
    """Rocket -- capsule body, nose cone, four fins, engine bell, a window
    subtracted out of the body."""
    k = 0.04
    return [
        Part('body', 'capsule', {'a': [0, -S * .50, 0], 'b': [0, S * .55, 0], 'r': S * .18}, Op.UNION),
        Part('nose', 'ellipsoid', {'c': [0, S * .72, 0], 'abc': [S * .14, S * .26, S * .14]}, Op.SMOOTH, k),
        Part('fin1', 'box', {'c': [S * .28, -S * .36, 0], 'h': [S * .14, S * .18, S * .02]}, Op.SMOOTH, k),
        Part('fin2', 'box', {'c': [-S * .28, -S * .36, 0], 'h': [S * .14, S * .18, S * .02]}, Op.SMOOTH, k),
        Part('fin3', 'box', {'c': [0, -S * .36, S * .28], 'h': [S * .02, S * .18, S * .14]}, Op.SMOOTH, k),
        Part('fin4', 'box', {'c': [0, -S * .36, -S * .28], 'h': [S * .02, S * .18, S * .14]}, Op.SMOOTH, k),
        Part('engine', 'cylinder', {'c': [0, -S * .66, 0], 'r': S * .20, 'h': S * .10}, Op.SMOOTH, k),
        Part('window', 'sphere', {'c': [0, S * .28, S * .18], 'r': S * .06}, Op.SUBTRACT),
    ]

def atlas_spiral(S: float = 0.80) -> List[Part]:
    k = 0.04
    parts = [Part('core', 'torus', {'c': [0, 0, 0], 'R': S * .20, 'r': S * .08}, Op.UNION)]
    n = 8
    for i in range(n):
        a = 2 * np.pi * i / n
        r_i = S * (0.25 + 0.60 * i / n)
        y_i = S * (-0.40 + 0.80 * i / n)
        x_i = r_i * np.cos(a)
        z_i = r_i * np.sin(a)
        parts.append(Part(f'seg_{i}', 'sphere',
                          {'c': [x_i, y_i, z_i], 'r': S * (0.08 + 0.04 * i / n)},
                          Op.SMOOTH, k))
    return parts


# ══════════════════════════════════════════════════════════════
# 4. ATLAS REGISTRY
# ══════════════════════════════════════════════════════════════

ATLAS: Dict[str, Callable] = {
    'bear': atlas_bear, 'grizzly': atlas_bear, 'animal': atlas_bear,
    'tree': atlas_tree, 'forest': atlas_tree, 'pine': atlas_tree,
    'human': atlas_human, 'person': atlas_human, 'figure': atlas_human,
    'mountain': atlas_mountain, 'peak': atlas_mountain, 'hill': atlas_mountain,
    # 'formation'/'ice' -- not 'geological_formation'/'ice_mass' verbatim:
    # definition_compiler._lemma_candidates splits WordNet's underscored
    # multi-word lemmas into individual tokens (head noun first), so the
    # registered key has to be the token it will actually look up.
    'formation': atlas_mountain, 'ice': atlas_mountain,
    'crystal': atlas_crystal, 'gem': atlas_crystal, 'diamond': atlas_crystal,
    'fire': atlas_fire, 'flame': atlas_fire, 'blaze': atlas_fire,
    'water': atlas_water, 'wave': atlas_water, 'ocean': atlas_water,
    'sphere': atlas_sphere, 'ball': atlas_sphere, 'globe': atlas_sphere,
    'cube': atlas_cube, 'box': atlas_cube, 'block': atlas_cube,
    'torus': atlas_torus, 'ring': atlas_torus, 'donut': atlas_torus,
    'void': atlas_void, 'empty': atlas_void, 'nothing': atlas_void,
    'spiral': atlas_spiral, 'helix': atlas_spiral, 'coil': atlas_spiral,
    'device': atlas_device, 'tool': atlas_device, 'instrument': atlas_device, 'machine': atlas_device,
    'instrumentality': atlas_device, 'equipment': atlas_device,
    'vehicle': atlas_vehicle, 'car': atlas_vehicle, 'automobile': atlas_vehicle,
    'fruit': atlas_fruit, 'berry': atlas_fruit,
    'food': atlas_food, 'foodstuff': atlas_food,
    'clothing': atlas_clothing, 'garment': atlas_clothing, 'covering': atlas_clothing,
    'cylinder': atlas_cylinder, 'tube': atlas_cylinder,
    # NOT 'substance': matching is by lemma string, not WordNet sense, and
    # "substance" is also a lemma of an unrelated synset meaning "the gist
    # of a message" (message/content/subject_matter/substance) -- "irony"
    # (sense 1: sarcasm -> wit -> message) hit exactly that collision and
    # resolved into a physical-material blob for a word about rhetoric.
    # 'material'/'wood' are specific enough not to have this problem and
    # already cover what 'substance' was added for (oak resolves via
    # 'wood' at a nearer hop regardless).
    'material': atlas_material, 'wood': atlas_material,
    'plant': atlas_plant, 'shrub': atlas_plant, 'flower': atlas_plant,
    'storm': atlas_storm,
    'structure': atlas_structure, 'tower': atlas_structure,
    'polyhedron': atlas_cube, 'solid': atlas_cube,
    'chair': atlas_chair, 'seat': atlas_chair,
    'skull': atlas_skull,
    'rocket': atlas_rocket,
    # NOT 'ship': a cargo ship and a rocket don't look alike -- same
    # discipline as dropping 'artifact'/'substance' earlier, an
    # imprecise-but-plausible-sounding synonym is still a real mismatch.
}
# Known residual risk, not fully closed: matching is by lemma *string*,
# not WordNet sense, so any registered key that's also a lemma of an
# unrelated synset can collide -- 'substance' (removed below) collided for
# real; 'plant' (factory buildings, plant.n.01) vs. the organism sense this
# key is meant for (plant.n.02) is a known, structurally identical risk
# that just hasn't been hit by a real word yet. Full word-sense
# disambiguation would close this properly; out of scope for this pass.
#
# Deliberately NOT registered: 'artifact', 'phenomenon'. Both look like
# reasonable generic catch-alls but are near-universal WordNet ancestors --
# "wildcat" (sense 1, an oil well) reaches 'artifact' in exactly 4 hops via
# well -> excavation -> artifact, which would have silently resolved a word
# about a cat into a generic structure blob from its most obscure sense.
# That's the same class of danger the sense-1-only rule in
# definition_compiler.py exists to prevent: a technically-real hypernym
# path through the *wrong* meaning is still confidently wrong, not merely
# imprecise. 'structure'/'tower' are kept because they're reached at a much
# nearer hop for words that actually mean a structure (bridge, lighthouse).

def lookup_atlas(word: str, S: float = 0.80) -> Optional[List[Part]]:
    fn = ATLAS.get(word.lower().strip())
    if fn:
        return fn(S)
    for key, fn in ATLAS.items():
        if key in word.lower() or word.lower() in key:
            return fn(S)
    return None
