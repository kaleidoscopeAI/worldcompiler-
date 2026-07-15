"""
world_compiler_core.py
======================
World Compiler — core dispatch, field operators, scene graph, and predicate resolver.

Architecture:
    W_t = W_{t-1} ⊕ Λ(u_t, W_{t-1})
    Λ   = Λ_res ∘ Λ_verb[stage] ∘ Λ_geom ∘ Λ_canon

Compilation targets:
    Physical     → atlas lookup or parametric primitive synthesis
    AbstractNoun → parametric RBF field  (identity lives in parameters, not silhouette)
    Relational   → lazy Predicate on scene graph  (binds at render time)
    FunctionWord → ControlToken  (staging, scope, CSG mode, retrieval flag)

⊕ dispatch:
    spatial verb       → SceneAppend
    physical_interaction verb → CSG(op, k)
    material_state verb      → FieldBlend(α)
"""

from __future__ import annotations
import numpy as np
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple
from enum import Enum, auto
import math


# ---------------------------------------------------------------------------
# 1.  Enumerations
# ---------------------------------------------------------------------------

class WordClass(Enum):
    PHYSICAL       = auto()
    ABSTRACT_NOUN  = auto()
    RELATIONAL     = auto()
    FUNCTION_WORD  = auto()

class VerbType(Enum):
    SPATIAL              = auto()   # sits_on, hovers_above  → SceneAppend
    PHYSICAL_INTERACTION = auto()   # carves, crushes        → CSG
    MATERIAL_STATE       = auto()   # melts, crystallises    → FieldBlend

class VerbStage(Enum):
    PRE_CANON  = auto()   # modifies base material / initialisation
    POST_CANON = auto()   # transforms canonical object after formation
    POST_GEOM  = auto()   # final field edits after geometric assembly

class CSGMode(Enum):
    UNION      = auto()
    SUBTRACT   = auto()
    INTERSECT  = auto()
    BLEND      = auto()

class RetrievalFlag(Enum):
    RETRIEVE   = auto()   # "the" → look up existing atlas entry
    SYNTHESIZE = auto()   # "a"   → create with variation


# ---------------------------------------------------------------------------
# 2.  Compiled token types
# ---------------------------------------------------------------------------

@dataclass
class PhysicalToken:
    word:        str
    gcode_path:  Optional[str]        = None   # path to .gcode file if known
    sdf_key:     Optional[str]        = None   # key into SDF atlas if pre-built
    occupancy:   Optional[np.ndarray] = None   # shape (R,R,R) bool; None until resolved

    def resolve_occupancy(self, resolution: int) -> np.ndarray:
        """
        Rasterize G-code → occupancy, or fall back to a unit-cube primitive.
        Result is cached on self.occupancy after first call.
        Note: parse_gcode_moves / rasterize_toolpath are defined in section 13.
        """
        if self.occupancy is not None:
            return self.occupancy

        if self.gcode_path and os.path.exists(self.gcode_path):
            moves = parse_gcode_moves(self.gcode_path)
            self.occupancy = rasterize_toolpath(moves, resolution)
        else:
            # Fallback: solid cube in the centre quarter of the volume
            occ = np.zeros((resolution,) * 3, dtype=bool)
            lo, hi = resolution // 4, 3 * resolution // 4
            occ[lo:hi, lo:hi, lo:hi] = True
            self.occupancy = occ

        return self.occupancy

@dataclass
class RBFField:
    """
    Abstract-noun canonical field.
    Evaluated at any point x ∈ ℝ³ as:
        φ(x) = amplitude · exp(-0.5 · (x-centre)ᵀ A⁻¹ (x-centre))
    Identity lives in (amplitude, σ, anisotropy), not in silhouette.
    """
    word:        str
    amplitude:   float                = 1.0
    sigma:       float                = 1.0
    centre:      np.ndarray           = field(default_factory=lambda: np.zeros(3))
    anisotropy:  np.ndarray           = field(default_factory=lambda: np.eye(3))  # 3×3

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate field at point x (shape (3,))."""
        d = x - self.centre
        exponent = -0.5 * float(d @ np.linalg.solve(self.anisotropy, d))
        return self.amplitude * math.exp(exponent)

    def to_sdf_volume(self, resolution: int = 32, iso: float = 0.5) -> np.ndarray:
        """Rasterise to a signed-distance-like volume for atlas insertion."""
        grid = np.linspace(-3 * self.sigma, 3 * self.sigma, resolution)
        xs, ys, zs = np.meshgrid(grid, grid, grid, indexing='ij')
        pts = np.stack([xs, ys, zs], axis=-1)  # (R,R,R,3)
        d   = pts - self.centre                 # broadcast
        inv = np.linalg.inv(self.anisotropy)
        # batched mahalanobis
        tmp = np.einsum('...i,ij->...j', d, inv)
        mah = np.einsum('...i,...i->...', tmp, d)
        density = self.amplitude * np.exp(-0.5 * mah)
        # convert to pseudo-SDF: negative inside, positive outside
        return iso - density


@dataclass
class Predicate:
    """
    Lazy relational constraint.  Binds at render time when both args are
    present in the scene graph.

    Example:
        above(cat, chair) → translate cat so cat.centroid.z > chair.top + margin
    """
    name:    str                               # "above", "inside", "between", …
    fn:      Callable[..., None]               # resolver: (node_a, node_b, scene) → None
    arity:   int                = 2
    args:    List[str]          = field(default_factory=list)   # bound node ids
    margin:  float              = 0.0
    resolved: bool              = False

    def bind(self, *node_ids: str) -> "Predicate":
        assert len(node_ids) == self.arity
        return Predicate(
            name=self.name, fn=self.fn, arity=self.arity,
            args=list(node_ids), margin=self.margin
        )

    def resolve(self, scene: "SceneGraph") -> bool:
        """Attempt to resolve; returns True if successful."""
        if len(self.args) < self.arity:
            return False
        nodes = [scene.get_node(a) for a in self.args]
        if any(n is None for n in nodes):
            return False
        self.fn(*nodes, scene=scene, margin=self.margin)
        self.resolved = True
        return True


@dataclass
class ControlToken:
    """
    Function-word compiled token.
    Modifies compilation context rather than adding geometry.
    """
    word:           str
    stage:          Optional[VerbStage]     = None
    csg_mode:       Optional[CSGMode]       = None
    scope:          Optional[str]           = None   # e.g. "negation", "universal"
    retrieval_flag: Optional[RetrievalFlag] = None
    noise_scale:    float                   = 0.0    # for indefinite articles

    def apply_to_context(self, ctx: "CompileContext") -> None:
        if self.csg_mode       is not None: ctx.csg_mode       = self.csg_mode
        if self.stage          is not None: ctx.current_stage  = self.stage
        if self.scope          is not None: ctx.scope_stack.append(self.scope)
        if self.retrieval_flag is not None: ctx.retrieval_flag = self.retrieval_flag
        if self.noise_scale    > 0:         ctx.noise_scale    = self.noise_scale


# ---------------------------------------------------------------------------
# 3.  Compile context  (mutable, per-sentence)
# ---------------------------------------------------------------------------

@dataclass
class CompileContext:
    csg_mode:       CSGMode        = CSGMode.UNION
    current_stage:  VerbStage      = VerbStage.POST_GEOM
    scope_stack:    List[str]      = field(default_factory=list)
    retrieval_flag: RetrievalFlag  = RetrievalFlag.SYNTHESIZE
    noise_scale:    float          = 0.0

    def is_negated(self) -> bool:
        return "negation" in self.scope_stack

    def pop_scope(self, scope: str) -> None:
        if scope in self.scope_stack:
            self.scope_stack.remove(scope)


# ---------------------------------------------------------------------------
# 4.  Word classifier  (rule-based stub; replace head with learned classifier)
# ---------------------------------------------------------------------------

# Minimal built-in lexicon — extend or replace with embedding-based classifier
_FUNCTION_WORDS = {
    "the": ControlToken("the",  retrieval_flag=RetrievalFlag.RETRIEVE),
    "a":   ControlToken("a",    retrieval_flag=RetrievalFlag.SYNTHESIZE, noise_scale=0.1),
    "an":  ControlToken("an",   retrieval_flag=RetrievalFlag.SYNTHESIZE, noise_scale=0.1),
    "not": ControlToken("not",  scope="negation"),
    "no":  ControlToken("no",   scope="negation"),
}

_RELATIONAL_WORDS = {
    "above", "below", "inside", "outside", "beside",
    "between", "near", "on", "under", "through",
}

_ABSTRACT_NOUNS = {
    "justice", "freedom", "entropy", "chaos", "order",
    "love", "time", "space", "truth", "beauty", "death",
}

def classify_word(word: str) -> WordClass:
    w = word.lower().strip()
    if w in _FUNCTION_WORDS:          return WordClass.FUNCTION_WORD
    if w in _RELATIONAL_WORDS:        return WordClass.RELATIONAL
    if w in _ABSTRACT_NOUNS:          return WordClass.ABSTRACT_NOUN
    # Default: treat as physical (atlas lookup will fail gracefully)
    return WordClass.PHYSICAL


# ---------------------------------------------------------------------------
# 5.  Word compiler  (dispatch to correct compilation target)
# ---------------------------------------------------------------------------

class WordClassifier:
    """
    Dispatches a word to its compiled token type.
    The `classify_fn` slot can be replaced with a learned classifier head.
    """

    def __init__(
        self,
        classify_fn:  Callable[[str], WordClass]       = classify_word,
        atlas:        Optional["SDFAtlas"]             = None,
        gcode_db:     Optional[Dict[str, str]]         = None,  # word → .gcode path
        rbf_params:   Optional[Dict[str, Dict]]        = None,  # word → RBF kwargs
        predicate_db: Optional[Dict[str, Predicate]]  = None,
    ):
        self.classify_fn  = classify_fn
        self.atlas        = atlas
        self.gcode_db     = gcode_db     or {}
        self.rbf_params   = rbf_params   or {}
        self.predicate_db = predicate_db or _default_predicate_db()

    def compile(self, word: str, ctx: CompileContext) -> Any:
        wclass = self.classify_fn(word)

        if wclass == WordClass.FUNCTION_WORD:
            tok = _FUNCTION_WORDS.get(word.lower(), ControlToken(word))
            tok.apply_to_context(ctx)
            return tok

        if wclass == WordClass.RELATIONAL:
            pred = self.predicate_db.get(word.lower())
            if pred is None:
                raise ValueError(f"No predicate registered for relational word '{word}'")
            return pred   # partially applied; args bound later by scene graph

        if wclass == WordClass.ABSTRACT_NOUN:
            kwargs = self.rbf_params.get(word.lower(), {})
            return RBFField(word=word, **kwargs)

        # Physical
        return PhysicalToken(
            word=word,
            gcode_path=self.gcode_db.get(word.lower()),
            sdf_key=word.lower() if (self.atlas and word.lower() in self.atlas) else None,
        )


# ---------------------------------------------------------------------------
# 6.  Scene graph
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    node_id:  str
    token:    Any                        # compiled token
    sdf:      Optional[np.ndarray] = None  # resolved SDF volume
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox:     Optional[Tuple[np.ndarray, np.ndarray]] = None  # (min, max)

@dataclass
class SceneEdge:
    predicate: Predicate
    node_a_id: str
    node_b_id: str

class SceneGraph:
    def __init__(self):
        self.nodes:    Dict[str, SceneNode] = {}
        self.edges:    List[SceneEdge]      = []
        self._counter: int                  = 0

    def add_node(self, token: Any) -> str:
        nid = f"n{self._counter}"
        self._counter += 1
        self.nodes[nid] = SceneNode(node_id=nid, token=token)
        return nid

    def add_edge(self, predicate: Predicate, node_a_id: str, node_b_id: str) -> None:
        bound = predicate.bind(node_a_id, node_b_id)
        self.edges.append(SceneEdge(predicate=bound, node_a_id=node_a_id, node_b_id=node_b_id))

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        return self.nodes.get(node_id)

    def resolve_predicates(self) -> List[str]:
        """
        Translate-mode resolver: moves/rotates objects to satisfy constraints.
        Returns list of resolved predicate names.
        """
        resolved = []
        for edge in self.edges:
            if not edge.predicate.resolved:
                if edge.predicate.resolve(self):
                    resolved.append(edge.predicate.name)
        return resolved

    def pending_predicates(self) -> List[SceneEdge]:
        return [e for e in self.edges if not e.predicate.resolved]


# ---------------------------------------------------------------------------
# 7.  Λ operator stack
# ---------------------------------------------------------------------------

class LambdaOperator:
    """
    Λ = Λ_res ∘ Λ_verb[stage] ∘ Λ_geom ∘ Λ_canon
    """

    SMOOTH_K = 4.0  # smoothness factor for smooth-min/max CSG

    def __init__(self, voxel_size: float = 0.01, residual_epsilon: float = 0.1,
                 resolution: int = 32):
        self.voxel_size       = voxel_size
        self.residual_epsilon = residual_epsilon
        self.resolution       = resolution

    # --- Λ_canon ----------------------------------------------------------

    def canon(self, token: Any, ctx: CompileContext) -> np.ndarray:
        """
        Produce a canonical SDF volume from any compiled token.
        Returns np.ndarray shape (R,R,R) — signed distances.
        """
        if isinstance(token, PhysicalToken):
            return self._canon_physical(token, ctx)
        if isinstance(token, RBFField):
            return token.to_sdf_volume()
        if isinstance(token, (Predicate, ControlToken)):
            return np.zeros((1, 1, 1))  # no geometry; handled elsewhere
        raise TypeError(f"Unknown token type: {type(token)}")

    def _canon_physical(self, token: PhysicalToken, ctx: CompileContext) -> np.ndarray:
        occ = token.resolve_occupancy(self.resolution)
        return occupancy_to_sdf(occ)

    # --- Λ_geom -----------------------------------------------------------

    def geom(self, sdf_a: np.ndarray, sdf_b: np.ndarray, mode: CSGMode) -> np.ndarray:
        """Smooth CSG operation between two SDF volumes (same resolution)."""
        if mode == CSGMode.UNION:
            return smooth_min(sdf_a, sdf_b, self.SMOOTH_K)
        if mode == CSGMode.SUBTRACT:
            return smooth_max(sdf_a, -sdf_b, self.SMOOTH_K)
        if mode == CSGMode.INTERSECT:
            return smooth_max(sdf_a, sdf_b, self.SMOOTH_K)
        if mode == CSGMode.BLEND:
            return 0.5 * sdf_a + 0.5 * sdf_b
        raise ValueError(f"Unknown CSGMode: {mode}")

    # --- Λ_verb -----------------------------------------------------------

    def verb(
        self,
        sdf:   np.ndarray,
        verb:  str,
        stage: VerbStage,
        vtype: VerbType,
        params: Dict,
    ) -> np.ndarray:
        """Apply verb transform to SDF field."""
        if stage == VerbStage.PRE_CANON:
            return self._verb_pre_canon(sdf, verb, params)
        if stage == VerbStage.POST_CANON:
            return self._verb_post_canon(sdf, verb, params)
        # POST_GEOM
        return self._verb_post_geom(sdf, verb, vtype, params)

    def _verb_pre_canon(self, sdf, verb, params):
        # Pre-canon: modify base material / density before shape stabilisation
        if verb == "soften":
            sigma = params.get("sigma", 1.0)
            return gaussian_smooth_sdf(sdf, sigma)
        return sdf

    def _verb_post_canon(self, sdf, verb, params):
        if verb in ("grow", "expand"):
            offset = params.get("offset", 0.05)
            return sdf - offset   # SDF offset = morphological dilation
        if verb in ("shrink", "erode"):
            offset = params.get("offset", 0.05)
            return sdf + offset
        if verb == "bend":
            axis   = np.array(params.get("axis",   [0, 0, 1]))
            angle  = params.get("angle",  0.3)
            radius = params.get("radius", 1.0)
            return bend_sdf(sdf, axis, angle, radius)
        return sdf

    def _verb_post_geom(self, sdf, verb, vtype, params):
        if verb == "melt":
            sigma    = params.get("sigma",    2.0)
            strength = params.get("strength", 0.5)
            smoothed = gaussian_smooth_sdf(sdf, sigma)
            return (1 - strength) * sdf + strength * smoothed
        if verb == "carve":
            tool = params.get("tool_sdf")
            if tool is not None:
                return self.geom(sdf, tool, CSGMode.SUBTRACT)
        return sdf

    # --- Λ_res ------------------------------------------------------------

    def residual(self, sdf: np.ndarray, net_output: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Hard-bounded residual correction.
        residual = ε · voxel_size · tanh(net_output)
        """
        if net_output is None:
            return sdf
        correction = self.residual_epsilon * self.voxel_size * np.tanh(net_output)
        return sdf + correction

    # --- Full Λ -----------------------------------------------------------

    def apply(
        self,
        W_prev:     np.ndarray,
        token:      Any,
        ctx:        CompileContext,
        verb_info:  Optional[Dict] = None,
        net_output: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Full operator: Λ(u_t, W_{t-1})
        Returns SDF delta to be composed with W_prev via ⊕.
        """
        if isinstance(token, (Predicate, ControlToken)):
            return W_prev   # no field contribution

        # Stage 1: canonicalize
        sdf = self.canon(token, ctx)

        # Stage 2: pre-canon verb
        if verb_info and verb_info.get("stage") == VerbStage.PRE_CANON:
            sdf = self.verb(sdf, verb_info["verb"], VerbStage.PRE_CANON,
                            verb_info.get("type", VerbType.SPATIAL), verb_info.get("params", {}))

        # Stage 3: geometry (if a second SDF is being combined)
        if verb_info and verb_info.get("secondary_sdf") is not None:
            sdf = self.geom(sdf, verb_info["secondary_sdf"], ctx.csg_mode)

        # Stage 4: post-canon verb
        if verb_info and verb_info.get("stage") == VerbStage.POST_CANON:
            sdf = self.verb(sdf, verb_info["verb"], VerbStage.POST_CANON,
                            verb_info.get("type", VerbType.SPATIAL), verb_info.get("params", {}))

        # Stage 5: post-geom verb
        if verb_info and verb_info.get("stage") == VerbStage.POST_GEOM:
            sdf = self.verb(sdf, verb_info["verb"], VerbStage.POST_GEOM,
                            verb_info.get("type", VerbType.SPATIAL), verb_info.get("params", {}))

        # Stage 6: residual correction
        sdf = self.residual(sdf, net_output)

        return sdf


# ---------------------------------------------------------------------------
# 8.  ⊕ compositor
# ---------------------------------------------------------------------------

def compose(
    W:       np.ndarray,
    delta:   np.ndarray,
    verb_type: VerbType,
    alpha:   float = 0.5,
    k:       float = 4.0,
) -> np.ndarray:
    """
    ⊕ dispatch:
        SPATIAL              → SceneAppend  (no field fusion; handled by scene graph)
        PHYSICAL_INTERACTION → CSG smooth subtract/union
        MATERIAL_STATE       → FieldBlend
    """
    if verb_type == VerbType.SPATIAL:
        return W   # composition handled by scene graph edges, not field fusion

    if verb_type == VerbType.PHYSICAL_INTERACTION:
        return smooth_min(W, delta, k)

    if verb_type == VerbType.MATERIAL_STATE:
        return (1 - alpha) * W + alpha * delta

    return W


# ---------------------------------------------------------------------------
# 9.  Full World Compiler  (entry point)
# ---------------------------------------------------------------------------

class WorldCompiler:
    """
    W_t = W_{t-1} ⊕ Λ(u_t, W_{t-1})

    Usage:
        wc = WorldCompiler()
        wc.compile_sentence("The cat sits on the chair")
        sdf = wc.render()
    """

    def __init__(
        self,
        resolution:  int   = 32,
        voxel_size:  float = 0.01,
        gcode_db:    Optional[Dict[str, str]] = None,
    ):
        self.resolution  = resolution
        self.voxel_size  = voxel_size
        self.scene_graph = SceneGraph()
        self.classifier  = WordClassifier(gcode_db=gcode_db)
        self.lambda_op   = LambdaOperator(voxel_size=voxel_size, resolution=resolution)
        self.W           = np.full((resolution,)*3, 1e9)   # empty world (all outside)
        self.ctx         = CompileContext()
        self._node_stack: List[str] = []   # MRU node ids for predicate binding

    def compile_sentence(self, sentence: str) -> None:
        tokens = sentence.strip().split()
        pending_predicate: Optional[Predicate] = None

        for word in tokens:
            token = self.classifier.compile(word, self.ctx)

            if isinstance(token, ControlToken):
                continue   # already applied context mutations

            if isinstance(token, Predicate):
                pending_predicate = token
                continue

            if isinstance(token, (PhysicalToken, RBFField)):
                # Canonicalize and add to scene graph
                sdf   = self.lambda_op.canon(token, self.ctx)
                nid   = self.scene_graph.add_node(token)
                node  = self.scene_graph.get_node(nid)
                node.sdf = sdf
                self._node_stack.append(nid)

                # Bind pending predicate if we now have two nodes
                if pending_predicate and len(self._node_stack) >= 2:
                    a, b = self._node_stack[-2], self._node_stack[-1]
                    self.scene_graph.add_edge(pending_predicate, a, b)
                    pending_predicate = None

    def resolve(self) -> List[str]:
        """Translate objects to satisfy all pending spatial predicates."""
        return self.scene_graph.resolve_predicates()

    def render(self) -> np.ndarray:
        """
        Fuse all scene nodes into a single SDF volume.
        For spatial verbs the scene graph holds layout; we just union all SDFs here.

        Implementation note: np.minimum (hard min) is used here, NOT smooth_min.
        smooth_min only blends correctly when values differ by < ~2k; for global
        volume fusion where objects are separated by large SDF distances the
        Quilez formula clips to the wrong operand.  Hard minimum preserves exact
        interior/exterior information everywhere.  Smooth CSG blending is
        reserved for explicit physical-interaction verbs in LambdaOperator.geom().
        """
        target_shape = (self.resolution,) * 3
        result: Optional[np.ndarray] = None

        for node in self.scene_graph.nodes.values():
            if node.sdf is None:
                continue
            if node.sdf.shape != target_shape:
                continue
            if result is None:
                result = node.sdf.copy()
            else:
                result = np.minimum(result, node.sdf)   # hard union of SDF fields

        if result is None:
            result = np.full(target_shape, 1e9)

        self.W = result
        return result


# ---------------------------------------------------------------------------
# 10.  SDF math utilities
# ---------------------------------------------------------------------------

def smooth_min(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    """Inigo Quilez smooth minimum (C² continuous)."""
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return a * (1 - h) + b * h - k * h * (1 - h)

def smooth_max(a: np.ndarray, b: np.ndarray, k: float) -> np.ndarray:
    return -smooth_min(-a, -b, k)

def occupancy_to_sdf(occupancy: np.ndarray) -> np.ndarray:
    """
    Convert binary occupancy grid to SDF via fast exact Euclidean distance transform.
    Requires scipy.
    """
    from scipy.ndimage import distance_transform_edt
    inside  = distance_transform_edt( occupancy)
    outside = distance_transform_edt(~occupancy)
    return outside - inside   # negative inside, positive outside

def gaussian_smooth_sdf(sdf: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(sdf, sigma=sigma)

def bend_sdf(sdf: np.ndarray, axis: np.ndarray, angle: float, radius: float) -> np.ndarray:
    """
    Coordinate warp: radial bend around `axis`.
    Approximate — remaps voxel coordinates before sampling SDF.
    """
    R = sdf.shape[0]
    grid   = np.linspace(-1, 1, R)
    xs, ys, zs = np.meshgrid(grid, grid, grid, indexing='ij')

    # Simple bend along Z axis: rotate (x,y) plane by angle·z
    bend_angle = angle * zs / radius
    cos_a = np.cos(bend_angle)
    sin_a = np.sin(bend_angle)

    xs_w = cos_a * xs - sin_a * ys
    ys_w = sin_a * xs + cos_a * ys
    zs_w = zs

    # Sample sdf at warped coordinates (trilinear approximation via rounding)
    i = np.clip(((xs_w + 1) / 2 * R).astype(int), 0, R - 1)
    j = np.clip(((ys_w + 1) / 2 * R).astype(int), 0, R - 1)
    k = np.clip(((zs_w + 1) / 2 * R).astype(int), 0, R - 1)
    return sdf[i, j, k]

def _unit_sphere_sdf(resolution: int = 32) -> np.ndarray:
    """Fallback canonical form: unit sphere SDF."""
    grid  = np.linspace(-1.5, 1.5, resolution)
    x, y, z = np.meshgrid(grid, grid, grid, indexing='ij')
    return np.sqrt(x**2 + y**2 + z**2) - 1.0

class SDFAtlas(dict):
    """Thin wrapper around dict for SDF key lookup. Extend with mmapped files."""
    pass


# ---------------------------------------------------------------------------
# 11.  Built-in predicate database
# ---------------------------------------------------------------------------

def _resolve_above(node_a: SceneNode, node_b: SceneNode,
                   scene: SceneGraph, margin: float = 0.05) -> None:
    """Translate node_a so its bottom rests on top of node_b."""
    if node_b.bbox is None or node_a.sdf is None:
        return
    b_top_z = float(node_b.bbox[1][2])   # max z of node_b
    a_min_z = float(node_a.bbox[0][2]) if node_a.bbox else 0.0
    offset_z = b_top_z + margin - a_min_z
    node_a.centroid[2] += offset_z

def _resolve_inside(node_a: SceneNode, node_b: SceneNode,
                    scene: SceneGraph, margin: float = 0.0) -> None:
    """Centre node_a at node_b's centroid."""
    node_a.centroid = node_b.centroid.copy()

def _resolve_beside(node_a: SceneNode, node_b: SceneNode,
                    scene: SceneGraph, margin: float = 0.1) -> None:
    """Place node_a to the right of node_b."""
    if node_b.bbox is None:
        return
    node_a.centroid[0] = node_b.bbox[1][0] + margin

def _default_predicate_db() -> Dict[str, Predicate]:
    return {
        "above":  Predicate("above",  _resolve_above,  arity=2, margin=0.05),
        "on":     Predicate("on",     _resolve_above,  arity=2, margin=0.02),
        "inside": Predicate("inside", _resolve_inside, arity=2),
        "beside": Predicate("beside", _resolve_beside, arity=2, margin=0.1),
        "near":   Predicate("near",   _resolve_beside, arity=2, margin=0.2),
    }


# ---------------------------------------------------------------------------
# 13.  G-code parser + rasterizer
# ---------------------------------------------------------------------------

@dataclass
class GCodeMove:
    cmd: str
    x:   float
    y:   float
    z:   float
    e:   float   # extruder position (positive = material deposited)


def parse_gcode_moves(gcode_path: str) -> List[GCodeMove]:
    """
    Parse G0/G1 motion commands from a .gcode file.
    Returns a list of GCodeMove with absolute XYZ+E coordinates.
    Handles incremental position tracking (last known value per axis).
    """
    moves: List[GCodeMove] = []
    cx = cy = cz = ce = 0.0   # current position

    _G01 = re.compile(
        r'^G[01]\b'            # G0 or G1
        r'(?:\s+X(-?\d*\.?\d+))?'
        r'(?:\s+Y(-?\d*\.?\d+))?'
        r'(?:\s+Z(-?\d*\.?\d+))?'
        r'(?:\s+E(-?\d*\.?\d+))?',
        re.IGNORECASE
    )

    with open(gcode_path, 'r', errors='replace') as fh:
        for raw in fh:
            line = raw.split(';')[0].strip()   # strip inline comments
            if not line:
                continue
            m = _G01.match(line)
            if not m:
                continue
            x_s, y_s, z_s, e_s = m.groups()
            cx = float(x_s) if x_s is not None else cx
            cy = float(y_s) if y_s is not None else cy
            cz = float(z_s) if z_s is not None else cz
            ce = float(e_s) if e_s is not None else ce
            moves.append(GCodeMove(cmd=line.split()[0].upper(),
                                   x=cx, y=cy, z=cz, e=ce))

    return moves


def rasterize_toolpath(
    moves:        List[GCodeMove],
    resolution:   int   = 64,
    tool_radius:  float = 1.5,   # in normalized voxel units
    extrude_only: bool  = True,  # only rasterize segments where E advanced
) -> np.ndarray:
    """
    Swept-segment rasterization: each consecutive extruding move pair (p_i, p_{i+1})
    is treated as a capsule (cylinder + hemisphere caps).  This fills inter-move gaps
    that the point-sphere approach misses, producing solid closed geometry.

    Capsule signed distance from point q to segment (p0, p1):
        t     = clamp(dot(q-p0, d) / dot(d,d), 0, 1)    d = p1-p0
        close = p0 + t*d
        dist² = ||q - close||²

    Capsule interior: dist² ≤ r²

    Shapes: GI/GJ/GK are (R,R,R); p0/p1/d are (3,); broadcast over voxels via
    an (R,R,R,3) grid.  No Python inner loops.
    """
    if not moves:
        return np.zeros((resolution,) * 3, dtype=bool)

    # ---- bounding-box normalisation ----------------------------------------
    xs = np.array([m.x for m in moves], dtype=np.float64)
    ys = np.array([m.y for m in moves], dtype=np.float64)
    zs = np.array([m.z for m in moves], dtype=np.float64)

    def _norm(v: np.ndarray) -> np.ndarray:
        lo, hi = v.min(), v.max()
        span = hi - lo if hi - lo > 1e-9 else 1.0
        pad  = tool_radius
        return pad + (v - lo) / span * (resolution - 2.0 * pad)

    xs_n = _norm(xs)
    ys_n = _norm(ys)
    zs_n = _norm(zs)

    # ---- voxel coordinate grid  (R,R,R,3)  ---------------------------------
    gi = np.arange(resolution, dtype=np.float64)
    GI, GJ, GK = np.meshgrid(gi, gi, gi, indexing='ij')
    GRID = np.stack([GI, GJ, GK], axis=-1)          # (R,R,R,3)

    occ  = np.zeros((resolution,) * 3, dtype=bool)
    r2   = float(tool_radius) ** 2
    prev_e = moves[0].e

    for i in range(len(moves) - 1):
        nxt = moves[i + 1]

        if extrude_only and nxt.e <= prev_e:
            prev_e = nxt.e
            continue
        prev_e = nxt.e

        p0 = np.array([xs_n[i],     ys_n[i],     zs_n[i]    ], dtype=np.float64)
        p1 = np.array([xs_n[i + 1], ys_n[i + 1], zs_n[i + 1]], dtype=np.float64)

        d     = p1 - p0                        # segment direction (3,)
        d2    = float(np.dot(d, d))            # ||d||²
        if d2 < 1e-12:                         # degenerate (coincident points)
            dist2 = np.sum((GRID - p0) ** 2, axis=-1)
            occ  |= dist2 <= r2
            continue

        # t = clamp(dot(q-p0, d)/d2, 0, 1)  —  closest point param on segment
        q_p0  = GRID - p0                      # (R,R,R,3)
        t     = np.einsum('...i,i->...', q_p0, d) / d2   # (R,R,R)
        t     = np.clip(t, 0.0, 1.0)

        # closest point on segment for each voxel
        # shape: (R,R,R,3) = (R,R,R,1) * (3,)  → broadcast OK
        closest = p0 + t[..., np.newaxis] * d  # (R,R,R,3)

        dist2 = np.sum((GRID - closest) ** 2, axis=-1)   # (R,R,R)
        occ  |= dist2 <= r2

    return occ


def load_gcode_atlas(gcode_dir: str) -> Dict[str, str]:
    """
    Scan a directory and return a word → .gcode path mapping.
    Filename convention: <word>.gcode  (e.g. cat.gcode, chair.gcode)
    Subdirectories are also scanned one level deep.
    """
    mapping: Dict[str, str] = {}
    if not os.path.isdir(gcode_dir):
        return mapping

    for entry in os.scandir(gcode_dir):
        if entry.is_file() and entry.name.lower().endswith('.gcode'):
            word = entry.name[:-6].lower()
            mapping[word] = entry.path
        elif entry.is_dir():
            for sub in os.scandir(entry.path):
                if sub.is_file() and sub.name.lower().endswith('.gcode'):
                    word = sub.name[:-6].lower()
                    mapping[word] = sub.path

    return mapping


def write_synthetic_gcode(path: str, shape: str = "helix") -> None:
    """
    Write a small synthetic G-code file useful for testing.
    shape: 'helix' | 'box'
    """
    lines = ["; synthetic test gcode", "G28 ; home"]
    if shape == "helix":
        steps = 60
        for i in range(steps):
            t  = i / steps * 4 * math.pi
            x  = 10 + 8 * math.cos(t)
            y  = 10 + 8 * math.sin(t)
            z  = i * 0.1
            e  = i * 0.05
            lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} E{e:.3f}")
    elif shape == "box":
        corners = [(5,5), (15,5), (15,15), (5,15), (5,5)]
        for layer in range(5):
            z = layer * 0.2
            e = layer * 2.0
            for (x, y) in corners:
                e += 0.4
                lines.append(f"G1 X{x} Y{y} Z{z:.1f} E{e:.2f}")
    lines.append("; end")
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


# ---------------------------------------------------------------------------
# 12.  Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile, pathlib
    print("=== World Compiler smoke test ===\n")

    wc = WorldCompiler(resolution=32)

    # Test 1: word classification
    for word in ["the", "cat", "justice", "above", "not", "a"]:
        wclass = classify_word(word)
        print(f"  classify('{word}') → {wclass.name}")

    print()

    # Test 2: RBF field for abstract noun
    rbf = RBFField("entropy", amplitude=1.0, sigma=0.8)
    vol = rbf.to_sdf_volume(resolution=16)
    print(f"  RBFField('entropy') sdf volume shape: {vol.shape}, "
          f"min={vol.min():.3f}, max={vol.max():.3f}")

    # Test 3: occupancy → SDF
    occ = np.zeros((16, 16, 16), dtype=bool)
    occ[6:10, 6:10, 6:10] = True
    sdf = occupancy_to_sdf(occ)
    print(f"  occupancy_to_sdf: shape={sdf.shape}, "
          f"min={sdf.min():.3f}, max={sdf.max():.3f}")

    # Test 4: smooth CSG
    sphere = _unit_sphere_sdf(32)
    box    = _unit_sphere_sdf(32) - 0.3
    union  = smooth_min(sphere, box, k=4.0)
    print(f"  smooth_union: shape={union.shape}")

    # Test 5: G-code parser + swept-segment rasterizer
    with tempfile.TemporaryDirectory() as tmpdir:
        # --- helix ---
        gcode_path = os.path.join(tmpdir, "helix.gcode")
        write_synthetic_gcode(gcode_path, shape="helix")
        moves_h = parse_gcode_moves(gcode_path)
        occ_h   = rasterize_toolpath(moves_h, resolution=32, tool_radius=2.0)
        sdf_h   = occupancy_to_sdf(occ_h)
        print(f"\n  helix: {len(moves_h)} moves, "
              f"{occ_h.sum()} voxels ({100*occ_h.mean():.1f}%), "
              f"sdf min={sdf_h.min():.2f}")
        assert occ_h.sum() > 0,       "helix: expected occupied voxels"
        assert sdf_h.min() < 0,       "helix: expected negative SDF interior"

        # --- box ---
        gcode_path_b = os.path.join(tmpdir, "box.gcode")
        write_synthetic_gcode(gcode_path_b, shape="box")
        moves_b = parse_gcode_moves(gcode_path_b)
        occ_b   = rasterize_toolpath(moves_b, resolution=32, tool_radius=2.0)
        sdf_b   = occupancy_to_sdf(occ_b)
        print(f"  box:   {len(moves_b)} moves, "
              f"{occ_b.sum()} voxels ({100*occ_b.mean():.1f}%), "
              f"sdf min={sdf_b.min():.2f}")
        assert occ_b.sum() > 0,       "box: expected occupied voxels"
        assert sdf_b.min() < 0,       "box: expected negative SDF interior"

        # --- atlas + full sentence compile ---
        atlas_dir = os.path.join(tmpdir, "atlas")
        os.makedirs(atlas_dir)
        write_synthetic_gcode(os.path.join(atlas_dir, "cat.gcode"),   shape="helix")
        write_synthetic_gcode(os.path.join(atlas_dir, "chair.gcode"), shape="box")
        db = load_gcode_atlas(atlas_dir)
        print(f"\n  load_gcode_atlas: {sorted(db.keys())}")

        wc2 = WorldCompiler(resolution=32, gcode_db=db)
        wc2.lambda_op.resolution = 32   # ensure consistent
        # Override tool_radius for test by monkey-patching resolve_occupancy
        # (production code would pass tool_radius through the token)
        _orig = wc2.lambda_op._canon_physical
        def _patched(token, ctx):
            if token.gcode_path and os.path.exists(token.gcode_path):
                moves = parse_gcode_moves(token.gcode_path)
                token.occupancy = rasterize_toolpath(moves, 32, tool_radius=2.0)
            return occupancy_to_sdf(token.resolve_occupancy(32))
        wc2.lambda_op._canon_physical = _patched

        wc2.compile_sentence("the cat sits on the chair")
        wc2.resolve()
        sdf_out = wc2.render()

        print(f"\n  compile_sentence (swept G-code):")
        print(f"  scene nodes: {list(wc2.scene_graph.nodes.keys())}")
        print(f"  scene edges: "
              f"{[(e.node_a_id, e.predicate.name, e.node_b_id) for e in wc2.scene_graph.edges]}")
        for nid, node in wc2.scene_graph.nodes.items():
            if node.sdf is not None:
                print(f"    node {nid} ({node.token.word}): "
                      f"sdf min={node.sdf.min():.3f}  occ={int((node.sdf<0).sum())} voxels inside")
        print(f"  render: shape={sdf_out.shape}, min={sdf_out.min():.3f}, max={sdf_out.max():.3f}")

        assert sdf_out.min() < 1e8, "Expected real geometry"
        assert (sdf_out < 0).any(), "Expected negative SDF in final render"

    print("\n=== ALL TESTS PASS ===")
