"""world_compiler.py — text -> world. The product layer.

    "Turn words into worlds. World Compiler takes text from an AI chat and
    creates 3D renderings from the words using a unique approach to
    semantics." (README)

This module is the compiler. It does not invent a new semantic engine — it
composes the five verified, gated modules already in this repo into one
pipeline, and adds exactly one new thing: a deterministic, meaning-preserving
map from the engine's output (latent DNA + kaleidoscope symmetry structure)
onto a 3D scene.

THE PIPELINE (each stage is a real, previously-verified module — nothing here
is decorative):

  1. INGEST      organic_ai_core.load_text
                 Text -> overlapping windows -> hashed-trigram embeddings.
                 No vocabulary, no training, no network: the same text always
                 embeds to the same vectors, on any machine.

  2. LIVE        genetic_manifold.GeneticManifold
                 Each text window becomes a node whose DNA *is* its embedding
                 (kaleidoscope_core.CompressionOrganism fits the manifold this
                 lives in). The population evolves under a closed energy
                 economy: nodes that hold a rare, uncovered view of their
                 symmetry orbit thrive; redundant duplicates starve. This is
                 natural selection over the text's own motifs, not a training
                 loop — nothing here is fit to a target.

  3. PURIFY      refinement_loop.RefinementEngine
                 The evolved population's DNA is repeatedly pulled toward the
                 canonical (symmetry-invariant) view of its kaleidoscope orbit
                 until it stops moving — raw ingested signal becomes pure,
                 inherited DNA. Only THEN do supernodes crystallize: one
                 supernode per settled semantic motif. This is a proven
                 contraction (gate_contraction), so the process provably
                 terminates at a fixed point rather than drifting forever.

  4. PROJECT     this module (WorldCompiler.compile)
                 Each crystallized supernode becomes one object in a 3D scene:
                 position from its dominant latent axes, geometric complexity
                 from how many latent axes its kaleidoscope invariant actually
                 uses, color from a stable hash of its DNA, size from how much
                 of the text it represents. Motifs the text repeats become
                 large, simple, central shapes; motifs it mentions once become
                 small, jagged, peripheral ones. Nothing here is arbitrary:
                 every visual property is a deterministic function of a
                 quantity the engine already computed and gated.

Same text + same seed -> byte-identical scene, anywhere (gate_determinism).
Different text -> a different world (gate_diverges) — the whole product claim
is that (3D world) is a faithful, falsifiable rendering of (evolved textual
structure), and both gates exist to keep that claim honest rather than
asserted.

Runtime: numpy + stdlib only. No network, no training, no pickle/exec.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import organic_ai_core as core
import kaleidoscope_group as kgroup
import genetic_manifold as gmod
import refinement_loop as refine
import semantic_embedding as sem

# Ordered by facet count. A supernode's shape family is a deterministic
# function of how many latent axes its kaleidoscope invariant actually uses —
# not an arbitrary skin, a readout of real structure (see _shape_kind).
_SHAPE_FAMILIES: Tuple[str, ...] = ("tetra", "cube", "octa", "icosa")


class WorldCompilerError(core.OrganicError):
    """Raised when text cannot be compiled into a world (e.g. too short)."""


# ---------------------------------------------------------------------------
# Scene data model
# ---------------------------------------------------------------------------


@dataclass
class WorldObject:
    id: str
    position: Tuple[float, float, float]
    scale: float
    color: Tuple[float, float, float]      # RGB in [0,1]
    shape: str                             # one of _SHAPE_FAMILIES
    mass: float                            # fraction of population, sums to 1
    members: int                           # surviving nodes crystallized here
    label: str                             # a real snippet of the source text
    spin_seed: float                       # deterministic per-object rotation phase


@dataclass
class WorldEdge:
    a: int
    b: int
    strength: float


@dataclass
class WorldScene:
    objects: List[WorldObject]
    edges: List[WorldEdge]
    background: Tuple[float, float, float]
    fingerprint: str
    title: str
    stats: Dict[str, object]

    def to_json_dict(self) -> dict:
        return {
            "title": self.title,
            "fingerprint": self.fingerprint,
            "background": list(self.background),
            "stats": self.stats,
            "objects": [
                {
                    "id": o.id, "pos": list(o.position), "scale": o.scale,
                    "color": list(o.color), "shape": o.shape, "mass": o.mass,
                    "members": o.members, "label": o.label,
                    "spin": o.spin_seed,
                }
                for o in self.objects
            ],
            "edges": [{"a": e.a, "b": e.b, "s": e.strength} for e in self.edges],
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CompilerConfig:
    seed: int = 0
    embed_dim: int = 16           # orthographic (hashed-trigram) channel width
    target_chunks: int = 110      # windowing is auto-tuned toward this count
    min_window: int = 24
    max_window: int = 96
    generations: int = 12
    mutation_scale: float = 0.06
    max_strains: int = 10
    group_max_elements: int = 64
    strain_resolution: float = 0.35
    refine_max_passes: int = 40
    refine_purify_rate: float = 0.5
    # semantic (word co-occurrence) channel — see semantic_embedding.py.
    # Set semantic_dim=0 to fall back to pure orthographic embedding.
    semantic_dim: int = 12
    semantic_weight: float = 1.6  # relative influence vs the trigram channel
    semantic_min_count: int = 2
    semantic_max_vocab: int = 600


# ---------------------------------------------------------------------------
# Deterministic helpers (same hashing discipline as organic_ai_core)
# ---------------------------------------------------------------------------


def _stable_unit(payload: bytes) -> float:
    """blake2b(payload) -> deterministic value in [0, 1)."""
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64 - 1)


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[float, float, float]:
    h = (h % 360.0) / 360.0
    if s <= 0:
        return (l, l, l)

    def hue(p: float, q: float, t: float) -> float:
        t %= 1.0
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < 1 / 2:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q
    return (hue(p, q, h + 1 / 3), hue(p, q, h), hue(p, q, h - 1 / 3))


def _auto_window(text_len: int, cfg: CompilerConfig) -> Tuple[int, int]:
    """Pick (window, stride) so the chunk count lands near target_chunks.

    Stride is solved directly from n_chunks = (text_len-window)//stride + 1,
    and is deliberately allowed to exceed window: for very long text this
    makes chunks a sparse, evenly-spaced sample across the whole document
    (a summary of themes spread throughout it) rather than capping out at a
    dense scan of only its first ~9,000 characters."""
    window = max(cfg.min_window, min(cfg.max_window, text_len // 8 or cfg.min_window))
    if text_len <= window:
        return window, max(1, window // 2)
    target = max(cfg.target_chunks, 2)
    stride = max(1, (text_len - window) // (target - 1))
    return window, stride


def _ingest(text: str, cfg: CompilerConfig,
           semantic: Optional[sem.SemanticSpace] = None
           ) -> Tuple[int, int, List[str], np.ndarray, sem.SemanticSpace]:
    """Chunk text and embed each chunk as [orthographic trigram features |
    semantic co-occurrence features] (see semantic_embedding.py for why both:
    trigrams see spelling and survive OOV/typos with zero vocabulary;
    co-occurrence sees topic and requires enough text to learn one).

    If `semantic` is passed in already fit, chunks are embedded through it
    unchanged rather than fitting a new one — this is how LiveWorld.feed()
    keeps a running world in the same semantic coordinate system its founding
    text established, exactly as it reuses the founding geometric manifold.
    Returns (window, stride, snippets, data, semantic_space_used).
    """
    window, stride = _auto_window(len(text), cfg)
    if len(text) < window + stride:
        raise WorldCompilerError(
            f"text too short ({len(text)} chars); need at least "
            f"{window + stride} for a stable world")
    starts = list(range(0, len(text) - window + 1, stride))
    snippets = [text[s:s + window] for s in starts]
    src = core.load_text(text, dim=cfg.embed_dim, window=window, stride=stride)
    n = len(starts)
    trigram = np.stack([src.observe(t) for t in range(n)])

    if cfg.semantic_dim <= 0:
        return window, stride, snippets, trigram, sem.SemanticSpace.empty()

    space = semantic if semantic is not None else sem.SemanticSpace.fit(
        text, dim=cfg.semantic_dim, min_count=cfg.semantic_min_count,
        max_vocab=cfg.semantic_max_vocab)
    if space.dim > 0:
        sem_frames = np.stack([space.vector_for(s) for s in snippets]) * cfg.semantic_weight
        data = np.concatenate([trigram, sem_frames], axis=1)
    else:
        data = trigram  # too little text to learn a semantic space; degrade gracefully
    return window, stride, snippets, data, space


def _shape_kind(rep: np.ndarray, group: kgroup.KaleidoscopeGroup) -> str:
    """Facet family from the PARTICIPATION RATIO of the DNA's kaleidoscope
    invariant spectrum: (Σx²)² / Σx⁴, the standard measure of how many axes
    a vector effectively spreads across (1 for a single dominant axis, up to
    the manifold rank for a uniform spread). A motif concentrated on one or
    two latent axes renders as a simple tetrahedron; a motif whose identity is
    smeared across most of the manifold renders as a many-faceted icosahedron.
    Real structure, not a random skin."""
    sig = group.invariant(rep)
    spectrum = sig[1:]
    sq = spectrum ** 2
    denom = float(np.sum(sq ** 2))
    eff_dim = float(np.sum(sq) ** 2 / denom) if denom > 1e-15 else 1.0
    breakpoints = (2.5, 4.0, 6.5)
    idx = sum(eff_dim > b for b in breakpoints)
    return _SHAPE_FAMILIES[min(idx, len(_SHAPE_FAMILIES) - 1)]


def _origin_hash(row: np.ndarray) -> str:
    """Must match GeneticManifold.seed_from_data's founder-provenance hash
    exactly, so surviving nodes N generations later can be traced back to the
    text chunk that founded their lineage (CodeGenome.origin_hash is inherited
    unchanged through every mutation)."""
    return hashlib.blake2b(row.tobytes(), digest_size=8).hexdigest()


def project_strains(strains: List["kgroup.Strain"], group: kgroup.KaleidoscopeGroup,
                    label_for_member: Callable[[int], str]) -> List[WorldObject]:
    """Strains (crystallized supernodes) -> 3D objects. Shared by the batch
    compiler and world_live.LiveWorld so both project identically. Every
    visual property is a deterministic function of something the engine
    already computed (see module docstring, stage 4) — nothing here is
    arbitrary, and nothing here mutates strains/group.

    ``label_for_member(local_index)`` resolves a strain's first member
    (an index into whatever code array the strains were formed from) to a
    human-readable provenance string.
    """
    if not strains:
        return []
    reps = np.stack([s.representative for s in strains])
    top = min(3, reps.shape[1])
    axes = reps[:, :top]
    mean = axes.mean(axis=0)
    spread = axes.std(axis=0)
    spread = np.where(spread < 1e-9, 1.0, spread)
    normed = (axes - mean) / spread
    if top < 3:
        normed = np.pad(normed, ((0, 0), (0, 3 - top)))

    objects: List[WorldObject] = []
    for k, strain in enumerate(strains):
        rep = strain.representative
        pos = tuple(float(v) * 2.4 for v in normed[k])
        hue = _stable_unit(np.round(rep, 6).tobytes()) * 360.0
        sat = 0.45 + 0.35 * min(strain.mass * len(strains), 1.0)
        lum = 0.42 + 0.16 * _stable_unit(strain.signature.tobytes() + b"l")
        color = _hsl_to_rgb(hue, sat, lum)
        shape = _shape_kind(rep, group)
        scale = 0.35 + 1.65 * (strain.mass ** 0.5)
        label = " ".join(label_for_member(strain.members[0]).split())
        spin = _stable_unit(rep.tobytes() + b"spin") * 6.28318

        objects.append(WorldObject(
            id=f"motif-{k}", position=pos, scale=scale, color=color,
            shape=shape, mass=float(strain.mass), members=len(strain.members),
            label=label, spin_seed=spin))
    return objects


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------


class WorldCompiler:
    """Text -> WorldScene. Stateless across calls; every call is a pure
    function of (text, cfg)."""

    def __init__(self, cfg: Optional[CompilerConfig] = None) -> None:
        self.cfg = cfg or CompilerConfig()

    def compile(self, text: str) -> WorldScene:
        cfg = self.cfg
        text = text.strip()
        window, stride, snippets, data, space = _ingest(text, cfg)
        n = len(snippets)
        origin_to_chunk = {_origin_hash(data[i]): i for i in range(n)}

        # --- LIVE: evolve the text's own DNA under the closed economy -----
        gcfg = gmod.GeneticConfig(
            seed=cfg.seed, generations=cfg.generations,
            mutation_scale=cfg.mutation_scale, max_strains=cfg.max_strains,
            group_max_elements=cfg.group_max_elements,
            strain_resolution=cfg.strain_resolution,
            total_energy=max(600.0, n * 10.0),
            max_population=min(300, max(60, n * 3)))
        gm = gmod.GeneticManifold(gcfg)
        gm.seed_from_data(data)
        gm.evolve()
        if not gm.nodes:
            raise WorldCompilerError(
                "the evolved population went extinct on this text; try "
                "longer or more varied input")

        ordered = sorted(gm.nodes)
        evolved_codes = np.stack([gm.nodes[nid].genome.code for nid in ordered])
        evolved_origins = [gm.nodes[nid].genome.origin_hash for nid in ordered]

        # --- PURIFY: kaleidoscope refinement to crystallized supernodes ---
        rcfg = refine.RefinementConfig(
            seed=cfg.seed, max_passes=cfg.refine_max_passes,
            purify_rate=cfg.refine_purify_rate,
            strain_resolution=cfg.strain_resolution,
            group_max_elements=cfg.group_max_elements)
        eng = refine.RefinementEngine(rcfg)
        eng.manifold = gm.manifold
        eng.group = gm.group
        eng.codes = evolved_codes
        report = eng.run()
        refine.gate_contraction(report)   # re-assert the fixed-point guarantee

        strains, labels = kgroup.form_strains(
            eng.codes, eng.group, resolution=cfg.strain_resolution)

        # --- PROJECT: strains -> 3D objects --------------------------------
        def label_for(member_idx: int) -> str:
            chunk_i = origin_to_chunk.get(evolved_origins[member_idx])
            return snippets[chunk_i] if chunk_i is not None else "(inherited motif)"

        objects = project_strains(strains, eng.group, label_for)
        edges = self._build_edges(objects)

        bg_hue = _stable_unit(np.round(report.one_entity, 6).tobytes()) * 360.0
        background = _hsl_to_rgb(bg_hue, 0.35, 0.10)

        stats = {
            "source_chars": len(text), "chunks": n, "window": window,
            "stride": stride, "manifold_rank": int(gm.manifold.rank),
            "group_order": int(gm.group.elements.shape[0]),
            "generations": cfg.generations,
            "survivors": len(gm.nodes),
            "refine_passes": report.passes_run,
            "refine_converged": report.converged,
            "motifs": len(objects),
            "seed": cfg.seed,
            "semantic_vocab": len(space.vocab),
            "semantic_dim": space.dim,
        }

        fp = self._fingerprint(objects, edges, background)
        title = self._title(text)
        return WorldScene(objects=objects, edges=edges, background=background,
                          fingerprint=fp, title=title, stats=stats)

    @staticmethod
    def _build_edges(objects: List[WorldObject]) -> List[WorldEdge]:
        n = len(objects)
        if n < 2:
            return []
        pos = np.array([o.position for o in objects])
        d = np.sqrt(np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2))
        np.fill_diagonal(d, np.inf)
        median = float(np.median(d[np.isfinite(d)])) if np.isfinite(d).any() else 1.0
        threshold = 0.75 * median
        seen = set()
        edges: List[WorldEdge] = []
        for i in range(n):
            nearest = int(np.argmin(d[i]))
            pair = (min(i, nearest), max(i, nearest))
            if pair not in seen:
                seen.add(pair)
                edges.append(WorldEdge(pair[0], pair[1],
                                       strength=float(1.0 / (1.0 + d[i, nearest]))))
        for i in range(n):
            for j in range(i + 1, n):
                if d[i, j] <= threshold and (i, j) not in seen:
                    seen.add((i, j))
                    edges.append(WorldEdge(i, j, strength=float(1.0 / (1.0 + d[i, j]))))
        return edges

    @staticmethod
    def _fingerprint(objects: List[WorldObject], edges: List[WorldEdge],
                     background: Tuple[float, float, float]) -> str:
        h = hashlib.blake2b(digest_size=16)
        for o in objects:
            h.update(f"{o.id}|{np.round(o.position,6).tolist()}|{o.scale:.6f}|"
                     f"{np.round(o.color,6).tolist()}|{o.shape}|{o.mass:.6f}|"
                     f"{o.members}".encode())
        for e in edges:
            h.update(f"e{e.a}-{e.b}:{e.strength:.6f}".encode())
        h.update(str(np.round(background, 6).tolist()).encode())
        return h.hexdigest()

    @staticmethod
    def _title(text: str) -> str:
        words = text.strip().split()
        snippet = " ".join(words[:8])
        return snippet if len(snippet) <= 60 else snippet[:57] + "..."


# ---------------------------------------------------------------------------
# Verification gates (product-level, layered on top of the engine's own)
# ---------------------------------------------------------------------------


def gate_determinism(text: str, cfg: Optional[CompilerConfig] = None) -> None:
    cfg = cfg or CompilerConfig()
    a = WorldCompiler(cfg).compile(text)
    b = WorldCompiler(cfg).compile(text)
    if a.fingerprint != b.fingerprint:
        raise AssertionError(
            f"world-determinism gate FAILED: {a.fingerprint} != {b.fingerprint}")


def gate_diverges(text_a: str, text_b: str,
                  cfg: Optional[CompilerConfig] = None) -> None:
    """Different text must compile to a different world. If it didn't, the
    'unique approach to semantics' claim would be empty."""
    cfg = cfg or CompilerConfig()
    a = WorldCompiler(cfg).compile(text_a)
    b = WorldCompiler(cfg).compile(text_b)
    if a.fingerprint == b.fingerprint:
        raise AssertionError(
            "world-diverges gate FAILED: distinct texts compiled to the same "
            "world")


# ---------------------------------------------------------------------------
# HTML rendering — self-contained canvas 3D viewer, no external dependencies
# ---------------------------------------------------------------------------


def render_html(scene: WorldScene, out_path: str) -> None:
    from world_render import build_html
    Path(out_path).write_text(build_html(scene.to_json_dict()), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Compile text into a 3D world.")
    parser.add_argument("input", nargs="?", help="path to a text file (default: stdin)")
    parser.add_argument("-o", "--output", default="world.html", help="output HTML path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gates", action="store_true",
                        help="run determinism/divergence gates before rendering")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8") if args.input else sys.stdin.read()
    cfg = CompilerConfig(seed=args.seed)

    if args.gates:
        gate_determinism(text, cfg)
        print("GATE world-determinism   PASS")

    scene = WorldCompiler(cfg).compile(text)
    render_html(scene, args.output)
    print(f"compiled {scene.stats['chunks']} chunks -> {len(scene.objects)} motifs "
         f"-> {args.output}")
    print(f"  fingerprint: {scene.fingerprint}")
    print(f"  stats: {scene.stats}")


if __name__ == "__main__":
    main()
