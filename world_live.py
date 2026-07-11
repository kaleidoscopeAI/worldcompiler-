"""world_live.py — the organism, not the document converter.

Every other file in this repo (including world_compiler.py) treats
"compile text to a world" as a batch job: ingest once, evolve for a fixed
number of generations, purify to a fixed point, render, done. That throws
away the one thing organic_ai_core and genetic_manifold were actually built
for — a population that keeps living in a closed energy economy, tick after
tick, indefinitely.

LiveWorld is the same engine used as what it is: a standing population you
can keep feeding. `seed()` founds it from an initial text and fits the
manifold that becomes its permanent "worldview" — the coordinate system every
later feed is interpreted through. `feed()` injects new text as new DNA into
the SAME running population, without resetting anything: new motifs are born
and immediately have to compete for energy against everything already alive.
`tick()` advances the economy by one generation — reproduction, death,
selection by orbit-coverage — exactly as genetic_manifold.GeneticManifold
already does, just never stopped. `scene()` takes a live, lightly-purified
snapshot for rendering without mutating the actual gene pool (visualization
is a lens, not a write path into evolution).

Energy accounting stays provable across an unbounded run: the only exogenous
energy the world ever receives is its founding budget (cfg.total_energy) and
whatever feed() explicitly mints per fed chunk. total_energy() at any instant
must equal the sum of those two — gate_conservation() checks exactly that,
no matter how many ticks or feeds have happened in between.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import kaleidoscope_group as kgroup
import genetic_manifold as gmod
import refinement_loop as refine
import semantic_embedding as sem
import gcode_embedding as gcode

import world_compiler as wc


@dataclass
class FeedEvent:
    t: float
    chars: int
    chunks: int
    tick_at_feed: int


class LiveWorld:
    """A standing, feedable population. Not thread-safe on its own — callers
    (e.g. world_server.py) must hold `.lock` around any seed/feed/tick/scene
    call that must not interleave with another."""

    def __init__(self, cfg: Optional[wc.CompilerConfig] = None) -> None:
        self.cfg = cfg or wc.CompilerConfig()
        self.gm: Optional[gmod.GeneticManifold] = None
        self.lock = threading.RLock()
        self.tick_count = 0
        self.founding_energy = 0.0
        self.total_fed_energy = 0.0
        self.total_fed_chars = 0
        self.origin_labels: Dict[str, str] = {}
        self.feed_log: List[FeedEvent] = []
        self._feed_epoch = 0
        self.semantic: Optional[sem.SemanticSpace] = None
        self.gcode_space: Optional[gcode.GcodeSpace] = None

    @property
    def seeded(self) -> bool:
        return self.gm is not None

    # -- founding -----------------------------------------------------------

    def seed(self, text: str) -> None:
        """Fit the manifold (the world's permanent 'worldview') and found the
        initial population. Can only be called once per LiveWorld."""
        with self.lock:
            if self.gm is not None:
                raise wc.WorldCompilerError("world already seeded; start a new one to reseed")
            cfg = self.cfg
            text = text.strip()
            window, stride, snippets, data, space, gcode_sp = wc._ingest(text, cfg)
            n = len(snippets)
            self.semantic = space
            self.gcode_space = gcode_sp

            gcfg = gmod.GeneticConfig(
                seed=cfg.seed, generations=0, mutation_scale=cfg.mutation_scale,
                max_strains=cfg.max_strains, group_max_elements=cfg.group_max_elements,
                strain_resolution=cfg.strain_resolution,
                total_energy=max(600.0, n * 10.0),
                max_population=min(220, max(80, n * 3)))
            gm = gmod.GeneticManifold(gcfg)
            gm.seed_from_data(data)
            self.gm = gm
            self.founding_energy = gcfg.total_energy
            for i in range(n):
                self.origin_labels[wc._origin_hash(data[i])] = " ".join(snippets[i].split())
            self.feed_log.append(FeedEvent(t=time.time(), chars=len(text), chunks=n, tick_at_feed=0))

    # -- feeding: new DNA into the SAME running population -------------------

    def feed(self, text: str) -> int:
        """Embed `text` through the world's existing manifold and inject the
        resulting DNA as new nodes with fresh (exogenous, tracked) energy.
        Returns the number of chunks added. The manifold is never refit —
        text arriving after founding is seen through the lens the founding
        text already shaped, which is the point: a world has a worldview."""
        with self.lock:
            if self.gm is None or self.gm.manifold is None:
                raise wc.WorldCompilerError("seed the world before feeding it")
            cfg = self.cfg
            text = text.strip()
            # Reuse the founding SemanticSpace (never refit) — a feed is seen
            # through the vocabulary the world already learned, exactly like
            # it's seen through the founding geometric manifold below.
            window, stride, snippets, data, _space, _ = wc._ingest(
                text, cfg, semantic=self.semantic, gcode_space=self.gcode_space)
            n = len(snippets)

            codes = self.gm.manifold.encode(data)
            self._feed_epoch += 1
            seed_energy = self.gm.cfg.seed_energy
            for i in range(n):
                origin = wc._origin_hash(data[i])
                self.origin_labels[origin] = " ".join(snippets[i].split())
                genome = gmod.CodeGenome(code=codes[i].copy(), generation=0, origin_hash=origin)
                node_id = f"feed{self._feed_epoch}-{i}"
                self.gm.nodes[node_id] = gmod.GeneticNode(node_id, genome, energy=seed_energy)

            self.total_fed_energy += n * seed_energy
            self.total_fed_chars += len(text)
            self.feed_log.append(FeedEvent(t=time.time(), chars=len(text), chunks=n,
                                           tick_at_feed=self.tick_count))
            del self.feed_log[:-20]
            return n

    def feed_gcode(self, gcode_text: str) -> int:
        """Inject a G-code toolpath into the running population.

        The G-code text is embedded through BOTH channels:
          • text channel: trigram + semantic features of the G-code tokens,
            seen through the founding semantic worldview.
          • geometry channel: point-cloud features extracted from the parsed
            3-D waypoints, normalised through the founding GcodeSpace.

        Raises ``WorldCompilerError`` if the world was not founded with a
        geometry channel (``cfg.gcode_dim == 0``).  Returns the number of
        chunks added.
        """
        with self.lock:
            if self.gm is None or self.gm.manifold is None:
                raise wc.WorldCompilerError("seed the world before feeding it")
            if self.cfg.gcode_dim <= 0 or self.gcode_space is None:
                raise wc.WorldCompilerError(
                    "world was not founded with a geometry channel "
                    "(set cfg.gcode_dim > 0 when seeding)")
            cfg = self.cfg
            gcode_text = gcode_text.strip()
            # Embed through both the founding semantic space and the founding
            # GcodeSpace — _ingest will detect G-code commands and compute
            # geometry features using the existing GcodeSpace normaliser.
            window, stride, snippets, data, _space, _ = wc._ingest(
                gcode_text, cfg,
                semantic=self.semantic,
                gcode_space=self.gcode_space)
            n = len(snippets)

            codes = self.gm.manifold.encode(data)
            self._feed_epoch += 1
            seed_energy = self.gm.cfg.seed_energy
            for i in range(n):
                origin = wc._origin_hash(data[i])
                self.origin_labels[origin] = f"[gcode] {' '.join(snippets[i].split())[:60]}"
                genome = gmod.CodeGenome(
                    code=codes[i].copy(), generation=0, origin_hash=origin)
                node_id = f"gcode{self._feed_epoch}-{i}"
                self.gm.nodes[node_id] = gmod.GeneticNode(
                    node_id, genome, energy=seed_energy)

            self.total_fed_energy += n * seed_energy
            self.total_fed_chars += len(gcode_text)
            self.feed_log.append(FeedEvent(
                t=time.time(), chars=len(gcode_text), chunks=n,
                tick_at_feed=self.tick_count))
            del self.feed_log[:-20]
            return n

    # -- living ---------------------------------------------------------------

    def tick(self) -> None:
        with self.lock:
            if self.gm is None:
                return
            self.gm.step()
            self.tick_count += 1

    def gate_conservation(self, tol: float = 1e-6) -> float:
        """total_energy() must equal exactly the founding budget plus every
        feed() mint, no matter how many ticks have run in between. Returns
        the drift; raises if it exceeds tolerance."""
        with self.lock:
            if self.gm is None:
                return 0.0
            expected = self.founding_energy + self.total_fed_energy
            drift = abs(self.gm.total_energy() - expected)
            if drift > tol:
                raise AssertionError(
                    f"live-conservation gate FAILED: drift {drift:.3e} > {tol:.0e} "
                    f"(have {self.gm.total_energy():.6f}, expected {expected:.6f})")
            return drift

    # -- snapshot for rendering (does not mutate the gene pool) --------------

    def scene(self, purify_passes: int = 6) -> wc.WorldScene:
        with self.lock:
            if self.gm is None or not self.gm.nodes:
                return wc.WorldScene(objects=[], edges=[], background=(0.02, 0.02, 0.04),
                                     fingerprint="unseeded", title="(empty world — feed it text)",
                                     stats=self._stats())

            cfg = self.cfg
            node_ids = sorted(self.gm.nodes)
            codes = np.stack([self.gm.nodes[nid].genome.code for nid in node_ids])
            origins = [self.gm.nodes[nid].genome.origin_hash for nid in node_ids]

            # Purification here is a DISPLAY LENS: it runs on a copy, over a
            # handful of passes (not to convergence), so the world always
            # looks like it is settling without ever mutating the actual
            # evolving gene pool that tick()/feed() operate on.
            rcfg = refine.RefinementConfig(
                seed=cfg.seed, max_passes=purify_passes, purify_rate=0.5,
                strain_resolution=cfg.strain_resolution,
                group_max_elements=cfg.group_max_elements)
            eng = refine.RefinementEngine(rcfg)
            eng.manifold = self.gm.manifold
            eng.group = self.gm.group
            eng.codes = codes.copy()
            report = eng.run()

            strains, labels = kgroup.form_strains(
                eng.codes, eng.group, resolution=cfg.strain_resolution)

            def label_for(member_idx: int) -> str:
                return self.origin_labels.get(origins[member_idx], "(inherited motif)")

            objects = wc.project_strains(strains, eng.group, label_for)
            edges = wc.WorldCompiler._build_edges(objects)

            bg_hue = wc._stable_unit(np.round(report.one_entity, 6).tobytes()) * 360.0
            background = wc._hsl_to_rgb(bg_hue, 0.35, 0.10)
            fp = wc.WorldCompiler._fingerprint(objects, edges, background)

            return wc.WorldScene(
                objects=objects, edges=edges, background=background, fingerprint=fp,
                title=f"a living world — tick {self.tick_count}",
                stats=self._stats(n_motifs=len(objects)))

    def _stats(self, n_motifs: int = 0) -> dict:
        gm = self.gm
        return {
            "seeded": gm is not None,
            "tick": self.tick_count,
            "population": len(gm.nodes) if gm else 0,
            "pool": round(float(gm.pool), 3) if gm else 0.0,
            "total_energy": round(float(gm.total_energy()), 3) if gm else 0.0,
            "fed_chars": self.total_fed_chars,
            "fed_energy": round(self.total_fed_energy, 3),
            "manifold_rank": int(gm.manifold.rank) if gm and gm.manifold else 0,
            "group_order": int(gm.group.elements.shape[0]) if gm and gm.group else 0,
            "semantic_vocab": len(self.semantic.vocab) if self.semantic else 0,
            "motifs": n_motifs,
            "feeds": len(self.feed_log),
        }


class Heartbeat:
    """Advances a LiveWorld on a fixed cadence in a background thread, so the
    world keeps evolving even if nobody is watching or feeding it."""

    def __init__(self, world: LiveWorld, interval_s: float = 1.2) -> None:
        self.world = world
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            if self.world.seeded:
                self.world.tick()
            self._stop.wait(self.interval_s)
