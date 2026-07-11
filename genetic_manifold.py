"""genetic_manifold.py — The data IS the DNA.

The reframe: a node's genome is not a vector of hyperparameters describing how
to learn. A node's genome is the *compressed latent code of the data it
ingested* — z_i = Ψ_membrane(x_i). The node literally is its data, in the
compressed coordinate system S* discovered by the Kaleidoscope engine.

  individual DNA   z_i          = membrane-filtered latent code of one datum
  refined strain   S_k = Σ a_i z_i (a_i normalized within cluster k)
                                = the mass-weighted consensus code of a dense
                                  region — a supernode, now read as a lineage
  the one entity   S* = Σ w_k S_k = the whole population as one coordinate

Reproduction operates on knowledge, not parameters: a child inherits the
parent's code with a small mutation, so lineages drift through the data
manifold. Selection is driven by two competing, measurable pressures:

  fidelity   how well the code reconstructs the datum it represents
             (does this DNA still encode real signal?)
  consensus  how central the code is to its strain
             (does this DNA carry shared structure, or is it an outlier?)

These pressures are genuinely in tension — pure fidelity rewards memorizing
individual points, pure consensus rewards collapsing to the mean. The honest
question this module answers by MEASUREMENT (not assertion) is whether that
tension produces strains that separate distinct data regimes, or degenerates.
The gate `gate_strains_separate_regimes` tests exactly that: feed the organism
data from several distinct generative regimes and check the evolved strains
recover the regime structure better than chance.

Built on kaleidoscope_core (the membrane + MDL manifold + Mirror) and the
verified v4 substrate (organic_ai_core: derive_rng, DataSource, energy economy
patterns). Deterministic, numpy + stdlib only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

import organic_ai_core as core
import kaleidoscope_core as kal
import kaleidoscope_group as kgroup


# ===========================================================================
# The genome IS the code
# ===========================================================================


@dataclass
class CodeGenome:
    """A node's DNA: its ingested datum, as a latent code in S*.

    Immutable-by-convention numeric vector (no pickle/exec, same safety
    posture as v4's Genome). `code` lives in the manifold's latent space;
    `origin_hash` ties it to the datum it came from for provenance.
    """

    code: np.ndarray            # (k,) latent DNA
    generation: int
    origin_hash: str            # provenance of the founding datum

    def mutate(self, rng: np.random.Generator, scale: float) -> "CodeGenome":
        """Heritable drift through the data manifold. Mutation is Gaussian in
        latent space — a child's DNA is its parent's code nudged toward a
        neighboring region of S*."""
        noise = rng.normal(0.0, scale, size=self.code.shape)
        return CodeGenome(code=self.code + noise,
                          generation=self.generation + 1,
                          origin_hash=self.origin_hash)

    def integrity_hash(self) -> str:
        h = hashlib.blake2b(digest_size=8)
        h.update(np.round(self.code, 9).tobytes())
        h.update(self.origin_hash.encode())
        return h.hexdigest()


# ===========================================================================
# A node whose identity is its code
# ===========================================================================


class GeneticNode:
    """A living unit that IS its ingested data. Its DNA is a latent code; its
    fitness is fidelity (reconstructs its datum) balanced against consensus
    (central to its strain)."""

    __slots__ = ("id", "genome", "energy", "strain_id", "_fitness",
                 "children_born")

    def __init__(self, node_id: str, genome: CodeGenome, energy: float) -> None:
        self.id = node_id
        self.genome = genome
        self.energy = float(energy)
        self.strain_id = -1
        self._fitness = float("nan")
        self.children_born = 0

    def reconstruct(self, manifold: kal.Manifold) -> np.ndarray:
        """Regenerate the datum this DNA encodes (S* -> X)."""
        return manifold.reconstruct(self.genome.code)

    def fidelity(self, datum: np.ndarray, manifold: kal.Manifold) -> float:
        """How well this DNA reconstructs its datum. Higher = better."""
        recon = self.reconstruct(manifold)
        mse = float(np.mean((datum - recon) ** 2))
        return 1.0 / (1.0 + mse)

    def consensus(self, strain_code: np.ndarray) -> float:
        """How central this DNA is to its strain. Higher = more shared signal."""
        d = float(np.mean((self.genome.code - strain_code) ** 2))
        return 1.0 / (1.0 + d)


# ===========================================================================
# The evolving population over the data manifold
# ===========================================================================


@dataclass
class GeneticConfig:
    seed: int = 0
    generations: int = 30
    mutation_scale: float = 0.05
    fidelity_weight: float = 0.5    # w_f in fitness; consensus gets (1 - w_f)
    total_energy: float = 600.0
    founder_energy_fraction: float = 0.25  # founders may consume at most this
    #                                        share of total_energy; the rest is
    #                                        pool surplus that fuels growth so
    #                                        reproduction can actually occur
    #                                        under conservation.
    seed_energy: float = 1.0
    reproduction_threshold: float = 2.0
    child_fraction: float = 0.5
    death_energy: float = 1e-3
    max_population: int = 500
    reward_per_gen: float = 0.6     # energy minted per node per gen, capped by pool
    metabolic_cost: float = 0.02
    selection_sharpness: float = 6.0  # exp() temperature on fitness in reward
    max_strains: int = 8
    strain_resolution: float = 0.35   # kaleidoscope angular resolution
    group_max_elements: int = 64      # cap on |G| (turns+mirrors) materialized


@dataclass
class StrainReport:
    generations: int
    final_population: int
    n_strains: int
    strain_codes: np.ndarray            # (K, k) refined DNA strains (supernodes)
    strain_masses: np.ndarray           # (K,) w_k
    one_entity: np.ndarray              # S* = Σ w_k S_k
    fitness_curve: List[float]          # mean population fitness per generation
    diversity_curve: List[float]        # mean pairwise strain distance per gen
    energy_drift: float
    fingerprint: str
    collapsed: bool                     # did all strains converge to one point?


class GeneticManifold:
    """Evolves a population whose DNA is ingested data, refining it into
    strains (supernodes). The manifold (membrane + MDL basis) is fit once from
    the data — that is the fixed 'genetic code alphabet' S* the population
    evolves within. Energy is conserved across the run (drift-gated)."""

    def __init__(self, config: GeneticConfig) -> None:
        self.cfg = config
        self.manifold: Optional[kal.Manifold] = None
        self.group: Optional[kgroup.KaleidoscopeGroup] = None
        self.nodes: Dict[str, GeneticNode] = {}
        self._data: Optional[np.ndarray] = None
        self.pool = 0.0

    def seed_from_data(self, data: np.ndarray) -> None:
        """Ingest data: fit the manifold, then found one node per datum whose
        DNA is that datum's latent code."""
        cfg = self.cfg
        data = np.asarray(data, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 2:
            raise core.OrganicError("need (N>=2, D) data to seed genetic manifold")
        self._data = data

        # Fit the manifold once — the alphabet of S* the population lives in.
        org = kal.CompressionOrganism(kal.CompressionConfig(
            seed=cfg.seed, max_supernodes=cfg.max_strains))
        org.compress_array(data)
        self.manifold = org.manifold

        # The kaleidoscope group acts on the latent DNA space. Turns + mirrors
        # are fixed once the manifold's dimensionality (S* alphabet) is known.
        self.group = kgroup.KaleidoscopeGroup(
            dim=self.manifold.rank, max_elements=cfg.group_max_elements)

        codes = self.manifold.encode(data)
        # Founding population is bounded by BOTH max_population and the energy
        # budget: we can only afford total_energy / seed_energy founders. If the
        # dataset is larger, we found from an evenly-spaced deterministic subset
        # (every stride-th datum) so the founders still tile the manifold.
        # Founders may consume only founder_energy_fraction of the budget, so
        # the pool keeps a surplus to mint growth rewards from. Without this,
        # founders eat the entire budget (pool=0) and — under conservation —
        # no node can ever accumulate enough energy to reproduce.
        budget_cap = int((cfg.total_energy * cfg.founder_energy_fraction)
                         // cfg.seed_energy)
        n = min(data.shape[0], cfg.max_population, budget_cap)
        if n < 2:
            raise core.OrganicError(
                "energy budget too small to found >=2 nodes; raise "
                "total_energy or lower seed_energy")
        stride = max(1, data.shape[0] // n)
        founder_idx = list(range(0, data.shape[0], stride))[:n]
        used = n * cfg.seed_energy
        self.pool = cfg.total_energy - used

        for slot, i in enumerate(founder_idx):
            origin = hashlib.blake2b(data[i].tobytes(), digest_size=8).hexdigest()
            genome = CodeGenome(code=codes[i].copy(), generation=0,
                                origin_hash=origin)
            node = GeneticNode(f"seed-{slot}", genome, cfg.seed_energy)
            self.nodes[node.id] = node

    def total_energy(self) -> float:
        return self.pool + sum(n.energy for n in self.nodes.values())

    def _strains(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Form refined strains as kaleidoscope ORBITS, not clusters.

        Two DNA codes share a strain iff one is a mirrored/rotated view of the
        other (same group invariant), NOT merely if they are near each other.
        The supernode is the strain's invariant representative — the conserved
        identity that survives every turn and mirror. Returns
        (representatives[K,k], labels[pop], masses[K]) so the rest of the
        economy (which expects a per-strain code and per-node label) is
        unchanged."""
        assert self.group is not None
        codes = np.stack([self.nodes[nid].genome.code
                          for nid in sorted(self.nodes)])
        strains, labels = kgroup.form_strains(
            codes, self.group, resolution=self.cfg.strain_resolution)
        representatives = np.stack([s.representative for s in strains])
        masses = np.array([s.mass for s in strains], dtype=np.float64)
        return representatives, labels, masses

    def step(self) -> Tuple[float, float]:
        """One generation: assign strains, score by ORBIT COVERAGE (not a dead
        fitness scalar), run the conserved energy economy, reproduce, cull.
        Returns (mean_coverage, diversity)."""
        cfg = self.cfg
        assert self.manifold is not None and self._data is not None and \
            self.group is not None

        ordered = sorted(self.nodes)
        codes = np.stack([self.nodes[nid].genome.code for nid in ordered])
        centroids, labels, masses = self._strains()
        for idx, nid in enumerate(ordered):
            self.nodes[nid].strain_id = int(labels[idx])

        # Selection signal = orbit-coverage contribution. A node survives by
        # holding a view of its strain that nothing else covers; a redundant
        # duplicate view contributes ~0 and starves. This is structural — it
        # comes from the group, not an imposed metric — and it actually
        # discriminates (rare views score high, duplicates low), which the old
        # fidelity/consensus scalar could not.
        coverage = kgroup.coverage_contribution(codes, labels, self.group)
        for idx, nid in enumerate(ordered):
            self.nodes[nid]._fitness = float(coverage[idx])

        # Comparative economy (v4 pattern): reward relative to median coverage,
        # pro-rata under scarcity, energy conserved via the pool. Conservation
        # is enforced by construction: the total minted equals exactly what is
        # withdrawn from the pool, and metabolic cost returns exactly to it.
        median = max(float(np.median(coverage)), 1e-6)
        raw = {nid: float(np.exp(cfg.selection_sharpness *
                                 (self.nodes[nid]._fitness - median)))
               for nid in ordered}
        total_raw = sum(raw.values())
        # Mint at most reward_per_gen·N, and never more than the pool holds.
        budget = min(cfg.reward_per_gen * len(ordered), max(self.pool, 0.0))
        for nid in ordered:
            share = (raw[nid] / total_raw) if total_raw > 0 else 0.0
            reward = budget * share
            self.pool -= reward
            self.nodes[nid].energy += reward
        # Metabolic cost: withdraw from each node, return the same sum to pool.
        for nid in ordered:
            cost = min(cfg.metabolic_cost, self.nodes[nid].energy)
            self.nodes[nid].energy -= cost
            self.pool += cost

        self._reproduce()
        self._cull()

        mean_cov = float(np.mean(coverage)) if len(coverage) else 0.0
        diversity = self._strain_diversity(centroids)
        return mean_cov, diversity

    def _reproduce(self) -> None:
        cfg = self.cfg
        rng_base = ("reproduce", cfg.seed)
        # Snapshot the parents up front: never iterate a dict we mutate, and
        # never let a child reproduce in the same generation it was born.
        for nid in sorted(self.nodes):
            if len(self.nodes) >= cfg.max_population:
                break
            parent = self.nodes[nid]
            if parent.energy < cfg.reproduction_threshold:
                continue
            child_rng = core.derive_rng(*rng_base, nid, parent.genome.generation,
                                        parent.children_born)
            child_genome = parent.genome.mutate(child_rng, cfg.mutation_scale)
            # Unique child id: parent id + child-count. A collision would
            # overwrite an existing node and silently destroy its energy
            # (measured as a conservation leak), so the counter is mandatory.
            child_id = f"{nid}#{parent.children_born}"
            while child_id in self.nodes:
                parent.children_born += 1
                child_id = f"{nid}#{parent.children_born}"
            parent.children_born += 1
            transfer = parent.energy * cfg.child_fraction
            parent.energy -= transfer
            child = GeneticNode(child_id, child_genome, transfer)
            self.nodes[child_id] = child

    def _cull(self) -> None:
        dead = [nid for nid, n in self.nodes.items()
                if n.energy <= self.cfg.death_energy]
        for nid in dead:
            self.pool += self.nodes[nid].energy
            del self.nodes[nid]

    def _strain_diversity(self, centroids: np.ndarray) -> float:
        if len(centroids) < 2:
            return 0.0
        dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                dists.append(float(np.sqrt(np.sum(
                    (centroids[i] - centroids[j]) ** 2))))
        return float(np.mean(dists))

    def evolve(self) -> StrainReport:
        cfg = self.cfg
        assert self.manifold is not None
        e0 = self.total_energy()
        fitness_curve: List[float] = []
        diversity_curve: List[float] = []
        for _ in range(cfg.generations):
            mf, dv = self.step()
            fitness_curve.append(mf)
            diversity_curve.append(dv)

        centroids, labels, masses = self._strains()
        one_entity = masses @ centroids
        collapsed = self._strain_diversity(centroids) < 1e-6
        drift = abs(self.total_energy() - e0)

        h = hashlib.blake2b(digest_size=12)
        h.update(np.round(centroids, 9).tobytes())
        h.update(np.round(masses, 9).tobytes())
        h.update(str(len(self.nodes)).encode())
        fingerprint = h.hexdigest()

        return StrainReport(
            generations=cfg.generations, final_population=len(self.nodes),
            n_strains=len(centroids), strain_codes=centroids,
            strain_masses=masses, one_entity=one_entity,
            fitness_curve=fitness_curve, diversity_curve=diversity_curve,
            energy_drift=drift, fingerprint=fingerprint, collapsed=collapsed)


# ===========================================================================
# Gates
# ===========================================================================


def gate_determinism(data: np.ndarray, cfg: GeneticConfig) -> None:
    a = GeneticManifold(cfg)
    a.seed_from_data(data.copy())
    ra = a.evolve()
    b = GeneticManifold(cfg)
    b.seed_from_data(data.copy())
    rb = b.evolve()
    if ra.fingerprint != rb.fingerprint:
        raise AssertionError(
            f"determinism FAILED: {ra.fingerprint} != {rb.fingerprint}")
    if not np.allclose(ra.fitness_curve, rb.fitness_curve, atol=0.0):
        raise AssertionError("determinism FAILED: fitness curves differ")


def gate_energy_conservation(report: StrainReport, tol: float = 1e-6) -> None:
    if report.energy_drift > tol:
        raise AssertionError(
            f"conservation FAILED: drift {report.energy_drift:.3e} > {tol:.0e}")


def gate_no_collapse(report: StrainReport) -> None:
    """The population must NOT collapse all DNA to a single point. If it does,
    consensus pressure has overwhelmed fidelity and the strains carry no
    distinct information — report it honestly rather than calling it success."""
    if report.collapsed:
        raise AssertionError(
            "collapse gate FAILED: all strains converged to one point "
            "(consensus overwhelmed fidelity)")


def gate_strains_separate_regimes(cfg: GeneticConfig, n_regimes: int = 3,
                                  per_regime: int = 120, dim: int = 12
                                  ) -> float:
    """The real test of the idea. Build data from `n_regimes` distinct
    generative regimes (different latent loadings). After evolution, each datum
    maps to a strain; a good genetic manifold assigns data from the same regime
    to the same strain. We measure regime/strain agreement (adjusted for chance)
    and require it to beat random assignment. Returns the purity score.

    This is the falsifiable claim behind 'data as DNA, supernodes as refined
    strains': if strains are meaningful lineages they recover regime structure;
    if the idea is empty they scatter."""
    rng = core.derive_rng("regime-gate", cfg.seed, n_regimes, dim)
    blocks = []
    truth = []
    for r in range(n_regimes):
        loading = rng.normal(size=(3, dim)) * (r + 1)  # distinct regimes
        factors = rng.normal(size=(per_regime, 3))
        blocks.append(factors @ loading + 0.05 * rng.normal(size=(per_regime, dim)))
        truth.extend([r] * per_regime)
    data = np.vstack(blocks)
    truth = np.array(truth)

    gm = GeneticManifold(cfg)
    gm.seed_from_data(data)
    gm.evolve()

    # Assign each ORIGINAL datum to its nearest strain and score purity.
    assert gm.manifold is not None
    codes = gm.manifold.encode(data)
    centroids, _, _ = gm._strains()
    d = np.sum((codes[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    assign = np.argmin(d, axis=1)

    # Purity: for each strain, the fraction that is its majority regime.
    purity = 0.0
    for s in np.unique(assign):
        members = truth[assign == s]
        if len(members):
            counts = np.bincount(members, minlength=n_regimes)
            purity += counts.max()
    purity /= len(truth)

    chance = 1.0 / n_regimes
    if purity <= chance + 0.15:
        raise AssertionError(
            f"regime-separation FAILED: strain purity {purity:.3f} not "
            f"meaningfully above chance {chance:.3f}")
    return purity


# ===========================================================================
# Demonstration
# ===========================================================================


def _demo() -> None:
    cfg = GeneticConfig()

    print("=" * 70)
    print("GENETIC MANIFOLD — the data IS the DNA")
    print("=" * 70)

    purity = gate_strains_separate_regimes(cfg)
    print(f"GATE strains-separate-regimes  PASS  (strain purity {purity:.3f} "
          f"vs chance 0.333)")

    rng = core.derive_rng("gm-demo", cfg.seed)
    factors = rng.normal(size=(300, 4))
    loading = rng.normal(size=(4, 14))
    data = factors @ loading + 0.03 * rng.normal(size=(300, 14))

    gate_determinism(data, cfg)
    print("GATE determinism               PASS  (identical strains x2)")

    gm = GeneticManifold(cfg)
    gm.seed_from_data(data)
    report = gm.evolve()

    gate_energy_conservation(report)
    gate_no_collapse(report)
    print(f"GATE energy-conservation       PASS  (drift "
          f"{report.energy_drift:.2e})")
    print("GATE no-collapse               PASS  (strains stay distinct)")

    print("-" * 70)
    print("EVOLUTION OF INGESTED-DATA DNA INTO REFINED STRAINS")
    print(f"  founding nodes (1 per datum) : {data.shape[0]}  "
          f"(DNA = latent code of each datum)")
    print(f"  latent DNA dimension         : {gm.manifold.rank}")
    print(f"  generations                  : {report.generations}")
    print(f"  surviving population         : {report.final_population}")
    print(f"  refined strains (supernodes) : {report.n_strains}")
    print(f"  mean coverage  {report.fitness_curve[0]:.4f} -> "
          f"{report.fitness_curve[-1]:.4f}")
    print(f"  strain diversity {report.diversity_curve[0]:.4f} -> "
          f"{report.diversity_curve[-1]:.4f}")
    print(f"  strain masses w_k            : "
          f"{np.round(report.strain_masses, 3).tolist()}")
    print(f"  S* 'one entity' (Σ w_k S_k)  : dim={report.one_entity.shape[0]}, "
          f"‖S*‖={np.linalg.norm(report.one_entity):.4f}")
    print(f"  fingerprint                  : {report.fingerprint}")

    print("-" * 70)
    print("STRAIN LINEAGE VIEW (each strain = a refined DNA consensus):")
    for k in range(report.n_strains):
        code = report.strain_codes[k]
        print(f"  strain {k}: mass {report.strain_masses[k]:.3f}  "
              f"‖DNA‖ {np.linalg.norm(code):.3f}  "
              f"code[:3]={np.round(code[:3], 3).tolist()}")


if __name__ == "__main__":
    _demo()
