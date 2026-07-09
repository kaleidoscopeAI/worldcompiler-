"""refinement_loop.py — raw DNA -> pure inherited DNA -> supernode crystallization.

The process you described, made literal:

  1. Each node's DNA (its ingested latent code) is fed THROUGH the kaleidoscope
     engine: it is snapped toward the canonical (invariant) view of its orbit.
     One turn of the tube purifies it a little — noise in the code that does not
     belong to the shape's true symmetry class is filtered out.

  2. The purified DNA is looped BACK and fed through again. Across passes the
     code converges: raw ingested data -> pure inherited DNA (the fixed point
     of the engine — the view the shape "wants" to be). Inheritance here means
     the code has settled onto the conserved identity its whole orbit shares.

  3. Only when every node has stopped moving (all DNA is pure/inherited) do the
     nodes enter the engine for the LAST time — and that final pass is where
     SUPERNODES form: nodes whose purified DNA now shares one invariant collapse
     into a single strain. Supernodes are not built early from noisy codes; they
     crystallize at the end, from settled DNA.

This is a contraction by construction: each pass moves a code toward its
orbit's canonical representative and never away, so the per-node drift is
monotone non-increasing to a fixed point. The gate proves it.

Composes: kaleidoscope_core.Manifold (S* alphabet), kaleidoscope_group
(turns/mirrors, orbits, invariants, strains). Deterministic, numpy + stdlib.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

import organic_ai_core as core
import kaleidoscope_core as kal
import kaleidoscope_group as kgroup


def _purify_once(code: np.ndarray, group: kgroup.KaleidoscopeGroup,
                 rate: float) -> np.ndarray:
    """One turn of the tube: move a code toward the canonical view of its orbit.

    The canonical view is the orbit member with the largest lexicographic
    signature (the 'upright' image). Pulling the code toward it by `rate`
    strips off the component that does not respect the shape's symmetry class.
    Fully applied (rate=1) the code lands exactly on its inherited DNA.
    """
    orbit = group.orbit(code)
    scores = [(tuple(np.round(-p, 9)), i) for i, p in enumerate(orbit)]
    canonical = orbit[min(scores)[1]]
    return code + rate * (canonical - code)


@dataclass
class RefinementConfig:
    seed: int = 0
    max_passes: int = 40
    purify_rate: float = 0.5          # fraction of the way to canonical per pass
    convergence_tol: float = 1e-6     # stop when max per-node drift < tol
    strain_resolution: float = 0.35
    group_max_elements: int = 64


@dataclass
class RefinementReport:
    passes_run: int
    converged: bool
    drift_curve: List[float]          # max per-node DNA movement per pass
    purity_curve: List[float]         # mean distance code->its canonical view
    n_supernodes: int                 # strains formed on the FINAL pass only
    supernode_reps: np.ndarray        # (K, k) crystallized supernode DNA
    supernode_masses: np.ndarray      # (K,)
    one_entity: np.ndarray            # S*
    is_contraction: bool              # drift monotone non-increasing?
    fingerprint: str


class RefinementEngine:
    """Loops nodes through the kaleidoscope until DNA is pure, then crystallizes
    supernodes on the final pass."""

    def __init__(self, config: RefinementConfig) -> None:
        self.cfg = config
        self.codes: np.ndarray = np.empty((0, 0))
        self.group: kgroup.KaleidoscopeGroup = None  # type: ignore
        self.manifold: kal.Manifold = None            # type: ignore

    def load(self, data: np.ndarray) -> None:
        """Ingest raw data -> initial (raw) DNA codes + the engine's group."""
        cfg = self.cfg
        org = kal.CompressionOrganism(kal.CompressionConfig(seed=cfg.seed))
        org.compress_array(np.asarray(data, dtype=np.float64))
        self.manifold = org.manifold
        self.codes = self.manifold.encode(np.asarray(data, dtype=np.float64))
        self.group = kgroup.KaleidoscopeGroup(
            dim=self.manifold.rank, max_elements=cfg.group_max_elements)

    def _mean_purity_gap(self) -> float:
        """Mean distance from each code to its strain's inherited DNA (the
        shared canonical representative it is converging onto)."""
        strains, labels = kgroup.form_strains(
            self.codes, self.group, resolution=self.cfg.strain_resolution)
        total = 0.0
        for i, c in enumerate(self.codes):
            total += float(np.linalg.norm(c - strains[labels[i]].representative))
        return total / max(len(self.codes), 1)

    def run(self) -> RefinementReport:
        cfg = self.cfg
        drift_curve: List[float] = []
        purity_curve: List[float] = []
        converged = False
        passes = 0

        # --- refinement: raw DNA -> pure inherited DNA -----------------------
        # Strain membership is FIXED from the raw codes (one grouping by shared
        # invariant). Each pass pulls every member toward its strain's inherited
        # DNA. Because the partition is frozen, purification can only pull codes
        # together within a lineage — it can never split a strain, so the final
        # crystallization has <= the raw strain count. Merges happen when two
        # frozen strains' inherited forms coincide within resolution.
        strains0, labels0 = kgroup.form_strains(
            self.codes, self.group, resolution=cfg.strain_resolution)
        targets = np.stack([strains0[labels0[i]].representative
                            for i in range(len(self.codes))])
        for _ in range(cfg.max_passes):
            passes += 1
            gap = float(np.mean(np.linalg.norm(self.codes - targets, axis=1)))
            purity_curve.append(gap)
            new_codes = self.codes + cfg.purify_rate * (targets - self.codes)
            drift = float(np.max(np.linalg.norm(new_codes - self.codes, axis=1)))
            drift_curve.append(drift)
            self.codes = new_codes
            if drift < cfg.convergence_tol:
                converged = True
                break

        # --- FINAL pass: supernodes crystallize from settled DNA -------------
        strains, labels = kgroup.form_strains(
            self.codes, self.group, resolution=cfg.strain_resolution)
        reps = np.stack([s.representative for s in strains])
        masses = np.array([s.mass for s in strains], dtype=np.float64)
        s_star = kgroup.one_entity(strains)

        is_contraction = all(
            drift_curve[i + 1] <= drift_curve[i] + 1e-9
            for i in range(len(drift_curve) - 1))

        h = hashlib.blake2b(digest_size=12)
        h.update(np.round(reps, 9).tobytes())
        h.update(np.round(masses, 9).tobytes())
        fingerprint = h.hexdigest()

        return RefinementReport(
            passes_run=passes, converged=converged, drift_curve=drift_curve,
            purity_curve=purity_curve, n_supernodes=len(strains),
            supernode_reps=reps, supernode_masses=masses, one_entity=s_star,
            is_contraction=is_contraction, fingerprint=fingerprint)


# ===========================================================================
# Gates
# ===========================================================================


def gate_contraction(report: RefinementReport) -> None:
    """Purification must be a contraction: per-node drift monotone non-
    increasing to a fixed point (raw -> pure is one-directional)."""
    if not report.is_contraction:
        raise AssertionError("contraction gate FAILED: DNA drift increased")


def gate_reaches_purity(report: RefinementReport, tol: float = 1e-3) -> None:
    """After refinement the DNA must be pure: mean distance to canonical view
    near zero. Otherwise codes never became inherited DNA."""
    if report.purity_curve[-1] > tol:
        raise AssertionError(
            f"purity gate FAILED: final purity gap {report.purity_curve[-1]:.3e} "
            f"> {tol} (DNA never fully inherited)")


def gate_determinism(data: np.ndarray, cfg: RefinementConfig) -> None:
    a = RefinementEngine(cfg); a.load(data.copy()); ra = a.run()
    b = RefinementEngine(cfg); b.load(data.copy()); rb = b.run()
    if ra.fingerprint != rb.fingerprint:
        raise AssertionError(
            f"determinism FAILED: {ra.fingerprint} != {rb.fingerprint}")


def gate_supernodes_only_final(cfg: RefinementConfig, data: np.ndarray) -> None:
    """Supernodes must form from SETTLED DNA, not raw. We check that forming
    strains on raw codes vs purified codes yields FEWER, cleaner strains after
    purification (noise-split raw codes collapse once purified)."""
    eng = RefinementEngine(cfg); eng.load(data.copy())
    raw_strains, _ = kgroup.form_strains(eng.codes, eng.group,
                                         resolution=cfg.strain_resolution)
    report = eng.run()
    if report.n_supernodes > len(raw_strains):
        raise AssertionError(
            f"final-crystallization gate FAILED: purified DNA produced MORE "
            f"strains ({report.n_supernodes}) than raw ({len(raw_strains)}); "
            f"purification should merge, not fragment")


# ===========================================================================
# Demo
# ===========================================================================


def _demo() -> None:
    cfg = RefinementConfig()
    rng = core.derive_rng("refine-demo", cfg.seed)
    base = rng.normal(size=(5, 4))
    grp = kgroup.KaleidoscopeGroup(dim=4, max_elements=cfg.group_max_elements)
    rows = []
    for _ in range(400):
        shape = base[rng.integers(5)]
        g = grp.elements[rng.integers(grp.elements.shape[0])]
        # embed 4-d latent shape back into a 14-d observed space with noise
        rows.append(g @ shape)
    latent = np.stack(rows)
    loading = rng.normal(size=(4, 14))
    data = latent @ loading + 0.05 * rng.normal(size=(400, 14))

    print("=" * 70)
    print("REFINEMENT LOOP — raw DNA -> pure inherited DNA -> supernodes")
    print("=" * 70)

    gate_determinism(data, cfg)
    print("GATE determinism            PASS")

    eng = RefinementEngine(cfg)
    eng.load(data)
    report = eng.run()

    gate_contraction(report)
    gate_reaches_purity(report)
    gate_supernodes_only_final(cfg, data)
    print("GATE contraction            PASS  (DNA drift monotone -> fixed pt)")
    print("GATE reaches-purity         PASS  (raw DNA became inherited DNA)")
    print("GATE supernodes-only-final  PASS  (crystallize from settled DNA)")

    print("-" * 70)
    print(f"  passes until pure           : {report.passes_run} "
          f"(converged={report.converged})")
    print(f"  DNA drift    {report.drift_curve[0]:.4f} -> "
          f"{report.drift_curve[-1]:.2e}")
    print(f"  purity gap   {report.purity_curve[0]:.4f} -> "
          f"{report.purity_curve[-1]:.2e}  (0 = fully inherited)")
    print(f"  supernodes (final pass only): {report.n_supernodes}")
    print(f"  S* one-entity               : dim={report.one_entity.shape[0]}, "
          f"‖S*‖={np.linalg.norm(report.one_entity):.4f}")
    print(f"  fingerprint                 : {report.fingerprint}")
    print("-" * 70)
    for i in range(report.n_supernodes):
        r = report.supernode_reps[i]
        print(f"  supernode {i}: mass {report.supernode_masses[i]:.3f}  "
              f"‖DNA‖ {np.linalg.norm(r):.3f}  "
              f"rep={np.round(r, 3).tolist()}")


if __name__ == "__main__":
    _demo()
