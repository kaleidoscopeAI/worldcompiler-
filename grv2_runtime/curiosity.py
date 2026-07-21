"""grv2_runtime/curiosity.py — negative-space sampler (ports curiosity_module.hpp/.cpp).

Proposes entities the player didn't explicitly name but the scene implies (a
cold scene implies breath fog; a hostile creature implies a blood trail),
scores each candidate on how much it would shift coherence/question-
tension/surprise/duality-risk, and keeps only candidates that pass an
envelope (coherence floor, minimum question-gain, duality-risk ceiling).

New relative to the C++ version: the candidate pool can also include
`extra_candidates` -- entities the caller (grv2_runtime.runtime.Runtime)
builds from the surviving motifs of a world_live.LiveWorld population, so the
"negative space" filled is grounded in whatever has actually been fed into
this session, not only a fixed lexicon.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set

from .sgr import Entity


@dataclass
class CuriosityConfig:
    min_coherence: float = 0.75
    min_question_gain: float = 0.12
    max_duality_risk: float = 0.85
    max_additions_per_turn: int = 4
    max_candidates: int = 24


@dataclass
class CandidateEval:
    candidate: Entity
    delta_coherence: float = 0.0
    delta_question: float = 0.0
    delta_surprise: float = 0.0
    delta_duality_risk: float = 0.0
    passes_envelope: bool = False


def _mk(id_: str, type_: str, pos=(0.0, 0.0, 0.0), **props: str) -> Entity:
    return Entity(id=id_, type=type_, position=pos, properties=dict(props))


class CuriosityModule:
    def __init__(self, cfg: Optional[CuriosityConfig] = None) -> None:
        self.cfg = cfg or CuriosityConfig()

    def sample_negative_space(self, known_ids: Set[str], sgr_entities: Sequence[Entity],
                              metrics, world_seed: int,
                              master_prompt: Optional[str] = None,
                              extra_candidates: Optional[Sequence[Entity]] = None) -> List[Entity]:
        evals = self.evaluate_all(known_ids, sgr_entities, metrics, world_seed,
                                  master_prompt, extra_candidates)
        out: List[Entity] = []
        for ev in evals:
            if not ev.passes_envelope:
                continue
            out.append(ev.candidate)
            if len(out) >= self.cfg.max_additions_per_turn:
                break
        return out

    def evaluate_all(self, known_ids: Set[str], sgr_entities: Sequence[Entity], metrics,
                     _world_seed: int, master_prompt: Optional[str] = None,
                     extra_candidates: Optional[Sequence[Entity]] = None) -> List[CandidateEval]:
        pool = self._build_candidate_pool(known_ids, sgr_entities, master_prompt, extra_candidates)
        evals = [self._score(c, metrics) for c in pool]
        for ev in evals:
            ev.passes_envelope = self._passes_envelope(ev, metrics)
        evals.sort(key=lambda e: (e.passes_envelope, e.delta_question + 0.5 * e.delta_surprise),
                  reverse=True)
        return evals

    def _build_candidate_pool(self, known_ids: Set[str], sgr_entities: Sequence[Entity],
                              master_prompt: Optional[str],
                              extra_candidates: Optional[Sequence[Entity]]) -> List[Entity]:
        has_cold = has_danger = has_fire = has_night = has_death = False
        for e in sgr_entities:
            for v in e.properties.values():
                lv = v.lower()
                if "cold" in lv or "snow" in lv:
                    has_cold = True
                if "hostile" in lv or "blood" in lv:
                    has_danger = True
                if "fire" in lv or "flame" in lv:
                    has_fire = True
                if "night" in lv or "midnight" in lv:
                    has_night = True
                if "dead" in lv or "resurrect" in lv:
                    has_death = True
            if "yeti" in e.type:
                has_danger = True
            if "mountain" in e.type:
                has_cold = True

        pool: List[Entity] = []

        def add(e: Entity) -> None:
            if e.id not in known_ids and e.id not in {p.id for p in pool}:
                pool.append(e)

        if has_cold:
            add(_mk("breath_fog", "atmospheric", density="thick", lingers="yes"))
            add(_mk("ice_shelf", "terrain", (2, 0, 3), stability="uncertain", implication="footing_risk"))
        if has_danger:
            add(_mk("blood_trail", "clue", (-1, 0, 0), age="hours", direction="southwest",
                    implication="prior_event"))
            add(_mk("watching_shadow", "entity", (5, 2, 0), intent="unknown", implication="not_alone"))
        if has_fire:
            add(_mk("smoke_column", "atmospheric", visibility_reduction="30%"))
            add(_mk("burn_scar", "terrain", (1, 0, 0), age="recent", implication="fire_was_here"))
        if has_night:
            add(_mk("ambient_distant_sound", "atmospheric", description="something_moving", distance="far"))
        if has_death:
            add(_mk("resonance_echo", "metaphysical", type="absence_reversing", sensation="pressure_drop"))
        if master_prompt:
            mp = master_prompt.lower()
            if "dark" in mp:
                add(_mk("shadow_deeper", "atmospheric", quality="unnatural", implication="something_within"))
            if "cost" in mp:
                add(_mk("cost_residue", "metaphysical", type="attention_marks", description="world_watching"))

        for e in sgr_entities:
            for implied in self._emit_implied_entities(e):
                add(implied)

        if extra_candidates:
            for e in extra_candidates:
                add(e)

        if len(pool) > self.cfg.max_candidates:
            pool = pool[: self.cfg.max_candidates]
        return pool

    @staticmethod
    def _emit_implied_entities(parent: Entity) -> Iterable[Entity]:
        t, id_ = parent.type, parent.id
        if "yeti" in t or t == "creature":
            px, py, pz = parent.position
            yield _mk(f"{id_}_territory_mark", "environmental_clue", (px + 3, py, pz),
                      kind="claw_grooves", implication="claimed_territory")
        if t == "player":
            for v in parent.properties.values():
                if "exhausted" in v:
                    yield _mk("labored_breathing", "atmospheric", implication="stamina_critical")
                    break
        if "storm" in t:
            yield _mk("wind_vector", "physics", speed="25ms", direction="crosswind")
        if "piton" in id_:
            yield _mk(f"{id_}_prior_climber", "absence", description="gone", implication="prior_event")

    def _score(self, c: Entity, metrics) -> CandidateEval:
        return CandidateEval(
            candidate=c,
            delta_coherence=self._coherence_delta(c),
            delta_question=self._question_delta(c),
            delta_surprise=self._surprise_delta(c),
            delta_duality_risk=self._duality_risk_delta(c, metrics.duality_risk),
        )

    def _passes_envelope(self, ev: CandidateEval, cur) -> bool:
        return ((cur.coherence + ev.delta_coherence) >= self.cfg.min_coherence
                and ev.delta_question >= self.cfg.min_question_gain
                and (cur.duality_risk + ev.delta_duality_risk) <= self.cfg.max_duality_risk)

    @staticmethod
    def _coherence_delta(e: Entity) -> float:
        if e.has_prop("implication"):
            return 0.12
        if e.type in ("atmospheric", "terrain", "clue", "trace", "metaphysical", "absence", "object"):
            return 0.08
        return 0.04

    @staticmethod
    def _question_delta(e: Entity) -> float:
        if e.type == "clue":
            return 0.42
        if e.type == "absence":
            return 0.38
        if e.type in ("entity", "creature"):
            return 0.33
        for v in e.properties.values():
            if "unknown" in v or "uncertain" in v:
                return 0.29
        if e.type == "metaphysical":
            return 0.25
        return 0.14

    @staticmethod
    def _surprise_delta(e: Entity) -> float:
        if e.type == "absence":
            return 0.35
        if e.type == "metaphysical":
            return 0.30
        return 0.15

    @staticmethod
    def _duality_risk_delta(e: Entity, current_risk: float) -> float:
        if any(v in ("game_over", "victory") for v in e.properties.values()):
            return 0.50
        if current_risk > 0.7:
            return -0.05
        return 0.02
