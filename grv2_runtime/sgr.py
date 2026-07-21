"""grv2_runtime/sgr.py — Scene Graph of Record (ports sgr_kernel.hpp/.cpp).

The deterministic entity kernel: a flat map of Entity objects, each mutated
only through a validated EntityDelta, each hashed so the whole graph reduces
to one Merkle-style root. Every other tier in grv2_runtime treats this as the
single source of truth for "what exists right now."

Differences from the C++ original, both deliberate:
  - Hashing is hashlib.blake2b, matching this repo's hashing convention
    everywhere else (organic_ai_core.derive_rng, every fingerprint() in
    kaleidoscope_core.py/genetic_manifold.py/etc.), not the C++'s FNV-1a.
    Nothing depends on bit-compatibility with the (unused) C++ server.
  - "Ephemeral seed" / wall-clock time windows are dropped. This runtime is
    turn-based, not framerate-based, so entities are tick-stamped with an
    explicit turn counter the caller passes in — a pure function of inputs,
    which is easier to test deterministically than wall-clock time.
"""
from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def _dist(a: Vec3, b: Vec3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


@dataclass
class Entity:
    id: str
    type: str
    position: Vec3 = (0.0, 0.0, 0.0)
    properties: Dict[str, str] = field(default_factory=dict)
    last_modified_tick: int = 0
    state_hash: str = ""
    immutable: bool = False

    def prop(self, key: str, default: str = "") -> str:
        return self.properties.get(key, default)

    def has_prop(self, key: str) -> bool:
        return key in self.properties


@dataclass
class EntityDelta:
    id: str
    new_position: Optional[Vec3] = None
    prop_updates: Dict[str, str] = field(default_factory=dict)
    new_type: Optional[str] = None
    is_new_entity: bool = False
    request_delete: bool = False


class ValidationOutcome(Enum):
    ACCEPTED = auto()
    REJECTED = auto()
    REROUTED = auto()


@dataclass
class ValidationResult:
    outcome: ValidationOutcome = ValidationOutcome.ACCEPTED
    reason: str = ""
    rerouted_delta: Optional[EntityDelta] = None


ValidationRule = Callable[["SceneGraphKernel", EntityDelta], ValidationResult]


def _merkle_combine(a: str, b: str) -> str:
    return hashlib.blake2b((a + "|" + b).encode(), digest_size=8).hexdigest()


class SceneGraphKernel:
    def __init__(self, world_seed: int = 0xDEADBEEF12345678) -> None:
        self.world_seed = world_seed
        self.entities: Dict[str, Entity] = {}
        self.rules: List[Tuple[str, ValidationRule]] = []
        self._merkle_dirty = True
        self._cached_root = "0" * 16

    # ---- hashing ------------------------------------------------------

    @staticmethod
    def compute_state_hash(e: Entity) -> str:
        h = hashlib.blake2b(digest_size=8)
        h.update(e.id.encode())
        h.update(e.type.encode())
        for c in e.position:
            h.update(repr(round(c, 6)).encode())
        for k in sorted(e.properties):
            h.update(k.encode())
            h.update(e.properties[k].encode())
        h.update(str(e.last_modified_tick).encode())
        return h.hexdigest()

    def get_merkle_root(self) -> str:
        if not self._merkle_dirty:
            return self._cached_root
        hashes = sorted(e.state_hash for e in self.entities.values())
        if not hashes:
            self._cached_root = "0" * 16
        else:
            level = hashes
            while len(level) > 1:
                nxt = []
                for i in range(0, len(level), 2):
                    if i + 1 < len(level):
                        nxt.append(_merkle_combine(level[i], level[i + 1]))
                    else:
                        nxt.append(level[i])
                level = nxt
            self._cached_root = level[0]
        self._merkle_dirty = False
        return self._cached_root

    def _mark_dirty(self) -> None:
        self._merkle_dirty = True

    # ---- mutation -------------------------------------------------------

    def apply_delta(self, delta: EntityDelta, tick: int) -> ValidationResult:
        for _name, rule in self.rules:
            r = rule(self, delta)
            if r.outcome == ValidationOutcome.REJECTED:
                return r
            if r.outcome == ValidationOutcome.REROUTED and r.rerouted_delta is not None:
                return self.apply_delta(r.rerouted_delta, tick)

        if delta.request_delete:
            e = self.entities.get(delta.id)
            if e is not None:
                if e.immutable:
                    return ValidationResult(ValidationOutcome.REJECTED, "Entity is immutable.")
                del self.entities[delta.id]
                self._mark_dirty()
            return ValidationResult(ValidationOutcome.ACCEPTED)

        e = self.entities.get(delta.id)
        if e is None:
            e = Entity(id=delta.id, type=delta.new_type or "unknown")
            self.entities[delta.id] = e
        if delta.new_type is not None:
            e.type = delta.new_type
        if delta.new_position is not None:
            e.position = delta.new_position
        e.properties.update(delta.prop_updates)
        e.last_modified_tick = tick
        e.state_hash = self.compute_state_hash(e)
        self._mark_dirty()
        return ValidationResult(ValidationOutcome.ACCEPTED)

    def apply_delta_batch(self, deltas: Sequence[EntityDelta], tick: int) -> bool:
        snapshot = copy.deepcopy(self.entities)
        for d in deltas:
            if self.apply_delta(d, tick).outcome == ValidationOutcome.REJECTED:
                self.entities = snapshot
                self._mark_dirty()
                return False
        return True

    def validate_and_apply_curiosity(self, candidates: Sequence[Entity], tick: int) -> List[Entity]:
        accepted: List[Entity] = []
        for c in candidates:
            d = EntityDelta(id=c.id, is_new_entity=True, new_type=c.type,
                            new_position=c.position, prop_updates=dict(c.properties))
            if self.apply_delta(d, tick).outcome == ValidationOutcome.ACCEPTED:
                stored = self.get_entity(c.id)
                if stored is not None:
                    accepted.append(stored)
        return accepted

    # ---- rules ----------------------------------------------------------

    def add_rule(self, name: str, rule: ValidationRule) -> None:
        for i, (n, _) in enumerate(self.rules):
            if n == name:
                self.rules[i] = (name, rule)
                return
        self.rules.append((name, rule))

    def remove_rule(self, name: str) -> None:
        self.rules = [(n, r) for n, r in self.rules if n != name]

    def register_default_rules(self) -> None:
        self.add_rule("no_instant_travel", _rule_no_instant_travel)
        self.add_rule("magic_cost", _rule_magic_cost)
        self.add_rule("immutability", _rule_immutability)
        self.add_rule("win_button_duality", _rule_win_button_duality)

    # ---- queries ----------------------------------------------------------

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_all_entities(self) -> List[Entity]:
        return list(self.entities.values())

    def entities_in_radius(self, center: Vec3, radius: float) -> List[str]:
        return [eid for eid, e in self.entities.items() if _dist(e.position, center) <= radius]

    def to_dict(self) -> dict:
        return {
            "world_seed": hex(self.world_seed),
            "merkle_root": self.get_merkle_root(),
            "entity_count": len(self.entities),
            "entities": [
                {"id": e.id, "type": e.type, "position": list(e.position),
                 "properties": dict(e.properties), "state_hash": e.state_hash}
                for e in self.entities.values()
            ],
        }


# ---- default rules (ported from sgr_kernel.cpp's register_default_rules) ----

def _rule_no_instant_travel(kernel: SceneGraphKernel, delta: EntityDelta) -> ValidationResult:
    if delta.new_position is None:
        return ValidationResult(ValidationOutcome.ACCEPTED)
    old = kernel.get_entity(delta.id)
    if old is None:
        return ValidationResult(ValidationOutcome.ACCEPTED)
    if _dist(old.position, delta.new_position) > 100.0:
        return ValidationResult(ValidationOutcome.REJECTED, "Movement > 100m")
    return ValidationResult(ValidationOutcome.ACCEPTED)


def _rule_magic_cost(_kernel: SceneGraphKernel, delta: EntityDelta) -> ValidationResult:
    if "magic_effect" not in delta.prop_updates:
        return ValidationResult(ValidationOutcome.ACCEPTED)
    if "attention" not in delta.prop_updates:
        return ValidationResult(ValidationOutcome.REJECTED, "Magic needs attention cost")
    try:
        attn = int(delta.prop_updates["attention"])
    except ValueError:
        return ValidationResult(ValidationOutcome.REJECTED, "attention must be an integer")
    if attn < 1:
        return ValidationResult(ValidationOutcome.REJECTED, "Attention must be >= 1")
    return ValidationResult(ValidationOutcome.ACCEPTED)


def _rule_immutability(kernel: SceneGraphKernel, delta: EntityDelta) -> ValidationResult:
    e = kernel.get_entity(delta.id)
    if e is not None and e.immutable and not delta.request_delete:
        return ValidationResult(ValidationOutcome.REJECTED, "Entity immutable")
    return ValidationResult(ValidationOutcome.ACCEPTED)


def _rule_win_button_duality(_kernel: SceneGraphKernel, delta: EntityDelta) -> ValidationResult:
    """The win button exists, is real, and is pressable -- pressing it doesn't
    end anything, it reroutes into a duality event. This is the single best
    idea in the GRV2 reference design; ported with no changes."""
    if delta.prop_updates.get("game_state") != "won":
        return ValidationResult(ValidationOutcome.ACCEPTED)
    rerouted = replace(delta, prop_updates={
        **delta.prop_updates, "game_state": "questioning", "duality_event": "win_button_pressed",
    })
    return ValidationResult(ValidationOutcome.REROUTED, "Win rerouted to duality event", rerouted)
