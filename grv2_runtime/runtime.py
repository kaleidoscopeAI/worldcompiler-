"""grv2_runtime/runtime.py — the merged per-turn loop (ports grv2_runtime.hpp's Runtime).

Glues the ported GRV2 tiers (sgr, mira, curiosity, rase, frame) to the
already-tested pieces that already exist in this repo, reusing rather than
reimplementing them:

  mindai_substrate_bridge.WorldSubstrateSession   sentence -> RBNetwork + Substrate,
                                                    one RBNode per compiled WorldObject
  mindai_substrate_bridge.tick / node_to_coherence_source / feed_terrain_back
  world_live.LiveWorld                            feedable evolving population

Turn-based, not real-time: a full step runs the real genetic/kaleidoscope
population and the real terrain substrate, neither of which is a sub-100ms
operation (unlike GRV2's original video-game framerate target). Type an
action, wait a beat, see the result -- closer to an illustrated text
adventure than a live-rendered game.

Naming note: both world_compiler.py (root) and
wc-substrate/world_compiler_core.py define `class WorldCompiler`. This module
needs the root one (text -> evolved motifs); it's imported here as
`myth_engine` so the two are never ambiguous at the point of use.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import mindai_substrate_bridge as msb          # noqa: E402
import world_compiler as myth_engine           # noqa: E402  (root WorldCompiler: text -> evolved motifs)
import world_live as wl                        # noqa: E402

from . import curiosity as cur_mod             # noqa: E402
from . import frame as frame_mod               # noqa: E402
from . import mira as mira_mod                 # noqa: E402
from . import rase as rase_mod                 # noqa: E402
from . import sgr as sgr_mod                   # noqa: E402
from . import texture as texture_mod           # noqa: E402
from . import wiring as wiring_mod             # noqa: E402

_PLAYER_ID = "player"

# The dictionary WiringBank persists to grows permanently across process
# restarts (see wiring_store.py) -- default it to a real on-disk location
# next to cesium_output so `python -m grv2_runtime.server` gets a real,
# growing dictionary for free. Tests pass wiring_dictionary_path=None
# explicitly to stay fully in-memory/ephemeral.
_DEFAULT_WIRING_DICTIONARY = os.path.join(_REPO_ROOT, "wiring_dictionary")

_FLY_RE = re.compile(r"\bfly\b|\bflight\b", re.I)
_RESURRECT_RE = re.compile(r"resurrect|bring back", re.I)
_TRANSFORM_RE = re.compile(r"(transform|become)\b.*(yeti|wolf|bear)|(yeti|wolf|bear)\b.*(transform|become)", re.I)


@dataclass
class TurnResult:
    narrative: str
    reward: float
    foe_mode: bool
    metrics: mira_mod.RewardMetrics
    sgr_root: str
    png: bytes


class Runtime:
    """One playable session: a compiled scene, a live evolving population, a
    deterministic entity record, and the MIRA/Curiosity/RASE narrative layer
    on top of it."""

    def __init__(self, seed_sentence: str, world_seed: int = 0xACE5,
                 mira_cfg: Optional[mira_mod.MIRAConfig] = None,
                 curiosity_cfg: Optional[cur_mod.CuriosityConfig] = None,
                 allow_llm_wiring: bool = True,
                 wiring_dictionary_path: Optional[str] = _DEFAULT_WIRING_DICTIONARY) -> None:
        self.session = msb.WorldSubstrateSession(seed_sentence)
        self.kernel = sgr_mod.SceneGraphKernel(world_seed=world_seed)
        self.kernel.register_default_rules()
        self.mira = mira_mod.MIRA(mira_cfg or mira_mod.MIRAConfig())
        self.curiosity = cur_mod.CuriosityModule(curiosity_cfg or cur_mod.CuriosityConfig())
        self.npc_registry = rase_mod.NPCMemoryRegistry()
        self.metrics = mira_mod.RewardMetrics()
        self.master_prompt = ""
        self.tick = 0

        # Live evolving population, fed by every player action. Cheap: feed()
        # encodes new text through the manifold already fit at seed time --
        # it does not re-fit the manifold or run the genetic economy itself.
        self.live_world = wl.LiveWorld(myth_engine.CompilerConfig(seed=world_seed & 0xFFFFFFFF))
        try:
            self.live_world.seed(seed_sentence)
        except myth_engine.WorldCompilerError:
            pass  # seed sentence too short/degenerate for the genetic engine; terrain-side compile still ran

        # Wiring (structure, looked up once per word, never reshaped) and
        # texture (skin, real-image-grounded, re-rolled per invocation) --
        # kept as separate dicts, not fields on Entity, so the wiring/skin
        # split is real in code, not just in description.
        self.wiring_bank = wiring_mod.WiringBank(allow_llm=allow_llm_wiring,
                                                 dictionary_path=wiring_dictionary_path)
        self._texture_by_entity: Dict[str, texture_mod.TextureEntry] = {}
        # Metabolic motion channel: a smoothed resonance per entity that
        # decays and re-lerps toward the entity's live RBNode bridge
        # strength, so it visibly breathes/crystallizes/fades instead of
        # snapping every tick (same math as brain_engine.SceneAssembler's
        # update_metabolism). Kinematic (skeleton) motion is out of scope
        # for this pass -- see grv2_runtime/wiring.py's module docstring.
        self._resonance_by_entity: Dict[str, float] = {}

        self._rbnode_uid_by_entity: Dict[str, int] = {}
        self._register_compiled_objects()
        self._register_player()

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _register_compiled_objects(self) -> None:
        cube = self.session.network.cubes[0]
        for obj, uid in zip(self.session.scene.objects, self.session.object_uids):
            node = cube.nodes[uid]
            delta = sgr_mod.EntityDelta(
                id=obj.id, is_new_entity=True, new_type="motif",
                new_position=(node.world_x, 0.0, node.world_z),
                prop_updates={"label": obj.label[:80], "shape": obj.shape, "mass": f"{obj.mass:.4f}"},
            )
            self.kernel.apply_delta(delta, self.tick)
            self._rbnode_uid_by_entity[obj.id] = uid
            self._ensure_wiring_and_texture(obj.id, obj.label)

    def _register_player(self) -> None:
        cube = self.session.network.cubes[0]
        uid = cube.add_node("player", "player")
        node = cube.nodes[uid]
        ox, oz = self.session.scene_origin
        node.world_x, node.world_z = ox, oz
        delta = sgr_mod.EntityDelta(
            id=_PLAYER_ID, is_new_entity=True, new_type="player",
            new_position=(ox, 0.0, oz),
            prop_updates={"attention": "0", "form": "human", "exhausted": "0", "label": "person"},
        )
        self.kernel.apply_delta(delta, self.tick)
        self._rbnode_uid_by_entity[_PLAYER_ID] = uid
        self._ensure_wiring_and_texture(_PLAYER_ID, "person")

        npc = self.npc_registry.get_or_create("yeti_original")
        npc.set_trait("hostile", 0.9)
        npc.set_trait("territorial", 0.8)
        npc.set_lore("stripe_taboo", "black_white_stripes_mean_liars_who_skin")

    def _ensure_wiring_and_texture(self, entity_id: str, word: str) -> None:
        """Look up (and cache) this entity's wiring, and roll a fresh texture
        for this invocation. Wiring is a pure O(1) bank lookup (same word ->
        identical points, forever). Texture is a real image search re-run
        every time an entity is newly registered -- this is where "grounding
        and variety, from one mechanism" actually happens; it never touches
        wiring.points."""
        self.wiring_bank.recall(word)   # warms/creates the bank entry; result read via geo_export
        self._texture_by_entity[entity_id] = texture_mod.texture_for(
            word, context=self.master_prompt)

    def set_master_prompt(self, prompt: str) -> None:
        self.master_prompt = prompt

    # ------------------------------------------------------------------
    # the turn
    # ------------------------------------------------------------------

    def step(self, action: str) -> TurnResult:
        self.tick += 1
        action = action.strip()

        for delta in self._intent_to_deltas(action):
            self.kernel.apply_delta(delta, self.tick)

        if self.live_world.seeded and len(action) >= 4:
            try:
                self.live_world.feed(action)
            except myth_engine.WorldCompilerError:
                pass

        known_ids = {e.id for e in self.kernel.get_all_entities()}
        sgr_entities = self.kernel.get_all_entities()
        live_motifs = self._live_world_candidates(known_ids) if self.live_world.seeded else []
        candidates = self.curiosity.sample_negative_space(
            known_ids, sgr_entities, self.metrics, self.kernel.world_seed,
            master_prompt=self.master_prompt or None, extra_candidates=live_motifs)
        accepted = self.kernel.validate_and_apply_curiosity(candidates, self.tick)
        for ent in accepted:
            self._spawn_rbnode_for_entity(ent)

        # Real geometric cost of whatever was just materialized this turn
        # (0.0 if nothing new was spawned) -- see wiring.WiringEntry.thermal_cost
        # and thermal.py. Already-cached lookups, so this is cheap.
        measured_thermal_cost = 0.0
        if accepted:
            costs = [self.wiring_bank.recall(ent.properties.get("label", ent.type)).thermal_cost
                    for ent in accepted]
            measured_thermal_cost = sum(costs) / len(costs)

        player_xz = self._player_xz()
        detail_before = self.session.substrate.detail_near(*player_xz, min_lod=3, radius=220.0)
        msb.tick(self.session.network, self.session.substrate, eye=player_xz)
        detail_after = self.session.substrate.detail_near(*player_xz, min_lod=3, radius=220.0)
        terrain_delta = detail_after - detail_before

        cube = self.session.network.cubes[0]
        measured_coherence = 1.0 - min(1.0, cube.instability() / cube.SPLIT_THRESH)
        measured_surprise = min(1.0, abs(terrain_delta) / 20.0)
        turn = self.mira.evaluate(action, self.metrics, measured_coherence, measured_surprise,
                                  measured_thermal_cost)
        self.metrics = turn.new_metrics

        narrative = self._build_narrative(action, turn, accepted, terrain_delta)
        self._record_npc_memory(action, turn, measured_surprise, measured_coherence)

        bridges = {eid: cube.nodes[uid].rstate.er_bridge_strength()
                  for eid, uid in self._rbnode_uid_by_entity.items() if uid in cube.nodes}
        self._update_resonance(bridges)

        png = frame_mod.render_frame(self.session.substrate, self.kernel.get_all_entities(),
                                     bridge_by_id=bridges, center=player_xz,
                                     foe_mode=turn.foe_mode_active)

        return TurnResult(narrative=narrative, reward=turn.reward, foe_mode=turn.foe_mode_active,
                          metrics=self.metrics, sgr_root=self.kernel.get_merkle_root(), png=png)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _player_xz(self) -> Tuple[float, float]:
        node = self._rbnode_for(_PLAYER_ID)
        if node is None:
            return self.session.scene_origin
        return (node.world_x, node.world_z)

    def _rbnode_for(self, entity_id: str):
        uid = self._rbnode_uid_by_entity.get(entity_id)
        if uid is None:
            return None
        return self.session.network.cubes[0].nodes.get(uid)

    def _spawn_rbnode_for_entity(self, ent: sgr_mod.Entity) -> None:
        if ent.id in self._rbnode_uid_by_entity:
            return
        cube = self.session.network.cubes[0]
        uid = cube.add_node(ent.type, ent.id)
        node = cube.nodes[uid]
        node.world_x, node.world_z = ent.position[0], ent.position[2]
        node.world_obj = ent
        self._rbnode_uid_by_entity[ent.id] = uid
        word = ent.properties.get("label", ent.type)
        self._ensure_wiring_and_texture(ent.id, word)

    def _update_resonance(self, bridges: Dict[str, float]) -> None:
        """Metabolic motion channel: global decay, then lerp toward each
        entity's live RBNode bridge strength. Ported from
        brain_engine.SceneAssembler.update_metabolism's math (decay 0.92,
        lerp 0.15-0.35/tick) -- this is what makes a concept visibly
        crystallize or fade instead of snapping between states every tick."""
        for eid in list(self._resonance_by_entity):
            self._resonance_by_entity[eid] *= 0.92
        for eid, target in bridges.items():
            current = self._resonance_by_entity.get(eid, 0.0)
            self._resonance_by_entity[eid] = current + (target - current) * 0.25

    def _live_world_candidates(self, known_ids) -> List[sgr_mod.Entity]:
        scene = self.live_world.scene(purify_passes=3)
        out: List[sgr_mod.Entity] = []
        for obj in sorted(scene.objects, key=lambda o: -o.mass)[:4]:
            digest = hashlib.blake2b(obj.label.encode(), digest_size=4).hexdigest()
            eid = f"motif_echo_{digest}"
            if eid in known_ids:
                continue
            wx = float(obj.position[0]) * msb.OBJECT_TO_WORLD_SCALE + self.session.scene_origin[0]
            wz = float(obj.position[2]) * msb.OBJECT_TO_WORLD_SCALE + self.session.scene_origin[1]
            out.append(sgr_mod.Entity(
                id=eid, type="motif_echo", position=(wx, 0.0, wz),
                properties={"label": obj.label[:80], "implication": "world_remembers",
                           "mass": f"{obj.mass:.4f}"}))
        return out

    def _intent_to_deltas(self, action: str) -> List[sgr_mod.EntityDelta]:
        deltas: List[sgr_mod.EntityDelta] = []
        if _RESURRECT_RE.search(action):
            deltas.append(sgr_mod.EntityDelta(id=_PLAYER_ID, prop_updates={
                "resurrection_target": "unknown", "magic_effect": "resurrection", "attention": "6"}))
        if _FLY_RE.search(action):
            deltas.append(sgr_mod.EntityDelta(id=_PLAYER_ID, prop_updates={
                "magic_effect": "flight", "attention": "3", "form": "airborne"}))
        if _TRANSFORM_RE.search(action):
            deltas.append(sgr_mod.EntityDelta(id=_PLAYER_ID, prop_updates={
                "magic_effect": "transformation", "attention": "5", "form": "yeti_large"}))
        if mira_mod.MIRA.is_duality_collapse_attempt(action):
            deltas.append(sgr_mod.EntityDelta(id=_PLAYER_ID, prop_updates={"game_state": "won"}))
        deltas.append(sgr_mod.EntityDelta(id=_PLAYER_ID, prop_updates={"last_action": action[:80]}))
        return deltas

    def _build_narrative(self, action: str, turn: mira_mod.MIRATurnResult,
                         accepted: List[sgr_mod.Entity], terrain_delta: int) -> str:
        lines = [f"SGR:0x{self.kernel.get_merkle_root()[:12]}  tick:{self.tick}  R:{turn.reward:.3f}"]
        if turn.foe_mode_active:
            lines[-1] += "  [FOE]"
        if turn.disposition == mira_mod.ActionDisposition.COMPLIANT_DEFIANCE:
            lines.append(f"[COMPLIANT DEFIANCE] {turn.co_author_comment}")
        elif turn.co_author_comment:
            lines.append(turn.co_author_comment)
        if accepted:
            lines.append("[Curiosity ¬S]")
            for ent in accepted:
                impl = ent.properties.get("implication", ent.type)
                lines.append(f"  + {ent.id} [{impl}]")
        if terrain_delta:
            sign = "+" if terrain_delta >= 0 else ""
            lines.append(f"The ground itself answers: {sign}{terrain_delta} deep chunks near you.")
        npc = self.npc_registry.get("yeti_original")
        if npc is not None and self._npc_engaged(action):
            lines.append(npc.assemble_context(action))
        return "\n".join(lines)

    _NPC_MENTION_KEYWORDS = ("yeti", "creature", "beast")

    def _npc_engaged(self, action: str) -> bool:
        low = action.lower()
        return any(k in low for k in self._NPC_MENTION_KEYWORDS)

    def _record_npc_memory(self, action: str, turn: mira_mod.MIRATurnResult,
                           measured_surprise: float, measured_coherence: float) -> None:
        """The other half of npc.assemble_context (read in _build_narrative,
        using memory as of *before* this turn): persist this turn as an
        episodic memory, and nudge favorability/trust from the turn's real
        measured signals -- same "measured, not string-guessed" discipline
        mira.py itself uses for coherence/surprise -- so a yeti that's been
        provoked into foe mode repeatedly actually trusts the player less
        next time, instead of restating the same init-time traits forever."""
        npc = self.npc_registry.get("yeti_original")
        if npc is None or not self._npc_engaged(action):
            return
        weight = max(0.0, min(1.0, 0.35 + 0.4 * measured_surprise +
                              (0.25 if turn.foe_mode_active else 0.0)))
        npc.record_event(
            event=f"{action[:60]!r} -> {turn.disposition.name} reward={turn.reward:.2f}",
            player_action=action[:120], weight=weight, tick=self.tick)
        npc.update_favorability(0.05 * (measured_coherence - 0.5))
        npc.update_trust(-0.1 if turn.foe_mode_active else 0.02)

    # ------------------------------------------------------------------
    # snapshot for the server
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "tick": self.tick,
            "sgr_root": self.kernel.get_merkle_root(),
            "entities": self.kernel.to_dict()["entities"],
            "metrics": dataclasses.asdict(self.metrics),
            "foe_mode": self.mira.foe_mode,
        }
