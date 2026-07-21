"""grv2_runtime/rase.py — per-NPC memory (ports mira_rase.hpp/.cpp's RASE half).

Three layers per NPC, exactly as in the reference design:
  episodic   — a bounded recent-events deque, compressed into a semantic gist
               once it overflows
  semantic   — hashed-trigram embeddings of those compressed gists, retrieved
               by cosine similarity
  procedural — traits/favorability/trust/lore, the "personality" a narrative
               layer reads from directly

The embedding scheme is the same hashed-trigram idea organic_ai_core.load_text
already uses for deterministic, vocabulary-free text embedding -- reused here
rather than inventing a second one.
"""
from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np

_EMBED_DIM = 64


@dataclass
class EpisodicMemory:
    turn_id: str
    who: str
    event: str
    player_action: str
    emotional_weight: float = 0.5
    tick: int = 0


@dataclass
class SemanticMemory:
    npc_id: str
    gist: str
    significance: float = 0.5
    embedding: np.ndarray = field(default_factory=lambda: np.zeros(_EMBED_DIM))


@dataclass
class ProceduralMemory:
    npc_id: str
    traits: Dict[str, float] = field(default_factory=dict)
    favorability: float = 0.0
    trust: float = 0.5
    lore: Dict[str, str] = field(default_factory=dict)


def _text_to_embedding(text: str) -> np.ndarray:
    vec = np.zeros(_EMBED_DIM)
    for i in range(len(text) - 2):
        d = hashlib.blake2b(text[i:i + 3].encode("utf-8"), digest_size=8).digest()
        vec[int.from_bytes(d[:4], "big") % _EMBED_DIM] += 1.0
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-8 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denom) if denom > 1e-8 else 0.0


class RASEMemory:
    EPISODIC_WINDOW = 50
    EPISODIC_COMPRESS = 40

    def __init__(self, npc_id: str) -> None:
        self.npc_id = npc_id
        self.episodic: Deque[EpisodicMemory] = deque()
        self.semantic: List[SemanticMemory] = []
        self.procedural = ProceduralMemory(npc_id=npc_id)

    def record_event(self, event: str, player_action: str, weight: float, tick: int) -> None:
        self.episodic.append(EpisodicMemory(
            turn_id=f"turn_{len(self.episodic)}", who=self.npc_id, event=event,
            player_action=player_action, emotional_weight=weight, tick=tick))
        self._maybe_compress()

    def recall_recent(self, n: int = 10) -> List[EpisodicMemory]:
        return list(self.episodic)[-n:][::-1]

    def _maybe_compress(self) -> None:
        if len(self.episodic) <= self.EPISODIC_WINDOW:
            return
        batch = [self.episodic.popleft() for _ in range(self.EPISODIC_COMPRESS)]
        self.compress_to_semantic(batch)

    def compress_to_semantic(self, batch: List[EpisodicMemory]) -> None:
        if not batch:
            return
        gist = ". ".join(m.event for m in batch)
        significance = float(np.clip(sum(m.emotional_weight for m in batch) / len(batch), 0.0, 1.0))
        self.semantic.append(SemanticMemory(
            npc_id=self.npc_id, gist=gist, significance=significance,
            embedding=_text_to_embedding(gist)))

    def recall_semantic(self, query: np.ndarray, k: int = 3) -> List[SemanticMemory]:
        scored = sorted(self.semantic, key=lambda s: -_cosine(query, s.embedding))
        return scored[:k]

    def set_trait(self, name: str, value: float) -> None:
        self.procedural.traits[name] = float(np.clip(value, 0.0, 1.0))

    def get_trait(self, name: str, default: float = 0.5) -> float:
        return self.procedural.traits.get(name, default)

    def update_favorability(self, delta: float) -> None:
        self.procedural.favorability = float(np.clip(self.procedural.favorability + delta, -1.0, 1.0))

    def update_trust(self, delta: float) -> None:
        self.procedural.trust = float(np.clip(self.procedural.trust + delta, 0.0, 1.0))

    def set_lore(self, key: str, value: str) -> None:
        self.procedural.lore[key] = value

    @property
    def favorability(self) -> float:
        return self.procedural.favorability

    @property
    def trust(self) -> float:
        return self.procedural.trust

    def assemble_context(self, player_action: str, n: int = 5) -> str:
        lines = [f"NPC:{self.npc_id} fav:{self.procedural.favorability:.2f} trust:{self.procedural.trust:.2f}"]
        recent = self.recall_recent(n)
        if recent:
            lines.append("Recent:" + " ".join(m.event for m in recent))
        if self.semantic:
            lines.append(f"Gist:{self.semantic[-1].gist}")
        lines.append(f"Now:{player_action}")
        return "\n".join(lines)


class NPCMemoryRegistry:
    def __init__(self) -> None:
        self._memories: Dict[str, RASEMemory] = {}

    def get_or_create(self, npc_id: str) -> RASEMemory:
        if npc_id not in self._memories:
            self._memories[npc_id] = RASEMemory(npc_id)
        return self._memories[npc_id]

    def get(self, npc_id: str) -> "RASEMemory | None":
        return self._memories.get(npc_id)

    @property
    def npc_count(self) -> int:
        return len(self._memories)
