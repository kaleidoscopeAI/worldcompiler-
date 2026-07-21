"""grv2_runtime — the merged narrative/terrain/relational runtime.

Ports the logic tiers of the GRV2 reference design (a supplied C++/Unity
prototype: deterministic entity kernel + duality co-author + negative-space
curiosity sampler + NPC memory) to Python, and wires them directly to the
already-tested pieces already in this repo instead of reimplementing them:

  mindai_substrate_bridge.WorldSubstrateSession   sentence -> RBNetwork + Substrate
  mindai_substrate_bridge.tick / node_to_coherence_source / feed_terrain_back
  world_live.LiveWorld                            feedable evolving population

See grv2_runtime.runtime.Runtime for the per-turn loop.
"""
from .runtime import Runtime, TurnResult

__all__ = ["Runtime", "TurnResult"]
