# World Compiler

Turn words into worlds. World Compiler takes text — from an AI chat, a
document, anything — and renders it as an explorable 3D world, using a
deterministic evolutionary/compression engine instead of a trained model.

```
python3 world_compiler.py mydoc.txt -o world.html --gates
```

Open `world.html` in a browser: drag to orbit, scroll to zoom, click a shape
to see the exact snippet of your text it crystallized from.

## How it works

Nothing here is a black box, and nothing is trained. The whole pipeline is
five small, independently-gated modules composed end to end:

1. **Ingest** (`organic_ai_core.py`) — text is sliced into overlapping
   windows and embedded with hashed-trigram features. No vocabulary, no
   training: the same text embeds identically on any machine, forever.

2. **Live** (`genetic_manifold.py`) — every window becomes a node whose DNA
   *is* its embedding. The population evolves under a closed energy economy:
   nodes holding a rare, uncovered view of their symmetry orbit thrive;
   redundant duplicates starve. This is natural selection over your text's
   own motifs, not fitting to a target.

3. **Purify** (`refinement_loop.py`) — the evolved DNA is pulled toward the
   canonical (symmetry-invariant) view of its kaleidoscope orbit, repeatedly,
   until it stops moving. This is a proven contraction: it provably reaches a
   fixed point rather than drifting forever. Only then do **supernodes**
   crystallize — one per settled semantic motif.

4. **Project** (`world_compiler.py`) — each crystallized motif becomes one
   object in the 3D scene. Position comes from its dominant latent axes;
   facet count comes from how many latent axes its symmetry invariant
   actually spans (a participation-ratio measure); color is a stable hash of
   its DNA; size is how much of the text it represents. A motif your text
   repeats becomes large and simple; a motif it mentions once becomes small
   and many-faceted. Every visual property is a deterministic function of a
   quantity the engine already computed — nothing is decorative.

5. **Render** (`world_render.py`) — a self-contained, dependency-free canvas
   3D viewer (no WebGL, no CDN) renders the scene with a hand-rolled
   perspective camera and flat shading.

Underneath, `world_compiler.py` sits on `kaleidoscope_core.py` (the manifold
+ MDL compression organism) and `kaleidoscope_group.py` (the symmetry group
of turns and mirrors that defines what "the same motif" means) — both used
internally by stages 2 and 3.

Same text + same seed compiles to a byte-identical world, anywhere
(`gate_determinism`). Different text compiles to a different world
(`gate_diverges`) — those two gates exist so "a faithful rendering of your
text's structure" is a falsifiable claim, not an assertion.

## Files

| file | role |
|---|---|
| `organic_ai_core.py` | deterministic RNG, universal data ingestion, evolvable-plasticity predictive nets, the closed energy economy |
| `kaleidoscope_core.py` | membrane + MDL manifold + Mirror: the compression organism (`X -> S*`) |
| `kaleidoscope_group.py` | the group of turns/mirrors; orbits, invariants, strains as symmetry classes |
| `genetic_manifold.py` | "the data IS the DNA" — a population that evolves ingested data itself |
| `refinement_loop.py` | raw DNA -> pure inherited DNA -> supernode crystallization |
| `world_compiler.py` | the product: composes the five modules, maps supernodes to a 3D scene |
| `world_render.py` | the self-contained HTML/canvas 3D viewer |

Every engine module (`organic_ai_core.py` through `refinement_loop.py`) has
its own `_demo()`/`main()` with console-printed verification gates — run any
of them directly (`python3 kaleidoscope_core.py`) to see it prove its own
claims before `world_compiler.py` ever composes them.

Runtime: numpy + stdlib only. No network, no training, no pickle/exec.
