# World Compiler

Turn words into worlds. World Compiler takes text — from an AI chat, a
document, anything — and renders it as an explorable 3D world, using a
deterministic evolutionary/compression engine instead of a trained model.

Two ways to run it:

```
# batch: compile a document once to a static HTML world
python3 world_compiler.py mydoc.txt -o world.html --gates

# live: an organism you keep feeding, evolving in real time in your browser
python3 world_server.py --seed-file mydoc.txt
```

Open the result in a browser: drag to orbit, scroll to zoom, click a shape
to see the exact snippet of your text it crystallized from. In live mode
there's also a text box — paste in more text at any point and watch new
motifs get born and compete for a place in the world that's already there.

## How it works

Nothing here is a black box, and nothing is trained. The pipeline is six
small, independently-gated modules composed end to end:

1. **Ingest** (`organic_ai_core.py` + `semantic_embedding.py`) — text is
   sliced into overlapping windows and embedded on two channels: hashed
   character trigrams (orthographic — sees spelling, needs no vocabulary,
   survives typos/OOV) and word co-occurrence PPMI-SVD vectors fit fresh from
   the document itself (semantic — sees topic: two chunks about the same
   thing cluster even when they share almost no words). Both are ordinary
   linear algebra with no gradient descent and no pretrained weights; the
   same text embeds identically everywhere, forever.

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

6. **Live** (`world_live.py` + `world_server.py`) — the same population, never
   stopped. Instead of evolving for a fixed number of generations and
   rendering once, a `LiveWorld` keeps ticking in a background thread and
   accepts new text at any time via `feed()`: new chunks are embedded through
   the SAME manifold and semantic space the world was founded with (a world
   has a worldview, fixed at founding) and injected as new DNA that
   immediately has to compete for energy against everything already alive. A
   tiny stdlib HTTP server exposes this as `POST /seed`, `POST /feed`,
   `GET /state`; the browser polls a few times a second and animates the
   difference between snapshots — births fade in, deaths fade out.

Underneath, `world_compiler.py` sits on `kaleidoscope_core.py` (the manifold
+ MDL compression organism) and `kaleidoscope_group.py` (the symmetry group
of turns and mirrors that defines what "the same motif" means) — both used
internally by stages 2 and 3.

## What's provably true, not just claimed

- Same text + same seed compiles to a byte-identical world, anywhere
  (`gate_determinism`). Different text compiles to a different world
  (`gate_diverges`).
- Same-topic text clusters closer than different-topic text in the semantic
  channel, by a real, checked margin, even with near-zero shared vocabulary
  (`semantic_embedding.gate_semantic_clustering`).
- In live mode, total energy at any instant equals exactly the founding
  budget plus everything ever fed in — checked after every tick and every
  feed, no matter how long the world has been running
  (`LiveWorld.gate_conservation`).

## Files

| file | role |
|---|---|
| `organic_ai_core.py` | deterministic RNG, universal data ingestion, evolvable-plasticity predictive nets, the closed energy economy |
| `semantic_embedding.py` | distributional semantics (PPMI + SVD) fit fresh from each document — no pretrained model |
| `kaleidoscope_core.py` | membrane + MDL manifold + Mirror: the compression organism (`X -> S*`) |
| `kaleidoscope_group.py` | the group of turns/mirrors; orbits, invariants, strains as symmetry classes |
| `genetic_manifold.py` | "the data IS the DNA" — a population that evolves ingested data itself |
| `refinement_loop.py` | raw DNA -> pure inherited DNA -> supernode crystallization |
| `world_compiler.py` | the batch product: composes the engine, maps supernodes to a 3D scene |
| `world_render.py` | the self-contained HTML/canvas 3D viewer (batch and live variants) |
| `world_live.py` | `LiveWorld` — the same engine as a standing, feedable population |
| `world_server.py` | a stdlib HTTP server around `LiveWorld`: seed it, feed it, watch it |

Every engine module has its own `_demo()`/`main()` with console-printed
verification gates — run any of them directly (`python3 kaleidoscope_core.py`)
to see it prove its own claims before `world_compiler.py` ever composes them.

Runtime: numpy + stdlib only. No network dependency in the engine itself —
`world_server.py` is a thin local application layer on top of it, same as
any local dev server. No training, no pickle/exec.
