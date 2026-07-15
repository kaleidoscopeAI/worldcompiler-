# World Compiler × Planetary Substrate

Binds `world_compiler_core.py` (language → scene graph of SDF objects) to the
verified planetary substrate (deterministic terrain whose detail is driven by a
coherence field). A sentence compiles to objects; the objects make the terrain
resolve around them.

```
"the wolf sits on the hill"
      │  WorldCompiler.compile_sentence + resolve
      ▼
  SceneGraph: objects at distinct centroids
      │  scene_to_coherence_sources  (centroid → planet (x,z,intensity,radius))
      ▼
  substrate_capi.so: stamp coherence + stream  (GCL-driven LOD)
      ▼
  terrain earns detail under each compiled object
```

## Verified (test_wc_substrate.py, all measured)

| Claim | Result |
|---|---|
| C1 — bbox fix gives objects distinct positions | PASS (the "on" predicate now actually moves objects) |
| C2 — terrain resolves under compiled objects | PASS (**+586** and **+222** deep chunks vs bare terrain) |
| C3 — richer scene drives more planet | PASS (1 object → 512 chunks; 5 objects → 920) |

The separated-placement demo (render_separated.py) shows three objects each
earning **+32 to +40** deep chunks at distinct terrain locations — see
`scene_separated.png`.

## Two real bugs this integration found and fixed in the compiler

The wiring is only meaningful if compiled objects occupy distinct positions.
They didn't, because of two coupled defects in `world_compiler_core.py`:

1. **`SceneNode.bbox` was never computed.** It defaults to `None` and nothing
   sets it. Every spatial-predicate resolver (`_resolve_above`, `_resolve_beside`,
   …) guards on `if node_b.bbox is None: return`, so `"on"`/`"above"`/`"beside"`
   silently did nothing and **all objects stayed at the origin**. The bridge
   computes each node's bbox from its SDF surface after canonicalization —
   exactly the data the resolvers needed — then re-runs `resolve()`. Now objects
   actually separate.

2. **Predicates reported success on a no-op.** `Predicate.resolve` marked itself
   resolved even when the resolver early-returned (bbox `None`). With the bbox
   populated this is no longer masked.

We did **not** edit `world_compiler_core.py` — it's an uploaded artifact. The
fix lives in `wc_substrate_bridge.py` (`compute_bbox_from_sdf`), applied as a
wrapper.

## Honest limitations (found, not hidden)

These are real and worth knowing; none break the coupling itself:

- **The classifier treats some verbs as objects.** "sits" in "the wolf sits on
  the hill" gets its own SDF and centroid and becomes a coherence source,
  instead of being the spatial *relation* between wolf and hill. The integration
  works regardless (the verb-object just adds coherence), but it's a
  word-classification gap in the compiler, not the bridge.
- **Spatial layout is coarse.** The `"on"` resolver only translates in z; objects
  in a single sentence don't separate in the x-plane. So `"wolf and bear and
  tree"` stacks all three at one (x,z) — visible in `scene_on_planet.png` as a
  single bloom. To place objects at genuinely distinct locations,
  `render_separated.py` compiles them as separate scenes at explicit origins.
  Fixing in-sentence horizontal layout is a compiler change (a real spatial
  solver, not just z-stacking).
- **Distant-object detail caps at LOD 3.** By design — the substrate's bounded
  LOD lift (verified last session) tops out a few levels over the eye-ring, so an
  object 400m from the eye earns LOD-3 detail, not LOD-5. Tested accordingly.

## Files

```
world_compiler_core.py     — the uploaded compiler (unmodified)
substrate.{c,h}            — deterministic terrain field + LOD oracle
substrate_stream.{c,h}     — GCL-driven quadtree streaming
entity_coherence.{c,h}     — objects → coherence field
substrate_capi.c           — C ABI over the substrate for ctypes
wc_substrate_bridge.py     — the binding: compiler → coherence sources (+ bbox fix)
test_wc_substrate.py       — the integration gate (C1–C3)
render_scene.py            — top-down proof, one sentence
render_separated.py        — top-down proof, objects at distinct locations
```

## Build & run

```bash
gcc -O3 -march=native -mavx2 -mfma -ffp-contract=fast -fPIC -shared \
    substrate.c substrate_stream.c entity_coherence.c substrate_capi.c \
    -o substrate_capi.so -lm
python3 test_wc_substrate.py      # runs C1–C3
python3 render_separated.py       # writes scene_separated.png
```
