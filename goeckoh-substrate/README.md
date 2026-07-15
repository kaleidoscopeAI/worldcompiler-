# Goeckoh-Coupled Planetary Substrate

The terrain layer the World Compiler was missing — and the one thing that makes
"as good as Google Earth, with character" a precise claim instead of a vibe.

## What this is

Every primitive the World Compiler spawns (fire, fauna, flora) carries distilled
motion and stays persistent through the Goeckoh lattice. They sit on terrain. In
the existing codebase that terrain was a TODO: scattered clipmap structs with no
generator and no continuity guarantee. This is the missing layer — a deterministic
planetary heightfield whose **level of detail is driven by the Goeckoh life-drive
itself.**

The novel idea, stated as one equation:

```
                    ⎡       focal · spacing₀ · (κ_floor + κ·slope)  ⎤
LOD(x) = ⌈ log₂  ⎢  ───────────────────────────────────────────  ⎥ ⌉
                    ⎣            dist(x, eye) · ( τ / (1 + β·GCL) )    ⎦
```

Terrain subdivides where **screen-space geometric error** exceeds a pixel budget —
and that budget is *sharpened* in regions of high Goeckoh coherence GCL(x). The
same scalar that keeps a wolf persistent now decides where the planet earns its
detail. Geometry and life-drive are one operator chain, not two subsystems.

Most engines allocate detail by distance-to-camera alone. That is exactly why
their terrain reads as bland: detail goes where the camera is, never where the
world is *interesting*. Here, a coastline the user lingered on, a ridge a living
creature anchored — these pull resolution toward themselves. That is the
"character" the brief demanded, made mechanical.

## Verified claims (all measured, none asserted)

Run `make test`. The gates prove:

| Claim | Test | Result |
|---|---|---|
| Scalar path ≡ AVX2 path | T1 | **bit-identical** over 100k samples (max\|Δ\|=0) |
| Same seed ⇒ same planet | T2 | stable; seed+1 differs by 9.97m |
| GCL allocates detail | T3 | dead→LOD0, alive→LOD4 at 2km: **256× density lift** |
| Chunk gen is finite & sane | T4 | 4096 verts, bounded height range |
| **Delta reuse** | binding A | identical frame ⇒ **0 regenerated**; 40m move ⇒ 34% |
| **Life-following detail** | binding B | GCL hotspot ⇒ **174% more deep chunks** vs dead terrain |

Throughput (single core, AVX2+FMA, scales with `OMP_NUM_THREADS`):
**16.8 M vertices/sec, 4112 chunks/sec.**

## Design defense

**Why a flat chart, not a spheroid.** Curvature over a session's reach is
sub-pixel. A flat chart gives O(1) integer tile addressing and an exactly periodic
noise basis — both of which the determinism contract requires. Curvature, when
wanted, is a vertex-shader post-transform; it must not live in the field generator
where it would couple height to position non-locally and break the pure-function
guarantee. Flat where it must be exact, curved where it's only cosmetic.

**Why bit-identical scalar/SIMD instead of "within tolerance."** The Δ-SIREN
renderer this feeds relies on delta-evaluation between adjacent points. If the two
height paths disagreed even slightly, that delta would be unsound. The noise basis
uses only multiply/add/bit-ops — no transcendental inside the lattice — precisely
so the scalar reference and the 8-wide AVX2 path cannot diverge on rounding. They
produce identical bits (verified), which is the strongest possible form of the
determinism claim.

**Why slope comes from the grid, not fresh samples.** The first chunk generator
sampled fBm three times per vertex (height + two finite-difference neighbors).
That tripled the cost of a gradient that's *already present* in the height
samples. The current generator samples a padded grid once and derives slope by
central difference across it — `(N+2)²` evaluations instead of `3N²`. Measured
result: **6.76 → 16.8 M verts/sec, a 2.46× gain.** This is the Δ-SIREN principle
("never recompute information you already have") applied to terrain.

**Why the split rule is distance-vs-node-size, not the raw oracle.** The
screen-space oracle correctly chooses a *leaf's* resolution, but driving the
quadtree recursion directly with it made every near node split to the cap (851
chunks for a 256m view). The standard quadtree metric — split only when
`dist < factor · node_size` — bounds the resident set to a detail *ring*
(O(factor²) per level). GCL and slope *widen* that ring where the world is
interesting, which is how the life-drive couples in without forcing max LOD
everywhere.

## Self-critique (the weaknesses, named before they bite)

1. **The 40m-move regenerates 34% of chunks.** When the eye moves, near chunks
   cross LOD boundaries and their keys change, forcing regeneration. 34% is
   acceptable but not great. The fix is *geomorphing* — interpolate a chunk's
   vertices toward the next LOD instead of hard-swapping — which also kills the
   visible pop. Not yet implemented.

2. **Slope is sampled at the node center for the split decision.** A node whose
   *center* is flat but whose *edge* is a cliff could under-split. The GCL channel
   already uses max-over-footprint to avoid exactly this; slope should too. One-line
   change, deferred.

3. **The ridged-fBm gradient is finite-difference, not analytic.** Analytic
   gradient of domain-warped ridged fBm is genuinely painful (the `(1-|n|)²` and
   the warp make the chain rule ugly), so slope is numerical. For shading normals
   the diagnostic renderer recomputes them; a production path would want the
   analytic Jacobian or a stored normal per vertex.

4. **No true planetary streaming from disk.** The resident set is in-memory with
   LRU eviction. A real planet needs out-of-core paging (the chunk key is already
   a stable content hash, so this is a clean extension — but it's an extension).

5. **`focal_px` and `tau` are hard-coded to a screen assumption.** They belong in
   `LODParams`, keyed to actual viewport size and FOV. Trivial, not done.

## Files

```
src/substrate.h          — types, the LOD oracle contract, determinism contract
src/substrate.c          — deterministic AVX2 fBm, oracle, GCL-coupled chunk gen
src/substrate_stream.h   — the binding layer: GCL-driven quadtree streaming
src/substrate_stream.c   — refinement, chunk reuse by key, LRU eviction
src/test_substrate.c     — the correctness gate (T1–T4 + benchmark)
src/test_binding.c       — proves delta-reuse and life-following
src/render_proof.c       — diagnostic raymarcher (visual proof, not production)
```

## Integration with GRV2 / Goeckoh

The substrate does **not** own the GCL field. The GRV2 runtime's Goeckoh lattice
owns it (it's the lattice's coherence output). The substrate reads it through the
`GCLField` view — one source of truth, shared, not duplicated. Wire the lattice's
coherence grid into a `GCLField` each frame and call `stream_update`; the planet's
detail will track wherever the life-drive concentrates.

## Build

```bash
make            # builds everything
make test       # runs the full gate + binding proof
make proof      # renders proof.png (needs PIL for PNG conversion)
```

---

## Integration layer: entities drive terrain detail

`entity_coherence.{h,c}` is the bridge between the GRV2 runtime's living entities
and the terrain substrate. The data flow is one direction, no shared mutable state:

```
entities (world position + intensity)
     │  coherence_stamp_all  (Gaussian stamp per entity)
     ▼
CoherenceRaster  ──coherence_as_gcl_field──▶  GCLField
     │
     ▼
stream_update  (substrate quadtree, GCL-sharpened screen-space LOD)
     ▼
resident chunks: detail concentrated where the entities are
```

A wolf standing on a hillside raises coherence there; the substrate's dual-split
oracle earns that hillside extra detail — because the world is *alive* there, not
merely because the camera is near. When the wolf moves, the detail follows.

**Verified (test_integration, slope held constant so geometry can't confound it):**
- An entity adds **+53 deep chunks** of terrain detail at its location.
- Moving the entity moves **+32 chunks** of detail to the new position.

**The split oracle has two independent reasons to subdivide:**
1. *Eye-centered ring* — standard quadtree LOD, bounds chunk count for ordinary terrain.
2. *Life-driven screen-space error* — where coherence is high, split by the
   GCL-sharpened pixel-error budget directly, regardless of eye distance, so detail
   reaches entities far from the camera (a creature 400m away gets detail the
   eye-ring alone cannot deliver). The lift is bounded (+3 LOD × coherence) so a
   large coherent region cannot blow the chunk budget.

This strengthened the GCL coupling from **174% → 933% more detail** under a
coherence hotspot versus dead terrain, because detail is now driven by true
screen-space error, not ring-widening.

### What this is NOT
This integration deliberately does not use the "hyperdimensional binding" /
resonance-scalar approach from some prior drafts. An entity's influence on terrain
is a spatial falloff at its position — a real, measurable, debuggable operation.
HD vectors, XOR "binding", and a resonance scalar that measures its own similarity
add no information the position doesn't already carry. The binding happens through
SPACE, which is the dimension terrain actually lives in.
