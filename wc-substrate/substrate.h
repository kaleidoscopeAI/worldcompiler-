/* substrate.h — Goeckoh-coupled planetary SDF substrate.
 *
 * WHY THIS FILE EXISTS
 * --------------------
 * The World Compiler spawns primitives (fire, fauna, flora) that carry distilled
 * motion fields and stay persistent via the Goeckoh lattice. They sit on terrain.
 * Every prior sketch tessellated that terrain by distance-to-camera — the same
 * thing every engine does, and the reason it reads as "bland CGI ground": detail
 * is allocated by proximity, not by where the world is actually interesting.
 *
 * This substrate inverts that. The Goeckoh Global Coherence Level GCL(x,t) — the
 * SAME scalar that keeps a wolf persistent — becomes the LOD oracle for the
 * planet. Detail flows to where geometric salience (slope, curvature) and life
 * salience (coherence, attention residue) are high. Terrain and life-drive are
 * one operator chain, not two subsystems bolted together.
 *
 * DETERMINISM CONTRACT (non-negotiable, per the build standard)
 * -------------------------------------------------------------
 *  - Height at a world point is a pure function of (x, z, seed). No frame state,
 *    no RNG draw, no global mutation. h(x,z) on machine A == h(x,z) on machine B.
 *  - LOD selection is a pure function of (salience, eye, params). Given identical
 *    inputs it yields an identical integer level. This is what makes the field
 *    reproducible and the renderer's delta-evaluation valid.
 *  - All float math is single-precision and FMA-contraction-safe: the scalar
 *    reference path and the AVX2 path agree to <1e-5 (verified in tests).
 *
 * COORDINATE MODEL
 * ----------------
 * We use a flat-chart "infinite plane" model rather than a true spheroid. This is
 * deliberate, not a shortcut — see DESIGN NOTE in substrate.c. World units are
 * meters. A planet-scale chart is addressed by signed 64-bit tile coordinates,
 * which gives a 2^64 * tile_size addressable extent: far past any session's reach.
 */
#ifndef GOECKOH_SUBSTRATE_H
#define GOECKOH_SUBSTRATE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 *  Compile-time configuration. Tunable, but the defaults are chosen   *
 *  so a CHUNK fits an L2 slice and a tile-row fits L1 on the hot path. *
 * ------------------------------------------------------------------ */
#define SUB_CHUNK_VERTS      64          /* vertices per chunk edge (64x64 grid)   */
#define SUB_MAX_LOD          12          /* deepest subdivision level              */
#define SUB_NOISE_OCTAVES    9           /* fBm octaves at max LOD                  */
#define SUB_METERS_PER_CHUNK 256.0f      /* world size of an LOD-0 chunk           */

/* A height sample plus the salience that drove its resolution choice.
 * Packed so a chunk's vertex array streams linearly through the cache. */
typedef struct {
    float height;        /* meters; pure f(x,z,seed)                      */
    float slope;         /* ||grad h||, the geometric-salience input      */
    float gcl;           /* Goeckoh coherence sampled here, life-salience */
} SubVertex;

/* The Goeckoh coupling field. This is the seam between the planet and the
 * life-drive. The substrate does NOT own this data — the GRV2 runtime's
 * Goeckoh lattice owns it. The substrate reads it through this view so the
 * two systems share one source of truth instead of duplicating state. */
typedef struct {
    const float* gcl_grid;   /* row-major coherence samples, [h*w]            */
    int          w, h;       /* grid dims                                     */
    float        origin_x;   /* world coord of grid cell (0,0)                */
    float        origin_z;
    float        cell_size;  /* world meters per coherence cell               */
    float        fill;       /* value returned outside the grid (background)  */
} GCLField;

/* Parameters of the LOD oracle. Naming maps 1:1 to the binding equation:
 *   LOD = ceil( log2( 1 + kappa*slope + beta*gcl + gamma*exp(-d/sigma) ) ) */
typedef struct {
    float kappa;   /* weight on geometric salience (slope)            */
    float beta;    /* weight on life salience (GCL) — the novel term  */
    float gamma;   /* weight on raw proximity                         */
    float sigma;   /* proximity falloff distance, meters              */
    int   lod_cap; /* hard ceiling, <= SUB_MAX_LOD                    */
} LODParams;

/* Sensible defaults: slope dominates at distance, GCL pulls detail toward
 * living regions, proximity is a gentle floor so the eye is never starved. */
LODParams sub_default_lod_params(void);

/* ---- The deterministic height field ---------------------------------- *
 * Pure functions. Thread-safe by construction (no shared mutable state).  */

/* Single sample. Reference correctness path. */
float sub_height(float wx, float wz, uint64_t seed);

/* Eight samples at once (AVX2 when available, scalar fallback otherwise).
 * wx/wz are 8-lane arrays; out is written 8-wide. The vectorized fBm here
 * is the performance core of the whole substrate. */
void  sub_height8(const float* wx, const float* wz, uint64_t seed, float* out);

/* ---- The LOD oracle --------------------------------------------------- *
 * Given salience inputs and the eye, return the integer subdivision level.
 * Pure, branch-light, identical on every machine. */
int   sub_select_lod(float slope, float gcl, float dist_to_eye,
                     const LODParams* p);

/* ---- Chunk generation ------------------------------------------------- *
 * Fill a SubVertex grid for the chunk at integer tile coords (cx, cz) at a
 * given LOD. Reads the GCL field for the life-salience channel. This is the
 * function the streaming layer calls when a chunk enters view or its LOD
 * changes. Returns the number of vertices written (always CHUNK_VERTS^2 for
 * a valid lod; 0 on bad args). */
size_t sub_generate_chunk(int64_t cx, int64_t cz, int lod,
                          uint64_t seed,
                          const GCLField* gcl,
                          SubVertex* out_verts);

/* GCL field sampler — bilinear, clamped to the grid, deterministic. Exposed
 * because the renderer wants it too (for shading, not just LOD). */
float sub_sample_gcl(const GCLField* f, float wx, float wz);

#ifdef __cplusplus
}
#endif
#endif /* GOECKOH_SUBSTRATE_H */
