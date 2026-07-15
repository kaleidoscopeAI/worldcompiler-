/* substrate_stream.h — the binding layer.
 *
 * This is the seam the whole exercise was for: the planetary substrate driven
 * by the Goeckoh life-drive. The streamer holds a quadtree of live chunks. Each
 * frame it walks the tree from the eye, and at every node asks the LOD oracle
 * (which reads GCL) whether to split, keep, or merge. Chunks whose LOD changes
 * are regenerated; everything else is reused. This is the Δ principle at the
 * scene scale: recompute only what the life-drive says has changed.
 *
 * Ownership: the streamer owns chunk geometry. The GRV2 runtime owns the GCL
 * field (it's the Goeckoh lattice's coherence output). They share through the
 * GCLField view in substrate.h — one source of truth.
 */
#ifndef GOECKOH_SUBSTRATE_STREAM_H
#define GOECKOH_SUBSTRATE_STREAM_H

#include "substrate.h"
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define STREAM_MAX_CHUNKS 3072   /* hard cap on resident chunks (memory bound) */

typedef struct {
    int64_t   cx, cz;        /* tile coords AT this chunk's LOD level          */
    int       lod;
    uint64_t  key;           /* (lod,cx,cz) hashed — identity for reuse        */
    uint32_t  last_touched;  /* frame index, for LRU eviction                  */
    bool      resident;      /* true = visible/counted THIS frame              */
    bool      cached;        /* true = holds valid geometry, reusable by key   */
    SubVertex verts[SUB_CHUNK_VERTS * SUB_CHUNK_VERTS];
} Chunk;

typedef struct {
    Chunk     pool[STREAM_MAX_CHUNKS];
    int       count;
    uint64_t  seed;
    LODParams lod;
    uint32_t  frame;

    /* stats from the last update — measured, surfaced for honesty */
    int       last_generated;   /* chunks regenerated this frame              */
    int       last_reused;      /* chunks served from cache                   */
    int       last_evicted;
} Streamer;

void   stream_init(Streamer* s, uint64_t seed, LODParams lod);

/* The per-frame entry point. Given eye position and the live GCL field, bring
 * the resident set up to date: split high-salience regions, merge low ones,
 * regenerate only what changed. Returns number of chunks regenerated. */
int    stream_update(Streamer* s, float eye_x, float eye_z, const GCLField* gcl,
                     float view_radius_m);

/* Iterate resident chunks for the renderer. */
int    stream_resident_count(const Streamer* s);
const Chunk* stream_chunk_at(const Streamer* s, int i);

#ifdef __cplusplus
}
#endif
#endif
