/* substrate_capi.c — a thin C ABI over the substrate, for Python (ctypes).
 *
 * The World Compiler is Python; the substrate is C. This is the seam. It
 * exposes exactly what the integration needs and nothing more:
 *
 *   - create/destroy a streamer
 *   - push N coherence sources (one per compiled scene object) at their
 *     world (x,z) positions with an intensity
 *   - run one streaming update against that coherence field
 *   - read back, for any probe point, how much terrain detail was allocated
 *
 * That last call is what lets Python verify the coupling: compile a sentence,
 * place its objects, and confirm the planet earned detail under them.
 *
 * Built as substrate_capi.so and loaded with ctypes, the same pattern the
 * existing brain_engine.py uses for reality_kernel.so.
 */
#include "substrate_stream.h"
#include "entity_coherence.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Opaque handle bundling everything a session needs. */
typedef struct {
    Streamer*        stream;
    float*           coh_grid;     /* coherence raster backing buffer */
    CoherenceRaster  raster;
} SubstrateSession;

/* --- lifecycle --- */

SubstrateSession* sub_session_create(uint64_t seed, int lod_cap,
                                     int grid_w, int grid_h,
                                     float origin_x, float origin_z,
                                     float cell_size) {
    SubstrateSession* s = (SubstrateSession*)calloc(1, sizeof(SubstrateSession));
    if (!s) return NULL;

    s->stream = (Streamer*)malloc(sizeof(Streamer));
    if (!s->stream) { free(s); return NULL; }

    LODParams lp = sub_default_lod_params();
    if (lod_cap > 0 && lod_cap <= SUB_MAX_LOD) lp.lod_cap = lod_cap;
    stream_init(s->stream, seed, lp);

    s->coh_grid = (float*)calloc((size_t)grid_w * grid_h, sizeof(float));
    if (!s->coh_grid) { free(s->stream); free(s); return NULL; }

    s->raster.grid      = s->coh_grid;
    s->raster.w         = grid_w;
    s->raster.h         = grid_h;
    s->raster.origin_x  = origin_x;
    s->raster.origin_z  = origin_z;
    s->raster.cell_size = cell_size;
    return s;
}

void sub_session_destroy(SubstrateSession* s) {
    if (!s) return;
    free(s->coh_grid);
    free(s->stream);
    free(s);
}

/* --- per-frame: stamp objects, stream, --- *
 * srcs is a flat array: [x0,z0,intensity0,radius0, x1,z1,...]. n is object
 * count. Clears the raster, stamps every object, runs `settle` update passes
 * so the resident set stabilizes. Returns chunks regenerated on the last pass. */
int sub_session_update(SubstrateSession* s, const float* srcs, int n,
                       float eye_x, float eye_z, float view_radius_m,
                       int settle_passes) {
    if (!s) return -1;

    coherence_clear(&s->raster);
    for (int i = 0; i < n; ++i) {
        CoherenceSource cs;
        cs.world_x   = srcs[i*4 + 0];
        cs.world_z   = srcs[i*4 + 1];
        cs.intensity = srcs[i*4 + 2];
        cs.radius_m  = srcs[i*4 + 3];
        coherence_stamp(&s->raster, &cs);
    }
    GCLField gcl = coherence_as_gcl_field(&s->raster, 0.0f);

    int passes = settle_passes < 1 ? 1 : settle_passes;
    int last = 0;
    for (int p = 0; p < passes; ++p)
        last = stream_update(s->stream, eye_x, eye_z, &gcl, view_radius_m);
    return last;
}

/* --- readback --- */

/* total resident chunks this frame */
int sub_session_resident_count(SubstrateSession* s) {
    return s ? stream_resident_count(s->stream) : 0;
}

/* deep-chunk count (lod >= min_lod) within `radius` of a probe point —
 * the measure of "how much detail the terrain earned here". */
int sub_session_detail_near(SubstrateSession* s, float wx, float wz,
                            int min_lod, float radius) {
    if (!s) return 0;
    int n = 0, rc = stream_resident_count(s->stream);
    for (int i = 0; i < rc; ++i) {
        const Chunk* c = stream_chunk_at(s->stream, i);
        if (!c || c->lod < min_lod) continue;
        float cm = SUB_METERS_PER_CHUNK / (float)(1u << c->lod);
        float cxw = ((float)c->cx + 0.5f) * cm, czw = ((float)c->cz + 0.5f) * cm;
        float dx = cxw - wx, dz = czw - wz;
        if (sqrtf(dx*dx + dz*dz) < radius) n++;
    }
    return n;
}

/* terrain height at a world point (for placing objects ON the terrain) */
float sub_session_height(SubstrateSession* s, float wx, float wz) {
    if (!s) return 0.0f;
    /* reach into the streamer's seed via a fresh sample */
    return sub_height(wx, wz, s->stream->seed);
}

/* coherence value at a point (for debugging the stamp) */
float sub_session_coherence(SubstrateSession* s, float wx, float wz) {
    if (!s) return 0.0f;
    GCLField gcl = coherence_as_gcl_field(&s->raster, 0.0f);
    return sub_sample_gcl(&gcl, wx, wz);
}
