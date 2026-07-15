/* test_binding.c — clean proof of the two binding claims.
 * Rewritten with explicit frame control to avoid measurement-ordering bugs.
 *
 *  A (delta reuse): after the resident set settles, re-running the SAME frame
 *  regenerates ~0 chunks. A small eye move regenerates a MINORITY.
 *
 *  B (life-following): a GCL hotspot causes MORE deep chunks near it than an
 *  identical region with no coherence. Measured on a settled set, same frame.
 */
#include "substrate_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float* hotspot(int gw, int hx, int hz, float spread) {
    float* g = malloc(sizeof(float)*gw*gw);
    for (int z=0; z<gw; ++z) for (int x=0; x<gw; ++x) {
        float dx=(float)(x-hx), dz=(float)(z-hz);
        g[z*gw+x] = expf(-(dx*dx+dz*dz)/spread);
    }
    return g;
}

/* count resident chunks with lod >= min_lod whose center is within `radius`
 * meters of (wx,wz). Correct: single pass over the resident set. */
static int deep_near(const Streamer* s, float wx, float wz, int min_lod, float radius) {
    int n=0, rc=stream_resident_count(s);
    for (int i=0;i<rc;++i) {
        const Chunk* c = stream_chunk_at(s,i);
        if (!c || c->lod < min_lod) continue;
        float cm = SUB_METERS_PER_CHUNK/(float)(1u<<c->lod);
        float cxw=((float)c->cx+0.5f)*cm, czw=((float)c->cz+0.5f)*cm;
        float dx=cxw-wx, dz=czw-wz;
        if (sqrtf(dx*dx+dz*dz) < radius) n++;
    }
    return n;
}

int main(void) {
    printf("=== Substrate Binding Test (clean) ===\n\n");
    Streamer* sp = malloc(sizeof(Streamer));
    LODParams lp = sub_default_lod_params(); lp.lod_cap = 5;
    stream_init(sp, 0xBEEF, lp);

    const int GW=128; const float CELL=24.0f, ORIG=-1536.0f;
    const float R = 350.0f;

    /* ---------- CLAIM A ---------- */
    float* g1 = hotspot(GW, 64, 64, 250.0f);
    GCLField gcl = { g1, GW, GW, ORIG, ORIG, CELL, 0.0f };

    /* settle: run the same frame twice; second run should regen ~0 */
    stream_update(sp, 0, 0, &gcl, R);
    int resident = stream_resident_count(sp);
    stream_update(sp, 0, 0, &gcl, R);
    int regen_same = sp->last_generated;
    printf("[A] resident set: %d chunks\n", resident);
    printf("[A] re-run identical frame:  regenerated=%d  (want 0)\n", regen_same);

    stream_update(sp, 40.0f, 0, &gcl, R);
    int regen_move = sp->last_generated;
    float frac = resident>0 ? (float)regen_move/resident : 1.0f;
    printf("[A] eye +40m:                regenerated=%d  (%.0f%% of set)\n",
           regen_move, frac*100.0f);

    int a_ok = (regen_same == 0) && (frac < 0.5f);
    printf("[A] %s\n\n", a_ok ? "PASS" : "PARTIAL");

    /* ---------- CLAIM B ---------- */
    /* reset, eye fixed at origin throughout so only GCL differs */
    stream_init(sp, 0xBEEF, lp);

    /* B1: hotspot at world (ORIG+64*CELL) */
    float hx_w = ORIG + 64*CELL, hz_w = ORIG + 64*CELL;
    stream_update(sp, hx_w, hz_w, &gcl, R);   /* eye AT the hotspot */
    stream_update(sp, hx_w, hz_w, &gcl, R);   /* settle */
    int deep_with_gcl = deep_near(sp, hx_w, hz_w, 4, R);

    /* B2: SAME eye position, SAME terrain, but NO coherence field */
    GCLField dead = {0}; dead.fill = 0.0f;
    stream_init(sp, 0xBEEF, lp);
    stream_update(sp, hx_w, hz_w, &dead, R);
    stream_update(sp, hx_w, hz_w, &dead, R);
    int deep_no_gcl = deep_near(sp, hx_w, hz_w, 4, R);

    printf("[B] deep chunks (lod>=4) within %.0fm of eye:\n", R);
    printf("[B]   with GCL hotspot: %d\n", deep_with_gcl);
    printf("[B]   no coherence:     %d\n", deep_no_gcl);
    printf("[B]   life-driven gain: +%d deep chunks (%.0f%% more)\n",
           deep_with_gcl - deep_no_gcl,
           deep_no_gcl>0 ? 100.0f*(deep_with_gcl-deep_no_gcl)/deep_no_gcl
                         : (deep_with_gcl>0?100.0f:0.0f));

    int b_ok = (deep_with_gcl > deep_no_gcl);
    printf("[B] %s\n", b_ok ? "PASS (GCL provably allocates more detail)" : "FAIL");

    free(g1); free(sp);
    int all = a_ok && b_ok;
    printf("\n=== %s ===\n", all ? "BINDING VERIFIED" : "BINDING INCOMPLETE");
    return all ? 0 : 1;
}
