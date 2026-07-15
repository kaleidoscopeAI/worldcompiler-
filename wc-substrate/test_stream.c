/* test_stream.c — proves the binding's two core claims, with measured numbers.
 *
 *  CLAIM A (delta): moving the eye a little regenerates FEW chunks; the rest
 *  are reused. If a small move regenerated everything, the "recompute only
 *  what changed" thesis would be false at the scene scale.
 *
 *  CLAIM B (life-following): moving the GCL hotspot moves where detail is
 *  allocated — the resident set gains deep-LOD chunks near the new hotspot.
 *  This is the whole point: the planet's detail tracks the life-drive.
 */
#include "substrate_stream.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static float* make_hotspot(int gw, int hot_x, int hot_z, float spread) {
    float* g = malloc(sizeof(float) * gw * gw);
    for (int z = 0; z < gw; ++z)
        for (int x = 0; x < gw; ++x) {
            float dx = (float)(x - hot_x), dz = (float)(z - hot_z);
            g[z*gw + x] = expf(-(dx*dx + dz*dz) / spread);
        }
    return g;
}

/* count resident chunks at or above a given LOD near a world point */
static int deep_chunks_near(const Streamer* s, float wx, float wz, int min_lod, float radius) {
    int n = 0;
    int rc = stream_resident_count(s);
    for (int i = 0; i < rc; ++i) {
        const Chunk* c = stream_chunk_at(s, i);
        if (c->lod < min_lod) continue;
        float cm = SUB_METERS_PER_CHUNK / (float)(1u << c->lod);
        float cxw = ((float)c->cx + 0.5f) * cm, czw = ((float)c->cz + 0.5f) * cm;
        float dx = cxw - wx, dz = czw - wz;
        if (sqrtf(dx*dx+dz*dz) < radius) n++;
    }
    return n;
}

int main(void) {
    printf("=== Substrate Streaming Binding Test ===\n\n");
    Streamer* sp = malloc(sizeof(Streamer));
    #define s (*sp)
    LODParams lp = sub_default_lod_params(); lp.lod_cap = 6;
    stream_init(&s, 0xBEEF, lp);

    const int GW = 128;
    const float CELL = 30.0f, ORIG = -1920.0f;
    float view_r = 400.0f;

    /* hotspot near world origin-ish */
    float* g1 = make_hotspot(GW, 64, 64, 300.0f);
    GCLField gcl = { g1, GW, GW, ORIG, ORIG, CELL, 0.0f };

    /* ---- frame 1: cold load at eye (0,0) ---- */
    float eye_x = 0, eye_z = 0;
    stream_update(&s, eye_x, eye_z, &gcl, view_r);
    int cold_gen = s.last_generated, resident = stream_resident_count(&s);
    printf("[A] cold load @ eye(0,0):  generated=%d  resident=%d\n", cold_gen, resident);

    /* settle one more identical frame so the resident set is stable */
    stream_update(&s, eye_x, eye_z, &gcl, view_r);
    printf("[A] re-run same frame:     generated=%d  reused=%d  (expect gen~0)\n",
           s.last_generated, s.last_reused);
    int stable_ok = (s.last_generated <= 4);   /* essentially nothing regenerates */

    /* ---- small eye move: 40m east ---- */
    eye_x = 40.0f;
    stream_update(&s, eye_x, eye_z, &gcl, view_r);
    int small_gen = s.last_generated;
    float frac = resident > 0 ? (float)small_gen / resident : 1.0f;
    printf("[A] eye +40m east:         generated=%d  reused=%d  (%.1f%% of set)\n",
           small_gen, s.last_reused, frac*100.0f);
    int delta_ok = (frac < 0.35f);   /* a small move touches a minority of chunks */

    printf("[A] %s\n\n", (stable_ok && delta_ok)
        ? "PASS (delta confirmed: small moves regenerate few chunks)"
        : "FAIL (regenerating too much — delta thesis broken)");

    /* ---- CLAIM B: move the hotspot, watch detail follow ---- */
    eye_x = 0; eye_z = 0;
    stream_update(&s, eye_x, eye_z, &gcl, view_r);
    /* hotspot is at grid (64,64) -> world (ORIG + 64*CELL) */
    float hot1_wx = ORIG + 64*CELL, hot1_wz = ORIG + 64*CELL;
    int deep_at_hot1 = deep_chunks_near(&s, hot1_wx, hot1_wz, 3, 400.0f);

    /* new hotspot shifted far across the view */
    float* g2 = make_hotspot(GW, 95, 70, 300.0f);
    GCLField gcl2 = { g2, GW, GW, ORIG, ORIG, CELL, 0.0f };
    float hot2_wx = ORIG + 95*CELL, hot2_wz = ORIG + 70*CELL;

    /* let the streamer react over a few frames (detail migrates as nodes split) */
    for (int f = 0; f < 4; ++f) stream_update(&s, eye_x, eye_z, &gcl2, view_r);
    int deep_at_hot2_new = deep_chunks_near(&s, hot2_wx, hot2_wz, 3, 400.0f);
    int deep_at_hot1_old = deep_chunks_near(&s, hot1_wx, hot1_wz, 3, 400.0f);

    printf("[B] deep chunks @ original hotspot (before move): %d\n", deep_at_hot1);
    printf("[B] deep chunks @ new hotspot      (after move):  %d\n", deep_at_hot2_new);
    printf("[B] deep chunks @ original hotspot (after move):  %d\n", deep_at_hot1_old);
    int follow_ok = (deep_at_hot2_new > 0) && (deep_at_hot2_new >= deep_at_hot1_old);
    printf("[B] %s\n", follow_ok
        ? "PASS (detail migrated to follow the life-drive)"
        : "FAIL (detail did not track GCL)");

    free(g1); free(g2); free(sp);
    #undef s
    int all = stable_ok && delta_ok && follow_ok;
    printf("\n=== %s ===\n", all ? "BINDING VERIFIED" : "BINDING FAILED");
    return all ? 0 : 1;
}
