/* test_substrate.c — correctness proofs + honest benchmark.
 *
 * This is not a demo. It is the verification gate. It asserts the three
 * claims the substrate makes and refuses to pass if any fails:
 *
 *   1. DETERMINISM ACROSS PATHS: the AVX2 height field and the scalar
 *      reference agree to < 1e-4 over a large random sample. If they
 *      diverge, the renderer's delta-evaluation is unsound and the whole
 *      thing is a lie. This test is the difference between "fast" and "fast
 *      AND correct."
 *
 *   2. REPRODUCIBILITY: identical seed => bit-stable height at a fixed
 *      point. This is what lets two machines render the same planet.
 *
 *   3. GCL COUPLING IS REAL: inject a coherence hotspot and show the LOD
 *      oracle allocates strictly more detail there than on identical
 *      terrain with no coherence. If beta did nothing, the novel claim
 *      would be decoration. We measure the actual LOD lift.
 *
 * Then it benchmarks chunk generation and reports MEASURED throughput —
 * vertices/sec, chunks/sec, and the implied frame budget. No invented
 * numbers; whatever the box does is what gets printed.
 */
#include "substrate.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* simple deterministic PRNG for generating test sample points (NOT the
 * terrain — just where we probe it). */
static uint64_t rng_state = 0x123456789ABCDEFULL;
static float frand(float lo, float hi) {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    float u = (float)((rng_state >> 11) & 0xFFFFFFu) / (float)0xFFFFFFu;
    return lo + u * (hi - lo);
}

/* ---- TEST 1: scalar vs AVX2 agreement -------------------------------- */
static int test_path_agreement(void) {
    const uint64_t seed = 0xC0FFEE1234ULL;
    const int N = 100000;
    float max_abs = 0.0f;
    double sum_abs = 0.0;

    for (int i = 0; i < N; i += 8) {
        float wx[8], wz[8], v8[8];
        for (int k = 0; k < 8; ++k) { wx[k] = frand(-50000, 50000); wz[k] = frand(-50000, 50000); }
        sub_height8(wx, wz, seed, v8);
        for (int k = 0; k < 8; ++k) {
            float ref = sub_height(wx[k], wz[k], seed);
            float d = fabsf(ref - v8[k]);
            if (d > max_abs) max_abs = d;
            sum_abs += d;
        }
    }
    double mean_abs = sum_abs / N;
    printf("  [T1] scalar vs AVX2:  max|Δ| = %.2e   mean|Δ| = %.2e   over %d samples\n",
           max_abs, mean_abs, N);

    /* tolerance: FMA-contraction and op-reorder can differ in the last few
     * ULPs of a ~400m-scale value. 1e-4 m = 0.1 mm. Far below render precision. */
    int ok = (max_abs < 1e-4f);
    printf("  [T1] %s\n", ok ? "PASS (paths agree below 0.1mm)" : "FAIL (paths diverge!)");
    return ok;
}

/* ---- TEST 2: reproducibility ----------------------------------------- */
static int test_reproducibility(void) {
    const uint64_t seed = 42;
    float h1 = sub_height(12345.678f, -9876.543f, seed);
    float h2 = sub_height(12345.678f, -9876.543f, seed);
    /* and a different seed must (almost surely) differ */
    float h3 = sub_height(12345.678f, -9876.543f, seed + 1);

    int stable = (h1 == h2);
    int seed_matters = (fabsf(h1 - h3) > 1e-3f);
    printf("  [T2] same seed:   h=%.6f == h=%.6f  -> %s\n", h1, h2, stable ? "stable" : "UNSTABLE");
    printf("  [T2] seed+1:      h=%.6f (Δ=%.4f)    -> %s\n", h3, fabsf(h1-h3),
           seed_matters ? "differs" : "NO EFFECT");
    int ok = stable && seed_matters;
    printf("  [T2] %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* ---- TEST 3: GCL coupling changes LOD -------------------------------- */
static int test_gcl_coupling(void) {
    LODParams p = sub_default_lod_params();

    /* identical geometric salience (slope) and distance for both probes;
     * only GCL differs. This isolates the beta term. */
    float slope = 0.15f;          /* moderate hillside */
    float dist  = 2000.0f;        /* same distance     */

    int lod_dead  = sub_select_lod(slope, 0.0f, dist, &p);   /* no coherence */
    int lod_alive = sub_select_lod(slope, 1.0f, dist, &p);   /* full coherence */

    printf("  [T3] identical terrain @ slope=%.2f dist=%.0fm:\n", slope, dist);
    printf("  [T3]   GCL=0.0 (dead)  -> LOD %d\n", lod_dead);
    printf("  [T3]   GCL=1.0 (alive) -> LOD %d\n", lod_alive);
    printf("  [T3]   life-driven detail lift: +%d levels (%.1fx vertex density)\n",
           lod_alive - lod_dead, powf(4.0f, (float)(lod_alive - lod_dead)));

    int ok = (lod_alive > lod_dead);
    printf("  [T3] %s\n", ok ? "PASS (GCL provably allocates detail)" : "FAIL (beta inert)");
    return ok;
}

/* ---- TEST 4: chunk generation sanity --------------------------------- */
static int test_chunk_sanity(void) {
    static SubVertex verts[SUB_CHUNK_VERTS * SUB_CHUNK_VERTS];
    GCLField empty = {0}; empty.fill = 0.0f;
    size_t n = sub_generate_chunk(3, -7, 4, 0xABCD, &empty, verts);

    int finite = 1;
    float hmin = 1e30f, hmax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        if (!isfinite(verts[i].height) || !isfinite(verts[i].slope)) { finite = 0; break; }
        if (verts[i].height < hmin) hmin = verts[i].height;
        if (verts[i].height > hmax) hmax = verts[i].height;
    }
    printf("  [T4] chunk(3,-7,lod4): %zu verts, height range [%.1f, %.1f]m\n", n, hmin, hmax);
    int ok = (n == (size_t)SUB_CHUNK_VERTS * SUB_CHUNK_VERTS) && finite && (hmax > hmin);
    printf("  [T4] %s\n", ok ? "PASS" : "FAIL");
    return ok;
}

/* ---- BENCHMARK: measured, not invented ------------------------------- */
static void benchmark(void) {
    printf("\n--- BENCHMARK (single core; scales with OMP threads on your box) ---\n");

    /* a coherence field with one hotspot, to exercise the GCL sampler too */
    const int GW = 64;
    static float grid[64 * 64];
    for (int z = 0; z < GW; ++z)
        for (int x = 0; x < GW; ++x) {
            float dx = (float)(x - 32), dz = (float)(z - 32);
            grid[z * GW + x] = expf(-(dx * dx + dz * dz) / 200.0f);
        }
    GCLField gcl = { grid, GW, GW, -4096.0f, -4096.0f, 128.0f, 0.0f };

    static SubVertex verts[SUB_CHUNK_VERTS * SUB_CHUNK_VERTS];
    const int CHUNKS = 4000;

    double t0 = now_sec();
    volatile float sink = 0.0f;
    for (int i = 0; i < CHUNKS; ++i) {
        int64_t cx = (i * 13) % 97;
        int64_t cz = (i * 7) % 89;
        int lod = (i % 6) + 2;
        sub_generate_chunk(cx, cz, lod, 0xBEEF, &gcl, verts);
        sink += verts[0].height + verts[100].slope;   /* defeat DCE */
    }
    double t1 = now_sec();
    (void)sink;

    double secs = t1 - t0;
    double verts_total = (double)CHUNKS * SUB_CHUNK_VERTS * SUB_CHUNK_VERTS;
    double vps = verts_total / secs;
    double cps = (double)CHUNKS / secs;

    printf("  generated %d chunks (%.0f vertices) in %.3f s\n", CHUNKS, verts_total, secs);
    printf("  throughput: %.2f M verts/sec   |   %.0f chunks/sec\n", vps / 1e6, cps);

    /* frame-budget framing: a typical view holds ~200 visible chunks at mixed
     * LOD. How much of a 16.6ms (60fps) frame does regenerating them cost? */
    double chunks_per_frame = 200.0;
    double frame_cost_ms = (chunks_per_frame / cps) * 1000.0;
    printf("  implied cost to regen 200 chunks: %.2f ms/frame (%.1f%% of a 60fps budget)\n",
           frame_cost_ms, frame_cost_ms / 16.6 * 100.0);
    printf("  NOTE: streaming only regenerates chunks whose LOD changed, so steady-state\n");
    printf("        cost is a small fraction of this. This is the cold worst case.\n");

#if defined(__AVX2__)
    printf("  build: AVX2 + FMA path active\n");
#else
    printf("  build: scalar fallback path (no AVX2 at compile time)\n");
#endif
}

int main(void) {
    printf("=== Goeckoh Substrate Verification Gate ===\n\n");
    int all = 1;
    all &= test_path_agreement();
    printf("\n");
    all &= test_reproducibility();
    printf("\n");
    all &= test_gcl_coupling();
    printf("\n");
    all &= test_chunk_sanity();

    benchmark();

    printf("\n=== %s ===\n", all ? "ALL CORRECTNESS GATES PASSED" : "FAILURE — DO NOT SHIP");
    return all ? 0 : 1;
}
