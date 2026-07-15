/* test_integration.c — full stack, with the slope confound controlled.
 *
 * The naive test ("detail under entity vs empty terrain") is confounded: this
 * terrain is mountainous everywhere, so the screen-space oracle gives every
 * location deep LOD from SLOPE alone, swamping the coherence signal. To isolate
 * what the entity actually contributes, we measure the DELTA on the SAME terrain:
 *
 *   detail at position P with the entity present
 *   minus
 *   detail at position P with NO entities at all
 *
 * The difference is the coherence-driven detail — slope is identical in both
 * cases, so it cancels. Same control test_binding uses.
 */
#include "entity_coherence.h"
#include "substrate_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static int detail_near(const Streamer* s, float wx, float wz, int min_lod, float radius) {
    int n = 0, rc = stream_resident_count(s);
    for (int i = 0; i < rc; ++i) {
        const Chunk* c = stream_chunk_at(s, i);
        if (!c || c->lod < min_lod) continue;
        float cm = SUB_METERS_PER_CHUNK / (float)(1u << c->lod);
        float cxw = ((float)c->cx + 0.5f) * cm, czw = ((float)c->cz + 0.5f) * cm;
        float dx = cxw - wx, dz = czw - wz;
        if (sqrtf(dx*dx + dz*dz) < radius) n++;
    }
    return n;
}

static int settled_detail(Streamer* s, CoherenceRaster* raster,
                          const CoherenceSource* ents, size_t n,
                          float eye_x, float eye_z, float view_r,
                          float probe_x, float probe_z) {
    coherence_clear(raster);
    if (n) coherence_stamp_all(raster, ents, n);
    GCLField gcl = coherence_as_gcl_field(raster, 0.0f);
    for (int f = 0; f < 5; ++f) stream_update(s, eye_x, eye_z, &gcl, view_r);
    return detail_near(s, probe_x, probe_z, 3, 220.0f);
}

int main(void) {
    printf("=== Full-Stack Integration (slope-controlled) ===\n\n");

    const int GW = 192;
    static float coh_grid[192 * 192];
    CoherenceRaster raster = { coh_grid, GW, GW, -2880.0f, -2880.0f, 30.0f };

    Streamer* sp = malloc(sizeof(Streamer));
    LODParams lp = sub_default_lod_params();
    lp.lod_cap = 5;
    stream_init(sp, 0xACE5, lp);

    const float VIEW_R = 600.0f;
    float eye_x = 0.0f, eye_z = 0.0f;
    float cre_x = 200.0f, cre_z = 150.0f;

    CoherenceSource creature = { cre_x, cre_z, 1.0f, 180.0f };

    int with = settled_detail(sp, &raster, &creature, 1, eye_x, eye_z, VIEW_R, cre_x, cre_z);
    stream_init(sp, 0xACE5, lp);
    int without = settled_detail(sp, &raster, NULL, 0, eye_x, eye_z, VIEW_R, cre_x, cre_z);

    printf("[I1] deep (LOD>=3) chunks at (%.0f,%.0f):\n", cre_x, cre_z);
    printf("[I1]   creature present: %d\n", with);
    printf("[I1]   no entities:      %d\n", without);
    printf("[I1]   coherence-driven detail (delta): +%d\n", with - without);
    int i1_ok = (with > without);
    printf("[I1] %s\n\n", i1_ok
        ? "PASS (the living entity adds terrain detail, slope held equal)"
        : "FAIL (entity contributed no detail beyond geometry)");

    float new_x = -150.0f, new_z = 400.0f;
    CoherenceSource moved = { new_x, new_z, 1.0f, 180.0f };

    stream_init(sp, 0xACE5, lp);
    int new_with = settled_detail(sp, &raster, &moved, 1, eye_x, eye_z, VIEW_R, new_x, new_z);
    stream_init(sp, 0xACE5, lp);
    int new_without = settled_detail(sp, &raster, NULL, 0, eye_x, eye_z, VIEW_R, new_x, new_z);

    printf("[I2] deep (LOD>=3) chunks at NEW position (%.0f,%.0f):\n", new_x, new_z);
    printf("[I2]   creature moved here: %d\n", new_with);
    printf("[I2]   no entities:         %d\n", new_without);
    printf("[I2]   coherence-driven detail (delta): +%d\n", new_with - new_without);
    int i2_ok = (new_with > new_without);
    printf("[I2] %s\n", i2_ok
        ? "PASS (detail followed the creature to its new position)"
        : "FAIL (moving the entity did not move the added detail)");

    free(sp);
    int all = i1_ok && i2_ok;
    printf("\n=== %s ===\n", all
        ? "INTEGRATION VERIFIED: entity presence drives terrain detail, isolated from slope"
        : "INTEGRATION INCOMPLETE");
    return all ? 0 : 1;
}
