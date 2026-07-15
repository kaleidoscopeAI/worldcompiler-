/* entity_coherence.c — implementation of the entity->terrain coherence bridge. */
#include "entity_coherence.h"
#include <math.h>
#include <string.h>

void coherence_clear(CoherenceRaster* r) {
    if (!r || !r->grid) return;
    memset(r->grid, 0, (size_t)r->w * (size_t)r->h * sizeof(float));
}

void coherence_stamp(CoherenceRaster* r, const CoherenceSource* src) {
    if (!r || !r->grid || !src || r->cell_size <= 0.0f) return;
    if (src->radius_m <= 0.0f || src->intensity <= 0.0f) return;

    /* entity center in grid coordinates */
    float gcx = (src->world_x - r->origin_x) / r->cell_size;
    float gcz = (src->world_z - r->origin_z) / r->cell_size;
    float grad = src->radius_m / r->cell_size;     /* radius in cells */

    /* bounding box of affected cells, clamped to the grid */
    int x0 = (int)floorf(gcx - grad), x1 = (int)ceilf(gcx + grad);
    int z0 = (int)floorf(gcz - grad), z1 = (int)ceilf(gcz + grad);
    if (x0 < 0) x0 = 0; if (z0 < 0) z0 = 0;
    if (x1 > r->w - 1) x1 = r->w - 1;
    if (z1 > r->h - 1) z1 = r->h - 1;

    /* Gaussian falloff: at the center -> intensity, at radius -> ~intensity/e².
     * sigma chosen so the kernel is ~0 by the stated radius (radius = 2.5 sigma).
     * Additive accumulation, clamped to 1.0. */
    float sigma = grad / 2.5f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);

    for (int z = z0; z <= z1; ++z) {
        float dz = (float)z - gcz;
        for (int x = x0; x <= x1; ++x) {
            float dx = (float)x - gcx;
            float d2 = dx * dx + dz * dz;
            float contrib = src->intensity * expf(-d2 * inv2s2);
            int idx = z * r->w + x;
            float v = r->grid[idx] + contrib;
            r->grid[idx] = v > 1.0f ? 1.0f : v;   /* clamp */
        }
    }
}

void coherence_stamp_all(CoherenceRaster* r, const CoherenceSource* srcs, size_t n) {
    for (size_t i = 0; i < n; ++i) coherence_stamp(r, &srcs[i]);
}

GCLField coherence_as_gcl_field(const CoherenceRaster* r, float fill) {
    GCLField f;
    f.gcl_grid  = r ? r->grid : NULL;
    f.w         = r ? r->w : 0;
    f.h         = r ? r->h : 0;
    f.origin_x  = r ? r->origin_x : 0.0f;
    f.origin_z  = r ? r->origin_z : 0.0f;
    f.cell_size = r ? r->cell_size : 1.0f;
    f.fill      = fill;
    return f;
}
