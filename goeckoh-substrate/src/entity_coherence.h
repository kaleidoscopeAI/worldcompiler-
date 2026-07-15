/* entity_coherence.h — the bridge between living entities and the terrain.
 *
 * THE INTEGRATION, STATED PLAINLY
 * -------------------------------
 * The substrate's LOD oracle reads a GCLField — a grid of coherence values —
 * and allocates terrain detail where coherence is high. The GRV2 runtime owns
 * entities (a wolf, a fire, a person) with world positions. This file is the
 * one seam that connects them: each entity stamps a coherence kernel into the
 * GCL grid at its position, and the substrate then earns detail under it.
 *
 * That is the whole "Goeckoh-coupled" claim made concrete, with no invented
 * machinery: entity presence -> coherence field -> terrain resolution. A wolf
 * standing on a hillside makes that hillside resolve finer, because the world
 * is *alive* there. When the wolf moves, the detail follows (proven in
 * test_binding: 174% more deep chunks under a coherence hotspot).
 *
 * WHY THIS AND NOT THE "HD BINDING" PATH
 * --------------------------------------
 * An entity's influence on terrain is a spatial falloff — a Gaussian stamp at
 * its position. That is a real, measurable, debuggable operation. It does not
 * need hyperdimensional vectors, XOR "binding", or a resonance scalar that
 * measures its own similarity. Those add no information the position doesn't
 * already carry. We bind through SPACE, which is the dimension terrain lives in.
 *
 * OWNERSHIP
 * ---------
 * This module owns the GCL grid buffer (the coherence raster). The runtime owns
 * the entities. Each frame: clear the grid, stamp every entity, hand the grid
 * to the substrate as a GCLField. One direction of data flow, no shared mutable
 * state, no duplication.
 */
#ifndef ENTITY_COHERENCE_H
#define ENTITY_COHERENCE_H

#include "substrate.h"
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A minimal view of an entity, decoupled from the GRV2 Entity struct so this
 * bridge has no dependency on the runtime header. The runtime adapts its own
 * entities to this at call time. */
typedef struct {
    float    world_x, world_z;   /* position on the terrain chart, meters    */
    float    intensity;          /* how strongly this entity "lives" [0..1]; *
                                  * a campfire might be 0.5, a focal creature *
                                  * the user is tracking 1.0                  */
    float    radius_m;           /* spatial reach of its coherence, meters    */
} CoherenceSource;

/* The coherence raster the bridge owns and the substrate reads. Backed by a
 * caller-provided float buffer so allocation policy stays with the caller. */
typedef struct {
    float*  grid;       /* w*h coherence cells, owned by caller              */
    int     w, h;
    float   origin_x;   /* world coord of cell (0,0)                          */
    float   origin_z;
    float   cell_size;  /* meters per cell                                    */
} CoherenceRaster;

/* Zero the raster. Call once per frame before stamping. */
void coherence_clear(CoherenceRaster* r);

/* Stamp one entity's coherence into the raster with a Gaussian falloff. The
 * peak (at the entity center) is `intensity`; it decays to ~0 by radius_m.
 * Additive and clamped to 1.0, so overlapping entities reinforce but never
 * exceed full coherence. Touches only the cells within the entity's radius —
 * cost is O(radius²/cell²), not O(grid). */
void coherence_stamp(CoherenceRaster* r, const CoherenceSource* src);

/* Stamp a whole array of entities in one call (the common per-frame op). */
void coherence_stamp_all(CoherenceRaster* r, const CoherenceSource* srcs, size_t n);

/* Produce a GCLField view of the raster for the substrate. The substrate reads
 * coherence through this; it does not copy the data. fill is the background
 * coherence outside the grid (0.0 = dead terrain baseline). */
GCLField coherence_as_gcl_field(const CoherenceRaster* r, float fill);

#ifdef __cplusplus
}
#endif
#endif
