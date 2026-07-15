/* substrate_stream.c — GCL-driven quadtree streaming. */
#include "substrate_stream.h"
#include <string.h>
#include <math.h>

static uint64_t chunk_key(int lod, int64_t cx, int64_t cz) {
    uint64_t h = (uint64_t)lod * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)cx + 0x165667B19E3779F9ULL + (h << 6) + (h >> 2);
    h ^= (uint64_t)cz + 0x27D4EB2F165667C5ULL + (h << 6) + (h >> 2);
    return h;
}

void stream_init(Streamer* s, uint64_t seed, LODParams lod) {
    memset(s, 0, sizeof(*s));
    s->seed = seed;
    s->lod  = lod;
    s->frame = 0;
}

/* Find a resident chunk by key, or NULL. Linear scan — count is bounded and
 * this stays in L2; a hashmap is the obvious upgrade once count is large, but
 * premature here. */
static Chunk* find_chunk(Streamer* s, uint64_t key) {
    /* Match by key regardless of the resident flag. A chunk demoted at the top
     * of this frame still holds valid geometry; if refine wants it again we
     * reuse it instead of regenerating. `valid` distinguishes a real cached
     * chunk from an empty pool slot. */
    for (int i = 0; i < s->count; ++i)
        if (s->pool[i].cached && s->pool[i].key == key)
            return &s->pool[i];
    return NULL;
}

/* Evict the least-recently-touched chunk to make room. Returns its slot. */
static Chunk* evict_lru(Streamer* s) {
    /* Reclaim the least-recently-touched slot that was NOT touched this frame
     * (touching this frame means refine needs it now — never evict those). An
     * empty (never-cached) slot is always preferred. */
    int oldest = -1; uint32_t best = 0xFFFFFFFFu;
    for (int i = 0; i < s->count; ++i) {
        if (!s->pool[i].cached) return &s->pool[i];          /* empty slot */
        if (s->pool[i].last_touched == s->frame) continue;   /* needed now  */
        if (s->pool[i].last_touched < best) { best = s->pool[i].last_touched; oldest = i; }
    }
    if (oldest < 0) oldest = 0;   /* pathological: pool full of this-frame chunks */
    s->last_evicted++;
    return &s->pool[oldest];
}

/* Acquire a chunk for (lod,cx,cz): reuse if cached, else generate into an
 * evicted/free slot. This is the delta gate — generation only happens on miss.
 * Either way the chunk is marked resident (visible this frame) and touched. */
static Chunk* acquire(Streamer* s, int lod, int64_t cx, int64_t cz, const GCLField* gcl) {
    uint64_t key = chunk_key(lod, cx, cz);
    Chunk* c = find_chunk(s, key);
    if (c) {
        c->resident = true;             /* re-mark visible this frame */
        c->last_touched = s->frame;
        s->last_reused++;
        return c;
    }

    /* miss — need to generate */
    if (s->count < STREAM_MAX_CHUNKS) c = &s->pool[s->count++];
    else c = evict_lru(s);

    c->cx = cx; c->cz = cz; c->lod = lod; c->key = key;
    c->resident = true; c->cached = true; c->last_touched = s->frame;
    sub_generate_chunk(cx, cz, lod, s->seed, gcl, c->verts);
    s->last_generated++;
    return c;
}

/* Recursively refine a region of the planet. At node (lod,cx,cz) covering a
 * world square, decide via the LOD oracle whether this node is detailed enough
 * or must split into 4 children. The oracle reads the GCL field sampled at the
 * node center and the worst-case slope, so living/steep regions split deeper. */
static void refine(Streamer* s, int lod, int64_t cx, int64_t cz,
                   float eye_x, float eye_z, const GCLField* gcl,
                   float view_radius_m, int max_lod) {

    float chunk_m = SUB_METERS_PER_CHUNK / (float)(1u << lod);
    float cxw = ((float)cx + 0.5f) * chunk_m;   /* node center, world */
    float czw = ((float)cz + 0.5f) * chunk_m;

    float dx = cxw - eye_x, dz = czw - eye_z;
    float dist = sqrtf(dx*dx + dz*dz);

    /* cull: node entirely outside view radius (plus its own half-extent) */
    if (dist - chunk_m * 0.71f > view_radius_m) return;

    /* SPLIT DECISION — two independent reasons to subdivide:
     *
     * (1) Eye-centered ring: the standard quadtree metric, split when the eye is
     *     close relative to node size. This gives near-the-camera detail and
     *     bounds the chunk count for ordinary terrain.
     *
     * (2) Life-driven screen-space error: where coherence is high, split based on
     *     the GCL-sharpened oracle DIRECTLY, regardless of eye distance. This is
     *     the crucial coupling — a living region earns detail because the world
     *     is alive there, even if it's far from the eye. Widening an eye-centered
     *     ring can't do this (a creature 400m away needs detail the ring can't
     *     reach); the oracle, which folds GCL into the pixel-error budget, can.
     *
     * We split if EITHER fires. */
    const float BASE_SPLIT = 1.6f;       /* eye-ring: split when dist < 1.6*size */

    /* max GCL over the node footprint (prevents popping at coherence edges) */
    float gcl_node = 0.0f;
    for (int sz = 0; sz <= 2; ++sz)
        for (int sx = 0; sx <= 2; ++sx) {
            float px = ((float)cx + 0.5f*(float)sx) * chunk_m;
            float pz = ((float)cz + 0.5f*(float)sz) * chunk_m;
            float g = sub_sample_gcl(gcl, px, pz);
            if (g > gcl_node) gcl_node = g;
        }

    /* slope proxy at node center */
    float e = chunk_m * 0.25f;
    float hl = sub_height(cxw - e, czw, s->seed), hr = sub_height(cxw + e, czw, s->seed);
    float hd = sub_height(cxw, czw - e, s->seed), hu = sub_height(cxw, czw + e, s->seed);
    float slope = sqrtf((hr-hl)*(hr-hl) + (hu-hd)*(hu-hd)) / (2.0f*e);

    /* reason (1): eye-centered ring */
    float split_dist = BASE_SPLIT * chunk_m;
    int split_ring = (dist < split_dist);

    /* reason (2): GCL-sharpened oracle wants this node finer than its current LOD.
     * The oracle returns the LOD that meets the pixel-error budget at this node's
     * distance and coherence. We BOUND its reach: coherence may pull detail at
     * most GCL_MAX_LIFT levels deeper than the eye-ring would give here, so a
     * large coherent blob can't split its entire footprint to max LOD and blow
     * the chunk budget. The lift is proportional to coherence, so the entity's
     * center (high GCL) gets the deepest allowed detail and its fringe less. */
    const int GCL_MAX_LIFT = 3;
    /* what LOD would the eye-ring alone settle this node at? approximate by the
     * level where the ring stops: dist ~ BASE_SPLIT * (chunk0 / 2^L). */
    float chunk0 = SUB_METERS_PER_CHUNK;
    int ring_lod = 0;
    {
        float L = log2f((BASE_SPLIT * chunk0) / fmaxf(dist, 1.0f));
        ring_lod = (int)floorf(L);
        if (ring_lod < 0) ring_lod = 0;
    }
    int gcl_target = sub_select_lod(slope, gcl_node, dist, &s->lod);
    int lift_cap = ring_lod + (int)ceilf(GCL_MAX_LIFT * gcl_node);
    if (gcl_target > lift_cap) gcl_target = lift_cap;
    int split_oracle = (gcl_target > lod);

    int should_split = (lod < max_lod) && (split_ring || split_oracle);

    if (!should_split) {
        /* leaf: make it resident at this LOD */
        acquire(s, lod, cx, cz, gcl);
        return;
    }

    /* split into 4 children at lod+1 (tile coords double) */
    int64_t c2x = cx * 2, c2z = cz * 2;
    refine(s, lod+1, c2x,   c2z,   eye_x, eye_z, gcl, view_radius_m, max_lod);
    refine(s, lod+1, c2x+1, c2z,   eye_x, eye_z, gcl, view_radius_m, max_lod);
    refine(s, lod+1, c2x,   c2z+1, eye_x, eye_z, gcl, view_radius_m, max_lod);
    refine(s, lod+1, c2x+1, c2z+1, eye_x, eye_z, gcl, view_radius_m, max_lod);
}

int stream_update(Streamer* s, float eye_x, float eye_z, const GCLField* gcl,
                  float view_radius_m) {
    s->frame++;
    s->last_generated = 0; s->last_reused = 0; s->last_evicted = 0;

    /* Demote everything to non-resident. refine() re-touches (and re-marks
     * resident) exactly the chunks the quadtree wants THIS frame, given the
     * current eye AND the current GCL field. Anything refine doesn't revisit —
     * because the eye moved OR because an entity's coherence dropped — falls out
     * of the resident set immediately. Its data is retained for reuse-by-key
     * until a later generation overwrites the slot, so re-entering a region is
     * still cheap, but it no longer COUNTS as resident detail. This is what makes
     * detail relax the instant the life-drive leaves a region. */
    for (int i = 0; i < s->count; ++i)
        s->pool[i].resident = false;

    /* find which LOD-0 tiles the view radius spans, refine each */
    float r = view_radius_m;
    int64_t t0x = (int64_t)floorf((eye_x - r) / SUB_METERS_PER_CHUNK);
    int64_t t1x = (int64_t)floorf((eye_x + r) / SUB_METERS_PER_CHUNK);
    int64_t t0z = (int64_t)floorf((eye_z - r) / SUB_METERS_PER_CHUNK);
    int64_t t1z = (int64_t)floorf((eye_z + r) / SUB_METERS_PER_CHUNK);

    for (int64_t tz = t0z; tz <= t1z; ++tz)
        for (int64_t tx = t0x; tx <= t1x; ++tx)
            refine(s, 0, tx, tz, eye_x, eye_z, gcl, view_radius_m, s->lod.lod_cap);

    return s->last_generated;
}

int stream_resident_count(const Streamer* s) {
    int n = 0;
    for (int i = 0; i < s->count; ++i) if (s->pool[i].resident) n++;
    return n;
}

const Chunk* stream_chunk_at(const Streamer* s, int i) {
    int n = 0;
    for (int j = 0; j < s->count; ++j)
        if (s->pool[j].resident) { if (n == i) return &s->pool[j]; n++; }
    return NULL;
}
