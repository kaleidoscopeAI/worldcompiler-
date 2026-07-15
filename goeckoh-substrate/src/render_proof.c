/* render_proof.c — visual proof the substrate has character.
 *
 * A number (16.6 M verts/sec) proves it is fast. It does not prove it looks
 * like anything. This raymarches the height field directly (sphere-tracing the
 * heightfield as an SDF lower bound) and shades it with slope-based rock/grass/
 * snow banding plus a GCL overlay, so the "life pulls detail here" claim is
 * visible, not just asserted. Output is a PPM you can open.
 *
 * This is a DIAGNOSTIC renderer, not the production path — it exists to make the
 * field inspectable. Production rendering is the existing Δ-SIREN tile kernel;
 * this proves the substrate that kernel will draw.
 */
#include "substrate.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define RW 800
#define RH 450

typedef struct { float x, y, z; } V3;
static V3 v(float x,float y,float z){V3 r={x,y,z};return r;}
static V3 add(V3 a,V3 b){return v(a.x+b.x,a.y+b.y,a.z+b.z);}
static V3 mul(V3 a,float s){return v(a.x*s,a.y*s,a.z*s);}
static float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static V3 norm(V3 a){float l=sqrtf(dot(a,a));return l>0?mul(a,1.0f/l):a;}
static float clampf(float x,float a,float b){return x<a?a:(x>b?b:x);}

static uint64_t g_seed = 0xBEEF;
static GCLField* g_gcl = NULL;

/* heightfield "SDF": vertical distance to terrain. Conservative for marching. */
static float terrain_dist(V3 p) {
    float h = sub_height(p.x, p.z, g_seed);
    return p.y - h;
}

/* analytic-ish normal via small central differences on the height field */
static V3 terrain_normal(float wx, float wz) {
    float e = 2.0f;
    float hl = sub_height(wx - e, wz, g_seed);
    float hr = sub_height(wx + e, wz, g_seed);
    float hd = sub_height(wx, wz - e, g_seed);
    float hu = sub_height(wx, wz + e, g_seed);
    return norm(v(hl - hr, 2.0f * e, hd - hu));
}

int main(void) {
    /* GCL field: a hotspot — imagine the user lingered here, or a living
     * primitive (the wolf) anchored the lattice's attention. */
    const int GW = 128;
    static float grid[128 * 128];
    for (int z = 0; z < GW; ++z)
        for (int x = 0; x < GW; ++x) {
            float dx = (float)(x - 70), dz = (float)(z - 64);
            float d2 = dx*dx + dz*dz;
            grid[z*GW + x] = expf(-d2 / 400.0f);  /* warm coherence blob */
        }
    GCLField gcl = { grid, GW, GW, -2000.0f, -2000.0f, 40.0f, 0.0f };
    g_gcl = &gcl;

    /* camera: high oblique aerial — the Google Earth "tilted satellite" look,
     * positioned ABOVE the highest terrain so we look down across the range. */
    V3 cam = v(-200.0f, 820.0f, -1100.0f);
    V3 target = v(500.0f, -120.0f, 500.0f);
    V3 fwd = norm(add(target, mul(cam, -1.0f)));
    V3 right = norm(v(fwd.z, 0, -fwd.x));
    V3 up = v(right.y*fwd.z - right.z*fwd.y,
              right.z*fwd.x - right.x*fwd.z,
              right.x*fwd.y - right.y*fwd.x);

    V3 sun = norm(v(-0.4f, 0.7f, 0.5f));

    static unsigned char img[RH][RW][3];

    for (int py = 0; py < RH; ++py) {
        for (int px = 0; px < RW; ++px) {
            float u = (2.0f*(px+0.5f)/RW - 1.0f) * ((float)RW/RH);
            float vv = 1.0f - 2.0f*(py+0.5f)/RH;
            V3 rd = norm(add(fwd, add(mul(right, u*0.55f), mul(up, vv*0.55f))));

            /* LINEAR heightfield march: advance in fixed world steps, detect the
             * frame where the ray drops below the terrain, then binary-refine the
             * crossing. This is the correct way to trace a heightfield — the
             * relaxed sphere-trace overshoots steep faces and ghosts. */
            float t = 0.0f; int hit = 0; V3 hp = cam;
            float dt = 3.0f;              /* base step, meters */
            float prev_gap = cam.y - sub_height(cam.x, cam.z, g_seed);
            for (int s = 0; s < 1400; ++s) {
                t += dt;
                hp = add(cam, mul(rd, t));
                float gap = hp.y - sub_height(hp.x, hp.z, g_seed);
                if (gap < 0.0f) {
                    /* crossed the surface between t-dt and t — binary refine */
                    float lo = t - dt, hi = t;
                    for (int it = 0; it < 24; ++it) {
                        float mid = 0.5f*(lo+hi);
                        V3 mp = add(cam, mul(rd, mid));
                        float gm = mp.y - sub_height(mp.x, mp.z, g_seed);
                        if (gm < 0.0f) hi = mid; else lo = mid;
                    }
                    t = hi; hp = add(cam, mul(rd, t));
                    hit = 1; break;
                }
                prev_gap = gap;
                /* step grows with distance for far terrain — bounded so we don't
                 * skip thin ridges up close */
                dt = fminf(3.0f + t*0.012f, 30.0f);
                if (t > 5000.0f) break;
            }
            (void)prev_gap;

            float r,g,b;
            if (hit) {
                V3 n = terrain_normal(hp.x, hp.z);
                float h = hp.y;
                float slope = 1.0f - clampf(n.y, 0.0f, 1.0f);  /* 0 flat .. 1 cliff */
                float diff = clampf(dot(n, sun), 0.0f, 1.0f);
                float amb = 0.12f;
                float light = amb + 1.15f * diff;

                /* slope/height banding: grass low+flat, rock steep, snow high */
                float snow = clampf((h - 180.0f) / 80.0f, 0.0f, 1.0f);
                float rock = clampf(slope * 2.2f, 0.0f, 1.0f);
                V3 grass = v(0.20f, 0.40f, 0.12f);
                V3 stone = v(0.40f, 0.32f, 0.26f);
                V3 white = v(0.90f, 0.92f, 0.96f);
                V3 base = add(mul(grass, (1.0f-rock)), mul(stone, rock));
                base = add(mul(base, (1.0f-snow)), mul(white, snow));

                /* GCL overlay: living regions get a warm verdigris lift, the
                 * naturalist-cabinet accent — visible proof the field is here */
                float gv = sub_sample_gcl(&gcl, hp.x, hp.z);
                V3 life = v(0.10f, 0.45f, 0.38f);
                base = add(mul(base, (1.0f - gv*0.6f)), mul(life, gv*0.6f));

                r = base.x*light; g = base.y*light; b = base.z*light;

                /* distance haze */
                float fog = clampf((t-400.0f)/6000.0f, 0.0f, 0.82f);
                V3 haze = v(0.62f, 0.66f, 0.74f);
                r = r*(1-fog)+haze.x*fog; g = g*(1-fog)+haze.y*fog; b = b*(1-fog)+haze.z*fog;
            } else {
                /* sky gradient */
                float k = clampf(rd.y*1.4f + 0.3f, 0.0f, 1.0f);
                r = 0.42f + 0.30f*k; g = 0.55f + 0.28f*k; b = 0.72f + 0.22f*k;
            }
            /* ACES-ish + gamma */
            #define TM(c) ((c)*(2.51f*(c)+0.03f)/((c)*(2.43f*(c)+0.59f)+0.14f))
            r=TM(r); g=TM(g); b=TM(b);
            #undef TM
            img[py][px][0]=(unsigned char)(powf(clampf(r,0,1),1/2.2f)*255);
            img[py][px][1]=(unsigned char)(powf(clampf(g,0,1),1/2.2f)*255);
            img[py][px][2]=(unsigned char)(powf(clampf(b,0,1),1/2.2f)*255);
        }
    }

    FILE* f = fopen("/home/claude/goeckoh-substrate/proof.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", RW, RH);
    fwrite(img, sizeof(img), 1, f);
    fclose(f);
    fprintf(stderr, "wrote proof.ppm (%dx%d)\n", RW, RH);
    return 0;
}
