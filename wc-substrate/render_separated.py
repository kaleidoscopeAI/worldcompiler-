#!/usr/bin/env python3
"""Demonstrate the coupling with objects placed at DISTINCT terrain locations,
bypassing the compiler's degenerate layout by setting scene origins explicitly.
This shows what the integration does when objects actually separate."""
import sys; sys.path.insert(0, ".")
import numpy as np
from wc_substrate_bridge import WorldCompilerOnTerrain, Substrate

# Place three objects at distinct spots on ONE planet by compiling three
# single-object scenes at different scene origins, accumulating coherence.
sub = Substrate()
placements = [
    ("the wolf", (-400.0, -200.0)),
    ("the bear", (350.0, 100.0)),
    ("the tree", (-50.0, 450.0)),
]
all_sources = []
for sentence, origin in placements:
    s = WorldCompilerOnTerrain(resolution=32, substrate=sub, scene_origin=origin)
    s.compile(sentence)
    srcs = s.scene_to_coherence_sources()
    all_sources.extend(srcs)

# stamp all of them together onto the one planet
regen = sub.update(all_sources, eye=(0.0, 0.0), view_radius=900.0, settle=5)
print(f"placed {len(all_sources)} objects from 3 sentences; {regen} chunks regenerated")
for sentence, origin in placements:
    d = sub.detail_near(origin[0], origin[1], min_lod=4, radius=200.0)
    print(f"  '{sentence}' @ {origin}: {d} deep chunks (lod>=4)")

# render top-down
N = 280; span = 1100.0
xs = np.linspace(-span, span, N); zs = np.linspace(-span, span, N)
height = np.zeros((N, N), dtype=np.float32); coh = np.zeros((N, N), dtype=np.float32)
for j, z in enumerate(zs):
    for i, x in enumerate(xs):
        height[j, i] = sub.height(float(x), float(z))
        coh[j, i] = sub.coherence(float(x), float(z))
hn = (height - height.min()) / (np.ptp(height) + 1e-6)
gz, gx = np.gradient(hn)
relief = np.clip(0.5 + 4.0*(gx*0.7 + gz*0.7), 0, 1)
base = 0.35 + 0.5*hn*relief
img = np.stack([base, base, base], axis=-1)
img[..., 0] *= (1 - coh*0.5)
img[..., 1] = img[..., 1]*(1 - coh*0.2) + coh*0.45
img[..., 2] = img[..., 2]*(1 - coh*0.3) + coh*0.40
for (x, z, inten, rad) in all_sources:
    i = int((x + span)/(2*span)*N); j = int((z + span)/(2*span)*N)
    for dj in range(-2,3):
        for di in range(-2,3):
            if 0<=j+dj<N and 0<=i+di<N: img[j+dj, i+di] = [1.0, 0.85, 0.2]
img8 = (np.clip(img,0,1)*255).astype(np.uint8)
from PIL import Image
Image.fromarray(img8).save("scene_separated.png")
print("wrote scene_separated.png")
