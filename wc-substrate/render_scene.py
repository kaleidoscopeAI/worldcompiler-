#!/usr/bin/env python3
"""Render the planet with a compiled sentence shaping its terrain detail.
Produces a top-down heatmap: terrain height + where the compiled objects
pulled detail (coherence overlay). Visual proof of the coupling."""
import sys; sys.path.insert(0, ".")
import numpy as np
from wc_substrate_bridge import WorldCompilerOnTerrain, Substrate

sentence = "the wolf and the bear and the tree"
sub = Substrate()
scene = WorldCompilerOnTerrain(resolution=32, substrate=sub)
scene.compile(sentence)
sources, regen = scene.project_to_terrain()

# sample a top-down grid: height as base, coherence as overlay
N = 240
span = 1400.0  # meters, centered on scene origin
xs = np.linspace(-span, span, N)
zs = np.linspace(-span, span, N)
height = np.zeros((N, N), dtype=np.float32)
coh = np.zeros((N, N), dtype=np.float32)
for j, z in enumerate(zs):
    for i, x in enumerate(xs):
        height[j, i] = sub.height(float(x), float(z))
        coh[j, i] = sub.coherence(float(x), float(z))

# compose an RGB image: terrain as grayscale relief, coherence as warm overlay
h = height
hn = (h - h.min()) / (np.ptp(h) + 1e-6)
# hillshade-ish: gradient magnitude for relief
gz, gx = np.gradient(hn)
relief = np.clip(0.5 + 4.0*(gx*0.7 + gz*0.7), 0, 1)
base = 0.35 + 0.5*hn*relief  # grayscale terrain

img = np.zeros((N, N, 3), dtype=np.float32)
img[..., 0] = base
img[..., 1] = base
img[..., 2] = base
# coherence overlay: teal-green where objects pulled detail
img[..., 0] = img[..., 0]*(1 - coh*0.5)
img[..., 1] = img[..., 1]*(1 - coh*0.2) + coh*0.45
img[..., 2] = img[..., 2]*(1 - coh*0.3) + coh*0.40

# mark object centers
for (x, z, inten, rad) in sources:
    i = int((x + span) / (2*span) * N)
    j = int((z + span) / (2*span) * N)
    if 0 <= i < N and 0 <= j < N:
        for dj in range(-2, 3):
            for di in range(-2, 3):
                if 0 <= j+dj < N and 0 <= i+di < N:
                    img[j+dj, i+di] = [1.0, 0.85, 0.2]  # amber dot

img8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
try:
    from PIL import Image
    Image.fromarray(img8).save("scene_on_planet.png")
    print(f"wrote scene_on_planet.png  ('{sentence}', {len(sources)} objects)")
except ImportError:
    import struct
    with open("scene_on_planet.ppm", "wb") as f:
        f.write(f"P6\n{N} {N}\n255\n".encode())
        f.write(img8.tobytes())
    print("wrote scene_on_planet.ppm")
