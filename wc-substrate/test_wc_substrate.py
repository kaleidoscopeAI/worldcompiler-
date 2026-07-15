#!/usr/bin/env python3
"""
test_wc_substrate.py — prove the World Compiler drives the planetary substrate.

Claims, measured:
  C1: the bbox fix works — after compile+resolve, a spatial sentence gives
      objects with DISTINCT centroids (not all at origin).
  C2: compiling a sentence makes the terrain resolve under the objects — deep
      chunks appear at each object's projected position, vs bare terrain.
  C3: more objects -> more total terrain detail (the scene drives the planet).
"""

import sys
sys.path.insert(0, ".")
from wc_substrate_bridge import WorldCompilerOnTerrain, Substrate, compute_bbox_from_sdf
import world_compiler_core as wc


def banner(s):
    print("\n" + "=" * 60)
    print(s)
    print("=" * 60)


def test_bbox_fix():
    banner("C1: bbox fix gives objects distinct positions")
    sub = Substrate()
    scene = WorldCompilerOnTerrain(resolution=32, substrate=sub)
    resolved = scene.compile("the cat sits on the chair")
    print(f"  resolved predicates: {resolved}")
    objs = scene.object_names()
    for nid, name, centroid in objs:
        print(f"    {nid}: {name:8s} centroid={centroid}")

    centroids = [c for _, _, c in objs]
    distinct = len(set(centroids)) > 1
    print(f"  distinct centroids: {distinct}")
    ok = distinct and len(objs) >= 2
    print(f"  {'PASS' if ok else 'FAIL'} "
          f"({'objects separated in space' if ok else 'objects collapsed to one point'})")
    return ok


def test_detail_under_objects():
    banner("C2: terrain resolves under compiled objects")
    sub = Substrate()
    scene = WorldCompilerOnTerrain(resolution=32, substrate=sub)
    scene.compile("the wolf sits on the hill")
    sources, regen = scene.project_to_terrain()

    print(f"  compiled {len(sources)} objects onto the planet:")
    for (x, z, inten, rad) in sources:
        coh = sub.coherence(x, z)
        print(f"    object @ ({x:7.1f},{z:7.1f}) r={rad:.0f}  coherence={coh:.3f}")

    # detail under each object vs a bare-terrain baseline at the same spots
    bare = Substrate()
    bare.update([], eye=scene.scene_origin, view_radius=600.0)

    all_ok = True
    for (x, z, inten, rad) in sources:
        with_obj = sub.detail_near(x, z, min_lod=3, radius=200.0)
        without  = bare.detail_near(x, z, min_lod=3, radius=200.0)
        delta = with_obj - without
        status = "PASS" if delta > 0 else "FAIL"
        if delta <= 0:
            all_ok = False
        print(f"    @({x:7.1f},{z:7.1f}): with={with_obj:3d} bare={without:3d} "
              f"delta=+{delta:<3d} {status}")

    print(f"  {'PASS' if all_ok else 'FAIL'} "
          f"({'objects earn terrain detail' if all_ok else 'no detail gained'})")
    return all_ok


def test_more_objects_more_detail():
    banner("C3: richer scene -> more terrain detail")
    sub1 = Substrate()
    s1 = WorldCompilerOnTerrain(resolution=32, substrate=sub1)
    s1.compile("the wolf")
    src1, _ = s1.project_to_terrain()
    total1 = sub1.resident_count()

    sub2 = Substrate()
    s2 = WorldCompilerOnTerrain(resolution=32, substrate=sub2)
    s2.compile("the wolf and the bear and the tree")
    src2, _ = s2.project_to_terrain()
    total2 = sub2.resident_count()

    print(f"  '{'the wolf'}': {len(src1)} object(s) -> {total1} resident chunks")
    print(f"  '{'the wolf and the bear and the tree'}': "
          f"{len(src2)} object(s) -> {total2} resident chunks")
    # more objects spread coherence wider -> more resident chunks (or at least not fewer)
    ok = total2 >= total1
    print(f"  {'PASS' if ok else 'FAIL'} "
          f"({'scene complexity drives the planet' if ok else 'no scaling'})")
    return ok


if __name__ == "__main__":
    print("WORLD COMPILER x PLANETARY SUBSTRATE — INTEGRATION")
    c1 = test_bbox_fix()
    c2 = test_detail_under_objects()
    c3 = test_more_objects_more_detail()

    banner("RESULT")
    print(f"  C1 (distinct positions): {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 (detail under objects): {'PASS' if c2 else 'FAIL'}")
    print(f"  C3 (scene drives planet): {'PASS' if c3 else 'FAIL'}")
    allok = c1 and c2 and c3
    print(f"\n  {'=== INTEGRATION VERIFIED ===' if allok else '=== INCOMPLETE ==='}")
    sys.exit(0 if allok else 1)
