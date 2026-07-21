"""Tests for grv2_runtime/geo_export.py -- Runtime state -> real 3D-Tiles output.

Mocks grv2_runtime.texture.texture_for so Runtime construction never touches
the network in the test suite (the texture module's own offline fallback is
tested directly in test_grv2_texture.py).
"""
import json
import struct

import pytest

import grv2_runtime.texture as texture_mod
from grv2_runtime import geo_export
from grv2_runtime.runtime import Runtime

_SEED_SENTENCE = "the wolf sits on the hill"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(texture_mod, "texture_for",
                        lambda word, context="": texture_mod.TextureEntry(
                            color=texture_mod.color_from_hash(word), source="hash_fallback"))


@pytest.fixture(scope="module")
def _runtime_module_scoped():
    # module-scoped since Runtime() does real, non-trivial work (terrain
    # compile, genetic population seed) -- the autouse fixture above still
    # re-patches per-test, but construction itself is shared for speed.
    import grv2_runtime.texture as t
    orig = t.texture_for
    t.texture_for = lambda word, context="": t.TextureEntry(
        color=t.color_from_hash(word), source="hash_fallback")
    try:
        rt = Runtime(_SEED_SENTENCE, allow_llm_wiring=False, wiring_dictionary_path=None)
    finally:
        t.texture_for = orig
    return rt


def _read_glb_json_chunk(glb_bytes: bytes) -> dict:
    magic, version, length = struct.unpack_from("<III", glb_bytes, 0)
    assert magic == 0x46546C67, "bad GLB magic"
    chunk_len, chunk_type = struct.unpack_from("<II", glb_bytes, 12)
    assert chunk_type == 0x4E4F534A, "first chunk must be JSON"
    json_bytes = glb_bytes[20:20 + chunk_len]
    return json.loads(json_bytes.decode("utf-8"))


def test_build_entities_dict_has_point_cloud_entities(_runtime_module_scoped):
    entities_dict = geo_export.build_entities_dict(_runtime_module_scoped)
    assert entities_dict["entities"], "expected at least one entity"
    for ent in entities_dict["entities"]:
        assert "points" in ent
        assert len(ent["points"]) > 0
        assert len(ent["points"][0]) == 3
        assert "lat" in ent["position"] and "lon" in ent["position"]


def test_export_scene_writes_valid_glb_with_points_mode(tmp_path, _runtime_module_scoped):
    out_dir = tmp_path / "cesium_output"
    entities_dict = geo_export.export_scene(_runtime_module_scoped, output_dir=str(out_dir))
    assert entities_dict["entities"]

    entities_json = out_dir / "entities.json"
    tileset_json = out_dir / "tileset.json"
    scene_glb = out_dir / "scene.glb"
    assert entities_json.exists() and entities_json.stat().st_size > 0
    assert tileset_json.exists() and tileset_json.stat().st_size > 0
    assert scene_glb.exists() and scene_glb.stat().st_size > 0

    json.loads(entities_json.read_text())   # must parse
    tileset = json.loads(tileset_json.read_text())
    assert "root" in tileset and "transform" in tileset["root"]

    gltf_json = _read_glb_json_chunk(scene_glb.read_bytes())
    modes = [prim["mode"] for mesh in gltf_json["meshes"] for prim in mesh["primitives"]]
    assert 0 in modes, "expected at least one POINTS-mode (mode=0) primitive from real wiring"
