"""Tests for grv2_runtime/wiring_store.py -- the persistent word dictionary."""
import numpy as np

from grv2_runtime.wiring_store import WiringStore


def _entry(word, n=5, source="atlas", cost=0.1):
    return {"word": word, "points": np.arange(n * 3, dtype=np.float32).reshape(n, 3),
           "node_count": n, "source": source, "thermal_cost": cost}


def test_round_trips_through_save_and_load(tmp_path):
    store = WiringStore(str(tmp_path / "dict"))
    entries = {"atlas:crystal": _entry("crystal"), "llm:castle": _entry("castle", source="llm")}
    store.save(entries)

    loaded = store.load()
    assert set(loaded.keys()) == set(entries.keys())
    for key, original in entries.items():
        got = loaded[key]
        assert got["word"] == original["word"]
        assert got["node_count"] == original["node_count"]
        assert got["source"] == original["source"]
        assert got["thermal_cost"] == original["thermal_cost"]
        np.testing.assert_array_equal(got["points"], original["points"])


def test_load_on_missing_store_returns_empty_dict(tmp_path):
    store = WiringStore(str(tmp_path / "does_not_exist_yet"))
    assert store.load() == {}


def test_load_on_corrupt_store_returns_empty_dict_not_raise(tmp_path):
    d = tmp_path / "dict"
    d.mkdir()
    (d / "points.npz").write_bytes(b"not actually an npz file")
    (d / "meta.json").write_text("{not valid json")
    store = WiringStore(str(d))
    assert store.load() == {}


def test_save_overwrites_rather_than_merges(tmp_path):
    store = WiringStore(str(tmp_path / "dict"))
    store.save({"atlas:crystal": _entry("crystal")})
    store.save({"atlas:sphere": _entry("sphere")})

    loaded = store.load()
    assert set(loaded.keys()) == {"atlas:sphere"}


def test_words_with_odd_characters_survive_as_npz_keys(tmp_path):
    store = WiringStore(str(tmp_path / "dict"))
    tricky_key = "llm:a majestic castle on the horizon"
    store.save({tricky_key: _entry("a majestic castle on the horizon", source="llm")})
    loaded = store.load()
    assert tricky_key in loaded
