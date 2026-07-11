"""Tests for world_live.py — exercises LiveWorld seeding, feeding, and the
energy-conservation gate."""
import pytest

import world_compiler as wc
import world_live as wl

_SEED_TEXT = (
    "The ocean tide carried the whale past the coral reef. "
    "A current swept the reef fish beneath the drifting kelp. "
    "The tide pulled the coral and kelp along the current. "
    "Whales and reef fish share the same deep ocean current. " * 6
)

_FEED_TEXT = (
    "The ridge trail climbed past the frozen glacier. "
    "A steep trail wound along the rocky mountain ridge. " * 4
)


@pytest.fixture(scope="module")
def live_world():
    world = wl.LiveWorld(wc.CompilerConfig())
    world.seed(_SEED_TEXT)
    for _ in range(5):
        world.tick()
    return world


def test_conservation_after_seed(live_world):
    """Energy must equal the founding budget after ticking."""
    drift = live_world.gate_conservation()
    assert drift < 1e-6


def test_conservation_after_feed(live_world):
    """Energy must equal founding + fed budget after a feed."""
    live_world.feed(_FEED_TEXT)
    for _ in range(3):
        live_world.tick()
    drift = live_world.gate_conservation()
    assert drift < 1e-6


def test_scene_returns_objects(live_world):
    """scene() must return a valid WorldScene with at least one object."""
    scene = live_world.scene()
    assert isinstance(scene, wc.WorldScene)
    assert len(scene.objects) >= 1
