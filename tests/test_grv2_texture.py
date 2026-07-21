"""Tests for grv2_runtime/texture.py — the skin, applied over the wiring.

Live network search is real (LexForge's DuckDuckGo scrape), but nothing
here depends on it succeeding -- consistent with the rest of this repo's
test suite never requiring live network. `texture_for`'s offline/no-results
fallback path is what's actually asserted.
"""
from grv2_runtime import texture


def test_color_from_hash_is_deterministic():
    a = texture.color_from_hash("bear")
    b = texture.color_from_hash("bear")
    assert a == b


def test_color_from_hash_differs_across_words():
    a = texture.color_from_hash("bear")
    b = texture.color_from_hash("river")
    assert a != b


def test_color_from_hash_in_valid_range():
    for word in ("bear", "river", "mountain", ""):
        r, g, b = texture.color_from_hash(word)
        for c in (r, g, b):
            assert 0.0 <= c <= 1.0


def test_texture_for_falls_back_when_search_returns_nothing(monkeypatch):
    monkeypatch.setattr(texture, "search_images", lambda query, max_results=8: [])
    entry = texture.texture_for("bear", context="forest")
    assert entry.source == "hash_fallback"
    assert entry.color == texture.color_from_hash("bear")


def test_texture_for_uses_image_color_when_available(monkeypatch):
    monkeypatch.setattr(texture, "search_images",
                        lambda query, max_results=8: [{"url": "http://example.invalid/x.jpg"}])
    monkeypatch.setattr(texture, "analyze_image_colors",
                        lambda url: [(0.1, 0.2, 0.3), (0.4, 0.5, 0.6)])
    entry = texture.texture_for("bear")
    assert entry.source == "image"
    assert entry.color == (0.1, 0.2, 0.3)
    assert entry.image_url == "http://example.invalid/x.jpg"


def test_analyze_image_colors_handles_bad_url_gracefully():
    # No network mocking needed: an invalid/unreachable URL must be caught
    # and return None, never raise.
    assert texture.analyze_image_colors("http://this.does.not.exist.invalid/x.jpg") is None
