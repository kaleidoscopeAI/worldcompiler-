"""Tests for grv2_runtime/wiring.py — the structural/anatomical layer.

Tests that intend to exercise the neural (last-resort) tier pass
allow_llm=False explicitly -- otherwise they'd depend on whether
ANTHROPIC_API_KEY happens to be set in the environment running them,
which is exactly the kind of live-network dependency this repo's test
suite never allows. The LLM tier itself is tested separately below with
llm_gcode.generate_gcode_via_llm monkeypatched, same pattern as
test_grv2_texture.py's search_images mocking.
"""
import numpy as np
import pytest

from grv2_runtime import llm_gcode, wiring


@pytest.mark.parametrize("word", sorted(wiring.GCODE_LIBRARY.keys()))
def test_every_library_word_produces_stable_points(word):
    """Every word in the library must produce a non-empty, deterministic
    point cloud -- the wiring is looked up once and never regenerated
    differently for the same word."""
    bank = wiring.WiringBank()
    a = bank.recall(word)
    b = bank.recall(word)
    assert a.node_count > 0
    assert a.points.shape == (a.node_count, 3)
    np.testing.assert_array_equal(a.points, b.points), "same word must give identical wiring"


def test_two_independent_banks_agree():
    """Determinism across processes/instances, not just within one bank's cache."""
    a = wiring.WiringBank().recall("bear")
    b = wiring.WiringBank().recall("bear")
    np.testing.assert_array_equal(a.points, b.points)


def test_empty_word_falls_back_to_default():
    bank = wiring.WiringBank()
    empty = bank.recall("   ")
    default = bank.recall("default")
    np.testing.assert_array_equal(empty.points, default.points)
    assert empty.source == "gcode"


def test_atlas_tier_gives_real_shape_not_default():
    """A word absent from GCODE_LIBRARY but present in atlas_csg.ATLAS
    (e.g. 'human') must get its own real CSG-derived skeleton, not the
    generic default blob -- this was the actual bug: callers pass whole
    entity labels, and before the atlas/neural tiers existed, anything
    that wasn't one of 12 exact single-word matches silently collapsed."""
    bank = wiring.WiringBank()
    human = bank.recall("human")
    default = bank.recall("default")
    assert human.source == "atlas"
    assert human.node_count > 0
    assert not np.array_equal(human.points, default.points)
    # determinism within the atlas tier too
    human2 = wiring.WiringBank().recall("human")
    np.testing.assert_array_equal(human.points, human2.points)


def test_atlas_tier_matches_token_inside_a_multiword_label():
    """Entity labels are whole phrases (e.g. 'a lone figure on the hill'),
    not single clean words -- recall() must tokenize and find 'figure'/
    'hill' rather than falling straight to the neural tier."""
    bank = wiring.WiringBank()
    phrase = bank.recall("a lone figure standing on the hill")
    figure = bank.recall("figure")
    assert phrase.source == "atlas"
    np.testing.assert_array_equal(phrase.points, figure.points)


def test_neural_tier_for_words_with_no_match_at_all():
    bank = wiring.WiringBank(allow_llm=False)
    unknown = bank.recall("xyzzy_not_a_real_concept")
    default = bank.recall("default")
    assert unknown.source == "neural"
    assert unknown.node_count > 0
    assert not np.array_equal(unknown.points, default.points)
    # deterministic across independent banks
    unknown2 = wiring.WiringBank(allow_llm=False).recall("xyzzy_not_a_real_concept")
    np.testing.assert_array_equal(unknown.points, unknown2.points)
    # different unknown phrases get different sculpted shapes
    other = bank.recall("a completely different unmatched phrase")
    assert not np.array_equal(unknown.points, other.points)


def test_allow_llm_false_never_calls_the_llm_tier(monkeypatch):
    """The hard override: even if the environment happens to have
    ANTHROPIC_API_KEY set, allow_llm=False must skip straight to neural."""
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("llm_gcode should never be called when allow_llm=False")
    monkeypatch.setattr(llm_gcode, "generate_gcode_via_llm", _fail_if_called)

    bank = wiring.WiringBank(allow_llm=False)
    entry = bank.recall("qzjxklm_unmatched_9000")
    assert entry.source == "neural"


def test_retrieval_reuses_llm_entry_for_a_near_duplicate_phrasing(monkeypatch):
    """The dictionary-retrieval tier's actual job: once 'zorblex' has been
    resolved via the LLM tier, a near-identical later phrasing of the same
    word should be served from the dictionary, not cost a second LLM call.
    A fabricated word, not a real one like the earlier drafts of this test
    used ("castle") -- once definition_compiler.py became a live tier,
    'castle' legitimately composes real geometry from its own WordNet
    definition (one hypernym hop to 'house') before ever reaching the LLM
    tier this test means to exercise, so a real word can no longer be
    trusted to reach here. A fabricated word has no WordNet synsets at
    all, so it's guaranteed to fall through every free tier first."""
    monkeypatch.setattr(llm_gcode, "generate_gcode_via_llm",
                        lambda word, context="": wiring.gcode_default())

    bank = wiring.WiringBank(allow_llm=True)
    first = bank.recall("zorblex")
    assert first.source == "llm"

    # Same exact subject token, different surrounding phrase -- must hit
    # the llm/retrieved cache, not issue a second LLM call.
    monkeypatch.setattr(llm_gcode, "generate_gcode_via_llm",
                        lambda word, context="": (_ for _ in ()).throw(
                            AssertionError("must not call the LLM again for the same subject")))
    second = bank.recall("a small zorblex nearby")  # "zorblex" stays the longest token
    np.testing.assert_array_equal(first.points, second.points)


def test_retrieval_never_fires_for_genuinely_unrelated_words():
    bank = wiring.WiringBank(allow_llm=False)
    entry = bank.recall("qzjx_completely_unrelated_klm")
    assert entry.source == "neural", (
        "a genuinely unrelated word must not be misclassified as a near-duplicate "
        "of something already in the dictionary")


def test_retrieval_rejects_wolf_woods_style_prefix_coincidence():
    """Regression test for a real false positive caught via a live server
    run: 'wolf' scored 0.956 cosine similarity against 'woods' (well above
    _RETRIEVAL_SIM_THRESHOLD) purely from a shared 'wo' prefix -- a wolf is
    not a tree. Retrieval must reject this because 'woods' is not a
    substring of 'wolf' (and vice versa), even though the embedder alone
    would have said yes."""
    bank = wiring.WiringBank(allow_llm=False)
    entry = bank.recall("wolf")
    assert entry.source != "retrieved"
    assert entry.word != "woods"


@pytest.mark.parametrize("plural,singular", [
    ("trees", "tree"), ("mountains", "mountain"), ("clouds", "cloud"),
])
def test_plural_gcode_words_reuse_the_singular_entry(plural, singular):
    bank = wiring.WiringBank(allow_llm=False)
    a = bank.recall(plural)
    b = bank.recall(singular)
    assert a.source == "gcode"
    np.testing.assert_array_equal(a.points, b.points)


def test_plural_atlas_word_reuses_the_singular_entry():
    bank = wiring.WiringBank(allow_llm=False)
    a = bank.recall("crystals")
    b = bank.recall("crystal")
    assert a.source == "atlas"
    np.testing.assert_array_equal(a.points, b.points)


def test_inflection_candidates_never_crashes_on_short_or_odd_input():
    for word in ("", "a", "ss", "es", "ing", "xyz's"):
        cands = wiring._inflection_candidates(word)
        assert word in cands


def test_llm_tier_used_when_it_succeeds(monkeypatch):
    calls = []

    def _fake_llm(word, context=""):
        calls.append((word, context))
        return wiring.gcode_default()

    monkeypatch.setattr(llm_gcode, "generate_gcode_via_llm", _fake_llm)

    bank = wiring.WiringBank(allow_llm=True)
    # "zorblex", not a real word ("castle" was tried here originally) --
    # since definition_compiler.py became a live tier, real dictionary
    # words can legitimately resolve for free before ever reaching the LLM
    # tier this test means to exercise. See the near-duplicate-phrasing
    # test above for the same reasoning.
    entry = bank.recall("a majestic zorblex on the horizon")
    assert entry.source == "llm"
    assert entry.node_count > 0
    assert len(calls) == 1
    # cache hit: recalling the same phrase must not call the LLM again
    bank.recall("a majestic zorblex on the horizon")
    assert len(calls) == 1


def test_llm_tier_falls_back_to_neural_when_llm_returns_none(monkeypatch):
    monkeypatch.setattr(llm_gcode, "generate_gcode_via_llm", lambda word, context="": None)
    bank = wiring.WiringBank(allow_llm=True)
    # Every content word here is fabricated (no WordNet synsets at all),
    # so none can accidentally resolve via definition_compiler's tier
    # before reaching the LLM/neural tiers this test exercises -- a real
    # sentence risks a real word composing for free instead.
    entry = bank.recall("zorblex quennorf tantivorous splegwhat")
    assert entry.source == "neural"


def test_points_are_already_y_up():
    """wiring._build swaps G-code's Z-up convention to glTF/Cesium's y-up --
    for the 'mountain' generator (which climbs to z=22 in G-code space) the
    tallest axis of the returned points should be index 1 (y), not index 2."""
    bank = wiring.WiringBank()
    entry = bank.recall("mountain")
    spans = entry.points.max(axis=0) - entry.points.min(axis=0)
    assert spans[1] >= spans[2], "expected the y axis to carry the greater vertical span after the swap"


@pytest.mark.parametrize("word,expected_source", [
    ("bear", "gcode"), ("human", "atlas"), ("totally_unmatched_xyz", "neural"),
])
def test_thermal_cost_is_populated_in_valid_range(word, expected_source):
    bank = wiring.WiringBank(allow_llm=False)
    entry = bank.recall(word)
    assert entry.source == expected_source
    assert 0.0 <= entry.thermal_cost <= 1.0
    # deterministic, like everything else about a WiringEntry
    entry2 = wiring.WiringBank(allow_llm=False).recall(word)
    assert entry.thermal_cost == pytest.approx(entry2.thermal_cost)


def test_parse_gcode_and_voxelize_smoke():
    raw = wiring.parse_gcode(wiring.gcode_default())
    assert raw.shape[1] == 3
    assert len(raw) > 0
    vox = wiring.voxelize(wiring.normalize_pts(raw), R=16)
    assert vox.shape == (16, 16, 16)
    assert vox.sum() > 0
