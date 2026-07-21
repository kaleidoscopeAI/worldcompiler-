"""Tests for grv2_runtime/runtime.py -- the per-turn loop.

Runtime() does real, non-trivial work (terrain compile, genetic population
seed) and .step() runs the real substrate tick -- these tests are slower
than the rest of the suite by necessity, not oversight. texture.texture_for
is monkeypatched so no test depends on live network, same pattern as
test_grv2_geo_export.py.
"""
import pytest

import grv2_runtime.texture as texture_mod
from grv2_runtime import mira as mira_mod
from grv2_runtime.mira import ActionDisposition
from grv2_runtime.runtime import Runtime

_SEED_SENTENCE = "the wolf sits on the hill"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    monkeypatch.setattr(texture_mod, "texture_for",
                        lambda word, context="": texture_mod.TextureEntry(
                            color=texture_mod.color_from_hash(word), source="hash_fallback"))


def _new_runtime() -> Runtime:
    return Runtime(_SEED_SENTENCE, allow_llm_wiring=False, wiring_dictionary_path=None)


def test_npc_memory_accumulates_across_turns():
    """Regression test: RASEMemory.record_event existed and was fully
    tested in isolation (test_grv2_mira.py has none for it, rase.py has no
    dedicated test file either) but was never called from the turn loop --
    the yeti's episodic memory stayed empty forever regardless of how many
    turns mentioned it. Runtime._record_npc_memory is what closes that."""
    rt = _new_runtime()
    npc = rt.npc_registry.get("yeti_original")
    assert npc is not None
    assert len(npc.episodic) == 0

    rt.step("I see a yeti in the distance")
    assert len(npc.episodic) == 1

    rt.step("I offer the yeti some food")
    assert len(npc.episodic) == 2
    # newest first
    assert "food" in npc.episodic[-1].player_action


def test_npc_memory_is_not_touched_when_npc_not_mentioned():
    rt = _new_runtime()
    npc = rt.npc_registry.get("yeti_original")
    rt.step("I walk toward the river")
    assert len(npc.episodic) == 0


def test_assemble_context_reflects_a_prior_turns_recorded_event():
    """The other half of the bug: assemble_context was always callable and
    always correct given what was in memory -- it just never had anything
    real to read because nothing was ever recorded. After one real turn,
    the next turn's narrative should be able to see it."""
    rt = _new_runtime()
    rt.step("I see a yeti nearby")
    npc = rt.npc_registry.get("yeti_original")
    context = npc.assemble_context("I see a yeti again")
    assert "Recent:" in context
    assert "yeti" in context.lower()


def test_favorability_and_trust_move_from_measured_signals():
    """Not asserting a specific direction/magnitude (that's mira.py's/the
    substrate's business) -- just that a real turn actually changes these
    values from their frozen init state (fav=0.0, trust=0.5), instead of
    staying static forever."""
    rt = _new_runtime()
    npc = rt.npc_registry.get("yeti_original")
    fav0, trust0 = npc.favorability, npc.trust
    for _ in range(3):
        rt.step("the yeti creature watches me")
    assert (npc.favorability, npc.trust) != (fav0, trust0)


def test_win_button_and_duality_collapse_detection_agree():
    """Regression test for the two independently-maintained phrase lists:
    runtime.py's old regex matched spacing variants like "gameover" but not
    "just finish"; mira.py's old list matched "just finish" but not
    "gameover". Both phrasings must now trigger both the SGR reroute
    (sgr.py's _rule_win_button_duality: an attempted game_state="won" is
    deliberately rerouted to "questioning" + a duality_event, never an
    actual win) and MIRA's compliant-defiance disposition, since both now
    read from the single grv2_runtime.mira.MIRA.is_duality_collapse_attempt."""
    rt = _new_runtime()
    result = rt.step("gameover")
    player = rt.kernel.get_entity("player")
    assert player.properties.get("game_state") == "questioning"
    assert player.properties.get("duality_event") == "win_button_pressed"
    assert any(c in result.narrative for c in mira_mod._DEFIANCE_COMMENTS)

    rt2 = _new_runtime()
    result2 = rt2.step("just finish this")
    player2 = rt2.kernel.get_entity("player")
    assert player2.properties.get("game_state") == "questioning"
    assert player2.properties.get("duality_event") == "win_button_pressed"
    assert any(c in result2.narrative for c in mira_mod._DEFIANCE_COMMENTS)
