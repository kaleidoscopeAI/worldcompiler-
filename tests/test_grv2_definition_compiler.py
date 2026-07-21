"""Tests for grv2_runtime/definition_compiler.py -- the free, offline,
WordNet-driven dictionary bootstrap compiler.

Network/GPU/cost note: nothing here calls an LLM or the network (WordNet's
corpus is a local, already-downloaded data file) -- these tests exercise
pure graph traversal and geometry composition over data already on disk.
"""
import numpy as np
import pytest

from grv2_runtime import definition_compiler as dc
from grv2_runtime.wiring import WiringBank, WiringEntry


def test_already_atomic_word_resolves_at_hop_zero():
    bank = WiringBank()
    trace = dc.compile_word("bear", bank)
    assert trace.resolved
    assert trace.hypernym_hops == 0
    assert trace.base_word == "bear"


def test_one_hop_hypernym_composes_a_real_shape():
    """'eagle' isn't in GCODE_LIBRARY/atlas_csg, but its WordNet hypernym
    chain reaches 'bird' (which is), one hop up."""
    bank = WiringBank()
    trace = dc.compile_word("eagle", bank)
    assert trace.resolved
    assert trace.base_word == "bird"
    assert trace.hypernym_hops == 1
    entry = bank._entries["compiled:eagle"]
    assert entry.source == "compiled"
    assert entry.node_count > 0
    assert np.isfinite(entry.points).all()


def test_sense_one_only_avoids_a_dangerous_mis_resolution():
    """Regression case found while prototyping: WordNet's sense 1 for
    'wildcat' is an oil well, sense 2 is 'a cruelly rapacious person'
    (which resolves to a human figure!), and only sense 3 is the actual
    animal. Trying sense 2 on a sense-1 miss silently gives a person shape
    to a word about a cat -- confidently wrong, not just imprecise. Only
    ever trying sense 1 means this word correctly falls through to
    film_siren instead (caller's responsibility, not this module's)."""
    bank = WiringBank()
    trace = dc.compile_word("wildcat", bank)
    assert not trace.resolved
    assert "compiled:wildcat" not in bank._entries


def test_atlas_catch_all_words_stay_out_of_the_registry():
    """'wildcat' (sense 1: an oil well) reaches 'artifact' in exactly 4
    hops via well -> excavation -> artifact -- registering 'artifact' as an
    atlas_csg key (it looked like a reasonable generic catch-all) silently
    resolved a word about a cat into a structure blob from its most
    obscure sense. Same danger class as the sense-1 test above, found the
    same way: by actually checking what a 'successful' resolution produced
    instead of trusting the resolved count. Locks in that these stay out."""
    from grv2_runtime import atlas_csg
    assert "artifact" not in atlas_csg.ATLAS
    assert "phenomenon" not in atlas_csg.ATLAS


def test_no_lemma_string_collision_between_unrelated_senses():
    """'irony' (sense 1: sarcasm) -> wit -> a synset whose lemma names are
    message/content/subject_matter/substance -- a completely different
    sense of the word 'substance' than the physical-material one
    atlas_csg.py's 'material' family was meant to cover. Matching is by
    lemma string, not WordNet sense, so the two collided: irony resolved
    into a physical-substance blob. 'material'/'wood' cover what
    'substance' was added for without the collision."""
    bank = WiringBank()
    trace = dc.compile_word("irony", bank)
    assert not trace.resolved


def test_unresolvable_word_does_not_write_to_the_bank():
    bank = WiringBank()
    trace = dc.compile_word("democracy", bank)
    assert not trace.resolved
    assert not any(k.startswith("compiled:democracy") for k in bank._entries)


def test_two_independent_banks_agree():
    a = WiringBank()
    b = WiringBank()
    dc.compile_word("puppy", a)
    dc.compile_word("puppy", b)
    np.testing.assert_array_equal(a._entries["compiled:puppy"].points,
                                  b._entries["compiled:puppy"].points)


def test_thermal_cost_populated_in_valid_range():
    bank = WiringBank()
    dc.compile_word("kitten", bank)
    entry = bank._entries["compiled:kitten"]
    assert 0.0 <= entry.thermal_cost <= 1.0


def test_fixed_point_reuses_a_word_compiled_earlier_in_the_run():
    """The key scaling mechanism: once a word is compiled, it becomes a
    usable ancestor for the next word, without growing the hand-built seed
    set. Proven synthetically here (rather than hunting for a real WordNet
    coincidence) by seeding the cache the way a prior compile_word call
    would have, then confirming _base_tier_entry finds it."""
    bank = WiringBank()
    fake = WiringEntry(word="gadget", points=np.zeros((20, 3), dtype=np.float32),
                       node_count=20, source="compiled", thermal_cost=0.1)
    bank._entries["compiled:gadget"] = fake
    found = dc._base_tier_entry(bank, "gadget")
    assert found is fake


def test_attach_places_sub_shape_near_base_surface_deterministically():
    base = np.random.RandomState(0).normal(size=(200, 3)).astype(np.float32) * 5
    sub = np.random.RandomState(1).normal(size=(50, 3)).astype(np.float32)

    placed_a = dc._attach(base, sub, "head", "ear")
    placed_b = dc._attach(base, sub, "head", "ear")
    np.testing.assert_array_equal(placed_a, placed_b), "same word pair must attach identically"

    placed_other = dc._attach(base, sub, "head", "horn")
    assert not np.array_equal(placed_a, placed_other), "different parts must attach differently"

    base_center = base.mean(axis=0)
    base_radius = float(np.linalg.norm(base - base_center, axis=1).max())
    dist = float(np.linalg.norm(placed_a.mean(axis=0) - base_center))
    assert dist < base_radius * 1.3, "attached part should land near the base's surface, not far outside it"


def test_lemma_candidates_prefers_specific_word_over_modifier():
    cands = dc._lemma_candidates("domestic_animal")
    assert cands.index("animal") < cands.index("domestic")
