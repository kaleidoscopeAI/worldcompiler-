"""grv2_runtime/definition_compiler.py -- bootstraps geometry for arbitrary
English words for free, entirely offline, by resolving each word against
words that ALREADY have real geometry, instead of generating new geometry
per word.

The insight: a dictionary is a graph, not a flat word list. Almost every
word is defined in terms of simpler words ("carburetor" -> a "device" that
"mixes" ...). Given a seed set of "atomic" concepts that already have real
geometry -- GCODE_LIBRARY (hand-authored) and atlas_csg.ATLAS (closed-form
CSG), the two free tiers wiring.py already has -- an arbitrary word's shape
can usually be *composed*, not regenerated, from:

  is-a (hypernym)    -> reuse the nearest resolvable ancestor's shape as a
                         base, walking WordNet's hypernym chain outward
                         from the word until some ancestor's lemma matches
                         an already-known base-tier word.
  part-of (meronym)  -> attach each resolvable part's shape as a smaller
                         sub-cluster near the base shape's surface,
                         deterministically positioned by hashing
                         (base_word, part_word) -- the same "ears attached
                         to a head" pattern wiring.gcode_bear already uses
                         by hand, generalized so it doesn't need a human to
                         write it per word.

No network call, no training, no GPU, no cost: just nltk.corpus.wordnet
(a local, offline, one-time ~10MB download) plus geometry the wiring
tiers already produce. Words with no resolvable ancestor anywhere up the
hypernym chain return None -- the caller (WiringBank) falls through to its
own film_siren tier, exactly as it does today for any unmatched word.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .wiring import WiringBank, WiringEntry

_MAX_HYPERNYM_DEPTH = 8
_MAX_MERONYMS = 4
_SUB_SCALE = 0.32          # attached parts are smaller than the base
_PLACEMENT_FRAC = 0.85     # how far out toward the base's surface a part sits


def _base_tier_entry(bank: "WiringBank", word: str) -> Optional["WiringEntry"]:
    """The free, deterministic, already-known tiers: GCODE_LIBRARY and
    atlas_csg.ATLAS (via bank._entry_for_known_word), plus any word this
    compiler has already resolved earlier in the same run (cached under
    "compiled:{word}"). That second half is what makes this a fixed-point
    compiler rather than a single shallow pass: once "puppy" resolves to
    "animal", "puppy" itself becomes a usable ancestor/part for whatever
    word gets compiled next -- coverage grows every round without the
    52-word hand-built seed set growing at all. Never LLM, retrieval, or
    neural -- this only ever answers "is this word already real geometry?"."""
    entry = bank._entry_for_known_word(word)
    if entry is not None:
        return entry
    return bank._entries.get(f"compiled:{word}")


def _lemma_candidates(lemma_name: str) -> List[str]:
    """'domestic_animal' -> ['domesticanimal', 'animal', 'domestic']. English
    noun compounds are head-final ("domestic animal" is a kind of animal,
    "motor vehicle" is a kind of vehicle), so the last word -- the actual
    head noun -- is tried before earlier modifier words. Whole-lemma-joined
    is tried first since a handful of base-tier words are themselves
    compounds with no separating space in this codebase's vocabulary."""
    whole = lemma_name.replace("_", "").lower()
    parts = [p.lower() for p in lemma_name.split("_") if p]
    head_first = list(reversed(parts))
    out = [whole] + head_first
    seen = set()
    uniq = []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


@dataclass
class ResolutionTrace:
    """Diagnostics for the prototype run -- not needed by WiringBank, only
    by the demo script proving the composition logic actually works before
    it's pointed at the full dictionary."""
    word: str
    resolved: bool
    base_word: Optional[str] = None
    hypernym_hops: Optional[int] = None
    meronyms_attached: Optional[List[str]] = None
    node_count: Optional[int] = None


def _nearest_resolvable_ancestor(bank: "WiringBank", synset) -> Optional[tuple]:
    """BFS up the hypernym tree; returns (WiringEntry, base_word, hops) for
    the first ancestor (closest first) with a lemma that's already a known
    base-tier word. None if nothing up to _MAX_HYPERNYM_DEPTH resolves."""
    frontier = [synset]
    seen_synsets = {synset}
    for hops in range(_MAX_HYPERNYM_DEPTH):
        next_frontier = []
        for s in frontier:
            for hyper in s.hypernyms():
                if hyper in seen_synsets:
                    continue
                seen_synsets.add(hyper)
                for lemma_name in hyper.lemma_names():
                    for cand in _lemma_candidates(lemma_name):
                        entry = _base_tier_entry(bank, cand)
                        if entry is not None:
                            return entry, cand, hops + 1
                next_frontier.append(hyper)
        frontier = next_frontier
        if not frontier:
            break
    return None


def _attach(base_points: np.ndarray, sub_points: np.ndarray,
           base_word: str, part_word: str) -> np.ndarray:
    """Place a deterministically-scaled, deterministically-positioned copy
    of `sub_points` near `base_points`'s surface. Direction and slight
    scale jitter are hashed from (base_word, part_word) -- same word pair
    always attaches the same way, across processes, like every other
    deterministic tier in wiring.py."""
    seed = int.from_bytes(hashlib.blake2b(f"{base_word}:{part_word}".encode(),
                                          digest_size=4).digest(), "big")
    rng = np.random.RandomState(seed)
    theta = float(rng.uniform(0, 2 * math.pi))
    phi = float(rng.uniform(0.15 * math.pi, 0.85 * math.pi))  # avoid exact poles
    direction = np.array([math.sin(phi) * math.cos(theta),
                          math.cos(phi),
                          math.sin(phi) * math.sin(theta)], dtype=np.float32)

    center = base_points.mean(axis=0)
    radius = float(np.linalg.norm(base_points - center, axis=1).max()) if len(base_points) else 1.0
    scale_jitter = float(rng.uniform(0.85, 1.15))

    sub_center = sub_points.mean(axis=0)
    placed = (sub_points - sub_center) * (_SUB_SCALE * scale_jitter)
    placed += center + direction * (radius * _PLACEMENT_FRAC)
    return placed


def compile_word(word: str, bank: "WiringBank", wn_module=None) -> Optional["ResolutionTrace"]:
    """Attempt to resolve `word` via WordNet definition-graph composition.
    Returns a ResolutionTrace with resolved=False (and no entry written
    to `bank`) if nothing up the hypernym chain of any noun synset is a
    known base-tier word. Never raises -- WordNet lookup failures and
    missing synsets are just "not resolvable this way", not errors."""
    from .wiring import WiringEntry  # local import: avoid a cycle at module load

    if wn_module is None:
        from nltk.corpus import wordnet as wn_module

    raw_key = word.lower().strip()
    direct = _base_tier_entry(bank, raw_key)
    if direct is not None:
        return ResolutionTrace(word=raw_key, resolved=True, base_word=raw_key,
                               hypernym_hops=0, meronyms_attached=[],
                               node_count=direct.node_count)

    try:
        synsets = wn_module.synsets(raw_key, pos=wn_module.NOUN) or wn_module.synsets(raw_key)
    except LookupError:
        return ResolutionTrace(word=raw_key, resolved=False)
    if not synsets:
        return ResolutionTrace(word=raw_key, resolved=False)

    # Only ever the word's #1 (most common) sense. WordNet orders synsets
    # by frequency, and trying sense 2/3 on a sense-1 miss sounds tempting
    # but is a trap: "wildcat" sense 1 is an oil well, sense 2 is "a
    # cruelly rapacious person" (resolves to a human figure!), and only
    # sense 3 is the actual animal. Guessing at rarer senses silently
    # produces confidently wrong shapes; failing over to film_siren on a
    # sense-1 miss is the honest outcome.
    for synset in synsets[:1]:
        found = _nearest_resolvable_ancestor(bank, synset)
        if found is None:
            continue
        base_entry, base_word, hops = found

        points = base_entry.points.copy()
        attached: List[str] = []
        for meronym in synset.part_meronyms()[:_MAX_MERONYMS]:
            for lemma_name in meronym.lemma_names()[:1]:
                for cand in _lemma_candidates(lemma_name):
                    part_entry = _base_tier_entry(bank, cand)
                    if part_entry is not None:
                        sub = _attach(base_entry.points, part_entry.points, base_word, cand)
                        points = np.concatenate([points, sub], axis=0)
                        attached.append(cand)
                        break
                if attached and attached[-1] == cand:
                    break

        entry = WiringEntry(word=raw_key, points=points.astype(np.float32),
                            node_count=len(points), source="compiled",
                            thermal_cost=base_entry.thermal_cost)
        bank._remember(f"compiled:{raw_key}", entry)
        return ResolutionTrace(word=raw_key, resolved=True, base_word=base_word,
                               hypernym_hops=hops, meronyms_attached=attached,
                               node_count=entry.node_count)

    return ResolutionTrace(word=raw_key, resolved=False)
