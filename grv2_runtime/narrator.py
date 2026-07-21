"""grv2_runtime/narrator.py -- turns real, already-measured turn signals
into prose instead of a debug dump.

Nothing here invents information. Every sentence is a phrasing of a value
that already exists elsewhere in the system (reward, curiosity-accepted
entities and their real `implication` tags, terrain delta, NPC
favorability/trust/recent memory) -- this module's only job is wording,
deterministic and free (no LLM call, no network), matching the rest of
this repo's free-tier-first discipline. There is no LLM-based narrator
anywhere else in grv2_runtime either; before this module, `_build_narrative`
and RASEMemory.assemble_context's raw telemetry format WAS what the
player saw -- ids like "motif_echo_04d89ff7" and "NPC:yeti_original
fav:0.00 trust:0.50" printed directly, not summarized into anything.
"""
from __future__ import annotations

from typing import List, Optional

from . import sgr as sgr_mod
from .rase import RASEMemory

# implication slug (from curiosity.py's _mk calls) -> a natural phrase
# describing what just appeared. Deliberately a fixed table, not a
# generator: every phrase here corresponds to a real implication tag that
# already exists in curiosity.py, so this can never describe something
# that didn't actually happen.
_IMPLICATION_PHRASES = {
    "footing_risk": "the ground here feels uncertain underfoot",
    "prior_event": "something happened here before you arrived",
    "not_alone": "you're not alone",
    "fire_was_here": "there was fire here recently",
    "something_within": "the shadows feel deeper than they should",
    "claimed_territory": "this place has been claimed by something",
    "stamina_critical": "you can hear labored breathing nearby",
    "world_remembers": "the world remembers something you once said",
}


def _humanize(slug: str) -> str:
    """Fallback for any implication/id slug not in the table above --
    'ice_shelf' -> 'ice shelf'. Never invents content, only reformats an
    identifier that's already real."""
    return slug.replace("_", " ").replace("-", " ").strip()


def describe_curiosity(accepted: List[sgr_mod.Entity]) -> Optional[str]:
    """Prose for whatever the curiosity system just filled in, using each
    entity's own real implication tag or (for motif echoes) its label --
    never the raw internal id."""
    if not accepted:
        return None
    phrases = []
    for ent in accepted:
        implication = ent.properties.get("implication")
        if implication and implication in _IMPLICATION_PHRASES:
            phrases.append(_IMPLICATION_PHRASES[implication])
        elif implication:
            phrases.append(_humanize(implication))
        else:
            # NOT ent.properties.get("label"): for motif-echo entities
            # (world_live's genetic population), "label" is an arbitrary
            # truncated substring of previously-fed text, not curated
            # content -- e.g. "the yeti and offer it fo". Showing that
            # verbatim as "you notice X" reads as nonsense, not detail.
            # The implication table above is the only trustworthy prose
            # source; this is the last-resort fallback, not a preference.
            phrases.append(_humanize(ent.id))
    seen = set()
    phrases = [p for p in phrases if not (p in seen or seen.add(p))]
    if len(phrases) == 1:
        return f"Something more: {phrases[0]}."
    return "Something more: " + "; ".join(phrases[:-1]) + f"; and {phrases[-1]}."


def describe_terrain(delta: int) -> Optional[str]:
    """Same real signal _build_narrative already used, phrased as one of
    a few real directions rather than a raw '+240 deep chunks' count --
    the exact number is still available structurally on TurnResult if
    something downstream wants it, it just doesn't need to be prose."""
    if not delta:
        return None
    if delta > 40:
        return "The ground shifts dramatically beneath the request."
    if delta > 0:
        return "The ground answers, quietly, in your favor."
    if delta < -40:
        return "The ground resists -- something pulls away."
    return "The ground answers, but reluctantly."


def describe_npc(npc: RASEMemory) -> str:
    """The yeti's current demeanor, from its own real favorability/trust
    state and (if any exists yet) its most recent actual memory -- never
    the player's own action echoed back, never a raw stat dump."""
    fav, trust = npc.favorability, npc.trust

    if trust < 0.2:
        trust_phrase = "watches you like it's ready to bolt"
    elif trust < 0.4:
        trust_phrase = "keeps its distance, wary"
    elif trust < 0.6:
        trust_phrase = "holds its ground, watching"
    elif trust < 0.8:
        trust_phrase = "seems to have stopped bracing for a threat"
    else:
        trust_phrase = "seems entirely at ease with you"

    if fav < -0.3:
        fav_clause = ", something like resentment in its posture"
    elif fav < -0.05:
        fav_clause = ", still unconvinced by you"
    elif fav <= 0.05:
        fav_clause = ""
    elif fav < 0.3:
        fav_clause = ", something almost warm in it"
    else:
        fav_clause = ", plainly glad you're here"

    sentence = f"The yeti {trust_phrase}{fav_clause}."

    recent = npc.recall_recent(1)
    if recent:
        # event is stored as "'<action text>' -> DISPOSITION reward=X.XX"
        # (see Runtime._record_npc_memory) -- take just the quoted action.
        last_action = recent[0].event.split(" -> ")[0].strip().strip("'")
        sentence += f" It hasn't forgotten: {last_action}."
    return sentence
