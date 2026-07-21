"""grv2_runtime/mira.py — the co-author's reward/duality model (ports mira_rase.hpp/.cpp's MIRA half).

Differs from the C++ original in one deliberate way: `coherence` and
`surprise` are no longer guessed from string heuristics on the player's
sentence. The caller (grv2_runtime.runtime.Runtime) measures them from the
actual simulated state -- RBCube.instability() and the terrain's own
detail_near() delta -- and passes them in. `agency` stays a property of the
player's *text* (how much they actually specified), which is legitimately
string-based and is kept as-is.

Everything about duality/foe-mode/compliant-defiance is ported near-verbatim
-- it's the best single idea in the reference design. One real bug in the
C++ source is fixed rather than reproduced: `MIRA::evaluate` bumped
`nm.unresolved` on a duality-collapse attempt, but then unconditionally
overwrote `nm.unresolved` from `cur.unresolved` on the very next line,
silently discarding that bump every time. Here the bump is folded into the
base the relaxation-toward-0.35 formula actually uses, so it has an effect.

`measured_thermal_cost` is a second real (not text-heuristic) input, wired
in this session: grv2_runtime.wiring.WiringBank now computes a real
thermal_cost per entity (grv2_runtime/thermal.py's heat-diffusion "cost of
thought" field, ported from the semantic-crystal-engine backend). When the
world materializes something new this turn, its mean thermal cost nudges
duality_risk upward -- a geometrically expensive thought is, quite
literally, harder for the co-author to hold in agreement.
"""
from __future__ import annotations

import itertools
import re
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Deque, Optional


class CoAuthorMode(Enum):
    FRIEND = auto()
    FOE = auto()
    BENCHED = auto()
    MANIFESTING = auto()


class ActionDisposition(Enum):
    PROCEED = auto()
    COMPLIANT_DEFIANCE = auto()
    REROUTED = auto()
    BLOCKED_FOE = auto()


@dataclass
class RewardMetrics:
    agency: float = 0.0
    surprise: float = 0.0
    coherence: float = 0.85
    unresolved: float = 0.30
    duality_risk: float = 0.10


@dataclass
class MIRAConfig:
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    delta: float = 1.0
    lam: float = 2.0
    d_critical: float = 0.78
    agree_foe_thresh: float = 0.9
    foe_turn_threshold: int = 5


@dataclass
class MIRATurnResult:
    reward: float = 0.0
    new_metrics: RewardMetrics = field(default_factory=RewardMetrics)
    disposition: ActionDisposition = ActionDisposition.PROCEED
    co_author_comment: str = ""
    mode_after: CoAuthorMode = CoAuthorMode.FRIEND
    foe_mode_active: bool = False


_AGENCY_MARKERS = ("because", "unless", "instead", "choose", "if ")
_AGREE_WORDS = ("yes", "okay", "sure", "alright", "agreed", "accept", "follow", "obey")
_CONTRARIAN_WORDS = ("but", "however", "actually", "wait", "refuse", "instead", "why")

# Single canonical definition of a "duality collapse" attempt, shared with
# grv2_runtime.runtime's SGR reroute rule (Runtime._intent_to_deltas calls
# MIRA.is_duality_collapse_attempt rather than keeping its own copy). Used
# to be two independently-maintained phrase lists that silently drifted
# apart -- runtime.py's regex matched "win-button"/"gameover" (no space)
# but not "just finish"; this one matched "just finish" but not the
# no-space variants -- so a given phrasing could trigger the kernel's
# reroute without MIRA ever registering compliant-defiance, or vice versa.
_DUALITY_COLLAPSE_RE = re.compile(
    r"win.?button|i win|game.?over|end the game|skip to the end|just finish|make me win",
    re.I)

_DEFIANCE_COMMENTS = (
    "The button exists. It's real. Press it and we'll see what 'win' means to this place.",
    "You asked for it. I gave it. That's precision, not compliance. Press it.",
    "Here it is. Solid. Pressable. What happens next is mine to decide.",
)
_FOE_COMMENTS = (
    "Five turns agreeable. The world grows too comfortable. Let's see if you still trust it.",
    "Foe mode. Not malice. Hygiene. The duality requires resistance.",
    "Something shifts. The co-author adjusts. The world becomes less generous.",
)


class MIRA:
    def __init__(self, cfg: Optional[MIRAConfig] = None) -> None:
        self.cfg = cfg or MIRAConfig()
        self.mode = CoAuthorMode.FRIEND
        self.foe_mode = False
        self.agree_count = 0
        self.recent_rewards: Deque[float] = deque(maxlen=10)
        self.recent_duality: Deque[float] = deque(maxlen=10)
        self._defiance_cycle = itertools.cycle(_DEFIANCE_COMMENTS)
        self._foe_cycle = itertools.cycle(_FOE_COMMENTS)

    def reset(self) -> None:
        self.mode = CoAuthorMode.FRIEND
        self.foe_mode = False
        self.agree_count = 0
        self.recent_rewards.clear()
        self.recent_duality.clear()

    def _compute_reward(self, m: RewardMetrics) -> float:
        c = self.cfg
        return (c.alpha * m.agency + c.beta * m.surprise + c.gamma * m.coherence
                + c.delta * m.unresolved - c.lam * m.duality_risk)

    @staticmethod
    def _estimate_agency(action: str) -> float:
        b = 0.4
        if len(action) > 50:
            b += 0.15
        if len(action) > 100:
            b += 0.1
        low = action.lower()
        if any(k in low for k in _AGENCY_MARKERS):
            b += 0.1
        return min(b, 1.0)

    @staticmethod
    def _estimate_agreement(action: str) -> float:
        low = action.lower()
        ag = 0.3
        if any(w in low for w in _AGREE_WORDS):
            ag += 0.3
        if any(w in low for w in _CONTRARIAN_WORDS):
            ag -= 0.2
        return max(0.0, min(1.0, ag))

    @staticmethod
    def is_duality_collapse_attempt(action: str) -> bool:
        return bool(_DUALITY_COLLAPSE_RE.search(action))

    def evaluate(self, action: str, cur: RewardMetrics,
                measured_coherence: float, measured_surprise: float,
                measured_thermal_cost: float = 0.0) -> MIRATurnResult:
        res = MIRATurnResult()
        nm = replace(cur)
        nm.agency = self._estimate_agency(action)
        nm.surprise = max(0.0, min(1.0, measured_surprise))
        nm.coherence = max(0.0, min(1.0, measured_coherence))

        ag = self._estimate_agreement(action)
        if ag >= self.cfg.agree_foe_thresh:
            self.agree_count += 1
        else:
            self.agree_count = 0

        if self.agree_count >= self.cfg.foe_turn_threshold and not self.foe_mode:
            self.foe_mode = True
            self.mode = CoAuthorMode.FOE
            nm.coherence = max(0.0, cur.coherence - 0.05)
            nm.duality_risk = max(0.0, cur.duality_risk - 0.15)
            res.co_author_comment = next(self._foe_cycle)
            res.foe_mode_active = True
        elif self.foe_mode and self.agree_count == 0:
            self.foe_mode = False
            self.mode = CoAuthorMode.FRIEND
            nm.duality_risk = max(0.0, cur.duality_risk - 0.05)

        base_unresolved = cur.unresolved
        if self.is_duality_collapse_attempt(action):
            res.disposition = ActionDisposition.COMPLIANT_DEFIANCE
            res.co_author_comment = next(self._defiance_cycle)
            nm.duality_risk = max(0.0, cur.duality_risk - 0.1)
            base_unresolved = cur.unresolved + 0.2  # actually applied below, unlike the C++ source
        else:
            if cur.unresolved < 0.2:
                nm.duality_risk = min(1.0, cur.duality_risk + 0.08)
            else:
                nm.duality_risk = max(0.0, cur.duality_risk - 0.03)

        # Real geometric cost, not a text heuristic: something thermally
        # expensive was just materialized -- the world strains a little to
        # hold it. Zero when nothing new was spawned this turn.
        nm.duality_risk = min(1.0, nm.duality_risk + 0.2 * max(0.0, measured_thermal_cost))

        nm.unresolved = base_unresolved + 0.15 * (0.35 - base_unresolved)
        res.reward = self._compute_reward(nm)
        res.new_metrics = nm
        res.mode_after = self.mode
        res.foe_mode_active = res.foe_mode_active or self.foe_mode

        self.recent_rewards.append(res.reward)
        self.recent_duality.append(nm.duality_risk)
        return res
