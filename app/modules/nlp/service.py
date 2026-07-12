from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

_TOKEN_RE = re.compile(r"[A-Za-z']+")

_POSITIVE_WORDS = {
    "love",
    "hope",
    "thanks",
    "grateful",
    "excited",
    "happy",
    "calm",
}
_NEGATIVE_WORDS = {
    "sad",
    "lonely",
    "afraid",
    "anxious",
    "hurt",
    "angry",
    "overwhelmed",
    "upset",
}
_SUPPORT_WORDS = {
    "help",
    "support",
    "stuck",
    "cope",
    "feel",
    "problem",
    "advice",
}


@dataclass(frozen=True)
class NLPAnalysis:
    tokens: list[str]
    intent: str
    sentiment: str
    empathy_tags: list[str]
    uncertainty: float


class NLPService(Protocol):
    def analyze(self, text: str) -> NLPAnalysis: ...


class HeuristicNLPService:
    """Lightweight BERT-inspired adapter with future model hook points."""

    def preprocess(self, text: str) -> str:
        return " ".join(text.strip().split()).lower()

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_RE.findall(text.lower())

    def analyze(self, text: str) -> NLPAnalysis:
        cleaned = self.preprocess(text)
        tokens = self.tokenize(cleaned)
        token_set = set(tokens)

        if not tokens:
            return NLPAnalysis(
                tokens=[],
                intent="unknown",
                sentiment="neutral",
                empathy_tags=["clarification_needed"],
                uncertainty=0.95,
            )

        if cleaned.endswith("?") or token_set & {"what", "why", "how", "when"}:
            intent = "question"
        elif token_set & _SUPPORT_WORDS:
            intent = "support_request"
        elif token_set & {"hello", "hi", "hey"}:
            intent = "greeting"
        else:
            intent = "inform"

        positive_hits = len(token_set & _POSITIVE_WORDS)
        negative_hits = len(token_set & _NEGATIVE_WORDS)
        if negative_hits > positive_hits:
            sentiment = "negative"
        elif positive_hits > negative_hits:
            sentiment = "positive"
        else:
            sentiment = "neutral"

        empathy_tags: list[str] = []
        if token_set & _SUPPORT_WORDS:
            empathy_tags.append("support-seeking")
        if sentiment == "negative":
            empathy_tags.append("reassurance-needed")
        if sentiment == "positive":
            empathy_tags.append("encouragement-ready")
        if not empathy_tags:
            empathy_tags.append("context-light")

        uncertainty = 0.2
        if len(tokens) < 3 or intent == "unknown":
            uncertainty = 0.7
        elif not (token_set & (_POSITIVE_WORDS | _NEGATIVE_WORDS | _SUPPORT_WORDS)):
            uncertainty = 0.45

        return NLPAnalysis(
            tokens=tokens,
            intent=intent,
            sentiment=sentiment,
            empathy_tags=empathy_tags,
            uncertainty=uncertainty,
        )
