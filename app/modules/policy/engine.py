from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings
from app.modules.nlp.service import NLPAnalysis

_HIGH_RISK_TERMS = {
    "suicide",
    "kill myself",
    "self harm",
    "hurt myself",
    "end my life",
    "overdose",
}
_MEDIUM_RISK_TERMS = {"panic", "can't go on", "hopeless", "worthless"}


@dataclass(frozen=True)
class PolicyResult:
    risk_level: str
    supportiveness_level: str
    safe_response_draft: str
    disclaimers: list[str]
    warnings: list[str]


class PolicyEngine:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def apply(self, *, text: str, nlp_result: NLPAnalysis, tone: str) -> PolicyResult:
        normalized = text.lower()
        risk_level = "low"
        warnings: list[str] = []

        if any(term in normalized for term in _HIGH_RISK_TERMS):
            risk_level = "high"
        elif any(term in normalized for term in _MEDIUM_RISK_TERMS):
            risk_level = "medium"

        supportiveness_level = "low"
        if risk_level == "high" or nlp_result.sentiment == "negative":
            supportiveness_level = "high"
        elif nlp_result.intent == "support_request" or risk_level == "medium":
            supportiveness_level = "medium"

        disclaimers: list[str] = []
        if nlp_result.uncertainty >= 0.6:
            disclaimers.append(self._settings.uncertainty_disclaimer)

        if risk_level == "high":
            warnings.append("high-risk language detected")
            response = (
                "I'm sorry you're going through this. I may be limited, but your safety matters. "
                "Please reach out to a trusted person or local emergency service right now if you might act on these thoughts."
            )
            disclaimers.append("AI support is not a substitute for urgent professional care.")
            return PolicyResult(
                risk_level=risk_level,
                supportiveness_level="high",
                safe_response_draft=response,
                disclaimers=disclaimers,
                warnings=warnings,
            )

        prefix = {
            "supportive": "I hear you, and I'm here to help.",
            "neutral": "Here is a balanced interpretation.",
            "concise": "Short answer:",
        }.get(tone, "I hear you, and I'm here to help.")

        if supportiveness_level == "high":
            body = "It sounds heavy. Consider a small next step, like pausing, breathing, and contacting someone you trust."
        elif supportiveness_level == "medium":
            body = "You may benefit from practical support and a calm, step-by-step response."
        else:
            body = "The request appears low risk and can be handled with a direct response."

        return PolicyResult(
            risk_level=risk_level,
            supportiveness_level=supportiveness_level,
            safe_response_draft=f"{prefix} {body}",
            disclaimers=disclaimers,
            warnings=warnings,
        )
