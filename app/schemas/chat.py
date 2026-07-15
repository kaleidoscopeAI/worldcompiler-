from __future__ import annotations

from pydantic import BaseModel, Field


class ChatAnalyzeRequest(BaseModel):
    text: str = Field(min_length=1)
    tone: str = Field(default="supportive")


class ChatAnalyzeResponse(BaseModel):
    intent: str
    sentiment: str
    empathy_tags: list[str]
    supportiveness_level: str
    risk_level: str
    safe_response_draft: str
    disclaimers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
