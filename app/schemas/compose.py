from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.chat import ChatAnalyzeResponse
from app.schemas.health import TraceMetadata
from app.schemas.knowledge import DefinitionResponse
from app.schemas.vision import VisionAnalyzeResponse


class ComposeRequest(BaseModel):
    text: str = Field(min_length=1)
    tone: str = Field(default="supportive")
    image_url: str | None = None
    knowledge_terms: list[str] = Field(default_factory=list)


class ComposeResponse(BaseModel):
    chat: ChatAnalyzeResponse
    vision: VisionAnalyzeResponse | None = None
    knowledge: list[DefinitionResponse] = Field(default_factory=list)
    trace: TraceMetadata
