from __future__ import annotations

from pydantic import BaseModel, Field


class DefinitionEntry(BaseModel):
    part_of_speech: str | None
    definition: str
    example: str | None
    synonyms: list[str] = Field(default_factory=list)


class DefinitionResponse(BaseModel):
    term: str
    source: str
    fallback_used: bool
    entries: list[DefinitionEntry]
    warnings: list[str] = Field(default_factory=list)
