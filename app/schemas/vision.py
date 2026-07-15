from __future__ import annotations

from pydantic import BaseModel, Field


class ColorSummary(BaseModel):
    r: int
    g: int
    b: int


class VisionAnalyzeResponse(BaseModel):
    source: str
    width: int | None
    height: int | None
    edge_density: float | None
    dominant_color: ColorSummary | None
    faces_detected: int | None
    object_hints: list[str] = Field(default_factory=list)
    cv_available: bool
    warnings: list[str] = Field(default_factory=list)
