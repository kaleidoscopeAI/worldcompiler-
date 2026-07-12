from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import get_orchestrator
from app.core.orchestrator import WorldCompilerOrchestrator
from app.schemas.vision import VisionAnalyzeResponse

router = APIRouter(prefix="/v1/vision", tags=["vision"])


@router.post("/analyze", response_model=VisionAnalyzeResponse)
async def analyze_vision(
    file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
    orchestrator: WorldCompilerOrchestrator = Depends(get_orchestrator),
) -> VisionAnalyzeResponse:
    try:
        if file is not None:
            return orchestrator.analyze_vision(
                image_bytes=await file.read(),
                filename=file.filename,
            )
        if image_url:
            return orchestrator.analyze_vision(image_url=image_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raise HTTPException(status_code=400, detail="Provide an upload file or image_url.")
