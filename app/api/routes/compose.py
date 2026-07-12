from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.core.orchestrator import WorldCompilerOrchestrator
from app.schemas.compose import ComposeRequest, ComposeResponse

router = APIRouter(prefix="/v1", tags=["compose"])


@router.post("/compose", response_model=ComposeResponse)
def compose(
    request: ComposeRequest,
    orchestrator: WorldCompilerOrchestrator = Depends(get_orchestrator),
) -> ComposeResponse:
    return orchestrator.compose(request)
