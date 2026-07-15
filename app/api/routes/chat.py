from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.dependencies import get_orchestrator
from app.core.orchestrator import WorldCompilerOrchestrator
from app.schemas.chat import ChatAnalyzeRequest, ChatAnalyzeResponse

router = APIRouter(prefix="/v1/chat", tags=["chat"])


@router.post("/analyze", response_model=ChatAnalyzeResponse)
def analyze_chat(
    request: ChatAnalyzeRequest,
    orchestrator: WorldCompilerOrchestrator = Depends(get_orchestrator),
) -> ChatAnalyzeResponse:
    return orchestrator.analyze_chat(request.text, request.tone)
