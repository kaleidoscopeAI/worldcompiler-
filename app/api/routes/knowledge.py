from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.dependencies import get_orchestrator
from app.core.orchestrator import WorldCompilerOrchestrator
from app.schemas.knowledge import DefinitionResponse

router = APIRouter(prefix="/v1/knowledge", tags=["knowledge"])


@router.get("/define", response_model=DefinitionResponse)
def define_term(
    term: str = Query(min_length=1),
    orchestrator: WorldCompilerOrchestrator = Depends(get_orchestrator),
) -> DefinitionResponse:
    return orchestrator.define_term(term)
