from __future__ import annotations

from functools import lru_cache

from app.core.orchestrator import WorldCompilerOrchestrator
from app.modules.knowledge.providers import HTTPDictionaryProvider
from app.modules.nlp.service import HeuristicNLPService
from app.modules.policy.engine import PolicyEngine
from app.modules.vision.service import OpenCVVisionAnalyzer


@lru_cache(maxsize=1)
def get_orchestrator() -> WorldCompilerOrchestrator:
    from app.core.config import get_settings

    settings = get_settings()
    return WorldCompilerOrchestrator(
        settings=settings,
        nlp_service=HeuristicNLPService(),
        vision_analyzer=OpenCVVisionAnalyzer(settings=settings),
        dictionary_provider=HTTPDictionaryProvider(settings=settings),
        policy_engine=PolicyEngine(settings=settings),
    )
