from __future__ import annotations

import time
from dataclasses import dataclass

from app.core.config import Settings
from app.modules.knowledge.providers import DictionaryProvider, KnowledgeLookupResult
from app.modules.nlp.service import NLPAnalysis, NLPService
from app.modules.policy.engine import PolicyEngine, PolicyResult
from app.modules.vision.service import OpenCVVisionAnalyzer, VisionAnalysis
from app.schemas.chat import ChatAnalyzeResponse
from app.schemas.compose import ComposeRequest, ComposeResponse
from app.schemas.health import TraceMetadata
from app.schemas.knowledge import DefinitionEntry, DefinitionResponse
from app.schemas.vision import ColorSummary, VisionAnalyzeResponse


@dataclass
class WorldCompilerOrchestrator:
    settings: Settings
    nlp_service: NLPService
    vision_analyzer: OpenCVVisionAnalyzer
    dictionary_provider: DictionaryProvider
    policy_engine: PolicyEngine

    def analyze_chat(self, text: str, tone: str) -> ChatAnalyzeResponse:
        nlp_result = self.nlp_service.analyze(text)
        policy_result = self.policy_engine.apply(text=text, nlp_result=nlp_result, tone=tone)
        return _build_chat_response(nlp_result=nlp_result, policy_result=policy_result)

    def analyze_vision(
        self,
        *,
        image_bytes: bytes | None = None,
        image_url: str | None = None,
        filename: str | None = None,
    ) -> VisionAnalyzeResponse:
        vision_result = self.vision_analyzer.analyze(
            image_bytes=image_bytes,
            image_url=image_url,
            filename=filename,
        )
        return _build_vision_response(vision_result)

    def define_term(self, term: str) -> DefinitionResponse:
        return _build_definition_response(self.dictionary_provider.define(term))

    def compose(self, request: ComposeRequest) -> ComposeResponse:
        timings_ms: dict[str, float] = {}
        warnings: list[str] = []
        modules_used: list[str] = []

        start = time.perf_counter()
        chat = self.analyze_chat(request.text, request.tone)
        timings_ms["nlp_policy"] = _elapsed_ms(start)
        modules_used.extend(["nlp", "policy"])
        warnings.extend(chat.warnings)

        knowledge_results: list[DefinitionResponse] = []
        if request.knowledge_terms:
            modules_used.append("knowledge")
            start = time.perf_counter()
            knowledge_results = [self.define_term(term) for term in request.knowledge_terms]
            timings_ms["knowledge"] = _elapsed_ms(start)
            for result in knowledge_results:
                warnings.extend(result.warnings)

        vision: VisionAnalyzeResponse | None = None
        if request.image_url:
            modules_used.append("vision")
            start = time.perf_counter()
            vision = self.analyze_vision(image_url=request.image_url)
            timings_ms["vision"] = _elapsed_ms(start)
            warnings.extend(vision.warnings)

        return ComposeResponse(
            chat=chat,
            vision=vision,
            knowledge=knowledge_results,
            trace=TraceMetadata(
                modules_used=modules_used,
                timings_ms=timings_ms,
                warnings=warnings,
            ),
        )


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def _build_chat_response(
    *,
    nlp_result: NLPAnalysis,
    policy_result: PolicyResult,
) -> ChatAnalyzeResponse:
    return ChatAnalyzeResponse(
        intent=nlp_result.intent,
        sentiment=nlp_result.sentiment,
        empathy_tags=nlp_result.empathy_tags,
        supportiveness_level=policy_result.supportiveness_level,
        risk_level=policy_result.risk_level,
        safe_response_draft=policy_result.safe_response_draft,
        disclaimers=policy_result.disclaimers,
        warnings=policy_result.warnings,
    )


def _build_vision_response(vision_result: VisionAnalysis) -> VisionAnalyzeResponse:
    color = None
    if vision_result.dominant_color is not None:
        color = ColorSummary(
            r=vision_result.dominant_color[0],
            g=vision_result.dominant_color[1],
            b=vision_result.dominant_color[2],
        )
    return VisionAnalyzeResponse(
        source=vision_result.source,
        width=vision_result.width,
        height=vision_result.height,
        edge_density=vision_result.edge_density,
        dominant_color=color,
        faces_detected=vision_result.faces_detected,
        object_hints=vision_result.object_hints,
        cv_available=vision_result.cv_available,
        warnings=vision_result.warnings,
    )


def _build_definition_response(result: KnowledgeLookupResult) -> DefinitionResponse:
    return DefinitionResponse(
        term=result.term,
        source=result.source,
        fallback_used=result.fallback_used,
        warnings=result.warnings,
        entries=[
            DefinitionEntry(
                part_of_speech=entry.part_of_speech,
                definition=entry.definition,
                example=entry.example,
                synonyms=entry.synonyms,
            )
            for entry in result.entries
        ],
    )
