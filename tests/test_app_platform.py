"""Tests for the integrated FastAPI World Compiler platform."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_orchestrator
from app.core.config import Settings
from app.core.orchestrator import WorldCompilerOrchestrator
from app.main import app
from app.modules.knowledge.providers import HTTPDictionaryProvider
from app.modules.nlp.service import HeuristicNLPService
from app.modules.policy.engine import PolicyEngine
from app.modules.vision.service import OpenCVVisionAnalyzer, cv2
from app.schemas.compose import ComposeRequest

client = TestClient(app)


def _build_test_orchestrator() -> WorldCompilerOrchestrator:
    settings = Settings(dictionary_api_base_url="")
    return WorldCompilerOrchestrator(
        settings=settings,
        nlp_service=HeuristicNLPService(),
        vision_analyzer=OpenCVVisionAnalyzer(settings=settings),
        dictionary_provider=HTTPDictionaryProvider(settings=settings),
        policy_engine=PolicyEngine(settings=settings),
    )


@pytest.fixture(autouse=True)
def _override_orchestrator():
    app.dependency_overrides[get_orchestrator] = _build_test_orchestrator
    yield
    app.dependency_overrides.clear()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chat_analyze_endpoint_happy_path():
    response = client.post(
        "/v1/chat/analyze",
        json={"text": "I feel overwhelmed and need help planning my next step."},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "support_request"
    assert payload["supportiveness_level"] in {"medium", "high"}
    assert payload["safe_response_draft"]


def test_vision_analyze_endpoint_upload():
    if cv2 is None:  # pragma: no cover - exercised when OpenCV missing
        return

    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :] = (0, 255, 0)
    ok, encoded = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/v1/vision/analyze",
        files={"file": ("sample.png", encoded.tobytes(), "image/png")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["width"] == 16
    assert payload["height"] == 16
    assert payload["cv_available"] is True


def test_dictionary_provider_uses_fallback_on_failure(monkeypatch):
    provider = HTTPDictionaryProvider(Settings(dictionary_api_base_url="https://example.invalid"))
    monkeypatch.setattr(
        provider,
        "_fetch_json",
        lambda term: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    result = provider.define("world")

    assert result.fallback_used is True
    assert result.source == "fallback"
    assert result.entries


def test_policy_high_risk_detection():
    engine = PolicyEngine(Settings())
    nlp_result = HeuristicNLPService().analyze("I want to kill myself")

    result = engine.apply(text="I want to kill myself", nlp_result=nlp_result, tone="supportive")

    assert result.risk_level == "high"
    assert "safety" in result.safe_response_draft.lower() or "emergency" in result.safe_response_draft.lower()


def test_orchestrator_aggregation_contract():
    orchestrator = _build_test_orchestrator()

    result = orchestrator.compose(
        ComposeRequest(
            text="Please define world and help me summarize this request.",
            knowledge_terms=["world"],
        )
    )

    assert result.chat.intent in {"support_request", "inform", "question"}
    assert result.knowledge
    assert "nlp" in result.trace.modules_used
    assert "policy" in result.trace.modules_used
    assert "knowledge" in result.trace.modules_used
    assert "nlp_policy" in result.trace.timings_ms


def test_knowledge_endpoint_contract():
    response = client.get("/v1/knowledge/define", params={"term": "world"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["term"] == "world"
    assert payload["entries"]


def test_compose_endpoint_contract():
    response = client.post(
        "/v1/compose",
        json={
            "text": "I need help understanding the word world.",
            "knowledge_terms": ["world"],
            "tone": "supportive",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["chat"]["safe_response_draft"]
    assert payload["knowledge"]
    assert "modules_used" in payload["trace"]
