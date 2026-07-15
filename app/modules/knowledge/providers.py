from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Protocol

from app.core.config import Settings

_LOCAL_FALLBACKS: dict[str, list[dict[str, str]]] = {
    "world": [
        {
            "part_of_speech": "noun",
            "definition": "The earth, its environments, and the shared human context built upon it.",
            "example": "World Compiler aims to reason across signals about the world.",
        }
    ],
    "compiler": [
        {
            "part_of_speech": "noun",
            "definition": "A system that transforms one representation into another in a structured way.",
            "example": "The platform compiles text, images, and lexical knowledge into a unified response.",
        }
    ],
    "empathy": [
        {
            "part_of_speech": "noun",
            "definition": "The ability to understand and respond sensitively to another person's emotional state.",
            "example": "The policy engine adds empathy-aware response guidance.",
        }
    ],
}


@dataclass(frozen=True)
class LexicalEntry:
    part_of_speech: str | None
    definition: str
    example: str | None
    synonyms: list[str]


@dataclass(frozen=True)
class KnowledgeLookupResult:
    term: str
    source: str
    entries: list[LexicalEntry]
    fallback_used: bool
    warnings: list[str]


class DictionaryProvider(Protocol):
    def define(self, term: str) -> KnowledgeLookupResult: ...


class HTTPDictionaryProvider:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def define(self, term: str) -> KnowledgeLookupResult:
        normalized_term = term.strip().lower()
        warnings: list[str] = []
        try:
            payload = self._fetch_json(normalized_term)
            entries = _normalize_dictionary_payload(payload)
            if entries:
                return KnowledgeLookupResult(
                    term=normalized_term,
                    source="http",
                    entries=entries,
                    fallback_used=False,
                    warnings=warnings,
                )
            warnings.append("Dictionary provider returned no normalized entries.")
        except Exception as exc:  # pragma: no cover - network dependent
            warnings.append(f"Dictionary provider unavailable: {exc}")

        fallback_entries = _fallback_entries(normalized_term)
        return KnowledgeLookupResult(
            term=normalized_term,
            source="fallback",
            entries=fallback_entries,
            fallback_used=True,
            warnings=warnings,
        )

    def _fetch_json(self, term: str) -> object:
        if not self._settings.dictionary_api_base_url:
            raise RuntimeError("Dictionary API base URL is not configured.")

        url = f"{self._settings.dictionary_api_base_url.rstrip('/')}/{urllib.parse.quote(term)}"
        attempts = self._settings.dictionary_max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                with urllib.request.urlopen(
                    url,
                    timeout=self._settings.dictionary_timeout_seconds,
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
                last_error = exc
                if attempt < attempts - 1:
                    time.sleep(0.1 * (attempt + 1))

        assert last_error is not None
        raise last_error


def _normalize_dictionary_payload(payload: object) -> list[LexicalEntry]:
    entries: list[LexicalEntry] = []
    if isinstance(payload, list):
        for item in payload:
            entries.extend(_normalize_dictionary_payload(item))
        return entries

    if not isinstance(payload, dict):
        return entries

    if "meanings" in payload and isinstance(payload["meanings"], list):
        for meaning in payload["meanings"]:
            if not isinstance(meaning, dict):
                continue
            part_of_speech = meaning.get("partOfSpeech")
            definitions = meaning.get("definitions", [])
            if not isinstance(definitions, list):
                continue
            for definition in definitions:
                if not isinstance(definition, dict) or not definition.get("definition"):
                    continue
                synonyms = definition.get("synonyms", [])
                entries.append(
                    LexicalEntry(
                        part_of_speech=part_of_speech,
                        definition=str(definition["definition"]),
                        example=str(definition["example"]) if definition.get("example") else None,
                        synonyms=[str(item) for item in synonyms[:5]],
                    )
                )
        return entries

    if payload.get("definition"):
        entries.append(
            LexicalEntry(
                part_of_speech=str(payload.get("part_of_speech")) if payload.get("part_of_speech") else None,
                definition=str(payload["definition"]),
                example=str(payload.get("example")) if payload.get("example") else None,
                synonyms=[str(item) for item in payload.get("synonyms", [])[:5]],
            )
        )
    return entries


def _fallback_entries(term: str) -> list[LexicalEntry]:
    raw_entries = _LOCAL_FALLBACKS.get(
        term,
        [
            {
                "part_of_speech": "noun",
                "definition": f"No external definition was available for '{term}', so this fallback entry preserves the lookup contract.",
                "example": None,
            }
        ],
    )
    return [
        LexicalEntry(
            part_of_speech=entry.get("part_of_speech"),
            definition=entry["definition"],
            example=entry.get("example"),
            synonyms=[],
        )
        for entry in raw_entries
    ]
