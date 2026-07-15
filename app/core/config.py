from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("WORLD_COMPILER_APP_NAME", "World Compiler")
    app_version: str = os.getenv("WORLD_COMPILER_APP_VERSION", "0.1.0")
    dictionary_api_base_url: str = os.getenv(
        "WORLD_COMPILER_DICTIONARY_API_URL",
        "https://api.dictionaryapi.dev/api/v2/entries/en",
    )
    dictionary_timeout_seconds: float = float(
        os.getenv("WORLD_COMPILER_DICTIONARY_TIMEOUT", "1.5")
    )
    dictionary_max_retries: int = int(
        os.getenv("WORLD_COMPILER_DICTIONARY_MAX_RETRIES", "1")
    )
    vision_remote_timeout_seconds: float = float(
        os.getenv("WORLD_COMPILER_VISION_TIMEOUT", "3.0")
    )
    vision_allow_remote_urls: bool = os.getenv(
        "WORLD_COMPILER_ALLOW_REMOTE_VISION_URLS", "true"
    ).lower() in {"1", "true", "yes", "on"}
    uncertainty_disclaimer: str = os.getenv(
        "WORLD_COMPILER_UNCERTAINTY_DISCLAIMER",
        "This AI-generated interpretation may be incomplete and should be verified.",
    )


_SETTINGS = Settings()


def get_settings() -> Settings:
    return _SETTINGS
