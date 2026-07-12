from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.chat import router as chat_router
from app.api.routes.compose import router as compose_router
from app.api.routes.health import router as health_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.vision import router as vision_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(vision_router)
app.include_router(knowledge_router)
app.include_router(compose_router)
