"""FastAPI application factory.

Deliberately a factory rather than a module-level `app = FastAPI()`: tests build
an isolated instance with their own `Settings`, and nothing is constructed as an
import side effect.

Routers land here in step 10 of the build order. For now the app exists so that
the configuration, the container and the deployment target can each be verified
independently of the pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import Settings, get_settings

APP_VERSION = "0.1.0"

log = logging.getLogger("unstitch")


class Health(BaseModel):
    """Reports the *effective* configuration, not the requested one.

    A deployment where `VISION_PROVIDER=gemini` but the secret never made it into
    the environment is the single most likely production failure here, and it is
    invisible from the outside unless the health endpoint admits to it.
    """

    status: str
    version: str
    vision_provider: str
    vision_provider_requested: str
    removal_mode: str


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # Created once at boot rather than per job, so a permissions problem on the
    # container's writable volume surfaces at startup instead of mid-render.
    settings.media_root_path.mkdir(parents=True, exist_ok=True)

    if settings.vision_downgraded:
        log.warning(
            "VISION_PROVIDER=%s was requested but GEMINI_API_KEY is empty - "
            "falling back to the offline stub detector. Overlay detection will "
            "be heuristic, not semantic.",
            settings.vision_provider.value,
        )
    log.info(
        "unstitch %s ready | vision=%s removal=%s media_root=%s",
        APP_VERSION,
        settings.effective_vision_provider.value,
        settings.removal_mode.value,
        settings.media_root_path,
    )
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Unstitch",
        version=APP_VERSION,
        summary="Break a short-form video back into its editable components.",
        lifespan=_lifespan,
    )
    # Stashed on app.state so `_lifespan` and the dependency layer read the same
    # instance the factory was handed - injection, not a module-level lookup.
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=Health, tags=["meta"])
    async def health() -> Health:
        return Health(
            status="ok",
            version=APP_VERSION,
            vision_provider=settings.effective_vision_provider.value,
            vision_provider_requested=settings.vision_provider.value,
            removal_mode=settings.removal_mode.value,
        )

    return app


app = create_app()
