"""FastAPI application factory.

Deliberately a factory rather than a module-level `app = FastAPI()`: tests build
an isolated instance with their own `Settings` and their own container, and
nothing is constructed as an import side effect.

The lifespan is where the background machinery starts and - just as importantly -
stops. A worker pool that is not drained on shutdown leaves ffmpeg subprocesses
orphaned, which on a small Space is the difference between a clean restart and a
container that comes back already loaded.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.errors import install_error_handlers
from app.api.routes_jobs import router as jobs_router
from app.api.routes_media import router as media_router
from app.config import Settings, get_settings
from app.dependencies import Container, build_container

APP_VERSION = "0.1.0"

log = logging.getLogger("unstitch")


class Health(BaseModel):
    """Reports the *effective* configuration, not the requested one.

    A deployment where `VISION_PROVIDER=gemini` but the secret never reached the
    environment is the most likely production failure here, and it is invisible
    from outside unless the health endpoint admits to it.
    """

    status: str
    version: str
    vision_provider: str
    vision_provider_requested: str
    removal_mode: str
    queued_jobs: int


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    container: Container = app.state.container
    settings = container.settings

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

    await container.runner.start()
    await container.sweeper.start()
    log.info(
        "unstitch %s ready | vision=%s removal=%s workers=%d media_root=%s",
        APP_VERSION,
        settings.effective_vision_provider.value,
        settings.removal_mode.value,
        settings.max_concurrent_jobs,
        settings.media_root_path,
    )
    try:
        yield
    finally:
        await container.sweeper.stop()
        await container.runner.stop()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Unstitch",
        version=APP_VERSION,
        summary="Break a short-form video back into its editable components.",
        lifespan=_lifespan,
    )
    # Stashed on app.state so the lifespan, the routes and a test all read the
    # same instance the factory was handed - injection, not a module-level global.
    app.state.container = build_container(settings)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    install_error_handlers(app)
    app.include_router(jobs_router)
    app.include_router(media_router)

    @app.get("/api/health", response_model=Health, tags=["meta"])
    async def health() -> Health:
        container: Container = app.state.container
        return Health(
            status="ok",
            version=APP_VERSION,
            vision_provider=settings.effective_vision_provider.value,
            vision_provider_requested=settings.vision_provider.value,
            removal_mode=settings.removal_mode.value,
            queued_jobs=container.runner.pending,
        )

    return app


app = create_app()
