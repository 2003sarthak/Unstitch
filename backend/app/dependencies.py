"""Dependency wiring. The only module that knows which adapter is which.

Everything else in the application talks to Protocols. This file is where they
are bound to concrete implementations, which means the entire "swap Gemini for
the stub" or "swap the in-memory store for Redis" decision is visible in one
screen rather than scattered across the call sites that happen to need it.

The container is built once at startup and lives on `app.state`. Constructing it
per request would rebuild an ffmpeg wrapper and a Gemini client on every poll -
and the job runner in particular *must* be a singleton, since it owns the worker
pool that outlives any single request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from fastapi import Request

from app.adapters.editor_ffmpeg import FfmpegVideoEditor
from app.adapters.ingest_ytdlp import YtDlpIngestor
from app.adapters.sampler_ffmpeg import FfmpegFrameSampler
from app.adapters.scenes_pyscenedetect import PySceneDetectDetector
from app.adapters.store_memory import InMemoryJobStore
from app.adapters.vision_gemini import GeminiVisionDetector
from app.adapters.vision_stub import StubVisionDetector
from app.config import Settings
from app.domain.models import VisionProvider
from app.domain.ports import JobStore, VisionDetector
from app.infra.ffmpeg import Ffmpeg
from app.infra.workspace import WorkspaceManager
from app.services.job_runner import JobRunner, WorkspaceSweeper
from app.services.pipeline import Pipeline

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Container:
    """Everything the request handlers need, assembled once."""

    settings: Settings
    workspaces: WorkspaceManager
    store: JobStore
    runner: JobRunner
    sweeper: WorkspaceSweeper


def build_vision_detector(settings: Settings) -> VisionDetector:
    """Choose the detector. The one branch that the stub adapter exists for.

    Note that it reads `effective_vision_provider`, not the requested one: asking
    for Gemini with no key degrades to the stub in `config.py`, and honouring
    that here is what lets the app boot and run with no secrets at all.
    """
    if settings.effective_vision_provider is VisionProvider.GEMINI:
        return GeminiVisionDetector(
            settings.gemini_api_key,
            model=settings.gemini_model,
            batch_size=settings.vision_batch_size,
            min_confidence=settings.min_detection_confidence,
        )
    return StubVisionDetector()


def build_pipeline(settings: Settings) -> Pipeline:
    ffmpeg = Ffmpeg(
        ffmpeg_path=settings.ffmpeg_path,
        ffprobe_path=settings.ffprobe_path,
    )
    return Pipeline(
        ingestor=YtDlpIngestor(
            settings.ytdlp_cookies_file,
            max_video_seconds=settings.max_video_seconds,
        ),
        scene_detector=PySceneDetectDetector(
            threshold=settings.scene_threshold,
            min_scene_seconds=settings.min_scene_seconds,
        ),
        sampler=FfmpegFrameSampler(
            ffmpeg,
            interval_s=settings.frame_sample_interval_s,
            max_frames=settings.max_frames,
        ),
        vision=build_vision_detector(settings),
        editor=FfmpegVideoEditor(
            ffmpeg,
            target_height=settings.target_height,
            max_video_seconds=settings.max_video_seconds,
        ),
        vision_provider=settings.effective_vision_provider,
        iou_threshold=settings.track_iou_threshold,
        max_gap_s=settings.track_max_gap_s,
        min_detection_confidence=settings.min_detection_confidence,
        frame_interval_s=settings.frame_sample_interval_s,
        mask_padding_pct=settings.mask_padding_pct,
    )


def build_container(settings: Settings) -> Container:
    workspaces = WorkspaceManager(settings.media_root_path)
    store: JobStore = InMemoryJobStore()
    runner = JobRunner(
        pipeline=build_pipeline(settings),
        store=store,
        workspaces=workspaces,
        max_concurrent=settings.max_concurrent_jobs,
    )
    sweeper = WorkspaceSweeper(
        store=store, workspaces=workspaces, ttl_minutes=settings.job_ttl_minutes
    )
    return Container(
        settings=settings,
        workspaces=workspaces,
        store=store,
        runner=runner,
        sweeper=sweeper,
    )


def get_container(request: Request) -> Container:
    """FastAPI dependency: hand routes the container built at startup.

    Routes depend on *this*, never on a concrete adapter, so a test can swap the
    whole container by assigning to `app.state` and every route follows.
    """
    return request.app.state.container
