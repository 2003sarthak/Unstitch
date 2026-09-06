"""Typed application settings, bound to `backend/.env`.

Every knob in the system is declared here exactly once. Nothing else in the
codebase reads `os.environ`, and no module carries its own default for a value
that appears in `.env.example` - that duplication is how config drifts.

Two deliberate choices worth naming:

1.  **The `.env` path is anchored to this file, not the working directory.**
    `uvicorn app.main:app` from `backend/`, `pytest` from the repo root and the
    Docker `CMD` all have different CWDs; resolving relative to `__file__` means
    all three load the same file.

2.  **Bounds live on the fields.** `Field(ge=..., le=...)` turns a typo like
    `TRACK_IOU_THRESHOLD=50` into a clear validation error at startup instead of
    an overlay tracker that silently matches nothing an hour later.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from app.domain.models import RemovalMode, VisionProvider

# backend/  -- the anchor for .env and for relative MEDIA_ROOT values.
BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Field names are the lowercase form of the `.env` keys.

    pydantic-settings matches case-insensitively, so `MAX_FRAMES` populates
    `max_frames`. Keeping the two lists identical is a hard rule: an undeclared
    key in `.env.example` is a lie to whoever clones the repo.
    """

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        # Ignore rather than forbid: the deployment environment (Hugging Face
        # Spaces) injects its own variables into the process, and none of them
        # are ours to validate.
        extra="ignore",
    )

    # --- Vision ---------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    vision_provider: VisionProvider = VisionProvider.GEMINI

    # --- Ingest ---------------------------------------------------------------
    ytdlp_cookies_file: Path | None = None
    max_upload_mb: int = Field(default=100, gt=0)
    max_video_seconds: int = Field(default=90, gt=0)
    target_height: int = Field(default=720, ge=240, le=1080)

    # --- Scene detection ------------------------------------------------------
    scene_threshold: float = Field(default=27.0, gt=0)
    min_scene_seconds: float = Field(default=0.6, ge=0)

    # --- Frame sampling (this is what governs API quota use) ------------------
    frame_sample_interval_s: float = Field(default=1.5, gt=0)
    max_frames: int = Field(default=24, gt=0)
    vision_batch_size: int = Field(default=6, gt=0)

    # --- Overlay tracking -----------------------------------------------------
    track_iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    track_max_gap_s: float = Field(default=3.5, ge=0.0)
    min_detection_confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    refine_boxes: bool = True

    # --- Removal --------------------------------------------------------------
    removal_mode: RemovalMode = RemovalMode.DELOGO
    mask_padding_pct: float = Field(default=2.0, ge=0.0, le=25.0)

    # --- Runtime --------------------------------------------------------------
    media_root: Path = Path("./work")
    job_ttl_minutes: int = Field(default=120, gt=0)
    max_concurrent_jobs: int = Field(default=2, gt=0)
    # `NoDecode` suppresses pydantic-settings' automatic JSON parsing of complex
    # types so the raw string reaches `_split_csv_origins` below. Without it the
    # settings *source* raises before any validator gets a chance to run.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"]
    )
    log_level: str = "INFO"
    ffmpeg_path: str = "ffmpeg"
    ffprobe_path: str = "ffprobe"

    # --- Normalisation --------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_csv_origins(cls, value: object) -> object:
        """Accept `a,b` as well as a real list.

        Complex-typed settings are JSON-decoded by default, so the natural
        `CORS_ORIGINS=http://localhost:5173,https://unstitch.vercel.app` would be
        rejected outright. Comma-separated is the readable form, and this is the
        one setting that has to change at deploy time - so it is worth the
        `NoDecode` annotation on the field above.
        """
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("ytdlp_cookies_file", mode="before")
    @classmethod
    def _blank_path_is_none(cls, value: object) -> object:
        """`YTDLP_COOKIES_FILE=` (the documented default) means "no cookies",
        not "a file named empty string"."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    # --- Derived values -------------------------------------------------------

    @property
    def media_root_path(self) -> Path:
        """`MEDIA_ROOT` resolved against `backend/`, so `./work` is the same
        directory no matter where the process was launched from."""
        return (BACKEND_DIR / self.media_root).resolve()

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def effective_vision_provider(self) -> VisionProvider:
        """The provider we can actually honour, as opposed to the one requested.

        Asking for Gemini without a key is a configuration mistake, and there are
        three defensible responses: refuse to start, run degraded, or start and
        fail at detection time. The third is strictly worst - the error surfaces
        minutes into a job, far from its cause - so it is off the table.

        We choose to run degraded. The whole point of the stub adapter is that a
        reviewer can clone the repo, add no secrets, and still watch the pipeline
        run end to end; booting into an error page would defeat that. The cost is
        that a genuine misconfiguration in production degrades quietly, which is
        why `main.py` logs a warning at startup and the health endpoint reports
        the effective provider rather than the requested one.

        To flip this to fail-fast instead, raise from a `model_validator` here.
        """
        if self.vision_provider is VisionProvider.GEMINI and not self.gemini_api_key:
            return VisionProvider.STUB
        return self.vision_provider

    @property
    def vision_downgraded(self) -> bool:
        """True when we fell back to the stub. Reported, never silent."""
        return self.effective_vision_provider is not self.vision_provider


@lru_cache
def get_settings() -> Settings:
    """Cached accessor - the `.env` file is read once per process.

    This is the only global in the codebase. Everything downstream receives
    `Settings` by injection (`app/dependencies.py`), so tests construct their own
    instance instead of monkey-patching the environment.
    """
    return Settings()
