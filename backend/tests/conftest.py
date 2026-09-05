"""Shared fixtures.

The video fixtures are *generated* by ffmpeg rather than committed as binary
files. That keeps the repository text-only, and it puts each clip's exact
properties in front of the reader of the test instead of inside an opaque .mp4.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from app.infra.ffmpeg import Ffmpeg

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


@pytest.fixture
def ffmpeg() -> Ffmpeg:
    """A real ffmpeg wrapper, or a skip.

    Skipping here rather than via a marker on each test means anything that
    needs a binary is opted out automatically just by asking for the fixture -
    there is no second list to keep in sync.
    """
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return Ffmpeg(default_timeout_s=60.0)


async def _synthesise(
    ffmpeg: Ffmpeg, path: Path, *, size: str, duration: float, audio: bool
) -> Path:
    inputs = ["-f", "lavfi", "-i", f"testsrc2=size={size}:rate=30:duration={duration}"]
    if audio:
        inputs += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}"]
    codecs = ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    codecs += ["-c:a", "aac", "-shortest"] if audio else ["-an"]
    await ffmpeg.run([*inputs, *codecs, str(path)])
    return path


@pytest.fixture
async def sample_video(ffmpeg: Ffmpeg, tmp_path: Path) -> Path:
    """2 seconds of 320x240 colour bars, with a 440Hz tone."""
    return await _synthesise(
        ffmpeg, tmp_path / "sample.mp4", size="320x240", duration=2.0, audio=True
    )


@pytest.fixture
async def tall_video(ffmpeg: Ffmpeg, tmp_path: Path) -> Path:
    """A silent 540x960 vertical clip - the shape this app actually targets, and
    tall enough that a 720-line cap has to do something."""
    return await _synthesise(
        ffmpeg, tmp_path / "tall.mp4", size="540x960", duration=1.0, audio=False
    )
