"""Choosing which frames to spend vision quota on, and extracting them.

Implements the `FrameSampler` port.

This is the module that decides what the app costs to run. Gemini's free tier is
roughly 10 requests/minute and 250/day; at `VISION_BATCH_SIZE=6` frames per
request and `MAX_FRAMES=24`, a video costs four requests, so the free tier is
about sixty de-edits a day. Every frame chosen here is quota spent, and every
frame skipped is an overlay that might be missed.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from app.domain.models import SampledFrame, Scene, VideoMeta
from app.infra.ffmpeg import Ffmpeg

log = logging.getLogger(__name__)


class FfmpegFrameSampler:
    """Extracts frames at scene mid-points plus a fixed interval."""

    def __init__(
        self,
        ffmpeg: Ffmpeg,
        *,
        interval_s: float = 1.5,
        max_frames: int = 24,
        jpeg_quality: int = 3,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._interval_s = interval_s
        self._max_frames = max_frames
        self._jpeg_quality = jpeg_quality

    async def sample(
        self, video: Path, meta: VideoMeta, scenes: Sequence[Scene], dest_dir: Path
    ) -> list[SampledFrame]:
        timestamps = choose_timestamps(meta.duration_s, scenes, self._interval_s, self._max_frames)
        dest_dir.mkdir(parents=True, exist_ok=True)

        frames = [
            SampledFrame(index=i, t_s=t, path=str(dest_dir / f"f{i:04d}.jpg"))
            for i, t in enumerate(timestamps)
        ]
        # Extraction is independent per frame and dominated by seek time, so
        # running them concurrently turns a serial 24-seek walk into a handful
        # of parallel ones. The ffmpeg wrapper is async, so this is free.
        await asyncio.gather(*(self._extract(video, frame) for frame in frames))
        log.info("sampled %d frame(s) from %s", len(frames), video.name)
        return frames

    async def _extract(self, video: Path, frame: SampledFrame) -> None:
        await self._ffmpeg.run(
            [
                "-ss",
                f"{frame.t_s:.3f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                "-q:v",
                str(self._jpeg_quality),
                frame.path,
            ]
        )


def choose_timestamps(
    duration_s: float,
    scenes: Sequence[Scene],
    interval_s: float,
    max_frames: int,
) -> list[float]:
    """Decide which moments to look at. Pure, and the interesting decision here.

    Two sources, combined:

    * **Every scene's mid-point.** A cut is where the picture changes most, so a
      scene is the natural unit of "something new might be on screen". Sampling
      the middle avoids the transition frames at either end, which are often
      cross-faded and unreadable.
    * **A fixed interval on top.** Overlays do not respect scene boundaries - a
      caption can appear halfway through a long shot, and a product pop-up often
      does. Scene mid-points alone would miss both.

    When the two produce more than `max_frames`, the list is thinned *evenly
    across the timeline* rather than truncated. Truncating would spend the whole
    budget on the first third of the video and analyse none of the rest - the
    kind of bug that looks like a bad model rather than a bad sampler.
    """
    if duration_s <= 0 or max_frames <= 0:
        return []

    candidates = {round((s.start_s + s.end_s) / 2, 2) for s in scenes if s.duration_s > 0}
    steps = int(duration_s / interval_s) if interval_s > 0 else 0
    candidates.update(round(i * interval_s, 2) for i in range(steps + 1))

    # Never sample the very last frame: seeking to exactly the duration lands
    # past the end on some containers and produces nothing.
    ordered = sorted(t for t in candidates if 0.0 <= t < duration_s - 0.05)
    if not ordered:
        ordered = [round(duration_s / 2, 2)]

    return _thin_evenly(ordered, max_frames)


def _thin_evenly(values: list[float], limit: int) -> list[float]:
    """Reduce to `limit` items while keeping the spread across the whole list.

    Index arithmetic rather than slicing, so the first and last entries always
    survive - the opening frame and the closing frame are where watermarks and
    end-cards live.
    """
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    step = (len(values) - 1) / (limit - 1)
    return [values[round(i * step)] for i in range(limit)]
