"""Scene detection via PySceneDetect.

Implements the `SceneDetector` port.

Why not an LLM: cut detection is a measurement, not a judgement. `ContentDetector`
compares consecutive frames in HSV space and fires when the delta crosses a
threshold - it is frame-accurate, deterministic, free, and identical on every
run. A model asked to guess timestamps is slower, costs quota, and is worse at
the one thing that has an exact answer. The vision budget is better spent on the
question only a model can answer, which is *what* is on screen.

PySceneDetect is synchronous and CPU-bound, so detection runs in a worker
thread; without that, one job's scene pass would stall the event loop and every
other request with it.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from app.domain.errors import ProcessingError
from app.domain.models import Scene

log = logging.getLogger(__name__)


class PySceneDetectDetector:
    """Finds cuts with HSV content-delta thresholding."""

    def __init__(self, threshold: float = 27.0, min_scene_seconds: float = 0.6) -> None:
        self._threshold = threshold
        self._min_scene_seconds = min_scene_seconds

    async def detect(self, video: Path) -> list[Scene]:
        cuts, duration_s = await asyncio.to_thread(self._detect_sync, video)
        scenes = build_scenes(cuts, duration_s, self._min_scene_seconds)
        log.info("%s: %d scene(s) over %.1fs", video.name, len(scenes), duration_s)
        return scenes

    def _detect_sync(self, video: Path) -> tuple[list[tuple[float, float]], float]:
        try:
            stream = open_video(str(video))
            manager = SceneManager()
            # `min_scene_len` is in frames, and suppressing flash-frames inside
            # the detector is cheaper than merging them afterwards - though
            # `build_scenes` still merges, because a cut one frame under the
            # threshold would otherwise survive.
            fps = stream.frame_rate or 30.0
            manager.add_detector(
                ContentDetector(
                    threshold=self._threshold,
                    min_scene_len=max(1, int(self._min_scene_seconds * fps)),
                )
            )
            manager.detect_scenes(stream)
            duration_s = stream.duration.seconds if stream.duration else 0.0
            cuts = [(start.seconds, end.seconds) for start, end in manager.get_scene_list()]
            return cuts, duration_s
        except Exception as exc:
            raise ProcessingError(f"scene detection failed for {video.name}") from exc


def build_scenes(
    cuts: list[tuple[float, float]], duration_s: float, min_scene_seconds: float
) -> list[Scene]:
    """Turn raw cut boundaries into the scene list the rest of the app expects.

    Pure, so the awkward cases are testable without decoding a video - and they
    are most of the value here:

    * **Never returns an empty list.** Footage with no cuts is one scene, not
      zero. Callers should not each have to special-case unedited video.
    * **Short scenes are merged backwards.** A one-frame flash between two cuts
      is a compression artefact or a transition, not a shot worth a thumbnail.
    * **Coverage is contiguous.** Scenes tile the whole video with no gaps, so
      the timeline strip cannot show a hole.
    """
    if duration_s <= 0:
        return []
    if not cuts:
        return [Scene(index=0, start_s=0.0, end_s=duration_s)]

    merged: list[list[float]] = []
    for start, end in cuts:
        if merged and (end - start) < min_scene_seconds:
            merged[-1][1] = end  # too short to stand alone; extend the previous shot
        else:
            merged.append([start, end])

    # A short *first* scene has no predecessor to merge into, so it absorbs the
    # one after it instead - otherwise the video would start with a 0.2s shot.
    if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_scene_seconds:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    merged[0][0] = 0.0  # detection can start a frame late; the first scene owns it
    merged[-1][1] = duration_s  # and the last one runs to the end
    return [
        Scene(index=i, start_s=round(start, 3), end_s=round(end, 3))
        for i, (start, end) in enumerate(merged)
    ]
