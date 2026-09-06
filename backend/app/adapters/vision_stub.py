"""Offline overlay detection. Implements the `VisionDetector` port.

This adapter exists so the product is not hostage to a secret. With it, anyone
can clone the repository, add no API key, and watch the entire pipeline run end
to end - and CI can exercise every stage with no network and no quota.

It is a **heuristic**, and it is honest about that. Burned-in text has a
distinctive signature in edge space: dense, high-contrast strokes that form
horizontal runs against a smoother background. Morphological gradient plus an
Otsu threshold isolates those strokes; closing them horizontally merges
characters into words and words into lines. What it cannot do is *read* the text
or tell a caption from a product name, which is exactly the boundary where the
real vision model earns its keep - and precisely the argument for why this
project uses a model for perception rather than for geometry.

Classification here is positional, not semantic: bottom-third wide bands are
captions, small corner marks are watermarks, and so on. That is a fair
approximation of short-form convention and a poor substitute for understanding.
`JobResult.vision_provider` reports which detector actually ran, so a stub result
is never mistaken for a real analysis.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import cv2
import numpy as np

from app.domain.models import BBox, Detection, OverlayKind, SampledFrame

log = logging.getLogger(__name__)

#: Regions smaller than this fraction of the frame are noise - compression
#: blocks, texture, a highlight on someone's jacket.
_MIN_AREA_FRACTION = 0.0015
#: And anything above this is the scene itself, not an overlay on top of it.
_MAX_AREA_FRACTION = 0.35
#: Text lines are wider than they are tall. This is the single most effective
#: filter, because almost nothing else in a frame has this signature.
_MIN_ASPECT_RATIO = 1.4


class StubVisionDetector:
    """Edge-density text detection with positional classification."""

    def __init__(self, *, max_per_frame: int = 6) -> None:
        self._max_per_frame = max_per_frame

    async def detect(self, frames: Sequence[SampledFrame]) -> list[Detection]:
        # OpenCV releases the GIL for most of this, but it is still CPU-bound
        # work that would otherwise stall the event loop for every other job.
        batches = await asyncio.gather(
            *(asyncio.to_thread(self._detect_one, frame) for frame in frames)
        )
        detections = [d for batch in batches for d in batch]
        log.info("stub detector found %d element(s) in %d frame(s)", len(detections), len(frames))
        return detections

    def _detect_one(self, frame: SampledFrame) -> list[Detection]:
        image = cv2.imread(frame.path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            log.warning("could not read sampled frame %s", frame.path)
            return []

        height, width = image.shape[:2]
        regions = _text_like_regions(image)

        detections = [
            Detection(
                t_s=frame.t_s,
                bbox=BBox(x=x / width, y=y / height, w=w / width, h=h / height),
                kind=_classify(x / width, y / height, w / width, h / height),
                text="",  # a heuristic cannot read; leaving it blank says so
                confidence=score,
            )
            for x, y, w, h, score in regions[: self._max_per_frame]
        ]
        return detections


def _text_like_regions(gray: np.ndarray) -> list[tuple[int, int, int, int, float]]:
    """Find horizontal runs of dense edges, strongest first.

    The pipeline is: gradient to isolate strokes, Otsu to binarise without a
    magic threshold, then a wide-and-short closing kernel to join characters into
    a line while *not* joining two separate captions stacked on top of each
    other - which is why the kernel is 25x3 rather than square.
    """
    height, width = gray.shape[:2]
    frame_area = float(height * width)

    gradient = cv2.morphologyEx(
        gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    connected = cv2.morphologyEx(
        binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))
    )

    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[tuple[int, int, int, int, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_fraction = (w * h) / frame_area
        if not (_MIN_AREA_FRACTION <= area_fraction <= _MAX_AREA_FRACTION):
            continue
        if h == 0 or (w / h) < _MIN_ASPECT_RATIO:
            continue

        # How much of the box is actually edge. Real text fills a large part of
        # its bounding box; a spurious contour around smooth background does not.
        density = float(np.count_nonzero(binary[y : y + h, x : x + w])) / float(w * h)
        if density < 0.08:
            continue
        # Deliberately capped below 1.0: a heuristic should never report the
        # confidence of a model that has actually read the words.
        regions.append((x, y, w, h, round(min(0.75, 0.35 + density), 3)))

    regions.sort(key=lambda r: r[4], reverse=True)
    return regions


def _classify(x: float, y: float, w: float, h: float) -> OverlayKind:
    """Guess the element type from where it sits and how big it is.

    Positional rules that follow short-form convention: subtitles are burned in
    low and wide, platform watermarks are small and cornered, hook text sits high
    in the frame. This is a convention-follower, not an understanding - which is
    the whole reason `vision_gemini` exists.
    """
    centre_y = y + h / 2
    if w < 0.25 and h < 0.12 and (x > 0.6 or x < 0.1) and (y < 0.15 or y > 0.85):
        return OverlayKind.WATERMARK
    if w > 0.45 and centre_y > 0.62:
        return OverlayKind.CAPTION
    if w > 0.3 and centre_y < 0.35:
        return OverlayKind.TEXT_OVERLAY
    if w > 0.25 and h > 0.2:
        return OverlayKind.IMAGE_POPUP
    return OverlayKind.TEXT_OVERLAY
