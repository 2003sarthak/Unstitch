"""Snapping model-estimated boxes onto the pixels they actually describe.

This adapter exists because of a measured defect, not a theoretical one. Running
the real Gemini path against a clip with known element positions showed boxes
that were consistently short:

    caption    model x 0.050-0.500    actual x 0.100-0.680
    headline   model x 0.100-0.900    actual x 0.109-0.998
    watermark  model x 0.650-0.850    actual x 0.660-0.890

Every coordinate the model returned was a multiple of ten, which is the tell: it
is estimating a plausible rectangle, not measuring one. Prompting for precision
did not help - an accuracy-focused rewrite moved the caption box by 0.02 and made
the watermark worse. The limitation is in what the model does, not in how it was
asked, and under-coverage is the expensive direction: a caption erased to 70% of
its width is a visible failure, while a mask a few percent too large is not.

So the fix follows the same principle as the rest of this project. The model is
kept for what only it can do - *"that is a burned-in subtitle and it reads this"*
- and the part with an exact answer is measured instead of guessed, exactly as
scene cuts are measured by PySceneDetect rather than guessed by an LLM.

The measurement: inside a search window around the model's box, isolate strokes
with a morphological gradient, threshold with Otsu, close them into contiguous
runs, and take the extent of the components that overlap what the model pointed
at. The result on the clip above:

    caption    refined x 0.092-0.686
    headline   refined x 0.107-1.000
    watermark  refined x 0.656-0.896
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

import cv2
import numpy as np

from app.domain.models import BBox, Detection, SampledFrame

log = logging.getLogger(__name__)

#: How far beyond the model's box to look, as a fraction of that box's own size.
#: Generous, because under-estimation is the failure mode being corrected - but
#: not unbounded, or a caption would be free to annex the whole frame.
_SEARCH_MARGIN = 0.6

#: Refusal threshold. If the measured region is more than this many times the
#: area the model proposed, the measurement has almost certainly latched onto
#: background texture rather than the overlay, and the model's box is kept. A
#: slightly small mask beats one covering half the video.
_MAX_GROWTH = 6.0

#: Wide and short: joins characters into words and words into a line, without
#: bridging to a separate overlay stacked above or below it.
_CLOSE_KERNEL = (25, 9)

#: Measurement is repeated from its own result until the box stops moving.
#: One pass is not enough, because the search window is sized from the box being
#: corrected: a box covering half a caption can only see a little past its own
#: edge, so it recovers part of the remainder and stops short. Feeding the wider
#: box back in lets the window grow with it, and the sequence converges within a
#: pass or two on anything real. The growth cap is applied against the *original*
#: box throughout, so iterating cannot creep past the limit one pass at a time.
_MAX_PASSES = 3

#: Two boxes this close are the same box; further passes would only jitter.
_CONVERGED = 0.005


class OpenCvBoxRefiner:
    """Refines detection boxes against the frames they came from."""

    def __init__(
        self,
        *,
        search_margin: float = _SEARCH_MARGIN,
        max_growth: float = _MAX_GROWTH,
        max_passes: int = _MAX_PASSES,
    ) -> None:
        self._search_margin = search_margin
        self._max_growth = max_growth
        self._max_passes = max_passes

    async def refine(
        self, detections: Sequence[Detection], frames: Sequence[SampledFrame]
    ) -> list[Detection]:
        if not detections:
            return []
        by_time = {round(frame.t_s, 3): frame.path for frame in frames}
        # OpenCV work is CPU-bound and would otherwise stall the event loop for
        # every other job on the box.
        return await asyncio.to_thread(self._refine_all, list(detections), by_time)

    def _refine_all(
        self, detections: list[Detection], by_time: dict[float, str]
    ) -> list[Detection]:
        # Frames are read once and shared, since several detections normally come
        # from the same frame.
        cache: dict[str, np.ndarray | None] = {}
        refined: list[Detection] = []
        adjusted = 0

        for detection in detections:
            path = by_time.get(round(detection.t_s, 3))
            image = cache.setdefault(path, cv2.imread(path, cv2.IMREAD_GRAYSCALE) if path else None)
            if image is None:
                refined.append(detection)
                continue

            box = self._refine_one(image, detection.bbox)
            if box is not detection.bbox:
                adjusted += 1
            refined.append(detection.model_copy(update={"bbox": box}))

        log.info("refined %d of %d detection box(es)", adjusted, len(detections))
        return refined

    def _refine_one(self, gray: np.ndarray, bbox: BBox) -> BBox:
        """Measure repeatedly until the box stops moving."""
        current = bbox
        for _ in range(self._max_passes):
            candidate = self._measure(gray, current)
            if candidate is None or candidate.area > self._max_growth * bbox.area:
                break
            settled = _is_close(candidate, current)
            current = candidate
            if settled:
                break
        return current

    def _measure(self, gray: np.ndarray, bbox: BBox) -> BBox | None:
        height, width = gray.shape[:2]
        x0, y0 = int(bbox.x * width), int(bbox.y * height)
        x1, y1 = int(bbox.right * width), int(bbox.bottom * height)
        box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)

        sx0 = max(0, int(x0 - self._search_margin * box_w))
        sx1 = min(width, int(x1 + self._search_margin * box_w))
        sy0 = max(0, int(y0 - self._search_margin * box_h))
        sy1 = min(height, int(y1 + self._search_margin * box_h))
        window = gray[sy0:sy1, sx0:sx1]
        if window.size == 0:
            return None

        gradient = cv2.morphologyEx(
            window, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        )
        _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        connected = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, _CLOSE_KERNEL)
        )
        count, _, stats, _ = cv2.connectedComponentsWithStats(connected, 8)

        # Only components overlapping what the model pointed at. Without this the
        # window's other contents - a face, a logo, the next caption down - would
        # be swept into the same box.
        mx0, my0, mx1, my1 = x0 - sx0, y0 - sy0, x1 - sx0, y1 - sy0
        parts = [
            (cx, cy, cx + cw, cy + ch)
            for cx, cy, cw, ch, _ in (stats[i] for i in range(1, count))
            if cx < mx1 and cx + cw > mx0 and cy < my1 and cy + ch > my0
        ]
        if not parts:
            return None

        rx0 = min(p[0] for p in parts) + sx0
        ry0 = min(p[1] for p in parts) + sy0
        rx1 = max(p[2] for p in parts) + sx0
        ry1 = max(p[3] for p in parts) + sy0

        try:
            return BBox(
                x=rx0 / width,
                y=ry0 / height,
                w=(rx1 - rx0) / width,
                h=(ry1 - ry0) / height,
            )
        except ValueError:  # collapsed to zero area
            return None


def _is_close(a: BBox, b: BBox) -> bool:
    return (
        abs(a.x - b.x) < _CONVERGED
        and abs(a.y - b.y) < _CONVERGED
        and abs(a.right - b.right) < _CONVERGED
        and abs(a.bottom - b.bottom) < _CONVERGED
    )


class NullBoxRefiner:
    """Returns detections untouched.

    Bound when `REFINE_BOXES=false`, so the refinement can be turned off and the
    difference shown side by side without the pipeline growing a conditional.
    """

    async def refine(
        self, detections: Sequence[Detection], frames: Sequence[SampledFrame]
    ) -> list[Detection]:
        return list(detections)
