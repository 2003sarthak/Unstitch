"""Overlay detection with Gemini 2.5 Flash. Implements the `VisionDetector` port.

This is the one place an LLM is used, and it is used for the one thing only a
model can do: look at a frame and say *"that band of text is a burned-in
subtitle, that box is a product pop-up, that mark is a TikTok watermark"*, and
read what they say. Where the cuts are, when an overlay starts and stops, and how
to erase it are all handled by tools that are exact - because a model asked to
produce frame-accurate timestamps is strictly worse than measuring them.

Three decisions shape this adapter:

**The schema is forced, not requested.** `response_schema` makes the API itself
guarantee well-formed JSON in our shape, so there is no parsing of prose, no
regex over a code fence, and no "the model returned markdown today" failure
class. Malformed output stops being a runtime concern.

**Frames are batched.** Gemini's free tier bills per *request*, not per image, so
six frames go into each call. Twenty-four frames therefore cost four requests,
which is what makes ~60 de-edits a day fit in the free tier. Each element carries
its `frame_index` back so the batch can be unpicked.

**Requests are sequential.** The free tier allows about ten per minute, and
firing four at once from several concurrent jobs is the reliable way to hit a
429. Four sequential requests over a couple of seconds is well inside the limit.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.domain.errors import ProcessingError
from app.domain.models import BBox, Detection, OverlayKind, SampledFrame

log = logging.getLogger(__name__)

#: Gemini reports boxes on a fixed 0-1000 grid, in [ymin, xmin, ymax, xmax]
#: order. Asking for its native convention rather than ours gets better
#: geometry; the conversion is one function and happens here at the boundary.
_GRID = 1000.0

_PROMPT = """\
You are analysing a single frame from a short-form social video (TikTok, Reels,
YouTube Shorts) so that overlaid graphics can be removed and re-edited.

Identify every element that was ADDED ON TOP of the footage in an editor. For
each one, return its bounding box, the text it contains, and its type.

Report ONLY overlays. Do NOT report:
- text that is part of the filmed world (a shop sign, a product label held on
  camera, a t-shirt print, a street sign)
- faces, people, or objects
- the footage itself

Element types:
- caption: subtitles transcribing speech, usually centred and low in the frame,
  often word-by-word or in short phrases
- text_overlay: headline, hook, call-to-action, list item, or sticker text added
  by the editor
- image_popup: an inserted picture, product shot, screenshot, meme or inset
  video pasted over the footage
- watermark: a platform logo or @handle identifying the poster or the app
- ui_chrome: fake or real interface furniture drawn over the video, such as a
  like/comment bar, a progress bar, or a mock message notification

Be precise with boxes: cover the full extent of the element including any
background pill, box or shadow behind the text, but do not include surrounding
footage. If the frame contains no overlays at all, return an empty list.
"""


class _Element(BaseModel):
    """One overlay as the model reports it. A wire format, not a domain model.

    Kept separate from `Detection` on purpose: this shape is dictated by what is
    convenient to ask a model for, and it should be free to change with the
    prompt without disturbing anything downstream.
    """

    frame_index: int = Field(description="Index of the frame this element appears in")
    box_2d: list[int] = Field(description="[ymin, xmin, ymax, xmax], each 0-1000")
    kind: OverlayKind
    text: str = Field(default="", description="Text content, or empty for images")
    confidence: float = Field(default=0.8, description="0-1 certainty this is an overlay")


class _Response(BaseModel):
    elements: list[_Element]


class GeminiVisionDetector:
    """Batched, schema-constrained frame analysis."""

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gemini-2.5-flash",
        batch_size: int = 6,
        max_retries: int = 3,
        min_confidence: float = 0.0,
    ) -> None:
        if not api_key:
            # Reaching here means DI bound the wrong adapter; the settings layer
            # is supposed to degrade to the stub long before this.
            raise ValueError("GeminiVisionDetector requires an API key")
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._min_confidence = min_confidence

    async def detect(self, frames: Sequence[SampledFrame]) -> list[Detection]:
        batches = [
            frames[i : i + self._batch_size] for i in range(0, len(frames), self._batch_size)
        ]
        detections: list[Detection] = []
        for number, batch in enumerate(batches, start=1):
            elements = await self._analyse(batch, number, len(batches))
            detections.extend(self._to_detections(elements, batch))

        log.info(
            "gemini found %d element(s) across %d frame(s) in %d request(s)",
            len(detections),
            len(frames),
            len(batches),
        )
        return detections

    # --- request -------------------------------------------------------------

    async def _analyse(
        self, batch: Sequence[SampledFrame], number: int, total: int
    ) -> list[_Element]:
        contents = self._build_contents(batch)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=_Response,
            # Detection should be reproducible: the same frame must not yield a
            # different box on a retry, or the tracker sees phantom movement.
            temperature=0.0,
        )

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model, contents=contents, config=config
                )
                parsed = response.parsed
                if isinstance(parsed, _Response):
                    return parsed.elements
                # `parsed` is None when the model returned nothing usable -
                # a safety block, or a truncated response.
                log.warning("gemini batch %d/%d returned no parsable payload", number, total)
                return []
            except Exception as exc:
                if attempt == self._max_retries or not _is_retryable(exc):
                    raise ProcessingError(
                        "vision analysis failed. If this is a quota limit, either wait "
                        "for the free tier to reset or set VISION_PROVIDER=stub."
                    ) from exc
                # Exponential backoff: the failure this is nearly always for is
                # a per-minute rate limit, which clears on its own.
                delay = 2.0**attempt
                log.warning(
                    "gemini batch %d/%d failed (attempt %d/%d), retrying in %.0fs: %s",
                    number,
                    total,
                    attempt,
                    self._max_retries,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)
        return []

    def _build_contents(self, batch: Sequence[SampledFrame]) -> list[types.Part]:
        """Interleave a label with each image.

        The model has to attribute every element to a frame, and images alone
        carry no index. Labelling each one immediately before its bytes is what
        makes `frame_index` reliable in the response.
        """
        parts: list[types.Part] = [types.Part.from_text(text=_PROMPT)]
        for local_index, frame in enumerate(batch):
            parts.append(types.Part.from_text(text=f"Frame {local_index}:"))
            parts.append(
                types.Part.from_bytes(data=Path(frame.path).read_bytes(), mime_type="image/jpeg")
            )
        return parts

    # --- translation ---------------------------------------------------------

    def _to_detections(
        self, elements: Sequence[_Element], batch: Sequence[SampledFrame]
    ) -> list[Detection]:
        detections: list[Detection] = []
        for element in elements:
            if not 0 <= element.frame_index < len(batch):
                # A hallucinated index would otherwise silently attach an
                # overlay to the wrong moment in the video.
                log.warning("gemini returned out-of-range frame_index %d", element.frame_index)
                continue
            if element.confidence < self._min_confidence:
                continue
            box = _to_bbox(element.box_2d)
            if box is None:
                continue
            detections.append(
                Detection(
                    t_s=batch[element.frame_index].t_s,
                    bbox=box,
                    kind=element.kind,
                    text=element.text.strip(),
                    confidence=max(0.0, min(1.0, element.confidence)),
                )
            )
        return detections


def _to_bbox(box_2d: Sequence[int]) -> BBox | None:
    """Convert `[ymin, xmin, ymax, xmax]` on a 0-1000 grid into a normalised box.

    Returns `None` rather than raising for a degenerate box: one bad rectangle in
    a batch of six frames should cost that rectangle, not the whole job.
    """
    if len(box_2d) != 4:
        return None
    ymin, xmin, ymax, xmax = (v / _GRID for v in box_2d)
    # Models occasionally emit the corners the other way round.
    if xmax < xmin:
        xmin, xmax = xmax, xmin
    if ymax < ymin:
        ymin, ymax = ymax, ymin
    try:
        return BBox(x=xmin, y=ymin, w=xmax - xmin, h=ymax - ymin)
    except ValueError:  # collapsed to zero area after clamping
        return None


def _is_retryable(exc: Exception) -> bool:
    """Rate limits and transient server errors are worth waiting out.

    A bad API key or a malformed request is not - retrying those three times
    just makes the user wait longer for the same failure.
    """
    message = str(exc).lower()
    retryable = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "503",
        "unavailable",
        "500",
        "internal",
    )
    permanent = ("api key", "permission", "401", "403", "invalid")
    if any(token in message for token in permanent):
        return False
    return any(token in message for token in retryable)
