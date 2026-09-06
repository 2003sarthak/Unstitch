"""The data shapes the whole application agrees on.

These pydantic models are the API contract, the pipeline's working vocabulary
and the mirror for the TypeScript types - one definition, no DTO layer. That is
why `Field(description=...)` is used liberally: it lands in the OpenAPI schema,
so the frontend contract documents itself.

Two conventions run through everything here and are worth stating once:

**Geometry is normalised.** Every box is expressed in 0..1 of frame width and
height, never pixels. The video is downscaled during normalisation, scene clips
are re-encoded, and the browser renders the preview at whatever size the layout
gives it - a pixel box would be wrong in at least two of those three places. A
normalised box is correct in all of them, and converts to pixels only at the
moment it meets ffmpeg.

**Time is seconds, not frames.** Frame indices are only meaningful alongside a
frame rate, and the sources here range from 24 to 60 fps - some of them
variable. Seconds survive the re-encodes; frame numbers would not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

# =============================================================================
#  Vocabulary
# =============================================================================


class VisionProvider(StrEnum):
    """Which implementation of the `VisionDetector` port to bind.

    The stub exists so the entire pipeline is runnable - and testable in CI -
    with no API key and no network.
    """

    GEMINI = "gemini"
    STUB = "stub"


class RemovalMode(StrEnum):
    """How detected overlays are erased from the frame.

    Selectable per job rather than fixed, because the right choice depends on
    the footage: `delogo` interpolates from the mask border and looks best over
    busy video, while `boxblur` never invents detail and is the more honest
    choice when the point is to show that something *was* there.

    OpenCV inpainting was considered and dropped rather than stubbed. It needs a
    per-frame Python loop at roughly 50ms a frame - about two minutes for a 90s
    clip - which does not fit a free-tier request, and it cannot share the single
    render pass the other two modes use. A `RemovalMode.INPAINT` that quietly did
    something else would be worse than not offering it.

    These are `StrEnum`s rather than `Literal[...]` aliases for two reasons:
    FastAPI renders them as a proper `enum` in the OpenAPI schema, and a
    `StrEnum` member formats as its bare value - `f"{RemovalMode.DELOGO}"` is
    `"delogo"`, which is what an ffmpeg filter string needs. The older
    `(str, Enum)` spelling renders `"RemovalMode.DELOGO"` there and produces a
    filter graph that fails at runtime.
    """

    DELOGO = "delogo"
    BOXBLUR = "boxblur"


class OverlayKind(StrEnum):
    """What an on-screen element *is*, not merely that text was found there.

    This taxonomy is the difference between OCR and understanding, and it drives
    real behaviour rather than sitting in the response as a label: a `caption`
    can be re-rendered with new text, an `image_popup` can be swapped for a
    different picture, and a `watermark` is usually the thing a user most wants
    gone. A single "text detected" class would make all three the same feature.
    """

    CAPTION = "caption"
    TEXT_OVERLAY = "text_overlay"
    IMAGE_POPUP = "image_popup"
    WATERMARK = "watermark"
    UI_CHROME = "ui_chrome"


class JobStatus(StrEnum):
    """Where a job is. Ordered as the pipeline runs.

    Named stages rather than a bare percentage because "detecting overlays" tells
    a waiting user something a spinner cannot, and because a failure is far
    easier to diagnose when the status says which stage it died in.
    """

    QUEUED = "queued"
    DOWNLOADING = "downloading"
    ANALYZING_SCENES = "analyzing_scenes"
    DETECTING_OVERLAYS = "detecting_overlays"
    RENDERING = "rendering"
    DONE = "done"
    FAILED = "failed"

    @property
    def progress(self) -> float:
        """Rough completion, 0..1.

        Deliberately not uniform: overlay detection is the slowest stage by a
        wide margin, so giving each stage an equal slice would show a progress
        bar that races to 60% and then appears to hang. These weights roughly
        track observed wall-clock.
        """
        return {
            JobStatus.QUEUED: 0.0,
            JobStatus.DOWNLOADING: 0.10,
            JobStatus.ANALYZING_SCENES: 0.25,
            JobStatus.DETECTING_OVERLAYS: 0.45,
            JobStatus.RENDERING: 0.80,
            JobStatus.DONE: 1.0,
            JobStatus.FAILED: 1.0,
        }[self]

    @property
    def is_terminal(self) -> bool:
        return self in (JobStatus.DONE, JobStatus.FAILED)


# =============================================================================
#  Geometry
# =============================================================================


class PixelBox(BaseModel):
    """An integer box in frame coordinates - the form ffmpeg filters want.

    Produced only at the boundary, by `BBox.to_pixels`. Nothing upstream should
    hold one, because a pixel box silently becomes wrong the moment the video is
    rescaled.
    """

    model_config = ConfigDict(frozen=True)

    x: int
    y: int
    w: int
    h: int

    def clamped_to_frame(self, width: int, height: int, *, inset: int = 0) -> PixelBox:
        """Pull the box inside the frame, optionally leaving a margin.

        `delogo` interpolates from the pixels immediately *outside* the mask, so
        a box flush against the frame edge has nothing to sample and ffmpeg
        rejects it. `inset=1` is what makes a full-width burned-in caption - very
        common in this footage - removable at all.
        """
        x = min(max(self.x, inset), max(width - inset - 1, inset))
        y = min(max(self.y, inset), max(height - inset - 1, inset))
        w = max(1, min(self.w, width - inset - x))
        h = max(1, min(self.h, height - inset - y))
        return PixelBox(x=x, y=y, w=w, h=h)


class BBox(BaseModel):
    """A rectangle in normalised frame coordinates: origin top-left, 0..1."""

    model_config = ConfigDict(frozen=True)

    x: float = Field(description="Left edge, as a fraction of frame width")
    y: float = Field(description="Top edge, as a fraction of frame height")
    w: float = Field(description="Width, as a fraction of frame width")
    h: float = Field(description="Height, as a fraction of frame height")

    @model_validator(mode="before")
    @classmethod
    def _clamp_into_frame(cls, data: object) -> object:
        """Clamp rather than reject boxes that poke outside the frame.

        Model-returned boxes are accurate to a few percent, so a caption running
        to the very edge routinely comes back as `x=-0.01` or `w=1.02`. Rejecting
        those would discard true detections over a rounding error; clamping keeps
        them and costs nothing, since a box outside the frame has no pixels to
        erase anyway. Genuinely degenerate boxes - zero or negative area - are
        still rejected below, because those signal a real failure rather than
        imprecision.
        """
        if not isinstance(data, dict):
            return data
        try:
            x, y = float(data["x"]), float(data["y"])
            w, h = float(data["w"]), float(data["h"])
        except (KeyError, TypeError, ValueError):
            return data  # let pydantic produce the proper error

        x, y = min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0)
        return {**data, "x": x, "y": y, "w": min(w, 1.0 - x), "h": min(h, 1.0 - y)}

    @model_validator(mode="after")
    def _reject_degenerate(self) -> BBox:
        if self.w <= 0 or self.h <= 0:
            raise ValueError(f"box has no area: w={self.w}, h={self.h}")
        return self

    # --- derived ------------------------------------------------------------

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def centre(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2

    # --- operations ---------------------------------------------------------

    def iou(self, other: BBox) -> float:
        """Intersection over union, 0..1.

        The similarity measure behind overlay tracking: two boxes in different
        sampled frames are taken to be the same element when they overlap enough
        and their text agrees. IoU is the right measure here precisely because it
        is scale-aware - a small box sitting inside a large one scores low, which
        is what stops a caption from being merged into a full-frame popup.
        """
        ix0, iy0 = max(self.x, other.x), max(self.y, other.y)
        ix1, iy1 = min(self.right, other.right), min(self.bottom, other.bottom)
        intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0

    def union(self, other: BBox) -> BBox:
        """The smallest box containing both.

        A track's box is the union of its detections', so a caption that drifts
        or grows by a word across frames still gets fully covered by one mask.
        """
        x, y = min(self.x, other.x), min(self.y, other.y)
        return BBox(
            x=x, y=y, w=max(self.right, other.right) - x, h=max(self.bottom, other.bottom) - y
        )

    def padded(self, percent: float) -> BBox:
        """Grow the box by `percent` of the frame on every side.

        Bounding boxes are accurate to roughly ±2-3% of the frame, and text is
        anti-aliased: a mask that exactly fits the reported box leaves a visible
        fringe of half-erased glyph edges. Padding trades a slightly larger
        blurred region for actually removing the thing.
        """
        margin = percent / 100.0
        x, y = max(0.0, self.x - margin), max(0.0, self.y - margin)
        return BBox(
            x=x,
            y=y,
            w=min(1.0, self.right + margin) - x,
            h=min(1.0, self.bottom + margin) - y,
        )

    def to_pixels(self, width: int, height: int) -> PixelBox:
        """Convert to integer frame coordinates. The only place geometry stops
        being resolution-independent."""
        x, y = round(self.x * width), round(self.y * height)
        return PixelBox(
            x=x,
            y=y,
            w=max(1, round(self.right * width) - x),
            h=max(1, round(self.bottom * height) - y),
        )


# =============================================================================
#  Media
# =============================================================================


class VideoMeta(BaseModel):
    """What we know about a video file.

    Lives in the domain rather than beside the ffmpeg wrapper because the
    `VideoEditor` port returns it, and a port that referenced an infra type would
    invert the dependency the architecture rests on. Carries no filesystem path:
    the caller passed the path in, and a server path has no business appearing in
    an API response.
    """

    model_config = ConfigDict(frozen=True)

    duration_s: float
    width: int
    height: int
    fps: float
    has_audio: bool
    video_codec: str
    rotation: int = Field(
        default=0,
        description=(
            "Degrees from the container's display matrix. Phone-shot vertical "
            "video stores landscape pixels plus a rotation, which ffmpeg applies "
            "on decode."
        ),
    )

    @property
    def display_width(self) -> int:
        return self.height if self.rotation % 180 else self.width

    @property
    def display_height(self) -> int:
        return self.width if self.rotation % 180 else self.height

    @property
    def is_vertical(self) -> bool:
        return self.display_height > self.display_width


class SampledFrame(BaseModel):
    """One frame handed to the vision layer, with the time it came from.

    The timestamp travels *with* the image because it is what turns a set of
    per-frame detections into time ranges. Losing it would leave us knowing what
    is on screen but not when - which is most of the product.
    """

    model_config = ConfigDict(frozen=True)

    index: int
    t_s: float
    path: str = Field(description="Server-side path; never serialised to a client")


# =============================================================================
#  Detection and tracking
# =============================================================================


class Detection(BaseModel):
    """One element the vision layer saw in one frame.

    Deliberately a per-frame observation, not an object with a lifetime. Turning
    these into things that persist over time is `track_builder`'s job, and
    keeping the two separate is what lets that algorithm be tested against
    hand-written detections with no model in the loop.
    """

    model_config = ConfigDict(frozen=True)

    t_s: float = Field(description="Timestamp of the frame this was seen in")
    bbox: BBox
    kind: OverlayKind
    text: str = Field(default="", description="Transcribed text; empty for image elements")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class OverlayTrack(BaseModel):
    """One overlay, followed across time. The centre of this product.

    A detection says "there is a caption here in this frame". A track says "this
    caption is on screen from 1.5s to 4.0s, in this region, and reads *this*" -
    which is the difference between a video that has been blurred and a video
    that has been taken apart into components you can put back together. Removal,
    the overlay list, caption re-rendering and image swapping are all views of
    this one model.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(description="Stable id, matching the crop filename, e.g. 't003'")
    kind: OverlayKind
    text: str = ""
    bbox: BBox = Field(description="Union of the member detections' boxes")
    start_s: float
    end_s: float
    confidence: float = Field(ge=0.0, le=1.0)
    detection_count: int = Field(description="How many sampled frames support this track")
    crop_url: str | None = Field(default=None, description="Thumbnail crop of the element")

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class Scene(BaseModel):
    """A continuous shot between two cuts."""

    model_config = ConfigDict(frozen=True)

    index: int
    start_s: float
    end_s: float
    thumb_url: str | None = None
    clip_url: str | None = None
    clean_clip_url: str | None = None

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


# =============================================================================
#  Jobs
# =============================================================================


class Job(BaseModel):
    """A unit of work and its progress. What `GET /api/jobs/{id}` returns."""

    id: str
    status: JobStatus = JobStatus.QUEUED
    error: str | None = Field(
        default=None, description="User-facing failure reason; set only when status is failed"
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def progress(self) -> float:
        return self.status.progress


class JobRequest(BaseModel):
    """What the caller asks for. Doubles as the POST body.

    `url` is absent for uploads - there is nothing to fetch, because the route
    has already written the bytes into the workspace.
    """

    url: str | None = Field(default=None, description="Video URL to download")
    removal_mode: RemovalMode = Field(
        default=RemovalMode.DELOGO, description="How detected overlays are erased"
    )


class JobResult(BaseModel):
    """The finished de-edit. What `GET /api/jobs/{id}/result` returns.

    Everything the frontend needs in one document: the two videos to compare, the
    scene strip, and the overlay tracks. Media are URLs rather than embedded
    bytes so the browser can stream and cache them normally.
    """

    job_id: str
    source_url: str = Field(description="The normalised original")
    clean_url: str = Field(description="Same video with detected overlays removed")
    meta: VideoMeta
    scenes: list[Scene] = Field(default_factory=list)
    tracks: list[OverlayTrack] = Field(default_factory=list)
    removal_mode: RemovalMode
    vision_provider: VisionProvider = Field(
        description=(
            "Which detector actually ran. Reported because a result produced by "
            "the offline stub should never be mistaken for a real analysis."
        )
    )

    @property
    def track_count(self) -> int:
        return len(self.tracks)
