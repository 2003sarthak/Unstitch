"""Domain vocabulary and data shapes.

Only the enums are defined so far - the pydantic models (BBox, Detection,
OverlayTrack, Scene, Job, JobResult) arrive with step 3 of the build order.

These are `StrEnum`s rather than `Literal[...]` aliases for two reasons: FastAPI
renders them as a proper `enum` in the OpenAPI schema (so `removal_mode` is
self-documenting to the frontend), and a `StrEnum` member formats as its bare
value - `f"{RemovalMode.DELOGO}"` is `"delogo"`, which is what an ffmpeg filter
string needs. The older `(str, Enum)` spelling would render
`"RemovalMode.DELOGO"` there and produce a filter graph that fails at runtime.
"""

from __future__ import annotations

from enum import StrEnum


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
    busy video, `boxblur` is an honest smear that never invents detail, and
    `inpaint` is slower but can beat both over flat backgrounds.
    """

    DELOGO = "delogo"
    BOXBLUR = "boxblur"
    INPAINT = "inpaint"
