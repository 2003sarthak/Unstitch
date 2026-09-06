"""Tests for box refinement.

Written against synthesised frames with known element positions, so "the box is
correct" is a measurable claim rather than an impression. The numbers in the
module docstring of `refine_cv.py` came from exactly this kind of comparison
against the live model.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.refine_cv import NullBoxRefiner, OpenCvBoxRefiner
from app.domain.models import BBox, Detection, OverlayKind, SampledFrame
from app.infra.ffmpeg import Ffmpeg


@pytest.fixture
async def frame_with_band(ffmpeg: Ffmpeg, tmp_path: Path) -> SampledFrame:
    """A flat frame with a white band spanning x 0.20-0.80, y 0.80-0.86.

    Exact by construction, so the refined box can be compared against a number
    rather than against a guess.
    """
    path = tmp_path / "f0000.jpg"
    await ffmpeg.run(
        [
            "-f",
            "lavfi",
            "-i",
            "color=c=steelblue:size=500x500:duration=0.1",
            "-vf",
            "drawbox=x=100:y=400:w=300:h=30:color=white:t=fill",
            "-frames:v",
            "1",
            str(path),
        ]
    )
    return SampledFrame(index=0, t_s=1.0, path=str(path))


def detection(bbox: BBox, t_s: float = 1.0) -> Detection:
    return Detection(t_s=t_s, bbox=bbox, kind=OverlayKind.CAPTION, text="x", confidence=0.9)


class TestRefinement:
    async def test_an_undersized_box_is_grown_to_the_real_extent(
        self, frame_with_band: SampledFrame
    ) -> None:
        """The measured defect this exists for: the model reports a box covering
        part of a caption, and the rest survives the render."""
        short = detection(BBox(x=0.20, y=0.80, w=0.25, h=0.06))  # covers 0.20-0.45

        [result] = await OpenCvBoxRefiner().refine([short], [frame_with_band])

        assert result.bbox.right == pytest.approx(0.80, abs=0.03), "should reach the band's end"
        assert result.bbox.x == pytest.approx(0.20, abs=0.03)

    async def test_an_oversized_box_is_tightened(self, frame_with_band: SampledFrame) -> None:
        """Refinement measures, so it corrects in both directions - a box far
        larger than its content wastes footage the mask did not need to touch."""
        loose = detection(BBox(x=0.05, y=0.70, w=0.90, h=0.25))

        [result] = await OpenCvBoxRefiner().refine([loose], [frame_with_band])

        assert result.bbox.area < loose.bbox.area
        assert result.bbox.x >= 0.15
        assert result.bbox.right <= 0.85

    async def test_an_accurate_box_is_left_close_to_where_it_was(
        self, frame_with_band: SampledFrame
    ) -> None:
        good = detection(BBox(x=0.20, y=0.80, w=0.60, h=0.06))

        [result] = await OpenCvBoxRefiner().refine([good], [frame_with_band])

        assert result.bbox.x == pytest.approx(0.20, abs=0.04)
        assert result.bbox.right == pytest.approx(0.80, abs=0.04)

    async def test_everything_but_the_box_is_preserved(self, frame_with_band: SampledFrame) -> None:
        """Refinement adjusts geometry only. The text the model read and the
        class it assigned are not its business to change."""
        original = Detection(
            t_s=1.0,
            bbox=BBox(x=0.20, y=0.80, w=0.25, h=0.06),
            kind=OverlayKind.WATERMARK,
            text="@handle",
            confidence=0.77,
        )

        [result] = await OpenCvBoxRefiner().refine([original], [frame_with_band])

        assert result.kind is OverlayKind.WATERMARK
        assert result.text == "@handle"
        assert result.confidence == 0.77
        assert result.t_s == 1.0


class TestSafety:
    async def test_a_runaway_measurement_falls_back_to_the_model(
        self, frame_with_band: SampledFrame
    ) -> None:
        """On busy footage the edge analysis can latch onto background texture.
        A mask covering half the video is worse than one slightly too small, so
        growth beyond the cap is refused."""
        tiny = detection(BBox(x=0.45, y=0.82, w=0.02, h=0.02))

        strict = OpenCvBoxRefiner(max_growth=1.2)
        [result] = await strict.refine([tiny], [frame_with_band])

        assert result.bbox == tiny.bbox

    async def test_a_missing_frame_leaves_the_detection_untouched(self, tmp_path: Path) -> None:
        """One unreadable frame must not fail a job, and must not silently drop
        the detections that came from it."""
        orphan = detection(BBox(x=0.1, y=0.1, w=0.2, h=0.2))
        missing = SampledFrame(index=0, t_s=1.0, path=str(tmp_path / "gone.jpg"))

        [result] = await OpenCvBoxRefiner().refine([orphan], [missing])

        assert result.bbox == orphan.bbox

    async def test_a_detection_with_no_matching_frame_survives(
        self, frame_with_band: SampledFrame
    ) -> None:
        [result] = await OpenCvBoxRefiner().refine(
            [detection(BBox(x=0.2, y=0.8, w=0.3, h=0.06), t_s=99.0)], [frame_with_band]
        )
        assert result.bbox.x == pytest.approx(0.2)

    async def test_no_detections_is_not_an_error(self) -> None:
        assert await OpenCvBoxRefiner().refine([], []) == []

    async def test_refinement_never_leaves_the_frame(self, frame_with_band: SampledFrame) -> None:
        """A box outside 0..1 would produce an ffmpeg filter ffmpeg rejects."""
        edge = detection(BBox(x=0.75, y=0.80, w=0.24, h=0.06))

        [result] = await OpenCvBoxRefiner().refine([edge], [frame_with_band])

        assert result.bbox.x >= 0.0
        assert result.bbox.right <= 1.0
        assert result.bbox.bottom <= 1.0


class TestNullRefiner:
    async def test_it_changes_nothing(self, frame_with_band: SampledFrame) -> None:
        """`REFINE_BOXES=false` has to be a true no-op, or the side-by-side
        comparison in the README compares two things that both moved."""
        given = [detection(BBox(x=0.20, y=0.80, w=0.25, h=0.06))]

        result = await NullBoxRefiner().refine(given, [frame_with_band])

        assert result == given
