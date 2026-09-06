"""Tests for frame sampling and the offline vision stub.

Sampling policy gets the most attention here because it is what the app *costs*
to run: every timestamp this returns is quota spent against a rate-limited free
tier, and every one it skips is an overlay that might be missed.

The Gemini adapter is tested only at its pure boundaries - the box conversion and
the retry classifier. Everything else about it is a network call, and a test that
mocks an SDK mostly asserts that the mock was written to match the code.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.sampler_ffmpeg import FfmpegFrameSampler, choose_timestamps
from app.adapters.vision_gemini import _is_retryable, _to_bbox
from app.adapters.vision_stub import StubVisionDetector
from app.domain.models import OverlayKind, SampledFrame, Scene, VideoMeta
from app.infra.ffmpeg import Ffmpeg


def scenes(*ranges: tuple[float, float]) -> list[Scene]:
    return [Scene(index=i, start_s=a, end_s=b) for i, (a, b) in enumerate(ranges)]


class TestSamplingPolicy:
    def test_every_scene_contributes_its_midpoint(self) -> None:
        """A cut is where the picture changes most, so a scene is the natural
        unit of "something new might be on screen". The middle avoids the
        cross-faded frames at either end."""
        chosen = choose_timestamps(10.0, scenes((0.0, 4.0), (4.0, 10.0)), 100.0, 24)
        assert 2.0 in chosen
        assert 7.0 in chosen

    def test_a_fixed_interval_is_sampled_too(self) -> None:
        """Overlays do not respect scene boundaries - a caption can appear
        halfway through a long shot, and pop-ups usually do."""
        chosen = choose_timestamps(9.0, scenes((0.0, 9.0)), interval_s=1.5, max_frames=24)
        assert len(chosen) > 1
        assert 1.5 in chosen and 3.0 in chosen

    def test_the_budget_is_never_exceeded(self) -> None:
        """This is the number that keeps the app inside the free tier."""
        chosen = choose_timestamps(90.0, scenes(*[(i, i + 1) for i in range(90)]), 0.5, 24)
        assert len(chosen) <= 24

    def test_thinning_keeps_the_whole_timeline_not_the_first_third(self) -> None:
        """The bug this prevents: truncating instead of thinning spends the
        entire budget on the opening seconds and analyses none of the rest -
        which looks like a bad model rather than a bad sampler."""
        chosen = choose_timestamps(60.0, scenes((0.0, 60.0)), interval_s=0.5, max_frames=10)

        assert len(chosen) == 10
        assert chosen[-1] > 50.0, "the end of the video must still be sampled"
        assert chosen == sorted(chosen)

    def test_first_and_last_candidates_survive_thinning(self) -> None:
        """Watermarks sit on the opening frame and end-cards on the closing one."""
        dense = choose_timestamps(30.0, scenes((0.0, 30.0)), interval_s=0.2, max_frames=8)
        assert dense[0] == 0.0
        assert dense[-1] >= 29.0

    def test_timestamps_are_unique_and_ordered(self) -> None:
        """Scene midpoints and interval marks overlap; sampling the same instant
        twice would pay for the same frame twice."""
        chosen = choose_timestamps(12.0, scenes((0.0, 3.0), (3.0, 12.0)), 1.5, 24)
        assert chosen == sorted(set(chosen))

    def test_nothing_is_sampled_past_the_end(self) -> None:
        """Seeking to exactly the duration lands past the last frame on some
        containers and silently produces no image."""
        chosen = choose_timestamps(5.0, scenes((0.0, 5.0)), 1.5, 24)
        assert all(t < 5.0 for t in chosen)

    def test_a_very_short_video_still_yields_a_frame(self) -> None:
        assert choose_timestamps(0.4, scenes((0.0, 0.4)), 1.5, 24)

    def test_a_zero_length_video_yields_nothing(self) -> None:
        assert choose_timestamps(0.0, [], 1.5, 24) == []


class TestSamplerAgainstRealFfmpeg:
    async def test_frames_are_written_and_described(
        self, ffmpeg: Ffmpeg, sample_video: Path, tmp_path: Path
    ) -> None:
        sampler = FfmpegFrameSampler(ffmpeg, interval_s=0.5, max_frames=4)
        meta = await ffmpeg.probe(sample_video)

        frames = await sampler.sample(sample_video, meta, scenes((0.0, 2.0)), tmp_path / "frames")

        assert 0 < len(frames) <= 4
        for frame in frames:
            assert Path(frame.path).stat().st_size > 0
            assert 0 <= frame.t_s < meta.duration_s
        assert [f.index for f in frames] == list(range(len(frames)))


class TestStubDetector:
    """The stub only has to be *plausible* - it exists so the pipeline runs with
    no API key. What must hold is that its output is well-formed enough for the
    tracker and the renderer to consume without special-casing."""

    @pytest.fixture
    async def frame_with_text(self, ffmpeg: Ffmpeg, tmp_path: Path) -> SampledFrame:
        """A caption-like band: a horizontal run of high-contrast blocks low in
        the frame, standing in for glyph strokes.

        Built from `drawbox` rather than `drawtext` on purpose. `drawtext` needs
        fontconfig and a font file on disk, which makes the test pass or fail
        depending on the machine - and the stub detector keys on edge density,
        not on letterforms, so blocks exercise exactly the same code path.
        """
        path = tmp_path / "f0000.jpg"
        strokes = ",".join(
            f"drawbox=x={60 + i * 24}:y=300:w=15:h=30:color=white:t=fill" for i in range(14)
        )
        await ffmpeg.run(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:size=640x360:duration=0.1",
                "-vf",
                strokes,
                "-frames:v",
                "1",
                str(path),
            ]
        )
        return SampledFrame(index=0, t_s=1.0, path=str(path))

    async def test_it_finds_burned_in_text(self, frame_with_text: SampledFrame) -> None:
        detections = await StubVisionDetector().detect([frame_with_text])
        assert detections, "high-contrast text on flat grey is the easy case"

    async def test_detections_are_usable_by_the_rest_of_the_pipeline(
        self, frame_with_text: SampledFrame
    ) -> None:
        for detection in await StubVisionDetector().detect([frame_with_text]):
            assert detection.t_s == frame_with_text.t_s
            assert detection.bbox.x >= 0.0 and detection.bbox.right <= 1.0
            assert detection.bbox.y >= 0.0 and detection.bbox.bottom <= 1.0
            assert isinstance(detection.kind, OverlayKind)

    async def test_it_never_claims_model_level_confidence(
        self, frame_with_text: SampledFrame
    ) -> None:
        """A heuristic that cannot read the text must not report the certainty of
        a model that has."""
        for detection in await StubVisionDetector().detect([frame_with_text]):
            assert detection.confidence <= 0.75
            assert detection.text == ""

    async def test_a_missing_frame_file_is_survivable(self, tmp_path: Path) -> None:
        """One unreadable frame must not fail a whole job."""
        missing = SampledFrame(index=0, t_s=0.0, path=str(tmp_path / "nope.jpg"))
        assert await StubVisionDetector().detect([missing]) == []

    async def test_a_blank_frame_produces_few_or_no_detections(
        self, ffmpeg: Ffmpeg, tmp_path: Path
    ) -> None:
        """A flat colour field has no edges, so a detector keyed on edge density
        should find nothing there."""
        path = tmp_path / "blank.jpg"
        await ffmpeg.run(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=gray:size=640x360:duration=0.1",
                "-frames:v",
                "1",
                str(path),
            ]
        )
        found = await StubVisionDetector().detect([SampledFrame(index=0, t_s=0.0, path=str(path))])
        assert found == []


class TestGeminiBoundaries:
    """Only the pure edges. Mocking the SDK would mostly test the mock."""

    def test_gemini_box_order_is_converted_correctly(self) -> None:
        """Gemini reports [ymin, xmin, ymax, xmax] on a 0-1000 grid - a different
        order *and* a different scale from ours. Getting this wrong would put
        every mask in the transposed position."""
        box = _to_bbox([800, 100, 900, 900])
        assert box is not None
        assert box.x == pytest.approx(0.1)
        assert box.y == pytest.approx(0.8)
        assert box.w == pytest.approx(0.8)
        assert box.h == pytest.approx(0.1)

    def test_inverted_corners_are_repaired(self) -> None:
        assert _to_bbox([900, 900, 800, 100]) == _to_bbox([800, 100, 900, 900])

    def test_a_malformed_box_costs_only_itself(self) -> None:
        """One bad rectangle in a batch of six frames should not fail the job."""
        assert _to_bbox([1, 2, 3]) is None
        assert _to_bbox([]) is None
        assert _to_bbox([500, 500, 500, 500]) is None  # zero area

    @pytest.mark.parametrize(
        "message",
        [
            "429 RESOURCE_EXHAUSTED",
            "rate limit exceeded",
            "503 Service Unavailable",
            "quota exceeded for model",
        ],
    )
    def test_transient_failures_are_retried(self, message: str) -> None:
        assert _is_retryable(Exception(message))

    @pytest.mark.parametrize(
        "message",
        [
            "API key not valid",
            "403 permission denied",
            "401 unauthorized",
            "invalid argument: bad request",
        ],
    )
    def test_permanent_failures_are_not_retried(self, message: str) -> None:
        """Retrying a bad API key three times just makes the user wait longer for
        the same error."""
        assert not _is_retryable(Exception(message))


def test_sampler_meta_is_unused_but_declared() -> None:
    """Guards the port signature: `sample` takes `VideoMeta` so an implementation
    can use duration or dimensions without a signature change rippling outward."""
    meta = VideoMeta(
        duration_s=5.0, width=720, height=1280, fps=30.0, has_audio=False, video_codec="h264"
    )
    assert choose_timestamps(meta.duration_s, scenes((0.0, 5.0)), 1.5, 24)
