"""Tests for scene detection.

`build_scenes` holds all the awkward cases and is pure, so it is tested directly
against cut lists. The adapter around it is then only responsible for running
PySceneDetect in a thread - and that gets one integration test against a clip
built with a real, deliberate cut in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.scenes_pyscenedetect import PySceneDetectDetector, build_scenes
from app.infra.ffmpeg import Ffmpeg


class TestBuildScenes:
    def test_footage_with_no_cuts_is_one_scene_not_zero(self) -> None:
        """Callers should never have to special-case unedited video."""
        scenes = build_scenes([], duration_s=8.0, min_scene_seconds=0.6)
        assert len(scenes) == 1
        assert (scenes[0].start_s, scenes[0].end_s) == (0.0, 8.0)

    def test_cuts_become_scenes_in_order(self) -> None:
        scenes = build_scenes([(0.0, 3.0), (3.0, 7.0), (7.0, 10.0)], 10.0, 0.6)
        assert [s.index for s in scenes] == [0, 1, 2]
        assert [s.start_s for s in scenes] == [0.0, 3.0, 7.0]

    def test_scenes_tile_the_video_with_no_gaps(self) -> None:
        """A gap would show as a hole in the timeline strip."""
        scenes = build_scenes([(0.0, 3.0), (3.1, 6.9), (7.0, 9.5)], 10.0, 0.6)
        assert scenes[0].start_s == 0.0
        assert scenes[-1].end_s == 10.0
        for earlier, later in zip(scenes, scenes[1:], strict=False):
            assert later.start_s >= earlier.start_s

    def test_a_flash_frame_is_merged_into_the_previous_shot(self) -> None:
        """A 0.1s "scene" between two cuts is a transition or a compression
        artefact, not a shot worth its own thumbnail."""
        scenes = build_scenes([(0.0, 3.0), (3.0, 3.1), (3.1, 8.0)], 8.0, 0.6)
        assert len(scenes) == 2
        assert scenes[0].end_s == pytest.approx(3.1)

    def test_a_short_first_scene_merges_forward_instead(self) -> None:
        """The first scene has no predecessor to merge into, so it absorbs the
        one after it - otherwise the video would open on a 0.2s shot."""
        scenes = build_scenes([(0.0, 0.2), (0.2, 5.0), (5.0, 8.0)], 8.0, 0.6)
        assert len(scenes) == 2
        assert scenes[0].start_s == 0.0
        assert scenes[0].end_s == pytest.approx(5.0)

    def test_the_first_scene_always_starts_at_zero(self) -> None:
        """Detection can report the first cut a frame late; the opening frames
        still belong to scene 0."""
        assert build_scenes([(0.05, 4.0), (4.0, 8.0)], 8.0, 0.6)[0].start_s == 0.0

    def test_the_last_scene_always_reaches_the_end(self) -> None:
        assert build_scenes([(0.0, 4.0), (4.0, 7.8)], 8.0, 0.6)[-1].end_s == 8.0

    def test_a_zero_length_video_yields_nothing(self) -> None:
        assert build_scenes([], duration_s=0.0, min_scene_seconds=0.6) == []

    def test_every_scene_has_positive_duration(self) -> None:
        scenes = build_scenes([(0.0, 0.1), (0.1, 0.2), (0.2, 5.0)], 5.0, 0.6)
        assert all(s.duration_s > 0 for s in scenes)

    def test_all_scenes_short_collapse_to_one(self) -> None:
        """Fast-cut footage under the threshold should not produce twenty
        unusable 0.2s scenes."""
        cuts = [(i * 0.2, (i + 1) * 0.2) for i in range(10)]
        scenes = build_scenes(cuts, 2.0, min_scene_seconds=0.6)
        assert len(scenes) == 1
        assert (scenes[0].start_s, scenes[0].end_s) == (0.0, 2.0)


class TestAgainstRealVideo:
    @pytest.fixture
    async def two_shot_video(self, ffmpeg: Ffmpeg, tmp_path: Path) -> Path:
        """Two visually unrelated halves spliced together, so there is exactly
        one cut to find. Generated rather than committed, like every other
        fixture here."""
        first, second, path = tmp_path / "a.mp4", tmp_path / "b.mp4", tmp_path / "cut.mp4"
        for out, source in (
            (first, "color=c=black:size=320x240:rate=30:duration=2"),
            (second, "testsrc2=size=320x240:rate=30:duration=2"),
        ):
            await ffmpeg.run(
                [
                    "-f",
                    "lavfi",
                    "-i",
                    source,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-pix_fmt",
                    "yuv420p",
                    str(out),
                ]
            )
        concat = tmp_path / "list.txt"
        concat.write_text(f"file '{first.as_posix()}'\nfile '{second.as_posix()}'\n")
        await ffmpeg.run(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-pix_fmt",
                "yuv420p",
                str(path),
            ]
        )
        return path

    async def test_a_real_cut_is_detected(self, two_shot_video: Path) -> None:
        scenes = await PySceneDetectDetector().detect(two_shot_video)

        assert len(scenes) == 2, "black -> colour bars is about as obvious as a cut gets"
        assert scenes[0].start_s == 0.0
        assert scenes[0].end_s == pytest.approx(2.0, abs=0.2)
        assert scenes[1].end_s == pytest.approx(4.0, abs=0.3)

    async def test_uniform_footage_yields_a_single_scene(self, sample_video: Path) -> None:
        scenes = await PySceneDetectDetector().detect(sample_video)
        assert len(scenes) == 1

    async def test_the_threshold_is_actually_wired_through(self, sample_video: Path) -> None:
        """Sensitivity has to change the answer, or the setting is decorative.

        Tested by lowering the threshold rather than raising it: the black ->
        colour-bars cut is such a large delta that no plausible threshold misses
        it, so a "high threshold finds nothing" test would pass whether or not
        the value reached the detector at all. Moving footage under a near-zero
        threshold is the case that genuinely distinguishes them.
        """
        default = await PySceneDetectDetector().detect(sample_video)
        sensitive = await PySceneDetectDetector(threshold=1.0, min_scene_seconds=0.0).detect(
            sample_video
        )

        assert len(default) == 1
        assert len(sensitive) > len(default)
