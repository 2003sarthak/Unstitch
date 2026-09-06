"""Tests for the normalisation adapter.

The scaling *policy* is asserted purely, against the filter graph. The
*behaviour* is asserted against real encodes, because the thing worth proving -
that a 540x960 clip comes back as 720 lines with the aspect ratio intact and an
even width - cannot be established by inspecting an argument list.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.editor_ffmpeg import FfmpegVideoEditor
from app.domain.errors import MediaTooLongError, UnsupportedMediaError
from app.infra.ffmpeg import Ffmpeg


@pytest.fixture
def editor(ffmpeg: Ffmpeg) -> FfmpegVideoEditor:
    return FfmpegVideoEditor(ffmpeg, target_height=720, max_video_seconds=90)


class TestScalingPolicy:
    """Pure: no encoding, no ffmpeg."""

    def test_height_is_capped_and_width_follows(self) -> None:
        graph = FfmpegVideoEditor(Ffmpeg(), target_height=720).normalise_graph()
        assert graph.to_args() == ["-vf", "scale=w=-2:h=min(ih\\,720)"]

    def test_min_expression_means_we_only_ever_downscale(self) -> None:
        """`min(ih,720)` rather than a bare `720`: upscaling a 480p source would
        invent detail the detector then has to read text out of."""
        assert "min(ih" in FfmpegVideoEditor(Ffmpeg()).normalise_graph().render()

    def test_target_height_is_injected_not_hard_coded(self) -> None:
        graph = FfmpegVideoEditor(Ffmpeg(), target_height=480).normalise_graph()
        assert graph.render() == "scale=w=-2:h=min(ih\\,480)"


class TestNormalise:
    async def test_a_tall_clip_is_capped_to_720_lines(
        self, editor: FfmpegVideoEditor, tall_video: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "source.mp4"
        info = await editor.normalise(tall_video, out)

        assert info.height == 720
        # 540x960 scaled to 720 lines is 405 wide, which is odd - `-2` rounds it
        # to 406, because H.264 4:2:0 cannot encode odd dimensions.
        assert info.width == 406
        assert info.width % 2 == 0
        assert info.video_codec == "h264"

    async def test_a_small_clip_is_not_upscaled(
        self, editor: FfmpegVideoEditor, sample_video: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "source.mp4"
        info = await editor.normalise(sample_video, out)
        assert (info.width, info.height) == (320, 240)

    async def test_audio_is_preserved_when_present(
        self, editor: FfmpegVideoEditor, sample_video: Path, tmp_path: Path
    ) -> None:
        info = await editor.normalise(sample_video, tmp_path / "out.mp4")
        assert info.has_audio

    async def test_a_silent_clip_stays_silent(
        self, editor: FfmpegVideoEditor, tall_video: Path, tmp_path: Path
    ) -> None:
        """The `-an` branch: asking ffmpeg to encode a stream that does not exist
        is an error, so the argument list has to depend on the probe."""
        info = await editor.normalise(tall_video, tmp_path / "out.mp4")
        assert not info.has_audio

    async def test_output_is_streamable(
        self, editor: FfmpegVideoEditor, sample_video: Path, tmp_path: Path
    ) -> None:
        """`+faststart` moves the moov atom to the front so the preview player
        can start before the whole file has arrived. Its absence is invisible
        locally and very visible over a network."""
        out = tmp_path / "out.mp4"
        await editor.normalise(sample_video, out)

        data = out.read_bytes()
        assert data.index(b"moov") < data.index(b"mdat")


class TestRejection:
    async def test_a_video_over_the_limit_is_rejected_not_trimmed(
        self, ffmpeg: Ffmpeg, sample_video: Path, tmp_path: Path
    ) -> None:
        """Returning a silently truncated video would look like a bug to the
        person who uploaded it, and `.env.example` promises rejection."""
        strict = FfmpegVideoEditor(ffmpeg, max_video_seconds=1)
        out = tmp_path / "out.mp4"

        with pytest.raises(MediaTooLongError, match="the limit is 1s"):
            await strict.normalise(sample_video, out)

        assert not out.exists(), "rejection must happen before any encoding work"

    async def test_a_non_video_file_is_rejected_as_unsupported(
        self, editor: FfmpegVideoEditor, tmp_path: Path
    ) -> None:
        not_a_video = tmp_path / "notes.txt"
        not_a_video.write_text("this is not an mp4")

        with pytest.raises(UnsupportedMediaError):
            await editor.normalise(not_a_video, tmp_path / "out.mp4")

    async def test_a_missing_file_is_rejected_as_unsupported(
        self, editor: FfmpegVideoEditor, tmp_path: Path
    ) -> None:
        with pytest.raises(UnsupportedMediaError):
            await editor.normalise(tmp_path / "gone.mp4", tmp_path / "out.mp4")

    async def test_ffmpeg_failures_surface_as_domain_errors_not_ffmpeg_errors(
        self, editor: FfmpegVideoEditor, tmp_path: Path
    ) -> None:
        """The point of the adapter: nothing above this layer should have to
        know that ffmpeg exists, so no `FFmpegError` may escape it."""
        from app.domain.errors import UnstitchError

        with pytest.raises(UnstitchError):
            await editor.normalise(tmp_path / "gone.mp4", tmp_path / "out.mp4")
