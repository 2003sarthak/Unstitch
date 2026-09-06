"""Tests for overlay removal.

The graph builder is pure, so what gets erased and when is pinned exactly. The
integration tests then prove those graphs are ones ffmpeg actually accepts -
which matters most for `boxblur`, whose three-chains-per-overlay shape is easy
to get subtly wrong in a way no string assertion would catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.adapters.editor_ffmpeg import FfmpegVideoEditor
from app.domain.models import BBox, OverlayKind, OverlayTrack, RemovalMode, VideoMeta
from app.infra.ffmpeg import Ffmpeg

META = VideoMeta(
    duration_s=2.0, width=720, height=1280, fps=30.0, has_audio=True, video_codec="h264"
)


def track(
    *,
    id: str = "t000",
    x: float = 0.1,
    y: float = 0.8,
    w: float = 0.8,
    h: float = 0.1,
    start: float = 0.5,
    end: float = 1.5,
) -> OverlayTrack:
    return OverlayTrack(
        id=id,
        kind=OverlayKind.CAPTION,
        text="sample",
        bbox=BBox(x=x, y=y, w=w, h=h),
        start_s=start,
        end_s=end,
        confidence=0.9,
        detection_count=2,
    )


@pytest.fixture
def editor() -> FfmpegVideoEditor:
    return FfmpegVideoEditor(Ffmpeg(default_timeout_s=60.0))


class TestDelogoGraph:
    def test_one_track_becomes_one_time_gated_filter(self, editor: FfmpegVideoEditor) -> None:
        graph = editor.removal_graph([track()], META, RemovalMode.DELOGO, padding_pct=0.0)
        flag, value = graph.to_args()
        assert flag == "-vf"
        assert value.startswith("delogo=x=72:y=1024:w=576:h=128")
        assert value.endswith("enable=between(t\\,0.500\\,1.500)")

    def test_every_track_shares_a_single_render_pass(self, editor: FfmpegVideoEditor) -> None:
        """The point of tracks carrying time ranges: N overlays, one encode."""
        tracks = [track(id=f"t{i:03d}", y=0.1 * i, start=i, end=i + 1) for i in range(5)]
        value = editor.removal_graph(tracks, META, RemovalMode.DELOGO).to_args()[1]
        assert value.count("delogo=") == 5
        assert ";" not in value, "a single chain, so ffmpeg makes one pass"

    def test_padding_grows_the_mask(self, editor: FfmpegVideoEditor) -> None:
        tight = editor.removal_graph([track()], META, padding_pct=0.0).render()
        padded = editor.removal_graph([track()], META, padding_pct=5.0).render()
        assert tight != padded
        assert "w=576" in tight
        assert "w=648" in padded  # +5% of frame width on each side

    def test_a_full_width_caption_is_inset_off_the_frame_edge(
        self, editor: FfmpegVideoEditor
    ) -> None:
        """delogo interpolates from just outside the mask, so a box flush to the
        border is rejected. This is the case that would otherwise fail on most
        real short-form video."""
        edge = track(x=0.0, y=0.9, w=1.0, h=0.1)
        value = editor.removal_graph([edge], META, padding_pct=0.0).render()
        assert "x=1:" in value
        assert "w=718" in value  # 720 less one pixel each side

    def test_zero_length_tracks_are_skipped(self, editor: FfmpegVideoEditor) -> None:
        """A track with no duration would produce a filter that never fires."""
        graph = editor.removal_graph([track(start=2.0, end=2.0)], META)
        assert graph.is_empty

    def test_no_tracks_produces_no_graph(self, editor: FfmpegVideoEditor) -> None:
        assert editor.removal_graph([], META).is_empty


class TestBoxblurGraph:
    def test_blur_is_confined_to_the_overlay_region(self, editor: FfmpegVideoEditor) -> None:
        """boxblur has no region parameter, so it takes split -> crop+blur ->
        overlay to avoid blurring the entire frame."""
        rendered = editor.removal_graph([track()], META, RemovalMode.BOXBLUR).render()
        assert "split" in rendered
        assert "crop=" in rendered
        assert "boxblur=" in rendered
        assert "overlay=" in rendered

    def test_the_composite_is_time_gated_not_the_blur(self, editor: FfmpegVideoEditor) -> None:
        """Gating the blur itself would leave the cropped copy pasted back
        permanently. The `overlay` is what must be gated."""
        rendered = editor.removal_graph([track()], META, RemovalMode.BOXBLUR).render()
        blur_chain = next(c for c in rendered.split(";") if "boxblur" in c)
        overlay_chain = next(c for c in rendered.split(";") if "overlay" in c)
        assert "enable=" not in blur_chain
        assert "enable=between(t\\,0.500\\,1.500)" in overlay_chain

    def test_it_uses_filter_complex_because_the_chains_are_labelled(
        self, editor: FfmpegVideoEditor
    ) -> None:
        graph = editor.removal_graph([track()], META, RemovalMode.BOXBLUR)
        assert not graph.is_simple
        assert graph.to_args()[0] == "-filter_complex"

    def test_tracks_chain_so_each_sees_the_previous_result(self, editor: FfmpegVideoEditor) -> None:
        """Two overlays must compose, not race for the same source label - a
        stream label may only be consumed once."""
        tracks = [track(id="t000", y=0.1), track(id="t001", y=0.5)]
        rendered = editor.removal_graph(tracks, META, RemovalMode.BOXBLUR).render()
        assert "[v0]" in rendered  # first overlay's output feeds the second
        assert rendered.count("split") == 2
        assert rendered.endswith("[vout]")

    @pytest.mark.parametrize(
        ("w", "h"),
        [(0.8, 0.02), (0.02, 0.8), (0.5, 0.5), (0.005, 0.005)],
    )
    def test_blur_radius_always_fits_inside_its_region(
        self, editor: FfmpegVideoEditor, w: float, h: float
    ) -> None:
        """boxblur rejects a radius above half the region's smaller side, so a
        thin caption strip needs a smaller radius than a square popup - and a
        fixed value would work on one and fail on the other.

        Asserted against the region the graph actually crops rather than the
        requested box, because padding grows it first.
        """
        rendered = editor.removal_graph([track(w=w, h=h)], META, RemovalMode.BOXBLUR).render()
        crop = rendered.split("crop=")[1]
        crop_w = int(crop.split("w=")[1].split(":")[0])
        crop_h = int(crop.split("h=")[1].split(":")[0])
        luma = int(rendered.split("luma_radius=")[1].split(":")[0])
        chroma = int(rendered.split("chroma_radius=")[1].split(":")[0])

        smaller = min(crop_w, crop_h)
        # Strictly less than half the plane's smaller side, per plane. The chroma
        # planes are half resolution in yuv420p, which is the constraint that
        # actually bites - and the one an earlier version of this test missed by
        # checking only luma.
        assert 0 <= luma < smaller / 2
        assert 0 <= chroma < max(1, smaller // 2) / 2


class TestAgainstRealFfmpeg:
    """Proof the graphs above are ones ffmpeg will actually run."""

    @pytest.fixture
    def editor(self, ffmpeg: Ffmpeg) -> FfmpegVideoEditor:
        return FfmpegVideoEditor(ffmpeg)

    @pytest.mark.parametrize("mode", [RemovalMode.DELOGO, RemovalMode.BOXBLUR])
    async def test_removal_renders_a_playable_video(
        self,
        editor: FfmpegVideoEditor,
        ffmpeg: Ffmpeg,
        sample_video: Path,
        tmp_path: Path,
        mode: RemovalMode,
    ) -> None:
        meta = await ffmpeg.probe(sample_video)
        out = tmp_path / f"clean_{mode}.mp4"

        await editor.remove_overlays(sample_video, out, [track(start=0.5, end=1.5)], meta, mode)

        result = await editor.probe(out)
        assert result.duration_s == pytest.approx(meta.duration_s, abs=0.2)
        assert (result.width, result.height) == (meta.width, meta.height)
        assert result.has_audio, "audio must survive the render"

    @pytest.mark.parametrize("mode", [RemovalMode.DELOGO, RemovalMode.BOXBLUR])
    async def test_multiple_overlapping_tracks_render(
        self,
        editor: FfmpegVideoEditor,
        ffmpeg: Ffmpeg,
        sample_video: Path,
        tmp_path: Path,
        mode: RemovalMode,
    ) -> None:
        meta = await ffmpeg.probe(sample_video)
        tracks = [
            track(id="t000", x=0.05, y=0.05, w=0.4, h=0.2, start=0.0, end=1.0),
            track(id="t001", x=0.5, y=0.7, w=0.45, h=0.25, start=0.5, end=2.0),
        ]
        out = tmp_path / f"multi_{mode}.mp4"
        await editor.remove_overlays(sample_video, out, tracks, meta, mode)
        assert out.exists()

    async def test_a_silent_video_renders_without_asking_for_audio(
        self, editor: FfmpegVideoEditor, ffmpeg: Ffmpeg, tall_video: Path, tmp_path: Path
    ) -> None:
        """The `-map 0:a?` branch: mapping a stream that is not there is an error."""
        meta = await ffmpeg.probe(tall_video)
        out = tmp_path / "silent_clean.mp4"
        await editor.remove_overlays(tall_video, out, [track()], meta, RemovalMode.BOXBLUR)
        assert not (await editor.probe(out)).has_audio

    async def test_no_detections_copies_the_video_through(
        self, editor: FfmpegVideoEditor, ffmpeg: Ffmpeg, sample_video: Path, tmp_path: Path
    ) -> None:
        """No overlays means no re-encode: a generation of quality loss for zero
        visual change would be absurd."""
        meta = await ffmpeg.probe(sample_video)
        out = tmp_path / "untouched.mp4"
        await editor.remove_overlays(sample_video, out, [], meta)

        assert out.exists()
        # A stream copy preserves the exact frame count and codec.
        assert (await editor.probe(out)).video_codec == meta.video_codec

    async def test_scene_cut_and_thumbnail_and_crop(
        self, editor: FfmpegVideoEditor, ffmpeg: Ffmpeg, sample_video: Path, tmp_path: Path
    ) -> None:
        meta = await ffmpeg.probe(sample_video)

        clip, thumb, crop = tmp_path / "s000.mp4", tmp_path / "s000.jpg", tmp_path / "t000.jpg"
        await editor.cut(sample_video, clip, 0.5, 1.5)
        await editor.thumbnail(sample_video, thumb, 1.0)
        await editor.crop(sample_video, crop, BBox(x=0.1, y=0.1, w=0.5, h=0.3), meta, 1.0)

        assert (await editor.probe(clip)).duration_s == pytest.approx(1.0, abs=0.2)
        assert thumb.stat().st_size > 0
        assert crop.stat().st_size > 0


class TestPixelsActuallyChange:
    """The claim the whole product rests on, checked in pixel space.

    Every other test here asserts on argument lists, exit codes and metadata -
    all of which a render that quietly did nothing would satisfy. This one opens
    both videos and compares them.
    """

    @pytest.fixture
    async def banded_video(self, ffmpeg: Ffmpeg, tmp_path: Path) -> Path:
        """Flat teal with a row of white strokes across the lower third.

        Strokes rather than one solid bar, because the two modes fail on
        different things and a solid bar is pathological for exactly one of them.
        `delogo` interpolates a region from its border and erases either, but a
        blur whose radius is bounded by the region it sits in cannot flatten a
        solid fill - the middle stays bright however hard it is smeared.

        Burned-in text is thin strokes with background showing between them,
        which is the case blurring genuinely destroys. Asserting against a solid
        bar would measure a limitation of the fixture and report it as a bug in
        `boxblur`.
        """
        path = tmp_path / "banded.mp4"
        strokes = ",".join(
            f"drawbox=x={36 + i * 22}:y=272:w=9:h=26:color=white:t=fill" for i in range(12)
        )
        await ffmpeg.run(
            [
                "-f",
                "lavfi",
                "-i",
                "color=c=teal:size=320x320:rate=24:duration=2",
                "-vf",
                strokes,
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

    @pytest.mark.parametrize("mode", [RemovalMode.DELOGO, RemovalMode.BOXBLUR])
    async def test_the_overlay_is_gone_and_the_rest_is_untouched(
        self, ffmpeg: Ffmpeg, banded_video: Path, tmp_path: Path, mode: RemovalMode
    ) -> None:
        import cv2
        import numpy as np

        meta = await ffmpeg.probe(banded_video)
        out = tmp_path / f"clean_{mode}.mp4"
        band = track(x=0.1, y=0.85, w=0.8, h=0.09, start=0.0, end=2.0)

        await FfmpegVideoEditor(ffmpeg).remove_overlays(banded_video, out, [band], meta, mode)

        before, after = cv2.VideoCapture(str(banded_video)), cv2.VideoCapture(str(out))
        for capture in (before, after):
            capture.set(cv2.CAP_PROP_POS_MSEC, 1000)
        ok_a, frame_a = before.read()
        ok_b, frame_b = after.read()
        before.release()
        after.release()
        assert ok_a and ok_b

        h, w = frame_a.shape[:2]
        y0, y1 = int(0.84 * h), int(0.95 * h)

        bright_before = float(np.mean(frame_a[y0:y1] > 200))
        bright_after = float(np.mean(frame_b[y0:y1] > 200))
        assert bright_before > 0.1, "the fixture must actually contain bright strokes"
        assert bright_after < bright_before / 4, f"{mode} left the overlay legible"

        # And the footage the mask never covered must survive untouched, or the
        # "removal" is really just degrading the whole video.
        untouched = cv2.absdiff(
            frame_a[int(0.2 * h) : int(0.6 * h)], frame_b[int(0.2 * h) : int(0.6 * h)]
        )
        assert float(np.mean(untouched)) < 3.0
