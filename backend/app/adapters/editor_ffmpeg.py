"""ffmpeg-backed video editing. Implements the `VideoEditor` port.

Note what this adapter does *not* do: it never spawns a process itself and never
formats a filter string by hand. It decides *what* should happen - policy, driven
by configuration - and hands the *how* to `infra.ffmpeg`. That separation is what
lets the filter-building logic be tested exhaustively without ffmpeg, and the
ffmpeg wrapper be tested without any domain rules.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from app.domain.errors import MediaTooLongError, ProcessingError, UnsupportedMediaError
from app.domain.models import BBox, OverlayTrack, PixelBox, RemovalMode, VideoMeta
from app.infra.ffmpeg import Ffmpeg, Filter, FilterChain, FilterGraph, between

log = logging.getLogger(__name__)

#: Keep masks a pixel clear of the frame edge. `delogo` interpolates from the
#: pixels immediately *outside* the mask, so a region flush against the border
#: has nothing to sample and ffmpeg refuses it. Full-width burned-in captions -
#: the most common overlay in this footage - hit this every single time.
_EDGE_INSET_PX = 1

#: The label the removal graph gives its final video pad when it needs one.
_OUT = "vout"

#: Shared output settings, defined once so the clean render, a scene clip and a
#: rebuild cannot drift into three subtly different encoders.
_H264_OUTPUT = (
    "-c:v",
    "libx264",
    "-preset",
    "veryfast",
    "-crf",
    "23",
    "-pix_fmt",
    "yuv420p",
    "-movflags",
    "+faststart",
)


class FfmpegVideoEditor:
    """Implements the video-editing side of the pipeline.

    Takes its limits by injection rather than importing `Settings`, so a test can
    construct one with a 2-second cap and not need an env file.
    """

    def __init__(
        self,
        ffmpeg: Ffmpeg,
        *,
        target_height: int = 720,
        max_video_seconds: int = 90,
    ) -> None:
        self._ffmpeg = ffmpeg
        self._target_height = target_height
        self._max_video_seconds = max_video_seconds

    # --- probing -------------------------------------------------------------

    async def probe(self, path: Path) -> VideoMeta:
        """Probe, with tool failures translated into domain errors."""
        try:
            return await self._ffmpeg.probe(path)
        except ValueError as exc:  # no video stream
            raise UnsupportedMediaError(str(exc)) from exc
        except Exception as exc:  # a broken or truncated container
            raise UnsupportedMediaError(f"could not read {path.name} as video") from exc

    # --- normalisation -------------------------------------------------------

    async def normalise(self, source: Path, dest: Path) -> VideoMeta:
        """Re-encode `source` into the one format the rest of the pipeline assumes.

        Everything downstream - scene detection, frame sampling, the removal
        render - gets to assume H.264 in yuv420p at a known height. Paying for one
        re-encode up front is cheaper than making four later stages each handle
        VP9, HEVC, odd pixel formats and 4K.

        Three things happen here, and each earns its place:

        * **Length is checked before any work.** A ten-minute upload is rejected
          in the time it takes to probe, not after a two-minute transcode.
        * **Height is capped, width follows.** `-2` keeps the aspect ratio and
          rounds to an even number, which H.264 chroma subsampling requires;
          `min(ih,720)` means we only ever downscale, so a 480p source is not
          upscaled into fake detail the detector would then have to read.
        * **Rotation is baked in.** ffmpeg auto-applies the display matrix on
          decode, so the output is stored the way it is *seen*. Afterwards the
          file carries no rotation metadata, and a bounding box in frame
          coordinates means the same thing everywhere in the pipeline.
        """
        info = await self.probe(source)
        if info.duration_s > self._max_video_seconds:
            raise MediaTooLongError(
                f"video is {info.duration_s:.0f}s; the limit is "
                f"{self._max_video_seconds}s. Trim it and try again."
            )

        await self._run(
            [
                "-i",
                str(source),
                *self.normalise_graph().to_args(),
                *_H264_OUTPUT,
                *(["-c:a", "aac", "-b:a", "128k"] if info.has_audio else ["-an"]),
                str(dest),
            ],
            what=f"normalise {source.name}",
        )

        normalised = await self.probe(dest)
        log.info(
            "normalised %s: %dx%d %s %.1fs -> %dx%d h264 %.1fs",
            source.name,
            info.display_width,
            info.display_height,
            info.video_codec,
            info.duration_s,
            normalised.width,
            normalised.height,
            normalised.duration_s,
        )
        return normalised

    def normalise_graph(self) -> FilterGraph:
        """The scaling policy, as a value - assertable without encoding a frame."""
        return FilterGraph(
            [FilterChain([Filter("scale", {"w": -2, "h": f"min(ih,{self._target_height})"})])]
        )

    # --- removal -------------------------------------------------------------

    async def remove_overlays(
        self,
        source: Path,
        dest: Path,
        tracks: Sequence[OverlayTrack],
        meta: VideoMeta,
        mode: RemovalMode = RemovalMode.DELOGO,
        padding_pct: float = 2.0,
    ) -> None:
        """Erase every track's region, each gated to its own time window.

        One render pass for all N overlays. That is the entire reason tracks carry
        time ranges: without them the only options are masking every frame - which
        destroys footage that was never covered - or N sequential passes, which
        would make this comfortably the slowest thing in the app.
        """
        graph = self.removal_graph(tracks, meta, mode, padding_pct)

        if graph.is_empty:
            # Nothing detected. Copy rather than re-encode: faster, and a
            # generation of quality loss for zero visual change would be absurd.
            log.info("no overlays to remove from %s; copying through", source.name)
            await self._run(
                ["-i", str(source), "-c", "copy", "-movflags", "+faststart", str(dest)],
                what=f"copy {source.name}",
            )
            return

        await self._run(
            [
                "-i",
                str(source),
                *graph.to_args(),
                *self._stream_map(graph, meta),
                *_H264_OUTPUT,
                *(["-c:a", "copy"] if meta.has_audio else ["-an"]),
                str(dest),
            ],
            what=f"remove {len(tracks)} overlay(s) from {source.name}",
        )
        log.info("removed %d overlay(s) from %s using %s", len(tracks), source.name, mode)

    @staticmethod
    def _stream_map(graph: FilterGraph, meta: VideoMeta) -> list[str]:
        """Explicit stream selection, but only when the graph forces it.

        A simple `-vf` graph keeps ffmpeg's default mapping. A labelled
        `-filter_complex` graph does not: naming the video output disables the
        automatic choice, and the audio then has to be asked for by hand. `0:a?`
        makes that request optional, so the same argument list also works for a
        silent video instead of failing on a stream that is not there.
        """
        if graph.is_simple:
            return []
        return ["-map", f"[{_OUT}]", *(["-map", "0:a?"] if meta.has_audio else [])]

    def removal_graph(
        self,
        tracks: Sequence[OverlayTrack],
        meta: VideoMeta,
        mode: RemovalMode = RemovalMode.DELOGO,
        padding_pct: float = 2.0,
    ) -> FilterGraph:
        """Build the removal filter graph.

        Pure and public precisely so it can be tested exhaustively. This is the
        function that decides what gets erased and when; without it as a value,
        the only way to observe that decision would be to watch a video.
        """
        boxes = [
            (track, self._mask_box(track, meta, padding_pct))
            for track in tracks
            if track.duration_s > 0
        ]
        if not boxes:
            return FilterGraph()
        if mode is RemovalMode.BOXBLUR:
            return self._boxblur_graph(boxes)
        return self._delogo_graph(boxes)

    def _mask_box(self, track: OverlayTrack, meta: VideoMeta, padding_pct: float) -> PixelBox:
        """Pad, convert to pixels, then pull inside the frame - in that order.

        The order matters: padding a box that already sits at the edge pushes it
        out of frame, and clamping afterwards is what brings it back to something
        ffmpeg will accept. Clamping first would let the padding undo the clamp.
        """
        return (
            track.bbox.padded(padding_pct)
            .to_pixels(meta.width, meta.height)
            .clamped_to_frame(meta.width, meta.height, inset=_EDGE_INSET_PX)
        )

    @staticmethod
    def _delogo_graph(boxes: Sequence[tuple[OverlayTrack, PixelBox]]) -> FilterGraph:
        """The default: one time-gated `delogo` per track, chained into one pass.

        `delogo` interpolates each masked region from its own border, so text over
        busy footage dissolves into something plausible rather than into a
        rectangle announcing that something was removed.
        """
        return FilterGraph(
            [
                FilterChain(
                    [
                        Filter(
                            "delogo",
                            {"x": box.x, "y": box.y, "w": box.w, "h": box.h},
                            enable=between(track.start_s, track.end_s),
                        )
                        for track, box in boxes
                    ]
                )
            ]
        )

    @staticmethod
    def _boxblur_graph(boxes: Sequence[tuple[OverlayTrack, PixelBox]]) -> FilterGraph:
        """The honest alternative: blur the region rather than invent pixels.

        `boxblur` has no region parameter, so confining it takes three chains per
        overlay - split the stream, crop and blur one copy, then composite it back
        over the original inside the track's time window. The result never
        fabricates detail, which makes it the more defensible choice when the
        point is to show that something *was* there.

        This is what labelled chains in `FilterGraph` exist for, and why
        `to_args()` switching to `-filter_complex` is derived from the graph
        rather than decided at the call site.
        """
        chains: list[FilterChain] = []
        current = "0:v"
        for i, (track, box) in enumerate(boxes):
            main, spare, blurred = f"m{i}", f"s{i}", f"b{i}"
            # The final overlay writes the graph's terminal label; the rest feed
            # the next iteration.
            nxt = _OUT if i == len(boxes) - 1 else f"v{i}"
            # A stream label may only be consumed once, hence the split.
            chains.append(FilterChain([Filter("split")], inputs=[current], outputs=[main, spare]))
            chains.append(
                FilterChain(
                    [
                        Filter("crop", {"w": box.w, "h": box.h, "x": box.x, "y": box.y}),
                        Filter(
                            "boxblur",
                            dict(
                                zip(
                                    ("luma_radius", "chroma_radius", "luma_power"),
                                    (*_blur_radii(box), 2),
                                    strict=True,
                                )
                            ),
                        ),
                    ],
                    inputs=[spare],
                    outputs=[blurred],
                )
            )
            chains.append(
                FilterChain(
                    [
                        Filter(
                            "overlay",
                            {"x": box.x, "y": box.y},
                            enable=between(track.start_s, track.end_s),
                        )
                    ],
                    inputs=[main, blurred],
                    outputs=[nxt],
                )
            )
            current = nxt
        return FilterGraph(chains)

    # --- cuts and stills -----------------------------------------------------

    async def cut(self, source: Path, dest: Path, start_s: float, end_s: float) -> None:
        """Extract `[start_s, end_s]` as its own clip.

        `-ss` goes *before* `-i` so ffmpeg seeks instead of decoding and
        discarding everything up to the cut - the difference between instant and
        linear in the video's length. Re-encoding rather than stream-copying is
        also deliberate: a copy can only cut on keyframes, which on this footage
        means a scene clip that starts up to two seconds early.
        """
        await self._run(
            [
                "-ss",
                f"{start_s:.3f}",
                "-i",
                str(source),
                "-t",
                f"{max(0.04, end_s - start_s):.3f}",
                *_H264_OUTPUT,
                "-an",  # scene clips are silent previews, not playback
                str(dest),
            ],
            what=f"cut {start_s:.1f}-{end_s:.1f}s",
        )

    async def thumbnail(self, source: Path, dest: Path, at_s: float) -> None:
        await self._run(
            ["-ss", f"{at_s:.3f}", "-i", str(source), "-frames:v", "1", "-q:v", "3", str(dest)],
            what=f"thumbnail at {at_s:.1f}s",
        )

    async def crop(
        self, source: Path, dest: Path, bbox: BBox, meta: VideoMeta, at_s: float
    ) -> None:
        """A still of one overlay's region - the thumbnail in the overlay list.

        This is what makes the result legible rather than merely correct: a row
        reading "caption, 0:03-0:07" is a claim, and the crop beside it is the
        evidence.
        """
        box = bbox.to_pixels(meta.width, meta.height).clamped_to_frame(meta.width, meta.height)
        graph = FilterGraph(
            [FilterChain([Filter("crop", {"w": box.w, "h": box.h, "x": box.x, "y": box.y})])]
        )
        await self._run(
            [
                "-ss",
                f"{at_s:.3f}",
                "-i",
                str(source),
                *graph.to_args(),
                "-frames:v",
                "1",
                "-q:v",
                "3",
                str(dest),
            ],
            what=f"crop overlay at {at_s:.1f}s",
        )

    # --- one place to translate failures -------------------------------------

    async def _run(self, args: Sequence[str], *, what: str) -> None:
        """Every ffmpeg call in this adapter goes through here.

        The point of an adapter is that nothing above it needs to know ffmpeg
        exists, so no `FFmpegError` may escape - and doing the translation in one
        place means a method added later cannot forget to.
        """
        try:
            await self._ffmpeg.run(args)
        except Exception as exc:
            raise ProcessingError(f"could not {what}") from exc


def _blur_radii(box: PixelBox) -> tuple[int, int]:
    """Luma and chroma blur radii, each inside what boxblur will accept.

    `boxblur` requires a radius strictly below half its plane's smaller side, and
    it blurs the chroma planes as well as luma. In yuv420p - which everything
    here is encoded as - the chroma planes are *half resolution*, so chroma is
    the binding constraint, at half the limit luma has.

    Leaving `chroma_radius` to default to the luma value is therefore a bug that
    only shows up on thin regions: a 41px-tall caption strip yields a luma radius
    of 10, which is legal for luma and rejected for chroma, and the whole render
    fails. Computing the two separately keeps the luma blur strong enough to
    destroy text while staying valid.
    """
    smaller = min(box.w, box.h)
    luma = max(0, min(smaller // 4, (smaller - 1) // 2))
    chroma_smaller = smaller // 2
    chroma = max(0, min(luma, (chroma_smaller - 1) // 2))
    return luma, chroma
