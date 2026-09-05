"""ffmpeg-backed video editing.

For now this is normalisation only - the removal render and the scene cuts land
in steps 5 and 8, on top of the same `Ffmpeg` wrapper and the same filter-graph
value objects.

Note what this adapter does *not* do: it never spawns a process itself and never
formats a filter string by hand. It decides *what* should happen - which is
policy, and depends on configuration - and hands the *how* to `infra.ffmpeg`.
That separation is what lets the filter-building logic be tested without ffmpeg
and the ffmpeg wrapper be tested without any domain rules.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.domain.errors import MediaTooLongError, ProcessingError, UnsupportedMediaError
from app.infra.ffmpeg import Ffmpeg, Filter, FilterChain, FilterGraph, MediaInfo

log = logging.getLogger(__name__)


class FfmpegVideoEditor:
    """Implements the video-editing side of the pipeline.

    Takes its limits by injection rather than importing `Settings`, so a test
    can construct one with a 2-second cap and not need an env file.
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

    async def probe(self, path: Path) -> MediaInfo:
        """Probe, with tool failures translated into domain errors."""
        try:
            return await self._ffmpeg.probe(path)
        except ValueError as exc:  # no video stream
            raise UnsupportedMediaError(str(exc)) from exc
        except Exception as exc:  # a broken or truncated container
            raise UnsupportedMediaError(f"could not read {path.name} as video") from exc

    async def normalise(self, source: Path, dest: Path) -> MediaInfo:
        """Re-encode `source` into the one format the rest of the pipeline assumes.

        Everything downstream - scene detection, frame sampling, the removal
        render - gets to assume H.264 in yuv420p at a known height. Paying for
        one re-encode up front is cheaper than making four later stages each
        handle VP9, HEVC, odd pixel formats and 4K.

        Three things happen here, and each earns its place:

        * **Length is checked before any work.** A 10-minute upload is rejected
          in the time it takes to probe, not after a two-minute transcode.
        * **Height is capped, width follows.** `-2` keeps the aspect ratio and
          rounds to an even number, which H.264 chroma subsampling requires;
          `min(ih,720)` means we only ever downscale, so a 480p source is not
          upscaled into fake detail that the detector would then have to read.
        * **Rotation is baked in.** ffmpeg auto-applies the display matrix on
          decode, so the output is stored the way it is *seen*. After this the
          normalised file has no rotation metadata left, and a bounding box in
          frame coordinates means the same thing everywhere in the pipeline.
        """
        info = await self.probe(source)

        if info.duration_s > self._max_video_seconds:
            raise MediaTooLongError(
                f"video is {info.duration_s:.0f}s; the limit is "
                f"{self._max_video_seconds}s. Trim it and try again."
            )

        graph = self._normalise_graph()
        args = [
            "-i",
            str(source),
            *graph.to_args(),
            # veryfast/crf 23 is the useful knee for this workload: the output is
            # an intermediate that gets re-encoded again by the removal pass, so
            # spending encoder time on it twice buys nothing visible.
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            # Puts the moov atom first so the browser can start playing before
            # the whole file has downloaded - the preview player depends on it.
            "-movflags",
            "+faststart",
            *(["-c:a", "aac", "-b:a", "128k"] if info.has_audio else ["-an"]),
            str(dest),
        ]

        try:
            await self._ffmpeg.run(args)
        except Exception as exc:
            raise ProcessingError(f"could not normalise {source.name}") from exc

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

    def _normalise_graph(self) -> FilterGraph:
        """Built as a value so the scaling policy is assertable in a unit test
        without encoding a single frame."""
        return FilterGraph(
            [
                FilterChain(
                    [
                        Filter(
                            "scale",
                            {"w": -2, "h": f"min(ih,{self._target_height})"},
                        )
                    ]
                )
            ]
        )
