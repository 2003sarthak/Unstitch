"""Fetching a video from a URL, via yt-dlp.

Implements the `MediaIngestor` port. Uploads deliberately do not come through
here - there is nothing to fetch, so the route writes the bytes straight into
the workspace.

yt-dlp is synchronous and network-bound, so every call is moved to a worker
thread. That decision lives here rather than in the caller: the port is `async`,
and how an implementation honours that is its own business.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from urllib.parse import urlparse

import yt_dlp

from app.domain.errors import InvalidInputError, MediaTooLongError

log = logging.getLogger(__name__)

#: Video and audio are selected separately and merged.
#:
#: The obvious spelling - `best[height<=1080][ext=mp4]` - asks for a single file
#: containing both streams, and YouTube has largely stopped serving those. A real
#: Shorts URL offers eighteen video-only renditions and five audio-only ones and
#: not one progressive MP4, so that selector fails with "Requested format is not
#: available" on a video that downloads perfectly well. `bv*+ba` asks for the two
#: halves and lets ffmpeg mux them, with progressive kept as a fallback for the
#: sites that still offer it.
#:
#: The height cap is about bandwidth, not quality: normalisation re-encodes to
#: 720 anyway, so pulling 1080 is the most that can ever be useful.
_FORMAT = "bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b"


class YtDlpIngestor:
    """Downloads a URL into the job workspace."""

    def __init__(
        self,
        cookies_file: Path | None = None,
        *,
        max_video_seconds: int = 90,
        socket_timeout_s: int = 30,
    ) -> None:
        self._cookies_file = cookies_file
        self._max_video_seconds = max_video_seconds
        self._socket_timeout_s = socket_timeout_s

    async def fetch(self, url: str, dest: Path) -> Path:
        _require_http_url(url)
        # Metadata first, download second. A ten-minute video is rejected in the
        # time it takes to read its manifest instead of after we have pulled
        # 200MB we are about to throw away.
        info = await asyncio.to_thread(self._extract_info, url)
        self._check_duration(info)
        return await asyncio.to_thread(self._download, url, dest)

    # --- internals -----------------------------------------------------------

    def _options(self, **overrides: object) -> dict[str, object]:
        options: dict[str, object] = {
            "format": _FORMAT,
            "noplaylist": True,  # a playlist URL should yield one video, not forty
            # Separate video and audio streams have to be muxed into something;
            # without this yt-dlp picks a container per-download and the result
            # is sometimes .webm, sometimes .mkv.
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "socket_timeout": self._socket_timeout_s,
            "retries": 2,
        }
        if self._cookies_file:
            options["cookiefile"] = str(self._cookies_file)
        options.update(overrides)
        return options

    def _extract_info(self, url: str) -> dict[str, object]:
        try:
            with yt_dlp.YoutubeDL(self._options(skip_download=True)) as ydl:
                return ydl.extract_info(url, download=False) or {}
        except yt_dlp.utils.DownloadError as exc:
            raise _friendly_error(url, exc) from exc

    def _check_duration(self, info: dict[str, object]) -> None:
        raw = info.get("duration")
        if raw is None:
            return  # a live stream or a site that hides it; the probe will catch it
        duration = float(raw)  # type: ignore[arg-type]
        if duration > self._max_video_seconds:
            raise MediaTooLongError(
                f"that video is {duration:.0f}s; the limit is "
                f"{self._max_video_seconds}s. Try a shorter clip."
            )

    def _download(self, url: str, dest: Path) -> Path:
        """Download into a scratch directory, then move the result into place.

        yt-dlp decides the final extension itself - it may remux, or merge
        separate audio and video streams - so the reliable way to know what it
        produced is to give it an empty directory and look. Writing straight to
        `dest` would leave us guessing at the name.
        """
        staging = dest.parent / ".download"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True, exist_ok=True)
        try:
            options = self._options(outtmpl=str(staging / "%(id)s.%(ext)s"))
            with yt_dlp.YoutubeDL(options) as ydl:
                ydl.download([url])

            downloaded = sorted(p for p in staging.iterdir() if p.is_file())
            if not downloaded:
                raise InvalidInputError(
                    "the download produced no file. The link may be private, "
                    "region-locked, or require a login."
                )
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(downloaded[0]), dest)
            log.info("downloaded %s -> %s", url, dest.name)
            return dest
        except yt_dlp.utils.DownloadError as exc:
            raise _friendly_error(url, exc) from exc
        finally:
            shutil.rmtree(staging, ignore_errors=True)


def _require_http_url(url: str) -> None:
    """Only http(s).

    Without this, `file:///etc/passwd` is a valid input to a download endpoint.
    The check is here rather than in the route because it is a property of
    fetching, and a second ingestor would need exactly the same rule.
    """
    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise InvalidInputError(f"{url!r} is not an http(s) URL")


def _friendly_error(url: str, exc: Exception) -> InvalidInputError:
    """Turn yt-dlp's diagnostics into something a user can act on.

    The raw message is a stack of extractor internals. What the person pasting
    the link needs to know is almost always one of three things, and the most
    useful advice - upload the file instead - is the one yt-dlp cannot give.
    """
    message = str(exc).lower()
    if "private" in message or "login" in message or "sign in" in message or "cookies" in message:
        advice = "that video requires a login. Download it and upload the file instead."
    elif "unavailable" in message or "removed" in message or "404" in message:
        advice = "that video is unavailable or has been removed."
    elif "unsupported url" in message or "no video" in message:
        advice = "that link does not point at a video we can download."
    elif "requested format is not available" in message:
        # Ours, not theirs. This one hid a broken format selector behind a
        # message telling the user their perfectly good link was the problem -
        # so it says plainly that the site is fine and we are not.
        advice = (
            "this video is available but not in a format we asked for. That is a "
            "bug on our side, not a problem with your link."
        )
    else:
        advice = "could not download that link. Uploading the file directly always works."
    log.warning("yt-dlp failed for %s: %s", url, exc)
    return InvalidInputError(advice)
