"""Per-job directory layout, and the single mapping between disk paths and URLs.

Every artefact a job produces lives under `MEDIA_ROOT/{job_id}/`, and every file
name is derived from a method here rather than being spelled out at the call
site. That matters more than it looks: the renderer writes `clean.mp4`, the API
serialises a URL for it, and the TTL sweeper deletes it - three modules that
would otherwise each hard-code the same string and drift apart.

    work/{job_id}/
        original.mp4          as ingested (download or upload), untouched
        source.mp4            normalised: H.264, capped height, faststart
        clean.mp4             overlays removed
        frames/f0000.jpg      frames sampled for the vision pass
        scenes/s000.mp4       per-scene cuts, `s000_clean.mp4` for the clean pass
        thumbs/s000.jpg       scene thumbnails for the timeline strip
        crops/t000.jpg        a crop of each detected overlay, for the overlay list
        rebuild/r001.mp4      bonus re-renders, one per revision

The layout is flat and predictable on purpose: a reviewer can open the workspace
directory and see the whole pipeline's output without reading any code.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

#: Job ids come from `uuid4().hex`. Validated on the way *in* rather than
#: sanitised on the way out, because an id also becomes a URL segment and a
#: directory name - and `../` in either of those is how a media route turns into
#: an arbitrary-file-read.
_JOB_ID_RE = re.compile(r"^[0-9a-zA-Z_-]{8,64}$")

MEDIA_URL_PREFIX = "/media"


class InvalidJobIdError(ValueError):
    """A job id that we refuse to turn into a path."""


class PathEscapeError(ValueError):
    """A resolved path landed outside the workspace it claimed to be in."""


def validate_job_id(job_id: str) -> str:
    if not _JOB_ID_RE.match(job_id):
        raise InvalidJobIdError(f"invalid job id: {job_id!r}")
    return job_id


@dataclass(frozen=True, slots=True)
class Workspace:
    """The set of paths belonging to one job."""

    job_id: str
    root: Path

    # --- canonical artefacts -------------------------------------------------

    @property
    def original(self) -> Path:
        return self.root / "original.mp4"

    @property
    def source(self) -> Path:
        return self.root / "source.mp4"

    @property
    def clean(self) -> Path:
        return self.root / "clean.mp4"

    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def scenes_dir(self) -> Path:
        return self.root / "scenes"

    @property
    def thumbs_dir(self) -> Path:
        return self.root / "thumbs"

    @property
    def crops_dir(self) -> Path:
        return self.root / "crops"

    @property
    def rebuild_dir(self) -> Path:
        return self.root / "rebuild"

    # --- indexed artefacts ---------------------------------------------------
    #
    # Zero-padded so that a plain lexicographic listing - `ls`, a glob, or the
    # order the frontend receives - is also chronological order. Unpadded names
    # would sort scene 10 before scene 2.

    def frame(self, index: int) -> Path:
        return self.frames_dir / f"f{index:04d}.jpg"

    def scene_clip(self, index: int, *, clean: bool = False) -> Path:
        suffix = "_clean" if clean else ""
        return self.scenes_dir / f"s{index:03d}{suffix}.mp4"

    def scene_thumb(self, index: int) -> Path:
        return self.thumbs_dir / f"s{index:03d}.jpg"

    def track_crop(self, index: int) -> Path:
        return self.crops_dir / f"t{index:03d}.jpg"

    def rebuild(self, revision: int) -> Path:
        return self.rebuild_dir / f"r{revision:03d}.mp4"

    # --- lifecycle -----------------------------------------------------------

    def ensure(self) -> Workspace:
        """Create every directory up front.

        One `mkdir` burst at job start beats scattering `parents=True` through
        the pipeline, and it means a read-only or full volume fails immediately
        rather than forty seconds into a render.
        """
        for directory in (
            self.root,
            self.frames_dir,
            self.scenes_dir,
            self.thumbs_dir,
            self.crops_dir,
            self.rebuild_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    def delete(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    # --- the path <-> URL boundary -------------------------------------------

    def url_for(self, path: Path) -> str:
        """The public URL for a file inside this workspace.

        Refuses paths from outside it. The check is not paranoia about our own
        code: this is the one function that decides what a client is allowed to
        ask for, so it is the right place to make "inside the workspace" a
        checked property rather than an assumption.
        """
        resolved = path.resolve()
        root = self.root.resolve()
        if not resolved.is_relative_to(root):
            raise PathEscapeError(f"{path} is not inside workspace {self.job_id}")
        relative = resolved.relative_to(root).as_posix()
        return f"{MEDIA_URL_PREFIX}/{self.job_id}/{relative}"


class WorkspaceManager:
    """Creates, opens and expires workspaces under a single media root.

    Injected with the root rather than reading `Settings`, for the same reason
    `Ffmpeg` is: it keeps `infra` free of inward imports and lets tests point at
    a `tmp_path`.
    """

    def __init__(self, media_root: Path) -> None:
        self._root = media_root.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def create(self, job_id: str) -> Workspace:
        return self.open(job_id).ensure()

    def open(self, job_id: str) -> Workspace:
        """A `Workspace` handle. Does not touch the filesystem, so it is safe to
        call for a job whose files have already been swept away."""
        valid = validate_job_id(job_id)
        return Workspace(job_id=valid, root=self._root / valid)

    def resolve_media_path(self, job_id: str, relative_path: str) -> Path:
        """Map a `/media/{job}/{rest}` request onto a real file.

        Two independent defences, because this is the only route that turns
        client-supplied text into a filesystem path: the job id must match the
        id pattern, and the *resolved* path must still be inside the workspace.
        The second catches `..` segments, absolute paths, and symlinks pointing
        out of the tree - which the first, on its own, would not.
        """
        workspace = self.open(job_id)
        candidate = (workspace.root / relative_path).resolve()
        if not candidate.is_relative_to(workspace.root.resolve()):
            raise PathEscapeError(f"{relative_path!r} escapes workspace {job_id}")
        return candidate

    def sweep(self, ttl_minutes: int, *, now: float | None = None) -> list[str]:
        """Delete workspaces older than the TTL; return the ids removed.

        The Space's disk is ephemeral but not infinite, and a demo that is left
        running will otherwise fill it with videos nobody will watch again. Age
        is taken from the directory's mtime, which the pipeline refreshes as it
        writes - so a long-running job is never swept out from under itself.

        `now` is injectable so the test suite does not have to wait two hours.
        """
        if not self._root.exists():
            return []

        cutoff = (now if now is not None else time.time()) - ttl_minutes * 60
        removed: list[str] = []
        for entry in self._root.iterdir():
            if not entry.is_dir():
                continue
            try:
                if entry.stat().st_mtime >= cutoff:
                    continue
                shutil.rmtree(entry, ignore_errors=True)
            except OSError:  # pragma: no cover - racing with an in-flight job
                log.warning("could not sweep %s", entry, exc_info=True)
                continue
            removed.append(entry.name)

        if removed:
            log.info("swept %d expired workspace(s): %s", len(removed), ", ".join(removed))
        return removed
