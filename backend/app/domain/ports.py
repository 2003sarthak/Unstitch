"""The seams.

Every external system the pipeline touches - yt-dlp, PySceneDetect, Gemini,
ffmpeg, job storage - enters through one of these Protocols. `services/pipeline.py`
imports from this module and never from `adapters`, which buys three things:

* The whole pipeline runs against `vision_stub` with no API key, so the app is
  demoable and the suite is runnable on a machine that has no secrets.
* Swapping an implementation is one line in `dependencies.py`. The in-memory job
  store becomes Redis without the pipeline noticing.
* The orchestration logic can be tested against fakes, in milliseconds, without
  a video file.

`Protocol` rather than ABC on purpose: adapters state nothing about the domain
and inherit from nothing, so the dependency arrow points inward only - a
structural relationship, checked statically, with no base class reaching back
out of the domain to bind an implementation to it.

All I/O methods are `async` because the job runner is an asyncio worker. A
blocking implementation - PySceneDetect is CPU-bound and synchronous - satisfies
the contract by moving its work to a thread, and that decision belongs in the
adapter rather than leaking into every caller.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.domain.models import (
    BBox,
    Detection,
    Job,
    JobResult,
    JobStatus,
    OverlayTrack,
    RemovalMode,
    SampledFrame,
    Scene,
    VideoMeta,
)

#: Called by the pipeline as it advances. The runner supplies one that persists
#: the new status; tests supply one that appends to a list. Progress is pushed
#: through a callback rather than written to a store by the pipeline itself,
#: which is what keeps the pipeline ignorant of how jobs are stored.
ProgressCallback = Callable[[JobStatus], Awaitable[None]]


@runtime_checkable
class MediaIngestor(Protocol):
    """Fetches a video from a URL.

    Uploads deliberately do not go through this port: there is nothing to fetch,
    so the route writes the bytes straight into the workspace. Inventing an
    `UploadIngestor` to make the two paths symmetrical would add an indirection
    that only ever wraps `shutil.copyfileobj`.
    """

    async def fetch(self, url: str, dest: Path) -> Path:
        """Download `url` to `dest`. Raises `InvalidInputError` for a URL that
        cannot be resolved, or one behind an auth wall."""
        ...


@runtime_checkable
class SceneDetector(Protocol):
    """Finds the cuts.

    Kept behind a port even though the implementation is deterministic and local,
    because "how do you decide where a scene starts" is exactly the decision a
    reviewer may want swapped - threshold detection, adaptive, or a model.
    """

    async def detect(self, video: Path) -> list[Scene]:
        """Return scenes covering the whole video, in order and without gaps.

        A video with no detected cuts yields one scene spanning it, never an
        empty list - callers should not have to special-case "unedited footage".
        """
        ...


@runtime_checkable
class FrameSampler(Protocol):
    """Chooses which frames to spend vision quota on, and extracts them.

    Its own port because sampling policy is the main lever on cost and accuracy:
    every frame this returns is part of a request to a rate-limited free tier.
    """

    async def sample(
        self, video: Path, meta: VideoMeta, scenes: Sequence[Scene], dest_dir: Path
    ) -> list[SampledFrame]:
        """Extract the chosen frames into `dest_dir` and describe them.

        The destination is passed in rather than derived, because the workspace
        layout belongs to `infra.workspace` and an adapter should not be a second
        place that knows where a job's files live.
        """
        ...


@runtime_checkable
class VisionDetector(Protocol):
    """Reads what is on screen.

    The one port with two implementations that matter: `vision_gemini` and
    `vision_stub`. Everything downstream is written against this signature, so
    the pipeline cannot tell which one it is talking to - which is precisely why
    a missing API key degrades the output instead of breaking the app.
    """

    async def detect(self, frames: Sequence[SampledFrame]) -> list[Detection]:
        """Detections for every frame given, in one call.

        Batching is the adapter's business, not the caller's: Gemini charges per
        request, so several frames go into each one. A per-frame signature would
        have made that optimisation impossible without changing the port.
        """
        ...


@runtime_checkable
class VideoEditor(Protocol):
    """Everything that writes a video file."""

    async def probe(self, path: Path) -> VideoMeta: ...

    async def normalise(self, source: Path, dest: Path) -> VideoMeta:
        """Re-encode to the one format the rest of the pipeline assumes."""
        ...

    async def remove_overlays(
        self,
        source: Path,
        dest: Path,
        tracks: Sequence[OverlayTrack],
        meta: VideoMeta,
        mode: RemovalMode,
        padding_pct: float,
    ) -> None:
        """Erase every track's region, each gated to its own time range.

        Takes the whole sequence rather than being called once per track, because
        the implementation composes them into a single filter graph and one
        render pass. A per-track method would force N re-encodes and quietly
        become the slowest thing in the app.
        """
        ...

    async def cut(self, source: Path, dest: Path, start_s: float, end_s: float) -> None: ...

    async def thumbnail(self, source: Path, dest: Path, at_s: float) -> None: ...

    async def crop(
        self, source: Path, dest: Path, bbox: BBox, meta: VideoMeta, at_s: float
    ) -> None:
        """A still of one overlay's region - the thumbnail in the overlay list."""
        ...


@runtime_checkable
class JobStore(Protocol):
    """Where job state lives between the 202 and the poll that follows it.

    In-memory for now. The port exists because that choice is the app's clearest
    scaling limit - it confines state to one process - and naming the seam is
    what makes "swap in Redis" a small, obvious change rather than a rewrite.
    """

    async def create(self, job: Job) -> Job: ...

    async def get(self, job_id: str) -> Job | None: ...

    async def update(self, job_id: str, **changes: object) -> Job:
        """Apply a partial change and return the new state.

        A partial update rather than a whole-object save because two writers -
        the request handler and the background worker - touch the same job, and
        read-modify-write would let one silently discard the other's progress.
        """
        ...

    async def save_result(self, job_id: str, result: JobResult) -> None: ...

    async def get_result(self, job_id: str) -> JobResult | None:
        """The finished analysis, or None if the job is not done.

        Stored separately from `Job` because the two have different lifetimes and
        very different sizes: status is polled every 1.5s and must stay cheap,
        while the result is fetched once and carries every scene and track.
        """
        ...

    async def delete(self, job_id: str) -> None:
        """Forget a job entirely. Used by the TTL sweeper, so that expiring a
        workspace on disk does not leave its status behind claiming success."""
        ...
