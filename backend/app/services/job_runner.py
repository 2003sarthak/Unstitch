"""Background job execution.

This module is why the backend is a container and not a serverless function. The
API answers `202 Accepted` in milliseconds and the work continues afterwards; a
function that is killed when its response is sent cannot do that. It is the
single most load-bearing deployment decision in the project.

The design is an `asyncio.Queue` plus a fixed pool of worker tasks. `Redis` and
`Celery` would buy durability across restarts, which is worth nothing here -
storage is ephemeral anyway, so a restart loses the output whether or not it
remembers the request. What the pool *does* buy is a hard concurrency ceiling:
video encoding is CPU-bound, and two jobs on two shared vCPUs is the point past
which everything gets slower without anything getting done sooner.
"""

from __future__ import annotations

import asyncio
import logging

from app.domain.errors import UnstitchError
from app.domain.models import Job, JobRequest, JobStatus
from app.domain.ports import JobStore
from app.infra.workspace import WorkspaceManager
from app.services.pipeline import Pipeline

log = logging.getLogger(__name__)


class JobRunner:
    """A bounded pool of workers draining a queue of jobs."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        store: JobStore,
        workspaces: WorkspaceManager,
        max_concurrent: int = 2,
    ) -> None:
        self._pipeline = pipeline
        self._store = store
        self._workspaces = workspaces
        self._max_concurrent = max_concurrent
        self._queue: asyncio.Queue[tuple[str, JobRequest]] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []

    # --- lifecycle -----------------------------------------------------------

    async def start(self) -> None:
        if self._workers:
            return
        self._workers = [
            asyncio.create_task(self._work(n), name=f"unstitch-worker-{n}")
            for n in range(self._max_concurrent)
        ]
        log.info("job runner started with %d worker(s)", self._max_concurrent)

    async def stop(self) -> None:
        """Cancel the workers and wait for them.

        Awaiting after cancelling matters: a worker mid-render owns an ffmpeg
        subprocess, and dropping the task without letting it unwind would leave
        that process orphaned for the life of the container.
        """
        for worker in self._workers:
            worker.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        log.info("job runner stopped")

    # --- submission ----------------------------------------------------------

    async def submit(self, job_id: str, request: JobRequest) -> Job:
        """Queue a job and return it already persisted as `queued`.

        Persisting *before* enqueuing, rather than inside the worker, is what
        makes the client's very next poll meaningful. The other order has a
        window where `GET /api/jobs/{id}` 404s for a job the API just accepted.
        """
        job = await self._store.create(Job(id=job_id, status=JobStatus.QUEUED))
        await self._queue.put((job_id, request))
        return job

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    # --- the worker loop -----------------------------------------------------

    async def _work(self, number: int) -> None:
        while True:
            job_id, request = await self._queue.get()
            try:
                await self._run_one(job_id, request)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - the loop must never die
                log.exception("worker %d crashed on job %s", number, job_id)
            finally:
                self._queue.task_done()

    async def _run_one(self, job_id: str, request: JobRequest) -> None:
        workspace = self._workspaces.create(job_id)

        async def on_progress(status: JobStatus) -> None:
            await self._store.update(job_id, status=status)

        try:
            result = await self._pipeline.run(workspace, request, on_progress)
        except UnstitchError as exc:
            # Ours, and phrased for a person: surface it as written.
            log.warning("job %s failed: %s", job_id, exc)
            await self._store.update(job_id, status=JobStatus.FAILED, error=str(exc))
        except asyncio.CancelledError:
            await self._store.update(
                job_id, status=JobStatus.FAILED, error="server shut down mid-job"
            )
            raise
        except Exception:
            # Not ours, so the text is a stack trace or a vendor's internals.
            # Log it in full, tell the user something true and useless to an
            # attacker.
            log.exception("job %s failed unexpectedly", job_id)
            await self._store.update(
                job_id,
                status=JobStatus.FAILED,
                error="something went wrong while processing this video",
            )
        else:
            await self._store.save_result(job_id, result)
            await self._store.update(job_id, status=JobStatus.DONE)


class WorkspaceSweeper:
    """Expires old jobs on a timer.

    The Space's disk is ephemeral but not infinite, and a demo left running would
    otherwise fill it with videos nobody will watch again. Files and job state are
    dropped together - expiring one without the other would leave a job that
    reports success and serves 404s for every URL in its result.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        workspaces: WorkspaceManager,
        ttl_minutes: int,
        interval_s: float = 600.0,
    ) -> None:
        self._store = store
        self._workspaces = workspaces
        self._ttl_minutes = ttl_minutes
        self._interval_s = interval_s
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop(), name="unstitch-sweeper")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
            self._task = None

    async def sweep_once(self) -> list[str]:
        # The filesystem walk is blocking; off the event loop it goes.
        removed = await asyncio.to_thread(self._workspaces.sweep, self._ttl_minutes)
        for job_id in removed:
            await self._store.delete(job_id)
        return removed

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self._interval_s)
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover
                log.exception("workspace sweep failed")
