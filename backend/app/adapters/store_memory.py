"""In-process job storage. Implements the `JobStore` port.

This is the app's clearest scaling limit, and naming it as a port is the point:
state lives in one process, so two containers would not see each other's jobs and
a restart forgets everything in flight. For a demo on a single free-tier Space
that is the right trade - Redis would cost money and add a moving part to
something with no persistence requirement.

Swapping it is genuinely small. `RedisJobStore` implements these five methods
against a hash and a TTL, and `dependencies.py` changes by one line; nothing else
in the codebase learns about it.
"""

from __future__ import annotations

import asyncio
import logging

from app.domain.models import Job, JobResult

log = logging.getLogger(__name__)


class InMemoryJobStore:
    """Two dicts behind a lock."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._results: dict[str, JobResult] = {}
        # A job is written by the request handler and then repeatedly by a
        # background worker while it is polled by another request. The lock keeps
        # a read-modify-write in `update` from interleaving with itself.
        self._lock = asyncio.Lock()

    async def create(self, job: Job) -> Job:
        async with self._lock:
            self._jobs[job.id] = job
            return job

    async def get(self, job_id: str) -> Job | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update(self, job_id: str, **changes: object) -> Job:
        """Apply a partial change, stamping `updated_at`.

        `model_copy` rather than mutation, because `Job` is handed out to
        request handlers - a caller holding a reference should not see it change
        underneath them mid-response.
        """
        async with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            updated = job.model_copy(update={**changes, "updated_at": _now()})
            self._jobs[job_id] = updated
            return updated

    async def save_result(self, job_id: str, result: JobResult) -> None:
        async with self._lock:
            self._results[job_id] = result

    async def get_result(self, job_id: str) -> JobResult | None:
        async with self._lock:
            return self._results.get(job_id)

    async def delete(self, job_id: str) -> None:
        async with self._lock:
            self._jobs.pop(job_id, None)
            self._results.pop(job_id, None)

    async def job_ids(self) -> list[str]:
        """Every known job. Used by the sweeper to expire state alongside files."""
        async with self._lock:
            return list(self._jobs)


def _now():  # noqa: ANN202 - trivial indirection, kept for testability
    from datetime import UTC, datetime

    return datetime.now(UTC)
