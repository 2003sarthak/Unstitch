"""Job endpoints: submit, poll, fetch result.

The shape is `202 Accepted` plus polling, because de-editing a video takes tens
of seconds and no sensible client holds an HTTP connection open for that. It is
also the reason the backend is a container rather than a serverless function -
the work has to outlive the response that acknowledged it.

Two submission routes rather than one endpoint that sniffs its content type.
A single `POST /api/jobs` accepting either JSON or multipart was considered;
splitting them gives each a precise OpenAPI schema, which is what the frontend's
types and the interactive docs read. One handler branching on `Content-Type`
would document as neither.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from app.dependencies import Container, get_container
from app.domain.errors import InvalidInputError, MediaTooLongError
from app.domain.models import Job, JobRequest, JobResult, JobStatus, RemovalMode

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

#: Uploads are streamed in chunks rather than read whole. A 100MB file read into
#: memory is survivable; several at once on a 2-vCPU Space is not.
_UPLOAD_CHUNK_BYTES = 1024 * 1024


class JobAccepted(BaseModel):
    """The 202 body. Carries the id the client will poll."""

    job_id: str
    status: JobStatus


class JobProgress(BaseModel):
    """The polled body. Kept small - this is requested every 1.5 seconds.

    `stage` and `progress` are both present because they answer different
    questions: the label says what is happening, the number says how far along.
    A bar alone cannot say "detecting overlays", and a label alone cannot show
    movement during a long stage.
    """

    job_id: str
    status: JobStatus
    stage: str = Field(description="Human-readable stage name")
    progress: float = Field(ge=0.0, le=1.0)
    error: str | None = None

    @classmethod
    def of(cls, job: Job) -> JobProgress:
        return cls(
            job_id=job.id,
            status=job.status,
            stage=job.status.value.replace("_", " "),
            progress=job.status.progress,
            error=job.error,
        )


@router.post("", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_job_from_url(
    request: JobRequest,
    container: Annotated[Container, Depends(get_container)],
) -> JobAccepted:
    """Start a de-edit from a video URL."""
    if not request.url:
        raise InvalidInputError("a url is required; use /api/jobs/upload for a file")

    job = await container.runner.submit(_new_job_id(), request)
    log.info("accepted job %s from url", job.id)
    return JobAccepted(job_id=job.id, status=job.status)


@router.post("/upload", response_model=JobAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_job_from_upload(
    container: Annotated[Container, Depends(get_container)],
    file: Annotated[UploadFile, File(description="The video file")],
    removal_mode: Annotated[RemovalMode, Form()] = RemovalMode.DELOGO,
) -> JobAccepted:
    """Start a de-edit from an uploaded file.

    The bytes are written straight into the job's workspace, which is why upload
    does not go through the ingest port: there is nothing to fetch, and wrapping
    a file copy in an adapter would be indirection for its own sake.
    """
    job_id = _new_job_id()
    workspace = container.workspaces.create(job_id)
    await _save_upload(file, workspace.original, container.settings.max_upload_bytes)

    job = await container.runner.submit(job_id, JobRequest(removal_mode=removal_mode))
    log.info("accepted job %s from upload (%s)", job.id, file.filename)
    return JobAccepted(job_id=job.id, status=job.status)


@router.get("/{job_id}", response_model=JobProgress)
async def get_job(
    job_id: str, container: Annotated[Container, Depends(get_container)]
) -> JobProgress:
    job = await container.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    return JobProgress.of(job)


@router.get("/{job_id}/result", response_model=JobResult)
async def get_job_result(
    job_id: str, container: Annotated[Container, Depends(get_container)]
) -> JobResult:
    """The finished analysis.

    Separate from the status endpoint because the two have very different costs
    and cadences: status is polled every 1.5s and stays tiny, while this is
    fetched once and carries every scene and every track.
    """
    job = await container.store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such job")
    if job.status is JobStatus.FAILED:
        raise HTTPException(status.HTTP_409_CONFLICT, job.error or "job failed")
    if job.status is not JobStatus.DONE:
        # 409 rather than 404: the job exists and the answer will arrive, which
        # is a different situation from a job that never existed.
        raise HTTPException(status.HTTP_409_CONFLICT, f"job is {job.status.value}")

    result = await container.store.get_result(job_id)
    if result is None:  # pragma: no cover - would mean the store lost it
        raise HTTPException(status.HTTP_404_NOT_FOUND, "result is no longer available")
    return result


# --- helpers -----------------------------------------------------------------


def _new_job_id() -> str:
    """`uuid4().hex` - unguessable, and already matching the id pattern the
    workspace validator enforces before it will build a path from one."""
    return uuid.uuid4().hex


async def _save_upload(file: UploadFile, dest, limit_bytes: int) -> None:
    """Stream an upload to disk, enforcing the size cap as it arrives.

    Checked while writing rather than from `Content-Length`, because that header
    is supplied by the client and a body can simply keep coming. Writing first
    and checking after would mean the disk is already full by the time the limit
    is noticed.
    """
    written = 0
    try:
        with dest.open("wb") as sink:
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > limit_bytes:
                    raise MediaTooLongError(f"file is larger than {limit_bytes // (1024 * 1024)}MB")
                sink.write(chunk)
    except MediaTooLongError:
        dest.unlink(missing_ok=True)  # do not leave a truncated file behind
        raise
    finally:
        await file.close()

    if written == 0:
        dest.unlink(missing_ok=True)
        raise InvalidInputError("the uploaded file was empty")
