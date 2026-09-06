"""Serving job artefacts: `/media/{job_id}/{path}`.

A route rather than a `StaticFiles` mount, for one reason: this is the only place
in the app where client-supplied text becomes a filesystem path, and routing it
through `WorkspaceManager.resolve_media_path` means it inherits the two defences
that already have tests - the job id must match the id pattern, and the
*resolved* path must still be inside that job's workspace.

Mounting the media root directly would serve any file under it, including
another job's workspace, without either check.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from app.dependencies import Container, get_container
from app.infra.workspace import InvalidJobIdError, PathEscapeError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/media", tags=["media"])

#: Job outputs are immutable once written and their ids are unguessable, so they
#: can be cached hard. This matters for the preview player, which re-requests
#: byte ranges as the user scrubs.
_CACHE_CONTROL = "public, max-age=3600"


@router.get("/{job_id}/{path:path}")
async def get_media(
    job_id: str,
    path: str,
    container: Annotated[Container, Depends(get_container)],
) -> FileResponse:
    try:
        resolved = container.workspaces.resolve_media_path(job_id, path)
    except (InvalidJobIdError, PathEscapeError):
        # Deliberately 404 rather than 403. A traversal attempt gets the same
        # answer as a typo, so probing cannot distinguish "blocked" from "absent"
        # and map what exists.
        log.warning("rejected media request: job=%r path=%r", job_id, path)
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found") from None

    if not resolved.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")

    return FileResponse(resolved, headers={"Cache-Control": _CACHE_CONTROL})
