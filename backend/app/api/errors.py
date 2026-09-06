"""Domain errors to HTTP, in one place.

Routes raise domain errors and never build an error response. That is why
`domain/errors.py` is split by whose fault a failure is rather than by which
component raised it: the mapping below is then a short table instead of a
per-endpoint judgement call, and a new error type inherits sensible behaviour by
choosing its base class.

The response shape is identical for every failure - `{"error", "type"}` - so the
frontend has exactly one thing to parse whatever went wrong.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.domain.errors import (
    InvalidInputError,
    MediaTooLongError,
    ProcessingError,
    UnstitchError,
    UnsupportedMediaError,
)

log = logging.getLogger(__name__)

#: Most specific first - `MediaTooLongError` is an `InvalidInputError`, so order
#: is what distinguishes 413 from 400.
_STATUS_BY_TYPE: tuple[tuple[type[UnstitchError], int], ...] = (
    (MediaTooLongError, status.HTTP_413_CONTENT_TOO_LARGE),
    (UnsupportedMediaError, status.HTTP_415_UNSUPPORTED_MEDIA_TYPE),
    (InvalidInputError, status.HTTP_400_BAD_REQUEST),
    (ProcessingError, status.HTTP_500_INTERNAL_SERVER_ERROR),
)


def status_for(error: UnstitchError) -> int:
    for error_type, code in _STATUS_BY_TYPE:
        if isinstance(error, error_type):
            return code
    return status.HTTP_500_INTERNAL_SERVER_ERROR


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(UnstitchError)
    async def _handle_domain_error(_: Request, exc: UnstitchError) -> JSONResponse:
        code = status_for(exc)
        if code >= 500:
            # A 5xx is our bug, so it needs a stack trace in the log. A 4xx is
            # the caller's mistake and would only be noise at that level.
            log.exception("request failed: %s", exc)
        return JSONResponse(
            status_code=code,
            content={"error": str(exc), "type": type(exc).__name__},
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        """Anything we did not anticipate.

        The message is deliberately generic: an unhandled exception's text is a
        stack trace or a vendor's internals, and neither belongs in a response.
        The full detail goes to the log, where it is useful and not public.
        """
        log.exception("unhandled error", exc_info=exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "internal server error", "type": "InternalError"},
        )
