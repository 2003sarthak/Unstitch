"""The application's error vocabulary.

Split by *whose fault it is*, not by which component raised it, because that is
the axis the API layer needs: `InvalidInputError` becomes a 4xx and is shown to
the user verbatim, `ProcessingError` becomes a 5xx and is logged with a stack
trace. A hierarchy organised by component instead would force the HTTP layer to
re-derive that distinction for every new error type.

Adapters translate foreign failures into these. `FFmpegError` from `infra` is
deliberately not one of them - keeping infra free of domain imports is what
makes the ffmpeg wrapper reusable and independently testable.
"""

from __future__ import annotations


class UnstitchError(Exception):
    """Base for every error this application raises deliberately."""


class InvalidInputError(UnstitchError):
    """The request cannot be served, and the caller can fix it.

    The message is user-facing, so it must say what to do differently.
    """


class MediaTooLongError(InvalidInputError):
    """Longer than `MAX_VIDEO_SECONDS`.

    Rejected rather than silently trimmed: a user who uploads a three-minute
    video and receives a de-edited ninety seconds back would reasonably call
    that a bug, and the config comment promises rejection.
    """


class UnsupportedMediaError(InvalidInputError):
    """Not a video we can process - no video stream, or an unreadable container."""


class ProcessingError(UnstitchError):
    """A step failed for a reason the caller could not have prevented.

    Wraps tool-level failures (a non-zero ffmpeg exit, a vision API error) so
    that the pipeline and the API never need to know which tool produced them.
    """
