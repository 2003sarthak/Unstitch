"""The only module in Unstitch that spawns a process.

Two ideas carry this file:

**One wrapper.** Every probe, cut, thumbnail and render in the app goes through
`Ffmpeg.run`. Timeouts, logging, and turning a non-zero exit into a useful
exception are written once. Nowhere else builds a shell string, so there is no
place for a filename containing a space - or a quote - to become a
command-injection bug.

**Filter graphs are data, not strings.** `Filter`/`FilterChain`/`FilterGraph`
are frozen dataclasses that *render* to ffmpeg syntax. The interesting logic in
this project - deciding which regions to erase, and when - therefore produces a
value we can assert on in a unit test, with no video and no subprocess. Only the
final `.render()` is stringly-typed, and it is exercised once rather than at
every call site.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.domain.models import VideoMeta

log = logging.getLogger(__name__)

# ffmpeg writes progress and diagnostics to stderr; on failure we keep the tail,
# which is where the actual reason lives. The head is banner noise.
_STDERR_TAIL_CHARS = 2_000

# Applied to every ffmpeg invocation:
#   -nostdin  a background render must never block waiting on a terminal
#   -y        we only ever write inside a workspace we own, so overwrite is right
_BASE_FLAGS: tuple[str, ...] = ("-hide_banner", "-nostdin", "-loglevel", "error", "-y")


# =============================================================================
#  Errors
# =============================================================================


class FFmpegError(RuntimeError):
    """A failed ffmpeg/ffprobe invocation, with enough context to debug it.

    Intentionally *not* a domain error. Adapters translate this into one, which
    is exactly what an adapter is for - it means nothing above the adapter layer
    has to know that ffmpeg is what failed.
    """

    def __init__(self, argv: Sequence[str], returncode: int, stderr: str) -> None:
        self.argv = list(argv)
        self.returncode = returncode
        self.stderr = stderr
        tail = stderr.strip()[-_STDERR_TAIL_CHARS:]
        super().__init__(f"ffmpeg exited {returncode}: {tail or '<no stderr>'}")

    @property
    def command(self) -> str:
        return shlex.join(self.argv)


class FFmpegTimeoutError(FFmpegError):
    """The process outlived its budget and was killed.

    Distinct from a generic failure because the caller's response differs: a
    timeout usually means the input was bigger than we planned for, not that the
    command was wrong.
    """


# =============================================================================
#  Filter graph value objects
# =============================================================================

# Inside a filter description ffmpeg gives these characters structural meaning:
# ":" separates parameters, "," chains filters, ";" separates chains, "[]" mark
# stream labels, "'" quotes and "\" escapes. A value containing any of them must
# escape it. We escape rather than quote because escaping composes - a quoted
# region that itself contains a quote needs escaping anyway, so quoting would
# only add a second rule on top of this one.
_ESCAPES = str.maketrans({ch: "\\" + ch for ch in "\\'[],;:"})


def escape_value(value: str) -> str:
    """Escape a filter parameter value.

    `str.translate` is a single pass over the input, so a backslash that was
    already in the value cannot be escaped twice.
    """
    return value.translate(_ESCAPES)


def _format_value(value: str | int | float) -> str:
    """Render a parameter value deterministically.

    Floats go through fixed-point rather than `str()`, because `str(1e-05)` is
    `"1e-05"` and ffmpeg's expression parser rejects that where it wants a
    number. Trailing zeros are trimmed so a rendered graph stays readable in a
    test assertion.
    """
    if isinstance(value, bool):  # bool is an int subclass - catch it first
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def between(start_s: float, end_s: float) -> str:
    """A timeline expression gating a filter to the window `[start_s, end_s]`.

    This is the mechanism the whole removal step rests on: rather than masking
    every frame, each detected overlay contributes one filter that is only
    active while that overlay is actually on screen. N overlays then cost one
    render pass, not N.
    """
    return f"between(t,{start_s:.3f},{end_s:.3f})"


@dataclass(frozen=True, slots=True)
class Filter:
    """One ffmpeg filter, e.g. `delogo=x=10:y=20:w=100:h=40:enable=...`."""

    name: str
    params: Mapping[str, str | int | float] = field(default_factory=dict)
    #: An ffmpeg timeline expression - see `between()`. This gets its own field
    #: rather than being another entry in `params` because time-gating is the
    #: crux of this app, and a caller should not be able to misspell the key.
    enable: str | None = None

    def render(self) -> str:
        parts = [f"{k}={escape_value(_format_value(v))}" for k, v in self.params.items()]
        if self.enable is not None:
            parts.append(f"enable={escape_value(self.enable)}")
        if not parts:
            return self.name
        joined = ":".join(parts)
        return f"{self.name}={joined}"


@dataclass(frozen=True, slots=True)
class FilterChain:
    """Filters applied in sequence to one stream, optionally labelled.

    Labels are only needed once a graph has more than one input - the caption
    re-render and image-swap features in step 13. Removal needs none, which is
    why `FilterGraph` can pick the cheaper `-vf` form for it.
    """

    filters: Sequence[Filter]
    inputs: Sequence[str] = ()
    outputs: Sequence[str] = ()

    def render(self) -> str:
        body = ",".join(f.render() for f in self.filters)
        lead = "".join(f"[{label}]" for label in self.inputs)
        trail = "".join(f"[{label}]" for label in self.outputs)
        return f"{lead}{body}{trail}"

    @property
    def is_labelled(self) -> bool:
        return bool(self.inputs or self.outputs)


@dataclass(frozen=True, slots=True)
class FilterGraph:
    """A complete filter graph, plus the knowledge of how to hand it to ffmpeg."""

    chains: Sequence[FilterChain] = ()

    @property
    def is_empty(self) -> bool:
        """True when there is nothing to do.

        A real case, not a degenerate one: a video where the detector finds no
        overlays should be copied through untouched rather than pushed through
        an empty graph, which ffmpeg rejects outright.
        """
        return not any(chain.filters for chain in self.chains)

    @property
    def is_simple(self) -> bool:
        """A single unlabelled chain can use `-vf`, which is cheaper to read and
        does not force ffmpeg into the full graph builder."""
        return len(self.chains) == 1 and not self.chains[0].is_labelled

    def render(self) -> str:
        return ";".join(chain.render() for chain in self.chains)

    def to_args(self) -> list[str]:
        """The flag *and* its value, so call sites never choose between `-vf` and
        `-filter_complex` themselves - that choice is derivable from the graph,
        so duplicating it at every call site would be one more thing to get
        wrong."""
        if self.is_empty:
            return []
        return ["-vf" if self.is_simple else "-filter_complex", self.render()]


# =============================================================================
#  Probing
# =============================================================================


class Ffmpeg:
    """Async wrapper around the ffmpeg and ffprobe binaries.

    Constructed with explicit paths rather than reading `Settings` itself, so it
    stays injectable and `infra` keeps its no-inward-imports property.
    """

    def __init__(
        self,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
        default_timeout_s: float = 300.0,
    ) -> None:
        self._ffmpeg = ffmpeg_path
        self._ffprobe = ffprobe_path
        self._default_timeout_s = default_timeout_s

    # --- process plumbing ----------------------------------------------------

    async def _exec(
        self, program: str, args: Sequence[str], timeout_s: float | None
    ) -> tuple[str, str]:
        argv = [program, *args]
        log.debug("exec: %s", shlex.join(argv))

        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        budget = self._default_timeout_s if timeout_s is None else timeout_s
        try:
            raw_out, raw_err = await asyncio.wait_for(proc.communicate(), budget)
        except TimeoutError:
            # Without this the process survives the cancelled await and keeps
            # burning a core for the remaining life of the container.
            proc.kill()
            await proc.wait()
            raise FFmpegTimeoutError(argv, -1, f"timed out after {budget:.0f}s") from None

        stdout = raw_out.decode("utf-8", errors="replace")
        stderr = raw_err.decode("utf-8", errors="replace")
        if proc.returncode != 0:
            raise FFmpegError(argv, proc.returncode or -1, stderr)
        return stdout, stderr

    async def run(self, args: Sequence[str], *, timeout_s: float | None = None) -> str:
        """Run ffmpeg with the standard flags prepended.

        Returns stderr, because that is where ffmpeg reports what it did.
        """
        _, stderr = await self._exec(self._ffmpeg, [*_BASE_FLAGS, *args], timeout_s)
        return stderr

    # --- probing -------------------------------------------------------------

    async def probe(self, path: Path, *, timeout_s: float | None = 30.0) -> VideoMeta:
        stdout, _ = await self._exec(
            self._ffprobe,
            [
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(path),
            ],
            timeout_s,
        )
        return parse_probe(path, json.loads(stdout))


def parse_probe(path: Path, payload: Mapping[str, object]) -> VideoMeta:
    """Pure translation of ffprobe JSON into `VideoMeta`.

    Split out from `Ffmpeg.probe` so the parsing - which is where the fiddly
    cases live - can be unit-tested against captured payloads rather than only
    against whatever media happens to be on disk.
    """
    streams: list[Mapping[str, object]] = list(payload.get("streams", []))  # type: ignore[arg-type]
    fmt: Mapping[str, object] = payload.get("format", {})  # type: ignore[assignment]

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path.name} contains no video stream")

    return VideoMeta(
        duration_s=_duration(fmt, video),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        fps=_fps(str(video.get("avg_frame_rate") or "0/0")),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
        video_codec=str(video.get("codec_name") or "unknown"),
        rotation=_rotation(video),
    )


def _duration(fmt: Mapping[str, object], video: Mapping[str, object]) -> float:
    """Container duration, falling back to the video stream's own.

    Some remuxed downloads carry no `format.duration` at all, and a zero here
    would silently produce a video we believe has no frames worth sampling.
    """
    for candidate in (fmt.get("duration"), video.get("duration")):
        try:
            value = float(candidate)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return 0.0


def _fps(avg_frame_rate: str) -> float:
    """ffprobe reports frame rate as an exact rational - `30000/1001`, not
    `29.97` - so NTSC rates stay lossless. Divide carefully: variable frame rate
    streams report `0/0`.
    """
    numerator, _, denominator = avg_frame_rate.partition("/")
    try:
        den = float(denominator or 1)
        return float(numerator) / den if den else 0.0
    except ValueError:
        return 0.0


def _rotation(video: Mapping[str, object]) -> int:
    """Read rotation from the display matrix, or the legacy `rotate` tag.

    Normalised to 0/90/180/270: a display matrix reports -90 where the old tag
    reported 270, and the two must not disagree downstream.
    """
    side_data_list = video.get("side_data_list") or []
    for side_data in side_data_list:  # type: ignore[union-attr]
        if "rotation" in side_data:
            return int(round(float(side_data["rotation"]))) % 360
    tags: Mapping[str, object] = video.get("tags", {})  # type: ignore[assignment]
    try:
        return int(round(float(tags.get("rotate")))) % 360  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
