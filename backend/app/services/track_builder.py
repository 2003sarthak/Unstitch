"""Turning per-frame detections into overlay tracks that persist over time.

This is the algorithm the product rests on, and the part that is genuinely ours -
PySceneDetect finds cuts, Gemini reads frames, ffmpeg renders pixels, but nothing
off the shelf turns *"there is a caption in frames 4, 5 and 7"* into *"this
caption is on screen from 1.5s to 4.0s"*.

Why it matters: a detection is a sighting, and a sighting can only be blurred. A
track has a lifetime, which is what makes an overlay a **component** - something
with a start, an end, a region and content, that can be removed in one gated
render pass, listed in a UI, re-rendered with different text, or swapped for a
different image. Every feature downstream is a view of this output.

The algorithm is greedy nearest-match association, the same shape used in simple
multi-object tracking:

1. Walk detections in time order.
2. For each, find the open track it most plausibly continues.
3. Extend that track, or open a new one.
4. Close tracks that have not been seen for longer than the gap tolerance.

Two detections are "the same thing" when their boxes overlap enough **and** their
text agrees. Geometry alone merges a caption with the headline that replaces it
in the same spot; text alone merges two copies of the same word in different
corners. Requiring both is what makes the output correspond to things a person
would point at and name.

Pure functions over pure data - no I/O, no model, no ffmpeg - so every rule below
is tested against hand-written detections in microseconds.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.domain.models import BBox, Detection, OverlayKind, OverlayTrack

log = logging.getLogger(__name__)

#: Below this, two text strings are treated as different content even when the
#: boxes line up. Deliberately forgiving: OCR of the same caption across frames
#: differs by punctuation and the odd character, and word-by-word captions grow
#: as they are spoken.
_TEXT_SIMILARITY_THRESHOLD = 0.6

#: When two sightings carry near-identical text, one box being largely inside
#: the other is enough to associate them even if IoU falls short. This is what
#: handles a caption that is revealed word by word: the box grows with the text,
#: so "we" and "we tried" score only ~0.43 IoU and would otherwise split into two
#: components - the single most common overlay style in short-form video.
_GROWTH_CONTAINMENT_THRESHOLD = 0.7

#: The bar for "strong" text agreement, which is what licenses that relaxation.
#: Higher than the ordinary threshold, because it is being used as a substitute
#: for geometric evidence rather than a complement to it.
_STRONG_TEXT_SIMILARITY = 0.8

#: A track seen in only one sampled frame is usually a misfire - a compression
#: artefact, or a one-off hallucination. Kept anyway when the detector was very
#: confident, because a genuine 1.5s pop-up can legitimately appear once.
#:
#: This value is coupled to what the *detectors* can actually report, and that
#: coupling is easy to break silently: a detector whose confidence ceiling sits
#: below this floor can never have a single-frame detection kept, no matter how
#: strong the evidence. `tests/test_track_builder.py` asserts every detector's
#: achievable range clears it.
DEFAULT_SINGLETON_CONFIDENCE_FLOOR = 0.7


@dataclass
class _OpenTrack:
    """A track still accepting detections. Internal to the algorithm."""

    kind: OverlayKind
    detections: list[Detection] = field(default_factory=list)

    @property
    def last_seen_s(self) -> float:
        return self.detections[-1].t_s

    @property
    def box(self) -> BBox:
        """Union of every member box.

        A caption that gains a word, or drifts a few pixels as it animates in,
        must still be fully covered by the one mask that removes it.
        """
        box = self.detections[0].bbox
        for detection in self.detections[1:]:
            box = box.union(detection.bbox)
        return box

    @property
    def text(self) -> str:
        """The longest observed text.

        Word-by-word captions grow across frames - "we" then "we tried" then "we
        tried three" - so the longest sighting is the complete phrase. Taking the
        first would truncate it and taking the last would miss a caption that
        fades out mid-word.
        """
        return max((d.text for d in self.detections), key=len, default="")

    @property
    def confidence(self) -> float:
        return sum(d.confidence for d in self.detections) / len(self.detections)


def build_tracks(
    detections: Sequence[Detection],
    *,
    iou_threshold: float = 0.5,
    max_gap_s: float = 3.5,
    min_confidence: float = 0.35,
    frame_interval_s: float = 1.5,
    singleton_confidence_floor: float = DEFAULT_SINGLETON_CONFIDENCE_FLOOR,
) -> list[OverlayTrack]:
    """Group detections into overlay tracks.

    `max_gap_s` is what makes this robust to a detector that misses a frame: an
    overlay may vanish for that long and still be treated as one continuous
    element, rather than splitting into two tracks with a hole between them that
    the removal pass would then fail to cover.

    It must therefore exceed **two** sampling intervals, not one. A single missed
    detection at a 1.5s interval leaves a 3.0s gap, so the 2.0s that looks like a
    generous tolerance would split the exact case it exists to absorb.
    """
    usable = sorted(
        (d for d in detections if d.confidence >= min_confidence),
        key=lambda d: (d.t_s, -d.bbox.area),
    )
    if not usable:
        return []

    open_tracks: list[_OpenTrack] = []
    closed: list[_OpenTrack] = []

    for detection in usable:
        # Retire anything that has not been seen recently. Doing this *before*
        # matching means a stale track cannot claim a detection that belongs to
        # a new overlay occupying the same region later in the video.
        still_open = []
        for track in open_tracks:
            (still_open if detection.t_s - track.last_seen_s <= max_gap_s else closed).append(track)
        open_tracks = still_open

        match = _best_match(detection, open_tracks, iou_threshold)
        if match is None:
            open_tracks.append(_OpenTrack(kind=detection.kind, detections=[detection]))
        else:
            match.detections.append(detection)

    closed.extend(open_tracks)
    tracks = [
        _finalise(track, index, frame_interval_s)
        for index, track in enumerate(_worth_keeping(closed, singleton_confidence_floor))
    ]
    log.info("built %d track(s) from %d detection(s)", len(tracks), len(detections))
    return tracks


def _best_match(
    detection: Detection, tracks: Sequence[_OpenTrack], iou_threshold: float
) -> _OpenTrack | None:
    """The open track this detection most plausibly continues.

    Two independent routes to a match, because the two signals fail in different
    places:

    * **Overlap plus agreeing text.** The ordinary case.
    * **Near-identical text plus containment.** IoU punishes a box that changes
      size, which is exactly what a caption does as it is revealed word by word -
      "we" against "we tried" scores about 0.43 and would split. When the text
      says these are the same element, one box sitting inside the other is
      sufficient geometric evidence.

    The second route requires *strong* text agreement precisely because it is
    standing in for geometry rather than confirming it, and it scores below a
    genuine overlap match so a better-fitting track always wins.

    Both are measured against the track's *most recent* box, not its union. The
    union only grows, so matching on it would make a track progressively easier
    to join until it swallowed anything that drifted through the region it had
    ever covered.
    """
    best: _OpenTrack | None = None
    best_score = 0.0
    for track in tracks:
        if track.kind is not detection.kind:
            continue  # a caption never becomes a watermark
        recent = track.detections[-1].bbox
        iou = recent.iou(detection.bbox)

        if iou >= iou_threshold and _text_matches(track.text, detection.text):
            score = iou
        elif _text_strongly_matches(track.text, detection.text) and (
            _containment(recent, detection.bbox) >= _GROWTH_CONTAINMENT_THRESHOLD
        ):
            # Ranked below any real overlap match, so this never outbids one.
            score = _containment(recent, detection.bbox) * 0.5
        else:
            continue

        if score > best_score:
            best, best_score = track, score
    return best


def _containment(a: BBox, b: BBox) -> float:
    """How much of the smaller box lies inside the larger one, 0..1.

    Unlike IoU this is indifferent to a size difference, which is the property
    needed for a growing caption. It is far too permissive on its own - a
    watermark sits fully inside a full-frame popup and would score 1.0 - which is
    why it is only ever consulted alongside strong text agreement.
    """
    ix0, iy0 = max(a.x, b.x), max(a.y, b.y)
    ix1, iy1 = min(a.right, b.right), min(a.bottom, b.bottom)
    intersection = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    smaller = min(a.area, b.area)
    return intersection / smaller if smaller > 0 else 0.0


def _text_strongly_matches(existing: str, incoming: str) -> bool:
    """Near-identical text. Both sides must actually have text.

    An empty string is not evidence of sameness, so image pop-ups and watermarks
    - and everything the stub detector produces - are excluded from the relaxed
    route and must still clear the IoU bar on geometry alone.
    """
    a, b = existing.strip().lower(), incoming.strip().lower()
    if not a or not b:
        return False
    if a.startswith(b) or b.startswith(a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= _STRONG_TEXT_SIMILARITY


def _text_matches(existing: str, incoming: str) -> bool:
    """Whether two sightings carry the same content.

    An empty string on either side is not evidence of difference - image pop-ups
    and watermarks legitimately have no text, and the stub detector never reads
    any - so those fall back to geometry alone. Where both have text, one being
    a prefix of the other counts as a match, which is what handles captions that
    are revealed word by word.
    """
    a, b = existing.strip().lower(), incoming.strip().lower()
    if not a or not b:
        return True
    if a.startswith(b) or b.startswith(a):
        return True
    return SequenceMatcher(None, a, b).ratio() >= _TEXT_SIMILARITY_THRESHOLD


def _worth_keeping(tracks: Sequence[_OpenTrack], floor: float) -> list[_OpenTrack]:
    """Drop one-frame tracks unless the detector was sure.

    Sampling every ~1.5s means a real overlay is almost always seen more than
    once; a single sighting is more often a compression artefact or a one-off
    misread. The confidence escape hatch keeps genuinely brief pop-ups, which do
    exist in this footage and are exactly the kind of thing a user wants removed.
    """
    return [track for track in tracks if len(track.detections) > 1 or track.confidence >= floor]


def _finalise(track: _OpenTrack, index: int, frame_interval_s: float) -> OverlayTrack:
    """Convert an open track into the immutable domain model.

    The time range is widened by half a sampling interval at each end. Sampling
    only proves the overlay was present *at* those instants; it almost certainly
    appeared somewhat before the first sighting and lingered after the last. Not
    padding would leave the overlay visible for a fraction of a second either
    side of the mask - which is far more noticeable than a mask that lingers over
    footage already covered.
    """
    half = frame_interval_s / 2
    start = max(0.0, track.detections[0].t_s - half)
    end = track.detections[-1].t_s + half
    return OverlayTrack(
        id=f"t{index:03d}",
        kind=track.kind,
        text=track.text,
        bbox=track.box,
        start_s=round(start, 3),
        end_s=round(end, 3),
        confidence=round(track.confidence, 3),
        detection_count=len(track.detections),
    )
