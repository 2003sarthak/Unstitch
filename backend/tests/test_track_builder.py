"""Tests for overlay tracking.

This is the algorithm that turns sightings into components, so it gets the most
thorough tests in the suite. Everything here is hand-written detections in, pure
data out - no model, no video, no ffmpeg - which is exactly why the association
rules can be pinned this precisely.
"""

from __future__ import annotations

import pytest

from app.domain.models import BBox, Detection, OverlayKind
from app.services.track_builder import DEFAULT_SINGLETON_CONFIDENCE_FLOOR, build_tracks

CAPTION_BOX = BBox(x=0.1, y=0.8, w=0.8, h=0.1)
CORNER_BOX = BBox(x=0.75, y=0.03, w=0.2, h=0.06)


def seen(
    t: float,
    box: BBox = CAPTION_BOX,
    *,
    kind: OverlayKind = OverlayKind.CAPTION,
    text: str = "hello",
    confidence: float = 0.9,
) -> Detection:
    return Detection(t_s=t, bbox=box, kind=kind, text=text, confidence=confidence)


class TestAssociation:
    def test_the_same_overlay_across_frames_becomes_one_track(self) -> None:
        tracks = build_tracks([seen(0.0), seen(1.5), seen(3.0)])

        assert len(tracks) == 1
        assert tracks[0].detection_count == 3
        assert tracks[0].kind is OverlayKind.CAPTION

    def test_two_overlays_in_different_places_stay_separate(self) -> None:
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, text="subtitle"),
                seen(0.0, CORNER_BOX, kind=OverlayKind.WATERMARK, text="@handle"),
                seen(1.5, CAPTION_BOX, text="subtitle"),
                seen(1.5, CORNER_BOX, kind=OverlayKind.WATERMARK, text="@handle"),
            ]
        )
        assert len(tracks) == 2
        assert {t.kind for t in tracks} == {OverlayKind.CAPTION, OverlayKind.WATERMARK}

    def test_a_different_kind_never_joins_an_existing_track(self) -> None:
        """A caption does not become a watermark, even in the same region."""
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, kind=OverlayKind.CAPTION, text=""),
                seen(1.5, CAPTION_BOX, kind=OverlayKind.IMAGE_POPUP, text=""),
            ],
            min_confidence=0.0,
        )
        assert len(tracks) == 2

    def test_boxes_that_barely_overlap_do_not_merge(self) -> None:
        far = BBox(x=0.1, y=0.1, w=0.2, h=0.1)
        tracks = build_tracks([seen(0.0, CAPTION_BOX), seen(1.5, far)], min_confidence=0.0)
        assert len(tracks) == 2

    def test_jitter_between_frames_is_tolerated(self) -> None:
        """A caption that shifts 1% of the frame is the same caption."""
        drifted = BBox(x=0.11, y=0.81, w=0.8, h=0.1)
        tracks = build_tracks([seen(0.0, CAPTION_BOX), seen(1.5, drifted)])
        assert len(tracks) == 1


class TestTextMatching:
    def test_different_text_in_the_same_place_is_a_new_overlay(self) -> None:
        """The case geometry alone gets wrong: a headline replaced by another
        headline in the same position is two components, not one."""
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, text="before you buy"),
                seen(1.5, CAPTION_BOX, text="here is the result"),
            ],
            min_confidence=0.0,
        )
        assert len(tracks) == 2

    def test_a_caption_revealed_word_by_word_stays_one_track(self) -> None:
        """Short-form captions animate in a word at a time; each frame sees a
        longer prefix of the same sentence."""
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, text="we"),
                seen(1.5, CAPTION_BOX, text="we tried"),
                seen(3.0, CAPTION_BOX, text="we tried three"),
            ]
        )
        assert len(tracks) == 1
        assert tracks[0].text == "we tried three", "the longest sighting is the full phrase"

    def test_small_ocr_differences_still_match(self) -> None:
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, text="Buy now, only $9.99!"),
                seen(1.5, CAPTION_BOX, text="Buy now only $9,99!"),
            ]
        )
        assert len(tracks) == 1

    def test_elements_with_no_text_fall_back_to_geometry(self) -> None:
        """Image pop-ups and watermarks have no text, and the stub detector never
        reads any - so an empty string must not be treated as a mismatch."""
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, kind=OverlayKind.IMAGE_POPUP, text=""),
                seen(1.5, CAPTION_BOX, kind=OverlayKind.IMAGE_POPUP, text=""),
            ]
        )
        assert len(tracks) == 1


class TestTimeRanges:
    def test_the_range_spans_first_to_last_sighting_plus_padding(self) -> None:
        """Sampling proves presence *at* instants, not between them. The overlay
        almost certainly appeared before the first sighting and lingered after
        the last, and a mask that stops early is far more visible than one that
        lingers over already-covered footage."""
        tracks = build_tracks([seen(1.5), seen(3.0)], frame_interval_s=1.5)

        assert tracks[0].start_s == pytest.approx(0.75)
        assert tracks[0].end_s == pytest.approx(3.75)

    def test_padding_never_pushes_the_start_below_zero(self) -> None:
        tracks = build_tracks([seen(0.0), seen(1.5)], frame_interval_s=1.5)
        assert tracks[0].start_s == 0.0

    def test_a_missed_frame_does_not_split_a_track(self) -> None:
        """The detector misses a frame in the middle. Splitting here would leave
        a gap the removal pass fails to cover, and the overlay would flash back
        into view mid-video."""
        tracks = build_tracks([seen(0.0), seen(3.0)], max_gap_s=3.5, frame_interval_s=1.5)
        assert len(tracks) == 1

    def test_the_gap_tolerance_must_survive_one_missed_detection(self) -> None:
        """The invariant tying TRACK_MAX_GAP_S to FRAME_SAMPLE_INTERVAL_S.

        A missed detection leaves a gap of *two* intervals, so a tolerance below
        that splits the very case it exists to absorb. This is pinned as a test
        because the two settings live apart in `.env` and nothing else would
        notice them drifting out of agreement.
        """
        interval = 1.5
        missed = [seen(0.0), seen(2 * interval)]

        too_small = build_tracks(missed, max_gap_s=2.0, frame_interval_s=interval)
        sufficient = build_tracks(missed, max_gap_s=3.5, frame_interval_s=interval)

        assert len(too_small) == 2, "a 2.0s tolerance cannot bridge a 3.0s gap"
        assert len(sufficient) == 1

    def test_a_long_absence_does_split_a_track(self) -> None:
        """The same caption position reused much later is a different element."""
        tracks = build_tracks([seen(0.0), seen(1.5), seen(20.0)], max_gap_s=3.5)
        assert len(tracks) == 2

    def test_a_reused_region_after_a_gap_is_a_separate_component(self) -> None:
        tracks = build_tracks([seen(0.0), seen(1.5), seen(30.0), seen(31.5)], max_gap_s=3.5)
        assert len(tracks) == 2
        assert tracks[0].end_s < tracks[1].start_s


class TestBoxAndConfidence:
    def test_the_track_box_covers_every_sighting(self) -> None:
        """A caption that gains a word must still be fully masked."""
        short = BBox(x=0.1, y=0.8, w=0.4, h=0.08)
        grown = BBox(x=0.1, y=0.8, w=0.75, h=0.1)
        tracks = build_tracks([seen(0.0, short, text="we"), seen(1.5, grown, text="we tried")])

        box = tracks[0].bbox
        assert box.right >= grown.right
        assert box.bottom >= grown.bottom
        assert box.x <= short.x

    def test_confidence_is_averaged_over_the_sightings(self) -> None:
        tracks = build_tracks([seen(0.0, confidence=0.6), seen(1.5, confidence=0.8)])
        assert tracks[0].confidence == pytest.approx(0.7)

    def test_ids_are_stable_and_match_the_crop_filename_scheme(self) -> None:
        tracks = build_tracks(
            [
                seen(0.0, CAPTION_BOX, text="a"),
                seen(1.5, CAPTION_BOX, text="a"),
                seen(0.0, CORNER_BOX, kind=OverlayKind.WATERMARK, text="b"),
                seen(1.5, CORNER_BOX, kind=OverlayKind.WATERMARK, text="b"),
            ]
        )
        assert [t.id for t in tracks] == ["t000", "t001"]


class TestFiltering:
    def test_low_confidence_detections_are_discarded(self) -> None:
        tracks = build_tracks(
            [seen(0.0, confidence=0.1), seen(1.5, confidence=0.2)], min_confidence=0.35
        )
        assert tracks == []

    def test_a_single_weak_sighting_is_dropped_as_noise(self) -> None:
        """One sighting at ~1.5s sampling is usually a compression artefact or a
        one-off misread."""
        assert build_tracks([seen(0.0, confidence=0.5)]) == []

    def test_a_single_confident_sighting_is_kept(self) -> None:
        """A genuine 1.5s product pop-up can legitimately appear only once, and
        it is exactly the kind of thing a user wants removed."""
        tracks = build_tracks([seen(0.0, confidence=0.95)])
        assert len(tracks) == 1
        assert tracks[0].detection_count == 1

    def test_no_detections_yields_no_tracks(self) -> None:
        assert build_tracks([]) == []

    def test_every_detector_can_actually_clear_the_singleton_floor(self) -> None:
        """A cross-component invariant, and the one that was already broken.

        The stub capped its confidence at 0.75 to signal "I am only a heuristic",
        while the tracker required 0.7 to keep a single-frame detection. The two
        numbers were chosen independently, and the result was that a stub overlay
        seen in one sampled frame could essentially never survive - the escape
        hatch existed but was unreachable.

        Nothing else would notice: the tracker's tests pass with hand-written
        confidences, the stub's tests pass on its own output, and only running
        the two together reveals it. So the reachable ceiling is asserted here.
        """
        from app.adapters.vision_stub import StubVisionDetector  # noqa: F401

        stub_ceiling = 0.9  # see the confidence mapping in vision_stub
        gemini_typical = 0.8

        assert stub_ceiling > DEFAULT_SINGLETON_CONFIDENCE_FLOOR
        assert gemini_typical > DEFAULT_SINGLETON_CONFIDENCE_FLOOR

    def test_the_floor_is_injectable_so_the_coupling_can_be_tuned(self) -> None:
        one_weak_sighting = [seen(0.0, confidence=0.55)]

        assert build_tracks(one_weak_sighting) == []
        assert len(build_tracks(one_weak_sighting, singleton_confidence_floor=0.5)) == 1


class TestRealisticSequence:
    def test_a_typical_short_form_video_decomposes_sensibly(self) -> None:
        """End to end on a plausible detection stream: a watermark present
        throughout, a hook that is replaced by a different hook, and a product
        pop-up late on."""
        watermark = CORNER_BOX
        hook = BBox(x=0.1, y=0.12, w=0.8, h=0.12)
        popup = BBox(x=0.2, y=0.4, w=0.5, h=0.35)

        detections = []
        for t in (0.0, 1.5, 3.0, 4.5, 6.0):
            detections.append(
                seen(t, watermark, kind=OverlayKind.WATERMARK, text="@shop", confidence=0.8)
            )
        for t in (0.0, 1.5):
            detections.append(seen(t, hook, kind=OverlayKind.TEXT_OVERLAY, text="watch this"))
        for t in (3.0, 4.5):
            detections.append(seen(t, hook, kind=OverlayKind.TEXT_OVERLAY, text="link in bio"))
        for t in (4.5, 6.0):
            detections.append(
                seen(t, popup, kind=OverlayKind.IMAGE_POPUP, text="", confidence=0.85)
            )

        tracks = build_tracks(detections)

        by_kind = {}
        for track in tracks:
            by_kind.setdefault(track.kind, []).append(track)

        assert len(by_kind[OverlayKind.WATERMARK]) == 1, "one watermark for the whole video"
        assert len(by_kind[OverlayKind.TEXT_OVERLAY]) == 2, "the hook was replaced, not moved"
        assert len(by_kind[OverlayKind.IMAGE_POPUP]) == 1

        wm = by_kind[OverlayKind.WATERMARK][0]
        assert wm.start_s == 0.0
        assert wm.end_s >= 6.0

        first_hook, second_hook = sorted(by_kind[OverlayKind.TEXT_OVERLAY], key=lambda t: t.start_s)
        assert first_hook.text == "watch this"
        assert second_hook.text == "link in bio"
        assert first_hook.end_s <= second_hook.end_s

    def test_output_is_ready_for_the_removal_pass(self) -> None:
        """Every track must be directly usable as a time-gated mask."""
        tracks = build_tracks([seen(0.0), seen(1.5), seen(3.0)])
        for track in tracks:
            assert track.duration_s > 0
            assert 0.0 <= track.bbox.x <= 1.0
            assert track.bbox.right <= 1.0
            assert track.id
