"""Tests for the domain models.

The geometry is the part that earns tests. `BBox.iou` is the similarity measure
the whole overlay-tracking algorithm rests on, `padded` decides whether text
actually disappears or leaves a fringe, and `clamped_to_frame` is what stops
ffmpeg rejecting a full-width caption. All three are pure, so all three can be
pinned exactly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.domain.models import (
    BBox,
    Detection,
    Job,
    JobStatus,
    OverlayKind,
    OverlayTrack,
    PixelBox,
    Scene,
)


class TestBBoxValidation:
    def test_a_box_outside_the_frame_is_clamped_not_rejected(self) -> None:
        """Detected boxes are accurate to a few percent, so a caption running to
        the edge routinely comes back slightly out of frame. Rejecting it would
        discard a true detection over a rounding error."""
        box = BBox(x=-0.02, y=-0.01, w=1.05, h=0.2)
        assert box.x == 0.0
        assert box.y == 0.0
        assert box.right <= 1.0
        assert box.bottom <= 1.0

    def test_a_box_with_no_area_is_rejected(self) -> None:
        """Zero area is a real failure, not imprecision - there is nothing to
        erase and nothing to show."""
        with pytest.raises(ValidationError, match="no area"):
            BBox(x=0.1, y=0.1, w=0.0, h=0.5)

    def test_a_box_entirely_off_frame_collapses_and_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="no area"):
            BBox(x=1.0, y=0.5, w=0.3, h=0.3)

    def test_a_normal_box_is_untouched(self) -> None:
        box = BBox(x=0.1, y=0.2, w=0.3, h=0.4)
        assert (box.x, box.y, box.w, box.h) == (0.1, 0.2, 0.3, 0.4)

    def test_boxes_are_immutable(self) -> None:
        with pytest.raises(ValidationError):
            BBox(x=0.1, y=0.1, w=0.1, h=0.1).x = 0.5  # type: ignore[misc]


class TestIoU:
    """The measure that decides whether two sightings are the same overlay."""

    def test_identical_boxes_score_one(self) -> None:
        box = BBox(x=0.1, y=0.1, w=0.4, h=0.2)
        assert box.iou(box) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self) -> None:
        left = BBox(x=0.0, y=0.0, w=0.2, h=0.2)
        right = BBox(x=0.5, y=0.5, w=0.2, h=0.2)
        assert left.iou(right) == 0.0

    def test_touching_boxes_score_zero(self) -> None:
        """Edge contact is not overlap."""
        a = BBox(x=0.0, y=0.0, w=0.5, h=1.0)
        b = BBox(x=0.5, y=0.0, w=0.5, h=1.0)
        assert a.iou(b) == 0.0

    def test_half_overlap_scores_one_third(self) -> None:
        a = BBox(x=0.0, y=0.0, w=0.5, h=1.0)
        b = BBox(x=0.25, y=0.0, w=0.5, h=1.0)
        assert a.iou(b) == pytest.approx(1 / 3)

    def test_a_small_box_inside_a_large_one_scores_low(self) -> None:
        """Why IoU rather than "does it overlap": a caption sitting inside a
        full-frame popup must not be merged into it. Containment alone would
        merge them; IoU keeps them apart."""
        caption = BBox(x=0.1, y=0.8, w=0.8, h=0.1)
        full_frame = BBox(x=0.0, y=0.0, w=1.0, h=1.0)
        assert caption.iou(full_frame) < 0.1

    def test_iou_is_symmetric(self) -> None:
        a = BBox(x=0.1, y=0.1, w=0.4, h=0.4)
        b = BBox(x=0.2, y=0.2, w=0.4, h=0.4)
        assert a.iou(b) == pytest.approx(b.iou(a))

    def test_slight_jitter_still_scores_above_the_default_threshold(self) -> None:
        """A caption that shifts by 1% of the frame between sampled frames is the
        same caption, and must clear the 0.5 default in `.env.example`."""
        frame_one = BBox(x=0.10, y=0.80, w=0.80, h=0.10)
        frame_two = BBox(x=0.11, y=0.81, w=0.80, h=0.10)
        assert frame_one.iou(frame_two) > 0.5


class TestUnionAndPadding:
    def test_union_covers_both_boxes(self) -> None:
        """A track's box is the union of its detections', so a caption that gains
        a word mid-sentence is still fully masked."""
        short = BBox(x=0.1, y=0.8, w=0.4, h=0.08)
        grown = BBox(x=0.1, y=0.8, w=0.7, h=0.09)
        merged = short.union(grown)
        assert merged.x == 0.1
        assert merged.right == pytest.approx(0.8)
        assert merged.bottom == pytest.approx(0.89)

    def test_union_is_symmetric(self) -> None:
        a = BBox(x=0.1, y=0.1, w=0.2, h=0.2)
        b = BBox(x=0.5, y=0.5, w=0.2, h=0.2)
        assert a.union(b) == b.union(a)

    def test_padding_grows_every_side(self) -> None:
        padded = BBox(x=0.4, y=0.4, w=0.2, h=0.2).padded(2.0)
        assert padded.x == pytest.approx(0.38)
        assert padded.y == pytest.approx(0.38)
        assert padded.w == pytest.approx(0.24)
        assert padded.h == pytest.approx(0.24)

    def test_padding_is_clipped_at_the_frame_edge(self) -> None:
        """A caption already touching the bottom cannot be padded past it."""
        padded = BBox(x=0.0, y=0.9, w=1.0, h=0.1).padded(2.0)
        assert padded.x == 0.0
        assert padded.bottom == pytest.approx(1.0)
        assert padded.right == pytest.approx(1.0)

    def test_padding_by_zero_changes_nothing(self) -> None:
        box = BBox(x=0.2, y=0.2, w=0.3, h=0.3)
        assert box.padded(0.0) == box


class TestPixelConversion:
    def test_normalised_box_maps_onto_frame_pixels(self) -> None:
        px = BBox(x=0.1, y=0.2, w=0.3, h=0.4).to_pixels(1000, 1000)
        assert px == PixelBox(x=100, y=200, w=300, h=400)

    def test_conversion_uses_edges_so_rounding_cannot_shrink_the_box(self) -> None:
        """Rounding x and w separately can lose a pixel; rounding both edges and
        subtracting cannot."""
        px = BBox(x=0.333, y=0.0, w=0.334, h=1.0).to_pixels(1080, 1920)
        assert px.x + px.w == round(0.667 * 1080)

    def test_a_tiny_box_never_collapses_to_zero_width(self) -> None:
        """ffmpeg rejects a zero-width filter region."""
        px = BBox(x=0.5, y=0.5, w=0.0001, h=0.0001).to_pixels(720, 1280)
        assert px.w >= 1
        assert px.h >= 1


class TestDelogoFraming:
    """`delogo` interpolates from pixels just outside the mask, so a box flush
    against the frame edge has nothing to sample and ffmpeg refuses it."""

    def test_a_full_width_caption_is_inset_off_the_edges(self) -> None:
        px = PixelBox(x=0, y=1180, w=720, h=100).clamped_to_frame(720, 1280, inset=1)
        assert px.x >= 1
        assert px.y >= 1
        assert px.x + px.w <= 719
        assert px.y + px.h <= 1279

    def test_a_box_overflowing_the_frame_is_pulled_back(self) -> None:
        px = PixelBox(x=700, y=1200, w=400, h=400).clamped_to_frame(720, 1280, inset=1)
        assert px.x + px.w <= 719
        assert px.y + px.h <= 1279
        assert px.w >= 1 and px.h >= 1

    def test_a_box_already_inside_is_left_alone(self) -> None:
        px = PixelBox(x=10, y=20, w=100, h=50)
        assert px.clamped_to_frame(720, 1280, inset=1) == px


class TestJobStatus:
    def test_progress_never_goes_backwards_through_the_pipeline(self) -> None:
        order = [
            JobStatus.QUEUED,
            JobStatus.DOWNLOADING,
            JobStatus.ANALYZING_SCENES,
            JobStatus.DETECTING_OVERLAYS,
            JobStatus.RENDERING,
            JobStatus.DONE,
        ]
        values = [s.progress for s in order]
        assert values == sorted(values)
        assert values[0] == 0.0 and values[-1] == 1.0

    def test_only_done_and_failed_are_terminal(self) -> None:
        terminal = {s for s in JobStatus if s.is_terminal}
        assert terminal == {JobStatus.DONE, JobStatus.FAILED}

    def test_a_failed_job_does_not_report_partial_progress(self) -> None:
        """A stalled bar at 45% would read as "still working"."""
        assert JobStatus.FAILED.progress == 1.0

    def test_status_serialises_as_its_bare_value(self) -> None:
        job = Job(id="abc")
        assert job.model_dump(mode="json")["status"] == "queued"


class TestDurations:
    def test_track_duration_is_derived_not_stored(self) -> None:
        track = OverlayTrack(
            id="t000",
            kind=OverlayKind.CAPTION,
            text="buy now",
            bbox=BBox(x=0.1, y=0.8, w=0.8, h=0.1),
            start_s=1.5,
            end_s=4.0,
            confidence=0.9,
            detection_count=3,
        )
        assert track.duration_s == pytest.approx(2.5)

    def test_scene_duration_is_derived(self) -> None:
        assert Scene(index=0, start_s=0.0, end_s=3.25).duration_s == pytest.approx(3.25)

    def test_a_reversed_range_reports_zero_rather_than_a_negative(self) -> None:
        assert Scene(index=0, start_s=5.0, end_s=2.0).duration_s == 0.0


class TestDetection:
    def test_confidence_is_bounded(self) -> None:
        with pytest.raises(ValidationError):
            Detection(
                t_s=1.0, bbox=BBox(x=0, y=0, w=0.1, h=0.1), kind=OverlayKind.CAPTION, confidence=1.5
            )

    def test_image_elements_may_carry_no_text(self) -> None:
        detection = Detection(
            t_s=1.0, bbox=BBox(x=0, y=0, w=0.1, h=0.1), kind=OverlayKind.IMAGE_POPUP
        )
        assert detection.text == ""

    def test_kind_round_trips_through_json(self) -> None:
        detection = Detection(
            t_s=1.0, bbox=BBox(x=0, y=0, w=0.1, h=0.1), kind=OverlayKind.WATERMARK
        )
        assert detection.model_dump(mode="json")["kind"] == "watermark"
