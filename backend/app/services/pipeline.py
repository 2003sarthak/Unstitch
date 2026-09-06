"""The de-editing pipeline: source video in, editable components out.

Every collaborator arrives through a Protocol from `domain.ports`. There is no
`import yt_dlp`, no `google.genai`, no `cv2` and no `scenedetect` anywhere in
this file - a rule `tests/test_architecture.py` enforces mechanically. That is
what makes the whole pipeline runnable against the vision stub with no API key,
and testable against fakes in milliseconds with no video at all.

The stages, and why they are in this order:

    ingest -> normalise -> scenes -> sample -> detect -> track -> render

Normalisation comes second because everything after it is allowed to assume
H.264 at a known height, which removes a pile of format handling from four later
stages. Scene detection comes before sampling because scene boundaries are the
best prior for *where to look*. Tracking comes before rendering because the
render needs time ranges, not sightings - that is the whole reason one pass can
erase N overlays.

Progress is pushed through a callback rather than written to a store here: the
runner owns persistence, the pipeline owns work.
"""

from __future__ import annotations

import asyncio
import logging

from app.domain.errors import UnsupportedMediaError
from app.domain.models import (
    JobRequest,
    JobResult,
    JobStatus,
    OverlayTrack,
    Scene,
    VideoMeta,
    VisionProvider,
)
from app.domain.ports import (
    FrameSampler,
    MediaIngestor,
    ProgressCallback,
    SceneDetector,
    VideoEditor,
    VisionDetector,
)
from app.infra.workspace import Workspace
from app.services.track_builder import build_tracks

log = logging.getLogger(__name__)


class Pipeline:
    """Orchestrates one de-edit.

    Holds no state between runs, so a single instance serves every job and
    concurrency is the runner's concern rather than something to reason about
    here.
    """

    def __init__(
        self,
        *,
        ingestor: MediaIngestor,
        scene_detector: SceneDetector,
        sampler: FrameSampler,
        vision: VisionDetector,
        editor: VideoEditor,
        vision_provider: VisionProvider,
        iou_threshold: float = 0.5,
        max_gap_s: float = 3.5,
        min_detection_confidence: float = 0.35,
        frame_interval_s: float = 1.5,
        mask_padding_pct: float = 2.0,
    ) -> None:
        self._ingestor = ingestor
        self._scenes = scene_detector
        self._sampler = sampler
        self._vision = vision
        self._editor = editor
        self._vision_provider = vision_provider
        self._iou_threshold = iou_threshold
        self._max_gap_s = max_gap_s
        self._min_detection_confidence = min_detection_confidence
        self._frame_interval_s = frame_interval_s
        self._mask_padding_pct = mask_padding_pct

    async def run(
        self, workspace: Workspace, request: JobRequest, on_progress: ProgressCallback
    ) -> JobResult:
        workspace.ensure()

        await on_progress(JobStatus.DOWNLOADING)
        await self._acquire_source(workspace, request)
        meta = await self._editor.normalise(workspace.original, workspace.source)

        await on_progress(JobStatus.ANALYZING_SCENES)
        scenes = await self._scenes.detect(workspace.source)
        scenes = await self._render_thumbnails(workspace, scenes)

        await on_progress(JobStatus.DETECTING_OVERLAYS)
        tracks = await self._find_overlays(workspace, meta, scenes)

        await on_progress(JobStatus.RENDERING)
        await self._editor.remove_overlays(
            workspace.source,
            workspace.clean,
            tracks,
            meta,
            request.removal_mode,
            self._mask_padding_pct,
        )
        tracks = await self._render_crops(workspace, meta, tracks)

        log.info(
            "job %s done: %d scene(s), %d overlay track(s)",
            workspace.job_id,
            len(scenes),
            len(tracks),
        )
        return JobResult(
            job_id=workspace.job_id,
            source_url=workspace.url_for(workspace.source),
            clean_url=workspace.url_for(workspace.clean),
            meta=meta,
            scenes=scenes,
            tracks=tracks,
            removal_mode=request.removal_mode,
            vision_provider=self._vision_provider,
        )

    # --- stages --------------------------------------------------------------

    async def _acquire_source(self, workspace: Workspace, request: JobRequest) -> None:
        """Download, or accept what the upload route already wrote.

        Uploads bypass the ingest port entirely - there is nothing to fetch - so
        this is the one place the two entry paths converge, and after it nothing
        downstream knows or cares which one was used.
        """
        if request.url:
            await self._ingestor.fetch(request.url, workspace.original)
        elif not workspace.original.exists():
            raise UnsupportedMediaError("no video was provided")

    async def _render_thumbnails(self, workspace: Workspace, scenes: list[Scene]) -> list[Scene]:
        """One still per scene, taken from the middle of the shot.

        Concurrent because each is an independent seek-and-decode, so a
        twelve-scene video costs about one thumbnail's wall-clock rather than
        twelve.

        Note what is *not* produced: per-scene video clips. Cutting every scene
        would mean N extra encodes for something the player already does by
        seeking, and `JobResult` carries exact scene boundaries for it to seek
        to. On a free tier that is the difference between a demo that responds
        and one that times out.
        """

        async def thumb(scene: Scene) -> Scene:
            dest = workspace.scene_thumb(scene.index)
            await self._editor.thumbnail(workspace.source, dest, (scene.start_s + scene.end_s) / 2)
            return scene.model_copy(update={"thumb_url": workspace.url_for(dest)})

        return list(await asyncio.gather(*(thumb(scene) for scene in scenes)))

    async def _find_overlays(
        self, workspace: Workspace, meta: VideoMeta, scenes: list[Scene]
    ) -> list[OverlayTrack]:
        frames = await self._sampler.sample(workspace.source, meta, scenes, workspace.frames_dir)
        detections = await self._vision.detect(frames)
        return build_tracks(
            detections,
            iou_threshold=self._iou_threshold,
            max_gap_s=self._max_gap_s,
            min_confidence=self._min_detection_confidence,
            frame_interval_s=self._frame_interval_s,
        )

    async def _render_crops(
        self, workspace: Workspace, meta: VideoMeta, tracks: list[OverlayTrack]
    ) -> list[OverlayTrack]:
        """A still of each overlay, cut from the *original* rather than the clean
        render - the clean one no longer contains the thing being illustrated.

        This is what makes the result legible instead of merely correct: a row
        reading "caption, 0:03-0:07" is a claim, and the crop beside it is the
        evidence that the detector saw what it says it saw.
        """

        async def crop(position: int, track: OverlayTrack) -> OverlayTrack:
            dest = workspace.track_crop(position)
            await self._editor.crop(
                workspace.source, dest, track.bbox, meta, (track.start_s + track.end_s) / 2
            )
            return track.model_copy(update={"crop_url": workspace.url_for(dest)})

        return list(
            await asyncio.gather(*(crop(position, track) for position, track in enumerate(tracks)))
        )
