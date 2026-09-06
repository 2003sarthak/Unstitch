/**
 * Mirrors `backend/app/domain/models.py` one-to-one.
 *
 * Hand-written rather than generated from the OpenAPI schema. For a surface this
 * small, a generator would add a build step and a lockstep dependency for types
 * that fit on one screen - and the hand-written version can carry the *reasons*,
 * which a generated file cannot. The tradeoff is that these must be kept in sync
 * by hand; the field comments below flag the ones where a mistake would be
 * silent rather than obvious.
 */

/** Server-side: `JobStatus`. Ordered as the pipeline runs. */
export type JobStatus =
  | "queued"
  | "downloading"
  | "analyzing_scenes"
  | "detecting_overlays"
  | "rendering"
  | "done"
  | "failed";

/** Server-side: `OverlayKind`. The taxonomy that makes this understanding rather than OCR. */
export type OverlayKind =
  | "caption"
  | "text_overlay"
  | "image_popup"
  | "watermark"
  | "ui_chrome";

/** Server-side: `RemovalMode`. */
export type RemovalMode = "delogo" | "boxblur";

/**
 * Normalised to 0..1 of frame width and height, never pixels.
 *
 * This is what lets a box be drawn over a `<video>` element at whatever size the
 * layout gives it: multiply by the rendered element's size, not the source
 * video's. Treating these as pixels would put every box in the wrong place at
 * every viewport width.
 */
export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface VideoMeta {
  duration_s: number;
  width: number;
  height: number;
  fps: number;
  has_audio: boolean;
  video_codec: string;
  rotation: number;
}

/** One overlay followed across time - the re-editable component. */
export interface OverlayTrack {
  id: string;
  kind: OverlayKind;
  text: string;
  bbox: BBox;
  start_s: number;
  end_s: number;
  confidence: number;
  detection_count: number;
  crop_url: string | null;
}

export interface Scene {
  index: number;
  start_s: number;
  end_s: number;
  thumb_url: string | null;
  clip_url: string | null;
  clean_clip_url: string | null;
}

export interface JobResult {
  job_id: string;
  source_url: string;
  clean_url: string;
  meta: VideoMeta;
  scenes: Scene[];
  tracks: OverlayTrack[];
  removal_mode: RemovalMode;
  /**
   * Which detector actually ran. Surfaced in the UI because a result produced by
   * the offline stub must not be mistaken for a real analysis.
   */
  vision_provider: "gemini" | "stub";
}

export interface JobAccepted {
  job_id: string;
  status: JobStatus;
}

export interface JobProgress {
  job_id: string;
  status: JobStatus;
  stage: string;
  progress: number;
  error: string | null;
}

export interface Health {
  status: string;
  version: string;
  vision_provider: string;
  vision_provider_requested: string;
  removal_mode: string;
  queued_jobs: number;
}

/** Server-side: the `{error, type}` shape every failure uses. */
export interface ApiErrorBody {
  error: string;
  type: string;
}
