/** Small shared formatters. Duplicating these across components is how two parts
 *  of one timeline end up disagreeing about what "0:07" means. */

/** `0:07`, `1:23` - the form a viewer reads off a video player. */
export function timecode(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
}

export function timeRange(start: number, end: number): string {
  return `${timecode(start)} – ${timecode(end)}`;
}

/** Human label for an `OverlayKind`. The wire values are snake_case. */
export const KIND_LABEL: Record<string, string> = {
  caption: "Caption",
  text_overlay: "Text overlay",
  image_popup: "Image pop-up",
  watermark: "Watermark",
  ui_chrome: "UI chrome",
};

/** One colour per kind, shared by the overlay list and the boxes drawn on the
 *  video - so a row and its rectangle are recognisably the same thing. */
export const KIND_COLOUR: Record<string, string> = {
  caption: "#38bdf8",
  text_overlay: "#a78bfa",
  image_popup: "#fb923c",
  watermark: "#f472b6",
  ui_chrome: "#4ade80",
};
