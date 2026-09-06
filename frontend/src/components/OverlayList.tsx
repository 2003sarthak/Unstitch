/**
 * The detected components, grouped by kind.
 *
 * Grouping by `OverlayKind` rather than listing chronologically is the whole
 * argument of the project made visible: the output is not "here are seventeen
 * boxes", it is "here are your captions, here is the watermark, here is the
 * product pop-up". A flat list would present the taxonomy as a label; grouping
 * presents it as structure.
 *
 * Each row carries its crop, which turns a claim into evidence - "caption,
 * 0:03–0:07" is an assertion, and the picture beside it is the proof.
 */

import { mediaUrl } from "../api/client";
import type { OverlayTrack } from "../api/types";
import { KIND_COLOUR, KIND_LABEL, timeRange } from "../lib/format";

interface Props {
  tracks: OverlayTrack[];
  highlightId: string | null;
  onHighlight: (id: string | null) => void;
  onSeek: (seconds: number) => void;
}

export function OverlayList({ tracks, highlightId, onHighlight, onSeek }: Props) {
  if (tracks.length === 0) {
    return (
      <section className="rounded-lg border border-neutral-800 p-4 text-sm text-neutral-500">
        No overlays were detected in this video.
      </section>
    );
  }

  const groups = new Map<string, OverlayTrack[]>();
  for (const track of tracks) {
    const bucket = groups.get(track.kind);
    if (bucket) bucket.push(track);
    else groups.set(track.kind, [track]);
  }

  return (
    <section className="space-y-4">
      <h2 className="text-sm font-medium text-neutral-300">
        Detected components <span className="text-neutral-600">({tracks.length})</span>
      </h2>

      {[...groups.entries()].map(([kind, group]) => (
        <div key={kind} className="space-y-1.5">
          <div className="flex items-center gap-2">
            <span
              className="h-2.5 w-2.5 rounded-sm"
              style={{ backgroundColor: KIND_COLOUR[kind] ?? "#38bdf8" }}
            />
            <h3 className="text-xs font-medium uppercase tracking-wide text-neutral-400">
              {KIND_LABEL[kind] ?? kind}
              <span className="ml-1 text-neutral-600">({group.length})</span>
            </h3>
          </div>

          <ul className="space-y-1.5">
            {group.map((track) => (
              <li key={track.id}>
                <button
                  onClick={() => onSeek(track.start_s)}
                  onMouseEnter={() => onHighlight(track.id)}
                  onMouseLeave={() => onHighlight(null)}
                  className={`flex w-full items-center gap-3 rounded-md border p-2 text-left transition ${
                    highlightId === track.id
                      ? "border-sky-500 bg-sky-500/10"
                      : "border-neutral-800 hover:border-neutral-600"
                  }`}
                >
                  {track.crop_url ? (
                    <img
                      src={mediaUrl(track.crop_url)}
                      alt=""
                      className="h-10 w-20 shrink-0 rounded border border-neutral-800 bg-black object-contain"
                    />
                  ) : (
                    <div className="h-10 w-20 shrink-0 rounded bg-neutral-800" />
                  )}

                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm text-neutral-200">
                      {track.text || <span className="text-neutral-500">(no text)</span>}
                    </p>
                    <p className="text-xs tabular-nums text-neutral-500">
                      {timeRange(track.start_s, track.end_s)}
                      <span className="text-neutral-700"> · </span>
                      seen in {track.detection_count} frame
                      {track.detection_count === 1 ? "" : "s"}
                    </p>
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}
