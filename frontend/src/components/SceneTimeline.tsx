/**
 * The scene strip: one thumbnail per shot, width proportional to duration.
 *
 * Proportional width is the point rather than decoration - it turns the strip
 * into a picture of the edit's rhythm, where a row of equal tiles would hide the
 * difference between a four-second hold and eight rapid cuts.
 *
 * Clicking seeks the main player instead of opening a separate clip. The backend
 * deliberately does not render per-scene videos: it returns exact boundaries, and
 * seeking to one costs nothing where cutting every scene would cost N encodes.
 */

import { mediaUrl } from "../api/client";
import type { Scene } from "../api/types";
import { timecode } from "../lib/format";

interface Props {
  scenes: Scene[];
  totalDuration: number;
  onSeek: (seconds: number) => void;
}

export function SceneTimeline({ scenes, totalDuration, onSeek }: Props) {
  if (scenes.length === 0) return null;

  return (
    <section className="space-y-2">
      <h2 className="text-sm font-medium text-neutral-300">
        Scenes <span className="text-neutral-600">({scenes.length})</span>
      </h2>
      <div className="flex gap-1 overflow-x-auto pb-1">
        {scenes.map((scene) => {
          const share = totalDuration > 0 ? (scene.end_s - scene.start_s) / totalDuration : 0;
          return (
            <button
              key={scene.index}
              onClick={() => onSeek(scene.start_s)}
              title={`Scene ${scene.index + 1}: ${timecode(scene.start_s)}`}
              style={{ flexGrow: Math.max(share, 0.05), flexBasis: 0 }}
              className="group relative min-w-[56px] overflow-hidden rounded border border-neutral-800 transition hover:border-sky-500"
            >
              {scene.thumb_url ? (
                <img
                  src={mediaUrl(scene.thumb_url)}
                  alt={`Scene ${scene.index + 1}`}
                  className="h-20 w-full object-cover"
                />
              ) : (
                <div className="h-20 w-full bg-neutral-800" />
              )}
              <span className="absolute bottom-0 left-0 right-0 bg-black/70 px-1 py-0.5 text-[10px] tabular-nums text-neutral-300">
                {timecode(scene.start_s)}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}
