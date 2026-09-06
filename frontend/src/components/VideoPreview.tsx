/**
 * Original ⇄ cleaned, with the detected overlays drawn on top.
 *
 * The comparison is a toggle on one element rather than two players side by
 * side, because the question a viewer is actually asking is "what changed
 * *here*" - and answering it by flicking one frame between two states is far
 * more legible than scanning between two small videos. Playback position is
 * carried across the switch for the same reason.
 *
 * The boxes are what make the result inspectable rather than merely plausible:
 * they are the model's claim, drawn over the pixels it was making the claim
 * about. Because `BBox` is normalised, they are positioned as percentages and
 * stay correct at any player size.
 */

import { useEffect, useRef, useState } from "react";

import { mediaUrl } from "../api/client";
import type { JobResult, OverlayTrack } from "../api/types";
import { KIND_COLOUR, KIND_LABEL } from "../lib/format";

interface Props {
  result: JobResult;
  /** Driven by the timeline and the overlay list, so one click can move the video. */
  seekTo: number | null;
  highlightId: string | null;
}

export function VideoPreview({ result, seekTo, highlightId }: Props) {
  const [showClean, setShowClean] = useState(true);
  const [showBoxes, setShowBoxes] = useState(true);
  const [now, setNow] = useState(0);
  const video = useRef<HTMLVideoElement>(null);
  // Survives the source swap so the toggle compares the same moment.
  const position = useRef(0);

  useEffect(() => {
    const element = video.current;
    if (!element) return;
    element.currentTime = position.current;
    if (!element.paused) void element.play();
  }, [showClean]);

  useEffect(() => {
    if (seekTo === null || !video.current) return;
    video.current.currentTime = seekTo;
    setNow(seekTo);
  }, [seekTo]);

  const visible = result.tracks.filter((t) => now >= t.start_s && now <= t.end_s);

  return (
    <section className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-neutral-700">
          {[
            { label: "Cleaned", value: true },
            { label: "Original", value: false },
          ].map((option) => (
            <button
              key={option.label}
              onClick={() => setShowClean(option.value)}
              className={`px-3 py-1.5 text-sm transition ${
                showClean === option.value
                  ? "bg-sky-600 text-white"
                  : "bg-neutral-900 text-neutral-400 hover:text-neutral-200"
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>

        <label className="ml-auto flex items-center gap-2 text-xs text-neutral-400">
          <input
            type="checkbox"
            checked={showBoxes}
            onChange={(e) => setShowBoxes(e.target.checked)}
            className="accent-sky-500"
          />
          Show detected regions
        </label>
      </div>

      <div className="relative mx-auto w-fit overflow-hidden rounded-lg bg-black">
        <video
          ref={video}
          key={showClean ? "clean" : "source"}
          src={mediaUrl(showClean ? result.clean_url : result.source_url)}
          controls
          playsInline
          onTimeUpdate={(e) => {
            const t = e.currentTarget.currentTime;
            position.current = t;
            setNow(t);
          }}
          className="max-h-[62vh] w-auto"
        />

        {showBoxes && (
          <div className="pointer-events-none absolute inset-0">
            {visible.map((track) => (
              <Box
                key={track.id}
                track={track}
                dimmed={highlightId !== null && highlightId !== track.id}
              />
            ))}
          </div>
        )}
      </div>

      <p className="text-center text-xs text-neutral-500">
        {visible.length > 0
          ? `${visible.length} overlay${visible.length === 1 ? "" : "s"} on screen at ${now.toFixed(1)}s`
          : "no overlays detected at this moment"}
      </p>
    </section>
  );
}

function Box({ track, dimmed }: { track: OverlayTrack; dimmed: boolean }) {
  const colour = KIND_COLOUR[track.kind] ?? "#38bdf8";
  return (
    <div
      className="absolute rounded-sm border-2 transition-opacity"
      style={{
        // Percentages, because the box is normalised and the player is fluid.
        left: `${track.bbox.x * 100}%`,
        top: `${track.bbox.y * 100}%`,
        width: `${track.bbox.w * 100}%`,
        height: `${track.bbox.h * 100}%`,
        borderColor: colour,
        backgroundColor: `${colour}22`,
        opacity: dimmed ? 0.25 : 1,
      }}
    >
      <span
        className="absolute -top-5 left-0 whitespace-nowrap rounded px-1 text-[10px] font-medium text-black"
        style={{ backgroundColor: colour }}
      >
        {KIND_LABEL[track.kind] ?? track.kind}
      </span>
    </div>
  );
}
