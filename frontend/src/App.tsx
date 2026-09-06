/**
 * One page, three sections: input, status, result.
 *
 * All cross-component state lives here - the seek target and the highlighted
 * track - because both are shared by three children and hoisting them is what
 * lets hovering a row in the overlay list dim the other boxes on the video.
 * `useJob` owns everything about the job itself.
 */

import { useEffect, useState } from "react";

import { api } from "./api/client";
import type { Health } from "./api/types";
import { OverlayList } from "./components/OverlayList";
import { SceneTimeline } from "./components/SceneTimeline";
import { StatusBar } from "./components/StatusBar";
import { SubmitForm } from "./components/SubmitForm";
import { VideoPreview } from "./components/VideoPreview";
import { useJob } from "./hooks/useJob";

export default function App() {
  const job = useJob();
  const [health, setHealth] = useState<Health | null>(null);
  const [seekTo, setSeekTo] = useState<number | null>(null);
  const [highlightId, setHighlightId] = useState<string | null>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  /** Bumped by a hair so repeated clicks on the same scene still re-seek. */
  const seek = (seconds: number) => setSeekTo(seconds + Math.random() * 1e-6);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <div className="mx-auto max-w-4xl space-y-8 px-4 py-10">
        <header className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Unstitch</h1>
          <p className="text-sm text-neutral-400">
            Take a short-form video apart: find its scenes, detect the captions,
            overlays and pop-ups burned into it, and render it clean.
          </p>
          {health && (
            <p className="pt-1 text-xs text-neutral-600">
              vision: <span className="text-neutral-400">{health.vision_provider}</span>
              {health.vision_provider !== health.vision_provider_requested && (
                <span className="text-amber-500">
                  {" "}
                  (requested {health.vision_provider_requested} — no API key, so
                  detection is heuristic)
                </span>
              )}
            </p>
          )}
        </header>

        <SubmitForm busy={job.busy} onSubmit={job.start} />

        {job.error && (
          <div className="space-y-2 rounded-lg border border-red-900 bg-red-950/40 p-4">
            <p className="text-sm text-red-200">{job.error}</p>
            <button
              onClick={job.reset}
              className="text-xs text-red-300 underline hover:text-red-100"
            >
              Try another video
            </button>
          </div>
        )}

        {job.busy && job.progress && <StatusBar progress={job.progress} />}

        {job.result && (
          <div className="space-y-8 border-t border-neutral-800 pt-8">
            <VideoPreview
              result={job.result}
              seekTo={seekTo}
              highlightId={highlightId}
            />
            <SceneTimeline
              scenes={job.result.scenes}
              totalDuration={job.result.meta.duration_s}
              onSeek={seek}
            />
            <OverlayList
              tracks={job.result.tracks}
              highlightId={highlightId}
              onHighlight={setHighlightId}
              onSeek={seek}
            />

            <footer className="border-t border-neutral-800 pt-4 text-xs text-neutral-600">
              {job.result.meta.width}×{job.result.meta.height} ·{" "}
              {job.result.meta.duration_s.toFixed(1)}s · removed with{" "}
              {job.result.removal_mode} · detected by {job.result.vision_provider}
              {job.result.vision_provider === "stub" && (
                <span className="text-amber-600">
                  {" "}
                  — heuristic detection, not a semantic analysis
                </span>
              )}
            </footer>
          </div>
        )}
      </div>
    </div>
  );
}
