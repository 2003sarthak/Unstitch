/**
 * Stage name plus a bar.
 *
 * Both, because they answer different questions: the label says what is
 * happening ("detecting overlays"), the bar says how far along. A spinner alone
 * cannot tell a waiting user that the slowest stage is the one they are in, and
 * that is exactly when people assume something has hung.
 */

import type { JobProgress } from "../api/types";

export function StatusBar({ progress }: { progress: JobProgress }) {
  const pct = Math.round(progress.progress * 100);
  return (
    <div className="space-y-2 rounded-lg border border-neutral-800 bg-neutral-900/50 p-4">
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium capitalize text-neutral-200">{progress.stage}</span>
        <span className="tabular-nums text-neutral-500">{pct}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-neutral-800">
        <div
          className="h-full rounded-full bg-sky-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
      <p className="text-xs text-neutral-500">
        Detection is the slow stage - it waits on the vision model.
      </p>
    </div>
  );
}
