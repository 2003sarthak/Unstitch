/**
 * The single source of truth for job state, and the only place that polls.
 *
 * The backend answers `202` and keeps working, so the client's job is to ask
 * again until the answer changes. Three things make that tolerable rather than
 * wasteful:
 *
 * - **Polling stops at a terminal status.** `done` and `failed` are final, so a
 *   finished job must not leave a timer running for the rest of the session.
 * - **The interval backs off.** A de-edit takes tens of seconds and detection
 *   alone can take twenty, so a fixed 1.5s tick spends most of a job asking a
 *   question whose answer has not changed. Starting fast keeps the first stage
 *   transitions snappy, and widening after that cuts the request count roughly
 *   in half.
 * - **Every effect cleans up.** Submitting a second video while the first is
 *   still running must not leave two pollers writing to the same state.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError, api } from "../api/client";
import type { JobProgress, JobResult, RemovalMode } from "../api/types";

const FIRST_INTERVAL_MS = 800;
const MAX_INTERVAL_MS = 3000;
const BACKOFF = 1.35;

export interface JobState {
  jobId: string | null;
  progress: JobProgress | null;
  result: JobResult | null;
  error: string | null;
  /** True from submission until a terminal status, so the UI can disable input. */
  busy: boolean;
}

const IDLE: JobState = {
  jobId: null,
  progress: null,
  result: null,
  error: null,
  busy: false,
};

export function useJob() {
  const [state, setState] = useState<JobState>(IDLE);
  // Identifies the current attempt. A poll loop belonging to a superseded job
  // checks this before writing, which is what stops a stale response from
  // overwriting a newer one.
  const attempt = useRef(0);

  const reset = useCallback(() => {
    attempt.current += 1;
    setState(IDLE);
  }, []);

  const start = useCallback(
    async (input: { url?: string; file?: File; removalMode: RemovalMode }) => {
      const mine = ++attempt.current;
      setState({ ...IDLE, busy: true });
      try {
        const accepted = input.file
          ? await api.createFromFile(input.file, input.removalMode)
          : await api.createFromUrl(input.url ?? "", input.removalMode);
        if (mine !== attempt.current) return;
        setState((s) => ({ ...s, jobId: accepted.job_id }));
      } catch (err) {
        if (mine !== attempt.current) return;
        setState({
          ...IDLE,
          error: err instanceof ApiError ? err.message : "Something went wrong",
        });
      }
    },
    [],
  );

  useEffect(() => {
    const jobId = state.jobId;
    if (!jobId || state.result || state.error) return;

    const mine = attempt.current;
    let timer: number | undefined;
    let interval = FIRST_INTERVAL_MS;
    let cancelled = false;

    const tick = async () => {
      try {
        const progress = await api.progress(jobId);
        if (cancelled || mine !== attempt.current) return;
        setState((s) => ({ ...s, progress }));

        if (progress.status === "failed") {
          setState((s) => ({
            ...s,
            busy: false,
            error: progress.error ?? "The job failed",
          }));
          return;
        }
        if (progress.status === "done") {
          const result = await api.result(jobId);
          if (cancelled || mine !== attempt.current) return;
          setState((s) => ({ ...s, result, busy: false }));
          return;
        }

        interval = Math.min(interval * BACKOFF, MAX_INTERVAL_MS);
        timer = window.setTimeout(tick, interval);
      } catch (err) {
        if (cancelled || mine !== attempt.current) return;
        setState((s) => ({
          ...s,
          busy: false,
          error: err instanceof ApiError ? err.message : "Lost contact with the backend",
        }));
      }
    };

    tick();
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
    };
  }, [state.jobId, state.result, state.error]);

  return { ...state, start, reset };
}
