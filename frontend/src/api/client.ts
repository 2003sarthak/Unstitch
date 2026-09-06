/**
 * The only module that talks to the backend.
 *
 * Every request goes through `request()`, so the base URL, JSON parsing and -
 * most importantly - error translation are written once. The backend answers
 * every failure with the same `{error, type}` body, which is what lets a single
 * function turn any non-2xx response into a thrown `ApiError` carrying a message
 * that is already fit to show a user.
 */

import type {
  ApiErrorBody,
  Health,
  JobAccepted,
  JobProgress,
  JobResult,
  RemovalMode,
} from "./types";

/**
 * Falls back to same-origin when unset, so a production build served behind the
 * same host needs no configuration at all.
 */
const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly type: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, init);
  } catch {
    // fetch only rejects for network-level failures, and "the backend is not
    // running" is by far the most likely one in development.
    throw new ApiError(
      `Cannot reach the backend at ${BASE || "this origin"}. Is it running?`,
      0,
      "NetworkError",
    );
  }

  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as ApiErrorBody | null;
    throw new ApiError(
      body?.error ?? `Request failed (${response.status})`,
      response.status,
      body?.type ?? "HttpError",
    );
  }
  return (await response.json()) as T;
}

/** Absolute URL for a `/media/...` path returned in a `JobResult`. */
export function mediaUrl(path: string): string {
  return `${BASE}${path}`;
}

export const api = {
  health: () => request<Health>("/api/health"),

  createFromUrl: (url: string, removalMode: RemovalMode) =>
    request<JobAccepted>("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, removal_mode: removalMode }),
    }),

  createFromFile: (file: File, removalMode: RemovalMode) => {
    const form = new FormData();
    form.append("file", file);
    form.append("removal_mode", removalMode);
    // No Content-Type header: the browser must set it itself so the multipart
    // boundary matches the body it generates.
    return request<JobAccepted>("/api/jobs/upload", { method: "POST", body: form });
  },

  progress: (jobId: string) => request<JobProgress>(`/api/jobs/${jobId}`),

  result: (jobId: string) => request<JobResult>(`/api/jobs/${jobId}/result`),
};
