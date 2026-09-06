/**
 * The input: a URL, or a dropped file, plus the removal mode.
 *
 * Both entry paths share one submit button rather than sitting behind tabs,
 * because they are the same request with different payloads and the user should
 * not have to decide which one they are "in" before starting.
 */

import { useRef, useState } from "react";

import type { RemovalMode } from "../api/types";

interface Props {
  busy: boolean;
  onSubmit: (input: { url?: string; file?: File; removalMode: RemovalMode }) => void;
}

export function SubmitForm({ busy, onSubmit }: Props) {
  const [url, setUrl] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [removalMode, setRemovalMode] = useState<RemovalMode>("delogo");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (file) onSubmit({ file, removalMode });
    else if (url.trim()) onSubmit({ url: url.trim(), removalMode });
  };

  const take = (dropped: FileList | null) => {
    const chosen = dropped?.[0];
    if (chosen) {
      setFile(chosen);
      setUrl(""); // one source at a time, so the button is never ambiguous
    }
  };

  const canSubmit = !busy && (file !== null || url.trim().length > 0);

  return (
    <form onSubmit={submit} className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          take(e.dataTransfer.files);
        }}
        onClick={() => fileInput.current?.click()}
        className={`cursor-pointer rounded-lg border-2 border-dashed p-6 text-center transition ${
          dragging
            ? "border-sky-400 bg-sky-400/10"
            : "border-neutral-700 hover:border-neutral-500"
        }`}
      >
        <input
          ref={fileInput}
          type="file"
          accept="video/*"
          className="hidden"
          onChange={(e) => take(e.target.files)}
        />
        {file ? (
          <p className="text-sm">
            <span className="font-medium text-sky-300">{file.name}</span>
            <span className="text-neutral-500">
              {" "}
              · {(file.size / (1024 * 1024)).toFixed(1)} MB
            </span>
          </p>
        ) : (
          <p className="text-sm text-neutral-400">
            Drop a video here, or click to choose one
          </p>
        )}
      </div>

      <div className="flex items-center gap-3 text-xs text-neutral-500">
        <span className="h-px flex-1 bg-neutral-800" />
        or paste a link
        <span className="h-px flex-1 bg-neutral-800" />
      </div>

      <input
        type="url"
        value={url}
        onChange={(e) => {
          setUrl(e.target.value);
          if (e.target.value) setFile(null);
        }}
        placeholder="https://www.youtube.com/shorts/..."
        className="w-full rounded-md border border-neutral-700 bg-neutral-900 px-3 py-2 text-sm outline-none focus:border-sky-500"
      />

      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-neutral-400">
          Removal
          <select
            value={removalMode}
            onChange={(e) => setRemovalMode(e.target.value as RemovalMode)}
            className="mt-1 block rounded-md border border-neutral-700 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 outline-none focus:border-sky-500"
          >
            <option value="delogo">delogo — interpolates, best over busy video</option>
            <option value="boxblur">boxblur — never invents detail</option>
          </select>
        </label>

        <button
          type="submit"
          disabled={!canSubmit}
          className="ml-auto rounded-md bg-sky-600 px-5 py-2 text-sm font-medium text-white transition hover:bg-sky-500 disabled:cursor-not-allowed disabled:bg-neutral-800 disabled:text-neutral-500"
        >
          {busy ? "Working…" : "De-edit"}
        </button>
      </div>

      <p className="text-xs text-neutral-600">
        Uploading is the reliable path. TikTok and Instagram links often sit behind
        a login wall that yt-dlp cannot pass.
      </p>
    </form>
  );
}
