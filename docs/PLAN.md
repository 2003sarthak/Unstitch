# LightNoteAI — Video De-Editing & Cloning · Plan

> Assignment: take a short-form video (URL or upload), automatically break it into
> editable components (scenes, captions/text overlays, image/product pop-ups),
> remove/isolate those overlays, and expose everything in an editable form.
> Deadline: **6 Sep 2026, 18:00**.

---

## 1. The core insight

The tempting reading of "AI-powered de-editing" is *"throw the video at a multimodal LLM
and hope."* That scores badly: an LLM is unreliable at frame-accurate timing, and it
cannot render pixels.

The reading that actually scores is: **an LLM is a great *perception* layer and a terrible
*execution* layer.** So we split the problem:

| Sub-problem | Best tool | Why |
|---|---|---|
| Where do scenes cut? | **PySceneDetect** (HSV content delta) | Deterministic, frame-accurate, free, no API. An LLM guessing timestamps is strictly worse. |
| What is on screen, and where? | **Gemini 2.5 Flash** (vision + structured output) | Only an LLM can say *"this box is a burned-in caption, that box is a product pop-up, that one is a TikTok watermark"* and read the text. Gemini returns normalised bounding boxes. |
| When does an overlay start/stop? | **Our own temporal clustering** | Per-frame detections → persistent "overlay tracks" via IoU + text similarity. This is the real engineering, and it's ours. |
| Remove the pixels | **ffmpeg `delogo` / `boxblur` with time gating** | One render pass, no per-frame Python loop. Fast enough for a free tier. |

That table *is* the answer to "AI Integration & Understanding" (25%) and "Architecture &
Engineering Decisions" (10%). The README will lead with it.

**The novel bit worth demoing:** most candidates will stop at "detected a box, blurred it."
We produce **overlay tracks with time ranges**, which makes the output genuinely
*re-editable* — you can change a caption's text and re-render it in the same place, or
swap a detected product pop-up for a new reference image. That's two bonus items falling
out of one good data model.

---

## 2. Pipeline

```
 URL ──► yt-dlp ──┐
                  ├──► normalise (ffmpeg: H.264, ≤720p, ≤60s cap) ──► work/{job}/source.mp4
 upload ──────────┘
                          │
                          ▼
        ┌──── PySceneDetect (ContentDetector) ──► Scene[] {start, end}
        │
        ▼
   sample frames  (scene mid-points + every ~1.5s, capped ~24 frames)
        │
        ▼
   Gemini 2.5 Flash  (batched 6 frames/request, JSON schema forced)
        │            → Detection[] {t, bbox, kind, text, confidence}
        ▼
   TrackBuilder  (IoU ≥ 0.5 + text similarity ⇒ same element)
        │            → OverlayTrack[] {kind, text, bbox, start_s, end_s}
        ▼
   ffmpeg render pass:  delogo(box, enable='between(t,s,e)') × N   ──► clean.mp4
        │
        ▼
   ffmpeg cuts: scene clips (raw + clean) + thumbnails + track crops
        │
        ▼
   JobResult (JSON)  ──►  frontend
```

**Overlay `kind` taxonomy** (Gemini classifies into these): `caption` (speech subtitle) ·
`text_overlay` (headline/CTA/sticker text) · `image_popup` (product shot, screenshot,
inset) · `watermark` (TikTok/IG logo, @handle) · `ui_chrome` (fake like/comment bar).
Distinguishing these is what makes it *understanding* rather than *OCR*.

---

## 3. Stack

| Layer | Choice | Rationale |
|---|---|---|
| Backend | **Python 3.11 + FastAPI** | Assignment suggests it; the CV/AI ecosystem is Python. Pydantic gives typed API contracts for free. |
| Video I/O | **ffmpeg** (subprocess) + **yt-dlp** | ffmpeg already installed locally. |
| Scenes | **PySceneDetect** + `opencv-python-headless` | Standard, defensible, offline. |
| Vision | **Gemini 2.5 Flash** via `google-genai` | Free tier. Native bbox output + `response_schema` for guaranteed-valid JSON. |
| Jobs | **In-process asyncio worker + queue** | Real background processing without paying for Redis/Celery. Behind a `JobStore` port so Redis is a 30-line swap. |
| Frontend | **React 18 + Vite + TypeScript + Tailwind** | Faster to build than Angular, and Vercel-native. |
| Deploy | **Hugging Face Spaces (Docker)** + **Vercel** | See §7. |

> ⚠️ Local Python here is **3.14**, which is ahead of some wheels (PySceneDetect/OpenCV).
> First build step is a `venv` on 3.11/3.12 if available, else verify 3.14 wheels resolve.
> The Docker image pins `python:3.11-slim` regardless, so deployment is unaffected.

**Free-tier budget:** ~~~10 RPM / ~250 requests-per-day~~ **Corrected after measuring
it in production:** Google's 429 body reports `GenerateRequestsPerDayPerProjectPerModel-FreeTier`
with `limit: 20` per day on 2.5 Flash. We batch 6 frames per request and cap at 24 frames
⇒ **4 requests per video** ⇒ about **5 de-edits per day**, not 60. Raising
`VISION_BATCH_SIZE` to 12 halves that cost. `vision_stub` adapter (heuristic
edge-density text detection) keeps the app demoable with zero API key or when quota is hit.

---

## 4. Repository layout

Fresh repo — the current directory holds the unrelated *VoiceAgent* assessment.

```
lightnote-deedit/
├── backend/
│   └── app/
│       ├── main.py                    # app factory, CORS, static mount, error handlers
│       ├── config.py                  # pydantic-settings; all env in one place
│       ├── dependencies.py            # ← DI wiring lives ONLY here
│       ├── api/
│       │   ├── routes_jobs.py         # create / status / result
│       │   ├── routes_rebuild.py      # bonus: edit captions, swap overlay image
│       │   └── errors.py              # AppError → HTTP, single handler
│       ├── domain/
│       │   ├── models.py              # BBox, Detection, OverlayTrack, Scene, Job, JobResult
│       │   └── ports.py               # Protocols — the seams
│       ├── services/
│       │   ├── pipeline.py            # orchestration; imports ports only, zero vendor code
│       │   ├── job_runner.py          # asyncio queue, progress callbacks, cancellation
│       │   └── track_builder.py       # detections → tracks  (pure, unit-tested)
│       ├── adapters/
│       │   ├── ingest_ytdlp.py        │ scenes_pyscenedetect.py
│       │   ├── vision_gemini.py       │ vision_stub.py
│       │   ├── editor_ffmpeg.py       │ store_memory.py
│       └── infra/
│           ├── ffmpeg.py              # ONE async subprocess wrapper — no shell strings elsewhere
│           └── workspace.py           # per-job dir layout, URL↔path mapping
│       └── tests/                     # track_builder, ffmpeg filter-graph builder, API contract
├── frontend/src/
│   ├── api/client.ts                  # typed fetch, mirrors pydantic models
│   ├── hooks/useJob.ts                # polling + backoff, single source of job state
│   └── components/                    # UrlUploadForm · StatusBar · SceneTimeline
│                                      # OverlayList · VideoPreview · RebuildPanel
└── README.md
```

### The anti-redundancy rules
1. **One ffmpeg wrapper.** `infra/ffmpeg.py` is the only module that spawns a process.
   Every filter is *built as a data structure*, then rendered to a string — so filter
   graphs are unit-testable without running ffmpeg.
2. **Ports = Protocols.** `pipeline.py` never imports `google.genai`, `yt_dlp`, or
   `scenedetect`. Swapping Gemini→stub is one line in `dependencies.py`.
3. **One model definition.** Pydantic models in `domain/models.py` serialise straight to
   the API; TypeScript types mirror them 1:1. No DTO layer.
4. **Progress is a callback**, not a global. `pipeline.run(src, on_progress)` — the runner
   owns persistence, the pipeline owns work.

---

## 5. API surface

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | `{url}` **or** `multipart file` → `{job_id, status}` (202) |
| `GET` | `/api/jobs/{id}` | `{status, stage, progress, error?}` — polled every 1.5 s |
| `GET` | `/api/jobs/{id}/result` | full `JobResult`: scenes, tracks, media URLs |
| `POST` | `/api/jobs/{id}/rebuild` | bonus: `{caption_edits[], overlay_swaps[]}` → new render |
| `GET` | `/media/{job}/...` | static: `source.mp4`, `clean.mp4`, `scenes/`, `thumbs/` |

`JobStatus`: `queued → downloading → analyzing_scenes → detecting_overlays → rendering → done | failed`.
Each maps to a progress % so the UI shows a real stage name, not a spinner.

---

## 6. Frontend (deliberately modest)

Single page, three sections, Tailwind, no component library:
1. **Input** — URL field + drag-drop, one "De-edit" button.
2. **Status** — stage label + progress bar while running.
3. **Result** —
   - `SceneTimeline`: horizontal strip of thumbnails, width ∝ duration, click to preview.
   - `OverlayList`: grouped by `kind`, each row = crop thumbnail + text + `0:03–0:07`.
   - `VideoPreview`: original ⇄ cleaned toggle.
   - `RebuildPanel` (bonus): edit caption text / upload replacement image → re-render.

Clean and readable, not designed. Matches "functionality over visual design."

---

## 7. Deployment

**Backend → Hugging Face Spaces (Docker SDK).** Free, **16 GB RAM / 2 vCPU / 50 GB disk**,
and Docker means we control the ffmpeg install. Render's free tier is 512 MB RAM with
spin-down — genuinely marginal for video work. HF Spaces is the right call here and is a
good "engineering decision" talking point. *(Render Docker stays documented as fallback.)*

**Frontend → Vercel.** `VITE_API_BASE` env var points at the Space.

Storage is the Space's ephemeral disk with a TTL sweeper — correct for a stateless demo,
and the README names S3/R2 as the production swap.

---

## 8. Build order

| # | Step | Est. |
|---|---|---|
| 1 | Scaffold repo, venv, `requirements.txt`, Dockerfile, `config.py` | 45 m |
| 2 | `infra/ffmpeg.py` + `workspace.py` + probe/normalise | 45 m |
| 3 | `domain/models.py` + `ports.py` (contracts first) | 30 m |
| 4 | Ingest adapters (yt-dlp + upload) | 30 m |
| 5 | Scene detection adapter + scene clips + thumbnails | 45 m |
| 6 | Frame sampler + **Gemini vision adapter** (schema, batching, retry) | 1.5 h |
| 7 | **`track_builder.py`** + unit tests ← *the interesting algorithm* | 1 h |
| 8 | `editor_ffmpeg.py`: delogo filter graph, clean render, clean clips | 1.5 h |
| 9 | `pipeline.py` + `job_runner.py` + DI wiring | 1 h |
| 10 | API routes + error handling + static media | 45 m |
| 11 | Frontend: client, `useJob`, input, status | 1.5 h |
| 12 | Frontend: timeline, overlay list, preview | 1.5 h |
| 13 | Bonus: `RebuildPanel` + `/rebuild` (drawtext + image overlay) | 1.5 h |
| 14 | Deploy both, smoke test | 1 h |
| 15 | README + demo video | 1 h |

Order is dependency-correct: **steps 1–10 alone satisfy the full checklist.** Step 13 is
bonus and droppable if time runs short. Everything ships incrementally testable.

---

## 9. Known limitations (state these openly in the README — it reads as maturity)

- `delogo` blurs/interpolates rather than true inpainting; large opaque overlays leave a
  soft patch. Real fix is E2FGVI/ProPainter video inpainting — GPU, out of free-tier scope.
- Frame sampling means a sub-1.5 s overlay flash can be missed. Tunable, costs API quota.
- yt-dlp against TikTok/Instagram can hit auth walls; upload is the reliable path and the
  UI says so. Demo video will show both.
- Bbox precision is ±2–3 % of frame; masks are padded to compensate.
- Ephemeral storage: jobs vanish on Space restart.

---

## 10. Settled decisions

1. **Backend host: Hugging Face Spaces (Docker), not Vercel.** Vercel is serverless: no
   ffmpeg binary, 250 MB bundle cap, function dies at end of request, ephemeral `/tmp` not
   shared between invocations. Our `202 + poll` job design *requires* a process that
   outlives the HTTP response ⇒ container, not function. Frontend stays on Vercel.
   Render Docker documented as fallback.
2. **Removal is selectable.** `delogo` is the default; `boxblur` and OpenCV `inpaint` are
   selectable per job via the `removal_mode` field. Free to add — removal already sits
   behind the `VideoEditor` port — and the README shows a side-by-side comparison.
3. **Bonus rebuild panel is committed scope**, not a stretch. The `OverlayTrack` model
   makes caption re-render (`drawtext`) and image swap (`overlay`) cheap, and it closes the
   loop the challenge statement asks for: *"editable components so it can be reconstructed."*
