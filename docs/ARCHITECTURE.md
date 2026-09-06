# Unstitch — Architecture & Code Walkthrough

> How the system is put together, what every file does, and the order things are
> called in. Written to be read top to bottom before an interview.
>
> ~5,250 lines of source across backend and frontend, 274 tests.

---

## 1. The one idea everything follows from

> **An LLM is an excellent *perception* layer and a terrible *execution* layer.**

The naive reading of "AI-powered de-editing" is *throw the video at a multimodal
model and hope*. That fails, because the model is asked to do three jobs and is
only good at one of them:

| Sub-problem | Has an exact answer? | Who does it here | Why |
|---|---|---|---|
| Where do the scenes cut? | **Yes** | PySceneDetect | Frame-accurate, deterministic, free. A model *guessing* timestamps is strictly worse than measuring them. |
| What is on screen, and what does it say? | **No** | Gemini 2.5 Flash | Only a model can say *"this is a burned-in subtitle, that is a product pop-up, that is a TikTok watermark"* and read the words. |
| Exactly which pixels does it cover? | **Yes** | OpenCV (`refine_cv.py`) | Measured, because the model's boxes were provably wrong — see §7. |
| When does an overlay start and stop? | Derived | `track_builder.py` | Our own algorithm. Per-frame sightings → persistent tracks. |
| Erase the pixels | **Yes** | ffmpeg filter graph | One render pass, time-gated. No per-frame Python loop. |

Every architectural decision below is downstream of that table. The model is
used for exactly one thing, and everything with a determinable answer is
determined.

**What makes the output re-editable:** most implementations stop at *"found a
box, blurred it"*. We produce **overlay tracks with time ranges**, which is the
difference between a video that has been smudged and one that has been taken
apart into components. Removal, the overlay list, caption re-rendering and image
swapping are then four *views of one model* rather than four features.

---

## 2. Layering

```
        ┌──────────────────────────────────────────┐
        │  api/        HTTP in, HTTP out           │
        └──────────────┬───────────────────────────┘
                       │  calls
        ┌──────────────▼───────────────────────────┐
        │  services/   orchestration & algorithms  │
        │              (imports ONLY Protocols)    │
        └──────────────┬───────────────────────────┘
                       │  depends on
        ┌──────────────▼───────────────────────────┐
        │  domain/     models + ports (Protocols)  │
        │              pure; imports only pydantic │
        └──────────────▲───────────────────────────┘
                       │  implements
        ┌──────────────┴───────────────────────────┐
        │  adapters/   one per external system     │
        │  infra/      ffmpeg wrapper, workspace   │
        └──────────────────────────────────────────┘

              dependencies.py  ← the ONLY module that
                                 knows which adapter is which
```

**The rule that matters:** `services/pipeline.py` never imports `yt_dlp`,
`scenedetect`, `cv2` or `google.genai`. That single constraint is what makes the
whole pipeline runnable with **no API key** (bind the stub detector instead) and
testable with **no network**.

This is not a convention we hope people follow — `tests/test_architecture.py`
parses every module's AST and fails the build if an arrow points the wrong way.
Verified by deliberately adding `import scenedetect` to the domain and watching
it fail:

```
AssertionError: app\domain\models.py imports ['scenedetect']:
a domain that knows about Gemini is not a domain
```

---

## 3. The full request flow

### 3.1 Startup (once)

```
uvicorn app.main:app
  │
  ├─ main.py :: create_app()
  │     ├─ config.py :: get_settings()          reads backend/.env, validates
  │     ├─ dependencies.py :: build_container()
  │     │     ├─ WorkspaceManager(media_root)
  │     │     ├─ InMemoryJobStore()
  │     │     ├─ build_pipeline(settings)
  │     │     │     ├─ Ffmpeg(...)                      infra
  │     │     │     ├─ YtDlpIngestor(...)               adapter
  │     │     │     ├─ PySceneDetectDetector(...)       adapter
  │     │     │     ├─ FfmpegFrameSampler(...)          adapter
  │     │     │     ├─ build_vision_detector(...)  ←── THE branch: gemini | stub
  │     │     │     ├─ build_box_refiner(...)      ←── OpenCV | null
  │     │     │     └─ FfmpegVideoEditor(...)           adapter
  │     │     ├─ JobRunner(pipeline, store, workspaces)
  │     │     └─ WorkspaceSweeper(store, workspaces, ttl)
  │     ├─ CORSMiddleware               from settings.cors_origins
  │     ├─ install_error_handlers()     api/errors.py
  │     └─ include_router(jobs, media)
  │
  └─ lifespan
        ├─ logging.basicConfig(level=LOG_LEVEL)
        ├─ media_root.mkdir()           fail at boot, not mid-render
        ├─ warn if vision degraded      gemini requested but no key
        ├─ runner.start()               spawns MAX_CONCURRENT_JOBS workers
        └─ sweeper.start()              TTL cleanup timer
```

### 3.2 A job, end to end

```
BROWSER                          BACKEND
───────                          ───────
SubmitForm  ──POST /api/jobs──►  routes_jobs.create_job_from_url
  or /upload                       │  (upload writes bytes to workspace.original first)
                                   ├─ runner.submit(job_id, request)
                                   │    ├─ store.create(Job status=queued)   ← persisted BEFORE
                                   │    └─ queue.put(...)                       enqueue, so the
   ◄────202 {job_id}───────────────┘                                            first poll works

                                 JobRunner._work (background worker)
                                   └─ _run_one
                                        ├─ workspaces.create(job_id)   makes work/{id}/ tree
                                        └─ pipeline.run(workspace, request, on_progress)
                                             │
useJob polls  ──GET /api/jobs/{id}──►  routes_jobs.get_job → store.get
   every 0.8→3s (backoff)                                    │
                                             ┌───────────────┘
                                             ▼
   ┌───────────────────── pipeline.run stages ─────────────────────┐
   │ 1. on_progress(DOWNLOADING)          10%                      │
   │    _acquire_source                                            │
   │      └─ ingest_ytdlp.fetch()    (URL only; upload skips this) │
   │    editor.normalise()                                         │
   │      └─ infra/ffmpeg.run()  H.264, ≤720p, rotation baked in   │
   │                                                               │
   │ 2. on_progress(ANALYZING_SCENES)     25%                      │
   │    scenes_pyscenedetect.detect()                              │
   │      └─ build_scenes()  merge flash-frames, tile the timeline │
   │    _render_thumbnails()  editor.thumbnail() × N, concurrent   │
   │                                                               │
   │ 3. on_progress(DETECTING_OVERLAYS)   45%   ← the slow stage   │
   │    sampler_ffmpeg.sample()                                    │
   │      └─ choose_timestamps()  scene mids + interval, thinned   │
   │    vision_gemini.detect()  OR  vision_stub.detect()           │
   │      └─ batched 6 frames/request, forced JSON schema          │
   │    refine_cv.refine()   snap boxes onto real pixels           │
   │    track_builder.build_tracks()  ← detections → components    │
   │                                                               │
   │ 4. on_progress(RENDERING)            80%                      │
   │    editor.remove_overlays()                                   │
   │      └─ removal_graph()  N time-gated filters, ONE pass       │
   │    _render_crops()  editor.crop() × N, concurrent             │
   └───────────────────────────────────────────────────────────────┘
                                             │
                                        returns JobResult
                                             ├─ store.save_result()
                                             └─ store.update(status=done)

useJob sees done ──GET /api/jobs/{id}/result──► routes_jobs.get_job_result
   ◄──── JobResult JSON ─────────────────────────────────┘

<video src>, <img src> ──GET /media/{job}/...──► routes_media.get_media
                                                  └─ workspaces.resolve_media_path()
                                                       (validates id + resolved path)
                                                  └─ FileResponse (supports Range → 206)
```

---

## 4. Backend files — what each one is and why it exists

### `app/main.py` — application factory
Builds the `FastAPI` app, installs CORS and error handlers, mounts routers,
exposes `/api/health`. The **lifespan** starts and — importantly — *stops* the
worker pool; a pool not drained on shutdown leaves ffmpeg subprocesses orphaned.
A factory rather than a module-level `app = FastAPI()` so tests can build an
isolated instance with their own settings.

**Called by:** uvicorn. **Calls:** `config`, `dependencies`, `api/*`.

### `app/config.py` — every knob, declared once
`pydantic-settings` bound to `backend/.env`. Nothing else in the codebase reads
`os.environ`.

Three things worth knowing:
- **`.env` is anchored to `__file__`, not the CWD** — uvicorn, pytest and Docker
  all have different working directories and must read the same file.
- **Bounds live on the fields.** `Field(ge=0, le=1)` turns a typo like
  `TRACK_IOU_THRESHOLD=50` into a clear startup error instead of a tracker that
  silently matches nothing an hour later.
- **`effective_vision_provider`** — asking for Gemini with no key *degrades to
  the stub* rather than refusing to boot, so the repo runs with zero secrets.
  Never silently: startup logs a warning and `/api/health` reports both the
  requested and effective provider.

### `app/dependencies.py` — the composition root
The **only** module that names a concrete adapter. `build_container()` assembles
everything once at startup; `get_container()` is the FastAPI dependency routes
use. `build_vision_detector()` is the single branch the stub exists for.

Swapping the in-memory store for Redis is one line here and nothing else changes.

### `app/domain/models.py` — the shared vocabulary
Pydantic models that are simultaneously the API contract, the pipeline's working
types, and the mirror for the TypeScript types. **One definition, no DTO layer.**

Two conventions run through everything:
- **Geometry is normalised 0..1**, never pixels. The video is downscaled, clips
  are re-encoded, and the browser renders at whatever size the layout gives it —
  a pixel box would be wrong in two of those three places. Converted to pixels
  only at `BBox.to_pixels()`, the moment it meets ffmpeg.
- **Time is seconds**, never frame indices. Sources run 24–60fps, some variable.

Key types: `BBox` (with `iou`, `union`, `padded`, `to_pixels`), `PixelBox`
(with `clamped_to_frame`), `VideoMeta`, `SampledFrame`, `Detection` (one sighting),
**`OverlayTrack`** (the component), `Scene`, `Job`, `JobRequest`, `JobResult`,
and the enums `OverlayKind` / `RemovalMode` / `JobStatus` / `VisionProvider`.

### `app/domain/ports.py` — the seams
Six `Protocol`s: `MediaIngestor`, `SceneDetector`, `FrameSampler`,
`VisionDetector`, `BoxRefiner`, `VideoEditor`, `JobStore`. `Protocol` rather
than ABC on purpose — adapters inherit from nothing, so the dependency arrow
points inward only, checked structurally.

All I/O methods are `async`; a blocking implementation (PySceneDetect is
synchronous and CPU-bound) honours that by moving work to a thread, and that
decision belongs in the adapter rather than leaking to every caller.

### `app/domain/errors.py` — the error vocabulary
Split by *whose fault it is*, not by which component raised it, because that is
the axis the HTTP layer needs. `InvalidInputError` → 4xx and shown verbatim;
`ProcessingError` → 5xx and logged with a trace.

### `app/infra/ffmpeg.py` — the only module that spawns a process
Timeouts, logging and turning a non-zero exit into a useful exception are written
once. Nowhere else builds a shell string, so a filename with a quote in it cannot
become a command-injection bug.

**Filter graphs are frozen dataclasses**, not strings: `Filter` → `FilterChain` →
`FilterGraph`. So the logic deciding *what gets erased and when* produces a value
you can assert on with no video and no subprocess. `to_args()` even derives
`-vf` vs `-filter_complex` from the graph's own shape.

The subtle part is **escaping**. A time gate like `between(t,0.5,1.5)` contains
commas, which ffmpeg reads as *filter separators* unless escaped. Uses
`str.translate` (single pass) so a backslash already in the value isn't escaped
twice.

Also parses `ffprobe` JSON into `VideoMeta`, including **rotation** from the
display matrix — phone video stores landscape pixels plus a rotation, and
ignoring it would express every bounding box against the wrong dimensions.

### `app/infra/workspace.py` — per-job layout, and the path↔URL boundary
```
work/{job_id}/
    original.mp4        as ingested
    source.mp4          normalised: H.264, ≤720p, faststart
    clean.mp4           overlays removed
    frames/f0000.jpg    frames sampled for the vision pass
    thumbs/s000.jpg     scene thumbnails
    crops/t000.jpg      one crop per detected overlay
    scenes/, rebuild/   reserved
```
Indexed names are zero-padded so lexicographic order *is* chronological order.

`resolve_media_path()` is the security boundary — the one place client text
becomes a filesystem path. **Two independent defences:** the job id must match
the id pattern, *and* the resolved path must still be inside the workspace. The
second catches `..` segments and symlinks the first would not.

### `app/adapters/ingest_ytdlp.py` — URL download
Implements `MediaIngestor`. yt-dlp is synchronous, so calls go to a thread.
Reads metadata *first* and rejects over-length video before spending bandwidth.
Downloads into a staging dir then moves, because yt-dlp decides the final
extension itself.

Format selector is `bv*[height<=1080]+ba/b[height<=1080]/bv*+ba/b` — asking for
a single progressive MP4 fails on YouTube, which serves separate video and audio
streams (see §7). Translates yt-dlp's extractor internals into advice a user can
act on.

**Uploads deliberately do not use this port** — there is nothing to fetch, so the
route writes bytes straight into the workspace.

### `app/adapters/scenes_pyscenedetect.py` — cut detection
Implements `SceneDetector`. `ContentDetector` compares consecutive frames in HSV
space. Runs in a thread.

`build_scenes()` is split out as a **pure function** so the awkward cases test
without decoding video: never returns an empty list (no cuts = one scene), merges
flash-frames, and guarantees scenes tile the whole timeline with no gaps.

### `app/adapters/sampler_ffmpeg.py` — which frames to spend quota on
Implements `FrameSampler`. **This module decides what the app costs to run.**

`choose_timestamps()` (pure) combines scene mid-points — a cut is where the
picture changes most — with a fixed interval, because overlays don't respect
scene boundaries. When that exceeds `MAX_FRAMES`, the list is **thinned evenly
across the timeline rather than truncated**; truncating would spend the entire
budget on the opening seconds and look like a bad model rather than a bad sampler.

### `app/adapters/vision_gemini.py` — the LLM
Implements `VisionDetector`. Three decisions:
- **Schema is forced, not requested.** `response_schema` makes the API guarantee
  well-formed JSON in our shape — no parsing prose, no regex over code fences.
- **Frames are batched** (6/request), because the free tier bills per *request*.
- **Requests are sequential**, so several concurrent jobs don't burst into a 429.

`_to_bbox()` converts Gemini's `[ymin, xmin, ymax, xmax]` on a 0–1000 grid into
our normalised box — a different order *and* scale. Returns `None` for a
degenerate box so one bad rectangle costs itself, not the job.

### `app/adapters/vision_stub.py` — the offline detector
Implements the same port with **no network and no key**. Morphological gradient +
Otsu isolates strokes; a wide-short closing kernel merges characters into lines;
classification is positional (bottom-third wide = caption, small corner =
watermark).

It is honest about being a heuristic: it never sets `text`, and
`JobResult.vision_provider` reports which detector actually ran so a stub result
is never mistaken for a real analysis.

**This adapter is why the end-to-end test can run in CI at all.**

### `app/adapters/refine_cv.py` — measuring what the model estimated
Implements `BoxRefiner`. Exists because of a **measured** defect (§7): Gemini's
boxes were consistently short. Snaps each box onto the pixels it describes with
local edge analysis, iterating to a fixed point because the search window is
sized from the box being corrected.

Safety: refuses any growth beyond `_MAX_GROWTH` × the original area, because a
mask covering half the video is worse than one slightly too small.

### `app/adapters/editor_ffmpeg.py` — everything that writes a video
Implements `VideoEditor`. Decides *what* should happen; hands the *how* to
`infra/ffmpeg`. Never spawns a process itself, never formats a filter by hand.

- `normalise()` — one re-encode up front so four later stages can assume H.264 at
  a known height. `-2:min(ih,720)` keeps aspect, rounds even (4:2:0 requires it),
  and only ever downscales.
- `removal_graph()` — **pure and public**, so what gets erased is testable
  exhaustively. `delogo` = one time-gated filter per track, all in one chain, one
  render pass. `boxblur` needs three chains per track (split → crop+blur →
  overlay) because it has no region parameter — this is what the labelled-chain
  support exists for.
- `cut()`, `thumbnail()`, `crop()` — `-ss` before `-i` for a seek rather than a
  linear decode.
- `_run()` — the single place `FFmpegError` becomes a domain error, so a method
  added later cannot forget to translate.

### `app/services/track_builder.py` — **the algorithm that is ours**
Turns per-frame `Detection`s into `OverlayTrack`s. Greedy nearest-match
association: walk detections in time order, find the open track each most
plausibly continues, extend or open, retire tracks unseen for `max_gap_s`.

Two sightings are the same thing when **boxes overlap enough AND text agrees**.
Geometry alone merges a caption with the headline that replaces it in the same
spot; text alone merges two copies of the same word in different corners.

Three refinements, each from a real failure:
1. **Containment route.** IoU punishes a box that changes size — which is exactly
   what a caption does as it's revealed word by word ("we" vs "we tried" scores
   0.43). Strong text agreement + containment associates them anyway.
2. **Scale-neutral IoU.** IoU is a ratio, so identical jitter costs a small box
   far more. Both boxes are padded by a fixed 1.5% of frame before comparing.
3. **Half-interval padding** on the time range: sampling proves presence *at*
   instants, so the overlay almost certainly appeared before the first sighting.

Pure functions over pure data — no I/O, no model, no ffmpeg.

### `app/services/pipeline.py` — orchestration
Owns stage order and nothing else. Imports only Protocols. Progress is pushed
through a callback: the runner owns persistence, the pipeline owns work.

### `app/services/job_runner.py` — background execution
**This module is why the backend is a container and not a serverless function.**
The API answers 202 and the work continues afterwards; a function killed when its
response is sent cannot do that.

`asyncio.Queue` + a fixed worker pool. Celery would buy durability across
restarts, worth nothing when storage is ephemeral anyway. The pool buys a hard
concurrency ceiling, which matters because encoding is CPU-bound.

`WorkspaceSweeper` expires files *and* job state together — dropping one without
the other leaves a job reporting success and serving 404s.

### `app/adapters/store_memory.py` — job state
The app's clearest scaling limit, named as a port so swapping it is obvious.
`model_copy` rather than mutation, since `Job` is handed out to request handlers.

### `app/api/routes_jobs.py`, `routes_media.py`, `errors.py`
Submit/poll/result; media serving via the validated resolver (a route rather than
a `StaticFiles` mount, so it inherits the traversal defences); and one table
mapping domain errors to status codes. Uploads are **streamed in chunks with the
size cap enforced as bytes arrive**, because `Content-Length` is client-supplied.

---

## 5. Frontend files

| File | Role |
|---|---|
| `main.tsx` | Mounts `<App/>`. |
| `App.tsx` | One page, three sections. Owns the cross-component state — the seek target and the highlighted track — because three children share them. |
| `api/types.ts` | Mirrors the pydantic models 1:1, hand-written so it can carry *reasons*. |
| `api/client.ts` | The only module that talks to the backend. One `request()` so base URL, JSON parsing and error translation are written once. |
| `hooks/useJob.ts` | The only thing that polls. Stops at a terminal status, backs off 0.8s→3s, and tags each attempt so a stale response can't overwrite a newer one. |
| `components/SubmitForm.tsx` | URL + drag-drop sharing one button; they're the same request with different payloads. |
| `components/StatusBar.tsx` | Stage label *and* bar — they answer different questions. |
| `components/VideoPreview.tsx` | One player with a Cleaned⇄Original toggle that preserves playback position, plus detected boxes drawn as **percentages** (possible only because `BBox` is normalised). |
| `components/SceneTimeline.tsx` | Thumbnails with width ∝ duration, so the strip is a picture of the edit's rhythm. Click seeks the main player. |
| `components/OverlayList.tsx` | Grouped by kind — the project's argument made visible. Each row carries its crop: the claim, and the evidence. |
| `lib/format.ts` | Shared timecode/label/colour helpers, so a row and its rectangle are recognisably the same thing. |

---

## 6. Testing strategy

274 tests, split by what they can actually prove:

| Layer | Approach |
|---|---|
| Pure logic (`track_builder`, `build_scenes`, `choose_timestamps`, filter graphs, `BBox`) | Hand-written data in, exact assertions out. Microseconds. |
| ffmpeg integration | Clips **synthesised by ffmpeg** in fixtures — the repo stays text-only and each clip's properties are visible in the test using it. |
| Pixel truth | `test_removal.py::TestPixelsActuallyChange` opens both videos and compares. Everything else would pass on a render that did nothing. |
| API contract | `test_api.py` drives real uploads through the real pipeline with the stub detector. |
| Architecture | `test_architecture.py` parses ASTs and enforces the layering. |
| Cross-component invariants | e.g. stub's confidence ceiling vs the tracker's floor; `TRACK_MAX_GAP_S` vs `FRAME_SAMPLE_INTERVAL_S`. |

Several tests were **mutation-checked** — the escaping, `+faststart`, and the
architecture guard were each deliberately broken to confirm the test fails.

---

## 7. Bugs found by running it, not by reading it

Worth knowing; they are the strongest evidence the system was actually exercised.

1. **`TRACK_MAX_GAP_S=2.0` could not do its job.** Sampling every 1.5s means one
   missed detection leaves a **3.0s** gap — larger than the tolerance meant to
   absorb it. Now 3.5s, with the relationship pinned by a test.

2. **Word-by-word captions split.** IoU("we", "we tried") = 0.43 because the box
   grows with the text. Fixed with the containment route.

3. **Gemini's boxes were consistently short.** Every coordinate came back a
   multiple of ten — it estimates, it doesn't measure.
   ```
   caption   model x 0.050-0.500   actual x 0.100-0.680   → right third survived
   card      model x 0.590-0.760   actual x 0.562-0.951
   ```
   Prompting for accuracy made the watermark *worse*. Fixed by measuring
   (`refine_cv.py`); after: caption `0.09-0.68`, card `0.56-0.95`.

4. **`boxblur` failed on every thin caption.** It blurs chroma too, and yuv420p
   chroma planes are half resolution — so chroma is the binding constraint at
   half the limit. `Invalid chroma_param radius value 10, must be >= 0 and < 10`.

5. **The tests passed on a pipeline that found nothing.** First real run: 8
   detections → 0 tracks, suite green, because *"every track is well-formed"* is
   trivially true of zero tracks.

6. **The YouTube URL path was entirely broken.** The format selector asked for a
   single progressive MP4; YouTube serves 18 video-only and 5 audio-only
   renditions and no combined file. And the error message *blamed the user's link*
   for our bug.

7. **Small overlays fragmented.** Nine tracks piled in one corner where there was
   one watermark, because IoU is scale-sensitive. Fixed with scale-neutral IoU.

8. **The free-tier quota was 12× smaller than assumed** — see §8.

---

## 8. Known limitations (state these openly)

- **Free tier is ~5 videos/day, not 60.** Measured from Google's own 429 body:
  `GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20` per day on
  gemini-2.5-flash. At 24 frames / 6 per request that is 4 requests per video.
  Mitigations: raise `VISION_BATCH_SIZE` to 12 (→2 requests), lower `MAX_FRAMES`,
  or `VISION_PROVIDER=stub`.
- **Retry burns quota on a daily limit.** The backoff retries 3× on any 429; for
  a *per-day* quota that spends two extra requests achieving nothing. It should
  read `quotaId` and give up immediately on a daily violation.
- **A quota failure fails the whole job** rather than degrading to the stub. A
  graceful fallback would make the demo always complete (and `vision_provider`
  already reports honestly), at the cost of a quieter failure.
- **Small overlays still fragment.** The watermark on a real Short splits into
  four: refinement snaps to whatever edges it finds, and under a small overlay the
  footage changes every frame, so the measured box oscillates `w=0.03`→`w=0.12`.
  The fix is stabilising refinement for small regions, not a bigger tolerance.
- **`delogo` interpolates, it does not inpaint.** Large opaque overlays leave a
  soft patch. Real fix is E2FGVI/ProPainter — GPU, out of free-tier scope.
- **OpenCV inpainting was dropped, not stubbed.** ~50ms/frame ≈ 2 minutes for a
  90s clip, and it cannot share the single render pass.
- **Sub-1.5s overlay flashes can be missed.** Tunable, costs quota.
- **Gemini returns `confidence: 1.0` for everything**, so
  `MIN_DETECTION_CONFIDENCE` is inert on the Gemini path (it still works for the
  stub).
- **Scene *clips* are not rendered.** Exact boundaries are returned and the player
  seeks; cutting every scene would be N extra encodes for no new capability.
- **Storage is ephemeral** and job state is in-process — a restart loses both.
- **Not verified:** the Docker build (Docker isn't installed on the dev machine)
  and deployment.

---

## 9. Where to start reading

1. `app/domain/models.py` — the vocabulary. Everything else refers to these.
2. `app/domain/ports.py` — the seams. Shows the shape of the system in one file.
3. `app/services/pipeline.py` — stage order, and proof the abstraction holds.
4. `app/services/track_builder.py` — the algorithm worth defending.
5. `app/adapters/refine_cv.py` — the clearest example of the project's thesis.
6. `tests/test_architecture.py` — the rules, made executable.
