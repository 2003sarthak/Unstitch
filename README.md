# Unstitch

**Take a short-form video apart.** Paste a link or upload a file, and Unstitch
finds its scenes, detects the captions, text overlays, product pop-ups and
watermarks burned into it, and renders a clean copy with those elements removed.

Every detected element comes back as a **component with a time range, a region
and its text** — not just a blurred rectangle.

---

## The idea in one table

> An LLM is an excellent **perception** layer and a terrible **execution** layer.

So the model is used for exactly one thing, and everything with a determinable
answer is determined:

| Sub-problem | Handled by | Why |
|---|---|---|
| Where do the scenes cut? | **PySceneDetect** | Frame-accurate and deterministic. A model *guessing* timestamps is strictly worse than measuring them. |
| What is on screen, and what does it say? | **Gemini 2.5 Flash** | Only a model can say *"that's a burned-in subtitle, that's a product pop-up, that's a watermark"* — and read the words. |
| Exactly which pixels does it cover? | **OpenCV** | Measured, because the model's boxes were provably short (a caption reported at `x 0.05–0.50` actually spanned `0.10–0.68`). |
| When does it start and stop? | **Our own tracker** | Per-frame sightings → persistent overlay tracks. |
| Erase the pixels | **ffmpeg** | One render pass, each mask time-gated to its own window. |

That last row is why N overlays cost **one** encode rather than N.

Full walkthrough — layering, every file, the call order, and the bugs found by
running it — is in **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

```
React + Vite + TS  ──HTTP──►  FastAPI  ──►  yt-dlp · PySceneDetect · Gemini · OpenCV · ffmpeg
   (frontend/)                (backend/)
```

---

## Running it locally

### You need

- **Python 3.11**
- **Node 18+**
- **ffmpeg and ffprobe** on your `PATH` — check with `ffmpeg -version`
- A **Gemini API key** (free, no card): https://aistudio.google.com/apikey

> The key is the *only* secret. Without one the app still runs end to end using an
> offline heuristic detector — see [Running with no API key](#running-with-no-api-key).

### 1. Backend

```bash
cd backend

# create the environment (uv installs Python 3.11 for you if you don't have it)
uv venv .venv --python 3.11
uv pip install --python .venv/Scripts/python.exe -r requirements.txt -r requirements-dev.txt

# configure
cp .env.example .env
```

Open `backend/.env` and paste your key into the one blank line:

```ini
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
VISION_PROVIDER=gemini
```

Everything else in that file has a working default and is documented inline.

Start it:

```bash
# Windows
.venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000

# macOS / Linux
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

- API → http://localhost:8000
- Interactive docs → http://localhost:8000/docs
- Health → http://localhost:8000/api/health

That health endpoint reports the detector **actually in use**. If it says
`"vision_provider": "stub"` while you asked for `gemini`, your key didn't load.

<details>
<summary>Using plain <code>pip</code> instead of <code>uv</code></summary>

```bash
cd backend
python3.11 -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt -r requirements-dev.txt
```
</details>

### 2. Frontend

In a **second terminal**:

```bash
cd frontend
npm install
cp .env.example .env            # already points at http://localhost:8000
npm run dev
```

Open **http://localhost:5173**.

### 3. Use it

Paste a short video URL or drop a file, choose a removal mode, press **De-edit**.
A 20-second clip takes roughly a minute — most of it waiting on the vision model.

A URL that is known to work:

```
https://www.youtube.com/shorts/nX4vsznovdg
```

Uploading is the more reliable path; TikTok and Instagram links often sit behind
a login wall that yt-dlp cannot pass.

---

## Running with no API key

Leave `GEMINI_API_KEY` blank. The app detects this at startup, logs a warning,
and binds an offline heuristic detector instead — the whole pipeline still runs,
and the entire test suite passes with no network and no secrets.

It is never silent about it: `/api/health` and every `JobResult` report
`vision_provider: "stub"`, so a heuristic result can't be mistaken for a real
analysis. You can also force it with `VISION_PROVIDER=stub`.

---

## Free-tier quota — read this before demoing

Measured from Google's own rate-limit response:

```
quotaId : GenerateRequestsPerDayPerProjectPerModel-FreeTier
limit   : 20 requests per day   (gemini-2.5-flash)
```

At the default 24 frames batched 6-per-request, **one video costs 4 requests** —
so the free tier is about **five videos per day**. When it runs out, jobs fail at
the "detecting overlays" stage with a message saying so.

To stretch it, in `backend/.env`:

| Change | Effect |
|---|---|
| `VISION_BATCH_SIZE=12` | 2 requests/video → ~10 videos/day |
| `MAX_FRAMES=12` | 2 requests/video, slightly lower recall |
| `VISION_PROVIDER=stub` | Unlimited, no key, heuristic |

---

## Tests

```bash
cd backend
.venv/Scripts/python.exe -m pytest -q      # 274 tests
.venv/Scripts/python.exe -m ruff check app tests
```

No API key or network needed — the offline detector covers the vision port, and
video fixtures are synthesised by ffmpeg rather than committed.

```bash
cd frontend
npm run build                              # type-check + production build
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/jobs` | `{url, removal_mode}` → `202 {job_id}` |
| `POST` | `/api/jobs/upload` | multipart file → `202 {job_id}` |
| `GET` | `/api/jobs/{id}` | status, stage, progress — polled |
| `GET` | `/api/jobs/{id}/result` | scenes, overlay tracks, media URLs |
| `GET` | `/media/{job}/…` | the videos, thumbnails and crops |
| `GET` | `/api/health` | version and the detector actually in use |

`202` plus polling, because a de-edit takes tens of seconds and no sensible
client holds a connection open for that. It's also why the backend must be a
container rather than a serverless function — the work has to outlive the
response that acknowledged it.

---

## Limitations

Stated openly; most were found by running the thing rather than reading it.

- **Free tier ≈ 5 videos/day** (above).
- **A quota failure fails the job** rather than degrading to the offline detector.
- **`delogo` interpolates, it doesn't inpaint.** Large opaque overlays leave a soft
  patch. The real fix is video inpainting (E2FGVI/ProPainter) — GPU, out of scope.
- **Small overlays can fragment.** A watermark may split into a few tracks: box
  refinement snaps to whatever edges it finds, and under a small overlay the
  footage changes every frame.
- **Sub-1.5s overlay flashes can be missed.** Tunable via `FRAME_SAMPLE_INTERVAL_S`,
  at the cost of quota.
- **Gemini returns `confidence: 1.0` for everything**, so `MIN_DETECTION_CONFIDENCE`
  is inert on that path (it still works for the stub).
- **Storage is ephemeral** and job state is in-process — a restart loses both.
- **Prototype, not deployed.** The Dockerfile is written but the image has not been
  built or hosted.

---

## Layout

```
backend/
  app/
    domain/      models + ports (Protocols) — pure, the centre
    services/    pipeline, job runner, track builder — imports ONLY ports
    adapters/    yt-dlp · PySceneDetect · Gemini · stub · OpenCV · ffmpeg · store
    infra/       the single ffmpeg wrapper, workspace paths
    api/         routes and error mapping
    dependencies.py   the only module that names a concrete adapter
  tests/         274 tests, incl. one that enforces the layering from the AST
frontend/
  src/api        typed client mirroring the pydantic models
  src/hooks      useJob — the only thing that polls
  src/components submit · status · preview · timeline · overlay list
docs/
  ARCHITECTURE.md
```
