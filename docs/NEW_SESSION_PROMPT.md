# Paste this into a fresh Claude Code session

```
Read docs/PLAN.md first - it has the full architecture and build order. Then continue from
where the previous session stopped.

PROJECT
Unstitch: an AI video de-editing app for the LightNoteAI Full-Stack AI Developer assignment
(brief in AssignmentDocs/, gitignored). Paste a short-form video URL or upload a file, and it
segments the video into scenes, detects captions / text overlays / product pop-ups with
bounding boxes and time ranges, removes them to produce a clean video, and lets you re-edit
those components. Deadline: 6 Sep 2026, 18:00.

STACK (already decided, see PLAN.md sections 1-3)
- Backend: Python + FastAPI, ports-and-adapters. pipeline.py imports only Protocols, never
  vendor SDKs. All DI wiring lives in dependencies.py. One ffmpeg subprocess wrapper for the
  whole codebase, with filter graphs built as data structures so they are unit-testable.
- Scenes: PySceneDetect (deterministic). Vision: Gemini 2.5 Flash with forced JSON schema
  (bounding boxes + text + element type). Removal: ffmpeg delogo with time-gated enable=.
- Frontend: React + Vite + TypeScript + Tailwind. Simple, functional, not designed.
- Deploy: backend to Hugging Face Spaces (Docker), frontend to Vercel. NOT Vercel for the
  backend - it is serverless, and our 202-plus-polling job design needs a process that
  outlives the HTTP response.
- Everything must stay on free tiers.

CURRENT STATE
- Repo root is this folder. Branch: dev (tracking origin/dev at
  github.com/2003sarthak/Unstitch.git). main exists as the stable branch. NOT pushed yet.
- Steps 1-3 of PLAN.md section 8 are DONE. 143 tests pass, ruff clean.
  * Step 1: backend scaffold, requirements.txt (exact pins), requirements-dev.txt,
    Dockerfile, .dockerignore, pyproject.toml, app/config.py, app/main.py (factory,
    CORS, /api/health).
  * Step 2: app/infra/ffmpeg.py (the ONLY module that spawns a process; filter graphs
    are frozen dataclasses that render to ffmpeg syntax), app/infra/workspace.py
    (per-job layout, path<->URL mapping, TTL sweeper), app/adapters/editor_ffmpeg.py
    (probe + normalise only so far), app/domain/errors.py.
  * Step 3: app/domain/models.py (BBox/PixelBox/VideoMeta/SampledFrame/Detection/
    OverlayTrack/Scene/Job/JobResult + enums) and app/domain/ports.py (Protocols:
    MediaIngestor, SceneDetector, FrameSampler, VisionDetector, VideoEditor, JobStore).
- Steps 4-15 are NOT started. Nothing exists yet under app/services/ or app/api/,
  and app/dependencies.py has not been written.
- Local venv: backend/.venv on CPython 3.11.15 (uv venv --python 3.11). Run everything
  as backend/.venv/Scripts/python.exe -m pytest / -m ruff / -m uvicorn.
- Python 3.14 verified to install and import scenedetect 0.7.1 + OpenCV 5.0 fine; 3.11
  was chosen for parity with the Dockerfile, not because of a wheel gap.
- scenedetect 0.7.1 hard-requires opencv-python (GUI build) and has no headless extra,
  so requirements.txt uses opencv-python and the Dockerfile apt-installs libgl1 +
  libglib2.0-0. Do NOT add opencv-python-headless - it installs a second copy of cv2.
- tests/test_architecture.py enforces the layering by walking each module's AST. If you
  make services/ import a vendor SDK or an adapter, the suite fails. That is deliberate.
- Conventions that are already load-bearing: geometry is normalised 0..1 (never pixels
  until BBox.to_pixels at the ffmpeg boundary); time is seconds, never frame indices.
- backend/.env and frontend/.env exist locally and are gitignored. Never commit them.
- GEMINI_API_KEY is still EMPTY in backend/.env - the user will paste it near the end,
  along with any other accounts that need manual signup. Settings.effective_vision_provider
  therefore degrades gemini -> stub, logs a warning at startup, and /api/health reports
  both requested and effective provider. Everything must stay runnable without the key.

NEXT STEP
Step 4 of the build order in PLAN.md section 8: the ingest adapters - adapters/
ingest_ytdlp.py implementing the MediaIngestor port (yt-dlp is synchronous, so move it
to a thread; honour YTDLP_COOKIES_FILE; translate failures into InvalidInputError), plus
the upload path, which deliberately does NOT go through the port - the route writes the
bytes straight into workspace.original. Then step 5, the PySceneDetect adapter.

HOW I WANT YOU TO WORK
- Clean architecture, dependency injection, no duplicated logic. Quality over quantity.
- Implement only what the assignment asks for, plus the bonus items already committed to in
  PLAN.md section 10. No unrequested extras.
- Commit to dev in small, working increments. Never commit a .env or any secret.
- Explain your technical decisions as you go - I have to defend them in an interview.
```
