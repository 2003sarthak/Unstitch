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
  github.com/2003sarthak/Unstitch.git). main exists as the stable branch.
- Step 1 of PLAN.md section 8 is DONE: backend scaffold, requirements.txt (exact pins),
  requirements-dev.txt, Dockerfile, .dockerignore, pyproject.toml (ruff + pytest),
  app/config.py bound to backend/.env, app/domain/models.py (VisionProvider + RemovalMode
  StrEnums only - the pydantic models come in step 3), app/main.py (app factory, CORS,
  /api/health), and 20 passing tests. Steps 2-15 are not started.
- Local venv: backend/.venv on CPython 3.11.15, created with `uv venv --python 3.11`.
  Run things as backend/.venv/Scripts/python.exe -m pytest / -m ruff / -m uvicorn.
- Python 3.14 was verified to install and import scenedetect 0.7.1 + OpenCV 5.0 fine.
  3.11 was chosen anyway for parity with the Dockerfile, not because of a wheel gap.
- scenedetect 0.7.1 hard-requires opencv-python (GUI build) with no headless extra, so
  requirements.txt uses opencv-python and the Dockerfile apt-installs libgl1 + libglib2.0-0.
  Do NOT add opencv-python-headless - it installs a second copy of cv2.
- backend/.env and frontend/.env exist locally and are gitignored. Never commit them.
- GEMINI_API_KEY is still EMPTY in backend/.env. Settings.effective_vision_provider
  therefore degrades gemini -> stub, logs a warning at startup, and /api/health reports
  both the requested and effective provider. Ask me before assuming a key is present.
  Build the vision_stub adapter alongside vision_gemini so the whole pipeline can be
  tested end-to-end without a key.

NEXT STEP
Step 2 of the build order in PLAN.md section 8: infra/ffmpeg.py (the single async
subprocess wrapper - filter graphs built as data structures so they unit-test without
running ffmpeg) plus infra/workspace.py (per-job directory layout and URL<->path mapping),
and probe/normalise on top of them. Local ffmpeg is 8.1.2 and has delogo, boxblur,
drawtext and overlay.

HOW I WANT YOU TO WORK
- Clean architecture, dependency injection, no duplicated logic. Quality over quantity.
- Implement only what the assignment asks for, plus the bonus items already committed to in
  PLAN.md section 10. No unrequested extras.
- Commit to dev in small, working increments. Never commit a .env or any secret.
- Explain your technical decisions as you go - I have to defend them in an interview.
```
