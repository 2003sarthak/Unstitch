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
- Repo root is this folder. Branch: dev (github.com/2003sarthak/Unstitch.git). NOT pushed yet.
- THE BACKEND IS COMPLETE. Steps 1-10 of PLAN.md section 8 are done. 261 tests pass,
  ruff clean, and a real upload has been driven end to end through a live uvicorn server.
- Verified working end to end with NO API key (stub detector): upload -> 202 -> poll ->
  scenes + overlay tracks + clean.mp4, every media URL serving, and the overlay provably
  erased in pixel space (42% bright pixels inside the mask -> 0%, footage outside it
  byte-identical).
- API: POST /api/jobs (json {url, removal_mode}), POST /api/jobs/upload (multipart),
  GET /api/jobs/{id}, GET /api/jobs/{id}/result, GET /media/{job}/{path}, GET /api/health.
  Interactive docs at /docs.
- Run it:  cd backend && .venv/Scripts/python.exe -m uvicorn app.main:app --reload --port 8000
  Test it: cd backend && .venv/Scripts/python.exe -m pytest -q
- Local venv is backend/.venv on CPython 3.11.15 (matches the Dockerfile).
- Steps 11-13 (frontend: client+useJob, timeline/overlay list/preview, rebuild panel),
  14 (deploy) and 15 (README + demo) are NOT started. frontend/ holds only .env.example.
- Scene *clips* are deliberately not rendered - JobResult carries exact scene boundaries
  and the player seeks. Cutting every scene would be N extra encodes for no new capability.
- RemovalMode is delogo | boxblur only. OpenCV inpaint was dropped, not stubbed; see the
  docstring in domain/models.py for why.
- tests/test_architecture.py enforces the layering from each module's AST. If you make
  services/ import an adapter or a vendor SDK, the suite fails. That is deliberate.
- GEMINI_API_KEY is still EMPTY in backend/.env. The user pastes it at the end. Settings
  degrades gemini -> stub, warns at startup, and /api/health reports both providers.
  When the key arrives, nothing needs rewiring: dependencies.build_vision_detector already
  branches on settings.effective_vision_provider.
- backend/.env and frontend/.env are gitignored. Never commit them.

NEXT STEP
Step 11 of PLAN.md section 8: scaffold the frontend (Vite + React + TS + Tailwind),
frontend/src/api/client.ts mirroring the pydantic models 1:1, hooks/useJob.ts (poll
GET /api/jobs/{id} every 1.5s with backoff, stop on done/failed), and the input + status
UI. VITE_API_BASE_URL is already in frontend/.env.example. Then steps 12-13.

HOW I WANT YOU TO WORK
- Clean architecture, dependency injection, no duplicated logic. Quality over quantity.
- Implement only what the assignment asks for, plus the bonus items already committed to in
  PLAN.md section 10. No unrequested extras.
- Commit to dev in small, working increments. Never commit a .env or any secret.
- Explain your technical decisions as you go - I have to defend them in an interview.
```
