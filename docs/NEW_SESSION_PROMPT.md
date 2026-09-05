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
- Done so far: ONLY the config scaffold - .gitignore, backend/.env.example,
  frontend/.env.example, docs/PLAN.md. No application code has been written yet.
- backend/.env and frontend/.env exist locally and are gitignored. Never commit them.
- GEMINI_API_KEY is the single required secret. I add it manually; ask me before assuming
  it is present. Build the vision_stub adapter alongside vision_gemini so the whole pipeline
  can be tested end-to-end without a key.

NEXT STEP
Step 1 of the build order in PLAN.md section 8: scaffold the backend, write requirements.txt
and config.py bound to .env, and verify PySceneDetect / opencv-python-headless actually
resolve on this machine's Python 3.14 (they may not have wheels yet - fall back to a 3.11 or
3.12 venv if needed; the Dockerfile pins python:3.11-slim regardless).

HOW I WANT YOU TO WORK
- Clean architecture, dependency injection, no duplicated logic. Quality over quantity.
- Implement only what the assignment asks for, plus the bonus items already committed to in
  PLAN.md section 10. No unrequested extras.
- Commit to dev in small, working increments. Never commit a .env or any secret.
- Explain your technical decisions as you go - I have to defend them in an interview.
```
