# Unstitch — Deployment Strategy

> What to deploy where, why, and exactly what you need to have ready.
>
> **Short version:** the frontend is trivial (Vercel, GitHub-connected, ~3 min).
> The backend is the whole problem, and the answer is **Hugging Face Spaces with
> the Docker SDK** — because this app needs a *process*, an *ffmpeg binary*, and
> a *writable disk*, and almost no free host gives you all three.

---

## 1. Why the backend can't go where the frontend goes

This is the single most important deployment decision in the project, and the
one most likely to be asked about.

The API answers **`202 Accepted` in milliseconds and keeps working afterwards**.
A de-edit takes 20–60 seconds. That design is not a stylistic choice — it is what
lets a browser poll instead of holding a connection open for a minute.

A serverless function is **killed when its response is sent**. So:

| Requirement | Serverless (Vercel/Netlify/Lambda) | Container |
|---|---|---|
| Work outliving the HTTP response | ❌ process dies at response | ✅ |
| `ffmpeg` binary | ❌ not present; ships as a ~70MB layer against a 250MB cap | ✅ apt-get |
| Writable disk shared between requests | ❌ `/tmp` is per-invocation | ✅ |
| Long CPU-bound work | ❌ 10–60s wall-clock caps | ✅ |
| OpenCV + PySceneDetect + yt-dlp (~400MB) | ❌ over bundle limits | ✅ |

**Four independent blockers.** So: frontend on Vercel, backend in a container.

---

## 2. The recommendation

```
   ┌─────────────────────┐        ┌──────────────────────────────┐
   │  Vercel             │        │  Hugging Face Spaces         │
   │  (static SPA)       │───────►│  (Docker, port 7860)         │
   │  GitHub-connected   │  HTTPS │  FastAPI + ffmpeg + workers  │
   │  free              │  CORS  │  free · 16GB RAM · 2 vCPU     │
   └─────────────────────┘        └──────────────┬───────────────┘
                                                 │
                                          ephemeral disk
                                          work/{job_id}/…
                                          + TTL sweeper
```

### Why Hugging Face Spaces for the backend

| | HF Spaces (Docker) | Render free | Railway | Fly.io |
|---|---|---|---|---|
| RAM | **16 GB** | 512 MB | 512 MB | 256 MB free |
| vCPU | **2** | shared | shared | shared |
| Disk | **50 GB** ephemeral | small | small | small |
| Cold start | sleeps after ~48h idle | **spins down after 15 min** | — | — |
| Cost | **free, no card** | free tier | trial credit | card required |
| Docker control | **full** | full | full | full |

Render's 512 MB is genuinely marginal here — one 720p x264 encode plus OpenCV
plus a loaded Python process gets close, and Render's **15-minute spin-down**
means a reviewer opening your link waits ~50s for a cold boot before anything
happens. HF Spaces sleeps far less aggressively and gives 32× the RAM.

The honest counter-argument: HF Spaces is designed for ML demos, so using it as a
general API host is slightly off-label, and it has no custom domain on free tier.
Both are fine for an assignment. **Render Docker is the documented fallback** and
needs no code changes — same Dockerfile, same env vars.

---

## 3. What you need before you start

- [ ] **GitHub repo pushed** — currently 12 commits on `dev`, **nothing pushed yet**.
      Merge to `main` first; Vercel and HF both track a branch.
- [ ] **Hugging Face account** — free, no card. https://huggingface.co/join
- [ ] **`GEMINI_API_KEY`** — you have one. It goes in HF **Secrets**, never in git.
- [ ] **Vercel account** — already connected to your GitHub.

That's it. No database, no object store, no Redis, no card.

---

## 4. Backend → Hugging Face Spaces

### 4.1 Create the Space

1. https://huggingface.co/new-space
2. **Name:** `unstitch` · **SDK: Docker** (blank template) · **Hardware:** CPU basic (free) · **Public**

### 4.2 One file to add

HF reads config from YAML front-matter in a `README.md` at the **repo root of the
Space**. Since our backend lives in `backend/`, the simplest path is to push the
`backend/` directory as the Space repo.

Create `backend/README.md`:

```markdown
---
title: Unstitch API
emoji: ✂️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

Backend API for Unstitch. See https://github.com/2003sarthak/Unstitch
```

`app_port: 7860` must match the Dockerfile's `CMD`. It already does.

### 4.3 Push

```bash
# from the repo root
cd backend
git init                       # the Space is its own git repo
git remote add space https://huggingface.co/spaces/<user>/unstitch
git add . && git commit -m "Deploy Unstitch API"
git push space main
```

> **Simpler alternative if the nested repo is fiddly:** on the Space, add the
> GitHub repo as a remote and push a subtree:
> `git subtree push --prefix backend space main`

The build takes ~5–8 minutes (apt + OpenCV wheels dominate).

### 4.4 Set secrets

Space → **Settings → Variables and secrets**:

| Key | Type | Value |
|---|---|---|
| `GEMINI_API_KEY` | **Secret** | your key |
| `CORS_ORIGINS` | Variable | `https://<your-app>.vercel.app` |
| `VISION_PROVIDER` | Variable | `gemini` |
| `MAX_CONCURRENT_JOBS` | Variable | `2` |
| `JOB_TTL_MINUTES` | Variable | `60` |

Everything else has a working default in `config.py`. **`CORS_ORIGINS` is the one
you cannot forget** — get it wrong and the frontend fails in the browser only,
which looks like a backend bug.

### 4.5 Verify

```bash
curl https://<user>-unstitch.hf.space/api/health
```

Expect `"vision_provider": "gemini"`. If it says `"stub"`, the secret didn't
land — the health endpoint deliberately reports the *effective* provider so this
is visible from outside.

---

## 5. Frontend → Vercel

Genuinely 3 minutes since your GitHub is connected.

1. Vercel → **Add New → Project** → pick the repo
2. **Root Directory:** `frontend` ← the only setting that matters
3. Framework preset **Vite** (auto-detected); build `npm run build`; output `dist`
4. **Environment Variables:**

| Key | Value |
|---|---|
| `VITE_API_BASE_URL` | `https://<user>-unstitch.hf.space` |

5. Deploy.

> ⚠️ `VITE_*` variables are **baked into the bundle at build time**, not read at
> runtime. Changing this later requires a **redeploy**, not just a restart. And
> anything prefixed `VITE_` is publicly readable — never put a key there.

`frontend/.npmrc` pins the public npm registry, so Vercel resolves packages
correctly regardless of the private CodeArtifact registry configured on this dev
machine.

---

## 6. The order to do it in

There is a circular dependency — the backend needs the frontend's URL for CORS,
and the frontend needs the backend's URL. Break it like this:

1. **Deploy backend** with `CORS_ORIGINS=*` temporarily. Note its URL.
2. **Deploy frontend** with `VITE_API_BASE_URL` = that URL. Note its URL.
3. **Update backend** `CORS_ORIGINS` to the exact Vercel URL. Restart the Space.
4. **Test**, then remember Vercel gives every deployment a *unique* preview URL —
   only the production domain is stable, so pin CORS to that.

---

## 7. Things that will bite you

**Storage is ephemeral.** A Space restart wipes `work/` and the in-process job
store. Old links 404. Correct for a stateless demo; the production swap is S3/R2
behind the existing workspace abstraction. The TTL sweeper keeps disk bounded.

**Free-tier quota is ~5 videos/day.** Measured, not guessed:
`GenerateRequestsPerDayPerProjectPerModel-FreeTier, limit: 20`, and a video costs
4 requests. **This is the most likely thing to break a live demo.** Mitigations,
cheapest first:
- `VISION_BATCH_SIZE=12` → 2 requests/video → ~10 videos/day
- `MAX_FRAMES=12` → 2 requests/video, at some detection recall
- `VISION_PROVIDER=stub` → unlimited, no key, honest about being heuristic
- Record the demo video **while you still have quota**, and don't burn it testing

**Cold start.** First request after sleep takes ~30–60s. Hit `/api/health` a
minute before demoing.

**Concurrency.** 2 vCPUs; `MAX_CONCURRENT_JOBS=2` is right. Higher makes
everything slower without finishing anything sooner.

**HF runs as UID 1000.** The Dockerfile already creates that user and chowns
`/app`. Don't change `WORKDIR` without also fixing ownership.

**The Docker build is unverified.** Docker isn't installed on this machine, so
the image has never actually been built. Watch that first build closely — the
likely failure points are the apt line (`ffmpeg fonts-dejavu-core libgl1
libglib2.0-0`) and OpenCV wheels.

---

## 8. Pre-deploy checklist

```bash
# backend
cd backend
.venv/Scripts/python.exe -m pytest -q          # expect 274 passed
.venv/Scripts/python.exe -m ruff check app tests

# frontend
cd ../frontend
npm run build                                   # tsc -b && vite build

# nothing secret is tracked
cd ..
git ls-files | grep -E "\.env$"                 # must return NOTHING
```

- [ ] `backend/.env` and `frontend/.env` are gitignored (they are)
- [ ] `backend/README.md` with HF front-matter exists
- [ ] Branch merged to `main` and pushed
- [ ] `GEMINI_API_KEY` set as a **Secret**, not a Variable
- [ ] `CORS_ORIGINS` = exact production Vercel URL
- [ ] `VITE_API_BASE_URL` = exact Space URL, then **redeploy** the frontend

---

## 9. If deployment goes wrong on the day

**Fallback A — backend only.** The Space serves interactive docs at `/docs`. A
reviewer can drive the whole API from there. Not pretty, but complete.

**Fallback B — run it locally in the demo video.** Everything works locally; the
recording is the deliverable, and the README says how to reproduce it:

```bash
cd backend && .venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
cd frontend && npm run dev
```

**Fallback C — Render Docker.** Same Dockerfile, same env vars, no code changes.
Set `PORT` handling if needed; accept the 15-minute spin-down.

---

## 10. What production would actually need

Worth saying out loud — it shows you know this is a prototype:

| Concern | Now | Production |
|---|---|---|
| Job state | in-process dict | Redis (the `JobStore` port already exists) |
| Storage | ephemeral disk | S3/R2 + presigned URLs |
| Workers | asyncio pool in the API process | separate worker service, shared queue |
| Quota | one free key, fails the job | paid tier + per-user rate limiting + stub fallback |
| Auth | none | API keys / sessions; jobs are currently readable by anyone with the id |
| Observability | stdout logs | structured logs, traces, per-stage timing |
| Scale | one container | horizontal API + worker autoscaling on queue depth |

Each of those is a swap behind a port that already exists, which was the point of
the architecture.
