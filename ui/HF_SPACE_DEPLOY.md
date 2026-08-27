# Deploying ui to a Hugging Face Space

This is for the maintainer pushing the leaderboard to HF Spaces. End
users don't need to read this — they interact via the deployed Space URL.

## One-time setup

1. **Create the Space** on HuggingFace:
   - Go to huggingface.co → New Space
   - SDK: **Docker**
   - Hardware: **CPU basic** (free) for read-only leaderboard, or **GPU
     T4 small** (~$0.40/hr, sleeps when idle) for live submissions
   - Visibility: **Public** (so anyone can see the leaderboard)
   - Name suggestion: `jax-aht-benchmark`

2. **Add Space secrets** (Settings → Variables and secrets):
   - `BENCHMARK_UI_STORAGE` = `/data` (persistent storage path)
   - (optional) `WANDB_API_KEY` if you want eval runs to log to wandb

3. **Configure persistent storage** (Settings → Persistent storage):
   - Add `/data` as a persistent volume so leaderboard.json survives
     container restarts (Space restarts when idle on free tier)

## Pushing the code

The Space is a git repo. Two ways to push:

### Option A: HF as a remote (cleanest for ongoing updates)

```bash
# add HF Space as a remote on your local jax-aht checkout
git remote add hf-space https://huggingface.co/spaces/<your-username>/jax-aht-benchmark

# push the subset of files HF needs
git subtree push --prefix=ui hf-space main
# OR push everything (Space ignores files outside Dockerfile's COPY scope)
git push hf-space main
```

### Option B: Manual upload (one-shot)

```bash
git clone https://huggingface.co/spaces/<your-username>/jax-aht-benchmark
cd jax-aht-benchmark
cp -r ../jax-aht/ui ./
cp -r ../jax-aht/agents ./
cp -r ../jax-aht/common ./
cp -r ../jax-aht/envs ./
cp -r ../jax-aht/evaluation ./
cp -r ../jax-aht/marl ./
git add . && git commit -m "deploy ui"
git push
```

## Required files at Space root

The Dockerfile expects this layout (already set up by the COPY directives
in `ui/Dockerfile`):

```
.
├── README.md            ← copy of ui/README_HF_SPACE.md
├── Dockerfile           ← symlink or copy of ui/Dockerfile
├── ui/
│   ├── backend/
│   ├── frontend/
│   └── requirements.txt
├── agents/
├── common/
├── envs/
├── evaluation/
└── marl/
```

The Space will auto-build on push. First build takes ~5-10 min (jaxlib
wheel download). Subsequent builds use cached layers.

## Deployment modes

### Mode A: Read-only leaderboard (CPU, free)

- Set hardware to **CPU basic**
- The leaderboard displays precomputed entries from `leaderboard.json`
- Submit endpoint returns 503 with "submissions disabled in read-only mode"
- New entries are added by maintainers via the CLI (see
  `ui/scripts/submit_offline.py`)

### Mode B: Live submissions (GPU, paid)

- Set hardware to **GPU T4 small**
- Enable Space sleep when idle (Settings → Sleep time → 30 min)
- Submit endpoint runs live evaluation against held-out partners
- Costs scale with usage; expected $5-30/month for moderate traffic

To toggle modes, set `BENCHMARK_UI_LIVE_SUBMIT=true` (mode B) or `false`
(mode A) in Space secrets.

## Preloading the leaderboard

The Space ships with an empty `leaderboard.json`. To preload it with
your existing trained methods, run on a machine with GPU:

```bash
# evaluate every built-in ego on every env, write entries to local leaderboard
python -m ui.backend.preload \
    --envs hanabi,mini-hanabi,lbf,overcooked-v1 \
    --num-episodes 256

# the resulting ui/backend/storage/leaderboard.json
# can be committed to the Space repo or uploaded as Space data
```

(See `ui/backend/preload.py` — separate file that loops over
all built-in egos × envs.)

## Smoke test before going public

```bash
# build locally
docker build -t jax-aht-benchmark -f ui/Dockerfile .

# run locally
docker run --rm -p 7860:7860 -v $(pwd)/data:/data jax-aht-benchmark

# hit endpoints
curl http://localhost:7860/api/healthz
curl http://localhost:7860/api/envs
curl http://localhost:7860/api/hanabi/leaderboard
```

Once smoke test passes locally, push to HF and the Space auto-builds.
