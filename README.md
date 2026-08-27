# jaxaht-website

Web app for the JaxAHT ad-hoc-teamwork benchmark: play co-op games with AI
partners, submit an ego policy, and browse the leaderboard.

Four environments: Hanabi (full + mini), Level-Based Foraging, Overcooked.

## what you can do

- **play** — pick an env and a partner, play a game in the browser
- **demo** — watch two policies play together
- **submit** — upload a checkpoint, get evaluated against the held-out partner suite
- **leaderboard** — see submissions ranked, download per-partner CSVs

## layout

- `ui/backend/` — Flask API (env adapters, eval, leaderboard storage)
- `ui/frontend/` — React app
- `agents/`, `envs/`, `common/`, `evaluation/` — the `jax-aht` runtime code the backend imports

## run locally

```
bash ui/dev.sh
```

## deploy

Deploys to a Hugging Face Space (Docker). See `ui/HF_SPACE_DEPLOY.md`.
