#!/usr/bin/env bash
# Start both backend and frontend in dev mode.
#   - Flask on :5174 (with reload)
#   - Vite on :5173 (with HMR), proxies /api/* to Flask
#
# Open http://localhost:5173 in the browser.
# Ctrl-C stops both cleanly.

set -euo pipefail

# Run everything from the repo root so PYTHONPATH works without surgery.
BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Pick Python. Prefer the local venv in ui/.venv.
if [ -x "$BENCH_DIR/.venv/bin/python" ]; then
    PY="$BENCH_DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "[dev] no python interpreter found" >&2
    exit 1
fi

# Ensure frontend deps are present.
if [ ! -d "$BENCH_DIR/frontend/node_modules" ]; then
    echo "[dev] installing frontend deps..." >&2
    (cd "$BENCH_DIR/frontend" && npm install)
fi

echo "[dev] backend: $PY -m ui.backend.app (port 5174)"
echo "[dev] frontend: vite dev (port 5173)"
echo "[dev] open http://localhost:5173"

# Backend. Log to /tmp/ui.backend.log so the user can tail.
PYTHONPATH="$REPO_ROOT" DEBUG=1 "$PY" -m ui.backend.app \
    >/tmp/ui.backend.log 2>&1 &
BE_PID=$!

# Give Flask a second to bind or fail fast so we can surface the error.
sleep 1
if ! kill -0 "$BE_PID" 2>/dev/null; then
    echo "[dev] backend failed to start. Log at /tmp/ui.backend.log:" >&2
    tail -n 30 /tmp/ui.backend.log >&2
    exit 1
fi
echo "[dev] backend pid=$BE_PID (log: tail -f /tmp/ui.backend.log)"

# Frontend runs in the foreground so Ctrl-C kills it immediately and
# the trap takes care of the backend.
trap "echo '[dev] shutting down'; kill $BE_PID 2>/dev/null || true; wait 2>/dev/null || true" EXIT INT TERM

(cd "$BENCH_DIR/frontend" && npm run dev)
