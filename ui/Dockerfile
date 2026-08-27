FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    GUNICORN_WORKERS=2 \
    GUNICORN_TIMEOUT=180 \
    PORT=7860 \
    BENCHMARK_UI_STORAGE=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential git \
    nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ui/requirements.txt /app/ui/requirements.txt
RUN pip install -r /app/ui/requirements.txt

# Build the frontend in-container so HF Space deploys don't depend on a
# pre-built dist/ committed to git (dist/ is gitignored).
COPY ui/frontend/package*.json /app/ui/frontend/
RUN cd /app/ui/frontend && npm ci
COPY ui/frontend /app/ui/frontend
RUN cd /app/ui/frontend && npm run build

COPY ui/backend /app/ui/backend
COPY ui/__init__.py /app/ui/__init__.py
COPY agents/ /app/agents/
COPY common/ /app/common/
COPY envs/ /app/envs/
COPY evaluation/ /app/evaluation/
COPY marl/ /app/marl/

RUN mkdir -p /data && chmod 777 /data

EXPOSE 7860

CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:7860", "--timeout", "180", \
     "ui.backend.app:create_app()"]
