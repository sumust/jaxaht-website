"""Background job queue for long-running evaluations.

Model:
  - a job is a persisted record with status pending|running|done|error
  - the in-process ThreadPoolExecutor runs the work
  - progress is reported by mutating the record via JobStore.update()
  - clients poll /submit/status/<id> to see status + progress

On process restart, pending/running jobs are orphaned (their threads
were lost). We mark them "stale" on startup by scanning the job store
and flipping any running -> error so the UI can tell users to resubmit.

Postgres + Celery future: replace the ThreadPoolExecutor with a real
queue; JobStore interface is unchanged.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from .storage import JobStore

log = logging.getLogger(__name__)

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"


class JobManager:
    def __init__(self, store: JobStore, max_workers: int = 2):
        self.store = store
        # Keep the worker count low by default: jax eval holds a GIL
        # lease on compile, and we don't want the UI thread starved.
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="bench-eval")
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._mark_orphaned_on_startup()

    def _mark_orphaned_on_startup(self) -> None:
        """Any running-status jobs left from a previous process crash
        are no longer running. Mark them errored so polling clients
        don't hang forever."""
        for summary in self.store.list(limit=1000):
            job = self.store.get(summary["id"])
            if job and job.get("status") == STATUS_RUNNING:
                log.warning("job %s orphaned by restart, marking error", summary["id"])
                self.store.update(summary["id"], {
                    "status": STATUS_ERROR,
                    "error": "server restarted while job was running",
                })

    def submit(
        self,
        kind: str,
        env: str,
        params: dict,
        fn: Callable[[Callable[[int, int, str], None]], Any],
    ) -> str:
        """Register a job and schedule its execution.

        ``fn`` is a closure that accepts a ``progress_cb`` and returns
        the result dict. We wrap it to manage status updates.
        """
        job = {
            "kind": kind,
            "env": env,
            "params": params,
            "status": STATUS_PENDING,
            "progress": {"completed": 0, "total": 0, "current": None},
            "result": None,
            "error": None,
        }
        job_id = self.store.create(job)

        def progress_cb(completed: int, total: int, current: str) -> None:
            self.store.update(job_id, {
                "progress": {"completed": completed, "total": total, "current": current},
            })

        def wrapped() -> None:
            self.store.update(job_id, {"status": STATUS_RUNNING, "started_at": time.time()})
            try:
                result = fn(progress_cb)
                self.store.update(job_id, {
                    "status": STATUS_DONE,
                    "result": result,
                    "finished_at": time.time(),
                })
                log.info("job %s done (kind=%s env=%s)", job_id, kind, env)
            except Exception as exc:  # noqa: BLE001
                tb = traceback.format_exc()
                log.exception("job %s failed: %s", job_id, exc)
                self.store.update(job_id, {
                    "status": STATUS_ERROR,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": tb,
                    "finished_at": time.time(),
                })

        future = self.executor.submit(wrapped)
        with self._lock:
            self._futures[job_id] = future
        return job_id

    def get(self, job_id: str) -> dict | None:
        return self.store.get(job_id)

    def list(self, env: str | None = None, limit: int = 50) -> list[dict]:
        return self.store.list(env=env, limit=limit)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
