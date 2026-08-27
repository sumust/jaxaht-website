"""File-backed storage. JSON on disk for structured data, raw files
for checkpoints. Good enough for local dev and small-scale hosted
deployments. Postgres impl can replace this without touching routes.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
import zipfile
from pathlib import Path

from .base import (
    Backends,
    CheckpointStore,
    JobStore,
    LeaderboardStore,
    SessionStore,
    TrajectoryStore,
)


def _atomic_write_json(path: Path, data) -> None:
    """Write JSON with a temp file + rename so a crash mid-write
    can't corrupt the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(path)


class FileLeaderboardStore(LeaderboardStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, env: str, version: str) -> Path:
        return self.root / f"{env}_{version}.json"

    def _read(self, env: str, version: str) -> list[dict]:
        p = self._path(env, version)
        if not p.exists():
            return []
        return json.loads(p.read_text())

    def add_entry(self, env: str, version: str, entry: dict) -> str:
        entry_id = entry.get("id") or uuid.uuid4().hex[:12]
        entry["id"] = entry_id
        entry.setdefault("created_at", time.time())
        with self._lock:
            entries = self._read(env, version)
            entries.append(entry)
            entries.sort(key=lambda e: -e.get("aggregate_score", 0))
            _atomic_write_json(self._path(env, version), entries)
        return entry_id

    def list_entries(self, env: str, version: str) -> list[dict]:
        return self._read(env, version)

    def get_entry(self, env: str, version: str, entry_id: str) -> dict | None:
        for e in self._read(env, version):
            if e.get("id") == entry_id:
                return e
        return None

    def clear(self, env: str, version: str) -> None:
        with self._lock:
            p = self._path(env, version)
            if p.exists():
                p.unlink()


class FileTrajectoryStore(TrajectoryStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def save(self, env: str, trajectory: dict) -> str:
        traj_id = uuid.uuid4().hex[:12]
        trajectory = {
            **trajectory,
            "id": traj_id,
            "env": env,
            "created_at": time.time(),
        }
        # Folder split (mirrors Johnny's collected_data_warmup{_source} vs
        # collected_data{_source} convention): keep warmup vs real apart and
        # tag the data source so anonymous + prolific runs don't get mixed.
        is_warmup = bool(trajectory.get("is_warmup"))
        source = ""
        prolific = trajectory.get("prolific")
        if isinstance(prolific, dict):
            source = prolific.get("data_source") or ""
        suffix = f"_{source}" if source else ""
        folder_kind = "warmup" if is_warmup else "real"
        target_dir = self.root / env / f"{folder_kind}{suffix}"
        with self._lock:
            _atomic_write_json(target_dir / f"{traj_id}.json", trajectory)
        return traj_id

    def load(self, trajectory_id: str) -> dict | None:
        for env_dir in self.root.iterdir() if self.root.exists() else []:
            p = env_dir / f"{trajectory_id}.json"
            if p.exists():
                return json.loads(p.read_text())
        return None

    def list(self, env: str | None = None, limit: int = 50) -> list[dict]:
        out = []
        if env:
            env_dirs = [self.root / env] if (self.root / env).exists() else []
        else:
            env_dirs = [d for d in (self.root.iterdir() if self.root.exists() else []) if d.is_dir()]
        for env_dir in env_dirs:
            for p in env_dir.glob("*.json"):
                data = json.loads(p.read_text())
                out.append({
                    "id": data["id"],
                    "env": data["env"],
                    "created_at": data.get("created_at"),
                    "score": data.get("score"),
                })
        out.sort(key=lambda e: -(e.get("created_at") or 0))
        return out[:limit]


class FileCheckpointStore(CheckpointStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def save_upload(self, zip_bytes: bytes, metadata: dict | None = None) -> str:
        ckpt_id = uuid.uuid4().hex[:12]
        sha = hashlib.sha256(zip_bytes).hexdigest()
        ckpt_dir = self.root / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        zip_path = ckpt_dir / "upload.zip"
        zip_path.write_bytes(zip_bytes)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(ckpt_dir / "extracted")
        meta = {
            "id": ckpt_id,
            "sha256": sha,
            "size_bytes": len(zip_bytes),
            "uploaded_at": time.time(),
            **(metadata or {}),
        }
        _atomic_write_json(ckpt_dir / "meta.json", meta)
        return ckpt_id

    def get_path(self, checkpoint_id: str) -> str | None:
        extracted = self.root / checkpoint_id / "extracted"
        return str(extracted) if extracted.exists() else None

    def get_sha256(self, checkpoint_id: str) -> str | None:
        meta_path = self.root / checkpoint_id / "meta.json"
        if not meta_path.exists():
            return None
        return json.loads(meta_path.read_text()).get("sha256")


class FileSessionStore(SessionStore):
    """In-memory with disk spill for crash recovery. Sessions hold
    live jax state which can't be pickled cleanly, so the on-disk copy
    stores only metadata; the env + state live in _live."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()
        self._live: dict[str, dict] = {}

    def create(self, env: str, config: dict) -> str:
        sid = uuid.uuid4().hex[:12]
        session = {
            "id": sid,
            "env": env,
            "config": config,
            "created_at": time.time(),
            "last_seen_at": time.time(),
        }
        with self._lock:
            self._live[sid] = session
        return sid

    def get(self, session_id: str) -> dict | None:
        with self._lock:
            s = self._live.get(session_id)
            if s:
                s["last_seen_at"] = time.time()
            return s

    def update(self, session_id: str, patch: dict) -> None:
        with self._lock:
            if session_id in self._live:
                self._live[session_id].update(patch)
                self._live[session_id]["last_seen_at"] = time.time()

    def delete(self, session_id: str) -> None:
        with self._lock:
            self._live.pop(session_id, None)

    def purge_older_than(self, seconds: int) -> int:
        cutoff = time.time() - seconds
        with self._lock:
            stale = [sid for sid, s in self._live.items() if s["last_seen_at"] < cutoff]
            for sid in stale:
                self._live.pop(sid, None)
            return len(stale)


class FileJobStore(JobStore):
    def __init__(self, root: Path):
        self.root = Path(root)
        self._lock = threading.Lock()

    def _path(self, job_id: str) -> Path:
        return self.root / f"{job_id}.json"

    def create(self, job: dict) -> str:
        job_id = job.get("id") or uuid.uuid4().hex[:12]
        job = {
            **job,
            "id": job_id,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        with self._lock:
            _atomic_write_json(self._path(job_id), job)
        return job_id

    def get(self, job_id: str) -> dict | None:
        p = self._path(job_id)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def update(self, job_id: str, patch: dict) -> None:
        with self._lock:
            current = self.get(job_id) or {}
            if not current:
                raise KeyError(f"job {job_id} does not exist")
            current.update(patch)
            current["updated_at"] = time.time()
            _atomic_write_json(self._path(job_id), current)

    def list(self, env: str | None = None, limit: int = 50) -> list[dict]:
        if not self.root.exists():
            return []
        out = []
        for p in self.root.glob("*.json"):
            try:
                job = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if env and job.get("env") != env:
                continue
            out.append({
                "id": job["id"],
                "env": job.get("env"),
                "status": job.get("status"),
                "created_at": job.get("created_at"),
            })
        out.sort(key=lambda e: -(e.get("created_at") or 0))
        return out[:limit]


def build_backends(root: str = "./ui/data") -> Backends:
    """File-backed backends rooted at ``root``."""
    root_path = Path(root)
    return Backends(
        leaderboard=FileLeaderboardStore(root_path / "leaderboards"),
        trajectory=FileTrajectoryStore(root_path / "trajectories"),
        checkpoint=FileCheckpointStore(root_path / "checkpoints"),
        session=FileSessionStore(root_path / "sessions"),
        job=FileJobStore(root_path / "jobs"),
    )
