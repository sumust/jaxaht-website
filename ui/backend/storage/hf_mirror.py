# HF Hub mirror for leaderboard + checkpoint stores.
# Local file backs writes; after each successful write we push to HF dataset.
# Bootstrap: snapshot_download pulls the dataset into the local root before serving.
from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .base import CheckpointStore, LeaderboardStore
from .file import FileCheckpointStore, FileLeaderboardStore

log = logging.getLogger("hf_mirror")


def bootstrap_from_hf(repo_id: str, local_root: Path) -> None:
    # Pull leaderboard JSON files only at boot (small, fast).
    # Checkpoints are lazy-loaded on demand by ensure_checkpoint_local() since
    # snapshot_download'ing all of them blows the container's network timeout.
    from huggingface_hub import snapshot_download
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_root),
            allow_patterns=["leaderboard/*"],
        )
        log.info("bootstrapped leaderboard from %s into %s", repo_id, local_root)
    except Exception as exc:
        log.warning("failed to bootstrap leaderboard from %s (%s); starting empty", repo_id, exc)


def ensure_checkpoint_local(repo_id: str, ckpt_id: str, local_root: Path) -> bool:
    # Pull one ckpt's files from HF dataset to local_root on demand.
    # Returns True if the ckpt now exists locally (either was there, or fetched).
    from huggingface_hub import snapshot_download
    target = Path(local_root) / "checkpoints" / ckpt_id
    if target.exists() and any(target.iterdir()):
        return True
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(local_root),
            allow_patterns=[f"checkpoints/{ckpt_id}/**"],
        )
        log.info("lazy-fetched ckpt %s from HF", ckpt_id)
        return target.exists()
    except Exception as exc:
        log.warning("failed to lazy-fetch ckpt %s (%s)", ckpt_id, exc)
        return False


class HFMirrorLeaderboardStore(LeaderboardStore):
    def __init__(self, inner: FileLeaderboardStore, repo_id: str):
        self.inner = inner
        self.repo_id = repo_id
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hf-push")
        self._lock = threading.Lock()

    def _push(self, env: str, version: str) -> None:
        from huggingface_hub import HfApi, CommitOperationAdd
        local = self.inner._path(env, version)
        if not local.exists():
            return
        rel = f"leaderboard/{env}_{version}.json"
        try:
            HfApi().create_commit(
                repo_id=self.repo_id, repo_type="dataset",
                operations=[CommitOperationAdd(path_in_repo=rel, path_or_fileobj=str(local))],
                commit_message=f"update {rel}",
            )
            log.info("pushed %s to HF", rel)
        except Exception as exc:
            log.exception("HF push failed for %s: %s", rel, exc)

    def add_entry(self, env: str, version: str, entry: dict) -> str:
        entry_id = self.inner.add_entry(env, version, entry)
        # fire-and-forget push so the API response isn't blocked
        self._executor.submit(self._push, env, version)
        return entry_id

    def list_entries(self, env: str, version: str) -> list[dict]:
        return self.inner.list_entries(env, version)

    def get_entry(self, env: str, version: str, entry_id: str):
        return self.inner.get_entry(env, version, entry_id)

    def clear(self, env: str, version: str) -> None:
        self.inner.clear(env, version)
        self._executor.submit(self._push, env, version)


class HFMirrorCheckpointStore(CheckpointStore):
    # mirrors user-uploaded ckpts to HF so they survive container restarts.
    # the local FileCheckpointStore stays the source of truth for in-process reads.
    def __init__(self, inner: FileCheckpointStore, repo_id: str):
        self.inner = inner
        self.repo_id = repo_id
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hf-ckpt-push")

    def _push(self, ckpt_id: str) -> None:
        # Push the whole ckpt dir (zip + extracted + meta) so a future bootstrap can rehydrate it fully.
        from huggingface_hub import HfApi
        ckpt_dir = self.inner.root / ckpt_id
        if not ckpt_dir.exists():
            return
        try:
            HfApi().upload_folder(
                folder_path=str(ckpt_dir),
                path_in_repo=f"checkpoints/{ckpt_id}",
                repo_id=self.repo_id,
                repo_type="dataset",
                commit_message=f"upload ckpt {ckpt_id}",
            )
            log.info("pushed ckpt %s to HF", ckpt_id)
        except Exception as exc:
            log.exception("HF ckpt push failed for %s: %s", ckpt_id, exc)

    def save_upload(self, zip_bytes: bytes, metadata: dict | None = None) -> str:
        ckpt_id = self.inner.save_upload(zip_bytes, metadata)
        self._executor.submit(self._push, ckpt_id)
        return ckpt_id

    def get_path(self, ckpt_id: str):
        return self.inner.get_path(ckpt_id)

    def get_sha256(self, ckpt_id: str):
        return self.inner.get_sha256(ckpt_id)
