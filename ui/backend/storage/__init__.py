"""Storage layer. Interface in ``base``; file impl in ``file``.

Set BENCHMARK_HF_REPO=<dataset_id> to mirror leaderboard + ckpts to HF
so they survive container rebuilds.
"""
import os
from pathlib import Path

from .base import (
    Backends,
    CheckpointStore,
    JobStore,
    LeaderboardStore,
    SessionStore,
    TrajectoryStore,
)
from .file import build_backends as _build_file


def build_backends() -> Backends:
    backend = os.environ.get("STORAGE", "file").lower()
    if backend != "file":
        if backend == "postgres":
            raise NotImplementedError("Postgres backend not implemented yet.")
        raise ValueError(f"Unknown STORAGE: {backend}")

    root = os.environ.get("DATA_ROOT", "./ui/data")
    hf_repo = os.environ.get("BENCHMARK_HF_REPO")

    if hf_repo:
        from .hf_mirror import bootstrap_from_hf, HFMirrorCheckpointStore, HFMirrorLeaderboardStore
        bootstrap_root = Path(root)
        bootstrap_from_hf(hf_repo, bootstrap_root)
        _migrate_bootstrap_paths(bootstrap_root)

    backends = _build_file(root)

    if hf_repo:
        backends = Backends(
            leaderboard=HFMirrorLeaderboardStore(backends.leaderboard, hf_repo),
            trajectory=backends.trajectory,
            checkpoint=HFMirrorCheckpointStore(backends.checkpoint, hf_repo),
            session=backends.session,
            job=backends.job,
        )
    return backends


def _migrate_bootstrap_paths(root: Path) -> None:
    # the HF dataset uses "leaderboard/" while FileLeaderboardStore reads from "leaderboards/"
    src = root / "leaderboard"
    dst = root / "leaderboards"
    if not src.exists():
        return
    dst.mkdir(exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            target = dst / f.name
            if not target.exists():
                f.replace(target)


__all__ = [
    "Backends",
    "CheckpointStore",
    "JobStore",
    "LeaderboardStore",
    "SessionStore",
    "TrajectoryStore",
    "build_backends",
]
