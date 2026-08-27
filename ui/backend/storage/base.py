"""Storage interface.

Flask routes call store methods; they never touch disk or DB directly.
Swap the implementation via STORAGE=file|postgres env var without
touching routes.
"""
from abc import ABC, abstractmethod
from typing import Any


class LeaderboardStore(ABC):
    @abstractmethod
    def add_entry(self, env: str, version: str, entry: dict) -> str:
        """Append an entry. Return the entry id."""

    @abstractmethod
    def list_entries(self, env: str, version: str) -> list[dict]:
        """Return entries sorted by aggregate_score descending."""

    @abstractmethod
    def get_entry(self, env: str, version: str, entry_id: str) -> dict | None:
        ...

    @abstractmethod
    def clear(self, env: str, version: str) -> None:
        ...


class TrajectoryStore(ABC):
    @abstractmethod
    def save(self, env: str, trajectory: dict) -> str:
        """Persist a trajectory. Return the trajectory id."""

    @abstractmethod
    def load(self, trajectory_id: str) -> dict | None:
        ...

    @abstractmethod
    def list(self, env: str | None = None, limit: int = 50) -> list[dict]:
        """Return summary metadata (id, env, timestamp, score)."""


class CheckpointStore(ABC):
    @abstractmethod
    def save_upload(self, zip_bytes: bytes, metadata: dict | None = None) -> str:
        """Persist an uploaded checkpoint zip. Return the checkpoint id."""

    @abstractmethod
    def get_path(self, checkpoint_id: str) -> str | None:
        """Return a filesystem path to the extracted checkpoint, or
        None if not found / not yet extracted."""

    @abstractmethod
    def get_sha256(self, checkpoint_id: str) -> str | None:
        ...


class SessionStore(ABC):
    @abstractmethod
    def create(self, env: str, config: dict) -> str:
        """Start a session. Return session id."""

    @abstractmethod
    def get(self, session_id: str) -> dict | None:
        ...

    @abstractmethod
    def update(self, session_id: str, patch: dict) -> None:
        ...

    @abstractmethod
    def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    def purge_older_than(self, seconds: int) -> int:
        """Clean up stale sessions. Return number purged."""


class JobStore(ABC):
    """Persist eval job state so the /submit/status endpoint can poll
    a completed or in-flight job. Jobs live as immutable rows: once a
    status is set it doesn't revert. Postgres version later."""

    @abstractmethod
    def create(self, job: dict) -> str:
        """Create a new pending job. Return the job id."""

    @abstractmethod
    def get(self, job_id: str) -> dict | None:
        ...

    @abstractmethod
    def update(self, job_id: str, patch: dict) -> None:
        """Merge ``patch`` into the job record. Used for progress +
        completion."""

    @abstractmethod
    def list(self, env: str | None = None, limit: int = 50) -> list[dict]:
        ...


class Backends:
    """Bundle of all five stores. ``build_backends()`` wires the right
    implementation per STORAGE env var.
    """
    def __init__(
        self,
        leaderboard: LeaderboardStore,
        trajectory: TrajectoryStore,
        checkpoint: CheckpointStore,
        session: SessionStore,
        job: JobStore,
    ):
        self.leaderboard = leaderboard
        self.trajectory = trajectory
        self.checkpoint = checkpoint
        self.session = session
        self.job = job
