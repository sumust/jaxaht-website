"""API request/response schemas. Pydantic models give us runtime
validation, IDE-autocomplete in the frontend codegen, and free
OpenAPI docs at /api/docs.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# --------- envs / partners ---------

class PartnerInfo(BaseModel):
    key: str
    display_name: str
    difficulty: str = ""
    description: str
    tags: list[str] = Field(default_factory=list)


class EnvInfo(BaseModel):
    env_name: str
    display_name: str
    modes: list[str]
    default_partner_key: str
    overview: str = ""
    stats: dict[str, str] = Field(default_factory=dict)
    accent: str = "blue"
    ready: bool = True
    num_partners: int = 0


class EnvsResponse(BaseModel):
    envs: list[EnvInfo]


class PartnersResponse(BaseModel):
    partners: list[PartnerInfo]


# --------- play mode ---------

class NewGameRequest(BaseModel):
    partner_key: str
    env_kwargs: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None


class NewGameResponse(BaseModel):
    session_id: str
    state: dict[str, Any]
    score: dict[str, Any]


class StepRequest(BaseModel):
    session_id: str
    action: dict[str, Any]


class StepResponse(BaseModel):
    state: dict[str, Any]
    score: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any] = Field(default_factory=dict)
    partner_acted: bool = False
    events: list[dict[str, Any]] = Field(default_factory=list)


class SaveTrajectoryRequest(BaseModel):
    session_id: str
    agent_name: str = "Anonymous"


class SaveTrajectoryResponse(BaseModel):
    trajectory_id: str


# --------- submit + leaderboard ---------

class HeldoutPartnerInfo(BaseModel):
    key: str
    display_name: str
    difficulty: str = ""
    description: str
    tags: list[str]
    normalize_bounds: tuple[float, float] | None = None


class HeldoutVersionInfo(BaseModel):
    env: str
    version: str
    partners: list[HeldoutPartnerInfo]


class SubmitRequest(BaseModel):
    agent_name: str = Field(default="Anonymous", max_length=64)
    version: str = "v1"
    num_episodes: int = Field(default=32, ge=1, le=500)
    eval_seed: int = 0
    ego_kind: str = "builtin"   # "builtin" or "upload"
    builtin_key: str | None = None
    notes: str | None = None
    actor_type: str | None = None
    arch_params: dict | None = None
    ckpt_key: str = "final_params"
    idx: int = 0


class SubmitResponse(BaseModel):
    job_id: str


class JobProgress(BaseModel):
    completed: int
    total: int
    current: str | None


class JobStatusResponse(BaseModel):
    id: str
    kind: str
    env: str
    status: str
    progress: JobProgress
    error: str | None = None
    created_at: float
    updated_at: float
    started_at: float | None = None
    finished_at: float | None = None


class PerPartnerScore(BaseModel):
    key: str
    display_name: str
    mean: float
    std: float
    ci_low: float
    ci_high: float
    normalized_mean: float
    normalized_ci_low: float
    normalized_ci_high: float
    n_episodes: int
    mean_steps: float


class AggregateScore(BaseModel):
    method: str
    mean: float
    ci_low: float
    ci_high: float
    num_partners: int


class ClientResultSubmission(BaseModel):
    agent_name: str = Field(default="Anonymous", max_length=64)
    version: str = "v1"
    eval_seed: int = 0
    num_episodes: int = Field(default=128, ge=1, le=10000)
    per_partner: list[PerPartnerScore]
    aggregate: AggregateScore
    checkpoint_sha256: str | None = None
    notes: str | None = None
    wall_clock_seconds: float = 0.0
    client_version: str = "unknown"


class LeaderboardEntry(BaseModel):
    id: str
    env: str
    version: str
    agent_name: str
    aggregate_score: float
    aggregate: AggregateScore
    per_partner: list[PerPartnerScore]
    ego_kind: str
    builtin_key: str | None = None
    checkpoint_sha256: str | None = None
    num_episodes: int
    eval_seed: int
    notes: str | None = None
    created_at: float
    wall_clock_seconds: float


class LeaderboardResponse(BaseModel):
    env: str
    version: str
    entries: list[LeaderboardEntry]


class BuiltinEgoInfo(BaseModel):
    key: str
    display_name: str
    description: str
    tags: list[str]


class BuiltinEgosResponse(BaseModel):
    egos: list[BuiltinEgoInfo]


# --------- error ---------

class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
