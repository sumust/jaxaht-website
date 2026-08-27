"""Full-suite evaluation pipeline.

Given an ego policy and an env, evaluate against every held-out
partner for that env+version, aggregate, return a structured result.

Generic over env: we drive everything through the EnvRenderer adapter
so mock, Hanabi, LBF, and future envs share one code path.

The returned payload is the leaderboard entry shape:
    {
      env, version, agent_name, checkpoint_sha256, eval_seed,
      num_episodes, per_partner: { key: { mean, std, ci_low, ci_high,
        n_episodes, normalized_mean, ... } },
      aggregate: { mean, ci_low, ci_high, method: "mean_normalized" },
    }
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .envs import EnvRenderer, get as get_env
from .helpers import current_player, next_rng
from .heldout_loader import HeldoutPartner, load_heldout_partners

log = logging.getLogger(__name__)

# Number of bootstrap resamples for CI on per-partner + aggregate scores.
BOOTSTRAP_N = 2000


EgoFn = Callable[[Any, Any, Any], int]
"""(obs, state, rng) -> action. Ego policies adapt to this shape."""


@dataclass
class EpisodeResult:
    ego_return: float
    steps: int


def run_episode(
    renderer: EnvRenderer,
    env,
    rng,
    ego_fn: EgoFn,
    partner_fn: EgoFn,
    max_steps: int = 500,
) -> EpisodeResult:
    """Single episode. Two agents: ego=human_idx, partner=1-human_idx.
    Whichever is current_player acts; if both or neither, we act for
    the human first by convention."""
    human_idx = renderer.HUMAN_AGENT_IDX
    partner_idx = 1 - human_idx

    obs, state = renderer.reset(env, rng)
    total = 0.0
    for step_i in range(max_steps):
        cp = current_player(state, human_idx)
        # Build action dict. Only the current player's action actually
        # matters; non-current gets a noop via the renderer's defaults.
        if cp == human_idx:
            action = int(ego_fn(obs, state, next_rng()))
            actions = {human_idx: action}
        elif cp == partner_idx:
            action = int(partner_fn(obs, state, next_rng()))
            actions = {partner_idx: action}
        else:
            # Unknown - fall back to human acting. Shouldn't happen on
            # well-behaved envs.
            action = int(ego_fn(obs, state, next_rng()))
            actions = {human_idx: action}

        obs, state, reward, done, _info = renderer.step(
            env, state, actions, next_rng(),
        )
        total += float(reward)
        if done:
            return EpisodeResult(ego_return=total, steps=step_i + 1)
    return EpisodeResult(ego_return=total, steps=max_steps)


def evaluate_ego_vs_partner(
    renderer: EnvRenderer,
    env,
    ego_fn: EgoFn,
    partner: HeldoutPartner,
    num_episodes: int,
    seed: int,
) -> dict[str, Any]:
    """Roll out ``num_episodes`` of ego-vs-partner, compute stats."""
    try:
        import jax  # noqa: WPS433
        rngs = jax.random.split(jax.random.PRNGKey(seed), num_episodes)
    except ImportError:
        import random
        r = random.Random(seed)
        rngs = [random.Random(r.random()) for _ in range(num_episodes)]

    partner_fn = partner.load_get_action()
    returns: list[float] = []
    steps: list[int] = []
    for ep_rng in rngs:
        result = run_episode(renderer, env, ep_rng, ego_fn, partner_fn)
        returns.append(result.ego_return)
        steps.append(result.steps)

    arr = np.asarray(returns, dtype=np.float64)
    lo, hi = partner.normalize_bounds or (0.0, 1.0)
    span = max(1e-9, hi - lo)
    normalized = (arr - lo) / span
    mean = float(arr.mean())
    std = float(arr.std())
    mean_norm = float(normalized.mean())
    ci_low, ci_high = _bootstrap_ci(arr)
    ci_low_n, ci_high_n = _bootstrap_ci(normalized)

    return {
        "key": partner.key,
        "display_name": partner.display_name,
        "mean": mean,
        "std": std,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "normalized_mean": mean_norm,
        "normalized_ci_low": ci_low_n,
        "normalized_ci_high": ci_high_n,
        "n_episodes": len(returns),
        "mean_steps": float(np.mean(steps)),
    }


def evaluate_full_suite(
    env_name: str,
    version: str,
    ego_fn: EgoFn,
    num_episodes: int = 32,
    seed: int = 0,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate ego against every held-out partner for (env, version).

    ``progress_cb(completed, total, current_partner_key)`` fires after
    each partner finishes, so the jobs layer can expose progress to
    polling clients.
    """
    renderer = get_env(env_name)
    env = renderer.make_env(renderer.DEFAULT_KWARGS)
    partners = load_heldout_partners(env_name, version)
    if not partners:
        raise ValueError(
            f"no held-out partners registered for {env_name}:{version}"
        )

    t0 = time.time()
    per_partner: list[dict[str, Any]] = []
    for i, partner in enumerate(partners):
        log.info("eval %s:%s vs %s (%d/%d)", env_name, version,
                 partner.key, i + 1, len(partners))
        per_partner.append(evaluate_ego_vs_partner(
            renderer, env, ego_fn, partner, num_episodes, seed + i,
        ))
        if progress_cb is not None:
            progress_cb(i + 1, len(partners), partner.key)

    # Aggregate: mean of per-partner normalized means.
    # CI on aggregate comes from bootstrapping the per-partner means.
    partner_means = np.asarray([p["normalized_mean"] for p in per_partner])
    agg_mean = float(partner_means.mean())
    agg_lo, agg_hi = _bootstrap_ci(partner_means)
    aggregate = {
        "method": "mean_normalized",
        "mean": agg_mean,
        "ci_low": agg_lo,
        "ci_high": agg_hi,
        "num_partners": len(partners),
    }

    return {
        "env": env_name,
        "version": version,
        "num_episodes": num_episodes,
        "eval_seed": seed,
        "per_partner": per_partner,
        "aggregate": aggregate,
        "wall_clock_seconds": time.time() - t0,
    }


def _bootstrap_ci(arr: np.ndarray, alpha: float = 0.05, n: int = BOOTSTRAP_N) -> tuple[float, float]:
    """Two-sided percentile-bootstrap CI. Falls back to min/max on
    tiny samples where bootstrap is degenerate."""
    if arr.size <= 1:
        return (float(arr.min(initial=0)), float(arr.max(initial=0)))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, arr.size, size=(n, arr.size))
    resampled_means = arr[idx].mean(axis=1)
    lo = float(np.percentile(resampled_means, 100 * alpha / 2))
    hi = float(np.percentile(resampled_means, 100 * (1 - alpha / 2)))
    return lo, hi


# ---------------- helpers that the jobs layer uses ----------------


def load_ego_from_checkpoint(env_name: str, checkpoint_path: str, overrides: dict | None = None) -> EgoFn:
    """Turn a disk path to an extracted orbax checkpoint into an ego
    callable. Routes through checkpoint_loader.build_ego_fn (which
    delegates to the adapter's load_ego_checkpoint if defined, or
    falls back to the generic MLP/S5/RNN dispatch via
    agents.initialize_agents).

    overrides keys: actor_type, arch_params, ckpt_key, idx.
    """
    from pathlib import Path
    from .checkpoint_loader import UploadedCheckpoint, build_ego_fn

    overrides = overrides or {}
    renderer = get_env(env_name)
    env = renderer.make_env(renderer.DEFAULT_KWARGS or {})

    saved_path = Path(checkpoint_path)
    if saved_path.name != "saved_train_run":
        candidate = saved_path / "saved_train_run"
        if candidate.is_dir():
            saved_path = candidate

    uploaded = UploadedCheckpoint(
        extracted_dir=saved_path.parent,
        saved_train_run_path=saved_path,
        actor_type=overrides.get("actor_type", "mlp"),
        arch_params=overrides.get("arch_params") or {},
        ckpt_key=overrides.get("ckpt_key", "final_params"),
        idx=int(overrides.get("idx", 0)),
    )
    return build_ego_fn(env_name=env_name, env=env, uploaded=uploaded)
