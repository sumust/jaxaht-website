"""Preload the leaderboard with built-in egos for every supported env.

Run on a machine with GPU:

    python -m ui.backend.preload \\
        --envs hanabi mini-hanabi lbf overcooked-v1 \\
        --num-episodes 256 \\
        --version v1

Each (env, builtin_ego) pair becomes one leaderboard entry. Entries
already in storage with the same (agent_name, version) tuple are
skipped unless --force is passed.

Useful for shipping a populated leaderboard with the HF Space — so
viewers don't see an empty table on launch.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .envs import get as get_env, registry as env_registry
from .eval import evaluate_full_suite
from .storage import build_backends

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="preload leaderboard with built-in egos")
    parser.add_argument(
        "--envs", nargs="+",
        default=["hanabi", "mini-hanabi", "lbf", "overcooked-v1"],
        help="env names to preload (must be registered)",
    )
    parser.add_argument(
        "--version", default="v1",
        help="held-out partner version to evaluate against",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=256,
        help="episodes per (ego, partner) pair",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="eval seed (deterministic across builtins)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing entries with same (env, version, agent_name)",
    )
    parser.add_argument(
        "--storage-dir", default=None,
        help="storage backend dir (default: BENCHMARK_UI_STORAGE env var or ./data)",
    )
    parser.add_argument(
        "--ego-keys", nargs="*", default=None,
        help="restrict to specific ego keys (default: all builtins for each env)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    backends = build_backends(args.storage_dir)
    leaderboard = backends.leaderboard

    available_envs = env_registry()
    missing = [e for e in args.envs if e not in available_envs]
    if missing:
        log.error("envs not registered: %s. available: %s", missing, sorted(available_envs))
        return 2

    total = 0
    skipped = 0
    failed = 0

    for env_name in args.envs:
        adapter = get_env(env_name)
        if not getattr(adapter, "READY", True):
            log.warning("env %s is not READY; skipping", env_name)
            continue

        builtins = []
        try:
            builtins = adapter.builtin_egos()
        except (AttributeError, NotImplementedError):
            log.warning("env %s has no builtin_egos; skipping", env_name)
            continue

        if args.ego_keys:
            builtins = [b for b in builtins if b.key in args.ego_keys]
            if not builtins:
                log.warning("no builtin egos matched --ego-keys for env %s", env_name)
                continue

        log.info("env=%s | %d builtin egos: %s",
                 env_name, len(builtins), [b.key for b in builtins])

        for ego_spec in builtins:
            agent_name = ego_spec.display_name

            existing = [
                e for e in leaderboard.list_entries(env_name, args.version)
                if e.get("agent_name") == agent_name
            ]
            if existing and not args.force:
                log.info("  skip %s (already in leaderboard, use --force to overwrite)", agent_name)
                skipped += 1
                continue

            log.info("  evaluating %s …", agent_name)
            try:
                ego_fn = ego_spec.load_fn()
                result = evaluate_full_suite(
                    env_name=env_name,
                    version=args.version,
                    ego_fn=ego_fn,
                    num_episodes=args.num_episodes,
                    seed=args.seed,
                    progress_cb=lambda done, total_n, key: log.info(
                        "    [%d/%d] %s done", done, total_n, key,
                    ),
                )
            except Exception as exc:
                log.error("  FAILED %s: %s", agent_name, exc, exc_info=True)
                failed += 1
                continue

            entry = {
                "env": env_name,
                "version": args.version,
                "agent_name": agent_name,
                "ego_kind": "builtin",
                "builtin_key": ego_spec.key,
                "num_episodes": args.num_episodes,
                "eval_seed": args.seed,
                "aggregate_score": result["aggregate"]["mean"],
                "aggregate": result["aggregate"],
                "per_partner": result["per_partner"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                "notes": f"preloaded baseline ({ego_spec.description})",
            }
            entry_id = leaderboard.add_entry(env_name, args.version, entry)
            log.info("  stored entry_id=%s score=%.4f",
                     entry_id, entry["aggregate_score"])
            total += 1

    log.info("preload complete: %d added, %d skipped, %d failed", total, skipped, failed)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
