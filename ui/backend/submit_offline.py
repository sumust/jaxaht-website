"""Offline submission for the read-only leaderboard mode.

When the public HF Space runs in read-only mode (CPU tier, no live eval),
new methods are added by maintainers running this CLI on a GPU machine
and committing the resulting leaderboard.json back to the Space.

Usage:

    # built-in ego (random/self-play/etc.)
    python -m ui.backend.submit_offline \\
        --env hanabi --builtin smartbot --num-episodes 256

    # custom ego from a checkpoint path (placeholder — wire up your loader)
    python -m ui.backend.submit_offline \\
        --env hanabi --ckpt-path results/hanabi/meliba_ego/.../saved_train_run \\
        --agent-name "MeLIBA-ego (1e9)" --num-episodes 256
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .envs import get as get_env
from .eval import evaluate_full_suite
from .storage import build_backends

log = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="offline leaderboard submission")
    parser.add_argument("--env", required=True)
    parser.add_argument("--version", default="v1")
    parser.add_argument("--num-episodes", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--storage-dir", default=None)
    parser.add_argument("--builtin", help="key of a built-in ego (mutually exclusive with --ckpt-path)")
    parser.add_argument("--ckpt-path", help="filesystem path to a trained ego checkpoint")
    parser.add_argument("--agent-name", help="display name (required when using --ckpt-path)")
    parser.add_argument("--notes", default="", help="freeform notes for the entry")
    parser.add_argument("--dry-run", action="store_true",
                        help="evaluate but don't write to leaderboard")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.builtin and args.ckpt_path:
        log.error("specify either --builtin or --ckpt-path, not both")
        return 2

    adapter = get_env(args.env)

    if args.builtin:
        builtins = {b.key: b for b in adapter.builtin_egos()}
        if args.builtin not in builtins:
            log.error("unknown builtin '%s' for env %s. known: %s",
                      args.builtin, args.env, sorted(builtins))
            return 2
        ego_spec = builtins[args.builtin]
        ego_fn = ego_spec.load_fn()
        agent_name = args.agent_name or ego_spec.display_name
        ego_kind = "builtin"
        builtin_key = ego_spec.key
    elif args.ckpt_path:
        if not args.agent_name:
            log.error("--agent-name is required with --ckpt-path")
            return 2
        log.error("ckpt-path loading is a stub — wire up your custom loader here")
        return 2
    else:
        log.error("specify --builtin or --ckpt-path")
        return 2

    log.info("evaluating %s on %s ...", agent_name, args.env)
    result = evaluate_full_suite(
        env_name=args.env,
        version=args.version,
        ego_fn=ego_fn,
        num_episodes=args.num_episodes,
        seed=args.seed,
    )
    log.info("aggregate score: %.4f (CI %.4f - %.4f)",
             result["aggregate"]["mean"],
             result["aggregate"].get("ci_low", 0),
             result["aggregate"].get("ci_high", 0))

    entry = {
        "env": args.env,
        "version": args.version,
        "agent_name": agent_name,
        "ego_kind": ego_kind,
        "builtin_key": builtin_key,
        "num_episodes": args.num_episodes,
        "eval_seed": args.seed,
        "aggregate_score": result["aggregate"]["mean"],
        "aggregate": result["aggregate"],
        "per_partner": result["per_partner"],
        "wall_clock_seconds": result["wall_clock_seconds"],
        "notes": args.notes,
    }

    if args.dry_run:
        print(json.dumps(entry, indent=2, default=str))
        return 0

    backends = build_backends(args.storage_dir)
    entry_id = backends.leaderboard.add_entry(args.env, args.version, entry)
    log.info("entry_id=%s stored", entry_id)
    return 0


if __name__ == "__main__":
    sys.exit(main())
