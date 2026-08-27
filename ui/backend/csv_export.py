"""CSV export for leaderboard entries.

Two flavors:
  * leaderboard_csv(entries) - one row per (method) with aggregate score
    + flattened per-partner columns. Suitable for paper tables.
  * detailed_csv(entry)      - one row per (method, partner) pair, with
    full statistics (mean, std, CI, n_episodes, normalized_mean).
    Suitable for plots and statistical work.
"""
from __future__ import annotations

import csv
import io
from typing import Any


def leaderboard_csv(entries: list[dict[str, Any]]) -> str:
    """One row per method, partner scores as columns."""
    if not entries:
        return ""

    partner_keys: list[str] = []
    seen = set()
    for e in entries:
        for k in e.get("per_partner", {}):
            if k not in seen:
                partner_keys.append(k)
                seen.add(k)
    partner_keys.sort()

    fieldnames = [
        "rank", "agent_name", "env", "version", "aggregate_score",
        "aggregate_ci_low", "aggregate_ci_high",
        "num_episodes", "eval_seed", "wall_clock_seconds", "notes",
    ] + [f"{k}__norm" for k in partner_keys] + [f"{k}__mean" for k in partner_keys]

    sorted_entries = sorted(
        entries, key=lambda e: e.get("aggregate_score", 0), reverse=True,
    )

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for rank, e in enumerate(sorted_entries, start=1):
        agg = e.get("aggregate", {})
        row = {
            "rank": rank,
            "agent_name": e.get("agent_name", ""),
            "env": e.get("env", ""),
            "version": e.get("version", ""),
            "aggregate_score": e.get("aggregate_score", agg.get("mean", "")),
            "aggregate_ci_low": agg.get("ci_low", ""),
            "aggregate_ci_high": agg.get("ci_high", ""),
            "num_episodes": e.get("num_episodes", ""),
            "eval_seed": e.get("eval_seed", ""),
            "wall_clock_seconds": e.get("wall_clock_seconds", ""),
            "notes": (e.get("notes") or "").replace("\n", " ").replace("\r", " "),
        }
        per_partner = e.get("per_partner", {})
        for pk in partner_keys:
            row[f"{pk}__norm"] = per_partner.get(pk, {}).get("normalized_mean", "")
            row[f"{pk}__mean"] = per_partner.get(pk, {}).get("mean", "")
        writer.writerow(row)
    return buf.getvalue()


def detailed_csv(entry: dict[str, Any]) -> str:
    """One row per (method, partner) pair, with full statistics."""
    fieldnames = [
        "agent_name", "env", "version", "partner_key", "partner_display_name",
        "mean", "std", "ci_low", "ci_high",
        "normalized_mean", "normalized_ci_low", "normalized_ci_high",
        "n_episodes", "mean_steps",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for pk, stats in (entry.get("per_partner") or {}).items():
        row = {
            "agent_name": entry.get("agent_name", ""),
            "env": entry.get("env", ""),
            "version": entry.get("version", ""),
            "partner_key": pk,
            "partner_display_name": stats.get("display_name", pk),
            "mean": stats.get("mean", ""),
            "std": stats.get("std", ""),
            "ci_low": stats.get("ci_low", ""),
            "ci_high": stats.get("ci_high", ""),
            "normalized_mean": stats.get("normalized_mean", ""),
            "normalized_ci_low": stats.get("normalized_ci_low", ""),
            "normalized_ci_high": stats.get("normalized_ci_high", ""),
            "n_episodes": stats.get("n_episodes", ""),
            "mean_steps": stats.get("mean_steps", ""),
        }
        writer.writerow(row)
    return buf.getvalue()


def comparison_markdown(entries_by_env: dict[str, list[dict[str, Any]]]) -> str:
    """Markdown table: rows = methods, columns = envs, cells = aggregate score."""
    if not entries_by_env:
        return "_no entries yet_\n"

    method_to_env_score: dict[str, dict[str, tuple[float, float, float]]] = {}
    for env, entries in entries_by_env.items():
        for e in entries:
            name = e.get("agent_name", "?")
            agg = e.get("aggregate", {})
            score = float(e.get("aggregate_score", agg.get("mean", 0)))
            ci_low = float(agg.get("ci_low", score))
            ci_high = float(agg.get("ci_high", score))
            method_to_env_score.setdefault(name, {})[env] = (score, ci_low, ci_high)

    envs = sorted(entries_by_env.keys())
    methods = sorted(method_to_env_score.keys(),
                     key=lambda m: -sum(s[0] for s in method_to_env_score[m].values()))

    lines = []
    lines.append("| method | " + " | ".join(envs) + " |")
    lines.append("|" + "---|" * (len(envs) + 1))
    for m in methods:
        row = [m]
        for env in envs:
            cell = method_to_env_score[m].get(env)
            if cell is None:
                row.append("—")
            else:
                score, lo, hi = cell
                row.append(f"{score:.3f} ({lo:.3f}–{hi:.3f})")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"
