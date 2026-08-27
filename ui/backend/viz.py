"""Diagnostic plot generation for leaderboard entries.

Produces PNG plots from leaderboard entries:
  * per-partner bar chart with CIs (ego return per held-out partner)
  * normalized-score bar chart (same data, [0,1] BR-normalized scale)
  * cross-method comparison bar chart (one bar per method, sorted)
  * partner-coverage matrix (methods x partners) for an env/version

Plots are generated on demand by routes in app.py. Output is a PNG byte
buffer the route can stream back to the frontend.
"""
from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


_PALETTE = {
    "primary": "#3b82f6",
    "primary_dim": "#93c5fd",
    "good": "#10b981",
    "warn": "#f59e0b",
    "bad": "#ef4444",
    "muted": "#6b7280",
    "bg": "#ffffff",
    "grid": "#e5e7eb",
}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PALETTE["muted"])
    ax.spines["bottom"].set_color(_PALETTE["muted"])
    ax.tick_params(colors=_PALETTE["muted"], labelsize=9)
    ax.grid(axis="y", color=_PALETTE["grid"], linestyle="-", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)


def _to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=_PALETTE["bg"])
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def per_partner_bars(entry: dict[str, Any], normalized: bool = False) -> bytes:
    """Render per-partner score bars with CI whiskers for one entry."""
    per_partner = entry.get("per_partner", {})
    if not per_partner:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no per-partner data", ha="center", va="center", color=_PALETTE["muted"])
        ax.axis("off")
        return _to_png(fig)

    items = sorted(per_partner.items(), key=lambda kv: kv[1].get("mean", 0))
    labels = [k for k, _ in items]
    if normalized:
        means = [v.get("normalized_mean", 0) for _, v in items]
        lo = [v.get("normalized_ci_low", 0) for _, v in items]
        hi = [v.get("normalized_ci_high", 0) for _, v in items]
    else:
        means = [v.get("mean", 0) for _, v in items]
        lo = [v.get("ci_low", 0) for _, v in items]
        hi = [v.get("ci_high", 0) for _, v in items]

    err_lo = [m - lo_i for m, lo_i in zip(means, lo)]
    err_hi = [hi_i - m for m, hi_i in zip(means, hi)]

    fig, ax = plt.subplots(figsize=(max(6, 0.35 * len(labels) + 3), 4))
    bar_color = _PALETTE["primary"]
    ax.barh(
        range(len(labels)), means,
        xerr=[err_lo, err_hi], color=bar_color,
        ecolor=_PALETTE["muted"], capsize=3, edgecolor="none",
    )
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    xlabel = "normalized return (0=worst, 1=BR upper bound)" if normalized else "mean episode return"
    ax.set_xlabel(xlabel, fontsize=10)
    ego_name = entry.get("agent_name", "ego")
    ax.set_title(f"{ego_name} — per-partner returns", fontsize=11, color="black")
    if normalized:
        ax.set_xlim(0, max(1.05, max(hi) + 0.05))
        ax.axvline(1.0, color=_PALETTE["good"], linestyle="--", linewidth=1, alpha=0.6)
    _style(ax)
    return _to_png(fig)


def comparison_bars(entries: list[dict[str, Any]], metric: str = "aggregate_score") -> bytes:
    """One bar per method, sorted descending."""
    if not entries:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no entries", ha="center", va="center", color=_PALETTE["muted"])
        ax.axis("off")
        return _to_png(fig)

    items = []
    for e in entries:
        score = e.get(metric, e.get("aggregate", {}).get("mean", 0))
        agg = e.get("aggregate", {})
        items.append({
            "name": e.get("agent_name", "?"),
            "score": float(score),
            "ci_low": float(agg.get("ci_low", score)),
            "ci_high": float(agg.get("ci_high", score)),
        })
    items.sort(key=lambda x: -x["score"])

    labels = [i["name"] for i in items]
    means = [i["score"] for i in items]
    err_lo = [i["score"] - i["ci_low"] for i in items]
    err_hi = [i["ci_high"] - i["score"] for i in items]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels) + 3), 4))
    colors = []
    for s in means:
        if s >= 0.8:
            colors.append(_PALETTE["good"])
        elif s >= 0.5:
            colors.append(_PALETTE["primary"])
        elif s >= 0.2:
            colors.append(_PALETTE["warn"])
        else:
            colors.append(_PALETTE["bad"])
    ax.bar(
        range(len(labels)), means,
        yerr=[err_lo, err_hi], color=colors,
        ecolor=_PALETTE["muted"], capsize=3, edgecolor="none",
    )
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("aggregate normalized score", fontsize=10)
    ax.set_title("method comparison", fontsize=11, color="black")
    _style(ax)
    return _to_png(fig)


def coverage_heatmap(entries: list[dict[str, Any]]) -> bytes:
    """methods x partners heatmap of normalized scores. Empty cells = no eval."""
    if not entries:
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.text(0.5, 0.5, "no entries", ha="center", va="center", color=_PALETTE["muted"])
        ax.axis("off")
        return _to_png(fig)

    methods = [e.get("agent_name", f"#{i}") for i, e in enumerate(entries)]
    partner_keys: list[str] = []
    seen = set()
    for e in entries:
        for k in e.get("per_partner", {}):
            if k not in seen:
                partner_keys.append(k)
                seen.add(k)

    matrix = np.full((len(methods), len(partner_keys)), np.nan, dtype=np.float64)
    for i, e in enumerate(entries):
        per_partner = e.get("per_partner", {})
        for j, pk in enumerate(partner_keys):
            v = per_partner.get(pk, {}).get("normalized_mean")
            if v is not None:
                matrix[i, j] = float(v)

    fig, ax = plt.subplots(
        figsize=(max(6, 0.35 * len(partner_keys) + 3),
                 max(3, 0.35 * len(methods) + 1.5)),
    )
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(partner_keys)))
    ax.set_xticklabels(partner_keys, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized return", fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title("method × partner coverage", fontsize=11, color="black")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.3 <= v <= 0.7 else "white",
                        fontsize=7)
    _style(ax)
    return _to_png(fig)
