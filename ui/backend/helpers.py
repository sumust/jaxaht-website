"""Small shared helpers used by both app.py and sessions.py, parked
here to avoid an import cycle.
"""
from __future__ import annotations

from typing import Any


def current_player(state, default_idx: int = 0) -> int:
    """Best-effort current-player extractor. Unwraps LogWrapper / WrappedEnvState
    if present, since cur_player_idx lives on the inner env state."""
    import numpy as np

    s = state
    for _ in range(4):
        if hasattr(s, "current_player") or hasattr(s, "cur_player_idx"):
            break
        inner = getattr(s, "env_state", None)
        if inner is None or inner is s:
            break
        s = inner

    cp = getattr(s, "current_player", None)
    if isinstance(cp, (int, np.integer)):
        return int(cp)
    if cp is not None:
        try:
            return int(np.asarray(cp).argmax())
        except Exception:
            pass

    cp = getattr(s, "cur_player_idx", None)
    if cp is not None:
        try:
            arr = np.asarray(cp).flatten()
            if arr.size > 0:
                return int(arr.argmax())
        except Exception:
            pass

    return default_idx


def scrub(info: dict[str, Any] | None) -> dict[str, Any]:
    """Make ``info`` JSON-serializable by converting numpy/jax scalars
    to Python primitives."""
    if not info:
        return {}
    out: dict[str, Any] = {}
    for k, v in info.items():
        try:
            if hasattr(v, "item"):
                out[k] = v.item()
            elif hasattr(v, "tolist"):
                out[k] = v.tolist()
            else:
                out[k] = v
        except Exception:
            out[k] = str(v)
    return out


def next_rng():
    """Return a fresh jax PRNGKey if jax is available, else a stdlib
    Random. Each env adapter decides what it needs."""
    try:
        import jax  # noqa: WPS433
        import time
        return jax.random.PRNGKey(int(time.time() * 1000) & 0xFFFFFFFF)
    except ImportError:
        import random
        return random.Random()
