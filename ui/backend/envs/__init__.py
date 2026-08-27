"""Env registry.

Envs self-register by name. Routes look up adapters here and never
import env modules directly.

Adding a new env:
  1. Create envs/<name>.py with a class subclassing EnvRenderer
  2. Import it in ``_load_all()`` below (lazy, so Hanabi-on-a-no-jax
     machine still loads LBF only)
  3. Done.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import EnvRenderer, ModeSpec, PartnerSpec

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

# Populated by _load_all on first access.
_REGISTRY: dict[str, EnvRenderer] = {}
_LOADED = False


def _load_all() -> None:
    """Import all env adapters. Failures are logged but non-fatal so a
    missing optional dep (e.g. jax) doesn't kill the whole server."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    # Each registration wraps import in try/except so a broken or
    # missing dep takes down only that env.
    try:
        from .hanabi import HanabiRenderer  # noqa: WPS433
        _REGISTRY["hanabi"] = HanabiRenderer()
    except Exception as exc:
        log.warning("hanabi adapter unavailable: %s", exc)

    try:
        from .mini_hanabi import MiniHanabiRenderer  # noqa: WPS433
        _REGISTRY["mini-hanabi"] = MiniHanabiRenderer()
    except Exception as exc:
        log.warning("mini-hanabi adapter unavailable: %s", exc)

    try:
        from .lbf import LBFRenderer  # noqa: WPS433
        _REGISTRY["lbf"] = LBFRenderer()
    except Exception as exc:
        log.warning("lbf adapter unavailable: %s", exc)

    try:
        from .overcooked import OvercookedRenderer  # noqa: WPS433
        _REGISTRY["overcooked-v1"] = OvercookedRenderer()
    except Exception as exc:
        log.warning("overcooked adapter unavailable: %s", exc)

    try:
        from .mock import MockRenderer  # noqa: WPS433
        _REGISTRY["mock"] = MockRenderer()
    except Exception as exc:
        log.warning("mock adapter unavailable: %s", exc)


def registry() -> dict[str, EnvRenderer]:
    _load_all()
    return dict(_REGISTRY)


def get(env_name: str) -> EnvRenderer:
    _load_all()
    if env_name not in _REGISTRY:
        raise KeyError(
            f"unknown env '{env_name}' (registered: {list(_REGISTRY)})"
        )
    return _REGISTRY[env_name]


def list_env_names() -> list[str]:
    return sorted(registry().keys())


__all__ = [
    "EnvRenderer",
    "ModeSpec",
    "PartnerSpec",
    "registry",
    "get",
    "list_env_names",
]
