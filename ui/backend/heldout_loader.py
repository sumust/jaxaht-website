"""Held-out partner catalog.

Reads ``evaluation/configs/global_heldout_settings.yaml`` as the source
of truth so the ui leaderboard and the training pipeline
agree on which agents count as "held-out". Versioning is per (env,
version_tag) tuples.

For envs whose adapters are already registered (Hanabi, LBF, Mock),
this module maps yaml partner entries to callables by delegating to
``EnvRenderer.available_partners()``. That keeps one source of truth
for partner construction (the env adapter) and lets the yaml add
bounds / inclusion-per-version metadata on top.

Version discovery:
  Today the yaml is unversioned. We treat each env's block as
  version "v1" by convention. When Caroline adds an explicit ``version:``
  field we'll honor it here.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .envs import get as get_env, registry as env_registry

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HELDOUT_YAML = _REPO_ROOT / "evaluation" / "configs" / "global_heldout_settings.yaml"


@dataclass
class HeldoutPartner:
    """One held-out partner as far as the leaderboard cares.

    ``load_get_action`` returns the ``(obs, state, rng) -> action``
    callable the eval loop needs. We delegate to the env adapter's
    partner spec when possible so we aren't duplicating construction
    logic (SmartBot card_counts, OBL weight paths, etc.).
    """
    env_name: str
    version: str
    key: str
    display_name: str
    difficulty: str
    description: str
    normalize_bounds: tuple[float, float] | None
    tags: list[str] = field(default_factory=list)
    _load_get_action: Callable[[], Callable] | None = field(default=None, repr=False)

    def load_get_action(self):
        if self._load_get_action is None:
            raise RuntimeError(
                f"held-out partner {self.env_name}:{self.key} has no "
                "construction hook; env adapter must register it."
            )
        adapter = self._load_get_action()
        # Adapter returns a "partner wrapper" object with .get_action().
        # We want a plain (obs, state, rng) -> int callable.
        if hasattr(adapter, "get_action"):
            return adapter.get_action
        if callable(adapter):
            return adapter
        raise TypeError(f"partner {self.key} adapter has no get_action")


# Cache: parsed yaml + computed HeldoutPartner lists.
_CACHE: dict[tuple[str, str], list[HeldoutPartner]] = {}


def _parse_yaml() -> dict:
    """Parse the yaml once on first use. We don't import pyyaml at
    module load time in case it's absent on a minimal install."""
    try:
        import yaml  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyyaml required to read global_heldout_settings.yaml; "
            "pip install pyyaml"
        ) from exc
    if not _HELDOUT_YAML.exists():
        log.warning("held-out yaml not found at %s", _HELDOUT_YAML)
        return {}
    return yaml.safe_load(_HELDOUT_YAML.read_text()) or {}


def _env_block(env_name: str) -> dict:
    data = _parse_yaml()
    heldout_set = data.get("heldout_set") or {}
    return heldout_set.get(env_name) or {}


def _normalize_bounds(entry: dict) -> tuple[float, float] | None:
    """Best-effort: pick the first (lo, hi) pair we can find under
    performance_bounds, preferring returned_episode_returns."""
    pb = entry.get("performance_bounds") or {}
    for key in ("returned_episode_returns", "base_return", "percent_eaten"):
        if key not in pb:
            continue
        val = pb[key]
        # Could be [lo, hi] or list-of-pairs for ego-multi-seed envs.
        if isinstance(val, list) and val:
            first = val[0]
            if isinstance(first, list) and len(first) == 2:
                return float(first[0]), float(first[1])
            if len(val) == 2 and not isinstance(first, list):
                return float(val[0]), float(val[1])
    return None


def load_heldout_partners(env_name: str, version: str = "v1") -> list[HeldoutPartner]:
    """Build the held-out catalog for (env, version). Cached on first
    call per (env, version)."""
    cache_key = (env_name, version)
    if cache_key in _CACHE:
        return _CACHE[cache_key]

    registry = env_registry()
    adapter = registry.get(env_name)
    if adapter is None:
        raise KeyError(f"env '{env_name}' has no registered adapter")

    # Build a lookup of adapter-provided PartnerSpecs by key.
    spec_by_key = {p.key: p for p in adapter.available_partners()}

    # Read yaml for bounds + any version-gate metadata.
    yaml_partners = _env_block(env_name)

    partners: list[HeldoutPartner] = []
    if yaml_partners:
        # For each yaml entry whose key matches an adapter partner,
        # create a HeldoutPartner. Entries without adapter support are
        # skipped with a log (e.g. "ippo_seed123" type RL-fixture entries
        # that the UI play-mode catalog doesn't expose).
        for yaml_key, yaml_entry in yaml_partners.items():
            if not isinstance(yaml_entry, dict):
                continue
            # Pick a matching adapter partner. For keys we've aligned
            # (iggi, piers, smartbot, etc.) the yaml key == adapter key.
            # For others we skip.
            spec = spec_by_key.get(yaml_key)
            if spec is None:
                log.debug("skipping %s:%s: no adapter partner", env_name, yaml_key)
                continue
            partners.append(HeldoutPartner(
                env_name=env_name,
                version=version,
                key=spec.key,
                display_name=spec.display_name,
                difficulty=spec.difficulty,
                description=spec.description,
                normalize_bounds=_normalize_bounds(yaml_entry),
                tags=list(spec.tags),
                _load_get_action=spec.load_fn,
            ))

    # If the yaml is empty for this env (e.g. mock, or a fresh env with
    # no yaml block), fall back to every partner the adapter exposes.
    if not partners:
        for spec in adapter.available_partners():
            partners.append(HeldoutPartner(
                env_name=env_name,
                version=version,
                key=spec.key,
                display_name=spec.display_name,
                difficulty=spec.difficulty,
                description=spec.description,
                normalize_bounds=None,
                tags=list(spec.tags),
                _load_get_action=spec.load_fn,
            ))

    _CACHE[cache_key] = partners
    return partners


def list_versions(env_name: str) -> list[str]:
    """Placeholder until the yaml has per-env version tags. Every env
    has one version ("v1") today."""
    try:
        get_env(env_name)
    except KeyError:
        return []
    return ["v1"]


def invalidate_cache() -> None:
    """Test-only hook so fixtures can reload the yaml."""
    _CACHE.clear()
