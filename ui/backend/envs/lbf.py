"""LBF (Level-Based Foraging) adapter.

Wraps the same LBFWrapper env Johnny's ``human_data/app.py`` drives,
reusing its heuristic agents: Random, SequentialFruit (six orderings),
GreedyHeuristic (three same-level variants), and Entitled. Follows the
deferred-import pattern (jax/jaxmarl loaded lazily) so the module can
be imported on machines without jax.

Action encoding (shared with Johnny's app):
    0 NOOP, 1 UP, 2 DOWN, 3 LEFT, 4 RIGHT, 5 LOAD
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import EnvRenderer, PartnerSpec

log = logging.getLogger(__name__)

ACTION_NOOP = 0
ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_LOAD = 5

_ACTION_NAMES = {
    ACTION_NOOP: "wait",
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_LEFT: "left",
    ACTION_RIGHT: "right",
    ACTION_LOAD: "load",
}

# Default LBF variant: 7x7 grid, 2 agents, 3 food items, same-level agents.
# Tuned to match the smallest variant Johnny's app offers so new players
# have a manageable starting grid.
_DEFAULT_KWARGS = {
    "grid_size": 7,
    "num_agents": 2,
    "num_food": 3,
    "different_levels": False,
}


@dataclass
class _LBFMeta:
    grid_size: int
    num_agents: int
    num_food: int


class LBFRenderer(EnvRenderer):
    ENV_NAME = "lbf"
    DISPLAY_NAME = "Level-Based Foraging"
    HUMAN_AGENT_IDX = 0
    DEFAULT_KWARGS = _DEFAULT_KWARGS
    OVERVIEW = (
        "Two foragers on a grid collecting fruit. Some fruit needs both "
        "agents loading at the same time to pick up."
    )
    STATS = {
        "players": "2",
        "grid": "7x7",
        "actions": "6",
        "fruits": "3",
    }
    ACCENT = "green"

    def __init__(self):
        self._meta_cache: dict[int, _LBFMeta] = {}

    # ---------- env lifecycle ----------

    def make_env(self, kwargs: dict | None = None):
        from envs import make_env  # jax-heavy; lazy
        kwargs = {**_DEFAULT_KWARGS, **(kwargs or {})}
        env = make_env("lbf-reward-shaping", kwargs)
        self._meta_cache[id(env)] = _LBFMeta(
            grid_size=int(kwargs["grid_size"]),
            num_agents=int(kwargs["num_agents"]),
            num_food=int(kwargs["num_food"]),
        )
        return env

    def reset(self, env, rng):
        obs, state = env.reset(rng)
        return obs, state

    def step(self, env, state, actions: dict[int, int], rng):
        # Missing agents get NOOP so partner-only or human-only steps
        # still round-trip cleanly.
        action_dict = {
            f"agent_{i}": int(actions.get(i, ACTION_NOOP))
            for i in range(env.num_agents)
        }
        obs, state, reward, done, info = env.step(rng, state, action_dict)
        r0 = float(reward.get("agent_0", 0.0))
        r1 = float(reward.get("agent_1", 0.0))
        total = r0 + r1
        done_flag = bool(done.get("__all__", done.get("agent_0", False)))
        return obs, state, total, done_flag, info

    # ---------- partners ----------

    def available_partners(self) -> list[PartnerSpec]:
        return [
            PartnerSpec(
                key="random",
                display_name="Random",
                description="Moves and loads uniformly at random. Baseline.",
                load_fn=lambda: _make_partner("random"),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="seq_lexi",
                display_name="Sequential (lexicographic)",
                description="Visits food in lexicographic order of position.",
                load_fn=lambda: _make_partner("seq", "lexicographic"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="seq_rlexi",
                display_name="Sequential (reverse-lexicographic)",
                description="Visits food in reverse lexicographic order.",
                load_fn=lambda: _make_partner("seq", "reverse_lexicographic"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="seq_col",
                display_name="Sequential (column-major)",
                description="Visits food column-by-column.",
                load_fn=lambda: _make_partner("seq", "column_major"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="seq_rcol",
                display_name="Sequential (reverse-column)",
                description="Visits food reverse-column-major.",
                load_fn=lambda: _make_partner("seq", "reverse_column_major"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="seq_nearest",
                display_name="Nearest-first",
                description="Goes to the food nearest to itself at each step.",
                load_fn=lambda: _make_partner("seq", "nearest_agent"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="seq_farthest",
                display_name="Farthest-first",
                description="Goes to the food farthest from itself - often counter-productive.",
                load_fn=lambda: _make_partner("seq", "farthest_agent"),
                tags=["heuristic", "sequential"],
            ),
            PartnerSpec(
                key="greedy_closest_self",
                display_name="Greedy (closest to self)",
                description="Targets the food closest to itself; ignores teammate position.",
                load_fn=lambda: _make_partner("greedy", "closest_self"),
                tags=["heuristic", "greedy"],
            ),
            PartnerSpec(
                key="greedy_closest_teammate",
                display_name="Greedy (closest to teammate)",
                description="Targets the food closest to its teammate; encourages joint loads.",
                load_fn=lambda: _make_partner("greedy", "closest_teammate"),
                tags=["heuristic", "greedy"],
            ),
            PartnerSpec(
                key="greedy_closest_avg",
                display_name="Greedy (closest average)",
                description="Targets the food closest to the midpoint of self + teammate.",
                load_fn=lambda: _make_partner("greedy", "closest_avg"),
                tags=["heuristic", "greedy"],
            ),
            PartnerSpec(
                key="entitled",
                display_name="Entitled",
                description="Waits for its teammate to arrive before loading. Strong baseline when fruit needs both.",
                load_fn=lambda: _make_partner("entitled"),
                tags=["heuristic", "entitled"],
            ),
        ]

    def default_partner_key(self) -> str:
        return "seq_lexi"

    def study_config(self):
        """LBF Prolific study: 2 warmup + 8 real games (matches Johnny's
        human_data_collecting NUM_WARMUP_GAMES=2 + NUM_REAL_GAMES=8)."""
        from ..study_session import StudyConfig
        # 2 warmup games to learn controls before counted games begin
        warmup = [
            {"grid_size": 7, "num_food": 3, "different_levels": False},
            {"grid_size": 7, "num_food": 3, "different_levels": True},
        ]
        # 4 base configs × 2 plays each = 8 real
        base = [
            {"grid_size": 7,  "num_food": 3, "different_levels": False},
            {"grid_size": 7,  "num_food": 3, "different_levels": True},
            {"grid_size": 12, "num_food": 6, "different_levels": False},
            {"grid_size": 12, "num_food": 6, "different_levels": True},
        ]
        real = base + base
        return StudyConfig(
            num_warmup=len(warmup),
            num_real=len(real),
            game_env_kwargs=warmup + real,
        )

    # ---------- UI contract ----------

    def serialize_state(self, state, obs) -> dict:
        """Payload consumed by frontend/src/envs/lbf/Board.tsx."""
        import numpy as np

        env_state = getattr(state, "env_state", state)

        agents_pos = np.asarray(env_state.agents.position).astype(int)
        agents_lvl = np.asarray(env_state.agents.level).astype(int)
        food_pos = np.asarray(env_state.food_items.position).astype(int)
        food_lvl = np.asarray(env_state.food_items.level).astype(int)
        food_eaten = np.asarray(env_state.food_items.eaten).astype(bool)

        grid_size = int(max(
            agents_pos.max(initial=0) + 1,
            food_pos.max(initial=0) + 1,
        ))
        # Fall back to cache if env state was reset very recently.
        for meta in self._meta_cache.values():
            grid_size = max(grid_size, meta.grid_size)
            break

        # Available actions for the human (agent 0). Default to all
        # actions if the state doesn't expose avail_actions.
        avail_actions = [True] * 6
        if hasattr(state, "avail_actions"):
            try:
                avail_actions = [bool(x) for x in np.asarray(state.avail_actions["agent_0"]).tolist()]
            except Exception:
                pass

        return {
            "grid_size": grid_size,
            "agents": [
                {"x": int(p[1]), "y": int(p[0]), "level": int(lvl)}
                for p, lvl in zip(agents_pos, agents_lvl)
            ],
            "food": [
                {"x": int(p[1]), "y": int(p[0]), "level": int(lvl), "eaten": bool(eaten)}
                for p, lvl, eaten in zip(food_pos, food_lvl, food_eaten)
            ],
            "avail_actions": avail_actions,
            "step_count": int(getattr(state, "step_count", 0)),
        }

    def action_from_ui(self, ui_payload: dict, state) -> int:
        """UI payload shape: {"type": "move", "dir": "up|down|left|right|load|wait"}
        or {"type": "noop"}."""
        t = ui_payload.get("type")
        d = ui_payload.get("dir", "wait")
        if t == "move" or t is None:
            mapping = {
                "wait": ACTION_NOOP,
                "up": ACTION_UP,
                "down": ACTION_DOWN,
                "left": ACTION_LEFT,
                "right": ACTION_RIGHT,
                "load": ACTION_LOAD,
            }
            if d not in mapping:
                raise ValueError(f"unknown direction {d!r}")
            return mapping[d]
        if t == "noop":
            return ACTION_NOOP
        raise ValueError(f"unknown ui action type {t!r}")

    def score_summary(self, state, info) -> dict:
        import numpy as np
        env_state = getattr(state, "env_state", state)
        food_eaten = np.asarray(env_state.food_items.eaten).astype(bool)
        eaten = int(food_eaten.sum())
        total = int(food_eaten.size)
        return {
            "fruits_eaten": eaten,
            "total_fruits": total,
            "step_count": int(getattr(state, "step_count", 0)),
            "return": float(getattr(state, "base_return_so_far", np.zeros(1)).sum())
                if hasattr(state, "base_return_so_far") else 0.0,
        }

    def keyboard_controls(self) -> dict[str, int]:
        return {
            "w": ACTION_UP,
            "s": ACTION_DOWN,
            "a": ACTION_LEFT,
            "d": ACTION_RIGHT,
            " ": ACTION_LOAD,
            "q": ACTION_NOOP,
        }

    def describe_action(self, action, state_before, state_after, reward, player) -> dict:
        event = {
            "player": player,
            "kind": "move",
            "direction": _ACTION_NAMES.get(int(action), f"action_{action}"),
            "reward": float(reward),
        }
        if reward > 0:
            event["ate_fruit"] = True
        return event


# ---------- partner adapters ----------

def _make_partner(kind: str, arg: str | None = None):
    """Instantiate one of the LBF policy wrappers as a get_action
    adapter the session manager can use. arg = ordering for "seq",
    heuristic for "greedy", ignored otherwise."""
    from agents.lbf.agent_policy_wrappers import (  # noqa: WPS433
        LBFRandomPolicyWrapper,
        LBFSequentialFruitPolicyWrapper,
        LBFEntitledPolicyWrapper,
        LBFGreedyHeuristicPolicyWrapper,
    )

    if kind == "random":
        policy = LBFRandomPolicyWrapper()
    elif kind == "seq":
        policy = LBFSequentialFruitPolicyWrapper(
            grid_size=_DEFAULT_KWARGS["grid_size"],
            num_fruits=_DEFAULT_KWARGS["num_food"],
            ordering_strategy=arg or "lexicographic",
            using_log_wrapper=False,
        )
    elif kind == "greedy":
        policy = LBFGreedyHeuristicPolicyWrapper(
            grid_size=_DEFAULT_KWARGS["grid_size"],
            num_fruits=_DEFAULT_KWARGS["num_food"],
            heuristic=arg or "closest_self",
            using_log_wrapper=False,
        )
    elif kind == "entitled":
        policy = LBFEntitledPolicyWrapper(
            grid_size=_DEFAULT_KWARGS["grid_size"],
            num_fruits=_DEFAULT_KWARGS["num_food"],
            using_log_wrapper=False,
        )
    else:
        raise ValueError(f"unknown LBF partner kind {kind}")
    return _LBFLivePartnerAdapter(policy)


class _LBFLivePartnerAdapter:
    """Thin wrapper over LBF*PolicyWrapper giving the
    ``get_action(obs, state, rng) -> int`` contract ui uses."""

    def __init__(self, policy):
        self.policy = policy
        self._hstate = None

    def get_action(self, obs, state, rng):
        import jax.numpy as jnp  # noqa: WPS433
        if self._hstate is None:
            self._hstate = self.policy.init_hstate(1, aux_info={"agent_id": 1})
        agent_obs = obs["agent_1"] if isinstance(obs, dict) else obs
        avail = jnp.ones((6,))
        done = jnp.array(False)
        action, new_hstate = self.policy.get_action(
            params=None,
            obs=agent_obs,
            done=done,
            avail_actions=avail,
            hstate=self._hstate,
            rng=rng,
            env_state=state,
            aux_obs=None,
            test_mode=True,
        )
        self._hstate = new_hstate
        return int(action)
