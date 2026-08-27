"""Overcooked-v1 adapter.

Wraps JaxMARL's Overcooked environment via envs.make_env('overcooked-v1').
Reuses existing OvercookedRandom/Static/Independent/Onion/Plate policy
wrappers for partners. Action encoding follows JaxMARL's Overcooked
action_set order: up, down, right, left, stay, interact.

Five layouts supported (matches the official Overcooked-v1 set):
    cramped_room, asymm_advantages, coord_ring, counter_circuit,
    forced_coord.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .base import EnvRenderer, PartnerSpec

log = logging.getLogger(__name__)

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_RIGHT = 2
ACTION_LEFT = 3
ACTION_STAY = 4
ACTION_INTERACT = 5

_ACTION_NAMES = {
    ACTION_UP: "up",
    ACTION_DOWN: "down",
    ACTION_RIGHT: "right",
    ACTION_LEFT: "left",
    ACTION_STAY: "stay",
    ACTION_INTERACT: "interact",
}

_DEFAULT_KWARGS = {
    "layout": "cramped_room",
}

_LAYOUTS = [
    "cramped_room",
    "asymm_advantages",
    "coord_ring",
    "counter_circuit",
    "forced_coord",
]


@dataclass
class _OvercookedMeta:
    layout: str
    height: int
    width: int


class OvercookedRenderer(EnvRenderer):
    ENV_NAME = "overcooked-v1"
    DISPLAY_NAME = "Overcooked"
    HUMAN_AGENT_IDX = 0
    DEFAULT_KWARGS = _DEFAULT_KWARGS
    OVERVIEW = (
        "Two cooks share a kitchen, picking onions, dropping them in pots, "
        "plating finished soup, and serving before the timer runs out."
    )
    STATS = {
        "players": "2",
        "layouts": str(len(_LAYOUTS)),
        "actions": "6",
        "horizon": "400",
    }
    ACCENT = "yellow"

    def __init__(self):
        self._meta_cache: dict[int, _OvercookedMeta] = {}

    def make_env(self, kwargs: dict | None = None):
        from envs import make_env  # noqa: WPS433
        from envs.overcooked.augmented_layouts import augmented_layouts  # noqa: WPS433

        kwargs = {**_DEFAULT_KWARGS, **(kwargs or {})}
        layout_name = kwargs.get("layout", "cramped_room")
        if layout_name not in augmented_layouts:
            raise ValueError(
                f"unknown overcooked layout '{layout_name}'. "
                f"available: {sorted(augmented_layouts.keys())}"
            )
        env = make_env("overcooked-v1", kwargs)
        layout = augmented_layouts[layout_name]
        self._meta_cache[id(env)] = _OvercookedMeta(
            layout=layout_name,
            height=int(layout["height"]),
            width=int(layout["width"]),
        )
        return env

    def reset(self, env, rng):
        obs, state = env.reset(rng)
        return obs, state

    def step(self, env, state, actions: dict[int, int], rng):
        action_dict = {
            f"agent_{i}": int(actions.get(i, ACTION_STAY))
            for i in range(env.num_agents)
        }
        obs, state, reward, done, info = env.step(rng, state, action_dict)
        r0 = float(reward.get("agent_0", 0.0))
        r1 = float(reward.get("agent_1", 0.0))
        total = r0 + r1
        done_flag = bool(done.get("__all__", done.get("agent_0", False)))
        return obs, state, total, done_flag, info

    def available_partners(self) -> list[PartnerSpec]:
        return [
            PartnerSpec(
                key="random",
                display_name="Random",
                description="Picks actions uniformly at random. Baseline.",
                load_fn=lambda: _make_partner("random"),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="static",
                display_name="Static",
                description="Stands still. Tests whether the human can carry the team.",
                load_fn=lambda: _make_partner("static"),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="onion_agent",
                display_name="Onion Specialist",
                description="Picks up onions and drops them in pots. Doesn't plate or serve.",
                load_fn=lambda: _make_partner("onion"),
                tags=["heuristic", "specialist"],
            ),
            PartnerSpec(
                key="plate_agent",
                display_name="Plate Specialist",
                description="Picks up plates, puts soup on them, serves. Doesn't handle onions.",
                load_fn=lambda: _make_partner("plate"),
                tags=["heuristic", "specialist"],
            ),
            PartnerSpec(
                key="independent",
                display_name="Independent",
                description="Tries to do the whole loop alone (onion -> pot -> plate -> serve).",
                load_fn=lambda: _make_partner("independent"),
                tags=["heuristic"],
            ),
        ]

    def default_partner_key(self) -> str:
        return "independent"

    def serialize_state(self, state, obs) -> dict:
        """Payload consumed by frontend/src/envs/overcooked/Board.tsx."""
        import numpy as np

        env_state = getattr(state, "env_state", state)

        height = 0
        width = 0
        layout_name = "cramped_room"
        for meta in self._meta_cache.values():
            height = meta.height
            width = meta.width
            layout_name = meta.layout
            break
        if not (height and width):
            try:
                shape = np.asarray(env_state.maze_map).shape
                height, width = int(shape[0]) - 2, int(shape[1]) - 2
            except Exception:
                height, width = 5, 5

        agents_pos = np.asarray(env_state.agent_pos).astype(int)
        agents_dir = np.asarray(env_state.agent_dir_idx).astype(int)
        agents_inv = np.asarray(env_state.agent_inv).astype(int)

        try:
            wall_map = np.asarray(env_state.wall_map).astype(bool).tolist()
        except AttributeError:
            wall_map = []
        try:
            pot_pos = np.asarray(env_state.pot_pos).astype(int)
        except AttributeError:
            pot_pos = np.empty((0, 2), dtype=int)

        return {
            "layout": layout_name,
            "height": height,
            "width": width,
            "agents": [
                {
                    "x": int(p[0]),
                    "y": int(p[1]),
                    "dir": int(d),
                    "holding": int(inv),
                }
                for p, d, inv in zip(agents_pos, agents_dir, agents_inv)
            ],
            "wall_map": wall_map,
            "pots": [{"x": int(p[0]), "y": int(p[1])} for p in pot_pos],
            "step_count": int(getattr(state, "step", getattr(state, "step_count", 0))),
            "avail_actions": [True] * 6,
        }

    def action_from_ui(self, ui_payload: dict, state) -> int:
        """UI payload: {"type": "move", "dir": "up|down|left|right|stay|interact"}."""
        t = ui_payload.get("type")
        d = ui_payload.get("dir", "stay")
        if t == "move" or t is None:
            mapping = {
                "up": ACTION_UP,
                "down": ACTION_DOWN,
                "right": ACTION_RIGHT,
                "left": ACTION_LEFT,
                "stay": ACTION_STAY,
                "interact": ACTION_INTERACT,
            }
            if d not in mapping:
                raise ValueError(f"unknown direction {d!r}")
            return mapping[d]
        if t == "noop":
            return ACTION_STAY
        raise ValueError(f"unknown ui action type {t!r}")

    def score_summary(self, state, info) -> dict:
        import numpy as np
        ret_attr = getattr(state, "base_return_so_far", None)
        if ret_attr is not None:
            try:
                total_return = float(np.asarray(ret_attr).sum())
            except Exception:
                total_return = 0.0
        else:
            total_return = 0.0
        return {
            "score": total_return,
            "step_count": int(getattr(state, "step", getattr(state, "step_count", 0))),
            "horizon": 400,
        }

    def keyboard_controls(self) -> dict[str, int]:
        return {
            "w": ACTION_UP,
            "s": ACTION_DOWN,
            "d": ACTION_RIGHT,
            "a": ACTION_LEFT,
            " ": ACTION_INTERACT,
            "q": ACTION_STAY,
        }

    def describe_action(self, action, state_before, state_after, reward, player) -> dict:
        return {
            "player": player,
            "kind": "interact" if action == ACTION_INTERACT else "move",
            "direction": _ACTION_NAMES.get(int(action), f"action_{action}"),
            "reward": float(reward),
        }


def _make_partner(kind: str):
    from envs.overcooked.augmented_layouts import augmented_layouts  # noqa: WPS433
    from agents.overcooked.agent_policy_wrappers import (  # noqa: WPS433
        OvercookedRandomPolicyWrapper,
        OvercookedStaticPolicyWrapper,
        OvercookedOnionPolicyWrapper,
        OvercookedPlatePolicyWrapper,
        OvercookedIndependentPolicyWrapper,
    )

    aug_layout = augmented_layouts[_DEFAULT_KWARGS["layout"]]

    if kind == "random":
        policy = OvercookedRandomPolicyWrapper(aug_layout, using_log_wrapper=False)
    elif kind == "static":
        policy = OvercookedStaticPolicyWrapper(aug_layout, using_log_wrapper=False)
    elif kind == "onion":
        policy = OvercookedOnionPolicyWrapper(
            aug_layout, using_log_wrapper=False, p_onion_on_counter=0.0,
        )
    elif kind == "plate":
        policy = OvercookedPlatePolicyWrapper(
            aug_layout, using_log_wrapper=False, p_plate_on_counter=0.0,
        )
    elif kind == "independent":
        policy = OvercookedIndependentPolicyWrapper(
            aug_layout, using_log_wrapper=False,
            p_onion_on_counter=0.0, p_plate_on_counter=0.0,
        )
    else:
        raise ValueError(f"unknown overcooked partner kind {kind!r}")
    return _OvercookedLivePartnerAdapter(policy)


class _OvercookedLivePartnerAdapter:
    """Thin wrapper giving get_action(obs, state, rng) -> int contract."""

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
