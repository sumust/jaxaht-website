"""A trivial pure-Python env used to smoke-test the Flask routes
without a jax install. Not exposed as a real AHT env to users; serves
only as a development aid.

Game: human and partner alternate pressing 0-4. Reward += 1 if the
human's number matches the partner's previous number. Episode ends
after 20 turns.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .base import BuiltinEgoSpec, EnvRenderer, PartnerSpec


@dataclass
class MockState:
    turn: int = 0
    current_player: int = 0
    last_partner_action: int = -1
    score: int = 0
    history: list[dict] = field(default_factory=list)


class _MockEnv:
    MAX_TURNS = 20


class MockRenderer(EnvRenderer):
    ENV_NAME = "mock"
    DISPLAY_NAME = "Mock Matching Game"
    HUMAN_AGENT_IDX = 0
    DEFAULT_KWARGS = {}
    # Dev-only. Routes + tests still exercise this env, but it's hidden
    # from the Home page so users don't see benchmark-UI dev scaffolding.
    HIDDEN = True

    def make_env(self, kwargs=None):
        return _MockEnv()

    def reset(self, env, rng):
        return {}, MockState()

    def step(self, env, state: MockState, actions: dict[int, int], rng):
        acted_by = state.current_player
        action = actions[acted_by]
        reward = 0
        if acted_by == 1 and state.last_partner_action == action:
            reward = 0   # partner matching itself no-ops
        if acted_by == 0 and state.last_partner_action == action:
            reward = 1
            state.score += 1

        state.history.append({
            "turn": state.turn,
            "player": acted_by,
            "action": action,
            "reward": reward,
        })
        if acted_by == 1:
            state.last_partner_action = action

        state.turn += 1
        state.current_player = 1 - state.current_player
        done = state.turn >= env.MAX_TURNS
        return {}, state, reward, done, {}

    def available_partners(self) -> list[PartnerSpec]:
        return [
            PartnerSpec(
                key="random",
                display_name="Random picker",
                difficulty="easy",
                description="Plays uniformly at random.",
                load_fn=lambda: _RandomMockPartner(),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="echo",
                display_name="Echo bot",
                difficulty="easy",
                description="Always plays the number the human played last.",
                load_fn=lambda: _EchoMockPartner(),
                tags=["heuristic"],
            ),
        ]

    def serialize_state(self, state: MockState, obs) -> dict:
        return {
            "turn": state.turn,
            "current_player": state.current_player,
            "last_partner_action": state.last_partner_action,
            "score": state.score,
            "max_turns": _MockEnv.MAX_TURNS,
            "history": state.history[-10:],
        }

    def action_from_ui(self, ui_payload: dict, state: MockState) -> int:
        action = ui_payload.get("action")
        if not isinstance(action, int) or not (0 <= action <= 4):
            raise ValueError(f"action must be int 0-4, got {action!r}")
        return action

    def score_summary(self, state: MockState, info) -> dict:
        return {
            "score": state.score,
            "turn": f"{state.turn} / {_MockEnv.MAX_TURNS}",
        }

    def keyboard_controls(self) -> dict[str, int]:
        return {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4}

    def describe_action(self, action, state_before, state_after, reward, player) -> dict:
        return {
            "player": player,
            "kind": "play",
            "action": int(action),
            "reward": float(reward),
            "scored": bool(reward > 0),
        }

    def builtin_egos(self):
        from .base import BuiltinEgoSpec

        def _random_ego(obs, state, rng) -> int:
            return random.randint(0, 4)

        def _constant_ego(obs, state, rng) -> int:
            return 2

        def _mimic_ego(obs, state, rng) -> int:
            # Match what the partner just did. Pairs well with an echo
            # partner for a high-scoring baseline.
            if isinstance(state, MockState):
                return state.last_partner_action if state.last_partner_action >= 0 else 0
            return 0

        return [
            BuiltinEgoSpec(
                key="random",
                display_name="Random",
                description="Uniform over 0-4. Floor baseline.",
                load_fn=lambda: _random_ego,
                tags=["baseline"],
            ),
            BuiltinEgoSpec(
                key="constant_2",
                display_name="Always 2",
                description="Always plays 2. Pairs well with constant partners.",
                load_fn=lambda: _constant_ego,
                tags=["baseline"],
            ),
            BuiltinEgoSpec(
                key="mimic",
                display_name="Mimic last partner",
                description="Plays whatever the partner just played.",
                load_fn=lambda: _mimic_ego,
                tags=["baseline"],
            ),
        ]


class _RandomMockPartner:
    def get_action(self, obs, state, rng) -> int:
        return random.randint(0, 4)


class _EchoMockPartner:
    def get_action(self, obs, state, rng) -> int:
        if not isinstance(state, MockState):
            return 0
        for entry in reversed(state.history):
            if entry["player"] == 0:
                return entry["action"]
        return 0
