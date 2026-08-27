"""Hanabi adapter.

Drives the human-vs-AI Hanabi mode. Wraps JaxMARL's Hanabi env via
this repo's HanabiWrapper and the agent_policy_wrappers in
agents.hanabi. Human is agent 0 by convention.

Heavy imports (jax, jaxmarl, flax) are deferred to ``make_env`` /
partner ``load_fn`` so the app still starts on a machine without a
jax install.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .base import EnvRenderer, PartnerSpec

log = logging.getLogger(__name__)

# Full-Hanabi defaults. Override per session via env_kwargs.
_FULL_KWARGS = {
    "num_agents": 2,
    "num_colors": 5,
    "num_ranks": 5,
    "hand_size": 5,
    "max_info_tokens": 8,
    "max_life_tokens": 3,
    "num_cards_of_rank": [3, 2, 2, 2, 1],
}
_MINI_KWARGS = {
    "num_agents": 2,
    "num_colors": 3,
    "num_ranks": 3,
    "hand_size": 3,
    "max_info_tokens": 5,
    "max_life_tokens": 3,
    "num_cards_of_rank": [2, 2, 1],
}

_COLOR_NAMES = ["red", "yellow", "green", "white", "blue"]


@dataclass
class _HanabiSessionMeta:
    """Shape-info derived from env_kwargs. Used for action decoding and
    render payloads so the frontend doesn't need to re-derive."""
    num_colors: int
    num_ranks: int
    hand_size: int
    num_actions: int
    variant: str   # "full" | "mini"

    @classmethod
    def from_kwargs(cls, kwargs: dict[str, Any]) -> "_HanabiSessionMeta":
        num_colors = int(kwargs["num_colors"])
        num_ranks = int(kwargs["num_ranks"])
        hand_size = int(kwargs["hand_size"])
        num_actions = 2 * hand_size + num_colors + num_ranks + 1
        variant = "mini" if num_colors < 5 else "full"
        return cls(num_colors, num_ranks, hand_size, num_actions, variant)


class HanabiRenderer(EnvRenderer):
    IS_TURN_BASED = True
    ENV_NAME = "hanabi"
    DISPLAY_NAME = "Hanabi"
    HUMAN_AGENT_IDX = 0
    DEFAULT_KWARGS = _FULL_KWARGS
    OVERVIEW = (
        "Cooperative card game. You can see your partner's hand but not "
        "your own; hints cost info tokens."
    )
    STATS = {
        "players": "2",
        "actions": "21",
        "obs dim": "658",
        "max score": "25",
    }
    ACCENT = "blue"

    def __init__(self):
        # Shape metadata is always derivable from the state's array
        # shapes (see _meta_for_state), so no cache needed. Kept the
        # method for compatibility with old callers.
        pass

    # ---------- env lifecycle ----------

    def make_env(self, kwargs: dict | None = None):
        from envs import make_env  # jax-heavy; imported lazily
        # use self.DEFAULT_KWARGS so MiniHanabiRenderer subclass picks up _MINI_KWARGS
        kwargs = {**self.DEFAULT_KWARGS, **(kwargs or {})}
        return make_env("hanabi", kwargs)

    def reset(self, env, rng):
        obs, state = env.reset(rng)
        return obs, state

    def step(self, env, state, actions: dict[int, int], rng):
        # Hanabi expects {"agent_0": int_action, "agent_1": int_action}.
        # Fill missing agents with noop (last action index).
        meta = self._meta_for_state(state)
        noop = meta.num_actions - 1
        action_dict = {
            f"agent_{i}": int(actions.get(i, noop)) for i in range(2)
        }
        obs, state, reward, done, info = env.step(rng, state, action_dict)
        r0 = float(reward.get("agent_0", 0.0))
        r1 = float(reward.get("agent_1", 0.0))
        total_reward = r0 + r1
        done_flag = bool(done.get("__all__", done.get("agent_0", False)))
        return obs, state, total_reward, done_flag, info

    # ---------- partners ----------

    def available_partners(self) -> list[PartnerSpec]:
        return [
            PartnerSpec(
                key="random_agent",
                display_name="Random",
                description="Uniformly random legal move. Floor baseline; scores near 0/25.",
                load_fn=lambda: self._make_partner("random"),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="iggi",
                display_name="IGGI",
                description="Walton-Rivers 2017 baseline: play safe, hint playable, osawa-discard, fallback.",
                load_fn=lambda: self._make_partner("iggi"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="piers",
                display_name="Piers",
                description="IGGI + probabilistic play + dispensable hints.",
                load_fn=lambda: self._make_partner("piers"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="cautious",
                display_name="Cautious",
                description="Conservative Walton-Rivers variant: hints whenever a hint helps, plays only when fully known.",
                load_fn=lambda: self._make_partner("cautious"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="internal",
                display_name="Internal",
                description="Walton-Rivers variant tracking what info has been conveyed inside the team.",
                load_fn=lambda: self._make_partner("internal"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="van_den_bergh",
                display_name="Van den Bergh",
                description="Probabilistic play + tell-most-info hints.",
                load_fn=lambda: self._make_partner("van_den_bergh"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="outer",
                display_name="Outer",
                description="Hint-heavy Walton-Rivers strategy.",
                load_fn=lambda: self._make_partner("outer"),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="flawed",
                display_name="Flawed",
                description="IGGI but with a lower play threshold (0.4 vs Piers' 0.6); plays cards it isn't sure about. Walton-Rivers low-competence partner.",
                load_fn=lambda: self._make_partner("flawed", play_threshold=0.4),
                tags=["heuristic", "walton-rivers"],
            ),
            PartnerSpec(
                key="smartbot",
                display_name="SmartBot",
                description="JAX port of Quuxplusone's C++ SmartBot. Convention-heavy, high-scoring rule-based agent.",
                load_fn=lambda: self._make_partner("smartbot"),
                tags=["heuristic"],
            ),
            PartnerSpec(
                key="obl_r2d2",
                display_name="OBL R2D2 (L1)",
                description="Off-Belief Learning L1 (Hu et al. 2021). Pretrained. ~21/25 self-play.",
                load_fn=lambda: self._make_partner("obl_r2d2"),
                tags=["learned", "pretrained"],
            ),
            PartnerSpec(
                key="bc_lstm_human_proxy",
                display_name="BC-LSTM (human proxy)",
                description="BC-LSTM trained on AH2AC2 human games. Human-style, not optimal. 3.4/25 self-play.",
                load_fn=lambda: self._make_partner("bc_lstm"),
                tags=["learned", "human-proxy"],
            ),
        ]

    def default_partner_key(self) -> str:
        return "iggi"

    # ---------- UI contract ----------

    def serialize_state(self, state, obs) -> dict:
        """Render-ready payload. Shape is documented in the frontend's
        HanabiState type. Agent 0 is the human.

        Hidden-information rule: the frontend never receives the
        human's own cards (identities). Partner cards are visible
        (that's the game). Card-knowledge arrays represent what each
        player believes about their hand.
        """
        import numpy as np

        if state is None:
            return {"error": "no state"}

        # Unwrap if state is a WrappedEnvState (BaseEnv wrapper).
        state = getattr(state, "env_state", state)

        meta = self._meta_for_state(state)

        fireworks = _to_list(state.fireworks)
        info_tokens = int(_sum_tokens(state.info_tokens))
        life_tokens = int(_sum_tokens(state.life_tokens))
        num_discarded = int(getattr(state, "num_cards_discarded", 0))
        current_player = _current_player(state)

        # player_hands has shape (num_agents, hand_size, num_colors, num_ranks).
        # Agent 1's hand is visible to agent 0 (the human).
        hands = np.asarray(state.player_hands)
        partner_hand = _decode_cards(hands[1], meta.num_colors, meta.num_ranks)
        human_hand_slots = int(hands[0].shape[0])

        # card_knowledge: what each player believes about their OWN hand.
        # Agent 0's card_knowledge is safe to show the human.
        knowledge = np.asarray(state.card_knowledge).reshape(
            2, meta.hand_size, meta.num_colors, meta.num_ranks,
        )
        human_beliefs = _decode_beliefs(knowledge[0])

        # Discard pile: a list of played/discarded card coords.
        discard_pile = _decode_cards(state.discard_pile, meta.num_colors, meta.num_ranks)
        discard_pile = discard_pile[:num_discarded]

        total_cards = int(sum(meta_cards_of_rank(meta)) * meta.num_colors)
        cards_in_hand = human_hand_slots + int(hands[1].shape[0])
        played = int(np.asarray(state.fireworks).sum())
        deck_size = max(0, total_cards - played - num_discarded - cards_in_hand)

        return {
            "variant": meta.variant,
            "num_colors": meta.num_colors,
            "num_ranks": meta.num_ranks,
            "hand_size": meta.hand_size,
            "color_names": _COLOR_NAMES[:meta.num_colors],
            "fireworks": fireworks,
            "info_tokens": info_tokens,
            "max_info_tokens": int(state.info_tokens.shape[0])
                if hasattr(state.info_tokens, "shape") else 8,
            "life_tokens": life_tokens,
            "max_life_tokens": int(state.life_tokens.shape[0])
                if hasattr(state.life_tokens, "shape") else 3,
            "current_player": current_player,
            "human_hand_slots": human_hand_slots,
            "partner_hand": partner_hand,
            "human_beliefs": human_beliefs,
            "discard_pile": discard_pile,
            "deck_size": deck_size,
            "num_cards_discarded": num_discarded,
            "score": played,
        }

    def action_from_ui(self, ui_payload: dict, state) -> int:
        """UI payload shape (examples):

            {"type": "play", "slot": 2}
            {"type": "discard", "slot": 0}
            {"type": "hint_color", "color": 1}   # index into color_names
            {"type": "hint_rank", "rank": 3}     # 0-indexed (rank 1 == 0)
            {"type": "noop"}
        """
        meta = self._meta_for_state(state)
        t = ui_payload.get("type")
        if t == "discard":
            slot = int(ui_payload["slot"])
            if not (0 <= slot < meta.hand_size):
                raise ValueError(f"discard slot {slot} out of range")
            return slot
        if t == "play":
            slot = int(ui_payload["slot"])
            if not (0 <= slot < meta.hand_size):
                raise ValueError(f"play slot {slot} out of range")
            return meta.hand_size + slot
        if t == "hint_color":
            c = int(ui_payload["color"])
            if not (0 <= c < meta.num_colors):
                raise ValueError(f"hint color {c} out of range")
            return 2 * meta.hand_size + c
        if t == "hint_rank":
            r = int(ui_payload["rank"])
            if not (0 <= r < meta.num_ranks):
                raise ValueError(f"hint rank {r} out of range")
            return 2 * meta.hand_size + meta.num_colors + r
        if t == "noop":
            return meta.num_actions - 1
        raise ValueError(f"unknown ui action type {t!r}")

    def score_summary(self, state, info) -> dict:
        if state is None:
            return {}
        import numpy as np
        state = getattr(state, "env_state", state)
        fireworks = _to_list(state.fireworks)
        return {
            "score": int(np.asarray(state.fireworks).sum()),
            "max_score": len(fireworks) * 5,
            "info_tokens": int(_sum_tokens(state.info_tokens)),
            "life_tokens": int(_sum_tokens(state.life_tokens)),
            "fireworks": fireworks,
        }

    def describe_action(
        self,
        action: int,
        state_before,
        state_after,
        reward: float,
        player: int,
    ) -> dict:
        """Decode an action index to a GameLog event. Detects bomb vs
        score by diffing life/fireworks tokens before/after."""
        state_before = getattr(state_before, "env_state", state_before) if state_before is not None else None
        state_after = getattr(state_after, "env_state", state_after) if state_after is not None else None
        meta = self._meta_for_state(state_before or state_after)
        hand_size = meta.hand_size
        num_colors = meta.num_colors

        if 0 <= action < hand_size:
            return {"player": player, "kind": "discard", "slot": int(action)}

        if hand_size <= action < 2 * hand_size:
            slot = int(action - hand_size)
            event: dict = {"player": player, "kind": "play", "slot": slot}
            # Did a bomb happen (life token lost)?
            if state_before is not None and state_after is not None:
                lives_before = int(_sum_tokens(state_before.life_tokens))
                lives_after = int(_sum_tokens(state_after.life_tokens))
                if lives_after < lives_before:
                    event["bombed"] = True
                elif reward > 0:
                    event["scored"] = True
                # Reveal the card identity by diffing the discard pile
                # (bomb -> card goes to discard) or fireworks (score ->
                # fireworks grows). Either way we can tell what was
                # there from the before-state's partner knowledge.
            return event

        hint_start = 2 * hand_size
        if hint_start <= action < hint_start + num_colors:
            return {
                "player": player,
                "kind": "hint_color",
                "color": int(action - hint_start),
            }

        rank_start = hint_start + num_colors
        if rank_start <= action < rank_start + meta.num_ranks:
            return {
                "player": player,
                "kind": "hint_rank",
                "rank": int(action - rank_start),
            }

        return {"player": player, "kind": "noop"}

    # ---------- internals ----------

    def _make_partner(self, actor_type: str, **kwargs):
        """Instantiate one of the agent_policy_wrappers for live play.
        Partner adapters all expose ``get_action(obs, state, rng)``.

        Extra kwargs flow to the underlying wrapper (e.g. ``strategy``
        for rule_based, ``mistake_prob`` for flawed).
        """
        from agents.hanabi.agent_policy_wrappers import (  # noqa: WPS433
            HanabiBCLSTMPolicyWrapper,
            HanabiCautiousPolicyWrapper,
            HanabiFlawedPolicyWrapper,
            HanabiIGGIPolicyWrapper,
            HanabiInternalPolicyWrapper,
            HanabiOBLPolicyWrapper,
            HanabiOuterPolicyWrapper,
            HanabiPiersPolicyWrapper,
            HanabiRandomPolicyWrapper,
            HanabiSmartBotPolicyWrapper,
            HanabiVanDenBerghPolicyWrapper,
        )

        common = dict(
            hand_size=self.DEFAULT_KWARGS["hand_size"],
            num_colors=self.DEFAULT_KWARGS["num_colors"],
            num_ranks=self.DEFAULT_KWARGS["num_ranks"],
            num_actions=2 * self.DEFAULT_KWARGS["hand_size"]
                + self.DEFAULT_KWARGS["num_colors"]
                + self.DEFAULT_KWARGS["num_ranks"]
                + 1,
            using_log_wrapper=False,
        )
        if actor_type == "random":
            policy = HanabiRandomPolicyWrapper(
                num_actions=common["num_actions"], using_log_wrapper=False,
            )
        elif actor_type == "iggi":
            policy = HanabiIGGIPolicyWrapper(**common)
        elif actor_type == "cautious":
            policy = HanabiCautiousPolicyWrapper(**common)
        elif actor_type == "internal":
            policy = HanabiInternalPolicyWrapper(**common)
        elif actor_type == "piers":
            policy = HanabiPiersPolicyWrapper(
                play_threshold=0.6, hint_threshold=4, **common,
            )
        elif actor_type == "van_den_bergh":
            policy = HanabiVanDenBerghPolicyWrapper(**common)
        elif actor_type == "outer":
            policy = HanabiOuterPolicyWrapper(**common)
        elif actor_type == "flawed":
            policy = HanabiFlawedPolicyWrapper(
                play_threshold=float(kwargs.get("play_threshold", 0.4)), **common,
            )
        elif actor_type == "smartbot":
            policy = HanabiSmartBotPolicyWrapper(**common)
        elif actor_type == "obl_r2d2":
            policy = HanabiOBLPolicyWrapper(
                weight_file="agents/hanabi/obl-r2d2-flax/icml_OBL1/OFF_BELIEF1_SHUFFLE_COLOR0_BZA0_BELIEF_a.safetensors",
                using_log_wrapper=False,
            )
        elif actor_type == "bc_lstm":
            policy = HanabiBCLSTMPolicyWrapper(
                weight_file="agents/bc_weights/hanabi_ah2ac2_bc.safetensors",
                using_log_wrapper=False,
            )
        else:
            raise ValueError(f"unknown Hanabi partner: {actor_type}")
        return _LivePartnerAdapter(policy)

    def _meta_for_state(self, state) -> _HanabiSessionMeta:
        """Derive meta from the env state itself (its array shapes).
        Robust to env_kwargs overrides we didn't cache."""
        try:
            import numpy as np
            hands = np.asarray(state.player_hands)
            # hands shape: (num_agents, hand_size, num_colors, num_ranks)
            _, hand_size, num_colors, num_ranks = hands.shape
            return _HanabiSessionMeta(
                num_colors=int(num_colors),
                num_ranks=int(num_ranks),
                hand_size=int(hand_size),
                num_actions=2 * hand_size + num_colors + num_ranks + 1,
                variant="mini" if num_colors < 5 else "full",
            )
        except Exception:
            return _HanabiSessionMeta.from_kwargs(_FULL_KWARGS)


class _LivePartnerAdapter:
    """Wraps the heavy HanabiXPolicyWrapper classes into the simpler
    ``get_action(obs, state, rng) -> int`` contract ui
    expects. Encapsulates hstate threading per partner policy."""

    def __init__(self, policy):
        self.policy = policy
        self._hstate = None

    def get_action(self, obs, state, rng):
        import jax.numpy as jnp  # noqa: WPS433
        if self._hstate is None:
            aux = {"agent_id": 1}
            self._hstate = self.policy.init_hstate(batch_size=1, aux_info=aux)
        agent_obs = obs["agent_1"] if isinstance(obs, dict) else obs
        # Assume fully-available actions; individual partners mask internally.
        avail_actions = jnp.ones((self._hstate_shape_dim(),))
        done = jnp.array(False)
        action, new_hstate = self.policy.get_action(
            params=None,
            obs=agent_obs,
            done=done,
            avail_actions=avail_actions,
            hstate=self._hstate,
            rng=rng,
            env_state=state,
            aux_obs=None,
            test_mode=True,
        )
        self._hstate = new_hstate
        return int(action)

    def _hstate_shape_dim(self) -> int:
        # Policies that need avail_actions read its length; we use the
        # policy's action dim if it exposes one, else a safe upper bound.
        return int(getattr(self.policy, "num_actions", 21))


# -------- state decoding helpers --------

def _to_list(arr) -> list[int]:
    import numpy as np
    return np.asarray(arr).astype(int).tolist()


def _sum_tokens(tokens) -> int:
    import numpy as np
    return int(np.asarray(tokens).sum())


def _current_player(state) -> int:
    """Hanabi stores cur_player_idx as a one-hot. Decode to int."""
    import numpy as np
    cp = getattr(state, "cur_player_idx", None)
    if cp is None:
        return 0
    arr = np.asarray(cp).flatten()
    if arr.size == 0:
        return 0
    return int(arr.argmax())


def _decode_cards(cards, num_colors: int, num_ranks: int) -> list[dict]:
    """Input: (n, num_colors, num_ranks) one-hot card grid. Output: list
    of {"color": int, "rank": int, "known": True} or
    {"color": None, "rank": None, "known": False} for empty slots.
    """
    import numpy as np

    arr = np.asarray(cards)
    if arr.ndim == 2:
        arr = arr[None, ...]
    out = []
    for card in arr:
        if card.sum() == 0:
            out.append({"color": None, "rank": None, "known": False})
            continue
        flat = card.reshape(-1)
        idx = int(flat.argmax())
        color = idx // num_ranks
        rank = idx % num_ranks
        out.append({"color": int(color), "rank": int(rank), "known": True})
    return out


def _decode_beliefs(knowledge) -> list[dict]:
    """Convert (hand_size, num_colors, num_ranks) knowledge tensor into
    a list of belief dicts per slot. Each dict lists possible
    (color, rank) pairs + 'hinted' booleans per axis.
    """
    import numpy as np
    arr = np.asarray(knowledge)
    out = []
    for slot in arr:
        possible = [
            {"color": int(c), "rank": int(r)}
            for c in range(slot.shape[0])
            for r in range(slot.shape[1])
            if slot[c, r] > 0
        ]
        color_possible = set(p["color"] for p in possible)
        rank_possible = set(p["rank"] for p in possible)
        out.append({
            "possible": possible,
            "color_hinted": len(color_possible) == 1,
            "rank_hinted": len(rank_possible) == 1,
            "color": next(iter(color_possible)) if len(color_possible) == 1 else None,
            "rank": next(iter(rank_possible)) if len(rank_possible) == 1 else None,
        })
    return out


def meta_cards_of_rank(meta: _HanabiSessionMeta) -> list[int]:
    """Standard-Hanabi card counts sliced to num_ranks. Mini overrides
    elsewhere; this is fine as a deck-size fallback."""
    standard = [3, 2, 2, 2, 1]
    if meta.variant == "mini":
        return [2, 2, 1][:meta.num_ranks]
    return standard[:meta.num_ranks]
