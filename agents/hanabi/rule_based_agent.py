import jax
import jax.numpy as jnp
from agents.hanabi.base_agent import BaseAgent, AgentState
from typing import Tuple

STRATEGY_WEIGHTS = {
    "cautious":       (1.0,   10.0,  1000.0),
    "aggressive":     (1000.0, 1.0,  10.0),
    "communicative":  (10.0,   1.0,  1000.0),
    "frugal":         (1.0,   1000.0, 10.0),
}

VALID_STRATEGIES = tuple(STRATEGY_WEIGHTS.keys())


class RuleBasedAgent(BaseAgent):
    def __init__(
        self,
        strategy: str = "cautious",
        hand_size: int = 5,
        num_colors: int = 5,
        num_ranks: int = 5,
        num_actions: int = 21,
        **kwargs,
    ):
        super().__init__(num_actions=num_actions, **kwargs)
        if strategy not in STRATEGY_WEIGHTS:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Valid strategies: {VALID_STRATEGIES}"
            )
        self.strategy = strategy
        self.hand_size = hand_size
        self.num_colors = num_colors
        self.num_ranks = num_ranks

        play_w, discard_w, hint_w = STRATEGY_WEIGHTS[strategy]
        self.play_weight = play_w
        self.discard_weight = discard_w
        self.hint_weight = hint_w

        weights = jnp.zeros(num_actions)
        discard_end = hand_size
        play_end = 2 * hand_size
        hint_end = play_end + num_colors + num_ranks
        weights = weights.at[:discard_end].set(discard_w)
        weights = weights.at[discard_end:play_end].set(play_w)
        weights = weights.at[play_end:hint_end].set(hint_w)
        self._logits = jnp.log(weights + 1e-10)

    def _get_action(
        self,
        obs: jnp.ndarray,
        env_state,
        avail_mask: jnp.ndarray,
        agent_state: AgentState,
        rng: jax.random.PRNGKey,
    ) -> Tuple[int, AgentState]:
        masked_logits = jnp.where(avail_mask > 0, self._logits, -1e9)
        action = jax.random.categorical(rng, masked_logits)
        return action, agent_state
