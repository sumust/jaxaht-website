"""Env wrapper that applies a sampled symmetry to one agent's view.

Implements Other-Play-style augmentation generically (Hu et al. 2020): a
single symmetry is sampled per episode and persists across all steps within
that episode, so the policy sees a consistent permuted view of the world.
On reset the wrapper samples a fresh symmetry; on step it carries the same
symmetry forward unless the episode ended, in which case it resamples for
the next (auto-reset) episode.
"""
from functools import partial
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from flax import struct
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.wrappers.baselines import JaxMARLWrapper

from envs.common.symmetry import EnvSymmetry


@struct.dataclass
class SymmetryWrappedState:
    env_state: State
    sym: jnp.ndarray  # the currently-applied symmetry (permutation, reflection, etc.)


class SymmetryAugmentationWrapper(JaxMARLWrapper):
    """Wraps an env so one agent sees a permuted view of every observation.

    The wrapper applies `symmetry` to `env.agents[agent_idx_to_augment]`'s
    obs and avail_actions, and inverse-applies it to that agent's actions
    before stepping the underlying env. All other agents see the env's
    native observations.
    """

    def __init__(self, env: MultiAgentEnv, symmetry: EnvSymmetry,
                 agent_idx_to_augment: int = 1):
        super().__init__(env)
        self.symmetry = symmetry
        self.agent_idx_to_augment = agent_idx_to_augment
        self.augmented_agent_name = env.agents[agent_idx_to_augment]

    def get_avail_actions(self, state):
        """Return avail_actions with the augmented agent's mask permuted."""
        if isinstance(state, SymmetryWrappedState):
            avail = self._env.get_avail_actions(state.env_state)
            sym = state.sym
        else:
            avail = self._env.get_avail_actions(state)
            return avail

        permuted = {**avail}
        permuted[self.augmented_agent_name] = self.symmetry.apply_to_avail_actions(
            avail[self.augmented_agent_name], sym
        )
        return permuted

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey):
        key, sym_key = jax.random.split(key)
        sym = self.symmetry.sample(sym_key)
        obs, env_state = self._env.reset(key)
        obs = {**obs}
        obs[self.augmented_agent_name] = self.symmetry.apply_to_obs(
            obs[self.augmented_agent_name], sym
        )
        return obs, SymmetryWrappedState(env_state=env_state, sym=sym)

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: chex.PRNGKey,
        state: SymmetryWrappedState,
        action: dict,
    ):
        # Inverse-permute the augmented agent's action (which was chosen
        # under the permuted view) back into the env's native action space.
        native_action = {**action}
        native_action[self.augmented_agent_name] = self.symmetry.apply_to_action(
            action[self.augmented_agent_name], state.sym
        )

        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, native_action
        )

        # Persist the same symmetry across steps in an episode; resample
        # only when the episode just ended (the underlying env has already
        # auto-reset, so the obs we received belongs to the next episode).
        key, sym_key = jax.random.split(key)
        resampled = self.symmetry.sample(sym_key)
        new_sym = jax.tree.map(
            lambda old, new: jnp.where(done["__all__"], new, old),
            state.sym, resampled,
        )

        obs = {**obs}
        obs[self.augmented_agent_name] = self.symmetry.apply_to_obs(
            obs[self.augmented_agent_name], new_sym
        )
        return obs, SymmetryWrappedState(env_state=env_state, sym=new_sym), \
               reward, done, info
