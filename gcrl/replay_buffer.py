from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


class ReplayBuffer(FrozenDict):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __len__(self):
        return self.observations.shape[0]

    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> "ReplayBuffer":
        idx = jax.random.choice(rng, len(self), shape=(batch_size,), replace=False)
        return ReplayBuffer(
            observations=self.observations[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_observations=self.next_observations[idx],
            dones=self.dones[idx],
        )

    @classmethod
    def create(
        cls,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        rewards: jnp.ndarray,
        next_observations: jnp.ndarray,
        dones: jnp.ndarray,
    ):
        return cls(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
        )


class HierarchicalReplayBuffer(FrozenDict):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    subgoal_observations: jnp.ndarray
    goals: jnp.ndarray
    episode_end_idx: jnp.ndarray

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __len__(self):
        return self.observations.shape[0]

    def sample(
        self,
        batch_size: int,
        rng: jax.random.PRNGKey,
        reward_type: Literal["neg_one_zero", "zero_one"] = "neg_one_zero",
        goal_threshold: float = 0.1,
    ) -> "HierarchicalReplayBuffer":
        rng, key_idx, key_goal = jax.random.split(rng, 3)

        valid_indices = jnp.array(np.where(~np.array(self.dones, dtype=bool))[0])
        idx = jax.random.choice(key_idx, valid_indices, shape=(batch_size,), replace=True)

        episode_ends = self.episode_end_idx[idx]
        steps_remaining = (episode_ends - idx).astype(jnp.float32)
        goal_offsets = jnp.floor(jax.random.uniform(key_goal, (batch_size,)) * steps_remaining).astype(jnp.int32)
        goal_idx = idx + goal_offsets + 1
        goals = self.observations[goal_idx]

        dist = jnp.linalg.norm(self.next_observations[idx] - goals, axis=-1)
        achieved = dist <= goal_threshold
        if reward_type == "neg_one_zero":
            rewards = jnp.where(achieved, 0.0, -1.0)
        else:  # zero_one
            rewards = jnp.where(achieved, 1.0, 0.0)

        return HierarchicalReplayBuffer(
            observations=self.observations[idx],
            actions=self.actions[idx],
            next_observations=self.next_observations[idx],
            rewards=rewards,
            dones=self.dones[idx],
            subgoal_observations=self.subgoal_observations[idx],
            goals=goals,
            episode_end_idx=self.episode_end_idx[idx],
        )

    @classmethod
    def create(
        cls,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        dones: jnp.ndarray,
        subgoal_steps: int = 20,
    ) -> "HierarchicalReplayBuffer":
        observations = jnp.array(observations)
        actions = jnp.array(actions)
        next_observations = jnp.array(next_observations)
        dones = jnp.array(dones)
        N = len(observations)

        dones_np = np.array(dones, dtype=bool)
        done_positions = np.where(dones_np)[0]
        if len(done_positions) == 0:
            done_positions = np.array([N - 1])

        episode_end_idx = np.zeros(N, dtype=np.int32)
        start = 0
        for end in done_positions:
            episode_end_idx[start : end + 1] = end
            start = end + 1
        if start < N:
            episode_end_idx[start:] = N - 1

        subgoal_idx = np.minimum(np.arange(N) + subgoal_steps, episode_end_idx)
        subgoal_observations = np.array(observations)[subgoal_idx]

        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            subgoal_observations=jnp.array(subgoal_observations),
            goals=jnp.zeros_like(observations),
            episode_end_idx=jnp.array(episode_end_idx),
        )
