import functools
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct


def relabel_rewards(
    next_observations: jnp.ndarray,
    goals: jnp.ndarray,
    reward_type: Literal["neg_one_zero", "zero_one"],
    goal_threshold: float,
) -> jnp.ndarray:
    dist = jnp.linalg.norm(next_observations - goals, axis=-1)
    achieved = dist <= goal_threshold
    if reward_type == "neg_one_zero":
        return jnp.where(achieved, 0.0, -1.0)
    else:  # zero_one
        return jnp.where(achieved, 1.0, 0.0)


def make_episode_indices(dones: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    dones_np = np.array(dones, dtype=bool)
    done_positions = np.where(dones_np)[0]
    if len(done_positions) == 0:
        done_positions = np.array([len(dones_np) - 1])

    episode_end_idx = np.zeros(len(dones_np), dtype=np.int32)
    start = 0
    for end in done_positions:
        episode_end_idx[start : end + 1] = end
        start = end + 1
    if start < len(dones_np):
        episode_end_idx[start:] = len(dones_np) - 1

    valid_indices = np.where(np.arange(len(dones_np)) < episode_end_idx)[0].astype(np.int32)
    return jnp.array(episode_end_idx), jnp.array(valid_indices)


class ReplayBuffer(struct.PyTreeNode):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return self.observations.shape[0]

    @functools.partial(jax.jit, static_argnames=("batch_size",))
    def sample(self, batch_size: int, rng: jax.random.PRNGKey) -> dict[str, jnp.ndarray]:
        idx = jax.random.choice(rng, len(self), shape=(batch_size,), replace=False)
        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_observations[idx],
            "dones": self.dones[idx],
        }

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


class GCRLReplayBuffer(struct.PyTreeNode):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    episode_end_idx: jnp.ndarray
    valid_indices: jnp.ndarray

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return self.observations.shape[0]

    def sample_goals(self, idx: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        batch_size = idx.shape[0]
        episode_ends = self.episode_end_idx[idx]
        steps_remaining = episode_ends - idx
        goal_offsets = jnp.floor(jax.random.uniform(rng, (batch_size,)) * steps_remaining).astype(jnp.int32)
        goal_idx = idx + goal_offsets + 1
        return self.observations[goal_idx]

    @functools.partial(jax.jit, static_argnames=("batch_size", "reward_type"))
    def sample(
        self,
        batch_size: int,
        rng: jax.random.PRNGKey,
        reward_type: Literal["neg_one_zero", "zero_one"] = "neg_one_zero",
        goal_threshold: float = 0.1,
    ) -> dict[str, jnp.ndarray]:
        rng, key_idx, key_goal = jax.random.split(rng, 3)
        idx = jax.random.choice(key_idx, self.valid_indices, shape=(batch_size,), replace=True)
        goals = self.sample_goals(idx, key_goal)
        rewards = relabel_rewards(self.next_observations[idx], goals, reward_type, goal_threshold)

        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "rewards": rewards,
            "dones": self.dones[idx],
            "goals": goals,
        }

    @classmethod
    def create(
        cls,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        next_observations: jnp.ndarray,
        dones: jnp.ndarray,
        subgoal_steps: int = 0, # ignored 
    ) -> "GCRLReplayBuffer":
        observations = jnp.array(observations)
        actions = jnp.array(actions)
        next_observations = jnp.array(next_observations)
        dones = jnp.array(dones)
        episode_end_idx, valid_indices = make_episode_indices(dones)

        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            episode_end_idx=episode_end_idx,
            valid_indices=valid_indices,
        )


class HierarchicalReplayBuffer(struct.PyTreeNode):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    subgoal_observations: jnp.ndarray
    episode_end_idx: jnp.ndarray
    valid_indices: jnp.ndarray

    def __getitem__(self, name):
        return getattr(self, name)

    def __len__(self):
        return self.observations.shape[0]

    @functools.partial(jax.jit, static_argnames=("batch_size", "reward_type"))
    def sample(
        self,
        batch_size: int,
        rng: jax.random.PRNGKey,
        reward_type: Literal["neg_one_zero", "zero_one"] = "neg_one_zero",
        goal_threshold: float = 0.1,
    ) -> dict[str, jnp.ndarray]:
        rng, key_idx, key_goal = jax.random.split(rng, 3)

        idx = jax.random.choice(key_idx, self.valid_indices, shape=(batch_size,), replace=True)

        episode_ends = self.episode_end_idx[idx]
        steps_remaining = (episode_ends - idx).astype(jnp.float32)
        goal_offsets = jnp.floor(jax.random.uniform(key_goal, (batch_size,)) * steps_remaining).astype(jnp.int32)
        goal_idx = idx + goal_offsets + 1
        goals = self.observations[goal_idx]

        rewards = relabel_rewards(self.next_observations[idx], goals, reward_type, goal_threshold)

        return {
            "observations": self.observations[idx],
            "actions": self.actions[idx],
            "next_observations": self.next_observations[idx],
            "rewards": rewards,
            "dones": self.dones[idx],
            "subgoal_observations": self.subgoal_observations[idx],
            "goals": goals,
        }

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

        episode_end_idx, valid_indices = make_episode_indices(dones)

        subgoal_idx = np.minimum(np.arange(N) + subgoal_steps, np.array(episode_end_idx))
        subgoal_observations = np.array(observations)[subgoal_idx]

        return cls(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            subgoal_observations=jnp.array(subgoal_observations),
            episode_end_idx=episode_end_idx,
            valid_indices=valid_indices,
        )
