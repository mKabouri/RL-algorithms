import numpy as np
import torch as th
from typing import Any, NamedTuple

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor

class HERBuffer:
    def __init__(
        self,
        buffer_size: int,
        envs,
        action_space,
        device="cpu",
        k_future=4
    ):
        self.buffer_size = buffer_size
        self.n_envs = envs.num_envs
        self.device = device
        self.k_future = k_future

        unwrapped_env = envs.envs[0].unwrapped
        self.compute_reward = unwrapped_env.compute_reward

        dict_space = unwrapped_env.observation_space
        self.ag_dim = dict_space.spaces["achieved_goal"].shape[0]
        self.dg_dim = dict_space.spaces["desired_goal"].shape[0]
        
        self.ag_start = 0
        self.ag_end = self.ag_dim
        self.dg_start = self.ag_dim
        self.dg_end = self.ag_dim + self.dg_dim

        self.obs_shape = envs.single_observation_space.shape
        self.action_dim = action_space.shape[0]

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        self.pos = 0
        self.full = False

        # store current episode
        self.episode_transitions = []

    def add(self, obs, next_obs, action, reward, done, infos):
        self.episode_transitions.append((obs.copy(), next_obs.copy(), action.copy(), reward.copy(), done.copy()))

        if done[0] or (infos and "final_info" in infos):
            self._relabel()

    def _store_transition(self, obs, next_obs, action, reward, done):
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _relabel(self):
        ep_len = len(self.episode_transitions)
        
        for t in range(ep_len):
            obs, next_obs, action, reward, done = self.episode_transitions[t]
            
            self._store_transition(obs, next_obs, action, reward, done)
            
            # HER future strategy
            future_indices = np.random.randint(t, ep_len, size=self.k_future)
            
            for f_idx in future_indices:
                future_obs, future_next_obs, _, _, _ = self.episode_transitions[f_idx]
                # relabelling
                new_goal = future_next_obs[0, self.ag_start:self.ag_end]
                her_obs = obs.copy()
                her_next_obs = next_obs.copy()
                her_obs[0, self.dg_start:self.dg_end] = new_goal
                her_next_obs[0, self.dg_start:self.dg_end] = new_goal
                
                current_ag = her_next_obs[0, self.ag_start:self.ag_end]
                
                her_reward = self.compute_reward(current_ag, new_goal, {})
                her_reward = np.array([her_reward], dtype=np.float32)
                
                self._store_transition(her_obs, her_next_obs, action, her_reward, done)
                
        self.episode_transitions = []

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        env_indices = np.zeros(batch_size, dtype=int)
        
        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            self.next_observations[batch_inds, env_indices, :],
            self.dones[batch_inds, env_indices].reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        
        return ReplayBufferSamples(*tuple(map(lambda x: th.tensor(x, device=self.device), data)))
