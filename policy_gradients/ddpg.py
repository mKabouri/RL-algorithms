"""
My source for this implementation is:
* Original paper: "Continuous control with deep reinforcement learning"
by Timothy P. Lillicrap et al.
at: https://arxiv.org/pdf/1509.02971.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple
import time


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = "cpu" if not torch.cuda.is_available() else "cuda"


# DDPG is an off-policy algorithm
# I want to solve Lunar Lander with continuous actions
env = gym.make("LunarLander-v2", continuous=True)
obs, _ = env.reset()

obs_dimension = env.observation_space.shape[0]
action_space = env.action_space
action_size = len(action_space.high)

# net_actor is a neural network representing the actor (the policy).
net_actor = nn.Sequential(
    nn.Linear(obs_dimension, 64), 
    nn.ReLU(),
    nn.Linear(64, action_size)
).to(device)

# net_critic is a neural network representing the critic (the action-value function).
net_critic = nn.Sequential(
    nn.Linear(obs_dimension+action_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Target networks from DDPG original paper
# We initialize them with the same weights
# as the networks critic and actor above
target_actor = nn.Sequential(
    nn.Linear(obs_dimension, 64), 
    nn.ReLU(),
    nn.Linear(64, action_size)
).to(device)
target_actor.load_state_dict(net_actor.state_dict())

target_critic = nn.Sequential(
    nn.Linear(obs_dimension+action_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)
target_critic.load_state_dict(net_critic.state_dict())


# In DDPG we learn a deterministic policies
# Returns an action given an observation
def policy(observation):
    observation = torch.tensor(observation).to(device)
    logits = net_actor(observation)
    action = torch.argmax(F.softmax(logits, dim=-1))
    return action


NORMAL_SCALAR=0.25

# Used for exploration
# I didn't use the same as the original article
def get_random_noise():
    return torch.tensor(np.random.randn(action_size)*NORMAL_SCALAR).to(device)

######################################################################
# Replay Memory from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
######################################################################

# Hyperparameters (inspired from section 7 of the original paper)
EPISODES=1000
BUFFER_CAPACITY=500
BATCH_SIZE=50
LEARNING_RATE_ACTOR=1e-4
LEARNING_RATE_CRITIC=1e-3
GAMMA=1
TAU = 1e-3

actor_optimizer = Adam(net_actor.parameters(), lr=LEARNING_RATE_ACTOR)
critic_optimizer = Adam(net_critic.parameters(), lr=LEARNING_RATE_CRITIC)

REPLAY_BUFFER = ReplayMemory(capacity=BUFFER_CAPACITY)

# For clarity of the code, we used this function call
def one_train_step():
    get_batch = REPLAY_BUFFER.sample(BATCH_SIZE)
    batch = Transition(*zip(*get_batch))

    state_batch = torch.tensor(batch.state, dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.float32).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32).to(device)

    current_Q_values = net_critic(torch.cat([state_batch, action_batch], dim=-1))


def DDPG_algorithm():
    # For example, I will update the target networks
    # each 20 time step
    target_update_counter = 0
    total_rewards = []
    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # We select an action according to current policy and random noise
            action = torch.clamp(policy(obs) + get_random_noise(),
                                 torch.tensor(action_space.low).to(device),
                                 torch.tensor(action_space.high).to(device))
            
            new_obs, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward
            # Store in the replay buffer
            REPLAY_BUFFER.push(obs, action, reward, new_obs)

            obs = new_obs
            if len(REPLAY_BUFFER) >= BATCH_SIZE:
                # For clarity of the code we used this function call
                one_train_step()

                target_update_counter += 1
                if target_update_counter%20 == 0:
                    # Update target networks
                    for target_param, param in zip(target_actor.parameters(), net_actor.parameters()):
                        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
                    for target_param, param in zip(target_critic.parameters(), net_critic.parameters()):
                            target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
        
        if episode%5==0:
            print(f"Episode: {episode}, total reward: {total_reward}")
            print("------------------------------")
            print()

        total_rewards.append(total_reward)
    return total_rewards

# Function to plot average reward per iteration of training
def plot_rewards(avg_rewards, title):
    plt.plot(avg_rewards)
    plt.title(title)
    plt.xlabel('EPISODE')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")

if __name__ == '__main__':
    print(f"Training with {device}")
    DDPG_algorithm()

