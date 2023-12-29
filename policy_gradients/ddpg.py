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
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple


device = "cpu" if not torch.cuda.is_available() else "cuda"


# DDPG is an off-policy algorithm
# I want to solve Lunar Lander with continuous actions
env = gym.make("LunarLanderContinuous-v2")
obs, _ = env.reset()

obs_dimension = env.observation_space.shape[0]
action_space = env.action_space
action_size = len(action_space.high)

# From: stable_baseline3.common.noise
action_noise = OrnsteinUhlenbeckActionNoise(mean=0.15*np.ones(action_size), sigma=0.2*np.ones(action_size))

# NORMAL_SCALAR=0.15
# # Used for exploration
# # I didn't use the same as the original article
# def get_random_noise():
#     return torch.tensor(np.random.randn(action_size)*NORMAL_SCALAR).to(device)

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
def policy(state):
    with torch.no_grad():
        state = torch.tensor(state).to(device)
        logits = net_actor(state)
        action = torch.tanh(logits)
        action = action.cpu().detach().numpy() + action_noise()
        return torch.tensor(action).to(device)

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
EPISODES=50000
BUFFER_CAPACITY=5000
BATCH_SIZE=64
LEARNING_RATE_ACTOR=1e-4
LEARNING_RATE_CRITIC=1e-3
GAMMA=0.99
TAU = 1e-3

actor_optimizer = Adam(net_actor.parameters(), lr=LEARNING_RATE_ACTOR)
critic_optimizer = Adam(net_critic.parameters(), lr=LEARNING_RATE_CRITIC)

REPLAY_BUFFER = ReplayMemory(capacity=BUFFER_CAPACITY)

# For clarity of the code, we used this function call
# It update networks by calculating the loss for actor and critic
def one_train_step():
    get_batch = REPLAY_BUFFER.sample(BATCH_SIZE)
    batch = Transition(*zip(*get_batch))

    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)

    # I compute target actions and target Q values for the loss
    with torch.no_grad(): # I don't compute gradients for actor and critic targets
        target_actor_actions = target_actor(state_batch).detach()
        target_Q_values = target_critic(torch.cat([next_state_batch, target_actor_actions], dim=1)).squeeze(-1).detach()

    # I compute current Q values as for the loss (squeeze just to match dimension below (I got a warn))
    current_Q_values = net_critic(torch.cat([state_batch, action_batch], dim=1)).squeeze(-1)

    # I update the critic
    target = reward_batch+GAMMA*target_Q_values
    critic_optimizer.zero_grad()
    # I compute the loss for the critic
    critic_loss = F.mse_loss(current_Q_values, target)
    critic_loss.backward()
    critic_optimizer.step()

    # I update the actor with equation 6 of the original paper 
    actor_optimizer.zero_grad()
    current_actions = net_actor(state_batch)
    # I compute the loss for the actor
    actor_loss = -net_critic(torch.cat([state_batch, current_actions], dim=1)).squeeze(-1).mean()
    actor_loss.backward()
    actor_optimizer.step()

    # Update target networks
    for target_param, param in zip(target_actor.parameters(), net_actor.parameters()):
        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)
    for target_param, param in zip(target_critic.parameters(), net_critic.parameters()):
        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)

def DDPG_algorithm():
    # For example, I will update the target networks
    # each 20 time step
    total_rewards = []
    for episode in range(EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            # We select an action according to current policy and random noise
            # action = torch.clamp(policy(obs) + get_random_noise(),
            #                      torch.tensor(action_space.low).to(device),
            #                      torch.tensor(action_space.high).to(device))

            # action = policy(obs) + get_random_noise()
            action = policy(obs)            
            new_obs, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward
            # Store in the replay buffer
            REPLAY_BUFFER.push(np.array(obs), action.cpu().detach().numpy(), reward, new_obs)
            
            if done or truncated:
                break

            obs = new_obs
            if len(REPLAY_BUFFER) >= BATCH_SIZE:
                # For clarity of the code we used this function call
                one_train_step()

        if episode%5==0 and len(REPLAY_BUFFER) >= BATCH_SIZE:
            print(f"Episode: {episode}, total reward: {total_reward}")
            print("------------------------------")
            print()

        total_rewards.append(total_reward)
    return total_rewards

# Function to plot total reward per iteration of training
def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('EPISODE')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")

def save_weights():...

if __name__ == '__main__':
    print(f"Training with {device}")
    total_rewards = DDPG_algorithm()
    plot_rewards(total_rewards, "Continuous LunarLander DDPG")
