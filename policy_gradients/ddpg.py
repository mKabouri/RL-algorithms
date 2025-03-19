"""
"Continuous control with deep reinforcement learning"
by Timothy P. Lillicrap et al.
Link: https://arxiv.org/pdf/1509.02971.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import gymnasium as gym
import numpy as np
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque, namedtuple


device = "cpu" if not torch.cuda.is_available() else "cuda"


# DDPG is an off-policy algorithm
env = gym.make("Pendulum-v1")
obs, _ = env.reset()

obs_dimension = env.observation_space.shape[0]
action_space = env.action_space
action_size = action_space.shape[0]

action_noise = OrnsteinUhlenbeckActionNoise(
    mean=np.array(0.0), sigma=0.2*np.ones(action_size)
)

# NORMAL_SCALAR=0.15
# # Used for exploration
# # I didn't use the same as the original article
# def get_random_noise():
#     return torch.tensor(np.random.randn(action_size)*NORMAL_SCALAR).to(device)

# net_actor is a neural network representing the actor (the policy).
class Actor(nn.Module):
    def __init__(self, obs_dimension, action_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )

    def forward(self, obs):
        return 2*self.net(obs)
net_actor = Actor(obs_dimension, action_size).to(device)

# net_critic is a neural network representing the critic (the action-value function).
net_critic = nn.Sequential(
    nn.Linear(obs_dimension+action_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# Target networks from DDPG original paper
# We initialize them with the same weights
# as the networks critic and actor above
target_actor = Actor(obs_dimension, action_size).to(device)
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
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = net_actor(state).cpu().numpy()
        # logits = net_actor(state)
        # action = torch.tanh(logits).cpu().numpy()  # Range [-1, 1]
        # Scale action to environment's action range
        # action = action * action_space.high
        action = action + action_noise()
        action = np.clip(action, action_space.low, action_space.high)
        return torch.tensor(action, dtype=torch.float32).to(device)

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
EPISODES=5000
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
    if len(REPLAY_BUFFER) < BATCH_SIZE:
        return

    # Sample a batch from replay memory
    get_batch = REPLAY_BUFFER.sample(BATCH_SIZE)
    batch = Transition(*zip(*get_batch))

    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(batch.action), dtype=torch.float32).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)

    with torch.no_grad():
        # Compute target Q values
        target_actions = torch.tanh(target_actor(next_state_batch))
        target_Q_values = target_critic(torch.cat([next_state_batch, target_actions], dim=1)).squeeze(-1)
        target = reward_batch + GAMMA*target_Q_values

    # Update Critic
    current_Q_values = net_critic(torch.cat([state_batch, action_batch], dim=1)).squeeze(-1)
    critic_loss = F.mse_loss(current_Q_values, target)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update Actor
    actor_optimizer.zero_grad()
    current_actions = torch.tanh(net_actor(state_batch))
    actor_loss = -net_critic(torch.cat([state_batch, current_actions], dim=1)).mean()
    actor_loss.backward()
    actor_optimizer.step()

    # Soft update of target networks
    for target_param, param in zip(target_actor.parameters(), net_actor.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
    for target_param, param in zip(target_critic.parameters(), net_critic.parameters()):
        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

def DDPG_algorithm():
    total_rewards = []
    best_reward = -np.inf
    for episode in tqdm(range(EPISODES)):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = policy(obs)
            new_obs, reward, done, truncated, _ = env.step(action.cpu().numpy())
            total_reward += reward
            REPLAY_BUFFER.push(np.array(obs), action.cpu().numpy(), reward, np.array(new_obs))
            obs = new_obs

            # Perform one training step
            if len(REPLAY_BUFFER) >= BATCH_SIZE:
                one_train_step()

            if done or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            save_weights()
        total_rewards.append(total_reward)

        if episode % 5 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    return total_rewards

# Function to plot total reward per iteration of training
def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('EPISODE')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")

def save_weights():
    torch.save(net_actor.state_dict(), "best_actor_weights.pth")
    torch.save(net_critic.state_dict(), "best_critic_weights.pth")

def load_weights():
    net_actor.load_state_dict(torch.load("best_actor_weights.pth"))
    net_critic.load_state_dict(torch.load("best_critic_weights.pth"))

def evaluate():
    load_weights()
    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = policy(obs)
        obs, reward, done, _, _ = env.step(action.cpu().numpy())
        total_reward += reward
    print(f"Total reward: {total_reward}")

if __name__ == '__main__':
    print(f"Training with {device}")
    total_rewards = DDPG_algorithm()
    plot_rewards(total_rewards, "Pendulum DDPG")
