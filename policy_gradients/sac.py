"""
"Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
Learning with a Stochastic Actor"
by Tuomas Haarnoja et al.
Link: https://arxiv.org/pdf/1801.01290.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np
import random
from torch.optim import Adam
from tqdm import tqdm
from collections import deque, namedtuple
import matplotlib.pyplot as plt


env = gym.make("Pendulum-v1")
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high[0]

print(f"Observation space: {obs_dim}, Action space: {action_dim}")
print(f"Action bounds: {action_bound}")

device = "cpu" if not torch.cuda.is_available() else "cuda"

# We define four networks like in the original paper:
# Policy approximator
# State-action value approximator (Q value)
# State value approximator
# Target state value approximator
class Actor(nn.Module):
    def __init__(self, obs_dimension, action_size):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(64, action_size)
        self.log_std_layer = nn.Linear(64, action_size)

    def forward(self, obs):
        x = self.net(obs)
        mu = self.mu_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample_action(self, obs):
        mu, std = self.forward(obs)
        normal_dist = torch.distributions.Normal(mu, std)
        action = normal_dist.rsample()
        squashed_action = torch.tanh(action)*action_bound
        return squashed_action, normal_dist.log_prob(action).sum(dim=-1)

class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(obs_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

actor = Actor(obs_dim, action_dim).to(device)
critic_1 = Critic(obs_dim, action_dim).to(device)
critic_2 = Critic(obs_dim, action_dim).to(device)
target_critic_1 = Critic(obs_dim, action_dim).to(device)
target_critic_2 = Critic(obs_dim, action_dim).to(device)

target_critic_1.load_state_dict(critic_1.state_dict())
target_critic_2.load_state_dict(critic_2.state_dict())
####################################################
# Replay Memory from:
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
######################################################################
# Hyperparameters
NB_ITERATION = 10000
BUFFER_SIZE = 1000
BATCH_SIZE = 100
LR_ACTOR = 3e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
TAU = 5e-2
ALPHA = 0.2

actor_optimizer = Adam(actor.parameters(), lr=LR_ACTOR)
critic_1_optimizer = Adam(critic_1.parameters(), lr=LR_CRITIC)
critic_2_optimizer = Adam(critic_2.parameters(), lr=LR_CRITIC)
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# This function role is updating the networks
def update():
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    batch = Transition(*zip(*replay_buffer.sample(BATCH_SIZE)))
    state = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action = torch.tensor(np.array(batch.action), dtype=torch.float32).to(device)
    reward = torch.tensor(batch.reward, dtype=torch.float32).to(device).unsqueeze(1)
    next_state = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)
    done = torch.tensor(batch.done, dtype=torch.float32).to(device).unsqueeze(1)

    # Target Q-value
    with torch.no_grad():
        next_action, next_log_prob = actor.sample_action(next_state)
        q1_target = target_critic_1(next_state, next_action)
        q2_target = target_critic_2(next_state, next_action)
        min_q_target = torch.min(q1_target, q2_target) - ALPHA*next_log_prob.unsqueeze(1)
        q_target = reward + (1-done)*GAMMA*min_q_target

    # Critic update
    q1 = critic_1(state, action)
    q2 = critic_2(state, action)
    loss_critic1 = F.mse_loss(q1, q_target)
    loss_critic2 = F.mse_loss(q2, q_target)

    critic_1_optimizer.zero_grad()
    loss_critic1.backward()
    critic_1_optimizer.step()

    critic_2_optimizer.zero_grad()
    loss_critic2.backward()
    critic_2_optimizer.step()

    # Actor update
    new_action, log_prob = actor.sample_action(state)
    q1_new = critic_1(state, new_action)
    q2_new = critic_2(state, new_action)
    min_q_new = torch.min(q1_new, q2_new)
    loss_actor = (ALPHA*log_prob - min_q_new).mean()

    actor_optimizer.zero_grad()
    loss_actor.backward()
    actor_optimizer.step()

    # Target update
    for target_param, param in zip(target_critic_1.parameters(), critic_1.parameters()):
        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)

    for target_param, param in zip(target_critic_2.parameters(), critic_2.parameters()):
        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)

def SAC_algorithm():
    total_rewards = []
    best_reward = -np.inf
    for episode in tqdm(range(NB_ITERATION)):
        state, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = actor.sample_action(torch.tensor(state, dtype=torch.float32).to(device))
            next_state, reward, done, truncated, _ = env.step(action.cpu().detach().numpy())
            total_reward += reward
            replay_buffer.push(state, action.cpu().detach().numpy(), reward, next_state, done or truncated)
            state = next_state
            if done or truncated:
                break

        if total_reward > best_reward:
            best_reward = total_reward
            save_best_weights()

        if len(replay_buffer) >= BATCH_SIZE:
            update()

        total_rewards.append(total_reward)
        if episode%10 == 0:
            print(f"Episode {episode}, Total reward: {total_reward}")
    return total_rewards

def save_best_weights():
    torch.save(actor.state_dict(), "best_sac_weights.pth")

def load_best_weights():
    actor.load_state_dict(torch.load("best_sac_weights.pth"))

# Function to plot total reward per iteration of training
def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('EPISODE')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")

def evaluate():
    load_best_weights()
    env = gym.make("Pendulum-v1", render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = actor.sample_action(torch.tensor(obs, dtype=torch.float32).to(device))
        obs, _, done, _, _ = env.step(action.cpu().detach().numpy())
        env.render()

if __name__ == '__main__':
    print(f"Training with {device}")
    total_rewards = SAC_algorithm()
    plot_rewards(total_rewards, "Pendulum SAC")
    evaluate()
