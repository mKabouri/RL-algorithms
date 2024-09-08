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
from collections import deque, namedtuple
import matplotlib.pyplot as plt


env = gym.make("CartPole-v1")
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

device = "cpu" if not torch.cuda.is_available() else "cuda"

# We define four networks like in the original paper:
# Policy approximator
# State-action value approximator (Q value)
# State value approximator
# Target state value approximator
net_actor = nn.Sequential(
    nn.Linear(obs_dim, 64), 
    nn.ReLU(),
    nn.Linear(64, n_acts)
).to(device)

# We add plus 1 for action (one action)
net_state_action_value = nn.Sequential(
    nn.Linear(obs_dim+1, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

net_state_value = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

target_state_value = nn.Sequential(
    nn.Linear(obs_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)
####################################################

# I use a stochastic policy
def policy(state):
    state = state.to(device)
    logits = net_actor(state)
    return Categorical(logits=logits)

# exploration_noise = 0.1
# def choose_action(state):
#     state = torch.tensor(state).to(device)
#     action = policy(state).sample().item()
#     action = np.clip(action+exploration_noise*np.random.randn(), 0, n_acts-1)
#     return int(action)
def choose_action(state):
    state = torch.tensor(state).to(device)
    return policy(state).sample().item()

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
# Hyperparameters
NB_ITERATION=10000
BUFFER_CAPACITY=5000
BATCH_SIZE=60
LEARNING_RATE_ACTOR=1e-4
LEARNING_RATE_STATE_ACTION=1e-3
LEARNING_RATE_STATE=1e-3
GAMMA=0.99
TAU=1e-3
ALPHA=0.9

actor_optimizer = Adam(net_actor.parameters(), lr=LEARNING_RATE_ACTOR)
state_action_optimizer = Adam(net_state_action_value.parameters(), lr=LEARNING_RATE_STATE_ACTION)
state_optimizer = Adam(net_state_value.parameters(), lr=LEARNING_RATE_STATE)
REPLAY_BUFFER = ReplayMemory(capacity=BUFFER_CAPACITY)

# This function role is updating the networks
def one_train_step():
    # Sample batch from REPLAY_BUFFER
    get_batch = REPLAY_BUFFER.sample(BATCH_SIZE)
    batch = Transition(*zip(*get_batch))

    state_batch = torch.tensor(np.array(batch.state), dtype=torch.float32).to(device)
    action_batch = torch.tensor(np.array(batch.action), dtype=torch.int64).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(device)
    next_state_batch = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(device)

    # Update state value network
    # I try to calculate the loss: equation (5) of the original paper
    current_values = net_state_value(state_batch)
    current_Q_values = net_state_action_value(torch.cat([state_batch, action_batch.unsqueeze(1)], dim=-1))
    entropy_term = Categorical(logits=net_actor(state_batch)).log_prob(action_batch)
    target_values = current_Q_values.squeeze(1) - ALPHA*entropy_term

    state_optimizer.zero_grad()
    value_loss = F.mse_loss(current_values.squeeze(1), target_values)
    value_loss.backward()
    state_optimizer.step()

    # Update action-state value network
    # I try to calculate the loss: equation (7) of the original paper
    with torch.no_grad():
        target_next_value = target_state_value(next_state_batch)
    target_Q_values = reward_batch + GAMMA*target_next_value.squeeze(1)

    current_Q_values = net_state_action_value(torch.cat([state_batch, action_batch.unsqueeze(1)], dim=-1))
    state_action_optimizer.zero_grad()
    state_action_value_loss = F.mse_loss(current_Q_values.squeeze(1), target_Q_values)
    state_action_value_loss.backward()
    state_action_optimizer.step()

    # Update policy network (actor network)
    # I try to calculate the loss: equation (12) of the original paper
    entropy_term = Categorical(logits=net_actor(state_batch)).log_prob(action_batch)
    current_Q_values = net_state_action_value(torch.cat([state_batch, action_batch.unsqueeze(1)], dim=-1))
    
    actor_optimizer.zero_grad()
    actor_loss = (ALPHA*entropy_term - current_Q_values).mean()
    actor_loss.backward()
    actor_optimizer.step()

    # Update target networks
    for target_param, param in zip(target_state_value.parameters(), net_state_value.parameters()):
        target_param.data.copy_(TAU*param.data + (1-TAU)*target_param.data)

def SAC_algorithm():
    total_rewards = []
    for iteration in range(NB_ITERATION):
        obs, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = choose_action(obs)
            new_obs, reward, done, truncated, _ = env.step(action)

            total_reward += reward
            # Store in the replay buffer
            REPLAY_BUFFER.push(np.array(obs), action, reward, np.array(new_obs))
            
            obs = new_obs

            if done or truncated:
                break

        total_rewards.append(total_reward)
        # Different from DDPG. Here, I do optimization after the episode.
        if len(REPLAY_BUFFER) >= BATCH_SIZE:
            # For clarity of the code we used this function call
            one_train_step()

        if iteration%10 == 0:
            print(f"Iteration: {iteration}, total reward: {total_reward}")
            print("--------------------------------------------")
            print()
    return total_rewards


# Function to plot total reward per iteration of training
def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('EPISODE')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")

if __name__ == '__main__':
    print(f"Training with {device}")
    total_rewards = SAC_algorithm()
    plot_rewards(total_rewards, "CartPole on SAC")

