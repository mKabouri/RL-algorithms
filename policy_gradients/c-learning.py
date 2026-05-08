"""
"C-Learning: Learning to Achieve Goals via Recursive Classification" by Eysenbach et al.
Link: https://arxiv.org/abs/2011.08909
"""
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
import random

# C-learning is an off-policy goal-conditioned RL algorithm.
# idea: instead of learning a value function, it trains a binary classifier
# that predicts whether goal g will be visited
# from (s, a) under the current policy.
env = gym.make("Pendulum-v1")
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = float(env.action_space.high[0])
goal_dim = obs_dim

device = "cpu" if not torch.cuda.is_available() else "cuda"

net_classifier = nn.Sequential(
    nn.Linear(obs_dim + act_dim + goal_dim, 256),
    nn.ReLU(),
    nn.Linear(256, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
    nn.Sigmoid()
).to(device)

def classifier(state, action, goal):
    inp = torch.cat([state, action, goal], dim=-1)
    return net_classifier(inp).squeeze(-1)


LOG_STD_MIN, LOG_STD_MAX = -20, 2

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(256, act_dim)
        self.log_std_head = nn.Linear(256, act_dim)

    def sample(self, state, goal):
        x = self.trunk(torch.cat([state, goal], dim=-1))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        raw = Normal(mean, log_std.exp()).rsample()
        return torch.tanh(raw) * act_limit

net_policy = PolicyNet().to(device)

def choose_action(obs, goal):
    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    goal_t = torch.from_numpy(goal).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action = net_policy.sample(obs_t, goal_t)
    return action.squeeze(0).cpu().numpy()


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state):
        self.buffer.append((state.copy(), action.copy(), next_state.copy()))

    def sample(self, n):
        batch = random.sample(self.buffer, n)
        s, a, ns = zip(*batch)
        return (np.array(s, dtype=np.float32),
                np.array(a, dtype=np.float32),
                np.array(ns, dtype=np.float32))

    def sample_goals(self, n):
        batch = random.sample(self.buffer, n)
        return np.array([t[2] for t in batch], dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


GAMMA = 0.99
BUFFER_SIZE = 100_000
BATCH_SIZE = 256
NB_ITERATIONS = 1000
WARMUP_STEPS = 1000
EPISODE_LENGTH = 200
UPDATES_PER_EPISODE = 50
LEARNING_RATE = 3e-4

EVAL_GOAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

replay_buffer = ReplayBuffer(BUFFER_SIZE)
classifier_optimizer = torch.optim.Adam(net_classifier.parameters(), lr=LEARNING_RATE)
policy_optimizer = torch.optim.Adam(net_policy.parameters(), lr=LEARNING_RATE)


def compute_classifier_loss(states, actions, next_states, goals):
    s_t = torch.from_numpy(states).to(device)
    a_t = torch.from_numpy(actions).to(device)
    s_next = torch.from_numpy(next_states).to(device)
    g = torch.from_numpy(goals).to(device)

    with torch.no_grad():
        a_next = net_policy.sample(s_next, g)

    c_next = classifier(s_next, a_next, g).clamp(1e-6, 1 - 1e-6).detach()
    w = (c_next / (1.0 - c_next)).clamp(max=20)

    c_pos = classifier(s_t, a_t, s_next).clamp(1e-6, 1 - 1e-6)
    term_a = (1 - GAMMA) * torch.log(c_pos)

    c_goal = classifier(s_t, a_t, g).clamp(1e-6, 1 - 1e-6)
    term_b = torch.log(1 - c_goal)
    term_c = GAMMA * w * torch.log(c_goal)

    return -(term_a + term_b + term_c).mean()


def compute_policy_loss(states, goals):
    s_t = torch.from_numpy(states).to(device)
    g = torch.from_numpy(goals).to(device)
    a_t = net_policy.sample(s_t, g)
    c = classifier(s_t, a_t, g).clamp(1e-6, 1 - 1e-6)
    return -torch.log(c).mean()


def C_learning_algorithm():
    total_reward = []
    training_time = time.time()

    obs, _ = env.reset()
    for _ in range(WARMUP_STEPS):
        action = env.action_space.sample()
        next_obs, _, done, truncated, _ = env.step(action)
        replay_buffer.push(obs, action, next_obs)
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()

    for iteration in range(NB_ITERATIONS):
        iteration_start_time = time.time()

        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(EPISODE_LENGTH):
            action = choose_action(obs, EVAL_GOAL)
            next_obs, reward, done, truncated, _ = env.step(action)
            replay_buffer.push(obs, action, next_obs)
            episode_reward += reward
            obs = next_obs
            if done or truncated:
                obs, _ = env.reset()
                break

        for _ in range(UPDATES_PER_EPISODE):
            states, actions, next_states = replay_buffer.sample(BATCH_SIZE)
            goals = replay_buffer.sample_goals(BATCH_SIZE)

            classifier_optimizer.zero_grad()
            clf_loss = compute_classifier_loss(states, actions, next_states, goals)
            clf_loss.backward()
            classifier_optimizer.step()

            policy_optimizer.zero_grad()
            pol_loss = compute_policy_loss(states, goals)
            pol_loss.backward()
            policy_optimizer.step()

        if iteration % 50 == 0:
            elapsed = time.time() - iteration_start_time
            time_format = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            print(f'iteration: {iteration}\t duration: {time_format}')
            print(f'Episode reward: {episode_reward:.3f}')
            print('-------------------------------------------------')
            print()
        total_reward.append(episode_reward)

    elapsed = time.time() - training_time
    time_format = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    print(f"Training Duration: {time_format}")
    return total_reward


def evaluate(goal=None, render=False):
    if goal is None:
        goal = EVAL_GOAL
    eval_env = gym.make("Pendulum-v1", render_mode="human" if render else None)
    obs, _ = eval_env.reset()
    total_reward = 0
    for _ in range(EPISODE_LENGTH):
        action = choose_action(obs, goal)
        obs, reward, done, truncated, _ = eval_env.step(action)
        total_reward += reward
        if done or truncated:
            break
    eval_env.close()
    return total_reward


def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")


if __name__ == '__main__':
    print(f"Training with {device}")
    net_classifier.train()
    net_policy.train()

    avg_rewards = C_learning_algorithm()
    plot_rewards(avg_rewards, "C-learning training")

    torch.save(net_classifier.state_dict(), "./C_learning_classifier_weights.pt")
    torch.save(net_policy.state_dict(), "./C_learning_policy_weights.pt")

    net_classifier.eval()
    net_policy.eval()

    net_classifier.load_state_dict(torch.load("./C_learning_classifier_weights.pt"))
    net_policy.load_state_dict(torch.load("./C_learning_policy_weights.pt"))
    policy_scores = [evaluate(render=True) for _ in range(10)]
    print("Average score of the policy: ", np.mean(policy_scores))

    env.close()
