import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from collections import namedtuple, deque
import random

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    map_name="4x4",
)

NB_STATES = env.observation_space.n
NB_ACTIONS = env.action_space.n

print(f'Number of states: {NB_STATES}')
print(f'Number of actions: {NB_ACTIONS}')
print(f'Action space: {env.action_space}')
print(f'Observation space: {env.observation_space}')

# Exploration and learning parameters
MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1
EPISODES = 5000
TIMEOUT = 100

# Epsilon and learning rate decay
def update_explore_rate(episode):
    return max(MIN_EXPLORE_RATE, min(1, 1.0-np.log10((episode+1)/25)))

def update_learning_rate(episode):
    return max(MIN_LEARNING_RATE, min(0.5, 1.0-np.log10((episode+1)/50)))

def run_episode(env, policy=None, render=False):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = env.action_space.sample() if policy is None else policy[obs]
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
    return total_reward

# Q-Learning algorithm
def Q_learning():
    Qvalue = np.zeros((NB_STATES, NB_ACTIONS))
    for episode in range(EPISODES):
        obs, _ = env.reset()
        learning_rate = update_learning_rate(episode)
        epsilon = update_explore_rate(episode)
        total_reward = 0
        done = False
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qvalue[obs])
            new_obs, reward, done, _, _ = env.step(action)
            Qvalue[obs, action] += learning_rate*(
                reward+np.max(Qvalue[new_obs])-Qvalue[obs, action]
            )
            total_reward += reward
            obs = new_obs

        if episode % 500 == 0:
            print(f'Q-Learning Episode #{episode} -- Total reward: {total_reward}')
    return Qvalue

# SARSA algorithm
def SARSA():
    Qvalue = np.zeros((NB_STATES, NB_ACTIONS))
    for episode in range(EPISODES):
        obs, _ = env.reset()
        learning_rate = update_learning_rate(episode)
        epsilon = update_explore_rate(episode)
        total_reward = 0
        done = False
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Qvalue[obs])
        while not done:
            new_obs, reward, done, _, _ = env.step(action)
            if np.random.rand() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Qvalue[new_obs])
            Qvalue[obs, action] += learning_rate*(
                reward+Qvalue[new_obs, next_action]-Qvalue[obs, action]
            )
            total_reward += reward
            obs, action = new_obs, next_action
        if episode % 500 == 0:
            print(f'SARSA Episode #{episode} -- Total reward: {total_reward}')
    return Qvalue

# Double Q-Learning algorithm
def double_Q_learning():
    Qvalue1 = np.zeros((NB_STATES, NB_ACTIONS))
    Qvalue2 = np.zeros((NB_STATES, NB_ACTIONS))
    for episode in range(EPISODES):
        obs, _ = env.reset()
        learning_rate = update_learning_rate(episode)
        epsilon = update_explore_rate(episode)
        total_reward = 0
        done = False

        while not done:
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qvalue1[obs]+Qvalue2[obs])
            new_obs, reward, done, _, _ = env.step(action)
            if np.random.rand() < 0.5:
                Qvalue1[obs, action] += learning_rate*(
                    reward+Qvalue2[new_obs, np.argmax(Qvalue1[new_obs])]-Qvalue1[obs, action]
                )
            else:
                Qvalue2[obs, action] += learning_rate*(
                    reward+Qvalue1[new_obs, np.argmax(Qvalue2[new_obs])]-Qvalue2[obs, action]
                )
            total_reward += reward
            obs = new_obs
        if episode % 500 == 0:
            print(f'Double Q-Learning Episode #{episode} -- Total reward: {total_reward}')
    return Qvalue1, Qvalue2

# ReplayMemory from a pytorch tutorial
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def __len__(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Q-learning with experience replay
def experience_replay_Q_learning():
    Qvalue = np.zeros((NB_STATES, NB_ACTIONS))
    memory = ReplayMemory(capacity=250)
    batch_size = 64

    for episode in range(EPISODES):
        obs, _ = env.reset()
        epsilon = update_explore_rate(episode)
        total_reward = 0
        done = False
        while not done:
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(Qvalue[obs])
            new_obs, reward, done, _, _ = env.step(action)
            memory.push(obs, action, new_obs, reward)
            if len(memory) >= batch_size:
                transitions = memory.sample(batch_size)
                for state, action, next_state, reward in transitions:
                    Qvalue[state, action] += update_learning_rate(episode)*(
                        reward+np.max(Qvalue[next_state])-Qvalue[state, action]
                    )
            total_reward += reward
            obs = new_obs
        if episode % 500 == 0:
            print(f'Experience Replay Q-Learning Episode #{episode} -- Total reward: {total_reward}')
    return Qvalue

def n_step_Q_learning(n=5):
    Qvalue = np.zeros((NB_STATES, NB_ACTIONS))
    for episode in range(EPISODES):
        obs, _ = env.reset()
        epsilon = update_explore_rate(episode)
        learning_rate = update_learning_rate(episode)
        total_reward = 0
        done = False

        # Store states, actions, and rewards
        states, actions, rewards = [obs], [], []
        t = 0
        while not done:
            # Epsilon-greedy action selection
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Qvalue[obs])
            new_obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            actions.append(action)
            rewards.append(reward)
            states.append(new_obs)
            if len(rewards) >= n or done:
                n_step_return = n_step_lookahead(rewards, Qvalue, states, n, done)
                Qvalue[states[0], actions[0]] += learning_rate*(
                    n_step_return-Qvalue[states[0], actions[0]]
                )
                states.pop(0)
                actions.pop(0)
                rewards.pop(0)
            obs = new_obs
            t += 1
        if episode % 500 == 0:
            print(f'n-Step Q-Learning Episode #{episode} -- Total reward: {total_reward}')
    return Qvalue

def n_step_lookahead(rewards, Qvalue, states, n, done):
    """
    Compute n-step return
    """
    n_step_return = sum(rewards[:n])
    if not done and len(rewards) >= n:
        n_step_return += np.max(Qvalue[states[n]])
    return n_step_return

def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()

if __name__ == '__main__':
    # Learning
    Qvalue = Q_learning()
    Qvalue1, Qvalue2 = double_Q_learning()
    QvalueReplay = experience_replay_Q_learning()
    n_step_Qvalue = n_step_Q_learning()
    # Policies
    policy_qlearning = np.argmax(Qvalue, axis=1)
    policy_double_qlearning = np.argmax(Qvalue1, axis=1)
    policy_replay = np.argmax(QvalueReplay, axis=1)
    policy_nsQlearning = np.argmax(n_step_Qvalue, axis=1)
    # Evaluation
    episodes = 500
    rewards_qlearning = [run_episode(env, policy_qlearning) for _ in range(episodes)]
    rewards_double_qlearning = [run_episode(env, policy_double_qlearning) for _ in range(episodes)]
    rewards_replay = [run_episode(env, policy_replay) for _ in range(episodes)]
    rewards_nsQlearning = [run_episode(env, policy_nsQlearning) for _ in range(episodes)]

    plt.plot(rewards_qlearning, label="Q-Learning")
    plt.plot(rewards_double_qlearning, label="Double Q-Learning")
    plt.plot(rewards_replay, label="Experience Replay Q-Learning")
    plt.plot(rewards_nsQlearning, label="n-Step Q-Learning")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('./q_learning_based_algos_frozen_lake.png')
    plt.show()
