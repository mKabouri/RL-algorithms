import numpy as np
import matplotlib.pyplot as plt
import gym
from collections import namedtuple, deque
import random

env = gym.make("CartPole-v1")

env.seed(0)
env.action_space.seed(0)
np.random.seed(0)

NB_STATES = 162 # 3 * 3 * 6 * 3

def obs_to_state(obv):
    x, x_dot, theta, theta_dot = obv

    # X_pos Pass
    if x < -.8:
        state = 0
    elif x < .8:
        state = 1
    else:
        state = 2

    # X_velocity Pass
    if x_dot < -.5:
        pass
    elif x_dot < .5:
        state += 3
    else:
        state += 6

    if theta < np.radians(-12):
        pass
    elif theta < np.radians(-1.5):
        state += 9
    elif theta < np.radians(0):  # goldzone
        state += 18
    elif theta < np.radians(1.5):
        state += 27
    elif theta < np.radians(12):
        state += 36
    else:
        state += 45

    if theta_dot < np.radians(-50):
        pass
    elif theta_dot < np.radians(50):
        state += 54
    else:
        state += 108

    return state

MIN_EXPLORE_RATE = 0.01
MIN_LEARNING_RATE = 0.1

def update_explore_rate(episode):
    return max(MIN_EXPLORE_RATE, min(1, 1.0 - np.log10((episode + 1) / 25)))

def update_learning_rate(episode):
    return max(MIN_LEARNING_RATE, (min(0.5, 1.0 - np.log10((episode + 1) / 50))))

def run_episode(env, policy = None, render = False):
  obs = env.reset()
  total_reward = 0
  done = False
  while not done:
    if render:
      env.render()
  
    if policy is None:
      action = env.action_space.sample()
    else:
      state = obs_to_state(obs)
      action = policy[state]

    obs, reward, done, _ = env.step(action)
    total_reward += reward
  return total_reward

run_episode(env)

EPISODES = 5000
TIMEOUT = 250

def Q_learning():
  Qvalue = np.random.rand(NB_STATES, env.action_space.n)
  for episode in range(EPISODES):
    obs = env.reset()
    learning_rate = update_learning_rate(episode)
    epsilon = update_explore_rate(episode)

    state = obs_to_state(obs)
    total_reward = 0
    done = False
    while not done:
      # choose action using epsilon-greedy strategy
      if np.random.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(Qvalue[state])

      # make a step and update the state
      obs, reward, done, _ = env.step(action)
      new_state = obs_to_state(obs)

      # update Qvalue with step = learning_rate
      Qvalue[state, action] += learning_rate*(reward + np.max(Qvalue[new_state]) - Qvalue[state, action])
      total_reward += reward
      state = new_state
    
    if episode % 200 == 0:
      print('Iteration #%d -- Total reward = %d.' %(episode+1, total_reward))
      
  return Qvalue

def SARSA():
  Qvalue = np.random.rand(NB_STATES, env.action_space.n)
  for episode in range(EPISODES):
    obs = env.reset()
    learning_rate = update_learning_rate(episode)
    epsilon = update_explore_rate(episode)

    state = obs_to_state(obs)
    total_reward = 0
    done = False
    while not done:
      # choose action using epsilon-greedy strategy
      if np.random.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(Qvalue[state])

      # make a step and update the state
      obs, reward, done, _ = env.step(action)
      new_state = obs_to_state(obs)

      if np.random.random() < epsilon:
        next_action = env.action_space.sample()
      else:
        next_action = np.argmax(Qvalue[state])

      # update Qvalue with step = learning_rate
      Qvalue[state, action] += learning_rate*(reward + Qvalue[new_state, next_action] - Qvalue[state, action])
      total_reward += reward
      state = new_state

    if episode % 200 == 0:
      print('Iteration #%d -- Total reward = %d.' %(episode+1, total_reward))

  return Qvalue

def double_Q_learning():
  Qvalue1 = np.random.rand(NB_STATES, env.action_space.n)
  Qvalue2 = np.random.rand(NB_STATES, env.action_space.n)
  for episode in range(EPISODES):
    obs = env.reset()
    learning_rate = update_learning_rate(episode)
    epsilon = update_explore_rate(episode)

    state = obs_to_state(obs)
    total_reward = 0
    done = False
    while not done:
      # choose action using epsilon-greedy strategy
      if np.random.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax((Qvalue1+ Qvalue2)[state])

      # make a step and update the state
      obs, reward, done, _ = env.step(action)
      new_state = obs_to_state(obs)

      # update Qvalue with step = learning_rate
      if np.random.rand() < 0.5:
        Qvalue1[state, action] += learning_rate*(reward + Qvalue2[new_state][np.argmax(Qvalue1[new_state])] - Qvalue1[state, action])
      else:
        Qvalue2[state, action] += learning_rate*(reward + Qvalue1[new_state][np.argmax(Qvalue2[new_state])] - Qvalue2[state, action])

      total_reward += reward
      state = new_state
  
    if episode % 200 == 0:
      print('Iteration #%d -- Total reward = %d.' %(episode+1, total_reward))
      
  return Qvalue1, Qvalue2

########################################################
# From "https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory"
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
###############################################################

CAPACITY = NB_STATES
BATCH_SIZE = 100
BUFFER = ReplayMemory(capacity=CAPACITY)

def experience_replay_Q_learning():
  Qvalue = np.random.rand(NB_STATES, env.action_space.n)
  for episode in range(EPISODES):
    obs = env.reset()    
    epsilon = update_explore_rate(episode)

    state = obs_to_state(obs)
    total_reward = 0
    done = False
    while not done:
      # choose action using epsilon-greedy strategy
      if np.random.random() < epsilon:
        action = env.action_space.sample()
      else:
        action = np.argmax(Qvalue[state])

      # make a step and update the state
      obs, reward, done, _ = env.step(action)
      new_state = obs_to_state(obs)

      # Store in our buffer
      BUFFER.push(state, action, reward, new_state)

      if BUFFER.__len__() >= BATCH_SIZE:
        batch = BUFFER.sample(BATCH_SIZE)
        states, actions, rewards, next_states = zip(*batch)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        for i in range(BATCH_SIZE):
          learning_rate = update_learning_rate(episode)
          # update Qvalue with step = learning_rate
          Qvalue[states[i], actions[i]] += learning_rate*(\
             rewards[i] + np.max(Qvalue[next_states[i]]) - Qvalue[states[i], actions[i]])
      total_reward += reward
      state = new_state
  
    if episode % 200 == 0:
      print('Iteration #%d -- Total reward = %d.' %(episode+1, total_reward))

  return Qvalue


def plot_rewards(rewards, title):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()


if __name__ == '__main__':
    Qvalue = Q_learning()
    # # Qvalue = SARSA()
    Qvalue1, Qvalue2 = double_Q_learning()
    QvalueReplay = experience_replay_Q_learning()

    policy_qlearning = np.argmax(Qvalue, axis=1)
    policy_double_qlearning = np.argmax(Qvalue1, axis=1)
    policy_double_qlearning_replay = np.argmax(QvalueReplay, axis=1)

    policy_scores = [run_episode(env, policy_qlearning) for _ in range(500)]
    print("Average score of the policy (qleaning): ", np.mean(policy_scores))

    policy_scores_double_qlearning = [run_episode(env, policy_double_qlearning) for _ in range(500)]
    print("Average score of the policy (double qleaning): ", np.mean(policy_scores_double_qlearning))

    policy_scores_qlearning_replay = [run_episode(env, policy_double_qlearning_replay) for _ in range(500)]
    print("Average score of the policy (Memory Replay): ", np.mean(policy_scores_qlearning_replay))

    x_axis = np.arange(500) 
    plt.plot(x_axis, policy_scores, label="Q-Learning")
    plt.plot(x_axis, policy_scores_double_qlearning, label="Double Q-Learning")
    plt.plot(x_axis, policy_scores_qlearning_replay, label="Q-Learning with experience replay")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()