"""
"Proximal Policy Optimization Algorithms" by Schulman et al.
Link: https://arxiv.org/pdf/1707.06347.pdf
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

# PPO is an on-policy algorithm
env = gym.make("CartPole-v1")
obs, _ = env.reset()

obs_dim = env.observation_space.shape[0]
n_acts = env.action_space.n

device = "cpu" if not torch.cuda.is_available() else "cuda"

# Define policy neural network
net_stochastic_policy = nn.Sequential(
        nn.Linear(obs_dim, 64), 
        nn.ReLU(),
        nn.Linear(64, n_acts)
    ).to(device)

def net_policy(input):
    input = input.to(device)
    return net_stochastic_policy(input)

# Learned policy
def policy(state):
    state = state.to(device)
    logits = net_stochastic_policy(state)
    return Categorical(logits=logits)

def choose_action(state):
    state = state.to(device)
    return policy(state).sample().item()

# We calculates loss by applying the formula in the original paper
def loss_clip(states,
              advantages,
              old_probs,
              actions,
              epsilon):
    new_probs = policy(states).log_prob(actions)
    ratio = torch.exp(new_probs-old_probs)
    clip_term = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    loss = torch.min(ratio*advantages, clip_term*advantages).mean()
    return loss

def loss_VF(values, returns):
    return F.mse_loss(values, returns)

def compute_loss(states,
                 advantages,
                 old_probs,
                 actions,
                 epsilon,
                 values,
                 returns,
                 vf_coef,
                 entropy_coef):
    loss_clip_term = loss_clip(states, advantages, old_probs, actions, epsilon)
    loss_VF_term = loss_VF(values, returns)
    entropy_term = policy(states).entropy().mean() 
    loss = -loss_clip_term + vf_coef*loss_VF_term - entropy_coef*entropy_term
    return loss

BATCH_SIZE=200
EPOCHS=10
NB_ITERATIONS=800
RUN_TIME_STEPS=600 # To adjust depending on the environment
EPSILON=0.2
VF_COEF=1
ENTROPY_COEF=0.01

LEARNING_RATE=0.001
optimizer = torch.optim.Adam(net_stochastic_policy.parameters(), lr=LEARNING_RATE)

def PPO_algorithm():
    total_reward = []
    training_time = time.time()
    for iteration in range(NB_ITERATIONS):
        iteration_start_time = time.time()
        # Collect samples
        obs, _ = env.reset()
        collect_actions = []
        collect_rewards = []
        collect_states = []
        collect_states.append(obs) 
        collect_advantages = []
        # Values and targets (returns) to compute VF loss
        collect_values = []
        collect_returns = []
        collect_old_probs = []

        for time_step in range(RUN_TIME_STEPS):
            # Choose an action using the current policy
            action = policy(torch.from_numpy(obs)).sample()
            collect_actions.append(action.clone().detach().item())
            
            # Old probabilities and values for losses
            old_probs = policy(torch.from_numpy(obs)).log_prob(action).clone().detach().item()
            value = net_policy(torch.from_numpy(obs)).mean().detach().item()
            collect_old_probs.append(old_probs)
            collect_values.append(value)

            obs, reward, done, truncated, _ = env.step(action.item())
            collect_rewards.append(reward)
            collect_states.append(obs)

            # I want to collect RUN_TIME_STEPS sample
            # But in the original paper they collect samples
            # for T timesteps and T is less than the episode
            # length but They do not said why.
            # I use episodic environments.
            if done or truncated:
                obs, _ = env.reset()
                break
        
        # Compute advantages estimates (GAMMA=1 since I work with episodic environments)
        # See those lines carefully please
        # I consider the value function as the mean of the output of the shared network
        final_value = net_policy(torch.from_numpy(collect_states[-1])).mean().detach().item()
        for time_step, reward in enumerate(collect_rewards):
            first_value = net_policy(torch.from_numpy(collect_states[time_step])).mean().detach().item()
            # I calculate advantage by the formula given in the original article
            advantage = -first_value + sum(collect_rewards[time_step:-1]) + final_value
            collect_advantages.append(advantage)

            # compute returns for loss (as target value)
            returns = sum(collect_rewards[time_step:])
            collect_returns.append(returns)    

        # Transform lists to numpy arrays to avoid a warning
        collect_states = np.array(collect_states)
        collect_actions = np.array(collect_actions)
        collect_advantages = np.array(collect_advantages)
        collect_values = np.array(collect_values)
        collect_returns = np.array(collect_returns)
        collect_old_probs = np.array(collect_old_probs)

        # Know I have all the material to do some optimization
        # Optimization
        for epoch in range(EPOCHS):
            optimizer.zero_grad()

            batch_indices = np.random.choice(len(collect_states)-1, BATCH_SIZE)

            batch_states = torch.from_numpy(collect_states[batch_indices]).to(device)
            batch_actions = torch.from_numpy(collect_actions[batch_indices]).to(device)
            batch_advantages = torch.from_numpy(collect_advantages[batch_indices]).to(device)
            batch_values = torch.from_numpy(collect_values[batch_indices]).to(device)
            batch_returns = torch.from_numpy(collect_returns[batch_indices]).to(device)
            bacth_old_probs = torch.from_numpy(collect_old_probs[batch_indices]).to(device)

            batch_loss = compute_loss(batch_states,
                                    batch_advantages,
                                    bacth_old_probs,
                                    batch_actions,
                                    EPSILON,
                                    batch_values,
                                    batch_returns,
                                    VF_COEF,
                                    ENTROPY_COEF)

            batch_loss.backward()

            optimizer.step()
        end_iteration_time = time.time() - iteration_start_time
        if iteration % 50 == 0:
            time_format = time.strftime("%H:%M:%S", time.gmtime(end_iteration_time))
            print(f'iteration: {iteration}\t duration: {time_format}')
            print(f'Total reward: {np.sum(collect_rewards):.3f}\t')
            print('-------------------------------------------------')
            print()
        total_reward.append(np.sum(collect_rewards))
    end_training = time.time() - training_time
    time_format = time.strftime("%H:%M:%S", time.gmtime(end_training))
    print(f"Training Duration: {time_format}")
    return total_reward

def evaluate(render=False):
    obs, _ = env.reset()
    total_reward = 0
    done = False
    while not done:
        if render:
            env.render()
        action = choose_action(torch.as_tensor(obs, dtype=torch.float32))
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        if done or truncated:
            break
    return total_reward

# Function to plot total reward per iteration of training
def plot_rewards(total_rewards, title):
    plt.plot(total_rewards)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Total reward')
    plt.savefig("./" + title + ".png")


if __name__ == '__main__':
    print(f"Training with {device}")
    net_stochastic_policy.train()
    
    avg_rewards = PPO_algorithm()
    plot_rewards(avg_rewards, "PPO training")
    
    torch.save(net_stochastic_policy.state_dict(), "./PPO_weights.pt")

    net_stochastic_policy.eval()

    # net_stochastic_policy.load_state_dict(torch.load("./PPO_weights.pt"))
    policy_scores = [evaluate(render=True) for _ in range(100)]
    print("Average score of the policy: ", np.mean(policy_scores))

    env.close()
