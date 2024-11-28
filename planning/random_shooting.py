"""
Random Shooting Algorithm.

This algorithm optimizes a sequence of future actions to maximize
the expected cumulative reward over a planning horizon using a
model of the environment. At each time step:

1. Randomly generate multiple candidate action sequences.
2. Simulate the future rewards for each sequence using the environment model.
3. Select the action sequence with the highest predicted cumulative reward.
4. Execute the first action of the optimal sequence in the environment and re-plan at the next step.
"""
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from multiprocessing import Pool
from typing import List

def random_shooting(
    horizon: int,
    num_samples: int,
    env: gym.Env,
    seed: int | None= None
):
    """
    Parameters:
        horizon: Length of each action sequence.
        num_samples: Number of random action sequences to sample.
        env: Gymnasium environment object.
        seed: Random seed.

    Returns:
        Total reward achieved using Random Shooting.
    """
    if seed is not None:
        np.random.seed(seed)

    total_reward = 0
    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        action_sequences = np.random.randint(
            0, env.action_space.n,
            size=(num_samples, horizon)
        )
        rewards = np.zeros(num_samples)

        # Evaluate action sequences
        for i in range(num_samples):
            sim_env = gym.make(env.unwrapped.spec.id)
            sim_env.reset(seed=seed)
            # Set to the true current state
            sim_env.unwrapped.state = state

            cumulative_reward = 0
            for action in action_sequences[i]:
                _, reward, terminated, truncated, _ = sim_env.step(action)
                cumulative_reward += reward
                if terminated or truncated:
                    break
            rewards[i] = cumulative_reward
            sim_env.close()

        best_action = action_sequences[np.argmax(rewards)][0]
        state, reward, done, truncated, _ = env.step(best_action)
        total_reward += reward

    return total_reward

def evaluate_seed(args):
    env_name, horizon, num_samples, seed = args
    env = gym.make(env_name)
    reward = random_shooting(horizon=horizon, num_samples=num_samples, env=env, seed=seed)
    print(f"Seed: {seed}, Total Reward: {reward}")
    env.close()
    return seed, reward

def evaluate_random_shooting(
    env_name: str,
    horizon: int,
    num_samples: int,
    seeds: List[int]
):
    """
    Evaluate Random Shooting across multiple random seeds.

    Parameters:
        env_name: Name of the Gymnasium environment.
        horizon: Length of each action sequence.
        num_samples: Number of random action sequences to sample.
        seeds: List of random seeds.

    Returns:
        List of total rewards for each seed.
    """
    args = [(env_name, horizon, num_samples, seed) for seed in seeds]
    with Pool(processes=len(seeds)) as pool:
        results = pool.map(evaluate_seed, args)
    results.sort(key=lambda x: seeds.index(x[0]))
    return [reward for _, reward in results]

def plot_results(
    rewards: List[float],
    seeds: List[int]
):
    """
    Plot the results with confidence intervals.

    Parameters:
        rewards: List of total rewards for each seed.
        seeds: List of random seeds used.
    """
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    ci = 1.96*(std_reward/np.sqrt(len(seeds)))
    plt.figure(figsize=(8, 5))
    plt.bar(["Random Shooting"], [mean_reward], yerr=[ci], capsize=10, color="skyblue")
    plt.ylabel("Total Reward")
    plt.title("Performance of Random Shooting with Confidence Interval")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig("./random_shooting_results.png")

if __name__ == "__main__":
    ENV_NAME = "CartPole-v1"
    HORIZON = 20
    NUM_SAMPLES = 1000
    SEEDS = [42, 123, 456, 789, 101112]
    rewards = evaluate_random_shooting(ENV_NAME, HORIZON, NUM_SAMPLES, SEEDS)
    print(f"Rewards for each seed: {rewards}")
    plot_results(rewards, SEEDS)
