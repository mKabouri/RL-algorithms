import os
import argparse
import torch
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from stable_baselines3.ppo import PPO
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm as SB3Algorithm
from stable_baselines3.common.policies import BasePolicy as SB3Policy
from typing import Type, Union, List

def load_rewards(files: List[str]):
    """
    Load the rewards from the given files.
    """
    rewards = []
    seeds = []

    for file in files:
        seed = int(file.split("_")[-1].split(".")[0])
        seeds.append(seed)
        with open(file, "r") as f:
            rewards.append([float(line) for line in f.readlines()])

    return np.array(rewards), np.array(seeds)

def plot_learning_curves(training_rewards: np.ndarray, seeds: list[int]):
    """
    Plot the learning curves for the training experiments.
    """
    sns.set(style="darkgrid")

    mean_rewards = np.mean(training_rewards, axis=0)
    std_rewards = np.std(training_rewards, axis=0)

    plt.figure(figsize=(10, 6))
    x_vals = np.arange(1, len(mean_rewards) + 1)

    for i, seed in enumerate(seeds):
        plt.plot(x_vals, training_rewards[i], label=f'Seed {seed}', alpha=0.5)

    plt.plot(x_vals, mean_rewards, label='Mean Reward', color='blue', lw=2)
    plt.fill_between(x_vals, mean_rewards - std_rewards, mean_rewards + std_rewards, color='blue', alpha=0.3)
    
    plt.title("Learning Curves During Training")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend(loc='upper left')
    plt.show()


class RewardCallback(BaseCallback):
    """
    Callback for logging the total rewards during training.
    """
    def __init__(
        self,
        seed: int,
        verbose: int = 0,
        log_dir: str | None=None,
    ):
        super(RewardCallback, self).__init__(verbose=verbose)
        self.seed = seed

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.f_rewards = os.path.join(self.log_dir, f"rewards_seed_{seed}.txt")
        if not os.path.exists(self.f_rewards):
            with open(self.f_rewards, "w") as f:
                f.write("")

        self.episode_rewards = 0

    def _reset_rewards(self):
        self.episode_rewards = 0

    def _on_step(self) -> bool:
        self.episode_rewards += self.locals['rewards'][0]
        if self.locals['dones'][0] or self.locals['infos'][0]['TimeLimit.truncated']:
            self._log_rewards()
            self._reset_rewards()
        return True

    def _log_rewards(self):
        with open(self.f_rewards, "a") as f:
            f.write(str(self.episode_rewards) + "\n")

def make_env():
    """
    Create and return the HalfCheetah environment.
    """
    return gym.make("HalfCheetah-v4", render_mode="human")


def learn_policy(
    timesteps: int,
    env: gym.Env,
    algo_cls: Type[SB3Algorithm],
    path_to_agent: str,
    callback: RewardCallback,
    policy: Union[str, Type[SB3Policy]] = "MlpPolicy",
    algo_seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Train or load a policy for the given environment.
    """
    print(f"--> Learn policy with seed {algo_seed}")

    if not path_to_agent:
        raise ValueError(f"Invalid path to agent.")

    env.reset(seed=algo_seed)

    # Initialize the agent
    agent = algo_cls(
        policy=policy,
        env=env,
        seed=algo_seed,
        device=device,
        verbose=1,
    )
    
    if os.path.exists(path_to_agent):
        print("--> Load agent from checkpoint")
        agent = algo_cls.load(path_to_agent, env=env)

    # Learn and save the agent
    agent.learn(
        total_timesteps=timesteps,
        callback=callback
    )
    agent.save(path_to_agent)
    
    return agent

def evaluate(env: gym.Env, agent: Type[SB3Algorithm], eval_timesteps: int):
    """
    Evaluate the trained agent on the environment.
    For visualization, render the environment.
    """
    print(f"--> Evaluate policy")
    obs, _ = env.reset()
    for _ in range(eval_timesteps):
        action, _ = agent.predict(observation=obs)
        obs, _, done, truncated, _ = env.step(action)
        if done or truncated:
            obs, _ = env.reset()

def run_experiment(
    learn_timesteps: int,
    eval_timesteps: int,
    algo_cls: Type[SB3Algorithm],
    path_to_agent: str,
    seeds: list[int],
    device: str,
):
    """
    Run training experiments with multiple seeds and plot learning curves.
    """
    env = make_env()

    for seed in seeds:
        callback = RewardCallback(seed=seed, log_dir="./logs")
        agent = learn_policy(
            timesteps=learn_timesteps,
            env=env,
            algo_cls=algo_cls,
            path_to_agent=path_to_agent,
            callback=callback,
            algo_seed=seed,
            device=device
        )
        evaluate(env, agent, eval_timesteps)

    rewards, seeds = load_rewards(files=[f"./logs/rewards_seed_{seed}.txt" for seed in seeds])
    plot_learning_curves(training_rewards=rewards, seeds=seeds)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate a policy on HalfCheetah-v4.")
    parser.add_argument('--timesteps', type=int, default=int(1e5), help="Total training timesteps.")
    parser.add_argument('--eval_timesteps', type=int, default=int(1e4), help="Total evaluation timesteps.")
    parser.add_argument('--path_to_agent', type=str, default="./half_cheetah_model.zip", help="Path to save/load the agent.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2023], help="List of seeds for experimentation.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    return parser.parse_args()

def main():
    args = parse_args()
    run_experiment(
        learn_timesteps=args.timesteps,
        eval_timesteps=args.eval_timesteps,
        algo_cls=SAC,
        path_to_agent=args.path_to_agent,
        seeds=args.seeds,
        device=args.device
    )

if __name__ == "__main__":
    main()
