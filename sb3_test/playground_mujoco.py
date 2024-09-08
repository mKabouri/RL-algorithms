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
from torch.utils.tensorboard import SummaryWriter
from typing import Type, Union, List, Tuple

def make_env(
    for_training: bool = True
) -> gym.Env:
    """
    Create and return the HalfCheetah environment.
    """
    render_mode = None if for_training else "human"
    return gym.make("HalfCheetah-v4", render_mode=render_mode)

class RewardCallback(BaseCallback):
    """
    Callback for logging the total rewards during training.
    """
    def __init__(
        self,
        seed: int,
        verbose: int = 0,
        log_dir: str | None = None,
    ) -> None:
        super(RewardCallback, self).__init__(verbose=verbose)
        self.seed = seed

        self.log_dir = log_dir
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.f_rewards = os.path.join(self.log_dir, f"rewards_seed_{seed}.txt")
        if os.path.exists(self.f_rewards):
            os.remove(self.f_rewards)
        with open(self.f_rewards, "w") as f:
            f.write("")

        self.writer = SummaryWriter(log_dir=os.path.join(self.log_dir, f"tensorboard_{seed}"))

        self.episode_rewards = 0

    def _reset_rewards(self):
        self.episode_rewards = 0

    def _on_step(self) -> bool:
        self.episode_rewards += self.locals['rewards'][0]
        if self.locals['dones'][0] or self.locals['infos'][0]['TimeLimit.truncated']:
            self._log_and_reset_rewards()
        return True

    def _log_and_reset_rewards(self):
        self.writer.add_scalar('Reward/Episode', self.episode_rewards, self.num_timesteps)
        with open(self.f_rewards, "a") as f:
            f.write(str(self.episode_rewards) + "\n")
        self._reset_rewards()

def learn_policy(
    timesteps: int,
    env: gym.Env,
    algo_cls: Type[SB3Algorithm],
    path_to_agent: str,
    callback: RewardCallback,
    policy: Union[str, Type[SB3Policy]] = "MlpPolicy",
    algo_seed: int | None = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> SB3Algorithm:
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

def evaluate(
    env: gym.Env,
    agent: Type[SB3Algorithm],
    eval_timesteps: int
):
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

def load_rewards(
    files: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
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

def plot_learning_curves(
    training_rewards: np.ndarray,
    seeds: list[int],
    save_path: str | None = None
):
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
    plt.tight_layout()
    plt.savefig(save_path) if save_path else plt.savefig("learning_curves.png")
    print(f"Saved learning curves to {save_path}") if save_path else print("Saved learning curves to learning_curves.png")
    plt.show()

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
    print(f"Using: {device}")

    for seed in seeds:
        log_dir = "./logs"
        callback = RewardCallback(seed=seed, log_dir=log_dir)
        env = make_env(for_training=True)
        agent = learn_policy(
            timesteps=learn_timesteps,
            env=env,
            algo_cls=algo_cls,
            path_to_agent=path_to_agent,
            callback=callback,
            algo_seed=seed,
            device=device
        )
        env.close()

        env = make_env(for_training=False)
        evaluate(env, agent, eval_timesteps)

        env.close()
        callback.writer.close()

    rewards, seeds = load_rewards(files=[f"{log_dir}/rewards_seed_{seed}.txt" for seed in seeds])
    plot_learning_curves(training_rewards=rewards, seeds=seeds)

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train and evaluate a policy on HalfCheetah-v4.")
    parser.add_argument('--timesteps', type=int, default=int(5e5), help="Total training timesteps.")
    parser.add_argument('--eval_timesteps', type=int, default=int(5e4), help="Total evaluation timesteps.")
    parser.add_argument('--path_to_agent', type=str, default=f"./half_cheetah_model_{'{seed}'}.zip", help="Path to save/load the agent.")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 100, 2023], help="List of seeds for experimentation.")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training.")
    # parser.add_argument('--device', type=str, default="cpu", help="Device to use for training.")
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
