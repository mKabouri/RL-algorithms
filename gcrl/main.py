import argparse
import importlib
import os
from datetime import datetime

import jax
import ml_collections
import numpy as np
import ogbench
from tqdm import tqdm

from agents import agents
from logging_utils import evaluate, init_wandb, log_metrics, record_video, save_checkpoint
from replay_buffer import HierarchicalReplayBuffer, GCRLReplayBuffer

AGENT_REGISTRY = {
    "hiql": "agents.hiql",
    "crl": "agents.crl",
}


def get_config() -> ml_collections.ConfigDict:
    """Generic experiment/training config. Agent hyperparameters live in agents/<agent>.py."""
    return ml_collections.ConfigDict(
        dict(
            env="antmaze-large-navigate-v0",
            seed=42,
            agent="hiql",
            # training
            num_steps=2_000_000,
            batch_size=1024,
            utd=1,
            # logging
            log_interval=10_000,
            eval_interval=200_000,
            num_eval_episodes=50,
            # video
            record_video=True,
            # checkpointing
            save_checkpoints=True,
            # checkpoint dir should be absolute
            checkpoint_dir=os.path.abspath("checkpoints"),
            # reward type: "neg_one_zero" or "zero_one"
            reward_type="neg_one_zero",
            goal_threshold=0.05
        )
    )


def build_agent_cfg(agent: str, overrides: dict = None) -> ml_collections.ConfigDict:
    module = importlib.import_module(AGENT_REGISTRY[agent])
    agent_cfg = module.get_default_config()
    if overrides:
        agent_cfg.update(overrides)
    return agent_cfg


def parse_args(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
    """Apply CLI overrides on top of get_config() defaults."""
    parser = argparse.ArgumentParser()
    for key, val in cfg.items():
        parser.add_argument(f"--{key}", type=type(val), default=val)
    args, _ = parser.parse_known_args()
    cfg.update(vars(args))
    return cfg


def main():
    cfg = parse_args(get_config())
    cfg.checkpoint_dir = os.path.join(cfg.checkpoint_dir, datetime.now().strftime("%Y-%m-%d-%H-%M"))
    agent_cfg = build_agent_cfg(cfg.agent)

    init_wandb(cfg, agent_cfg=agent_cfg)

    rng = jax.random.PRNGKey(cfg.seed)
    np.random.seed(cfg.seed)

    # create environment, agent, replay buffer
    env, train_dataset, _ = ogbench.make_env_and_datasets(cfg.env)
    replay_buffers = {
        "hiql": HierarchicalReplayBuffer,
        "crl": GCRLReplayBuffer,
    }

    replay_buffer = replay_buffers[cfg.agent].create(
        observations=train_dataset["observations"],
        actions=train_dataset["actions"],
        next_observations=train_dataset["next_observations"],
        dones=train_dataset["terminals"],
        subgoal_steps=agent_cfg.subgoal_steps if hasattr(agent_cfg, "subgoal_steps") else 0,
    )
    agent = agents[cfg.agent].create(rng, env.observation_space.shape[0], env.action_space.shape[0], agent_cfg)

    # main training loop: sample batch, update agent, log, eval
    for i in tqdm(range(1, cfg.num_steps + 1)):
        rng, batch_rng = jax.random.split(rng)
        batch = replay_buffer.sample(
            cfg.batch_size,
            batch_rng,
            reward_type=cfg.reward_type,
            goal_threshold=cfg.goal_threshold,
        )
        agent = agent.replace(rng=rng)
        agent, update_logs = agent.update(batch)

        if i % cfg.log_interval == 0:
            # log agent
            log_metrics(update_logs, step=i, prefix="train")

        if i % cfg.eval_interval == 0:
            # evaluate agent on val_dataset
            eval_logs = evaluate(agent, env, num_episodes=cfg.num_eval_episodes, rng=rng)
            log_metrics(eval_logs, step=i, prefix="eval")

            # save a video if enabled
            if cfg.record_video:
                record_video(agent, env, rng, step=i, key="eval/video")
            if cfg.save_checkpoints:
                # save checkpoint
                save_checkpoint(agent, cfg.checkpoint_dir, step=i)

    # cleanup, save final model
    if cfg.save_checkpoints:
        save_checkpoint(agent, cfg.checkpoint_dir, step=cfg.num_steps)


if __name__ == "__main__":
    main()
