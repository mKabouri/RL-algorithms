import os

import jax
import ml_collections
import numpy as np
import orbax.checkpoint as ocp
import wandb


def _config_value(value):
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    if isinstance(value, tuple):
        return list(value)
    return value


def _config_dict(cfg: ml_collections.ConfigDict) -> dict:
    return {key: _config_value(value) for key, value in cfg.items()}


def init_wandb(cfg: ml_collections.ConfigDict, agent_cfg: ml_collections.ConfigDict = None, **kwargs):
    config = {"training": _config_dict(cfg)}
    if agent_cfg is not None:
        config["agent"] = _config_dict(agent_cfg)

    wandb.init(
        project="gcrl",
        config=config,
        reinit=True,
        **kwargs,
    )


def _flatten(d: dict, parent_key: str = "", sep: str = "/") -> dict:
    items = {}
    for k, v in d.items():
        key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, key, sep))
        else:
            items[key] = v
    return items


def _to_loggable(value):
    value = jax.device_get(value)
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, np.generic):
        return value.item()
    return value


def log_metrics(metrics: dict, step: int, prefix: str = ""):
    flat = _flatten(metrics)
    log_dict = {f"{prefix}/{k}" if prefix else k: _to_loggable(v) for k, v in flat.items()}
    wandb.log(log_dict, step=step)


def evaluate(agent, env, num_episodes: int, rng: jax.random.PRNGKey) -> dict:
    """
    Run the agent greedily for num_episodes.
    Expects the env to provide info['goal'] at reset (ogbench convention).
    The agent must implement sample_actions(obs, goal, rng, deterministic=True).
    """
    returns, successes, lengths = [], [], []

    for _ in range(num_episodes):
        obs, info = env.reset()
        goal = info.get("goal", None)
        done = False
        ep_return, ep_length = 0.0, 0

        while not done:
            rng, key = jax.random.split(rng)
            action = np.array(agent.sample_actions(obs, goal, key, deterministic=True))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_return += reward
            ep_length += 1

        returns.append(ep_return)
        lengths.append(ep_length)
        successes.append(float(info.get("success", info.get("is_success", False))))

    return {
        "episode_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "episode_length": float(np.mean(lengths)),
    }


def record_video(
    agent,
    env,
    rng: jax.random.PRNGKey,
    step: int,
    fps: int = 30,
    key: str = "eval/video",
):
    """
    Roll out one episode, capture frames via env.render(), and log to wandb.
    The env must be created with render_mode='rgb_array'.
    wandb.Video expects shape (T, C, H, W) uint8.
    """
    frames = []
    obs, info = env.reset()
    goal = info.get("goal", None)
    done = False

    while not done:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        rng, subkey = jax.random.split(rng)
        action = np.array(agent.sample_actions(obs, goal, subkey, deterministic=True))
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    if not frames:
        return

    # (T, H, W, C) → (T, C, H, W)
    video = np.stack(frames).transpose(0, 3, 1, 2).astype(np.uint8)
    wandb.log({key: wandb.Video(video, fps=fps, format="mp4")}, step=step)


def save_checkpoint(agent, checkpoint_dir: str, step: int):
    path = os.path.join(checkpoint_dir, f"step_{step}")
    if os.path.exists(path):
        return

    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, agent)


def restore_checkpoint(agent, checkpoint_dir: str, step: int):
    path = os.path.join(checkpoint_dir, f"step_{step}")
    checkpointer = ocp.PyTreeCheckpointer()
    return checkpointer.restore(path, item=agent)
