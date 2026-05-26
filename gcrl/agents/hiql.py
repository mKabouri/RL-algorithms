"""
Inspired from: https://github.com/seohongpark/ogbench
"""
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from networks import ActorNetwork, ValueNetwork


class HIQLAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Any
    cfg: ml_collections.ConfigDict

    def expectile_loss(self, diff: jnp.ndarray) -> jnp.ndarray:
        weights = jnp.where(diff > 0, self.cfg.expectile_coeff, 1 - self.cfg.expectile_coeff)
        return (weights * diff**2).mean()

    def update_value(self, batch: Any):
        obs_goal = jnp.concatenate([batch["observations"], batch["goals"]], axis=-1)
        next_obs_goal = jnp.concatenate([batch["next_observations"], batch["goals"]], axis=-1)

        v_tp1 = self.train_states["target_value"].apply_fn(
            {"params": self.train_states["target_value"].params}, next_obs_goal
        )
        td_target = batch["rewards"] + self.cfg.discount * (1.0 - batch["dones"]) * v_tp1
        td_target = jax.lax.stop_gradient(td_target)

        def loss_fn(params):
            v = self.train_states["value"].apply_fn({"params": params}, obs_goal)
            loss = self.expectile_loss(td_target - v)
            return loss, {"value_loss": loss, "v_mean": v.mean()}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.train_states["value"].params)
        new_ts = self.train_states["value"].apply_gradients(grads=grads)
        return self.replace(train_states={**self.train_states, "value": new_ts}), metrics

    def update_high_level(self, batch: Any):
        obs_goal = jnp.concatenate([batch["observations"], batch["goals"]], axis=-1)
        subgoal_goal = jnp.concatenate([batch["subgoal_observations"], batch["goals"]], axis=-1)

        v_s = self.train_states["value"].apply_fn({"params": self.train_states["value"].params}, obs_goal)
        v_subgoal = self.train_states["value"].apply_fn({"params": self.train_states["value"].params}, subgoal_goal)
        advantage = v_subgoal - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(params):
            dist = self.train_states["high_level_actor"].apply_fn({"params": params}, obs_goal)
            log_prob = dist.log_prob(batch["subgoal_observations"])
            loss = -(weights * log_prob).mean()
            return loss, {"high_level_actor_loss": loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.train_states["high_level_actor"].params)
        new_ts = self.train_states["high_level_actor"].apply_gradients(grads=grads)
        return self.replace(train_states={**self.train_states, "high_level_actor": new_ts}), metrics

    def update_low_level(self, batch: Any):
        obs_subgoal = jnp.concatenate([batch["observations"], batch["subgoal_observations"]], axis=-1)
        next_obs_subgoal = jnp.concatenate([batch["next_observations"], batch["subgoal_observations"]], axis=-1)

        v_s = self.train_states["value"].apply_fn({"params": self.train_states["value"].params}, obs_subgoal)
        v_tp1 = self.train_states["value"].apply_fn({"params": self.train_states["value"].params}, next_obs_subgoal)
        advantage = v_tp1 - v_s
        weights = jnp.clip(jnp.exp(self.cfg.beta * advantage), 0, 100)
        weights = jax.lax.stop_gradient(weights)

        def loss_fn(params):
            dist = self.train_states["low_level_actor"].apply_fn({"params": params}, obs_subgoal)
            log_prob = dist.log_prob(batch["actions"])
            loss = -(weights * log_prob).mean()
            return loss, {"low_level_actor_loss": loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.train_states["low_level_actor"].params)
        new_ts = self.train_states["low_level_actor"].apply_gradients(grads=grads)
        return self.replace(train_states={**self.train_states, "low_level_actor": new_ts}), metrics

    def soft_update_target_value(self):
        tau = self.cfg.tau
        new_params = jax.tree_util.tree_map(
            lambda target, online: tau * target + (1 - tau) * online,
            self.train_states["target_value"].params,
            self.train_states["value"].params,
        )
        new_ts = self.train_states["target_value"].replace(params=new_params)
        return self.replace(train_states={**self.train_states, "target_value": new_ts})

    def update(self, batch: Any):
        agent, value_logs = self.update_value(batch)
        agent, low_logs = agent.update_low_level(batch)
        agent, high_logs = agent.update_high_level(batch)
        agent = agent.soft_update_target_value()
        return agent, {**value_logs, **low_logs, **high_logs}

    def sample_actions(
        self, obs_goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        rng, key_h, key_l = jax.random.split(rng, 3)
        obs_dim = obs_goal.shape[-1] // 2
        obs = obs_goal[..., :obs_dim]

        high_dist = self.train_states["high_level_actor"].apply_fn(
            {"params": self.train_states["high_level_actor"].params}, obs_goal
        )
        subgoal = high_dist.mode() if deterministic else high_dist.sample(seed=key_h)

        obs_subgoal = jnp.concatenate([obs, subgoal], axis=-1)
        low_dist = self.train_states["low_level_actor"].apply_fn(
            {"params": self.train_states["low_level_actor"].params}, obs_subgoal
        )
        return low_dist.mode() if deterministic else low_dist.sample(seed=key_l)

    @classmethod
    def create(cls, rng: jax.random.PRNGKey, obs_dim: int, action_dim: int, cfg: ml_collections.ConfigDict):
        input_dim = 2 * obs_dim
        dummy_input = jnp.zeros((1, input_dim))

        networks = {
            "value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims, activations=cfg.activations, kernel_init=cfg.kernel_init
            ),
            "target_value": ValueNetwork(
                hidden_dims=cfg.value_hidden_dims, activations=cfg.activations, kernel_init=cfg.kernel_init
            ),
            "high_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=obs_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
            ),
            "low_level_actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
            ),
        }

        rng, *keys = jax.random.split(rng, 5)
        params = {
            "value": networks["value"].init(keys[0], dummy_input)["params"],
            "target_value": networks["target_value"].init(keys[1], dummy_input)["params"],
            "high_level_actor": networks["high_level_actor"].init(keys[2], dummy_input)["params"],
            "low_level_actor": networks["low_level_actor"].init(keys[3], dummy_input)["params"],
        }

        train_states = {
            "value": TrainState.create(
                apply_fn=networks["value"].apply, params=params["value"], tx=optax.adam(cfg.value_lr)
            ),
            "target_value": TrainState.create(
                apply_fn=networks["target_value"].apply, params=params["target_value"], tx=optax.set_to_zero()
            ),
            "high_level_actor": TrainState.create(
                apply_fn=networks["high_level_actor"].apply,
                params=params["high_level_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "low_level_actor": TrainState.create(
                apply_fn=networks["low_level_actor"].apply,
                params=params["low_level_actor"],
                tx=optax.adam(cfg.actor_lr),
            ),
        }
        return cls(rng=rng, train_states=train_states, cfg=cfg)


def get_default_config():
    return ml_collections.ConfigDict(
        dict(
            # networks
            actor_hidden_dims=(256, 256),
            value_hidden_dims=(256, 256),
            activations=nn.gelu,
            kernel_init=nn.initializers.orthogonal(),
            # training
            actor_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            expectile_coeff=0.7,
            beta=3.0,
            # hierarchy
            subgoal_steps=20,
        )
    )
