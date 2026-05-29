"""
Contrastive Reinforcement Learning (CRL) agent implementation.
"""
import functools
from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax.training.train_state import TrainState

from agents.utils import reduce_ensemble, with_ensemble_axis
from networks import ActorNetwork, BilinearCriticNetwork


class CRLAgent(flax.struct.PyTreeNode):
    rng: jax.random.PRNGKey
    train_states: Any
    cfg: ml_collections.ConfigDict = flax.struct.field(pytree_node=False)

    def _compute_contrastive_loss(self, batch: Any, params: Any) -> tuple[jnp.ndarray, jnp.ndarray]:
        q, phi, psi = self.train_states["critic"].apply_fn(
            {"params": params}, batch["observations"], batch["goals"], batch["actions"]
        )
        phi = with_ensemble_axis(phi, sample_ndim=2)
        psi = with_ensemble_axis(psi, sample_ndim=2)

        logits = jnp.einsum("ebk,egk -> bge", phi, psi)
        labels = jnp.eye(logits.shape[0])[:, :, None]
        labels = jnp.broadcast_to(labels, logits.shape)
        loss = optax.losses.sigmoid_binary_cross_entropy(logits, labels).mean()
        return loss, q

    def update_value(self, batch: Any):
        def loss_fn(params):
            critic_loss, critic_q = self._compute_contrastive_loss(batch, params)
            critic_q = reduce_ensemble(critic_q, self.cfg.critic_ensemble_reduce, name="critic")
            return critic_loss, {
                "critic_loss": critic_loss,
                "critic_q_mean": critic_q.mean(),
            }

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.train_states["critic"].params)
        new_ts = self.train_states["critic"].apply_gradients(grads=grads)
        return self.replace(train_states={**self.train_states, "critic": new_ts}), metrics

    def update_actor(self, batch: Any, rng: jax.random.PRNGKey):
        def loss_fn(params):
            dist = self.train_states["actor"].apply_fn({"params": params}, batch["observations"], batch["goals"])
            policy_actions = dist.sample(seed=rng)
            critic_q, _, _ = self.train_states["critic"].apply_fn(
                {"params": self.train_states["critic"].params}, batch["observations"], batch["goals"], policy_actions
            )
            critic_q = reduce_ensemble(critic_q, self.cfg.critic_ensemble_reduce, name="critic")
            actor_loss = -critic_q.mean()
            return actor_loss, {"actor_loss": actor_loss}

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(self.train_states["actor"].params)
        new_ts = self.train_states["actor"].apply_gradients(grads=grads)
        return self.replace(train_states={**self.train_states, "actor": new_ts}), metrics

    @jax.jit
    def update(self, batch: Any):
        rng, sub_rng = jax.random.split(self.rng)
        agent, critic_metrics = self.update_value(batch)
        agent, actor_metrics = agent.update_actor(batch, sub_rng)
        metrics = {**critic_metrics, **actor_metrics}
        agent = agent.replace(rng=rng)
        return agent, metrics

    @functools.partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(
        self, obs: jnp.ndarray, goal: jnp.ndarray, rng: jax.random.PRNGKey, deterministic: bool = False
    ) -> jnp.ndarray:
        rng, key_l = jax.random.split(rng)
        dist = self.train_states["actor"].apply_fn({"params": self.train_states["actor"].params}, obs, goal)
        return dist.mode() if deterministic else dist.sample(seed=key_l)

    @classmethod
    def create(cls, rng: jax.random.PRNGKey, obs_dim: int, action_dim: int, cfg: ml_collections.ConfigDict):
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_goal = jnp.zeros((1, obs_dim))
        dummy_action = jnp.zeros((1, action_dim))

        networks = {
            "actor": ActorNetwork(
                hidden_dims=cfg.actor_hidden_dims,
                action_dim=action_dim,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
            ),
            "critic": BilinearCriticNetwork(
                hidden_dims=cfg.critic_hidden_dims,
                activations=cfg.activations,
                kernel_init=cfg.kernel_init,
                ensemble_size=cfg.critic_ensemble_size,
                latent_dim=cfg.latent_dim,
            ),
        }

        rng, actor_key, critic_key = jax.random.split(rng, 3)
        train_states = {
            "actor": TrainState.create(
                apply_fn=networks["actor"].apply,
                params=networks["actor"].init(actor_key, dummy_obs, dummy_goal)["params"],
                tx=optax.adam(cfg.actor_lr),
            ),
            "critic": TrainState.create(
                apply_fn=networks["critic"].apply,
                params=networks["critic"].init(critic_key, dummy_obs, dummy_goal, dummy_action)["params"],
                tx=optax.adam(cfg.value_lr),
            ),
        }
        return cls(rng=rng, train_states=train_states, cfg=cfg)


def get_default_config():
    return ml_collections.ConfigDict(
        dict(
            # network
            actor_hidden_dims=(512, 512, 512),
            critic_hidden_dims=(512, 512, 512),
            activations=nn.gelu,
            kernel_init=nn.initializers.orthogonal(),
            critic_ensemble_size=2,
            critic_ensemble_reduce="min",
            # training
            actor_lr=3e-4,
            value_lr=3e-4,
            discount=0.99,
            tau=0.005,
            # CRL specific
            latent_dim=256,
        )
    )
