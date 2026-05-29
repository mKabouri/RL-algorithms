from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init():
    return nn.initializers.orthogonal()


def make_ensemble(cls, num_members: int):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=None,
        out_axes=0,
        axis_size=num_members,
    )


def combine_inputs(obs: jnp.ndarray, goals: jnp.ndarray | None = None, actions: jnp.ndarray | None = None):
    inputs = [obs]
    if goals is not None:
        inputs.append(goals)
    if actions is not None:
        inputs.append(actions)
    return jnp.concatenate(inputs, axis=-1) if len(inputs) > 1 else obs


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Sequence[callable] = nn.gelu
    activate_final: bool = False
    kernel_init: callable = default_init()
    layer_norm: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        return x


class ValueNetwork(nn.Module):
    hidden_dims: Sequence[int]
    activations: Sequence[callable] = nn.gelu
    kernel_init: callable = default_init()
    ensemble_size: int = 2

    def setup(self):
        if self.ensemble_size > 1:
            self.mlp = make_ensemble(LayerNormMLP, self.ensemble_size)(
                hidden_dims=(*self.hidden_dims, 1),
                activations=self.activations,
                activate_final=False,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )
        else:
            self.mlp = LayerNormMLP(
                hidden_dims=(*self.hidden_dims, 1),
                activations=self.activations,
                activate_final=False,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        goals: jnp.ndarray | None = None,
        actions: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x = combine_inputs(obs, goals, actions)
        v = self.mlp(x)
        return v.squeeze(axis=-1)


class ActionValueNetwork(ValueNetwork):

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        return super().__call__(obs, actions=actions)


class ActorNetwork(nn.Module):
    """
    Gaussian policy network
    """
    hidden_dims: Sequence[int]
    action_dim: int
    activations: callable = nn.gelu
    kernel_init: callable = default_init()
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    def setup(self):
        self.mlp = LayerNormMLP(
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            kernel_init=self.kernel_init,
            layer_norm=True,
        )
        self.mean_layer = nn.Dense(self.action_dim, kernel_init=self.kernel_init)
        self.log_std_layer = nn.Dense(self.action_dim, kernel_init=self.kernel_init)

    @nn.compact
    def __call__(self, obs: jnp.ndarray, goals: jnp.ndarray | None = None) -> distrax.Distribution:
        x = combine_inputs(obs, goals)
        x = self.mlp(x)
        means = self.mean_layer(x)
        log_stds = self.log_std_layer(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        stds = jnp.exp(log_stds)
        dist = distrax.MultivariateNormalDiag(means, stds)
        return dist


class BilinearValueNetwork(nn.Module):
    hidden_dims: Sequence[int]
    activations: Sequence[callable] = nn.gelu
    kernel_init: callable = default_init()
    latent_dim: int = 256
    ensemble_size: int = 2

    def setup(self):
        if self.ensemble_size > 1:
            self.phi = make_ensemble(LayerNormMLP, self.ensemble_size)(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activations=self.activations,
                activate_final=True,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )
            self.psi = make_ensemble(LayerNormMLP, self.ensemble_size)(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activations=self.activations,
                activate_final=True,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )
        else:
            self.phi = LayerNormMLP(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activations=self.activations,
                activate_final=True,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )
            self.psi = LayerNormMLP(
                hidden_dims=(*self.hidden_dims, self.latent_dim),
                activations=self.activations,
                activate_final=True,
                kernel_init=self.kernel_init,
                layer_norm=True,
            )

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, goal_obs: jnp.ndarray, actions: jnp.ndarray | None = None
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = combine_inputs(obs, actions=actions)
        phi_out = self.phi(x)
        psi_out = self.psi(goal_obs)
        v = jnp.sum(phi_out * psi_out, axis=-1)
        v = v / jnp.sqrt(phi_out.shape[-1])
        return v, phi_out, psi_out


class BilinearCriticNetwork(BilinearValueNetwork):

    @nn.compact
    def __call__(
        self, obs: jnp.ndarray, goal_obs: jnp.ndarray, actions: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        return super().__call__(obs, goal_obs, actions)


if __name__ == "__main__":
    # test ensemble of critics
    ensemble_value = BilinearValueNetwork(hidden_dims=[256, 256], ensemble_size=3)
    obs = jnp.zeros((1, 10))
    goal_obs = jnp.zeros((1, 10))
    v, _, _ = ensemble_value.apply(
        {"params": ensemble_value.init(jax.random.PRNGKey(0), obs, goal_obs)["params"]}, obs, goal_obs
    )
    print(v.shape)

    # test non ensemble critic
    non_ensemble_value = BilinearValueNetwork(hidden_dims=[256, 256], ensemble_size=1)
    v, _, _ = non_ensemble_value.apply(
        {"params": non_ensemble_value.init(jax.random.PRNGKey(0), obs, goal_obs)["params"]}, obs, goal_obs
    )
    print(v.shape)
