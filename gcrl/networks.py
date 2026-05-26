from typing import Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp


def default_init():
    return nn.initializers.orthogonal()


class LayerNormMLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Sequence[callable] = nn.gelu
    activate_final: bool = False
    kernel_init: callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, hidden_dim in enumerate(self.hidden_dims):
            x = nn.Dense(hidden_dim, kernel_init=self.kernel_init)(x)
            x = nn.LayerNorm()(x)
            if i < len(self.hidden_dims) - 1 or self.activate_final:
                x = self.activations(x)
        return x


class ValueNetwork(nn.Module):
    hidden_dims: Sequence[int]
    activations: Sequence[callable] = nn.gelu
    kernel_init: callable = default_init()

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = LayerNormMLP(
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            activate_final=False,
            kernel_init=self.kernel_init,
        )(x)
        return nn.Dense(1, kernel_init=self.kernel_init)(x).squeeze(-1)


class ActorNetwork(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    activations: callable = nn.gelu
    kernel_init: callable = default_init()
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> distrax.Distribution:
        x = LayerNormMLP(
            hidden_dims=self.hidden_dims,
            activations=self.activations,
            activate_final=True,
            kernel_init=self.kernel_init,
        )(x)
        means = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(x)
        log_stds = nn.Dense(self.action_dim, kernel_init=self.kernel_init)(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)
        return distrax.MultivariateNormalDiag(means, jnp.exp(log_stds))
