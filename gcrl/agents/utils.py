import jax.numpy as jnp


def reduce_ensemble(values: jnp.ndarray, reduction: str, name: str) -> jnp.ndarray:
    if values.ndim == 1:
        return values

    if reduction == "mean":
        return values.mean(axis=0)
    if reduction == "min":
        return values.min(axis=0)
    raise ValueError(f"Unknown {name} ensemble reduction: {reduction}")


def ensemble_std(values: jnp.ndarray) -> jnp.ndarray:
    if values.ndim == 1:
        return jnp.array(0.0)
    return values.std(axis=0).mean()


def with_ensemble_axis(values: jnp.ndarray, sample_ndim: int) -> jnp.ndarray:
    if values.ndim == sample_ndim:
        return values[jnp.newaxis]
    return values
