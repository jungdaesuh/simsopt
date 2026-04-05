import jax
import numpy as np


def host_array(value, *, dtype=np.float64) -> np.ndarray:
    """Materialize a value on the host through an explicit JAX boundary."""
    return np.asarray(jax.device_get(value), dtype=dtype)


def host_float(value) -> float:
    """Materialize a scalar on the host through an explicit JAX boundary."""
    return float(host_array(value))


def host_bool(value) -> bool:
    """Materialize a boolean on the host through an explicit JAX boundary."""
    return bool(host_array(value, dtype=np.bool_))
