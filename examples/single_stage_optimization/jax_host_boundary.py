import numpy as np

from simsopt._core.jax_host_boundary import (
    host_array,
    host_bool,
    host_scalar,
)


def host_float(value) -> float:
    """Materialize a scalar on the host through an explicit JAX boundary."""
    return float(host_scalar(value, dtype=np.float64))
