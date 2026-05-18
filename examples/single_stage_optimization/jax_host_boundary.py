import numpy as np

from simsopt._core.jax_host_boundary import (
    host_array as host_array,
    host_bool as host_bool,
    host_int as host_int,
    host_scalar,
)


def host_float(value) -> float:
    """Materialize a scalar on the host through an explicit JAX boundary."""
    return float(host_scalar(value, dtype=np.float64))
