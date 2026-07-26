"""Finite-difference Jacobian kernels for explicit JAX state vectors."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import PartitionSpec as P

from simsopt_jax.runtime.host_boundary import host_bool

from ._math_utils import runtime_device_put as _runtime_device_put
from ._math_utils import runtime_init_array as _runtime_init_array
from ._math_utils import runtime_init_scalar as _runtime_init_scalar

__all__ = [
    "forward_jacobian_shard_map_columns",
    "forward_jacobian_shard_map",
    "forward_jacobian_vmap",
]


def _residual_vector(fn: Callable[[jax.Array], jax.Array], x: jax.Array) -> jax.Array:
    return jnp.ravel(fn(x))


def _as_index_vector(value: object) -> jax.Array:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return jnp.ravel(value)
    return jnp.ravel(_runtime_device_put(np.asarray(value)))


def _as_runtime_array(value: object) -> jax.Array:
    if isinstance(value, (jax.Array, jax.core.Tracer)):
        return jnp.asarray(value)
    return _runtime_device_put(np.asarray(value))


def _index_range(size: int) -> jax.Array:
    return jax.lax.iota(jnp.int32, int(size))


def _device_zero_like(value: jax.Array) -> jax.Array:
    return value - value


def _step_vector(x_flat: jax.Array, abs_step: float, rel_step: float) -> jax.Array:
    if abs_step < 0.0:
        raise ValueError("abs_step must be >= 0")
    if rel_step < 0.0:
        raise ValueError("rel_step must be >= 0")
    dtype = np.dtype(x_flat.dtype)
    abs_step_host = np.asarray(abs_step, dtype=dtype)
    rel_step_host = np.asarray(rel_step, dtype=dtype)
    if isinstance(x_flat, jax.core.Tracer) and abs_step_host == 0.0:
        raise ValueError("Finite difference step size cannot be 0. Increase abs_step.")
    if abs_step_host == 0.0 and rel_step_host == 0.0:
        raise ValueError("Finite difference step size cannot be 0. Increase abs_step.")
    rel_step_device = _runtime_init_scalar(rel_step, x_flat.dtype)
    abs_step_device = _runtime_init_scalar(abs_step, x_flat.dtype)
    steps = jnp.maximum(jnp.abs(x_flat) * rel_step_device, abs_step_device)
    if not isinstance(x_flat, jax.core.Tracer) and host_bool(
        jnp.any(steps == _runtime_init_scalar(0, steps.dtype))
    ):
        raise ValueError("Finite difference step size cannot be 0. Increase abs_step.")
    return steps


def _finite_difference_column(
    fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    x_shape: tuple[int, ...],
    f0: jax.Array,
    steps: jax.Array,
    index: jax.Array,
    *,
    diff_method: str,
) -> jax.Array:
    x_flat = jnp.ravel(x0)
    positions = _index_range(x_flat.size).astype(index.dtype)
    selected = positions == index
    step = jnp.sum(jnp.where(selected, steps, _device_zero_like(steps)))
    perturbation = jnp.where(selected, step, _device_zero_like(x_flat))

    if diff_method == "forward":
        value = _residual_vector(fn, jnp.reshape(x_flat + perturbation, x_shape))
        return (value - f0) / step
    if diff_method == "centered":
        value_plus = _residual_vector(fn, jnp.reshape(x_flat + perturbation, x_shape))
        value_minus = _residual_vector(fn, jnp.reshape(x_flat - perturbation, x_shape))
        return (value_plus - value_minus) / (step + step)
    raise ValueError(f"Unsupported finite-difference method {diff_method!r}.")


def forward_jacobian_vmap(
    fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    abs_step: float,
    rel_step: float = 0.0,
    diff_method: str = "forward",
) -> jax.Array:
    """Return ``d r / d x`` for ``r = fn(x)`` using vectorized perturbations."""
    x = _as_runtime_array(x0)
    x_flat = jnp.ravel(x)
    f0 = _residual_vector(fn, x)
    steps = _step_vector(x_flat, abs_step, rel_step)
    columns = jax.vmap(
        lambda index: _finite_difference_column(
            fn,
            x,
            x.shape,
            f0,
            steps,
            index,
            diff_method=diff_method,
        )
    )(_index_range(x_flat.size))
    return jnp.transpose(columns)


def forward_jacobian_shard_map_columns(
    fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    column_indices: jax.Array,
    abs_step: float,
    rel_step: float = 0.0,
    diff_method: str = "forward",
    *,
    mesh,
) -> jax.Array:
    """Return selected ``d r / d x`` columns sharded over ``'dof'``.

    Callers that compile this function under strict transfer guards must pass
    ``x0`` and any closed-over residual data already placed on ``mesh``.
    """
    x = _as_runtime_array(x0)
    x_flat = jnp.ravel(x)
    dof_count = int(x_flat.size)
    columns = _as_index_vector(column_indices)
    selected_count = int(columns.size)
    device_count = int(mesh.shape["dof"])
    padded_selected_count = selected_count + (-selected_count) % device_count
    pad_count = padded_selected_count - selected_count
    if pad_count:
        columns = jnp.concatenate(
            (
                columns,
                _runtime_init_array((pad_count,), dof_count - 1, columns.dtype),
            )
        )
    valid_columns = _runtime_init_array((selected_count,), True, jnp.bool_)
    if pad_count:
        valid_columns = jnp.concatenate(
            (
                valid_columns,
                _runtime_init_array((pad_count,), False, jnp.bool_),
            )
        )
    f0 = _residual_vector(fn, x)
    steps = _step_vector(x_flat, abs_step, rel_step)

    @partial(
        jax.shard_map,
        mesh=mesh,
        in_specs=(P("dof"), P("dof"), P(), P(), P()),
        out_specs=P("dof", None),
        check_vma=True,
    )
    def sharded_columns(index_block, valid_block, x_value, f0_value, step_values):
        def column(index, valid):
            finite_difference = _finite_difference_column(
                fn,
                x_value,
                x_value.shape,
                f0_value,
                step_values,
                index,
                diff_method=diff_method,
            )
            return jnp.where(
                valid, finite_difference, _device_zero_like(finite_difference)
            )

        return jax.vmap(column)(index_block, valid_block)

    padded_columns = sharded_columns(columns, valid_columns, x, f0, steps)
    return jnp.transpose(padded_columns[:selected_count])


def forward_jacobian_shard_map(
    fn: Callable[[jax.Array], jax.Array],
    x0: jax.Array,
    abs_step: float,
    rel_step: float = 0.0,
    diff_method: str = "forward",
    *,
    mesh,
) -> jax.Array:
    """Return ``d r / d x`` with finite-difference columns sharded over ``'dof'``."""
    dof_count = int(jnp.ravel(_as_runtime_array(x0)).size)
    return forward_jacobian_shard_map_columns(
        fn,
        x0,
        _index_range(dof_count),
        abs_step,
        rel_step,
        diff_method,
        mesh=mesh,
    )
