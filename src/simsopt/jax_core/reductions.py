"""Shared reduction helpers for parity-sensitive JAX kernels."""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

VALID_REDUCTION_MODES = ("default", "strict_oracle")
_VALID_SCALAR_BASELINES = ("pairwise", "vdot")

__all__ = [
    "VALID_REDUCTION_MODES",
    "compensated_sum_flat",
    "pairwise_sum_axis",
    "pairwise_sum_flat",
    "scalar_square_sum",
    "validate_reduction_mode",
]


def _zero_scalar(dtype):
    return jnp.array(0, dtype=dtype)


def _next_power_of_two(size: int) -> int:
    if size <= 1:
        return 1
    return 1 << (size - 1).bit_length()


def _pad_axis0(array, padded_size: int):
    pad_rows = padded_size - array.shape[0]
    if pad_rows <= 0:
        return array
    padding_config = [(0, pad_rows, 0)] + [(0, 0, 0)] * (array.ndim - 1)
    return lax.pad(array, _zero_scalar(array.dtype), padding_config)


def validate_reduction_mode(reduction_mode: str) -> str:
    if reduction_mode not in VALID_REDUCTION_MODES:
        raise ValueError(
            f"Unknown reduction_mode={reduction_mode!r}. "
            f"Accepted: {VALID_REDUCTION_MODES}"
        )
    return reduction_mode


def _pairwise_reduce_axis0(array):
    reduced = array
    while reduced.shape[0] > 1:
        pair_shape = (reduced.shape[0] // 2, 2) + tuple(reduced.shape[1:])
        paired = jnp.reshape(reduced, pair_shape)
        reduced = paired[:, 0, ...] + paired[:, 1, ...]
    return reduced


def pairwise_sum_axis(array, *, axis: int):
    """Reduce ``array`` along ``axis`` using a fixed binary addition tree."""
    axis_index = axis if axis >= 0 else array.ndim + axis
    axis_size = array.shape[axis_index]
    if axis_size == 0:
        return jnp.sum(array, axis=axis_index)

    reduced = jnp.moveaxis(array, axis_index, 0)
    reduced = _pad_axis0(reduced, _next_power_of_two(axis_size))
    return jnp.squeeze(_pairwise_reduce_axis0(reduced), axis=0)


def pairwise_sum_flat(array):
    """Reduce all entries of ``array`` using a fixed binary addition tree."""
    reduced = jnp.ravel(array)
    size = reduced.shape[0]
    if size == 0:
        return jnp.sum(reduced)

    reduced = _pad_axis0(reduced[:, None], _next_power_of_two(size))
    return _pairwise_reduce_axis0(reduced)[0, 0]


def compensated_sum_flat(array):
    """Reduce all entries using a compensated Kahan summation loop."""
    reduced = jnp.ravel(array)
    size = reduced.shape[0]
    if size == 0:
        return jnp.sum(reduced)

    total = _zero_scalar(reduced.dtype)
    compensation = _zero_scalar(reduced.dtype)

    def body(index, state):
        running_total, running_compensation = state
        value = reduced[index]
        adjusted = value - running_compensation
        updated_total = running_total + adjusted
        updated_compensation = (updated_total - running_total) - adjusted
        return updated_total, updated_compensation

    total, compensation = lax.fori_loop(0, size, body, (total, compensation))
    del compensation
    return total


def scalar_square_sum(
    array,
    *,
    reduction_mode: str = "default",
    default: str = "pairwise",
):
    """Return ``sum(array * array)`` using the selected escalation tier."""
    reduction_mode = validate_reduction_mode(reduction_mode)
    if default not in _VALID_SCALAR_BASELINES:
        raise ValueError(
            f"Unknown scalar reduction baseline {default!r}. "
            f"Accepted: {_VALID_SCALAR_BASELINES}"
        )

    flat = jnp.ravel(jnp.asarray(array))
    squared = (jnp.conj(flat) * flat).real

    if reduction_mode == "strict_oracle":
        return compensated_sum_flat(squared)
    if default == "pairwise":
        return pairwise_sum_flat(squared)
    return jnp.vdot(flat, flat).real
