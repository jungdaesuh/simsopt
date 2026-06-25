"""Pure JAX composition primitives for ``MagneticFieldSum`` / ``MagneticFieldMultiply``.

This module exposes pure functional JAX kernels that compose Biot-Savart /
analytic-field-style results (B, dB, A, dA, d2B, d2A) across a sequence of
JAX-native child fields without materializing intermediate device arrays
back to host. It is the on-device pure-JAX equivalent of the
``MagneticFieldSum._B_impl`` / ``MagneticFieldMultiply._B_impl`` host-side
NumPy sum + scalar multiply pattern in ``simsopt.field.magneticfield``.

Contract
--------

Each composition primitive takes:

* ``children`` — a Python tuple of ``Callable[[jax.Array], jax.Array]`` that
  accept points of shape ``(N, 3)`` and return the corresponding field
  quantity. The tuple length is fixed at trace time. Each callable should
  close over its child's frozen JAX spec so the composition primitive does
  not need to dispatch on spec type.
* ``points`` — a JAX array of cartesian evaluation points with shape
  ``(N, 3)``.

The return value is a JAX array with the same shape as the underlying
single-child kernel output.

Strict mode is enforced at the ``MagneticFieldSum`` / ``MagneticFieldMultiply``
boundary in ``simsopt.field.magneticfield``; this module remains pure
JAX and does no environment / backend dispatch.
"""

from __future__ import annotations

from typing import Callable

import jax

from ._math_utils import as_jax_float64 as _as_jax_float64


__all__ = [
    "compose_B_sum",
    "compose_dB_sum",
    "compose_d2B_sum",
    "compose_A_sum",
    "compose_dA_sum",
    "compose_d2A_sum",
    "compose_B_scaled",
    "compose_dB_scaled",
    "compose_d2B_scaled",
    "compose_A_scaled",
    "compose_dA_scaled",
    "compose_d2A_scaled",
]


KernelCallable = Callable[[jax.Array], jax.Array]


def _validated_points(points: object) -> jax.Array:
    points_arr = _as_jax_float64(points)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3); got {tuple(points_arr.shape)!r}."
        )
    return points_arr


def _compose_sum(children: tuple[KernelCallable, ...], points: jax.Array) -> jax.Array:
    """Sum the outputs of ``children(points)`` on device with no host round-trip."""
    if len(children) == 0:
        raise ValueError("compose_*_sum requires at least one child kernel.")
    total = children[0](points)
    for child in children[1:]:
        total = total + child(points)
    return total


def compose_B_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device B = sum_k child_k.B(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def compose_dB_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device dB/dX = sum_k child_k.dB/dX(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def compose_d2B_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device d2B/dXdX = sum_k child_k.d2B/dXdX(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def compose_A_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device A = sum_k child_k.A(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def compose_dA_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device dA/dX = sum_k child_k.dA/dX(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def compose_d2A_sum(children: tuple[KernelCallable, ...], points: object) -> jax.Array:
    """On-device d2A/dXdX = sum_k child_k.d2A/dXdX(points)."""
    return _compose_sum(tuple(children), _validated_points(points))


def _compose_scaled(
    child: KernelCallable,
    scalar: object,
    points: jax.Array,
) -> jax.Array:
    # Route the host scalar through ``as_jax_float64`` so the host-to-device
    # staging step uses an explicit ``jax.device_put``, which is the only
    # form allowed under ``jax.transfer_guard("disallow")``.
    scaled = _as_jax_float64(scalar)
    return scaled * child(points)


def compose_B_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.B(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))


def compose_dB_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.dB/dX(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))


def compose_d2B_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.d2B/dXdX(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))


def compose_A_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.A(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))


def compose_dA_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.dA/dX(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))


def compose_d2A_scaled(
    child: KernelCallable,
    scalar: object,
    points: object,
) -> jax.Array:
    """On-device ``scalar * child.d2A/dXdX(points)``."""
    return _compose_scaled(child, scalar, _validated_points(points))
