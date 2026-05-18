"""Pure JAX XYZ-Fourier curve kernels."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ._device_scalars import device_one, two_pi
from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_runtime_float64 as _as_runtime_float64,
)


def _mode_numbers(order, *, reference):
    one = jax.lax.stop_gradient(device_one(reference))
    return jnp.cumsum(jnp.broadcast_to(one, (int(order),)))


def _constant_row(length, value, *, reference):
    one = jax.lax.stop_gradient(device_one(reference))
    zero = one - one
    scalar = one if value == 1.0 else zero
    return jnp.broadcast_to(scalar, (1, int(length)))


def _interleave_harmonics(first, second):
    return jnp.reshape(jnp.stack((first, second), axis=1), (-1, first.shape[1]))


def _fourier_basis_terms(quadpoints, order):
    quadpoints = _as_runtime_float64(quadpoints, reference=quadpoints)
    angle_scale = jax.lax.stop_gradient(two_pi(quadpoints))
    points = angle_scale * quadpoints
    mode_numbers = _mode_numbers(order, reference=points)
    phase = jnp.expand_dims(mode_numbers, axis=1) * jnp.expand_dims(points, axis=0)
    sin_phase = jnp.sin(phase)
    cos_phase = jnp.cos(phase)
    mode_scale = angle_scale * mode_numbers
    mode_scale_sq = mode_scale * mode_scale
    mode_scale_cu = mode_scale_sq * mode_scale
    zero_row = _constant_row(points.shape[0], 0.0, reference=points)

    basis = jnp.concatenate(
        (
            _constant_row(points.shape[0], 1.0, reference=points),
            _interleave_harmonics(sin_phase, cos_phase),
        ),
        axis=0,
    )
    dash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                jnp.expand_dims(mode_scale, axis=1) * cos_phase,
                -jnp.expand_dims(mode_scale, axis=1) * sin_phase,
            ),
        ),
        axis=0,
    )
    dashdash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                -jnp.expand_dims(mode_scale_sq, axis=1) * sin_phase,
                -jnp.expand_dims(mode_scale_sq, axis=1) * cos_phase,
            ),
        ),
        axis=0,
    )
    dashdashdash_basis = jnp.concatenate(
        (
            zero_row,
            _interleave_harmonics(
                -jnp.expand_dims(mode_scale_cu, axis=1) * cos_phase,
                jnp.expand_dims(mode_scale_cu, axis=1) * sin_phase,
            ),
        ),
        axis=0,
    )
    return basis, dash_basis, dashdash_basis, dashdashdash_basis


def jaxfouriercurve_pure(dofs, quadpoints, order):
    """Return XYZ-Fourier curve positions."""
    dofs = _as_jax_float64(dofs)
    coeffs = jnp.reshape(dofs, (3, dofs.shape[0] // 3))
    basis, _, _, _ = _fourier_basis_terms(quadpoints, order)
    gamma = coeffs @ basis
    return jnp.moveaxis(gamma, 0, -1)


def jaxfouriercurve_geometry_pure(dofs, quadpoints, order):
    """Return XYZ-Fourier geometry and its first three quadpoint derivatives."""
    dofs = _as_jax_float64(dofs)
    coeffs = jnp.reshape(dofs, (3, dofs.shape[0] // 3))
    basis, dash_basis, dashdash_basis, dashdashdash_basis = _fourier_basis_terms(
        quadpoints,
        order,
    )
    gamma = coeffs @ basis
    gammadash = coeffs @ dash_basis
    gammadashdash = coeffs @ dashdash_basis
    gammadashdashdash = coeffs @ dashdashdash_basis
    return tuple(
        jnp.moveaxis(component, 0, -1)
        for component in (gamma, gammadash, gammadashdash, gammadashdashdash)
    )
