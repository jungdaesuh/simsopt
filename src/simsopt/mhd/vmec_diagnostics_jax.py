# coding: utf-8
# Copyright (c) HiddenSymmetries Development Team.
# Distributed under the terms of the MIT License

"""JAX-backed public VMEC diagnostic entrypoints."""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from simsopt.jax_core.vmec_fieldlines import theta_vmec_from_theta_pest_implicit_jax
from simsopt.jax_core.vmec_geometry import VmecGeometryResultsJAX
from simsopt.jax_core.vmec_geometry import (
    vmec_compute_geometry_jax as _vmec_compute_geometry_jax,
)

from ._vmec_frozen import VmecFrozenSplineState, vmec_freeze_splines, vmec_spline_eval

__all__ = [
    "VmecGeometryResultsJAX",
    "VmecFrozenSplineState",
    "vmec_compute_geometry_jax",
    "vmec_fieldlines_jax",
    "vmec_freeze_splines",
]


def _frozen_state(vs) -> VmecFrozenSplineState:
    return vs if isinstance(vs, VmecFrozenSplineState) else vmec_freeze_splines(vs)


def vmec_compute_geometry_jax(vs, s, theta, phi, phi_center: float = 0.0):
    """Evaluate VMEC geometry diagnostics through the pure JAX kernel.

    ``vs`` may be a :class:`~simsopt.mhd.vmec.Vmec` object, the host
    ``vmec_splines`` structure, or an already frozen
    :class:`VmecFrozenSplineState`. Differentiated callers must avoid the
    inherited ``s = 0`` drift-normalization pole and zero-``grad_B`` diagnostic
    pole unless they provide a model-specific limiting treatment.
    """
    frozen_state = _frozen_state(vs)
    return _vmec_compute_geometry_jax(frozen_state, s, theta, phi, phi_center)


def vmec_fieldlines_jax(
    vs,
    s,
    alpha,
    theta1d=None,
    phi1d=None,
    phi_center: float = 0.0,
):
    """Evaluate VMEC fieldline diagnostics through JAX kernels.

    Differentiated callers inherit the same VMEC diagnostic-domain limits as
    :func:`vmec_compute_geometry_jax`.
    """
    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")

    frozen_state = _frozen_state(vs)
    s_array = jnp.atleast_1d(jnp.asarray(s, dtype=jnp.float64))
    alpha_array = jnp.atleast_1d(jnp.asarray(alpha, dtype=jnp.float64))
    iota = vmec_spline_eval(frozen_state.iota, s_array)
    if theta1d is None:
        phi1d_array = jnp.atleast_1d(jnp.asarray(phi1d, dtype=jnp.float64))
        phi = jnp.broadcast_to(
            phi1d_array.reshape(1, 1, phi1d_array.shape[0]),
            (s_array.shape[0], alpha_array.shape[0], phi1d_array.shape[0]),
        )
        theta_pest = alpha_array[None, :, None] + iota[:, None, None] * (
            phi1d_array[None, None, :] - phi_center
        )
        theta1d_array = None
    else:
        theta1d_array = jnp.atleast_1d(jnp.asarray(theta1d, dtype=jnp.float64))
        theta_pest = jnp.broadcast_to(
            theta1d_array.reshape(1, 1, theta1d_array.shape[0]),
            (s_array.shape[0], alpha_array.shape[0], theta1d_array.shape[0]),
        )
        phi = (
            phi_center
            + (theta1d_array[None, None, :] - alpha_array[None, :, None])
            / iota[:, None, None]
        )
        phi1d_array = None

    lmns = vmec_spline_eval(frozen_state.lmns, s_array)
    theta_vmec = theta_vmec_from_theta_pest_implicit_jax(
        theta_pest,
        phi,
        frozen_state.xm,
        frozen_state.xn,
        lmns,
        max_iter=20,
        tol=1e-13,
    )
    results = _vmec_compute_geometry_jax(
        frozen_state,
        s_array,
        theta_vmec,
        phi,
        phi_center,
    )
    return replace(
        results,
        nalpha=int(alpha_array.shape[0]),
        nl=int(theta_pest.shape[2]),
        alpha=alpha_array,
        theta1d=theta1d_array,
        phi1d=phi1d_array,
    )
