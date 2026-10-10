"""Native ``NonQuasiSymmetricRatio``'s objective, in JAX.

:func:`non_quasi_symmetric_ratio` evaluates, on a surface spec and a grouped
coil spec, native ``NonQuasiSymmetricRatio``'s

    J = mean(dS B_nonQS^2) / mean(dS B_QS^2),

with ``dS = |n|``, ``B_QS`` the ``dS``-weighted mean of ``|B|`` along
``axis`` (``0``: over ``phi``, quasi-axisymmetry; ``1``: over ``theta``,
quasi-poloidal symmetry) and ``B_nonQS = |B| - B_QS``, in native's order of
operations, together with its partial derivatives: with respect to the native
surface DOF vector (all of them, native's ``dJ_by_dsurfacecoefficients``) and
with respect to the coils (native's ``B_vjp(dJ_by_dB)``, as cotangents of the
coil spec). Zero fields and degenerate normals give native's non-finite
values; the float64 limits of :mod:`simsopt_jax.core.surface_geometry`
apply.
"""

from __future__ import annotations

from functools import partial
from typing import cast

import jax
import jax.numpy as jnp

from .field import grouped_biot_savart_B_from_spec
from .specs import GroupedCoilSetSpec, SurfaceSpec
from .surface_fourier_series import surface_get_dofs, surface_spec_with_dofs
from .surface_geometry import _norm, surface_gamma, surface_normal

__all__ = ["non_quasi_symmetric_ratio"]


def _ratio(surface: SurfaceSpec, coils: GroupedCoilSetSpec, axis: int) -> jax.Array:
    gamma = surface_gamma(surface)
    nphi, ntheta = gamma.shape[:2]
    B = cast(jax.Array, grouped_biot_savart_B_from_spec(gamma.reshape(-1, 3), coils))
    modB = _norm(B.reshape(nphi, ntheta, 3))
    dS = _norm(surface_normal(surface))
    B_QS = jnp.expand_dims(jnp.mean(modB * dS, axis=axis) / jnp.mean(dS, axis=axis), axis)
    B_nonQS = modB - B_QS
    return jnp.mean(dS * B_nonQS**2) / jnp.mean(dS * B_QS**2)


@partial(jax.jit, static_argnames=("axis",))
def non_quasi_symmetric_ratio(
    surface: SurfaceSpec, coils: GroupedCoilSetSpec, *, axis: int
) -> tuple[jax.Array, jax.Array, GroupedCoilSetSpec]:
    """``(J, dJ/dsurface_dofs, dJ/dcoils)``: ``J`` at ``surface`` and the
    coils, its gradient over the surface DOFs (the field following the
    points) and the coil-spec cotangent of its field (the points held). New
    coefficient and coil values reuse the compiled program; ``axis`` selects
    one.

    Args:
        surface (SurfaceSpec): Immutable surface coefficients and quadrature grid; all
            native DOFs are included.
        coils (GroupedCoilSetSpec): Grouped coil geometry in meters and currents in
            amperes.
        axis (int): Average field magnitude over phi (0, quasi-axisymmetric) or theta
            (1, quasi-poloidal).

    Returns:
        tuple[jax.Array, jax.Array, GroupedCoilSetSpec]: Dimensionless ratio scalar
            shape (), gradient over all native surface DOFs of shape (nsurface,), and
            coil cotangents with matching leaf shapes. Ratio is mean(dS * B_nonQS^2) /
            mean(dS * B_QS^2), with dS-weighted B_QS.
    """

    def ratio(dofs: jax.Array, coils: GroupedCoilSetSpec) -> jax.Array:
        return _ratio(surface_spec_with_dofs(surface, dofs), coils, axis)

    value, (dsurface, dcoils) = jax.value_and_grad(ratio, argnums=(0, 1))(
        surface_get_dofs(surface), coils
    )
    return value, dsurface, dcoils
