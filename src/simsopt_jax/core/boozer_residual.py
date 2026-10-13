"""Boozer residual of native ``boozer_surface_residual`` and its derivatives.

The unweighted equation is Eq. (5a) of Giuliani et al., "Direct computation
of magnetic surfaces in Boozer coordinates and coil optimization for
quasi-symmetry" (2022), https://doi.org/10.1017/S0022377822000563.
As in that paper, the surface coordinates are normalized turns in [0, 1).
The optional inverse-field weighting below is the native simsopt convention.

At each surface quadrature point the residual is

.. math::

    \\mathbf r = w \\bigl(G \\mathbf B - |\\mathbf B|^2
        (\\mathbf x_\\varphi + \\iota \\mathbf x_\\theta)\\bigr),
    \\qquad w = 1/|\\mathbf B| \\text{ if weight_inv_modB, else } 1,

with :math:`\\mathbf B` the field at the point and :math:`\\mathbf x_\\varphi`,
:math:`\\mathbf x_\\theta` the native ``gammadash1`` and ``gammadash2``. The
unknowns are ``x = [surface DOFs, iota, G]``, or ``[surface DOFs, iota]`` when
``G`` is not optimized and is a constant.

Derivatives with respect to ``x`` are JAX transforms of the residual as a
function of ``x``. The field enters through its Taylor model at the points
(:func:`taylor_field`): ``B``, ``dB/dX`` and ``d2B/dXdX`` there, up to the
order requested, so no derivative in ``x`` passes through the Biot-Savart
kernel, while the coefficients keep their coil dependence. The nesting is
fixed in :func:`residual_derivatives` (forward over forward) and
:func:`objective_derivatives` (forward over reverse): reverse over forward
keeps every forward tangent for the reverse sweep, which on large grids does
not fit in memory.

Every function is traceable and meant to run inside a jitted program.
Residual rows are point-major and component-minor, as natively.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import lax

from simsopt_jax.pytree import pytree_dataclass

__all__ = [
    "BoozerPoints",
    "boozer_least_squares",
    "boozer_point_residual",
    "boozer_residual",
    "objective_derivatives",
    "residual_derivatives",
    "taylor_field",
]


@pytree_dataclass(
    data=(
        "B",
        "xphi",
        "xtheta",
        "dB_by_dX",
        "d2B_by_dXdX",
        "dgamma_by_dcoeff",
        "dgammadash1_by_dcoeff",
        "dgammadash2_by_dcoeff",
    )
)
class BoozerPoints:
    """Field and surface at the quadrature points, flattened over the points.

    ``B``, ``xphi`` and ``xtheta`` are ``(npoints, 3)``; ``dB_by_dX``
    ``(npoints, 3, 3)`` and ``d2B_by_dXdX`` ``(npoints, 3, 3, 3)`` have the
    native layout (derivative directions first, field component last); the
    ``*_by_dcoeff`` derivatives of ``gamma``, ``gammadash1`` and ``gammadash2``
    are ``(npoints, 3, nsurface)``. Values need only the first three; first
    derivatives add ``dB_by_dX`` and the coefficient derivatives, second
    derivatives ``d2B_by_dXdX``. Unneeded fields may be ``None``.

    Args:
        B (jax.Array): Magnetic field in teslas, shape (npoints, 3).
        xphi (jax.Array): Surface toroidal tangent in meters per turn, shape (npoints,
            3).
        xtheta (jax.Array): Surface poloidal tangent in meters per turn, shape (npoints,
            3).
        dB_by_dX (jax.Array | None): Field gradient in teslas per meter, shape (npoints,
            3, 3), spatial direction before field component; needed for first
            derivatives.
        d2B_by_dXdX (jax.Array | None): Field curvature in teslas per meter squared,
            shape (npoints, 3, 3, 3), two spatial directions before field component;
            needed for second derivatives.
        dgamma_by_dcoeff (jax.Array | None): Position derivative with respect to all
            nsurface native coefficients, shape (npoints, 3, nsurface); needed for first
            derivatives.
        dgammadash1_by_dcoeff (jax.Array | None): Toroidal tangent coefficient
            derivative, shape (npoints, 3, nsurface); needed for first derivatives.
        dgammadash2_by_dcoeff (jax.Array | None): Poloidal tangent coefficient
            derivative, shape (npoints, 3, nsurface); needed for first derivatives.
    """

    B: jax.Array
    xphi: jax.Array
    xtheta: jax.Array
    dB_by_dX: jax.Array | None = None
    d2B_by_dXdX: jax.Array | None = None
    dgamma_by_dcoeff: jax.Array | None = None
    dgammadash1_by_dcoeff: jax.Array | None = None
    dgammadash2_by_dcoeff: jax.Array | None = None


def _required(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"these derivatives need BoozerPoints.{name}.")
    return value


def _local_residual(local: jax.Array, weight_inv_modB: bool) -> jax.Array:
    """The residual ``(3,)`` of one point's local values ``(B, tangent, G)``."""
    B, tangent, G = local[:3], local[3:6], local[6]
    B2 = B[0] * B[0] + B[1] * B[1] + B[2] * B[2]
    residual = G * B - B2 * tangent
    return residual / jnp.sqrt(B2) if weight_inv_modB else residual


def boozer_point_residual(
    G, iota, B: jax.Array, xphi: jax.Array, xtheta: jax.Array, *, weight_inv_modB: bool
) -> jax.Array:
    """The residual at every point, flattened point-major to ``(3 npoints,)``.

    Args:
        G (float | jax.Array): Scalar shape () covariant Boozer field constant in tesla
            meters.
        iota (float | jax.Array): Scalar shape () dimensionless rotational transform.
        B (jax.Array): Magnetic field in teslas, shape (npoints, 3).
        xphi (jax.Array): Surface toroidal tangent in meters per turn, shape (npoints, 3).
        xtheta (jax.Array): Surface poloidal tangent in meters per turn, shape (npoints,
            3).
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        jax.Array: Residual shape (3 * npoints,), in tesla squared meters, or tesla
            meters when weighted.
    """
    G_column = jnp.broadcast_to(G, (B.shape[0], 1))
    local = jnp.concatenate((B, xphi + iota * xtheta, G_column), axis=1)
    residual_of_local = partial(_local_residual, weight_inv_modB=weight_inv_modB)
    return jax.vmap(residual_of_local)(local).reshape(-1)


def taylor_field(
    coefficients_at: Callable[[jax.Array], tuple[jax.Array, ...]], points: jax.Array
) -> jax.Array:
    """The field at ``points`` ``(npoints, 3)`` from its Taylor model at the
    points themselves, ``q0 = stop_gradient(points)``.

    ``coefficients_at(q0)`` returns ``B``, then optionally ``dB/dX`` and
    ``d2B/dXdX`` at ``q0`` (native layouts); the model ``B + dB (q - q0) +
    d2B[q - q0, q - q0] / 2`` equals ``B`` and has the field's derivatives in
    ``points`` up to the order given. Only ``q0`` is held constant: the
    coefficients keep their dependence on what they are computed from, such
    as the coils.

    Args:
        coefficients_at (Callable): Expansion points (npoints, 3) to the coefficients,
            shapes (npoints, 3), (npoints, 3, 3) and (npoints, 3, 3, 3).
        points (jax.Array): Evaluation points in meters, shape (npoints, 3).

    Returns:
        jax.Array: Magnetic field in teslas, shape (npoints, 3).
    """
    expansion = lax.stop_gradient(points)
    coefficients = coefficients_at(expansion)
    displacement = points - expansion
    field = coefficients[0]
    if len(coefficients) > 1:
        field = field + jnp.einsum("pj,pjl->pl", displacement, coefficients[1])
    if len(coefficients) > 2:
        field = field + 0.5 * jnp.einsum(
            "pj,pk,pjkl->pl", displacement, displacement, coefficients[2]
        )
    return field


def residual_derivatives(
    residual: Callable[[jax.Array], jax.Array], x: jax.Array, derivatives: int
) -> tuple[jax.Array, ...]:
    """``(r,)``, ``(r, J)`` or ``(r, J, H)`` of ``r = residual(x)``: ``J`` by
    ``jax.jacfwd`` and the per-residual Hessians ``H`` ``(nr, nx, nx)`` by
    ``jax.jacfwd`` of it.

    Args:
        residual (Callable): Shape (nx,) to the residual, shape (nr,).
        x (jax.Array): Shape (nx,) point of evaluation.
        derivatives (int): Derivative order, 0, 1 or 2.

    Returns:
        tuple[jax.Array, ...]: Residual (nr,), optionally Jacobian (nr, nx) and
            Hessians (nr, nx, nx).
    """
    if derivatives == 0:
        return (residual(x),)

    def residual_and_aux(point):
        value = residual(point)
        return value, value

    if derivatives == 1:
        jacobian, value = jax.jacfwd(residual_and_aux, has_aux=True)(x)
        return value, jacobian

    def jacobian_and_aux(point):
        jacobian, value = jax.jacfwd(residual_and_aux, has_aux=True)(point)
        return jacobian, (value, jacobian)

    hessian, (value, jacobian) = jax.jacfwd(jacobian_and_aux, has_aux=True)(x)
    return value, jacobian, hessian


def objective_derivatives(
    objective: Callable[[jax.Array], jax.Array], x: jax.Array, derivatives: int
) -> tuple[jax.Array, ...]:
    """``(f,)``, ``(f, g)`` or ``(f, g, H)`` of the scalar ``f = objective(x)``:
    ``g`` by ``jax.value_and_grad``, ``H`` by ``jax.jacfwd`` of it (forward
    over reverse, as ``jax.hessian``).

    Args:
        objective (Callable): Shape (nx,) to a scalar shape ().
        x (jax.Array): Shape (nx,) point of evaluation.
        derivatives (int): Derivative order, 0, 1 or 2.

    Returns:
        tuple[jax.Array, ...]: Value (), optionally gradient (nx,) and Hessian (nx, nx).
    """
    if derivatives == 0:
        return (objective(x),)
    if derivatives == 1:
        return jax.value_and_grad(objective)(x)

    def gradient_and_aux(point):
        value, gradient = jax.value_and_grad(objective)(point)
        return gradient, (value, gradient)

    hessian, (value, gradient) = jax.jacfwd(gradient_and_aux, has_aux=True)(x)
    return value, gradient, hessian


def _residual_of_x(
    G, iota, points: BoozerPoints, *, derivatives: int, optimize_G: bool, weight_inv_modB: bool
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """The residual as a function of ``x = [surface coefficient change, iota,
    G]`` (``G`` only if ``optimize_G``) and ``x`` at the points: position and
    tangents move linearly with the coefficients, the field by its Taylor
    model of order ``derivatives``."""
    dgamma = _required(points.dgamma_by_dcoeff, "dgamma_by_dcoeff")
    dxphi = _required(points.dgammadash1_by_dcoeff, "dgammadash1_by_dcoeff")
    dxtheta = _required(points.dgammadash2_by_dcoeff, "dgammadash2_by_dcoeff")
    coefficients = (points.B, _required(points.dB_by_dX, "dB_by_dX"))
    if derivatives == 2:
        coefficients += (_required(points.d2B_by_dXdX, "d2B_by_dXdX"),)
    nsurface = dgamma.shape[-1]

    def residual(x):
        change = x[:nsurface]
        B = taylor_field(lambda _: coefficients, dgamma @ change)
        return boozer_point_residual(
            x[nsurface + 1] if optimize_G else G,
            x[nsurface],
            B,
            points.xphi + dxphi @ change,
            points.xtheta + dxtheta @ change,
            weight_inv_modB=weight_inv_modB,
        )

    unknowns = (iota, G) if optimize_G else (iota,)
    x = jnp.concatenate((jnp.zeros(nsurface, dgamma.dtype), jnp.asarray(unknowns, dgamma.dtype)))
    return residual, x


def boozer_residual(
    G, iota, points: BoozerPoints, *, derivatives: int, optimize_G: bool, weight_inv_modB: bool
) -> tuple[jax.Array, ...]:
    """Native ``boozer_surface_residual``: ``(r,)``, ``(r, J)`` or ``(r, J, H)``.

    ``r`` is ``(3 npoints,)``, ``J = dr/dx`` ``(3 npoints, nx)`` and ``H`` the
    second derivative of every residual, ``(3 npoints, nx, nx)``; ``nx`` counts
    ``G`` only if ``optimize_G``. Derivatives are those of
    :func:`residual_derivatives` with the position and tangents linear in the
    coefficients through the ``*_by_dcoeff`` arrays.

    Args:
        G (float | jax.Array): Scalar shape () covariant Boozer field constant in tesla
            meters.
        iota (float | jax.Array): Scalar shape () dimensionless rotational transform.
        points (BoozerPoints): Flattened field/tangents and the derivatives needed for
            the requested order.
        derivatives (int): Derivative order, 0 for values, 1 to add first derivatives,
            or 2 to add second derivatives.
        optimize_G (bool): Include the G column in derivative outputs; the supplied G
            value is always used.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        tuple[jax.Array, ...]: Residual shape (3 * npoints,), optionally Jacobian (3 *
            npoints, nx) and Hessians (3 * npoints, nx, nx), in [surface DOFs, iota,
            optional G] order. Unweighted residual units are tesla squared meters;
            weighted units are tesla meters.
    """
    if derivatives == 0:
        return (
            boozer_point_residual(
                G, iota, points.B, points.xphi, points.xtheta, weight_inv_modB=weight_inv_modB
            ),
        )
    residual, x = _residual_of_x(
        G,
        iota,
        points,
        derivatives=derivatives,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )
    return residual_derivatives(residual, x, derivatives)


def boozer_least_squares(
    G, iota, points: BoozerPoints, *, derivatives: int, optimize_G: bool, weight_inv_modB: bool
) -> tuple[jax.Array, ...]:
    """``0.5 |r|^2`` over all residuals and, up to ``derivatives``, its gradient
    and Hessian in ``x``: native ``boozer_residual``, ``boozer_residual_ds`` or
    ``boozer_residual_ds2`` as a tuple (with the ``G`` entries only if
    ``optimize_G``), not divided by the number of residuals. Derivatives are
    those of :func:`objective_derivatives`, with the position and tangents as
    in :func:`boozer_residual`.

    Args:
        G (float | jax.Array): Scalar shape () covariant Boozer field constant in tesla
            meters.
        iota (float | jax.Array): Scalar shape () dimensionless rotational transform.
        points (BoozerPoints): Flattened field/tangents and the derivatives needed for
            the requested order.
        derivatives (int): Derivative order, 0 for values, 1 to add first derivatives,
            or 2 to add second derivatives.
        optimize_G (bool): Include the G column in derivative outputs; the supplied G
            value is always used.
        weight_inv_modB (bool): Divide each point's Boozer residual by the field
            magnitude in teslas.

    Returns:
        tuple[jax.Array, ...]: Half the squared residual norm as scalar shape (),
            optionally gradient (nx,) and Hessian (nx, nx), in [surface DOFs, iota,
            optional G] order. No normalization by the number of residuals is applied;
            units are squared residual units.
    """
    if derivatives == 0:
        (residual,) = boozer_residual(
            G, iota, points, derivatives=0, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB
        )
        return (0.5 * jnp.sum(residual * residual),)
    residual, x = _residual_of_x(
        G,
        iota,
        points,
        derivatives=derivatives,
        optimize_G=optimize_G,
        weight_inv_modB=weight_inv_modB,
    )

    def half_square(point):
        value = residual(point)
        return 0.5 * jnp.sum(value * value)

    return objective_derivatives(half_square, x, derivatives)
