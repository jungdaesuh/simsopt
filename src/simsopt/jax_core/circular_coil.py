"""JAX port of :class:`simsopt.field.CircularCoil`.

This module is the single source of truth for the JAX kernels that
reproduce the upstream cartesian closed-form ``B`` and ``dB/dX`` of
``CircularCoil``. The kernels are exposed through public symbols:

* :class:`CircularCoilSpec` -- frozen, hashable, JAX-tree-registered
  payload that mirrors the CPU constructor.
* :func:`circular_coil_B` -- ``(N, 3) -> (N, 3)`` evaluation of the
  field at cartesian points.
* :func:`circular_coil_A` -- ``(N, 3) -> (N, 3)`` evaluation of the
  vector potential at cartesian points.
* :func:`circular_coil_dB` -- ``(N, 3) -> (N, 3, 3)`` evaluation of the
  first derivative, in the same ``dB[p, j, l] = ∂_j B_l`` axis layout
  the CPU oracle uses.

Two internal helpers, :func:`_rotation_matrix` and
:func:`_rotation_matrix_inv`, build the world<->coil-axis rotation from
the spec exactly as :meth:`simsopt.field.CircularCoil._rotmat` does.

Why a JAX-native elliptic helper
--------------------------------

``CircularCoil`` evaluates complete elliptic integrals ``K(m)`` and
``E(m)``. ``jax.scipy.special.ellipk`` / ``ellipe`` are not exposed by
``jaxlib`` 0.10.0, so we depend on the local Carlson-symmetric
implementation in :mod:`simsopt.jax_core._elliptic`. That helper is
fully traceable under ``jit`` / ``vmap`` / ``grad`` with no host
callback, so :func:`circular_coil_B` and :func:`circular_coil_dB`
remain pure JAX under ``transfer_guard("disallow")``.

Singular-regime contract
------------------------

The CPU oracle protects its divisions with an additive ``1e-31``
regularization on the ``rho`` denominators. The JAX kernels copy that
floor verbatim so the ``direct_kernel`` parity lane is bit-tight even
near the coil axis. No defensive ``where`` masks are added beyond the
upstream guards.

Convention
----------

* ``B`` and ``dB`` axes follow the upstream public layout
  (``CLAUDE.md`` ``Tensor convention`` section).
* The coil-axis frame has the coil in the ``z = 0`` plane with the
  current flowing CCW as viewed from ``+z``.
* Local-frame ``B`` is rotated back to world coordinates with
  :func:`_rotation_matrix` and ``dB`` is conjugated by the rotation.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._elliptic import ellipe, ellipk
from ._math_utils import as_jax_float64 as _as_jax_float64


__all__ = [
    "CircularCoilSpec",
    "circular_coil_A",
    "circular_coil_B",
    "circular_coil_dB",
]


# ── Spec ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CircularCoilSpec:
    """Immutable payload for a pure JAX ``CircularCoil`` evaluation.

    Parameters
    ----------
    r0
        Coil radius in metres.
    center
        Cartesian coil centre in metres, three floats.
    Inorm
        Upstream-normalised current ``I * 4e-7``.
    normal
        Either ``(theta, phi)`` spherical angles or a three-component
        ``(nx, ny, nz)`` direction vector payload, matching the CPU
        ``CircularCoil`` constructor.
    normal_kind
        ``"spherical"`` when ``normal`` is ``(theta, phi)`` and
        ``"cartesian"`` when ``normal`` is ``(nx, ny, nz)``. Derived
        automatically from ``len(normal)`` if left at the default
        ``"auto"`` value.
    """

    r0: float
    center: tuple[float, float, float]
    Inorm: float
    normal: tuple[float, ...]
    normal_kind: str = "auto"


jax.tree_util.register_dataclass(
    CircularCoilSpec,
    data_fields=[],
    meta_fields=["r0", "center", "Inorm", "normal", "normal_kind"],
)


def _resolve_normal_kind(spec: CircularCoilSpec) -> str:
    if spec.normal_kind == "auto":
        if len(spec.normal) == 2:
            return "spherical"
        if len(spec.normal) == 3:
            return "cartesian"
        raise ValueError("CircularCoil normal must have length 2 or 3.")
    if spec.normal_kind not in ("spherical", "cartesian"):
        raise ValueError(
            f"normal_kind must be 'spherical' or 'cartesian'; got {spec.normal_kind!r}."
        )
    return spec.normal_kind


def _spherical_angles(spec: CircularCoilSpec) -> tuple[float, float]:
    kind = _resolve_normal_kind(spec)
    if kind == "spherical":
        if len(spec.normal) != 2:
            raise ValueError(
                "Spherical normal payload must have exactly two entries "
                f"(theta, phi); got {len(spec.normal)}."
            )
        return float(spec.normal[0]), float(spec.normal[1])
    if len(spec.normal) != 3:
        raise ValueError(
            "Cartesian normal payload must have exactly three entries "
            f"(nx, ny, nz); got {len(spec.normal)}."
        )
    nx, ny, nz = (float(value) for value in spec.normal)
    theta = float(np.arctan2(ny, nx))
    phi = float(np.arctan2(np.sqrt(nx * nx + ny * ny), nz))
    return theta, phi


# ── Rotation helpers ─────────────────────────────────────────────────


def _rotation_matrix_from_angles(
    theta: jax.Array | float, phi: jax.Array | float
) -> jax.Array:
    """Build the coil-axis -> world rotation matrix from ``(theta, phi)``.

    Mirrors :meth:`simsopt.field.CircularCoil._rotmat` literally so the
    parity gate stays bit-tight against the CPU oracle.
    """

    theta_arr = jnp.asarray(theta)
    phi_arr = jnp.asarray(phi)
    cos_theta = jnp.cos(theta_arr)
    sin_theta = jnp.sin(theta_arr)
    cos_phi = jnp.cos(phi_arr)
    sin_phi = jnp.sin(phi_arr)
    sin_phi_half_sq = jnp.sin(phi_arr / 2.0) ** 2
    return jnp.stack(
        (
            jnp.stack(
                (
                    cos_phi * cos_theta**2 + sin_theta**2,
                    -sin_phi_half_sq * jnp.sin(2.0 * theta_arr),
                    cos_theta * sin_phi,
                )
            ),
            jnp.stack(
                (
                    -sin_phi_half_sq * jnp.sin(2.0 * theta_arr),
                    cos_theta**2 + cos_phi * sin_theta**2,
                    sin_phi * sin_theta,
                )
            ),
            jnp.stack((-cos_theta * sin_phi, -sin_phi * sin_theta, cos_phi)),
        ),
        axis=0,
    )


def _rotation_matrix(spec: CircularCoilSpec) -> jax.Array:
    """Return the coil-axis -> world rotation matrix for ``spec``."""

    theta, phi = _spherical_angles(spec)
    return _rotation_matrix_from_angles(theta, phi)


def _rotation_matrix_inv(spec: CircularCoilSpec) -> jax.Array:
    """Return the world -> coil-axis rotation matrix for ``spec``.

    The rotation is orthogonal so the inverse is the transpose; this
    helper makes the intent explicit and matches the CPU
    ``_rotmatinv`` semantics.
    """

    return _rotation_matrix(spec).T


# ── Local-frame kernels ──────────────────────────────────────────────


def _local_geometry(
    point: jax.Array, r0: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    x = point[0]
    y = point[1]
    z = point[2]
    rho = jnp.sqrt(x * x + y * y)
    r = jnp.sqrt(x * x + y * y + z * z)
    r0_sq = r0 * r0
    r_sq = r * r
    alpha = jnp.sqrt(r0_sq + r_sq - 2.0 * r0 * rho)
    beta = jnp.sqrt(r0_sq + r_sq + 2.0 * r0 * rho)
    m = 1.0 - (alpha * alpha) / (beta * beta)
    ellipe_m = ellipe(m)
    ellipk_m = ellipk(m)
    return rho, r, alpha, beta, ellipe_m, ellipk_m


def _B_local_pointwise(point: jax.Array, r0: jax.Array, Inorm: jax.Array) -> jax.Array:
    """B at one point in the coil-axis frame, mirroring ``CircularCoil._B_impl``."""

    x = point[0]
    y = point[1]
    z = point[2]
    rho, _, alpha, beta, ellipe_m, ellipk_m = _local_geometry(point, r0)
    r0_sq = r0 * r0
    r_sq = x * x + y * y + z * z
    alpha_sq = alpha * alpha
    radial_term = (r0_sq + r_sq) * ellipe_m - alpha_sq * ellipk_m
    denom_xy = 2.0 * alpha_sq * beta * rho * rho + 1.0e-31
    denom_z = 2.0 * alpha_sq * beta + 1.0e-31
    bx = Inorm * x * z / denom_xy * radial_term
    by = Inorm * y * z / denom_xy * radial_term
    bz = Inorm / denom_z * ((r0_sq - r_sq) * ellipe_m + alpha_sq * ellipk_m)
    return jnp.stack((bx, by, bz))


def _A_local_pointwise(point: jax.Array, r0: jax.Array, Inorm: jax.Array) -> jax.Array:
    """A at one point in the coil-axis frame, mirroring ``CircularCoil._A_impl``."""

    x = point[0]
    y = point[1]
    z = point[2]
    rho = jnp.sqrt(x * x + y * y)
    r_sq = x * x + y * y + z * z
    r0_sq = r0 * r0
    alpha = jnp.sqrt(r0_sq + r_sq - 2.0 * r0 * rho)
    beta_arg = r0_sq + x * x + y * y + 2.0 * r0 * rho + z * z + 1.0e-31
    beta_guarded = jnp.sqrt(beta_arg)
    k_sq = 1.0 - (alpha * alpha) / (beta_guarded * beta_guarded)
    ellipe_k_sq = ellipe(k_sq)
    ellipk_k_sq = ellipk(k_sq)
    num = 2.0 * r0 + rho * ellipe_k_sq + (r0_sq + r_sq) * (ellipe_k_sq - ellipk_k_sq)
    denom = (x * x + y * y + 1.0e-31) * beta_guarded
    factor = num / denom
    zero = jnp.zeros_like(x)
    local_a = factor * jnp.stack((-y, x, zero))
    return -0.5 * Inorm * local_a


def _dB_local_pointwise(point: jax.Array, r0: jax.Array, Inorm: jax.Array) -> jax.Array:
    """``dB[j, l]`` at one point in the coil-axis frame.

    The formulas are a direct JAX transcription of
    :meth:`simsopt.field.CircularCoil._dB_by_dX_impl` before its
    rotation back to global coordinates. Axis 0 is the derivative
    direction and axis 1 is the ``B`` component, matching the CPU
    public layout.
    """

    x = point[0]
    y = point[1]
    z = point[2]
    rho, r, alpha, beta, ellipe_m, ellipk_m = _local_geometry(point, r0)
    r0_sq = r0 * r0
    r0_4 = r0_sq * r0_sq
    r0_6 = r0_4 * r0_sq
    r_sq = r * r
    r_4 = r_sq * r_sq
    x_sq = x * x
    y_sq = y * y
    z_sq = z * z
    z_4 = z_sq * z_sq
    rho_sq = rho * rho
    rho_4 = rho_sq * rho_sq
    alpha_sq = alpha * alpha
    alpha_4 = alpha_sq * alpha_sq
    beta_3 = beta * beta * beta
    gamma = x_sq - y_sq
    x_4 = x_sq * x_sq
    y_4 = y_sq * y_sq

    dBxdx = (
        Inorm
        * z
        * (
            ellipk_m
            * alpha_sq
            * (
                (2.0 * x_4 + gamma * (y_sq + z_sq)) * r_sq
                + r0_sq
                * (gamma * (r0_sq + 2.0 * z_sq) - (3.0 * x_sq - 2.0 * y_sq) * rho_sq)
            )
            + ellipe_m
            * (
                -((2.0 * x_4 + gamma * (y_sq + z_sq)) * r_4)
                + r0_4
                * (-(gamma * (r0_sq + 3.0 * z_sq)) + (8.0 * x_sq - y_sq) * rho_sq)
                - r0_sq
                * (
                    3.0 * gamma * z_4
                    - 2.0 * (2.0 * x_sq + y_sq) * z_sq * rho_sq
                    + (5.0 * x_sq + y_sq) * rho_4
                )
            )
        )
        / (2.0 * alpha_4 * beta_3 * rho_4 + 1.0e-31)
    )

    dBydx = (
        Inorm
        * x
        * y
        * z
        * (
            ellipk_m
            * alpha_sq
            * (
                2.0 * r0_4
                + r_sq * (2.0 * r_sq + rho_sq)
                - r0_sq * (-4.0 * z_sq + 5.0 * rho_sq)
            )
            + ellipe_m
            * (
                -2.0 * r0_6
                - r_4 * (2.0 * r_sq + rho_sq)
                + 3.0 * r0_4 * (-2.0 * z_sq + 3.0 * rho_sq)
                - 2.0 * r0_sq * (3.0 * z_4 - z_sq * rho_sq + 2.0 * rho_4)
            )
        )
        / (2.0 * alpha_4 * beta_3 * rho_4 + 1.0e-31)
    )

    dBzdx = (
        Inorm
        * x
        * (
            -(ellipk_m * alpha_sq * ((-r0_sq + rho_sq) ** 2 + z_sq * (r0_sq + rho_sq)))
            + ellipe_m
            * (
                z_4 * (r0_sq + rho_sq)
                + (-r0_sq + rho_sq) ** 2 * (r0_sq + rho_sq)
                + 2.0 * z_sq * (r0_4 - 6.0 * r0_sq * rho_sq + rho_4)
            )
        )
        / (2.0 * alpha_4 * beta_3 * rho_sq + 1.0e-31)
    )
    dBxdy = dBydx

    dBydy = (
        Inorm
        * z
        * (
            ellipk_m
            * alpha_sq
            * (
                (2.0 * y_4 - gamma * (x_sq + z_sq)) * r_sq
                + r0_sq
                * (
                    -(gamma * (r0_sq + 2.0 * z_sq))
                    - (-2.0 * x_sq + 3.0 * y_sq) * rho_sq
                )
            )
            + ellipe_m
            * (
                -((2.0 * y_4 - gamma * (x_sq + z_sq)) * r_4)
                + r0_4 * (gamma * (r0_sq + 3.0 * z_sq) + (-x_sq + 8.0 * y_sq) * rho_sq)
                - r0_sq
                * (
                    -3.0 * gamma * z_4
                    - 2.0 * (x_sq + 2.0 * y_sq) * z_sq * rho_sq
                    + (x_sq + 5.0 * y_sq) * rho_4
                )
            )
        )
        / (2.0 * alpha_4 * beta_3 * rho_4 + 1.0e-31)
    )

    dBzdy = dBzdx * y / (x + 1.0e-31)
    dBxdz = dBzdx
    dBydz = dBzdy
    dBzdz = (
        Inorm
        * z
        * (
            ellipk_m * alpha_sq * (r0_sq - r_sq)
            + ellipe_m * (-7.0 * r0_4 + r_4 + 6.0 * r0_sq * (-z_sq + rho_sq))
        )
        / (2.0 * alpha_4 * beta_3 + 1.0e-31)
    )

    return jnp.stack(
        (
            jnp.stack((dBxdx, dBydx, dBzdx)),
            jnp.stack((dBxdy, dBydy, dBzdy)),
            jnp.stack((dBxdz, dBydz, dBzdz)),
        ),
        axis=0,
    )


# ── World-frame point-wise kernels ───────────────────────────────────


def _B_pointwise(
    point: jax.Array,
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
) -> jax.Array:
    rot = _rotation_matrix_from_angles(theta, phi)
    local_point = rot.T @ (point - center)
    return rot @ _B_local_pointwise(local_point, r0, Inorm)


def _A_pointwise(
    point: jax.Array,
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
) -> jax.Array:
    rot = _rotation_matrix_from_angles(theta, phi)
    local_point = rot.T @ (point - center)
    return rot @ _A_local_pointwise(local_point, r0, Inorm)


def _dB_pointwise(
    point: jax.Array,
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
) -> jax.Array:
    rot = _rotation_matrix_from_angles(theta, phi)
    local_point = rot.T @ (point - center)
    local_dB = _dB_local_pointwise(local_point, r0, Inorm)
    return rot @ local_dB @ rot.T


_B_vmap = jax.vmap(_B_pointwise, in_axes=(0, None, None, None, None, None))
_A_vmap = jax.vmap(_A_pointwise, in_axes=(0, None, None, None, None, None))
_dB_vmap = jax.vmap(_dB_pointwise, in_axes=(0, None, None, None, None, None))


@jax.jit
def _B_jit(
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    points: jax.Array,
) -> jax.Array:
    return _B_vmap(points, r0, center, Inorm, theta, phi)


@jax.jit
def _A_jit(
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    points: jax.Array,
) -> jax.Array:
    return _A_vmap(points, r0, center, Inorm, theta, phi)


@jax.jit
def _dB_jit(
    r0: jax.Array,
    center: jax.Array,
    Inorm: jax.Array,
    theta: jax.Array,
    phi: jax.Array,
    points: jax.Array,
) -> jax.Array:
    return _dB_vmap(points, r0, center, Inorm, theta, phi)


# ── Public API ───────────────────────────────────────────────────────


def _validate_points(points: jax.Array) -> jax.Array:
    points_arr = _as_jax_float64(points)
    if points_arr.ndim != 2 or points_arr.shape[1] != 3:
        raise ValueError(
            f"points must have shape (N, 3); got {tuple(points_arr.shape)!r}."
        )
    return points_arr


def _scalars(
    spec: CircularCoilSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    theta, phi = _spherical_angles(spec)
    return (
        _as_jax_float64(spec.r0),
        _as_jax_float64(spec.center),
        _as_jax_float64(spec.Inorm),
        _as_jax_float64(theta),
        _as_jax_float64(phi),
    )


def circular_coil_B(spec: CircularCoilSpec, points: jax.Array) -> jax.Array:
    """``B(x)`` at cartesian ``points`` for a circular coil described by ``spec``.

    The output has shape ``(N, 3)`` with ``B[p, l] = B_l(x_p)`` in the
    world frame, matching :meth:`simsopt.field.CircularCoil._B_impl`.
    """

    r0, center, Inorm, theta, phi = _scalars(spec)
    return _B_jit(r0, center, Inorm, theta, phi, _validate_points(points))


def circular_coil_A(spec: CircularCoilSpec, points: jax.Array) -> jax.Array:
    """``A(x)`` at cartesian ``points`` for a circular coil described by ``spec``.

    The output has shape ``(N, 3)`` with ``A[p, l] = A_l(x_p)`` in the
    world frame, matching :meth:`simsopt.field.CircularCoil._A_impl`.
    """

    r0, center, Inorm, theta, phi = _scalars(spec)
    return _A_jit(r0, center, Inorm, theta, phi, _validate_points(points))


def circular_coil_dB(spec: CircularCoilSpec, points: jax.Array) -> jax.Array:
    """``dB[p, j, l] = ∂_j B_l(x_p)`` for a circular coil.

    The output has shape ``(N, 3, 3)`` with axis 1 the derivative
    direction and axis 2 the ``B`` component, matching the upstream
    public layout in :meth:`simsopt.field.CircularCoil._dB_by_dX_impl`.
    """

    r0, center, Inorm, theta, phi = _scalars(spec)
    return _dB_jit(r0, center, Inorm, theta, phi, _validate_points(points))
