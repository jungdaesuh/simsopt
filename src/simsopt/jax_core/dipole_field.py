"""Pure JAX port of ``src/simsoptpp/dipole_field.cpp``.

The functions in this module mirror the raw C++ kernels exposed through
``simsoptpp``. Inputs are Cartesian arrays unless ``dipole_field_Bn`` is given
``coordinate_flag="cylindrical"`` or ``"toroidal"``, matching the permanent
magnet optimization matrix convention in the C++ oracle.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ._math_utils import (
    as_jax_float64 as _as_jax_float64,
    explicit_rsqrt as _explicit_rsqrt,
)

__all__ = [
    "DipoleFieldSpec",
    "define_a_uniform_cartesian_grid_between_two_toroidal_surfaces",
    "dipole_field_A",
    "dipole_field_A_from_spec",
    "dipole_field_B",
    "dipole_field_B_from_spec",
    "dipole_field_Bn",
    "dipole_field_dA",
    "dipole_field_dA_from_spec",
    "dipole_field_dB",
    "dipole_field_dB_from_spec",
    "make_dipole_field_spec",
]

_MU0_OVER_4PI = np.float64(1e-7)
_GRID_RAY_COUNT = 2000
_GRID_RAY_LENGTH = np.float64(4.0)


@dataclass(frozen=True)
class DipoleFieldSpec:
    """Immutable raw dipole payload for pure JAX field evaluation.

    ``dipole_points`` and ``dipole_moments`` are both shaped ``(M, 3)``. The
    arrays must already include any stellarator or field-period symmetries,
    exactly as ``DipoleField.dipole_grid`` / ``DipoleField.m_vec`` do before
    calling the C++ kernels.
    """

    dipole_points: jax.Array
    dipole_moments: jax.Array


jax.tree_util.register_dataclass(
    DipoleFieldSpec,
    data_fields=["dipole_points", "dipole_moments"],
    meta_fields=[],
)


def _scale(reference: jax.Array) -> jax.Array:
    return jnp.asarray(_MU0_OVER_4PI, dtype=reference.dtype)


def _scalar(reference: jax.Array, value: float) -> jax.Array:
    return jnp.asarray(np.float64(value), dtype=reference.dtype)


def _require_xyz_matrix(name: str, value: object) -> jax.Array:
    array = _as_jax_float64(value)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3); got {array.shape!r}.")
    return array


def _require_matching_dipoles(
    dipole_points: object, dipole_moments: object
) -> tuple[jax.Array, jax.Array]:
    points = _require_xyz_matrix("dipole_points", dipole_points)
    moments = _require_xyz_matrix("dipole_moments", dipole_moments)
    if points.shape != moments.shape:
        raise ValueError(
            "dipole_points and dipole_moments must have matching shape; "
            f"got {points.shape!r} and {moments.shape!r}."
        )
    return points, moments


def make_dipole_field_spec(
    dipole_points: object, dipole_moments: object
) -> DipoleFieldSpec:
    """Build an immutable raw dipole spec from shaped Cartesian arrays."""

    points, moments = _require_matching_dipoles(dipole_points, dipole_moments)
    return DipoleFieldSpec(dipole_points=points, dipole_moments=moments)


def _geometry(points: jax.Array, dipole_points: jax.Array):
    r = points[:, None, :] - dipole_points[None, :, :]
    r2 = jnp.sum(r * r, axis=-1)
    rinv = _explicit_rsqrt(r2)
    rinv2 = rinv * rinv
    rinv3 = rinv * rinv2
    rinv5 = rinv3 * rinv2
    return r, rinv2, rinv3, rinv5


@jax.jit
def _dipole_field_B_jit(
    points: jax.Array, dipole_points: jax.Array, dipole_moments: jax.Array
) -> jax.Array:
    r, _, rinv3, rinv5 = _geometry(points, dipole_points)
    rdotm = jnp.sum(r * dipole_moments[None, :, :], axis=-1)
    three = _scalar(points, 3.0)
    contribution = (
        three * rdotm[:, :, None] * r * rinv5[:, :, None]
        - dipole_moments[None, :, :] * rinv3[:, :, None]
    )
    return _scale(points) * jnp.sum(contribution, axis=1)


def dipole_field_B(
    points: object, dipole_points: object, dipole_moments: object
) -> jax.Array:
    """Magnetic field from Cartesian point dipoles.

    Mirrors ``simsoptpp.dipole_field_B``:
    ``mu0 / (4*pi) * sum(3 * dot(m, r) * r / |r|^5 - m / |r|^3)``.
    """

    points_arr = _require_xyz_matrix("points", points)
    dipole_points_arr, moments = _require_matching_dipoles(
        dipole_points, dipole_moments
    )
    return _dipole_field_B_jit(points_arr, dipole_points_arr, moments)


def dipole_field_B_from_spec(points: object, spec: DipoleFieldSpec) -> jax.Array:
    return dipole_field_B(points, spec.dipole_points, spec.dipole_moments)


@jax.jit
def _dipole_field_A_jit(
    points: jax.Array, dipole_points: jax.Array, dipole_moments: jax.Array
) -> jax.Array:
    r, _, rinv3, _ = _geometry(points, dipole_points)
    contribution = jnp.cross(dipole_moments[None, :, :], r, axis=-1) * rinv3[:, :, None]
    return _scale(points) * jnp.sum(contribution, axis=1)


def dipole_field_A(
    points: object, dipole_points: object, dipole_moments: object
) -> jax.Array:
    """Vector potential from Cartesian point dipoles.

    Mirrors ``simsoptpp.dipole_field_A``:
    ``mu0 / (4*pi) * sum(cross(m, r) / |r|^3)``.
    """

    points_arr = _require_xyz_matrix("points", points)
    dipole_points_arr, moments = _require_matching_dipoles(
        dipole_points, dipole_moments
    )
    return _dipole_field_A_jit(points_arr, dipole_points_arr, moments)


def dipole_field_A_from_spec(points: object, spec: DipoleFieldSpec) -> jax.Array:
    return dipole_field_A(points, spec.dipole_points, spec.dipole_moments)


@jax.jit
def _dipole_field_dB_jit(
    points: jax.Array, dipole_points: jax.Array, dipole_moments: jax.Array
) -> jax.Array:
    r, rinv2, _, rinv5 = _geometry(points, dipole_points)
    moments_b = dipole_moments[None, :, :]
    rdotm = jnp.sum(r * moments_b, axis=-1)
    delta = jnp.eye(3, dtype=points.dtype)[None, None, :, :]
    rr = r[:, :, :, None] * r[:, :, None, :]
    mj_rk = moments_b[:, :, :, None] * r[:, :, None, :]
    mk_rj = r[:, :, :, None] * moments_b[:, :, None, :]
    three = _scalar(points, 3.0)
    five = _scalar(points, 5.0)
    contribution = (
        three
        * rinv5[:, :, None, None]
        * (
            mj_rk
            + mk_rj
            + rdotm[:, :, None, None] * delta
            - five * rdotm[:, :, None, None] * rr * rinv2[:, :, None, None]
        )
    )
    return _scale(points) * jnp.sum(contribution, axis=1)


def dipole_field_dB(
    points: object, dipole_points: object, dipole_moments: object
) -> jax.Array:
    """Cartesian gradient of ``dipole_field_B``.

    The returned layout is ``dB[p, j, k] = d B_j(x_p) / d x_k``, matching the
    C++ ``dipole_field_dB`` array assignments and ``DipoleField.dB_by_dX()``.
    """

    points_arr = _require_xyz_matrix("points", points)
    dipole_points_arr, moments = _require_matching_dipoles(
        dipole_points, dipole_moments
    )
    return _dipole_field_dB_jit(points_arr, dipole_points_arr, moments)


def dipole_field_dB_from_spec(points: object, spec: DipoleFieldSpec) -> jax.Array:
    return dipole_field_dB(points, spec.dipole_points, spec.dipole_moments)


def _dipole_cross_derivative(moments: jax.Array) -> jax.Array:
    mx = moments[None, :, 0]
    my = moments[None, :, 1]
    mz = moments[None, :, 2]
    zero = jnp.zeros_like(mx)
    row_x = jnp.stack((zero, -mz, my), axis=-1)
    row_y = jnp.stack((mz, zero, -mx), axis=-1)
    row_z = jnp.stack((-my, mx, zero), axis=-1)
    return jnp.stack((row_x, row_y, row_z), axis=-2)


@jax.jit
def _dipole_field_dA_jit(
    points: jax.Array, dipole_points: jax.Array, dipole_moments: jax.Array
) -> jax.Array:
    r, _, rinv3, rinv5 = _geometry(points, dipole_points)
    mcrossr = jnp.cross(dipole_moments[None, :, :], r, axis=-1)
    skew = _dipole_cross_derivative(dipole_moments)
    three = _scalar(points, 3.0)
    contribution = (
        skew * rinv3[:, :, None, None]
        - three * mcrossr[:, :, :, None] * r[:, :, None, :] * rinv5[:, :, None, None]
    )
    return _scale(points) * jnp.sum(contribution, axis=1)


def dipole_field_dA(
    points: object, dipole_points: object, dipole_moments: object
) -> jax.Array:
    """Cartesian gradient of ``dipole_field_A``.

    The returned layout is ``dA[p, j, k] = d A_j(x_p) / d x_k``, matching the
    C++ ``dipole_field_dA`` array assignments.
    """

    points_arr = _require_xyz_matrix("points", points)
    dipole_points_arr, moments = _require_matching_dipoles(
        dipole_points, dipole_moments
    )
    return _dipole_field_dA_jit(points_arr, dipole_points_arr, moments)


def dipole_field_dA_from_spec(points: object, spec: DipoleFieldSpec) -> jax.Array:
    return dipole_field_dA(points, spec.dipole_points, spec.dipole_moments)


def _basis_angles(dipole_points: jax.Array, R0: object):
    x = dipole_points[:, 0]
    y = dipole_points[:, 1]
    z = dipole_points[:, 2]
    phi = jnp.atan2(y, x)
    theta = jnp.atan2(z, jnp.sqrt(x * x + y * y) - jnp.asarray(R0, dtype=x.dtype))
    return jnp.sin(phi), jnp.cos(phi), jnp.sin(theta), jnp.cos(theta)


def _symmetry_location(
    dipole_points: jax.Array, phi0: jax.Array, stell_sign: jax.Array
) -> jax.Array:
    x = dipole_points[:, 0]
    y = dipole_points[:, 1]
    z = dipole_points[:, 2]
    sphi0 = jnp.sin(phi0)
    cphi0 = jnp.cos(phi0)
    return jnp.stack(
        (
            x * cphi0 - y * sphi0 * stell_sign,
            x * sphi0 + y * cphi0 * stell_sign,
            z * stell_sign,
        ),
        axis=-1,
    )


def _normal_field_matrix(
    points: jax.Array, dipole_points: jax.Array, unitnormal: jax.Array
) -> jax.Array:
    r, _, rinv3, rinv5 = _geometry(points, dipole_points)
    rdotn = jnp.sum(r * unitnormal[:, None, :], axis=-1)
    three = _scalar(points, 3.0)
    return (
        three * rdotn[:, :, None] * r * rinv5[:, :, None]
        - unitnormal[:, None, :] * rinv3[:, :, None]
    )


def _rotate_normal_matrix_to_cartesian_basis(
    normal_matrix: jax.Array,
    phi0: jax.Array,
    stell_sign: jax.Array,
) -> jax.Array:
    sphi0 = jnp.sin(phi0)
    cphi0 = jnp.cos(phi0)
    return jnp.stack(
        (
            (normal_matrix[:, :, 0] * cphi0 + normal_matrix[:, :, 1] * sphi0)
            * stell_sign,
            -normal_matrix[:, :, 0] * sphi0 + normal_matrix[:, :, 1] * cphi0,
            normal_matrix[:, :, 2],
        ),
        axis=-1,
    )


def _rotate_normal_matrix_to_cylindrical_basis(
    normal_matrix: jax.Array,
    phi0: jax.Array,
    stell_sign: jax.Array,
    sphi: jax.Array,
    cphi: jax.Array,
) -> jax.Array:
    cartesian = _rotate_normal_matrix_to_cartesian_basis(
        normal_matrix, phi0, stell_sign
    )
    ax_temp = cartesian[:, :, 0]
    ay_temp = cartesian[:, :, 1]
    return jnp.stack(
        (
            ax_temp * cphi[None, :] + ay_temp * sphi[None, :],
            -ax_temp * sphi[None, :] + ay_temp * cphi[None, :],
            cartesian[:, :, 2],
        ),
        axis=-1,
    )


def _rotate_normal_matrix_to_toroidal_basis(
    normal_matrix: jax.Array,
    phi0: jax.Array,
    stell_sign: jax.Array,
    sphi: jax.Array,
    cphi: jax.Array,
    stheta: jax.Array,
    ctheta: jax.Array,
) -> jax.Array:
    cartesian = _rotate_normal_matrix_to_cartesian_basis(
        normal_matrix, phi0, stell_sign
    )
    ax_temp = cartesian[:, :, 0]
    ay_temp = cartesian[:, :, 1]
    az_temp = cartesian[:, :, 2]
    return jnp.stack(
        (
            ax_temp * cphi[None, :] * ctheta[None, :]
            + ay_temp * sphi[None, :] * ctheta[None, :]
            + az_temp * stheta[None, :],
            -ax_temp * sphi[None, :] + ay_temp * cphi[None, :],
            -ax_temp * cphi[None, :] * stheta[None, :]
            - ay_temp * sphi[None, :] * stheta[None, :]
            + az_temp * ctheta[None, :],
        ),
        axis=-1,
    )


def dipole_field_Bn(
    points: object,
    dipole_points: object,
    unitnormal: object,
    nfp: int,
    stellsym: int | bool,
    b: object,
    coordinate_flag: str = "cartesian",
    R0: float = 0.0,
) -> jax.Array:
    """Permanent-magnet normal-field matrix from the C++ oracle.

    ``b`` is accepted for signature parity with ``simsoptpp.dipole_field_Bn``;
    the C++ implementation checks its storage order but does not use its
    contents in the returned matrix.
    """

    points_arr = _require_xyz_matrix("points", points)
    dipole_points_arr = _require_xyz_matrix("dipole_points", dipole_points)
    unitnormal_arr = _require_xyz_matrix("unitnormal", unitnormal)
    _as_jax_float64(b)
    if unitnormal_arr.shape != points_arr.shape:
        raise ValueError(
            "unitnormal must have the same shape as points; "
            f"got {unitnormal_arr.shape!r} and {points_arr.shape!r}."
        )

    nfp_int = int(nfp)
    stellsym_int = int(stellsym)
    sphi, cphi, stheta, ctheta = _basis_angles(dipole_points_arr, R0)
    acc = jnp.zeros(
        (points_arr.shape[0], dipole_points_arr.shape[0], 3),
        dtype=points_arr.dtype,
    )
    for stell in range(stellsym_int + 1):
        stell_sign = jnp.asarray((-1.0) ** stell, dtype=points_arr.dtype)
        for fp in range(nfp_int):
            phi0 = jnp.asarray(2.0 * np.pi * fp / nfp_int, dtype=points_arr.dtype)
            sym_points = _symmetry_location(dipole_points_arr, phi0, stell_sign)
            normal_matrix = _normal_field_matrix(points_arr, sym_points, unitnormal_arr)
            if coordinate_flag == "cylindrical":
                contribution = _rotate_normal_matrix_to_cylindrical_basis(
                    normal_matrix, phi0, stell_sign, sphi, cphi
                )
            elif coordinate_flag == "toroidal":
                contribution = _rotate_normal_matrix_to_toroidal_basis(
                    normal_matrix, phi0, stell_sign, sphi, cphi, stheta, ctheta
                )
            else:
                contribution = _rotate_normal_matrix_to_cartesian_basis(
                    normal_matrix, phi0, stell_sign
                )
            acc = acc + contribution
    return _scale(points_arr) * acc


def _nearest_index_and_distance(surface_points: jax.Array, point: jax.Array):
    distances = jnp.sum((surface_points - point[None, :]) ** 2, axis=-1)
    index = jnp.argmin(distances)
    return index, distances[index]


def _filter_uniform_grid_point(
    point: jax.Array,
    normal_inner: jax.Array,
    normal_outer: jax.Array,
    xyz_inner: jax.Array,
    xyz_outer: jax.Array,
) -> jax.Array:
    inner_loc, min_dist_inner = _nearest_index_and_distance(xyz_inner, point)
    outer_loc, min_dist_outer = _nearest_index_and_distance(xyz_outer, point)
    normal = jnp.where(
        min_dist_inner < min_dist_outer,
        normal_inner[inner_loc],
        normal_outer[outer_loc],
    )
    ray = normal * _explicit_rsqrt(jnp.sum(normal * normal))
    ray_steps = jnp.arange(_GRID_RAY_COUNT, dtype=point.dtype)
    ray_points = (
        point[None, :]
        + ray[None, :]
        * (jnp.asarray(_GRID_RAY_LENGTH, dtype=point.dtype) / _GRID_RAY_COUNT)
        * ray_steps[:, None]
    )
    inner_anchor = xyz_inner[inner_loc]
    outer_anchor = xyz_outer[outer_loc]
    dist_inner_ray = jnp.sum((inner_anchor[None, :] - ray_points) ** 2, axis=-1)
    dist_outer_ray = jnp.sum((outer_anchor[None, :] - ray_points) ** 2, axis=-1)
    nearest_loc_inner = jnp.argmin(dist_inner_ray)
    nearest_loc_outer = jnp.argmin(dist_outer_ray)
    keep = jnp.logical_and(nearest_loc_inner <= 0, nearest_loc_outer > 0)
    return jnp.where(keep, point, jnp.zeros_like(point))


def define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
    normal_inner: object,
    normal_outer: object,
    xyz_uniform: object,
    xyz_inner: object,
    xyz_outer: object,
) -> jax.Array:
    """Filter a Cartesian candidate grid between two toroidal surfaces.

    This mirrors the C++ helper of the same name, including the hard-coded
    2000-sample ray test and zero rows for rejected candidate points.
    """

    normal_inner_arr = _require_xyz_matrix("normal_inner", normal_inner)
    normal_outer_arr = _require_xyz_matrix("normal_outer", normal_outer)
    xyz_uniform_arr = _require_xyz_matrix("xyz_uniform", xyz_uniform)
    xyz_inner_arr = _require_xyz_matrix("xyz_inner", xyz_inner)
    xyz_outer_arr = _require_xyz_matrix("xyz_outer", xyz_outer)
    return jax.vmap(
        _filter_uniform_grid_point,
        in_axes=(0, None, None, None, None),
    )(xyz_uniform_arr, normal_inner_arr, normal_outer_arr, xyz_inner_arr, xyz_outer_arr)
