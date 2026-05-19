"""Pure JAX SurfaceRZFourier geometry built on immutable specs.

The RZ derivative kernels use XLA reductions over Fourier modes and can
therefore differ across machines at the reduction-order floor. Parity checks
for this path should use the direct-kernel tolerance lane around 1e-12 rather
than byte identity.

The RZ Hessian helpers carry an explicit memory-work guard for extreme
``mpol=ntor=20`` production-shape requests where reverse-over-forward
autodiff would materialize an impractically large quadrature-by-dof tape.
"""

from __future__ import annotations

from functools import partial
from math import comb

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from ._math_utils import as_jax_float64 as _as_jax_float64
from ._math_utils import runtime_device_put
from ._device_scalars import device_one, float_scalar, two_pi
from ._vector_norms import norm3 as _norm3
from ._vector_norms import unit_vector3 as _unit_vector3
from .specs import (
    SurfaceRZFourierSpec,
    _surface_rz_fourier_block_mode_positions as _block_mode_positions,
    make_surface_rzfourier_spec,
    surface_rz_fourier_dofs_from_spec as _surface_rz_fourier_dofs_from_spec,
)

_HESSIAN_WORK_BYTES_LIMIT = 32 * 1024**3
_SCATTER_SET_DIMS_1D = lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,),
    operand_batching_dims=(),
    scatter_indices_batching_dims=(),
)


def _zero_based_mode_range(count: int, reference: jax.Array) -> jax.Array:
    ones = jnp.broadcast_to(device_one(reference), (count,))
    return jnp.cumsum(ones) - ones


def _poloidal_modes(spec: SurfaceRZFourierSpec) -> jax.Array:
    return _zero_based_mode_range(spec.mpol + 1, spec.quadpoints_theta)


def _toroidal_modes(spec: SurfaceRZFourierSpec) -> jax.Array:
    zero_based = _zero_based_mode_range(2 * spec.ntor + 1, spec.quadpoints_phi)
    return zero_based - float_scalar(spec.ntor, zero_based)


def _mode_terms(
    spec: SurfaceRZFourierSpec,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    angle_scale = two_pi(spec.quadpoints_theta)
    theta = angle_scale * spec.quadpoints_theta
    phi = angle_scale * spec.quadpoints_phi
    m = _poloidal_modes(spec)
    n = _toroidal_modes(spec)
    nfp = float_scalar(spec.nfp, n)
    angles = (
        m[None, None, :, None] * theta[None, :, None, None]
        - nfp * n[None, None, None, :] * phi[:, None, None, None]
    )
    return phi, m, n, jnp.cos(angles), jnp.sin(angles)


def _mode_angles(spec: SurfaceRZFourierSpec) -> tuple[jax.Array, jax.Array, jax.Array]:
    phi, _, _, cos_terms, sin_terms = _mode_terms(spec)
    return phi, cos_terms, sin_terms


def _sum_fourier_modes(
    cos_coeffs: jax.Array,
    sin_coeffs: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
) -> jax.Array:
    return jnp.sum(
        cos_coeffs[None, None, :, :] * cos_terms
        + sin_coeffs[None, None, :, :] * sin_terms,
        axis=(2, 3),
    )


def _radius_height_from_modes(
    spec: SurfaceRZFourierSpec,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return (
        _sum_fourier_modes(spec.rc, spec.rs, cos_terms, sin_terms),
        _sum_fourier_modes(spec.zc, spec.zs, cos_terms, sin_terms),
    )


def _differentiated_fourier_modes(
    cos_coeffs: jax.Array,
    sin_coeffs: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
    phi_factor: jax.Array,
    theta_factor: jax.Array,
    phi_order: int,
    theta_order: int,
) -> jax.Array:
    scale = jnp.ones_like(cos_terms)
    if phi_order:
        scale = scale * phi_factor**phi_order
    if theta_order:
        scale = scale * theta_factor**theta_order

    cos_coeffs = cos_coeffs[None, None, :, :]
    sin_coeffs = sin_coeffs[None, None, :, :]
    derivative_order = (phi_order + theta_order) % 4
    if derivative_order == 0:
        values = cos_coeffs * cos_terms + sin_coeffs * sin_terms
    elif derivative_order == 1:
        values = -cos_coeffs * sin_terms + sin_coeffs * cos_terms
    elif derivative_order == 2:
        values = -cos_coeffs * cos_terms - sin_coeffs * sin_terms
    else:
        values = cos_coeffs * sin_terms - sin_coeffs * cos_terms
    return jnp.sum(values * scale, axis=(2, 3))


def _radius_height_derivative_from_modes(
    spec: SurfaceRZFourierSpec,
    m: jax.Array,
    n: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
    angle_scale: jax.Array,
    phi_order: int,
    theta_order: int,
) -> tuple[jax.Array, jax.Array]:
    nfp = float_scalar(spec.nfp, n)
    phi_factor = -angle_scale * nfp * n[None, None, None, :]
    theta_factor = angle_scale * m[None, None, :, None]
    return (
        _differentiated_fourier_modes(
            spec.rc,
            spec.rs,
            cos_terms,
            sin_terms,
            phi_factor,
            theta_factor,
            phi_order,
            theta_order,
        ),
        _differentiated_fourier_modes(
            spec.zc,
            spec.zs,
            cos_terms,
            sin_terms,
            phi_factor,
            theta_factor,
            phi_order,
            theta_order,
        ),
    )


def _phi_frame(phi: jax.Array) -> tuple[jax.Array, jax.Array]:
    return jnp.cos(phi)[:, None], jnp.sin(phi)[:, None]


def _surface_rz_fourier_gamma_from_terms(
    r: jax.Array,
    z: jax.Array,
    cos_phi: jax.Array,
    sin_phi: jax.Array,
) -> jax.Array:
    return jnp.stack([r * cos_phi, r * sin_phi, z], axis=-1)


def _surface_rz_fourier_gammadash1_from_terms(
    spec: SurfaceRZFourierSpec,
    r: jax.Array,
    n: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
    cos_phi: jax.Array,
    sin_phi: jax.Array,
    angle_scale: jax.Array,
) -> jax.Array:
    nfp = float_scalar(spec.nfp, n)
    scale = angle_scale * nfp * n[None, None, None, :]
    d_r = jnp.sum(
        spec.rc[None, None, :, :] * sin_terms * scale
        - spec.rs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    d_z = jnp.sum(
        spec.zc[None, None, :, :] * sin_terms * scale
        - spec.zs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    return jnp.stack(
        [
            d_r * cos_phi - r * (angle_scale * sin_phi),
            d_r * sin_phi + r * (angle_scale * cos_phi),
            d_z,
        ],
        axis=-1,
    )


def _surface_rz_fourier_gammadash2_from_terms(
    spec: SurfaceRZFourierSpec,
    m: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
    cos_phi: jax.Array,
    sin_phi: jax.Array,
    angle_scale: jax.Array,
) -> jax.Array:
    scale = angle_scale * m[None, None, :, None]
    d_r = jnp.sum(
        -spec.rc[None, None, :, :] * sin_terms * scale
        + spec.rs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    d_z = jnp.sum(
        -spec.zc[None, None, :, :] * sin_terms * scale
        + spec.zs[None, None, :, :] * cos_terms * scale,
        axis=(2, 3),
    )
    return jnp.stack([d_r * cos_phi, d_r * sin_phi, d_z], axis=-1)


def _surface_rz_fourier_derivative_from_terms(
    spec: SurfaceRZFourierSpec,
    phi_order: int,
    theta_order: int,
    m: jax.Array,
    n: jax.Array,
    cos_terms: jax.Array,
    sin_terms: jax.Array,
    cos_phi: jax.Array,
    sin_phi: jax.Array,
    angle_scale: jax.Array,
) -> jax.Array:
    radial = jnp.zeros(cos_terms.shape[:2], dtype=cos_terms.dtype)
    toroidal = jnp.zeros_like(radial)
    radial_signs = (1.0, 0.0, -1.0, 0.0)
    toroidal_signs = (0.0, 1.0, 0.0, -1.0)

    for basis_order in range(phi_order + 1):
        radius_derivative, _ = _radius_height_derivative_from_modes(
            spec,
            m,
            n,
            cos_terms,
            sin_terms,
            angle_scale,
            phi_order - basis_order,
            theta_order,
        )
        scale = float_scalar(comb(phi_order, basis_order), cos_terms)
        if basis_order:
            scale = scale * angle_scale**basis_order
        phase = basis_order % 4
        radial = radial + scale * radial_signs[phase] * radius_derivative
        toroidal = toroidal + scale * toroidal_signs[phase] * radius_derivative

    _, z = _radius_height_derivative_from_modes(
        spec,
        m,
        n,
        cos_terms,
        sin_terms,
        angle_scale,
        phi_order,
        theta_order,
    )
    return jnp.stack(
        [
            radial * cos_phi - toroidal * sin_phi,
            radial * sin_phi + toroidal * cos_phi,
            z,
        ],
        axis=-1,
    )


def _scatter_zero_like(reference: jax.Array, size: int) -> jax.Array:
    zero = jnp.sum(lax.slice_in_dim(reference, 0, 0, axis=0))
    return jnp.broadcast_to(zero, (int(size),))


def _scatter_coefficients_impl(
    positions: tuple[int, ...],
    dofs: jax.Array,
    target_size: int,
    source_offset: int,
) -> jax.Array:
    positions_array = np.asarray(positions, dtype=np.int32)
    indices = runtime_device_put(positions_array, dtype=np.int32).reshape(-1, 1)
    start = int(source_offset)
    values = lax.slice_in_dim(dofs, start, start + int(positions_array.size), axis=0)
    base = _scatter_zero_like(dofs, target_size)
    return lax.scatter(
        base,
        indices,
        values,
        _SCATTER_SET_DIMS_1D,
        indices_are_sorted=True,
        unique_indices=True,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


@partial(jax.custom_vjp, nondiff_argnums=(0, 2, 3))
def _scatter_coefficients_vjp_impl(
    positions: tuple[int, ...],
    dofs: jax.Array,
    target_size: int,
    source_offset: int,
) -> jax.Array:
    return _scatter_coefficients_impl(
        positions,
        dofs,
        target_size,
        source_offset,
    )


def _scatter_coefficients_fwd(
    positions: tuple[int, ...],
    dofs: jax.Array,
    target_size: int,
    source_offset: int,
) -> tuple[jax.Array, tuple[int]]:
    return _scatter_coefficients_impl(
        positions,
        dofs,
        target_size,
        source_offset,
    ), (int(dofs.shape[0]),)


def _scatter_coefficients_bwd(
    positions: tuple[int, ...],
    target_size: int,
    source_offset: int,
    residual: tuple[int],
    cotangent: jax.Array,
) -> tuple[jax.Array]:
    del target_size
    (dofs_size,) = residual
    positions_array = np.asarray(positions, dtype=np.int32)
    columns = np.arange(
        int(source_offset),
        int(source_offset) + int(positions_array.size),
        dtype=np.int32,
    )
    updates = jnp.take(
        cotangent,
        runtime_device_put(positions_array, dtype=np.int32),
        axis=0,
    )
    base = _scatter_zero_like(cotangent, dofs_size)
    return (
        lax.scatter(
            base,
            runtime_device_put(columns, dtype=np.int32).reshape(-1, 1),
            updates,
            _SCATTER_SET_DIMS_1D,
            indices_are_sorted=True,
            unique_indices=True,
            mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
        ),
    )


_scatter_coefficients_vjp_impl.defvjp(
    _scatter_coefficients_fwd,
    _scatter_coefficients_bwd,
)


def _scatter_coefficients_raw(
    positions: np.ndarray,
    dofs: jax.Array,
    *,
    target_size: int,
    source_offset: int,
) -> jax.Array:
    return _scatter_coefficients_impl(
        tuple(int(position) for position in positions),
        dofs,
        int(target_size),
        int(source_offset),
    )


def _scatter_coefficients(
    positions: np.ndarray,
    dofs: jax.Array,
    *,
    target_size: int,
    source_offset: int,
) -> jax.Array:
    return _scatter_coefficients_vjp_impl(
        tuple(int(position) for position in positions),
        dofs,
        int(target_size),
        int(source_offset),
    )


def _surface_rz_hessian_work_bytes(spec: SurfaceRZFourierSpec, dofs: jax.Array) -> int:
    n_dofs = int(dofs.size)
    n_quad = int(spec.quadpoints_phi.size) * int(spec.quadpoints_theta.size)
    return n_dofs * n_dofs * n_quad * np.dtype(np.float64).itemsize


def _check_surface_rz_hessian_memory(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    *,
    name: str,
) -> None:
    work_bytes = _surface_rz_hessian_work_bytes(spec, dofs)
    if work_bytes > _HESSIAN_WORK_BYTES_LIMIT:
        raise MemoryError(
            f"{name} estimated RZ Hessian work is {work_bytes} bytes, exceeding "
            f"the {_HESSIAN_WORK_BYTES_LIMIT} byte guard. Reduce Fourier modes "
            "or quadrature before requesting the dense Hessian."
        )


def _coefficients_from_dofs(
    dofs: jax.Array,
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
    use_custom_vjp: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    coeff_shape = (mpol + 1, 2 * ntor + 1)
    flat_size = coeff_shape[0] * coeff_shape[1]
    include_positions = _block_mode_positions(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=True,
    )
    exclude_positions = _block_mode_positions(
        mpol=mpol,
        ntor=ntor,
        include_zero_mode=False,
    )

    rc_count = int(include_positions.size)
    tail_count = int(exclude_positions.size)
    dofs = _as_jax_float64(dofs)
    scatter_coefficients = (
        _scatter_coefficients if use_custom_vjp else _scatter_coefficients_raw
    )

    def scatter_block(positions: np.ndarray, source_offset: int) -> jax.Array:
        return jnp.reshape(
            scatter_coefficients(
                positions,
                dofs,
                target_size=flat_size,
                source_offset=source_offset,
            ),
            coeff_shape,
        )

    rc = scatter_block(include_positions, 0)
    if stellsym:
        zs = scatter_block(exclude_positions, rc_count)
        zero = lax.stop_gradient(
            jnp.reshape(_scatter_zero_like(dofs, flat_size), coeff_shape)
        )
        return rc, zero, zero, zs

    rs_start = rc_count
    zc_start = rs_start + tail_count
    zs_start = zc_start + rc_count

    return (
        rc,
        scatter_block(exclude_positions, rs_start),
        scatter_block(include_positions, zc_start),
        scatter_block(exclude_positions, zs_start),
    )


def surface_rz_fourier_dofs_from_spec(spec: SurfaceRZFourierSpec) -> jax.Array:
    return _surface_rz_fourier_dofs_from_spec(spec)


def surface_rz_fourier_spec_from_dofs(
    dofs: jax.Array,
    *,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    use_custom_vjp: bool = False,
) -> SurfaceRZFourierSpec:
    rc, rs, zc, zs = _coefficients_from_dofs(
        dofs,
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        use_custom_vjp=use_custom_vjp,
    )
    return make_surface_rzfourier_spec(
        rc=rc,
        rs=rs,
        zc=zc,
        zs=zs,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        stellsym=stellsym,
    )


def _spec_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    *,
    use_custom_vjp: bool = False,
) -> SurfaceRZFourierSpec:
    return surface_rz_fourier_spec_from_dofs(
        dofs,
        quadpoints_phi=spec.quadpoints_phi,
        quadpoints_theta=spec.quadpoints_theta,
        mpol=spec.mpol,
        ntor=spec.ntor,
        nfp=spec.nfp,
        stellsym=spec.stellsym,
        use_custom_vjp=use_custom_vjp,
    )


def _evaluate_from_dofs(
    evaluator,
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
):
    return evaluator(_spec_from_dofs(spec, dofs))


def _evaluate_jacobian_from_dofs(
    evaluator,
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
):
    dofs_jax = _as_jax_float64(dofs)
    return jax.jacfwd(lambda x: evaluator(_spec_from_dofs(spec, x)))(dofs_jax)


def _evaluate_vjp_from_dofs(
    evaluator,
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    cotangent: jax.Array,
):
    dofs_jax = _as_jax_float64(dofs)
    cotangent_jax = _as_jax_float64(cotangent)
    _, pullback = jax.vjp(
        lambda x: evaluator(_spec_from_dofs(spec, x, use_custom_vjp=True)),
        dofs_jax,
    )
    return pullback(cotangent_jax)[0]


def surface_rz_fourier_gamma_from_spec(spec: SurfaceRZFourierSpec):
    phi, cos_terms, sin_terms = _mode_angles(spec)
    r, z = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_gamma_from_terms(r, z, cos_phi, sin_phi)


def surface_rz_fourier_gammadash1_from_spec(spec: SurfaceRZFourierSpec):
    phi, _, n, cos_terms, sin_terms = _mode_terms(spec)
    r, _ = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_gammadash1_from_terms(
        spec,
        r,
        n,
        cos_terms,
        sin_terms,
        cos_phi,
        sin_phi,
        two_pi(spec.quadpoints_theta),
    )


def surface_rz_fourier_gammadash2_from_spec(spec: SurfaceRZFourierSpec):
    phi, m, _, cos_terms, sin_terms = _mode_terms(spec)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_gammadash2_from_terms(
        spec,
        m,
        cos_terms,
        sin_terms,
        cos_phi,
        sin_phi,
        two_pi(spec.quadpoints_theta),
    )


def surface_rz_fourier_gammadash1dash1_from_spec(spec: SurfaceRZFourierSpec):
    phi, m, n, cos_terms, sin_terms = _mode_terms(spec)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_derivative_from_terms(
        spec,
        2,
        0,
        m,
        n,
        cos_terms,
        sin_terms,
        cos_phi,
        sin_phi,
        two_pi(spec.quadpoints_theta),
    )


def surface_rz_fourier_gammadash1dash2_from_spec(spec: SurfaceRZFourierSpec):
    phi, m, n, cos_terms, sin_terms = _mode_terms(spec)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_derivative_from_terms(
        spec,
        1,
        1,
        m,
        n,
        cos_terms,
        sin_terms,
        cos_phi,
        sin_phi,
        two_pi(spec.quadpoints_theta),
    )


def surface_rz_fourier_gammadash2dash2_from_spec(spec: SurfaceRZFourierSpec):
    phi, m, n, cos_terms, sin_terms = _mode_terms(spec)
    cos_phi, sin_phi = _phi_frame(phi)
    return _surface_rz_fourier_derivative_from_terms(
        spec,
        0,
        2,
        m,
        n,
        cos_terms,
        sin_terms,
        cos_phi,
        sin_phi,
        two_pi(spec.quadpoints_theta),
    )


def _surface_rz_fourier_derivative_lin_from_spec(
    spec: SurfaceRZFourierSpec,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    phi_order: int,
    theta_order: int,
) -> jax.Array:
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)

    def evaluate_pair(phi: jax.Array, theta: jax.Array):
        pair_spec = make_surface_rzfourier_spec(
            rc=spec.rc,
            rs=spec.rs,
            zc=spec.zc,
            zs=spec.zs,
            quadpoints_phi=jnp.atleast_1d(phi),
            quadpoints_theta=jnp.atleast_1d(theta),
            nfp=spec.nfp,
            stellsym=spec.stellsym,
        )
        phi_grid, m, n, cos_terms, sin_terms = _mode_terms(pair_spec)
        cos_phi, sin_phi = _phi_frame(phi_grid)
        return _surface_rz_fourier_derivative_from_terms(
            pair_spec,
            phi_order,
            theta_order,
            m,
            n,
            cos_terms,
            sin_terms,
            cos_phi,
            sin_phi,
            two_pi(pair_spec.quadpoints_theta),
        )[0, 0]

    return jax.vmap(evaluate_pair)(quadpoints_phi_jax, quadpoints_theta_jax)


def surface_rz_fourier_gammadash1dash1dash1_lin_from_spec(
    spec: SurfaceRZFourierSpec,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return _surface_rz_fourier_derivative_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        3,
        0,
    )


def surface_rz_fourier_gammadash1dash1dash2_lin_from_spec(
    spec: SurfaceRZFourierSpec,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return _surface_rz_fourier_derivative_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        2,
        1,
    )


def surface_rz_fourier_gammadash1dash2dash2_lin_from_spec(
    spec: SurfaceRZFourierSpec,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return _surface_rz_fourier_derivative_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        1,
        2,
    )


def surface_rz_fourier_gammadash2dash2dash2_lin_from_spec(
    spec: SurfaceRZFourierSpec,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return _surface_rz_fourier_derivative_lin_from_spec(
        spec,
        quadpoints_phi,
        quadpoints_theta,
        0,
        3,
    )


def surface_rz_fourier_first_fund_form_from_spec(spec: SurfaceRZFourierSpec):
    drd1 = surface_rz_fourier_gammadash1_from_spec(spec)
    drd2 = surface_rz_fourier_gammadash2_from_spec(spec)
    return jnp.stack(
        [
            jnp.sum(drd1 * drd1, axis=-1),
            jnp.sum(drd1 * drd2, axis=-1),
            jnp.sum(drd2 * drd2, axis=-1),
        ],
        axis=-1,
    )


def surface_rz_fourier_second_fund_form_from_spec(spec: SurfaceRZFourierSpec):
    unitnormal = surface_rz_fourier_unitnormal_from_spec(spec)
    d2rd1d1 = surface_rz_fourier_gammadash1dash1_from_spec(spec)
    d2rd1d2 = surface_rz_fourier_gammadash1dash2_from_spec(spec)
    d2rd2d2 = surface_rz_fourier_gammadash2dash2_from_spec(spec)
    return jnp.stack(
        [
            jnp.sum(unitnormal * d2rd1d1, axis=-1),
            jnp.sum(unitnormal * d2rd1d2, axis=-1),
            jnp.sum(unitnormal * d2rd2d2, axis=-1),
        ],
        axis=-1,
    )


def surface_rz_fourier_surface_curvatures_from_spec(spec: SurfaceRZFourierSpec):
    first = surface_rz_fourier_first_fund_form_from_spec(spec)
    second = surface_rz_fourier_second_fund_form_from_spec(spec)
    e = first[:, :, 0]
    f = first[:, :, 1]
    g = first[:, :, 2]
    ell = second[:, :, 0]
    m = second[:, :, 1]
    n = second[:, :, 2]
    denom = e * g - f * f
    mean_curvature = (ell * g - 2.0 * f * m + n * e) / (2.0 * denom)
    gaussian_curvature = (ell * n - m * m) / denom
    principal_offset = jnp.sqrt(mean_curvature * mean_curvature - gaussian_curvature)
    return jnp.stack(
        [
            mean_curvature,
            gaussian_curvature,
            mean_curvature + principal_offset,
            mean_curvature - principal_offset,
        ],
        axis=-1,
    )


def surface_rz_fourier_geometry_from_spec(
    spec: SurfaceRZFourierSpec,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    phi, m, n, cos_terms, sin_terms = _mode_terms(spec)
    r, z = _radius_height_from_modes(spec, cos_terms, sin_terms)
    cos_phi, sin_phi = _phi_frame(phi)
    angle_scale = two_pi(spec.quadpoints_theta)
    return (
        _surface_rz_fourier_gamma_from_terms(r, z, cos_phi, sin_phi),
        _surface_rz_fourier_gammadash1_from_terms(
            spec,
            r,
            n,
            cos_terms,
            sin_terms,
            cos_phi,
            sin_phi,
            angle_scale,
        ),
        _surface_rz_fourier_gammadash2_from_terms(
            spec,
            m,
            cos_terms,
            sin_terms,
            cos_phi,
            sin_phi,
            angle_scale,
        ),
    )


def surface_rz_fourier_normal_from_spec(spec: SurfaceRZFourierSpec):
    return jnp.cross(
        surface_rz_fourier_gammadash1_from_spec(spec),
        surface_rz_fourier_gammadash2_from_spec(spec),
    )


def surface_rz_fourier_unitnormal_from_spec(spec: SurfaceRZFourierSpec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    return _unit_vector3(normal)


def _normal_norm(normal: jax.Array) -> jax.Array:
    return _norm3(normal)


def surface_rz_fourier_unitnormal_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_unitnormal_from_spec, spec, dofs)


def surface_rz_fourier_area_from_spec(spec: SurfaceRZFourierSpec):
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = normal.shape[:2]
    return jnp.sum(_normal_norm(normal)[..., 0]) / (nphi * ntheta)


def surface_rz_fourier_volume_from_spec(spec: SurfaceRZFourierSpec):
    gamma = surface_rz_fourier_gamma_from_spec(spec)
    normal = surface_rz_fourier_normal_from_spec(spec)
    nphi, ntheta = gamma.shape[:2]
    return jnp.sum(jnp.sum(gamma * normal, axis=-1)) / (3.0 * nphi * ntheta)


def surface_rz_fourier_mean_cross_sectional_area_from_spec(
    spec: SurfaceRZFourierSpec,
):
    gamma = surface_rz_fourier_gamma_from_spec(spec)
    gammadash1 = surface_rz_fourier_gammadash1_from_spec(spec)
    gammadash2 = surface_rz_fourier_gammadash2_from_spec(spec)
    x = gamma[:, :, 0]
    y = gamma[:, :, 1]
    radius_squared = x * x + y * y
    j00 = (x * gammadash1[:, :, 1] - y * gammadash1[:, :, 0]) / radius_squared
    j01 = (x * gammadash2[:, :, 1] - y * gammadash2[:, :, 0]) / radius_squared
    dz_dtheta = gammadash2[:, :, 2] - gammadash1[:, :, 2] * j01 / j00
    signed_area = jnp.mean(jnp.sqrt(radius_squared) * dz_dtheta * j00) / (
        two_pi(spec.quadpoints_theta)
    )
    return jnp.abs(signed_area)


def surface_rz_fourier_minor_radius_from_spec(spec: SurfaceRZFourierSpec):
    mean_area = surface_rz_fourier_mean_cross_sectional_area_from_spec(spec)
    pi = two_pi(spec.quadpoints_theta) / 2.0
    return jnp.sqrt(mean_area / pi)


def surface_rz_fourier_major_radius_from_spec(spec: SurfaceRZFourierSpec):
    volume = surface_rz_fourier_volume_from_spec(spec)
    minor_radius = surface_rz_fourier_minor_radius_from_spec(spec)
    pi = two_pi(spec.quadpoints_theta) / 2.0
    return jnp.abs(volume) / (2.0 * pi * pi * minor_radius * minor_radius)


def surface_rz_fourier_aspect_ratio_from_spec(spec: SurfaceRZFourierSpec):
    return (
        surface_rz_fourier_major_radius_from_spec(spec)
        / surface_rz_fourier_minor_radius_from_spec(spec)
    )


def surface_rz_fourier_dnormal_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_jacobian_from_dofs(surface_rz_fourier_normal_from_spec, spec, dofs)


def surface_rz_fourier_dunitnormal_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_unitnormal_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gamma_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_gamma_from_spec, spec, dofs)


def surface_rz_fourier_gammadash1_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_gammadash1_from_spec, spec, dofs)


def surface_rz_fourier_gammadash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_gammadash2_from_spec, spec, dofs)


def surface_rz_fourier_gammadash1dash1_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(
        surface_rz_fourier_gammadash1dash1_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gammadash1dash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(
        surface_rz_fourier_gammadash1dash2_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gammadash2dash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(
        surface_rz_fourier_gammadash2dash2_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gammadash1dash1dash1_lin_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return surface_rz_fourier_gammadash1dash1dash1_lin_from_spec(
        _spec_from_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_rz_fourier_gammadash1dash1dash2_lin_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return surface_rz_fourier_gammadash1dash1dash2_lin_from_spec(
        _spec_from_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_rz_fourier_gammadash1dash2dash2_lin_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return surface_rz_fourier_gammadash1dash2dash2_lin_from_spec(
        _spec_from_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_rz_fourier_gammadash2dash2dash2_lin_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
) -> jax.Array:
    return surface_rz_fourier_gammadash2dash2dash2_lin_from_spec(
        _spec_from_dofs(spec, dofs),
        quadpoints_phi,
        quadpoints_theta,
    )


def surface_rz_fourier_dgammadash1dash1_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_gammadash1dash1_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_dgammadash1dash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_gammadash1dash2_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_dgammadash2dash2_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_gammadash2dash2_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_gammadash1dash1_vjp_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    cotangent: jax.Array,
):
    return _evaluate_vjp_from_dofs(
        surface_rz_fourier_gammadash1dash1_from_spec,
        spec,
        dofs,
        cotangent,
    )


def surface_rz_fourier_gammadash1dash2_vjp_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    cotangent: jax.Array,
):
    return _evaluate_vjp_from_dofs(
        surface_rz_fourier_gammadash1dash2_from_spec,
        spec,
        dofs,
        cotangent,
    )


def surface_rz_fourier_gammadash2dash2_vjp_from_dofs(
    spec: SurfaceRZFourierSpec,
    dofs: jax.Array,
    cotangent: jax.Array,
):
    return _evaluate_vjp_from_dofs(
        surface_rz_fourier_gammadash2dash2_from_spec,
        spec,
        dofs,
        cotangent,
    )


def surface_rz_fourier_first_fund_form_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_first_fund_form_from_spec, spec, dofs)


def surface_rz_fourier_second_fund_form_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_second_fund_form_from_spec, spec, dofs)


def surface_rz_fourier_surface_curvatures_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(
        surface_rz_fourier_surface_curvatures_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_dfirst_fund_form_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_first_fund_form_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_dsecond_fund_form_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_second_fund_form_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_dsurface_curvatures_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_jacobian_from_dofs(
        surface_rz_fourier_surface_curvatures_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_mean_cross_sectional_area_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(
        surface_rz_fourier_mean_cross_sectional_area_from_spec,
        spec,
        dofs,
    )


def surface_rz_fourier_minor_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_minor_radius_from_spec, spec, dofs)


def surface_rz_fourier_major_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_major_radius_from_spec, spec, dofs)


def surface_rz_fourier_aspect_ratio_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return _evaluate_from_dofs(surface_rz_fourier_aspect_ratio_from_spec, spec, dofs)


def surface_rz_fourier_darea_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(lambda x: surface_rz_fourier_area_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_dvolume_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(lambda x: surface_rz_fourier_volume_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_dmean_cross_sectional_area_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(
        lambda x: surface_rz_fourier_mean_cross_sectional_area_from_dofs(spec, x)
    )(_as_jax_float64(dofs))


def surface_rz_fourier_dminor_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(lambda x: surface_rz_fourier_minor_radius_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_dmajor_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(lambda x: surface_rz_fourier_major_radius_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_daspect_ratio_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.grad(lambda x: surface_rz_fourier_aspect_ratio_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_d2area_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    dofs = _as_jax_float64(dofs)
    _check_surface_rz_hessian_memory(spec, dofs, name="surface_rz_fourier_d2area")
    return jax.hessian(lambda x: surface_rz_fourier_area_from_dofs(spec, x))(
        dofs
    )


def surface_rz_fourier_d2volume_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    dofs = _as_jax_float64(dofs)
    _check_surface_rz_hessian_memory(spec, dofs, name="surface_rz_fourier_d2volume")
    return jax.hessian(lambda x: surface_rz_fourier_volume_from_dofs(spec, x))(
        dofs
    )


def surface_rz_fourier_d2minor_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.hessian(lambda x: surface_rz_fourier_minor_radius_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_d2major_radius_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.hessian(lambda x: surface_rz_fourier_major_radius_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_d2aspect_ratio_from_dofs(
    spec: SurfaceRZFourierSpec, dofs: jax.Array
):
    return jax.hessian(lambda x: surface_rz_fourier_aspect_ratio_from_dofs(spec, x))(
        _as_jax_float64(dofs)
    )


def surface_rz_fourier_normal_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_normal_from_spec, spec, dofs)


def surface_rz_fourier_area_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_area_from_spec, spec, dofs)


def surface_rz_fourier_volume_from_dofs(spec: SurfaceRZFourierSpec, dofs: jax.Array):
    return _evaluate_from_dofs(surface_rz_fourier_volume_from_spec, spec, dofs)
