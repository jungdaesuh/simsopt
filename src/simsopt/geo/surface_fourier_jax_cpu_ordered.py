"""CPU-ordered JAX twins for ``SurfaceXYZTensorFourier`` evaluation.

Production hot-path kernels in :mod:`simsopt.geo.surface_fourier_jax` use
``einsum``/matmul (``V @ coeffs.T @ W.T``) which lower to XLA reductions whose
floating-point accumulation order does not match the C++ oracle in
``src/simsoptpp/surfacexyztensorfourier.h``. The
``docs/boozer_derivative_bit_identity_impl_plan_2026-05-07.md`` Phase 2
ladder requires per-build, per-host bit identity between CPU and JAX boundary
inputs to the Boozer LS callback, so this module supplies parallel kernels
that mirror the C++ accumulation order operator-for-operator:

* loop nesting ``k1 → k2 → m → n`` with serial scalar accumulators
  (``lax.fori_loop`` over ``(m, n)``);
* the rotation ``data(k1, k2, 0) = xhat*cos(phi) - yhat*sin(phi)`` applied
  once at the outer assembly;
* the ``2π`` chain-rule factor for ``gammadash{1,2}`` applied once outside
  the rotation, *not* distributed inside the basis.

These twins are diagnostic-grade: only the parity backend
(``SIMSOPT_BACKEND_MODE=jax_cpu_parity``/``jax_gpu_parity`` →
:func:`simsopt.backend.is_parity_mode`) routes through them. The production
fast paths in :mod:`simsopt.geo.surface_fourier_jax` are unchanged.

Pure JAX, no ``simsoptpp`` import (M1 contract).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from ..jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
)
from .surface_fourier_jax import (
    _dofs_to_xyzc_any,
    _two_pi,
)


__all__ = (
    "build_xyztensorfourier_basis_cache",
    "surface_gamma_cpu_ordered",
    "surface_gammadash1_cpu_ordered",
    "surface_gammadash2_cpu_ordered",
    "surface_gamma_from_dofs_cpu_ordered",
    "surface_gammadash1_from_dofs_cpu_ordered",
    "surface_gammadash2_from_dofs_cpu_ordered",
    "dgamma_by_dcoeff_cpu_ordered",
    "dgammadash1_by_dcoeff_cpu_ordered",
    "dgammadash2_by_dcoeff_cpu_ordered",
)


# ---------------------------------------------------------------------------
# Basis caches — identical to ``SurfaceXYZTensorFourier::build_cache``


def _build_phi_basis_cache(quadpoints_phi: jax.Array, ntor: int, nfp: int):
    """Mirror ``cache_basis_fun_phi`` and ``cache_basis_fun_phi_dash``.

    Returns:
        cache_phi: (nphi, 2*ntor+1) — ``basis_fun_phi(n, 2π·quadpoints_phi[k1])``.
        cache_phi_dash: (nphi, 2*ntor+1) — ``basis_fun_phi_dash`` at the
            same point. Note this is the derivative w.r.t. ``phi`` (not
            quadpoints_phi); the ``2π`` chain-rule factor is applied at the
            outer assembly to match C++.
    """
    quadpoints_phi = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi)
    phi = two_pi * quadpoints_phi  # (nphi,)
    # n in [0, ntor]: cos(nfp*n*phi); n in [ntor+1, 2*ntor]: sin(nfp*(n-ntor)*phi)
    n_cos_idx = jnp.arange(ntor + 1, dtype=jnp.int32)
    n_sin_idx = jnp.arange(1, ntor + 1, dtype=jnp.int32)
    arg_cos = _as_jax_float64(nfp * n_cos_idx)[None, :] * phi[:, None]  # (nphi, ntor+1)
    arg_sin = _as_jax_float64(nfp * n_sin_idx)[None, :] * phi[:, None]  # (nphi, ntor)
    cos_block = jnp.cos(arg_cos)
    sin_block = jnp.sin(arg_sin)
    cache_phi = jnp.concatenate([cos_block, sin_block], axis=1)
    # basis_fun_phi_dash:
    #   n=0..ntor:  -nfp*n*sin(nfp*n*phi)
    #   n=ntor+1..2*ntor: nfp*(n-ntor)*cos(nfp*(n-ntor)*phi)
    cos_block_dash = -_as_jax_float64(nfp * n_cos_idx)[None, :] * jnp.sin(arg_cos)
    sin_block_dash = _as_jax_float64(nfp * n_sin_idx)[None, :] * jnp.cos(arg_sin)
    cache_phi_dash = jnp.concatenate([cos_block_dash, sin_block_dash], axis=1)
    return cache_phi, cache_phi_dash


def _build_theta_basis_cache(quadpoints_theta: jax.Array, mpol: int):
    """Mirror ``cache_basis_fun_theta`` and ``cache_basis_fun_theta_dash``."""
    quadpoints_theta = _as_jax_float64(quadpoints_theta)
    two_pi = _two_pi(quadpoints_theta)
    theta = two_pi * quadpoints_theta  # (ntheta,)
    m_cos_idx = jnp.arange(mpol + 1, dtype=jnp.int32)
    m_sin_idx = jnp.arange(1, mpol + 1, dtype=jnp.int32)
    arg_cos = _as_jax_float64(m_cos_idx)[None, :] * theta[:, None]
    arg_sin = _as_jax_float64(m_sin_idx)[None, :] * theta[:, None]
    cos_block = jnp.cos(arg_cos)
    sin_block = jnp.sin(arg_sin)
    cache_theta = jnp.concatenate([cos_block, sin_block], axis=1)
    # basis_fun_theta_dash:
    #   m=0..mpol:    -m*sin(m*theta)
    #   m=mpol+1..2*mpol: (m-mpol)*cos((m-mpol)*theta)
    cos_block_dash = -_as_jax_float64(m_cos_idx)[None, :] * jnp.sin(arg_cos)
    sin_block_dash = _as_jax_float64(m_sin_idx)[None, :] * jnp.cos(arg_sin)
    cache_theta_dash = jnp.concatenate([cos_block_dash, sin_block_dash], axis=1)
    return cache_theta, cache_theta_dash


def build_xyztensorfourier_basis_cache(
    quadpoints_phi: jax.Array,
    quadpoints_theta: jax.Array,
    mpol: int,
    ntor: int,
    nfp: int,
):
    """Mirror the relevant portion of C++ ``SurfaceXYZTensorFourier::build_cache``.

    Returns ``(cache_phi, cache_phi_dash, cache_theta, cache_theta_dash,
    cosphi, sinphi)``. The two ``cache_phi*`` arrays are ``(nphi, 2*ntor+1)``
    and match ``cache_basis_fun_phi[_dash](k1, n)`` at the same indices. The
    cached angles ``cosphi/sinphi`` come from the same ``2π·quadpoints_phi``
    used by C++; storing them avoids re-running ``cos``/``sin`` per loop.
    """
    cache_phi, cache_phi_dash = _build_phi_basis_cache(quadpoints_phi, ntor, nfp)
    cache_theta, cache_theta_dash = _build_theta_basis_cache(quadpoints_theta, mpol)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_angles = two_pi * quadpoints_phi_jax
    cosphi = jnp.cos(phi_angles)
    sinphi = jnp.sin(phi_angles)
    return (
        cache_phi,
        cache_phi_dash,
        cache_theta,
        cache_theta_dash,
        cosphi,
        sinphi,
    )


# ---------------------------------------------------------------------------
# Per-cell evaluators


def _hat_sum_cpu_ordered(
    cache_phi_row: jax.Array,
    cache_theta_row: jax.Array,
    coeffs: jax.Array,
    *,
    ncols: int,
    nrows: int,
):
    """Serial ``Σ_m Σ_n coeffs(m, n) · phi(n) · theta(m)``.

    Loop nesting matches C++ ``gamma_impl``: outer ``m``, inner ``n``,
    scalar accumulator. Both bounds are static (Python ints) so the loop
    unrolls under ``jax.jit``.
    """
    zero = jnp.zeros((), dtype=cache_phi_row.dtype)

    def _m_body(m, accum):
        def _n_body(n, accum_inner):
            bf = cache_phi_row[n] * cache_theta_row[m]
            return accum_inner + coeffs[m, n] * bf

        return lax.fori_loop(0, ncols, _n_body, accum)

    return lax.fori_loop(0, nrows, _m_body, zero)


def _gamma_cell_cpu_ordered(
    cache_phi_row,
    cache_theta_row,
    xc,
    yc,
    zc,
    cosphi,
    sinphi,
    *,
    mpol: int,
    ntor: int,
):
    nrows = 2 * mpol + 1
    ncols = 2 * ntor + 1
    xhat = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_row, xc, ncols=ncols, nrows=nrows
    )
    yhat = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_row, yc, ncols=ncols, nrows=nrows
    )
    z = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_row, zc, ncols=ncols, nrows=nrows
    )
    return jnp.stack([xhat * cosphi - yhat * sinphi, xhat * sinphi + yhat * cosphi, z])


def _gammadash1_cell_cpu_ordered(
    cache_phi_row,
    cache_phi_dash_row,
    cache_theta_row,
    xc,
    yc,
    zc,
    cosphi,
    sinphi,
    *,
    mpol: int,
    ntor: int,
    two_pi,
):
    nrows = 2 * mpol + 1
    ncols = 2 * ntor + 1
    xhat = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_row, xc, ncols=ncols, nrows=nrows
    )
    yhat = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_row, yc, ncols=ncols, nrows=nrows
    )
    xhatdash = _hat_sum_cpu_ordered(
        cache_phi_dash_row, cache_theta_row, xc, ncols=ncols, nrows=nrows
    )
    yhatdash = _hat_sum_cpu_ordered(
        cache_phi_dash_row, cache_theta_row, yc, ncols=ncols, nrows=nrows
    )
    zdash = _hat_sum_cpu_ordered(
        cache_phi_dash_row, cache_theta_row, zc, ncols=ncols, nrows=nrows
    )
    xdash = xhatdash * cosphi - yhatdash * sinphi - xhat * sinphi - yhat * cosphi
    ydash = xhatdash * sinphi + yhatdash * cosphi + xhat * cosphi - yhat * sinphi
    return jnp.stack([two_pi * xdash, two_pi * ydash, two_pi * zdash])


def _gammadash2_cell_cpu_ordered(
    cache_phi_row,
    cache_theta_row,
    cache_theta_dash_row,
    xc,
    yc,
    zc,
    cosphi,
    sinphi,
    *,
    mpol: int,
    ntor: int,
    two_pi,
):
    nrows = 2 * mpol + 1
    ncols = 2 * ntor + 1
    xhatdash = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_dash_row, xc, ncols=ncols, nrows=nrows
    )
    yhatdash = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_dash_row, yc, ncols=ncols, nrows=nrows
    )
    zdash = _hat_sum_cpu_ordered(
        cache_phi_row, cache_theta_dash_row, zc, ncols=ncols, nrows=nrows
    )
    xdash = xhatdash * cosphi - yhatdash * sinphi
    ydash = xhatdash * sinphi + yhatdash * cosphi
    return jnp.stack([two_pi * xdash, two_pi * ydash, two_pi * zdash])


# ---------------------------------------------------------------------------
# Top-level kernels


def surface_gamma_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol: int,
    ntor: int,
    nfp: int,
):
    """CPU-ordered ``surface_gamma`` matching ``gamma_impl`` in
    ``surfacexyztensorfourier.h:127``."""
    cache_phi, _, cache_theta, _, cosphi, sinphi = build_xyztensorfourier_basis_cache(
        quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
    )
    cell = partial(_gamma_cell_cpu_ordered, mpol=mpol, ntor=ntor)
    # vmap over k2 first (innermost) then over k1
    cell_over_theta = jax.vmap(
        cell,
        in_axes=(None, 0, None, None, None, None, None),
    )
    cell_over_phi_then_theta = jax.vmap(
        cell_over_theta,
        in_axes=(0, None, None, None, None, 0, 0),
    )
    return cell_over_phi_then_theta(
        cache_phi,
        cache_theta,
        xc,
        yc,
        zc,
        cosphi,
        sinphi,
    )


def surface_gammadash1_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol: int,
    ntor: int,
    nfp: int,
):
    """CPU-ordered ``surface_gammadash1`` matching ``gammadash1_impl``."""
    cache_phi, cache_phi_dash, cache_theta, _, cosphi, sinphi = (
        build_xyztensorfourier_basis_cache(
            quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
        )
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    cell = partial(
        _gammadash1_cell_cpu_ordered,
        mpol=mpol,
        ntor=ntor,
        two_pi=two_pi,
    )
    cell_over_theta = jax.vmap(
        cell,
        in_axes=(None, None, 0, None, None, None, None, None),
    )
    cell_over_phi = jax.vmap(
        cell_over_theta,
        in_axes=(0, 0, None, None, None, None, 0, 0),
    )
    return cell_over_phi(
        cache_phi,
        cache_phi_dash,
        cache_theta,
        xc,
        yc,
        zc,
        cosphi,
        sinphi,
    )


def surface_gammadash2_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol: int,
    ntor: int,
    nfp: int,
):
    """CPU-ordered ``surface_gammadash2`` matching ``gammadash2_impl``."""
    cache_phi, _, cache_theta, cache_theta_dash, cosphi, sinphi = (
        build_xyztensorfourier_basis_cache(
            quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
        )
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    cell = partial(
        _gammadash2_cell_cpu_ordered,
        mpol=mpol,
        ntor=ntor,
        two_pi=two_pi,
    )
    cell_over_theta = jax.vmap(
        cell,
        in_axes=(None, 0, 0, None, None, None, None, None),
    )
    cell_over_phi = jax.vmap(
        cell_over_theta,
        in_axes=(0, None, None, None, None, None, 0, 0),
    )
    return cell_over_phi(
        cache_phi,
        cache_theta,
        cache_theta_dash,
        xc,
        yc,
        zc,
        cosphi,
        sinphi,
    )


# ---------------------------------------------------------------------------
# Analytic dgamma_by_dcoeff* kernels


def _stellsym_skip_mask(
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
):
    """Mirror ``SurfaceXYZTensorFourier::skip``.

    Returns a ``(3, 2*mpol+1, 2*ntor+1)`` boolean mask: ``True`` where the
    coefficient is skipped (constrained by stellarator symmetry, so it is
    not a free DOF). Matches the C++ ``skip(dim, m, n)`` predicate.
    """
    mask = np.zeros((3, 2 * mpol + 1, 2 * ntor + 1), dtype=bool)
    if not stellsym:
        return mask
    for m in range(2 * mpol + 1):
        for n in range(2 * ntor + 1):
            cos_phi_block = n <= ntor
            cos_theta_block = m <= mpol
            # dim 0 (x):  cos-cos + sin-sin
            x_skip = (cos_phi_block and not cos_theta_block) or (
                (not cos_phi_block) and cos_theta_block
            )
            # dim 1 (y):  cos-sin + sin-cos
            y_skip = (cos_phi_block and cos_theta_block) or (
                (not cos_phi_block) and (not cos_theta_block)
            )
            # dim 2 (z):  same skip pattern as y
            z_skip = y_skip
            mask[0, m, n] = x_skip
            mask[1, m, n] = y_skip
            mask[2, m, n] = z_skip
    return mask


def _coeff_dof_index_table(
    *,
    mpol: int,
    ntor: int,
    stellsym: bool,
):
    """Return the per-(dim, m, n) DOF counter used by C++ ``dgamma_by_dcoeff_impl``.

    The C++ kernel walks ``d → m → n`` and increments a ``counter`` for each
    non-skipped coefficient. The result is a mapping from coefficient
    position to DOF index. Returns ``-1`` for skipped entries.
    """
    skip_mask = _stellsym_skip_mask(mpol=mpol, ntor=ntor, stellsym=stellsym)
    counters = np.full(skip_mask.shape, -1, dtype=np.int64)
    counter = 0
    for d in range(3):
        for m in range(2 * mpol + 1):
            for n in range(2 * ntor + 1):
                if skip_mask[d, m, n]:
                    continue
                counters[d, m, n] = counter
                counter += 1
    return counters, counter


def _dgamma_by_dcoeff_dense(
    cache_phi,
    cache_theta,
    *,
    cosphi,
    sinphi,
    counters: np.ndarray,
    ndofs: int,
    mpol: int,
    ntor: int,
    factor: float = 1.0,
    cache_phi_dash=None,
    cache_theta_dash=None,
    derivative_kind: str = "value",
):
    """Build the (nphi, ntheta, 3, ndofs) Jacobian by mirroring C++ loops.

    Args:
        derivative_kind: ``"value"`` for ``dgamma_by_dcoeff_impl``,
            ``"dphi"`` for ``dgammadash1_by_dcoeff_impl``, ``"dtheta"`` for
            ``dgammadash2_by_dcoeff_impl``.
        factor: scalar multiplier applied at the OUTSIDE of the rotation
            (matches C++ ``2*M_PI`` for the dash variants).
    """
    nphi = cache_phi.shape[0]
    ntheta = cache_theta.shape[0]
    if derivative_kind == "value":
        if cache_phi_dash is not None or cache_theta_dash is not None:
            raise ValueError("derivative_kind='value' must not receive _dash caches")
    elif derivative_kind == "dphi":
        if cache_phi_dash is None:
            raise ValueError("derivative_kind='dphi' requires cache_phi_dash")
    elif derivative_kind == "dtheta":
        if cache_theta_dash is None:
            raise ValueError("derivative_kind='dtheta' requires cache_theta_dash")
    else:
        raise ValueError(f"derivative_kind={derivative_kind!r} not supported")

    # We materialize a per-(d, m, n) plane of size (nphi, ntheta) and add it
    # at the right slot in the output. Doing this on the host with explicit
    # loops over (d, m, n) preserves the C++ scatter order (the values are
    # independent per DOF, so accumulation order is irrelevant — what
    # matters is that the per-cell ``wivj`` values use the same basis cache
    # entries the C++ kernel reads).
    out_zero = jnp.zeros((nphi, ntheta, 3, ndofs), dtype=cache_phi.dtype)

    def per_dof_planes():
        for d in range(3):
            for m in range(2 * mpol + 1):
                for n in range(2 * ntor + 1):
                    counter = int(counters[d, m, n])
                    if counter < 0:
                        continue
                    if derivative_kind == "value":
                        wivj = (
                            cache_phi[:, n][:, None] * cache_theta[:, m][None, :]
                        )  # (nphi, ntheta)
                        if d == 0:
                            dx = wivj * cosphi[:, None]
                            dy = wivj * sinphi[:, None]
                            yield d, n, m, counter, (dx, dy, None)
                        elif d == 1:
                            dx = -wivj * sinphi[:, None]
                            dy = wivj * cosphi[:, None]
                            yield d, n, m, counter, (dx, dy, None)
                        else:
                            yield d, n, m, counter, (None, None, wivj)
                    elif derivative_kind == "dphi":
                        wivj = cache_phi[:, n][:, None] * cache_theta[:, m][None, :]
                        wivjdash = (
                            cache_phi_dash[:, n][:, None] * cache_theta[:, m][None, :]
                        )
                        if d == 0:
                            dx = factor * (
                                wivjdash * cosphi[:, None] - wivj * sinphi[:, None]
                            )
                            dy = factor * (
                                wivjdash * sinphi[:, None] + wivj * cosphi[:, None]
                            )
                            yield d, n, m, counter, (dx, dy, None)
                        elif d == 1:
                            dx = factor * (
                                -wivjdash * sinphi[:, None] - wivj * cosphi[:, None]
                            )
                            dy = factor * (
                                wivjdash * cosphi[:, None] - wivj * sinphi[:, None]
                            )
                            yield d, n, m, counter, (dx, dy, None)
                        else:
                            yield d, n, m, counter, (None, None, factor * wivjdash)
                    else:  # dtheta
                        wivjdash = (
                            cache_phi[:, n][:, None] * cache_theta_dash[:, m][None, :]
                        )
                        if d == 0:
                            dx = factor * wivjdash * cosphi[:, None]
                            dy = factor * wivjdash * sinphi[:, None]
                            yield d, n, m, counter, (dx, dy, None)
                        elif d == 1:
                            dx = -factor * wivjdash * sinphi[:, None]
                            dy = factor * wivjdash * cosphi[:, None]
                            yield d, n, m, counter, (dx, dy, None)
                        else:
                            yield d, n, m, counter, (None, None, factor * wivjdash)

    out = out_zero
    for _, _, _, counter, (dx, dy, dz) in per_dof_planes():
        if dx is not None:
            out = out.at[:, :, 0, counter].set(dx)
        if dy is not None:
            out = out.at[:, :, 1, counter].set(dy)
        if dz is not None:
            out = out.at[:, :, 2, counter].set(dz)
    return out


def _coefficients_from_dofs(
    sdofs,
    mpol: int,
    ntor: int,
    stellsym: bool,
    scatter_indices,
):
    return _dofs_to_xyzc_any(sdofs, mpol, ntor, stellsym, scatter_indices)


def surface_gamma_from_dofs_cpu_ordered(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    scatter_indices=None,
):
    xc, yc, zc = _coefficients_from_dofs(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gamma_cpu_ordered(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
    )


def surface_gammadash1_from_dofs_cpu_ordered(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    scatter_indices=None,
):
    xc, yc, zc = _coefficients_from_dofs(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1_cpu_ordered(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
    )


def surface_gammadash2_from_dofs_cpu_ordered(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
    scatter_indices=None,
):
    xc, yc, zc = _coefficients_from_dofs(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2_cpu_ordered(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
    )


def dgamma_by_dcoeff_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
):
    cache_phi, _, cache_theta, _, cosphi, sinphi = build_xyztensorfourier_basis_cache(
        quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
    )
    counters, ndofs = _coeff_dof_index_table(mpol=mpol, ntor=ntor, stellsym=stellsym)
    return _dgamma_by_dcoeff_dense(
        cache_phi,
        cache_theta,
        cosphi=cosphi,
        sinphi=sinphi,
        counters=counters,
        ndofs=ndofs,
        mpol=mpol,
        ntor=ntor,
        derivative_kind="value",
    )


def dgammadash1_by_dcoeff_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
):
    cache_phi, cache_phi_dash, cache_theta, _, cosphi, sinphi = (
        build_xyztensorfourier_basis_cache(
            quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
        )
    )
    counters, ndofs = _coeff_dof_index_table(mpol=mpol, ntor=ntor, stellsym=stellsym)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    return _dgamma_by_dcoeff_dense(
        cache_phi,
        cache_theta,
        cosphi=cosphi,
        sinphi=sinphi,
        counters=counters,
        ndofs=ndofs,
        mpol=mpol,
        ntor=ntor,
        factor=two_pi,
        cache_phi_dash=cache_phi_dash,
        derivative_kind="dphi",
    )


def dgammadash2_by_dcoeff_cpu_ordered(
    quadpoints_phi,
    quadpoints_theta,
    *,
    mpol: int,
    ntor: int,
    nfp: int,
    stellsym: bool,
):
    cache_phi, _, cache_theta, cache_theta_dash, cosphi, sinphi = (
        build_xyztensorfourier_basis_cache(
            quadpoints_phi, quadpoints_theta, mpol, ntor, nfp
        )
    )
    counters, ndofs = _coeff_dof_index_table(mpol=mpol, ntor=ntor, stellsym=stellsym)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    return _dgamma_by_dcoeff_dense(
        cache_phi,
        cache_theta,
        cosphi=cosphi,
        sinphi=sinphi,
        counters=counters,
        ndofs=ndofs,
        mpol=mpol,
        ntor=ntor,
        factor=two_pi,
        cache_theta_dash=cache_theta_dash,
        derivative_kind="dtheta",
    )
