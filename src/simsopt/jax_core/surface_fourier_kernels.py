"""
Pure JAX evaluation of SurfaceXYZTensorFourier geometry.

Replaces the C++ ``sopp.SurfaceXYZTensorFourier`` evaluation methods
(``gamma``, ``gammadash1``, ``gammadash2``, ``normal``,
``dgamma_by_dcoeff``) with JIT-compilable, autodiff-compatible functions.

Coefficient layout
------------------
The coefficient matrices ``xc``, ``yc``, ``zc`` each have shape
``(2*mpol+1, 2*ntor+1)`` matching the basis ordering:

* **rows** (theta basis):  ``{1, cos θ, …, cos(mpol·θ), sin θ, …, sin(mpol·θ)}``
* **cols** (phi basis):    ``{1, cos(nfp·φ), …, cos(ntor·nfp·φ),
                               sin(nfp·φ), …, sin(ntor·nfp·φ)}``

where ``θ = 2π·quadpoints_theta`` and ``φ = 2π·quadpoints_phi``.

For stellarator symmetry the caller must zero out the forbidden entries
*before* calling these functions; no masking is applied here.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

from simsopt.jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
    as_runtime_float64 as _as_runtime_float64,
    zeros as _zeros,
)
from simsopt.jax_core._vector_norms import unit_vector3 as _unit_vector3
from simsopt.jax_core.surface_integrals import surface_area, surface_volume
from simsopt.jax_core.surface_fourier_indices import stellsym_scatter_indices

__all__ = [
    "build_theta_basis",
    "build_phi_basis",
    "surface_gamma",
    "surface_gamma_lin",
    "surface_gammadash1",
    "surface_gammadash1_lin",
    "surface_gammadash2",
    "surface_gammadash2_lin",
    "surface_gammadash1dash1",
    "surface_gammadash1dash2",
    "surface_gammadash2dash2",
    "surface_normal",
    "surface_gamma_from_dofs",
    "surface_gamma_lin_from_dofs",
    "surface_gammadash1_from_dofs",
    "surface_gammadash1_lin_from_dofs",
    "surface_gammadash2_from_dofs",
    "surface_gammadash2_lin_from_dofs",
    "surface_gammadash1dash1_from_dofs",
    "surface_gammadash1dash1_lin_from_dofs",
    "surface_gammadash1dash2_from_dofs",
    "surface_gammadash1dash2_lin_from_dofs",
    "surface_gammadash2dash2_from_dofs",
    "surface_gammadash2dash2_lin_from_dofs",
    "surface_gammadash1dash1dash1_lin_from_dofs",
    "surface_gammadash1dash1dash2_lin_from_dofs",
    "surface_gammadash1dash2dash2_lin_from_dofs",
    "surface_gammadash2dash2dash2_lin_from_dofs",
    "surface_normal_from_dofs",
    "surface_unitnormal_from_dofs",
    "surface_area_from_dofs",
    "surface_volume_from_dofs",
    "surface_xyzfourier_gamma_from_dofs",
    "surface_xyzfourier_gamma_lin_from_dofs",
    "surface_xyzfourier_gammadash1_from_dofs",
    "surface_xyzfourier_gammadash1_lin_from_dofs",
    "surface_xyzfourier_gammadash2_from_dofs",
    "surface_xyzfourier_gammadash2_lin_from_dofs",
    "surface_xyzfourier_gammadash1dash1_from_dofs",
    "surface_xyzfourier_gammadash1dash1_lin_from_dofs",
    "surface_xyzfourier_gammadash1dash2_from_dofs",
    "surface_xyzfourier_gammadash1dash2_lin_from_dofs",
    "surface_xyzfourier_gammadash2dash2_from_dofs",
    "surface_xyzfourier_gammadash2dash2_lin_from_dofs",
    "surface_xyzfourier_gammadash1dash1dash1_lin_from_dofs",
    "surface_xyzfourier_gammadash1dash1dash2_lin_from_dofs",
    "surface_xyzfourier_gammadash1dash2dash2_lin_from_dofs",
    "surface_xyzfourier_gammadash2dash2dash2_lin_from_dofs",
    "surface_xyzfourier_normal_from_dofs",
    "surface_xyzfourier_unitnormal_from_dofs",
    "surface_xyzfourier_area_from_dofs",
    "surface_xyzfourier_volume_from_dofs",
    "surface_volume",
    "surface_area",
    "stellsym_scatter_indices",
    "dofs_to_xyzc",
    "dgamma_by_dcoeff",
    "dgammadash1_by_dcoeff",
    "dgammadash2_by_dcoeff",
    "dgammadash1dash1_by_dcoeff",
    "dgammadash1dash2_by_dcoeff",
    "dgammadash2dash2_by_dcoeff",
    "dnormal_by_dcoeff",
    "d2normal_by_dcoeffdcoeff",
    "dunitnormal_by_dcoeff",
    "darea_by_dcoeff",
    "d2area_by_dcoeffdcoeff",
    "dvolume_by_dcoeff",
    "d2volume_by_dcoeffdcoeff",
    "surface_xyzfourier_dgamma_by_dcoeff",
    "surface_xyzfourier_dgammadash1_by_dcoeff",
    "surface_xyzfourier_dgammadash2_by_dcoeff",
    "surface_xyzfourier_dgammadash1dash1_by_dcoeff",
    "surface_xyzfourier_dgammadash1dash2_by_dcoeff",
    "surface_xyzfourier_dgammadash2dash2_by_dcoeff",
    "surface_xyzfourier_dnormal_by_dcoeff",
    "surface_xyzfourier_d2normal_by_dcoeffdcoeff",
    "surface_xyzfourier_dunitnormal_by_dcoeff",
    "surface_xyzfourier_darea_by_dcoeff",
    "surface_xyzfourier_d2area_by_dcoeffdcoeff",
    "surface_xyzfourier_dvolume_by_dcoeff",
    "surface_xyzfourier_d2volume_by_dcoeffdcoeff",
]


_TWO_PI_HOST = np.float64(2.0 * np.pi)
_ONE_HOST = np.float64(1.0)
_HALF_HOST = np.float64(0.5)
_BASIS_SELECTORS3_HOST = np.eye(3, dtype=np.float64)


def _two_pi(reference):
    return _as_runtime_float64(_TWO_PI_HOST, reference=reference)


def _one(reference):
    return _as_runtime_float64(_ONE_HOST, reference=reference)


def _half(reference):
    return _as_runtime_float64(_HALF_HOST, reference=reference)


def _basis_selector(index: int, *, reference):
    return _as_runtime_float64(_BASIS_SELECTORS3_HOST[int(index)], reference=reference)


def _mode_range(start, stop):
    return _as_jax_float64(np.arange(start, stop, dtype=np.float64))


def _selector_matrix(size, positions):
    matrix = np.zeros((len(positions), size), dtype=np.float64)
    if positions:
        matrix[np.arange(len(positions)), positions] = 1.0
    return _as_jax_float64(matrix)


def _slice_indices(start: int, size: int):
    return _as_jax_int32(start) + jnp.arange(size, dtype=jnp.int32)


_SCATTER_SET_DIMS_1D = lax.ScatterDimensionNumbers(
    update_window_dims=(),
    inserted_window_dims=(0,),
    scatter_dims_to_operand_dims=(0,),
    operand_batching_dims=(),
    scatter_indices_batching_dims=(),
)


# ---------------------------------------------------------------------------
# Basis matrix construction
# ---------------------------------------------------------------------------


def build_theta_basis(quadpoints_theta, mpol):
    """Build theta basis ``W`` and its derivative ``dW``.

    Args:
        quadpoints_theta: (ntheta,) array in [0, 1).
        mpol: maximum poloidal mode number.

    Returns:
        W:  (ntheta, 2*mpol+1) basis values.
        dW: (ntheta, 2*mpol+1) derivatives d/d(quadpoints_theta).
    """
    quadpoints_theta = _as_jax_float64(quadpoints_theta)
    two_pi = _two_pi(quadpoints_theta)
    theta = two_pi * quadpoints_theta  # (ntheta,)

    m_cos = _mode_range(0, mpol + 1)  # [0 .. mpol]
    m_sin = _mode_range(1, mpol + 1)  # [1 .. mpol]

    arg_cos = m_cos[None, :] * theta[:, None]  # (ntheta, mpol+1)
    arg_sin = m_sin[None, :] * theta[:, None]  # (ntheta, mpol)

    W = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)

    # d/d(quadpoints_theta) = d/dθ_param  (chain rule: dθ/dθ_param = 2π)
    dW = jnp.concatenate(
        [
            -m_cos[None, :] * two_pi * jnp.sin(arg_cos),
            m_sin[None, :] * two_pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )

    return W, dW


def _build_theta_basis_with_second(quadpoints_theta, mpol):
    quadpoints_theta = _as_jax_float64(quadpoints_theta)
    two_pi = _two_pi(quadpoints_theta)
    theta = two_pi * quadpoints_theta

    m_cos = _mode_range(0, mpol + 1)
    m_sin = _mode_range(1, mpol + 1)

    arg_cos = m_cos[None, :] * theta[:, None]
    arg_sin = m_sin[None, :] * theta[:, None]
    cos_factor = (m_cos[None, :] * two_pi) ** 2
    sin_factor = (m_sin[None, :] * two_pi) ** 2

    W = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)
    dW = jnp.concatenate(
        [
            -m_cos[None, :] * two_pi * jnp.sin(arg_cos),
            m_sin[None, :] * two_pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )
    ddW = jnp.concatenate(
        [
            -cos_factor * jnp.cos(arg_cos),
            -sin_factor * jnp.sin(arg_sin),
        ],
        axis=1,
    )

    return W, dW, ddW


def build_phi_basis(quadpoints_phi, ntor, nfp):
    """Build phi basis ``V`` and its derivative ``dV``.

    Args:
        quadpoints_phi: (nphi,) array (typically in [0, 1/nfp)).
        ntor: maximum toroidal mode number.
        nfp: number of field periods.

    Returns:
        V:  (nphi, 2*ntor+1) basis values.
        dV: (nphi, 2*ntor+1) derivatives d/d(quadpoints_phi).
    """
    quadpoints_phi = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi)
    phi = two_pi * quadpoints_phi  # (nphi,)

    # frequencies: [0, nfp, 2*nfp, …, ntor*nfp]
    nfp_scale = _as_jax_float64(nfp)
    n_cos = _mode_range(0, ntor + 1) * nfp_scale
    n_sin = _mode_range(1, ntor + 1) * nfp_scale

    arg_cos = n_cos[None, :] * phi[:, None]  # (nphi, ntor+1)
    arg_sin = n_sin[None, :] * phi[:, None]  # (nphi, ntor)

    V = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)

    dV = jnp.concatenate(
        [
            -n_cos[None, :] * two_pi * jnp.sin(arg_cos),
            n_sin[None, :] * two_pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )

    return V, dV


def _build_phi_basis_with_second(quadpoints_phi, ntor, nfp):
    quadpoints_phi = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi)
    phi = two_pi * quadpoints_phi

    nfp_scale = _as_jax_float64(nfp)
    n_cos = _mode_range(0, ntor + 1) * nfp_scale
    n_sin = _mode_range(1, ntor + 1) * nfp_scale

    arg_cos = n_cos[None, :] * phi[:, None]
    arg_sin = n_sin[None, :] * phi[:, None]
    cos_factor = (n_cos[None, :] * two_pi) ** 2
    sin_factor = (n_sin[None, :] * two_pi) ** 2

    V = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)
    dV = jnp.concatenate(
        [
            -n_cos[None, :] * two_pi * jnp.sin(arg_cos),
            n_sin[None, :] * two_pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )
    ddV = jnp.concatenate(
        [
            -cos_factor * jnp.cos(arg_cos),
            -sin_factor * jnp.sin(arg_sin),
        ],
        axis=1,
    )

    return V, dV, ddV


# ---------------------------------------------------------------------------
# Surface evaluation primitives
# ---------------------------------------------------------------------------


def _eval_hat(V, W, coeffs):
    """Evaluate Σ_ij c_ij · w_i(θ) · v_j(φ)  →  (nphi, ntheta).

    Uses two matrix multiplications:  V @ coeffs^T @ W^T.
    """
    # V: (nphi, 2*ntor+1),  coeffs: (2*mpol+1, 2*ntor+1),  W: (ntheta, 2*mpol+1)
    # Result: (nphi, ntheta)
    return (V @ coeffs.T) @ W.T


def _eval_hat_paired(V, W, coeffs):
    return jnp.sum((V @ coeffs.T) * W, axis=1)


# ---------------------------------------------------------------------------
# BC enforcer for SurfaceXYZTensorFourier ``clamped_dims`` (CPU parity helper)
# ---------------------------------------------------------------------------
#
# The C++ ``SurfaceXYZTensorFourier`` multiplies basis functions on the
# ``(m <= mpol, n <= ntor)`` cos-cos block by
# ``E(phi, theta) = sin(nfp*phi/2)^2 + sin(theta/2)^2`` whenever
# ``clamped_dims[dim]`` is true for that Cartesian component (see
# ``src/simsoptpp/surfacexyztensorfourier.h:903-913``). The JAX kernel
# computes the unclamped hat first, then adds a correction term
# ``hat_block * (E - 1)`` for each clamped dim, where ``hat_block`` is
# the sub-evaluation over the cos-cos coefficient quadrant only.


def _bc_enforcer_angles(quadpoints_phi, quadpoints_theta, nfp):
    qp = _as_jax_float64(quadpoints_phi)
    qt = _as_jax_float64(quadpoints_theta)
    two_pi_phi = _two_pi(qp)
    two_pi_theta = _two_pi(qt)
    nfp_f = _as_jax_float64(nfp)
    phi_arg = nfp_f * (two_pi_phi * qp) * _half(qp)
    theta_arg = (two_pi_theta * qt) * _half(qt)
    return phi_arg, theta_arg, nfp_f, two_pi_phi, two_pi_theta


def _bc_enforcer_grid(quadpoints_phi, quadpoints_theta, nfp):
    """Return ``E(phi, theta) = sin(nfp*phi/2)^2 + sin(theta/2)^2``.

    Shape: ``(nphi, ntheta)``. The arguments are the C++
    quadpoints_phi/theta in [0, 1); the function uses
    ``phi = 2*pi*quadpoints_phi`` internally to match the
    ``cache_enforcer`` build at
    ``src/simsoptpp/surfacexyztensorfourier.h:889-898``.
    """
    phi_arg, theta_arg, _, _, _ = _bc_enforcer_angles(
        quadpoints_phi, quadpoints_theta, nfp
    )
    sin_phi_half = jnp.sin(phi_arg)
    sin_theta_half = jnp.sin(theta_arg)
    return (sin_phi_half * sin_phi_half)[:, None] + (sin_theta_half * sin_theta_half)[
        None, :
    ]


def _bc_enforcer_grid_with_derivatives(quadpoints_phi, quadpoints_theta, nfp):
    """Return the BC enforcer plus derivatives w.r.t. normalized phi/theta."""
    phi_arg, theta_arg, nfp_f, two_pi_phi, two_pi_theta = _bc_enforcer_angles(
        quadpoints_phi, quadpoints_theta, nfp
    )
    sin_phi_half = jnp.sin(phi_arg)
    sin_theta_half = jnp.sin(theta_arg)
    enforcer = (sin_phi_half * sin_phi_half)[:, None] + (
        sin_theta_half * sin_theta_half
    )[None, :]

    dphi_term = jnp.sin(phi_arg + phi_arg) * nfp_f * two_pi_phi * _half(phi_arg)
    dtheta_term = jnp.sin(theta_arg + theta_arg) * two_pi_theta * _half(theta_arg)
    zero_phi = sin_phi_half - sin_phi_half
    zero_theta = sin_theta_half - sin_theta_half
    d_enforcer_dphi = dphi_term[:, None] + zero_theta[None, :]
    d_enforcer_dtheta = zero_phi[:, None] + dtheta_term[None, :]
    return enforcer, d_enforcer_dphi, d_enforcer_dtheta


def _bc_enforcer_grid_lin(quadpoints_phi, quadpoints_theta, nfp):
    """Paired-point variant of :func:`_bc_enforcer_grid`. Shape: ``(npairs,)``."""
    phi_arg, theta_arg, _, _, _ = _bc_enforcer_angles(
        _as_jax_float64(quadpoints_phi).reshape(-1),
        _as_jax_float64(quadpoints_theta).reshape(-1),
        nfp,
    )
    sin_phi_half = jnp.sin(phi_arg)
    sin_theta_half = jnp.sin(theta_arg)
    return sin_phi_half * sin_phi_half + sin_theta_half * sin_theta_half


def _cos_cos_block_mask(coeffs, mpol, ntor):
    mask = np.zeros(tuple(int(dim) for dim in coeffs.shape), dtype=np.float64)
    mask[: int(mpol) + 1, : int(ntor) + 1] = 1.0
    return _as_runtime_float64(mask, reference=coeffs)


def _eval_hat_block(V, W, coeffs, mpol, ntor):
    """Sub-evaluation over the ``(m <= mpol, n <= ntor)`` cos-cos block.

    The cos-cos block sits in the first ``(mpol + 1, ntor + 1)``
    sub-matrix of ``coeffs`` and corresponds to the first ``mpol + 1``
    columns of ``W`` and the first ``ntor + 1`` columns of ``V``.
    """
    return _eval_hat(V, W, coeffs * _cos_cos_block_mask(coeffs, mpol, ntor))


def _eval_hat_block_paired(V, W, coeffs, mpol, ntor):
    return _eval_hat_paired(V, W, coeffs * _cos_cos_block_mask(coeffs, mpol, ntor))


def _normalize_clamped_dims(clamped_dims):
    """Validate the ``clamped_dims`` argument and coerce to a bool tuple."""
    flags = tuple(bool(flag) for flag in clamped_dims)
    if len(flags) != 3:
        raise ValueError(
            "clamped_dims must have exactly 3 boolean flags (x, y, z); "
            f"got length {len(flags)}"
        )
    return flags


def _apply_clamped_correction(hats, coeffs, clamped_dims, correction, eval_block):
    return tuple(
        hat + eval_block(coeff) * correction if clamped else hat
        for hat, coeff, clamped in zip(hats, coeffs, clamped_dims)
    )


def _hats_with_clamping(
    V,
    W,
    xc,
    yc,
    zc,
    *,
    quadpoints_phi,
    quadpoints_theta,
    nfp,
    mpol,
    ntor,
    clamped_dims,
):
    """Return (xhat, yhat, zhat) on the dense grid with BC enforcer applied."""
    hats = (_eval_hat(V, W, xc), _eval_hat(V, W, yc), _eval_hat(V, W, zc))
    if not any(clamped_dims):
        return hats
    enforcer = _bc_enforcer_grid(quadpoints_phi, quadpoints_theta, nfp)
    return _apply_clamped_correction(
        hats,
        (xc, yc, zc),
        clamped_dims,
        enforcer - _one(enforcer),
        lambda coeff: _eval_hat_block(V, W, coeff, mpol, ntor),
    )


def _hats_with_clamping_paired(
    V,
    W,
    xc,
    yc,
    zc,
    *,
    quadpoints_phi,
    quadpoints_theta,
    nfp,
    mpol,
    ntor,
    clamped_dims,
):
    """Paired-point version of :func:`_hats_with_clamping`."""
    hats = (
        _eval_hat_paired(V, W, xc),
        _eval_hat_paired(V, W, yc),
        _eval_hat_paired(V, W, zc),
    )
    if not any(clamped_dims):
        return hats
    enforcer = _bc_enforcer_grid_lin(quadpoints_phi, quadpoints_theta, nfp)
    return _apply_clamped_correction(
        hats,
        (xc, yc, zc),
        clamped_dims,
        enforcer - _one(enforcer),
        lambda coeff: _eval_hat_block_paired(V, W, coeff, mpol, ntor),
    )


def _rotate_hat_components(quadpoints_phi, radial, toroidal):
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]
    return radial * cphi - toroidal * sphi, radial * sphi + toroidal * cphi


def _rotate_hat_components_lin(quadpoints_phi, radial, toroidal):
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)
    sphi = jnp.sin(phi_angle)
    return radial * cphi - toroidal * sphi, radial * sphi + toroidal * cphi


def surface_gamma(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate surface Cartesian coordinates on the quadrature grid.

    Args:
        quadpoints_phi:   (nphi,)
        quadpoints_theta: (ntheta,)
        xc: (2*mpol+1, 2*ntor+1) coefficients for x̂.
        yc: (2*mpol+1, 2*ntor+1) coefficients for ŷ.
        zc: (2*mpol+1, 2*ntor+1) coefficients for z.
        mpol, ntor, nfp: integers (static).
        clamped_dims: 3-tuple of Python bools selecting which Cartesian
            components apply the C++ BC enforcer
            ``E(phi, theta) = sin(nfp*phi/2)^2 + sin(theta/2)^2`` on the
            ``(m <= mpol, n <= ntor)`` cos-cos coefficient block. Matches
            ``SurfaceXYZTensorFourier::apply_bc_enforcer`` at
            ``src/simsoptpp/surfacexyztensorfourier.h:903``.

    Returns:
        gamma: (nphi, ntheta, 3)  Cartesian [x, y, z].
    """
    clamped_flags = _normalize_clamped_dims(clamped_dims)
    W, _ = build_theta_basis(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)

    xhat, yhat, z = _hats_with_clamping(
        V,
        W,
        xc,
        yc,
        zc,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        clamped_dims=clamped_flags,
    )

    x, y = _rotate_hat_components(quadpoints_phi, xhat, yhat)
    return jnp.stack([x, y, z], axis=-1)


def surface_gamma_lin(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate surface Cartesian coordinates at paired ``(phi[i], theta[i])``."""
    clamped_flags = _normalize_clamped_dims(clamped_dims)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)
    W, _ = build_theta_basis(quadpoints_theta_jax, mpol)
    V, _ = build_phi_basis(quadpoints_phi_jax, ntor, nfp)

    xhat, yhat, z = _hats_with_clamping_paired(
        V,
        W,
        xc,
        yc,
        zc,
        quadpoints_phi=quadpoints_phi_jax,
        quadpoints_theta=quadpoints_theta_jax,
        nfp=nfp,
        mpol=mpol,
        ntor=ntor,
        clamped_dims=clamped_flags,
    )

    x, y = _rotate_hat_components_lin(quadpoints_phi_jax, xhat, yhat)
    return jnp.stack([x, y, z], axis=-1)


def surface_gammadash1_lin(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/d(quadpoints_phi) at paired ``(phi[i], theta[i])``."""
    # Derivatives w.r.t. quadrature parameters are obtained by autodiff
    # through :func:`surface_gamma_lin`, which encodes the clamped BC
    # enforcer multiplicatively. This keeps the analytic and clamped
    # derivative formulas consistent with the C++ basis_fun_dphi/dtheta
    # product rule at ``src/simsoptpp/surfacexyztensorfourier.h:966-1010``.
    if any(_normalize_clamped_dims(clamped_dims)):

        def _eval_single(qp_scalar):
            qp_vec = jnp.atleast_1d(qp_scalar)
            return surface_gamma_lin(
                qp_vec,
                quadpoints_theta,
                xc,
                yc,
                zc,
                mpol,
                ntor,
                nfp,
                clamped_dims=clamped_dims,
            )[0]

        qp = _as_jax_float64(quadpoints_phi).reshape(-1)
        return jax.vmap(jax.jacfwd(_eval_single))(qp)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)
    W, _ = build_theta_basis(quadpoints_theta_jax, mpol)
    V, dV = build_phi_basis(quadpoints_phi_jax, ntor, nfp)

    xhat = _eval_hat_paired(V, W, xc)
    yhat = _eval_hat_paired(V, W, yc)
    dxhat_dphi = _eval_hat_paired(dV, W, xc)
    dyhat_dphi = _eval_hat_paired(dV, W, yc)
    dz_dphi = _eval_hat_paired(dV, W, zc)

    two_pi = _two_pi(quadpoints_phi_jax)
    radial = dxhat_dphi - two_pi * yhat
    toroidal = dyhat_dphi + two_pi * xhat
    dx, dy = _rotate_hat_components_lin(quadpoints_phi_jax, radial, toroidal)
    return jnp.stack([dx, dy, dz_dphi], axis=-1)


def surface_gammadash2_lin(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/d(quadpoints_theta) at paired ``(phi[i], theta[i])``."""
    if any(_normalize_clamped_dims(clamped_dims)):

        def _eval_single(qt_scalar):
            qt_vec = jnp.atleast_1d(qt_scalar)
            return surface_gamma_lin(
                quadpoints_phi,
                qt_vec,
                xc,
                yc,
                zc,
                mpol,
                ntor,
                nfp,
                clamped_dims=clamped_dims,
            )[0]

        qt = _as_jax_float64(quadpoints_theta).reshape(-1)
        return jax.vmap(jax.jacfwd(_eval_single))(qt)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)
    _, dW = build_theta_basis(quadpoints_theta_jax, mpol)
    V, _ = build_phi_basis(quadpoints_phi_jax, ntor, nfp)

    dxhat_dtheta = _eval_hat_paired(V, dW, xc)
    dyhat_dtheta = _eval_hat_paired(V, dW, yc)
    dz_dtheta = _eval_hat_paired(V, dW, zc)

    dx, dy = _rotate_hat_components_lin(
        quadpoints_phi_jax,
        dxhat_dtheta,
        dyhat_dtheta,
    )
    return jnp.stack([dx, dy, dz_dtheta], axis=-1)


def _gammadash1_clamped(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    clamped_flags,
):
    """Differentiate :func:`surface_gamma` w.r.t. quadpoints_phi."""
    W, _ = build_theta_basis(quadpoints_theta, mpol)
    V, dV = build_phi_basis(quadpoints_phi, ntor, nfp)
    enforcer, d_enforcer_dphi, _ = _bc_enforcer_grid_with_derivatives(
        quadpoints_phi, quadpoints_theta, nfp
    )
    enforcer_correction = enforcer - _one(enforcer)
    cx, cy, cz = clamped_flags

    def _hat_and_dphi(coeffs, clamped_flag):
        hat = _eval_hat(V, W, coeffs)
        dhat_dphi = _eval_hat(dV, W, coeffs)
        if not clamped_flag:
            return hat, dhat_dphi
        block_hat = _eval_hat_block(V, W, coeffs, mpol, ntor)
        dblock_dphi = _eval_hat_block(dV, W, coeffs, mpol, ntor)
        return (
            hat + block_hat * enforcer_correction,
            dhat_dphi + dblock_dphi * enforcer_correction + block_hat * d_enforcer_dphi,
        )

    xhat, dxhat_dphi = _hat_and_dphi(xc, cx)
    yhat, dyhat_dphi = _hat_and_dphi(yc, cy)
    _, dz_dphi = _hat_and_dphi(zc, cz)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_angle = two_pi * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]

    dx = (
        dxhat_dphi * cphi
        - xhat * (two_pi * sphi)
        - dyhat_dphi * sphi
        - yhat * (two_pi * cphi)
    )
    dy = (
        dxhat_dphi * sphi
        + xhat * (two_pi * cphi)
        + dyhat_dphi * cphi
        - yhat * (two_pi * sphi)
    )
    return jnp.stack([dx, dy, dz_dphi], axis=-1)


def _gammadash2_clamped(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    clamped_flags,
):
    """Differentiate :func:`surface_gamma` w.r.t. quadpoints_theta (clamped path)."""
    W, dW = build_theta_basis(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)
    enforcer, _, d_enforcer_dtheta = _bc_enforcer_grid_with_derivatives(
        quadpoints_phi, quadpoints_theta, nfp
    )
    enforcer_correction = enforcer - _one(enforcer)
    cx, cy, cz = clamped_flags

    def _hat_dtheta(coeffs, clamped_flag):
        dhat_dtheta = _eval_hat(V, dW, coeffs)
        if not clamped_flag:
            return dhat_dtheta
        block_hat = _eval_hat_block(V, W, coeffs, mpol, ntor)
        dblock_dtheta = _eval_hat_block(V, dW, coeffs, mpol, ntor)
        return (
            dhat_dtheta
            + dblock_dtheta * enforcer_correction
            + block_hat * d_enforcer_dtheta
        )

    dxhat_dtheta = _hat_dtheta(xc, cx)
    dyhat_dtheta = _hat_dtheta(yc, cy)
    dz_dtheta = _hat_dtheta(zc, cz)
    dx, dy = _rotate_hat_components(quadpoints_phi, dxhat_dtheta, dyhat_dtheta)
    return jnp.stack([dx, dy, dz_dtheta], axis=-1)


def surface_gammadash1(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/d(quadpoints_phi) — the toroidal tangent vector.

    Returns:
        gammadash1: (nphi, ntheta, 3).
    """
    clamped_flags = _normalize_clamped_dims(clamped_dims)
    if any(clamped_flags):
        return _gammadash1_clamped(
            quadpoints_phi,
            quadpoints_theta,
            xc,
            yc,
            zc,
            mpol,
            ntor,
            nfp,
            clamped_flags,
        )
    W, _ = build_theta_basis(quadpoints_theta, mpol)
    V, dV = build_phi_basis(quadpoints_phi, ntor, nfp)

    xhat = _eval_hat(V, W, xc)
    yhat = _eval_hat(V, W, yc)

    dxhat_dphi = _eval_hat(dV, W, xc)
    dyhat_dphi = _eval_hat(dV, W, yc)
    dz_dphi = _eval_hat(dV, W, zc)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_angle = two_pi * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]

    # d/d(phi_param):  x = x̂·cos(2πφ) − ŷ·sin(2πφ)
    # dx/dφ = dx̂/dφ·cosφ − x̂·2π·sinφ − dŷ/dφ·sinφ − ŷ·2π·cosφ
    dx = (
        dxhat_dphi * cphi
        - xhat * (two_pi * sphi)
        - dyhat_dphi * sphi
        - yhat * (two_pi * cphi)
    )
    dy = (
        dxhat_dphi * sphi
        + xhat * (two_pi * cphi)
        + dyhat_dphi * cphi
        - yhat * (two_pi * sphi)
    )
    dz = dz_dphi

    return jnp.stack([dx, dy, dz], axis=-1)


def surface_gammadash2(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/d(quadpoints_theta) — the poloidal tangent vector.

    Returns:
        gammadash2: (nphi, ntheta, 3).
    """
    clamped_flags = _normalize_clamped_dims(clamped_dims)
    if any(clamped_flags):
        return _gammadash2_clamped(
            quadpoints_phi,
            quadpoints_theta,
            xc,
            yc,
            zc,
            mpol,
            ntor,
            nfp,
            clamped_flags,
        )
    _, dW = build_theta_basis(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)

    dxhat_dtheta = _eval_hat(V, dW, xc)
    dyhat_dtheta = _eval_hat(V, dW, yc)
    dz_dtheta = _eval_hat(V, dW, zc)

    dx, dy = _rotate_hat_components(quadpoints_phi, dxhat_dtheta, dyhat_dtheta)
    return jnp.stack([dx, dy, dz_dtheta], axis=-1)


def surface_gammadash1dash1(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/d(quadpoints_phi)²."""
    if any(_normalize_clamped_dims(clamped_dims)):

        def _eval_gd1(qp_full):
            return surface_gammadash1(
                qp_full,
                quadpoints_theta,
                xc,
                yc,
                zc,
                mpol,
                ntor,
                nfp,
                clamped_dims=clamped_dims,
            )

        qp = _as_jax_float64(quadpoints_phi)

        def _diag(k1):
            phi_slice = jnp.atleast_1d(qp[k1])
            jac = jax.jacfwd(lambda v: _eval_gd1(v))(phi_slice)
            # jac shape: (1, ntheta, 3, 1) -> (ntheta, 3)
            return jac[0, :, :, 0]

        return jax.vmap(_diag)(jnp.arange(qp.shape[0]))
    W, _ = build_theta_basis(quadpoints_theta, mpol)
    V, dV, ddV = _build_phi_basis_with_second(quadpoints_phi, ntor, nfp)

    xhat = _eval_hat(V, W, xc)
    yhat = _eval_hat(V, W, yc)
    dxhat_dphi = _eval_hat(dV, W, xc)
    dyhat_dphi = _eval_hat(dV, W, yc)
    d2xhat_dphi2 = _eval_hat(ddV, W, xc)
    d2yhat_dphi2 = _eval_hat(ddV, W, yc)
    d2z_dphi2 = _eval_hat(ddV, W, zc)

    two_pi = _two_pi(_as_jax_float64(quadpoints_phi))
    radial = d2xhat_dphi2 - 2.0 * two_pi * dyhat_dphi - two_pi**2 * xhat
    toroidal = d2yhat_dphi2 + 2.0 * two_pi * dxhat_dphi - two_pi**2 * yhat
    dx, dy = _rotate_hat_components(quadpoints_phi, radial, toroidal)
    return jnp.stack([dx, dy, d2z_dphi2], axis=-1)


def surface_gammadash1dash2(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/d(quadpoints_phi)d(quadpoints_theta)."""
    if any(_normalize_clamped_dims(clamped_dims)):

        def _eval_gd2(qp_full):
            return surface_gammadash2(
                qp_full,
                quadpoints_theta,
                xc,
                yc,
                zc,
                mpol,
                ntor,
                nfp,
                clamped_dims=clamped_dims,
            )

        qp = _as_jax_float64(quadpoints_phi)

        def _diag(k1):
            phi_slice = jnp.atleast_1d(qp[k1])
            jac = jax.jacfwd(lambda v: _eval_gd2(v))(phi_slice)
            # jac shape: (1, ntheta, 3, 1) -> (ntheta, 3)
            return jac[0, :, :, 0]

        return jax.vmap(_diag)(jnp.arange(qp.shape[0]))
    _, dW = build_theta_basis(quadpoints_theta, mpol)
    V, dV = build_phi_basis(quadpoints_phi, ntor, nfp)

    dxhat_dtheta = _eval_hat(V, dW, xc)
    dyhat_dtheta = _eval_hat(V, dW, yc)
    d2xhat_dphidtheta = _eval_hat(dV, dW, xc)
    d2yhat_dphidtheta = _eval_hat(dV, dW, yc)
    d2z_dphidtheta = _eval_hat(dV, dW, zc)

    two_pi = _two_pi(_as_jax_float64(quadpoints_phi))
    radial = d2xhat_dphidtheta - two_pi * dyhat_dtheta
    toroidal = d2yhat_dphidtheta + two_pi * dxhat_dtheta
    dx, dy = _rotate_hat_components(quadpoints_phi, radial, toroidal)
    return jnp.stack([dx, dy, d2z_dphidtheta], axis=-1)


def surface_gammadash2dash2(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/d(quadpoints_theta)²."""
    if any(_normalize_clamped_dims(clamped_dims)):

        def _eval_gd2(qt_full):
            return surface_gammadash2(
                quadpoints_phi,
                qt_full,
                xc,
                yc,
                zc,
                mpol,
                ntor,
                nfp,
                clamped_dims=clamped_dims,
            )

        qt = _as_jax_float64(quadpoints_theta)

        def _diag(k2):
            theta_slice = jnp.atleast_1d(qt[k2])
            jac = jax.jacfwd(lambda v: _eval_gd2(v))(theta_slice)
            # jac shape: (nphi, 1, 3, 1) -> (nphi, 3)
            return jac[:, 0, :, 0]

        return jnp.transpose(jax.vmap(_diag)(jnp.arange(qt.shape[0])), (1, 0, 2))
    _, _, ddW = _build_theta_basis_with_second(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)

    d2xhat_dtheta2 = _eval_hat(V, ddW, xc)
    d2yhat_dtheta2 = _eval_hat(V, ddW, yc)
    d2z_dtheta2 = _eval_hat(V, ddW, zc)

    dx, dy = _rotate_hat_components(
        quadpoints_phi,
        d2xhat_dtheta2,
        d2yhat_dtheta2,
    )
    return jnp.stack([dx, dy, d2z_dtheta2], axis=-1)


def surface_normal(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
    *,
    clamped_dims=(False, False, False),
):
    """Compute the (unnormalized) surface normal n = gammadash1 × gammadash2.

    Returns:
        normal: (nphi, ntheta, 3).
    """
    gd1 = surface_gammadash1(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )
    gd2 = surface_gammadash2(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )
    return jnp.cross(gd1, gd2)


def _unitnormal(normal):
    return _unit_vector3(normal)


# ---------------------------------------------------------------------------
# Coefficient-parametric evaluation (for autodiff w.r.t. dofs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stellsym DOF ↔ coefficient mapping
# ---------------------------------------------------------------------------


def _split_flat_to_xyzc(flat, mpol, ntor):
    """Split a flat super-vector into (xc, yc, zc) coefficient matrices.

    Args:
        flat: (3 * n_per_coord,) flat array.
        mpol, ntor: surface resolution.

    Returns:
        xc, yc, zc: each (2*mpol+1, 2*ntor+1).
    """
    n_per_coord = int((2 * mpol + 1) * (2 * ntor + 1))
    shape = (int(2 * mpol + 1), int(2 * ntor + 1))
    flat_jax = _as_jax_float64(flat)
    stacked = jnp.reshape(flat_jax, (3, n_per_coord))
    return (
        jnp.reshape(jnp.dot(_basis_selector(0, reference=stacked), stacked), shape),
        jnp.reshape(jnp.dot(_basis_selector(1, reference=stacked), stacked), shape),
        jnp.reshape(jnp.dot(_basis_selector(2, reference=stacked), stacked), shape),
    )


def dofs_to_xyzc(sdofs, scatter_indices, mpol, ntor):
    """Scatter surface DOFs into full ``(xc, yc, zc)`` coefficient matrices.

    JAX-traceable: supports autodiff through the scatter operation.

    Args:
        sdofs: (ndofs,) free surface DOFs.
        scatter_indices: (ndofs,) int array from :func:`stellsym_scatter_indices`.
        mpol, ntor: surface resolution.

    Returns:
        xc, yc, zc: each (2*mpol+1, 2*ntor+1).
    """
    sdofs_jax = _as_jax_float64(sdofs)
    scatter_operand = scatter_indices
    scatter_ndim = getattr(scatter_operand, "ndim", np.ndim(scatter_operand))
    if scatter_ndim == 2:
        flat = _as_jax_float64(scatter_operand) @ sdofs_jax
        return _split_flat_to_xyzc(flat, mpol, ntor)

    n_per_coord = int((2 * mpol + 1) * (2 * ntor + 1))
    scatter_indices_1d = _as_jax_int32(scatter_operand).reshape(-1, 1)
    flat = lax.scatter(
        _zeros(3 * n_per_coord, sdofs_jax.dtype),
        scatter_indices_1d,
        sdofs_jax,
        _SCATTER_SET_DIMS_1D,
        indices_are_sorted=True,
        unique_indices=True,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    return _split_flat_to_xyzc(flat, mpol, ntor)


def _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices):
    """Internal helper: unpack DOFs to (xc, yc, zc) for both stellsym modes."""
    if stellsym:
        if scatter_indices is None:
            raise ValueError(
                "scatter_indices required for stellsym=True. "
                "Precompute with stellsym_scatter_indices(mpol, ntor)."
            )
        return dofs_to_xyzc(dofs, scatter_indices, mpol, ntor)
    return _split_flat_to_xyzc(dofs, mpol, ntor)


def _scatter_surface_xyzfourier_dofs(
    dofs,
    mpol,
    ntor,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Unpack ``SurfaceXYZFourier`` DOFs into six coefficient matrices."""
    shape = (mpol + 1, 2 * ntor + 1)
    n_per = shape[0] * shape[1]
    if scatter_indices is not None and coeff_template is not None:
        return _scatter_surface_xyzfourier_dofs_from_template(
            dofs,
            shape,
            scatter_indices,
            coeff_template,
        )

    cos_count = n_per - ntor
    sin_count = n_per - (ntor + 1)

    def _scatter_segment(source, start, count, fill_start):
        flat = _zeros(n_per, source.dtype)
        values = jnp.take(source, _slice_indices(start, count), axis=0)
        return flat.at[_slice_indices(fill_start, count)].set(values)

    if stellsym:
        xc = _scatter_segment(dofs, 0, cos_count, ntor).reshape(shape)
        ys = _scatter_segment(dofs, cos_count, sin_count, ntor + 1).reshape(shape)
        zs = _scatter_segment(
            dofs,
            cos_count + sin_count,
            sin_count,
            ntor + 1,
        ).reshape(shape)
        zeros = _zeros(shape, dofs.dtype)
        return xc, zeros, zeros, ys, zeros, zs

    offset = 0
    xc = _scatter_segment(dofs, offset, cos_count, ntor).reshape(shape)
    offset += cos_count
    xs = _scatter_segment(dofs, offset, sin_count, ntor + 1).reshape(shape)
    offset += sin_count
    yc = _scatter_segment(dofs, offset, cos_count, ntor).reshape(shape)
    offset += cos_count
    ys = _scatter_segment(dofs, offset, sin_count, ntor + 1).reshape(shape)
    offset += sin_count
    zc = _scatter_segment(dofs, offset, cos_count, ntor).reshape(shape)
    offset += cos_count
    zs = _scatter_segment(dofs, offset, sin_count, ntor + 1).reshape(shape)
    return xc, xs, yc, ys, zc, zs


def _scatter_surface_xyzfourier_dofs_from_template(
    dofs,
    shape,
    scatter_indices,
    coeff_template,
):
    coeffs = lax.scatter(
        coeff_template,
        _as_jax_int32(scatter_indices).reshape(-1, 1),
        _as_jax_float64(dofs),
        _SCATTER_SET_DIMS_1D,
        indices_are_sorted=True,
        unique_indices=True,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )
    return tuple(jnp.reshape(coeffs, (6, *shape)))


def _surface_xyzfourier_basis(quadpoints_phi, quadpoints_theta, mpol, ntor, nfp):
    """Return ``SurfaceXYZFourier`` phase terms and mode indices."""
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    theta = _two_pi(quadpoints_theta_jax) * quadpoints_theta_jax
    phi = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    m = _mode_range(0, mpol + 1)
    n = _mode_range(-ntor, ntor + 1) * _as_jax_float64(nfp)

    angle = (
        theta[None, :, None, None] * m[None, None, :, None]
        - phi[:, None, None, None] * n[None, None, None, :]
    )
    return jnp.cos(angle), jnp.sin(angle), m, n


def _surface_xyzfourier_basis_paired(
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
):
    """Return ``SurfaceXYZFourier`` phase terms at paired points."""
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    theta = _two_pi(quadpoints_theta_jax) * quadpoints_theta_jax
    phi = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    m = _mode_range(0, mpol + 1)
    n = _mode_range(-ntor, ntor + 1) * _as_jax_float64(nfp)

    angle = theta[:, None, None] * m[None, :, None]
    angle -= phi[:, None, None] * n[None, None, :]
    return jnp.cos(angle), jnp.sin(angle), m, n


def _surface_xyzfourier_hat(cos_coeffs, sin_coeffs, cos_angle, sin_angle):
    coeff_term = cos_coeffs[None, None, :, :] * cos_angle
    coeff_term += sin_coeffs[None, None, :, :] * sin_angle
    return jnp.sum(coeff_term, axis=(2, 3))


def _surface_xyzfourier_hat_paired(cos_coeffs, sin_coeffs, cos_angle, sin_angle):
    coeff_term = cos_coeffs[None, :, :] * cos_angle
    coeff_term += sin_coeffs[None, :, :] * sin_angle
    return jnp.sum(coeff_term, axis=(1, 2))


def _surface_xyzfourier_derivative_hat(
    cos_coeffs,
    sin_coeffs,
    mode_factor,
    cos_angle,
    sin_angle,
):
    return _surface_xyzfourier_hat(
        cos_coeffs,
        sin_coeffs,
        -mode_factor * sin_angle,
        mode_factor * cos_angle,
    )


def _surface_xyzfourier_derivative_hat_paired(
    cos_coeffs,
    sin_coeffs,
    mode_factor,
    cos_angle,
    sin_angle,
):
    return _surface_xyzfourier_hat_paired(
        cos_coeffs,
        sin_coeffs,
        -mode_factor * sin_angle,
        mode_factor * cos_angle,
    )


def _surface_xyzfourier_mixed_derivative_hat(
    cos_coeffs,
    sin_coeffs,
    phi_factor,
    theta_factor,
    phi_order,
    theta_order,
    cos_angle,
    sin_angle,
):
    factor = (phi_factor**phi_order) * (theta_factor**theta_order)
    phase = (phi_order + theta_order) % 4
    if phase == 0:
        return _surface_xyzfourier_hat(
            cos_coeffs,
            sin_coeffs,
            factor * cos_angle,
            factor * sin_angle,
        )
    if phase == 1:
        return _surface_xyzfourier_hat(
            cos_coeffs,
            sin_coeffs,
            -factor * sin_angle,
            factor * cos_angle,
        )
    if phase == 2:
        return _surface_xyzfourier_hat(
            cos_coeffs,
            sin_coeffs,
            -factor * cos_angle,
            -factor * sin_angle,
        )
    return _surface_xyzfourier_hat(
        cos_coeffs,
        sin_coeffs,
        factor * sin_angle,
        -factor * cos_angle,
    )


def _surface_xyzfourier_rotate(phi_angle, xhat, yhat):
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]
    return xhat * cphi - yhat * sphi, xhat * sphi + yhat * cphi


def _surface_xyzfourier_rotate_lin(phi_angle, xhat, yhat):
    cphi = jnp.cos(phi_angle)
    sphi = jnp.sin(phi_angle)
    return xhat * cphi - yhat * sphi, xhat * sphi + yhat * cphi


def _paired_gamma_derivative(eval_gamma_at, quadpoints_phi, quadpoints_theta, *, phi_order, theta_order):
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)

    def _derivative_at(phi_value, theta_value):
        fn = eval_gamma_at
        for _ in range(phi_order):
            previous = fn
            fn = lambda local_phi, local_theta, previous=previous: jax.jacfwd(
                lambda phi_arg: previous(phi_arg, local_theta)
            )(local_phi)
        for _ in range(theta_order):
            previous = fn
            fn = lambda local_phi, local_theta, previous=previous: jax.jacfwd(
                lambda theta_arg: previous(local_phi, theta_arg)
            )(local_theta)
        return fn(phi_value, theta_value)

    return jax.vmap(_derivative_at)(quadpoints_phi_jax, quadpoints_theta_jax)


def surface_xyzfourier_gamma_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gamma()`` as a pure JAX function."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, _m, _n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )

    xhat = _surface_xyzfourier_hat(xc, xs, cos_angle, sin_angle)
    yhat = _surface_xyzfourier_hat(yc, ys, cos_angle, sin_angle)
    z = _surface_xyzfourier_hat(zc, zs, cos_angle, sin_angle)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    x, y = _surface_xyzfourier_rotate(phi_angle, xhat, yhat)
    return jnp.stack([x, y, z], axis=-1)


def surface_xyzfourier_gamma_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gamma_lin()`` as a pure JAX function."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, _m, _n = _surface_xyzfourier_basis_paired(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )

    xhat = _surface_xyzfourier_hat_paired(xc, xs, cos_angle, sin_angle)
    yhat = _surface_xyzfourier_hat_paired(yc, ys, cos_angle, sin_angle)
    z = _surface_xyzfourier_hat_paired(zc, zs, cos_angle, sin_angle)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    x, y = _surface_xyzfourier_rotate_lin(phi_angle, xhat, yhat)
    return jnp.stack([x, y, z], axis=-1)


def surface_xyzfourier_gammadash1_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash1()`` as a pure JAX function."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, _m, n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    n_factor = two_pi * n[None, None, None, :]

    xhat = _surface_xyzfourier_hat(xc, xs, cos_angle, sin_angle)
    yhat = _surface_xyzfourier_hat(yc, ys, cos_angle, sin_angle)
    dxhat_dphi = _surface_xyzfourier_derivative_hat(
        xc, xs, -n_factor, cos_angle, sin_angle
    )
    dyhat_dphi = _surface_xyzfourier_derivative_hat(
        yc, ys, -n_factor, cos_angle, sin_angle
    )
    dz_dphi = _surface_xyzfourier_derivative_hat(
        zc, zs, -n_factor, cos_angle, sin_angle
    )

    phi_angle = two_pi * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]

    dx = dxhat_dphi * cphi - xhat * (two_pi * sphi)
    dx -= dyhat_dphi * sphi + yhat * (two_pi * cphi)
    dy = dxhat_dphi * sphi + xhat * (two_pi * cphi)
    dy += dyhat_dphi * cphi - yhat * (two_pi * sphi)

    return jnp.stack([dx, dy, dz_dphi], axis=-1)


def surface_xyzfourier_gammadash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash1_lin()`` as pure JAX."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, _m, n = _surface_xyzfourier_basis_paired(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    two_pi = _two_pi(quadpoints_phi_jax)
    n_factor = two_pi * n[None, None, :]

    xhat = _surface_xyzfourier_hat_paired(xc, xs, cos_angle, sin_angle)
    yhat = _surface_xyzfourier_hat_paired(yc, ys, cos_angle, sin_angle)
    dxhat_dphi = _surface_xyzfourier_derivative_hat_paired(
        xc, xs, -n_factor, cos_angle, sin_angle
    )
    dyhat_dphi = _surface_xyzfourier_derivative_hat_paired(
        yc, ys, -n_factor, cos_angle, sin_angle
    )
    dz_dphi = _surface_xyzfourier_derivative_hat_paired(
        zc, zs, -n_factor, cos_angle, sin_angle
    )

    radial = dxhat_dphi - two_pi * yhat
    toroidal = dyhat_dphi + two_pi * xhat
    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate_lin(phi_angle, radial, toroidal)
    return jnp.stack([dx, dy, dz_dphi], axis=-1)


def surface_xyzfourier_gammadash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash2()`` as a pure JAX function."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, m, _n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    m_factor = two_pi * m[None, None, :, None]

    dxhat_dtheta = _surface_xyzfourier_derivative_hat(
        xc, xs, m_factor, cos_angle, sin_angle
    )
    dyhat_dtheta = _surface_xyzfourier_derivative_hat(
        yc, ys, m_factor, cos_angle, sin_angle
    )
    dz_dtheta = _surface_xyzfourier_derivative_hat(
        zc, zs, m_factor, cos_angle, sin_angle
    )

    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate(phi_angle, dxhat_dtheta, dyhat_dtheta)
    return jnp.stack([dx, dy, dz_dtheta], axis=-1)


def surface_xyzfourier_gammadash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash2_lin()`` as pure JAX."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, m, _n = _surface_xyzfourier_basis_paired(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    two_pi = _two_pi(quadpoints_phi_jax)
    m_factor = two_pi * m[None, :, None]

    dxhat_dtheta = _surface_xyzfourier_derivative_hat_paired(
        xc, xs, m_factor, cos_angle, sin_angle
    )
    dyhat_dtheta = _surface_xyzfourier_derivative_hat_paired(
        yc, ys, m_factor, cos_angle, sin_angle
    )
    dz_dtheta = _surface_xyzfourier_derivative_hat_paired(
        zc, zs, m_factor, cos_angle, sin_angle
    )

    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate_lin(phi_angle, dxhat_dtheta, dyhat_dtheta)
    return jnp.stack([dx, dy, dz_dtheta], axis=-1)


def surface_xyzfourier_gammadash1dash1_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash1dash1()`` as pure JAX."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, _m, n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_factor = -two_pi * n[None, None, None, :]
    theta_factor = jnp.ones_like(phi_factor)

    xhat = _surface_xyzfourier_hat(xc, xs, cos_angle, sin_angle)
    yhat = _surface_xyzfourier_hat(yc, ys, cos_angle, sin_angle)
    dxhat_dphi = _surface_xyzfourier_mixed_derivative_hat(
        xc, xs, phi_factor, theta_factor, 1, 0, cos_angle, sin_angle
    )
    dyhat_dphi = _surface_xyzfourier_mixed_derivative_hat(
        yc, ys, phi_factor, theta_factor, 1, 0, cos_angle, sin_angle
    )
    d2xhat_dphi2 = _surface_xyzfourier_mixed_derivative_hat(
        xc, xs, phi_factor, theta_factor, 2, 0, cos_angle, sin_angle
    )
    d2yhat_dphi2 = _surface_xyzfourier_mixed_derivative_hat(
        yc, ys, phi_factor, theta_factor, 2, 0, cos_angle, sin_angle
    )
    d2z_dphi2 = _surface_xyzfourier_mixed_derivative_hat(
        zc, zs, phi_factor, theta_factor, 2, 0, cos_angle, sin_angle
    )

    radial = d2xhat_dphi2 - 2.0 * two_pi * dyhat_dphi - two_pi**2 * xhat
    toroidal = d2yhat_dphi2 + 2.0 * two_pi * dxhat_dphi - two_pi**2 * yhat
    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate(phi_angle, radial, toroidal)
    return jnp.stack([dx, dy, d2z_dphi2], axis=-1)


def surface_xyzfourier_gammadash1dash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash1dash2()`` as pure JAX."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, m, n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_factor = -two_pi * n[None, None, None, :]
    theta_factor = two_pi * m[None, None, :, None]

    dxhat_dtheta = _surface_xyzfourier_mixed_derivative_hat(
        xc, xs, phi_factor, theta_factor, 0, 1, cos_angle, sin_angle
    )
    dyhat_dtheta = _surface_xyzfourier_mixed_derivative_hat(
        yc, ys, phi_factor, theta_factor, 0, 1, cos_angle, sin_angle
    )
    d2xhat_dphidtheta = _surface_xyzfourier_mixed_derivative_hat(
        xc, xs, phi_factor, theta_factor, 1, 1, cos_angle, sin_angle
    )
    d2yhat_dphidtheta = _surface_xyzfourier_mixed_derivative_hat(
        yc, ys, phi_factor, theta_factor, 1, 1, cos_angle, sin_angle
    )
    d2z_dphidtheta = _surface_xyzfourier_mixed_derivative_hat(
        zc, zs, phi_factor, theta_factor, 1, 1, cos_angle, sin_angle
    )

    radial = d2xhat_dphidtheta - two_pi * dyhat_dtheta
    toroidal = d2yhat_dphidtheta + two_pi * dxhat_dtheta
    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate(phi_angle, radial, toroidal)
    return jnp.stack([dx, dy, d2z_dphidtheta], axis=-1)


def surface_xyzfourier_gammadash2dash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.gammadash2dash2()`` as pure JAX."""
    xc, xs, yc, ys, zc, zs = _scatter_surface_xyzfourier_dofs(
        dofs,
        mpol,
        ntor,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    cos_angle, sin_angle, m, _n = _surface_xyzfourier_basis(
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
    )
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    two_pi = _two_pi(quadpoints_phi_jax)
    phi_factor = jnp.ones((1, 1, 1, 1), dtype=cos_angle.dtype)
    theta_factor = two_pi * m[None, None, :, None]

    d2xhat_dtheta2 = _surface_xyzfourier_mixed_derivative_hat(
        xc, xs, phi_factor, theta_factor, 0, 2, cos_angle, sin_angle
    )
    d2yhat_dtheta2 = _surface_xyzfourier_mixed_derivative_hat(
        yc, ys, phi_factor, theta_factor, 0, 2, cos_angle, sin_angle
    )
    d2z_dtheta2 = _surface_xyzfourier_mixed_derivative_hat(
        zc, zs, phi_factor, theta_factor, 0, 2, cos_angle, sin_angle
    )

    phi_angle = two_pi * quadpoints_phi_jax
    dx, dy = _surface_xyzfourier_rotate(phi_angle, d2xhat_dtheta2, d2yhat_dtheta2)
    return jnp.stack([dx, dy, d2z_dtheta2], axis=-1)


def _surface_xyzfourier_paired_derivative_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    coeff_template,
    *,
    phi_order,
    theta_order,
):
    def _eval_gamma_at(phi_value, theta_value):
        return surface_xyzfourier_gamma_lin_from_dofs(
            dofs,
            jnp.reshape(phi_value, (1,)),
            jnp.reshape(theta_value, (1,)),
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )[0]

    return _paired_gamma_derivative(
        _eval_gamma_at,
        quadpoints_phi,
        quadpoints_theta,
        phi_order=phi_order,
        theta_order=theta_order,
    )


def surface_xyzfourier_gammadash1dash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=2,
        theta_order=0,
    )


def surface_xyzfourier_gammadash1dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=1,
        theta_order=1,
    )


def surface_xyzfourier_gammadash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=0,
        theta_order=2,
    )


def surface_xyzfourier_gammadash1dash1dash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=3,
        theta_order=0,
    )


def surface_xyzfourier_gammadash1dash1dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=2,
        theta_order=1,
    )


def surface_xyzfourier_gammadash1dash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=1,
        theta_order=2,
    )


def surface_xyzfourier_gammadash2dash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    return _surface_xyzfourier_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
        phi_order=0,
        theta_order=3,
    )


def surface_xyzfourier_normal_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.normal()`` as pure JAX."""
    gd1 = surface_xyzfourier_gammadash1_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    gd2 = surface_xyzfourier_gammadash2_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    return jnp.cross(gd1, gd2)


def surface_xyzfourier_unitnormal_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.unitnormal()`` as pure JAX."""
    return _unitnormal(
        surface_xyzfourier_normal_from_dofs(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )
    )


def surface_xyzfourier_area_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.area()`` as pure JAX."""
    return surface_area(
        surface_xyzfourier_normal_from_dofs(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )
    )


def surface_xyzfourier_volume_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    coeff_template=None,
):
    """Evaluate ``SurfaceXYZFourier.volume()`` as pure JAX."""
    gamma = surface_xyzfourier_gamma_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    normal = surface_xyzfourier_normal_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        coeff_template,
    )
    return surface_volume(gamma, normal)


def surface_gamma_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate gamma as a pure function of the flat DOF vector.

    This is the entry point for JAX autodiff w.r.t. surface degrees of
    freedom: ``jax.grad(f)(dofs, ...)`` where ``f`` composes this function
    with downstream objectives.

    Args:
        dofs: flat DOF vector (free coefficients only if stellsym).
        quadpoints_phi, quadpoints_theta: quadrature grids.
        mpol, ntor, nfp: surface resolution and field periods.
        stellsym: whether stellarator symmetry is active.
        scatter_indices: precomputed from :func:`stellsym_scatter_indices`.
            Required when ``stellsym=True``.
        clamped_dims: optional 3-tuple of Python bools that mirrors the
            CPU ``SurfaceXYZTensorFourier.clamped_dims`` BC enforcer.

    Returns:
        gamma: (nphi, ntheta, 3) Cartesian positions.
    """
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gamma(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gamma_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate paired-point gamma as a pure function of the flat DOF vector."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gamma_lin(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash1_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/dφ as a pure function of DOFs (autodiff-compatible)."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate paired-point dγ/dφ as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1_lin(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate dγ/dθ as a pure function of DOFs (autodiff-compatible)."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate paired-point dγ/dθ as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2_lin(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash1dash1_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/dφ² as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1dash1(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash1dash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/dφdθ as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1dash2(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_gammadash2dash2_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate d²γ/dθ² as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2dash2(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def _surface_tensor_paired_derivative_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    *,
    clamped_dims,
    phi_order,
    theta_order,
):
    def _eval_gamma_at(phi_value, theta_value):
        return surface_gamma_lin_from_dofs(
            dofs,
            jnp.reshape(phi_value, (1,)),
            jnp.reshape(theta_value, (1,)),
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            clamped_dims=clamped_dims,
        )[0]

    return _paired_gamma_derivative(
        _eval_gamma_at,
        quadpoints_phi,
        quadpoints_theta,
        phi_order=phi_order,
        theta_order=theta_order,
    )


def surface_gammadash1dash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=2,
        theta_order=0,
    )


def surface_gammadash1dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=1,
        theta_order=1,
    )


def surface_gammadash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=0,
        theta_order=2,
    )


def surface_gammadash1dash1dash1_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=3,
        theta_order=0,
    )


def surface_gammadash1dash1dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=2,
        theta_order=1,
    )


def surface_gammadash1dash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=1,
        theta_order=2,
    )


def surface_gammadash2dash2dash2_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    return _surface_tensor_paired_derivative_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
        phi_order=0,
        theta_order=3,
    )


def surface_normal_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate unnormalized normal as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_normal(
        quadpoints_phi,
        quadpoints_theta,
        xc,
        yc,
        zc,
        mpol,
        ntor,
        nfp,
        clamped_dims=clamped_dims,
    )


def surface_unitnormal_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate unit normal as a pure function of DOFs."""
    return _unitnormal(
        surface_normal_from_dofs(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            clamped_dims=clamped_dims,
        )
    )


def surface_area_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate tensor-surface area as a pure function of DOFs."""
    return surface_area(
        surface_normal_from_dofs(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            clamped_dims=clamped_dims,
        )
    )


def surface_volume_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
    *,
    clamped_dims=(False, False, False),
):
    """Evaluate tensor-surface volume as a pure function of DOFs."""
    gamma = surface_gamma_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
    )
    normal = surface_normal_from_dofs(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
        clamped_dims=clamped_dims,
    )
    return surface_volume(gamma, normal)


# ---------------------------------------------------------------------------
# Surface coefficient Jacobians (M3)
# ---------------------------------------------------------------------------


def _dcoeff_jacobian(fn):
    """Build a surface coefficient Jacobian function from a ``_from_dofs`` evaluator."""

    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
    ):
        return jax.jacfwd(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
        )

    return wrapper


def _dcoeff_hessian(fn):
    """Build an explicit heavy coefficient Hessian from a ``_from_dofs`` evaluator."""

    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
    ):
        return jax.jacfwd(jax.jacfwd(fn, argnums=0), argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
        )

    return wrapper


dgamma_by_dcoeff = _dcoeff_jacobian(surface_gamma_from_dofs)
dgamma_by_dcoeff.__doc__ = """\
Jacobian of gamma w.r.t. surface DOFs via forward-mode autodiff.

Replaces ``sopp.SurfaceXYZTensorFourier.dgamma_by_dcoeff()``.

Returns:
    (nphi, ntheta, 3, ndofs) where ``result[i,j,l,k] = ∂γ_l(φ_i,θ_j)/∂dof_k``.
"""

dgammadash1_by_dcoeff = _dcoeff_jacobian(surface_gammadash1_from_dofs)
dgammadash1_by_dcoeff.__doc__ = """\
Jacobian of gammadash1 w.r.t. surface DOFs via forward-mode autodiff.

Replaces ``sopp.SurfaceXYZTensorFourier.dgammadash1_by_dcoeff()``.

Returns:
    (nphi, ntheta, 3, ndofs) where ``result[i,j,l,k] = ∂(∂γ/∂φ)_l(φ_i,θ_j)/∂dof_k``.
"""

dgammadash2_by_dcoeff = _dcoeff_jacobian(surface_gammadash2_from_dofs)
dgammadash2_by_dcoeff.__doc__ = """\
Jacobian of gammadash2 w.r.t. surface DOFs via forward-mode autodiff.

Replaces ``sopp.SurfaceXYZTensorFourier.dgammadash2_by_dcoeff()``.

Returns:
    (nphi, ntheta, 3, ndofs) where ``result[i,j,l,k] = ∂(∂γ/∂θ)_l(φ_i,θ_j)/∂dof_k``.
"""

dgammadash1dash1_by_dcoeff = _dcoeff_jacobian(surface_gammadash1dash1_from_dofs)
dgammadash1dash1_by_dcoeff.__doc__ = """\
Jacobian of gammadash1dash1 w.r.t. tensor-surface DOFs via forward-mode autodiff.
"""

dgammadash1dash2_by_dcoeff = _dcoeff_jacobian(surface_gammadash1dash2_from_dofs)
dgammadash1dash2_by_dcoeff.__doc__ = """\
Jacobian of gammadash1dash2 w.r.t. tensor-surface DOFs via forward-mode autodiff.
"""

dgammadash2dash2_by_dcoeff = _dcoeff_jacobian(surface_gammadash2dash2_from_dofs)
dgammadash2dash2_by_dcoeff.__doc__ = """\
Jacobian of gammadash2dash2 w.r.t. tensor-surface DOFs via forward-mode autodiff.
"""

dnormal_by_dcoeff = _dcoeff_jacobian(surface_normal_from_dofs)
dnormal_by_dcoeff.__doc__ = """\
Jacobian of tensor-surface normal w.r.t. surface DOFs via forward-mode autodiff.
"""

d2normal_by_dcoeffdcoeff = _dcoeff_hessian(surface_normal_from_dofs)
d2normal_by_dcoeffdcoeff.__doc__ = """\
Explicit heavy Hessian of tensor-surface normal w.r.t. surface DOFs.

Replaces ``sopp.SurfaceXYZTensorFourier.d2normal_by_dcoeffdcoeff()``.
"""

dunitnormal_by_dcoeff = _dcoeff_jacobian(surface_unitnormal_from_dofs)
dunitnormal_by_dcoeff.__doc__ = """\
Jacobian of tensor-surface unit normal w.r.t. surface DOFs via forward-mode autodiff.
"""


def _surface_scalar_grad(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
    ):
        return jax.grad(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
        )

    return wrapper


def _surface_scalar_hessian(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
    ):
        return jax.hessian(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
        )

    return wrapper


darea_by_dcoeff = _surface_scalar_grad(surface_area_from_dofs)
d2area_by_dcoeffdcoeff = _surface_scalar_hessian(surface_area_from_dofs)
dvolume_by_dcoeff = _surface_scalar_grad(surface_volume_from_dofs)
d2volume_by_dcoeffdcoeff = _surface_scalar_hessian(surface_volume_from_dofs)


def _surface_xyzfourier_dcoeff_jacobian(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
        coeff_template=None,
    ):
        return jax.jacfwd(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )

    return wrapper


def _surface_xyzfourier_dcoeff_hessian(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
        coeff_template=None,
    ):
        return jax.jacfwd(jax.jacfwd(fn, argnums=0), argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )

    return wrapper


def _surface_xyzfourier_scalar_grad(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
        coeff_template=None,
    ):
        return jax.grad(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )

    return wrapper


def _surface_xyzfourier_scalar_hessian(fn):
    def wrapper(
        dofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices=None,
        coeff_template=None,
    ):
        return jax.hessian(fn, argnums=0)(
            dofs,
            quadpoints_phi,
            quadpoints_theta,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
            coeff_template,
        )

    return wrapper


surface_xyzfourier_dgamma_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gamma_from_dofs
)
surface_xyzfourier_dgammadash1_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gammadash1_from_dofs
)
surface_xyzfourier_dgammadash2_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gammadash2_from_dofs
)
surface_xyzfourier_dnormal_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_normal_from_dofs
)
surface_xyzfourier_d2normal_by_dcoeffdcoeff = _surface_xyzfourier_dcoeff_hessian(
    surface_xyzfourier_normal_from_dofs
)
surface_xyzfourier_dunitnormal_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_unitnormal_from_dofs
)
surface_xyzfourier_dgammadash1dash1_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gammadash1dash1_from_dofs
)
surface_xyzfourier_dgammadash1dash2_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gammadash1dash2_from_dofs
)
surface_xyzfourier_dgammadash2dash2_by_dcoeff = _surface_xyzfourier_dcoeff_jacobian(
    surface_xyzfourier_gammadash2dash2_from_dofs
)
surface_xyzfourier_darea_by_dcoeff = _surface_xyzfourier_scalar_grad(
    surface_xyzfourier_area_from_dofs
)
surface_xyzfourier_d2area_by_dcoeffdcoeff = _surface_xyzfourier_scalar_hessian(
    surface_xyzfourier_area_from_dofs
)
surface_xyzfourier_dvolume_by_dcoeff = _surface_xyzfourier_scalar_grad(
    surface_xyzfourier_volume_from_dofs
)
surface_xyzfourier_d2volume_by_dcoeffdcoeff = _surface_xyzfourier_scalar_hessian(
    surface_xyzfourier_volume_from_dofs
)
