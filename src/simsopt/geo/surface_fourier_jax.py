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
    "surface_gammadash1dash2_from_dofs",
    "surface_gammadash2dash2_from_dofs",
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
    "surface_xyzfourier_gammadash1dash2_from_dofs",
    "surface_xyzfourier_gammadash2dash2_from_dofs",
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


def _as_jax_float64(value):
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.float64)
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _as_runtime_float64(value, *, reference):
    del reference
    return _as_jax_float64(value)


def _as_jax_int32(value):
    if isinstance(value, jax.Array):
        return jnp.asarray(value, dtype=jnp.int32)
    return jax.device_put(np.asarray(value, dtype=np.int32))


def _zeros(shape, dtype):
    return jax.device_put(np.zeros(shape, dtype=np.dtype(dtype)))


_TWO_PI_HOST = np.float64(2.0 * np.pi)
_BASIS_SELECTORS3_HOST = np.eye(3, dtype=np.float64)


def _two_pi(reference):
    return _as_runtime_float64(_TWO_PI_HOST, reference=reference)


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


def surface_gamma(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp):
    """Evaluate surface Cartesian coordinates on the quadrature grid.

    Args:
        quadpoints_phi:   (nphi,)
        quadpoints_theta: (ntheta,)
        xc: (2*mpol+1, 2*ntor+1) coefficients for x̂.
        yc: (2*mpol+1, 2*ntor+1) coefficients for ŷ.
        zc: (2*mpol+1, 2*ntor+1) coefficients for z.
        mpol, ntor, nfp: integers (static).

    Returns:
        gamma: (nphi, ntheta, 3)  Cartesian [x, y, z].
    """
    W, _ = build_theta_basis(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)

    xhat = _eval_hat(V, W, xc)  # (nphi, ntheta)
    yhat = _eval_hat(V, W, yc)
    z = _eval_hat(V, W, zc)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax  # (nphi,)
    cphi = jnp.cos(phi_angle)[:, None]  # (nphi, 1)
    sphi = jnp.sin(phi_angle)[:, None]

    x = xhat * cphi - yhat * sphi
    y = xhat * sphi + yhat * cphi

    return jnp.stack([x, y, z], axis=-1)


def surface_gamma_lin(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp):
    """Evaluate surface Cartesian coordinates at paired ``(phi[i], theta[i])``."""
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi).reshape(-1)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta).reshape(-1)
    W, _ = build_theta_basis(quadpoints_theta_jax, mpol)
    V, _ = build_phi_basis(quadpoints_phi_jax, ntor, nfp)

    xhat = _eval_hat_paired(V, W, xc)
    yhat = _eval_hat_paired(V, W, yc)
    z = _eval_hat_paired(V, W, zc)

    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)
    sphi = jnp.sin(phi_angle)
    x = xhat * cphi - yhat * sphi
    y = xhat * sphi + yhat * cphi
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
):
    """Evaluate dγ/d(quadpoints_phi) at paired ``(phi[i], theta[i])``."""
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
):
    """Evaluate dγ/d(quadpoints_theta) at paired ``(phi[i], theta[i])``."""
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


def surface_gammadash1(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp):
    """Evaluate dγ/d(quadpoints_phi) — the toroidal tangent vector.

    Returns:
        gammadash1: (nphi, ntheta, 3).
    """
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


def surface_gammadash2(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp):
    """Evaluate dγ/d(quadpoints_theta) — the poloidal tangent vector.

    Returns:
        gammadash2: (nphi, ntheta, 3).
    """
    _, dW = build_theta_basis(quadpoints_theta, mpol)
    V, _ = build_phi_basis(quadpoints_phi, ntor, nfp)

    dxhat_dtheta = _eval_hat(V, dW, xc)
    dyhat_dtheta = _eval_hat(V, dW, yc)
    dz_dtheta = _eval_hat(V, dW, zc)

    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    phi_angle = _two_pi(quadpoints_phi_jax) * quadpoints_phi_jax
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]

    dx = dxhat_dtheta * cphi - dyhat_dtheta * sphi
    dy = dxhat_dtheta * sphi + dyhat_dtheta * cphi
    dz = dz_dtheta

    return jnp.stack([dx, dy, dz], axis=-1)


def surface_gammadash1dash1(
    quadpoints_phi,
    quadpoints_theta,
    xc,
    yc,
    zc,
    mpol,
    ntor,
    nfp,
):
    """Evaluate d²γ/d(quadpoints_phi)²."""
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
):
    """Evaluate d²γ/d(quadpoints_phi)d(quadpoints_theta)."""
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
):
    """Evaluate d²γ/d(quadpoints_theta)²."""
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


def surface_normal(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp):
    """Compute the (unnormalized) surface normal n = gammadash1 × gammadash2.

    Returns:
        normal: (nphi, ntheta, 3).
    """
    gd1 = surface_gammadash1(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
    )
    gd2 = surface_gammadash2(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
    )
    return jnp.cross(gd1, gd2)


def _unitnormal(normal):
    return normal / jnp.sqrt(jnp.sum(normal * normal, axis=-1))[..., None]


# ---------------------------------------------------------------------------
# Coefficient-parametric evaluation (for autodiff w.r.t. dofs)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stellsym DOF ↔ coefficient mapping
# ---------------------------------------------------------------------------


def _is_stellsym_xy(m, n, mpol, ntor):
    """True if (m, n) index is free for x̂/ŷ under stellsym.

    Allowed quadrants: cos-cos (rows 0..mpol, cols 0..ntor)
    and sin-sin (rows mpol+1..2*mpol, cols ntor+1..2*ntor).
    """
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_cos_phi) or (is_sin_theta and is_sin_phi)


def _is_stellsym_z(m, n, mpol, ntor):
    """True if (m, n) index is free for z under stellsym.

    Allowed quadrants: cos-sin (rows 0..mpol, cols ntor+1..2*ntor)
    and sin-cos (rows mpol+1..2*mpol, cols 0..ntor).
    """
    is_cos_theta = m <= mpol
    is_sin_theta = m > mpol
    is_cos_phi = n <= ntor
    is_sin_phi = n > ntor
    return (is_cos_theta and is_sin_phi) or (is_sin_theta and is_cos_phi)


def stellsym_scatter_indices(mpol, ntor):
    """Compute scatter indices for stellsym DOF unpacking.

    The returned array maps DOF index ``i`` to position in the flattened
    ``[xc, yc, zc]`` super-vector (each block of length
    ``(2*mpol+1)*(2*ntor+1)``).

    Returns:
        indices: (ndofs,) int array.
    """
    n_per_coord = (2 * mpol + 1) * (2 * ntor + 1)
    indices = []
    # Stellsym convention: x uses cos-cos + sin-sin (even-even),
    # y and z use cos-sin + sin-cos (odd-odd).  This matches
    # CPU SurfaceXYZTensorFourier where y transforms like z
    # under the stellarator symmetry (φ,θ) → (−φ,−θ).
    for coord_offset, allowed_fn in [
        (0, _is_stellsym_xy),  # x: cos-cos + sin-sin
        (n_per_coord, _is_stellsym_z),  # y: cos-sin + sin-cos
        (2 * n_per_coord, _is_stellsym_z),  # z: cos-sin + sin-cos
    ]:
        for m in range(2 * mpol + 1):
            for n in range(2 * ntor + 1):
                if allowed_fn(m, n, mpol, ntor):
                    indices.append(coord_offset + m * (2 * ntor + 1) + n)
    return np.array(indices, dtype=np.int32)


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

    Returns:
        gamma: (nphi, ntheta, 3) Cartesian positions.
    """
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gamma(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp)


def surface_gamma_lin_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
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
):
    """Evaluate dγ/dφ as a pure function of DOFs (autodiff-compatible)."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
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
):
    """Evaluate dγ/dθ as a pure function of DOFs (autodiff-compatible)."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
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
):
    """Evaluate d²γ/dφ² as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1dash1(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
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
):
    """Evaluate d²γ/dφdθ as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash1dash2(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
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
):
    """Evaluate d²γ/dθ² as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_gammadash2dash2(
        quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp
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
):
    """Evaluate unnormalized normal as a pure function of DOFs."""
    xc, yc, zc = _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices)
    return surface_normal(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp)


def surface_unitnormal_from_dofs(
    dofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices=None,
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
    )
    return surface_volume(gamma, normal)


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------


def surface_volume(gamma, normal):
    """Compute the volume enclosed by a toroidal surface.

    Uses the divergence theorem:
    ``V = (1/3) ∫∫ γ · n dφ dθ``
    where ``n = gammadash1 × gammadash2`` is the unnormalized normal.

    The ``nfp`` factor cancels with the quadrature step size.

    Args:
        gamma:  (nphi, ntheta, 3) surface positions.
        normal: (nphi, ntheta, 3) unnormalized normal vectors.

    Returns:
        Scalar volume.
    """
    nphi, ntheta = gamma.shape[:2]
    integrand = jnp.sum(gamma * normal, axis=-1)  # (nphi, ntheta)
    return jnp.sum(integrand) / _as_jax_float64(3.0 * nphi * ntheta)


def surface_area(normal):
    """Compute the surface area of a toroidal surface.

    ``A = ∫∫ |n| dφ dθ``
    where ``n = gammadash1 × gammadash2`` is the unnormalized normal.

    Args:
        normal: (nphi, ntheta, 3) unnormalized normal vectors.

    Returns:
        Scalar area.
    """
    nphi, ntheta = normal.shape[:2]
    norm_n = jnp.sqrt(jnp.sum(normal * normal, axis=-1))  # (nphi, ntheta)
    return jnp.sum(norm_n) / _as_jax_float64(nphi * ntheta)


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
