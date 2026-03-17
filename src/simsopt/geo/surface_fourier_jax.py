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
import jax.numpy as jnp

__all__ = [
    "build_theta_basis",
    "build_phi_basis",
    "surface_gamma",
    "surface_gammadash1",
    "surface_gammadash2",
    "surface_normal",
    "surface_gamma_from_dofs",
    "surface_gammadash1_from_dofs",
    "surface_gammadash2_from_dofs",
    "surface_normal_from_dofs",
    "surface_volume",
    "stellsym_scatter_indices",
    "dofs_to_xyzc",
]

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
    theta = 2.0 * jnp.pi * quadpoints_theta  # (ntheta,)

    m_cos = jnp.arange(0, mpol + 1, dtype=theta.dtype)  # [0 .. mpol]
    m_sin = jnp.arange(1, mpol + 1, dtype=theta.dtype)  # [1 .. mpol]

    arg_cos = m_cos[None, :] * theta[:, None]  # (ntheta, mpol+1)
    arg_sin = m_sin[None, :] * theta[:, None]  # (ntheta, mpol)

    W = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)

    # d/d(quadpoints_theta) = d/dθ_param  (chain rule: dθ/dθ_param = 2π)
    dW = jnp.concatenate(
        [
            -m_cos[None, :] * 2.0 * jnp.pi * jnp.sin(arg_cos),
            m_sin[None, :] * 2.0 * jnp.pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )

    return W, dW


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
    phi = 2.0 * jnp.pi * quadpoints_phi  # (nphi,)

    # frequencies: [0, nfp, 2*nfp, …, ntor*nfp]
    n_cos = jnp.arange(0, ntor + 1, dtype=phi.dtype) * nfp
    n_sin = jnp.arange(1, ntor + 1, dtype=phi.dtype) * nfp

    arg_cos = n_cos[None, :] * phi[:, None]  # (nphi, ntor+1)
    arg_sin = n_sin[None, :] * phi[:, None]  # (nphi, ntor)

    V = jnp.concatenate([jnp.cos(arg_cos), jnp.sin(arg_sin)], axis=1)

    dV = jnp.concatenate(
        [
            -n_cos[None, :] * 2.0 * jnp.pi * jnp.sin(arg_cos),
            n_sin[None, :] * 2.0 * jnp.pi * jnp.cos(arg_sin),
        ],
        axis=1,
    )

    return V, dV


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

    phi_angle = 2.0 * jnp.pi * quadpoints_phi  # (nphi,)
    cphi = jnp.cos(phi_angle)[:, None]  # (nphi, 1)
    sphi = jnp.sin(phi_angle)[:, None]

    x = xhat * cphi - yhat * sphi
    y = xhat * sphi + yhat * cphi

    return jnp.stack([x, y, z], axis=-1)


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

    phi_angle = 2.0 * jnp.pi * quadpoints_phi
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]
    two_pi = 2.0 * jnp.pi

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

    phi_angle = 2.0 * jnp.pi * quadpoints_phi
    cphi = jnp.cos(phi_angle)[:, None]
    sphi = jnp.sin(phi_angle)[:, None]

    dx = dxhat_dtheta * cphi - dyhat_dtheta * sphi
    dy = dxhat_dtheta * sphi + dyhat_dtheta * cphi
    dz = dz_dtheta

    return jnp.stack([dx, dy, dz], axis=-1)


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
    for coord_offset, allowed_fn in [
        (0, _is_stellsym_xy),
        (n_per_coord, _is_stellsym_xy),
        (2 * n_per_coord, _is_stellsym_z),
    ]:
        for m in range(2 * mpol + 1):
            for n in range(2 * ntor + 1):
                if allowed_fn(m, n, mpol, ntor):
                    indices.append(coord_offset + m * (2 * ntor + 1) + n)
    return np.array(indices, dtype=np.int32)


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
    n_per_coord = (2 * mpol + 1) * (2 * ntor + 1)
    flat = jnp.zeros(3 * n_per_coord).at[scatter_indices].set(sdofs)
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = flat[:n_per_coord].reshape(shape)
    yc = flat[n_per_coord : 2 * n_per_coord].reshape(shape)
    zc = flat[2 * n_per_coord :].reshape(shape)
    return xc, yc, zc


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
    if stellsym:
        if scatter_indices is None:
            raise ValueError(
                "scatter_indices required for stellsym=True. "
                "Precompute with stellsym_scatter_indices(mpol, ntor)."
            )
        xc, yc, zc = dofs_to_xyzc(dofs, scatter_indices, mpol, ntor)
    else:
        n_per_coord = (2 * mpol + 1) * (2 * ntor + 1)
        xc = dofs[:n_per_coord].reshape((2 * mpol + 1, 2 * ntor + 1))
        yc = dofs[n_per_coord : 2 * n_per_coord].reshape((2 * mpol + 1, 2 * ntor + 1))
        zc = dofs[2 * n_per_coord :].reshape((2 * mpol + 1, 2 * ntor + 1))
    return surface_gamma(quadpoints_phi, quadpoints_theta, xc, yc, zc, mpol, ntor, nfp)


def _dofs_to_xyzc_any(dofs, mpol, ntor, stellsym, scatter_indices):
    """Internal helper: unpack DOFs to (xc, yc, zc) for both stellsym modes."""
    if stellsym:
        return dofs_to_xyzc(dofs, scatter_indices, mpol, ntor)
    n_per_coord = (2 * mpol + 1) * (2 * ntor + 1)
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = dofs[:n_per_coord].reshape(shape)
    yc = dofs[n_per_coord : 2 * n_per_coord].reshape(shape)
    zc = dofs[2 * n_per_coord :].reshape(shape)
    return xc, yc, zc


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
    return jnp.sum(integrand) / (3.0 * nphi * ntheta)
