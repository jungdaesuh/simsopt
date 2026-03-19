"""
Pure JAX Boozer residual functions.

This module provides two layers:

**M1 primitives** — ``boozer_residual_scalar``, ``boozer_residual_grad``,
``boozer_residual_hessian`` operate on pre-computed B/xphi/xtheta arrays
and differentiate only through (iota, G).  Surface DOF derivatives are
zero because field data is treated as constant.

**M3 composed pipeline** — ``boozer_penalty_composed``,
``boozer_penalty_grad_composed``, ``boozer_residual_jacobian_composed``,
``boozer_residual_coil_vjp`` trace through the full
DOFs → surface geometry → Biot-Savart → residual chain via JAX autodiff,
replacing the C++ kernels ``sopp.boozer_residual_ds``,
``sopp.boozer_residual_ds2``, and ``sopp.boozer_dresidual_dc``.

The residual at each grid point is

.. math::

    \\tilde{r}_{ij} = w_{ij}\\bigl[G\\,\\mathbf B_{ij}
        - |\\mathbf B_{ij}|^2 (\\mathbf x_\\varphi + \\iota\\,\\mathbf x_\\theta)\\bigr]

with ``w = 1/|B|`` when *weight_inv_modB* is True, else ``w = 1``.

The scalar objective is

.. math::

    J = \\frac{1}{2 N}\\sum_{i,j} \\|\\tilde{r}_{ij}\\|^2

where ``N = 3 · nphi · ntheta`` (matching the C++ normalization).
"""

import jax
import jax.numpy as jnp

__all__ = [
    "boozer_residual_scalar",
    "boozer_residual_grad",
    "boozer_residual_hessian",
    "boozer_residual_vector",
    "boozer_penalty_composed",
    "boozer_penalty_grad_composed",
    "boozer_residual_jacobian_composed",
    "boozer_residual_coil_vjp",
]


def boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB=True):
    """Boozer residual scalar objective (forward pass).

    Args:
        G:     scalar (Boozer G constant).
        iota:  scalar (rotational transform).
        B:     (nphi, ntheta, 3)  magnetic field on the surface.
        xphi:  (nphi, ntheta, 3)  surface tangent dγ/dφ.
        xtheta:(nphi, ntheta, 3)  surface tangent dγ/dθ.
        weight_inv_modB: if True, weight residual by 1/|B|.

    Returns:
        J: scalar objective value.
    """
    nphi, ntheta, _ = B.shape
    num_res = 3 * nphi * ntheta

    tang = xphi + iota * xtheta  # (nphi, ntheta, 3)
    B2 = jnp.sum(B * B, axis=-1)  # (nphi, ntheta)
    residual = G * B - B2[..., None] * tang  # (nphi, ntheta, 3)

    if weight_inv_modB:
        w = 1.0 / jnp.sqrt(B2)  # (nphi, ntheta)
        rtil = w[..., None] * residual  # (nphi, ntheta, 3)
    else:
        rtil = residual

    return 0.5 * jnp.sum(rtil * rtil) / num_res


# ---------------------------------------------------------------------------
# M1 gradient / Hessian wrappers (iota/G only, surface DOFs are constants).
# For the full composed pipeline through surface DOFs, use the M3 functions:
# boozer_penalty_grad_composed() and boozer_residual_jacobian_composed().
# ---------------------------------------------------------------------------


def _pack(surface_dofs, iota, G):
    """Pack (surface_dofs, iota, G) into a single vector for autodiff."""
    return jnp.concatenate([surface_dofs, jnp.array([iota, G])])


def _unpack(x, nsurfdofs):
    """Unpack a single vector into (surface_dofs, iota, G)."""
    return x[:nsurfdofs], x[nsurfdofs], x[nsurfdofs + 1]


def _boozer_objective_from_packed(x, nsurfdofs, B, xphi, xtheta, weight_inv_modB):
    """Scalar objective as a function of the packed decision vector."""
    _, iota, G = _unpack(x, nsurfdofs)
    return boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)


def boozer_residual_grad(G, iota, B, xphi, xtheta, nsurfdofs, weight_inv_modB=True):
    """Gradient of the Boozer residual w.r.t. [surface_dofs, iota, G].

    Surface DOF gradient entries are zero because B, xphi, xtheta are
    treated as constants (not differentiated through surface evaluation).
    Only iota and G entries are non-trivial.

    For the full composed pipeline (DOFs → geometry → field → residual),
    use :func:`boozer_penalty_grad_composed` instead.

    Returns:
        grad: (nsurfdofs + 2,) gradient vector.
    """
    x0 = _pack(jnp.zeros(nsurfdofs), iota, G)
    grad_fn = jax.grad(
        lambda x: _boozer_objective_from_packed(
            x, nsurfdofs, B, xphi, xtheta, weight_inv_modB
        )
    )
    return grad_fn(x0)


def boozer_residual_hessian(G, iota, B, xphi, xtheta, nsurfdofs, weight_inv_modB=True):
    """Hessian of the Boozer residual w.r.t. [surface_dofs, iota, G].

    Surface DOF Hessian blocks are zero because B, xphi, xtheta are
    treated as constants.  For the full composed pipeline, use
    ``jax.hessian(boozer_penalty_composed)`` instead.

    Returns:
        H: (nsurfdofs + 2, nsurfdofs + 2) Hessian matrix.
    """
    x0 = _pack(jnp.zeros(nsurfdofs), iota, G)
    hess_fn = jax.hessian(
        lambda x: _boozer_objective_from_packed(
            x, nsurfdofs, B, xphi, xtheta, weight_inv_modB
        )
    )
    return hess_fn(x0)


# ---------------------------------------------------------------------------
# M3: Composed derivative path (DOFs → geometry → field → residual)
# ---------------------------------------------------------------------------


def boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB=True):
    """Boozer residual vector (not the scalar 0.5||r||²/N).

    Returns the weighted residual at each grid point, flattened.

    Args:
        G, iota: Boozer parameters.
        B:      (nphi, ntheta, 3) magnetic field on the surface.
        xphi:   (nphi, ntheta, 3) toroidal tangent.
        xtheta: (nphi, ntheta, 3) poloidal tangent.
        weight_inv_modB: weight by 1/|B| if True.

    Returns:
        (nphi*ntheta*3,) flattened residual vector.
    """
    tang = xphi + iota * xtheta
    B2 = jnp.sum(B * B, axis=-1)
    residual = G * B - B2[..., None] * tang

    if weight_inv_modB:
        w = 1.0 / jnp.sqrt(B2)
        residual = w[..., None] * residual

    return residual.ravel()


def _get_surface_fns():
    """Lazily import surface geometry functions (avoids simsopt top-level)."""
    from simsopt.geo.surface_fourier_jax import (
        surface_gamma_from_dofs,
        surface_gammadash1_from_dofs,
        surface_gammadash2_from_dofs,
    )

    return (
        surface_gamma_from_dofs,
        surface_gammadash1_from_dofs,
        surface_gammadash2_from_dofs,
    )


def _get_grouped_biot_savart():
    """Lazily import grouped Biot-Savart (avoids simsopt top-level)."""
    from simsopt.field.biotsavart_jax import grouped_biot_savart_B

    return grouped_biot_savart_B


def _surface_geometry_from_dofs(
    sdofs,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
):
    """Evaluate gamma, gammadash1, gammadash2 from surface DOFs.

    Pure function suitable for JAX tracing.  Used by both the composed
    M3 pipeline and the M4 solver (``boozersurface_jax.py``).
    """
    sgf, sg1f, sg2f = _get_surface_fns()
    args = (
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )
    return sgf(*args), sg1f(*args), sg2f(*args)


def _unpack_decision_vector(x, coil_arrays, optimize_G):
    """Unpack decision vector into (sdofs, iota, G).

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
    """
    if optimize_G:
        return x[:-2], x[-2], x[-1]
    mu0 = 4.0 * jnp.pi * 1e-7
    all_currents = jnp.concatenate([c for _, _, c in coil_arrays])
    return x[:-1], x[-1], mu0 * jnp.sum(jnp.abs(all_currents))


def _composed_pipeline(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G,
):
    """Shared pipeline: unpack x → surface geometry → Biot-Savart field.

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.

    Returns (sdofs, iota, G, gamma, xphi, xtheta, B).
    """
    sdofs, iota, G = _unpack_decision_vector(x, coil_arrays, optimize_G)

    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        sdofs,
        quadpoints_phi,
        quadpoints_theta,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )

    grouped_bs_B = _get_grouped_biot_savart()
    B = grouped_bs_B(gamma.reshape(-1, 3), coil_arrays)
    B = B.reshape(gamma.shape)

    return sdofs, iota, G, gamma, xphi, xtheta, B


def boozer_penalty_composed(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G,
    weight_inv_modB=True,
):
    """Composed scalar penalty objective: DOFs → geometry → field → residual → scalar.

    The decision vector is ``x = [surface_dofs, iota]`` (optimize_G=False)
    or ``x = [surface_dofs, iota, G]`` (optimize_G=True).

    Args:
        x: (n,) flat decision vector.
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
        quadpoints_phi, quadpoints_theta: quadrature grids.
        mpol, ntor, nfp: surface resolution.
        stellsym: stellarator symmetry flag.
        scatter_indices: stellsym DOF scatter indices (or None).
        optimize_G: whether G is in the decision vector.
        weight_inv_modB: weight residual by 1/|B|.

    Returns:
        Scalar objective value.
    """
    _, iota, G, _, xphi, xtheta, B = _composed_pipeline(
        x,
        coil_arrays=coil_arrays,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        optimize_G=optimize_G,
    )
    return boozer_residual_scalar(G, iota, B, xphi, xtheta, weight_inv_modB)


def boozer_penalty_grad_composed(x, **kwargs):
    """VJP-based gradient of the composed penalty objective.

    Uses reverse-mode autodiff through the full pipeline:
    DOFs → surface geometry → Biot-Savart → residual → scalar.

    Args:
        x: (n,) flat decision vector.
        **kwargs: forwarded to :func:`boozer_penalty_composed`.

    Returns:
        (val, grad): scalar objective value and (n,) gradient vector.
    """
    return jax.value_and_grad(boozer_penalty_composed)(x, **kwargs)


def _boozer_residual_vector_composed(
    x,
    *,
    coil_arrays,
    quadpoints_phi,
    quadpoints_theta,
    mpol,
    ntor,
    nfp,
    stellsym,
    scatter_indices,
    optimize_G=True,
    weight_inv_modB=False,
):
    """Composed residual vector: DOFs → geometry → field → residual vector.

    The decision vector is ``x = [surface_dofs, iota, G]`` (optimize_G=True,
    default for BoozerExact) or ``x = [surface_dofs, iota]``
    (optimize_G=False).

    Args:
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.

    Returns:
        (nphi*ntheta*3,) flattened residual vector.
    """
    _, iota, G, _, xphi, xtheta, B = _composed_pipeline(
        x,
        coil_arrays=coil_arrays,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        scatter_indices=scatter_indices,
        optimize_G=optimize_G,
    )
    return boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)


def boozer_residual_jacobian_composed(
    x,
    **kwargs,
):
    """Explicit Jacobian of the composed residual vector.

    Uses ``jax.jacfwd`` to compute the full Jacobian matrix
    ``J[i,k] = ∂r_i/∂x_k`` where ``r`` is the residual vector and
    ``x = [surface_dofs, iota, G]`` (optimize_G=True) or
    ``x = [surface_dofs, iota]`` (optimize_G=False).

    This replaces the hand-coded C++ chain:
    ``sopp.boozer_dresidual_dc`` + ``dgamma_by_dcoeff`` + ``dB_by_dX``.

    Args:
        x: (n,) flat decision vector.
        **kwargs: forwarded to :func:`_boozer_residual_vector_composed`.

    Returns:
        (r, J): residual vector (n_res,) and Jacobian (n_res, n).
    """
    r = _boozer_residual_vector_composed(x, **kwargs)
    J = jax.jacfwd(_boozer_residual_vector_composed)(x, **kwargs)
    return r, J


def boozer_residual_coil_vjp(
    adjoint,
    *,
    gamma,
    xphi,
    xtheta,
    coil_arrays,
    iota,
    G,
    weight_inv_modB=False,
):
    """VJP of Boozer residual w.r.t. coil parameters (outer path).

    Given an adjoint vector (from the outer optimization), computes
    sensitivities of ``adjoint^T @ r`` w.r.t. coil geometry and currents
    via reverse-mode autodiff through Biot-Savart.

    This replaces the CPU chain:
    ``boozer_surface_residual_dB()`` → ``B_vjp()`` →
    ``sopp.biot_savart_vjp_graph()``.

    The surface geometry (gamma, xphi, xtheta) is held fixed — this
    function computes how the residual changes when the magnetic field
    changes due to coil parameter variations.

    Args:
        adjoint: (nphi*ntheta*3,) adjoint vector from outer solve.
        gamma:   (nphi, ntheta, 3) fixed surface positions.
        xphi:    (nphi, ntheta, 3) fixed toroidal tangent.
        xtheta:  (nphi, ntheta, 3) fixed poloidal tangent.
        coil_arrays: list of ``(gammas, gammadashs, currents)`` tuples.
        iota: rotational transform (scalar).
        G: Boozer G constant (scalar).
        weight_inv_modB: weight residual by 1/|B|.

    Returns:
        ``(d_coil_arrays,)`` — 1-tuple of grouped cotangent list matching
        the ``coil_arrays`` pytree structure.
    """
    nphi, ntheta = gamma.shape[:2]
    expected = nphi * ntheta * 3
    if adjoint.shape != (expected,):
        raise ValueError(
            f"adjoint shape {adjoint.shape} != expected ({expected},) "
            f"for nphi={nphi}, ntheta={ntheta}"
        )
    grouped_bs_B = _get_grouped_biot_savart()

    def residual_of_coils(ca):
        points = gamma.reshape(-1, 3)
        B = grouped_bs_B(points, ca)
        B = B.reshape(nphi, ntheta, 3)
        return boozer_residual_vector(G, iota, B, xphi, xtheta, weight_inv_modB)

    _, vjp_fn = jax.vjp(residual_of_coils, coil_arrays)
    return vjp_fn(adjoint)
