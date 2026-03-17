"""
Pure JAX Boozer residual scalar objective.

Replaces the C++ kernels ``sopp.boozer_residual``,
``sopp.boozer_residual_ds``, and ``sopp.boozer_residual_ds2``
with a single forward function whose gradient and Hessian are
obtained automatically via ``jax.grad`` and ``jax.hessian``.

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
        w = jnp.sqrt(1.0 / B2)  # (nphi, ntheta)
        rtil = w[..., None] * residual  # (nphi, ntheta, 3)
    else:
        rtil = residual

    return 0.5 * jnp.sum(rtil * rtil) / num_res


# ---------------------------------------------------------------------------
# M1-only gradient / Hessian wrappers (will be replaced by composed
# surface → BiotSavart → residual pipeline in M2)
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

    The gradient w.r.t. *surface_dofs* is zero here because B, xphi,
    xtheta are provided as constants (not differentiated through the
    surface evaluation).  The gradient w.r.t. *iota* and *G* is
    non-trivial.

    For the full pipeline (M2+), the caller should compose surface
    evaluation → BiotSavart → this function and differentiate the
    composition.

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
