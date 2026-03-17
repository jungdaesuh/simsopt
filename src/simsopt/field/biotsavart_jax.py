"""
Pure JAX implementation of the Biot-Savart magnetic field computation.

This module provides JIT-compilable, autodiff-compatible functions that
replace the C++ ``simsoptpp.BiotSavart`` kernel for GPU execution.

All functions accept and return JAX arrays and are fully traceable
by ``jax.grad``, ``jax.jacfwd``, ``jax.jacrev``, and ``jax.hessian``.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "biot_savart_B",
    "biot_savart_dB_by_dX",
    "biot_savart_B_and_dB",
]

# μ₀ / (4π) in SI units  [T·m/A]
_MU0_OVER_4PI = 1e-7


def _biot_savart_one_point(x, gammas, gammadashs, currents):
    """Biot-Savart B at a single evaluation point.

    Args:
        x: (3,) evaluation point.
        gammas: (ncoils, nquad, 3) coil curve positions.
        gammadashs: (ncoils, nquad, 3) coil curve tangent vectors dγ/dφ.
        currents: (ncoils,) coil currents [A].

    Returns:
        B: (3,) magnetic field [T].
    """
    # diff[c, q, :] = γ_cq − x
    diff = gammas - x  # broadcast (ncoils, nquad, 3)

    # |diff|² and |diff|⁻³  (double-where to keep gradients clean)
    r2 = jnp.sum(diff * diff, axis=-1)  # (ncoils, nquad)
    safe_r2 = jnp.where(r2 > 0, r2, 1.0)
    r_inv3 = jnp.where(r2 > 0, safe_r2 ** (-1.5), 0.0)

    # (γ − x) × γ'
    cross = jnp.cross(diff, gammadashs)  # (ncoils, nquad, 3)

    # integrand weighted by 1/|r|³, averaged over quadrature
    integrand = cross * r_inv3[..., None]  # (ncoils, nquad, 3)
    integral = jnp.mean(integrand, axis=1)  # (ncoils, 3)

    # weight by I_k and sum over coils
    B = _MU0_OVER_4PI * jnp.einsum("c,cj->j", currents, integral)
    return B


@jax.jit
def biot_savart_B(points, gammas, gammadashs, currents):
    """Compute the Biot-Savart magnetic field at many evaluation points.

    .. math::

        \\mathbf B(\\mathbf x) = \\frac{\\mu_0}{4\\pi}
        \\sum_k I_k \\int_0^1
        \\frac{(\\Gamma_k - \\mathbf x) \\times \\Gamma_k'}{
               \\|\\Gamma_k - \\mathbf x\\|^3}\\,d\\varphi

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        B: (npoints, 3) magnetic field [T].
    """
    return jax.vmap(_biot_savart_one_point, in_axes=(0, None, None, None))(
        points, gammas, gammadashs, currents
    )


@jax.jit
def biot_savart_dB_by_dX(points, gammas, gammadashs, currents):
    """Compute the spatial Jacobian dB/dX at many evaluation points.

    Uses forward-mode autodiff on the single-point kernel (3→3, so
    ``jacfwd`` requires exactly 3 JVP evaluations).

    Follows the SIMSOPT convention from ``fields.rst``:
    ``dB_dX[p, j, l] = ∂_j B_l(x_p)``, i.e. axis 1 is the derivative
    direction, axis 2 is the B component.

    Args:
        points: (npoints, 3) evaluation points.
        gammas: (ncoils, nquad, 3) coil positions.
        gammadashs: (ncoils, nquad, 3) coil tangent vectors.
        currents: (ncoils,) coil currents [A].

    Returns:
        dB_dX: (npoints, 3, 3) where ``dB_dX[p, j, l] = ∂_j B_l``.
    """
    jac_fn = jax.jacfwd(_biot_savart_one_point, argnums=0)
    # jacfwd gives J[i,j] = ∂B_i/∂X_j; SIMSOPT wants ∂_j B_l, so transpose
    raw = jax.vmap(
        lambda x: jac_fn(x, gammas, gammadashs, currents),
        in_axes=0,
    )(points)
    return jnp.swapaxes(raw, -1, -2)


@jax.jit
def biot_savart_B_and_dB(points, gammas, gammadashs, currents):
    """Compute B and dB/dX together (shares JIT compilation overhead).

    Returns:
        (B, dB_dX) with shapes (npoints, 3) and (npoints, 3, 3),
        where ``dB_dX[p, j, l] = ∂_j B_l`` (SIMSOPT convention).
    """

    def _val_and_jac(x):
        f = lambda xx: _biot_savart_one_point(xx, gammas, gammadashs, currents)
        primals, tangents_fn = jax.linearize(f, x)
        # linearize JVP pushforward: tangents_fn(e_j) = J[:, j].
        # vmapping over eye(3) gives jac_T[j, i] = ∂B_i/∂X_j = ∂_j B_i,
        # which is already the SIMSOPT convention ∂_j B_l.
        return primals, jax.vmap(tangents_fn)(jnp.eye(3))

    B, dB_dX = jax.vmap(_val_and_jac)(points)
    return B, dB_dX
