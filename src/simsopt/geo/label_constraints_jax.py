"""
Pure JAX label constraint functions for the Boozer inner solve.

These replace the CPU ``Volume.J()`` / ``ToroidalFlux.J()`` calls
inside the penalty objective so the entire inner solve stays on-device.

Both functions are JAX-traceable, meaning ``jax.grad`` through the
composed penalty objective automatically produces label constraint
derivatives — no hand-coded ``dJ``, ``d2J`` needed.
"""

import jax.numpy as jnp

from simsopt.geo.surface_fourier_jax import surface_area as area_jax
from simsopt.geo.surface_fourier_jax import surface_volume as volume_jax

__all__ = [
    "volume_jax",
    "area_jax",
    "toroidal_flux_jax",
    "compute_G_from_currents",
]


def toroidal_flux_jax(A, gammadash2_at_phi, ntheta):
    """Compute toroidal flux at a fixed toroidal angle.

    Via Stokes' theorem:

    .. math::

        \\Phi_\\text{tor} = \\oint \\mathbf A \\cdot \\mathbf t\\,dl
        \\approx \\frac{1}{N_\\theta} \\sum_j
        \\mathbf A_j \\cdot \\gamma_{\\theta,j}

    Args:
        A: (ntheta, 3) vector potential at surface points
           on the chosen phi slice.
        gammadash2_at_phi: (ntheta, 3) poloidal tangent vectors
           on the same phi slice.
        ntheta: number of poloidal quadrature points.

    Returns:
        Scalar toroidal flux.
    """
    return jnp.sum(A * gammadash2_at_phi) / ntheta


def compute_G_from_currents(currents):
    """Compute Boozer G from coil currents.

    ``G = 2π Σ|I_k| · μ₀/(2π) = μ₀ Σ|I_k|``

    Args:
        currents: (ncoils,) coil currents [A].

    Returns:
        Scalar G.
    """
    mu0 = 4.0 * jnp.pi * 1e-7
    return mu0 * jnp.sum(jnp.abs(currents))
