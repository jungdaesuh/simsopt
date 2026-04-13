"""Shared helpers for SurfaceXYZTensorFourier stellarator-symmetry masks."""

import numpy as np

from ..jax_core._math_utils import as_jax_int32 as _as_jax_int32


def surface_stellsym_mask_for_grid(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    """Return the exact-residual boolean mask for a specific quadrature grid."""
    phis = np.asarray(quadpoints_phi, dtype=float)
    thetas = np.asarray(quadpoints_theta, dtype=float)
    mask = np.ones((phis.size, thetas.size), dtype=bool)
    if not stellsym:
        return mask

    def _same_grid(lhs, rhs):
        return lhs.shape == rhs.shape and np.allclose(lhs, rhs)

    full_phi = np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False)
    full_theta = np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False)
    half_theta = np.linspace(0.0, 0.5, mpol + 1, endpoint=False)
    half_phi = np.linspace(0.0, 1.0 / (2.0 * nfp), ntor + 1, endpoint=False)

    if _same_grid(phis, full_phi) and _same_grid(thetas, full_theta):
        mask[:, mpol + 1 :] = False
        mask[ntor + 1 :, 0] = False
        return mask
    if _same_grid(phis, full_phi) and _same_grid(thetas, half_theta):
        mask[ntor + 1 :, 0] = False
        return mask
    if _same_grid(phis, half_phi) and _same_grid(thetas, full_theta):
        mask[0, mpol + 1 :] = False
        return mask
    raise Exception(
        "Stellarator symmetric BoozerExact surfaces require a specific set of "
        "quadrature points on the surface. See the "
        "SurfaceXYZTensorFourier.get_stellsym_mask() docstring for more "
        "information."
    )


def compute_stellsym_mask_indices_for_grid(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    """Return flattened exact-residual mask indices for a quadrature grid."""
    mask = np.repeat(
        surface_stellsym_mask_for_grid(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )[..., None],
        3,
        axis=2,
    )
    if stellsym:
        mask[0, 0, 0] = False
    return _as_jax_int32(np.flatnonzero(mask))
