"""Pure array reducers for MHD objective values."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

__all__ = [
    "boozer_quasisymmetry_mode_indices",
    "boozer_quasisymmetry_residuals",
    "iota_target_metric_j",
    "iota_weighted_j",
    "well_weighted_j",
]


def boozer_quasisymmetry_mode_indices(
    xm_b,
    xn_b,
    nfp: int,
    helicity_m: int,
    helicity_n: int,
):
    """Return fixed-shape mode-index metadata for ``Quasisymmetry.J()``."""
    if helicity_m != 0 and helicity_m != 1:
        raise ValueError("m for quasisymmetry should be 0 or 1.")

    xm = np.asarray(xm_b)
    xn = np.asarray(xn_b) / nfp

    if helicity_n == 0:
        symmetric = xn == 0
    elif helicity_m == 0:
        symmetric = xm == 0
    else:
        symmetric = (xm * helicity_n + xn * helicity_m) == 0

    symmetric_indices = np.nonzero(symmetric)[0]
    nonsymmetric_indices = np.nonzero(np.logical_not(symmetric))[0]
    return symmetric_indices, nonsymmetric_indices


def boozer_quasisymmetry_residuals(
    bmnc_b,
    symmetric_indices,
    nonsymmetric_indices,
    surface_indices,
    s_used,
    *,
    normalization: str = "B00",
    weight: str = "even",
):
    """Reduce frozen Boozer spectra to ``Quasisymmetry.J()`` residuals.

    Boozer mode classification is fixed metadata for one spectral layout.
    Compute it outside JAX transforms with ``boozer_quasisymmetry_mode_indices``
    and pass the resulting fixed-shape index arrays here.
    """
    symmetric_indices_jax = jnp.asarray(symmetric_indices, dtype=jnp.int32)
    nonsymmetric_indices_jax = jnp.asarray(nonsymmetric_indices, dtype=jnp.int32)

    bmnc = jnp.take(
        jnp.asarray(bmnc_b),
        jnp.asarray(surface_indices, dtype=jnp.int32),
        axis=1,
    )
    bmnc = jnp.moveaxis(bmnc, 0, 1)

    if normalization == "B00":
        bnorm = bmnc[:, 0]
    elif normalization == "symmetric":
        symmetric_bmnc = jnp.take(bmnc, symmetric_indices_jax, axis=1)
        bnorm = jnp.sqrt(jnp.sum(symmetric_bmnc * symmetric_bmnc, axis=1))
    else:
        raise ValueError("Unrecognized value for normalization in Quasisymmetry")

    normalized_bmnc = bmnc / bnorm[:, None]
    nonsymmetric_bmnc = jnp.take(normalized_bmnc, nonsymmetric_indices_jax, axis=1)

    if weight == "even":
        return jnp.ravel(nonsymmetric_bmnc)
    if weight == "stellopt":
        radial_weight = jnp.asarray(s_used) * jnp.asarray(s_used)
        return jnp.ravel(nonsymmetric_bmnc / radial_weight[:, None])
    if weight == "stellopt_ornl":
        return jnp.sqrt(jnp.sum(nonsymmetric_bmnc * nonsymmetric_bmnc, axis=1))

    raise ValueError("Unrecognized value for weight in Quasisymmetry")


def iota_target_metric_j(iotas_half_grid, target_iotas_half_grid, ds):
    """Return ``IotaTargetMetric.J()`` from frozen half-grid arrays."""
    delta = jnp.asarray(iotas_half_grid) - jnp.asarray(target_iotas_half_grid)
    return 0.5 * jnp.sum(delta * delta) * ds


def iota_weighted_j(iotas_half_grid, weights_half_grid):
    """Return ``IotaWeighted.J()`` from frozen half-grid arrays."""
    weights = jnp.asarray(weights_half_grid)
    return jnp.sum(weights * jnp.asarray(iotas_half_grid)) / jnp.sum(weights)


def well_weighted_j(vp_half_grid, weights1_half_grid, weights2_half_grid):
    """Return ``WellWeighted.J()`` from frozen half-grid arrays."""
    vp = jnp.asarray(vp_half_grid)
    weights1 = jnp.asarray(weights1_half_grid)
    weights2 = jnp.asarray(weights2_half_grid)
    return jnp.sum((weights1 - weights2) * vp) / jnp.sum((weights1 + weights2) * vp)
