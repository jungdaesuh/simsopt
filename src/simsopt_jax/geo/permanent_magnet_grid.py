"""JAX fixed-state payload for permanent-magnet optimization grids."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.core._math_utils import (
    as_runtime_array as _as_runtime_array,
    runtime_host_dtype as _runtime_host_dtype,
)
from simsopt_jax.core.dipole_field import dipole_field_Bn

__all__ = [
    "MWPGP_ALPHA_SAFETY_FACTOR",
    "PermanentMagnetGridJAX",
    "mwpgp_alpha_from_grid",
    "permanent_magnet_grid_to_jax",
]

MWPGP_ALPHA_SAFETY_FACTOR: float = 1.0 - 1.0e-5


def _reshape_moments(name: str, value: object, ndipoles: int) -> jax.Array:
    array = _as_runtime_array(value)
    if array.shape == (ndipoles * 3,):
        return jnp.reshape(array, (ndipoles, 3))
    if array.shape != (ndipoles, 3):
        raise ValueError(f"{name} must have shape ({ndipoles}, 3).")
    return array


@dataclass(frozen=True)
class PermanentMagnetGridJAX:
    """Immutable JAX payload for a fixed permanent-magnet optimization state."""

    A_obj: jax.Array
    b_obj: jax.Array
    ATb: jax.Array
    ATA_scale: jax.Array
    m0: jax.Array
    m: jax.Array
    m_proxy: jax.Array
    m_maxima: jax.Array
    dipole_grid_xyz: jax.Array
    coordinate_flag: str
    R0: float
    nfp: int
    stellsym: bool
    nphi: int
    ntheta: int
    ndipoles: int
    pol_vectors: jax.Array | None = None

    @classmethod
    def from_cpu(cls, pm_grid) -> "PermanentMagnetGridJAX":
        """Stage an already-initialized CPU ``PermanentMagnetGrid`` payload.

        ``pm_grid.ATA_scale`` must still be the raw spectral scale: native
        ``rescale_for_opt`` mutates it in place to ``ATA_scale + 2 reg_l2 +
        1/nu``, and the MwPGP validator derives its step bound by applying
        that same shift to the staged value — a grid staged after
        ``rescale_for_opt`` gets the shift twice and legitimate native steps
        are rejected.
        """

        ndipoles = int(pm_grid.ndipoles)
        if hasattr(pm_grid, "m_proxy"):
            m_proxy_source = pm_grid.m_proxy
        else:
            m_proxy_source = pm_grid.m
        if hasattr(pm_grid, "pol_vectors") and pm_grid.pol_vectors is not None:
            pol_vectors = _as_runtime_array(pm_grid.pol_vectors)
        else:
            pol_vectors = None
        return cls(
            A_obj=_as_runtime_array(pm_grid.A_obj),
            b_obj=_as_runtime_array(pm_grid.b_obj),
            ATb=_reshape_moments("ATb", pm_grid.ATb, ndipoles),
            ATA_scale=jnp.reshape(_as_runtime_array(pm_grid.ATA_scale), ()),
            m0=_reshape_moments("m0", pm_grid.m0, ndipoles),
            m=_reshape_moments("m", pm_grid.m, ndipoles),
            m_proxy=_reshape_moments("m_proxy", m_proxy_source, ndipoles),
            m_maxima=_as_runtime_array(pm_grid.m_maxima).reshape((ndipoles,)),
            dipole_grid_xyz=_as_runtime_array(pm_grid.dipole_grid_xyz),
            coordinate_flag=str(pm_grid.coordinate_flag),
            R0=float(pm_grid.R0),
            nfp=int(pm_grid.plasma_boundary.nfp),
            stellsym=bool(pm_grid.plasma_boundary.stellsym),
            nphi=int(pm_grid.nphi),
            ntheta=int(pm_grid.ntheta),
            ndipoles=ndipoles,
            pol_vectors=pol_vectors,
        )

    @classmethod
    def from_fixed_state(
        cls,
        *,
        plasma_points: object,
        normal: object,
        Bn: object,
        dipole_grid_xyz: object,
        m_maxima: object,
        nfp: int,
        stellsym: bool,
        coordinate_flag: str = "cartesian",
        R0: float = 0.0,
        m0: object | None = None,
        m: object | None = None,
        m_proxy: object | None = None,
        pol_vectors: object | None = None,
        nphi: int | None = None,
        ntheta: int | None = None,
    ) -> "PermanentMagnetGridJAX":
        """Build the fixed PM matrix from explicit host arrays.

        This mirrors ``PermanentMagnetGrid._optimization_setup`` after the host
        geometry/FAMUS setup has already produced plasma points, surface normals,
        and dipole locations.
        """

        points = _as_runtime_array(plasma_points)
        normal_arr = _as_runtime_array(normal)
        dipoles = _as_runtime_array(dipole_grid_xyz)
        ndipoles = int(dipoles.shape[0])
        Bn_host = np.asarray(Bn, dtype=_runtime_host_dtype())
        if nphi is None:
            nphi_value = int(Bn_host.shape[0])
        else:
            nphi_value = int(nphi)
        if ntheta is None:
            ntheta_value = int(Bn_host.shape[1])
        else:
            ntheta_value = int(ntheta)
        b_obj_unscaled = _as_runtime_array(-Bn_host.reshape(nphi_value * ntheta_value))
        normal_norms = jnp.linalg.norm(normal_arr, axis=1)
        unitnormal = normal_arr / normal_norms[:, None]

        A_raw = dipole_field_Bn(
            points,
            dipoles,
            unitnormal,
            nfp,
            stellsym,
            b_obj_unscaled,
            coordinate_flag,
            R0,
        )
        A_obj_unscaled = jnp.reshape(A_raw, (nphi_value * ntheta_value, ndipoles * 3))
        quadrature_count = _as_runtime_array(float(nphi_value * ntheta_value))
        scale = jnp.sqrt(normal_norms / quadrature_count.astype(normal_norms.dtype))
        A_obj = A_obj_unscaled * scale[:, None]
        b_obj = b_obj_unscaled * scale
        ATb = jnp.reshape(A_obj.T @ b_obj, (ndipoles, 3))
        singular_values = jnp.linalg.svd(A_obj, compute_uv=False)
        ATA_scale = singular_values[0] * singular_values[0]

        if m0 is None:
            m0_arr = dipoles - dipoles
        else:
            m0_arr = _reshape_moments("m0", m0, ndipoles)
        if m is None:
            m_arr = m0_arr
        else:
            m_arr = _reshape_moments("m", m, ndipoles)
        if m_proxy is None:
            m_proxy_arr = m_arr
        else:
            m_proxy_arr = _reshape_moments("m_proxy", m_proxy, ndipoles)
        if pol_vectors is None:
            pol_vectors_arr = None
        else:
            pol_vectors_arr = _as_runtime_array(pol_vectors)

        return cls(
            A_obj=A_obj,
            b_obj=b_obj,
            ATb=ATb,
            ATA_scale=ATA_scale,
            m0=m0_arr,
            m=m_arr,
            m_proxy=m_proxy_arr,
            m_maxima=_as_runtime_array(m_maxima).reshape((ndipoles,)),
            dipole_grid_xyz=dipoles,
            coordinate_flag=coordinate_flag,
            R0=float(R0),
            nfp=int(nfp),
            stellsym=bool(stellsym),
            nphi=nphi_value,
            ntheta=ntheta_value,
            ndipoles=ndipoles,
            pol_vectors=pol_vectors_arr,
        )


jax.tree_util.register_dataclass(
    PermanentMagnetGridJAX,
    data_fields=[
        "A_obj",
        "b_obj",
        "ATb",
        "ATA_scale",
        "m0",
        "m",
        "m_proxy",
        "m_maxima",
        "dipole_grid_xyz",
        "pol_vectors",
    ],
    meta_fields=[
        "coordinate_flag",
        "R0",
        "nfp",
        "stellsym",
        "nphi",
        "ntheta",
        "ndipoles",
    ],
)


def permanent_magnet_grid_to_jax(pm_grid) -> PermanentMagnetGridJAX:
    """Stage an existing CPU ``PermanentMagnetGrid`` as a JAX payload."""

    return PermanentMagnetGridJAX.from_cpu(pm_grid)


def mwpgp_alpha_from_grid(grid: PermanentMagnetGridJAX) -> jax.Array:
    """Return the upstream MwPGP step-size rule ``2 * (1 - 1e-5) / ATA_scale``."""

    numerator = _as_runtime_array(2.0 * MWPGP_ALPHA_SAFETY_FACTOR)
    return numerator.astype(grid.ATA_scale.dtype) / grid.ATA_scale
