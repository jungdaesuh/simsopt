"""Pure JAX MHD helpers."""

from simsopt_jax.core.mhd_bootstrap import compute_trapped_fraction_jax
from simsopt_jax.core.redl_current import RedlDetailsJAX, j_dot_B_Redl_jax_from_arrays
from .vmec_diagnostics import (
    VmecFrozenSplineState,
    VmecGeometryResultsJAX,
    vmec_compute_geometry_jax,
    vmec_fieldlines_jax,
    vmec_freeze_splines,
)

__all__ = (
    "RedlDetailsJAX",
    "VmecFrozenSplineState",
    "VmecGeometryResultsJAX",
    "compute_trapped_fraction_jax",
    "j_dot_B_Redl_jax_from_arrays",
    "vmec_compute_geometry_jax",
    "vmec_fieldlines_jax",
    "vmec_freeze_splines",
)
