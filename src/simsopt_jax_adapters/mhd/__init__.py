"""Legacy MHD object/profile adapters for ``simsopt_jax``."""

from simsopt_jax.mhd import (
    RedlDetailsJAX,
    compute_trapped_fraction_jax,
    j_dot_B_Redl_jax_from_arrays,
)

from .bootstrap import RedlBootstrapJAX, j_dot_B_Redl_jax
from .profiles import (
    ProfilePolynomialJAX,
    ProfilePressureJAX,
    ProfileScaledJAX,
    ProfileSplineJAX,
)
from .vmec_diagnostics import (
    VmecFrozenSplineState,
    VmecGeometryResultsJAX,
    vmec_compute_geometry_jax,
    vmec_fieldlines_jax,
    vmec_freeze_splines,
)
from .vmec_host import (
    VmecHostEvaluation,
    boundary_sha256,
    validate_vmec_host_evaluation,
)

__all__ = (
    "ProfilePolynomialJAX",
    "ProfilePressureJAX",
    "ProfileScaledJAX",
    "ProfileSplineJAX",
    "RedlBootstrapJAX",
    "RedlDetailsJAX",
    "VmecFrozenSplineState",
    "VmecGeometryResultsJAX",
    "VmecHostEvaluation",
    "boundary_sha256",
    "compute_trapped_fraction_jax",
    "j_dot_B_Redl_jax",
    "j_dot_B_Redl_jax_from_arrays",
    "validate_vmec_host_evaluation",
    "vmec_compute_geometry_jax",
    "vmec_fieldlines_jax",
    "vmec_freeze_splines",
)
