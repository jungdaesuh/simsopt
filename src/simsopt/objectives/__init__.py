from .constrained import *
from .least_squares import *
from .utilities import *

# Check simsoptpp availability once; probe a compiled symbol to
# distinguish the real extension from the src/simsoptpp/ namespace package.
try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False

_cpu_flux_all = []
if _has_simsoptpp:
    from .fluxobjective import *

    _cpu_flux_all = fluxobjective.__all__

# JAX modules (optional — requires jax)
_jax_flux_all = []
try:
    from .fluxobjective_jax import *

    _jax_flux_all = fluxobjective_jax.__all__
except (ImportError, AttributeError):
    pass

__all__ = (
    _cpu_flux_all
    + _jax_flux_all
    + least_squares.__all__
    + utilities.__all__
    + constrained.__all__
)
