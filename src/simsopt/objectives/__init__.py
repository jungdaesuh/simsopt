import sys

from .constrained import *
from .least_squares import *
from .utilities import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)

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

    _cpu_flux_all = _module_all("fluxobjective")

# JAX modules (optional — requires jax)
_jax_flux_all = []
try:
    from .fluxobjective_jax import *

    _jax_flux_all = _module_all("fluxobjective_jax")
except (ImportError, AttributeError):
    pass

__all__ = (
    _cpu_flux_all
    + _jax_flux_all
    + _module_all("least_squares")
    + _module_all("utilities")
    + _module_all("constrained")
)


def __getattr__(name):
    if name != "SquaredFluxJAX":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .fluxobjective_jax import SquaredFluxJAX

    globals()[name] = SquaredFluxJAX
    return SquaredFluxJAX
