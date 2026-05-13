import sys

from .serial import *
from .mpi import *
from .permanent_magnet_optimization import *
from .wireframe_optimization import *

try:
    import jax as _  # noqa: F401

    _has_jax = True
except ImportError:
    _has_jax = False

if _has_jax:
    from .permanent_magnet_optimization_jax import *
    from .wireframe_optimization_jax import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("serial")
    + _module_all("mpi")
    + _module_all("permanent_magnet_optimization")
    + _module_all("wireframe_optimization")
)
if _has_jax:
    __all__ += _module_all("permanent_magnet_optimization_jax") + _module_all(
        "wireframe_optimization_jax"
    )
