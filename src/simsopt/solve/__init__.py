import sys

from .serial import *
from .mpi import *
from .permanent_magnet_optimization import *
from .wireframe_optimization import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("serial")
    + _module_all("mpi")
    + _module_all("permanent_magnet_optimization")
    + _module_all("wireframe_optimization")
)
