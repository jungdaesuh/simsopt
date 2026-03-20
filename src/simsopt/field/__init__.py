import sys

from .biotsavart import *
from .boozermagneticfield import *
from .coil import *
from .coilset import *
from .magneticfield import *
from .magneticfieldclasses import *
from .mgrid import *
from .normal_field import *
from .tracing import *
from .coilobjective import *
from .wireframefield import *
from .selffield import *
from .magnetic_axis_helpers import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("biotsavart")
    + _module_all("boozermagneticfield")
    + _module_all("coil")
    + _module_all("coilset")
    + _module_all("magneticfield")
    + _module_all("magneticfieldclasses")
    + _module_all("mgrid")
    + _module_all("normal_field")
    + _module_all("tracing")
    + _module_all("coilobjective")
    + _module_all("wireframefield")
    + _module_all("selffield")
    + _module_all("magnetic_axis_helpers")
)
