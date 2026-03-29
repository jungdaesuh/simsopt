import sys

# Check simsoptpp availability once; probe a compiled symbol to
# distinguish the real extension from the src/simsoptpp/ namespace package.
try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


_cpu_field_all = []
if _has_simsoptpp:
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
    from .force import *

    _cpu_field_all = (
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
        + _module_all("force")
    )

# JAX modules (optional — requires jax)
_jax_field_all = []
try:
    from .biotsavart_jax_backend import *

    _jax_field_all = _module_all("biotsavart_jax_backend")
except (ImportError, AttributeError):
    pass

__all__ = _cpu_field_all + _jax_field_all
