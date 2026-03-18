# Check simsoptpp availability once; probe a compiled symbol to
# distinguish the real extension from the src/simsoptpp/ namespace package.
try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False

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

    _cpu_field_all = (
        biotsavart.__all__
        + boozermagneticfield.__all__
        + coil.__all__
        + coilset.__all__
        + magneticfield.__all__
        + magneticfieldclasses.__all__
        + mgrid.__all__
        + normal_field.__all__
        + tracing.__all__
        + coilobjective.__all__
        + wireframefield.__all__
        + selffield.__all__
        + magnetic_axis_helpers.__all__
    )

# JAX modules (optional — requires jax)
_jax_field_all = []
try:
    from .biotsavart_jax_backend import *

    _jax_field_all = biotsavart_jax_backend.__all__
except (ImportError, AttributeError):
    pass

__all__ = _cpu_field_all + _jax_field_all
