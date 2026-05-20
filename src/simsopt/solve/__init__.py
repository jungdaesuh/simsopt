import importlib.util

from .._lazy_exports import build_lazy_export_map, resolve_lazy_export

try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False

_has_jax = importlib.util.find_spec("jax") is not None

_CORE_SOLVE_MODULES = ("serial",)
_SIMSOPTPP_SOLVE_MODULES = (
    "mpi",
    "permanent_magnet_optimization",
    "wireframe_optimization",
)
_JAX_SOLVE_MODULES = (
    "serial_jax",
    "mpi_jax",
    "permanent_magnet_optimization_jax",
)
_SIMSOPTPP_JAX_SOLVE_MODULES = ("wireframe_optimization_jax",)

_solve_modules = _CORE_SOLVE_MODULES
if _has_simsoptpp:
    _solve_modules += _SIMSOPTPP_SOLVE_MODULES
if _has_jax:
    _solve_modules += _JAX_SOLVE_MODULES
if _has_simsoptpp and _has_jax:
    _solve_modules += _SIMSOPTPP_JAX_SOLVE_MODULES

_EXPORT_TO_MODULE, __all__ = build_lazy_export_map(__file__, _solve_modules)


def __getattr__(name):
    value = resolve_lazy_export(__name__, _EXPORT_TO_MODULE, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
