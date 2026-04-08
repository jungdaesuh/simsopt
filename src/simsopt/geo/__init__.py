import importlib

from .._lazy_exports import build_lazy_export_map, resolve_lazy_export
from .config import parameters

# Check simsoptpp availability once; probe a compiled symbol to
# distinguish the real extension from the src/simsoptpp/ namespace package.
try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False

try:
    import jax as _  # noqa: F401

    _has_jax = True
except ImportError:
    _has_jax = False

_BASE_CPU_GEO_MODULES = (
    "curve",
    "curverzfourier",
    "curvexyzfourier",
    "curveperturbed",
    "curveplanarfourier",
    "plotting",
    "boozersurface",
    "qfmsurface",
    "surface",
    "surfacegarabedian",
    "surfacehenneberg",
    "surfaceobjectives",
    "surfacerzfourier",
    "surfacexyzfourier",
    "surfacexyztensorfourier",
    "hull",
    "wireframe_toroidal",
    "ports",
    "permanent_magnet_grid",
)
_ORDERED_JAX_CPU_GEO_BLOCK = (
    "curvehelical",
    "curvexyzfouriersymmetries",
    "curveobjectives",
    "framedcurve",
    "finitebuild",
    "strain_optimization",
    "orientedcurve",
    "accessibility",
    "curvecwsfourier",
)
_JAX_GEO_MODULES = ("boozersurface_jax",)
_SIMSOPT_JAX_GEO_MODULES = ("surfaceobjectives_jax",)
_DYNAMIC_JAX_EXPORTS = {
    "CurveCWSFourier": "curve",
    "SurfaceSurfaceDistance": "surfaceobjectives",
}

_cpu_geo_modules = ()
if _has_simsoptpp:
    _cpu_geo_modules = _BASE_CPU_GEO_MODULES
    if _has_jax:
        _cpu_geo_modules += _ORDERED_JAX_CPU_GEO_BLOCK

_jax_geo_modules = _JAX_GEO_MODULES if _has_jax else ()
if _has_simsoptpp and _has_jax:
    _jax_geo_modules += _SIMSOPT_JAX_GEO_MODULES

_EXPORT_TO_MODULE, _lazy_all = build_lazy_export_map(
    __file__, _cpu_geo_modules + _jax_geo_modules
)
__all__ = list(_lazy_all)
if _has_simsoptpp and _has_jax:
    for export_name, module_name in _DYNAMIC_JAX_EXPORTS.items():
        _EXPORT_TO_MODULE[export_name] = module_name
        __all__.append(export_name)


def _import_geo_block_through(target_module):
    for module_name in _ORDERED_JAX_CPU_GEO_BLOCK:
        importlib.import_module(f".{module_name}", __name__)
        if module_name == target_module:
            return


def __getattr__(name):
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if module_name in _ORDERED_JAX_CPU_GEO_BLOCK:
        _import_geo_block_through(module_name)
    value = resolve_lazy_export(__name__, _EXPORT_TO_MODULE, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
