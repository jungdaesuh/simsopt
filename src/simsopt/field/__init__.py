from .._lazy_exports import build_lazy_export_map, resolve_lazy_export

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
except (ImportError, AttributeError):
    _has_jax = False

_CPU_FIELD_MODULES = (
    "biotsavart",
    "boozermagneticfield",
    "coil",
    "coilset",
    "magneticfield",
    "magneticfieldclasses",
    "mgrid",
    "normal_field",
    "tracing",
    "coilobjective",
    "wireframefield",
    "selffield",
    "magnetic_axis_helpers",
    "force",
)
_JAX_FIELD_MODULES = ("biotsavart_jax_backend",)
_JAX_FIELD_SIMSOPTPP_MODULES = ("magneticfieldclasses_jax",)

_cpu_field_modules = _CPU_FIELD_MODULES if _has_simsoptpp else ()
_jax_field_modules = _JAX_FIELD_MODULES if _has_jax else ()
_jax_simsoptpp_field_modules = (
    _JAX_FIELD_SIMSOPTPP_MODULES if _has_jax and _has_simsoptpp else ()
)
_field_modules = _cpu_field_modules + _jax_field_modules + _jax_simsoptpp_field_modules

_EXPORT_TO_MODULE, __all__ = build_lazy_export_map(__file__, _field_modules)


def __getattr__(name):
    value = resolve_lazy_export(__name__, _EXPORT_TO_MODULE, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
