# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================

# JAX runtime config.
#
# Package import validates pre-existing CUDA determinism settings, then applies
# JAX runtime policy only when an explicit backend selector asks for a JAX lane.
# Unconfigured imports do not mutate JAX x64; callers that need runtime
# configuration should set a backend selector or call ``simsopt.config`` helpers.
#
# Important: this import hook is recovery-oriented, not the primary launcher
# contract. Scripts that import ``jax`` directly must set platform/backend env
# vars before their first ``import jax`` so JAX initializes on the intended
# runtime. Importing ``simsopt`` can eagerly validate an explicit selector, but
# it cannot retroactively fix a backend that was already initialized too early.

from .backend import (
    apply_jax_runtime_config as _apply_jax_runtime_config,
    should_eagerly_configure_jax as _should_eagerly_configure_jax,
    validate_cuda_determinism_environment as _validate_cuda_determinism_environment,
)

_validate_cuda_determinism_environment()

if _should_eagerly_configure_jax():
    _apply_jax_runtime_config()

del (
    _apply_jax_runtime_config,
    _should_eagerly_configure_jax,
    _validate_cuda_determinism_environment,
)

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

_CORE_EXPORTS = frozenset(("make_optimizable", "load", "save"))

__all__ = [
    "make_optimizable",
    "load",
    "save",
    "__version__",
    "__built_with_xsimd__",
]

# VERSION info
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

def __getattr__(name):
    if name in _CORE_EXPORTS:
        from . import _core as _core_mod

        value = getattr(_core_mod, name)
        globals()[name] = value
        return value
    if name == "__built_with_xsimd__":
        try:
            from simsoptpp import using_xsimd as value
        except (ImportError, AttributeError):
            value = False
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
