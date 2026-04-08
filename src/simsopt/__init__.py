# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================

# JAX runtime config.
#
# We keep the global x64 toggle at package import time so existing scientific
# code paths and optimizer smoke tests continue to see the expected float64
# runtime. Platform pinning still routes through the backend selector.
#
# Important: this import hook is recovery-oriented, not the primary launcher
# contract. Scripts that import ``jax`` directly must set platform/backend env
# vars before their first ``import jax`` so JAX initializes on the intended
# runtime. Importing ``simsopt`` can eagerly validate an explicit selector, but
# it cannot retroactively fix a backend that was already initialized too early.
try:
    import os as _os
    import sys as _sys

    from .backend import (
        apply_jax_runtime_config as _apply_jax_runtime_config,
        should_eagerly_configure_jax as _should_eagerly_configure_jax,
    )

    if "jax" in _sys.modules:
        import jax as _jax

        _jax.config.update("jax_enable_x64", True)
        del _jax
    else:
        _os.environ.setdefault("JAX_ENABLE_X64", "True")

    if _should_eagerly_configure_jax():
        _apply_jax_runtime_config()

    del _apply_jax_runtime_config, _should_eagerly_configure_jax, _os, _sys
except ImportError:
    pass

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

from ._core import make_optimizable, load, save

# VERSION info
try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0+unknown"

# Expose XSIMD dependency in simsoptpp (optional: absent in JAX-only envs)
try:
    from simsoptpp import using_xsimd as __built_with_xsimd__
except (ImportError, AttributeError):
    __built_with_xsimd__ = False
