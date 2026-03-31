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
try:
    from .backend import (
        apply_jax_runtime_config as _apply_jax_runtime_config,
        should_eagerly_configure_jax as _should_eagerly_configure_jax,
    )
    import jax as _jax

    _jax.config.update("jax_enable_x64", True)
    if _should_eagerly_configure_jax():
        _apply_jax_runtime_config()
    del _apply_jax_runtime_config, _should_eagerly_configure_jax, _jax
except ImportError:
    pass

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

from ._core import make_optimizable, load, save

# VERSION info
from ._version import version as __version__

# Expose XSIMD dependency in simsoptpp (optional: absent in JAX-only envs)
try:
    from simsoptpp import using_xsimd as __built_with_xsimd__
except (ImportError, AttributeError):
    __built_with_xsimd__ = False
