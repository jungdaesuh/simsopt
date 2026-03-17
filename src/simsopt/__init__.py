# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================

# JAX platform config — must run before any submodule imports JAX.
# Placing it here (package root) guarantees it executes before simsopt.field,
# simsopt.geo, or any other subpackage that does `import jax` at module level.
import os as _os

_VALID_JAX_BACKENDS = {"cpu", "cuda"}
_JAX_BACKEND = _os.environ.get("SIMSOPT_JAX_BACKEND", "cpu")
if _JAX_BACKEND not in _VALID_JAX_BACKENDS:
    raise ValueError(
        f"SIMSOPT_JAX_BACKEND={_JAX_BACKEND!r} is not valid. "
        f"Accepted values: {sorted(_VALID_JAX_BACKENDS)}"
    )

import jax as _jax

_jax.config.update("jax_platforms", _JAX_BACKEND)
del _os, _jax

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

from ._core import make_optimizable, load, save

# VERSION info
from ._version import version as __version__

# Expose XSIMD depedency in simsoptpp
from simsoptpp import using_xsimd as __built_with_xsimd__
