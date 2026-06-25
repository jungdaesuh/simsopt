# ===================ATTENTION=================================================
# Don't abuse this file by importing all variables from all modules to top-level.
# Import only the important classes that should be at top-level.
# Follow the same logic in the sub-packages.
# ===================END ATTENTION=============================================

# Two ways of achieving the above-mentioned objective
# Use "from xyz import XYZ" style
# Define __all__ dunder at module and subpackage level. Then you could do
# "from xyz import *".  If xyz[.py] contains __all__ = ['XYZ'], only XYZ is
# imported

from ._core import make_optimizable, load, save

# VERSION info. Editable/source checkouts may not have the setuptools_scm
# generated file until build time; keep raw source imports usable.
try:
    from ._version import version as __version__
except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}._version":
        raise
    __version__ = "0.0.dev0+source"

# Expose XSIMD depedency in simsoptpp
from simsoptpp import using_xsimd as __built_with_xsimd__
