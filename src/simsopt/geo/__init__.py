import jax
import sys

jax.config.update("jax_enable_x64", True)

from .config import *

from .curve import *
from .curvehelical import *
from .curverzfourier import *
from .curvexyzfourier import *
from .curvexyzfouriersymmetries import *
from .curveperturbed import *
from .curveobjectives import *
from .curveplanarfourier import *
from .framedcurve import *

from .finitebuild import *
from .plotting import *

from .boozersurface import *
from .qfmsurface import *
from .surface import *
from .surfacegarabedian import *
from .surfacehenneberg import *
from .surfaceobjectives import *
from .surfacerzfourier import *
from .surfacexyzfourier import *
from .surfacexyztensorfourier import *
from .strain_optimization import *
from .hull import *
from .wireframe_toroidal import *
from .ports import *

from .permanent_magnet_grid import *
from .orientedcurve import *
from .accessibility import *
from .curvecwsfourier import *


def _module_all(name):
    return list(sys.modules[f"{__name__}.{name}"].__all__)


__all__ = (
    _module_all("curve")
    + _module_all("curvehelical")
    + _module_all("curverzfourier")
    + _module_all("curvexyzfourier")
    + _module_all("curvexyzfouriersymmetries")
    + _module_all("curveperturbed")
    + _module_all("curveobjectives")
    + _module_all("curveplanarfourier")
    + _module_all("finitebuild")
    + _module_all("plotting")
    + _module_all("boozersurface")
    + _module_all("qfmsurface")
    + _module_all("surface")
    + _module_all("surfacegarabedian")
    + _module_all("surfacehenneberg")
    + _module_all("surfacerzfourier")
    + _module_all("surfacexyzfourier")
    + _module_all("surfacexyztensorfourier")
    + _module_all("surfaceobjectives")
    + _module_all("permanent_magnet_grid")
    + _module_all("orientedcurve")
    + _module_all("strain_optimization")
    + _module_all("framedcurve")
    + _module_all("hull")
    + _module_all("accessibility")
    + _module_all("curvecwsfourier")
    + _module_all("wireframe_toroidal")
    + _module_all("ports")
)
