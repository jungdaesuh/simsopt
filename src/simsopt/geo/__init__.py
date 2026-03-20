from . import config as config
from .config import *

# Check simsoptpp availability once; probe a compiled symbol to
# distinguish the real extension from the src/simsoptpp/ namespace package.
try:
    from simsoptpp import Curve as _  # noqa: F401

    _has_simsoptpp = True
except (ImportError, AttributeError):
    _has_simsoptpp = False

_cpu_geo_all = []
if _has_simsoptpp:
    from . import curve as curve
    from . import curvehelical as curvehelical
    from . import curverzfourier as curverzfourier
    from . import curvexyzfourier as curvexyzfourier
    from . import curvexyzfouriersymmetries as curvexyzfouriersymmetries
    from . import curveperturbed as curveperturbed
    from . import curveobjectives as curveobjectives
    from . import curveplanarfourier as curveplanarfourier
    from . import framedcurve as framedcurve
    from . import finitebuild as finitebuild
    from . import plotting as plotting
    from . import boozersurface as boozersurface
    from . import qfmsurface as qfmsurface
    from . import surface as surface
    from . import surfacegarabedian as surfacegarabedian
    from . import surfacehenneberg as surfacehenneberg
    from . import surfaceobjectives as surfaceobjectives
    from . import surfacerzfourier as surfacerzfourier
    from . import surfacexyzfourier as surfacexyzfourier
    from . import surfacexyztensorfourier as surfacexyztensorfourier
    from . import strain_optimization as strain_optimization
    from . import hull as hull
    from . import wireframe_toroidal as wireframe_toroidal
    from . import ports as ports
    from . import permanent_magnet_grid as permanent_magnet_grid
    from . import orientedcurve as orientedcurve
    from . import accessibility as accessibility
    from . import curvecwsfourier as curvecwsfourier

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

    _cpu_geo_all = (
        curve.__all__
        + curvehelical.__all__
        + curverzfourier.__all__
        + curvexyzfourier.__all__
        + curvexyzfouriersymmetries.__all__
        + curveperturbed.__all__
        + curveobjectives.__all__
        + curveplanarfourier.__all__
        + finitebuild.__all__
        + plotting.__all__
        + boozersurface.__all__
        + qfmsurface.__all__
        + surface.__all__
        + surfacegarabedian.__all__
        + surfacehenneberg.__all__
        + surfacerzfourier.__all__
        + surfacexyzfourier.__all__
        + surfacexyztensorfourier.__all__
        + surfaceobjectives.__all__
        + permanent_magnet_grid.__all__
        + orientedcurve.__all__
        + strain_optimization.__all__
        + framedcurve.__all__
        + hull.__all__
        + accessibility.__all__
        + curvecwsfourier.__all__
        + wireframe_toroidal.__all__
        + ports.__all__
    )

# JAX modules (optional — requires jax)
_jax_geo_all = []
try:
    from . import boozersurface_jax as boozersurface_jax
    from .boozersurface_jax import *

    _jax_geo_all += boozersurface_jax.__all__
except (ImportError, AttributeError):
    pass
if _has_simsoptpp:
    try:
        from . import surfaceobjectives_jax as surfaceobjectives_jax
        from .surfaceobjectives_jax import *

        _jax_geo_all += surfaceobjectives_jax.__all__
    except (ImportError, AttributeError):
        pass

__all__ = _cpu_geo_all + _jax_geo_all
