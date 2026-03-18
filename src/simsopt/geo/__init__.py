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
    from .boozersurface_jax import *

    _jax_geo_all += boozersurface_jax.__all__
except (ImportError, AttributeError):
    pass
try:
    from .surfaceobjectives_jax import *

    _jax_geo_all += surfaceobjectives_jax.__all__
except (ImportError, AttributeError):
    pass

__all__ = _cpu_geo_all + _jax_geo_all
