"""JSON compatibility helpers for examples-side finite-current artifacts."""

from simsopt._core.json import GSONDecoder
from simsopt._core.optimizable import load
from simsopt.geo.boozersurface import BoozerSurface

from .boozer_finite_current import BoozerSurfaceFiniteI


_LEGACY_BOOZER_SURFACE_MODULES = {
    "simsopt.geo.boozersurface",
    "simsopt.geo",
}


def _rewrite_legacy_boozer_surface(value):
    if isinstance(value, list):
        for item in value:
            _rewrite_legacy_boozer_surface(item)
        return value
    if not isinstance(value, dict):
        return value

    for item in value.values():
        _rewrite_legacy_boozer_surface(item)
    if (
        value.get("@module") in _LEGACY_BOOZER_SURFACE_MODULES
        and value.get("@class") == "BoozerSurface"
        and "I" in value
    ):
        value["@module"] = BoozerSurfaceFiniteI.__module__
        value["@class"] = BoozerSurfaceFiniteI.__name__
    return value


class BoozerFiniteIDecoder(GSONDecoder):
    """Decode legacy ``BoozerSurface(I=...)`` JSON as ``BoozerSurfaceFiniteI``."""

    def process_decoded(self, d, serial_objs_dict=None, recon_objs=None):
        return super().process_decoded(
            _rewrite_legacy_boozer_surface(d),
            serial_objs_dict,
            recon_objs,
        )


def load_boozer_finite_i(filename, *args, **kwargs):
    kwargs.setdefault("cls", BoozerFiniteIDecoder)
    return load(filename, *args, **kwargs)


def save_boozer_finite_i(boozer_surface, filename) -> None:
    """Sanctioned save for Boozer-surface artifacts -- the counterpart to
    :func:`load_boozer_finite_i`.

    Refuses to write a *surface-only* artifact: ``boozer_surface`` must be a
    :class:`~simsopt.geo.boozersurface.BoozerSurface` (the finite-I subclass
    included) whose ``biotsavart`` field carries at least one coil, so the saved
    SIMSON graph embeds the ``BiotSavart`` + coils that the clearance viewer and
    the topology/Poincare consumers render and trace. Saving a bare ``.surface``
    under a ``*_boozer_surface.json`` name yields a graph with no ``BiotSavart``
    node -- the recurring viewer ``NoBiotSavart`` failure this guard prevents.
    """
    field = getattr(boozer_surface, "biotsavart", None)
    coils = getattr(field, "coils", None)
    if not isinstance(boozer_surface, BoozerSurface) or not coils:
        raise ValueError(
            "save_boozer_finite_i requires a BoozerSurface embedding a BiotSavart "
            f"with >=1 coil; got {type(boozer_surface).__name__} carrying "
            f"{0 if not coils else len(coils)} coils. Save the BoozerSurface "
            "itself, not its `.surface`."
        )
    boozer_surface.save(str(filename))
