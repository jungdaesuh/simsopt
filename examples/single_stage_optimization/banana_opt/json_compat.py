"""JSON compatibility helpers for examples-side finite-current artifacts."""

from simsopt._core.json import GSONDecoder
from simsopt._core.optimizable import load

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
