import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np


def curve_poloidal_half_extent(curve, winding_major_radius, winding_z=0.0):
    """Return max inboard poloidal angle magnitude for one banana curve."""
    gamma = np.asarray(curve.gamma())
    cylindrical_r = np.linalg.norm(gamma[:, :2], axis=-1)
    theta_inboard = np.arctan2(
        gamma[:, 2] - float(winding_z),
        -(cylindrical_r - float(winding_major_radius)),
    )
    return float(np.max(np.abs(theta_inboard)))


def surface_vessel_clearance(surface, vessel_major_radius, vessel_minor_radius):
    """Return minimum radial clearance to the concentric circular HBT vessel."""
    gamma = np.asarray(surface.gamma())
    cylindrical_r = np.linalg.norm(gamma[..., :2], axis=-1)
    cross_section_r = np.sqrt(
        (cylindrical_r - float(vessel_major_radius)) ** 2 + gamma[..., 2] ** 2
    )
    return float(np.min(float(vessel_minor_radius) - cross_section_r))


def surface_shape_metrics(surface):
    """Return scalar LCFS radius and volume metrics used by new-HW ranking."""
    return {
        "final_volume": float(surface.volume()),
        "surface_major_radius_m": float(surface.major_radius()),
        "surface_minor_radius_m": float(surface.minor_radius()),
    }


def boozer_json_vacuum_lineage(path):
    """Validate saved Boozer JSON lineage for the vacuum-current campaign."""
    json_path = Path(path)
    data = json.loads(json_path.read_text())
    boozer_objects = [
        node for node in _walk_mappings(data)
        if node.get("@class") == "BoozerSurface"
    ]
    plain_boozer_surface = any(
        node.get("@module") == "simsopt.geo.boozersurface"
        for node in boozer_objects
    )
    references_finite_i = _contains_string(data, "BoozerSurfaceFiniteI")
    has_i_field = _contains_key(data, "I")
    return {
        "vacuum_lineage_ok": bool(
            plain_boozer_surface and not references_finite_i and not has_i_field
        ),
        "boozer_surface_class": "BoozerSurface" if plain_boozer_surface else None,
        "boozer_surface_module": (
            "simsopt.geo.boozersurface" if plain_boozer_surface else None
        ),
        "boozer_json_has_I_field": bool(has_i_field),
        "boozer_json_references_finite_i": bool(references_finite_i),
    }


def _walk_mappings(node) -> Iterator[Mapping[str, object]]:
    if isinstance(node, Mapping):
        yield node
        for value in node.values():
            yield from _walk_mappings(value)
    elif isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        for value in node:
            yield from _walk_mappings(value)


def _contains_key(node, key):
    if isinstance(node, Mapping):
        if key in node:
            return True
        return any(_contains_key(value, key) for value in node.values())
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return any(_contains_key(value, key) for value in node)
    return False


def _contains_string(node, needle):
    if isinstance(node, str):
        return needle in node
    if isinstance(node, Mapping):
        return any(
            _contains_string(key, needle) or _contains_string(value, needle)
            for key, value in node.items()
        )
    if isinstance(node, Sequence) and not isinstance(node, (str, bytes, bytearray)):
        return any(_contains_string(value, needle) for value in node)
    return False
