"""Canonical dependency ownership for curve-surface distance objectives."""

from typing import Sequence, Tuple

from simsopt._core.optimizable import Optimizable


def curve_surface_distance_owners(
    curves: Sequence[Optimizable],
    surface: Optimizable,
) -> Tuple[Optimizable, ...]:
    """Return curve owners followed by the exact surface owner."""
    return (*curves, surface)
