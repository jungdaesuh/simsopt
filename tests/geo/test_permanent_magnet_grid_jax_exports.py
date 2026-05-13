"""Package export checks for the JAX permanent-magnet grid wrapper."""

from __future__ import annotations

import simsopt.geo as geo
from simsopt.geo import permanent_magnet_grid_jax


def test_permanent_magnet_grid_jax_is_public_geo_export():
    assert (
        geo.PermanentMagnetGridJAX is permanent_magnet_grid_jax.PermanentMagnetGridJAX
    )
    assert (
        geo.permanent_magnet_grid_to_jax
        is permanent_magnet_grid_jax.permanent_magnet_grid_to_jax
    )
