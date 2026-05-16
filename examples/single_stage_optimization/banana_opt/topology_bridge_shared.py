"""Shared infrastructure for the Phase 3 topology-bridge modules.

The Phase 3 plan (``autoresearch/docs/stage2_single_stage_boozer_handoff_impl_plan_2026-05-15.md``)
splits the topology-bridge diagnostics into two siblings:

- :mod:`banana_opt.topology_bridge` — Phase 3a, the live in-loop field-line
  iota proxy with structured failure tags (``FAILURE_ESCAPED`` etc.).
- :mod:`banana_opt.boozer_topology_bridge` — Phase 3b, the helical-content
  ``S_HEL`` objective and the convergence-validated proxy variant.

Both modules depend on identical coordinate-axis adapter geometry and on the
same tmax/tol/nfieldlines defaults. This module is the single source of truth
for that shared infrastructure so the two siblings do not drift apart.
"""
from __future__ import annotations

import numpy as np


__all__ = [
    "DEFAULT_NFIELDLINES",
    "DEFAULT_TMAX",
    "DEFAULT_TOL",
    "SurfaceCentroidCoordinateAxis",
    "build_surface_centroid_axis",
]


# Plan line 273: "start with tol=1e-8".
DEFAULT_TOL = 1.0e-8

# Plan line 271: "hard tmax cap". The number of toroidal revolutions
# completed by a confined field line during ``tmax`` is approximately
# ``|B| * tmax / (2 * pi * R0)`` (the integrator's time variable is arc
# length divided by |B|; the toroidal arc length per revolution is
# ``2 * pi * R0``). The HBT-EP campaign sits at R0 ~ 1 m and |B| ~ 0.4 T
# on the working surface, so ``tmax = 3000`` yields roughly 190
# revolutions per line — comfortably above the ``n_transits_target =
# 100`` rung the post-solve diagnostic uses for convergence verdicts,
# while remaining well below the cost ceiling for a per-checkpoint
# diagnostic (a single full sweep on a healthy field is sub-minute at
# this scale). Larger devices (R0 ~ 2 m) achieve ~95 revolutions at the
# same default and the caller can override via ``--topology-bridge-tmax``
# when needed. Smaller-aspect-ratio bench tests (R0 ~ 0.3 m) achieve
# ~600+ revolutions, well past the convergence floor.
DEFAULT_TMAX = 3000.0
DEFAULT_NFIELDLINES = 5


class SurfaceCentroidCoordinateAxis:
    """Coordinate axis adapter required by SIMSOPT poloidal-transit counting.

    The axis is sampled once at module-build time as the centroid of
    ``surface.cross_section`` over a uniform ``phi`` grid; ``gamma_impl``
    then linearly interpolates that centroid loop for any requested ``phi``
    (parameterized as a fraction in ``[0, 1)``).
    """

    def __init__(self, centers: np.ndarray) -> None:
        self._centers = centers
        self._nphi = int(centers.shape[0])

    def gamma_impl(self, gamma: np.ndarray, phi: float) -> None:
        phase = float(phi) % 1.0
        position = phase * self._nphi
        index = int(np.floor(position)) % self._nphi
        fraction = position - np.floor(position)
        center = (
            (1.0 - fraction) * self._centers[index]
            + fraction * self._centers[(index + 1) % self._nphi]
        )
        gamma[0, :] = center


def build_surface_centroid_axis(
    surface,
    *,
    nphi: int = 128,
    ntheta: int = 64,
) -> SurfaceCentroidCoordinateAxis:
    """Sample the surface centroid loop on a uniform ``phi`` grid.

    Returns a :class:`SurfaceCentroidCoordinateAxis` whose ``gamma_impl``
    interpolates the sampled centroid loop linearly between adjacent ``phi``
    knots. Used by both Phase 3a (live diagnostic) and Phase 3b (convergence-
    validated diagnostic) so the two reports describe the same magnetic-axis
    proxy.
    """
    phis = np.linspace(0.0, 1.0, int(nphi), endpoint=False)
    centers = np.asarray(
        [
            np.mean(
                np.asarray(
                    surface.cross_section(float(phi), thetas=int(ntheta)),
                    dtype=float,
                ),
                axis=0,
            )
            for phi in phis
        ],
        dtype=float,
    )
    return SurfaceCentroidCoordinateAxis(centers)
