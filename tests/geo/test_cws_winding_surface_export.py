"""Realized CWS winding-torus export tests (2026-06-10 laneR0920 fix).

Banana coils are ``CurveCWSFourierCPP`` angle-dofs pinned to the winding torus
serialized inside the curve. Results artifacts must record THAT torus — not
the 0.903 spec constant and not the ``--banana-surf-major-radius`` reference —
because warm resumes optimize on the embedded surface (the M-family lineage is
embedded on 0.976/0.21). These tests pin ``realized_cws_winding_radii``:
realized values for CWS lineages, ``None`` fallback for non-CWS coil sets.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.coil_order_upgrade import realized_cws_winding_radii  # noqa: E402

from simsopt.field import Coil, Current  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
)


def _build_cws_coil(major_radius: float, minor_radius: float) -> Coil:
    surface = SurfaceRZFourier(nfp=5, stellsym=True)
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, minor_radius)
    surface.set_zs(1, 0, minor_radius)
    quadpoints = np.linspace(0.0, 1.0, 32, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order=2, surf=surface, G=3, H=5)
    curve.set("phic(0)", 0.23)
    curve.set("thetac(0)", 0.41)
    curve.set("thetas(1)", 0.09)
    return Coil(curve, Current(1.1e4))


def _build_xyz_coil() -> Coil:
    curve = CurveXYZFourier(np.linspace(0.0, 1.0, 32, endpoint=False), order=1)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    return Coil(curve, Current(1.0e4))


class RealizedCwsWindingRadiiTest(unittest.TestCase):
    def test_reports_embedded_torus_not_spec_constant(self) -> None:
        coil = _build_cws_coil(0.976, 0.21)
        radii = realized_cws_winding_radii([coil])
        self.assertIsNotNone(radii)
        self.assertAlmostEqual(radii[0], 0.976, places=12)
        self.assertAlmostEqual(radii[1], 0.21, places=12)

    def test_reports_recentered_torus(self) -> None:
        coil = _build_cws_coil(0.993, 0.21)
        radii = realized_cws_winding_radii([coil])
        self.assertIsNotNone(radii)
        self.assertAlmostEqual(radii[0], 0.993, places=12)

    def test_master_resolved_among_symmetry_wrappers(self) -> None:
        # Loaded graphs interleave the CWS master with non-CWS symmetry
        # wrapper curves; the master wins regardless of position.
        coils = [_build_xyz_coil(), _build_cws_coil(0.976, 0.21)]
        radii = realized_cws_winding_radii(coils)
        self.assertIsNotNone(radii)
        self.assertAlmostEqual(radii[0], 0.976, places=12)

    def test_non_cws_lineage_returns_none(self) -> None:
        self.assertIsNone(realized_cws_winding_radii([_build_xyz_coil()]))

    def test_empty_returns_none(self) -> None:
        self.assertIsNone(realized_cws_winding_radii([]))


if __name__ == "__main__":
    unittest.main()
