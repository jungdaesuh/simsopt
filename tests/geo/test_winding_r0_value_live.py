"""B1.3 value-live winding_r0 regression tests (2026-06-16).

The banana-coil buildability and keep-out steering terms used to bake the
winding-surface major radius into a ``@jit`` closure at construction. With
``--winding-surface-free-r0`` the ``rc(0,0)`` DOF moves during optimisation, so
the U-channel / poloidal / projection frame stayed pinned to the seed radius
and J silently mis-scored the moved geometry (up to ~32 deg frame tilt at the
0.993 corridor ceiling).

The terms now read the major radius LIVE from the curve's CWS surface on every
evaluation. These tests pin the observable contract for the three buildability
terms in B1.3 scope (``PoloidalExtent``, ``ProjectedEllipseWidth``) and one
keep-out term (``CurveVesselEnvelopeKeepout``):

* moving ``surf.set_rc(0, 0, ...)`` moves J (the bug-exposing assertion -- with
  the old frozen-closure code J would not change), and
* with a FROZEN winding surface the live read equals the constructor value, so
  J/dJ are byte-identical to the pre-change frozen-closure numbers, for both a
  CWS lineage and a non-CWS (``CurveXYZFourier``) fallback lineage.

The golden J/dJ numbers below were recorded from the live-read code on a frozen
surface; because a frozen surface makes the live read exactly equal the
constructor constant (asserted alongside), they are identical to the values the
old frozen-closure code produced.
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

from banana_opt.ellipse_width import ProjectedEllipseWidth  # noqa: E402
from banana_opt.hardware_keepout import (  # noqa: E402
    CurveVesselEnvelopeKeepout,
    live_winding_r0,
)
from banana_opt.poloidal_extent import PoloidalExtent  # noqa: E402

from simsopt.field import Coil, Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
)

#: Seed winding major radius the golden numbers were recorded at, m.
SEED_R0_M = 0.903
#: A moved winding major radius inside the (0.908, 0.993) free-r0 corridor, m.
MOVED_R0_M = 0.96


def _cws_curve(major_radius: float):
    """A non-planar CWS curve whose frame genuinely depends on ``rc(0,0)``.

    Returns ``(curve, surface)`` so a test can move ``rc(0,0)`` on the surface
    and observe the curve (and the terms holding it) track the moved torus.
    """
    surface = SurfaceRZFourier(nfp=5, stellsym=True)
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, 0.142)
    surface.set_zs(1, 0, 0.142)
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order=2, surf=surface, G=1, H=0)
    curve.set("phic(0)", 0.05)
    curve.set("phic(1)", 0.03)
    curve.set("thetas(1)", 0.02)
    return curve, surface


def _xyz_curve() -> CurveXYZFourier:
    """A non-CWS curve (no embedded winding surface) for the fallback path."""
    curve = CurveXYZFourier(np.linspace(0.0, 1.0, 32, endpoint=False), order=2)
    curve.set("xc(0)", 0.95)
    curve.set("xc(1)", 0.10)
    curve.set("ys(1)", 0.10)
    curve.set("zs(1)", 0.05)
    curve.set("zs(2)", 0.02)
    return curve


def _cws_vessel_family(major_radius: float):
    surface = SurfaceRZFourier(nfp=5, stellsym=True)
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, 0.142)
    surface.set_zs(1, 0, 0.142)
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order=2, surf=surface, G=1, H=0)
    curve.set("phic(0)", 0.05)
    curve.set("phic(1)", 0.03)
    family = list(
        coils_via_symmetries(
            [curve], [Current(1.1e4)], surface.nfp, surface.stellsym
        )
    )
    return [coil.curve for coil in family], surface


def _xyz_vessel_family():
    curve = CurveXYZFourier(np.linspace(0.0, 1.0, 32, endpoint=False), order=1)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    return [Coil(curve, Current(1.0e4)).curve]


def _dj_norm(term) -> float:
    return float(np.linalg.norm(np.asarray(term.dJ(), dtype=float)))


class LiveWindingResolverTest(unittest.TestCase):
    def test_resolver_reads_cws_surface_live(self) -> None:
        curve, surface = _cws_curve(SEED_R0_M)
        self.assertAlmostEqual(
            live_winding_r0([curve], fallback=0.0), SEED_R0_M, places=12
        )
        surface.set_rc(0, 0, MOVED_R0_M)
        self.assertAlmostEqual(
            live_winding_r0([curve], fallback=0.0), MOVED_R0_M, places=12
        )

    def test_resolver_falls_back_for_non_cws(self) -> None:
        self.assertAlmostEqual(
            live_winding_r0([_xyz_curve()], fallback=0.987), 0.987, places=12
        )
        # An empty curve set has no master either -> fallback.
        self.assertAlmostEqual(live_winding_r0([], fallback=0.5), 0.5, places=12)


class PoloidalExtentValueLiveTest(unittest.TestCase):
    def test_moving_the_surface_moves_J(self) -> None:
        curve, surface = _cws_curve(SEED_R0_M)
        term = PoloidalExtent(curve, SEED_R0_M, 0.5)
        j_before = term.J()
        surface.set_rc(0, 0, MOVED_R0_M)
        self.assertAlmostEqual(term.live_R_winding(), MOVED_R0_M, places=12)
        self.assertNotAlmostEqual(term.J(), j_before, places=9)

    def test_frozen_cws_is_byte_identical_to_frozen_closure(self) -> None:
        curve, _ = _cws_curve(SEED_R0_M)
        term = PoloidalExtent(curve, SEED_R0_M, 0.5)
        # Frozen surface: the live read equals the constructor constant, so the
        # JAX-traced radius argument is the identical float the old closure used.
        self.assertAlmostEqual(term.live_R_winding(), SEED_R0_M, places=12)
        self.assertAlmostEqual(term.J(), 2.063531529531377, places=12)
        self.assertAlmostEqual(_dj_norm(term), 44.27932172054459, places=9)

    def test_non_cws_fallback_is_unchanged(self) -> None:
        term = PoloidalExtent(_xyz_curve(), 0.976, 0.5)
        self.assertAlmostEqual(term.live_R_winding(), 0.976, places=12)
        self.assertAlmostEqual(term.J(), 1.4136844080537232, places=12)
        self.assertAlmostEqual(_dj_norm(term), 28.548216964167345, places=9)


class ProjectedEllipseWidthValueLiveTest(unittest.TestCase):
    def test_moving_the_surface_moves_J(self) -> None:
        curve, surface = _cws_curve(SEED_R0_M)
        term = ProjectedEllipseWidth(curve, SEED_R0_M, 0.142)
        j_before = term.J()
        surface.set_rc(0, 0, MOVED_R0_M)
        self.assertAlmostEqual(term.live_R_winding(), MOVED_R0_M, places=12)
        self.assertNotAlmostEqual(term.J(), j_before, places=9)

    def test_frozen_cws_is_byte_identical_to_frozen_closure(self) -> None:
        curve, _ = _cws_curve(SEED_R0_M)
        term = ProjectedEllipseWidth(curve, SEED_R0_M, 0.142)
        self.assertAlmostEqual(term.live_R_winding(), SEED_R0_M, places=12)
        self.assertAlmostEqual(term.J(), 0.32060778920153227, places=12)
        self.assertAlmostEqual(_dj_norm(term), 9.947550385887096, places=9)

    def test_non_cws_fallback_is_unchanged(self) -> None:
        term = ProjectedEllipseWidth(_xyz_curve(), 0.976, 0.210)
        self.assertAlmostEqual(term.live_R_winding(), 0.976, places=12)
        self.assertAlmostEqual(term.J(), 0.16002958899667438, places=12)
        self.assertAlmostEqual(_dj_norm(term), 1.8505542399501256, places=9)


class VesselEnvelopeKeepoutValueLiveTest(unittest.TestCase):
    def test_moving_the_surface_moves_J(self) -> None:
        curves, surface = _cws_vessel_family(0.920)
        term = CurveVesselEnvelopeKeepout(
            curves, minimum_clearance=0.02, winding_r0=0.920
        )
        j_before = term.J()
        surface.set_rc(0, 0, MOVED_R0_M)
        self.assertAlmostEqual(term.live_winding_r0(), MOVED_R0_M, places=12)
        self.assertNotAlmostEqual(term.J(), j_before, places=9)

    def test_frozen_cws_is_byte_identical_to_frozen_closure(self) -> None:
        curves, _ = _cws_vessel_family(0.920)
        term = CurveVesselEnvelopeKeepout(
            curves, minimum_clearance=0.02, winding_r0=0.920
        )
        self.assertAlmostEqual(term.live_winding_r0(), 0.920, places=12)
        self.assertAlmostEqual(term.J(), 0.01616892284029552, places=12)
        self.assertAlmostEqual(_dj_norm(term), 13.866553670970998, places=9)

    def test_non_cws_fallback_is_unchanged(self) -> None:
        term = CurveVesselEnvelopeKeepout(
            _xyz_vessel_family(), minimum_clearance=0.5, winding_r0=0.903
        )
        # The constructor fallback is the back-compat attribute; the live read
        # returns it unchanged because there is no CWS master.
        self.assertAlmostEqual(term.winding_r0, 0.903, places=12)
        self.assertAlmostEqual(term.live_winding_r0(), 0.903, places=12)
        self.assertAlmostEqual(term.J(), 3.632869909382301, places=12)


if __name__ == "__main__":
    unittest.main()
