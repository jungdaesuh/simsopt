"""Keep-out winding-frame regression tests (2026-06-10 laneLOW xval fix).

The single-stage solver used to construct ``CurveHardwareKeepout`` and
``CurveVesselEnvelopeKeepout`` without ``winding_r0``, so both penalties
oriented their swept U-channel frames about the 0.903 spec constant even when
the banana lineage is re-centered (0.920/0.934/0.993 winding tori are in
active use). Measured consequence: true-frame hazard violations of
J = 0.04-0.057 read exactly 0 in the 0.903 frame. The solver now resolves the
realized CWS winding torus via ``resolve_keepout_winding_r0`` and passes its
major radius to both constructors. These tests pin that wiring:

* the resolver reports the realized torus for CWS lineages and the spec
  default for legacy non-CWS lineages (including the bundle's empty default),
* a hardware keep-out constructed the way the solver now constructs it sees a
  hazard point the 0.903-default construction is blind to (J > 0 vs exactly
  0), and
* the vessel-envelope term measures the violation in the realized frame, not
  the 0.903 frame.
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

from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
    HARDWARE_KEEPOUT_SAFETY_MARGIN_M,
)
from banana_opt.hardware_keepout import (  # noqa: E402
    CurveHardwareKeepout,
    CurveVesselEnvelopeKeepout,
    resolve_keepout_winding_r0,
)

from simsopt.field import Coil, Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveCWSFourierCPP,
    CurveXYZFourier,
    SurfaceRZFourier,
)

#: Realized winding torus of the re-centered test lineage (the +17 mm remap
#: family the 2026-06-10 xval measured the frame bug on).
REALIZED_WINDING_R0_M = 0.920

#: Surface patch area per hazard point, m^2 (arbitrary but fixed; J scales
#: linearly in it, so the J assertions below are written against this value).
POINT_WEIGHT_M2 = 1.0e-4

#: Hazard point placed 3.5 mm outside the realized-frame (0.920) U-channel
#: corner edge near the bottom of the master loop, where the 0.903-vs-0.920
#: frame roll peaks (~5.3 deg). The realized frame measures a 3.5 mm envelope
#: gap (inside the 5 mm safety margin, so J > 0) while the 0.903 frame
#: measures ~5.49 mm (outside the margin, so exactly J = 0): the measured
#: 2026-06-10 bug signature in miniature.
FRAME_SPLIT_HAZARD_POINT = np.array(
    [[0.85237908331592116, 0.29310824766910343, -0.13057059619573846]]
)


def _build_cws_banana_family(major_radius: float) -> list[Coil]:
    """Symmetry-expanded CWS banana family on a circular winding torus: one
    poloidal transit (G=1) with a toroidal wobble (phic(1)) so the loop is
    non-planar and the keep-out frame genuinely depends on ``winding_r0``."""
    surface = SurfaceRZFourier(nfp=5, stellsym=True)
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, 0.142)
    surface.set_zs(1, 0, 0.142)
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order=2, surf=surface, G=1, H=0)
    curve.set("phic(0)", 0.05)
    curve.set("phic(1)", 0.03)
    return list(
        coils_via_symmetries(
            [curve], [Current(1.1e4)], surface.nfp, surface.stellsym
        )
    )


def _build_legacy_xyz_family() -> list[Coil]:
    curve = CurveXYZFourier(np.linspace(0.0, 1.0, 32, endpoint=False), order=1)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    return [Coil(curve, Current(1.0e4))]


def _solver_style_hardware_keepout(banana_coils, points):
    """``CurveHardwareKeepout`` exactly as the objective bundle now builds it:
    ``winding_r0`` resolved from the banana coils' realized CWS torus."""
    return CurveHardwareKeepout(
        [coil.curve for coil in banana_coils],
        points,
        HARDWARE_KEEPOUT_MIN_DISTANCE_M,
        POINT_WEIGHT_M2,
        winding_r0=resolve_keepout_winding_r0(banana_coils),
    )


def _solver_style_vessel_keepout(banana_coils, minimum_clearance):
    """``CurveVesselEnvelopeKeepout`` exactly as the objective bundle now
    builds it: ``winding_r0`` resolved from the realized CWS torus."""
    return CurveVesselEnvelopeKeepout(
        [coil.curve for coil in banana_coils],
        minimum_clearance=minimum_clearance,
        winding_r0=resolve_keepout_winding_r0(banana_coils),
    )


class ResolveKeepoutWindingR0Test(unittest.TestCase):
    def test_cws_lineage_resolves_realized_torus(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        self.assertAlmostEqual(
            resolve_keepout_winding_r0(family),
            REALIZED_WINDING_R0_M,
            places=12,
        )

    def test_legacy_non_cws_lineage_resolves_spec_default(self) -> None:
        self.assertAlmostEqual(
            resolve_keepout_winding_r0(_build_legacy_xyz_family()),
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )

    def test_bundle_default_empty_coils_resolves_spec_default(self) -> None:
        # build_single_stage_objective_bundle defaults banana_coils=(): legacy
        # callers that omit it must keep the historical 0.903 frame exactly.
        self.assertAlmostEqual(
            resolve_keepout_winding_r0(()),
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )


class HardwareKeepoutWindingFrameTest(unittest.TestCase):
    def test_realized_frame_sees_hazard_the_default_frame_misses(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        curves = [coil.curve for coil in family]

        realized = _solver_style_hardware_keepout(
            family, FRAME_SPLIT_HAZARD_POINT
        )
        self.assertAlmostEqual(
            realized.winding_r0, REALIZED_WINDING_R0_M, places=12
        )
        # Pre-fix solver construction: no winding_r0, so the 0.903 default.
        default_frame = CurveHardwareKeepout(
            curves,
            FRAME_SPLIT_HAZARD_POINT,
            HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            POINT_WEIGHT_M2,
        )
        self.assertAlmostEqual(
            default_frame.winding_r0,
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )

        # True frame: 3.5 mm envelope gap, inside the 5 mm margin -> active.
        self.assertLess(
            realized.shortest_distance(), HARDWARE_KEEPOUT_SAFETY_MARGIN_M
        )
        j_realized = realized.J()
        self.assertAlmostEqual(
            j_realized,
            POINT_WEIGHT_M2
            * ((HARDWARE_KEEPOUT_SAFETY_MARGIN_M - 0.0035) / HARDWARE_KEEPOUT_SAFETY_MARGIN_M) ** 2
            / HARDWARE_KEEPOUT_SAFETY_MARGIN_M**2,
            delta=0.02,
            msg="realized-frame J should read the 3.5 mm corner violation",
        )

        # 0.903 frame: same point measures ~5.49 mm, beyond the margin -> the
        # default construction is exactly blind to the violation.
        self.assertGreater(
            default_frame.shortest_distance(), HARDWARE_KEEPOUT_SAFETY_MARGIN_M
        )
        self.assertEqual(
            default_frame.J(),
            0.0,
            msg="0.903-default frame must reproduce the bug signature J=0",
        )


class VesselEnvelopeWindingFrameTest(unittest.TestCase):
    def test_vessel_violation_is_measured_in_the_realized_frame(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        curves = [coil.curve for coil in family]
        # 20 mm required clearance puts the inboard envelope corners inside
        # the hinge, so J is nonzero and frame-sensitive.
        minimum_clearance = 0.02

        realized = _solver_style_vessel_keepout(family, minimum_clearance)
        self.assertAlmostEqual(
            realized.winding_r0, REALIZED_WINDING_R0_M, places=12
        )
        default_frame = CurveVesselEnvelopeKeepout(
            curves, minimum_clearance=minimum_clearance
        )
        self.assertAlmostEqual(
            default_frame.winding_r0,
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )

        j_realized = realized.J()
        j_default = default_frame.J()
        self.assertGreater(j_realized, 0.0)
        self.assertGreater(j_default, 0.0)
        # Frame roll moves the envelope corners, so the two frames measure
        # materially different violations (~5.6% here): the term must score
        # the realized geometry, not the 0.903 spec frame.
        self.assertGreater(
            abs(j_realized - j_default) / max(j_realized, j_default),
            0.02,
            msg=(
                f"vessel J realized={j_realized} vs default={j_default}: "
                "winding_r0 did not reach the vessel term"
            ),
        )


if __name__ == "__main__":
    unittest.main()
