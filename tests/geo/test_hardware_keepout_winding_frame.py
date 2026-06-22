"""Keep-out winding-frame regression tests (2026-06-10 laneLOW xval fix;
2026-06-16 B1.3 value-live winding_r0).

The single-stage solver used to construct ``CurveHardwareKeepout`` and
``CurveVesselEnvelopeKeepout`` without ``winding_r0``, so both penalties
oriented their swept U-channel frames about the 0.903 spec constant even when
the banana lineage is re-centered (0.920/0.934/0.993 winding tori are in
active use). Measured consequence: true-frame hazard violations of
J = 0.04-0.057 read exactly 0 in the 0.903 frame. The solver resolves the
realized CWS winding torus via ``resolve_keepout_winding_r0`` and passes its
major radius to both constructors.

B1.3 closed the remaining gap: the constructor value was frozen into the JIT
closure, so with ``--winding-surface-free-r0`` the ``rc(0,0)`` DOF moved during
optimisation while the frame stayed pinned to the seed radius. The terms now
read the winding major radius LIVE from the curves' CWS surface on every
evaluation, so J scores the true moving frame. The constructor ``winding_r0``
is now only the fallback for non-CWS lineages (exposed as the back-compat
``.winding_r0`` attribute; the live value is ``.live_winding_r0()``).

These tests pin that wiring:

* the resolver reports the realized torus for CWS lineages and the spec
  default for legacy non-CWS lineages (including the bundle's empty default),
* ``.winding_r0`` reports the constructor fallback while the term scores the
  LIVE surface radius (so a stale constructor value no longer mis-scores a
  CWS lineage, and moving ``rc(0,0)`` moves J), and
* the vessel-envelope term measures the violation in the live frame.
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
    CurveHardwareSdfClearanceHinge,
    CurveHardwareSdfFreeSpaceReward,
    CurveHardwareSdfKeepout,
    CurveVesselEnvelopeKeepout,
    HardwareSdfData,
    HardwareSdfGroup,
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
    def test_realized_frame_is_read_live_not_from_the_constructor(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        curves = [coil.curve for coil in family]

        realized = _solver_style_hardware_keepout(
            family, FRAME_SPLIT_HAZARD_POINT
        )
        self.assertAlmostEqual(
            realized.winding_r0, REALIZED_WINDING_R0_M, places=12
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

        # B1.3: a STALE constructor value no longer mis-scores. This term is
        # built with the 0.903 fallback, but its curves' CWS surface is the
        # realized 0.920 torus -- the frame is read live from that surface, so
        # ``.winding_r0`` (the fallback) is 0.903 while ``.live_winding_r0()``
        # and J track the live 0.920 surface and match the realized term exactly.
        stale_ctor = CurveHardwareKeepout(
            curves,
            FRAME_SPLIT_HAZARD_POINT,
            HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            POINT_WEIGHT_M2,
        )
        self.assertAlmostEqual(
            stale_ctor.winding_r0,
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )
        self.assertAlmostEqual(
            stale_ctor.live_winding_r0(), REALIZED_WINDING_R0_M, places=12
        )
        self.assertAlmostEqual(stale_ctor.J(), j_realized, places=12)

    def test_moving_the_winding_surface_moves_J(self) -> None:
        # The bug-exposing assertion: with the OLD frozen-closure code, J was
        # baked against the construction-time radius and would NOT change when
        # rc(0,0) moved. Live-reading makes J track the moved surface.
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surface = family[0].curve.surf
        term = _solver_style_hardware_keepout(family, FRAME_SPLIT_HAZARD_POINT)

        j_before = term.J()
        surface.set_rc(0, 0, 0.96)
        self.assertAlmostEqual(term.live_winding_r0(), 0.96, places=12)
        self.assertNotAlmostEqual(term.J(), j_before, places=9)

    def test_dJ_drc00_matches_fd_off_midplane(self) -> None:
        # Off-midplane, with keepout active, winding_r0 = rc(0,0) rolls the
        # U-channel frame, so dJ/d rc(0,0) has a frame-orientation partial the
        # centerline VJP cannot carry (the deferred B1.3 term). Before it was
        # added the analytic over-counted dJ/d rc(0,0) by ~40% (ana/FD = 1.40) on
        # this fixture; this pins analytic == central-FD.
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surface = family[0].curve.surf
        surface.fix_all()
        surface.unfix("rc(0,0)")
        term = _solver_style_hardware_keepout(family, FRAME_SPLIT_HAZARD_POINT)
        self.assertGreater(term.J(), 0.0)  # hinge active -> a frame partial exists

        leaf = term.dJ(partials=True)(surface, as_derivative=True).data[surface]
        idx = list(surface.local_full_dof_names).index("rc(0,0)")
        analytic = float(np.asarray(leaf)[idx])

        x0 = surface.get_rc(0, 0)
        eps = 1e-5
        surface.set_rc(0, 0, x0 + eps)
        term.recompute_bell()
        j_plus = term.J()
        surface.set_rc(0, 0, x0 - eps)
        term.recompute_bell()
        j_minus = term.J()
        surface.set_rc(0, 0, x0)
        term.recompute_bell()
        fd = (j_plus - j_minus) / (2.0 * eps)

        self.assertGreater(abs(fd), 1.0)  # a real, well-scaled slope, not noise
        self.assertAlmostEqual(
            analytic / fd,
            1.0,
            delta=1.0e-2,
            msg="frame-orientation partial missing: analytic dJ/d rc(0,0) != FD",
        )


class VesselEnvelopeWindingFrameTest(unittest.TestCase):
    def test_vessel_violation_is_measured_in_the_live_frame(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        curves = [coil.curve for coil in family]
        # 20 mm required clearance puts the inboard envelope corners inside
        # the hinge, so J is nonzero and frame-sensitive.
        minimum_clearance = 0.02

        realized = _solver_style_vessel_keepout(family, minimum_clearance)
        self.assertAlmostEqual(
            realized.winding_r0, REALIZED_WINDING_R0_M, places=12
        )
        # B1.3: built with the 0.903 fallback, but its curves embed the realized
        # 0.920 torus. ``.winding_r0`` reports the fallback; the term scores the
        # LIVE 0.920 surface, so its J matches the realized construction exactly.
        stale_ctor = CurveVesselEnvelopeKeepout(
            curves, minimum_clearance=minimum_clearance
        )
        self.assertAlmostEqual(
            stale_ctor.winding_r0,
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
            places=12,
        )
        self.assertAlmostEqual(
            stale_ctor.live_winding_r0(), REALIZED_WINDING_R0_M, places=12
        )

        j_realized = realized.J()
        self.assertGreater(j_realized, 0.0)
        self.assertAlmostEqual(stale_ctor.J(), j_realized, places=12)

    def test_moving_the_winding_surface_moves_vessel_J(self) -> None:
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surface = family[0].curve.surf
        term = _solver_style_vessel_keepout(family, minimum_clearance=0.02)

        j_before = term.J()
        surface.set_rc(0, 0, 0.96)
        self.assertAlmostEqual(term.live_winding_r0(), 0.96, places=12)
        self.assertNotAlmostEqual(term.J(), j_before, places=9)

    def test_dJ_drc00_matches_fd(self) -> None:
        # Phase 3 frame-orientation partial for the vessel-envelope term: with the
        # hinge active and rc(0,0) free, analytic dJ/d rc(0,0) must match central
        # FD. The deferred closure was ~0.4% low (ana/FD ~0.996), so the 1e-3
        # tolerance discriminates the fix.
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surface = family[0].curve.surf
        surface.fix_all()
        surface.unfix("rc(0,0)")
        term = _solver_style_vessel_keepout(family, minimum_clearance=0.02)
        self.assertGreater(term.J(), 0.0)  # hinge active -> a frame partial exists

        leaf = term.dJ(partials=True)(surface, as_derivative=True).data[surface]
        idx = list(surface.local_full_dof_names).index("rc(0,0)")
        analytic = float(np.asarray(leaf)[idx])

        x0 = surface.get_rc(0, 0)
        eps = 1e-5
        surface.set_rc(0, 0, x0 + eps)
        j_plus = term.J()
        surface.set_rc(0, 0, x0 - eps)
        j_minus = term.J()
        surface.set_rc(0, 0, x0)
        fd = (j_plus - j_minus) / (2.0 * eps)

        self.assertGreater(abs(fd), 1.0)  # a real, well-scaled slope, not noise
        self.assertAlmostEqual(analytic / fd, 1.0, delta=1.0e-3)


# --- SDF-backed keep-out winding-frame coverage (2026-06-21) -----------------
# The CAD distance-field keep-out terms also route winding_r0 (= the master CWS
# rc(0,0)) through the swept-bracket frame, so under --winding-surface-free-r0
# their dJ/d rc(0,0) carries a frame-orientation partial on top of the centerline
# VJP. This matters in production because under --hardware-keepout-backend sdf the
# SDF keep-out IS the primary keep-out (the role the point-cloud term holds by
# default), and the live free-R0 space-squeeze campaign runs SDF + free-R0
# together. These tests pin the partial for all three SDF terms: it matches
# central FD, is isolated to rc(0,0), and is suppressed when rc(0,0) is fixed.


def _tilted_plane_sdf(plane_x=1.04, z_tilt=1.0, spacing=0.01, margin=0.005):
    """A synthetic single-group hardware SDF: signed distance to the tilted plane
    ``x + z_tilt*z = plane_x`` over the banana family's outboard +x lobe. The
    z-tilt couples the field to the U-channel's normal extent, so the frame ROLL
    that winding_r0 induces off-midplane moves the sampled SDF -- making the
    keep-out hinge both ACTIVE (J != 0) and genuinely frame-sensitive (the
    frame-orientation partial is ~10-14% of dJ/d rc(0,0) on this fixture). Built
    in-memory to exercise the gradient directly; loader/manifest validation is
    covered separately in test_hardware_sdf_keepout.py."""
    ox, oy, oz = 0.80, -0.30, -0.30
    nx = int(round((1.15 - ox) / spacing)) + 1
    ny = int(round((0.30 - oy) / spacing)) + 1
    nz = int(round((0.30 - oz) / spacing)) + 1
    xs = ox + spacing * np.arange(nx)
    ys = oy + spacing * np.arange(ny)
    zs = oz + spacing * np.arange(nz)
    xx, _, zz = np.meshgrid(xs, ys, zs, indexing="ij")
    grid = plane_x - (xx + z_tilt * zz)
    group = HardwareSdfGroup(
        label="sensors",
        grid=np.asarray(grid, dtype=float),
        origin_m=np.array([ox, oy, oz]),
        spacing_m=spacing,
        narrow_band_m=1.0,
        effective_margin_m=margin,
        sign_method="synthetic_tilted_plane",
        patches=(),
    )
    return HardwareSdfData(
        manifest_path="-",
        data_path="-",
        manifest_sha256="-",
        data_sha256="-",
        groups=(group,),
        safety_margin_m=margin,
        error_budget_m={},
        documented_gate_only={},
        covered_by_other_in_loop={},
        provenance={},
    )


def _central_fd_drc00(term, surface, eps=1e-5):
    """Central finite difference of J in the master surf's rc(0,0)."""
    x0 = surface.get_rc(0, 0)
    surface.set_rc(0, 0, x0 + eps)
    term.recompute_bell()
    j_plus = term.J()
    surface.set_rc(0, 0, x0 - eps)
    term.recompute_bell()
    j_minus = term.J()
    surface.set_rc(0, 0, x0)
    term.recompute_bell()
    return (j_plus - j_minus) / (2.0 * eps)


def _analytic_drc00(term, surface):
    leaf = term.dJ(partials=True)(surface, as_derivative=True).data[surface]
    idx = list(surface.local_full_dof_names).index("rc(0,0)")
    return float(np.asarray(leaf)[idx])


class SdfKeepoutWindingFrameTest(unittest.TestCase):
    def test_dJ_drc00_matches_fd(self) -> None:
        # Primary SDF keep-out, hinge active off-midplane with rc(0,0) free: the
        # frame-orientation partial is ~10% of dJ/d rc(0,0) here, so analytic must
        # match central FD, and a regression dropping the partial (ana/FD ~ 0.90)
        # would miss by ~100x the 1e-3 tolerance.
        family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surface = family[0].curve.surf
        surface.fix_all()
        surface.unfix("rc(0,0)")
        term = CurveHardwareSdfKeepout(
            [coil.curve for coil in family], _tilted_plane_sdf()
        )
        self.assertGreater(term.J(), 0.0)  # hinge active -> a frame partial exists

        fd = _central_fd_drc00(term, surface)
        self.assertGreater(abs(fd), 1.0)  # a real, well-scaled slope, not noise
        self.assertAlmostEqual(
            _analytic_drc00(term, surface) / fd,
            1.0,
            delta=1.0e-3,
            msg="SDF keep-out frame-orientation partial missing or wrong",
        )

    def test_frame_partial_is_isolated_to_rc00_and_suppressed_when_fixed(
        self,
    ) -> None:
        # The partial is added to the master surf's rc(0,0) ONLY, and only when
        # rc(0,0) is free. So freeing rc(0,0) lifts its leaf to the FD value, while
        # fixing it drops the leaf by exactly that (meaningful) partial -- and the
        # other surf dofs (rc(1,0)/zs(1,0)) stay byte-identical either way.
        sdf = _tilted_plane_sdf()

        fam_free = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surf_free = fam_free[0].curve.surf  # all three surf dofs free
        term_free = CurveHardwareSdfKeepout([c.curve for c in fam_free], sdf)
        leaf_free = np.asarray(
            term_free.dJ(partials=True)(surf_free, as_derivative=True).data[
                surf_free
            ]
        )
        fd = _central_fd_drc00(term_free, surf_free)

        fam_fixed = _build_cws_banana_family(REALIZED_WINDING_R0_M)
        surf_fixed = fam_fixed[0].curve.surf
        surf_fixed.fix("rc(0,0)")  # -> free_winding_r0 guard suppresses the partial
        term_fixed = CurveHardwareSdfKeepout([c.curve for c in fam_fixed], sdf)
        leaf_fixed = np.asarray(
            term_fixed.dJ(partials=True)(surf_fixed, as_derivative=True).data[
                surf_fixed
            ]
        )

        names = list(surf_free.local_full_dof_names)
        rc00 = names.index("rc(0,0)")
        other = [i for i in range(len(names)) if i != rc00]
        # free-rc(0,0) leaf matches FD (the frame partial is included)
        self.assertAlmostEqual(leaf_free[rc00] / fd, 1.0, delta=1.0e-3)
        # fixing rc(0,0) removes a meaningful (~10%) frame partial from its leaf
        self.assertGreater((leaf_free[rc00] - leaf_fixed[rc00]) / fd, 0.05)
        # the partial is isolated to rc(0,0): every other surf dof is byte-identical
        np.testing.assert_array_equal(leaf_free[other], leaf_fixed[other])

    def test_all_sdf_terms_match_fd_when_free(self) -> None:
        # All three SDF terms share the single-pass frame-partial fold
        # (grad(J_jax, argnums=3) + _winding_r0_frame_derivative); each must match
        # central FD with rc(0,0) free and an active term.
        cases = [
            ("keepout", CurveHardwareSdfKeepout, {}),
            (
                "clearance_hinge",
                CurveHardwareSdfClearanceHinge,
                {"target_margin": 0.1},  # scale so the hinge slope clears FD noise
            ),
            ("free_space_reward", CurveHardwareSdfFreeSpaceReward, {}),
        ]
        for label, cls, kwargs in cases:
            with self.subTest(term=label):
                family = _build_cws_banana_family(REALIZED_WINDING_R0_M)
                surface = family[0].curve.surf
                surface.fix_all()
                surface.unfix("rc(0,0)")
                term = cls(
                    [c.curve for c in family], _tilted_plane_sdf(), **kwargs
                )
                self.assertNotEqual(term.J(), 0.0)  # term active
                fd = _central_fd_drc00(term, surface)
                self.assertGreater(abs(fd), 1.0e-3)  # a real slope above FD noise
                self.assertAlmostEqual(
                    _analytic_drc00(term, surface) / fd, 1.0, delta=1.0e-3
                )


if __name__ == "__main__":
    unittest.main()
