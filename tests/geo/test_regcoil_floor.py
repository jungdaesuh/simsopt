"""Tests for the offline REGCOIL/NESCOIL winding-surface flux floor kernel.

The kernel lives in examples/single_stage_optimization/banana_opt/regcoil_floor.py.
These tests anchor on observable physics (closed-form far fields, net-current
identities, the SquaredFlux reduction), the Tikhonov response of the augmented
solve, the inductance operator's rank/assembly, and the floor<=achieved /
floor<=secular lower-bound properties — never on the kernel's internal expressions.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surface import Surface
from simsopt.objectives import SquaredFlux

# F4 SSOT: MU0 is owned by banana_opt.current_contracts, not the kernel's copy.
from banana_opt.current_contracts import MU0
from banana_opt.regcoil_floor import (
    CurrentPotentialBasis,
    biot_savart_normal_field,
    build_current_potential_basis,
    compute_regcoil_floor,
    net_poloidal_current_amperes,
    quadratic_flux_objective,
    surface_current_from_potential_derivatives,
    surface_quadrature,
)

REAL_ARTIFACT_DIR = Path(
    "/Users/suhjungdae/code/columbia/autoresearch/campaigns/"
    "banana_box_merge_2026-06-10/laneM9_volpush/m9k_derate_v052"
)


def _circular_winding_surface(nfp, major, minor, nphi, ntheta):
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi, ntheta=ntheta, range="full torus", nfp=nfp
    )
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_rc(0, 0, major)
    surface.set_rc(1, 0, minor)
    surface.set_zs(1, 0, minor)
    return surface


def _shaped_plasma_surface(nfp, major, minor, nphi, ntheta, bump):
    """Non-axisymmetric plasma surface (an n!=0 boundary bump).

    On such a surface the purely toroidal secular (TF) field is NOT tangent, so the
    secular B.n is genuinely nonzero and the single-valued basis has real work to
    do — the regime where floor < secular is a meaningful discriminator.
    """
    quadpoints_phi, quadpoints_theta = Surface.get_quadpoints(
        nphi=nphi, ntheta=ntheta, range="full torus", nfp=nfp
    )
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=2,
        ntor=2,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    surface.set_rc(0, 0, major)
    surface.set_rc(1, 0, minor)
    surface.set_zs(1, 0, minor)
    surface.set_rc(1, 1, bump)
    surface.set_zs(1, 1, bump)
    return surface


class SecularCurrentPhysicsTests(unittest.TestCase):
    """The secular current potential must reproduce known closed-form physics."""

    def test_secular_current_reproduces_analytic_toroidal_field_on_axis(self):
        # A pure poloidal (TF) current potential Phi = G_amperes * phi on a circular
        # torus produces, on the symmetry axis circle R=major, the analytic toroidal
        # field B_y = mu0 * G_amperes / (2*pi*R). This tests the K-from-potential
        # formula AND the Biot-Savart kernel against an independent closed form.
        nfp, major, minor = 1, 0.903, 0.142
        surface = _circular_winding_surface(nfp, major, minor, nphi=180, ntheta=120)
        quad = surface_quadrature(surface)
        g_amperes = -1.6e6

        dphi_secular = np.full(quad.shape, g_amperes)
        dtheta_secular = np.zeros(quad.shape)
        surface_current = surface_current_from_potential_derivatives(
            quad, dphi_secular, dtheta_secular
        ).reshape(-1, 3)
        source_points = quad.gamma.reshape(-1, 3)
        area_weights = (quad.abs_normal / quad.n_quad).reshape(-1)

        eval_point = np.array([[major, 0.0, 0.0]])
        eval_unit_normal = np.array([[0.0, 1.0, 0.0]])  # project onto +y (toroidal)
        bn = biot_savart_normal_field(
            source_points, surface_current, area_weights, eval_point, eval_unit_normal
        )
        analytic_b_toroidal = MU0 * g_amperes / (2.0 * np.pi * major)
        self.assertAlmostEqual(
            float(bn[0]),
            analytic_b_toroidal,
            places=4,
            msg="secular K Biot-Savart toroidal field disagrees with mu0*G/(2piR)",
        )

    def test_secular_current_integrates_to_net_poloidal_current(self):
        # The secular potential Phi = I_pol * phi must carry net poloidal current
        # I_pol linking the torus. Net current crossing a theta=const ribbon is
        # sum_phi K . (n_hat x e_phi) dphi. Tests the K formula against the physical
        # definition of net current, independent of any field evaluation.
        nfp, major, minor = 5, 0.903, 0.143
        nphi, ntheta = 128, 96
        surface = _circular_winding_surface(nfp, major, minor, nphi, ntheta)
        quad = surface_quadrature(surface)
        i_pol = 20 * (-80000.0)

        dphi_secular = np.full(quad.shape, i_pol)
        dtheta_secular = np.zeros(quad.shape)
        surface_current = surface_current_from_potential_derivatives(
            quad, dphi_secular, dtheta_secular
        )
        unit_normal = quad.normal / quad.abs_normal[..., None]
        theta_index = 0
        d_phi = 1.0 / nphi
        cross_normal_ephi = np.cross(
            unit_normal[:, theta_index, :], quad.gammadash1[:, theta_index, :]
        )
        net_poloidal = (
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    surface_current[:, theta_index, :],
                    cross_normal_ephi,
                )
            )
            * d_phi
        )
        # The K formula carries the right MAGNITUDE; the global sign is an
        # orientation convention that the least-squares floor is invariant to.
        self.assertAlmostEqual(abs(net_poloidal), abs(i_pol), places=2)


class QuadraticFluxReductionTests(unittest.TestCase):
    """The floor reduction must equal SquaredFlux 'quadratic flux' exactly."""

    def test_reduction_matches_hand_computed_value_on_small_grid(self):
        # 0.5 * mean(Bn^2 * |n|), hand-computed on a tiny deterministic grid.
        bn = np.array([[0.1, -0.2, 0.3], [0.0, 0.5, -0.4]])
        abs_normal = np.array([[1.0, 2.0, 0.5], [3.0, 1.5, 2.5]])
        expected = 0.5 * float(np.mean(bn**2 * abs_normal))
        self.assertAlmostEqual(
            quadratic_flux_objective(bn, abs_normal), expected, places=15
        )

    def test_reduction_matches_simsopt_squaredflux_on_a_real_field(self):
        # Independent check against simsopt's own C++ reduction: build a B.n on a
        # surface from a simple analytic field, reduce with both, and require
        # agreement. Uses a uniform B so the comparison is grid-exact.
        surface = _circular_winding_surface(3, 1.0, 0.3, nphi=40, ntheta=30)
        normal = surface.normal()
        abs_normal = np.linalg.norm(normal, axis=2)
        unit_normal = normal / abs_normal[..., None]
        uniform_b = np.array([0.7, -0.2, 0.4])
        bn = unit_normal @ uniform_b
        bcoil = np.broadcast_to(uniform_b, normal.shape).copy()

        import simsoptpp as sopp

        target = np.zeros(abs_normal.shape)
        simsopt_value = sopp.integral_BdotN(
            np.ascontiguousarray(bcoil), target, np.ascontiguousarray(normal),
            "quadratic flux",
        )
        self.assertAlmostEqual(
            quadratic_flux_objective(bn, abs_normal), simsopt_value, places=12
        )


class InductanceMatrixAssemblyTests(unittest.TestCase):
    """F3: the inductance operator (K -> B.n columns) must be well-posed and the
    K->B.n assembly must actually map a single mode to a nonzero plasma-grid field."""

    def test_inductance_matrix_has_full_column_rank(self):
        # At a production-like grid the basis B.n responses must be linearly
        # independent (full column rank == n_basis); a rank-deficient operator would
        # make the unrestricted floor an ill-posed null-space artifact. The kernel
        # reports rank from the SAME A_w it factorizes.
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 64, 40, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 64, 40)
        plasma_quad = surface_quadrature(plasma)
        winding_quad = surface_quadrature(winding)
        basis = build_current_potential_basis(
            winding_quad, mpol=4, ntor=4, nfp=nfp, stellsym=True
        )
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=net_poloidal_current_amperes(20, -80000.0),
            net_toroidal_current_amperes=0.0,
        )
        self.assertEqual(
            result.inductance_rank,
            basis.n_basis,
            msg=(
                f"inductance matrix rank {result.inductance_rank} < n_basis "
                f"{basis.n_basis}: basis B.n responses are linearly dependent."
            ),
        )
        self.assertTrue(np.isfinite(result.inductance_cond))
        self.assertGreaterEqual(result.inductance_cond, 1.0)

    def test_rank_deficient_inductance_matrix_is_reported_below_n_basis(self):
        # The full-rank test above cannot catch a broken rank tolerance: on an
        # already-full-rank operator a degenerate tol (e.g. rcond=-1, which counts
        # EVERY singular value) still reports rank == n_basis and ships green. This
        # test feeds the kernel a GENUINELY rank-deficient inductance operator and
        # requires the reported rank to drop below n_basis. We make the basis itself
        # rank-deficient by duplicating one current-potential column (mode 1 := mode
        # 0): two identical K columns produce two identical B.n columns, so A_w has a
        # provably zero singular value and true rank n_basis - 1. A correct tolerance
        # discards that ~0 singular value (rank == n_basis - 1); a broken tolerance
        # counts it and reports the full n_basis -- which this assertion rejects.
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 40, 28, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 40, 28)
        plasma_quad = surface_quadrature(plasma)
        winding_quad = surface_quadrature(winding)
        full_basis = build_current_potential_basis(
            winding_quad, mpol=3, ntor=3, nfp=nfp, stellsym=True
        )
        self.assertGreaterEqual(
            full_basis.n_basis, 2, "need >= 2 modes to duplicate a column"
        )
        # Duplicate column 0 onto column 1 (the rest of the basis is untouched), so
        # exactly one redundant direction is introduced: true rank == n_basis - 1.
        dphi = full_basis.dphi.copy()
        dtheta = full_basis.dtheta.copy()
        modes = full_basis.modes.copy()
        dphi[1] = dphi[0]
        dtheta[1] = dtheta[0]
        modes[1] = modes[0]
        degenerate_basis = CurrentPotentialBasis(
            dphi=dphi, dtheta=dtheta, modes=modes
        )
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=degenerate_basis,
            net_poloidal_current_amperes=net_poloidal_current_amperes(20, -80000.0),
            net_toroidal_current_amperes=0.0,
        )
        self.assertEqual(
            result.inductance_rank,
            degenerate_basis.n_basis - 1,
            msg=(
                f"rank-deficient inductance operator reported rank "
                f"{result.inductance_rank}, expected n_basis - 1 = "
                f"{degenerate_basis.n_basis - 1}: the rank tolerance failed to "
                "discard the duplicated column's zero singular value."
            ),
        )
        self.assertLess(
            result.inductance_rank,
            degenerate_basis.n_basis,
            msg="reported rank did not drop below n_basis on a degenerate operator.",
        )
        # The degeneracy must also surface as an effectively singular operator.
        self.assertGreater(
            result.inductance_cond,
            1e12,
            msg=(
                f"duplicated-column operator has cond {result.inductance_cond:.3e}; "
                "a rank-deficient A_w must be near-singular."
            ),
        )

    def test_single_coefficient_bn_matches_independent_recompute(self):
        # A single unit current-potential coefficient must produce the B.n the
        # independent K->B.n physics predicts, VALUE for value -- not merely a
        # nonzero, correctly-shaped column. This pins the column SCALE of the full
        # K -> Biot-Savart -> B.n assembly the floor relies on, which no other test
        # touches directly. The reference B.n is recomputed here from the mode's
        # (m, n, parity) LABEL via the documented analytic K-from-potential form and
        # a hand-rolled direct Biot-Savart double sum -- it never calls
        # surface_current_from_potential_derivatives nor biot_savart_normal_field,
        # so it is a genuinely independent check (mis-scaling a basis column by 2
        # leaves the analytic reference untouched and trips this assertion). A bare
        # linearity invariant is blind to a column scale error (it holds for any
        # scaling), so the value match is the load-bearing check.
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 40, 28, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 40, 28)
        plasma_quad = surface_quadrature(plasma)
        winding_quad = surface_quadrature(winding)
        basis = build_current_potential_basis(
            winding_quad, mpol=3, ntor=3, nfp=nfp, stellsym=True
        )
        coeff = np.zeros(basis.n_basis)
        coeff[0] = 1.0
        dphi = np.einsum("j,jpt->pt", coeff, basis.dphi)
        dtheta = np.einsum("j,jpt->pt", coeff, basis.dtheta)
        k = surface_current_from_potential_derivatives(
            winding_quad, dphi, dtheta
        ).reshape(-1, 3)
        source_points = winding_quad.gamma.reshape(-1, 3)
        area_weights = (winding_quad.abs_normal / winding_quad.n_quad).reshape(-1)
        eval_points = plasma_quad.gamma.reshape(-1, 3)
        unit_normals = (
            plasma_quad.normal / plasma_quad.abs_normal[..., None]
        ).reshape(-1, 3)
        bn = biot_savart_normal_field(
            source_points, k, area_weights, eval_points, unit_normals
        )
        self.assertEqual(bn.shape, (plasma_quad.n_quad,))
        self.assertGreater(
            float(np.max(np.abs(bn))),
            0.0,
            msg="a unit current-potential coefficient produced an all-zero B.n.",
        )

        # --- Independent reference: rebuild mode 0's surface current from its label
        # and Biot-Savart it by hand. Mode 0 is the stellarator-symmetric sine mode
        # f = sin(2*pi*(m*theta - n*nfp*phi)) (parity 0), so the (phi_quad, theta_quad)
        # derivatives are dPhi/dphi = cos(angle)*(-2pi n nfp), dPhi/dtheta =
        # cos(angle)*(2pi m). K = (1/|N|)(dPhi/dtheta e_phi - dPhi/dphi e_theta).
        m0, n0, parity0 = (int(v) for v in basis.modes[0])
        self.assertEqual(parity0, 0, "stellsym basis mode 0 must be a sine mode")
        phi_q = winding_quad.quadpoints_phi[:, None]
        theta_q = winding_quad.quadpoints_theta[None, :]
        angle = 2.0 * np.pi * (m0 * theta_q - n0 * nfp * phi_q)
        cos_angle = np.cos(angle)
        dphi_ref = np.broadcast_to(
            cos_angle * (-2.0 * np.pi * n0 * nfp), winding_quad.shape
        )
        dtheta_ref = np.broadcast_to(
            cos_angle * (2.0 * np.pi * m0), winding_quad.shape
        )
        k_ref = (
            (
                dtheta_ref[..., None] * winding_quad.gammadash1
                - dphi_ref[..., None] * winding_quad.gammadash2
            )
            / winding_quad.abs_normal[..., None]
        ).reshape(-1, 3)
        prefactor = MU0 / (4.0 * np.pi)
        bn_ref = np.empty(plasma_quad.n_quad)
        for p in range(plasma_quad.n_quad):
            diff = eval_points[p][None, :] - source_points  # (S, 3)
            dist = np.linalg.norm(diff, axis=1)  # (S,)
            cross = np.cross(k_ref, diff)  # (S, 3)
            b_vec = prefactor * np.sum(
                cross * (area_weights / dist**3)[:, None], axis=0
            )
            bn_ref[p] = b_vec @ unit_normals[p]

        # Value match to a tight relative tolerance: a column mis-scaled by 2 would
        # leave bn 2x the analytic reference, blowing this to ~1.0.
        scale = float(np.max(np.abs(bn_ref)))
        self.assertGreater(scale, 0.0, "independent reference B.n is identically zero")
        rel_err = float(np.max(np.abs(bn - bn_ref))) / scale
        self.assertLess(
            rel_err,
            1e-11,
            msg=(
                "single-mode B.n disagrees with the independent K->B.n recompute "
                f"(rel err {rel_err:.3e}); a basis-column scale error would show here."
            ),
        )

        # Linearity of the field in the coefficient: bn(2 c) == 2 bn(c) to machine
        # precision. Necessary (a nonlinear assembly bug fails this) but not
        # sufficient (it is invariant to a uniform column rescale) -- the value
        # match above is what pins the scale.
        bn_double = biot_savart_normal_field(
            source_points, 2.0 * k, area_weights, eval_points, unit_normals
        )
        lin_err = float(np.max(np.abs(bn_double - 2.0 * bn))) / float(
            np.max(np.abs(bn))
        )
        self.assertLess(
            lin_err,
            1e-13,
            msg=f"B.n is not linear in the coefficient (rel err {lin_err:.3e}).",
        )


class TikhonovResponseTests(unittest.TestCase):
    """F3: the augmented (lambda>0) least-squares branch must regularize correctly —
    increasing lambda raises the achievable floor and shrinks the coefficients."""

    def _solve(self, lam):
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 48, 32, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 48, 32)
        return compute_regcoil_floor(
            plasma_quad=surface_quadrature(plasma),
            winding_quad=surface_quadrature(winding),
            basis=build_current_potential_basis(
                surface_quadrature(winding), mpol=3, ntor=3, nfp=nfp, stellsym=True
            ),
            net_poloidal_current_amperes=net_poloidal_current_amperes(20, -80000.0),
            net_toroidal_current_amperes=0.0,
            tikhonov_lambda=lam,
        )

    def test_floor_monotone_increasing_and_coefficients_shrink_with_lambda(self):
        lambdas = [0.0, 1e-10, 1e-8]
        results = [self._solve(lam) for lam in lambdas]
        floors = [r.floor_field_objective for r in results]
        max_coeffs = [float(np.max(np.abs(r.coefficients))) for r in results]

        # Monotone non-decreasing: more regularization can only raise the floor.
        for lo, hi in zip(floors, floors[1:]):
            self.assertLessEqual(
                lo,
                hi * (1.0 + 1e-9),
                msg=f"floor decreased as lambda grew: {floors}",
            )
        # Strictly increasing across at least one decade (the augmented branch is
        # actually biting, not a no-op): the 1e-10 -> 1e-8 decade must rise.
        self.assertGreater(
            floors[2],
            floors[1] * (1.0 + 1e-6),
            msg=f"floor did not strictly rise across a lambda decade: {floors}",
        )
        # max|coefficient| strictly decreases as lambda grows (penalty shrinks c).
        for lo, hi in zip(max_coeffs, max_coeffs[1:]):
            self.assertLess(
                hi,
                lo,
                msg=f"max|coefficient| did not shrink as lambda grew: {max_coeffs}",
            )


class FloorLowerBoundPropertyTests(unittest.TestCase):
    """F3: the unregularized floor is a genuine lower bound. It must not exceed the
    c=0 secular baseline (a no-op / wrong-sign solve returning ~secular fails this),
    and the LSQ optimum must satisfy the area-weighted normal equations."""

    def test_floor_strictly_below_secular_when_basis_can_act(self):
        # Non-axisymmetric plasma: the secular B.n is genuinely nonzero, so the
        # single-valued basis can (and must) reduce it. The floor must be STRICTLY
        # below the c=0 secular baseline — a solve that returned c=0 would tie.
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 48, 32, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 48, 32)
        plasma_quad = surface_quadrature(plasma)
        winding_quad = surface_quadrature(winding)
        basis = build_current_potential_basis(
            winding_quad, mpol=3, ntor=3, nfp=nfp, stellsym=True
        )
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=net_poloidal_current_amperes(20, -80000.0),
            net_toroidal_current_amperes=0.0,
        )
        self.assertGreater(result.secular_field_objective, 0.0)
        self.assertLessEqual(
            result.floor_field_objective,
            result.secular_field_objective,
            msg="floor exceeds the c=0 secular baseline; LSQ did worse than no-op.",
        )
        # And it actually reduced it (not a tie within noise): require a clear margin.
        self.assertLess(
            result.floor_field_objective,
            0.5 * result.secular_field_objective,
            msg=(
                "floor did not strictly improve on secular "
                f"(floor={result.floor_field_objective:.6e}, "
                f"secular={result.secular_field_objective:.6e})."
            ),
        )

    def test_lsq_optimum_satisfies_area_weighted_normal_equations(self):
        # Sharp optimality (constructed known optimum): at the true minimizer the
        # residual B.n must be area-weight-orthogonal to EVERY basis column's B.n
        # (the normal equations A_w^T (A_w c - b_w) = 0). This is the hand-reduced
        # optimality condition; a wrong-sign / under-solved coefficient vector would
        # leave a large gradient. Reconstructed independently of the kernel's solve.
        nfp = 5
        plasma = _shaped_plasma_surface(nfp, 0.90, 0.07, 40, 28, bump=0.02)
        winding = _circular_winding_surface(nfp, 0.903, 0.143, 40, 28)
        plasma_quad = surface_quadrature(plasma)
        winding_quad = surface_quadrature(winding)
        basis = build_current_potential_basis(
            winding_quad, mpol=3, ntor=3, nfp=nfp, stellsym=True
        )
        g_amperes = net_poloidal_current_amperes(20, -80000.0)
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=g_amperes,
            net_toroidal_current_amperes=0.0,
        )

        source_points = winding_quad.gamma.reshape(-1, 3)
        area_weights = (winding_quad.abs_normal / winding_quad.n_quad).reshape(-1)
        eval_points = plasma_quad.gamma.reshape(-1, 3)
        unit_normals = (
            plasma_quad.normal / plasma_quad.abs_normal[..., None]
        ).reshape(-1, 3)
        plasma_weight = plasma_quad.abs_normal.reshape(-1)

        k_sec = surface_current_from_potential_derivatives(
            winding_quad, np.full(winding_quad.shape, g_amperes), np.zeros(winding_quad.shape)
        ).reshape(-1, 3)
        bn_secular = biot_savart_normal_field(
            source_points, k_sec, area_weights, eval_points, unit_normals
        )
        k_basis = surface_current_from_potential_derivatives(
            winding_quad, basis.dphi, basis.dtheta
        ).reshape(basis.n_basis, -1, 3)
        bn_basis = biot_savart_normal_field(
            source_points, k_basis, area_weights, eval_points, unit_normals
        )
        bn_total = bn_secular + result.coefficients @ bn_basis

        gradient = bn_basis @ (plasma_weight * bn_total)  # one entry per basis column
        scale = np.max(np.abs(bn_basis @ (plasma_weight * bn_secular)))
        self.assertLess(
            float(np.max(np.abs(gradient))) / scale,
            1e-10,
            msg="LSQ residual is not orthogonal to the basis columns; not optimal.",
        )


@unittest.skipUnless(
    REAL_ARTIFACT_DIR.exists(),
    f"real artifact dir unavailable: {REAL_ARTIFACT_DIR}",
)
class FloorBelowAchievedTests(unittest.TestCase):
    """V2: on the real artifact, the unrestricted floor must be a lower bound on
    the achieved banana-coil flux, on the identical scaled plasma grid."""

    def test_floor_is_a_lower_bound_on_achieved(self):
        import json

        from simsopt import load

        from banana_opt.stage2_geometry import load_plasma_geometry

        results = json.loads(
            (REAL_ARTIFACT_DIR / "results.json").read_text(encoding="utf-8")
        )
        wout = results["PLASMA_SURF_PATH"]
        if not Path(wout).exists():
            self.skipTest(f"wout file unavailable: {wout}")

        nphi, ntheta = 255, 64
        plasma_geometry = load_plasma_geometry(
            float(results["FINAL_LCFS_MAJOR_RADIUS_M"]),
            float(results["TOROIDAL_FLUX"]),
            wout,
            nphi,
            ntheta,
        )
        working_surface = plasma_geometry.working_surface
        biot_savart = load(REAL_ARTIFACT_DIR / "biot_savart_opt.json")
        achieved = float(SquaredFlux(working_surface, biot_savart).J())

        nfp = int(results["NFP"])
        winding = _circular_winding_surface(
            nfp,
            float(results["BANANA_WINDING_SURFACE_MAJOR_RADIUS_M"]),
            float(results["banana_surf_radius"]),
            nphi=128,
            ntheta=64,
        )
        plasma_quad = surface_quadrature(working_surface)
        winding_quad = surface_quadrature(winding)
        basis = build_current_potential_basis(
            winding_quad, mpol=6, ntor=6, nfp=nfp, stellsym=True
        )
        result = compute_regcoil_floor(
            plasma_quad=plasma_quad,
            winding_quad=winding_quad,
            basis=basis,
            net_poloidal_current_amperes=net_poloidal_current_amperes(
                int(results["NUM_TF_COILS"]), float(results["TF_CURRENT_A"])
            ),
            net_toroidal_current_amperes=float(
                results.get("PROXY_PLASMA_CURRENT_A", 0.0)
            ),
        )
        self.assertLessEqual(
            result.floor_field_objective,
            achieved * (1.0 + 1e-6),
            msg=(
                f"floor {result.floor_field_objective:.6e} exceeds achieved "
                f"{achieved:.6e}; lower-bound property violated."
            ),
        )
        # The floor must also not do worse than the c=0 secular baseline.
        self.assertLessEqual(
            result.floor_field_objective,
            result.secular_field_objective * (1.0 + 1e-9),
            msg="floor exceeds the secular baseline on the real artifact.",
        )


if __name__ == "__main__":
    unittest.main()
