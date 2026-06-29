"""Tests for the differentiable edge-iota proxy and Stage-2 soft-mode wiring.

The proxy (``banana_opt.edge_iota_proxy``) is a cheap, differentiable surrogate
for the field-line-trace oracle. These tests pin the load-bearing correctness
claims: the analytic gradient matches finite differences, the zero-banana-current
limit delivers zero transform, the tokamak-only proxy iota agrees with a direct
trace of the same field, the steering objective projects its gradient onto the
FULL Stage-2 field, and the soft-mode hinge steers below target and is inert
above it. Fields are built synthetically (no heavyweight artifacts) so the suite
stays fast.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))
# Make the sibling oracle-test module importable for its synthetic-EQDSK writer.
_TEST_DIR = str(Path(__file__).resolve().parent)
if _TEST_DIR not in sys.path:
    sys.path.insert(0, _TEST_DIR)

from simsopt.field import BiotSavart, Coil, Current  # noqa: E402
from simsopt.geo import create_equally_spaced_curves  # noqa: E402

from banana_opt.edge_delivered_iota import read_eqdsk, trace_iota  # noqa: E402
from banana_opt.edge_iota_proxy import (  # noqa: E402
    EdgeIotaProxyContours,
    build_edge_iota_proxy_contours,
    edge_iota_proxy_value_and_grad,
)
from banana_opt.stage2_objectives import (  # noqa: E402
    EDGE_IOTA_HINGE_LINEAR,
    Stage2EdgeIotaSteeringObjective,
    _add_stage2_edge_iota_objective,
)
from STAGE_2.banana_coil_solver import validate_stage2_edge_iota_cli_args  # noqa: E402

# Reuse the synthetic G-EQDSK writer from the oracle test suite.
from test_edge_delivered_iota import _write_synthetic_eqdsk  # noqa: E402


def _banana_biot_savart(ncoils=3, current=1.0e4):
    curves = create_equally_spaced_curves(
        ncoils, 1, stellsym=True, R0=0.9, R1=0.25, order=1
    )
    coils = [Coil(curve, Current(current)) for curve in curves]
    return BiotSavart(coils)


def _finite_difference_gradient(x0, value_fn, *, set_x):
    """Centered finite-difference gradient of ``value_fn`` over the DOF vector."""
    x0 = np.asarray(x0, dtype=float)
    fd = np.empty(x0.size, dtype=float)
    for j in range(x0.size):
        h = 1.0e-7 * max(1.0, abs(x0[j]))
        xp = x0.copy()
        xp[j] += h
        set_x(xp)
        f_plus = value_fn()
        xm = x0.copy()
        xm[j] -= h
        set_x(xm)
        f_minus = value_fn()
        fd[j] = (f_plus - f_minus) / (2.0 * h)
    set_x(x0)
    return fd


def _proxy_setup(tmpdir, *, edge_band=(0.30, 0.60), sample_count=3):
    eqdsk_path = Path(tmpdir) / "synthetic.eqdsk"
    _write_synthetic_eqdsk(eqdsk_path)
    eqdsk = read_eqdsk(eqdsk_path)
    tokamak_field = eqdsk.build_axisymmetric_field()
    minor_radius_m = 0.18
    contours = build_edge_iota_proxy_contours(
        tokamak_field,
        eqdsk=eqdsk,
        minor_radius_m=minor_radius_m,
        edge_band=edge_band,
        sample_count=sample_count,
        helicity_sign=1,
    )
    return eqdsk, tokamak_field, minor_radius_m, contours


class _SetPointsCountingBiotSavart:
    def __init__(self):
        self.points = None
        self.set_points_calls = 0

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float).copy()
        self.set_points_calls += 1

    def B(self):
        if self.points is None:
            raise AssertionError("B() called before set_points")
        return np.tile(np.array([[1.0, 1.0, 0.0]]), (self.points.shape[0], 1))

    def B_vjp(self, cotangent):
        if self.points is None:
            raise AssertionError("B_vjp() called before set_points")

        def grad(_target):
            return np.array([float(np.sum(cotangent))], dtype=float)

        return grad


def _minimal_proxy_contours():
    points_xyz = np.array(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
        ],
        dtype=float,
    )
    return EdgeIotaProxyContours(
        axis_r_m=1.0,
        axis_z_m=0.0,
        helicity_sign=1,
        radial_labels=(0.5,),
        phi_planes=(0.0,),
        points_xyz=points_xyz,
        point_R=np.ones(2, dtype=float),
        point_phi=np.zeros(2, dtype=float),
        dl_pol=np.ones(2, dtype=float),
        segment_label=np.zeros(2, dtype=int),
        segment_plane=np.zeros(2, dtype=int),
        tokamak_cyl_B=np.zeros((2, 3), dtype=float),
        iota_tokamak=np.zeros(1, dtype=float),
    )


class EdgeIotaProxyTests(unittest.TestCase):
    def test_value_and_grad_reuses_contour_points_for_b_vjp(self):
        banana_bs = _SetPointsCountingBiotSavart()

        result = edge_iota_proxy_value_and_grad(banana_bs, _minimal_proxy_contours())

        self.assertEqual(banana_bs.set_points_calls, 1)
        self.assertEqual(result.grad_delta_abs_mean.shape, (1,))

    def test_tokamak_only_proxy_iota_matches_a_direct_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            eqdsk, tokamak_field, minor_radius_m, contours = _proxy_setup(tmp)
            for label, proxy_iota in zip(contours.radial_labels, contours.iota_tokamak):
                seed_r = float(eqdsk.rmaxis) + label * minor_radius_m
                traced = trace_iota(
                    tokamak_field,
                    seed_r_m=seed_r,
                    seed_z_m=float(eqdsk.zmaxis),
                    axis_r_m=float(eqdsk.rmaxis),
                    axis_z_m=float(eqdsk.zmaxis),
                    turns=40,
                    steps_per_turn=80,
                )
                # Same field, two methods (surface average vs trace): magnitudes agree.
                self.assertAlmostEqual(abs(proxy_iota), abs(traced.iota), places=2)

    def test_zero_banana_current_delivers_zero_edge_transform(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, contours = _proxy_setup(tmp)
            banana_bs = _banana_biot_savart(current=0.0)
            result = edge_iota_proxy_value_and_grad(banana_bs, contours)
            np.testing.assert_allclose(result.delta_abs, 0.0, atol=1.0e-12)
            self.assertEqual(result.delta_abs_mean, 0.0)

    def test_analytic_gradient_matches_finite_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, contours = _proxy_setup(tmp)
            banana_bs = _banana_biot_savart()
            grad = edge_iota_proxy_value_and_grad(
                banana_bs, contours
            ).grad_delta_abs_mean
            x0 = np.array(banana_bs.x, dtype=float)
            self.assertEqual(grad.size, x0.size)
            fd_grad = _finite_difference_gradient(
                x0, lambda: edge_iota_proxy_value_and_grad(
                    banana_bs, contours
                ).delta_abs_mean,
                set_x=lambda xv: setattr(banana_bs, "x", xv),
            )
            # rtol governs signal DOFs; atol absorbs the ~1e-8 centered-FD noise
            # floor on structurally-zero-gradient DOFs.
            np.testing.assert_allclose(grad, fd_grad, rtol=1.0e-3, atol=1.0e-6)

    def test_contour_construction_fails_closed_when_band_escapes_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            eqdsk, tokamak_field, _, _ = _proxy_setup(tmp)
            with self.assertRaises(ValueError):
                build_edge_iota_proxy_contours(
                    tokamak_field,
                    eqdsk=eqdsk,
                    minor_radius_m=0.18,
                    edge_band=(5.0, 6.0),  # far outside the EQDSK psi domain
                    sample_count=3,
                    helicity_sign=1,
                )


class Stage2EdgeIotaSteeringTests(unittest.TestCase):
    def test_steering_objective_projects_gradient_onto_full_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, contours = _proxy_setup(tmp)
            tf = _banana_biot_savart(ncoils=2, current=5.0e3)
            banana = _banana_biot_savart(ncoils=3, current=1.0e4)
            full = BiotSavart(list(tf.coils) + list(banana.coils))
            banana_slice = BiotSavart(list(full.coils)[2:])
            obj = Stage2EdgeIotaSteeringObjective(banana_slice, full, contours)
            grad = obj.dJ_by_dcoils()
            x0 = np.array(full.x, dtype=float)
            self.assertEqual(grad.size, x0.size)
            # The full-field FD comparison IS the projection proof: it covers every
            # DOF, so the banana-slice B_vjp must reproduce the true dependence of
            # the proxy on every coil DOF (zero where independent, the true value
            # where shared) or this allclose fails. A blanket "TF block == 0" check
            # is invalid here because create_equally_spaced_curves shares some DOFs
            # across the two sets, so those DOFs legitimately carry banana gradient.
            fd_grad = _finite_difference_gradient(
                x0,
                lambda: Stage2EdgeIotaSteeringObjective(
                    banana_slice, full, contours
                ).J(),
                set_x=lambda xv: setattr(full, "x", xv),
            )
            np.testing.assert_allclose(grad, fd_grad, rtol=1.0e-3, atol=1.0e-6)
            self.assertGreater(np.linalg.norm(grad), 0.0)

    def test_hinge_steers_below_target_and_is_inert_above(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, contours = _proxy_setup(tmp)
            banana = _banana_biot_savart()
            full = BiotSavart(list(banana.coils))
            obj = Stage2EdgeIotaSteeringObjective(banana, full, contours)
            delta = obj.J()
            grad_delta = obj.dJ_by_dcoils()
            base_grad = np.zeros(np.asarray(full.x).size)

            # target above current delta -> active hinge, positive penalty, pushes up.
            value_active, grad_active = _add_stage2_edge_iota_objective(
                7.0, base_grad,
                edge_iota_objective=obj,
                edge_iota_weight=3.0,
                edge_iota_target_min=delta + 0.05,
            )
            self.assertGreater(value_active, 7.0)
            # the penalty gradient opposes grad(delta_abs): minimizing it grows delta.
            self.assertLess(float(np.dot(grad_active, grad_delta)), 0.0)

            # target below current delta -> inert: value and gradient unchanged.
            value_inert, grad_inert = _add_stage2_edge_iota_objective(
                7.0, base_grad,
                edge_iota_objective=obj,
                edge_iota_weight=3.0,
                edge_iota_target_min=delta - 0.05,
            )
            self.assertEqual(value_inert, 7.0)
            np.testing.assert_array_equal(grad_inert, base_grad)

    def test_linear_hinge_pull_is_constant_in_shortfall(self):
        # The L1 hinge exists so the steering gradient does NOT vanish as the mean
        # approaches target_min (the quadratic one does). Observable consequence:
        # for two different positive shortfalls the linear penalty gradient is
        # IDENTICAL (= -weight * grad(delta_abs)), whereas the quadratic one scales
        # with the shortfall. That constant pull is what drives a converged seed off
        # its hardware-minimum vertex.
        with tempfile.TemporaryDirectory() as tmp:
            _, _, _, contours = _proxy_setup(tmp)
            banana = _banana_biot_savart()
            full = BiotSavart(list(banana.coils))
            obj = Stage2EdgeIotaSteeringObjective(banana, full, contours)
            delta = obj.J()
            grad_delta = obj.dJ_by_dcoils()
            base_grad = np.zeros(np.asarray(full.x).size)
            weight = 3.0

            def fold(target_min, shape):
                return _add_stage2_edge_iota_objective(
                    7.0, base_grad,
                    edge_iota_objective=obj,
                    edge_iota_weight=weight,
                    edge_iota_target_min=target_min,
                    edge_iota_hinge_shape=shape,
                )

            small, large = delta + 0.05, delta + 0.20
            v_lin_s, g_lin_s = fold(small, EDGE_IOTA_HINGE_LINEAR)
            v_lin_l, g_lin_l = fold(large, EDGE_IOTA_HINGE_LINEAR)
            # Closed form: penalty grad == -weight * grad(delta_abs), shortfall-free.
            np.testing.assert_allclose(g_lin_s, -weight * grad_delta, rtol=1e-12)
            np.testing.assert_allclose(g_lin_l, g_lin_s, rtol=1e-12)
            # Value is linear in the shortfall (0.20 vs 0.05 -> 4x the lift over 7.0).
            self.assertAlmostEqual((v_lin_l - 7.0) / (v_lin_s - 7.0), 4.0, places=6)

            # Contrast: the quadratic hinge gradient DOES scale with the shortfall.
            _, g_quad_s = fold(small, "quadratic")
            _, g_quad_l = fold(large, "quadratic")
            self.assertAlmostEqual(
                float(np.linalg.norm(g_quad_l) / np.linalg.norm(g_quad_s)),
                4.0,
                places=6,
            )

            # Linear hinge is still inert above target (no pull, unchanged).
            v_inert, g_inert = fold(delta - 0.05, EDGE_IOTA_HINGE_LINEAR)
            self.assertEqual(v_inert, 7.0)
            np.testing.assert_array_equal(g_inert, base_grad)

    def test_soft_mode_requires_positive_weight(self):
        base = dict(
            stage2_edge_iota_mode="soft",
            stage2_edge_iota_eqdsk="x.eqdsk",
            stage2_edge_iota_lcfs="x.json",
            stage2_edge_iota_radial_band="0.75,1.0",
            stage2_edge_iota_sample_count=3,
            stage2_edge_iota_target_min=0.10,
            stage2_edge_iota_helicity="+1",
            stage2_edge_iota_trace_turns=40,
            stage2_edge_iota_steps_per_turn=80,
            stage2_edge_iota_q_validation_rel_tol=2.0e-3,
            stage2_edge_iota_survival_fraction_min=1.0,
            stage2_edge_iota_width_max=None,
        )
        with self.assertRaises(ValueError):
            validate_stage2_edge_iota_cli_args(
                SimpleNamespace(**base, stage2_edge_iota_weight=0.0)
            )
        # a positive weight passes CLI validation (config/inputs validated elsewhere).
        validate_stage2_edge_iota_cli_args(
            SimpleNamespace(**base, stage2_edge_iota_weight=1.0e-3)
        )


if __name__ == "__main__":
    unittest.main()
