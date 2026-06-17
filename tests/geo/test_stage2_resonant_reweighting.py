"""Unit tests for the 2026-06-11 formulation-audit-8 (active half) Stage-2
static resonant flux reweighting.

These pin the four sanity contracts of the audit-8 spec:

* w_res = 0 (the default) constructs NOTHING: the weight-gated solver helper
  returns ``(None, 0.0, ())`` without touching the surface/field arguments,
  and constructing/evaluating the penalty on a shared field leaves the stock
  ``SquaredFlux`` value and gradient bit-identical.
* The resonant-mode penalty detects a synthetically injected resonant
  perturbation (J_res > 0) and is EXACTLY 0 for the opposite helicity and
  for a non-resonant mode (pins the ``exp(i(m theta - n phi))``, iota = n/m
  sign convention).
* The analytic FFT-adjoint gradient matches finite differences, both for the
  pure spectral kernel and end-to-end through ``BiotSavart.B_vjp`` on a
  small synthetic coil/surface case.
* Misconfiguration raises loudly: negative weight, nonzero weight without an
  iota target, empty rational window, q above the campaign-band cap, and an
  NFP-resonant harmonic set the quadrature grid cannot represent.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
import uuid
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.stage2_resonant_flux import (  # noqa: E402
    MAX_RESONANT_DENOMINATOR,
    ResonantFluxPenalty,
    build_resonant_mode_mask,
    build_stage2_resonant_flux_penalty,
    enumerate_resonant_rationals,
    resonant_mode_power,
    resonant_mode_power_dbn,
)

from simsopt.field import BiotSavart, Coil, Current  # noqa: E402
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier  # noqa: E402
from simsopt.objectives import SquaredFlux  # noqa: E402

SOLVER_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"


def _load_solver_module():
    spec = importlib.util.spec_from_file_location(
        f"banana_coil_solver_audit8_{uuid.uuid4().hex}", SOLVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SOLVER = _load_solver_module()

NPHI = 36
NTHETA = 30


def _full_torus_surface(nphi=NPHI, ntheta=NTHETA, nfp=1):
    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.arange(nphi) / nphi,
        quadpoints_theta=np.arange(ntheta) / ntheta,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.3)
    surface.set_zs(1, 0, 0.3)
    return surface


def _synthetic_coil_field(seed=7):
    """A deliberately asymmetric coil so B.n carries a broad spectrum,
    including the resonant bins."""
    rng = np.random.default_rng(seed)
    curve = CurveXYZFourier(64, 2)
    dofs = 0.05 * rng.standard_normal(len(curve.get_dofs()))
    # base ring of radius 1.6 in the xy-plane, lifted to z = 0.5
    dofs[2] += 1.6   # x cos(1)
    dofs[6] += 1.6   # y sin(1)
    dofs[10] += 0.5  # z constant
    curve.set_dofs(dofs)
    return BiotSavart([Coil(curve, Current(1.0e5))])


def _mode_grid(m, n, nphi=NPHI, ntheta=NTHETA, helicity=-1.0):
    """cos(2 pi (m theta_hat + helicity * n phi_hat)) on the FFT grid; the
    module's resonant convention is helicity = -1 (i.e. m theta - n phi)."""
    phi_hat = np.arange(nphi)[:, None] / nphi
    theta_hat = np.arange(ntheta)[None, :] / ntheta
    return np.cos(2.0 * np.pi * (m * theta_hat + helicity * n * phi_hat))


class TestRationalEnumeration(unittest.TestCase):
    def test_campaign_window_at_iota_0143(self):
        rationals = enumerate_resonant_rationals(0.143, 0.02, 8)
        self.assertEqual(rationals, (Fraction(1, 8), Fraction(1, 7)))

    def test_low_iota_campaign_band_needs_q_thirteen(self):
        self.assertEqual(enumerate_resonant_rationals(0.0954, 0.02, 8), ())
        self.assertEqual(
            enumerate_resonant_rationals(
                0.0954,
                0.02,
                MAX_RESONANT_DENOMINATOR,
            ),
            (
                Fraction(1, 13),
                Fraction(1, 12),
                Fraction(1, 11),
                Fraction(1, 10),
                Fraction(1, 9),
            ),
        )

    def test_exact_newton_low_iota_branch_reaches_one_ninth_and_one_tenth(self):
        self.assertEqual(
            enumerate_resonant_rationals(0.1186, 0.02, 8),
            (Fraction(1, 8),),
        )
        self.assertEqual(
            enumerate_resonant_rationals(
                0.1186,
                0.02,
                MAX_RESONANT_DENOMINATOR,
            ),
            (
                Fraction(1, 10),
                Fraction(1, 9),
                Fraction(1, 8),
            ),
        )

    def test_one_sixth_outside_default_window(self):
        self.assertNotIn(Fraction(1, 6), enumerate_resonant_rationals(0.143, 0.02, 8))
        self.assertIn(Fraction(1, 6), enumerate_resonant_rationals(0.143, 0.03, 8))

    def test_lowest_terms_dedupe(self):
        rationals = enumerate_resonant_rationals(0.5, 0.01, 8)
        self.assertEqual(rationals, (Fraction(1, 2),))

    def test_signed_target(self):
        rationals = enumerate_resonant_rationals(-0.143, 0.02, 8)
        self.assertEqual(rationals, (Fraction(-1, 7), Fraction(-1, 8)))

    def test_empty_window_returns_empty_tuple(self):
        self.assertEqual(enumerate_resonant_rationals(0.1234, 1e-4, 8), ())

    def test_invalid_config_raises(self):
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(None, 0.02, 8)
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(float("nan"), 0.02, 8)
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(0.143, 0.0, 8)
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(0.143, -0.01, 8)
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(0.143, 0.02, 0)
        with self.assertRaises(ValueError):
            enumerate_resonant_rationals(0.143, 0.02, MAX_RESONANT_DENOMINATOR + 1)


class TestModeMask(unittest.TestCase):
    def test_bin_convention_nfp1(self):
        mask = build_resonant_mode_mask(NPHI, NTHETA, 1, (Fraction(1, 7),))
        # harmonics of 1/7 on a 36x30 grid: (m, n) = (7, 1) and (14, 2)
        expected = {
            ((-1) % NPHI, 7), (1 % NPHI, (-7) % NTHETA),
            ((-2) % NPHI, 14), (2 % NPHI, (-14) % NTHETA),
        }
        self.assertEqual({tuple(idx) for idx in np.argwhere(mask)}, expected)

    def test_nfp5_marks_only_nfp_multiples(self):
        # 1/7 with NFP=5: first representable harmonic is (m, n) = (35, 5)
        mask = build_resonant_mode_mask(64, 128, 5, (Fraction(1, 7),))
        marked = {tuple(idx) for idx in np.argwhere(mask)}
        self.assertIn(((-5) % 64, 35), marked)
        for a, b in marked:
            n = a if a <= 32 else a - 64
            self.assertEqual(abs(n) % 5, 0)

    def test_unrepresentable_nfp_harmonics_raise(self):
        # NFP=5 pushes 1/7 to m=35 > Nyquist of ntheta=64
        with self.assertRaises(ValueError) as ctx:
            build_resonant_mode_mask(64, 64, 5, (Fraction(1, 7),))
        self.assertIn("ntheta >= 71", str(ctx.exception))

    def test_empty_rational_set_raises(self):
        with self.assertRaises(ValueError):
            build_resonant_mode_mask(NPHI, NTHETA, 1, ())


class TestSpectralKernel(unittest.TestCase):
    def setUp(self):
        self.mask = build_resonant_mode_mask(NPHI, NTHETA, 1, (Fraction(1, 7),))

    def test_detects_resonant_mode(self):
        bn = _mode_grid(7, 1)  # cos(7 theta - phi): resonant at iota = 1/7
        self.assertAlmostEqual(resonant_mode_power(bn, self.mask), 0.5, places=12)

    def test_amplitude_and_resolution_scaling(self):
        bn = 3.0 * _mode_grid(7, 1)
        self.assertAlmostEqual(resonant_mode_power(bn, self.mask), 4.5, places=12)
        mask2 = build_resonant_mode_mask(2 * NPHI, 2 * NTHETA, 1, (Fraction(1, 7),))
        bn2 = 3.0 * _mode_grid(7, 1, nphi=2 * NPHI, ntheta=2 * NTHETA)
        self.assertAlmostEqual(resonant_mode_power(bn2, mask2), 4.5, places=12)

    # The off-mask projections are mathematically exactly zero (orthogonal
    # Fourier modes on the uniform grid); numpy's FFT leaves O(machine-eps)
    # amplitude (~2e-16, power ~5e-32) of pure roundoff in foreign bins, so
    # the assertions bound at the roundoff floor -- 30 orders of magnitude
    # below the resonant signal of 0.5.

    def test_opposite_helicity_is_zero_to_roundoff(self):
        bn = _mode_grid(7, 1, helicity=+1.0)  # cos(7 theta + phi): iota = -1/7
        self.assertLess(resonant_mode_power(bn, self.mask), 1e-30)

    def test_nonresonant_mode_is_zero_to_roundoff(self):
        bn = _mode_grid(6, 1)  # iota = 1/6, not in the mask
        self.assertLess(resonant_mode_power(bn, self.mask), 1e-30)

    def test_kernel_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        bn = rng.standard_normal((NPHI, NTHETA))
        direction = rng.standard_normal((NPHI, NTHETA))
        grad = resonant_mode_power_dbn(bn, self.mask)
        eps = 1e-6
        fd = (
            resonant_mode_power(bn + eps * direction, self.mask)
            - resonant_mode_power(bn - eps * direction, self.mask)
        ) / (2.0 * eps)
        self.assertAlmostEqual(fd, float(np.sum(grad * direction)), places=9)


class TestResonantFluxPenaltyOptimizable(unittest.TestCase):
    def test_w0_default_path_constructs_nothing_and_reads_nothing(self):
        args = SimpleNamespace(stage2_resonant_flux_weight=0.0)
        term, weight, rationals = _SOLVER.build_stage2_resonant_flux_term_if_requested(
            args, None, None
        )
        self.assertIsNone(term)
        self.assertEqual(weight, 0.0)
        self.assertEqual(rationals, ())

    def test_stock_squared_flux_bit_identical_alongside_penalty(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        jf = SquaredFlux(surface, field)
        stock_value = jf.J()
        stock_grad = jf.dJ()
        term, rationals = build_stage2_resonant_flux_penalty(
            surface,
            field,
            iota_target=1.0 / 7.0,
            delta=1e-3,
            q_max=8,
        )
        self.assertEqual(rationals, (Fraction(1, 7),))
        term.J()
        term.dJ()
        # sharing the field with the new term must not perturb the stock
        # objective in any bit
        self.assertEqual(jf.J(), stock_value)
        np.testing.assert_array_equal(jf.dJ(), stock_grad)

    def test_end_to_end_gradient_matches_finite_differences(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        term, _ = build_stage2_resonant_flux_penalty(
            surface,
            field,
            iota_target=1.0 / 7.0,
            delta=1e-3,
            q_max=8,
        )
        self.assertGreater(term.J(), 0.0)
        grad = term.dJ()
        rng = np.random.default_rng(11)
        direction = rng.standard_normal(grad.shape)
        direction /= np.linalg.norm(direction)
        dot = float(np.dot(grad, direction))
        self.assertGreater(abs(dot), 0.0)
        x0 = np.array(term.x)
        eps = 1e-6
        term.x = x0 + eps * direction
        j_plus = term.J()
        term.x = x0 - eps * direction
        j_minus = term.J()
        term.x = x0
        fd = (j_plus - j_minus) / (2.0 * eps)
        np.testing.assert_allclose(fd, dot, rtol=1e-5)

    def test_penalty_value_is_resonant_power_of_actual_bn(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        mask = build_resonant_mode_mask(NPHI, NTHETA, 1, (Fraction(1, 7),))
        term = ResonantFluxPenalty(surface, field, mask)
        normal = surface.normal()
        unitn = normal / np.linalg.norm(normal, axis=2)[:, :, None]
        field.set_points(np.ascontiguousarray(surface.gamma().reshape((-1, 3))))
        bn = np.sum(field.B().reshape(normal.shape) * unitn, axis=2)
        self.assertAlmostEqual(term.J(), resonant_mode_power(bn, mask), places=14)

    def test_constructor_contract_raises(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        with self.assertRaises(ValueError):  # non-boolean mask
            ResonantFluxPenalty(surface, field, np.zeros((NPHI, NTHETA)))
        with self.assertRaises(ValueError):  # shape mismatch
            ResonantFluxPenalty(surface, field, np.zeros((NPHI + 1, NTHETA), dtype=bool))
        with self.assertRaises(ValueError):  # all-false mask = silent no-op
            ResonantFluxPenalty(surface, field, np.zeros((NPHI, NTHETA), dtype=bool))


class TestBuilderAndSolverValidation(unittest.TestCase):
    def _args(self, **overrides):
        base = dict(
            stage2_resonant_flux_weight=0.0,
            stage2_resonant_iota_target=None,
            stage2_iota_target=None,
            stage2_resonant_delta=0.02,
            stage2_resonant_qmax=MAX_RESONANT_DENOMINATOR,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_negative_weight_raises(self):
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_resonant_flux_cli_args(
                self._args(stage2_resonant_flux_weight=-1.0)
            )
        with self.assertRaises(ValueError):
            _SOLVER.build_stage2_resonant_flux_term_if_requested(
                self._args(stage2_resonant_flux_weight=-1.0), None, None
            )

    def test_nonzero_weight_without_target_raises(self):
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_resonant_flux_cli_args(
                self._args(stage2_resonant_flux_weight=0.5)
            )

    def test_existing_iota_target_plumbing_is_reused(self):
        args = self._args(stage2_resonant_flux_weight=0.5, stage2_iota_target=0.143)
        _SOLVER.validate_stage2_resonant_flux_cli_args(args)
        self.assertAlmostEqual(
            _SOLVER.resolve_stage2_resonant_iota_target(args), 0.143
        )
        # the dedicated flag wins over the shared plumbing
        args = self._args(
            stage2_resonant_flux_weight=0.5,
            stage2_iota_target=0.143,
            stage2_resonant_iota_target=0.125,
        )
        self.assertAlmostEqual(
            _SOLVER.resolve_stage2_resonant_iota_target(args), 0.125
        )

    def test_q_above_cap_raises(self):
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_resonant_flux_cli_args(
                self._args(
                    stage2_resonant_flux_weight=0.5,
                    stage2_iota_target=0.143,
                    stage2_resonant_qmax=MAX_RESONANT_DENOMINATOR + 1,
                )
            )

    def test_q_above_cap_raises_even_when_weight_is_zero(self):
        args = self._args(
            stage2_resonant_flux_weight=0.0,
            stage2_resonant_qmax=MAX_RESONANT_DENOMINATOR + 1,
        )
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_resonant_flux_cli_args(args)
        with self.assertRaises(ValueError):
            _SOLVER.build_stage2_resonant_flux_term_if_requested(
                args,
                None,
                None,
            )

    def test_empty_rational_window_raises(self):
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_resonant_flux_cli_args(
                self._args(
                    stage2_resonant_flux_weight=0.5,
                    stage2_iota_target=0.1234,
                    stage2_resonant_delta=1e-4,
                )
            )

    def test_builder_raises_on_empty_window(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        with self.assertRaises(ValueError):
            build_stage2_resonant_flux_penalty(
                surface, field, iota_target=0.1234, delta=1e-4, q_max=8
            )

    def test_builder_raises_on_partial_torus_grid(self):
        surface = SurfaceRZFourier(
            nfp=2,
            stellsym=True,
            mpol=1,
            ntor=0,
            quadpoints_phi=np.arange(NPHI) / NPHI / 2,  # half torus
            quadpoints_theta=np.arange(NTHETA) / NTHETA,
        )
        surface.set_rc(0, 0, 1.0)
        surface.set_rc(1, 0, 0.3)
        surface.set_zs(1, 0, 0.3)
        field = _synthetic_coil_field()
        with self.assertRaises(ValueError):
            build_stage2_resonant_flux_penalty(
                surface, field, iota_target=0.5, delta=1e-3, q_max=8
            )

    def test_solver_helper_builds_working_term(self):
        surface = _full_torus_surface()
        field = _synthetic_coil_field()
        args = self._args(
            stage2_resonant_flux_weight=2.0,
            stage2_resonant_iota_target=1.0 / 7.0,
            stage2_resonant_delta=1e-3,
        )
        term, weight, rationals = _SOLVER.build_stage2_resonant_flux_term_if_requested(
            args, surface, field
        )
        self.assertEqual(weight, 2.0)
        self.assertEqual(rationals, (Fraction(1, 7),))
        self.assertGreater(term.J(), 0.0)


if __name__ == "__main__":
    unittest.main()
