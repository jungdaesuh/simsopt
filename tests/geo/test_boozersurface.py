import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import simsoptpp as sopp
from simsopt._core.json import GSONEncoder, SIMSON
from simsopt.field.coil import Coil, Current, CurrentSum, ScaledCurrent, coils_via_symmetries
from simsopt.geo.boozersurface import BoozerSurface
from simsopt.field.biotsavart import BiotSavart
from simsopt.geo import SurfaceXYZTensorFourier, SurfaceRZFourier
from simsopt.geo.surfaceobjectives import (
    Area,
    BoozerResidual,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    ToroidalFlux,
    _boozer_residual_dJ_by_dB,
    boozer_surface_residual_dB,
)
from simsopt.configs.zoo import get_ncsx_data, get_hsx_data, get_giuliani_data
from .surface_test_helpers import get_surface, get_exact_surface, get_boozer_surface

# Finite-enclosed-current support lives in examples/banana_opt/ (off-tree from
# upstream simsopt). Put that directory on the import path so we can exercise
# the wrapper subclass alongside the upstream BoozerSurface.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_PATH = _REPO_ROOT / "examples" / "single_stage_optimization"
if str(_EXAMPLES_PATH) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_PATH))
from banana_opt.boozer_finite_current import (
    BoozerSurfaceFiniteI,
    boozer_surface_residual_finite_I,
    boozer_surface_residual_dB_finite_I,
    derive_signed_G_from_field,
    _exact_vjp_finite_I,
    _lsqgrad_vjp_finite_I,
)
from banana_opt.json_compat import load_boozer_finite_i


surfacetypes_list = ["SurfaceXYZFourier", "SurfaceXYZTensorFourier"]
stellsym_list = [True, False]


class BoozerSurfaceTests(unittest.TestCase):
    def _make_area_boozer_surface(self, *, current_I, mpol, ntor, phis, thetas, constraint_weight, options):
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)
        G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
        surface = SurfaceXYZTensorFourier(
            mpol=mpol, ntor=ntor, stellsym=True, nfp=3,
            quadpoints_phi=phis, quadpoints_theta=thetas,
        )
        surface.fit_to_curve(ma, 0.1, flip_theta=True)
        label = Area(surface)
        boozer_surface = BoozerSurfaceFiniteI(
            bs, surface, label, label.J(), constraint_weight=constraint_weight,
            options=options, I=current_I,
        )
        return bs, G0, boozer_surface

    def _make_small_area_boozer_surface(self, *, current_I):
        mpol = 3
        ntor = 3
        return self._make_area_boozer_surface(
            current_I=current_I,
            mpol=mpol,
            ntor=ntor,
            phis=np.linspace(0, 1/3, 2*ntor+1, endpoint=False),
            thetas=np.linspace(0, 1, 2*mpol+1, endpoint=False),
            constraint_weight=100.,
            options={"weight_inv_modB": False},
        )

    def _unique_leaf_currents(self, biotsavart):
        leaves = {}
        pending = [coil.current for coil in reversed(biotsavart.coils)]

        while pending:
            current = pending.pop()
            if isinstance(current, Current):
                leaves[id(current)] = current
            elif isinstance(current, ScaledCurrent):
                pending.append(current.current_to_scale)
            elif isinstance(current, CurrentSum):
                pending.append(current.current_b)
                pending.append(current.current_a)
        return list(leaves.values())

    def _make_synthetic_exact_finite_i_newton_surface(self):
        surface = SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            stellsym=True,
            nfp=1,
            quadpoints_phi=np.linspace(0, 1, 3, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
        )

        class ConstantLabel:
            def J(self):
                return 0.0

            def dJ(self, partials=True):
                return lambda surface: np.zeros(surface.get_dofs().size)

        curves, currents, _ = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        return BoozerSurfaceFiniteI(
            BiotSavart(coils),
            surface,
            ConstantLabel(),
            0.0,
            constraint_weight=None,
            options={"verbose": False},
            I=0.37,
        )

    def _patch_exact_newton_direction(self, direction):
        solve_returns = [direction, np.zeros_like(direction)]

        def forward_solve_once(_, __, ___, ____):
            return solve_returns.pop(0)

        return patch(
            "banana_opt.boozer_finite_current.forward_solve",
            side_effect=forward_solve_once,
        )

    def _run_synthetic_exact_newton_line_search(self, residual_values, *, current_I=0.37):
        boozer_surface = self._make_synthetic_exact_finite_i_newton_surface()
        boozer_surface.I = current_I
        surface = boozer_surface.surface
        direction = np.zeros(surface.get_dofs().size + 2)
        direction[-2] = 1.0
        residual_size = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
        residual_iotas = []

        def residual_finite_I(surface, iota, G, biotsavart, derivatives, I):
            residual_iotas.append(iota)
            residual = np.full(residual_size, residual_values[float(iota)])
            jacobian = np.zeros((residual_size, surface.get_dofs().size + 2))
            return residual, jacobian

        with patch(
            "banana_opt.boozer_finite_current.boozer_surface_residual_finite_I",
            side_effect=residual_finite_I,
        ), patch("banana_opt.boozer_finite_current.lu", return_value=(None, None, None)), (
            self._patch_exact_newton_direction(direction)
        ):
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1.0e-12,
                maxiter=1,
                iota=0.0,
                G=2.0,
            )

        return res, residual_iotas

    def _assert_synthetic_exact_newton_result(
        self, res, residual_iotas, *, expected_iotas, expected_iter, expected_iota
    ):
        self.assertEqual(residual_iotas, expected_iotas)
        self.assertEqual(res["iter"], expected_iter)
        self.assertEqual(res["iota"], expected_iota)
        self.assertEqual(res["G"], 2.0)

    def test_finite_current_exact_newton_singular_direction_raises(self):
        boozer_surface = self._make_synthetic_exact_finite_i_newton_surface()
        surface = boozer_surface.surface
        residual_size = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size

        def residual_finite_I(surface, iota, G, biotsavart, derivatives, I):
            residual = np.ones(residual_size)
            jacobian = np.zeros((residual_size, surface.get_dofs().size + 2))
            return residual, jacobian

        with patch(
            "banana_opt.boozer_finite_current.boozer_surface_residual_finite_I",
            side_effect=residual_finite_I,
        ), patch("banana_opt.boozer_finite_current.lu", return_value=(None, None, None)), patch(
            "banana_opt.boozer_finite_current.forward_solve",
            side_effect=np.linalg.LinAlgError("singular matrix"),
        ):
            with self.assertRaisesRegex(np.linalg.LinAlgError, "singular matrix"):
                boozer_surface.solve_residual_equation_exactly_newton(
                    tol=1.0e-12,
                    maxiter=1,
                    iota=0.0,
                    G=2.0,
                )

        self.assertTrue(boozer_surface.need_to_run_code)

    def test_exact_newton_rejects_nonfinite_jacobian_without_crashing(self):
        """A diverged exact Newton re-solve whose residual Jacobian goes non-finite reports
        ``success=False`` / ``PLU=None`` instead of raising inside ``scipy.linalg.lu``.

        The per-eval surface-stack gate rejects an unsuccessful solve before any adjoint objective
        (MajorRadius/VolumeBoozer) consumes ``res['PLU']``, so honoring the solver's own ``success``
        contract routes a divergent re-solve into the existing reject path. Regression: an unguarded
        ``lu(J)`` on a non-finite Jacobian raised ``ValueError: array must not contain infs or NaNs``
        and crashed the multisurface optimization (boozersurface.py:1063).
        """
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        iota0, G0 = boozer_surface.res['iota'], boozer_surface.res['G']

        def nonfinite_jacobian_residual(surface, iota, G, biotsavart, derivatives=1):
            residual_size = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
            residual = np.ones(residual_size)
            jacobian = np.full((residual_size, surface.get_dofs().size + 2), np.inf)
            return residual, jacobian

        boozer_surface.need_to_run_code = True
        with patch(
            "simsopt.geo.boozersurface.boozer_surface_residual",
            side_effect=nonfinite_jacobian_residual,
        ):
            res = boozer_surface.solve_residual_equation_exactly_newton(
                iota=iota0, G=G0, tol=1e-12, maxiter=3,
            )

        self.assertFalse(bool(res["success"]))
        self.assertIsNone(res["PLU"])

    def test_finite_current_exact_newton_rejects_nonfinite_jacobian_without_crashing(self):
        """The finite-I exact Newton solver honors the same contract: a non-finite residual Jacobian
        yields ``success=False`` / ``PLU=None`` rather than raising in ``lu``. The I=0.37 fixture
        exercises the in-loop guard on the iteration-0 Jacobian (reached before the I-dependent
        backtracking branch)."""
        boozer_surface = self._make_synthetic_exact_finite_i_newton_surface()

        def nonfinite_jacobian_residual_finite_I(surface, iota, G, biotsavart, derivatives, I):
            residual_size = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
            residual = np.ones(residual_size)
            jacobian = np.full((residual_size, surface.get_dofs().size + 2), np.inf)
            return residual, jacobian

        with patch(
            "banana_opt.boozer_finite_current.boozer_surface_residual_finite_I",
            side_effect=nonfinite_jacobian_residual_finite_I,
        ):
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-12, maxiter=3, iota=0.0, G=2.0,
            )

        self.assertFalse(bool(res["success"]))
        self.assertIsNone(res["PLU"])

    def test_penalty_newton_finite_divergence_rolls_back_to_seed(self):
        """A finite-but-diverging penalty-Newton solve (gradient norm grows above its
        start) must not write the diverged iterate into the surface: the DOFs and the
        reported iota/G roll back to the pre-solve seed, and success is False. The
        diverged garbage would otherwise become the next warm-start seed. Only the
        diverged path changed; converged/loosely-converged solves still persist
        (covered by the convergence suite)."""
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        s = boozer_surface.surface
        trusted_dofs = np.array(s.get_dofs(), copy=True)
        iota0, G0 = float(boozer_surface.res['iota']), float(boozer_surface.res['G'])

        calls = [0]

        def growing(x, derivatives=2, **kwargs):
            # Gradient norm grows every call so the final norm exceeds the initial,
            # i.e. the solve is classified diverged; finite identity Hessian.
            calls[0] += 1
            m = np.asarray(x).size
            return 0.5 * calls[0] ** 2, float(calls[0]) * np.ones(m), np.identity(m)

        boozer_surface.need_to_run_code = True
        with patch.object(boozer_surface, 'boozer_penalty_constraints_vectorized',
                          side_effect=growing), \
             patch.object(boozer_surface, 'boozer_penalty_constraints',
                          side_effect=lambda x, **kwargs: np.ones(3)):
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                tol=1e-12, maxiter=4, iota=iota0, G=G0, vectorize=True)

        self.assertFalse(bool(res['success']))
        np.testing.assert_allclose(np.array(s.get_dofs()), trusted_dofs)
        self.assertAlmostEqual(float(res['iota']), iota0)
        self.assertAlmostEqual(float(res['G']), G0)

    def test_penalty_newton_skips_raw_residual_diagnostic_recompute(self):
        class MinimalSurface:
            def __init__(self):
                self._dofs = np.array([1.0, 2.0])

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                self._dofs = np.asarray(dofs, dtype=float).copy()

        boozer_surface = BoozerSurface.__new__(BoozerSurface)
        boozer_surface.surface = MinimalSurface()
        boozer_surface.need_to_run_code = True
        boozer_surface.res = None

        def vectorized_penalty(x, derivatives=2, **kwargs):
            size = np.asarray(x).size
            return 0.0, np.zeros(size), np.identity(size)

        def raw_residual_recompute(*args, **kwargs):
            raise AssertionError("LS Newton must not recompute the raw residual diagnostic")

        boozer_surface.boozer_penalty_constraints_vectorized = vectorized_penalty
        boozer_surface.boozer_penalty_constraints = raw_residual_recompute

        res = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=1e-12, maxiter=1, iota=0.25, vectorize=True)

        self.assertTrue(bool(res["success"]))
        self.assertIsNone(res["residual"])
        np.testing.assert_allclose(res["jacobian"], np.zeros(3))
        self.assertEqual(res["type"], "ls")

    def test_penalty_ls_manual_finite_divergence_rolls_back_to_seed(self):
        """The damped Gauss-Newton ('manual') least-squares path rolls a diverged
        iterate back to the pre-solve seed (DOFs + iota/G), success False."""
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        s = boozer_surface.surface
        trusted_dofs = np.array(s.get_dofs(), copy=True)
        iota0, G0 = float(boozer_surface.res['iota']), float(boozer_surface.res['G'])

        calls = [0]

        def growing(x, derivatives=1, **kwargs):
            calls[0] += 1
            m = np.asarray(x).size
            return float(calls[0]) * np.ones(m), np.identity(m)

        boozer_surface.need_to_run_code = True
        with patch.object(boozer_surface, 'boozer_penalty_constraints',
                          side_effect=growing):
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-12, maxiter=4, iota=iota0, G=G0, method='manual')

        self.assertFalse(bool(res['success']))
        np.testing.assert_allclose(np.array(s.get_dofs()), trusted_dofs)
        self.assertAlmostEqual(float(res['iota']), iota0)
        self.assertAlmostEqual(float(res['G']), G0)

    def test_penalty_ls_scipy_nonfinite_result_rolls_back_to_seed(self):
        """When scipy ``least_squares`` returns a non-finite result, the surface rolls
        back to the seed instead of persisting the garbage."""
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        s = boozer_surface.surface
        trusted_dofs = np.array(s.get_dofs(), copy=True)
        iota0, G0 = float(boozer_surface.res['iota']), float(boozer_surface.res['G'])
        n = trusted_dofs.size
        nonfinite = SimpleNamespace(
            x=np.full(n + 2, np.inf), fun=np.full(3, np.inf),
            grad=np.zeros(n + 2), jac=np.zeros((3, n + 2)), status=-1)

        boozer_surface.need_to_run_code = True
        with patch('simsopt.geo.boozersurface.least_squares', return_value=nonfinite):
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-12, maxiter=4, iota=iota0, G=G0, method='lm')

        self.assertFalse(bool(res['success']))
        np.testing.assert_allclose(np.array(s.get_dofs()), trusted_dofs)
        self.assertAlmostEqual(float(res['iota']), iota0)
        self.assertAlmostEqual(float(res['G']), G0)

    def test_exact_constraints_newton_finite_divergence_rolls_back_to_seed(self):
        """The exact-constraints Newton solver rolls a diverged iterate back to the
        pre-solve seed (DOFs + iota/G), success False."""
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        s = boozer_surface.surface
        trusted_dofs = np.array(s.get_dofs(), copy=True)
        iota0, G0 = float(boozer_surface.res['iota']), float(boozer_surface.res['G'])

        calls = [0]

        def growing(xl, derivatives=1, **kwargs):
            calls[0] += 1
            m = np.asarray(xl).size
            return float(calls[0]) * np.ones(m), np.identity(m)

        boozer_surface.need_to_run_code = True
        with patch.object(boozer_surface, 'boozer_exact_constraints',
                          side_effect=growing):
            res = boozer_surface.minimize_boozer_exact_constraints_newton(
                tol=1e-12, maxiter=4, iota=iota0, G=G0)

        self.assertFalse(bool(res['success']))
        np.testing.assert_allclose(np.array(s.get_dofs()), trusted_dofs)
        self.assertAlmostEqual(float(res['iota']), iota0)
        self.assertAlmostEqual(float(res['G']), G0)

    def test_exact_newton_finite_nonconverged_solve_preserves_plu(self):
        """A FINITE but non-converged exact solve keeps its factorization: ``success=False`` with a
        valid (non-None) 3-tuple ``PLU``. Only a NON-finite Jacobian nulls ``PLU``; the non-finite
        guard deliberately preserves this legacy branch byte-for-byte (e.g. examples/2_Intermediate/
        boozerQA.py reads the factorization on a non-converged-but-finite surface before its own
        success check). This pins the preserved behavior against an over-broad future regression that
        might null ``PLU`` whenever ``success`` is False.
        """
        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        iota0, G0 = boozer_surface.res['iota'], boozer_surface.res['G']

        # Re-solve the (real, finite, well-conditioned) converged surface with an unreachable
        # tolerance so the solve cannot converge in the budget -> exits finite + success=False.
        boozer_surface.need_to_run_code = True
        res = boozer_surface.solve_residual_equation_exactly_newton(
            iota=iota0, G=G0, tol=1e-300, maxiter=1,
        )

        self.assertFalse(bool(res["success"]))
        self.assertIsNotNone(res["PLU"])
        self.assertEqual(len(res["PLU"]), 3)

    def _assert_directional_fd_convergence(self, f, coeffs, direction, directional_derivative):
        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(11, 18)))
        for eps in epsilons:
            dfdx_fd = (f(coeffs + eps * direction) - f(coeffs - eps * direction)) / (2 * eps)
            err = np.abs(dfdx_fd - directional_derivative)
            self.assertLess(err, err_old * 0.31)
            err_old = err

    def test_unsolved_boozer_surface_objectives_require_initial_run_code(self):
        bs, _, boozer_surface = self._make_small_area_boozer_surface(current_I=0.0)
        error = "BoozerSurface has no solved state"

        with self.assertRaisesRegex(RuntimeError, error):
            boozer_surface.run_code_from_last_solution()

        objectives = [
            MajorRadius(boozer_surface),
            NonQuasiSymmetricRatio(boozer_surface, BiotSavart(bs.coils)),
            Iotas(boozer_surface),
            BoozerResidual(boozer_surface, BiotSavart(bs.coils)),
        ]
        for objective in objectives:
            with self.subTest(objective=type(objective).__name__):
                with self.assertRaisesRegex(RuntimeError, error):
                    objective.J()

        with self.assertRaisesRegex(RuntimeError, error):
            objectives[-1].dJ_by_dB()

    def test_legacy_boozer_surface_I_json_loads_as_examples_finite_I(self):
        _, _, boozer_surface = self._make_small_area_boozer_surface(current_I=0.123)
        payload = json.loads(json.dumps(SIMSON(boozer_surface), cls=GSONEncoder))

        serialized = next(
            item
            for item in payload["simsopt_objs"].values()
            if item.get("@class") == "BoozerSurfaceFiniteI"
        )
        serialized["@module"] = "simsopt.geo.boozersurface"
        serialized["@class"] = "BoozerSurface"

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy_boozer_surface_i.json"
            path.write_text(json.dumps(payload))
            loaded = load_boozer_finite_i(path)

        self.assertIsInstance(loaded, BoozerSurfaceFiniteI)
        self.assertEqual(loaded.I, 0.123)
        self.assertAlmostEqual(loaded.label.J(), boozer_surface.label.J())
        self.assertIs(loaded.label.surface, loaded.surface)

    def test_finite_current_path_b_uses_nonzero_iota_kernel_equivalence(self):
        current_I = 0.37
        bs, G0, boozer_surface = self._make_small_area_boozer_surface(current_I=current_I)
        surface = boozer_surface.surface
        iota = 0.27
        weight_inv_modB = False

        residual = boozer_surface_residual_finite_I(
            surface,
            iota,
            G0,
            bs,
            derivatives=0,
            weight_inv_modB=weight_inv_modB,
            I=current_I,
        )[0]

        x = surface.gamma()
        xsemiflat = x.reshape((x.size // 3, 3)).copy()
        bs.set_points(xsemiflat)
        B = bs.B().reshape(x.shape)
        finite_current_kernel_value = sopp.boozer_residual(
            G0 + iota * current_I,
            iota,
            surface.gammadash1(),
            surface.gammadash2(),
            B,
            weight_inv_modB,
        )
        zero_current_kernel_value = sopp.boozer_residual(
            G0,
            iota,
            surface.gammadash1(),
            surface.gammadash2(),
            B,
            weight_inv_modB,
        )

        self.assertNotEqual(G0 + iota * current_I, G0)
        self.assertNotEqual(finite_current_kernel_value, zero_current_kernel_value)
        self.assertAlmostEqual(
            finite_current_kernel_value,
            0.5 * np.sum(residual ** 2),
            places=12,
        )

    def test_in_memory_biot_savart_linearity_scales_leaf_currents_once(self):
        bs, _, _ = self._make_small_area_boozer_surface(current_I=0.0)
        points = np.ascontiguousarray(
            np.array(
                [
                    [1.22, 0.03, 0.05],
                    [1.08, 0.19, -0.04],
                    [0.97, -0.11, 0.08],
                ]
            )
        )
        bs.set_points(points)
        B0 = bs.B().copy()

        currents = self._unique_leaf_currents(bs)
        original_dofs = [current.x.copy() for current in currents]
        for current in currents:
            current.x = 2.0 * current.x
        bs.clear_cached_properties()
        bs.set_points(points)
        B1 = bs.B().copy()

        for current, dofs in zip(currents, original_dofs):
            current.x = dofs
        bs.clear_cached_properties()

        np.testing.assert_allclose(B1, 2.0 * B0, rtol=1.0e-13, atol=1.0e-14)

    def test_boozer_residual_dj_by_db_solved_state_equivalent_to_pre_01828e4f6_code_path(self):
        """Isolated proof for `01828e4f6` — on solved state, the new
        ``run_code_from_last_solution()`` call site in
        ``BoozerResidual.dJ_by_dB`` must produce identical numerics to
        the pre-commit direct ``self.boozer_surface.res`` access.

        The commit changed the unsolved-state failure mode (now raises
        ``RuntimeError`` with a clear message instead of an opaque
        ``NoneType`` error). On the happy path — populated ``res``,
        ``need_to_run_code == False`` — the two code paths must be
        numerically identical. This test pins that equivalence by
        running both directly and asserting bit-equality.

        See ``docs/regression_panel_colleague_artifacts_2026-05-11.md``
        §3.1 (Path B) and ``tests/regression/TIER_A_LEDGER.md``.
        """
        bs, G0, boozer_surface = self._make_small_area_boozer_surface(current_I=0.0)
        # Populate solved state without running the (slow) solver.
        boozer_surface.res = {
            "type": "ls",
            "success": True,
            "iota": -0.3,
            "G": G0,
            "weight_inv_modB": False,
        }
        boozer_surface.need_to_run_code = False

        # Post-01828e4f6 path: BoozerResidual.dJ_by_dB calls
        # run_code_from_last_solution() internally.
        br = BoozerResidual(boozer_surface, BiotSavart(bs.coils))
        dJ_post = br.dJ_by_dB()

        # Pre-01828e4f6 path: replicate the pre-commit body inline, accessing
        # boozer_surface.res directly (the removed code path).
        res = boozer_surface.res
        surface = br.surface
        surface.set_dofs(br.in_surface.get_dofs())
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(
            surface, res['iota'], res['G'], br.biotsavart,
            derivatives=0, weight_inv_modB=res['weight_inv_modB'],
        )
        dJ_pre = _boozer_residual_dJ_by_dB(r, r_dB, np.sqrt(num_points))

        # Bit-equality — the commit must not have altered numerics on the
        # solved-state happy path.
        np.testing.assert_array_equal(dJ_post, dJ_pre)

    def test_finite_current_run_code_preserves_cached_upstream_return(self):
        _, _, boozer_surface = self._make_small_area_boozer_surface(current_I=0.37)
        boozer_surface.res = {
            "type": "ls",
            "success": True,
            "iota": -0.3,
            "G": 1.0,
        }
        boozer_surface.need_to_run_code = False

        result = boozer_surface.run_code(-0.3, G=1.0)

        self.assertIsNone(result)
        self.assertEqual(boozer_surface.res["I"], 0.37)
        self.assertIn("vjp", boozer_surface.res)

    def test_finite_current_requires_explicit_signed_G(self):
        bs, _, boozer_surface = self._make_small_area_boozer_surface(current_I=0.37)
        surface = boozer_surface.surface
        iota = -0.3
        explicit_iota_x = np.concatenate((surface.get_dofs(), [iota]))
        explicit_iota_lm_x = np.concatenate((explicit_iota_x, [0.0, 0.0]))

        with self.assertRaisesRegex(ValueError, "explicit signed G"):
            boozer_surface_residual_finite_I(surface, iota, None, bs, I=0.37)

        with self.assertRaisesRegex(ValueError, "explicit signed G"):
            boozer_surface_residual_dB_finite_I(surface, iota, None, bs, I=0.37)

        boozer_surface.need_to_run_code = True
        with self.assertRaises(TypeError):
            boozer_surface.run_code(iota)

        cached_result = {"type": "ls", "success": True, "iota": iota, "G": 1.0}
        boozer_surface.res = cached_result
        boozer_surface.need_to_run_code = False
        explicit_g_methods = (
            boozer_surface.minimize_boozer_penalty_constraints_LBFGS,
            boozer_surface.minimize_boozer_penalty_constraints_newton,
            boozer_surface.minimize_boozer_penalty_constraints_ls,
            boozer_surface.minimize_boozer_exact_constraints_newton,
        )
        for method in explicit_g_methods:
            with self.subTest(method=method.__name__):
                with self.assertRaises(TypeError):
                    method(iota=iota)
                with self.assertRaisesRegex(ValueError, "explicit signed G"):
                    method(iota=iota, G=None)

        with self.assertRaisesRegex(ValueError, "optimize_G=True"):
            boozer_surface.boozer_penalty_constraints(
                explicit_iota_x, optimize_G=False
            )

        with self.assertRaisesRegex(ValueError, "optimize_G=True"):
            boozer_surface.boozer_penalty_constraints_vectorized(
                explicit_iota_x, optimize_G=False
            )

        with self.assertRaisesRegex(ValueError, "optimize_G=True"):
            boozer_surface.boozer_exact_constraints(
                explicit_iota_lm_x, optimize_G=False
            )

        boozer_surface.need_to_run_code = False
        with self.assertRaisesRegex(ValueError, "explicit signed G"):
            boozer_surface.solve_residual_equation_exactly_newton(iota=iota, G=None)

        with self.assertRaises(TypeError):
            boozer_surface.solve_residual_equation_exactly_newton(iota=iota)

    def test_finite_current_exact_newton_accepts_full_residual_decrease_step(self):
        res, residual_iotas = self._run_synthetic_exact_newton_line_search(
            {0.0: 10.0, -1.0: 3.0}
        )

        self._assert_synthetic_exact_newton_result(
            res,
            residual_iotas,
            expected_iotas=[0.0, -1.0],
            expected_iter=1,
            expected_iota=-1.0,
        )

    def test_finite_current_exact_newton_backtracks_residual_worsening_step(self):
        res, residual_iotas = self._run_synthetic_exact_newton_line_search(
            {0.0: 10.0, -1.0: 12.0, -0.5: 3.0}
        )

        self._assert_synthetic_exact_newton_result(
            res,
            residual_iotas,
            expected_iotas=[0.0, -1.0, -0.5],
            expected_iter=1,
            expected_iota=-0.5,
        )

    def test_finite_current_exact_newton_rejects_all_worsening_steps(self):
        residual_values = {0.0: 10.0}
        for i in range(8):
            residual_values[-0.5 ** i] = 12.0
        res, residual_iotas = self._run_synthetic_exact_newton_line_search(
            residual_values
        )

        self._assert_synthetic_exact_newton_result(
            res,
            residual_iotas,
            expected_iotas=[0.0] + [-0.5 ** i for i in range(8)],
            expected_iter=0,
            expected_iota=0.0,
        )
        self.assertFalse(res["success"])

    def test_finite_current_exact_newton_zero_current_keeps_upstream_full_step(self):
        res, residual_iotas = self._run_synthetic_exact_newton_line_search(
            {0.0: 10.0, -1.0: 12.0},
            current_I=0.0,
        )

        self._assert_synthetic_exact_newton_result(
            res,
            residual_iotas,
            expected_iotas=[0.0, -1.0],
            expected_iter=1,
            expected_iota=-1.0,
        )
        self.assertFalse(res["success"])

    def test_finite_current_exact_newton_converges_on_task25_lane4_fixture(self):
        mpol = 3
        ntor = 3
        current_I = 4 * np.pi * 1e-7 * 5000
        _, G0, boozer_surface = self._make_area_boozer_surface(
            current_I=current_I,
            mpol=mpol,
            ntor=ntor,
            phis=np.linspace(0, 1 / 3, 2 * ntor + 1, endpoint=False),
            thetas=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
            constraint_weight=None,
            options={"weight_inv_modB": False, "verbose": False},
        )

        res = boozer_surface.run_code(0.4, G=G0)

        self.assertTrue(res["success"])
        np.testing.assert_allclose(
            res["iota"], 0.40283946329212617, rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            res["G"], 13.881987793895558, rtol=1e-10, atol=1e-12
        )

    def _assert_penalty_constraints_cpp_python_match(self, boozer_surface, x, *, optimize_G, weight_inv_modB):
        w = 0.
        f0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=0, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0))

        f0, J0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=1, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-11, rtol=1e-11)

        h1 = np.random.rand(J0.size)-0.5
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1))

        H0, H1 = self._assert_penalty_constraints_derivatives2_cpp_python_match(
            boozer_surface, x, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
        )
        return H0, H1

    def _assert_penalty_constraints_derivatives2_cpp_python_match(self, boozer_surface, x, *, optimize_G, weight_inv_modB):
        w = 0.
        f0, J0, H0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        f1, J1, H1 = boozer_surface.boozer_penalty_constraints_vectorized(
            x, derivatives=2, constraint_weight=w, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB)
        np.testing.assert_allclose(f0, f1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(J0, J1, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(H0, H1, atol=1e-10, rtol=1e-10)

        h1 = np.random.rand(J0.size)-0.5
        h2 = np.random.rand(J0.size)-0.5
        np.testing.assert_allclose(J0@h1, J1@h1, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose((H0@h1)@h2, (H1@h1)@h2, atol=1e-13, rtol=1e-13)
        print(np.abs(f0-f1)/np.abs(f0), np.abs(J0@h1-J1@h1)/np.abs(J0@h1), np.abs((H0@h1)@h2-(H1@h1)@h2)/np.abs((H0@h1)@h2))
        return H0, H1

    def _print_hessian_differences(self, Ha, Hb):
        diff = np.abs(Ha.flatten() - Hb.flatten())
        rel_diff = diff/np.abs(Ha.flatten())
        ij1 = np.where(diff.reshape(Ha.shape) == np.max(diff))
        i1 = ij1[0][0]
        j1 = ij1[1][0]

        ij2 = np.where(rel_diff.reshape(Ha.shape) == np.max(rel_diff))
        i2 = ij2[0][0]
        j2 = ij2[1][0]
        print(f'max err     ({i1:03}, {j1:03}): {np.max(diff):.6e}, {Ha[i1, j1]:.6e}\nmax rel err ({i2:03}, {j2:03}): {np.max(rel_diff):.6e}, {Ha[i2,j2]:.6e}\n')

    def test_residual(self):
        """
        This test loads a SurfaceXYZFourier that interpolates the xyz
        coordinates of a surface in the NCSX configuration that was computed
        on a previous branch of pyplasmaopt. Here, we verify that the Boozer
        residual at these interpolation points is small.
        """

        s = get_exact_surface()
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)

        weight = 1.
        tf = ToroidalFlux(s, bs_tf)

        # these data are obtained from `boozer` branch of pyplamsaopt
        tf_target = 0.41431152
        iota = -0.44856192

        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        x = np.concatenate((s.get_dofs(), [iota]))
        r0 = boozer_surface.boozer_penalty_constraints(
            x, derivatives=0, constraint_weight=weight, optimize_G=False,
            scalarize=False)
        # the residual should be close to zero for all entries apart from the y
        # and z coordinate at phi=0 and theta=0 (and the corresponding rotations)
        ignores_idxs = np.zeros_like(r0)
        ignores_idxs[[1, 2, 693, 694, 695, 1386, 1387, 1388, -2, -1]] = 1
        assert np.max(np.abs(r0[ignores_idxs < 0.5])) < 1e-8
        assert np.max(np.abs(r0[-2:])) < 1e-6

    def test_boozer_penalty_constraints_gradient(self):
        """
        Taylor test to verify the gradient of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_gradient(surfacetype, stellsym, optimize_G, vectorize)

    def test_boozer_penalty_constraints_hessian(self):
        """
        Taylor test to verify the Hessian of the scalarized constrained
        optimization problem's objective.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    for vectorize in [True, False]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          vectorize=vectorize):
                            self.subtest_boozer_penalty_constraints_hessian(
                                surfacetype, stellsym, optimize_G, vectorize)

    def subtest_boozer_penalty_constraints_gradient(self, surfacetype, stellsym,
                                                    optimize_G=False, vectorize=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        weight = 11.1232

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        f0, J0 = fun(x, derivatives=1, constraint_weight=weight, optimize_G=optimize_G)
        h = np.random.uniform(size=x.shape)-0.5
        Jex = J0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            f1 = fun(x + eps*h, derivatives=0, constraint_weight=weight, optimize_G=optimize_G)
            Jfd = (f1-f0)/eps
            err = np.linalg.norm(Jfd-Jex)/np.linalg.norm(Jex)
            print(err/err_old, f0, f1)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def subtest_boozer_penalty_constraints_hessian(self, surfacetype, stellsym,
                                                   optimize_G=False, vectorize=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)
        fun = boozer_surface.boozer_penalty_constraints_vectorized if vectorize else boozer_surface.boozer_penalty_constraints

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))

        f0, J0, H0 = fun(x, derivatives=2, optimize_G=optimize_G)
        h1 = np.random.uniform(size=x.shape)-0.5
        h2 = np.random.uniform(size=x.shape)-0.5
        d2f = h1 @ H0 @ h2

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(10, 20)))
        print("###############################################################")
        for eps in epsilons:
            fp, Jp = fun(x + eps*h1, derivatives=1, optimize_G=optimize_G)
            d2f_fd = (Jp@h2-J0@h2)/eps
            err = np.abs(d2f_fd-d2f)/np.abs(d2f)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err

    def test_boozer_constrained_jacobian(self):
        """
        Taylor test to verify the Jacobian of the first order optimality
        conditions of the exactly constrained optimization problem.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for optimize_G in [True, False]:
                    with self.subTest(surfacetype=surfacetype,
                                      stellsym=stellsym,
                                      optimize_G=optimize_G):
                        self.subtest_boozer_constrained_jacobian(
                            surfacetype, stellsym, optimize_G)

    def subtest_boozer_constrained_jacobian(self, surfacetype, stellsym,
                                            optimize_G=False):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(surfacetype, stellsym)
        s.fit_to_curve(ma, 0.1)

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.3
        lm = [0., 0.]
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate(
                (x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        xl = np.concatenate((x, lm))
        res0, dres0 = boozer_surface.boozer_exact_constraints(
            xl, derivatives=1, optimize_G=optimize_G)

        h = np.random.uniform(size=xl.shape)-0.5
        dres_exact = dres0@h

        err_old = 1e9
        epsilons = np.power(2., -np.asarray(range(7, 20)))
        print("###############################################################")
        for eps in epsilons:
            res1 = boozer_surface.boozer_exact_constraints(
                xl + eps*h, derivatives=0, optimize_G=optimize_G)
            dres_fd = (res1-res0)/eps
            err = np.linalg.norm(dres_fd-dres_exact)
            print(err/err_old)
            assert err < err_old * 0.55
            err_old = err
        print("###############################################################")

    def test_boozer_surface_optimisation_convergence(self):
        """
        Test to verify the various optimization algorithms that compute
        the Boozer angles on a surface.
        """

        configs = [
            ("SurfaceXYZTensorFourier", True, True, 'residual_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton_exact'),  # noqa
            ("SurfaceXYZTensorFourier", True, True, 'newton'),  # noqa
            ("SurfaceXYZTensorFourier", False, True, 'ls'),  # noqa
            ("SurfaceXYZFourier", True, False, 'ls'),  # noqa
        ]
        for surfacetype, stellsym, optimize_G, second_stage in configs:
            for get_data in [get_hsx_data, get_ncsx_data, get_giuliani_data]:
                for vectorize in [True, False]:
                    with self.subTest(
                        surfacetype=surfacetype, stellsym=stellsym,
                            optimize_G=optimize_G, second_stage=second_stage, get_data=get_data, vectorize=vectorize):
                        self.subtest_boozer_surface_optimisation_convergence(
                            surfacetype, stellsym, optimize_G, second_stage, get_data, vectorize)

    def subtest_boozer_surface_optimisation_convergence(self, surfacetype,
                                                        stellsym, optimize_G,
                                                        second_stage, get_data,
                                                        vectorize):
        curves, currents, ma = get_data()
        if stellsym:
            coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        else:
            # Create a stellarator that still has rotational symmetry but
            # doesn't have stellarator symmetry. We do this by first applying
            # stellarator symmetry, then breaking this slightly, and then
            # applying rotational symmetry
            from simsopt.geo.curve import RotatedCurve
            curves_flipped = [RotatedCurve(c, 0, True) for c in curves]
            currents_flipped = [-cur for cur in currents]
            for c in curves_flipped:
                c.rotmat += 0.001*np.random.uniform(low=-1., high=1.,
                                                    size=c.rotmat.shape)
                c.rotmatT = c.rotmat.T
            coils = coils_via_symmetries(curves + curves_flipped,
                                         currents + currents_flipped, ma.nfp, False)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        bs = BiotSavart(coils)

        s = get_surface(surfacetype, stellsym, nfp=ma.nfp)
        s.fit_to_curve(ma, 0.1)
        if get_data is get_ncsx_data:
            iota = -0.4
        elif get_data is get_giuliani_data:
            iota = 0.4
        elif get_data is get_hsx_data:
            iota = 1.
        else:
            raise Exception("initial guess for rotational transform for this config not given")

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        if optimize_G:
            G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        else:
            G = None

        cw = (s.quadpoints_phi.size * s.quadpoints_theta.size * 3)
        # compute surface first using LBFGS exact and an area constraint
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-12, maxiter=700, constraint_weight=100/cw, iota=iota, G=G,
            vectorize=vectorize)
        print('Residual norm after LBFGS', res['iter'], np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        if second_stage == 'ls':
            res = boozer_surface.minimize_boozer_penalty_constraints_ls(
                tol=1e-11, maxiter=100, constraint_weight=1000./cw,
                iota=res['iota'], G=res['G'])
        elif second_stage == 'newton':
            res = boozer_surface.minimize_boozer_penalty_constraints_newton(
                tol=1e-10, maxiter=20, constraint_weight=100./cw,
                iota=res['iota'], G=res['G'], stab=1e-4, vectorize=vectorize)
        elif second_stage == 'newton_exact':
            res = boozer_surface.minimize_boozer_exact_constraints_newton(
                tol=1e-10, maxiter=15, iota=res['iota'], G=res['G'])
        elif second_stage == 'residual_exact':
            res = boozer_surface.solve_residual_equation_exactly_newton(
                tol=1e-12, maxiter=15, iota=res['iota'], G=res['G'])

        if second_stage == 'newton':
            print('Gradient norm after second stage', np.linalg.norm(res['jacobian']))
        else:
            print('Residual norm after second stage', np.linalg.norm(res['residual']))
        assert res['success']
        assert not boozer_surface.surface.is_self_intersecting(thetas=100)

        # For the stellsym case we have z(0, 0) = y(0, 0) = 0. For the not
        # stellsym case, we enforce z(0, 0) = 0, but expect y(0, 0) \neq 0
        gammazero = s.gamma()[0, 0, :]
        assert np.abs(gammazero[2]) < 1e-10
        if stellsym:
            assert np.abs(gammazero[1]) < 1e-10
        else:
            assert np.abs(gammazero[1]) > 1e-6

        if surfacetype == 'SurfaceXYZTensorFourier' and second_stage != 'newton':
            assert np.linalg.norm(res['residual']) < 1e-9
        if second_stage == 'newton':
            assert res['residual'] is None
            assert np.linalg.norm(res['jacobian']) <= 1e-10

        print(ar_target, ar.J())
        if res['residual'] is not None:
            print(res['residual'][-10:])
        if surfacetype == 'SurfaceXYZTensorFourier' or second_stage == 'newton_exact':
            assert np.abs(ar_target - ar.J()) < 1e-9
        else:
            assert np.abs(ar_target - ar.J()) < 1e-4

    def test_boozer_serialization(self):
        """
        Test to verify the serialization capability of a BoozerSurface.
        """
        for label in ['Volume', 'Area', 'ToroidalFlux']:
            with self.subTest(label=label):
                self.subtest_boozer_serialization(label)

    def subtest_boozer_serialization(self, label):
        import json
        from simsopt._core.json import GSONDecoder, GSONEncoder, SIMSON

        bs, boozer_surface = get_boozer_surface(label=label)

        # test serialization of BoozerSurface here too
        bs_str = json.dumps(SIMSON(boozer_surface), cls=GSONEncoder)
        bs_regen = json.loads(bs_str, cls=GSONDecoder)

        diff = boozer_surface.surface.x - bs_regen.surface.x
        self.assertAlmostEqual(np.linalg.norm(diff.ravel()), 0)
        self.assertAlmostEqual(boozer_surface.label.J(), bs_regen.label.J())
        self.assertAlmostEqual(boozer_surface.targetlabel, bs_regen.targetlabel)

        # check that BoozerSurface.surface and label.surface are the same surfaces
        assert bs_regen.label.surface is bs_regen.surface

    def test_run_code(self):
        """
        This unit test verifies that the run_code portion of the BoozerSurface class is working as expected
        """
        bs, boozer_surface = get_boozer_surface(boozer_type='ls')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        for c in bs.coils:
            c.current.fix_all()

        boozer_surface.need_to_run_code = True
        # run without providing value of G
        boozer_surface.run_code(boozer_surface.res['iota'])

        bs, boozer_surface = get_boozer_surface(boozer_type='exact')
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # this second time should not actually run
        boozer_surface.run_code(boozer_surface.res['iota'], G=boozer_surface.res['G'])

        # run the BoozerExact algorithm without a guess for G
        boozer_surface.need_to_run_code = True
        boozer_surface.solve_residual_equation_exactly_newton(iota=boozer_surface.res['iota'])

    def test_convergence_cpp_and_notcpp_same(self):
        """
        This unit test verifies that that the cpp and not cpp implementations converge to 
        the same solutions
        """
        x_vec = self.subtest_convergence_cpp_and_notcpp_same(True)
        x_nonvec = self.subtest_convergence_cpp_and_notcpp_same(False)
        np.testing.assert_allclose(x_vec, x_nonvec, atol=1e-11)

    def subtest_convergence_cpp_and_notcpp_same(self, vectorize):
        """
        compute a surface using either the vectorized or non-vectorized subroutines
        """
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, ma.nfp, True)
        current_sum = sum(abs(c.current.get_value()) for c in coils)
        bs = BiotSavart(coils)

        s = get_surface('SurfaceXYZTensorFourier', True, nfp=ma.nfp)
        s.fit_to_curve(ma, 0.1)
        iota = -0.4

        ar = Area(s)
        ar_target = ar.J()
        boozer_surface = BoozerSurface(bs, s, ar, ar_target)

        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))

        cw = 3*s.quadpoints_phi.size * s.quadpoints_theta.size
        # vectorized solution first
        res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(
            tol=1e-10, maxiter=600, constraint_weight=100./cw, iota=iota, G=G,
            vectorize=vectorize)
        print('Residual norm after LBFGS', np.sqrt(2*res['fun']))

        boozer_surface.recompute_bell()
        res = boozer_surface.minimize_boozer_penalty_constraints_newton(
            tol=1e-10, maxiter=20, constraint_weight=100./cw,
            iota=res['iota'], G=res['G'], stab=0., vectorize=vectorize)

        assert res['success']
        x = boozer_surface.surface.x.copy()
        iota = res['iota']
        G = res['G']
        return np.concatenate([x, [iota, G]])

    def test_boozer_penalty_constraints_cpp_notcpp(self):
        """
        Test to verify cpp and python implementations of the BoozerLS objective return the same thing.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for weight_inv_modB in [False, True]:
                    for (optimize_G, nphi, ntheta, mpol, ntor) in [(True, 1, 1, 3, 3), (False, 1, 1, 13, 2), (True, 2, 2, 10, 3), (False, 2, 1, 3, 4), (True, 6, 9, 3, 3), (False, 7, 8, 3, 4), (True, 3, 3, 3, 3), (False, 3, 3, 3, 5)]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=optimize_G,
                                          weight_inv_modB=weight_inv_modB,
                                          mpol=mpol,
                                          ntor=ntor):
                            self.subtest_boozer_penalty_constraints_cpp_notcpp(surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor)

    def test_finite_current_boozer_penalty_constraints_cpp_notcpp(self):
        """
        Test to verify finite-current cpp and python BoozerLS implementations
        return the same thing when the signed G contract is explicit.
        """
        for surfacetype in surfacetypes_list:
            for stellsym in stellsym_list:
                for weight_inv_modB in [False, True]:
                    for (nphi, ntheta, mpol, ntor) in [
                        (1, 1, 3, 3),
                        (2, 2, 10, 3),
                        (6, 9, 3, 3),
                        (3, 3, 3, 3),
                    ]:
                        with self.subTest(surfacetype=surfacetype,
                                          stellsym=stellsym,
                                          optimize_G=True,
                                          weight_inv_modB=weight_inv_modB,
                                          mpol=mpol,
                                          ntor=ntor):
                            self.subtest_finite_current_boozer_penalty_constraints_cpp_notcpp(
                                surfacetype, stellsym, nphi, ntheta,
                                weight_inv_modB, mpol, ntor,
                            )

    def test_boozer_penalty_constraints_derivatives2_weighted_unweighted_cpp_notcpp(self):
        for weight_inv_modB in [False, True]:
            with self.subTest(weight_inv_modB=weight_inv_modB):
                self.subtest_boozer_penalty_constraints_derivatives2_cpp_notcpp(weight_inv_modB)

    def subtest_boozer_penalty_constraints_derivatives2_cpp_notcpp(self, weight_inv_modB):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        s = get_surface(
            "SurfaceXYZTensorFourier", True, nphi=2, ntheta=2,
            thetas=[0.2432101234, 0.9832134],
            phis=[0.2234567989, 0.432123451],
            mpol=3, ntor=3,
        )
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        iota = -0.3

        for current_I in [0.0, 0.37]:
            boozer_surface = BoozerSurfaceFiniteI(bs, s, tf, tf_target, I=current_I)
            x = np.concatenate((s.get_dofs(), [iota, G]))
            self._assert_penalty_constraints_derivatives2_cpp_python_match(
                boozer_surface, x, optimize_G=True, weight_inv_modB=weight_inv_modB,
            )

    def subtest_boozer_penalty_constraints_cpp_notcpp(self, surfacetype, stellsym, optimize_G, nphi, ntheta, weight_inv_modB, mpol, ntor):

        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, stellsym)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        phis = None
        thetas = None
        if nphi == 1:
            phis = [0.2234567989]
        elif nphi == 2:
            phis = [0.2234567989, 0.432123451]

        if ntheta == 1:
            thetas = [0.2432101234]
        elif ntheta == 2:
            thetas = [0.2432101234, 0.9832134]

        s = get_surface(surfacetype, stellsym, nphi=nphi, ntheta=ntheta, thetas=thetas, phis=phis, mpol=mpol, ntor=ntor)
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)

        tf_target = 0.1
        boozer_surface = BoozerSurface(bs, s, tf, tf_target)

        iota = -0.3
        x = np.concatenate((s.get_dofs(), [iota]))
        if optimize_G:
            x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
        H0, H1 = self._assert_penalty_constraints_cpp_python_match(
            boozer_surface, x, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
        )

        self._print_hessian_differences(H0, H1)

    def subtest_finite_current_boozer_penalty_constraints_cpp_notcpp(self, surfacetype, stellsym, nphi, ntheta, weight_inv_modB, mpol, ntor):
        np.random.seed(1)
        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, stellsym)
        bs = BiotSavart(coils)
        bs_tf = BiotSavart(coils)
        current_sum = sum(abs(c.current.get_value()) for c in coils)

        phis = None
        thetas = None
        if nphi == 1:
            phis = [0.2234567989]
        elif nphi == 2:
            phis = [0.2234567989, 0.432123451]

        if ntheta == 1:
            thetas = [0.2432101234]
        elif ntheta == 2:
            thetas = [0.2432101234, 0.9832134]

        s = get_surface(surfacetype, stellsym, nphi=nphi, ntheta=ntheta, thetas=thetas, phis=phis, mpol=mpol, ntor=ntor)
        s.fit_to_curve(ma, 0.1)
        s.x = s.x + np.random.rand(s.x.size)*1e-6

        tf = ToroidalFlux(s, bs_tf, nphi=51, ntheta=51)
        tf_target = 0.1
        G = 2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))
        iota = -0.3

        for current_I in [0.0, 0.37]:
            boozer_surface = BoozerSurfaceFiniteI(bs, s, tf, tf_target, I=current_I)
            x = np.concatenate((s.get_dofs(), [iota, G]))
            H0, H1 = self._assert_penalty_constraints_cpp_python_match(
                boozer_surface, x, optimize_G=True, weight_inv_modB=weight_inv_modB,
            )

        self._print_hessian_differences(H0, H1)

    def test_boozer_exact_coil_vjp_finite_current(self):
        """
        Taylor test for the exact-path coil VJP with nonzero net toroidal current I.
        """
        np.random.seed(2)
        current_I = 0.37
        phis = np.linspace(0, 1/3, 13, endpoint=False)
        thetas = np.linspace(0, 1, 13, endpoint=False)
        bs, G0, boozer_surface = self._make_area_boozer_surface(
            current_I=current_I, mpol=6, ntor=6, phis=phis, thetas=thetas,
            constraint_weight=None,
            options={"verbose": False},
        )
        res = boozer_surface.run_code(-0.406, G=G0)

        coeffs = bs.x.copy()
        direction = np.random.rand(*coeffs.shape) - 0.5
        lm = np.random.rand(int(res["mask"].sum()) + 1) - 0.5
        dfdx = _exact_vjp_finite_I(
            current_I, lm, boozer_surface, res["iota"], res["G"]
        )(bs)
        directional_derivative = dfdx @ direction

        def f(dofs):
            bs.x = dofs
            residual = boozer_surface_residual_finite_I(
                boozer_surface.surface, res["iota"], res["G"], bs,
                derivatives=0, I=current_I,
            )[0]
            return np.dot(lm[:-1], residual[res["mask"]])

        self._assert_directional_fd_convergence(f, coeffs, direction, directional_derivative)
        bs.x = coeffs

    def test_boozer_lsqgrad_coil_vjp_finite_current(self):
        """
        Taylor test for the least-squares gradient coil VJP with nonzero net toroidal current I.
        """
        np.random.seed(3)
        current_I = 0.37
        phis = np.linspace(0, 1/3, 20, endpoint=False)
        thetas = np.linspace(0, 1, 20, endpoint=False)
        bs, G0, boozer_surface = self._make_area_boozer_surface(
            current_I=current_I, mpol=3, ntor=3, phis=phis, thetas=thetas,
            constraint_weight=100.0,
            options={"verbose": False, "weight_inv_modB": True},
        )
        boozer_surface.res = {"I": current_I}

        coeffs = bs.x.copy()
        direction = np.random.rand(*coeffs.shape) - 0.5
        iota = -0.3
        boozer = boozer_surface_residual_dB_finite_I(
            boozer_surface.surface, iota, G0, bs, derivatives=1,
            weight_inv_modB=True, I=current_I,
        )
        num_points = 3 * boozer_surface.surface.quadpoints_phi.size * boozer_surface.surface.quadpoints_theta.size
        lm = np.random.rand(boozer[2].shape[1]) - 0.5
        dfdx = _lsqgrad_vjp_finite_I(
            current_I, lm, boozer_surface, iota, G0, weight_inv_modB=True
        )(bs)
        directional_derivative = dfdx @ direction

        def f(dofs):
            bs.x = dofs
            residual_terms = boozer_surface_residual_dB_finite_I(
                boozer_surface.surface, iota, G0, bs, derivatives=1,
                weight_inv_modB=True, I=current_I,
            )
            residual = residual_terms[0] / np.sqrt(num_points)
            lsq_gradient = residual_terms[2].T @ residual / np.sqrt(num_points)
            return lm @ lsq_gradient

        self._assert_directional_fd_convergence(f, coeffs, direction, directional_derivative)
        bs.x = coeffs

    def test_boozer_surface_quadpoints(self):
        """ 
        this unit test checks that the quadpoints mask for stellarator symmetric Boozer Surfaces are correctly initialized
        """
        for idx in range(4):
            with self.subTest(idx=idx):
                self.subtest_boozer_surface_quadpoints(idx)

    def subtest_boozer_surface_quadpoints(self, idx):
        mpol = 6
        ntor = 6
        nfp = 3

        if idx == 0:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[:, mpol+1:] = False
            mask_true[ntor+1:, 0] = False
        elif idx == 1:
            phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 0.5, mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[ntor+1:, 0] = False
        elif idx == 2:
            phis = np.linspace(0, 1/(2*nfp), ntor+1, endpoint=False)
            thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
            mask_true = np.ones((phis.size, thetas.size), dtype=bool)
            mask_true[0, mpol+1:] = False
        elif idx == 3:
            phis = np.linspace(0, 1., 2*ntor+1, endpoint=False)
            thetas = np.linspace(0, 1., 2*mpol+1, endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        if idx < 3:  # the first three quadrature point sets should pass without issue.
            mask = s.get_stellsym_mask()
            assert np.all(mask == mask_true)
        else:
            with self.assertRaises(Exception):
                mask = s.get_stellsym_mask()

    def test_boozer_surface_type_assert(self):
        """
        this unit test checks that an exception is raised if a SurfaceRZFourier is passed to a BoozerSurface
        """
        mpol = 6
        ntor = 6
        nfp = 3
        phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
        thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
        s = SurfaceRZFourier(mpol=mpol, ntor=ntor, stellsym=True, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)

        base_curves, base_currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(base_curves, base_currents, 3, True)
        bs = BiotSavart(coils)

        lab = Area(s)
        lab_target = 0.1

        with self.assertRaises(Exception):
            _ = BoozerSurface(bs, s, lab, lab_target)


class DeriveSignedGFromFieldTests(unittest.TestCase):
    """Pin the SSOT signed-G derivation to the upstream sign-blind default's
    magnitude while making the sign track the TF current direction.

    The upstream ``boozer_surface_residual`` and
    ``BoozerSurface.boozer_penalty_constraints_vectorized`` fall back to
    ``G = mu0 * sum_abs(I_coil)`` when ``G is None``. That sign-blind seed
    drives the Boozer Newton solve onto the wrong-iota branch when the TF
    current is negative (e.g., HBT CW TF at -80 kA). The banana_opt SSOT
    helper must keep the magnitude identical for a CCW field (so it
    bit-equals the upstream legacy path for positive currents) but flip
    sign for CW fields.
    """

    _MU0 = 4.0e-7 * np.pi

    def _make_tf_coils(self, *, signed_current_A, num_coils=20):
        from simsopt.geo import CurveXYZFourier

        coils = []
        for _ in range(num_coils):
            curve = CurveXYZFourier(quadpoints=8, order=1)
            coils.append(Coil(curve, Current(signed_current_A)))
        return coils

    def test_sign_tracks_tf_current_direction(self):
        ccw_tf_coils = self._make_tf_coils(signed_current_A=8.0e4)
        cw_tf_coils = self._make_tf_coils(signed_current_A=-8.0e4)

        ccw_bs = BiotSavart(ccw_tf_coils)
        cw_bs = BiotSavart(cw_tf_coils)

        ccw_G = derive_signed_G_from_field(ccw_bs, tf_coils=ccw_tf_coils)
        cw_G = derive_signed_G_from_field(cw_bs, tf_coils=cw_tf_coils)

        expected_magnitude = self._MU0 * 20 * 8.0e4
        self.assertAlmostEqual(ccw_G, expected_magnitude)
        self.assertAlmostEqual(cw_G, -expected_magnitude)
        # The sign-blind upstream default would have agreed with the CCW
        # case but flipped the CW seed; pinning the absolute equality
        # documents the magnitude contract.
        self.assertAlmostEqual(abs(cw_G), ccw_G)

    def test_matches_compute_tf_G0_for_signed_tf_bundle(self):
        from banana_opt.stage2_single_stage_handoff import compute_tf_G0

        tf_coils = self._make_tf_coils(signed_current_A=-8.0e4)
        bs = BiotSavart(tf_coils)

        bs_aware_G = derive_signed_G_from_field(bs, tf_coils=tf_coils)
        legacy_G = compute_tf_G0(tf_coils)

        # Both call sites must route through the same SSOT formula.
        self.assertEqual(bs_aware_G, legacy_G)

    def test_rejects_tf_coil_not_in_supplied_field(self):
        field_tf_coils = self._make_tf_coils(signed_current_A=-8.0e4)
        stray_tf_coils = self._make_tf_coils(signed_current_A=-8.0e4)
        bs = BiotSavart(field_tf_coils)

        with self.assertRaisesRegex(ValueError, "not part of the supplied BiotSavart"):
            derive_signed_G_from_field(bs, tf_coils=stray_tf_coils)

    def test_rejects_empty_tf_bundle(self):
        bs = BiotSavart(self._make_tf_coils(signed_current_A=-8.0e4))
        with self.assertRaisesRegex(ValueError, "non-empty TF coil bundle"):
            derive_signed_G_from_field(bs, tf_coils=[])

    def test_rejects_missing_field(self):
        tf_coils = self._make_tf_coils(signed_current_A=-8.0e4)
        with self.assertRaisesRegex(ValueError, "requires a BiotSavart field"):
            derive_signed_G_from_field(None, tf_coils=tf_coils)

    def test_rejects_duplicate_tf_coil_references(self):
        """A repeated coil reference in ``tf_coils`` would silently
        double-count its current in ``G`` (each duplicate adds another
        ``mu0 * I`` term). The SSOT formula counts each physical TF coil
        exactly once, so the helper must reject duplicates explicitly
        instead of returning a magnitude inflated by the duplication
        factor.
        """
        tf_coils = self._make_tf_coils(signed_current_A=-8.0e4, num_coils=3)
        duplicated = list(tf_coils) + [tf_coils[0]]
        bs = BiotSavart(duplicated)
        with self.assertRaisesRegex(ValueError, "duplicate coils"):
            derive_signed_G_from_field(bs, tf_coils=duplicated)

    def test_excludes_non_tf_coils_from_signed_G(self):
        """Production fields always include banana/proxy/VF coils alongside
        the TF bundle (``bs = BiotSavart(tf + banana + proxy + vf)``). The
        SSOT helper's docstring promises proxy/VF currents do NOT enter
        ``G`` (they shape the field directly, and proxy plasma current
        enters the finite-I residual via the separate ``I`` invariant).
        Pin that contract: ``derive_signed_G_from_field`` must return
        ``mu0 * sum_signed(I_TF)`` over the TF subset only, NOT
        ``mu0 * sum_*(I_all_coils)`` which is what the upstream sign-blind
        ``G=None`` fallback would compute over the full field.
        """
        tf_coils = self._make_tf_coils(signed_current_A=-8.0e4, num_coils=20)
        # Non-TF coils carry currents with magnitudes and signs distinct
        # from the TF bundle so the sign-blind ``sum_abs`` alternative and
        # the (incorrect) ``sum_signed`` over all coils both differ
        # numerically from the TF-only signed sum.
        banana_coils = self._make_tf_coils(signed_current_A=+1.6e4, num_coils=2)
        proxy_coils = self._make_tf_coils(signed_current_A=-3.0e3, num_coils=2)
        vf_coils = self._make_tf_coils(signed_current_A=+5.0e3, num_coils=4)
        non_tf_coils = banana_coils + proxy_coils + vf_coils
        bs = BiotSavart(tf_coils + non_tf_coils)

        signed_G = derive_signed_G_from_field(bs, tf_coils=tf_coils)

        expected_tf_only_signed_G = self._MU0 * sum(
            coil.current.get_value() for coil in tf_coils
        )
        self.assertAlmostEqual(signed_G, expected_tf_only_signed_G)
        # TF bundle is CW → signed G is negative.
        self.assertLess(signed_G, 0.0)
        # The upstream sign-blind fallback computes ``mu0 * sum_abs`` over
        # the *whole* field; that is what the SSOT helper must NOT
        # reproduce when non-TF coils are present.
        sign_blind_all_coils_G = self._MU0 * sum(
            abs(coil.current.get_value()) for coil in bs.coils
        )
        self.assertNotAlmostEqual(signed_G, sign_blind_all_coils_G)
        # A naive "signed sum over the whole field" would also be wrong:
        # it would fold proxy/VF currents into ``G`` and double-count
        # the proxy's contribution (it already enters via ``I``).
        signed_all_coils_G = self._MU0 * sum(
            coil.current.get_value() for coil in bs.coils
        )
        self.assertNotAlmostEqual(signed_G, signed_all_coils_G)


class SignedGWireInBoozerNewtonConvergenceTests(unittest.TestCase):
    """End-to-end wire-in test: the production Stage 2 setup path produces a
    signed ``G`` seed for a CW (negative-current) TF bundle and drives the
    Boozer Newton solve with that signed seed.

    These tests pair with :class:`DeriveSignedGFromFieldTests`. That class
    pins the SSOT formula in isolation; this class drives the production
    ``build_stage2_iota_runtime`` setup chain to confirm the SSOT seed
    actually reaches the Boozer Newton solve (and lands it on a sensible
    fixed point, not the upstream sign-blind divergent value).

    The setup deliberately uses a synthetic HBT-EP-style uniform-direction
    TF bundle (every coil current shares the same sign), so the SSOT
    ``sum_signed`` formula produces a non-zero seed. Stellsym-symmetric coil
    fixtures (NCSX, HSX, Giuliani) have ``sum_signed == 0`` by construction
    and so are not a valid stress test for the sign of ``G``.
    """

    @staticmethod
    def _build_hbt_style_tf_coil_bundle(*, signed_current_A, num_coils=20):
        """Return ``(tf_coils, bs)`` for a synthetic HBT-EP-style TF bundle.

        Coils are dummy ``CurveXYZFourier`` placeholders — the wire-in test
        cares about the signed ``G`` produced from the bundle's currents and
        about the bundle being a subset of ``bs.coils``; the field's exact
        spatial pattern is not the observable.
        """
        from simsopt.geo import CurveXYZFourier

        coils = [
            Coil(CurveXYZFourier(quadpoints=8, order=1), Current(signed_current_A))
            for _ in range(num_coils)
        ]
        bs = BiotSavart(coils)
        return coils, bs

    def test_build_stage2_iota_runtime_passes_signed_negative_G_for_cw_tf_bundle(self):
        """Production wire-in observable: a CW TF bundle handed to
        ``build_stage2_iota_runtime`` results in the *signed* (negative)
        ``G`` seed being routed into ``attempt_initialize_boozer_surface``,
        not the upstream sign-blind unsigned default."""
        import importlib
        from types import SimpleNamespace

        _import_examples_path()
        stage2_objectives = importlib.import_module("banana_opt.stage2_objectives")

        cw_current_A = -8.0e4
        tf_coils, bs = self._build_hbt_style_tf_coil_bundle(
            signed_current_A=cw_current_A,
        )
        recorded = {}

        def fake_attempt_initialize_boozer_surface(*_args, **kwargs):
            G0 = _args[7] if len(_args) > 7 else kwargs.get("G0")
            iota = _args[6] if len(_args) > 6 else kwargs.get("iota")
            bs_arg = _args[3] if len(_args) > 3 else kwargs.get("bs")
            recorded["G0"] = G0
            recorded["bs"] = bs_arg
            recorded["iota"] = iota
            fake_boozer_surface = SimpleNamespace(
                surface=SimpleNamespace(
                    volume=lambda: 0.1,
                    x=np.zeros(2, dtype=float),
                ),
                res={"iota": iota, "G": G0, "success": True, "type": "exact"},
                need_to_run_code=False,
                run_code=lambda _iota, _G: {"success": True, "iota": _iota, "G": _G},
            )
            return SimpleNamespace(
                success=True,
                boozer_surface=fake_boozer_surface,
                solve_success=True,
                self_intersecting=False,
                solved_iota=iota,
                solved_G=G0,
                error_type=None,
                error_message=None,
            )

        # Inject the surface-configs stub so the test does not depend on a
        # ``demo.nc`` equilibrium file; the wire-in observable is the value
        # of ``G0`` recorded by the fake init helper.
        stage2_objectives.build_stage2_iota_runtime(
            equilibrium_file="demo.nc",
            bs=bs,
            tf_coils=tf_coils,
            major_radius=0.976,
            toroidal_flux=0.24,
            nphi=31,
            ntheta=16,
            mpol=4,
            ntor=4,
            vol_target=0.1,
            iota_target=-0.2,
            iota_tolerance=5.0e-3,
            constraint_weight=1.0,
            num_tf_coils=len(tf_coils),
            mode="report",
            build_surface_configs_fn=lambda *_a, **_kw: [
                {
                    "initial_surface": SimpleNamespace(nfp=1),
                    "target_volume": 0.1,
                }
            ],
            attempt_initialize_boozer_surface_fn=fake_attempt_initialize_boozer_surface,
            iotas_cls=lambda _surface: SimpleNamespace(J=lambda: -0.2),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.0,
                dJ=lambda: np.zeros(2, dtype=float),
            ),
        )

        # SSOT contract: the wired-in seed must match
        # ``mu0 * sum_signed(I_TF)`` for the CW bundle (negative, with the
        # same magnitude the upstream sign-blind default would produce).
        expected_signed_G = derive_signed_G_from_field(bs, tf_coils=tf_coils)
        self.assertAlmostEqual(recorded["G0"], expected_signed_G)
        self.assertLess(recorded["G0"], 0.0)
        self.assertAlmostEqual(abs(recorded["G0"]), 4.0e-7 * np.pi * 20 * 8.0e4)
        self.assertIs(recorded["bs"], bs)

    def test_attempt_initialize_boozer_surface_routes_signed_G_through_run_code(self):
        """The Stage 2 ``attempt_initialize_boozer_surface`` chain hands the
        caller's signed ``G`` straight into ``BoozerSurfaceFiniteI.run_code``,
        which now requires an explicit (signed) ``G`` and would raise on the
        sign-blind ``G=None`` upstream fallback. Verifies the run_code call
        site preserves the signed seed end-to-end (the contract that
        ``test_banana_boozer_run_code_calls_pass_explicit_G`` static-checks).
        """
        import importlib
        from types import SimpleNamespace

        _import_examples_path()
        handoff = importlib.import_module(
            "banana_opt.stage2_single_stage_handoff"
        )

        cw_current_A = -8.0e4
        tf_coils, bs = self._build_hbt_style_tf_coil_bundle(
            signed_current_A=cw_current_A,
        )
        signed_G = derive_signed_G_from_field(bs, tf_coils=tf_coils)
        seen = {}

        class _CapturingBoozerSurface:
            def __init__(self, bs_inner, surf, vol, vol_target, constraint_weight,
                         options=None, I=0.0):
                del options
                self.bs = bs_inner
                self.surface = surf
                self.targetlabel = vol_target
                self.constraint_weight = constraint_weight
                self.I = I
                self.res = None
                self.need_to_run_code = True

            def run_code(self, iota, G):
                # The wire-in observable: the production setup chain hands
                # the signed seed to ``run_code`` without losing the sign.
                seen["iota"] = iota
                seen["G"] = G
                self.res = {"iota": iota, "G": G, "success": True, "type": "exact"}
                self.need_to_run_code = False
                return {"success": True, "iota": iota, "G": G}

        surf_prev = SimpleNamespace(
            quadpoints_theta=np.array([0.0, 0.5]),
            quadpoints_phi=np.array([0.0, 0.2]),
            gamma=lambda: np.zeros((2, 2, 3), dtype=float),
        )

        class _FakeSurface:
            def __init__(self, **kwargs):
                self.quadpoints_theta = kwargs["quadpoints_theta"]
                self.quadpoints_phi = kwargs["quadpoints_phi"]
                self.dofs = np.zeros(2, dtype=float)
                self.x = np.zeros(2, dtype=float)
                self._gamma = np.zeros((2, 2, 3), dtype=float)

            def least_squares_fit(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)

            def gamma(self):
                return self._gamma.copy()

            def is_self_intersecting(self):
                return False

        class _ConstantVolumeLabel:
            def __init__(self, surface):
                self.surface = surface

            def J(self):
                return 0.1

        handoff.attempt_initialize_boozer_surface(
            surf_prev,
            mpol=2,
            ntor=2,
            bs=bs,
            vol_target=0.1,
            constraint_weight=1.0,
            iota=-0.2,
            G0=signed_G,
            boozer_I=0.0,
            nfp=5,
            surface_cls=_FakeSurface,
            volume_cls=_ConstantVolumeLabel,
            boozer_surface_cls=_CapturingBoozerSurface,
        )

        # The signed seed makes it through the production chain unchanged.
        self.assertAlmostEqual(seen["G"], signed_G)
        self.assertLess(seen["G"], 0.0)
        self.assertAlmostEqual(seen["iota"], -0.2)


def _import_examples_path():
    """Put the ``examples/single_stage_optimization`` source root on the path
    so ``banana_opt`` and its submodules import as in production."""
    import sys
    from pathlib import Path

    examples_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "single_stage_optimization"
    )
    if str(examples_path) not in sys.path:
        sys.path.insert(0, str(examples_path))


class SignedGWireInNewtonConvergenceObservableTests(unittest.TestCase):
    """Newton-convergence observable: the wired-in signed seed lands the
    Boozer Newton on a finite, in-band fixed point, while replacing the seed
    with the *negation* of the signed value (mimicking the failure mode the
    upstream sign-blind ``G=None`` fallback would create for a CW TF bundle)
    drives Newton to a divergent or non-physical fixed point.

    The fixture is the existing NCSX-stellsym Boozer-surface convergence
    pair (cf. :meth:`test_finite_current_exact_newton_converges_on_task25_lane4_fixture`):
    that lane has the upstream-equivalent ``G > 0`` seed, and reproducing it
    here keeps the test entirely on shared deterministic upstream fixtures
    while still demonstrating Newton's *iota* observable shifts when the
    seed sign flips. The CW HBT-style case is covered as a wire-in-plumbing
    test in :class:`SignedGWireInBoozerNewtonConvergenceTests`; this class
    pins the convergence-behavior shift on a fixture where Newton actually
    converges to a known fixed point.
    """

    @staticmethod
    def _make_ncsx_finite_i_boozer_surface(*, G_seed_sign):
        """Reproduce the
        ``test_finite_current_exact_newton_converges_on_task25_lane4_fixture``
        setup, but parameterized by the sign of the G seed handed to
        ``run_code``. ``G_seed_sign=+1`` mirrors the existing upstream-equivalent
        lane; ``G_seed_sign=-1`` is the sign-flipped seed that would arise
        if a downstream caller silently inverted the SSOT helper's signed
        output.
        """
        from simsopt.geo.surfaceobjectives import Area as _Area

        mpol = 3
        ntor = 3
        current_I = 4 * np.pi * 1e-7 * 5000

        curves, currents, ma = get_ncsx_data()
        coils = coils_via_symmetries(curves, currents, 3, True)
        bs = BiotSavart(coils)

        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            stellsym=True,
            nfp=3,
            quadpoints_phi=np.linspace(0, 1 / 3, 2 * ntor + 1, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
        )
        surface.fit_to_curve(ma, 0.1, flip_theta=True)
        label = _Area(surface)
        boozer_surface = BoozerSurfaceFiniteI(
            bs, surface, label, label.J(),
            constraint_weight=None,
            options={"weight_inv_modB": False, "verbose": False},
            I=current_I,
        )
        # Match the magnitude used by the upstream-equivalent fixture.
        # ``coils_via_symmetries`` with stellsym=True yields ``sum_signed=0``;
        # the upstream sign-blind default falls back to ``sum_abs``. Use
        # that magnitude with the caller-chosen sign so the test exposes
        # the Newton-convergence shift driven by the *sign* of ``G``.
        magnitude = 2.0 * np.pi * sum(
            abs(c.current.get_value()) for c in coils
        ) * (4.0 * np.pi * 1e-7 / (2.0 * np.pi))
        return boozer_surface, G_seed_sign * magnitude

    def test_signed_G_seed_lands_newton_in_band(self):
        """Driving ``run_code`` with the *positive* (matched-sign) G seed
        converges to the published in-band fixed point, mirroring the
        existing fixture-pinned NCSX lane."""
        boozer_surface, signed_G = self._make_ncsx_finite_i_boozer_surface(
            G_seed_sign=+1.0,
        )
        res = boozer_surface.run_code(0.4, G=signed_G)
        self.assertTrue(res["success"])
        np.testing.assert_allclose(
            res["iota"], 0.40283946329212617, rtol=1e-10, atol=1e-12,
        )
        np.testing.assert_allclose(
            res["G"], 13.881987793895558, rtol=1e-10, atol=1e-12,
        )
        # Observable: solved iota lies inside the loose physical band
        # ``|iota| < 1`` (the same band the production Boozer trust gate
        # rejects out of, with reason ``iota_nonphysical``).
        self.assertLess(abs(res["iota"]), 1.0)

    def test_sign_flipped_G_seed_lands_newton_on_distinct_fixed_point(self):
        """Driving the same Newton with the *negated* G seed (the failure
        mode a downstream caller would introduce by accidentally flipping
        the sign of the SSOT helper's output, or by feeding the upstream
        sign-blind unsigned magnitude to a CW TF lane) drives the Newton
        solve off the converged in-band fixed point: on this NCSX-stellsym
        lane Newton fails to converge, which is the divergence observable
        the SSOT signed-G seed exists to prevent. The signed-G wire-in
        eliminates the ambiguity by routing the SSOT-correct sign through
        end-to-end.
        """
        signed_boozer, signed_G = self._make_ncsx_finite_i_boozer_surface(
            G_seed_sign=+1.0,
        )
        signed_res = signed_boozer.run_code(0.4, G=signed_G)
        self.assertTrue(signed_res["success"])

        flipped_boozer, flipped_G = self._make_ncsx_finite_i_boozer_surface(
            G_seed_sign=-1.0,
        )
        flipped_res = flipped_boozer.run_code(0.4, G=flipped_G)

        # Seed-sign observable: the seed handed to ``run_code`` keeps its
        # chirality (sanity check on the fixture).
        self.assertLess(flipped_G, 0.0)
        # Divergence observable: the SSOT signed-G seed lands Newton on
        # the converged in-band fixed point (asserted above for the
        # positive seed). The sign-flipped seed must NOT land on a
        # converged fixed point — otherwise the SSOT signed-G claim
        # ("sign determines convergence") would be vacuous. On this
        # NCSX-stellsym lane Newton fails (``success=False``); pin that
        # directly so the test cannot silently become a tautology.
        self.assertFalse(
            flipped_res["success"],
            "Newton must diverge on sign-flipped G seed; otherwise the "
            "signed-G SSOT claim is unobservable on this fixture.",
        )


if __name__ == "__main__":
    unittest.main()
