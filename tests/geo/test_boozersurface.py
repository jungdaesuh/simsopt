import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import simsoptpp as sopp
from simsopt._core.json import GSONEncoder, SIMSON
from simsopt.field.coil import Current, CurrentSum, ScaledCurrent, coils_via_symmetries
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

        if surfacetype == 'SurfaceXYZTensorFourier':
            assert np.linalg.norm(res['residual']) < 1e-9

        print(ar_target, ar.J())
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
        for current_I in [0.0, 0.37]:
            boozer_surface = BoozerSurfaceFiniteI(bs, s, tf, tf_target, I=current_I)

            iota = -0.3
            x = np.concatenate((s.get_dofs(), [iota]))
            if optimize_G:
                x = np.concatenate((x, [2.*np.pi*current_sum*(4*np.pi*10**(-7)/(2 * np.pi))]))
            H0, H1 = self._assert_penalty_constraints_cpp_python_match(
                boozer_surface, x, optimize_G=optimize_G, weight_inv_modB=weight_inv_modB,
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


if __name__ == "__main__":
    unittest.main()
