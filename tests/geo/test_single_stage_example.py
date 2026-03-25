import importlib.util
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simsopt.geo.surfaceobjectives import boozer_surface_residual, boozer_surface_residual_dB
from simsopt.objectives.utilities import forward_backward


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0


def load_single_stage_example_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        EXAMPLE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeSurfPrev:
    def __init__(self):
        self.nfp = 5
        self.quadpoints_phi = np.linspace(0, 1 / self.nfp, 13, endpoint=False)
        self.quadpoints_theta = np.linspace(0, 1, 17, endpoint=False)

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeSurfaceXYZTensorFourier:
    instances = []

    def __init__(
        self,
        *,
        mpol,
        ntor,
        nfp,
        stellsym,
        quadpoints_theta,
        quadpoints_phi,
        dofs=None,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.dofs = np.array([1.0]) if dofs is None else np.asarray(dofs)
        FakeSurfaceXYZTensorFourier.instances.append(self)

    def least_squares_fit(self, gamma):
        self.fitted_gamma = gamma

    def is_self_intersecting(self):
        return False

    def volume(self):
        return 1.0


class FakeVolume:
    def __init__(self, surface):
        self.surface = surface


class FakeBoozerSurface:
    def __init__(self, bs, surface, label, targetlabel, constraint_weight, options=None):
        self.bs = bs
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.options = options or {}
        self.res = {"success": True, "iota": 0.15, "G": 1.0}

    def run_code(self, iota, G):
        return self.res


class FakeAlgebraicObjective:
    def __init__(self, value, gradient):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)

    def J(self):
        return self._value

    def dJ(self):
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return FakeAlgebraicObjective(self._value + other._value, self._gradient + other._gradient)

    __radd__ = __add__

    def __mul__(self, scalar):
        return FakeAlgebraicObjective(self._value * scalar, self._gradient * scalar)

    __rmul__ = __mul__


class SingleStageExampleTests(unittest.TestCase):
    def setUp(self):
        FakeSurfaceXYZTensorFourier.instances = []

    def load_module(self):
        return load_single_stage_example_module()

    def initialize_boozer_surface(self, module, surf_prev, *, constraint_weight):
        with patch.object(module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier), patch.object(
            module, "Volume", FakeVolume
        ), patch.object(module, "BoozerSurface", FakeBoozerSurface):
            return module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=constraint_weight,
                iota=TEST_IOTA,
                G0=TEST_G0,
            )

    def test_exact_boozer_helpers_are_imported(self):
        module = self.load_module()

        self.assertIs(module.boozer_surface_residual, boozer_surface_residual)
        self.assertIs(module.boozer_surface_residual_dB, boozer_surface_residual_dB)
        self.assertIs(module.forward_backward, forward_backward)

    def test_initialize_boozer_surface_exact_uses_ntor_phi_quadrature(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=None)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 2)

        exact_surface = FakeSurfaceXYZTensorFourier.instances[1]
        expected_phi = np.linspace(0, 1 / surf_prev.nfp, 2 * TEST_NTOR + 1, endpoint=False)

        self.assertEqual(exact_surface.quadpoints_theta.size, 2 * TEST_MPOL + 1)
        self.assertEqual(exact_surface.quadpoints_phi.size, 2 * TEST_NTOR + 1)
        np.testing.assert_allclose(exact_surface.quadpoints_phi, expected_phi)

    def test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=0.0)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])

    def test_fun_fallback_returns_elevated_j_and_same_sign_gradient(self):
        """Issue #2: failed Boozer must return elevated J + same-sign gradient,
        not (J_old, -dJ_old)."""
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)
            def is_self_intersecting(self):
                return False
            def volume(self):
                return 1.0
            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": False, "iota": TEST_IOTA, "G": TEST_G0}
            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [{"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(5), "lscount": 0,
            "surface_state": {"sdofs": [np.ones(3)], "iota": [TEST_IOTA], "G": [TEST_G0]},
            "J": last_J, "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertIsNot(dJ_out, module.run_dict["dJ"])

    def test_evaluate_topology_gate_reports_early_surface_exit(self):
        module = self.load_module()

        class _Surface:
            def cross_section(self, phi, thetas):
                theta = np.asarray(thetas)
                return np.column_stack(
                    [
                        1.0 + 0.1 * np.cos(2 * np.pi * theta),
                        np.zeros_like(theta),
                        0.1 * np.sin(2 * np.pi * theta),
                    ]
                )

        class _Stop:
            def __init__(self, *args, **kwargs):
                pass

        fake_hits = [
            np.array([[0.8, -1.0, 1.0, 0.0, 0.0]]),
            np.array([]),
        ]

        with patch.object(module, "SurfaceClassifier", return_value=SimpleNamespace(dist=lambda xyz: 1.0)), patch.object(
            module, "LevelsetStoppingCriterion", _Stop
        ), patch.object(module, "MaxZStoppingCriterion", _Stop), patch.object(
            module, "MinZStoppingCriterion", _Stop
        ), patch.object(module, "MinRStoppingCriterion", _Stop), patch.object(
            module, "MaxRStoppingCriterion", _Stop
        ), patch.object(module, "compute_fieldlines", return_value=([], fake_hits)):
            status = module.evaluate_topology_gate(_Surface(), object(), 2, 2.0, 1e-7, 0.75)

        self.assertTrue(status["enabled"])
        self.assertFalse(status["success"])
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)
        self.assertEqual(status["first_exit_reason"], "surface_exit")
        self.assertAlmostEqual(status["first_exit_time"], 0.8)

    def test_topology_gate_rejection_increment_scales_with_deficit(self):
        module = self.load_module()

        status = {
            "enabled": True,
            "survival_fraction": 0.5,
            "survival_threshold": 0.75,
        }

        self.assertAlmostEqual(module.topology_gate_deficit(status), 0.25)
        self.assertAlmostEqual(
            module.topology_gate_rejection_increment(42.0, status, 4.0),
            84.0,
        )

    def test_fun_rejects_candidate_on_topology_gate_failure(self):
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [{"boozer_surface": _BoozerSurface()}, {"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {"sdofs": [np.ones(3), np.ones(3)], "iota": [TEST_IOTA, TEST_IOTA], "G": [TEST_G0, TEST_G0]},
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [SimpleNamespace(J=lambda: TEST_IOTA), SimpleNamespace(J=lambda: TEST_IOTA)]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = object()
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = object()
        module.CC_WEIGHT = 1.0
        module.JCurveSurface = object()
        module.CS_WEIGHT = 1.0
        module.JCurvature = object()
        module.CURVATURE_WEIGHT = 1.0
        module.JSurfSurf = None
        module.SURF_DIST_WEIGHT = 0.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 4
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value={
                "success": True,
                "solve_success": [True, True],
                "self_intersections": [False, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
                "adjacent_gaps": [0.01],
                "outer_vessel_gap": 0.1,
                "bad_nesting_phis": [],
            },
        ), patch.object(
            module,
            "evaluate_total_objective",
            return_value={"total": 7.0, "grad": np.arange(5, dtype=float), "surface_weights": np.array([1.0, 1.0])},
        ), patch.object(module, "restore_surface_states") as restore_mock, patch.object(
            module,
            "evaluate_topology_gate",
            return_value={
                "enabled": True,
                "success": False,
                "nfieldlines": 4,
                "survived_lines": 0,
                "survival_fraction": 0.0,
                "survival_threshold": 0.25,
                "tmax": 2.0,
                "tol": 1e-7,
                "stop_reason_counts": {"surface_exit": 4},
                "first_exit_time": 0.4,
                "first_exit_angle": 0.2,
                "first_exit_reason": "surface_exit",
            },
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, 126.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()

    def test_build_surface_configs_two_surface_mode_derives_inner_target_volume(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale ** 3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=2,
                inner_surface_ratio=0.8,
            )

        self.assertEqual([config["name"] for config in configs], ["inner", "outer"])
        self.assertAlmostEqual(configs[0]["seed_label"], 0.20)
        self.assertAlmostEqual(configs[1]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.08)
        self.assertAlmostEqual(configs[1]["target_volume"], 0.10)

    def test_build_surface_configs_single_surface_mode_keeps_outer_only_contract(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale ** 3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "outer")
        self.assertAlmostEqual(configs[0]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.10)

    def test_average_surface_objectives_uses_mean_and_preserves_single_surface_scale(self):
        module = self.load_module()

        single = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        single_avg = module.average_surface_objectives([single])
        self.assertAlmostEqual(single_avg.J(), 2.0)
        np.testing.assert_allclose(single_avg.dJ(), [2.0, -1.0])

        left = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        right = FakeAlgebraicObjective(6.0, [4.0, 3.0])
        pair_avg = module.average_surface_objectives([left, right])
        self.assertAlmostEqual(pair_avg.J(), 4.0)
        np.testing.assert_allclose(pair_avg.dJ(), [3.0, 1.0])

    def test_build_surface_search_weights_ramps_inner_surface_only(self):
        module = self.load_module()

        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=1,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.0, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=2,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.4, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=5,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0, 1.0],
        )

    def test_build_surface_search_gate_ramps_thresholds_and_nesting(self):
        module = self.load_module()

        single = module.build_surface_search_gate(
            num_surfaces=1,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertEqual(single["surface_gap_threshold"], 0.005)
        self.assertEqual(single["vessel_gap_threshold"], 0.04)
        self.assertTrue(single["enforce_nesting"])
        self.assertEqual(single["gate_scale"], 1.0)

        start = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertEqual(start["surface_gap_threshold"], 0.0)
        self.assertEqual(start["vessel_gap_threshold"], 0.0)
        self.assertFalse(start["enforce_nesting"])
        self.assertEqual(start["gate_scale"], 0.0)

        mid = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=2,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertAlmostEqual(mid["surface_gap_threshold"], 0.002)
        self.assertAlmostEqual(mid["vessel_gap_threshold"], 0.016)
        self.assertFalse(mid["enforce_nesting"])
        self.assertAlmostEqual(mid["gate_scale"], 0.4)

        done = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=5,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertAlmostEqual(done["surface_gap_threshold"], 0.005)
        self.assertAlmostEqual(done["vessel_gap_threshold"], 0.04)
        self.assertTrue(done["enforce_nesting"])
        self.assertAlmostEqual(done["gate_scale"], 1.0)

    def test_build_scaled_outer_problem_scales_coordinates_gradients_and_callback(self):
        module = self.load_module()
        seen = {"fun": [], "callback": []}

        def base_fun(x):
            seen["fun"].append(np.asarray(x, dtype=float).copy())
            return 7.5, np.array([3.0, -4.0])

        def base_callback(x):
            seen["callback"].append(np.asarray(x, dtype=float).copy())

        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            np.array([10.0, 20.0]),
            0.1,
        )

        J, dJ = scaled_fun(np.array([1.0, -2.0]))
        self.assertAlmostEqual(J, 7.5)
        np.testing.assert_allclose(dJ, [0.3, -0.4])
        np.testing.assert_allclose(seen["fun"][0], [10.1, 19.8])

        scaled_callback(np.array([1.0, -2.0]))
        np.testing.assert_allclose(seen["callback"][0], [10.1, 19.8])

    def test_evaluate_total_objective_uses_surface_weights_for_qs_and_boozer_terms(self):
        module = self.load_module()

        nonqs = [
            FakeAlgebraicObjective(2.0, [2.0, 0.0]),
            FakeAlgebraicObjective(6.0, [4.0, 0.0]),
        ]
        brs = [
            FakeAlgebraicObjective(10.0, [1.0, 1.0]),
            FakeAlgebraicObjective(20.0, [3.0, 3.0]),
        ]
        zero = FakeAlgebraicObjective(0.0, [0.0, 0.0])

        outer_only = module.evaluate_total_objective(
            np.array([0.0, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(outer_only["J_QS"], 6.0)
        self.assertAlmostEqual(outer_only["J_Boozer"], 20.0)
        np.testing.assert_allclose(outer_only["dJ_QS"], [4.0, 0.0])
        np.testing.assert_allclose(outer_only["dJ_Boozer"], [3.0, 3.0])
        self.assertAlmostEqual(outer_only["total"], 46.0)
        np.testing.assert_allclose(outer_only["grad"], [10.0, 6.0])

        ramped = module.evaluate_total_objective(
            np.array([0.5, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(ramped["J_QS"], (0.5 * 2.0 + 6.0) / 1.5)
        self.assertAlmostEqual(ramped["J_Boozer"], (0.5 * 10.0 + 20.0) / 1.5)
        np.testing.assert_allclose(ramped["dJ_QS"], [10.0 / 3.0, 0.0])
        np.testing.assert_allclose(ramped["dJ_Boozer"], [7.0 / 3.0, 7.0 / 3.0])
        self.assertAlmostEqual(ramped["total"], 38.0)
        np.testing.assert_allclose(ramped["grad"], [8.0, 14.0 / 3.0])

    def test_build_total_objective_skips_missing_surface_vessel_term_without_injecting_int(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            FakeAlgebraicObjective(7.0, [2.0, 0.0]),
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            SURF_DIST_WEIGHT=1000.0,
            JSurfSurf=None,
        )

        self.assertAlmostEqual(total.J(), 1 + 2*3 + 4*5 + 6*7 + 8*9 + 10*11 + 12*13)
        np.testing.assert_allclose(total.dJ(), [33.0, 28.0])

    def test_evaluate_surface_stack_rejects_unordered_or_too_close_surfaces(self):
        module = self.load_module()

        class _FakeSurface:
            def __init__(self, volume, points, self_intersecting=False):
                self._volume = volume
                self._points = np.asarray(points, dtype=float).reshape((-1, 1, 3))
                self._self_intersecting = self_intersecting

            def volume(self):
                return self._volume

            def gamma(self):
                return self._points

            def is_self_intersecting(self):
                return self._self_intersecting

        good_stack = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        vessel = _FakeSurface(0.2, [[1.0, 0.0, 0.0]])
        good_status = module.evaluate_surface_stack(good_stack, vessel_surface=vessel, surface_gap_threshold=0.1, vessel_gap_threshold=0.1)
        self.assertTrue(good_status["success"])
        self.assertEqual(good_status["adjacent_gaps"], [0.4])

        bad_order = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.11, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        order_status = module.evaluate_surface_stack(bad_order)
        self.assertFalse(order_status["success"])
        self.assertFalse(order_status["volumes_ordered"])

        bad_gap = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.05, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        gap_status = module.evaluate_surface_stack(bad_gap, surface_gap_threshold=0.1)
        self.assertFalse(gap_status["success"])
        self.assertFalse(gap_status["gap_ok"])

    def test_evaluate_surface_stack_rejects_cross_section_nesting_failure(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, cross_section):
                self._volume = volume
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._cross_section.reshape((-1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, inner_crossing), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, outer_box), res={"success": True, "iota": 0.15})},
        ]

        status = module.evaluate_surface_stack(surface_data)
        self.assertFalse(status["success"])
        self.assertFalse(status["nesting_ok"])
        self.assertTrue(status["bad_nesting_phis"])

    def test_evaluate_surface_stack_can_skip_nesting_during_search_continuation(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [0.0, 0.0, 0.0], inner_crossing), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [0.4, 0.0, 0.0], outer_box), res={"success": True, "iota": 0.15})},
        ]

        relaxed = module.evaluate_surface_stack(surface_data, enforce_nesting=False)
        self.assertTrue(relaxed["success"])
        self.assertTrue(relaxed["nesting_ok"])
        self.assertEqual(relaxed["bad_nesting_phis"], [])

        strict = module.evaluate_surface_stack(surface_data, enforce_nesting=True)
        self.assertFalse(strict["success"])
        self.assertFalse(strict["nesting_ok"])

    def test_fun_multisurface_fallback_restores_all_surface_state(self):
        module = self.load_module()
        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            nfp = 5

            def __init__(self, volume, point):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return np.array([[1.0, 0.0, -0.1], [1.1, 0.0, 0.0], [1.0, 0.0, 0.1], [0.9, 0.0, 0.0]])

        class _BoozerSurface:
            def __init__(self, surface, success):
                self.surface = surface
                self.res = {"success": success, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                self.res["iota"] = iota + 0.1
                self.res["G"] = G + 0.2
                return self.res

        class _JF:
            x = np.zeros(5)

        inner = _BoozerSurface(_Surface(0.08, [0.0, 0.0, 0.0]), True)
        outer = _BoozerSurface(_Surface(0.10, [0.4, 0.0, 0.0]), False)
        module.surface_data = [
            {"boozer_surface": inner},
            {"boozer_surface": outer},
        ]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.array([0.08]), np.array([0.10])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        np.testing.assert_array_equal(inner.surface.x, np.array([0.08]))
        np.testing.assert_array_equal(outer.surface.x, np.array([0.10]))
        self.assertEqual(inner.res["iota"], 0.12)
        self.assertEqual(outer.res["iota"], 0.15)

    def test_callback_multisurface_records_status_and_log(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(self._value + other.J(), self.dJ() + other.dJ())

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        inner_cs = [[0.85, 0.0, -0.1], [1.05, 0.0, -0.1], [1.05, 0.0, 0.1], [0.85, 0.0, 0.1]]
        outer_cs = [[0.7, 0.0, -0.3], [1.3, 0.0, -0.3], [1.3, 0.0, 0.3], [0.7, 0.0, 0.3]]
        inner = SimpleNamespace(surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_cs), res={"success": True, "iota": 0.12, "G": 1.0})
        outer = SimpleNamespace(surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_cs), res={"success": True, "iota": 0.15, "G": 1.1})

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {"name": "inner", "seed_label": 0.16, "target_volume": 0.08, "boozer_surface": inner},
                {"name": "outer", "seed_label": 0.20, "target_volume": 0.10, "boozer_surface": outer},
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.0
            module.SS_DIST = 0.0
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JSurfSurf = None
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CS_WEIGHT = 1.0
            module.CURVATURE_WEIGHT = 0.1
            module.SURF_DIST_WEIGHT = 1000.0
            module.banana_curve = _Curve()
            module.curvelength = _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "topology_gate_status": {"enabled": False, "success": True, "nfieldlines": 0, "survived_lines": 0, "survival_fraction": 1.0, "survival_threshold": 0.25, "tmax": 2.0, "tol": 1e-7, "stop_reason_counts": {}, "first_exit_time": None, "first_exit_angle": None, "first_exit_reason": None},
            }

            module.callback(np.zeros(2))

            self.assertEqual(module.run_dict["surface_status"]["adjacent_gaps"], [0.4])
            self.assertEqual(module.run_dict["search_surface_status"]["adjacent_gaps"], [0.4])
            self.assertTrue(module.run_dict["surface_status"]["nesting_ok"])
            log_path = Path(tmpdir) / "log.txt"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text()
            self.assertIn("Adjacent surface gaps", log_text)
            self.assertIn("Surfaces nested", log_text)
            self.assertIn("Surface gate scale", log_text)

    def test_callback_tracks_relaxed_search_status_separately_from_full_status(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(self._value + other.J(), self.dJ() + other.dJ())

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        inner = SimpleNamespace(surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_crossing), res={"success": True, "iota": 0.12, "G": 1.0})
        outer = SimpleNamespace(surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_box), res={"success": True, "iota": 0.15, "G": 1.1})

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {"name": "inner", "seed_label": 0.16, "target_volume": 0.08, "boozer_surface": inner},
                {"name": "outer", "seed_label": 0.20, "target_volume": 0.10, "boozer_surface": outer},
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.005
            module.SS_DIST = 0.0
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JSurfSurf = None
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CS_WEIGHT = 1.0
            module.CURVATURE_WEIGHT = 0.1
            module.SURF_DIST_WEIGHT = 1000.0
            module.banana_curve = _Curve()
            module.curvelength = _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "topology_gate_status": {"enabled": False, "success": True, "nfieldlines": 0, "survived_lines": 0, "survival_fraction": 1.0, "survival_threshold": 0.25, "tmax": 2.0, "tol": 1e-7, "stop_reason_counts": {}, "first_exit_time": None, "first_exit_angle": None, "first_exit_reason": None},
            }

            module.callback(np.zeros(2))

            self.assertTrue(module.run_dict["search_surface_status"]["success"])
            self.assertTrue(module.run_dict["search_surface_status"]["nesting_ok"])
            self.assertFalse(module.run_dict["surface_status"]["success"])
            self.assertFalse(module.run_dict["surface_status"]["nesting_ok"])

    def test_finalize_surface_stack_reverts_to_last_accepted_state_when_final_endpoint_is_invalid(self):
        module = self.load_module()

        class _Objective:
            def __init__(self):
                self.x = np.array([0.0])

            def J(self):
                return float(self.x[0] + 10.0)

            def dJ(self):
                return np.array([self.x[0] + 1.0])

        class _Surface:
            nfp = 5

            def __init__(self, accepted_x, accepted_volume, point):
                self.x = np.array([accepted_x], dtype=float)
                self._volume = accepted_volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                radius = self._point[0]
                return np.array([
                    [radius - 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, 0.05],
                    [radius - 0.05, 0.0, 0.05],
                ])

        class _BoozerSurface:
            def __init__(self, surface, objective, valid_limit, success_iota):
                self.surface = surface
                self._objective = objective
                self._valid_limit = valid_limit
                self._success_iota = success_iota
                self.res = {"success": True, "iota": success_iota, "G": 1.0}

            def run_code(self, iota, G):
                current = float(self._objective.x[0])
                self.surface.x = np.array([current], dtype=float)
                self.surface._volume = 0.08 if self._success_iota < 0.2 else 0.10
                self.res["success"] = current <= self._valid_limit
                self.res["iota"] = self._success_iota if self.res["success"] else -1.0
                self.res["G"] = G
                return self.res

        objective = _Objective()
        inner_surface = _Surface(1.0, 0.08, [0.2, 0.0, 0.0])
        outer_surface = _Surface(1.0, 0.10, [0.6, 0.0, 0.0])
        surface_data = [
            {"boozer_surface": _BoozerSurface(inner_surface, objective, valid_limit=1.5, success_iota=0.12)},
            {"boozer_surface": _BoozerSurface(outer_surface, objective, valid_limit=1.5, success_iota=0.15)},
        ]
        run_state = {
            "surface_state": {
                "sdofs": [np.array([1.0]), np.array([1.0])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "accepted_x": np.array([1.0]),
            "J": 11.0,
            "dJ": np.array([2.0]),
            "intersecting": False,
        }

        status = module.finalize_surface_stack(np.array([2.0]), objective, surface_data, run_state)

        self.assertFalse(status["success"])
        np.testing.assert_allclose(objective.x, [1.0])
        np.testing.assert_allclose(run_state["accepted_x"], [1.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [1.0])
        np.testing.assert_allclose(surface_data[1]["boozer_surface"].surface.x, [1.0])
        self.assertEqual(run_state["surface_state"]["iota"], [0.12, 0.15])

    def test_collect_surface_run_metadata_serializes_multisurface_fields(self):
        module = self.load_module()
        surface_data = [
            {"name": "inner", "seed_label": 0.16, "target_volume": 0.08},
            {"name": "outer", "seed_label": 0.20, "target_volume": 0.10},
        ]
        run_status = {
            "self_intersections": [False, False],
            "adjacent_gaps": [0.4],
            "outer_vessel_gap": 0.6,
            "nesting_ok": True,
            "bad_nesting_phis": [],
        }
        payload = module.collect_surface_run_metadata(
            surface_data,
            run_status,
            initial_surface_volumes=[0.08, 0.10],
            initial_surface_iotas=[0.12, 0.15],
            final_surface_volumes=[0.081, 0.101],
            final_surface_iotas=[0.121, 0.151],
        )

        self.assertEqual(payload["SURFACE_NAMES"], ["inner", "outer"])
        self.assertEqual(payload["ADJACENT_SURFACE_GAPS"], [0.4])
        self.assertTrue(payload["SURFACES_NESTED"])
        self.assertEqual(payload["FINAL_SURFACE_VOLUMES"], [0.081, 0.101])


class BoozerFallbackLBFGSBTests(unittest.TestCase):
    """Issue #2: elevated-J fallback must not flush L-BFGS-B Hessian memory."""

    def test_elevated_j_stale_gradient_preserves_bfgs_memory(self):
        from scipy.optimize import minimize

        def rosenbrock(x):
            f = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                    for i in range(len(x) - 1))
            g = np.zeros_like(x)
            for i in range(len(x) - 1):
                g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
                g[i+1] += 200*(x[i+1] - x[i]**2)
            return f, g

        rng = np.random.RandomState(42)
        x0 = rng.randn(10) * 0.5
        state = {"x_good": x0.copy(), "J": None, "dJ": None}

        def fun_with_fallback(x):
            f, g = rosenbrock(x)
            if np.linalg.norm(x - state["x_good"]) > 0.5 and state["J"] is not None:
                return state["J"] + max(abs(state["J"]), 1.0), state["dJ"].copy()
            state["J"] = f
            state["dJ"] = g.copy()
            state["x_good"] = x.copy()
            return f, g

        res = minimize(fun_with_fallback, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "maxcor": 10})

        self.assertTrue(res.success, f"L-BFGS-B did not converge: {res.message}")
        self.assertGreater(res.hess_inv.n_corrs, 0)


STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)


def _load_segment_distance_from_source():
    """Extract the deployed segment_segment_distance from banana_coil_solver.py via AST.

    Parses the source file, extracts just the _clamp01 and segment_segment_distance
    function definitions (stripping @njit decorators), and compiles them in an
    isolated namespace. This executes the REAL deployed algorithm without requiring
    numba or triggering module-level code (arg parsing, VMEC file loading, etc.).
    """
    import ast
    source = STAGE2_MODULE_PATH.read_text()
    tree = ast.parse(source)
    func_nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_clamp01", "segment_segment_distance"):
            node.decorator_list = []
            func_nodes.append(node)
    extracted = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {"np": np}
    exec(compile(extracted, str(STAGE2_MODULE_PATH), "exec"), namespace)
    return namespace["segment_segment_distance"]


_segment_segment_distance = _load_segment_distance_from_source()


def _brute_force_segment_distance(P1, P2, Q1, Q2):
    """Reference distance via interior + 4 edge candidates on [0,1]^2."""
    u, v, w0 = P2 - P1, Q2 - Q1, P1 - Q1
    a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d_val, e = np.dot(u, w0), np.dot(v, w0)
    cands = []
    denom = a * c - bv * bv
    if denom > 1e-30:
        sn = (bv * e - c * d_val) / denom
        tn = (a * e - bv * d_val) / denom
        if 0.0 <= sn <= 1.0 and 0.0 <= tn <= 1.0:
            dp = w0 + sn * u - tn * v
            cands.append(np.dot(dp, dp))
    for sf in [0.0, 1.0]:
        to = max(0.0, min(1.0, (e + sf * bv) / c)) if c > 1e-30 else 0.0
        dp = w0 + sf * u - to * v
        cands.append(np.dot(dp, dp))
    for tf in [0.0, 1.0]:
        so = max(0.0, min(1.0, (tf * bv - d_val) / a)) if a > 1e-30 else 0.0
        dp = w0 + so * u - tf * v
        cands.append(np.dot(dp, dp))
    return np.sqrt(min(cands))


class SegmentDistanceTests(unittest.TestCase):
    """Issue #5/#6: segment-segment distance with Sunday/Lumelsky re-projection."""

    def _d(self, p1, p2, q1, q2):
        return _segment_segment_distance(
            np.array(p1, dtype=float), np.array(p2, dtype=float),
            np.array(q1, dtype=float), np.array(q2, dtype=float),
        )

    def test_skew_segments_reprojection(self):
        """Issue #5: buggy=1.414, correct=sqrt(1.8) after re-projection."""
        d = self._d([0, 0, 0], [2, 1, 0], [-1, 3, 0], [1, 2, 0])
        self.assertAlmostEqual(d, np.sqrt(1.8), places=10)

    def test_parallel_overlapping_segments(self):
        """Issue #6: buggy=8.06, correct=1.0 for overlapping parallel segments."""
        d = self._d([0, 0, 0], [10, 0, 0], [8, 1, 0], [20, 1, 0])
        self.assertAlmostEqual(d, 1.0, places=10)

    def test_collinear_gap(self):
        d = self._d([0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_perpendicular_touching(self):
        d = self._d([0, 0, 0], [1, 0, 0], [0.5, 0, 0], [0.5, 1, 0])
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_point_to_segment(self):
        d = self._d([0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_parallel_non_overlapping(self):
        d = self._d([0, 0, 0], [3, 0, 0], [5, 1, 0], [8, 1, 0])
        self.assertAlmostEqual(d, np.sqrt(5.0), places=10)

    def test_t_shaped(self):
        d = self._d([0, 0, 0], [2, 0, 0], [1, 0.5, 0], [1, 3, 0])
        self.assertAlmostEqual(d, 0.5, places=10)

    def test_both_degenerate(self):
        d = self._d([1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6])
        self.assertAlmostEqual(d, np.linalg.norm([3, 3, 3]), places=10)

    def test_near_parallel_interior_minimum(self):
        """Near-parallel segments where the true minimum is at an interior point.

        P along x-axis, Q nearly parallel with tiny z-tilt and small y-offset.
        Endpoint projections all return sqrt(d^2 + eps^2) but the true minimum
        at (s=0.5, t=0.5) is just d (z components cancel at the midpoint).
        """
        eps = 9e-6
        d_offset = 1e-6
        d = self._d([-1, 0, 0], [1, 0, 0], [-1, d_offset, -eps], [1, d_offset, eps])
        self.assertAlmostEqual(d, d_offset, places=10)

    def test_near_parallel_brute_force(self):
        """Stress-test the near-parallel branch with 1000 adversarial pairs."""
        rng = np.random.RandomState(77777)
        PAR_EPS = 1e-10
        n_parallel = 0
        for _ in range(1000):
            base = rng.randn(3)
            base /= np.linalg.norm(base)
            P1 = rng.randn(3)
            P2 = P1 + rng.uniform(0.5, 5.0) * base
            angle = rng.uniform(1e-7, 1e-5)
            perp = rng.randn(3)
            perp -= np.dot(perp, base) * base
            perp /= np.linalg.norm(perp)
            q_dir = base + angle * perp
            q_dir /= np.linalg.norm(q_dir)
            Q1 = P1 + rng.randn(3) * rng.uniform(1e-7, 1e-4)
            Q2 = Q1 + rng.uniform(0.5, 5.0) * q_dir
            u, v, _ = P2 - P1, Q2 - Q1, P1 - Q1
            a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
            denom = a * c - bv * bv
            if denom < PAR_EPS * a * c:
                n_parallel += 1
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(d_algo, d_brute, places=9,
                                   msg=f"Near-parallel mismatch: algo={d_algo}, brute={d_brute}")
        self.assertGreater(n_parallel, 900, "Not enough pairs hit the near-parallel branch")

    def test_random_brute_force(self):
        """Verify against exhaustive interior + edge search on 1000 random pairs."""
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            P1, P2, Q1, Q2 = rng.randn(4, 3)
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(d_algo, d_brute, places=9,
                                   msg=f"Mismatch: algo={d_algo}, brute={d_brute}")


class CrossSectionNormalizationTests(unittest.TestCase):
    """Issue #8/#9: cross_section phi argument must be normalized to [0,1]."""

    PLOTTING_UTILS_PATH = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "single_stage_optimization"
        / "plotting_utils.py"
    )

    def test_plotting_utils_source_divides_by_2pi(self):
        """Verify the shared plotting_utils uses phi_slice / (2 * np.pi), not * 2 * np.pi."""
        source = self.PLOTTING_UTILS_PATH.read_text()
        self.assertIn("phi_slice / (2 * np.pi)", source)
        self.assertNotIn("phi_slice * 2 * np.pi", source)


class FtolGtolDefaultTests(unittest.TestCase):
    """Issue #31: ftol/gtol must not be None for any mpol value."""

    def test_ftol_gtol_have_defaults_for_all_mpol(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        for mpol in range(1, 30):
            ftol = ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)
            gtol = gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)
            self.assertIsNotNone(ftol, f"ftol is None for mpol={mpol}")
            self.assertIsNotNone(gtol, f"gtol is None for mpol={mpol}")
            self.assertIsInstance(ftol, float, f"ftol not float for mpol={mpol}")
            self.assertIsInstance(gtol, float, f"gtol not float for mpol={mpol}")
            self.assertGreater(ftol, 0, f"ftol not positive for mpol={mpol}")
            self.assertGreater(gtol, 0, f"gtol not positive for mpol={mpol}")

    def test_defaults_match_dictionary_endpoints(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        self.assertEqual(ftol_by_mpol.get(7, 1e-5 if 7 < 8 else 1e-10), 1e-5)
        self.assertEqual(ftol_by_mpol.get(19, 1e-5 if 19 < 8 else 1e-10), 1e-10)
        self.assertEqual(gtol_by_mpol.get(7, 1e-2 if 7 < 8 else 1e-7), 1e-2)
        self.assertEqual(gtol_by_mpol.get(19, 1e-2 if 19 < 8 else 1e-7), 1e-7)

    def test_source_uses_default_argument(self):
        """The deployed .get() calls must include a default, not bare .get(mpol)."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol.get(mpol)", source)
        self.assertNotIn("gtol_by_mpol.get(mpol)", source)


if __name__ == "__main__":
    unittest.main()
