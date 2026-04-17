import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


BASIN_HOPPING_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "basin_hopping.py"
)
EXPECTED_BASIN_TELEMETRY = {
    "basin_accepted_hops": 1,
    "basin_rejected_hops": 1,
    "basin_completed_hops": 2,
    "basin_best_objective": 2.0,
    "basin_initial_objective": 3.0,
    "basin_best_hop_objective": 2.0,
    "basin_best_hop_index": 1,
    "basin_best_result_source": "hop",
    "basin_objective_improvement": 1.0,
    "basin_accept_test_rejections": 1,
    "basin_accept_test_triggered": True,
}
EXPECTED_BASIN_TELEMETRY_FIELDS = (
    "basin_accepted_hops",
    "basin_rejected_hops",
    "basin_best_objective",
    "basin_accept_test_rejections",
    "basin_accept_test_triggered",
)


def load_basin_hopping_module():
    module_name = f"basin_hopping_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        BASIN_HOPPING_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    return module


class BasinHoppingHelperTests(unittest.TestCase):
    def test_normalized_step_rms_uses_relative_scale(self):
        module = load_basin_hopping_module()

        self.assertAlmostEqual(
            module._normalized_step_rms(np.array([10.0, 1.0]), np.array([15.0, 3.0])),
            np.sqrt(2.125),
        )

    def test_accept_test_rejects_nonfinite_candidates(self):
        module = load_basin_hopping_module()
        monitor = module.BasinHoppingMonitor()

        accepted = monitor.accept_test(
            f_new=np.inf,
            x_new=np.array([1.0, 2.0]),
            f_old=1.0,
            x_old=np.array([1.0, 2.0]),
        )

        self.assertFalse(accepted)
        self.assertEqual(monitor.accept_test_rejections, 1)
        self.assertEqual(monitor.nonfinite_rejections, 1)
        self.assertTrue(monitor.accept_test_triggered)

    def test_accept_test_rejects_large_normalized_steps(self):
        module = load_basin_hopping_module()
        monitor = module.BasinHoppingMonitor(normalized_step_rms_limit=0.1)

        accepted = monitor.accept_test(
            f_new=1.0,
            x_new=np.array([3.0, 3.0]),
            f_old=1.5,
            x_old=np.array([1.0, 1.0]),
        )

        self.assertFalse(accepted)
        self.assertEqual(monitor.accept_test_rejections, 1)
        self.assertEqual(monitor.normalized_step_rejections, 1)
        self.assertTrue(monitor.accept_test_triggered)

    def test_callback_skips_initial_minimum_when_counting_hops(self):
        module = load_basin_hopping_module()
        monitor = module.BasinHoppingMonitor()

        self.assertFalse(monitor.callback(np.array([0.0]), 5.0, True))
        self.assertFalse(monitor.callback(np.array([1.0]), 4.0, True))
        self.assertFalse(monitor.callback(np.array([2.0]), 6.0, False))

        self.assertEqual(monitor.accepted_hops, 1)
        self.assertEqual(monitor.rejected_hops, 1)
        self.assertEqual(monitor.completed_hops, 2)
        self.assertEqual(monitor.initial_objective, 5.0)
        self.assertEqual(monitor.best_hop_objective, 4.0)
        self.assertEqual(monitor.best_hop_index, 1)
        self.assertEqual(monitor.best_result_source, "hop")
        self.assertEqual(monitor.best_objective, 4.0)

    def test_run_basin_hopping_passes_rng_accept_test_and_callback(self):
        module = load_basin_hopping_module()
        captured = {}
        minima = []

        def fake_basinhopping(fun, dofs, **kwargs):
            del fun
            captured["dofs"] = np.asarray(dofs)
            captured.update(kwargs)
            self.assertIsInstance(kwargs["rng"], np.random.Generator)
            self.assertFalse(
                kwargs["accept_test"](
                    1.0,
                    np.array([20.0, 20.0]),
                    1.0,
                    np.array([1.0, 1.0]),
                )
            )
            kwargs["callback"](np.array([0.0, 0.0]), 3.0, True)
            kwargs["callback"](np.array([1.0, 1.0]), 2.0, True)
            kwargs["callback"](np.array([2.0, 2.0]), 5.0, False)
            return SimpleNamespace(nit=2, fun=2.0)

        with patch.object(module, "basinhopping", side_effect=fake_basinhopping):
            result, telemetry = module.run_basin_hopping(
                lambda x: (float(np.sum(x * x)), 2.0 * np.asarray(x)),
                np.array([1.0, -1.0]),
                basin_hops=2,
                basin_stepsize=0.25,
                basin_temperature=2.5,
                basin_niter_success=4,
                rng_seed=17,
                minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
                disp=False,
                local_minimum_callback=lambda x, f, accept: minima.append(
                    (x.tolist(), f, accept)
                ),
            )

        self.assertEqual(result.nit, 2)
        np.testing.assert_allclose(captured["dofs"], [1.0, -1.0])
        self.assertEqual(captured["niter"], 2)
        self.assertEqual(captured["stepsize"], 0.25)
        self.assertEqual(captured["T"], 2.5)
        self.assertEqual(captured["niter_success"], 4)
        self.assertEqual(
            captured["minimizer_kwargs"],
            {"method": "L-BFGS-B", "jac": True},
        )
        self.assertFalse(captured["disp"])
        self.assertEqual(
            minima,
            [
                ([0.0, 0.0], 3.0, True),
                ([1.0, 1.0], 2.0, True),
                ([2.0, 2.0], 5.0, False),
            ],
        )
        for key, expected in EXPECTED_BASIN_TELEMETRY.items():
            self.assertEqual(telemetry[key], expected)

    def test_run_basin_hopping_integrates_with_real_scipy_basinhopping(self):
        module = load_basin_hopping_module()

        def fun(x):
            x = np.asarray(x, dtype=float)
            value = float((x[0] ** 2 - 1.0) ** 2 + 0.1 * x[0])
            grad = np.array([4.0 * x[0] * (x[0] ** 2 - 1.0) + 0.1])
            return value, grad

        result, telemetry = module.run_basin_hopping(
            fun,
            np.array([1.5]),
            basin_hops=4,
            basin_stepsize=2.0,
            basin_temperature=0.5,
            basin_niter_success=4,
            rng_seed=0,
            minimizer_kwargs={"method": "L-BFGS-B", "jac": True},
            disp=False,
        )

        self.assertLess(float(result.fun), 0.0)
        self.assertEqual(telemetry["basin_completed_hops"], 4)
        self.assertEqual(
            telemetry["basin_accepted_hops"] + telemetry["basin_rejected_hops"],
            4,
        )
        self.assertEqual(telemetry["basin_best_result_source"], "hop")
        self.assertLess(float(telemetry["basin_best_objective"]), 0.0)
        self.assertGreater(float(telemetry["basin_objective_improvement"]), 0.0)

    def test_telemetry_values_uses_declared_field_order(self):
        module = load_basin_hopping_module()

        self.assertEqual(module.BASIN_TELEMETRY_FIELDS, EXPECTED_BASIN_TELEMETRY_FIELDS)
        self.assertEqual(
            module.telemetry_values(EXPECTED_BASIN_TELEMETRY),
            (1, 1, 2.0, 1, True),
        )
