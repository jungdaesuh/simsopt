import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
sys.path.insert(0, str(EXAMPLES_ROOT))
import benchmark_banana_impact  # noqa: E402
import benchmark_lbfgsb_maxcor  # noqa: E402
del sys.path[0]


class BananaImpactBenchmarkTests(unittest.TestCase):
    def test_measure_operation_records_timing_and_memory_contract(self):
        result = benchmark_banana_impact.measure_operation(
            name="unit-fixture",
            description="Unit-test fixture.",
            build=lambda: lambda: 3.5,
            repeat=2,
            warmup=1,
        )

        self.assertEqual(result["name"], "unit-fixture")
        self.assertEqual(result["repeat"], 2)
        self.assertEqual(result["warmup"], 1)
        self.assertGreaterEqual(result["seconds_min"], 0.0)
        self.assertGreaterEqual(result["seconds_median"], result["seconds_min"])
        self.assertGreaterEqual(result["python_peak_bytes"], 0)
        self.assertGreaterEqual(result["process_maxrss_after_bytes"], result["process_maxrss_before_bytes"])
        self.assertEqual(result["checksum_first"], 3.5)
        self.assertEqual(result["checksum_last"], 3.5)

    def test_report_is_json_serializable_and_declares_selected_fixtures(self):
        fixtures = {
            "unit-fixture": benchmark_banana_impact.BenchmarkFixture(
                name="unit-fixture",
                description="Unit-test fixture.",
                build=lambda: lambda: 2.0,
            )
        }
        report = benchmark_banana_impact.build_report(
            ["unit-fixture"],
            repeat=1,
            warmup=0,
            fixtures=fixtures,
        )

        payload = json.loads(json.dumps(report))
        self.assertEqual(payload["schema_version"], benchmark_banana_impact.SCHEMA_VERSION)
        self.assertEqual(payload["fixtures"], ["unit-fixture"])
        self.assertEqual(payload["results"][0]["name"], "unit-fixture")

    def test_lbfgsb_maxcor_benchmark_records_progress_and_memory_contract(self):
        result = benchmark_lbfgsb_maxcor.measure_maxcor(
            maxcor=20,
            dimension=8,
            maxiter=3,
            repeat=1,
            warmup=0,
        )

        self.assertEqual(result["maxcor"], 20)
        self.assertEqual(result["dimension"], 8)
        self.assertGreaterEqual(result["seconds_min"], 0.0)
        self.assertGreaterEqual(result["python_peak_bytes"], 0)
        self.assertGreaterEqual(result["process_peak_rss_bytes"], 0)
        self.assertGreaterEqual(result["iterations_median"], 0)
        self.assertGreaterEqual(result["function_evaluations_median"], 1)
        self.assertGreaterEqual(result["final_objective_median"], 0.0)
        self.assertGreaterEqual(result["gradient_inf_norm_median"], 0.0)

    def test_lbfgsb_maxcor_report_declares_default_and_requested_values(self):
        report = benchmark_lbfgsb_maxcor.build_report(
            [20, 40],
            dimension=8,
            maxiter=2,
            repeat=1,
            warmup=0,
        )

        self.assertEqual(report["schema_version"], benchmark_lbfgsb_maxcor.SCHEMA_VERSION)
        self.assertEqual(report["default_maxcor"], 40)
        self.assertEqual([entry["maxcor"] for entry in report["results"]], [20, 40])


if __name__ == "__main__":
    unittest.main()
