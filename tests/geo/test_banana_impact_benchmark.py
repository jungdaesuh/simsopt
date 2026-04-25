import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
BANANA_IMPACT_BENCHMARK = EXAMPLES_ROOT / "benchmark_banana_impact.py"
sys.path.insert(0, str(EXAMPLES_ROOT))
import benchmark_banana_impact  # noqa: E402
import benchmark_lbfgsb_maxcor  # noqa: E402
del sys.path[0]


def _json_completed_process(payload):
    return SimpleNamespace(stdout=json.dumps(payload))


def _banana_impact_measure_one_command(fixture, repeat, warmup):
    return [
        sys.executable,
        benchmark_banana_impact.__file__,
        "--measure-one",
        "--fixture",
        fixture,
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]


def _lbfgsb_maxcor_measure_one_command(maxcor, dimension, maxiter, repeat, warmup):
    return [
        sys.executable,
        benchmark_lbfgsb_maxcor.__file__,
        "--measure-one",
        "--maxcor",
        str(maxcor),
        "--dimension",
        str(dimension),
        "--maxiter",
        str(maxiter),
        "--repeat",
        str(repeat),
        "--warmup",
        str(warmup),
    ]


class BananaImpactBenchmarkTests(unittest.TestCase):
    def test_measure_operation_records_timing_and_memory_contract(self):
        calls = {"build": 0, "operation": 0}

        def build():
            calls["build"] += 1

            def operation():
                calls["operation"] += 1
                return float(calls["operation"] + 9)

            return operation

        with patch.object(
            benchmark_banana_impact.gc,
            "collect",
            return_value=0,
        ) as gc_collect, patch.object(
            benchmark_banana_impact.tracemalloc,
            "start",
        ) as tracemalloc_start, patch.object(
            benchmark_banana_impact.tracemalloc,
            "get_traced_memory",
            return_value=(123, 456),
        ) as get_traced_memory, patch.object(
            benchmark_banana_impact.tracemalloc,
            "stop",
        ) as tracemalloc_stop, patch.object(
            benchmark_banana_impact,
            "_maxrss_bytes",
            side_effect=[4096, 8192],
        ) as maxrss_bytes, patch.object(
            benchmark_banana_impact.time,
            "perf_counter",
            side_effect=[10.0, 10.125, 20.0, 20.375],
        ) as perf_counter:
            result = benchmark_banana_impact.measure_operation(
                name="unit-fixture",
                description="Unit-test fixture.",
                build=build,
                repeat=2,
                warmup=1,
            )

        self.assertEqual(calls, {"build": 1, "operation": 3})
        gc_collect.assert_called_once_with()
        tracemalloc_start.assert_called_once_with()
        get_traced_memory.assert_called_once_with()
        tracemalloc_stop.assert_called_once_with()
        self.assertEqual(maxrss_bytes.call_count, 2)
        self.assertEqual(perf_counter.call_count, 4)
        self.assertEqual(
            result,
            {
                "name": "unit-fixture",
                "description": "Unit-test fixture.",
                "repeat": 2,
                "warmup": 1,
                "seconds_min": 0.125,
                "seconds_median": 0.25,
                "seconds_mean": 0.25,
                "python_peak_bytes": 456,
                "process_peak_rss_bytes": 8192,
                "process_maxrss_before_bytes": 4096,
                "process_maxrss_after_bytes": 8192,
                "checksum_first": 11.0,
                "checksum_last": 12.0,
            },
        )

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
        self.assertIn("process_peak_rss_bytes", payload["results"][0])

    def test_default_report_measures_each_fixture_in_subprocess(self):
        subprocess_payloads = [
            {
                "name": "biot-savart",
                "description": "subprocess fixture a",
                "repeat": 2,
                "warmup": 1,
                "seconds_min": 0.01,
                "seconds_median": 0.02,
                "seconds_mean": 0.03,
                "python_peak_bytes": 128,
                "process_peak_rss_bytes": 1024,
                "process_maxrss_before_bytes": 512,
                "process_maxrss_after_bytes": 1024,
                "checksum_first": 1.0,
                "checksum_last": 2.0,
            },
            {
                "name": "magnetic-field-sum",
                "description": "subprocess fixture b",
                "repeat": 2,
                "warmup": 1,
                "seconds_min": 0.04,
                "seconds_median": 0.05,
                "seconds_mean": 0.06,
                "python_peak_bytes": 256,
                "process_peak_rss_bytes": 4096,
                "process_maxrss_before_bytes": 2048,
                "process_maxrss_after_bytes": 4096,
                "checksum_first": 3.0,
                "checksum_last": 4.0,
            },
        ]

        with patch.object(
            benchmark_banana_impact.platform,
            "platform",
            return_value="test-platform",
        ), patch.object(
            benchmark_banana_impact.subprocess,
            "run",
            side_effect=[_json_completed_process(payload) for payload in subprocess_payloads],
        ) as run:
            report = benchmark_banana_impact.build_report(
                ["biot-savart", "magnetic-field-sum"],
                repeat=2,
                warmup=1,
            )

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    _banana_impact_measure_one_command("biot-savart", 2, 1),
                    check=True,
                    capture_output=True,
                    text=True,
                ),
                call(
                    _banana_impact_measure_one_command("magnetic-field-sum", 2, 1),
                    check=True,
                    capture_output=True,
                    text=True,
                ),
            ],
        )
        self.assertEqual(
            [entry["name"] for entry in report["results"]],
            ["biot-savart", "magnetic-field-sum"],
        )
        self.assertEqual(report["results"], subprocess_payloads)

    def test_cli_runs_from_outside_repository(self):
        command = [
            sys.executable,
            str(BANANA_IMPACT_BENCHMARK),
            "--fixture",
            "biot-savart",
            "--repeat",
            "1",
            "--warmup",
            "0",
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                cwd=tmpdir,
                text=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(payload["fixtures"], ["biot-savart"])
        self.assertEqual(payload["results"][0]["name"], "biot-savart")

    def test_markdown_report_renders_fixture_impact_table(self):
        report = {
            "schema_version": benchmark_banana_impact.SCHEMA_VERSION,
            "created_at_utc": "2026-04-25T00:00:00+00:00",
            "repeat": 1,
            "warmup": 0,
            "results": [
                {
                    "name": "unit-fixture",
                    "seconds_median": 0.125,
                    "seconds_mean": 0.125,
                    "python_peak_bytes": 1048576,
                    "process_peak_rss_bytes": 2097152,
                    "checksum_first": 2.0,
                    "checksum_last": 2.0,
                }
            ],
        }

        markdown = benchmark_banana_impact.render_markdown_report(report)

        self.assertIn("# Banana Impact Benchmark", markdown)
        self.assertIn("| Fixture | Median seconds | Mean seconds | Python peak | Process peak RSS |", markdown)
        self.assertIn("| unit-fixture | 0.125 | 0.125 | 1.000 MiB | 2.000 MiB | 2 | 2 |", markdown)

    def test_lbfgsb_maxcor_subprocess_wrapper_uses_exact_command(self):
        payload = {
            "maxcor": 20,
            "dimension": 6,
            "maxiter": 3,
            "repeat": 2,
            "warmup": 1,
            "seconds_min": 0.01,
            "seconds_median": 0.02,
            "seconds_mean": 0.03,
            "python_peak_bytes": 1234,
            "process_peak_rss_bytes": 5678,
            "iterations_median": 3,
            "function_evaluations_median": 4,
            "final_objective_median": 0.5,
            "gradient_inf_norm_median": 0.125,
            "success_count": 1,
        }
        completed = _json_completed_process(payload)

        with patch.object(
            benchmark_lbfgsb_maxcor.subprocess,
            "run",
            return_value=completed,
        ) as run:
            result = benchmark_lbfgsb_maxcor.measure_maxcor(
                maxcor=20,
                dimension=6,
                maxiter=3,
                repeat=2,
                warmup=1,
            )

        run.assert_called_once_with(
            _lbfgsb_maxcor_measure_one_command(20, 6, 3, 2, 1),
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result, payload)

    def test_lbfgsb_fixture_objective_gradient_matches_central_difference(self):
        scale, target = benchmark_lbfgsb_maxcor._fixture_parameters(6)
        objective = benchmark_lbfgsb_maxcor._fixture_objective(scale, target)
        x = np.array([-0.2, -0.05, 0.1, 0.2, 0.35, 0.5], dtype=float)
        eps = 1.0e-6

        _, grad = objective(x)
        finite_difference_grad = np.empty_like(x)
        for index in range(x.size):
            step = np.zeros_like(x)
            step[index] = eps
            value_plus, _ = objective(x + step)
            value_minus, _ = objective(x - step)
            finite_difference_grad[index] = (value_plus - value_minus) / (2.0 * eps)

        np.testing.assert_allclose(
            grad,
            finite_difference_grad,
            atol=1.0e-6,
            rtol=1.0e-6,
        )

    def test_lbfgsb_maxcor_in_process_records_payload_contract(self):
        result = benchmark_lbfgsb_maxcor._measure_maxcor_in_process(
            maxcor=20,
            dimension=6,
            maxiter=3,
            repeat=1,
            warmup=0,
        )

        self.assertEqual(result["maxcor"], 20)
        self.assertEqual(result["dimension"], 6)
        self.assertEqual(result["maxiter"], 3)
        self.assertEqual(result["repeat"], 1)
        self.assertEqual(result["warmup"], 0)
        self.assertGreaterEqual(result["seconds_min"], 0.0)
        self.assertGreaterEqual(result["seconds_median"], result["seconds_min"])
        self.assertGreaterEqual(result["seconds_mean"], result["seconds_min"])
        self.assertGreaterEqual(result["python_peak_bytes"], 0)
        self.assertGreaterEqual(result["process_peak_rss_bytes"], 0)
        self.assertGreaterEqual(result["iterations_median"], 0)
        self.assertGreaterEqual(result["function_evaluations_median"], 1)
        self.assertGreaterEqual(result["final_objective_median"], 0.0)
        self.assertGreaterEqual(result["gradient_inf_norm_median"], 0.0)
        self.assertIn(result["success_count"], (0, 1))

    def test_lbfgsb_maxcor_report_declares_default_and_requested_values(self):
        measured_entries = [
            {"maxcor": 20, "dimension": 8},
            {"maxcor": 40, "dimension": 8},
        ]

        with patch.object(
            benchmark_lbfgsb_maxcor,
            "measure_maxcor",
            side_effect=measured_entries,
        ) as measure_maxcor:
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
        self.assertEqual(report["results"], measured_entries)
        self.assertEqual(
            measure_maxcor.call_args_list,
            [
                call(maxcor=20, dimension=8, maxiter=2, repeat=1, warmup=0),
                call(maxcor=40, dimension=8, maxiter=2, repeat=1, warmup=0),
            ],
        )


if __name__ == "__main__":
    unittest.main()
