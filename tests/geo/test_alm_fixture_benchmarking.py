import json
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
sys.path.insert(0, str(EXAMPLES_ROOT))
from alm_utils import ALMSettings  # noqa: E402
from banana_opt import alm_fixture_benchmarking  # noqa: E402
del sys.path[0]


class AlmFixtureBenchmarkingTests(unittest.TestCase):
    def test_fixture_raw_and_normalized_constraints_share_feasible_set(self):
        fixture = alm_fixture_benchmarking.default_fixtures()[0]
        x = fixture.upper_bounds

        self.assertEqual(
            alm_fixture_benchmarking._max_positive(
                fixture.raw_signed_values(x),
            ),
            0.0,
        )
        self.assertEqual(
            alm_fixture_benchmarking._max_positive(
                fixture.normalized_signed_values(x),
            ),
            0.0,
        )

    def test_run_fixture_benchmark_emits_raw_and_normalized_rows(self):
        fixture = alm_fixture_benchmarking.default_fixtures()[0]
        payload = alm_fixture_benchmarking.run_fixture_benchmark(
            fixtures=(fixture,),
            settings=ALMSettings(
                max_outer_iterations=3,
                penalty_init=1.0,
                penalty_scale=10.0,
                penalty_max=1.0e6,
                feasibility_tol=1.0e-7,
                stationarity_tol=1.0e-7,
                multiplier_max=1.0e6,
            ),
            inner_options={
                "maxiter": 40,
                "ftol": 1.0e-12,
                "gtol": 1.0e-9,
                "maxls": 30,
            },
        )

        self.assertEqual(
            payload["schema_version"],
            alm_fixture_benchmarking.SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["settings"]["seed"],
            alm_fixture_benchmarking.DEFAULT_FIXTURE_SEED,
        )
        self.assertEqual(
            payload["solver_checkout"],
            str(Path(__file__).resolve().parents[2]),
        )
        self.assertEqual(len(payload["solver_commit"]), 40)
        int(payload["solver_commit"], 16)
        self.assertEqual(len(payload["fixture_rows"]), 2)
        self.assertEqual(len(payload["comparisons"]), 1)
        formulations = {row["formulation"] for row in payload["fixture_rows"]}
        self.assertEqual(
            formulations,
            {
                alm_fixture_benchmarking.FORMULATION_RAW,
                alm_fixture_benchmarking.FORMULATION_NORMALIZED,
            },
        )
        comparison = payload["comparisons"][0]
        self.assertEqual(comparison["fixture"], fixture.name)
        self.assertIn("eval_count_delta_raw_minus_normalized", comparison)

    def test_run_fixture_benchmark_preserves_explicit_empty_inner_options(self):
        payload = alm_fixture_benchmarking.run_fixture_benchmark(
            fixtures=(),
            inner_options={},
            seed=123,
        )

        self.assertEqual(payload["inner_options"], {})
        self.assertEqual(payload["settings"]["seed"], 123)

    def test_cli_does_not_default_to_user_specific_autoresearch_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fixture.json"
            args = alm_fixture_benchmarking._parse_args(["--output", str(output_path)])

        self.assertEqual(args.output, output_path)
        self.assertIsNone(args.autoresearch_root)

    def test_evaluate_fixture_rejects_unknown_formulation(self):
        fixture = alm_fixture_benchmarking.default_fixtures()[0]

        with self.assertRaisesRegex(ValueError, "unknown ALM fixture formulation"):
            alm_fixture_benchmarking._evaluate_fixture(
                fixture,
                fixture.x0,
                multipliers=[0.0, 0.0],
                penalty=1.0,
                formulation="typo",
            )

    def test_write_fixture_benchmark_writes_json_payload(self):
        payload = {
            "schema_version": alm_fixture_benchmarking.SCHEMA_VERSION,
            "fixture_rows": [],
            "comparisons": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "fixture_benchmark.json"
            written_path = alm_fixture_benchmarking.write_fixture_benchmark(
                payload,
                output_path,
            )
            loaded = json.loads(written_path.read_text(encoding="utf-8"))

        self.assertEqual(written_path, output_path)
        self.assertEqual(
            loaded["schema_version"],
            alm_fixture_benchmarking.SCHEMA_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
