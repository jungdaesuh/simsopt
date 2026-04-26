import csv
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
sys.path.insert(0, str(EXAMPLES_ROOT))
from banana_opt import alm_benchmarking  # noqa: E402
del sys.path[0]


def _make_roots(tmpdir):
    root = Path(tmpdir)
    roots = alm_benchmarking.AutoresearchArtifactRoots.from_autoresearch_root(root)
    roots.registry_dir.mkdir(parents=True)
    roots.runs_dir.mkdir()
    roots.artifact_exports_dir.mkdir()
    roots.harvested_seeds_dir.mkdir()
    return roots


def _write_registry(registry_db):
    connection = sqlite3.connect(registry_db)
    connection.executescript(
        """
        CREATE TABLE runs (
            run_id TEXT,
            created_at TEXT,
            runtime_s REAL,
            solver TEXT,
            method TEXT,
            iterations INTEGER,
            objective_J REAL,
            optimizer_converged INTEGER,
            validation_status TEXT,
            solver_commit TEXT,
            artifact_dir TEXT,
            single_stage_artifact_dir TEXT,
            results_json_path TEXT
        );
        CREATE TABLE params (
            run_id TEXT,
            constraint_method TEXT,
            alm_enabled INTEGER,
            alm_outer_iters INTEGER,
            alm_formulation TEXT,
            alm_penalty_init REAL,
            alm_penalty_scale REAL,
            alm_feas_tol REAL,
            alm_stationarity_tol REAL,
            alm_qs_threshold REAL,
            alm_boozer_threshold REAL,
            alm_iota_penalty_threshold REAL,
            alm_length_penalty_threshold REAL,
            alm_penalty_max REAL
        );
        CREATE TABLE metrics (
            run_id TEXT,
            field_error REAL,
            qs_error REAL,
            boozer_residual REAL,
            iota_actual REAL
        );
        CREATE TABLE seeds (
            run_id TEXT,
            seed_path TEXT
        );
        """
    )
    connection.execute(
        """
        INSERT INTO runs (
            run_id,
            created_at,
            runtime_s,
            solver,
            method,
            iterations,
            objective_J,
            optimizer_converged,
            validation_status,
            solver_commit,
            artifact_dir,
            single_stage_artifact_dir,
            results_json_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "registry-run",
            "2026-04-26T00:00:00+00:00",
            12.5,
            "single_stage",
            "alm",
            7,
            0.125,
            1,
            "pass",
            "abc123",
            "/tmp/artifact",
            None,
            "/tmp/artifact/results.json",
        ),
    )
    connection.execute(
        """
        INSERT INTO params (
            run_id,
            constraint_method,
            alm_enabled,
            alm_outer_iters,
            alm_formulation,
            alm_penalty_init,
            alm_penalty_scale,
            alm_feas_tol,
            alm_stationarity_tol,
            alm_qs_threshold,
            alm_boozer_threshold,
            alm_iota_penalty_threshold,
            alm_length_penalty_threshold,
            alm_penalty_max
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "registry-run",
            "alm",
            1,
            3,
            "weighted_sum",
            1.0,
            2.0,
            1.0e-4,
            1.0e-5,
            None,
            None,
            None,
            None,
            1.0e6,
        ),
    )
    connection.execute(
        "INSERT INTO metrics VALUES (?, ?, ?, ?, ?)",
        ("registry-run", 0.25, 0.01, 0.02, 0.3),
    )
    connection.execute(
        "INSERT INTO seeds VALUES (?, ?)",
        ("registry-run", "/tmp/seed"),
    )
    connection.commit()
    connection.close()


class AlmBenchmarkingTests(unittest.TestCase):
    def test_alm_relevance_prefers_explicit_constraint_method(self):
        payload = {"params": {"constraint_method": "alm", "alm_enabled": False}}

        self.assertEqual(
            alm_benchmarking.alm_relevance_reason(payload),
            "params.constraint_method",
        )

    def test_build_baseline_summary_reads_registry_ledger_artifacts_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            roots = _make_roots(tmpdir)
            _write_registry(roots.registry_dir / "registry.db")
            roots.ledger_path.write_text(
                json.dumps(
                    {
                        "run_id": "ledger-run",
                        "solver_commit": "def456",
                        "params": {"alm": True},
                        "elapsed": 3.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_dir = roots.runs_dir / "run-artifact"
            run_dir.mkdir()
            (run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "run_id": "artifact-run",
                        "ALM_FINAL_MAX_NORMALIZED_VIOLATION": 0.0625,
                        "ALM_SCHEMA_VERSION": "alm_normalized_v1",
                    }
                ),
                encoding="utf-8",
            )
            invalid_run_dir = roots.runs_dir / "invalid-artifact"
            invalid_run_dir.mkdir()
            (invalid_run_dir / "results.json").write_text(
                '{"run_id": "invalid", "params": ',
                encoding="utf-8",
            )
            seed_dir = roots.harvested_seeds_dir / "seed-artifact"
            seed_dir.mkdir()
            (seed_dir / "results.json").write_text(
                json.dumps({"run_id": "seed-run", "params": {"alm_enabled": True}}),
                encoding="utf-8",
            )

            summary = alm_benchmarking.build_baseline_summary(
                roots,
                created_at_utc="2026-04-26T00:00:00+00:00",
            )

        self.assertEqual(summary["schema_version"], alm_benchmarking.SCHEMA_VERSION)
        self.assertEqual(summary["counts"]["registry_rows"], 1)
        self.assertEqual(summary["counts"]["ledger_rows"], 1)
        self.assertEqual(summary["counts"]["run_artifact_rows"], 1)
        self.assertEqual(summary["counts"]["harvested_seed_fixtures"], 1)
        self.assertEqual(summary["counts"]["skipped_artifacts"], 1)
        self.assertEqual(summary["counts"]["baseline_rows"], 3)
        self.assertEqual(summary["skipped_artifacts"][0]["skip_reason"], "invalid_json")
        artifact_row = next(
            row for row in summary["baseline_rows"] if row["run_id"] == "artifact-run"
        )
        self.assertTrue(artifact_row["normalized_fields_available"])
        self.assertEqual(
            artifact_row["normalization_contract_version"],
            "alm_normalized_v1",
        )

    def test_write_baseline_outputs_preserves_comparison_schema(self):
        summary = {
            "schema_version": alm_benchmarking.SCHEMA_VERSION,
            "created_at_utc": "2026-04-26T00:00:00+00:00",
            "counts": {
                "registry_rows": 0,
                "ledger_rows": 1,
                "run_artifact_rows": 0,
                "harvested_seed_fixtures": 0,
                "skipped_artifacts": 0,
                "baseline_rows": 1,
            },
            "baseline_rows": [
                {
                    "source_kind": "ledger",
                    "source_path": "/tmp/results.jsonl",
                    "run_id": "case-a",
                    "relevance_reason": "params.alm",
                    "normalized_fields_available": False,
                    "solver_commit": "abc",
                    "hard_feasible_success": True,
                    "best_feasible_base_objective": 0.25,
                    "max_raw_hard_violation": 0.0,
                    "max_normalized_violation": None,
                    "outer_iterations": 2,
                    "objective_eval_count": 10,
                    "wall_time_s": 4.0,
                    "penalty_cap_hit_count": 0,
                    "multiplier_cap_hit_count": 0,
                    "blocking_constraint_name": None,
                }
            ],
            "fixture_manifest": [],
            "skipped_artifacts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = alm_benchmarking.write_baseline_outputs(
                summary,
                Path(tmpdir),
                stamp="20260426T000000Z",
            )
            baseline_lines = paths["baseline"].read_text(encoding="utf-8").splitlines()
            with paths["comparison_csv"].open(encoding="utf-8", newline="") as handle:
                comparison_rows = list(csv.DictReader(handle))

        self.assertEqual(len(baseline_lines), 1)
        self.assertEqual(json.loads(baseline_lines[0])["run_id"], "case-a")
        self.assertEqual(comparison_rows[0]["case"], "case-a")
        self.assertEqual(comparison_rows[0]["before_success"], "True")
        self.assertEqual(comparison_rows[0]["after_success"], "")
        self.assertIn("fixture_manifest", paths)
        self.assertIn("skipped_artifacts", paths)


if __name__ == "__main__":
    unittest.main()
