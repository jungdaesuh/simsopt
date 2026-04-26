import csv
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXAMPLES_ROOT = Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
sys.path.insert(0, str(EXAMPLES_ROOT))
from alm_utils import ALM_SCHEMA_VERSION  # noqa: E402
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

    def test_baseline_row_detects_nested_normalized_alm_fields(self):
        row = alm_benchmarking.baseline_row(
            source_kind="run_artifact",
            source_path=Path("/tmp/results.json"),
            payload={
                "params": {"constraint_method": "alm"},
                "results_payload": {
                    "ALM_SCHEMA_VERSION": ALM_SCHEMA_VERSION,
                    "ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES": [0.0],
                },
            },
            relevance_reason="params.constraint_method",
        )

        self.assertTrue(row["normalized_fields_available"])
        self.assertEqual(row["normalization_contract_version"], ALM_SCHEMA_VERSION)

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
                + "\n"
                + '{"run_id": "bad", "params": '
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
                        "ALM_SCHEMA_VERSION": ALM_SCHEMA_VERSION,
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
        self.assertEqual(summary["counts"]["skipped_artifacts"], 2)
        self.assertEqual(summary["counts"]["baseline_rows"], 3)
        self.assertEqual(
            {entry["skip_reason"] for entry in summary["skipped_artifacts"]},
            {"invalid_json", "invalid_jsonl"},
        )
        artifact_row = next(
            row for row in summary["baseline_rows"] if row["run_id"] == "artifact-run"
        )
        self.assertTrue(artifact_row["normalized_fields_available"])
        self.assertEqual(
            artifact_row["normalization_contract_version"],
            ALM_SCHEMA_VERSION,
        )

    def test_autoresearch_root_requires_explicit_arg_or_env(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "--autoresearch-root"):
                alm_benchmarking.autoresearch_root_from_arg(None)

        with mock.patch.dict(
            os.environ,
            {alm_benchmarking.AUTORESEARCH_ROOT_ENV: "/tmp/autoresearch"},
            clear=True,
        ):
            self.assertEqual(
                alm_benchmarking.autoresearch_root_from_arg(None),
                Path("/tmp/autoresearch"),
            )

        self.assertEqual(
            alm_benchmarking.autoresearch_root_from_arg(Path("/tmp/explicit")),
            Path("/tmp/explicit"),
        )

    def test_registry_rows_raise_on_missing_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_db = Path(tmpdir) / "registry.db"

            with self.assertRaises(FileNotFoundError):
                alm_benchmarking._registry_rows(registry_db)

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
                    "ledger_row_index": 12,
                    "registry_database": None,
                    "created_at": "2026-04-26T00:00:00+00:00",
                    "normalization_contract_version": "pre_normalization",
                    "relevance_reason": "params.alm",
                    "normalized_fields_available": False,
                    "solver_commit": "abc",
                    "artifact_export_path": None,
                    "seed_artifact_path": "/tmp/seed",
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
        self.assertEqual(comparison_rows[0]["case"], "ledger:/tmp/results.jsonl:12")
        self.assertEqual(comparison_rows[0]["before_run_id"], "case-a")
        self.assertEqual(comparison_rows[0]["before_ledger_row_index"], "12")
        self.assertEqual(comparison_rows[0]["before_solver_commit"], "abc")
        self.assertEqual(
            comparison_rows[0]["before_normalization_contract_version"],
            "pre_normalization",
        )
        self.assertEqual(comparison_rows[0]["before_seed_artifact_path"], "/tmp/seed")
        self.assertEqual(comparison_rows[0]["after_run_id"], "")
        self.assertEqual(comparison_rows[0]["before_success"], "True")
        self.assertEqual(comparison_rows[0]["after_success"], "")
        self.assertIn("fixture_manifest", paths)
        self.assertIn("skipped_artifacts", paths)

    def test_comparison_rows_join_ledger_rows_by_row_index(self):
        source_path = "/tmp/results.jsonl"
        before_rows = [
            {
                "source_kind": "ledger",
                "source_path": source_path,
                "ledger_row_index": 3,
                "hard_feasible_success": False,
                "max_raw_hard_violation": 0.001,
                "max_normalized_violation": None,
            },
            {
                "source_kind": "ledger",
                "source_path": source_path,
                "ledger_row_index": 4,
                "hard_feasible_success": True,
                "max_raw_hard_violation": 0.0,
                "max_normalized_violation": None,
            },
        ]
        after_rows = [
            {
                "source_kind": "ledger",
                "source_path": source_path,
                "ledger_row_index": 4,
                "hard_feasible_success": True,
                "max_raw_hard_violation": 0.0,
                "max_normalized_violation": 0.0,
            }
        ]

        rows = alm_benchmarking.comparison_rows(before_rows, after_rows)

        self.assertEqual(rows[0]["case"], f"ledger:{source_path}:3")
        self.assertEqual(rows[0]["after_success"], None)
        self.assertEqual(rows[1]["case"], f"ledger:{source_path}:4")
        self.assertEqual(rows[1]["after_success"], True)
        self.assertEqual(rows[1]["after_max_normalized_violation"], 0.0)

    def test_comparison_rows_join_run_artifacts_by_source_path(self):
        before_rows = [
            {
                "source_kind": "run_artifact",
                "source_path": "/tmp/run-a/results.json",
                "run_id": "shared-run",
                "hard_feasible_success": False,
            },
            {
                "source_kind": "run_artifact",
                "source_path": "/tmp/run-b/results.json",
                "run_id": "shared-run",
                "hard_feasible_success": True,
            },
        ]
        after_rows = [
            {
                "source_kind": "run_artifact",
                "source_path": "/tmp/run-b/results.json",
                "run_id": "shared-run",
                "hard_feasible_success": True,
                "max_normalized_violation": 0.0,
            }
        ]

        rows = alm_benchmarking.comparison_rows(before_rows, after_rows)

        self.assertEqual(rows[0]["case"], "run_artifact:/tmp/run-a/results.json")
        self.assertEqual(rows[0]["after_success"], None)
        self.assertEqual(rows[1]["case"], "run_artifact:/tmp/run-b/results.json")
        self.assertEqual(rows[1]["after_success"], True)
        self.assertEqual(rows[1]["after_max_normalized_violation"], 0.0)

    def test_write_after_outputs_writes_after_rows_and_joined_comparison(self):
        baseline_row = {
            "source_kind": "registry",
            "source_path": "/tmp/registry.db",
            "run_id": "case-a",
            "ledger_row_index": None,
            "registry_database": "/tmp/registry.db",
            "solver_commit": "before-sha",
            "created_at": "2026-04-26T00:00:00+00:00",
            "normalization_contract_version": "pre_normalization",
            "normalized_fields_available": False,
            "artifact_export_path": "/tmp/export-before",
            "seed_artifact_path": "/tmp/seed-before",
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
        after_row = {
            **baseline_row,
            "solver_commit": "after-sha",
            "created_at": "2026-04-26T01:00:00+00:00",
            "normalized_fields_available": True,
            "normalization_contract_version": ALM_SCHEMA_VERSION,
            "artifact_export_path": "/tmp/export-after",
            "seed_artifact_path": "/tmp/seed-after",
            "max_normalized_violation": 0.0,
            "wall_time_s": 3.5,
        }
        after_summary = {
            "schema_version": alm_benchmarking.SCHEMA_VERSION,
            "created_at_utc": "2026-04-26T00:00:00+00:00",
            "counts": {
                "registry_rows": 1,
                "ledger_rows": 0,
                "run_artifact_rows": 0,
                "harvested_seed_fixtures": 0,
                "skipped_artifacts": 0,
                "baseline_rows": 1,
            },
            "baseline_rows": [after_row],
            "skipped_artifacts": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            baseline_path = Path(tmpdir) / "baseline.jsonl"
            baseline_path.write_text(json.dumps(baseline_row) + "\n", encoding="utf-8")
            paths = alm_benchmarking.write_after_outputs(
                after_summary,
                Path(tmpdir),
                stamp="20260426T010000Z",
                baseline_path=baseline_path,
                baseline_rows=alm_benchmarking.read_benchmark_rows(baseline_path),
            )
            after_lines = paths["after"].read_text(encoding="utf-8").splitlines()
            with paths["comparison_csv"].open(encoding="utf-8", newline="") as handle:
                comparison_rows = list(csv.DictReader(handle))
            comparison_markdown = paths["comparison_markdown"].read_text(encoding="utf-8")

        self.assertEqual(len(after_lines), 1)
        self.assertEqual(json.loads(after_lines[0])["run_id"], "case-a")
        self.assertEqual(comparison_rows[0]["case"], "registry:case-a")
        self.assertEqual(comparison_rows[0]["before_solver_commit"], "before-sha")
        self.assertEqual(comparison_rows[0]["after_solver_commit"], "after-sha")
        self.assertEqual(
            comparison_rows[0]["after_normalization_contract_version"],
            ALM_SCHEMA_VERSION,
        )
        self.assertEqual(
            comparison_rows[0]["after_artifact_export_path"],
            "/tmp/export-after",
        )
        self.assertEqual(comparison_rows[0]["after_seed_artifact_path"], "/tmp/seed-after")
        self.assertEqual(comparison_rows[0]["after_success"], "True")
        self.assertEqual(comparison_rows[0]["after_max_normalized_violation"], "0.0")
        self.assertIn("does not execute solver fixtures", comparison_markdown)
        self.assertIn("solver_commit=before-sha", comparison_markdown)
        self.assertIn(f"normalization_contract_version={ALM_SCHEMA_VERSION}", comparison_markdown)


if __name__ == "__main__":
    unittest.main()
