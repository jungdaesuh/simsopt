import importlib.util
import json
import tempfile
import unittest
import uuid
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "stage2_seed_report.py"
)


def load_stage2_seed_report_module():
    spec = importlib.util.spec_from_file_location(
        f"stage2_seed_report_{uuid.uuid4().hex}",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_run(
    root: Path,
    name: str,
    *,
    field_error: float,
    objective: float,
    final_cc_distance: float,
    max_curvature: float,
    final_curve_length: float = 0.9,
    optimizer_success: bool = True,
    hardware_ok: bool = True,
    self_intersecting: bool = False,
    write_restart_artifacts: bool = True,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    payload = {
        "FIELD_ERROR": field_error,
        "FINAL_OBJECTIVE": objective,
        "FINAL_CC_DISTANCE": final_cc_distance,
        "MAX_CURVATURE": max_curvature,
        "FINAL_CURVE_LENGTH": final_curve_length,
        "LENGTH_TARGET": 1.0,
        "CC_THRESHOLD": 0.05,
        "CURVATURE_THRESHOLD": 20.0,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "TERMINATION_MESSAGE": (
            "success" if optimizer_success else "maximum iterations reached"
        ),
        "HARDWARE_CONSTRAINTS_OK": hardware_ok,
        "HARDWARE_CONSTRAINT_VIOLATIONS": [] if hardware_ok else ["coil too close"],
        "SELF_INTERSECTING": self_intersecting,
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    if write_restart_artifacts:
        (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
        (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
    return run_dir


class Stage2SeedReportTests(unittest.TestCase):
    def test_evaluate_candidate_reports_research_grade_for_complete_clean_seed(self):
        module = load_stage2_seed_report_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_run(
                Path(tmpdir),
                "run-a",
                field_error=0.01,
                objective=1e-3,
                final_cc_distance=0.09,
                max_curvature=8.0,
            )
            report = module.evaluate_stage2_seed_candidate(run_dir)

        self.assertEqual(report["status"], "research_grade")
        self.assertTrue(report["downstream_eligible"])
        self.assertTrue(report["research_grade"])
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["warnings"], [])
        self.assertGreater(report["margins"]["coil_coil_margin"], 0.0)

    def test_evaluate_candidate_marks_results_only_seed_as_salvageable(self):
        module = load_stage2_seed_report_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _write_run(
                Path(tmpdir),
                "run-b",
                field_error=0.02,
                objective=2e-3,
                final_cc_distance=0.08,
                max_curvature=9.0,
                optimizer_success=False,
                write_restart_artifacts=False,
            )
            (run_dir / "surf_opt.vts").write_text("legacy", encoding="utf-8")
            report = module.evaluate_stage2_seed_candidate(run_dir)

        self.assertEqual(report["status"], "salvageable")
        self.assertFalse(report["downstream_eligible"])
        self.assertIn("restart artifacts are incomplete", report["failures"])
        self.assertTrue(
            any("legacy surf_opt.vts exists" in warning for warning in report["warnings"])
        )

    def test_build_catalog_sorts_best_candidate_first(self):
        module = load_stage2_seed_report_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_run(
                root,
                "worse",
                field_error=0.03,
                objective=3e-3,
                final_cc_distance=0.07,
                max_curvature=10.0,
            )
            best_run = _write_run(
                root,
                "better",
                field_error=0.01,
                objective=1e-3,
                final_cc_distance=0.09,
                max_curvature=8.0,
            )
            catalog = module.build_stage2_seed_catalog(root)

        self.assertEqual(catalog["candidate_count"], 2)
        self.assertEqual(catalog["eligible_count"], 2)
        self.assertEqual(catalog["research_grade_count"], 2)
        self.assertEqual(catalog["best_candidate"]["run_dir"], str(best_run))
        self.assertTrue(catalog["passed"])

    def test_evaluate_candidate_rejects_unreadable_results_without_crashing(self):
        module = load_stage2_seed_report_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run-corrupt"
            run_dir.mkdir(parents=True)
            (run_dir / "results.json").write_text(
                '{"FIELD_ERROR": 0.01, "FINAL_OBJECTIVE": }',
                encoding="utf-8",
            )
            (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")

            report = module.evaluate_stage2_seed_candidate(run_dir)

        self.assertEqual(report["status"], "rejected")
        self.assertFalse(report["downstream_eligible"])
        self.assertTrue(
            any(
                "results.json is unreadable: JSONDecodeError"
                in failure
                for failure in report["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
