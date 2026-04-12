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
    / "candidate_ledger.py"
)


def load_candidate_ledger_module():
    spec = importlib.util.spec_from_file_location(
        f"candidate_ledger_{uuid.uuid4().hex}",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_stage2_run(root: Path, name: str, *, field_error: float, objective: float):
    run_dir = root / name
    run_dir.mkdir(parents=True)
    payload = {
        "FIELD_ERROR": field_error,
        "FINAL_OBJECTIVE": objective,
        "FINAL_CC_DISTANCE": 0.09,
        "MAX_CURVATURE": 8.0,
        "FINAL_CURVE_LENGTH": 0.9,
        "LENGTH_TARGET": 1.0,
        "CC_THRESHOLD": 0.05,
        "CURVATURE_THRESHOLD": 20.0,
        "OPTIMIZER_SUCCESS": True,
        "TERMINATION_MESSAGE": "success",
        "HARDWARE_CONSTRAINTS_OK": True,
        "HARDWARE_CONSTRAINT_VIOLATIONS": [],
        "SELF_INTERSECTING": False,
    }
    (run_dir / "results.json").write_text(json.dumps(payload), encoding="utf-8")
    (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
    (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
    return run_dir


def _write_continuation_validation(
    root: Path,
    name: str,
    *,
    passed: bool,
    field_error: float,
    abs_iota_error: float,
    final_non_qs: float,
    research_grade_ready: bool,
):
    run_root = root / name
    run_root.mkdir(parents=True)
    payload = {
        "passed": passed,
        "research_verdicts": {
            "full_convergence": passed,
            "hardware_feasible_final_coils": passed,
            "acceptable_final_field_error": passed,
            "acceptable_iota_target": passed,
            "acceptable_non_qs_behavior": passed,
            "physics_gate_pass": passed,
            "research_grade_ready": research_grade_ready,
        },
        "final_stage": {
            "abs_iota_error": abs_iota_error,
            "metrics": {
                "FIELD_ERROR": field_error,
                "FINAL_IOTA": 0.21,
                "FINAL_NON_QS": final_non_qs,
                "FINAL_BOOZER_RESIDUAL": 1e-3,
                "MAX_CURVATURE": 12.0,
                "CURVE_CURVE_MIN_DIST": 0.08,
                "CURVE_SURFACE_MIN_DIST": 0.07,
                "SURFACE_VESSEL_MIN_DIST": 0.06,
                "CC_DIST": 0.05,
                "CS_DIST": 0.04,
                "SS_DIST": 0.03,
                "CURVATURE_THRESHOLD": 20.0,
            },
        },
        "profiling": {
            "total_accepted_step_count": 3,
            "total_objective_eval_count": 12,
            "total_gradient_eval_count": 12,
            "objective_evals_per_accepted_step": 4.0,
        },
        "failures": [] if passed else ["failed"],
        "warnings": [],
    }
    (run_root / "continuation_validation.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return run_root


class CandidateLedgerTests(unittest.TestCase):
    def test_build_candidate_ledger_ranks_stage2_and_single_stage_candidates(self):
        module = load_candidate_ledger_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_root = root / "stage2"
            single_stage_root = root / "single-stage"
            best_stage2 = _write_stage2_run(
                stage2_root,
                "best-seed",
                field_error=0.01,
                objective=1e-3,
            )
            _write_stage2_run(
                stage2_root,
                "worse-seed",
                field_error=0.03,
                objective=3e-3,
            )
            best_single = _write_continuation_validation(
                single_stage_root,
                "best-run",
                passed=True,
                field_error=0.001,
                abs_iota_error=0.002,
                final_non_qs=0.01,
                research_grade_ready=True,
            )
            _write_continuation_validation(
                single_stage_root,
                "worse-run",
                passed=True,
                field_error=0.01,
                abs_iota_error=0.01,
                final_non_qs=0.05,
                research_grade_ready=False,
            )

            ledger = module.build_candidate_ledger(
                stage2_root=stage2_root,
                single_stage_root=single_stage_root,
                stage2_max_field_error=None,
                single_stage_max_final_field_error=None,
                single_stage_max_final_abs_iota_error=None,
                single_stage_max_final_non_qs=None,
            )

        self.assertEqual(
            ledger["stage2"]["best_candidate"]["run_dir"],
            str(best_stage2),
        )
        self.assertEqual(
            ledger["single_stage"]["best_candidate"]["run_root"],
            str(best_single),
        )
        self.assertEqual(
            ledger["single_stage"]["best_candidate"]["status"],
            "research_grade",
        )
        self.assertTrue(ledger["single_stage"]["best_candidate"]["research_usable"])
        self.assertEqual(ledger["single_stage"]["research_usable_count"], 2)
        self.assertTrue(ledger["single_stage"]["best_candidate_reason"]["reasons"])
        self.assertEqual(
            ledger["cross_workflow_summary"]["best_stage2_seed_reason"]["run_dir"],
            str(best_stage2),
        )

    def test_build_candidate_ledger_can_rebuild_validation_from_summary(self):
        module = load_candidate_ledger_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_root = root / "stage2"
            single_stage_root = root / "single-stage"
            _write_stage2_run(stage2_root, "seed", field_error=0.01, objective=1e-3)

            run_root = single_stage_root / "summary-only"
            run_dir = run_root / "stage-04-final" / "run-a"
            run_dir.mkdir(parents=True)
            (run_dir / "results.json").write_text("{}", encoding="utf-8")
            (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
            summary = {
                "stages": [
                    {
                        "name": "final",
                        "status": "completed",
                        "run_dir": str(run_dir),
                        "results": {
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "FIELD_ERROR": 0.001,
                            "TARGET_IOTA": 0.21,
                            "FINAL_IOTA": 0.209,
                            "FINAL_G": 1.0,
                            "FINAL_NON_QS": 0.02,
                            "FINAL_BOOZER_RESIDUAL": 1e-3,
                            "TIMINGS": {"script_total_s": 12.0},
                        },
                    }
                ]
            }
            (run_root / "continuation_summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            ledger = module.build_candidate_ledger(
                stage2_root=stage2_root,
                single_stage_root=single_stage_root,
                stage2_max_field_error=None,
                single_stage_max_final_field_error=0.01,
                single_stage_max_final_abs_iota_error=0.01,
                single_stage_max_final_non_qs=0.05,
            )

        self.assertEqual(ledger["single_stage"]["candidate_count"], 1)
        self.assertEqual(
            ledger["single_stage"]["best_candidate"]["status"],
            "research_grade",
        )

    def test_build_candidate_ledger_threads_campaign_context_and_schedule(self):
        module = load_candidate_ledger_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_root = root / "stage2"
            single_stage_root = root / "single-stage"
            _write_stage2_run(stage2_root, "seed", field_error=0.01, objective=1e-3)

            campaign_root = single_stage_root / "campaign-run-001"
            donor_run_root = campaign_root / "donor-01-seed-a"
            _write_continuation_validation(
                campaign_root,
                "donor-01-seed-a",
                passed=True,
                field_error=0.001,
                abs_iota_error=0.002,
                final_non_qs=0.01,
                research_grade_ready=True,
            )
            (donor_run_root / "continuation_summary.json").write_text(
                json.dumps(
                    {
                        "run_mode": "new",
                        "trial_policy": "validated-fast",
                        "backend": "jax",
                        "optimizer_backend": "ondevice",
                        "use_target_lane_fast_trials": True,
                        "stages": [
                            {
                                "name": "coarse",
                                "maxiter": 1,
                                "minimal_artifacts": True,
                                "outer_maxls": 4,
                                "shape": {
                                    "mpol": 2,
                                    "ntor": 2,
                                    "nphi": 31,
                                    "ntheta": 16,
                                },
                            },
                            {
                                "name": "final",
                                "maxiter": 300,
                                "minimal_artifacts": False,
                                "shape": {
                                    "mpol": 8,
                                    "ntor": 6,
                                    "nphi": 255,
                                    "ntheta": 64,
                                },
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (campaign_root / "campaign_summary.json").write_text(
                json.dumps(
                    {
                        "run_id": "run-001",
                        "trial_policy": "validated-fast",
                        "backend": "jax",
                        "optimizer_backend": "ondevice",
                        "candidate_count": 1,
                        "status_counts": {"research_grade": 1},
                        "best_candidate": {
                            "run_root": str(donor_run_root.resolve()),
                        },
                        "reports": [
                            {
                                "run_root": str(donor_run_root.resolve()),
                                "donor_index": 1,
                                "donor_label": "01-seed-a",
                                "status": "research_grade",
                                "research_grade": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            ledger = module.build_candidate_ledger(
                stage2_root=stage2_root,
                single_stage_root=single_stage_root,
                stage2_max_field_error=None,
                single_stage_max_final_field_error=None,
                single_stage_max_final_abs_iota_error=None,
                single_stage_max_final_non_qs=None,
            )

        best_candidate = ledger["single_stage"]["best_candidate"]
        self.assertEqual(best_candidate["campaign"]["run_id"], "run-001")
        self.assertEqual(best_candidate["campaign"]["donor_label"], "01-seed-a")
        self.assertTrue(best_candidate["campaign"]["best_candidate"])
        self.assertEqual(best_candidate["continuation"]["trial_policy"], "validated-fast")
        self.assertEqual(best_candidate["continuation"]["backend"], "jax")
        self.assertEqual(len(best_candidate["continuation"]["stages"]), 2)

    def test_build_candidate_ledger_tolerates_unreadable_stage2_results(self):
        module = load_candidate_ledger_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_root = root / "stage2"
            single_stage_root = root / "single-stage"
            best_stage2 = _write_stage2_run(
                stage2_root,
                "good-seed",
                field_error=0.01,
                objective=1e-3,
            )
            corrupt_run = stage2_root / "corrupt-seed"
            corrupt_run.mkdir(parents=True)
            (corrupt_run / "results.json").write_text(
                '{"FIELD_ERROR": 0.02, "FINAL_OBJECTIVE": }',
                encoding="utf-8",
            )
            (corrupt_run / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
            (corrupt_run / "surf_opt.json").write_text("{}", encoding="utf-8")
            _write_continuation_validation(
                single_stage_root,
                "run-a",
                passed=True,
                field_error=0.001,
                abs_iota_error=0.002,
                final_non_qs=0.01,
                research_grade_ready=True,
            )

            ledger = module.build_candidate_ledger(
                stage2_root=stage2_root,
                single_stage_root=single_stage_root,
                stage2_max_field_error=None,
                single_stage_max_final_field_error=None,
                single_stage_max_final_abs_iota_error=None,
                single_stage_max_final_non_qs=None,
            )

        self.assertEqual(
            ledger["stage2"]["best_candidate"]["run_dir"],
            str(best_stage2),
        )
        corrupt_report = next(
            report
            for report in ledger["stage2"]["reports"]
            if report["run_dir"] == str(corrupt_run)
        )
        self.assertEqual(corrupt_report["status"], "rejected")
        self.assertTrue(
            any(
                "results.json is unreadable: JSONDecodeError" in failure
                for failure in corrupt_report["failures"]
            )
        )


if __name__ == "__main__":
    unittest.main()
