import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "run_single_stage_continuation.py"
)


def load_continuation_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_continuation_{uuid.uuid4().hex}",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SingleStageContinuationTests(unittest.TestCase):
    def load_module(self):
        return load_continuation_module()

    def _handle_compile_seed_spec_command(self, command: list[str]) -> bool:
        if "--compile-jax-runtime-seed-spec" not in command:
            return False
        spec_path = Path(command[command.index("--jax-runtime-seed-spec") + 1])
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text("{}", encoding="utf-8")
        return True

    def test_build_default_continuation_stages_for_default_final_shape(self):
        module = self.load_module()

        stages = module.build_default_continuation_stages(
            final_mpol=8,
            final_ntor=6,
            final_nphi=255,
            final_ntheta=64,
            final_maxiter=3,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
        )

        self.assertEqual(
            [
                (stage.name, stage.mpol, stage.ntor, stage.nphi, stage.ntheta, stage.maxiter)
                for stage in stages
            ],
            [
                ("coarse", 2, 2, 31, 16, 1),
                ("medium", 4, 4, 63, 32, 1),
                ("prefinal", 6, 6, 127, 48, 2),
                ("final", 8, 6, 255, 64, 3),
            ],
        )
        self.assertEqual(stages[0].outer_maxls, 2)
        self.assertEqual(stages[1].outer_maxls, 4)
        self.assertEqual(stages[2].outer_maxls, 6)
        self.assertEqual(stages[0].initial_step_scale, 0.1)
        self.assertEqual(stages[0].initial_step_maxiter, 0)
        self.assertEqual(stages[1].initial_step_scale, 0.25)
        self.assertEqual(stages[1].initial_step_maxiter, 1)
        self.assertEqual(stages[2].initial_step_scale, 0.5)
        self.assertEqual(stages[2].initial_step_maxiter, 1)
        self.assertIsNone(stages[3].outer_maxls)
        self.assertIsNone(stages[3].initial_step_scale)
        self.assertIsNone(stages[3].initial_step_maxiter)

    def test_build_default_continuation_stages_keeps_real_final_stage_when_shapes_collapse(
        self,
    ):
        module = self.load_module()

        stages = module.build_default_continuation_stages(
            final_mpol=2,
            final_ntor=2,
            final_nphi=31,
            final_ntheta=16,
            final_maxiter=7,
            coarse_maxiter=1,
            medium_maxiter=2,
            prefinal_maxiter=3,
        )

        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0].name, "final")
        self.assertEqual(stages[0].maxiter, 7)
        self.assertFalse(stages[0].minimal_artifacts)

    def test_full_resolution_warm_start_runs_final_stage_only(self):
        module = self.load_module()

        stages = [
            module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
            module.ContinuationStage("medium", 4, 4, 63, 32, 2),
            module.ContinuationStage("final", 10, 10, 255, 64, 3),
        ]

        selected = module.select_continuation_stages_for_initial_resolution(
            stages,
            initial_resolution=(10, 10, 255, 64),
        )

        self.assertEqual(selected, [stages[-1]])

    def test_lower_resolution_warm_start_skips_completed_stage(self):
        module = self.load_module()

        stages = [
            module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
            module.ContinuationStage("medium", 4, 4, 63, 32, 2),
            module.ContinuationStage("final", 10, 10, 255, 64, 3),
        ]

        selected = module.select_continuation_stages_for_initial_resolution(
            stages,
            initial_resolution=(2, 2, 31, 16),
        )

        self.assertEqual(selected, stages[1:])

    def test_high_resolution_warm_start_runs_next_dominating_stage(self):
        module = self.load_module()

        stages = [
            module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
            module.ContinuationStage("medium", 4, 4, 63, 32, 2),
            module.ContinuationStage("prefinal", 6, 6, 127, 48, 2),
            module.ContinuationStage("final", 10, 10, 255, 64, 3),
        ]

        selected = module.select_continuation_stages_for_initial_resolution(
            stages,
            initial_resolution=(8, 6, 127, 32),
        )

        self.assertEqual(selected, [stages[-1]])

    def test_strip_overridden_passthrough_args(self):
        module = self.load_module()

        stripped = module.strip_overridden_passthrough_args(
            [
                "--backend",
                "jax",
                "--output-root",
                "/tmp/out",
                "--stage2-bs-path",
                "/tmp/seed.json",
                "--warm-start-run-dir",
                "/tmp/run",
                "--iota-target",
                "0.15",
            ]
        )

        self.assertEqual(
            stripped,
            ["--backend", "jax", "--iota-target", "0.15"],
        )

    def test_strip_overridden_passthrough_args_handles_equals_syntax(self):
        module = self.load_module()

        stripped = module.strip_overridden_passthrough_args(
            [
                "--backend=jax",
                "--stage2-bs-path=/tmp/seed.json",
                "--warm-start-run-dir=/tmp/run",
                "--output-root=/tmp/out",
                "--iota-target=0.15",
            ]
        )

        self.assertEqual(stripped, ["--backend=jax", "--iota-target=0.15"])

    def test_resolve_initial_stage_inputs_uses_warm_start_seed_when_needed(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_run_dir = Path(tmpdir) / "prev"
            warm_start_run_dir.mkdir()
            (warm_start_run_dir / "biot_savart_opt.json").write_text(
                "{}",
                encoding="utf-8",
            )

            stage2_seed_path, resolved_warm_start_run_dir = (
                module.resolve_initial_stage_inputs(
                    initial_stage2_bs_path=None,
                    initial_warm_start_run_dir=str(warm_start_run_dir),
                )
            )

        self.assertEqual(
            stage2_seed_path,
            (warm_start_run_dir / "biot_savart_opt.json").resolve(),
        )
        self.assertEqual(resolved_warm_start_run_dir, warm_start_run_dir.resolve())

    def test_build_stage_command_threads_seed_and_warm_start(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_seed_path = Path(tmpdir) / "seed.json"
            stage2_seed_path.write_text("{}", encoding="utf-8")
            warm_start_run_dir = Path(tmpdir) / "prev"
            warm_start_run_dir.mkdir()
            command = module.build_stage_command(
                python_executable="/usr/bin/python3",
                passthrough_args=["--backend", "jax"],
                stage=module.ContinuationStage("final", 8, 6, 255, 64, 3),
                stage_output_root=Path(tmpdir) / "stage",
                stage2_seed_path=stage2_seed_path,
                warm_start_run_dir=warm_start_run_dir,
                jax_runtime_seed_spec_path=Path(tmpdir) / "stage-seed.json",
                jax_profile_dir=None,
                use_target_lane_fast_trials=True,
            )

        self.assertIn("--stage2-bs-path", command)
        self.assertIn(str(stage2_seed_path), command)
        self.assertIn("--warm-start-run-dir", command)
        self.assertIn(str(warm_start_run_dir), command)
        self.assertIn("--jax-runtime-seed-spec", command)
        self.assertIn(str(Path(tmpdir) / "stage-seed.json"), command)
        self.assertEqual(command[-2:], ["--backend", "jax"])

    def test_load_warm_start_contract_overrides_reads_results_contract(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            warm_start_run_dir = Path(tmpdir) / "prev"
            warm_start_run_dir.mkdir()
            (warm_start_run_dir / "results.json").write_text(
                json.dumps(
                    {
                        "TARGET_VOLUME": 0.04,
                        "TARGET_IOTA": 0.25,
                        "CURVATURE_THRESHOLD": 100.0,
                        "CC_DIST": 0.05,
                        "CS_DIST": 0.015,
                        "SS_DIST": 0.04,
                        "BANANA_CURRENT_MAX_A": 16000.0,
                        "LENGTH_TARGET": 1.7,
                    }
                ),
                encoding="utf-8",
            )

            overrides = module.load_warm_start_contract_overrides(warm_start_run_dir)

        self.assertEqual(
            overrides,
            {
                "--vol-target": 0.04,
                "--iota-target": 0.25,
                "--curvature-threshold": 100.0,
                "--cc-dist": 0.05,
                "--cs-dist": 0.015,
                "--ss-dist": 0.04,
                "--banana-current-max-A": 16000.0,
                "--length-target": 1.7,
            },
        )

    def test_build_stage_command_threads_warm_start_target_contract(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=["--backend", "jax"],
            stage=module.ContinuationStage("final", 8, 6, 255, 64, 3),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=Path("/tmp/prev"),
            jax_runtime_seed_spec_path=Path("/tmp/stage-seed.json"),
            jax_profile_dir=None,
            use_target_lane_fast_trials=True,
            warm_start_target_overrides={
                "--vol-target": 0.04,
                "--iota-target": 0.25,
                "--curvature-threshold": 100.0,
            },
        )

        self.assertEqual(command[command.index("--vol-target") + 1], "0.04")
        self.assertEqual(command[command.index("--iota-target") + 1], "0.25")
        self.assertEqual(
            command[command.index("--curvature-threshold") + 1],
            "100.0",
        )

    def test_build_stage_command_respects_explicit_target_passthrough(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=[
                "--backend",
                "jax",
                "--vol-target",
                "0.06",
                "--iota-target=0.18",
            ],
            stage=module.ContinuationStage("final", 8, 6, 255, 64, 3),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=Path("/tmp/prev"),
            jax_runtime_seed_spec_path=Path("/tmp/stage-seed.json"),
            jax_profile_dir=None,
            use_target_lane_fast_trials=True,
            warm_start_target_overrides={"--vol-target": 0.04, "--iota-target": 0.25},
        )

        self.assertEqual(command.count("--vol-target"), 1)
        self.assertNotIn("0.04", command)
        self.assertIn("0.06", command)
        self.assertIn("--iota-target=0.18", command)
        self.assertNotIn("0.25", command)

    def test_build_stage_command_threads_stage_specific_jax_profile_dir(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=["--backend", "jax"],
            stage=module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=None,
            jax_runtime_seed_spec_path=None,
            jax_profile_dir=Path("/tmp/xprof/stage-01-coarse"),
            use_target_lane_fast_trials=True,
        )

        self.assertIn("--jax-profile-dir", command)
        self.assertIn("/tmp/xprof/stage-01-coarse", command)

    def test_build_stage_jax_runtime_seed_spec_command_uses_stage_resolution(self):
        module = self.load_module()

        command = module.build_stage_jax_runtime_seed_spec_command(
            python_executable="/usr/bin/python3",
            passthrough_args=["--backend", "jax", "--optimizer-backend", "ondevice"],
            stage=module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
            warm_start_run_dir=Path("/tmp/donor-final-resolution"),
            jax_runtime_seed_spec_path=Path("/tmp/stage/single_stage_jax_runtime_spec.json"),
        )

        self.assertIn("--compile-jax-runtime-seed-spec", command)
        self.assertEqual(command[command.index("--mpol") + 1], "2")
        self.assertEqual(command[command.index("--ntor") + 1], "2")
        self.assertEqual(command[command.index("--nphi") + 1], "31")
        self.assertEqual(command[command.index("--ntheta") + 1], "16")
        self.assertEqual(
            command[command.index("--jax-runtime-seed-spec") + 1],
            "/tmp/stage/single_stage_jax_runtime_spec.json",
        )

    def test_existing_stage_jax_runtime_seed_spec_path_reuses_matching_shape(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            donor_dir = Path(tmpdir)
            runtime_spec_path = donor_dir / "single_stage_jax_runtime_spec.json"
            runtime_spec_path.write_text(
                json.dumps(
                    {
                        "surface": {"mpol": 10, "ntor": 10},
                        "quadrature": {"nphi": 255, "ntheta": 64},
                    }
                ),
                encoding="utf-8",
            )

            resolved = module.existing_stage_jax_runtime_seed_spec_path(
                donor_dir,
                module.ContinuationStage("final", 10, 10, 255, 64, 3),
            )

        self.assertEqual(resolved, runtime_spec_path)

    def test_existing_stage_jax_runtime_seed_spec_path_rejects_other_shape(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            donor_dir = Path(tmpdir)
            runtime_spec_path = donor_dir / "single_stage_jax_runtime_spec.json"
            runtime_spec_path.write_text(
                json.dumps(
                    {
                        "surface": {"mpol": 8, "ntor": 6},
                        "quadrature": {"nphi": 255, "ntheta": 64},
                    }
                ),
                encoding="utf-8",
            )

            resolved = module.existing_stage_jax_runtime_seed_spec_path(
                donor_dir,
                module.ContinuationStage("final", 10, 10, 255, 64, 3),
            )

        self.assertIsNone(resolved)

    def test_build_stage_command_injects_target_lane_fast_trial_budgets(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=["--backend", "jax"],
            stage=module.ContinuationStage(
                "medium",
                4,
                4,
                63,
                32,
                1,
                minimal_artifacts=True,
                outer_maxls=6,
                maxcor=12,
                initial_step_scale=0.25,
                initial_step_maxiter=1,
                target_lane_boozer_bfgs_tol=3e-6,
                target_lane_boozer_bfgs_maxiter=32,
            ),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=None,
            jax_runtime_seed_spec_path=None,
            jax_profile_dir=None,
            use_target_lane_fast_trials=True,
        )

        self.assertIn("--minimal-artifacts", command)
        self.assertIn("--outer-maxls", command)
        self.assertIn("6", command)
        self.assertIn("--maxcor", command)
        self.assertIn("12", command)
        self.assertIn("--initial-step-scale", command)
        self.assertIn("0.25", command)
        self.assertIn("--initial-step-maxiter", command)
        self.assertIn("1", command)
        self.assertIn("--target-lane-boozer-bfgs-tol", command)
        self.assertIn("3e-06", command)
        self.assertIn("--target-lane-boozer-bfgs-maxiter", command)
        self.assertIn("32", command)

    def test_build_stage_command_keeps_full_artifacts_for_final_stage(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=["--backend", "jax"],
            stage=module.ContinuationStage("final", 8, 6, 255, 64, 3),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=None,
            jax_runtime_seed_spec_path=None,
            jax_profile_dir=None,
            use_target_lane_fast_trials=True,
        )

        self.assertNotIn("--minimal-artifacts", command)
        self.assertNotIn("--outer-maxls", command)

    def test_build_stage_command_respects_explicit_minimal_artifacts_passthrough(self):
        module = self.load_module()

        command = module.build_stage_command(
            python_executable="/usr/bin/python3",
            passthrough_args=[
                "--backend",
                "jax",
                "--minimal-artifacts",
                "--outer-maxls",
                "9",
                "--initial-step-scale",
                "0.4",
                "--initial-step-maxiter",
                "2",
            ],
            stage=module.ContinuationStage(
                "medium",
                4,
                4,
                63,
                32,
                1,
                minimal_artifacts=True,
                outer_maxls=6,
            ),
            stage_output_root=Path("/tmp/stage"),
            stage2_seed_path=None,
            warm_start_run_dir=None,
            jax_runtime_seed_spec_path=None,
            jax_profile_dir=None,
            use_target_lane_fast_trials=True,
        )

        self.assertEqual(command.count("--minimal-artifacts"), 1)
        self.assertEqual(command.count("--outer-maxls"), 1)
        self.assertEqual(command.count("--initial-step-scale"), 1)
        self.assertEqual(command.count("--initial-step-maxiter"), 1)
        self.assertIn("9", command)
        self.assertIn("0.4", command)
        self.assertIn("2", command)

    def test_parse_args_collects_campaign_donor_run_dirs(self):
        module = self.load_module()

        args, passthrough = module.parse_args(
            [
                "--campaign-donor-run-dir",
                "/tmp/donor-a",
                "--campaign-donor-run-dir",
                "/tmp/donor-b",
                "--backend",
                "jax",
            ]
        )

        self.assertEqual(
            args.campaign_donor_run_dir,
            ["/tmp/donor-a", "/tmp/donor-b"],
        )
        self.assertEqual(passthrough, ["--backend", "jax"])

    def test_build_continuation_campaign_summary_prefers_research_grade_status(self):
        module = self.load_module()

        research_grade = module.build_campaign_candidate_record(
            donor_index=1,
            donor_label="01-seed-a",
            donor_run_dir=Path("/tmp/donor-a"),
            outcome=module.ContinuationRunOutcome(
                run_root=Path("/tmp/campaign/donor-a"),
                summary_path=Path("/tmp/campaign/donor-a/continuation_summary.json"),
                summary={},
                report_path=Path(
                    "/tmp/campaign/donor-a/continuation_validation.json"
                ),
                report={
                    "passed": True,
                    "research_verdicts": {"research_grade_ready": True},
                    "profiling": {
                        "total_stage_script_time_s": 34.5,
                        "total_accepted_step_count": 5,
                        "total_objective_eval_count": 21,
                        "total_gradient_eval_count": 21,
                        "objective_evals_per_accepted_step": 4.2,
                        "total_value_and_grad_compile_overhead_s": 2.8,
                    },
                    "final_stage": {
                        "abs_iota_error": 0.01,
                        "metrics": {
                            "FIELD_ERROR": 1e-3,
                            "FINAL_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_BOOZER_RESIDUAL": 2e-5,
                        },
                    },
                    "failures": [],
                    "warnings": [],
                },
                exit_code=0,
            ),
        )
        eligible = module.build_campaign_candidate_record(
            donor_index=2,
            donor_label="02-seed-b",
            donor_run_dir=Path("/tmp/donor-b"),
            outcome=module.ContinuationRunOutcome(
                run_root=Path("/tmp/campaign/donor-b"),
                summary_path=Path("/tmp/campaign/donor-b/continuation_summary.json"),
                summary={},
                report_path=Path(
                    "/tmp/campaign/donor-b/continuation_validation.json"
                ),
                report={
                    "passed": True,
                    "research_verdicts": {"research_grade_ready": False},
                    "profiling": {
                        "total_stage_script_time_s": 40.0,
                        "total_accepted_step_count": 4,
                        "total_objective_eval_count": 18,
                        "total_gradient_eval_count": 18,
                        "objective_evals_per_accepted_step": 4.5,
                        "total_value_and_grad_compile_overhead_s": 3.1,
                    },
                    "final_stage": {
                        "abs_iota_error": 0.002,
                        "metrics": {
                            "FIELD_ERROR": 1e-4,
                            "FINAL_IOTA": 0.208,
                            "FINAL_NON_QS": 0.02,
                            "FINAL_BOOZER_RESIDUAL": 1e-5,
                        },
                    },
                    "failures": [],
                    "warnings": [],
                },
                exit_code=0,
            ),
        )

        summary = module.build_continuation_campaign_summary(
            campaign_root=Path("/tmp/campaign"),
            run_id="run-001",
            donor_records=[eligible, research_grade],
            passthrough_args=["--backend", "jax"],
            trial_policy="validated-fast",
            validation_thresholds={
                "max_final_field_error": 5e-4,
                "max_final_abs_iota_error": 0.01,
                "max_final_non_qs": 0.05,
            },
        )

        self.assertEqual(summary["best_candidate"]["status"], "research_grade")
        self.assertEqual(summary["status_counts"]["research_grade"], 1)
        self.assertEqual(summary["status_counts"]["eligible"], 1)
        self.assertEqual(
            summary["best_candidate"]["profiling"]["total_objective_eval_count"],
            21,
        )
        self.assertEqual(summary["profiling"]["profiled_candidate_count"], 2)
        self.assertEqual(summary["profiling"]["total_accepted_step_count"], 9)
        self.assertEqual(summary["profiling"]["total_objective_eval_count"], 39)
        self.assertEqual(summary["profiling"]["total_gradient_eval_count"], 39)
        self.assertIsNone(summary["profiling"]["total_outer_optimizer_s"])
        self.assertIsNone(summary["profiling"]["total_outer_optimizer_initial_phase_s"])
        self.assertIsNone(summary["profiling"]["total_outer_optimizer_main_s"])
        self.assertIsNone(summary["profiling"]["total_target_lane_bundle_setup_s"])
        self.assertAlmostEqual(
            summary["profiling"]["objective_evals_per_accepted_step"],
            39 / 9,
        )
        self.assertAlmostEqual(
            summary["profiling"]["total_value_and_grad_compile_overhead_s"],
            5.9,
        )
        self.assertEqual(
            summary["branch_decision"]["category"],
            "campaign_ready_for_convergence",
        )

    def test_build_continuation_campaign_summary_marks_zero_progress_profile_as_reevaluation_dominated(
        self,
    ):
        module = self.load_module()

        zero_progress_a = module.build_campaign_candidate_record(
            donor_index=1,
            donor_label="01-seed-a",
            donor_run_dir=Path("/tmp/donor-a"),
            outcome=module.ContinuationRunOutcome(
                run_root=Path("/tmp/campaign/donor-a"),
                summary_path=Path("/tmp/campaign/donor-a/continuation_summary.json"),
                summary={},
                report_path=Path(
                    "/tmp/campaign/donor-a/continuation_validation.json"
                ),
                report={
                    "passed": False,
                    "research_verdicts": {"research_grade_ready": False},
                    "profiling": {
                        "total_stage_script_time_s": 417.2,
                        "total_outer_optimizer_s": 327.6,
                        "total_outer_optimizer_initial_phase_s": 101.7,
                        "total_outer_optimizer_main_s": 225.9,
                        "total_target_lane_bundle_setup_s": 10.0,
                        "total_accepted_step_count": 0,
                        "total_objective_eval_count": 6,
                        "total_gradient_eval_count": 6,
                        "objective_evals_per_accepted_step": None,
                    },
                    "final_stage": {
                        "abs_iota_error": 0.05,
                        "metrics": {
                            "FIELD_ERROR": 0.0039,
                            "FINAL_IOTA": 0.095,
                            "FINAL_NON_QS": 0.00011,
                            "FINAL_BOOZER_RESIDUAL": 6e-5,
                        },
                    },
                    "failures": ["coarse stage failed contract"],
                    "warnings": [],
                },
                exit_code=1,
            ),
        )
        zero_progress_b = module.build_campaign_candidate_record(
            donor_index=2,
            donor_label="02-seed-b",
            donor_run_dir=Path("/tmp/donor-b"),
            outcome=module.ContinuationRunOutcome(
                run_root=Path("/tmp/campaign/donor-b"),
                summary_path=Path("/tmp/campaign/donor-b/continuation_summary.json"),
                summary={},
                report_path=Path(
                    "/tmp/campaign/donor-b/continuation_validation.json"
                ),
                report={
                    "passed": False,
                    "research_verdicts": {"research_grade_ready": False},
                    "profiling": {
                        "total_stage_script_time_s": 642.1,
                        "total_outer_optimizer_s": 552.8,
                        "total_outer_optimizer_initial_phase_s": 306.2,
                        "total_outer_optimizer_main_s": 246.6,
                        "total_target_lane_bundle_setup_s": 9.7,
                        "total_accepted_step_count": 0,
                        "total_objective_eval_count": 6,
                        "total_gradient_eval_count": 6,
                        "objective_evals_per_accepted_step": None,
                    },
                    "final_stage": {
                        "abs_iota_error": 0.05,
                        "metrics": {
                            "FIELD_ERROR": 0.0046,
                            "FINAL_IOTA": 0.0995,
                            "FINAL_NON_QS": 0.00012,
                            "FINAL_BOOZER_RESIDUAL": 8e-5,
                        },
                    },
                    "failures": ["coarse stage failed contract"],
                    "warnings": [],
                },
                exit_code=1,
            ),
        )

        summary = module.build_continuation_campaign_summary(
            campaign_root=Path("/tmp/campaign"),
            run_id="run-reeval-001",
            donor_records=[zero_progress_b, zero_progress_a],
            passthrough_args=["--backend", "jax"],
            trial_policy="validated-fast",
            validation_thresholds={
                "max_final_field_error": 5e-4,
                "max_final_abs_iota_error": None,
                "max_final_non_qs": 0.05,
            },
        )

        self.assertAlmostEqual(summary["profiling"]["total_outer_optimizer_s"], 880.4)
        self.assertAlmostEqual(
            summary["profiling"]["total_target_lane_bundle_setup_s"], 19.7
        )
        self.assertEqual(
            summary["branch_decision"]["category"],
            "reevaluation_or_host_stall_dominated",
        )
        self.assertIn(
            "Skip the validated-fast coarse scaled initial outer phase and re-profile the same donor set.",
            summary["branch_decision"]["recommended_actions"],
        )

    def test_build_continuation_campaign_summary_marks_zero_progress_after_phase_skip_as_line_search_budget_dominated(
        self,
    ):
        module = self.load_module()

        zero_progress = module.build_campaign_candidate_record(
            donor_index=1,
            donor_label="01-seed-a",
            donor_run_dir=Path("/tmp/donor-a"),
            outcome=module.ContinuationRunOutcome(
                run_root=Path("/tmp/campaign/donor-a"),
                summary_path=Path("/tmp/campaign/donor-a/continuation_summary.json"),
                summary={},
                report_path=Path(
                    "/tmp/campaign/donor-a/continuation_validation.json"
                ),
                report={
                    "passed": False,
                    "research_verdicts": {"research_grade_ready": False},
                    "profiling": {
                        "total_stage_script_time_s": 178.7,
                        "total_outer_optimizer_s": 99.2,
                        "total_outer_optimizer_initial_phase_s": None,
                        "total_outer_optimizer_main_s": 99.2,
                        "total_target_lane_bundle_setup_s": 9.9,
                        "total_accepted_step_count": 0,
                        "total_objective_eval_count": 6,
                        "total_gradient_eval_count": 6,
                        "objective_evals_per_accepted_step": None,
                    },
                    "final_stage": {
                        "abs_iota_error": 0.05,
                        "metrics": {
                            "FIELD_ERROR": 0.0040,
                            "FINAL_IOTA": 0.0953,
                            "FINAL_NON_QS": 0.00012,
                            "FINAL_BOOZER_RESIDUAL": 6e-5,
                        },
                    },
                    "failures": ["coarse stage failed contract"],
                    "warnings": [],
                },
                exit_code=1,
            ),
        )

        summary = module.build_continuation_campaign_summary(
            campaign_root=Path("/tmp/campaign"),
            run_id="run-reeval-002",
            donor_records=[zero_progress],
            passthrough_args=["--backend", "jax"],
            trial_policy="validated-fast",
            validation_thresholds={
                "max_final_field_error": 5e-4,
                "max_final_abs_iota_error": None,
                "max_final_non_qs": 0.05,
            },
        )

        self.assertEqual(
            summary["branch_decision"]["category"],
            "reevaluation_or_host_stall_dominated",
        )
        self.assertIn(
            "Tighten the validated-fast non-final outer line-search budget and re-profile the same donor set.",
            summary["branch_decision"]["recommended_actions"],
        )
        self.assertIn(
            "The validated-fast coarse scaled phase is already absent, so the remaining waste is in the main outer loop.",
            summary["branch_decision"]["rationale"],
        )

    def test_continuation_uses_target_lane_fast_trials_only_for_nonbenchmark_jax_ondevice(
        self,
    ):
        module = self.load_module()

        self.assertTrue(
            module.continuation_uses_target_lane_fast_trials(
                ["--backend", "jax"]
            )
        )
        self.assertFalse(
            module.continuation_uses_target_lane_fast_trials(
                ["--backend", "jax", "--benchmark-mode"]
            )
        )
        self.assertFalse(
            module.continuation_uses_target_lane_fast_trials(
                ["--backend", "jax", "--optimizer-backend", "scipy"]
            )
        )
        self.assertFalse(module.continuation_uses_target_lane_fast_trials([]))

    def test_continuation_uses_target_lane_fast_trials_honors_env_defaults(self):
        module = self.load_module()

        with patch.dict(
            "os.environ",
            {"SIMSOPT_BACKEND": "jax", "OPTIMIZER_BACKEND": "ondevice"},
            clear=False,
        ):
            self.assertTrue(module.continuation_uses_target_lane_fast_trials([]))

    def test_find_single_stage_run_dir_requires_unique_result(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage_root = Path(tmpdir)
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected exactly one single-stage results.json",
            ):
                module.find_single_stage_run_dir(stage_root)

            first = stage_root / "run-a"
            first.mkdir()
            (first / "results.json").write_text("{}", encoding="utf-8")
            self.assertEqual(module.find_single_stage_run_dir(stage_root), first)

            second = stage_root / "run-b"
            second.mkdir()
            (second / "results.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError,
                "Expected exactly one single-stage results.json",
            ):
                module.find_single_stage_run_dir(stage_root)

    def test_load_single_stage_results(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            payload = {"FINAL_IOTA": 0.123, "FINAL_G": 4.5}
            (run_dir / "results.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            loaded = module.load_single_stage_results(run_dir)

        self.assertEqual(loaded, payload)

    def _write_stage_run(
        self,
        root: Path,
        stage_name: str,
        results_payload: dict[str, object],
        *,
        include_surface: bool = True,
        status: str = "completed",
    ) -> dict[str, object]:
        run_dir = root / stage_name
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            json.dumps(results_payload),
            encoding="utf-8",
        )
        (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
        if include_surface:
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
        return {
            "name": stage_name,
            "status": status,
            "run_dir": str(run_dir),
            "results": results_payload,
        }

    def _write_jax_stage_run(
        self,
        root: Path,
        stage_name: str,
        results_payload: dict[str, object],
        *,
        include_runtime_spec: bool = True,
        status: str = "completed",
    ) -> dict[str, object]:
        run_dir = root / stage_name
        run_dir.mkdir(parents=True)
        payload = {
            "backend": "jax",
            "optimizer_backend": "ondevice",
            **results_payload,
        }
        (run_dir / "results.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        if include_runtime_spec:
            (run_dir / "single_stage_jax_runtime_spec.json").write_text(
                "{}",
                encoding="utf-8",
            )
        return {
            "name": stage_name,
            "status": status,
            "run_dir": str(run_dir),
            "results": payload,
        }

    def _write_existing_stage_output(
        self,
        run_root: Path,
        stage_dir_name: str,
        run_dir_name: str,
        results_payload: dict[str, object],
        *,
        include_surface: bool = True,
    ) -> Path:
        stage_output_root = run_root / stage_dir_name
        run_dir = stage_output_root / run_dir_name
        run_dir.mkdir(parents=True)
        results_payload = {
            "TARGET_VOLUME": 0.1,
            "TARGET_IOTA": 0.21,
            "CURVATURE_THRESHOLD": 100.0,
            "CC_DIST": 0.05,
            "CS_DIST": 0.015,
            "SS_DIST": 0.04,
            "BANANA_CURRENT_MAX_A": 16000.0,
            "LENGTH_TARGET": 1.7,
            **results_payload,
        }
        (run_dir / "results.json").write_text(
            json.dumps(results_payload),
            encoding="utf-8",
        )
        (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")
        if include_surface:
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
        return stage_output_root

    def test_build_continuation_validation_report_passes_for_valid_final_stage(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "coarse",
                        {
                            "FINAL_IOTA": 0.18,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.08,
                            "FINAL_G": 4.0,
                            "FIELD_ERROR": 5e-4,
                            "INITIAL_IOTA": 0.17,
                            "INITIAL_FIELD_ERROR": 8e-4,
                            "iterations": 1,
                            "OPTIMIZER_SUCCESS": False,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "TERMINATION_MESSAGE": "Line search failed.",
                            "TIMINGS": {"script_total_s": 12.5},
                        },
                    ),
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "TIMINGS": {"script_total_s": 22.0},
                        },
                    ),
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=1e-3,
                max_final_abs_iota_error=0.01,
                max_final_non_qs=0.05,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(report["completed_stage_count"], 2)
        self.assertTrue(report["research_verdicts"]["full_convergence"])
        self.assertTrue(report["research_verdicts"]["hardware_feasible_final_coils"])
        self.assertTrue(report["research_verdicts"]["acceptable_non_qs_behavior"])
        self.assertTrue(report["research_verdicts"]["research_grade_ready"])
        self.assertAlmostEqual(
            report["aggregate"]["field_error_improvement_vs_first_completed"],
            2.5e-4,
        )
        self.assertAlmostEqual(report["aggregate"]["total_stage_script_time_s"], 34.5)

    def test_build_continuation_validation_report_accepts_jax_runtime_spec_artifact(
        self,
    ):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_jax_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "TIMINGS": {"script_total_s": 22.0},
                        },
                    )
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=1e-3,
                max_final_abs_iota_error=0.01,
                max_final_non_qs=0.05,
            )

        self.assertTrue(report["passed"])
        self.assertIn(
            "single_stage_jax_runtime_spec.json",
            report["stage_reports"][0]["artifacts"]["files"],
        )
        self.assertNotIn(
            "biot_savart_opt.json",
            report["stage_reports"][0]["artifacts"]["files"],
        )

    def test_build_continuation_validation_report_rejects_missing_jax_runtime_spec(
        self,
    ):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_jax_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        },
                        include_runtime_spec=False,
                    )
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=None,
                max_final_abs_iota_error=None,
                max_final_non_qs=None,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "single_stage_jax_runtime_spec.json" in failure
                for failure in report["failures"]
            )
        )

    def test_build_continuation_validation_report_includes_compact_profiling_summary(
        self,
    ):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "coarse",
                        {
                            "FINAL_IOTA": 0.18,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.08,
                            "FINAL_G": 4.0,
                            "FIELD_ERROR": 5e-4,
                            "INITIAL_IOTA": 0.17,
                            "INITIAL_FIELD_ERROR": 8e-4,
                            "INITIAL_PHASE_ITERATIONS": 1,
                            "iterations": 2,
                            "OPTIMIZER_SUCCESS": False,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "OPTIMIZER_NFEV": 9,
                            "OPTIMIZER_NJEV": 9,
                            "TIMINGS": {
                                "script_total_s": 12.5,
                                "outer_optimizer_s": 10.0,
                                "outer_optimizer_initial_phase_s": 2.0,
                                "outer_optimizer_main_s": 8.0,
                                "target_lane_bundle_setup_s": 1.5,
                            },
                            "TARGET_LANE_PROFILE": {
                                "solve_success": True,
                                "inner_solve": {
                                    "compile_overhead_s": 0.4,
                                    "first": {"total_s": 1.0},
                                    "warm": {"total_s": 0.6},
                                },
                                "value_and_grad_pipeline": {
                                    "compile_overhead_s": 1.2,
                                    "first": {"total_s": 2.5},
                                    "warm": {"total_s": 1.3},
                                },
                            },
                        },
                    ),
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "INITIAL_PHASE_ITERATIONS": 1,
                            "iterations": 3,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "OPTIMIZER_NFEV": 12,
                            "OPTIMIZER_NJEV": 12,
                            "TIMINGS": {
                                "script_total_s": 22.0,
                                "outer_optimizer_s": 20.0,
                                "outer_optimizer_initial_phase_s": 3.0,
                                "outer_optimizer_main_s": 17.0,
                                "target_lane_bundle_setup_s": 2.5,
                            },
                            "TARGET_LANE_PROFILE": {
                                "solve_success": True,
                                "inner_solve": {
                                    "compile_overhead_s": 0.5,
                                    "first": {"total_s": 1.1},
                                    "warm": {"total_s": 0.6},
                                },
                                "value_and_grad_pipeline": {
                                    "compile_overhead_s": 1.6,
                                    "first": {"total_s": 2.8},
                                    "warm": {"total_s": 1.2},
                                },
                            },
                        },
                    ),
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=1e-3,
                max_final_abs_iota_error=0.01,
                max_final_non_qs=0.05,
            )

        coarse_profiling = report["profiling"]["stages"]["coarse"]
        final_profiling = report["profiling"]["stages"]["final"]

        self.assertEqual(report["profiling"]["profiled_stage_count"], 2)
        self.assertEqual(report["profiling"]["target_lane_profiled_stage_count"], 2)
        self.assertAlmostEqual(report["profiling"]["total_stage_script_time_s"], 34.5)
        self.assertAlmostEqual(report["profiling"]["total_outer_optimizer_s"], 30.0)
        self.assertAlmostEqual(
            report["profiling"]["total_target_lane_bundle_setup_s"],
            4.0,
        )
        self.assertEqual(report["profiling"]["total_accepted_step_count"], 5)
        self.assertEqual(report["profiling"]["total_objective_eval_count"], 21)
        self.assertEqual(report["profiling"]["total_gradient_eval_count"], 21)
        self.assertAlmostEqual(
            report["profiling"]["objective_evals_per_accepted_step"],
            21 / 5,
        )
        self.assertAlmostEqual(
            report["profiling"]["total_value_and_grad_compile_overhead_s"],
            2.8,
        )
        self.assertAlmostEqual(
            report["profiling"]["total_inner_solve_compile_overhead_s"],
            0.9,
        )
        self.assertEqual(coarse_profiling["accepted_step_count"], 2)
        self.assertEqual(coarse_profiling["initial_phase_iterations"], 1)
        self.assertEqual(coarse_profiling["objective_eval_count"], 9)
        self.assertAlmostEqual(
            coarse_profiling["objective_evals_per_accepted_step"],
            4.5,
        )
        self.assertAlmostEqual(
            coarse_profiling["target_lane_profile"][
                "value_and_grad_compile_overhead_s"
            ],
            1.2,
        )
        self.assertEqual(final_profiling["accepted_step_count"], 3)
        self.assertEqual(final_profiling["objective_eval_count"], 12)

    def test_build_continuation_profiling_report_markdown_emits_stage_metrics(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "coarse",
                        {
                            "FINAL_IOTA": 0.18,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.08,
                            "FINAL_G": 4.0,
                            "FIELD_ERROR": 5e-4,
                            "INITIAL_IOTA": 0.17,
                            "INITIAL_FIELD_ERROR": 8e-4,
                            "INITIAL_PHASE_ITERATIONS": 1,
                            "iterations": 2,
                            "OPTIMIZER_SUCCESS": False,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "OPTIMIZER_NFEV": 9,
                            "OPTIMIZER_NJEV": 9,
                            "JAX_PROFILE_DIR": "/tmp/profiles/coarse",
                            "TIMINGS": {
                                "script_total_s": 12.5,
                                "outer_optimizer_s": 10.0,
                                "target_lane_bundle_setup_s": 1.5,
                            },
                            "TARGET_LANE_PROFILE": {
                                "solve_success": True,
                                "inner_solve": {"compile_overhead_s": 0.4},
                                "value_and_grad_pipeline": {
                                    "compile_overhead_s": 1.2,
                                },
                            },
                        },
                    ),
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "iterations": 3,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "OPTIMIZER_NFEV": 12,
                            "OPTIMIZER_NJEV": 12,
                            "TIMINGS": {"script_total_s": 22.0},
                        },
                    ),
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=1e-3,
                max_final_abs_iota_error=0.01,
                max_final_non_qs=0.05,
            )
            markdown = module.build_continuation_profiling_report_markdown(report)

        self.assertIn("# Continuation Profiling Report", markdown)
        self.assertIn("- Total objective evaluations: 21", markdown)
        self.assertIn("- Objective evaluations per accepted step: 4.2", markdown)
        self.assertIn("## Stage `coarse`", markdown)
        self.assertIn("- JAX profile dir: `/tmp/profiles/coarse`", markdown)
        self.assertIn("- Value-and-grad compile overhead (s): 1.2", markdown)

    def test_build_campaign_profiling_report_markdown_emits_candidate_metrics(self):
        module = self.load_module()

        summary = {
            "campaign_root": "/tmp/campaign-run-001",
            "run_id": "run-001",
            "profiling": {
                "profiled_candidate_count": 2,
                "total_stage_script_time_s": 56.5,
                "total_outer_optimizer_s": 48.0,
                "total_outer_optimizer_initial_phase_s": 8.0,
                "total_outer_optimizer_main_s": 40.0,
                "total_target_lane_bundle_setup_s": 3.2,
                "total_accepted_step_count": 7,
                "total_objective_eval_count": 29,
                "objective_evals_per_accepted_step": 29 / 7,
                "total_gradient_eval_count": 29,
                "total_value_and_grad_compile_overhead_s": 3.5,
            },
            "branch_decision": {
                "category": "campaign_ready_for_convergence",
                "rationale": [
                    "The campaign already produced continuation-valid donors."
                ],
                "recommended_actions": [
                    "Run longer multi-donor convergence campaigns."
                ],
            },
            "reports": [
                {
                    "donor_label": "01-good-donor",
                    "status": "research_grade",
                    "research_grade": True,
                    "run_root": "/tmp/campaign-run-001/donor-01",
                    "profiling": {
                        "total_stage_script_time_s": 24.0,
                        "total_outer_optimizer_s": 19.0,
                        "total_outer_optimizer_initial_phase_s": 3.0,
                        "total_outer_optimizer_main_s": 16.0,
                        "total_accepted_step_count": 4,
                        "total_objective_eval_count": 14,
                        "objective_evals_per_accepted_step": 3.5,
                        "total_gradient_eval_count": 14,
                        "total_value_and_grad_compile_overhead_s": 1.2,
                    },
                },
                {
                    "donor_label": "02-borderline-donor",
                    "status": "eligible",
                    "research_grade": False,
                    "run_root": "/tmp/campaign-run-001/donor-02",
                    "profiling": {
                        "total_stage_script_time_s": 32.5,
                        "total_outer_optimizer_s": 29.0,
                        "total_outer_optimizer_initial_phase_s": 5.0,
                        "total_outer_optimizer_main_s": 24.0,
                        "total_accepted_step_count": 3,
                        "total_objective_eval_count": 15,
                        "objective_evals_per_accepted_step": 5.0,
                        "total_gradient_eval_count": 15,
                        "total_value_and_grad_compile_overhead_s": 2.3,
                    },
                },
            ],
        }

        markdown = module.build_campaign_profiling_report_markdown(summary)

        self.assertIn("# Campaign Profiling Report", markdown)
        self.assertIn("- Profiled candidates: 2", markdown)
        self.assertIn("- Total outer optimizer time (s): 48", markdown)
        self.assertIn("- Total initial outer phase time (s): 8", markdown)
        self.assertIn("- Total main outer phase time (s): 40", markdown)
        self.assertIn("- Total objective evaluations: 29", markdown)
        self.assertIn("## Branch Decision", markdown)
        self.assertIn("- Category: campaign_ready_for_convergence", markdown)
        self.assertIn("## Donor `01-good-donor`", markdown)
        self.assertIn("- Total outer optimizer time (s): 19", markdown)
        self.assertIn("- Objective evaluations per accepted step: 3.5", markdown)

    def test_build_continuation_validation_report_rejects_missing_surface_artifact(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        },
                        include_surface=False,
                    )
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=None,
                max_final_abs_iota_error=None,
                max_final_non_qs=None,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("surf_opt.json" in failure for failure in report["failures"])
        )

    def test_build_continuation_validation_report_rejects_failed_final_contract(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": False,
                            "HARDWARE_CONSTRAINTS_OK": False,
                        },
                    )
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=None,
                max_final_abs_iota_error=None,
                max_final_non_qs=None,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("OPTIMIZER_SUCCESS" in failure for failure in report["failures"])
        )
        self.assertTrue(
            any(
                "HARDWARE_CONSTRAINTS_OK" in failure for failure in report["failures"]
            )
        )

    def test_build_continuation_validation_report_enforces_optional_thresholds(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.16,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.11,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 3e-3,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        },
                    )
                ],
            }

            report = module.build_continuation_validation_report(
                summary,
                max_final_field_error=1e-3,
                max_final_abs_iota_error=0.02,
                max_final_non_qs=0.05,
            )

        self.assertFalse(report["passed"])
        self.assertTrue(
            any("FIELD_ERROR" in failure for failure in report["failures"])
        )
        self.assertTrue(
            any("FINAL_IOTA - TARGET_IOTA" in failure for failure in report["failures"])
        )
        self.assertTrue(
            any("FINAL_NON_QS" in failure for failure in report["failures"])
        )
        self.assertFalse(report["research_verdicts"]["acceptable_non_qs_behavior"])

    def test_validate_summary_json_mode_writes_report(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = {
                "run_root": str(root),
                "stages": [
                    self._write_stage_run(
                        root,
                        "final",
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        },
                    )
                ],
            }
            summary_path = root / "continuation_summary.json"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            module.main(
                [
                    "--validate-summary-json",
                    str(summary_path),
                    "--max-final-non-qs",
                    "0.05",
                    "--strict-validation",
                ]
            )

            report_path = root / "continuation_validation.json"
            self.assertTrue(report_path.exists())
            report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(report["passed"])

    def test_main_summarize_run_root_reconstructs_partial_existing_run(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "continuation-existing"
            summary_path = run_root / "continuation_summary.json"
            self._write_existing_stage_output(
                run_root,
                "stage-01-coarse",
                "coarse-run",
                {
                    "FINAL_IOTA": 0.18,
                    "TARGET_IOTA": 0.21,
                    "FINAL_G": 4.0,
                    "FIELD_ERROR": 5e-4,
                    "OPTIMIZER_SUCCESS": False,
                    "HARDWARE_CONSTRAINTS_OK": True,
                },
            )
            (run_root / "stage-02-medium" / "partial-run").mkdir(parents=True)

            module.main(["--summarize-run-root", str(run_root)])

            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (run_root / "continuation_validation.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(summary["run_mode"], "summarize")
        self.assertEqual(
            [stage["status"] for stage in summary["stages"]],
            [
                "completed",
                "incomplete_existing",
                "not_started",
                "not_started",
            ],
        )
        self.assertTrue(summary["stages"][0]["reused_existing_run"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["stage_reports"][1]["status"], "incomplete_existing")

    def test_main_resume_run_root_skips_completed_stages(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "continuation-existing"
            summary_path = run_root / "continuation_summary.json"
            self._write_existing_stage_output(
                run_root,
                "stage-01-coarse",
                "coarse-run",
                {
                    "FINAL_IOTA": 0.18,
                    "TARGET_IOTA": 0.21,
                    "FINAL_G": 4.0,
                    "FIELD_ERROR": 5e-4,
                    "INITIAL_IOTA": 0.17,
                    "INITIAL_FIELD_ERROR": 8e-4,
                    "iterations": 1,
                    "OPTIMIZER_SUCCESS": False,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "TERMINATION_MESSAGE": "Line search failed.",
                },
            )
            seen_commands: list[list[str]] = []

            def fake_run(command, check):
                self.assertTrue(check)
                if self._handle_compile_seed_spec_command(command):
                    return subprocess.CompletedProcess(command, 0)
                seen_commands.append(command)
                stage_output_root = Path(command[command.index("--output-root") + 1])
                run_dir = stage_output_root / "final-run"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            stages = [
                module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
                module.ContinuationStage("final", 8, 6, 255, 64, 3),
            ]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    module.main(
                        [
                            "--resume-run-root",
                            str(run_root),
                            "--strict-validation",
                        ]
                    )

            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (run_root / "continuation_validation.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(seen_commands), 1)
        self.assertIn("stage-02-final", seen_commands[0][seen_commands[0].index("--output-root") + 1])
        self.assertEqual(summary["run_mode"], "resume")
        self.assertTrue(summary["stages"][0]["reused_existing_run"])
        self.assertEqual(
            [stage["status"] for stage in summary["stages"]],
            ["completed", "completed"],
        )
        self.assertTrue(report["passed"])

    def test_main_summarize_existing_stage_does_not_claim_missing_jax_profile_dir(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "continuation-existing"
            self._write_existing_stage_output(
                run_root,
                "stage-01-coarse",
                "coarse-run",
                {
                    "FINAL_IOTA": 0.19,
                    "TARGET_IOTA": 0.21,
                    "FINAL_G": 4.1,
                    "FIELD_ERROR": 4.5e-4,
                    "INITIAL_IOTA": 0.18,
                    "INITIAL_FIELD_ERROR": 7e-4,
                    "iterations": 1,
                    "OPTIMIZER_SUCCESS": True,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "JAX_PROFILE_DIR": "/tmp/profiles/coarse",
                },
            )

            stages = [module.ContinuationStage("coarse", 2, 2, 31, 16, 1)]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                module.main(
                    [
                        "--summarize-run-root",
                        str(run_root),
                        "--jax-profile-dir",
                        "xprof",
                    ]
                )

            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["jax_profile_dir"], str((run_root / "xprof").resolve()))
        self.assertEqual(summary["stages"][0]["jax_profile_dir"], "/tmp/profiles/coarse")

    def test_main_threads_stage_specific_jax_profile_dirs(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            seen_commands: list[list[str]] = []

            def fake_run(command, check):
                self.assertTrue(check)
                if self._handle_compile_seed_spec_command(command):
                    return subprocess.CompletedProcess(command, 0)
                seen_commands.append(command)
                stage_output_root = Path(command[command.index("--output-root") + 1])
                run_dir = stage_output_root / "run"
                run_dir.mkdir(parents=True, exist_ok=True)
                is_final_stage = stage_output_root.name.endswith("final")
                results_payload = {
                    "FINAL_IOTA": 0.205 if is_final_stage else 0.19,
                    "TARGET_IOTA": 0.21,
                    "TARGET_VOLUME": 0.1,
                    "CURVATURE_THRESHOLD": 100.0,
                    "CC_DIST": 0.05,
                    "CS_DIST": 0.015,
                    "SS_DIST": 0.04,
                    "BANANA_CURRENT_MAX_A": 16000.0,
                    "LENGTH_TARGET": 1.7,
                    "FINAL_NON_QS": 0.03 if is_final_stage else 0.07,
                    "FINAL_G": 4.5 if is_final_stage else 4.1,
                    "FIELD_ERROR": 2.5e-4 if is_final_stage else 4.5e-4,
                    "INITIAL_IOTA": 0.18,
                    "INITIAL_FIELD_ERROR": 7e-4,
                    "iterations": 3 if is_final_stage else 1,
                    "OPTIMIZER_SUCCESS": True,
                    "HARDWARE_CONSTRAINTS_OK": True,
                }
                (run_dir / "results.json").write_text(
                    json.dumps(results_payload),
                    encoding="utf-8",
                )
                (run_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (run_dir / "surf_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            stages = [
                module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
                module.ContinuationStage("final", 8, 6, 255, 64, 3),
            ]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    module.main(
                        [
                            "--output-root",
                            str(output_root),
                            "--run-id",
                            "profiled",
                            "--jax-profile-dir",
                            "xprof",
                            "--strict-validation",
                        ]
                    )

            run_root = output_root / "continuation-profiled"
            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(summary["jax_profile_dir"], str((run_root / "xprof").resolve()))
        self.assertEqual(len(seen_commands), 2)
        coarse_profile_dir = (
            run_root / "xprof" / "stage-01-coarse"
        ).resolve()
        final_profile_dir = (
            run_root / "xprof" / "stage-02-final"
        ).resolve()
        self.assertEqual(
            seen_commands[0][seen_commands[0].index("--jax-profile-dir") + 1],
            str(coarse_profile_dir),
        )
        self.assertEqual(
            seen_commands[1][seen_commands[1].index("--jax-profile-dir") + 1],
            str(final_profile_dir),
        )
        self.assertEqual(summary["stages"][0]["jax_profile_dir"], str(coarse_profile_dir))
        self.assertEqual(summary["stages"][1]["jax_profile_dir"], str(final_profile_dir))

    def test_main_resume_run_root_reruns_invalid_completed_nonfinal_stage(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "continuation-existing"
            summary_path = run_root / "continuation_summary.json"
            self._write_existing_stage_output(
                run_root,
                "stage-01-coarse",
                "coarse-run",
                {
                    "FINAL_IOTA": 0.18,
                    "TARGET_IOTA": 0.21,
                    "FINAL_G": 4.0,
                    "FIELD_ERROR": 5e-4,
                    "OPTIMIZER_SUCCESS": False,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "TERMINATION_MESSAGE": (
                        "Optimization failed with non-finite objective or gradient."
                    ),
                },
            )
            seen_commands: list[list[str]] = []
            live_summaries: list[dict[str, object]] = []

            def fake_run(command, check):
                self.assertTrue(check)
                if self._handle_compile_seed_spec_command(command):
                    return subprocess.CompletedProcess(command, 0)
                seen_commands.append(command)
                stage_output_root = Path(command[command.index("--output-root") + 1])
                live_summaries.append(
                    json.loads(summary_path.read_text(encoding="utf-8"))
                )
                run_dir = stage_output_root / "rerun"
                run_dir.mkdir(parents=True, exist_ok=True)
                stage_name = stage_output_root.name
                results_payload = {
                    "FINAL_IOTA": 0.205 if stage_name.endswith("final") else 0.19,
                    "TARGET_IOTA": 0.21,
                    "TARGET_VOLUME": 0.1,
                    "CURVATURE_THRESHOLD": 100.0,
                    "CC_DIST": 0.05,
                    "CS_DIST": 0.015,
                    "SS_DIST": 0.04,
                    "BANANA_CURRENT_MAX_A": 16000.0,
                    "LENGTH_TARGET": 1.7,
                    "FINAL_NON_QS": 0.03 if stage_name.endswith("final") else 0.07,
                    "FINAL_G": 4.5 if stage_name.endswith("final") else 4.1,
                    "FIELD_ERROR": 2.5e-4 if stage_name.endswith("final") else 4.5e-4,
                    "INITIAL_IOTA": 0.18,
                    "INITIAL_FIELD_ERROR": 7e-4,
                    "iterations": 1 if not stage_name.endswith("final") else 3,
                    "OPTIMIZER_SUCCESS": True,
                    "HARDWARE_CONSTRAINTS_OK": True,
                }
                (run_dir / "results.json").write_text(
                    json.dumps(results_payload),
                    encoding="utf-8",
                )
                (run_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            stages = [
                module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
                module.ContinuationStage("final", 8, 6, 255, 64, 3),
            ]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    module.main(
                        [
                            "--resume-run-root",
                            str(run_root),
                            "--strict-validation",
                        ]
                    )

            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(len(seen_commands), 2)
        self.assertIn(
            "stage-01-coarse",
            seen_commands[0][seen_commands[0].index("--output-root") + 1],
        )
        coarse_live_summary = live_summaries[0]
        coarse_live_stage = coarse_live_summary["stages"][0]
        self.assertEqual(coarse_live_stage["status"], "running")
        self.assertFalse(coarse_live_stage["reused_existing_run"])
        self.assertEqual(
            coarse_live_stage["stage_output_root"],
            seen_commands[0][seen_commands[0].index("--output-root") + 1],
        )
        self.assertNotIn("run_dir", coarse_live_stage)
        self.assertNotIn("results", coarse_live_stage)
        self.assertNotIn("artifacts", coarse_live_stage)
        self.assertNotIn("stage_contract", coarse_live_stage)
        self.assertEqual(
            coarse_live_stage["preexisting_stage_snapshot"]["run_dir"],
            str((run_root / "stage-01-coarse" / "coarse-run").resolve()),
        )
        self.assertIn(
            "stage-02-final",
            seen_commands[1][seen_commands[1].index("--output-root") + 1],
        )
        self.assertEqual(summary["run_mode"], "resume")
        self.assertFalse(summary["stages"][0]["reused_existing_run"])
        self.assertEqual(summary["stages"][0]["resume_replaces_existing_status"], "completed")
        self.assertFalse(summary["stages"][0]["preexisting_stage_validation"]["passed"])

    def test_main_resume_run_root_rewrites_stale_summary_before_rerun(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_root = Path(tmpdir) / "continuation-existing"
            self._write_existing_stage_output(
                run_root,
                "stage-01-coarse",
                "coarse-run",
                {
                    "FINAL_IOTA": 0.18,
                    "TARGET_IOTA": 0.21,
                    "FINAL_G": 4.0,
                    "FIELD_ERROR": 5e-4,
                    "INITIAL_IOTA": 0.17,
                    "INITIAL_FIELD_ERROR": 8e-4,
                    "iterations": 1,
                    "OPTIMIZER_SUCCESS": False,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "TERMINATION_MESSAGE": "Line search failed.",
                },
            )
            summary_path = run_root / "continuation_summary.json"
            report_path = run_root / "continuation_validation.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "stale_marker": True,
                        "stages": [{"name": "stale", "status": "subprocess_failed"}],
                    }
                ),
                encoding="utf-8",
            )
            report_path.write_text(
                json.dumps({"passed": False, "stale_marker": True}),
                encoding="utf-8",
            )

            def fake_run(command, check):
                self.assertTrue(check)
                if self._handle_compile_seed_spec_command(command):
                    return subprocess.CompletedProcess(command, 0)
                live_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                self.assertNotIn("stale_marker", live_summary)
                self.assertEqual(live_summary["run_mode"], "resume")
                self.assertEqual(live_summary["stages"][0]["status"], "completed")
                self.assertTrue(live_summary["stages"][0]["reused_existing_run"])
                self.assertEqual(live_summary["stages"][1]["status"], "running")
                self.assertFalse(report_path.exists())

                stage_output_root = Path(command[command.index("--output-root") + 1])
                run_dir = stage_output_root / "final-run"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "FINAL_IOTA": 0.205,
                            "TARGET_IOTA": 0.21,
                            "FINAL_NON_QS": 0.03,
                            "FINAL_G": 4.5,
                            "FIELD_ERROR": 2.5e-4,
                            "OPTIMIZER_SUCCESS": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (run_dir / "surf_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return subprocess.CompletedProcess(command, 0)

            stages = [
                module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
                module.ContinuationStage("final", 8, 6, 255, 64, 3),
            ]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    module.main(
                        [
                            "--resume-run-root",
                            str(run_root),
                            "--strict-validation",
                        ]
                    )

            final_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            final_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertNotIn("stale_marker", final_summary)
        self.assertNotIn("stale_marker", final_report)
        self.assertTrue(final_report["passed"])

    def test_main_stops_promotion_when_nonfinal_stage_contract_fails(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"
            seen_commands: list[list[str]] = []

            def fake_run(command, check):
                self.assertTrue(check)
                if self._handle_compile_seed_spec_command(command):
                    return subprocess.CompletedProcess(command, 0)
                seen_commands.append(command)
                stage_output_root = Path(command[command.index("--output-root") + 1])
                run_dir = stage_output_root / "coarse-run"
                run_dir.mkdir(parents=True, exist_ok=True)
                (run_dir / "results.json").write_text(
                    json.dumps(
                        {
                            "FINAL_IOTA": 0.18,
                            "TARGET_IOTA": 0.21,
                            "FINAL_G": 4.0,
                            "FIELD_ERROR": 5e-4,
                            "OPTIMIZER_SUCCESS": False,
                            "iterations": 0,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "TERMINATION_MESSAGE": (
                                "Optimization failed with non-finite objective or gradient."
                            ),
                        }
                    ),
                    encoding="utf-8",
                )
                (run_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
                return subprocess.CompletedProcess(command, 0)

            stages = [
                module.ContinuationStage("coarse", 2, 2, 31, 16, 1),
                module.ContinuationStage("final", 8, 6, 255, 64, 3),
            ]
            with patch.object(
                module,
                "build_default_continuation_stages",
                return_value=stages,
            ):
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    with self.assertRaises(SystemExit) as exc_info:
                        module.main(
                            [
                                "--output-root",
                                str(output_root),
                                "--run-id",
                                "invalid-coarse",
                            ]
                        )

            self.assertEqual(exc_info.exception.code, 1)
            run_root = output_root / "continuation-invalid-coarse"
            summary = json.loads(
                (run_root / "continuation_summary.json").read_text(encoding="utf-8")
            )
            report = json.loads(
                (run_root / "continuation_validation.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(len(seen_commands), 1)
        self.assertEqual(summary["stages"][0]["status"], "completed")
        self.assertEqual(summary["stages"][0]["failure_kind"], "stage_contract_failed")
        self.assertIn(
            "non-final stage recorded no accepted optimizer progress",
            summary["stages"][0]["failure_message"],
        )
        self.assertFalse(report["passed"])
        self.assertTrue(
            any(
                "coarse: " in failure
                and (
                    "accepted optimizer progress" in failure
                    or "optimizer ended in an invalid state" in failure
                )
                for failure in report["failures"]
            )
        )

    def test_main_writes_summary_and_report_for_failed_stage_subprocess(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir) / "out"

            def fail_run(*_args, **_kwargs):
                raise module.subprocess.CalledProcessError(
                    7,
                    ["/usr/bin/python3", "single_stage_banana_example.py"],
                )

            with patch.object(module.subprocess, "run", side_effect=fail_run):
                with self.assertRaises(SystemExit) as exc_info:
                    module.main(
                        [
                            "--output-root",
                            str(output_root),
                            "--run-id",
                            "failed-stage",
                            "--mpol",
                            "2",
                            "--ntor",
                            "2",
                            "--nphi",
                            "31",
                            "--ntheta",
                            "16",
                            "--maxiter",
                            "1",
                        ]
                    )
            self.assertEqual(exc_info.exception.code, 7)
            run_root = output_root / "continuation-failed-stage"
            summary_path = run_root / "continuation_summary.json"
            report_path = run_root / "continuation_validation.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(report_path.exists())
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["stages"][0]["status"], "subprocess_failed")
            self.assertEqual(summary["stages"][0]["subprocess_returncode"], 7)
            self.assertFalse(report["passed"])
            self.assertEqual(report["stage_reports"][0]["status"], "subprocess_failed")


if __name__ == "__main__":
    unittest.main()
