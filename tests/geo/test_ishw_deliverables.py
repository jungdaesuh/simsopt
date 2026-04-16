import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
IOTA_SWEEP_PATH = EXAMPLE_ROOT / "run_single_stage_iota_target_sweep.py"
BANANA_SCAN_PATH = EXAMPLE_ROOT / "run_banana_current_scan.py"
PLOT_PATH = EXAMPLE_ROOT / "plot_ishw_tradeoffs.py"
WORKFLOW_COMMON_PATH = EXAMPLE_ROOT / "workflow_runner_common.py"
STAGE2_ENTRYPOINT_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_iota_sweep_module():
    return load_module(IOTA_SWEEP_PATH, "run_single_stage_iota_target_sweep")


def load_banana_scan_module():
    return load_module(BANANA_SCAN_PATH, "run_banana_current_scan")


def load_plot_module():
    return load_module(PLOT_PATH, "plot_ishw_tradeoffs")


def load_workflow_common_module():
    return load_module(WORKFLOW_COMMON_PATH, "workflow_runner_common")


def load_stage2_module():
    return load_module(STAGE2_ENTRYPOINT_PATH, "banana_coil_solver")


class IotaTargetSweepTests(unittest.TestCase):
    def test_dry_run_writes_summary_and_csv(self):
        module = load_iota_sweep_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            summary_csv_path = tmpdir_path / "summary.csv"
            stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "stage2" / "results.json"

            with patch.object(
                module.goal_mode_runner,
                "maybe_load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path,
                    stage2_results_path,
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "BANANA_CURRENT_A": 12000.0,
                    },
                ),
            ), patch.object(
                module.goal_mode_runner,
                "run_goal_mode_case",
                side_effect=[
                    {"command": ["python", "single_stage.py", "--iota-target", "0.15"]},
                    {"command": ["python", "single_stage.py", "--iota-target", "0.20"]},
                ],
            ):
                result = module.main(
                    [
                        "--dry-run",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(tmpdir_path / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--summary-csv",
                        str(summary_csv_path),
                        "--iota-targets",
                        "0.15,0.20",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["iota_targets"], [0.15, 0.2])
            self.assertEqual(len(summary["cases"]), 2)
            self.assertEqual(summary["cases"][0]["status"], "dry_run")
            self.assertIn("--iota-target", summary["cases"][0]["command"])
            csv_text = summary_csv_path.read_text(encoding="utf-8")
            self.assertIn("case_id", csv_text)
            self.assertIn("iota_0p15", csv_text)


class BananaCurrentScanTests(unittest.TestCase):
    def test_dry_run_scales_optimized_current_and_writes_csv(self):
        module = load_banana_scan_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            summary_csv_path = tmpdir_path / "summary.csv"
            stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "stage2" / "results.json"

            with patch.object(
                module.goal_mode_runner,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path,
                    stage2_results_path,
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "BANANA_CURRENT_A": 16000.0,
                        "init_only": False,
                    },
                ),
            ), patch.object(
                module.goal_mode_runner,
                "run_goal_mode_case",
                side_effect=[
                    {"command": ["python", "single_stage.py"]},
                    {"command": ["python", "single_stage.py"]},
                    {"command": ["python", "single_stage.py"]},
                ],
            ):
                result = module.main(
                    [
                        "--dry-run",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(tmpdir_path / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--summary-csv",
                        str(summary_csv_path),
                        "--banana-current-scales",
                        "0,0.5,1.0",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [case["banana_current_a"] for case in summary["cases"]],
                [0.0, 8000.0, 16000.0],
            )
            self.assertTrue(
                all(case["poincare_status"] == "dry_run" for case in summary["cases"])
            )
            self.assertIn("banana_current_scale", summary_csv_path.read_text(encoding="utf-8"))

    def test_failed_boozer_case_can_still_report_poincare_only_fallback(self):
        module = load_banana_scan_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "stage2" / "results.json"
            variant_bs_path = tmpdir_path / "variant" / "biot_savart_opt.json"
            variant_results_path = tmpdir_path / "variant" / "results.json"
            fallback_root = tmpdir_path / "fallback"
            poincare_metrics_path = fallback_root / "PoincareMetrics_init.json"

            with patch.object(
                module.goal_mode_runner,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path,
                    stage2_results_path,
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "BANANA_CURRENT_A": 16000.0,
                        "init_only": False,
                    },
                ),
            ), patch.object(
                module,
                "_materialize_stage2_seed_variant",
                return_value=(variant_bs_path, variant_results_path),
            ), patch.object(
                module.goal_mode_runner,
                "run_goal_mode_case",
                side_effect=subprocess.CalledProcessError(1, ["python"]),
            ), patch.object(
                module,
                "_materialize_poincare_fallback_inputs",
                return_value=fallback_root,
            ), patch.object(
                module,
                "run_poincare_artifact",
                return_value=["python", "poincare_surfaces.py"],
            ), patch.object(
                module,
                "_load_poincare_metrics",
                return_value=(
                    poincare_metrics_path,
                    {"validation_status": "ok"},
                ),
            ):
                result = module.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(tmpdir_path / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--banana-current-scales",
                        "0.0",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            case = summary["cases"][0]
            self.assertEqual(case["classification"], "poincare_only_fallback")
            self.assertEqual(case["single_stage_status"], "failed")
            self.assertEqual(case["poincare_status"], "completed")

    def test_missing_poincare_fallback_inputs_do_not_abort_scan(self):
        module = load_banana_scan_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "stage2" / "results.json"
            variant_bs_path = tmpdir_path / "variant" / "biot_savart_opt.json"
            variant_results_path = tmpdir_path / "variant" / "results.json"

            with patch.object(
                module.goal_mode_runner,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path,
                    stage2_results_path,
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "BANANA_CURRENT_A": 16000.0,
                        "TOROIDAL_FLUX": 0.24,
                        "MAJOR_RADIUS": 0.915,
                        "init_only": False,
                    },
                ),
            ), patch.object(
                module,
                "_materialize_stage2_seed_variant",
                return_value=(variant_bs_path, variant_results_path),
            ), patch.object(
                module.goal_mode_runner,
                "run_goal_mode_case",
                side_effect=subprocess.CalledProcessError(1, ["python"]),
            ):
                result = module.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(tmpdir_path / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--banana-current-scales",
                        "0.0",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            case = summary["cases"][0]
            self.assertEqual(case["classification"], "boozer_failed")
            self.assertEqual(case["single_stage_status"], "failed")
            self.assertEqual(case["poincare_status"], "failed")
            self.assertIn("poincare_fallback_setup_failed", case["error_message"])


class IshwPlotTests(unittest.TestCase):
    def test_plot_manifest_records_generated_files(self):
        module = load_plot_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            iota_summary_path = tmpdir_path / "iota_summary.json"
            banana_summary_path = tmpdir_path / "banana_summary.json"
            output_root = tmpdir_path / "plots"

            iota_summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "iota_0p15",
                                "status": "completed",
                                "iota_target": 0.15,
                                "results_summary": {
                                    "coil_length": 1.68,
                                    "max_curvature": 24.0,
                                    "nonqs_ratio": 0.011,
                                    "field_error": 0.02,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            banana_summary_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case_id": "banana_scale_1p0",
                                "banana_current_scale": 1.0,
                                "banana_current_a": 16000.0,
                                "classification": "success",
                                "results_summary": {
                                    "nonqs_ratio": 0.012,
                                    "field_error": 0.02,
                                    "final_iota": 0.16,
                                },
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = module.main(
                [
                    "--iota-sweep-summary",
                    str(iota_summary_path),
                    "--banana-current-scan-summary",
                    str(banana_summary_path),
                    "--output-root",
                    str(output_root),
                ]
            )

            self.assertEqual(result, 0)
            manifest = json.loads(
                (output_root / module.DEFAULT_MANIFEST_JSON).read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(manifest["generated_plots"]["iota_target_vs_coil_length"])
            self.assertTrue(manifest["generated_plots"]["banana_current_scale_vs_iota"])
            self.assertTrue(
                Path(
                    manifest["generated_plots"]["field_error_vs_coil_length"][0]
                ).exists()
            )


class Stage2IotaReportingTests(unittest.TestCase):
    def test_build_stage2_command_forwards_stage2_iota_hot_loop_flags(self):
        module = load_workflow_common_module()

        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir=None,
            tf_current_A=8.0e4,
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=100.0,
            banana_surf_radius=0.21,
            order=2,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            stage2_iota_mode="soft",
            stage2_iota_target=0.2,
            stage2_iota_tolerance=1.0e-2,
            stage2_iota_weight=3.0,
            stage2_iota_vol_target=0.12,
            stage2_iota_constraint_weight=-1.0,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertEqual(
            command[command.index("--stage2-iota-mode") + 1],
            "soft",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-target") + 1],
            "0.2",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-tolerance") + 1],
            "0.01",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-weight") + 1],
            "3.0",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-vol-target") + 1],
            "0.12",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-constraint-weight") + 1],
            "-1.0",
        )

    def test_stage2_artifact_config_rejects_iota_alm_without_alm_constraint_method(self):
        module = load_workflow_common_module()

        with self.assertRaisesRegex(
            ValueError,
            "stage2_iota_mode='alm' requires constraint_method='alm'",
        ):
            module.Stage2ArtifactConfig(
                plasma_surf_filename="demo.nc",
                output_root=Path("/tmp/stage2"),
                equilibria_dir=None,
                tf_current_A=8.0e4,
                major_radius=0.915,
                toroidal_flux=0.24,
                length_weight=0.0005,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=100.0,
                banana_surf_radius=0.21,
                order=2,
                constraint_method="penalty",
                alm_max_outer_iters=10,
                alm_penalty_init=1.0,
                alm_penalty_scale=10.0,
                basin_hops=0,
                basin_stepsize=0.01,
                stage2_iota_mode="alm",
                stage2_iota_target=0.2,
            )

    def test_run_stage2_alm_expected_metadata_canonicalizes_exact_constraint_weight(self):
        module = load_module(EXAMPLE_ROOT / "run_stage2_alm.py", "run_stage2_alm")

        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir=None,
            tf_current_A=8.0e4,
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=100.0,
            banana_surf_radius=0.21,
            order=2,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            stage2_iota_mode="report",
            stage2_iota_target=0.2,
            stage2_iota_constraint_weight=0.0,
        )

        expected_metadata = module._expected_stage2_artifact_metadata(config)

        self.assertIsNone(expected_metadata["STAGE2_IOTA_CONSTRAINT_WEIGHT"])

    def test_run_stage2_alm_rejects_enabled_iota_mode_without_target_before_launch(self):
        module = load_module(EXAMPLE_ROOT / "run_stage2_alm.py", "run_stage2_alm")

        with self.assertRaisesRegex(
            ValueError,
            "--stage2-iota-target is required when --stage2-iota-mode is enabled.",
        ):
            module.main(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--profile",
                    "standard_80ka",
                    "--stage2-iota-mode",
                    "report",
                ]
            )

    def test_stage2_iota_report_payload_reuses_bootability_schema_without_recovery_fields(self):
        module = load_stage2_module()

        args = SimpleNamespace(
            stage2_iota_mode="report",
            stage2_iota_target=0.2,
            stage2_iota_tolerance=5.0e-3,
            equilibria_dir="/tmp/equilibria",
            equilibrium_path=None,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
            stage2_iota_vol_target=0.1,
            stage2_iota_constraint_weight=1.0,
            plasma_surf_filename="demo.nc",
        )

        with patch.object(
            module,
            "probe_stage2_seed_bootability",
            return_value={
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": True,
                "BOOTABILITY_REASON": "ok",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.2,
                "BOOTABILITY_SOLVED_IOTA": 0.201,
                "BOOTABILITY_SELF_INTERSECTING": False,
            },
        ):
            payload = module.build_stage2_iota_report_payload(
                args=args,
                stage2_bs_artifact_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_payload={},
            )

        self.assertTrue(payload["STAGE2_ROOT_FIX_ENABLED"])
        self.assertEqual(payload["STAGE2_IOTA_MODE"], "report")
        self.assertTrue(payload["BOOZER_BOOTABLE"])
        self.assertTrue(payload["IOTA_FEASIBLE"])
        self.assertNotIn("RECOVERY_ATTEMPTED", payload)
        self.assertEqual(
            payload["BOOTABILITY_STAGE2_BS_PATH"],
            "/tmp/stage2/biot_savart_opt.json",
        )
        self.assertEqual(
            payload["BOOTABILITY_STAGE2_RESULTS_PATH"],
            "/tmp/stage2/results.json",
        )
        self.assertIsNotNone(payload["STAGE2_IOTA_PROBE_SECONDS"])

    def test_stage2_iota_report_payload_maps_nonpositive_constraint_weight_to_exact_mode(self):
        module = load_stage2_module()

        args = SimpleNamespace(
            stage2_iota_mode="report",
            stage2_iota_target=0.2,
            stage2_iota_tolerance=5.0e-3,
            equilibria_dir="/tmp/equilibria",
            equilibrium_path=None,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
            stage2_iota_vol_target=0.1,
            stage2_iota_constraint_weight=0.0,
            plasma_surf_filename="demo.nc",
        )

        with patch.object(
            module,
            "probe_stage2_seed_bootability",
            return_value={
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": True,
                "BOOTABILITY_REASON": "ok",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.2,
                "BOOTABILITY_SOLVED_IOTA": 0.2,
                "BOOTABILITY_SELF_INTERSECTING": False,
            },
        ) as probe_mock:
            payload = module.build_stage2_iota_report_payload(
                args=args,
                stage2_bs_artifact_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_payload={},
            )

        self.assertIsNone(probe_mock.call_args.kwargs["constraint_weight"])
        self.assertIsNone(payload["STAGE2_IOTA_CONSTRAINT_WEIGHT"])


if __name__ == "__main__":
    unittest.main()
