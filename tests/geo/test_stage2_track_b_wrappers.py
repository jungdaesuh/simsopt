import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
DECISION_GATE_PATH = EXAMPLE_ROOT / "run_stage2_iota_decision_gate.py"
UNIFIED_RUNNER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_decision_gate_module():
    return load_module(DECISION_GATE_PATH, "run_stage2_iota_decision_gate")


def load_unified_runner_module():
    return load_module(UNIFIED_RUNNER_PATH, "run_stage2_to_single_stage")


class UnifiedRunnerStage2InputTests(unittest.TestCase):
    def test_generated_stage2_input_threads_constraint_metadata(self):
        module = load_unified_runner_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "stage2" / "biot_savart_opt.json"
            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                python_executable="python",
                dry_run=False,
                stage2_output_root=None,
                stage2_profile="standard_80ka",
                stage2_spec_json=None,
                equilibria_dir=None,
                stage2_timeout_seconds=0.0,
                stage2_cc_threshold=None,
                stage2_curvature_threshold=None,
                stage2_order=None,
                stage2_tf_current_A=None,
                stage2_toroidal_flux=None,
                stage2_basin_seed=None,
                iota_target=0.2,
                vol_target=0.10,
                alm_fix_signal_mismatch_guard=False,
            )

            with patch.object(
                module,
                "ensure_stage2_artifact_result",
                return_value=SimpleNamespace(
                    artifact_path=artifact_path,
                    artifact_reused=True,
                ),
            ) as ensure_mock, patch.object(
                module.stage2_alm_runner,
                "load_validated_stage2_artifact",
                return_value=(
                    artifact_path.with_name("results.json"),
                    {"OPTIMIZER_SUCCESS": True},
                ),
            ) as load_mock:
                stage2_input = module.resolve_stage2_input(
                    args,
                    output_root=root / "outputs",
                )

            ensure_kwargs = ensure_mock.call_args.kwargs
            self.assertEqual(
                ensure_kwargs["constraint_profile_label"],
                "profile:standard_80ka",
            )
            self.assertIn("constraint_metadata", load_mock.call_args.kwargs)
            self.assertIn(
                "--constraint-profile-label",
                stage2_input["command"],
            )
            self.assertTrue(stage2_input["artifact_reused"])


class Stage2DecisionGateTests(unittest.TestCase):
    def test_run_mode_case_threads_stage2_constraint_metadata(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact_path = root / "stage2" / "biot_savart_opt.json"
            args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--profile",
                    "standard_80ka",
                    "--stage2-iota-target",
                    "0.2",
                    "--output-root",
                    str(root / "outputs"),
                ]
            )

            with patch.object(
                module.stage2_alm_runner,
                "ensure_stage2_artifact_result",
                return_value=SimpleNamespace(
                    artifact_path=artifact_path,
                    artifact_reused=True,
                ),
            ) as ensure_mock, patch.object(
                module.stage2_alm_runner,
                "load_validated_stage2_artifact",
                return_value=(
                    artifact_path.with_name("results.json"),
                    {
                        "OPTIMIZER_SUCCESS": True,
                        "HARDWARE_CONSTRAINTS_OK": True,
                        "BOOZER_BOOTABLE": True,
                        "IOTA_NEAR_TARGET": True,
                        "IOTA_FEASIBLE": True,
                    },
                ),
            ) as load_mock:
                payload = module.run_mode_case(
                    args,
                    mode="report",
                    output_root=root / "case",
                )

            ensure_kwargs = ensure_mock.call_args.kwargs
            self.assertEqual(
                ensure_kwargs["constraint_profile_label"],
                "profile:standard_80ka",
            )
            self.assertIn("constraint_metadata", load_mock.call_args.kwargs)
            self.assertIn(
                "--constraint-profile-label",
                payload["command"],
            )
            self.assertTrue(payload["artifact_reused"])

    def test_run_mode_case_report_uses_alm_stage2_mode(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            args = module.parse_args(
                [
                    "--dry-run",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--profile",
                    "standard_80ka",
                    "--stage2-iota-target",
                    "0.2",
                    "--output-root",
                    str(root / "outputs"),
                    "--benchmark-modes",
                    "off,report",
                ]
            )

            payload = module.run_mode_case(
                args,
                mode="report",
                output_root=root / "report",
            )
            mode_args = module.build_stage2_mode_args(
                args,
                mode="report",
                output_root=root / "report",
            )
            resolved_spec, _ = module.stage2_alm_runner.resolve_stage2_spec_payload(
                mode_args
            )
            config = module.stage2_alm_runner.build_stage2_alm_config(
                mode_args,
                resolved_spec=resolved_spec,
            )

        command = payload["command"]
        self.assertEqual(
            command[command.index("--constraint-method") + 1],
            "alm",
        )
        self.assertEqual(
            command[command.index("--stage2-iota-mode") + 1],
            "report",
        )
        self.assertNotIn("--stage2-iota-weight", command)
        self.assertEqual(
            payload["resolved_stage2_config"]["constraint_method"],
            "alm",
        )
        expected_metadata = module.stage2_alm_runner._expected_stage2_artifact_metadata(
            config
        )
        self.assertEqual(expected_metadata["ALM_PENALTY_INIT"], config.alm_penalty_init)

    def test_dry_run_summary_uses_requested_modes(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_csv_path = root / "summary.csv"

            with patch.object(
                module,
                "run_mode_case",
                side_effect=[
                    {"mode": "off", "status": "dry_run", "command": ["python", "off"]},
                    {"mode": "report", "status": "dry_run", "command": ["python", "report"]},
                ],
            ):
                result = module.main(
                    [
                        "--dry-run",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--profile",
                        "standard_80ka",
                        "--stage2-iota-target",
                        "0.2",
                        "--output-root",
                        str(root / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--summary-csv",
                        str(summary_csv_path),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["benchmark_modes"], ["off", "report"])
            self.assertEqual(
                summary["recommendation"]["recommendation"],
                "insufficient_runtime_data",
            )
            self.assertIn("mode", summary_csv_path.read_text(encoding="utf-8"))

    def test_decision_gate_nonbootable_report_stays_on_supported_seam(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"

            with patch.object(
                module,
                "run_mode_case",
                side_effect=[
                    {
                        "mode": "off",
                        "status": "completed",
                        "run_wallclock_seconds": 8.0,
                        "stage2_iota_abs_error": None,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                    {
                        "mode": "report",
                        "status": "completed",
                        "run_wallclock_seconds": 10.0,
                        "stage2_iota_abs_error": 0.02,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                ],
            ):
                result = module.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--profile",
                        "standard_80ka",
                        "--stage2-iota-target",
                        "0.2",
                        "--summary-json",
                        str(summary_path),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["recommendation"]["recommendation"],
                "stop_at_unified_runner_or_reporting_probe",
            )
            for payload in (summary, summary["recommendation"]):
                self.assertTrue(
                    all(not key.startswith("donor_") for key in payload)
                )

    def test_decision_gate_stops_at_reporting_probe(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"

            with patch.object(
                module,
                "run_mode_case",
                side_effect=[
                    {
                        "mode": "off",
                        "status": "completed",
                        "run_wallclock_seconds": 8.0,
                        "stage2_iota_abs_error": None,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                    {
                        "mode": "report",
                        "status": "completed",
                        "run_wallclock_seconds": 10.0,
                        "stage2_iota_abs_error": 5.0e-4,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": True,
                        "iota_feasible": True,
                    },
                ],
            ):
                result = module.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--profile",
                        "standard_80ka",
                        "--stage2-iota-target",
                        "0.2",
                        "--summary-json",
                        str(summary_path),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["recommendation"]["recommendation"],
                "stop_at_unified_runner_or_reporting_probe",
            )


if __name__ == "__main__":
    unittest.main()
