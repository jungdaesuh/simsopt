import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
DONOR_REPAIR_PATH = EXAMPLE_ROOT / "run_single_stage_donor_repair.py"
DECISION_GATE_PATH = EXAMPLE_ROOT / "run_stage2_iota_decision_gate.py"


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_donor_repair_module():
    return load_module(DONOR_REPAIR_PATH, "run_single_stage_donor_repair")


def load_decision_gate_module():
    return load_module(DECISION_GATE_PATH, "run_stage2_iota_decision_gate")


class DonorRepairWrapperTests(unittest.TestCase):
    def test_dry_run_writes_summary_and_case_commands(self):
        module = load_donor_repair_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"
            summary_csv_path = root / "summary.csv"
            stage2_bs_path = root / "stage2" / "biot_savart_opt.json"
            stage2_results_path = root / "stage2" / "results.json"

            with patch.object(
                module.unified_runner,
                "resolve_stage2_input",
                return_value={
                    "source": "generated_artifact",
                    "stage2_bs_path": stage2_bs_path,
                    "stage2_results_path": stage2_results_path,
                    "stage2_results": None,
                    "artifact_reused": False,
                    "command": ["python", "run_stage2_alm.py"],
                    "config_source": "profile:standard_80ka",
                },
            ), patch.object(
                module.unified_runner,
                "build_recovery_command",
                return_value=["python", "run_single_stage_thresholded_physics_alm.py"],
            ):
                result = module.main(
                    [
                        "--dry-run",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-profile",
                        "standard_80ka",
                        "--output-root",
                        str(root / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--summary-csv",
                        str(summary_csv_path),
                        "--iota-targets",
                        "0.18,0.2",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["best_case_id"], None)
            self.assertEqual([case["status"] for case in summary["cases"]], ["dry_run", "dry_run"])
            self.assertIn(
                "run_single_stage_thresholded_physics_alm.py",
                summary["cases"][0]["recovery_command"],
            )
            self.assertIn("case_id", summary_csv_path.read_text(encoding="utf-8"))

    def test_best_donor_manifest_prefers_bootable_case(self):
        module = load_donor_repair_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path = root / "stage2" / "biot_savart_opt.json"
            stage2_results_path = root / "stage2" / "results.json"
            summary_path = root / "summary.json"
            best_donor_path = root / "best.json"

            probe_failed = {
                "BOOZER_BOOTABLE": False,
                "IOTA_FEASIBLE": False,
                "BOOTABILITY_REASON": "self_intersection",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.18,
                "BOOTABILITY_SOLVED_IOTA": 0.01,
                "BOOTABILITY_ABS_IOTA_ERROR": 0.17,
            }
            probe_bootable = {
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": True,
                "BOOTABILITY_REASON": "ok",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.20,
                "BOOTABILITY_SOLVED_IOTA": 0.2005,
                "BOOTABILITY_ABS_IOTA_ERROR": 5.0e-4,
            }
            recovered_probe = {
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": True,
                "BOOTABILITY_REASON": "ok",
                "BOOTABILITY_STAGE": "recovery",
                "BOOTABILITY_TARGET_IOTA": 0.18,
                "BOOTABILITY_SOLVED_IOTA": 0.1801,
                "BOOTABILITY_ABS_IOTA_ERROR": 1.0e-4,
            }

            with patch.object(
                module.unified_runner,
                "resolve_stage2_input",
                return_value={
                    "source": "existing_artifact",
                    "stage2_bs_path": stage2_bs_path,
                    "stage2_results_path": stage2_results_path,
                    "stage2_results": {"PLASMA_SURF_FILENAME": "demo.nc"},
                    "artifact_reused": True,
                    "command": None,
                    "config_source": None,
                },
            ), patch.object(
                module.unified_runner,
                "build_probe_status",
                side_effect=[probe_failed, probe_bootable],
            ), patch.object(
                module.unified_runner,
                "run_recovery_stage",
                return_value={
                    "status": "completed",
                    "results_path": str(root / "recovery" / "results.json"),
                    "recovered_bs_path": str(root / "recovery" / "biot_savart_opt.json"),
                    "recovery_probe": recovered_probe,
                    "recovery_succeeded": True,
                    "recovery_iters": 9,
                    "recovery_termination_reason": "bootable",
                },
            ):
                result = module.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(root / "outputs"),
                        "--summary-json",
                        str(summary_path),
                        "--best-donor-json",
                        str(best_donor_path),
                        "--iota-targets",
                        "0.18,0.20",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["best_case_id"], "repair_iota_0p18")
            best_donor = json.loads(best_donor_path.read_text(encoding="utf-8"))
            self.assertTrue(best_donor["handoff_bootable"])
            self.assertEqual(
                best_donor["selected_seed_source"],
                module.unified_runner.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )


class Stage2DecisionGateTests(unittest.TestCase):
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
                    {"mode": "report", "status": "dry_run", "command": ["python", "report"]},
                    {"mode": "soft", "status": "dry_run", "command": ["python", "soft"]},
                    {"mode": "alm", "status": "dry_run", "command": ["python", "alm"]},
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
            self.assertEqual(summary["benchmark_modes"], ["report", "soft", "alm"])
            self.assertEqual(
                summary["recommendation"]["recommendation"],
                "insufficient_runtime_data",
            )
            self.assertIn("mode", summary_csv_path.read_text(encoding="utf-8"))

    def test_decision_gate_prefers_donor_repair_when_soft_signal_is_weak(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            donor_repair_summary = root / "donor_repair_summary.json"
            donor_repair_summary.write_text(
                json.dumps(
                    {
                        "best_case": {
                            "case_id": "repair_iota_0p2",
                            "handoff_bootable": True,
                            "selected_seed_source": "recovered_stage2_donor",
                        }
                    }
                ),
                encoding="utf-8",
            )
            summary_path = root / "summary.json"

            with patch.object(
                module,
                "run_mode_case",
                side_effect=[
                    {
                        "mode": "report",
                        "status": "completed",
                        "run_wallclock_seconds": 10.0,
                        "stage2_iota_abs_error": 0.02,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                    {
                        "mode": "soft",
                        "status": "completed",
                        "run_wallclock_seconds": 28.0,
                        "stage2_iota_abs_error": 0.0195,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                    {
                        "mode": "alm",
                        "status": "completed",
                        "run_wallclock_seconds": 40.0,
                        "stage2_iota_abs_error": 0.001,
                        "hardware_constraints_ok": False,
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
                        "--donor-repair-summary",
                        str(donor_repair_summary),
                        "--summary-json",
                        str(summary_path),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(
                summary["recommendation"]["recommendation"],
                "prefer_unified_runner_donor_repair",
            )

    def test_decision_gate_can_recommend_hard_alm(self):
        module = load_decision_gate_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary_path = root / "summary.json"

            with patch.object(
                module,
                "run_mode_case",
                side_effect=[
                    {
                        "mode": "report",
                        "status": "completed",
                        "run_wallclock_seconds": 10.0,
                        "stage2_iota_abs_error": 0.02,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": False,
                        "iota_feasible": False,
                    },
                    {
                        "mode": "soft",
                        "status": "completed",
                        "run_wallclock_seconds": 14.0,
                        "stage2_iota_abs_error": 0.005,
                        "hardware_constraints_ok": True,
                        "boozer_bootable": True,
                        "iota_feasible": True,
                    },
                    {
                        "mode": "alm",
                        "status": "completed",
                        "run_wallclock_seconds": 18.0,
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
                "proceed_to_hard_stage2_alm_iota",
            )


if __name__ == "__main__":
    unittest.main()
