import importlib
import importlib.util
import json
import hashlib
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
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

IOTA_SWEEP_PATH = EXAMPLE_ROOT / "run_single_stage_iota_target_sweep.py"
BANANA_SCAN_PATH = EXAMPLE_ROOT / "run_banana_current_scan.py"
PERTURBED_SEED_PATH = EXAMPLE_ROOT / "make_perturbed_banana_seed.py"
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


def load_perturbed_seed_module():
    return load_module(PERTURBED_SEED_PATH, "make_perturbed_banana_seed")


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def load_plot_module():
    return load_module(PLOT_PATH, "plot_ishw_tradeoffs")


def load_workflow_common_module():
    return load_module(WORKFLOW_COMMON_PATH, "workflow_runner_common")


def load_stage2_module():
    return load_module(STAGE2_ENTRYPOINT_PATH, "banana_coil_solver")


def write_stage2_results_with_digest(stage2_bs_path: Path, stage2_results: dict) -> Path:
    results_path = stage2_bs_path.with_name("results.json")
    payload = dict(stage2_results)
    payload["STAGE2_BS_SHA256"] = hashlib.sha256(
        stage2_bs_path.read_bytes()
    ).hexdigest()
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    return results_path


class IotaTargetSweepTests(unittest.TestCase):
    def test_parse_args_accepts_independent_banana_current_mode(self):
        module = load_iota_sweep_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "seed.json",
                "--single-stage-banana-current-mode",
                "independent",
            ]
        )

        self.assertEqual(args.single_stage_banana_current_mode, "independent")

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
            self.assertEqual(summary["output_contract"], "dry_run_summary_only")
            self.assertFalse(summary["contains_solver_outputs"])
            self.assertNotIn("output_materialization", summary)
            self.assertEqual(summary["iota_targets"], [0.15, 0.2])
            self.assertEqual(len(summary["cases"]), 2)
            self.assertEqual(summary["cases"][0]["status"], "dry_run")
            self.assertIn("--iota-target", summary["cases"][0]["command"])
            csv_text = summary_csv_path.read_text(encoding="utf-8")
            self.assertIn("case_id", csv_text)
            self.assertIn("iota_0p15", csv_text)

    def test_build_summary_dry_run_contract_overrides_case_payload_shape(self):
        module = load_iota_sweep_module()
        args = SimpleNamespace(
            dry_run=True,
            init_only=False,
            plasma_surf_filename="demo.nc",
        )

        summary = module.build_summary(
            args,
            iota_targets=[0.15],
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            stage2_results_path=None,
            stage2_results=None,
            case_payloads=[
                {
                    "case_id": "iota_0p15",
                    "results_path": "/tmp/results.json",
                }
            ],
            summary_csv_path=Path("/tmp/summary.csv"),
        )

        self.assertEqual(summary["output_contract"], "dry_run_summary_only")
        self.assertFalse(summary["contains_solver_outputs"])

    def test_result_summary_keeps_missing_numeric_metrics_json_portable(self):
        module = load_iota_sweep_module()

        summary = module._result_summary(
            {
                "SINGLE_STAGE_GOAL_MODE": "target",
                "TARGET_IOTA": 0.15,
                "FIELD_ERROR": None,
                "BANANA_CURRENT_A": None,
                "BANANA_CURRENT_MODE": None,
                "BANANA_CURRENTS_A": None,
                "BANANA_CURRENT_MAX_ABS_A": None,
                "PLASMA_CURRENT_A": None,
                "INITIAL_IOTA": None,
            }
        )

        self.assertEqual(summary["goal_mode"], "target")
        self.assertEqual(summary["target_iota"], 0.15)
        self.assertIsNone(summary["banana_current_mode"])
        self.assertIsNone(summary["banana_currents_a"])
        for key in (
            "field_error",
            "banana_current_a",
            "banana_current_max_abs_a",
            "plasma_current_a",
            "initial_iota",
            "final_iota",
            "coil_length",
        ):
            self.assertIsNone(summary[key])


class BananaCurrentScanTests(unittest.TestCase):
    _EXPECTED_BANANA_CURRENTS_A = [0.0, 8000.0, 16000.0]
    _EXPECTED_CASE_IDS = [
        "banana_current_0A",
        "banana_current_8000A",
        "banana_current_16000A",
    ]
    _DEFAULT_DONOR_BANANA_CURRENT_A = 15828.0
    _EXPECTED_DEFAULT_BANANA_CURRENTS_A = [0.0, 3957.0, 7914.0, 11871.0, 15828.0]
    _EXPECTED_DEFAULT_CASE_IDS = [
        "banana_current_0A",
        "banana_current_3957A",
        "banana_current_7914A",
        "banana_current_11871A",
        "banana_current_15828A",
    ]

    def _assert_amp_based_summary_contract(self, summary: dict) -> None:
        self.assertEqual(
            [case["banana_current_a"] for case in summary["cases"]],
            self._EXPECTED_BANANA_CURRENTS_A,
        )
        self.assertEqual(
            [case["case_id"] for case in summary["cases"]],
            self._EXPECTED_CASE_IDS,
        )
        self.assertEqual(
            summary["requested_banana_currents_a"],
            self._EXPECTED_BANANA_CURRENTS_A,
        )

    def _assert_default_amp_summary_contract(self, summary: dict) -> None:
        self.assertEqual(
            summary["requested_banana_currents_a"],
            self._EXPECTED_DEFAULT_BANANA_CURRENTS_A,
        )
        self.assertEqual(
            [case["case_id"] for case in summary["cases"]],
            self._EXPECTED_DEFAULT_CASE_IDS,
        )

    def test_dry_run_writes_amp_based_summary_and_csv(self):
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
                        "--banana-currents-A",
                        "0,8000,16000",
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._assert_amp_based_summary_contract(summary)
            self.assertTrue(
                all(case["poincare_status"] == "dry_run" for case in summary["cases"])
            )
            self.assertIn("banana_current_a", summary_csv_path.read_text(encoding="utf-8"))

    def test_omitted_banana_currents_uses_default_amp_setpoints(self):
        module = load_banana_scan_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            summary_path = tmpdir_path / "summary.json"
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
                        "BANANA_CURRENT_A": self._DEFAULT_DONOR_BANANA_CURRENT_A,
                        "init_only": False,
                    },
                ),
            ), patch.object(
                module.goal_mode_runner,
                "run_goal_mode_case",
                side_effect=[
                    {"command": ["python", "single_stage.py"]}
                    for _ in module.DEFAULT_BANANA_CURRENT_FRACTIONS
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
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self._assert_default_amp_summary_contract(summary)

    def test_removed_scale_cli_is_rejected(self):
        module = load_banana_scan_module()

        with self.assertRaises(SystemExit) as raised:
            module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    "/tmp/seed/biot_savart_opt.json",
                    "--banana-current-scales",
                    "0,0.5,1.0",
                ]
            )

        self.assertEqual(raised.exception.code, 2)

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
                        "--banana-currents-A",
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
                        "MAJOR_RADIUS": 0.976,
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
                        "--banana-currents-A",
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

class BananaCurrentScanSummaryTests(unittest.TestCase):
    def test_summary_uses_shared_dry_run_output_contract(self):
        module = load_banana_scan_module()
        args = SimpleNamespace(dry_run=True, plasma_surf_filename="demo.nc")

        summary = module.build_summary(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={"init_only": False, "BANANA_CURRENT_A": 12000.0},
            banana_currents_a=[0.0],
            case_payloads=[
                {
                    "case_id": "current_0p0",
                    "results_path": "/tmp/results.json",
                }
            ],
            summary_csv_path=Path("/tmp/summary.csv"),
        )

        self.assertEqual(summary["output_contract"], "dry_run_summary_only")
        self.assertFalse(summary["contains_solver_outputs"])
        self.assertNotIn("output_materialization", summary)

    def test_summary_reports_solver_outputs_when_case_results_exist(self):
        module = load_banana_scan_module()
        args = SimpleNamespace(dry_run=False, plasma_surf_filename="demo.nc")

        summary = module.build_summary(
            args,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={"init_only": False, "BANANA_CURRENT_A": 12000.0},
            banana_currents_a=[0.0],
            case_payloads=[
                {
                    "case_id": "current_0p0",
                    "results_path": "/tmp/results.json",
                }
            ],
            summary_csv_path=Path("/tmp/summary.csv"),
        )

        self.assertEqual(summary["output_contract"], "materialized_scan_outputs")
        self.assertTrue(summary["contains_solver_outputs"])


class BananaCurrentChainScalingTests(unittest.TestCase):
    """Regression tests for the banana-coil current mutation used by
    run_banana_current_scan._materialize_stage2_seed_variant. The previous
    implementation called coil.current.set_value(banana_current_a), which raised
    AttributeError on ScaledCurrent (no set_value method) and would also have
    clobbered stellsym sign flips had set_value existed.
    """

    @staticmethod
    def _build_banana_partitions(
        banana_init_current_A: float,
        *,
        num_tf_coils: int = 20,
        fixed_leaf_current: bool = False,
    ):
        from simsopt._core.optimizable import load as load_optimizable
        from simsopt.field import BiotSavart, Coil, Current
        from simsopt.field.coil import ScaledCurrent, coils_via_symmetries
        from simsopt.geo import CurveXYZFourier

        handoff = importlib.import_module(
            "banana_opt.stage2_single_stage_handoff"
        )

        banana_curve = CurveXYZFourier(96, 1)
        banana_curve.set_dofs([0.9, 0.2, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
        leaf_current = Current(1.0)
        if fixed_leaf_current:
            leaf_current.fix_all()
        base_banana_current = ScaledCurrent(leaf_current, banana_init_current_A)
        banana_coils = coils_via_symmetries(
            [banana_curve],
            [base_banana_current],
            5,
            True,
        )
        tf_coils = []
        for coil_index in range(num_tf_coils):
            tf_curve = CurveXYZFourier(96, 1)
            tf_curve.set_dofs(
                [0.9 + 0.01 * coil_index, 0.18, 0.0, 0.0, 0.18, 0.0, 0.0, 0.0, 0.0]
            )
            tf_coils.append(Coil(tf_curve, Current(-8.0e4)))
        bs = BiotSavart([*tf_coils, *banana_coils])
        stage2_results = {
            "NUM_TF_COILS": num_tf_coils,
            "NUM_BANANA_COILS": 10,
            "NUM_PROXY_COILS": 0,
            "NUM_VF_COILS": 0,
            "SURFACE_VESSEL_MIN_DIST": 0.04,
        }
        return bs, stage2_results, handoff, load_optimizable

    def test_scale_banana_current_chain_preserves_stellsym_signs_under_round_trip(
        self,
    ):
        module = load_banana_scan_module()
        bs, stage2_results, handoff, load_optimizable = self._build_banana_partitions(
            11000.0
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "seed.json"
            bs.save(str(save_path))
            loaded = load_optimizable(str(save_path))
            partitions = handoff.partition_loaded_stage2_coils(
                loaded.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )
            self.assertEqual(len(partitions.banana_coils), 10)
            original_signs = [
                1.0 if coil.current.get_value() >= 0.0 else -1.0
                for coil in partitions.banana_coils
            ]

            module._scale_banana_current_chain(
                partitions.banana_coils,
                target_banana_current_a=5500.0,
            )
            scaled_values = [
                coil.current.get_value() for coil in partitions.banana_coils
            ]
            self.assertTrue(
                all(
                    abs(abs(value) - 5500.0) < 1.0e-9
                    for value in scaled_values
                )
            )
            self.assertEqual(
                [1.0 if value >= 0.0 else -1.0 for value in scaled_values],
                original_signs,
            )

            module._scale_banana_current_chain(
                partitions.banana_coils,
                target_banana_current_a=0.0,
            )
            for coil in partitions.banana_coils:
                self.assertAlmostEqual(coil.current.get_value(), 0.0, places=9)

            module._scale_banana_current_chain(
                partitions.banana_coils,
                target_banana_current_a=11000.0,
            )
            restored_values = [
                coil.current.get_value() for coil in partitions.banana_coils
            ]
            self.assertTrue(
                all(
                    abs(abs(value) - 11000.0) < 1.0e-9
                    for value in restored_values
                )
            )
            self.assertEqual(
                [1.0 if value >= 0.0 else -1.0 for value in restored_values],
                original_signs,
            )

    def test_scale_banana_current_chain_rejects_empty_partition(self):
        module = load_banana_scan_module()

        with self.assertRaisesRegex(
            ValueError,
            "at least one banana coil",
        ):
            module._scale_banana_current_chain(
                (),
                target_banana_current_a=1.0,
            )

    def test_scale_banana_current_chain_updates_fixed_leaf_current(self):
        module = load_banana_scan_module()
        bs, stage2_results, handoff, load_optimizable = self._build_banana_partitions(
            11000.0,
            num_tf_coils=1,
            fixed_leaf_current=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "seed.json"
            bs.save(str(save_path))
            loaded = load_optimizable(str(save_path))
            partitions = handoff.partition_loaded_stage2_coils(
                loaded.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=stage2_results["NUM_TF_COILS"],
            )

            module._scale_banana_current_chain(
                partitions.banana_coils,
                target_banana_current_a=5500.0,
            )

            scaled_values = [
                coil.current.get_value() for coil in partitions.banana_coils
            ]
            self.assertTrue(
                all(
                    abs(abs(value) - 5500.0) < 1.0e-9
                    for value in scaled_values
                )
            )

    def test_materialize_stage2_seed_variant_emits_matching_checksum(self):
        module = load_banana_scan_module()

        class _FakeBS:
            coils = [object()]

            @staticmethod
            def save(path):
                Path(path).write_text('{"variant": true}', encoding="utf-8")

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "load",
            return_value=_FakeBS(),
        ), patch.object(
            module,
            "partition_loaded_stage2_coils",
            return_value=SimpleNamespace(
                tf_coils=[
                    SimpleNamespace(
                        current=SimpleNamespace(get_value=lambda: -8.0e4)
                    )
                ],
                banana_coils=[object()],
            ),
        ), patch.object(
            module,
            "_scale_banana_current_chain",
        ):
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            seed_bs_path.write_text('{"seed": true}', encoding="utf-8")

            variant_bs_path, variant_results_path = module._materialize_stage2_seed_variant(
                stage2_bs_path=seed_bs_path,
                stage2_results={},
                variant_root=tmpdir_path / "variant",
                banana_current_a=5500.0,
                requested_num_tf_coils=20,
            )

            variant_results = json.loads(variant_results_path.read_text(encoding="utf-8"))
            expected_digest = module.compute_stage2_bs_sha256(variant_bs_path)

        self.assertEqual(
            variant_results["STAGE2_BS_SHA256"],
            expected_digest,
        )
        self.assertEqual(variant_results["BANANA_CURRENT_A"], 5500.0)
        self.assertEqual(variant_results["STAGE2_BS_PATH"], str(seed_bs_path))
        self.assertEqual(variant_results["TF_CURRENT_A"], -8.0e4)

    def test_materialize_stage2_seed_variant_rejects_nonuniform_signed_tf_current(self):
        module = load_banana_scan_module()

        class _FakeBS:
            coils = [object()]

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "load",
            return_value=_FakeBS(),
        ), patch.object(
            module,
            "partition_loaded_stage2_coils",
            return_value=SimpleNamespace(
                tf_coils=[
                    SimpleNamespace(current=SimpleNamespace(get_value=lambda: -8.0e4)),
                    SimpleNamespace(current=SimpleNamespace(get_value=lambda: 8.0e4)),
                ],
                banana_coils=[object()],
            ),
        ):
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            seed_bs_path.write_text('{"seed": true}', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "signed current"):
                module._materialize_stage2_seed_variant(
                    stage2_bs_path=seed_bs_path,
                    stage2_results={},
                    variant_root=tmpdir_path / "variant",
                    banana_current_a=5500.0,
                    requested_num_tf_coils=20,
                )

    def test_materialize_stage2_seed_variant_from_currents_round_trips_vector(self):
        module = load_banana_scan_module()
        handoff_module = load_handoff_module()
        bs, stage2_results, handoff, load_optimizable = self._build_banana_partitions(
            11000.0
        )
        bs.set_points(
            [
                [0.20, -0.10, 0.05],
                [0.35, 0.15, -0.20],
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))

            donor_loaded = load_optimizable(str(seed_bs_path))
            donor_partitions = handoff.partition_loaded_stage2_coils(
                donor_loaded.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )
            donor_tf_currents = [
                coil.current.get_value() for coil in donor_partitions.tf_coils
            ]
            target_banana_currents = [
                float((coil_index + 1) * 1000.0)
                if coil_index % 2 == 0
                else float(-((coil_index + 1) * 1000.0))
                for coil_index in range(len(donor_partitions.banana_coils))
            ]

            variant_bs_path, variant_results_path = (
                module.materialize_stage2_seed_variant_from_currents(
                    stage2_bs_path=seed_bs_path,
                    stage2_results=stage2_results,
                    variant_root=tmpdir_path / "variant",
                    banana_currents_a=target_banana_currents,
                    requested_num_tf_coils=20,
                    extra_results_updates={"CUSTOM_PROVENANCE": "test-vector"},
                )
            )

            variant_loaded = load_optimizable(str(variant_bs_path))
            variant_partitions = handoff.partition_loaded_stage2_coils(
                variant_loaded.coils,
                stage2_results=stage2_results,
                requested_num_tf_coils=20,
            )
            variant_results = json.loads(
                variant_results_path.read_text(encoding="utf-8")
            )
            expected_digest = module.compute_stage2_bs_sha256(variant_bs_path)
            variant_points = variant_loaded.get_points_cart_ref().tolist()
            donor_points = donor_loaded.get_points_cart_ref().tolist()

        self.assertEqual(
            [coil.current.get_value() for coil in variant_partitions.banana_coils],
            target_banana_currents,
        )
        self.assertEqual(
            [coil.current.get_value() for coil in variant_partitions.tf_coils],
            donor_tf_currents,
        )
        self.assertEqual(
            variant_results["BANANA_CURRENT_MODE"],
            "independent",
        )
        self.assertEqual(
            variant_results["BANANA_CURRENTS_A"],
            target_banana_currents,
        )
        self.assertEqual(
            variant_results["BANANA_CURRENT_A"],
            max(abs(value) for value in target_banana_currents),
        )
        self.assertEqual(variant_results["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(variant_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)
        self.assertEqual(
            variant_results["DONOR_BANANA_CURRENTS_A"],
            [
                coil.current.get_value()
                for coil in donor_partitions.banana_coils
            ],
        )
        self.assertEqual(variant_results["DONOR_BANANA_CURRENT_A"], 11000.0)
        self.assertEqual(variant_results["CUSTOM_PROVENANCE"], "test-vector")
        self.assertEqual(variant_results["STAGE2_BS_SHA256"], expected_digest)
        self.assertEqual(variant_points, donor_points)
        handoff_module.validate_stage2_seed_contract(variant_results)

    def test_materialize_stage2_seed_variant_from_currents_rejects_length_mismatch(self):
        module = load_banana_scan_module()
        bs, stage2_results, _, _ = self._build_banana_partitions(11000.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))

            with self.assertRaisesRegex(
                ValueError,
                "vector length mismatch",
            ):
                module.materialize_stage2_seed_variant_from_currents(
                    stage2_bs_path=seed_bs_path,
                    stage2_results=stage2_results,
                    variant_root=tmpdir_path / "variant",
                    banana_currents_a=[1.0, -1.0],
                    requested_num_tf_coils=20,
                )


class PerturbedBananaSeedTests(unittest.TestCase):
    def test_main_materializes_explicit_independent_seed_bundle(self):
        banana_scan_module = load_banana_scan_module()
        handoff_module = load_handoff_module()
        module = load_perturbed_seed_module()
        bs, stage2_results, _, _ = (
            BananaCurrentChainScalingTests._build_banana_partitions(11000.0)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))
            seed_results_path = write_stage2_results_with_digest(
                seed_bs_path,
                stage2_results,
            )
            output_root = tmpdir_path / "variant"
            target_banana_currents = [
                9000.0 if index % 2 == 0 else -9500.0
                for index in range(stage2_results["NUM_BANANA_COILS"])
            ]

            result = module.main(
                [
                    "--stage2-bs-path",
                    str(seed_bs_path),
                    "--output-root",
                    str(output_root),
                    "--banana-currents-A",
                    ",".join(str(value) for value in target_banana_currents),
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(
                (output_root / module.DEFAULT_SUMMARY_JSON).read_text(
                    encoding="utf-8"
                )
            )
            variant_results = json.loads(
                (output_root / "results.json").read_text(encoding="utf-8")
            )
            expected_digest = banana_scan_module.compute_stage2_bs_sha256(
                output_root / "biot_savart_opt.json"
            )

        self.assertEqual(summary["experiment_family"], "perturbed_banana_seed")
        self.assertEqual(summary["banana_current_mode"], "independent")
        self.assertEqual(
            summary["perturbed_banana_currents_a"],
            target_banana_currents,
        )
        self.assertEqual(
            summary["recommended_single_stage_flags"],
            [
                "--stage2-bs-path",
                str((output_root / "biot_savart_opt.json").resolve()),
                "--single-stage-banana-current-mode",
                "independent",
                "--single-stage-banana-current-coordinate-scaling",
                "seed-relative",
                "--banana-current-diagnostics",
            ],
        )
        self.assertEqual(
            variant_results["BANANA_CURRENTS_A"],
            target_banana_currents,
        )
        self.assertEqual(
            variant_results["DONOR_STAGE2_BS_PATH"],
            str(seed_bs_path.resolve()),
        )
        self.assertEqual(
            variant_results["DONOR_STAGE2_RESULTS_PATH"],
            str(seed_results_path.resolve()),
        )
        self.assertEqual(
            variant_results["PERTURBATION_MODE"],
            "explicit_vector",
        )
        self.assertEqual(
            variant_results["DONOR_BANANA_CURRENTS_A"],
            [11000.0 if index % 2 == 0 else -11000.0 for index in range(10)],
        )
        self.assertEqual(variant_results["DONOR_BANANA_CURRENT_A"], 11000.0)
        self.assertEqual(variant_results["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(variant_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)
        self.assertEqual(
            variant_results["BANANA_CURRENT_A"],
            9500.0,
        )
        self.assertEqual(variant_results["STAGE2_BS_SHA256"], expected_digest)
        handoff_module.validate_stage2_seed_contract(variant_results)

    def test_main_materializes_target_envelope_seed_bundle(self):
        banana_scan_module = load_banana_scan_module()
        module = load_perturbed_seed_module()
        bs, stage2_results, _, _ = (
            BananaCurrentChainScalingTests._build_banana_partitions(11000.0)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))
            write_stage2_results_with_digest(seed_bs_path, stage2_results)
            output_root = tmpdir_path / "variant"

            result = module.main(
                [
                    "--stage2-bs-path",
                    str(seed_bs_path),
                    "--output-root",
                    str(output_root),
                    "--relative-perturbation-max",
                    "0.0",
                    "--target-banana-current-max-abs-A",
                    "9500",
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(
                (output_root / module.DEFAULT_SUMMARY_JSON).read_text(
                    encoding="utf-8"
                )
            )
            variant_results = json.loads(
                (output_root / "results.json").read_text(encoding="utf-8")
            )
            expected_digest = banana_scan_module.compute_stage2_bs_sha256(
                output_root / "biot_savart_opt.json"
            )

        target_banana_currents = [
            9500.0 if index % 2 == 0 else -9500.0
            for index in range(stage2_results["NUM_BANANA_COILS"])
        ]
        self.assertEqual(summary["perturbed_banana_currents_a"], target_banana_currents)
        self.assertEqual(summary["target_banana_current_max_abs_a"], 9500.0)
        self.assertEqual(
            summary["banana_current_envelope_scale_factor"],
            9500.0 / 11000.0,
        )
        self.assertEqual(variant_results["BANANA_CURRENTS_A"], target_banana_currents)
        self.assertEqual(variant_results["BANANA_CURRENT_A"], 9500.0)
        self.assertEqual(
            variant_results["TARGET_BANANA_CURRENT_MAX_ABS_A"],
            9500.0,
        )
        self.assertEqual(
            variant_results["BANANA_CURRENT_ENVELOPE_SCALE_FACTOR"],
            9500.0 / 11000.0,
        )
        self.assertEqual(variant_results["STAGE2_BS_SHA256"], expected_digest)

    def test_main_rejects_negative_target_envelope(self):
        module = load_perturbed_seed_module()
        bs, stage2_results, _, _ = (
            BananaCurrentChainScalingTests._build_banana_partitions(11000.0)
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))
            write_stage2_results_with_digest(seed_bs_path, stage2_results)

            with self.assertRaisesRegex(
                ValueError,
                "--target-banana-current-max-abs-A must be positive",
            ):
                module.main(
                    [
                        "--stage2-bs-path",
                        str(seed_bs_path),
                        "--output-root",
                        str(tmpdir_path / "variant"),
                        "--relative-perturbation-max",
                        "0.0",
                        "--target-banana-current-max-abs-A",
                        "-9500",
                    ]
                )

    def test_main_keeps_init_only_donor_metadata_without_unsupported_single_stage_flag(self):
        module = load_perturbed_seed_module()
        bs, stage2_results, _, _ = (
            BananaCurrentChainScalingTests._build_banana_partitions(11000.0)
        )
        stage2_results = dict(stage2_results)
        stage2_results["init_only"] = True

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            seed_bs_path = tmpdir_path / "seed" / "biot_savart_opt.json"
            seed_bs_path.parent.mkdir(parents=True, exist_ok=True)
            bs.save(str(seed_bs_path))
            write_stage2_results_with_digest(seed_bs_path, stage2_results)
            output_root = tmpdir_path / "variant"

            result = module.main(
                [
                    "--stage2-bs-path",
                    str(seed_bs_path),
                    "--output-root",
                    str(output_root),
                    "--banana-currents-A",
                    ",".join(["9000.0", "-9500.0"] * 5),
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(
                (output_root / module.DEFAULT_SUMMARY_JSON).read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(summary["donor_init_only"])
        self.assertEqual(
            summary["recommended_single_stage_flags"][-5:],
            [
                "--single-stage-banana-current-mode",
                "independent",
                "--single-stage-banana-current-coordinate-scaling",
                "seed-relative",
                "--banana-current-diagnostics",
            ],
        )
        self.assertNotIn(
            "--allow-init-only-stage2-seed",
            summary["recommended_single_stage_flags"],
        )


class IshwPlotTests(unittest.TestCase):
    def test_resolved_error_metric_key_ignores_missing_and_nan_nonqs_ratio(self):
        module = load_plot_module()

        self.assertEqual(
            module._resolved_error_metric_key(
                [
                    {"nonqs_ratio": "", "field_error": 0.01},
                    {"nonqs_ratio": None, "field_error": 0.02},
                    {"nonqs_ratio": float("nan"), "field_error": 0.03},
                ]
            ),
            "field_error",
        )
        self.assertEqual(
            module._resolved_error_metric_key(
                [
                    {"nonqs_ratio": 0.011, "field_error": 0.02},
                ]
            ),
            "nonqs_ratio",
        )

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
                                "case_id": "banana_current_16000A",
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
            self.assertTrue(manifest["generated_plots"]["banana_current_a_vs_iota"])
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
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
                tf_current_A=-8.0e4,
                major_radius=0.976,
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

    def test_stage2_artifact_config_rejects_iota_soft_with_alm_constraint_method(self):
        module = load_workflow_common_module()

        with self.assertRaisesRegex(
            ValueError,
            "stage2_iota_mode='soft' is incompatible with constraint_method='alm'",
        ):
            module.Stage2ArtifactConfig(
                plasma_surf_filename="demo.nc",
                output_root=Path("/tmp/stage2"),
                equilibria_dir=None,
                tf_current_A=-8.0e4,
                major_radius=0.976,
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
            )

    def test_run_stage2_alm_expected_metadata_canonicalizes_exact_constraint_weight(self):
        module = load_module(EXAMPLE_ROOT / "run_stage2_alm.py", "run_stage2_alm")

        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir=None,
            tf_current_A=-8.0e4,
            major_radius=0.976,
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

    def test_stage2_iota_report_payload_persists_hot_loop_and_probe_timings(self):
        module = load_stage2_module()

        args = SimpleNamespace(
            stage2_iota_mode="alm",
            stage2_iota_target=0.2,
            stage2_iota_tolerance=5.0e-3,
            stage2_iota_weight=1.0,
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
        runtime = SimpleNamespace(
            stats=SimpleNamespace(
                bootstrap_seconds=0.25,
                runtime_seconds=1.5,
                runtime_calls=7,
            ),
            initial_state=SimpleNamespace(iota=0.18, penalty=0.03),
            penalty_threshold=5.0e-3,
            effective_weight=2.5,
        )

        with patch.object(
            module,
            "evaluate_stage2_iota_state",
            return_value=SimpleNamespace(iota=0.201, penalty=0.0),
        ), patch.object(
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
                "BOOTABILITY_SOLVE_SUCCESS": True,
                "BOOTABILITY_ABS_IOTA_ERROR": 1.0e-3,
                "BOOTABILITY_ERROR_TYPE": None,
                "BOOTABILITY_ERROR_MESSAGE": None,
            },
        ):
            payload = module.build_stage2_iota_report_payload(
                args=args,
                stage2_bs_artifact_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_payload={},
                stage2_iota_runtime=runtime,
            )

        self.assertTrue(payload["STAGE2_IOTA_HOT_LOOP_ENABLED"])
        self.assertEqual(payload["STAGE2_IOTA_BOOTSTRAP_SECONDS"], 0.25)
        self.assertEqual(payload["STAGE2_IOTA_RUNTIME_SECONDS"], 1.5)
        self.assertEqual(payload["STAGE2_IOTA_RUNTIME_CALLS"], 7)
        self.assertIsNotNone(payload["STAGE2_IOTA_PROBE_SECONDS"])

    def test_stage2_iota_hot_loop_payload_nulls_final_values_after_solve_failure(self):
        module = load_stage2_module()

        args = SimpleNamespace(
            stage2_iota_weight=1.0,
            stage2_iota_vol_target=0.1,
            stage2_iota_constraint_weight=1.0,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
        )
        runtime = SimpleNamespace(
            stats=SimpleNamespace(
                bootstrap_seconds=0.25,
                runtime_seconds=1.5,
                runtime_calls=7,
            ),
            initial_state=SimpleNamespace(iota=0.18, penalty=0.03),
            penalty_threshold=5.0e-3,
            effective_weight=2.5,
        )

        with patch.object(
            module,
            "evaluate_stage2_iota_state",
            return_value=SimpleNamespace(iota=0.201, penalty=0.0, solve_failed=True),
        ):
            payload = module.build_stage2_iota_hot_loop_payload(
                args=args,
                stage2_iota_runtime=runtime,
            )

        self.assertEqual(payload["STAGE2_IOTA_INITIAL"], 0.18)
        self.assertEqual(payload["STAGE2_IOTA_INITIAL_PENALTY"], 0.03)
        self.assertEqual(payload["STAGE2_IOTA_EFFECTIVE_WEIGHT"], 2.5)
        self.assertIsNone(payload["STAGE2_IOTA_FINAL"])
        self.assertIsNone(payload["STAGE2_IOTA_FINAL_PENALTY"])

    def test_stage2_secondary_artifact_helpers_return_standard_bundle_paths(self):
        module = load_stage2_module()

        bs_path, results_path = module.build_stage2_secondary_artifact_paths(
            "/tmp/stage2/biot_savart_opt.json"
        )
        metadata = module.build_stage2_secondary_artifact_metadata(
            secondary_stage2_bs_path=bs_path,
            secondary_stage2_results_path=results_path,
            secondary_source="accepted_iterate",
        )

        self.assertTrue(bs_path.endswith("secondary_exact_hardware_pass_iota_fail/biot_savart_opt.json"))
        self.assertTrue(results_path.endswith("secondary_exact_hardware_pass_iota_fail/results.json"))
        self.assertTrue(metadata["STAGE2_SECONDARY_ARTIFACT_PRESERVED"])
        self.assertEqual(
            metadata["STAGE2_SECONDARY_ARTIFACT_REASON"],
            "exact_hardware_pass_iota_fail",
        )
        self.assertEqual(
            metadata["STAGE2_SECONDARY_ARTIFACT_SOURCE"],
            "accepted_iterate",
        )
        self.assertEqual(metadata["STAGE2_SECONDARY_BS_PATH"], bs_path)
        self.assertEqual(metadata["STAGE2_SECONDARY_RESULTS_PATH"], results_path)

    def test_secondary_stage2_results_drop_primary_alm_final_diagnostics(self):
        module = load_stage2_module()

        with patch.object(module, "is_self_intersecting", return_value=False):
            secondary_kwargs = module.build_secondary_stage2_results_kwargs(
                stage2_results_kwargs={
                    "alm_result": SimpleNamespace(
                        penalty=123.0,
                        multipliers=[1.0],
                        constraint_values=[2.0],
                    ),
                    "optimizer_success": True,
                },
                secondary_state={
                    "banana_current_A": 9000.0,
                    "max_curvature": 22.0,
                    "coil_length": 1.8,
                    "curve_curve_min_dist": 0.06,
                    "curve_surface_min_dist": 0.05,
                    "hardware_status": {"success": True, "violations": []},
                },
                tf_current_A=-80000.0,
                new_banana_curve=SimpleNamespace(),
                new_surf=SimpleNamespace(volume=lambda: 0.11),
                termination_message="solver_done",
            )

        self.assertIsNone(secondary_kwargs["alm_result"])
        self.assertFalse(secondary_kwargs["optimizer_success"])
        self.assertIn(
            "preserved_secondary_exact_hardware_pass_iota_fail",
            secondary_kwargs["termination_message"],
        )


class WorkflowJsonTests(unittest.TestCase):
    def test_write_json_normalizes_nonfinite_floats_to_null(self):
        module = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"

            module.write_json(
                path,
                {
                    "scalar_nan": float("nan"),
                    "nested": {"value": float("inf")},
                    "rows": [1.0, float("-inf"), {"inner": float("nan")}],
                },
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertIsNone(payload["scalar_nan"])
        self.assertIsNone(payload["nested"]["value"])
        self.assertEqual(payload["rows"][0], 1.0)
        self.assertIsNone(payload["rows"][1])
        self.assertIsNone(payload["rows"][2]["inner"])


if __name__ == "__main__":
    unittest.main()
