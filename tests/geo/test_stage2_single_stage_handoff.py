import importlib
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
WRAPPER_PATH = EXAMPLE_ROOT / "run_stage2_to_single_stage.py"
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_wrapper_module():
    return load_module(WRAPPER_PATH, "run_stage2_to_single_stage")


def load_hardware_schema_module():
    return importlib.import_module("banana_opt.hardware_constraint_schema")


def load_artifact_contracts_module():
    return importlib.import_module("banana_opt.artifact_contracts")


def load_handoff_module():
    return importlib.import_module("banana_opt.stage2_single_stage_handoff")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _bootability_status(
    handoff_module,
    *,
    stage: str,
    reason: str,
    bootable: bool,
    iota_feasible: bool,
    solved_iota: float | None,
    self_intersecting: bool | None = None,
) -> dict[str, object]:
    abs_iota_error = None
    if solved_iota is not None:
        abs_iota_error = abs(float(solved_iota) - 0.2)
    return {
        "BOOZER_BOOTABLE": bootable,
        "IOTA_FEASIBLE": iota_feasible,
        "BOOTABILITY_REASON": reason,
        "BOOTABILITY_STAGE": stage,
        "BOOTABILITY_TARGET_IOTA": 0.2,
        "BOOTABILITY_SOLVED_IOTA": solved_iota,
        "BOOTABILITY_SELF_INTERSECTING": self_intersecting,
        "BOOTABILITY_SOLVE_SUCCESS": bootable,
        "BOOTABILITY_ABS_IOTA_ERROR": abs_iota_error,
        "BOOTABILITY_ERROR_TYPE": None,
        "BOOTABILITY_ERROR_MESSAGE": None,
    }


class HandoffSchemaTests(unittest.TestCase):
    def test_build_bootability_recovery_payload_fields_shapes_all_expected_keys(self):
        module = load_hardware_schema_module()

        payload = module.build_bootability_recovery_payload_fields(
            {
                "BOOZER_BOOTABLE": True,
                "IOTA_FEASIBLE": False,
                "BOOTABILITY_REASON": "iota_mismatch",
                "BOOTABILITY_STAGE": "probe",
                "BOOTABILITY_TARGET_IOTA": 0.2,
                "BOOTABILITY_SOLVED_IOTA": 0.18,
                "BOOTABILITY_SELF_INTERSECTING": False,
                "BOOTABILITY_SOLVE_SUCCESS": True,
                "BOOTABILITY_ABS_IOTA_ERROR": 0.02,
                "BOOTABILITY_ERROR_TYPE": None,
                "BOOTABILITY_ERROR_MESSAGE": None,
            },
            stage2_bs_path="/tmp/stage2/biot_savart_opt.json",
            stage2_results_path="/tmp/stage2/results.json",
            recovery_attempted=True,
            recovery_succeeded=False,
            recovery_iters=7,
            recovery_termination_reason="not_bootable_after_budget",
        )

        self.assertEqual(
            module.bootability_recovery_payload_field_names(),
            (
                "BOOZER_BOOTABLE",
                "IOTA_FEASIBLE",
                "BOOTABILITY_REASON",
                "BOOTABILITY_STAGE",
                "BOOTABILITY_TARGET_IOTA",
                "BOOTABILITY_SOLVED_IOTA",
                "BOOTABILITY_SELF_INTERSECTING",
                "BOOTABILITY_SOLVE_SUCCESS",
                "BOOTABILITY_ABS_IOTA_ERROR",
                "BOOTABILITY_ERROR_TYPE",
                "BOOTABILITY_ERROR_MESSAGE",
                "STAGE2_BS_PATH",
                "STAGE2_RESULTS_PATH",
                "RECOVERY_ATTEMPTED",
                "RECOVERY_SUCCEEDED",
                "RECOVERY_ITERS",
                "RECOVERY_TERMINATION_REASON",
            ),
        )
        self.assertTrue(payload["BOOZER_BOOTABLE"])
        self.assertFalse(payload["IOTA_FEASIBLE"])
        self.assertEqual(payload["BOOTABILITY_REASON"], "iota_mismatch")
        self.assertAlmostEqual(payload["BOOTABILITY_ABS_IOTA_ERROR"], 0.02)
        self.assertEqual(payload["RECOVERY_ITERS"], 7)
        self.assertEqual(
            payload["RECOVERY_TERMINATION_REASON"],
            "not_bootable_after_budget",
        )

    def test_upgrade_legacy_stage2_artifact_results_backfills_handoff_defaults(self):
        module = load_artifact_contracts_module()

        upgraded = module.upgrade_legacy_stage2_artifact_results({})

        self.assertIsNone(upgraded["BOOZER_BOOTABLE"])
        self.assertIsNone(upgraded["BOOTABILITY_REASON"])
        self.assertFalse(upgraded["RECOVERY_ATTEMPTED"])
        self.assertFalse(upgraded["RECOVERY_SUCCEEDED"])
        self.assertIsNone(upgraded["RECOVERY_ITERS"])
        self.assertIsNone(upgraded["RECOVERY_TERMINATION_REASON"])
        self.assertFalse(upgraded["STAGE2_SECONDARY_ARTIFACT_PRESERVED"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_ARTIFACT_REASON"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_ARTIFACT_SOURCE"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_BS_PATH"])
        self.assertIsNone(upgraded["STAGE2_SECONDARY_RESULTS_PATH"])
        self.assertEqual(upgraded["FINITE_CURRENT_MODE"], "boozer_surrogate")
        self.assertEqual(upgraded["FINITE_CURRENT_MODE_SOURCE"], "legacy_assumed_default")
        self.assertEqual(upgraded["BOOZER_CURRENT_CONVENTION"], "mu0")
        self.assertEqual(upgraded["NUM_PROXY_COILS"], 0)
        self.assertEqual(upgraded["NUM_VF_COILS"], 0)
        self.assertEqual(upgraded["PROXY_PLASMA_CURRENT_A"], 0.0)
        self.assertEqual(upgraded["VF_CURRENT_A"], 0.0)
        self.assertIsNone(upgraded["VF_TEMPLATE_PATH"])


class HandoffModuleTests(unittest.TestCase):
    def test_classify_bootability_result_rejects_iota_mismatch(self):
        module = load_handoff_module()

        status = module.classify_bootability_result(
            module.BoozerInitializationResult(
                boozer_surface=None,
                solve_success=True,
                self_intersecting=False,
                success=True,
                solved_iota=0.12,
                solved_G=1.0,
                volume=0.1,
            ),
            stage=module.BOOTABILITY_STAGE_PROBE,
            target_iota=0.2,
            iota_tolerance=1.0e-3,
        )

        self.assertFalse(module.bootability_passes(status))
        self.assertEqual(status["BOOTABILITY_REASON"], module.BOOTABILITY_REASON_IOTA_MISMATCH)
        self.assertAlmostEqual(status["BOOTABILITY_ABS_IOTA_ERROR"], 0.08)

    def test_probe_stage2_seed_bootability_reports_missing_metadata_without_loading_bs(self):
        module = load_handoff_module()

        status = module.probe_stage2_seed_bootability(
            stage2_bs_path="/tmp/demo/biot_savart_opt.json",
            stage2_artifact_results={"PLASMA_SURF_FILENAME": "demo.nc"},
            plasma_surf_filename="demo.nc",
            equilibria_dir="/tmp/equilibria",
            num_tf_coils=20,
            nphi=91,
            ntheta=32,
            mpol=8,
            ntor=6,
            vol_target=0.1,
            iota_target=0.2,
            iota_tolerance=5.0e-3,
            constraint_weight=1.0,
        )

        self.assertEqual(
            status["BOOTABILITY_REASON"],
            module.BOOTABILITY_REASON_MISSING_ARTIFACT_METADATA,
        )
        self.assertFalse(module.bootability_passes(status))

    def test_partition_loaded_stage2_coils_uses_recorded_proxy_and_vf_counts(self):
        module = load_handoff_module()
        coils = [object() for _ in range(24)]

        partitions = module.partition_loaded_stage2_coils(
            coils,
            stage2_results={
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 2,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 1,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
            },
            requested_num_tf_coils=20,
        )

        self.assertEqual(len(partitions.tf_coils), 20)
        self.assertEqual(len(partitions.banana_coils), 2)
        self.assertEqual(len(partitions.proxy_coils), 1)
        self.assertEqual(len(partitions.vf_coils), 1)
        self.assertEqual(partitions.finite_current_mode, "wataru_proxy_field")

    def test_partition_loaded_stage2_coils_rejects_inconsistent_partition_total(self):
        module = load_handoff_module()
        coils = [object() for _ in range(22)]

        with self.assertRaisesRegex(ValueError, "partition metadata expects 24"):
            module.partition_loaded_stage2_coils(
                coils,
                stage2_results={
                    "NUM_TF_COILS": 20,
                    "NUM_BANANA_COILS": 2,
                    "NUM_PROXY_COILS": 1,
                    "NUM_VF_COILS": 1,
                },
                requested_num_tf_coils=20,
            )


class UnifiedRunnerTests(unittest.TestCase):
    def _stage2_seed_paths(self, root: Path) -> tuple[Path, Path]:
        stage2_dir = root / "stage2_seed"
        stage2_bs_path = stage2_dir / "biot_savart_opt.json"
        stage2_results_path = stage2_dir / "results.json"
        stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
        stage2_bs_path.write_text("{}", encoding="utf-8")
        _write_json(
            stage2_results_path,
            {
                "PLASMA_SURF_FILENAME": "demo.nc",
                "init_only": False,
            },
        )
        return stage2_bs_path, stage2_results_path

    def test_probe_only_writes_summary_with_bootability_status(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, _ = self._stage2_seed_paths(root)
            summary_root = root / "summary"

            with patch.object(
                wrapper,
                "build_probe_status",
                return_value=_bootability_status(
                    handoff,
                    stage=handoff.BOOTABILITY_STAGE_PROBE,
                    reason=handoff.BOOTABILITY_REASON_IOTA_MISMATCH,
                    bootable=False,
                    iota_feasible=False,
                    solved_iota=0.05,
                    self_intersecting=False,
                ),
            ):
                result = wrapper.main(
                    [
                        "--probe-only",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(summary_root),
                    ]
                )

            self.assertEqual(result, 0)
            summary = json.loads(
                (summary_root / wrapper.DEFAULT_SUMMARY_JSON).read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(summary["mode"], "probe_only")
            self.assertEqual(
                summary["bootability_probe"]["BOOTABILITY_REASON"],
                handoff.BOOTABILITY_REASON_IOTA_MISMATCH,
            )
            self.assertIsNone(summary["recovery"])
            self.assertIsNone(summary["full_single_stage"])

    def test_load_stage2_seed_metadata_for_handoff_backfills_legacy_tf_current_from_cli(self):
        wrapper = load_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            _write_json(
                stage2_results_path,
                {
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "init_only": False,
                },
            )
            args = wrapper.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--stage2-seed-tf-current-A",
                    "12345.0",
                    "--num-tf-coils",
                    "18",
                ]
            )

            _, stage2_results = wrapper.load_stage2_seed_metadata_for_handoff(
                args,
                stage2_bs_path=stage2_bs_path,
            )

            self.assertEqual(stage2_results["TF_CURRENT_A"], 12345.0)
            self.assertEqual(stage2_results["NUM_TF_COILS"], 18)
            self.assertEqual(stage2_results["TF_CURRENT_SUM_ABS_A"], 222210.0)

    def test_build_probe_status_uses_exact_boozer_semantics_for_negative_constraint_weight(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--constraint-weight",
                "-1.0",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={"PLASMA_SURF_FILENAME": "demo.nc"},
                stage="probe",
            )

        self.assertIsNone(probe.call_args.kwargs["constraint_weight"])

    def test_build_probe_status_derives_boozer_current_from_physical_current(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
                "--plasma-current-A",
                "9000.0",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "FINITE_CURRENT_MODE": "boozer_surrogate",
                    "PROXY_PLASMA_CURRENT_A": 0.0,
                },
                stage="probe",
            )

        self.assertAlmostEqual(
            probe.call_args.kwargs["boozer_I"],
            4.0e-7 * 3.141592653589793 * 9000.0,
        )

    def test_build_probe_status_uses_stage2_proxy_current_default_in_wataru_mode(self):
        wrapper = load_wrapper_module()

        args = wrapper.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ]
        )

        with patch.object(wrapper, "probe_stage2_seed_bootability", return_value={}) as probe:
            wrapper.build_probe_status(
                args,
                stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
                stage2_results={
                    "PLASMA_SURF_FILENAME": "demo.nc",
                    "FINITE_CURRENT_MODE": "wataru_proxy_field",
                    "PROXY_PLASMA_CURRENT_A": 9000.0,
                },
                stage="probe",
            )

        self.assertAlmostEqual(
            probe.call_args.kwargs["boozer_I"],
            4.0e-7 * 3.141592653589793 * 9000.0,
        )

    def test_recovery_only_updates_recovery_results_with_handoff_metadata(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            recovery_case_dir = output_root / "recovery" / "mpol=8-ntor=6-test"

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                self.assertEqual(Path(output_root), output_root)
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    recovery_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "iterations": 7,
                    },
                )
                (recovery_case_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "final",
                    recovery_case_dir / "results.json",
                    json.loads(
                        (recovery_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

            initial_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_SELF_INTERSECTION,
                bootable=False,
                iota_feasible=False,
                solved_iota=0.0003,
                self_intersecting=True,
            )
            recovered_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_RECOVERY,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.2004,
                self_intersecting=False,
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=[initial_probe, recovered_probe],
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ):
                result = wrapper.main(
                    [
                        "--recovery-only",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(result, 0)
            recovered_results = json.loads(
                (recovery_case_dir / "results.json").read_text(encoding="utf-8")
            )
            expected_stage2_bs_path = str(stage2_bs_path.resolve())
            expected_stage2_results_path = str(stage2_results_path.resolve())
            self.assertTrue(recovered_results["BOOZER_BOOTABLE"])
            self.assertTrue(recovered_results["IOTA_FEASIBLE"])
            self.assertTrue(recovered_results["RECOVERY_ATTEMPTED"])
            self.assertTrue(recovered_results["RECOVERY_SUCCEEDED"])
            self.assertEqual(recovered_results["RECOVERY_ITERS"], 7)
            self.assertEqual(
                recovered_results["RECOVERY_TERMINATION_REASON"],
                "bootable",
            )
            self.assertEqual(
                recovered_results["STAGE2_BS_PATH"],
                expected_stage2_bs_path,
            )
            self.assertEqual(
                recovered_results["STAGE2_RESULTS_PATH"],
                expected_stage2_results_path,
            )
            self.assertEqual(
                recovered_results["UNIFIED_SEED_SOURCE"],
                wrapper.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )

    def test_full_mode_augments_final_results_with_recovered_handoff_metadata(self):
        wrapper = load_wrapper_module()
        handoff = load_handoff_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._stage2_seed_paths(root)
            output_root = root / "outputs"
            recovery_case_dir = output_root / "recovery" / "mpol=8-ntor=6-test"
            full_case_dir = output_root / "full" / "target" / "mpol=8-ntor=6-test"

            def fake_recovery_run(command, *, output_root, timeout_seconds):
                recovery_case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    recovery_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "iterations": 11,
                    },
                )
                (recovery_case_dir / "biot_savart_opt.json").write_text(
                    "{}",
                    encoding="utf-8",
                )
                return (
                    "final",
                    recovery_case_dir / "results.json",
                    json.loads(
                        (recovery_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                )

            def fake_full_run(args, *, stage2_bs_path, full_output_root):
                full_case_dir.mkdir(parents=True, exist_ok=True)
                _write_json(
                    full_case_dir / "results.json",
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "OPTIMIZER_SUCCESS": True,
                    },
                )
                return {
                    "status": "completed",
                    "command": ["python", "single_stage_banana_example.py"],
                    "output_root": str(full_output_root),
                    "results_path": str(full_case_dir / "results.json"),
                    "result_source": "final",
                    "results": json.loads(
                        (full_case_dir / "results.json").read_text(
                            encoding="utf-8"
                        )
                    ),
                }

            initial_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_PROBE,
                reason=handoff.BOOTABILITY_REASON_SELF_INTERSECTION,
                bootable=False,
                iota_feasible=False,
                solved_iota=0.0002,
                self_intersecting=True,
            )
            recovered_probe = _bootability_status(
                handoff,
                stage=handoff.BOOTABILITY_STAGE_RECOVERY,
                reason=handoff.BOOTABILITY_REASON_OK,
                bootable=True,
                iota_feasible=True,
                solved_iota=0.1999,
                self_intersecting=False,
            )

            with patch.object(
                wrapper,
                "build_probe_status",
                side_effect=[initial_probe, recovered_probe],
            ), patch.object(
                wrapper,
                "run_single_stage_command_with_salvage",
                side_effect=fake_recovery_run,
            ), patch.object(
                wrapper,
                "run_full_single_stage",
                side_effect=fake_full_run,
            ):
                result = wrapper.main(
                    [
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(output_root),
                    ]
                )

            self.assertEqual(result, 0)
            final_results = json.loads(
                (full_case_dir / "results.json").read_text(encoding="utf-8")
            )
            expected_stage2_bs_path = str(stage2_bs_path.resolve())
            expected_stage2_results_path = str(stage2_results_path.resolve())
            self.assertTrue(final_results["BOOZER_BOOTABLE"])
            self.assertTrue(final_results["IOTA_FEASIBLE"])
            self.assertTrue(final_results["RECOVERY_ATTEMPTED"])
            self.assertTrue(final_results["RECOVERY_SUCCEEDED"])
            self.assertEqual(final_results["RECOVERY_ITERS"], 11)
            self.assertEqual(
                final_results["RECOVERY_TERMINATION_REASON"],
                "bootable",
            )
            self.assertEqual(
                final_results["STAGE2_BS_PATH"],
                expected_stage2_bs_path,
            )
            self.assertEqual(
                final_results["STAGE2_RESULTS_PATH"],
                expected_stage2_results_path,
            )
            self.assertEqual(
                final_results["UNIFIED_SEED_SOURCE"],
                wrapper.SEED_SOURCE_RECOVERED_STAGE2_DONOR,
            )

    def test_recovery_only_conflict_with_skip_recovery_is_rejected(self):
        wrapper = load_wrapper_module()

        with self.assertRaisesRegex(
            ValueError,
            "--recovery-only cannot be combined with --skip-recovery",
        ):
            wrapper.main(
                [
                    "--recovery-only",
                    "--skip-recovery",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    "/tmp/stage2/biot_savart_opt.json",
                ]
            )
