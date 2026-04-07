import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
WORKFLOW_HELPERS_PATH = EXAMPLE_ROOT / "workflow_helpers.py"
WORKFLOW_COMMON_PATH = EXAMPLE_ROOT / "workflow_runner_common.py"
BASELINE_SWEEP_PATH = EXAMPLE_ROOT / "run_80ka_baseline_tradeoff_sweep.py"
FINITE_CURRENT_SMOKE_PATH = EXAMPLE_ROOT / "run_finite_current_smoke.py"


def load_module(path: Path, stem: str):
    spec = importlib.util.spec_from_file_location(f"{stem}_{uuid.uuid4().hex}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_workflow_helpers_module():
    return load_module(WORKFLOW_HELPERS_PATH, "workflow_helpers")


def load_workflow_common_module():
    return load_module(WORKFLOW_COMMON_PATH, "workflow_runner_common")


def load_baseline_sweep_module():
    return load_module(BASELINE_SWEEP_PATH, "run_80ka_baseline_tradeoff_sweep")


def load_finite_current_smoke_module():
    return load_module(FINITE_CURRENT_SMOKE_PATH, "run_finite_current_smoke")


class WorkflowHelpersTests(unittest.TestCase):
    def test_format_local_stage2_run_dir_includes_constraint_and_basin_suffix(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=8.0e4,
            order=2,
        )

        run_dir = module.format_local_stage2_run_dir(
            spec,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=3,
            basin_stepsize=0.01,
            basin_seed=7,
        )

        self.assertIn("TFC=80000", run_dir)
        self.assertIn("-CM=penalty", run_dir)
        self.assertIn("-BH=3-BS=0.01-BSeed=7", run_dir)

    def test_local_stage2_bs_path_matches_current_stage2_contract(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=8.0e4,
            order=2,
        )

        artifact_path = module.local_stage2_bs_path(
            "/tmp/stage2-root",
            spec,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_seed=None,
        )

        self.assertEqual(
            str(artifact_path),
            "/tmp/stage2-root/outputs-demo.nc/"
            "R0=0.915-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-TFC=80000-Order=2-CM=penalty/"
            "biot_savart_opt.json",
        )

    def test_select_non_dominated_records_uses_augmented_iota_error(self):
        module = load_workflow_helpers_module()
        records = [
            {"CASE_NAME": "best", "FIELD_ERROR": 0.1, "FINAL_IOTA": 0.15, "TARGET_IOTA": 0.15},
            {"CASE_NAME": "worse", "FIELD_ERROR": 0.2, "FINAL_IOTA": 0.2, "TARGET_IOTA": 0.15},
        ]

        front = module.select_non_dominated_records(records, ["FIELD_ERROR", "IOTA_ERROR_ABS"])

        self.assertEqual([record["CASE_NAME"] for record in front], ["best"])


class WorkflowRunnerCommonTests(unittest.TestCase):
    def test_build_stage2_command_contains_seed_contract(self):
        module = load_workflow_common_module()
        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir="/tmp/equilibria",
            tf_current_A=8.0e4,
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_seed=None,
            init_only=True,
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertEqual(command[0], "python")
        self.assertIn("--tf-current-A", command)
        self.assertIn("--output-root", command)
        self.assertIn("--init-only", command)

    def test_discover_single_results_path_requires_unique_match(self):
        module = load_workflow_common_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            case_dir = output_root / "mpol=4-ntor=4-hash"
            case_dir.mkdir(parents=True)
            (case_dir / "results.json").write_text("{}", encoding="utf-8")

            self.assertEqual(
                module.discover_single_results_path(output_root),
                case_dir / "results.json",
            )

    def test_discover_single_results_path_uses_new_match_from_previous_snapshot(self):
        module = load_workflow_common_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            old_dir = output_root / "mpol=4-ntor=4-old"
            old_dir.mkdir(parents=True)
            (old_dir / "results.json").write_text("{}", encoding="utf-8")

            previous_snapshot = module.snapshot_single_results_paths(output_root)

            new_dir = output_root / "mpol=4-ntor=4-new"
            new_dir.mkdir(parents=True)
            (new_dir / "results.json").write_text("{}", encoding="utf-8")

            self.assertEqual(
                module.discover_single_results_path(
                    output_root,
                    previous_snapshot=previous_snapshot,
                ),
                new_dir / "results.json",
            )

    def test_discover_single_results_path_uses_updated_match_from_previous_snapshot(self):
        module = load_workflow_common_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            old_dir = output_root / "mpol=4-ntor=4-old"
            rerun_dir = output_root / "mpol=4-ntor=4-rerun"
            old_dir.mkdir(parents=True)
            rerun_dir.mkdir(parents=True)
            (old_dir / "results.json").write_text("{}", encoding="utf-8")
            rerun_path = rerun_dir / "results.json"
            rerun_path.write_text("{\"version\": 1}", encoding="utf-8")

            previous_snapshot = module.snapshot_single_results_paths(output_root)

            rerun_path.write_text("{\"version\": 2}", encoding="utf-8")

            self.assertEqual(
                module.discover_single_results_path(
                    output_root,
                    previous_snapshot=previous_snapshot,
                ),
                rerun_path,
            )

    def test_parse_csv_rejects_empty_input(self):
        module = load_workflow_common_module()

        with self.assertRaisesRegex(ValueError, "at least one"):
            module.parse_csv(" , ", float)


class BaselineSweepScriptTests(unittest.TestCase):
    def _make_args(self):
        return SimpleNamespace(
            python_executable="python",
            plasma_surf_filename="demo.nc",
            equilibria_dir=None,
            single_stage_constraint_method="penalty",
            single_stage_maxiter=25,
            single_stage_init_only=True,
            plasma_current_A=0.0,
            res_weight=1000.0,
            iotas_weight=100.0,
            cc_weight=100.0,
            curvature_weight=0.1,
            length_weight=1.0,
            cs_weight=1.0,
            surf_dist_weight=1000.0,
            scan_weights="res_weight,cc_weight",
            weight_multipliers="0.5,1.0,2.0",
        )

    def test_build_single_stage_command_uses_zero_plasma_current(self):
        module = load_baseline_sweep_module()
        helpers = load_workflow_helpers_module()
        args = self._make_args()
        case = helpers.SingleStageWeightCase(
            name="baseline",
            res_weight=1000.0,
            iotas_weight=100.0,
            cc_weight=100.0,
            curvature_weight=0.1,
            length_weight=1.0,
            cs_weight=1.0,
            surf_dist_weight=1000.0,
        )

        command = module.build_single_stage_command(
            args,
            case=case,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            case_output_root=Path("/tmp/sweep/baseline"),
        )

        self.assertIn("--plasma-current-A", command)
        self.assertIn("0.0", command)
        self.assertIn("--init-only", command)

    def test_build_summary_reports_non_dominated_cases_and_artifact_provenance(self):
        module = load_baseline_sweep_module()
        common = load_workflow_common_module()
        records = [
            {
                "CASE_NAME": "better",
                "FIELD_ERROR": 0.1,
                "FINAL_IOTA": 0.15,
                "TARGET_IOTA": 0.15,
                "COIL_LENGTH": 1.2,
                "MAX_CURVATURE": 30.0,
                "NEG_CURVE_CURVE_MIN_DIST": -0.08,
            },
            {
                "CASE_NAME": "worse",
                "FIELD_ERROR": 0.2,
                "FINAL_IOTA": 0.20,
                "TARGET_IOTA": 0.15,
                "COIL_LENGTH": 1.4,
                "MAX_CURVATURE": 35.0,
                "NEG_CURVE_CURVE_MIN_DIST": -0.05,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path = stage2_dir / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            stage2_results_path = stage2_dir / "results.json"
            stage2_results_path.write_text(
                json.dumps(
                    {
                        "TF_CURRENT_A": 1.0e5,
                        "MAJOR_RADIUS": 0.915,
                        "TOROIDAL_FLUX": 0.24,
                    }
                ),
                encoding="utf-8",
            )
            requested_config = common.Stage2ArtifactConfig(
                plasma_surf_filename="demo.nc",
                output_root=stage2_dir,
                equilibria_dir=None,
                tf_current_A=8.0e4,
                major_radius=0.915,
                toroidal_flux=0.24,
                length_weight=0.0005,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=40.0,
                banana_surf_radius=0.22,
                order=2,
                constraint_method="penalty",
                alm_max_outer_iters=10,
                alm_penalty_init=1.0,
                alm_penalty_scale=10.0,
                basin_hops=0,
                basin_stepsize=0.01,
                basin_seed=None,
                init_only=True,
            )

            summary = module.build_summary(stage2_bs_path, requested_config, records)

        self.assertEqual(summary["non_dominated_case_names"], ["better"])
        self.assertEqual(summary["stage2_requested_config"]["tf_current_A"], 8.0e4)
        self.assertEqual(summary["stage2_artifact_results"]["TF_CURRENT_A"], 1.0e5)
        self.assertEqual(summary["stage2_results_path"], str(stage2_results_path))
        self.assertEqual(summary["stage2_bs_path"], str(stage2_bs_path))


class FiniteCurrentSmokeScriptTests(unittest.TestCase):
    def _make_args(self):
        return SimpleNamespace(
            python_executable="python",
            plasma_surf_filename="demo.nc",
            equilibria_dir=None,
            nphi=41,
            ntheta=16,
            mpol=4,
            ntor=4,
        )

    def test_build_smoke_command_includes_init_only_and_resolution(self):
        module = load_finite_current_smoke_module()
        args = self._make_args()

        command = module.build_smoke_command(
            args,
            current_A=8000.0,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            case_output_root=Path("/tmp/smoke/current_8000"),
        )

        self.assertIn("--init-only", command)
        self.assertIn("--nphi", command)
        self.assertIn("--plasma-current-A", command)

    def test_validate_smoke_results_checks_current_contract(self):
        module = load_finite_current_smoke_module()
        results = {
            "PLASMA_CURRENT_A": -35200.0,
            "PLASMA_CURRENT_INPUT_SOURCE": "physical_A",
            "BOOZER_I": -0.00704,
            "STAGE2_TF_CURRENT_A": 8.0e4,
            "FINITE_CURRENT_MODE": "boozer_surrogate",
        }

        validation = module.validate_smoke_results(
            results,
            requested_current_A=-35200.0,
            expected_stage2_tf_current_A=8.0e4,
        )

        self.assertTrue(validation["passed"])

    def test_validate_smoke_results_uses_actual_artifact_tf_current(self):
        module = load_finite_current_smoke_module()
        results = {
            "PLASMA_CURRENT_A": 0.0,
            "PLASMA_CURRENT_INPUT_SOURCE": "physical_A",
            "BOOZER_I": 0.0,
            "STAGE2_TF_CURRENT_A": 1.0e5,
            "FINITE_CURRENT_MODE": "boozer_surrogate",
        }

        validation = module.validate_smoke_results(
            results,
            requested_current_A=0.0,
            expected_stage2_tf_current_A=1.0e5,
        )

        self.assertTrue(validation["passed"])

    def test_resolve_expected_stage2_tf_current_A_requires_artifact_metadata(self):
        module = load_finite_current_smoke_module()

        with self.assertRaisesRegex(ValueError, "missing TF_CURRENT_A"):
            module.resolve_expected_stage2_tf_current_A({})
