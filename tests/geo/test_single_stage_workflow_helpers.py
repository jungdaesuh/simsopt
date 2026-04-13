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


EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
)
WORKFLOW_HELPERS_PATH = EXAMPLE_ROOT / "workflow_helpers.py"
WORKFLOW_COMMON_PATH = EXAMPLE_ROOT / "workflow_runner_common.py"
BASELINE_SWEEP_PATH = EXAMPLE_ROOT / "run_80ka_baseline_tradeoff_sweep.py"
FINITE_CURRENT_SMOKE_PATH = EXAMPLE_ROOT / "run_finite_current_smoke.py"
GOAL_MODE_COMPARISON_PATH = EXAMPLE_ROOT / "run_single_stage_goal_mode_comparison.py"
SINGLE_STAGE_ENTRYPOINT_PATH = EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
STAGE2_ENTRYPOINT_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
IMPORT_PROVENANCE_PATH = EXAMPLE_ROOT / "import_provenance.py"
EXPECTED_LOCAL_SIMSOPT_INIT = (
    Path(__file__).resolve().parents[2] / "src" / "simsopt" / "__init__.py"
)


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


def load_goal_mode_comparison_module():
    return load_module(
        GOAL_MODE_COMPARISON_PATH,
        "run_single_stage_goal_mode_comparison",
    )


def _run_python_snippet(source: str, *args: str) -> str:
    command = [
        sys.executable,
        "-c",
        source,
        *args,
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def imported_simsopt_init_for_entrypoint(script_path: Path) -> Path:
    return Path(
        _run_python_snippet(
            (
                "import pathlib, runpy, sys; "
                "runpy.run_path(sys.argv[1], run_name='__probe__'); "
                "import simsopt; "
                "print(pathlib.Path(simsopt.__file__).resolve())"
            ),
            str(script_path),
        )
    )


class WorkflowHelpersTests(unittest.TestCase):
    def test_stage2_seed_spec_rejects_out_of_range_toroidal_flux(self):
        module = load_workflow_helpers_module()

        with self.assertRaisesRegex(ValueError, "between 0 and 1 inclusive"):
            module.Stage2SeedSpec(
                plasma_surf_filename="demo.nc",
                major_radius=0.915,
                toroidal_flux=1.2,
                length_weight=0.0005,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=40.0,
                banana_surf_radius=0.22,
                tf_current_A=8.0e4,
                order=2,
            )

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
            basin_temperature=2.5,
            basin_niter_success=8,
            basin_seed=7,
        )

        self.assertIn("TFC=80000", run_dir)
        self.assertIn("INITC=10000", run_dir)
        self.assertIn("-CM=penalty", run_dir)
        self.assertIn("-BH=3-BS=0.01-BSeed=7-BT=2.5-BNS=8", run_dir)

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
            "R0=0.915-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2-CM=penalty/"
            "biot_savart_opt.json",
        )

    def test_format_local_stage2_run_dir_includes_alm_penalty_cap_when_enabled(self):
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
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=1.0e8,
            basin_hops=0,
            basin_stepsize=0.01,
        )

        self.assertIn("-CM=alm-ALMOuter=10-ALMMu=1-ALMScale=10-ALMMax=1e+08", run_dir)

    def test_select_non_dominated_records_uses_augmented_iota_error(self):
        module = load_workflow_helpers_module()
        records = [
            {"CASE_NAME": "best", "FIELD_ERROR": 0.1, "FINAL_IOTA": 0.15, "TARGET_IOTA": 0.15},
            {"CASE_NAME": "worse", "FIELD_ERROR": 0.2, "FINAL_IOTA": 0.2, "TARGET_IOTA": 0.15},
        ]

        front = module.select_non_dominated_records(records, ["FIELD_ERROR", "IOTA_ERROR_ABS"])

        self.assertEqual([record["CASE_NAME"] for record in front], ["best"])


class WorkflowRunnerCommonTests(unittest.TestCase):
    def test_stage2_artifact_config_rejects_out_of_range_toroidal_flux(self):
        module = load_workflow_common_module()

        with self.assertRaisesRegex(ValueError, "between 0 and 1 inclusive"):
            module.Stage2ArtifactConfig(
                plasma_surf_filename="demo.nc",
                output_root=Path("/tmp/stage2"),
                equilibria_dir=None,
                tf_current_A=8.0e4,
                major_radius=0.915,
                toroidal_flux=-0.01,
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

    def test_single_stage_entrypoint_imports_local_simsopt(self):
        imported_path = imported_simsopt_init_for_entrypoint(SINGLE_STAGE_ENTRYPOINT_PATH)

        self.assertEqual(imported_path, EXPECTED_LOCAL_SIMSOPT_INIT)

    def test_stage2_entrypoint_imports_local_simsopt(self):
        imported_path = imported_simsopt_init_for_entrypoint(STAGE2_ENTRYPOINT_PATH)

        self.assertEqual(imported_path, EXPECTED_LOCAL_SIMSOPT_INIT)

    def test_import_provenance_moves_local_paths_to_front_when_already_present(self):
        imported_path = Path(
            _run_python_snippet(
                (
                "import pathlib, sys, tempfile; "
                "from importlib.util import module_from_spec, spec_from_file_location; "
                "fake_root = pathlib.Path(tempfile.mkdtemp()); "
                "(fake_root / 'simsopt').mkdir(); "
                "(fake_root / 'simsopt' / '__init__.py').write_text('ORIGIN = \"fake\"\\n', encoding='utf-8'); "
                "sys.path[:] = [str(fake_root), str(pathlib.Path(sys.argv[2]).resolve()), str(pathlib.Path(sys.argv[3]).resolve()), *sys.path]; "
                "spec = spec_from_file_location('import_provenance_test', sys.argv[1]); "
                "module = module_from_spec(spec); "
                "sys.modules[spec.name] = module; "
                "spec.loader.exec_module(module); "
                "module.configure_local_simsopt_imports(sys.argv[4]); "
                "import simsopt; "
                "print(pathlib.Path(simsopt.__file__).resolve())"
                ),
                str(IMPORT_PROVENANCE_PATH),
                str(EXAMPLE_ROOT.parent.parent / "src"),
                str(EXAMPLE_ROOT),
                str(SINGLE_STAGE_ENTRYPOINT_PATH),
            )
        )

        self.assertEqual(imported_path, EXPECTED_LOCAL_SIMSOPT_INIT)

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
        self.assertIn("--banana-init-current-A", command)
        self.assertIn("--banana-current-max-A", command)
        self.assertIn("--output-root", command)
        self.assertIn("--init-only", command)

    def test_build_stage2_command_threads_extended_basin_controls(self):
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
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=3,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=8,
            basin_seed=11,
            init_only=False,
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertIn("--basin-hops", command)
        self.assertIn("--basin-stepsize", command)
        self.assertIn("--basin-temperature", command)
        self.assertIn("--basin-niter-success", command)
        self.assertIn("--basin-seed", command)

    def test_locked_baseline_stage2_metadata_includes_basin_identity(self):
        common = load_workflow_common_module()
        module = load_baseline_sweep_module()
        config = common.Stage2ArtifactConfig(
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
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=3,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=8,
            basin_seed=11,
            init_only=False,
        )

        metadata = module.expected_locked_baseline_stage2_artifact_metadata(config)

        self.assertEqual(metadata["basin_hops"], 3)
        self.assertEqual(metadata["basin_stepsize"], 0.01)
        self.assertEqual(metadata["basin_temperature"], 2.5)
        self.assertEqual(metadata["basin_niter_success"], 8)
        self.assertEqual(metadata["basin_seed"], 11)

    def test_locked_baseline_stage2_metadata_drops_basin_seed_without_basin_hops(self):
        common = load_workflow_common_module()
        module = load_baseline_sweep_module()
        config = common.Stage2ArtifactConfig(
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
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=8,
            basin_seed=11,
            init_only=False,
        )

        metadata = module.expected_locked_baseline_stage2_artifact_metadata(config)

        self.assertIsNone(metadata["basin_stepsize"])
        self.assertIsNone(metadata["basin_temperature"])
        self.assertIsNone(metadata["basin_niter_success"])
        self.assertIsNone(metadata["basin_seed"])

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
    def _make_stage2_artifact_results(self, **overrides):
        results = {
            "PLASMA_SURF_FILENAME": "demo.nc",
            "TF_CURRENT_A": 8.0e4,
            "TF_CURRENT_SUM_ABS_A": 1.6e6,
            "NUM_TF_COILS": 20,
            "MAJOR_RADIUS": 0.915,
            "TOROIDAL_FLUX": 0.24,
            "LENGTH_WEIGHT": 0.0005,
            "CC_WEIGHT": 100.0,
            "CC_THRESHOLD": 0.05,
            "CURVATURE_WEIGHT": 0.0001,
            "CURVATURE_THRESHOLD": 40.0,
            "banana_surf_radius": 0.22,
            "order": 2,
            "CONSTRAINT_METHOD": "penalty",
            "basin_hops": 0,
            "basin_stepsize": None,
            "basin_seed": None,
            "init_only": False,
        }
        results.update(overrides)
        return results

    def _make_expected_stage2_config(self, common, output_root: Path, **overrides):
        config = {
            "plasma_surf_filename": "demo.nc",
            "output_root": output_root,
            "equilibria_dir": None,
            "tf_current_A": 8.0e4,
            "major_radius": 0.915,
            "toroidal_flux": 0.24,
            "length_weight": 0.0005,
            "cc_weight": 100.0,
            "cc_threshold": 0.05,
            "curvature_weight": 0.0001,
            "curvature_threshold": 40.0,
            "banana_surf_radius": 0.22,
            "order": 2,
            "constraint_method": "penalty",
            "alm_max_outer_iters": 10,
            "alm_penalty_init": 1.0,
            "alm_penalty_scale": 10.0,
            "basin_hops": 0,
            "basin_stepsize": 0.01,
            "basin_seed": None,
            "init_only": False,
            "banana_init_current_A": 1.0e4,
            "banana_current_max_A": 1.6e4,
        }
        config.update(overrides)
        return common.Stage2ArtifactConfig(**config)

    def _make_args(self):
        return SimpleNamespace(
            python_executable="python",
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            equilibria_dir=None,
            tf_current_A=8.0e4,
            major_radius=0.915,
            toroidal_flux=0.24,
            stage2_length_weight=0.0005,
            stage2_cc_weight=100.0,
            stage2_cc_threshold=0.05,
            stage2_curvature_weight=0.0001,
            stage2_curvature_threshold=40.0,
            banana_surf_radius=0.22,
            stage2_order=2,
            stage2_constraint_method="penalty",
            stage2_basin_hops=0,
            stage2_basin_stepsize=0.01,
            stage2_basin_seed=-1,
            stage2_init_only=False,
            allow_init_only_stage2_seed=False,
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

    def test_validate_locked_baseline_args_rejects_nonzero_plasma_current(self):
        module = load_baseline_sweep_module()
        args = self._make_args()
        args.plasma_current_A = 8000.0

        with self.assertRaisesRegex(ValueError, "--plasma-current-A"):
            module.validate_locked_baseline_args(args)

    def test_validate_locked_baseline_args_rejects_non_80ka_tf_current(self):
        module = load_baseline_sweep_module()
        args = self._make_args()
        args.tf_current_A = 1.0e5

        with self.assertRaisesRegex(ValueError, "--tf-current-A"):
            module.validate_locked_baseline_args(args)

    def test_validate_locked_baseline_args_rejects_nonbaseline_stage2_geometry(self):
        module = load_baseline_sweep_module()
        args = self._make_args()
        args.major_radius = 1.23

        with self.assertRaisesRegex(ValueError, "--major-radius"):
            module.validate_locked_baseline_args(args)

    def test_validate_locked_baseline_args_rejects_nonbaseline_constraint_method(self):
        module = load_baseline_sweep_module()
        args = self._make_args()
        args.stage2_constraint_method = "alm"

        with self.assertRaisesRegex(ValueError, "--stage2-constraint-method"):
            module.validate_locked_baseline_args(args)

    def test_load_locked_baseline_stage2_artifact_rejects_wrong_tf_current(self):
        module = load_baseline_sweep_module()
        common = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path = stage2_dir / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (stage2_dir / "results.json").write_text(
                json.dumps(
                    self._make_stage2_artifact_results(
                        TF_CURRENT_A=1.0e5,
                        TF_CURRENT_SUM_ABS_A=2.0e6,
                    )
                ),
                encoding="utf-8",
            )
            expected_config = self._make_expected_stage2_config(common, stage2_dir)

            with self.assertRaisesRegex(ValueError, "TF_CURRENT_A"):
                module.load_locked_baseline_stage2_artifact(
                    stage2_bs_path,
                    expected_config,
                )

    def test_load_locked_baseline_stage2_artifact_rejects_wrong_geometry_identity(self):
        module = load_baseline_sweep_module()
        common = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path = stage2_dir / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (stage2_dir / "results.json").write_text(
                json.dumps(
                    self._make_stage2_artifact_results(MAJOR_RADIUS=1.23)
                ),
                encoding="utf-8",
            )
            expected_config = self._make_expected_stage2_config(common, stage2_dir)

            with self.assertRaisesRegex(ValueError, "MAJOR_RADIUS"):
                module.load_locked_baseline_stage2_artifact(
                    stage2_bs_path,
                    expected_config,
                )

    def test_load_locked_baseline_stage2_artifact_rejects_wrong_total_tf_current(self):
        module = load_baseline_sweep_module()
        common = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path = stage2_dir / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (stage2_dir / "results.json").write_text(
                json.dumps(
                    self._make_stage2_artifact_results(
                        TF_CURRENT_SUM_ABS_A=8.0e5,
                        NUM_TF_COILS=10,
                    )
                ),
                encoding="utf-8",
            )
            expected_config = self._make_expected_stage2_config(common, stage2_dir)

            with self.assertRaisesRegex(ValueError, "TF_CURRENT_SUM_ABS_A|NUM_TF_COILS"):
                module.load_locked_baseline_stage2_artifact(
                    stage2_bs_path,
                    expected_config,
                )

    def test_load_locked_baseline_stage2_artifact_upgrades_legacy_tf_metadata(self):
        module = load_baseline_sweep_module()
        common = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path = stage2_dir / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            legacy_results = self._make_stage2_artifact_results()
            legacy_results.pop("TF_CURRENT_SUM_ABS_A")
            legacy_results.pop("NUM_TF_COILS")
            stage2_results_path = stage2_dir / "results.json"
            stage2_results_path.write_text(
                json.dumps(legacy_results),
                encoding="utf-8",
            )
            expected_config = self._make_expected_stage2_config(common, stage2_dir)

            loaded_results_path, loaded_results = module.load_locked_baseline_stage2_artifact(
                stage2_bs_path,
                expected_config,
            )

        self.assertEqual(loaded_results_path, stage2_results_path)
        self.assertEqual(loaded_results["NUM_TF_COILS"], 20)
        self.assertEqual(loaded_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)
        self.assertEqual(loaded_results["BANANA_INIT_CURRENT_A"], 1.0e4)
        self.assertEqual(loaded_results["BANANA_CURRENT_MAX_A"], 1.6e4)

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
                json.dumps(self._make_stage2_artifact_results()),
                encoding="utf-8",
            )
            requested_config = self._make_expected_stage2_config(common, stage2_dir)

            summary = module.build_summary(stage2_bs_path, requested_config, records)

        self.assertEqual(summary["experiment_family"], "coil_only_baseline")
        self.assertEqual(summary["plasma_current_locked_A"], 0.0)
        self.assertEqual(summary["tf_current_locked_A"], 8.0e4)
        self.assertEqual(summary["non_dominated_case_names"], ["better"])
        self.assertEqual(summary["stage2_requested_config"]["tf_current_A"], 8.0e4)
        self.assertEqual(summary["stage2_artifact_results"]["TF_CURRENT_A"], 8.0e4)
        self.assertEqual(summary["stage2_results_path"], str(stage2_results_path))
        self.assertEqual(summary["stage2_bs_path"], str(stage2_bs_path))


class FiniteCurrentSmokeScriptTests(unittest.TestCase):
    def _make_smoke_results(self, **overrides):
        results = {
            "PLASMA_CURRENT_A": 0.0,
            "PLASMA_CURRENT_INPUT_SOURCE": "physical_A",
            "BOOZER_I": 0.0,
            "EFFECTIVE_CURRENT_MODE": "vacuum",
            "STAGE2_TF_CURRENT_A": 8.0e4,
            "STAGE2_TF_CURRENT_SUM_ABS_A": 1.6e6,
            "FINITE_CURRENT_MODE": "boozer_surrogate",
        }
        results.update(overrides)
        return results

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
        results = self._make_smoke_results(
            PLASMA_CURRENT_A=-35200.0,
            BOOZER_I=-0.00704,
            EFFECTIVE_CURRENT_MODE="boozer_surrogate",
        )

        validation = module.validate_smoke_results(
            results,
            requested_current_A=-35200.0,
            expected_stage2_tf_current_A=8.0e4,
            expected_stage2_tf_current_sum_abs_A=1.6e6,
        )

        self.assertTrue(validation["passed"])
        self.assertTrue(validation["effective_mode_matches"])

    def test_validate_smoke_results_uses_actual_artifact_tf_current(self):
        module = load_finite_current_smoke_module()
        results = self._make_smoke_results(
            STAGE2_TF_CURRENT_A=1.0e5,
            STAGE2_TF_CURRENT_SUM_ABS_A=2.0e6,
        )

        validation = module.validate_smoke_results(
            results,
            requested_current_A=0.0,
            expected_stage2_tf_current_A=1.0e5,
            expected_stage2_tf_current_sum_abs_A=2.0e6,
        )

        self.assertTrue(validation["passed"])

    def test_validate_smoke_results_uses_actual_artifact_total_tf_current(self):
        module = load_finite_current_smoke_module()
        results = self._make_smoke_results()

        validation = module.validate_smoke_results(
            results,
            requested_current_A=0.0,
            expected_stage2_tf_current_A=8.0e4,
            expected_stage2_tf_current_sum_abs_A=1.6e6,
        )

        self.assertTrue(validation["passed"])

    def test_resolve_expected_stage2_tf_current_A_requires_artifact_metadata(self):
        module = load_finite_current_smoke_module()

        with self.assertRaisesRegex(ValueError, "missing TF_CURRENT_A"):
            module.resolve_expected_stage2_tf_current_A({})

    def test_resolve_expected_stage2_tf_current_sum_abs_A_requires_artifact_metadata(self):
        module = load_finite_current_smoke_module()

        with self.assertRaisesRegex(ValueError, "missing TF_CURRENT_SUM_ABS_A"):
            module.resolve_expected_stage2_tf_current_sum_abs_A({})

    def test_legacy_smoke_artifact_upgrades_total_tf_current_when_num_coils_present(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "TF_CURRENT_A": 8.0e4,
            "NUM_TF_COILS": 20,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)

    def test_legacy_smoke_artifact_upgrades_total_tf_current_from_negative_tf_current(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "TF_CURRENT_A": -8.0e4,
            "NUM_TF_COILS": 20,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)

    def test_legacy_smoke_artifact_still_fails_when_total_tf_current_is_ambiguous(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "TF_CURRENT_A": 8.0e4,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        with self.assertRaisesRegex(ValueError, "missing TF_CURRENT_SUM_ABS_A"):
            module.resolve_expected_stage2_tf_current_sum_abs_A(upgraded_results)

    def test_legacy_smoke_artifact_upgrades_banana_current_metadata_for_fresh_runs(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "BANANA_CURRENT_A": 9500.0,
            "STAGE2_BS_PATH": None,
            "init_only": False,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["BANANA_INIT_CURRENT_A"], 1.0e4)
        self.assertEqual(upgraded_results["BANANA_CURRENT_MAX_A"], 1.6e4)

    def test_legacy_smoke_artifact_upgrades_banana_current_max_conservatively(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "BANANA_CURRENT_A": -2.1e4,
            "STAGE2_BS_PATH": "/tmp/seed.json",
            "init_only": False,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["BANANA_CURRENT_MAX_A"], 2.1e4)
        self.assertNotIn("BANANA_INIT_CURRENT_A", upgraded_results)


class GoalModeComparisonScriptTests(unittest.TestCase):
    def _make_args(self):
        return SimpleNamespace(
            python_executable="python",
            dry_run=False,
            plasma_surf_filename="demo.nc",
            stage2_bs_path="relative/seed.json",
            equilibria_dir=None,
            output_root="outputs",
            summary_json=None,
            single_stage_timeout_seconds=0.0,
            nphi=91,
            ntheta=32,
            mpol=8,
            ntor=6,
            maxiter=300,
            maxcor=300,
            ftol=1e-15,
            gtol=1e-15,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_feas_tol=1e-6,
            alm_stationarity_tol=1e-6,
            alm_trust_radius_init=0.05,
            alm_trust_radius_min=1e-4,
            alm_trust_radius_shrink=0.5,
            alm_trust_radius_grow=1.5,
            alm_max_inner_attempts=4,
            alm_max_subproblem_continuations=20,
            alm_distance_smoothing=0.005,
            alm_curvature_smoothing=0.05,
            alm_formulation="weighted_sum",
            alm_qs_threshold=0.01,
            alm_boozer_threshold=0.02,
            alm_iota_penalty_threshold=0.03,
            alm_length_penalty_threshold=0.04,
            iota_target=0.15,
            vol_target=0.10,
            boozer_I=0.123,
            plasma_current_A=8000.0,
            banana_surf_radius=0.22,
            num_surfaces=2,
            inner_surface_ratio=0.8,
            surface_gap_threshold=0.01,
            multisurface_ramp_iterations=5,
            inner_surface_initial_weight=0.2,
            multisurface_initial_step_scale=0.5,
            multisurface_initial_step_maxiter=7,
            boozer_stage="initial",
            boozer_stage_refinement=True,
            refinement_boozer_stage="final",
            refinement_maxiter=100,
            refinement_chunk_maxiter=20,
            refinement_max_stalled_chunks=2,
            res_weight=1000.0,
            iotas_weight=100.0,
            frontier_volume_weight=None,
            cc_weight=100.0,
            curvature_weight=0.1,
            length_weight=1.0,
            length_target=1.75,
            cs_weight=1.0,
            surf_dist_weight=1000.0,
            cc_dist=0.05,
            cs_dist=0.02,
            ss_dist=0.04,
            curvature_threshold=40.0,
            checkpoint_every=5,
            topology_gate_fieldlines=6,
            topology_gate_tmax=3.0,
            topology_gate_tol=1e-6,
            topology_gate_survival_threshold=0.5,
            topology_gate_penalty_scale=5.0,
            topology_scorer_every=4,
            topology_scorer_nfieldlines=10,
            topology_scorer_tmax=40.0,
            confinement_objective_weight=0.3,
            confinement_surrogate_worst_k=5,
            confinement_surrogate_early_threshold=0.25,
            confinement_surrogate_mean_weight=0.2,
            confinement_surrogate_worst_weight=0.6,
            confinement_surrogate_early_weight=0.2,
            hardware_search_mode="adaptive",
            hardware_search_soft_iterations=3,
            basin_hops=2,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=6,
            basin_seed=7,
            init_only=False,
        )

    def _minimal_goal_mode_payload(
        self,
        output_root: Path,
        *,
        goal_mode: str,
        result_source: str,
        final_iota: float,
        final_volume: float,
        nonqs_ratio: float,
        boozer_residual: float,
        optimizer_success: bool,
    ) -> dict:
        results_name = (
            "results.json"
            if result_source == "final"
            else "results_best_feasible.partial.json"
        )
        return {
            "command": ["python", "single_stage.py", "--single-stage-goal-mode", goal_mode],
            "results_path": output_root / goal_mode / results_name,
            "result_source": result_source,
            "results": {
                "SINGLE_STAGE_GOAL_MODE": goal_mode,
                "SINGLE_STAGE_GOAL_MODE_IMPL": (
                    "target" if goal_mode == "target" else "frontier_tradeoff_score_v2"
                ),
                "FINAL_IOTA": final_iota,
                "FINAL_VOLUME": final_volume,
                "NONQS_RATIO": nonqs_ratio,
                "BOOZER_RESIDUAL": boozer_residual,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "OPTIMIZER_SUCCESS": optimizer_success,
            },
        }

    def _run_goal_mode_case(
        self,
        module,
        *,
        goal_mode: str,
        output_root: Path,
    ):
        return module.run_goal_mode_case(
            self._make_args(),
            goal_mode=goal_mode,
            stage2_bs_path=Path("seed.json").resolve(),
            output_root=output_root,
        )

    def test_build_single_stage_goal_mode_command_resolves_paths_and_threads_mode(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.equilibria_dir = "eqdir"

        target_command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="target",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/target").resolve(),
        )
        frontier_command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertEqual(
            target_command[target_command.index("--single-stage-goal-mode") + 1],
            "target",
        )
        self.assertEqual(
            frontier_command[frontier_command.index("--single-stage-goal-mode") + 1],
            "frontier",
        )
        self.assertEqual(
            target_command[target_command.index("--stage2-bs-path") + 1],
            str(Path("relative/seed.json").resolve()),
        )
        self.assertEqual(
            target_command[target_command.index("--equilibria-dir") + 1],
            str(Path("eqdir").resolve()),
        )
        self.assertEqual(
            target_command[
                target_command.index("--hardware-search-soft-iterations") + 1
            ],
            "3",
        )
        self.assertEqual(
            target_command[target_command.index("--alm-formulation") + 1],
            "weighted_sum",
        )
        self.assertEqual(
            target_command[target_command.index("--num-surfaces") + 1],
            "2",
        )
        self.assertEqual(
            target_command[target_command.index("--plasma-current-A") + 1],
            "8000.0",
        )
        self.assertEqual(
            target_command[target_command.index("--banana-surf-radius") + 1],
            "0.22",
        )
        self.assertIn("--boozer-stage-refinement", target_command)
        self.assertEqual(
            target_command[target_command.index("--basin-seed") + 1],
            "7",
        )

    def test_goal_mode_comparison_wrapper_defaults_match_single_stage_entrypoint(self):
        module = load_goal_mode_comparison_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_goal_mode_comparison.py",
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "seed.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.cs_dist, 0.015)
        self.assertEqual(args.curvature_threshold, 100.0)

    def test_build_single_stage_goal_mode_command_forwards_frontier_volume_weight(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.frontier_volume_weight = 200.0

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertIn("--frontier-volume-weight", command)
        self.assertEqual(
            command[command.index("--frontier-volume-weight") + 1],
            "200.0",
        )

    def test_build_single_stage_goal_mode_command_omits_zero_frontier_volume_weight(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertNotIn("--frontier-volume-weight", command)

    def test_goal_mode_comparison_wrapper_rejects_stage2_surface_mismatch(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (tmpdir_path / "results.json").write_text(
                json.dumps({"PLASMA_SURF_FILENAME": "other_surface.nc"}),
                encoding="utf-8",
            )
            args = self._make_args()
            args.plasma_surf_filename = "demo.nc"
            args.stage2_bs_path = str(stage2_bs_path)

            with self.assertRaisesRegex(ValueError, "Stage 2 artifact surface mismatch"):
                module.load_validated_stage2_seed_metadata(args)

    def test_goal_mode_comparison_wrapper_rejects_init_only_stage2_seed(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (tmpdir_path / "results.json").write_text(
                json.dumps(
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": True,
                    }
                ),
                encoding="utf-8",
            )
            args = self._make_args()
            args.plasma_surf_filename = "demo.nc"
            args.stage2_bs_path = str(stage2_bs_path)

            with self.assertRaisesRegex(ValueError, "non-init-only Stage 2 artifact"):
                module.load_validated_stage2_seed_metadata(args)

    def test_goal_mode_comparison_wrapper_allows_init_only_stage2_seed_with_override(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            results_path = tmpdir_path / "results.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            results_path.write_text(
                json.dumps(
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": True,
                    }
                ),
                encoding="utf-8",
            )
            args = self._make_args()
            args.plasma_surf_filename = "demo.nc"
            args.stage2_bs_path = str(stage2_bs_path)
            args.allow_init_only_stage2_seed = True

            _, loaded_results_path, loaded_results = (
                module.load_validated_stage2_seed_metadata(args)
            )

        self.assertEqual(loaded_results_path, results_path.resolve())
        self.assertTrue(loaded_results["init_only"])

    def test_build_summary_reports_mode_results_and_deltas(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.output_root = "/tmp/comparison"
        commands_by_mode = {
            "target": ["python", "single_stage.py", "--single-stage-goal-mode", "target"],
            "frontier": ["python", "single_stage.py", "--single-stage-goal-mode", "frontier"],
        }
        mode_payloads = {
            "target": {
                "results_path": Path("/tmp/comparison/target/results.json"),
                "result_source": "final",
                "results": {
                    "SINGLE_STAGE_GOAL_MODE": "target",
                    "SINGLE_STAGE_GOAL_MODE_IMPL": "target",
                    "TERMINATION_MESSAGE": "target_ok",
                    "OPTIMIZER_SUCCESS": True,
                    "FINAL_FEASIBILITY_OK": True,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                    "FINAL_IOTA": 0.15,
                    "FINAL_VOLUME": 0.10,
                    "NONQS_RATIO": 0.012,
                    "BOOZER_RESIDUAL": 0.008,
                    "COIL_LENGTH": 1.7,
                    "MAX_CURVATURE": 39.0,
                    "CURVE_CURVE_MIN_DIST": 0.08,
                    "CURVE_SURFACE_MIN_DIST": 0.03,
                    "SURFACE_VESSEL_MIN_DIST": 0.05,
                    "INVALID_STATE_REJECTS_TOTAL": 2,
                    "TOPOLOGY_GATE_REJECTS": 1,
                    "HARDWARE_REJECTS": 1,
                    "SURFACE_SOLVE_REJECTS": 0,
                    "BEST_FEASIBLE_AVAILABLE": True,
                    "BEST_FEASIBLE_STAGE": "initial",
                    "BEST_FEASIBLE_FINAL_IOTA": 0.151,
                    "BEST_FEASIBLE_FINAL_VOLUME": 0.101,
                    "BEST_FEASIBLE_QA_OBJECTIVE": 0.011,
                    "BEST_FEASIBLE_BOOZER_OBJECTIVE": 0.007,
                    "BEST_FEASIBLE_SEARCH_OBJECTIVE_J": 0.95,
                    "BEST_FEASIBLE_BASE_OBJECTIVE_J": 0.9,
                    "BEST_FEASIBLE_CURVE_CURVE_MIN_DIST": 0.081,
                    "BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST": 0.031,
                    "BEST_FEASIBLE_SURFACE_VESSEL_MIN_DIST": 0.051,
                    "BEST_FEASIBLE_MAX_CURVATURE": 38.5,
                    "BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK": True,
                    "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS": True,
                    "SEARCH_OBJECTIVE_J": 1.0,
                    "OBJECTIVE_J": 1.0,
                    "BASE_OBJECTIVE_J": 1.0,
                },
            },
            "frontier": {
                "results_path": Path("/tmp/comparison/frontier/results.json"),
                "result_source": "best_feasible_partial",
                "results": {
                    "SINGLE_STAGE_GOAL_MODE": "frontier",
                    "SINGLE_STAGE_GOAL_MODE_IMPL": "frontier_tradeoff_score_v2",
                    "TARGET_IOTA": None,
                    "TARGET_VOLUME": None,
                    "BOOZER_SURFACE_TARGET_VOLUMES": [0.10],
                    "TERMINATION_MESSAGE": "frontier_ok",
                    "OPTIMIZER_SUCCESS": True,
                    "FINAL_FEASIBILITY_OK": True,
                    "HARDWARE_CONSTRAINTS_OK": True,
                    "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                    "FINAL_IOTA": 0.18,
                    "FINAL_VOLUME": 0.11,
                    "NONQS_RATIO": 0.014,
                    "BOOZER_RESIDUAL": 0.010,
                    "COIL_LENGTH": 1.72,
                    "MAX_CURVATURE": 39.5,
                    "CURVE_CURVE_MIN_DIST": 0.079,
                    "CURVE_SURFACE_MIN_DIST": 0.029,
                    "SURFACE_VESSEL_MIN_DIST": 0.051,
                    "INVALID_STATE_REJECTS_TOTAL": 1,
                    "TOPOLOGY_GATE_REJECTS": 1,
                    "HARDWARE_REJECTS": 0,
                    "SURFACE_SOLVE_REJECTS": 0,
                    "FRONTIER_TRUST_REJECTS": 1,
                    "FRONTIER_TRUST_OK": True,
                    "FRONTIER_BOOZER_TRUST_THRESHOLD": 1.0e-5,
                    "FRONTIER_BOOZER_TRUST_EXCESS": 0.0,
                    "FRONTIER_BOOZER_TRUST_EXCESS_RATIO": 0.0,
                    "FRONTIER_BOOZER_TRUST_PENALTY_SCALE": 5.0e-5,
                    "FRONTIER_TRUST_PENALTY": 0.0,
                    "FRONTIER_REFERENCE_IOTA": 0.15,
                    "FRONTIER_REFERENCE_VOLUME": 0.10,
                    "FRONTIER_REFERENCE_QA": 0.012,
                    "FRONTIER_REFERENCE_BOOZER": 0.008,
                    "FRONTIER_EFFECTIVE_IOTA_WEIGHT": 1.0,
                    "FRONTIER_EFFECTIVE_VOLUME_WEIGHT": 1.0,
                    "FRONTIER_EFFECTIVE_BOOZER_WEIGHT": 1.0,
                    "FRONTIER_VOLUME_OBJECTIVE": -0.4,
                    "BEST_FEASIBLE_AVAILABLE": True,
                    "BEST_FEASIBLE_STAGE": "final",
                    "BEST_FEASIBLE_FRONTIER_RANK_OBJECTIVE_J": -10.5,
                    "BEST_FEASIBLE_FRONTIER_TRUST_OK": True,
                    "BEST_FEASIBLE_FINAL_IOTA": 0.181,
                    "BEST_FEASIBLE_FINAL_VOLUME": 0.111,
                    "BEST_FEASIBLE_QA_OBJECTIVE": 0.013,
                    "BEST_FEASIBLE_BOOZER_OBJECTIVE": 0.009,
                    "BEST_FEASIBLE_SEARCH_OBJECTIVE_J": -10.5,
                    "BEST_FEASIBLE_BASE_OBJECTIVE_J": -10.2,
                    "BEST_FEASIBLE_CURVE_CURVE_MIN_DIST": 0.08,
                    "BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST": 0.03,
                    "BEST_FEASIBLE_SURFACE_VESSEL_MIN_DIST": 0.052,
                    "BEST_FEASIBLE_MAX_CURVATURE": 39.0,
                    "BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK": True,
                    "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS": True,
                    "SEARCH_OBJECTIVE_J": -10.0,
                    "FRONTIER_RANK_OBJECTIVE_J": -10.0,
                    "OBJECTIVE_J": -10.0,
                    "BASE_OBJECTIVE_J": -10.0,
                },
            },
        }

        summary = module.build_summary(
            args,
            commands_by_mode,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={
                "PLASMA_SURF_FILENAME": "demo.nc",
                "init_only": False,
                "BANANA_CURRENT_A": 12000.0,
                "BANANA_CURRENT_MAX_A": 16000.0,
            },
            mode_payloads=mode_payloads,
        )

        self.assertFalse(summary["search_objective_values_comparable"])
        self.assertFalse(summary["stage2_artifact_init_only"])
        self.assertEqual(summary["stage2_banana_current_a"], 12000.0)
        self.assertEqual(summary["mode_runs"]["target"]["result_source"], "final")
        self.assertEqual(
            summary["mode_runs"]["frontier"]["result_source"],
            "best_feasible_partial",
        )
        self.assertEqual(summary["mode_runs"]["target"]["results"]["goal_mode"], "target")
        self.assertEqual(
            summary["mode_runs"]["frontier"]["results"]["goal_mode_impl"],
            "frontier_tradeoff_score_v2",
        )
        self.assertIsNone(summary["mode_runs"]["frontier"]["results"]["target_iota"])
        self.assertEqual(
            summary["mode_runs"]["frontier"]["results"]["boozer_surface_target_volumes"],
            [0.10],
        )
        self.assertTrue(summary["mode_runs"]["frontier"]["results"]["frontier_trust_ok"])
        self.assertEqual(summary["mode_runs"]["frontier"]["results"]["frontier_trust_penalty"], 0.0)
        self.assertEqual(
            summary["mode_runs"]["frontier"]["results"]["frontier_boozer_trust_penalty_scale"],
            5.0e-5,
        )
        self.assertTrue(summary["mode_runs"]["target"]["results"]["best_feasible_available"])
        self.assertEqual(summary["mode_runs"]["target"]["results"]["invalid_state_rejects_total"], 2)
        self.assertAlmostEqual(summary["comparison"]["frontier_minus_target_final_iota"], 0.03)
        self.assertAlmostEqual(summary["comparison"]["frontier_minus_target_final_volume"], 0.01)
        self.assertAlmostEqual(summary["comparison"]["frontier_minus_target_nonqs_ratio"], 0.002)
        self.assertAlmostEqual(summary["comparison"]["frontier_minus_target_boozer_residual"], 0.002)
        self.assertTrue(summary["comparison"]["both_final_feasibility_ok"])
        self.assertTrue(summary["comparison"]["both_hardware_feasible"])
        self.assertTrue(summary["comparison"]["both_optimizer_success"])

    def test_delta_returns_none_when_either_side_missing(self):
        module = load_goal_mode_comparison_module()

        self.assertIsNone(module._delta(None, 1.0))
        self.assertIsNone(module._delta(1.0, None))

    def test_maybe_load_validated_stage2_seed_metadata_returns_loaded_results_when_present(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "results.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            stage2_results_path.write_text(
                json.dumps({"PLASMA_SURF_FILENAME": "demo.nc"}),
                encoding="utf-8",
            )
            args = self._make_args()
            args.stage2_bs_path = str(stage2_bs_path)
            args.plasma_surf_filename = "demo.nc"

            loaded_bs_path, loaded_results_path, loaded_results = (
                module.maybe_load_validated_stage2_seed_metadata(args)
            )

        self.assertEqual(loaded_bs_path, stage2_bs_path.resolve())
        self.assertEqual(loaded_results_path, stage2_results_path.resolve())
        self.assertEqual(loaded_results["PLASMA_SURF_FILENAME"], "demo.nc")

    def test_run_goal_mode_case_executes_and_loads_results(self):
        module = load_goal_mode_comparison_module()
        output_root = Path(tempfile.mkdtemp())

        with patch.object(module, "run_command") as run_command, patch.object(
            module,
            "discover_single_results_path",
            return_value=output_root / "target" / "results.json",
        ) as discover_results, patch.object(
            module,
            "load_json",
            return_value={"SINGLE_STAGE_GOAL_MODE": "target"},
        ) as load_json:
            payload = self._run_goal_mode_case(
                module,
                goal_mode="target",
                output_root=output_root,
            )

        run_command.assert_called_once()
        discover_results.assert_called_once()
        load_json.assert_called_once()
        self.assertEqual(payload["result_source"], "final")
        self.assertEqual(payload["results"]["SINGLE_STAGE_GOAL_MODE"], "target")

    def test_discover_single_stage_salvage_results_path_skips_stale_best_feasible(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            stale_feasible_dir = output_root / "mpol=8-ntor=6-old"
            stale_feasible_dir.mkdir(parents=True)
            (stale_feasible_dir / "results_best_feasible.partial.json").write_text(
                "{}",
                encoding="utf-8",
            )

            previous_snapshot = module.snapshot_single_stage_preserved_results_paths(
                output_root
            )

            updated_accepted_dir = output_root / "mpol=8-ntor=6-new"
            updated_accepted_dir.mkdir(parents=True)
            updated_accepted_path = (
                updated_accepted_dir / "results_best_accepted.partial.json"
            )
            updated_accepted_path.write_text("{}", encoding="utf-8")

            result_source, results_path = (
                module.discover_single_stage_salvage_results_path(
                    output_root,
                    previous_snapshot=previous_snapshot,
                )
            )

        self.assertEqual(result_source, "best_accepted_partial")
        self.assertEqual(results_path, updated_accepted_path)

    def test_run_goal_mode_case_salvages_best_feasible_partial_when_final_results_are_truncated(self):
        module = load_goal_mode_comparison_module()
        output_root = Path(tempfile.mkdtemp())
        final_results_path = output_root / "frontier" / "mpol=8-ntor=6" / "results.json"
        partial_results_path = (
            output_root
            / "frontier"
            / "mpol=8-ntor=6"
            / "results_best_feasible.partial.json"
        )

        def load_json_side_effect(path):
            if path == final_results_path:
                raise json.JSONDecodeError("bad", "}", 0)
            if path == partial_results_path:
                return {"SINGLE_STAGE_GOAL_MODE": "frontier"}
            raise AssertionError(f"unexpected load_json path: {path}")

        with patch.object(module, "run_command") as run_command, patch.object(
            module,
            "discover_single_results_path",
            return_value=final_results_path,
        ) as discover_results, patch.object(
            module,
            "discover_single_stage_salvage_results_path",
            return_value=("best_feasible_partial", partial_results_path),
        ) as discover_salvage, patch.object(
            module,
            "load_json",
            side_effect=load_json_side_effect,
        ) as load_json:
            payload = self._run_goal_mode_case(
                module,
                goal_mode="frontier",
                output_root=output_root,
            )

        run_command.assert_called_once()
        discover_results.assert_called_once()
        discover_salvage.assert_called_once()
        self.assertEqual(load_json.call_count, 2)
        self.assertEqual(payload["result_source"], "best_feasible_partial")
        self.assertEqual(payload["results_path"], partial_results_path)
        self.assertEqual(payload["results"]["SINGLE_STAGE_GOAL_MODE"], "frontier")

    def test_run_goal_mode_case_salvages_partial_when_run_command_times_out(self):
        module = load_goal_mode_comparison_module()
        output_root = Path(tempfile.mkdtemp())
        partial_results_path = (
            output_root
            / "frontier"
            / "mpol=8-ntor=6"
            / "results_best_feasible.partial.json"
        )

        with patch.object(
            module,
            "run_command",
            side_effect=subprocess.TimeoutExpired(cmd=["python"], timeout=1.0),
        ) as run_command, patch.object(
            module,
            "discover_single_results_path",
            side_effect=FileNotFoundError("no final results"),
        ) as discover_results, patch.object(
            module,
            "discover_single_stage_salvage_results_path",
            return_value=("best_feasible_partial", partial_results_path),
        ) as discover_salvage, patch.object(
            module,
            "load_json",
            return_value={"SINGLE_STAGE_GOAL_MODE": "frontier"},
        ) as load_json:
            payload = self._run_goal_mode_case(
                module,
                goal_mode="frontier",
                output_root=output_root,
            )

        run_command.assert_called_once()
        discover_results.assert_called_once()
        discover_salvage.assert_called_once()
        load_json.assert_called_once_with(partial_results_path)
        self.assertEqual(payload["result_source"], "best_feasible_partial")
        self.assertEqual(payload["results_path"], partial_results_path)
        self.assertEqual(payload["results"]["SINGLE_STAGE_GOAL_MODE"], "frontier")

    def test_run_goal_mode_case_re_raises_timeout_when_no_results_can_be_salvaged(self):
        module = load_goal_mode_comparison_module()
        output_root = Path(tempfile.mkdtemp())
        timeout_error = subprocess.TimeoutExpired(cmd=["python"], timeout=1.0)

        with patch.object(
            module,
            "run_command",
            side_effect=timeout_error,
        ) as run_command, patch.object(
            module,
            "discover_single_results_path",
            side_effect=FileNotFoundError("no final results"),
        ) as discover_results, patch.object(
            module,
            "discover_single_stage_salvage_results_path",
            side_effect=FileNotFoundError("no partial results"),
        ) as discover_salvage:
            with self.assertRaises(subprocess.TimeoutExpired):
                self._run_goal_mode_case(
                    module,
                    goal_mode="frontier",
                    output_root=output_root,
                )

        run_command.assert_called_once()
        discover_results.assert_called_once()
        discover_salvage.assert_called_once()

    def test_discover_single_stage_salvage_results_path_raises_when_no_partials_exist(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with self.assertRaises(FileNotFoundError):
                module.discover_single_stage_salvage_results_path(output_root)

    def test_goal_mode_comparison_dry_run_does_not_require_existing_stage2_artifact(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            missing_stage2_bs_path = tmpdir_path / "missing" / "biot_savart_opt.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_goal_mode_comparison.py",
                    "--dry-run",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(missing_stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                ],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["stage2_bs_path"], str(missing_stage2_bs_path.resolve()))
            self.assertTrue(summary["dry_run"])
            self.assertIn("target", summary["mode_runs"])
            self.assertIn("frontier", summary["mode_runs"])
            self.assertEqual(
                summary["mode_runs"]["target"]["command"][
                    summary["mode_runs"]["target"]["command"].index("--single-stage-goal-mode") + 1
                ],
                "target",
            )
            self.assertEqual(
                summary["mode_runs"]["frontier"]["command"][
                    summary["mode_runs"]["frontier"]["command"].index("--single-stage-goal-mode") + 1
                ],
                "frontier",
            )
            self.assertNotIn("stage2_results_path", summary)

    def test_goal_mode_comparison_main_writes_summary_with_mixed_result_sources(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "stage2" / "results.json"
            stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
            stage2_bs_path.write_text("{}", encoding="utf-8")
            stage2_results_path.write_text("{}", encoding="utf-8")

            target_payload = self._minimal_goal_mode_payload(
                output_root,
                goal_mode="target",
                result_source="final",
                final_iota=0.15,
                final_volume=0.10,
                nonqs_ratio=0.01,
                boozer_residual=1.0e-6,
                optimizer_success=True,
            )
            frontier_payload = self._minimal_goal_mode_payload(
                output_root,
                goal_mode="frontier",
                result_source="best_feasible_partial",
                final_iota=0.16,
                final_volume=0.11,
                nonqs_ratio=0.011,
                boozer_residual=1.1e-6,
                optimizer_success=False,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_goal_mode_comparison.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                ],
            ), patch.object(
                module,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                        "BANANA_CURRENT_A": 12000.0,
                        "BANANA_CURRENT_MAX_A": 16000.0,
                    },
                ),
            ), patch.object(
                module,
                "run_goal_mode_case",
                side_effect=[target_payload, frontier_payload],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["mode_runs"]["target"]["result_source"], "final")
            self.assertEqual(
                summary["mode_runs"]["frontier"]["result_source"],
                "best_feasible_partial",
            )
            self.assertEqual(
                summary["mode_runs"]["frontier"]["results"]["goal_mode_impl"],
                "frontier_tradeoff_score_v2",
            )
            self.assertFalse(summary["comparison"]["both_optimizer_success"])
