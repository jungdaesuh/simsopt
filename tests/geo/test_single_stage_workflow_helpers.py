import importlib.util
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import threading
import time
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
FRONTIER_CAMPAIGN_PATH = EXAMPLE_ROOT / "run_single_stage_frontier_campaign.py"
SINGLE_STAGE_ENTRYPOINT_PATH = EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
STAGE2_ENTRYPOINT_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
IMPORT_PROVENANCE_PATH = EXAMPLE_ROOT / "import_provenance.py"
EXPECTED_LOCAL_SIMSOPT_INIT = (
    Path(__file__).resolve().parents[2] / "src" / "simsopt" / "__init__.py"
)
EXPECTED_FINITE_CURRENT_MODE = "wataru_proxy_field"


def stage2_results_with_digest(stage2_bs_path: Path, payload: dict) -> dict:
    results = dict(payload)
    results.setdefault(
        "STAGE2_BS_SHA256",
        hashlib.sha256(stage2_bs_path.read_bytes()).hexdigest(),
    )
    return results


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


def load_frontier_campaign_module():
    return load_module(
        FRONTIER_CAMPAIGN_PATH,
        "run_single_stage_frontier_campaign",
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
                major_radius=0.976,
                toroidal_flux=1.2,
                length_weight=0.0005,
                cc_weight=100.0,
                cc_threshold=0.05,
                curvature_weight=0.0001,
                curvature_threshold=40.0,
                banana_surf_radius=0.22,
                tf_current_A=-8.0e4,
                order=2,
            )

    def test_validate_stage2_iota_args_rejects_soft_mode_under_alm(self):
        module = load_workflow_helpers_module()

        with self.assertRaisesRegex(
            ValueError,
            "--stage2-iota-mode=soft is incompatible with --constraint-method=alm",
        ):
            module.validate_stage2_iota_args(
                stage2_iota_mode="soft",
                stage2_iota_target=0.2,
                stage2_iota_tolerance=5.0e-3,
                stage2_iota_vol_target=0.1,
                stage2_iota_num_tf_coils=20,
                stage2_iota_nphi=91,
                stage2_iota_ntheta=32,
                stage2_iota_mpol=8,
                stage2_iota_ntor=6,
                stage2_iota_weight=3.0,
                constraint_method="alm",
            )

    def test_format_local_stage2_run_dir_includes_constraint_and_basin_suffix(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
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

        self.assertIn("TFC=-80000", run_dir)
        self.assertIn("INITC=10000", run_dir)
        self.assertIn("-CM=penalty", run_dir)
        self.assertIn("-BH=3-BS=0.01-BSeed=7-BT=2.5-BNS=8", run_dir)

    def test_local_stage2_bs_path_matches_current_stage2_contract(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
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
            "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=10000-MAXC=16000-TFC=-80000-Order=2-CM=penalty/"
            "biot_savart_opt.json",
        )

    def test_local_stage2_bs_path_includes_nondefault_length_target(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
            length_target=1.6,
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

        self.assertIn("-LT=1.6-", str(artifact_path))

    def test_local_stage2_bs_path_includes_nondefault_lcfs_ceiling(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
            target_lcfs_max_major_radius_m=0.91,
            target_lcfs_max_minor_radius_m=0.14,
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

        self.assertIn("-LCFSMR=0.91-LCFSMN=0.14-", str(artifact_path))

    def test_format_local_stage2_run_dir_canonicalizes_exact_iota_constraint_weight(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
        )

        run_dir = module.format_local_stage2_run_dir(
            spec,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            stage2_iota_mode="soft",
            stage2_iota_target=0.2,
            stage2_iota_tolerance=5.0e-3,
            stage2_iota_weight=3.0,
            stage2_iota_vol_target=0.12,
            stage2_iota_constraint_weight=0.0,
            stage2_iota_num_tf_coils=20,
            stage2_iota_nphi=91,
            stage2_iota_ntheta=32,
            stage2_iota_mpol=8,
            stage2_iota_ntor=6,
        )

        self.assertIn("-IM=soft", run_dir)
        self.assertIn("-ITarget=0.2", run_dir)
        self.assertIn("-ITol=0.005", run_dir)
        self.assertIn("-IW=3", run_dir)
        self.assertIn("-IVol=0.12", run_dir)
        self.assertIn("-ICW=exact", run_dir)

    def test_format_local_stage2_run_dir_includes_alm_penalty_cap_when_enabled(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
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

    def test_local_stage2_run_dir_encodes_nondefault_extended_alm_controls(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
        )

        run_dir = module.format_local_stage2_run_dir(
            spec,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=50.0,
            alm_feas_tol=1.0e-4,
            alm_stationarity_tol=2.0e-4,
            alm_trust_radius_init=0.15,
            alm_trust_radius_min=1.0e-3,
            alm_trust_radius_shrink=0.4,
            alm_trust_radius_grow=1.8,
            alm_max_inner_attempts=5,
            alm_max_subproblem_continuations=9,
            alm_distance_smoothing=0.01,
            alm_curvature_smoothing=0.5,
            basin_hops=0,
            basin_stepsize=0.01,
        )

        self.assertIn("-ALMSub=9", run_dir)
        self.assertIn("-ALMFeas=0.0001", run_dir)
        self.assertIn("-ALMStat=0.0002", run_dir)
        self.assertIn("-ALMTR=0.15", run_dir)
        self.assertIn("-ALMInner=5", run_dir)
        self.assertIn("-ALMDist=0.01", run_dir)
        self.assertIn("-ALMCurv=0.5", run_dir)

    def test_format_local_stage2_run_dir_includes_wataru_field_suffix(self):
        module = load_workflow_helpers_module()
        spec = module.Stage2SeedSpec(
            plasma_surf_filename="demo.nc",
            major_radius=0.976,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=-8.0e4,
            order=2,
            finite_current_mode="wataru_proxy_field",
            proxy_plasma_current_A=9000.0,
            vf_current_A=500.0,
            vf_template_path="/tmp/vf_template.json",
        )

        run_dir = module.format_local_stage2_run_dir(
            spec,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
        )

        self.assertIn("-FCM=wataru_proxy_field", run_dir)
        self.assertIn("-PPC=9000", run_dir)
        self.assertIn("-VFC=500", run_dir)
        self.assertIn("-VFT=vf_template", run_dir)

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
                tf_current_A=-8.0e4,
                major_radius=0.976,
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
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
        self.assertIn("--length-target", command)
        self.assertIn("--target-lcfs-max-major-radius-m", command)
        self.assertIn("--target-lcfs-max-minor-radius-m", command)
        self.assertIn("--output-root", command)
        self.assertIn("--init-only", command)

    def test_build_stage2_command_adds_offspec_engineering_flag(self):
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
            curvature_threshold=150.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="penalty",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_seed=None,
            init_only=False,
            length_target=3.0,
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertIn("--allow-offspec-engineering-constraints", command)

    def test_build_stage2_command_threads_extended_basin_controls(self):
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

    def test_build_stage2_command_threads_extended_alm_controls(self):
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
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            order=2,
            constraint_method="alm",
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_penalty_max=50.0,
            alm_feas_tol=1.0e-4,
            alm_stationarity_tol=2.0e-4,
            alm_trust_radius_init=0.15,
            alm_trust_radius_min=1.0e-3,
            alm_trust_radius_shrink=0.4,
            alm_trust_radius_grow=1.8,
            alm_max_inner_attempts=5,
            alm_max_subproblem_continuations=9,
            alm_distance_smoothing=0.01,
            alm_curvature_smoothing=0.05,
            basin_hops=0,
            basin_stepsize=0.01,
            basin_seed=None,
            init_only=False,
        )

        command = module.build_stage2_command(config, python_executable="python")

        for flag, expected in (
            ("--alm-max-subproblem-continuations", "9"),
            ("--alm-feas-tol", "0.0001"),
            ("--alm-stationarity-tol", "0.0002"),
            ("--alm-trust-radius-init", "0.15"),
            ("--alm-trust-radius-min", "0.001"),
            ("--alm-trust-radius-shrink", "0.4"),
            ("--alm-trust-radius-grow", "1.8"),
            ("--alm-max-inner-attempts", "5"),
            ("--alm-distance-smoothing", "0.01"),
            ("--alm-curvature-smoothing", "0.05"),
        ):
            self.assertIn(flag, command)
            self.assertEqual(command[command.index(flag) + 1], expected)

    def test_build_stage2_command_threads_wataru_field_controls(self):
        module = load_workflow_common_module()
        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir="/tmp/equilibria",
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
            init_only=False,
            finite_current_mode="wataru_proxy_field",
            proxy_plasma_current_A=9000.0,
            vf_current_A=500.0,
            vf_template_path="/tmp/vf_template.json",
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertIn("--finite-current-mode", command)
        self.assertEqual(
            command[command.index("--finite-current-mode") + 1],
            "wataru_proxy_field",
        )
        self.assertIn("--proxy-plasma-current-A", command)
        self.assertEqual(command[command.index("--proxy-plasma-current-A") + 1], "9000.0")
        self.assertIn("--vf-current-A", command)
        self.assertEqual(command[command.index("--vf-current-A") + 1], "500.0")
        self.assertIn("--vf-template-path", command)
        self.assertEqual(
            command[command.index("--vf-template-path") + 1],
            "/tmp/vf_template.json",
        )

    def test_build_stage2_command_uses_repo_default_vf_template_for_wataru_mode(self):
        module = load_workflow_common_module()
        helpers = load_workflow_helpers_module()
        config = module.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir="/tmp/equilibria",
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
            init_only=False,
            finite_current_mode="wataru_proxy_field",
            proxy_plasma_current_A=9000.0,
            vf_current_A=500.0,
            vf_template_path=None,
        )

        command = module.build_stage2_command(config, python_executable="python")

        self.assertIn("--vf-template-path", command)
        self.assertEqual(
            command[command.index("--vf-template-path") + 1],
            helpers.default_wataru_vf_template_path(),
        )

    def test_stage2_artifact_config_preserves_raw_template_and_resolves_identity_default(self):
        common = load_workflow_common_module()
        baseline = load_baseline_sweep_module()
        helpers = load_workflow_helpers_module()
        config = common.Stage2ArtifactConfig(
            plasma_surf_filename="demo.nc",
            output_root=Path("/tmp/stage2"),
            equilibria_dir="/tmp/equilibria",
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
            init_only=False,
            finite_current_mode="wataru_proxy_field",
            proxy_plasma_current_A=9000.0,
            vf_current_A=500.0,
            vf_template_path=None,
        )

        self.assertIsNone(config.vf_template_path)
        self.assertEqual(
            config.effective_vf_template_path,
            helpers.default_wataru_vf_template_path(),
        )
        artifact_path = common.resolve_stage2_artifact_path(config)
        self.assertIn("-VFT=wataru_vf_template", str(artifact_path))
        metadata = baseline.expected_locked_baseline_stage2_artifact_metadata(config)
        self.assertEqual(
            metadata["VF_TEMPLATE_PATH"],
            helpers.default_wataru_vf_template_path(),
        )

    def test_locked_baseline_stage2_metadata_includes_basin_identity(self):
        common = load_workflow_common_module()
        module = load_baseline_sweep_module()
        config = common.Stage2ArtifactConfig(
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
            tf_current_A=-8.0e4,
            major_radius=0.976,
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
            "TF_CURRENT_A": -8.0e4,
            "TF_CURRENT_SUM_ABS_A": 1.6e6,
            "NUM_TF_COILS": 20,
            "MAJOR_RADIUS": 0.976,
            "TOROIDAL_FLUX": 0.24,
            "LENGTH_WEIGHT": 0.0005,
            "CC_WEIGHT": 100.0,
            "CC_THRESHOLD": 0.05,
            "CURVATURE_WEIGHT": 0.0001,
            "CURVATURE_THRESHOLD": 100.0,
            "banana_surf_radius": 0.21,
            "order": 2,
            "CONSTRAINT_METHOD": "penalty",
            "CONTRACT_SCHEMA_VERSION": 1,
            "basin_hops": 0,
            "basin_stepsize": None,
            "basin_seed": None,
            "init_only": False,
        }
        results.update(overrides)
        return results

    def _write_stage2_artifact_results(
        self,
        stage2_bs_path: Path,
        stage2_results: dict,
    ) -> Path:
        stage2_results_path = stage2_bs_path.with_name("results.json")
        payload = stage2_results_with_digest(stage2_bs_path, stage2_results)
        stage2_results_path.write_text(json.dumps(payload), encoding="utf-8")
        return stage2_results_path

    def _make_expected_stage2_config(self, common, output_root: Path, **overrides):
        config = {
            "plasma_surf_filename": "demo.nc",
            "output_root": output_root,
            "equilibria_dir": None,
            "tf_current_A": -8.0e4,
            "major_radius": 0.976,
            "toroidal_flux": 0.24,
            "length_weight": 0.0005,
            "cc_weight": 100.0,
            "cc_threshold": 0.05,
            "curvature_weight": 0.0001,
            "curvature_threshold": 100.0,
            "banana_surf_radius": 0.21,
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
            tf_current_A=-8.0e4,
            major_radius=0.976,
            toroidal_flux=0.24,
            stage2_length_weight=0.0005,
            stage2_cc_weight=100.0,
            stage2_cc_threshold=0.05,
            stage2_curvature_weight=0.0001,
            stage2_curvature_threshold=40.0,
            banana_surf_radius=0.21,
            stage2_order=2,
            stage2_constraint_method="penalty",
            stage2_basin_hops=0,
            stage2_basin_stepsize=0.01,
            stage2_basin_seed=-1,
            stage2_init_only=False,
            allow_init_only_stage2_seed=False,
            seed_order_upgrade=None,
            single_stage_constraint_method="penalty",
            single_stage_maxiter=25,
            single_stage_init_only=True,
            single_stage_banana_current_mode="shared",
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
        self.assertEqual(
            command[command.index("--single-stage-banana-current-mode") + 1],
            "shared",
        )

    def test_build_single_stage_command_forwards_seed_order_upgrade(self):
        module = load_baseline_sweep_module()
        helpers = load_workflow_helpers_module()
        args = self._make_args()
        args.seed_order_upgrade = 4
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

        self.assertEqual(command[command.index("--seed-order-upgrade") + 1], "4")

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

    def test_make_stage2_config_rejects_offspec_major_radius(self):
        module = load_baseline_sweep_module()
        args = self._make_args()
        args.major_radius = 0.915

        with self.assertRaisesRegex(ValueError, "vacuum-vessel major radius"):
            module.make_stage2_config(args)

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
            self._write_stage2_artifact_results(
                stage2_bs_path,
                self._make_stage2_artifact_results(
                    TF_CURRENT_A=1.0e5,
                    TF_CURRENT_SUM_ABS_A=2.0e6,
                ),
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
            self._write_stage2_artifact_results(
                stage2_bs_path,
                self._make_stage2_artifact_results(MAJOR_RADIUS=1.23),
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
            self._write_stage2_artifact_results(
                stage2_bs_path,
                self._make_stage2_artifact_results(
                    TF_CURRENT_SUM_ABS_A=8.0e5,
                    NUM_TF_COILS=10,
                ),
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
            stage2_results_path = self._write_stage2_artifact_results(
                stage2_bs_path,
                legacy_results,
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
            stage2_results_path = self._write_stage2_artifact_results(
                stage2_bs_path,
                self._make_stage2_artifact_results(),
            )
            requested_config = self._make_expected_stage2_config(common, stage2_dir)

            summary = module.build_summary(stage2_bs_path, requested_config, records)

        self.assertEqual(summary["experiment_family"], "coil_only_baseline")
        self.assertEqual(summary["plasma_current_locked_A"], 0.0)
        self.assertEqual(summary["tf_current_locked_A"], -8.0e4)
        self.assertEqual(summary["non_dominated_case_names"], ["better"])
        self.assertEqual(summary["stage2_requested_config"]["tf_current_A"], -8.0e4)
        self.assertEqual(summary["stage2_artifact_results"]["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(summary["stage2_results_path"], str(stage2_results_path))
        self.assertEqual(summary["stage2_bs_path"], str(stage2_bs_path))


class WorkflowRunnerCommonArtifactTests(unittest.TestCase):
    def _write_stage2_artifact_pair(
        self,
        root: Path,
        *,
        stage2_results: dict,
        include_digest: bool = True,
    ) -> tuple[Path, Path]:
        stage2_bs_path = root / "biot_savart_opt.json"
        stage2_bs_path.write_text('{"coils": []}', encoding="utf-8")
        stage2_results_path = root / "results.json"
        payload = dict(stage2_results)
        if include_digest:
            payload = stage2_results_with_digest(stage2_bs_path, payload)
        stage2_results_path.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        return stage2_bs_path, stage2_results_path

    def test_load_stage2_artifact_results_accepts_matching_checksum(self):
        module = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path, stage2_results_path = self._write_stage2_artifact_pair(
                stage2_dir,
                stage2_results={},
            )
            expected_digest = module.compute_stage2_bs_sha256(stage2_bs_path)
            stage2_results_path.write_text(
                json.dumps({"STAGE2_BS_SHA256": expected_digest}),
                encoding="utf-8",
            )

            loaded_results_path, loaded_results = module.load_stage2_artifact_results(
                stage2_bs_path
            )

        self.assertEqual(loaded_results_path, stage2_results_path)
        self.assertEqual(loaded_results["STAGE2_BS_SHA256"], expected_digest)

    def test_load_stage2_artifact_results_rejects_missing_checksum(self):
        module = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path, _ = self._write_stage2_artifact_pair(
                stage2_dir,
                stage2_results={},
                include_digest=False,
            )

            with self.assertRaisesRegex(ValueError, "missing STAGE2_BS_SHA256"):
                module.load_stage2_artifact_results(stage2_bs_path)

    def test_load_stage2_artifact_results_rejects_checksum_mismatch(self):
        module = load_workflow_common_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_dir = Path(tmpdir)
            stage2_bs_path, _ = self._write_stage2_artifact_pair(
                stage2_dir,
                stage2_results={"STAGE2_BS_SHA256": "not-the-real-digest"},
            )

            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                module.load_stage2_artifact_results(stage2_bs_path)


class FiniteCurrentSmokeScriptTests(unittest.TestCase):
    def _assert_upgraded_boozer_current_convention(
        self,
        legacy_results,
        *,
        expected_convention,
    ):
        module = load_finite_current_smoke_module()

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["FINITE_CURRENT_MODE"], EXPECTED_FINITE_CURRENT_MODE)
        self.assertEqual(
            upgraded_results["FINITE_CURRENT_MODE_SOURCE"],
            "legacy_assumed_default",
        )
        self.assertEqual(
            upgraded_results["BOOZER_CURRENT_CONVENTION"],
            expected_convention,
        )

    def _make_smoke_results(self, **overrides):
        results = {
            "PLASMA_CURRENT_A": 0.0,
            "PLASMA_CURRENT_INPUT_SOURCE": "physical_A",
            "BOOZER_I": 0.0,
            "EFFECTIVE_CURRENT_MODE": "vacuum",
            "STAGE2_TF_CURRENT_A": -8.0e4,
            "STAGE2_TF_CURRENT_SUM_ABS_A": 1.6e6,
            "FINITE_CURRENT_MODE": EXPECTED_FINITE_CURRENT_MODE,
            "BOOZER_CURRENT_CONVENTION": "mu0",
        }
        results.update(overrides)
        return results

    def _make_args(self):
        return SimpleNamespace(
            python_executable="python",
            plasma_surf_filename="demo.nc",
            equilibria_dir=None,
            stage2_output_root="/tmp/stage2",
            tf_current_A=-8.0e4,
            major_radius=0.976,
            toroidal_flux=0.24,
            stage2_length_weight=0.0005,
            stage2_cc_weight=100.0,
            stage2_cc_threshold=0.05,
            stage2_curvature_weight=0.0001,
            stage2_curvature_threshold=100.0,
            banana_surf_radius=0.21,
            stage2_order=2,
            seed_order_upgrade=None,
            single_stage_banana_current_mode="shared",
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
        self.assertEqual(
            command[command.index("--single-stage-banana-current-mode") + 1],
            "shared",
        )

    def test_build_smoke_command_forwards_seed_order_upgrade(self):
        module = load_finite_current_smoke_module()
        args = self._make_args()
        args.seed_order_upgrade = 4

        command = module.build_smoke_command(
            args,
            current_A=8000.0,
            stage2_bs_path=Path("/tmp/stage2/biot_savart_opt.json"),
            case_output_root=Path("/tmp/smoke/current_8000"),
        )

        self.assertEqual(command[command.index("--seed-order-upgrade") + 1], "4")

    def test_make_stage2_config_rejects_offspec_major_radius(self):
        module = load_finite_current_smoke_module()
        args = self._make_args()
        args.major_radius = 0.915

        with self.assertRaisesRegex(ValueError, "vacuum-vessel major radius"):
            module.make_stage2_config(args)

    def test_validate_smoke_results_checks_current_contract(self):
        module = load_finite_current_smoke_module()
        results = self._make_smoke_results(
            PLASMA_CURRENT_A=-35200.0,
            BOOZER_I=4.0e-7 * math.pi * -35200.0,
            EFFECTIVE_CURRENT_MODE=EXPECTED_FINITE_CURRENT_MODE,
        )

        validation = module.validate_smoke_results(
            results,
            requested_current_A=-35200.0,
            expected_stage2_tf_current_A=-8.0e4,
            expected_stage2_tf_current_sum_abs_A=1.6e6,
        )

        self.assertTrue(validation["passed"])
        self.assertTrue(validation["effective_mode_matches"])

    def test_validate_smoke_results_uses_actual_artifact_tf_current(self):
        module = load_finite_current_smoke_module()
        results = self._make_smoke_results(
            STAGE2_TF_CURRENT_A=-7.0e4,
            STAGE2_TF_CURRENT_SUM_ABS_A=1.4e6,
        )

        validation = module.validate_smoke_results(
            results,
            requested_current_A=0.0,
            expected_stage2_tf_current_A=-7.0e4,
            expected_stage2_tf_current_sum_abs_A=1.4e6,
        )

        self.assertTrue(validation["passed"])

    def test_validate_smoke_results_uses_actual_artifact_total_tf_current(self):
        module = load_finite_current_smoke_module()
        results = self._make_smoke_results()

        validation = module.validate_smoke_results(
            results,
            requested_current_A=0.0,
            expected_stage2_tf_current_A=-8.0e4,
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
            "TF_CURRENT_A": -8.0e4,
            "NUM_TF_COILS": 20,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(legacy_results)

        self.assertEqual(upgraded_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)

    def test_legacy_smoke_artifact_backfills_tf_current_from_known_seed_contract(self):
        module = load_finite_current_smoke_module()
        legacy_results = {
            "NUM_TF_COILS": 20,
        }

        upgraded_results = module.upgrade_legacy_stage2_artifact_results(
            legacy_results,
            known_tf_current_A=-8.0e4,
        )

        self.assertEqual(upgraded_results["TF_CURRENT_A"], -8.0e4)
        self.assertEqual(upgraded_results["TF_CURRENT_SUM_ABS_A"], 1.6e6)
        self.assertEqual(upgraded_results["FINITE_CURRENT_MODE"], EXPECTED_FINITE_CURRENT_MODE)
        self.assertEqual(upgraded_results["BOOZER_CURRENT_CONVENTION"], "mu0")

    def test_legacy_smoke_artifact_preserves_old_boozer_I_convention_when_inferable(self):
        self._assert_upgraded_boozer_current_convention(
            {
                "PLASMA_CURRENT_A": 8000.0,
                "BOOZER_I": 0.0016,
            },
            expected_convention="mu0_over_2pi",
        )

    def test_legacy_smoke_artifact_infers_mu0_convention_from_normalized_api_value(self):
        self._assert_upgraded_boozer_current_convention(
            {
                "PLASMA_CURRENT_A": 8000.0,
                "BOOZER_I": 4.0e-7 * math.pi * 8000.0,
            },
            expected_convention="mu0",
        )

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
            "TF_CURRENT_A": -8.0e4,
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
            stage2_seed_surf_path=None,
            seed_order_upgrade=None,
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
            single_stage_banana_current_mode="shared",
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
            frontier_scalarization_type=None,
            frontier_reference_iota=None,
            frontier_reference_iota_scale=None,
            frontier_reference_volume=None,
            frontier_reference_volume_scale=None,
            frontier_reference_qa=None,
            frontier_reference_boozer=None,
            frontier_boozer_trust_threshold=None,
            frontier_boozer_trust_penalty_scale=None,
            frontier_chebyshev_rho=None,
            frontier_chebyshev_weight_iota=None,
            frontier_chebyshev_weight_volume=None,
            frontier_chebyshev_weight_qa=None,
            frontier_chebyshev_weight_boozer=None,
            epsilon_constraint_qa_max=None,
            epsilon_constraint_boozer_max=None,
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
            target_command[
                target_command.index("--single-stage-banana-current-mode") + 1
            ],
            "shared",
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
        self.assertEqual(args.single_stage_banana_current_mode, "shared")
        self.assertEqual(args.maxcor, module.DEFAULT_LBFGSB_MAXCOR)
        self.assertEqual(args.maxcor, 40)

    def test_goal_mode_comparison_wrapper_parse_args_accepts_seed_order_upgrade(self):
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
                "--seed-order-upgrade",
                "4",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_order_upgrade, 4)

    def test_goal_mode_comparison_wrapper_parse_args_accepts_warm_start_surface_stem(self):
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
                "--warm-start-surface-stem",
                "recovery/surf_best_feasible",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.warm_start_surface_stem, "recovery/surf_best_feasible")

    def test_goal_mode_comparison_wrapper_parse_args_accepts_stage2_seed_surf_path(self):
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
                "--stage2-seed-surf-path",
                "stage2/surf_opt_boozer_surface.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.stage2_seed_surf_path,
            "stage2/surf_opt_boozer_surface.json",
        )

    def test_goal_mode_comparison_wrapper_parse_args_accepts_independent_banana_current_mode(self):
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
                "--single-stage-banana-current-mode",
                "independent",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_banana_current_mode, "independent")

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

    def test_build_single_stage_goal_mode_command_forwards_resume_solver_checkpoint(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.resume_solver_checkpoint = "/tmp/checkpoints/solver_state_checkpoint.json"

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertIn("--resume-solver-checkpoint", command)
        self.assertEqual(
            command[command.index("--resume-solver-checkpoint") + 1],
            "/tmp/checkpoints/solver_state_checkpoint.json",
        )

    def test_build_single_stage_goal_mode_command_forwards_stage2_handoff_flags(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.allow_init_only_stage2_seed = True
        args.equilibrium_path = "eq/demo.nc"
        args.stage2_seed_surf_path = "stage2/surf_opt_boozer_surface.json"
        args.warm_start_surface_stem = "recovery/surf_best_feasible"
        args.seed_order_upgrade = 4
        args.constraint_weight = -1.0
        args.num_tf_coils = 18
        args.stage2_seed_tf_current_A = 12345.0

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="target",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/target").resolve(),
        )

        self.assertIn("--allow-init-only-stage2-seed", command)
        self.assertEqual(
            command[command.index("--equilibrium-path") + 1],
            str(Path("eq/demo.nc").resolve()),
        )
        self.assertEqual(
            command[command.index("--stage2-seed-surf-path") + 1],
            str(Path("stage2/surf_opt_boozer_surface.json").resolve()),
        )
        self.assertEqual(
            command[command.index("--warm-start-surface-stem") + 1],
            str(Path("recovery/surf_best_feasible").resolve()),
        )
        self.assertEqual(
            command[command.index("--seed-order-upgrade") + 1],
            "4",
        )
        self.assertEqual(
            command[command.index("--constraint-weight") + 1],
            "-1.0",
        )
        self.assertEqual(
            command[command.index("--num-tf-coils") + 1],
            "18",
        )
        self.assertEqual(
            command[command.index("--stage2-seed-tf-current-A") + 1],
            "12345.0",
        )

    def test_build_single_stage_goal_mode_command_adds_offspec_flag(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.length_target = 3.0
        args.curvature_threshold = 150.0

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="target",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/target").resolve(),
        )

        self.assertIn("--allow-offspec-engineering-constraints", command)

    def test_build_single_stage_goal_mode_command_forwards_chebyshev_flags(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.frontier_scalarization_type = "achievement_chebyshev_sweep_v1"
        args.frontier_chebyshev_rho = 0.02
        args.frontier_chebyshev_weight_iota = 2.0
        args.frontier_chebyshev_weight_volume = 1.5
        args.frontier_chebyshev_weight_qa = 1.0
        args.frontier_chebyshev_weight_boozer = 0.5

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertEqual(
            command[command.index("--frontier-scalarization-type") + 1],
            "achievement_chebyshev_sweep_v1",
        )
        self.assertEqual(command[command.index("--frontier-chebyshev-rho") + 1], "0.02")
        self.assertEqual(
            command[command.index("--frontier-chebyshev-weight-iota") + 1],
            "2.0",
        )
        self.assertEqual(
            command[command.index("--frontier-chebyshev-weight-volume") + 1],
            "1.5",
        )

    def test_build_single_stage_goal_mode_command_forwards_epsilon_flags(self):
        module = load_goal_mode_comparison_module()
        args = self._make_args()
        args.frontier_scalarization_type = "epsilon_constraint_sweep_v1"
        args.epsilon_constraint_qa_max = 0.011
        args.epsilon_constraint_boozer_max = 0.007

        command = module.build_single_stage_goal_mode_command(
            args,
            goal_mode="frontier",
            stage2_bs_path=Path("relative/seed.json").resolve(),
            case_output_root=Path("outputs/frontier").resolve(),
        )

        self.assertEqual(
            command[command.index("--frontier-scalarization-type") + 1],
            "epsilon_constraint_sweep_v1",
        )
        self.assertEqual(
            command[command.index("--epsilon-constraint-qa-max") + 1],
            "0.011",
        )
        self.assertEqual(
            command[command.index("--epsilon-constraint-boozer-max") + 1],
            "0.007",
        )

    def test_goal_mode_comparison_wrapper_rejects_stage2_surface_mismatch(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            (tmpdir_path / "results.json").write_text(
                json.dumps(
                    stage2_results_with_digest(
                        stage2_bs_path,
                        {"PLASMA_SURF_FILENAME": "other_surface.nc"},
                    )
                ),
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
                    stage2_results_with_digest(
                        stage2_bs_path,
                        {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": True,
                        },
                    )
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
                    stage2_results_with_digest(
                        stage2_bs_path,
                        {
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": True,
                        },
                    )
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
                    "BANANA_CURRENT_A": 12000.0,
                    "BANANA_CURRENT_MODE": "shared",
                    "BANANA_CURRENTS_A": [12000.0],
                    "BANANA_CURRENT_MAX_ABS_A": 12000.0,
                    "BANANA_CURRENT_CONTROL_METRIC": "max_abs",
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
                    "BANANA_CURRENT_A": 15000.0,
                    "BANANA_CURRENT_MODE": "independent",
                    "BANANA_CURRENTS_A": [12000.0, -15000.0],
                    "BANANA_CURRENT_MAX_ABS_A": 15000.0,
                    "BANANA_CURRENT_CONTROL_METRIC": "max_abs",
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
        self.assertEqual(summary["mode_runs"]["target"]["results"]["banana_current_mode"], "shared")
        self.assertEqual(
            summary["mode_runs"]["frontier"]["results"]["banana_currents_a"],
            [12000.0, -15000.0],
        )
        self.assertEqual(
            summary["mode_runs"]["frontier"]["results"]["banana_current_max_abs_a"],
            15000.0,
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

        self.assertIsNone(module.delta(None, 1.0))
        self.assertIsNone(module.delta(1.0, None))

    def test_maybe_load_validated_stage2_seed_metadata_returns_loaded_results_when_present(self):
        module = load_goal_mode_comparison_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "results.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            stage2_results_path.write_text(
                json.dumps(
                    stage2_results_with_digest(
                        stage2_bs_path,
                        {"PLASMA_SURF_FILENAME": "demo.nc"},
                    )
                ),
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

    def test_single_stage_artifact_bundle_from_results_maps_preserved_partial_artifacts(self):
        module = load_goal_mode_comparison_module()
        results_path = Path(
            "/tmp/recovery/mpol=8-ntor=6/results_best_feasible.partial.json"
        )

        bundle = module.single_stage_artifact_bundle_from_results(
            "best_feasible_partial",
            results_path,
        )

        self.assertEqual(
            bundle["bs_path"],
            results_path.with_name("biot_savart_best_feasible.json"),
        )
        self.assertEqual(
            bundle["surface_stem"],
            results_path.with_name("surf_best_feasible"),
        )
        self.assertEqual(
            bundle["outer_boozer_surface_path"],
            results_path.with_name("surf_best_feasible_outer_boozer_surface.json"),
        )

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


class FrontierCampaignScriptTests(unittest.TestCase):
    def _minimal_target_payload(
        self,
        output_root: Path,
    ) -> dict:
        return {
            "command": ["python", "single_stage.py", "--single-stage-goal-mode", "target"],
            "results_path": output_root / "target_baseline" / "target" / "results.json",
            "result_source": "final",
            "results": {
                "SINGLE_STAGE_GOAL_MODE": "target",
                "SINGLE_STAGE_GOAL_MODE_IMPL": "target",
                "FINAL_IOTA": 0.15,
                "FINAL_VOLUME": 0.10,
                "NONQS_RATIO": 0.012,
                "BOOZER_RESIDUAL": 0.008,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "OPTIMIZER_SUCCESS": True,
            },
        }

    def _minimal_frontier_payload(
        self,
        output_root: Path,
        *,
        lane_id: str,
        final_iota: float,
        final_volume: float,
        nonqs_ratio: float,
        boozer_residual: float,
        result_source: str = "final",
    ) -> dict:
        results_name = (
            "results.json"
            if result_source == "final"
            else "results_best_feasible.partial.json"
        )
        return {
            "command": ["python", "single_stage.py", "--single-stage-goal-mode", "frontier"],
            "results_path": (
                output_root / "lanes" / lane_id / "frontier" / results_name
            ),
            "result_source": result_source,
            "results": {
                "SINGLE_STAGE_GOAL_MODE": "frontier",
                "SINGLE_STAGE_GOAL_MODE_IMPL": "frontier_tradeoff_score_v2",
                "TERMINATION_MESSAGE": "ok",
                "OPTIMIZER_SUCCESS": True,
                "FINAL_FEASIBILITY_OK": True,
                "HARDWARE_CONSTRAINTS_OK": True,
                "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                "FINAL_IOTA": final_iota,
                "FINAL_VOLUME": final_volume,
                "NONQS_RATIO": nonqs_ratio,
                "BOOZER_RESIDUAL": boozer_residual,
                "FRONTIER_TRUST_OK": True,
                "FRONTIER_REFERENCE_IOTA": 0.15,
                "FRONTIER_REFERENCE_VOLUME": 0.10,
                "FRONTIER_REFERENCE_QA": 0.012,
                "FRONTIER_REFERENCE_BOOZER": 0.008,
                "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                "SEARCH_OBJECTIVE_J": -1.0,
            },
        }

    def _write_stage2_seed_artifact(
        self,
        tmpdir_path: Path,
        *,
        overrides: dict | None = None,
    ) -> tuple[Path, Path, dict]:
        stage2_results = {
            "PLASMA_SURF_FILENAME": "demo.nc",
            "init_only": False,
            "TF_CURRENT_A": -8.0e4,
            "NUM_TF_COILS": 20,
            "TF_CURRENT_SUM_ABS_A": 1.6e6,
            "SURFACE_VESSEL_MIN_DIST": 0.04,
        }
        if overrides is not None:
            stage2_results.update(overrides)
        stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
        stage2_results_path = tmpdir_path / "stage2" / "results.json"
        stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
        stage2_bs_path.write_text("{}", encoding="utf-8")
        payload = stage2_results_with_digest(stage2_bs_path, stage2_results)
        stage2_results_path.write_text(json.dumps(payload), encoding="utf-8")
        return stage2_bs_path, stage2_results_path, stage2_results

    def test_frontier_campaign_dry_run_writes_manifest_and_summary(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            missing_stage2_bs_path = tmpdir_path / "missing" / "biot_savart_opt.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--dry-run",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(missing_stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                ],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(
                (output_root / "campaign_manifest.json").read_text(encoding="utf-8")
            )
            progress = json.loads(
                (output_root / "campaign_progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["frontier_num_lanes"], 3)
            self.assertEqual(len(summary["frontier_lanes"]), 3)
            self.assertEqual(
                [lane["lane_budget"] for lane in summary["frontier_lanes"]],
                [300, 300, 300],
            )
            self.assertEqual(summary["frontier_archive_size"], 0)
            self.assertEqual(
                summary["recommended_member"],
                {
                    "schema_version": "frontier_campaign_recommended_v1",
                    "recommended_member_id": None,
                    "policy_name": "balanced",
                    "policy_inputs": None,
                    "policy_rationale": None,
                    "policy_score": None,
                    "recommended_metrics": None,
                    "frontier_archive_size": 0,
                },
            )
            self.assertEqual(summary["target_run"]["status"], "dry_run")
            self.assertEqual(
                summary["progress_path"],
                str((output_root / "campaign_progress.json").resolve()),
            )
            self.assertEqual(
                progress["schema_version"],
                "frontier_campaign_progress_v1",
            )
            self.assertEqual(manifest["FRONTIER_ENGINE"], "multilane_local")

    def test_frontier_campaign_main_writes_archive_recommendation_and_survives_failed_lane(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )

            target_payload = self._minimal_target_payload(output_root)
            lane_01 = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_01",
                final_iota=0.165,
                final_volume=0.108,
                nonqs_ratio=0.011,
                boozer_residual=0.0075,
            )
            lane_03 = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_03",
                final_iota=0.19,
                final_volume=0.095,
                nonqs_ratio=0.015,
                boozer_residual=0.011,
                result_source="best_feasible_partial",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[
                    target_payload,
                    lane_01,
                    RuntimeError("lane failed"),
                    lane_03,
                ],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            archive = json.loads(
                (output_root / "frontier_archive.json").read_text(encoding="utf-8")
            )
            recommended = json.loads(
                (output_root / "frontier_recommended.json").read_text(encoding="utf-8")
            )
            progress_payload = json.loads(
                (output_root / "campaign_progress.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary["frontier_archive_size"], 2)
            self.assertEqual(
                [lane["lane_budget"] for lane in summary["frontier_lanes"]],
                [300, 300, 300],
            )
            self.assertIsNotNone(summary["frontier_hypervolume"])
            self.assertEqual(len(summary["frontier_hypervolume_history"]), 3)
            self.assertEqual(summary["frontier_lanes"][1]["status"], "failed")
            self.assertEqual(summary["frontier_lanes"][1]["error_type"], "RuntimeError")
            self.assertEqual(
                [lane["provisional_member_ids"] for lane in summary["frontier_lanes"]],
                [[], [], []],
            )
            self.assertIsNotNone(archive["hypervolume_total"])
            self.assertTrue(
                all(
                    member["hypervolume_contribution"] is not None
                    for member in archive["members"]
                )
            )
            self.assertEqual(archive["best_by_metric"]["iota"]["member_id"].split(":")[-1], "lane_03")
            self.assertEqual(recommended["recommended_member_id"].split(":")[-1], "lane_01")
            self.assertEqual(
                summary["recommended_member"]["recommended_member_id"].split(":")[-1],
                "lane_01",
            )
            self.assertEqual(
                len(progress_payload["provisional_archive_members"]),
                2,
            )
            self.assertTrue(
                all(
                    member["archive_state"] == "provisional"
                    and member["member_id"].endswith(":provisional")
                    for member in progress_payload["provisional_archive_members"]
                )
            )
            self.assertEqual(
                progress_payload["lane_records"][0]["provisional_member_ids"],
                [
                    f"{summary['frontier_campaign_id']}:lane_01:provisional",
                ],
            )
            self.assertEqual(
                progress_payload["lane_records"][2]["provisional_member_ids"],
                [
                    f"{summary['frontier_campaign_id']}:lane_03:provisional",
                ],
            )
            self.assertEqual(
                progress_payload["lane_records"][0]["certified_member_ids"],
                [f"{summary['frontier_campaign_id']}:lane_01"],
            )
            self.assertAlmostEqual(
                summary["target_comparison"]["recommended_minus_target_final_iota"],
                0.015,
            )

    def test_frontier_campaign_runs_independent_seed_lane_group_with_workers(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )
            active_lane_count = 0
            max_active_lane_count = 0
            active_lane_lock = threading.Lock()

            def run_goal_mode_case(args, *, goal_mode, stage2_bs_path, output_root):
                nonlocal active_lane_count
                nonlocal max_active_lane_count
                self.assertEqual(goal_mode, "frontier")
                with active_lane_lock:
                    active_lane_count += 1
                    max_active_lane_count = max(
                        max_active_lane_count,
                        active_lane_count,
                    )
                time.sleep(0.05)
                with active_lane_lock:
                    active_lane_count -= 1
                lane_id = output_root.name
                lane_index = int(lane_id.removeprefix("lane_"))
                return self._minimal_frontier_payload(
                    output_root.parents[1],
                    lane_id=lane_id,
                    final_iota=0.16 + 0.01 * lane_index,
                    final_volume=0.10 + 0.001 * lane_index,
                    nonqs_ratio=0.011,
                    boozer_residual=0.0075,
                )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "2",
                    "--frontier-lane-workers",
                    "2",
                    "--frontier-early-stop-patience-lanes",
                    "0",
                    "--skip-target",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=run_goal_mode_case,
            ) as run_case:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            progress = json.loads(
                (output_root / "campaign_progress.json").read_text(encoding="utf-8")
            )

            self.assertEqual(run_case.call_count, 2)
            self.assertGreaterEqual(max_active_lane_count, 2)
            self.assertEqual(len(summary["frontier_lanes"]), 2)
            self.assertEqual(len(progress["lane_records"]), 2)
            self.assertEqual(summary["frontier_feasible_lane_count"], 2)
            self.assertEqual(summary["frontier_archive_size"], 1)

    def test_frontier_campaign_parallel_seed_group_matches_serial_archive_outputs(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )

            def run_goal_mode_case(args, *, goal_mode, stage2_bs_path, output_root):
                self.assertEqual(goal_mode, "frontier")
                lane_id = output_root.name
                lane_index = int(lane_id.removeprefix("lane_"))
                return self._minimal_frontier_payload(
                    output_root.parents[1],
                    lane_id=lane_id,
                    final_iota=0.16 + 0.01 * lane_index,
                    final_volume=0.104 - 0.001 * lane_index,
                    nonqs_ratio=0.011,
                    boozer_residual=0.0075,
                )

            def run_campaign(*, output_root: Path, summary_path: Path, lane_workers: int):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "run_single_stage_frontier_campaign.py",
                        "--plasma-surf-filename",
                        "demo.nc",
                        "--stage2-bs-path",
                        str(stage2_bs_path),
                        "--output-root",
                        str(output_root),
                        "--summary-json",
                        str(summary_path),
                        "--frontier-num-lanes",
                        "2",
                        "--frontier-lane-workers",
                        str(lane_workers),
                        "--frontier-early-stop-patience-lanes",
                        "0",
                        "--skip-target",
                    ],
                ), patch.object(
                    module.goal_mode_comparison,
                    "load_validated_stage2_seed_metadata",
                    return_value=(
                        stage2_bs_path.resolve(),
                        stage2_results_path.resolve(),
                        stage2_results,
                    ),
                ), patch.object(
                    module.goal_mode_comparison,
                    "run_goal_mode_case",
                    side_effect=run_goal_mode_case,
                ):
                    self.assertEqual(module.main(), 0)
                return json.loads(summary_path.read_text(encoding="utf-8"))

            serial_summary = run_campaign(
                output_root=tmpdir_path / "serial_outputs",
                summary_path=tmpdir_path / "serial_summary.json",
                lane_workers=1,
            )
            parallel_summary = run_campaign(
                output_root=tmpdir_path / "parallel_outputs",
                summary_path=tmpdir_path / "parallel_summary.json",
                lane_workers=2,
            )

            self.assertEqual(
                serial_summary["frontier_feasible_lane_count"],
                parallel_summary["frontier_feasible_lane_count"],
            )
            self.assertEqual(
                serial_summary["frontier_archive_size"],
                parallel_summary["frontier_archive_size"],
            )
            self.assertEqual(
                [
                    member["objective_metrics"]
                    for member in serial_summary["frontier_archive"]["members"]
                ],
                [
                    member["objective_metrics"]
                    for member in parallel_summary["frontier_archive"]["members"]
                ],
            )
            self.assertEqual(
                [
                    entry["archive_size"]
                    for entry in serial_summary["frontier_hypervolume_history"]
                ],
                [
                    entry["archive_size"]
                    for entry in parallel_summary["frontier_hypervolume_history"]
                ],
            )
            self.assertEqual(
                serial_summary["recommended_member"]["recommended_metrics"],
                parallel_summary["recommended_member"]["recommended_metrics"],
            )

    def test_frontier_campaign_parse_args_accepts_v4_resume_contract_flags(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-reference-mode",
                "reference_point_sweep_v1",
                "--frontier-hypervolume-reference",
                "0.15,0.10,0.012,0.008",
                "--frontier-reference-points-file",
                "/tmp/reference_points.json",
                "--frontier-epsilon-spec-file",
                "/tmp/epsilon.json",
                "--resume",
            ]
        )

        self.assertEqual(args.frontier_reference_mode, "reference_point_sweep_v1")
        self.assertEqual(
            args.frontier_hypervolume_reference,
            "0.15,0.10,0.012,0.008",
        )
        self.assertEqual(
            args.frontier_reference_points_file,
            "/tmp/reference_points.json",
        )
        self.assertEqual(
            args.frontier_epsilon_spec_file,
            "/tmp/epsilon.json",
        )
        self.assertEqual(args.frontier_lane_warm_start_mode, "seed")
        self.assertTrue(args.resume)

    def test_frontier_campaign_parse_args_accepts_runtime_calibration_and_early_stop_flags(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-runtime-calibration-profile",
                "canonical_seed_v1",
                "--frontier-early-stop-patience-lanes",
                "4",
                "--frontier-early-stop-min-certified",
                "2",
                "--frontier-early-stop-min-hypervolume-gain",
                "0.0015",
            ]
        )

        self.assertEqual(
            args.frontier_runtime_calibration_profile,
            "canonical_seed_v1",
        )
        self.assertEqual(args.frontier_early_stop_patience_lanes, 4)
        self.assertEqual(args.frontier_early_stop_min_certified, 2)
        self.assertEqual(
            args.frontier_early_stop_min_hypervolume_gain,
            0.0015,
        )

    def test_frontier_campaign_parse_args_accepts_lane_worker_count(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-lane-workers",
                "4",
            ]
        )

        self.assertEqual(args.frontier_lane_workers, 4)
        with self.assertRaises(SystemExit):
            module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    "/tmp/demo/biot_savart_opt.json",
                    "--frontier-lane-workers",
                    "0",
                ]
            )

    def test_frontier_campaign_groups_only_independent_seed_lanes_for_parallel_run(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-num-lanes",
                "3",
            ]
        )
        lane_specs = module.generate_multilane_local_specs(
            num_lanes=3,
            iotas_weight=args.iotas_weight,
            frontier_volume_weight=args.frontier_volume_weight,
            res_weight=args.res_weight,
            lane_budget=args.frontier_lane_budget,
        )

        parallel_groups = module.build_frontier_lane_execution_groups(
            lane_specs,
            lane_records_by_id={},
            warm_start_mode=module.FRONTIER_LANE_WARM_START_MODE_SEED,
            early_stop_patience_lanes=0,
            lane_workers=2,
        )
        serial_groups = module.build_frontier_lane_execution_groups(
            lane_specs,
            lane_records_by_id={},
            warm_start_mode=module.FRONTIER_LANE_WARM_START_MODE_SEED,
            early_stop_patience_lanes=0,
            lane_workers=1,
        )
        reuse_groups = module.build_frontier_lane_execution_groups(
            lane_specs,
            lane_records_by_id={},
            warm_start_mode=module.FRONTIER_LANE_WARM_START_MODE_REUSE_LATEST_CERTIFIED,
            early_stop_patience_lanes=0,
            lane_workers=2,
        )
        early_stop_groups = module.build_frontier_lane_execution_groups(
            lane_specs,
            lane_records_by_id={},
            warm_start_mode=module.FRONTIER_LANE_WARM_START_MODE_SEED,
            early_stop_patience_lanes=2,
            lane_workers=2,
        )

        self.assertEqual(
            [[lane.lane_id for _index, lane in group] for group in parallel_groups],
            [["lane_01", "lane_02", "lane_03"]],
        )
        self.assertEqual(
            [[lane.lane_id for _index, lane in group] for group in serial_groups],
            [["lane_01"], ["lane_02"], ["lane_03"]],
        )
        self.assertEqual(
            [[lane.lane_id for _index, lane in group] for group in reuse_groups],
            [["lane_01"], ["lane_02"], ["lane_03"]],
        )
        self.assertEqual(
            [[lane.lane_id for _index, lane in group] for group in early_stop_groups],
            [["lane_01"], ["lane_02"], ["lane_03"]],
        )

    def test_frontier_campaign_parse_args_accepts_recommendation_and_warm_start_modes(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-recommendation-policy",
                "closest_to_seed",
                "--frontier-lane-warm-start-mode",
                "reuse_latest_certified",
            ]
        )

        self.assertEqual(args.frontier_recommendation_policy, "closest_to_seed")
        self.assertEqual(
            args.frontier_lane_warm_start_mode,
            "reuse_latest_certified",
        )

    def test_frontier_campaign_parse_args_accepts_normalization_and_full_simplex_flags(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-reference-mode",
                "achievement_chebyshev_full_simplex_v1",
                "--frontier-full-simplex-partitions",
                "2",
                "--frontier-normalization-kind",
                "fixed_ideal_nadir_span_with_floor",
                "--frontier-normalization-spec-file",
                "/tmp/frontier_norm.json",
            ]
        )

        self.assertEqual(
            args.frontier_reference_mode,
            "achievement_chebyshev_full_simplex_v1",
        )
        self.assertEqual(args.frontier_full_simplex_partitions, 2)
        self.assertEqual(
            args.frontier_normalization_kind,
            "fixed_ideal_nadir_span_with_floor",
        )
        self.assertEqual(
            args.frontier_normalization_spec_file,
            "/tmp/frontier_norm.json",
        )

    def test_frontier_campaign_manifest_records_resolved_hypervolume_reference_metrics(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-num-lanes",
                "1",
                "--frontier-hypervolume-reference",
                "0.15,0.10,0.012,0.008",
            ]
        )
        lane_specs = [
            module.FrontierLaneSpec(
                lane_id="lane_01",
                scalarization_type="weight_schedule_v1",
                scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
                iotas_weight=150.0,
                frontier_volume_weight=150.0,
                res_weight=1000.0,
                lane_budget=30,
            )
        ]

        manifest = module.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            stage2_results_path=None,
            stage2_results=None,
            lane_specs=lane_specs,
        )

        self.assertEqual(
            manifest["FRONTIER_HYPERVOLUME_REFERENCE_METRICS"],
            {
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        self.assertEqual(
            manifest["PARETO_OBJECTIVE_NORMALIZATION"]["reference_metrics"],
            {
                "iota": 0.15,
                "volume": 0.10,
                "qa_error": 0.012,
                "boozer_residual": 0.008,
            },
        )
        self.assertEqual(
            manifest["PARETO_OBJECTIVE_NORMALIZATION"]["metric_rules"]["iota"],
            {
                "direction": "max",
                "scale_kind": "reference_fraction_with_floor",
                "reference_fraction": 0.25,
                "floor": 0.05,
            },
        )

    def test_frontier_campaign_manifest_records_fixed_ideal_nadir_normalization(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            normalization_spec_path = Path(tmpdir) / "normalization.json"
            normalization_spec_path.write_text(
                json.dumps(
                    {
                        "schema_version": "frontier_pareto_normalization_spec_v1",
                        "ideal_metrics": {
                            "iota": 0.22,
                            "volume": 0.13,
                            "qa_error": 0.008,
                            "boozer_residual": 0.004,
                        },
                        "nadir_metrics": {
                            "iota": 0.14,
                            "volume": 0.09,
                            "qa_error": 0.02,
                            "boozer_residual": 0.012,
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    "/tmp/demo/biot_savart_opt.json",
                    "--frontier-num-lanes",
                    "1",
                    "--frontier-normalization-kind",
                    "fixed_ideal_nadir_span_with_floor",
                    "--frontier-normalization-spec-file",
                    str(normalization_spec_path),
                ]
            )
            lane_specs = [
                module.FrontierLaneSpec(
                    lane_id="lane_01",
                    scalarization_type="weight_schedule_v1",
                    scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
                    iotas_weight=150.0,
                    frontier_volume_weight=150.0,
                    res_weight=1000.0,
                    lane_budget=30,
                )
            ]

            manifest = module.build_frontier_campaign_manifest(
                args,
                campaign_id="campaign",
                stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
                stage2_results_path=None,
                stage2_results=None,
                lane_specs=lane_specs,
            )

        self.assertEqual(
            manifest["PARETO_OBJECTIVE_NORMALIZATION"]["kind"],
            "fixed_ideal_nadir_span_with_floor",
        )
        self.assertEqual(
            manifest["PARETO_OBJECTIVE_NORMALIZATION"]["ideal_metrics"]["iota"],
            0.22,
        )
        self.assertEqual(
            manifest["PARETO_OBJECTIVE_NORMALIZATION"]["nadir_metrics"]["boozer_residual"],
            0.012,
        )

    def test_frontier_campaign_manifest_uses_effective_lane_budget_from_lane_specs(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-num-lanes",
                "1",
            ]
        )
        lane_specs = [
            module.FrontierLaneSpec(
                lane_id="lane_iota",
                scalarization_type="reference_point_sweep_v1",
                scalarization_params={},
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=25,
            )
        ]

        manifest = module.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            stage2_results_path=None,
            stage2_results=None,
            lane_specs=lane_specs,
        )

        self.assertEqual(manifest["LANE_BUDGET"], 25)
        self.assertEqual(manifest["TOTAL_BUDGET"], 25)
        self.assertEqual(manifest["LANE_SPECS"][0]["lane_budget"], 25)

    def test_frontier_campaign_manifest_reports_lane_constraint_mode_family(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-num-lanes",
                "1",
            ]
        )
        lane_specs = [
            module.FrontierLaneSpec(
                lane_id="lane_tradeoff",
                scalarization_type="achievement_chebyshev_sweep_v1",
                scalarization_params={"frontier_chebyshev_rho": 0.02},
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=25,
            )
        ]

        manifest = module.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            stage2_results_path=None,
            stage2_results=None,
            lane_specs=lane_specs,
        )

        self.assertEqual(
            manifest["FRONTIER_CONSTRAINT_MODE"],
            "frontier_achievement_chebyshev_v1",
        )

    def test_frontier_campaign_manifest_uses_null_lane_budget_for_mixed_lane_budgets(self):
        module = load_frontier_campaign_module()

        args = module.parse_args(
            [
                "--plasma-surf-filename",
                "demo.nc",
                "--stage2-bs-path",
                "/tmp/demo/biot_savart_opt.json",
                "--frontier-num-lanes",
                "2",
            ]
        )
        lane_specs = [
            module.FrontierLaneSpec(
                lane_id="lane_iota",
                scalarization_type="reference_point_sweep_v1",
                scalarization_params={},
                iotas_weight=100.0,
                frontier_volume_weight=200.0,
                res_weight=1000.0,
                lane_budget=25,
            ),
            module.FrontierLaneSpec(
                lane_id="lane_volume",
                scalarization_type="reference_point_sweep_v1",
                scalarization_params={},
                iotas_weight=120.0,
                frontier_volume_weight=180.0,
                res_weight=1000.0,
                lane_budget=40,
            ),
        ]

        manifest = module.build_frontier_campaign_manifest(
            args,
            campaign_id="campaign",
            stage2_bs_path=Path("/tmp/demo/biot_savart_opt.json"),
            stage2_results_path=None,
            stage2_results=None,
            lane_specs=lane_specs,
        )

        self.assertIsNone(manifest["LANE_BUDGET"])
        self.assertEqual(manifest["TOTAL_BUDGET"], 65)

    def test_frontier_campaign_resume_reuses_progress_and_only_runs_missing_lanes(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )

            base_args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                ]
            )
            lane_specs = module.generate_multilane_local_specs(
                num_lanes=3,
                iotas_weight=base_args.iotas_weight,
                frontier_volume_weight=base_args.frontier_volume_weight,
                res_weight=base_args.res_weight,
                lane_budget=base_args.frontier_lane_budget,
            )
            campaign_id = "resume123abc"
            module.write_json(
                output_root / "campaign_manifest.json",
                module.build_frontier_campaign_manifest(
                    base_args,
                    campaign_id=campaign_id,
                    stage2_bs_path=stage2_bs_path.resolve(),
                    stage2_results_path=stage2_results_path.resolve(),
                    stage2_results=stage2_results,
                    lane_specs=lane_specs,
                ),
            )

            target_payload = {
                "status": "completed",
                **self._minimal_target_payload(output_root),
            }
            target_payload["results_summary"] = module.goal_mode_comparison.result_metric_subset(
                target_payload["results"]
            )
            lane_01_payload = {
                "status": "completed",
                **self._minimal_frontier_payload(
                    output_root,
                    lane_id="lane_01",
                    final_iota=0.165,
                    final_volume=0.108,
                    nonqs_ratio=0.011,
                    boozer_residual=0.0075,
                ),
            }
            lane_01_payload["results_summary"] = module.goal_mode_comparison.result_metric_subset(
                lane_01_payload["results"]
            )
            lane_01_args = module.build_frontier_lane_args(base_args, lane_specs[0])
            lane_01_contract = module.build_frontier_lane_contract_for_spec(
                lane_01_args,
                lane_specs[0],
                campaign_id=campaign_id,
                stage2_bs_path=stage2_bs_path.resolve(),
                warm_start_source=str(stage2_bs_path.resolve()),
                lane_budget=int(lane_01_args.maxiter),
                lane_index=0,
            )
            lane_01_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
            )
            lane_01_provisional_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
                archive_state=module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
            )
            lane_01_record = module.build_lane_record_from_payload(
                lane_01_contract,
                lane_specs[0],
                int(lane_01_args.maxiter),
                lane_01_payload,
                provisional_archive_member=lane_01_provisional_member,
                archive_member=lane_01_member,
                archive_update={
                    "action": "inserted",
                    "member_id": lane_01_member.member_id,
                    "dominated_members": [],
                },
            )
            module.persist_campaign_progress(
                output_root / "campaign_progress.json",
                campaign_id=campaign_id,
                frontier_version=base_args.frontier_version,
                frontier_engine=base_args.frontier_engine,
                target_payload=target_payload,
                lane_records=[lane_01_record],
                provisional_archive_members=[lane_01_provisional_member],
                archive_members=[lane_01_member],
            )
            progress_payload = json.loads(
                (output_root / "campaign_progress.json").read_text(encoding="utf-8")
            )
            progress_payload["archive_members"] = []
            progress_payload["provisional_archive_members"] = []
            (output_root / "campaign_progress.json").write_text(
                json.dumps(progress_payload, indent=2),
                encoding="utf-8",
            )

            lane_02_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_02",
                final_iota=0.172,
                final_volume=0.107,
                nonqs_ratio=0.0105,
                boozer_residual=0.0070,
            )
            lane_03_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_03",
                final_iota=0.181,
                final_volume=0.101,
                nonqs_ratio=0.0115,
                boozer_residual=0.0085,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[lane_02_payload, lane_03_payload],
            ) as run_case:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            progress = module.load_frontier_campaign_progress(
                output_root / "campaign_progress.json"
            )
            expected_provisional_ids = [
                f"{campaign_id}:lane_01:provisional",
                f"{campaign_id}:lane_02:provisional",
                f"{campaign_id}:lane_03:provisional",
            ]

            self.assertEqual(run_case.call_count, 2)
            self.assertEqual(summary["frontier_campaign_id"], campaign_id)
            self.assertEqual(summary["target_run"]["status"], "completed")
            self.assertEqual(len(summary["frontier_lanes"]), 3)
            self.assertEqual(
                [lane["lane_id"] for lane in summary["frontier_lanes"]],
                ["lane_01", "lane_02", "lane_03"],
            )
            self.assertEqual(summary["frontier_archive_size"], 3)
            self.assertEqual(len(progress.lane_records), 3)
            self.assertEqual(len(progress.provisional_archive_members), 3)
            self.assertEqual(len(progress.archive_members), 3)
            self.assertEqual(
                [member.member_id for member in progress.provisional_archive_members],
                expected_provisional_ids,
            )

    def test_frontier_campaign_resume_preserves_existing_contract_metadata(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            original_stage2_bs_path = tmpdir_path / "stage2-original" / "biot_savart_opt.json"
            original_stage2_results_path = (
                tmpdir_path / "stage2-original" / "results.json"
            )
            new_stage2_bs_path = tmpdir_path / "stage2-new" / "biot_savart_opt.json"
            new_stage2_results_path = tmpdir_path / "stage2-new" / "results.json"
            original_stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
            new_stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
            original_stage2_bs_path.write_text("{}", encoding="utf-8")
            new_stage2_bs_path.write_text("{}", encoding="utf-8")
            original_stage2_results_path.write_text(
                json.dumps(
                    stage2_results_with_digest(
                        original_stage2_bs_path,
                        {"PLASMA_SURF_FILENAME": "demo.nc", "init_only": False},
                    )
                ),
                encoding="utf-8",
            )
            new_stage2_results_path.write_text(
                json.dumps(
                    stage2_results_with_digest(
                        new_stage2_bs_path,
                        {"PLASMA_SURF_FILENAME": "demo.nc", "init_only": False},
                    )
                ),
                encoding="utf-8",
            )

            base_args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(original_stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "1",
                ]
            )
            lane_specs = module.generate_multilane_local_specs(
                num_lanes=1,
                iotas_weight=base_args.iotas_weight,
                frontier_volume_weight=base_args.frontier_volume_weight,
                res_weight=base_args.res_weight,
                lane_budget=base_args.frontier_lane_budget,
            )
            campaign_id = "resume-contract"
            module.write_json(
                output_root / "campaign_manifest.json",
                module.build_frontier_campaign_manifest(
                    base_args,
                    campaign_id=campaign_id,
                    stage2_bs_path=original_stage2_bs_path.resolve(),
                    stage2_results_path=original_stage2_results_path.resolve(),
                    stage2_results={
                        "PLASMA_SURF_FILENAME": "demo.nc",
                        "init_only": False,
                    },
                    lane_specs=lane_specs,
                ),
            )
            module.persist_campaign_progress(
                output_root / "campaign_progress.json",
                campaign_id=campaign_id,
                frontier_version="original_frontier_version",
                frontier_engine="original_frontier_engine",
                target_payload=None,
                lane_records=[],
                provisional_archive_members=[],
                archive_members=[],
            )

            lane_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_01",
                final_iota=0.165,
                final_volume=0.108,
                nonqs_ratio=0.011,
                boozer_residual=0.0075,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(new_stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-version",
                    "mutated_frontier_version",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[lane_payload],
            ):
                self.assertEqual(module.main(), 0)

            progress_payload = json.loads(
                (output_root / "campaign_progress.json").read_text(encoding="utf-8")
            )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(
                progress_payload["frontier_version"],
                "original_frontier_version",
            )
            self.assertEqual(
                progress_payload["frontier_engine"],
                "original_frontier_engine",
            )
            self.assertEqual(
                summary["frontier_version"],
                "original_frontier_version",
            )
            self.assertEqual(
                summary["stage2_bs_path"],
                str(original_stage2_bs_path.resolve()),
            )
            self.assertEqual(
                summary["frontier_lanes"][0]["warm_start_source"],
                str(original_stage2_bs_path.resolve()),
            )

    def test_frontier_campaign_resume_salvages_missing_lane_from_partial_artifact_before_rerun(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )

            base_args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                ]
            )
            lane_specs = module.generate_multilane_local_specs(
                num_lanes=3,
                iotas_weight=base_args.iotas_weight,
                frontier_volume_weight=base_args.frontier_volume_weight,
                res_weight=base_args.res_weight,
                lane_budget=base_args.frontier_lane_budget,
            )
            campaign_id = "resume-salvage"
            module.write_json(
                output_root / "campaign_manifest.json",
                module.build_frontier_campaign_manifest(
                    base_args,
                    campaign_id=campaign_id,
                    stage2_bs_path=stage2_bs_path.resolve(),
                    stage2_results_path=stage2_results_path.resolve(),
                    stage2_results=stage2_results,
                    lane_specs=lane_specs,
                ),
            )

            target_payload = {
                "status": "completed",
                **self._minimal_target_payload(output_root),
            }
            target_payload["results_summary"] = (
                module.goal_mode_comparison.result_metric_subset(
                    target_payload["results"]
                )
            )
            lane_01_payload = {
                "status": "completed",
                **self._minimal_frontier_payload(
                    output_root,
                    lane_id="lane_01",
                    final_iota=0.165,
                    final_volume=0.108,
                    nonqs_ratio=0.011,
                    boozer_residual=0.0075,
                ),
            }
            lane_01_payload["results_summary"] = (
                module.goal_mode_comparison.result_metric_subset(
                    lane_01_payload["results"]
                )
            )
            lane_01_args = module.build_frontier_lane_args(base_args, lane_specs[0])
            lane_01_contract = module.build_frontier_lane_contract_for_spec(
                lane_01_args,
                lane_specs[0],
                campaign_id=campaign_id,
                stage2_bs_path=stage2_bs_path.resolve(),
                warm_start_source=str(stage2_bs_path.resolve()),
                lane_budget=int(lane_01_args.maxiter),
                lane_index=0,
            )
            lane_01_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
            )
            lane_01_provisional_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
                archive_state=module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
            )
            lane_01_record = module.build_lane_record_from_payload(
                lane_01_contract,
                lane_specs[0],
                int(lane_01_args.maxiter),
                lane_01_payload,
                provisional_archive_member=lane_01_provisional_member,
                archive_member=lane_01_member,
                archive_update={
                    "action": "inserted",
                    "member_id": lane_01_member.member_id,
                    "dominated_members": [],
                },
            )
            module.persist_campaign_progress(
                output_root / "campaign_progress.json",
                campaign_id=campaign_id,
                frontier_version=base_args.frontier_version,
                frontier_engine=base_args.frontier_engine,
                target_payload=target_payload,
                lane_records=[lane_01_record],
                provisional_archive_members=[lane_01_provisional_member],
                archive_members=[lane_01_member],
            )

            lane_02_partial = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_02",
                final_iota=0.172,
                final_volume=0.106,
                nonqs_ratio=0.0108,
                boozer_residual=0.0072,
                result_source="best_feasible_partial",
            )
            lane_02_partial_path = (
                output_root
                / "lanes"
                / "lane_02"
                / "frontier"
                / "mpol=8-ntor=6"
                / "results_best_feasible.partial.json"
            )
            lane_02_partial_path.parent.mkdir(parents=True, exist_ok=True)
            lane_02_partial_path.write_text(
                json.dumps(lane_02_partial["results"]),
                encoding="utf-8",
            )

            lane_03_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_03",
                final_iota=0.181,
                final_volume=0.101,
                nonqs_ratio=0.0115,
                boozer_residual=0.0085,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[lane_03_payload],
            ) as run_case:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            lane_02_summary = summary["frontier_lanes"][1]

            self.assertEqual(run_case.call_count, 1)
            self.assertEqual(lane_02_summary["lane_id"], "lane_02")
            self.assertEqual(lane_02_summary["status"], "completed")
            self.assertEqual(
                lane_02_summary["result_source"],
                "best_feasible_partial",
            )

    def test_frontier_campaign_resume_uses_solver_checkpoint_when_results_are_missing(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(tmpdir_path)
            )

            resume_checkpoint = (
                output_root
                / "lanes"
                / "lane_01"
                / "frontier"
                / "mpol=8-ntor=6"
                / "solver_state_checkpoint.json"
            )
            resume_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            resume_checkpoint.write_text("{}", encoding="utf-8")

            seen_resume_checkpoint: dict[str, str | None] = {}

            def run_goal_mode_case(args, *, goal_mode, stage2_bs_path, output_root):
                seen_resume_checkpoint["path"] = args.resume_solver_checkpoint
                self.assertEqual(goal_mode, "frontier")
                return self._minimal_frontier_payload(
                    output_root.parent.parent,
                    lane_id="lane_01",
                    final_iota=0.171,
                    final_volume=0.107,
                    nonqs_ratio=0.0109,
                    boozer_residual=0.0071,
                )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "1",
                    "--skip-target",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=run_goal_mode_case,
            ) as run_case:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(run_case.call_count, 1)
            self.assertEqual(
                Path(seen_resume_checkpoint["path"]).resolve(),
                resume_checkpoint.resolve(),
            )
            self.assertEqual(summary["frontier_lanes"][0]["lane_id"], "lane_01")
            self.assertEqual(summary["frontier_lanes"][0]["status"], "completed")

    def test_frontier_campaign_resume_matches_clean_archive_on_deterministic_smoke_fixture(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(tmpdir_path)
            )

            clean_output_root = tmpdir_path / "clean_outputs"
            clean_summary_path = tmpdir_path / "clean_summary.json"
            clean_target_payload = self._minimal_target_payload(clean_output_root)
            clean_lane_01_payload = self._minimal_frontier_payload(
                clean_output_root,
                lane_id="lane_01",
                final_iota=0.165,
                final_volume=0.108,
                nonqs_ratio=0.011,
                boozer_residual=0.0075,
            )
            clean_lane_02_payload = self._minimal_frontier_payload(
                clean_output_root,
                lane_id="lane_02",
                final_iota=0.172,
                final_volume=0.107,
                nonqs_ratio=0.0105,
                boozer_residual=0.0070,
            )
            clean_lane_03_payload = self._minimal_frontier_payload(
                clean_output_root,
                lane_id="lane_03",
                final_iota=0.181,
                final_volume=0.101,
                nonqs_ratio=0.0115,
                boozer_residual=0.0085,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(clean_output_root),
                    "--summary-json",
                    str(clean_summary_path),
                    "--frontier-num-lanes",
                    "3",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[
                    clean_target_payload,
                    clean_lane_01_payload,
                    clean_lane_02_payload,
                    clean_lane_03_payload,
                ],
            ):
                self.assertEqual(module.main(), 0)

            clean_summary = json.loads(clean_summary_path.read_text(encoding="utf-8"))

            resume_output_root = tmpdir_path / "resume_outputs"
            resume_summary_path = tmpdir_path / "resume_summary.json"
            base_args = module.parse_args(
                [
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(resume_output_root),
                    "--summary-json",
                    str(resume_summary_path),
                    "--frontier-num-lanes",
                    "3",
                ]
            )
            lane_specs = module.generate_multilane_local_specs(
                num_lanes=3,
                iotas_weight=base_args.iotas_weight,
                frontier_volume_weight=base_args.frontier_volume_weight,
                res_weight=base_args.res_weight,
                lane_budget=base_args.frontier_lane_budget,
            )
            campaign_id = "resume-equivalence"
            module.write_json(
                resume_output_root / "campaign_manifest.json",
                module.build_frontier_campaign_manifest(
                    base_args,
                    campaign_id=campaign_id,
                    stage2_bs_path=stage2_bs_path.resolve(),
                    stage2_results_path=stage2_results_path.resolve(),
                    stage2_results=stage2_results,
                    lane_specs=lane_specs,
                ),
            )
            resume_target_payload = {
                "status": "completed",
                **self._minimal_target_payload(resume_output_root),
            }
            resume_target_payload["results_summary"] = (
                module.goal_mode_comparison.result_metric_subset(
                    resume_target_payload["results"]
                )
            )
            resume_lane_01_payload = {
                "status": "completed",
                **self._minimal_frontier_payload(
                    resume_output_root,
                    lane_id="lane_01",
                    final_iota=0.165,
                    final_volume=0.108,
                    nonqs_ratio=0.011,
                    boozer_residual=0.0075,
                ),
            }
            resume_lane_01_payload["results_summary"] = (
                module.goal_mode_comparison.result_metric_subset(
                    resume_lane_01_payload["results"]
                )
            )
            lane_01_args = module.build_frontier_lane_args(base_args, lane_specs[0])
            lane_01_contract = module.build_frontier_lane_contract_for_spec(
                lane_01_args,
                lane_specs[0],
                campaign_id=campaign_id,
                stage2_bs_path=stage2_bs_path.resolve(),
                warm_start_source=str(stage2_bs_path.resolve()),
                lane_budget=int(lane_01_args.maxiter),
                lane_index=0,
            )
            lane_01_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=resume_lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
            )
            lane_01_provisional_member = module.build_archive_member_from_results(
                campaign_id=campaign_id,
                lane_id="lane_01",
                payload=resume_lane_01_payload,
                rerun_contract=lane_01_contract.rerun_contract,
                archive_state=module.FRONTIER_ARCHIVE_STATE_PROVISIONAL,
            )
            lane_01_record = module.build_lane_record_from_payload(
                lane_01_contract,
                lane_specs[0],
                int(lane_01_args.maxiter),
                resume_lane_01_payload,
                provisional_archive_member=lane_01_provisional_member,
                archive_member=lane_01_member,
                archive_update={
                    "action": "inserted",
                    "member_id": lane_01_member.member_id,
                    "dominated_members": [],
                },
            )
            module.persist_campaign_progress(
                resume_output_root / "campaign_progress.json",
                campaign_id=campaign_id,
                frontier_version=base_args.frontier_version,
                frontier_engine=base_args.frontier_engine,
                target_payload=resume_target_payload,
                lane_records=[lane_01_record],
                provisional_archive_members=[lane_01_provisional_member],
                archive_members=[lane_01_member],
            )

            resume_lane_02_payload = self._minimal_frontier_payload(
                resume_output_root,
                lane_id="lane_02",
                final_iota=0.172,
                final_volume=0.107,
                nonqs_ratio=0.0105,
                boozer_residual=0.0070,
            )
            resume_lane_03_payload = self._minimal_frontier_payload(
                resume_output_root,
                lane_id="lane_03",
                final_iota=0.181,
                final_volume=0.101,
                nonqs_ratio=0.0115,
                boozer_residual=0.0085,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(resume_output_root),
                    "--summary-json",
                    str(resume_summary_path),
                    "--frontier-num-lanes",
                    "3",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[resume_lane_02_payload, resume_lane_03_payload],
            ):
                self.assertEqual(module.main(), 0)

            resume_summary = json.loads(
                resume_summary_path.read_text(encoding="utf-8")
            )

            self.assertEqual(
                [
                    member["objective_metrics"]
                    for member in clean_summary["frontier_archive"]["members"]
                ],
                [
                    member["objective_metrics"]
                    for member in resume_summary["frontier_archive"]["members"]
                ],
            )
            self.assertEqual(
                clean_summary["recommended_member"]["recommended_metrics"],
                resume_summary["recommended_member"]["recommended_metrics"],
            )
            self.assertEqual(
                clean_summary["target_comparison"],
                resume_summary["target_comparison"],
            )

    def test_frontier_campaign_early_stop_stops_after_archive_stagnation(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )

            lane_01_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_01",
                final_iota=0.181,
                final_volume=0.111,
                nonqs_ratio=0.0105,
                boozer_residual=0.0070,
            )
            lane_02_payload = self._minimal_frontier_payload(
                output_root,
                lane_id="lane_02",
                final_iota=0.170,
                final_volume=0.106,
                nonqs_ratio=0.0115,
                boozer_residual=0.0078,
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--frontier-num-lanes",
                    "3",
                    "--skip-target",
                    "--frontier-early-stop-patience-lanes",
                    "1",
                    "--frontier-early-stop-min-certified",
                    "1",
                    "--frontier-early-stop-min-hypervolume-gain",
                    "1.0",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module.goal_mode_comparison,
                "run_goal_mode_case",
                side_effect=[lane_01_payload, lane_02_payload],
            ) as run_case:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(run_case.call_count, 2)
            self.assertEqual(
                [lane["lane_id"] for lane in summary["frontier_lanes"]],
                ["lane_01", "lane_02"],
            )
            self.assertTrue(summary["frontier_early_stop"]["triggered"])
            self.assertEqual(
                summary["frontier_early_stop"]["reason"],
                "archive_stagnation",
            )
            self.assertEqual(
                summary["frontier_early_stop"]["stopped_after_lane_id"],
                "lane_02",
            )
            self.assertEqual(summary["frontier_archive_size"], 1)

    def test_frontier_campaign_nsga3_records_generation_summary(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )
            archive_payload = {
                "status": "completed",
                **self._minimal_frontier_payload(
                    output_root,
                    lane_id="gen_0001_cand_0000",
                    final_iota=0.182,
                    final_volume=0.112,
                    nonqs_ratio=0.0104,
                    boozer_residual=0.0069,
                ),
            }
            archive_member = module.build_archive_member_from_results(
                campaign_id="nsga3-campaign",
                lane_id="gen_0001_cand_0000",
                payload=archive_payload,
                rerun_contract={
                    "frontier_engine": "nsga3",
                    "candidate_x": [0.0, 1.0],
                },
            )
            engine_artifacts = SimpleNamespace(
                evaluator_spec={
                    "schema_version": "single_stage_frontier_evaluator_spec_v1",
                    "run_identity": "nsga3-test",
                },
                evaluator_spec_path=str(
                    output_root / "global_engine_nsga3" / "evaluator_spec.json"
                ),
                generation_history=[
                    {
                        "generation": 1,
                        "population_size": 3,
                        "feasible_count": 2,
                        "archive_size": 1,
                        "archive_growth": 1,
                        "cv_min": 0.0,
                        "cv_mean": 0.1,
                        "cv_max": 0.3,
                        "failure_histogram": {"evaluator_candidate_valid": 2},
                        "cache_hits": 3,
                        "cache_misses": 3,
                        "hypervolume": 1.0e-4,
                    }
                ],
                archive_members=[archive_member],
                provisional_archive_members=[],
                population_checkpoint_path=str(
                    output_root / "global_engine_nsga3" / "population_checkpoint.json"
                ),
                generation_history_path=str(
                    output_root / "global_engine_nsga3" / "generation_history.json"
                ),
                engine_stats={
                    "population_size": 3,
                    "generations": 1,
                    "archive_size": 1,
                    "cache_hits": 3,
                    "cache_misses": 3,
                },
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--skip-target",
                    "--frontier-engine",
                    "nsga3",
                    "--frontier-reference-mode",
                    "achievement_chebyshev_full_simplex_v1",
                    "--frontier-full-simplex-partitions",
                    "1",
                    "--frontier-num-lanes",
                    "3",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module,
                "run_nsga3_frontier_campaign",
                return_value=engine_artifacts,
            ) as run_engine:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(run_engine.call_count, 1)
            self.assertEqual(summary["frontier_engine"], "nsga3")
            self.assertEqual(summary["frontier_archive_size"], 1)
            self.assertEqual(summary["frontier_feasible_lane_count"], 2)
            self.assertEqual(summary["frontier_generation_history"][0]["generation"], 1)
            self.assertEqual(
                summary["frontier_hypervolume_history"][0]["lane_id"],
                "generation_0001",
            )
            self.assertEqual(
                summary["frontier_hypervolume_history"][0]["hypervolume"],
                1.0e-4,
            )
            module.validate_frontier_campaign_summary_payload(summary)
            self.assertEqual(
                summary["frontier_evaluator_spec_path"],
                engine_artifacts.evaluator_spec_path,
            )
            self.assertEqual(
                summary["frontier_population_checkpoint_path"],
                engine_artifacts.population_checkpoint_path,
            )
            self.assertEqual(
                summary["frontier_generation_history_path"],
                engine_artifacts.generation_history_path,
            )
            self.assertEqual(
                summary["frontier_engine_stats"]["cache_hits"],
                3,
            )
            self.assertEqual(
                summary["frontier_evaluator_spec"]["schema_version"],
                "single_stage_frontier_evaluator_spec_v1",
            )

    def test_frontier_campaign_nsga3_resume_reuses_saved_engine_artifacts(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            stage2_bs_path, stage2_results_path, stage2_results = (
                self._write_stage2_seed_artifact(
                    tmpdir_path,
                    overrides={
                        "FINAL_IOTA": 0.15,
                        "FINAL_VOLUME": 0.10,
                        "NONQS_RATIO": 0.012,
                        "BOOZER_RESIDUAL": 0.008,
                    },
                )
            )
            archive_payload = {
                "status": "completed",
                **self._minimal_frontier_payload(
                    output_root,
                    lane_id="gen_0001_cand_0000",
                    final_iota=0.182,
                    final_volume=0.112,
                    nonqs_ratio=0.0104,
                    boozer_residual=0.0069,
                ),
            }
            archive_member = module.build_archive_member_from_results(
                campaign_id="nsga3-campaign",
                lane_id="gen_0001_cand_0000",
                payload=archive_payload,
                rerun_contract={
                    "frontier_engine": "nsga3",
                    "candidate_x": [0.0, 1.0],
                },
            )
            progress = module.FrontierCampaignProgress(
                schema_version="frontier_campaign_progress_v1",
                campaign_id="nsga3-campaign",
                frontier_version="frontier_v3_multilane_local_v1",
                frontier_engine="nsga3",
                target_payload=None,
                lane_records=[],
                provisional_archive_members=[],
                archive_members=[archive_member],
            )
            engine_dir = output_root / "global_engine_nsga3"
            engine_dir.mkdir(parents=True, exist_ok=True)
            module.write_frontier_campaign_progress(
                output_root / "campaign_progress.json",
                progress,
            )
            (engine_dir / "evaluator_spec.json").write_text(
                json.dumps(
                    {
                        "schema_version": "single_stage_frontier_evaluator_spec_v1",
                        "args_payload": {
                            "single_stage_goal_mode": "frontier",
                            "frontier_engine": "nsga3",
                        },
                        "stage2_bs_path": str(stage2_bs_path),
                        "stage2_results_path": str(stage2_results_path),
                        "stage2_results": dict(stage2_results),
                        "run_identity": "nsga3-test",
                        "decision_variables": [
                            {
                                "name": "phic(1)",
                                "semantic_role": "phic",
                                "harmonic_index": 1,
                                "lower_bound": -1.0,
                                "upper_bound": 1.0,
                            },
                            {
                                "name": "zs(1)",
                                "semantic_role": "zs",
                                "harmonic_index": 1,
                                "lower_bound": 0.0,
                                "upper_bound": 1.0,
                            },
                        ],
                        "lower_bounds": [-1.0, 0.0],
                        "upper_bounds": [1.0, 1.0],
                        "seed_x": [0.0, 1.0],
                        "reference_metrics": {
                            "iota": 0.15,
                            "volume": 0.10,
                            "qa_error": 0.012,
                            "boozer_residual": 0.008,
                        },
                        "cv_bucket_names": [
                            "surface_solve_failed",
                            "geometry_state_unrestorable",
                            "missing_search_eval",
                            "nonfinite_evaluation",
                            "topology_broken",
                            "topology_deficit",
                            "hardware_violation_ratio",
                            "frontier_trust_excess_ratio",
                        ],
                        "surface_weight_schedule": [1.0],
                        "search_gate": {"surface_gap_threshold": 0.0},
                    }
                ),
                encoding="utf-8",
            )
            (engine_dir / "population_checkpoint.json").write_text(
                json.dumps(
                    {
                        "population_size": 3,
                        "generations": 1,
                        "ref_dirs": [[1.0, 0.0, 0.0, 0.0]],
                        "X": [[0.0, 1.0]],
                        "F": [[-0.182, -0.112, 0.0104, 0.0069]],
                    }
                ),
                encoding="utf-8",
            )
            (engine_dir / "generation_history.json").write_text(
                json.dumps(
                    [
                        {
                            "generation": 1,
                            "population_size": 3,
                            "feasible_count": 2,
                            "archive_size": 1,
                            "archive_growth": 1,
                            "cv_min": 0.0,
                            "cv_mean": 0.1,
                            "cv_max": 0.3,
                            "failure_histogram": {
                                "evaluator_candidate_valid": 2
                            },
                            "cache_hits": 3,
                            "cache_misses": 3,
                            "hypervolume": 1.0e-4,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_frontier_campaign.py",
                    "--plasma-surf-filename",
                    "demo.nc",
                    "--stage2-bs-path",
                    str(stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                    "--skip-target",
                    "--frontier-engine",
                    "nsga3",
                    "--frontier-reference-mode",
                    "achievement_chebyshev_full_simplex_v1",
                    "--frontier-full-simplex-partitions",
                    "1",
                    "--frontier-num-lanes",
                    "3",
                    "--resume",
                ],
            ), patch.object(
                module.goal_mode_comparison,
                "load_validated_stage2_seed_metadata",
                return_value=(
                    stage2_bs_path.resolve(),
                    stage2_results_path.resolve(),
                    stage2_results,
                ),
            ), patch.object(
                module,
                "run_nsga3_frontier_campaign",
            ) as run_engine:
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))

            self.assertEqual(run_engine.call_count, 0)
            self.assertEqual(summary["frontier_engine"], "nsga3")
            self.assertEqual(summary["frontier_archive_size"], 1)
            self.assertEqual(summary["frontier_feasible_lane_count"], 2)
            self.assertEqual(
                Path(summary["frontier_population_checkpoint_path"]).resolve(),
                (engine_dir / "population_checkpoint.json").resolve(),
            )

    def test_resolve_frontier_lane_warm_start_reuses_latest_certified_final_artifact(self):
        module = load_frontier_campaign_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            base_stage2_bs_path = tmpdir_path / "stage2" / "biot_savart_opt.json"
            base_stage2_bs_path.parent.mkdir(parents=True, exist_ok=True)
            base_stage2_bs_path.write_text("{}", encoding="utf-8")

            lane_results_path = (
                tmpdir_path / "outputs" / "lanes" / "lane_01" / "frontier" / "results.json"
            )
            lane_results_path.parent.mkdir(parents=True, exist_ok=True)
            lane_results_path.write_text("{}", encoding="utf-8")
            warmed_bs_path = lane_results_path.with_name("biot_savart_opt.json")
            warmed_bs_path.write_text("{}", encoding="utf-8")

            lane_contract = module.build_frontier_lane_contract(
                campaign_id="campaign",
                lane_id="lane_01",
                engine="multilane_local",
                scalarization_type="weight_schedule_v1",
                scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
                constraint_mode="frontier_v2_single_lane_contract",
                warm_start_source=str(base_stage2_bs_path),
                optimizer_budget=25,
                rng_seed=0,
                rerun_contract={},
            )
            lane_record = module.build_frontier_lane_record(
                lane_contract,
                command=["python"],
                weights={
                    "iotas_weight": 150.0,
                    "frontier_volume_weight": 150.0,
                    "res_weight": 1000.0,
                },
                lane_budget=25,
                status="completed",
                result_source="final",
                success=True,
                archive_state="certified",
                archive_member=module.build_archive_member_from_results(
                    campaign_id="campaign",
                    lane_id="lane_01",
                    payload={
                        "result_source": "final",
                        "results_path": str(lane_results_path),
                        "results": {
                            "FINAL_IOTA": 0.17,
                            "FINAL_VOLUME": 0.11,
                            "NONQS_RATIO": 0.011,
                            "BOOZER_RESIDUAL": 0.007,
                            "FINAL_FEASIBILITY_OK": True,
                            "HARDWARE_CONSTRAINTS_OK": True,
                            "FINAL_TOPOLOGY_GATE_SUCCESS": True,
                            "FRONTIER_TRUST_OK": True,
                            "FRONTIER_REFERENCE_IOTA": 0.15,
                            "FRONTIER_REFERENCE_VOLUME": 0.10,
                            "FRONTIER_REFERENCE_QA": 0.012,
                            "FRONTIER_REFERENCE_BOOZER": 0.008,
                            "FRONTIER_RANK_OBJECTIVE_J": -1.0,
                            "OPTIMIZER_SUCCESS": True,
                            "TERMINATION_MESSAGE": "ok",
                        },
                    },
                    rerun_contract={},
                ),
                results_path=str(lane_results_path),
                results={"OPTIMIZER_SUCCESS": True},
            )
            lane_specs = [
                module.FrontierLaneSpec(
                    lane_id="lane_01",
                    scalarization_type="weight_schedule_v1",
                    scalarization_params={"iota_share": 0.5, "volume_share": 0.5},
                    iotas_weight=150.0,
                    frontier_volume_weight=150.0,
                    res_weight=1000.0,
                    lane_budget=25,
                ),
                module.FrontierLaneSpec(
                    lane_id="lane_02",
                    scalarization_type="weight_schedule_v1",
                    scalarization_params={"iota_share": 0.7, "volume_share": 0.3},
                    iotas_weight=210.0,
                    frontier_volume_weight=90.0,
                    res_weight=1000.0,
                    lane_budget=25,
                ),
            ]

            warm_start_path, warm_start_source = (
                module.resolve_frontier_lane_warm_start(
                    base_stage2_bs_path=base_stage2_bs_path,
                    lane_records_by_id={"lane_01": lane_record},
                    lane_specs=lane_specs,
                    lane_index=1,
                    warm_start_mode="reuse_latest_certified",
                )
            )

            self.assertEqual(warm_start_path, warmed_bs_path.resolve())
            self.assertEqual(warm_start_source, str(warmed_bs_path.resolve()))
