import argparse
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path, PurePosixPath


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "runpod_single_stage_continuation.py"
)


def load_module_globals():
    return runpy.run_path(str(MODULE_PATH))


class RunpodSingleStageContinuationTests(unittest.TestCase):
    def test_map_local_path_to_remote(self):
        module = load_module_globals()
        local_root = Path("/tmp/columbia")
        local_path = local_root / "autoresearch" / "results" / "case"

        remote_path = module["map_local_path_to_remote"](
            local_path,
            local_columbia_root=local_root,
            remote_columbia_root=PurePosixPath("/workspace/columbia"),
        )

        self.assertEqual(
            remote_path,
            PurePosixPath("/workspace/columbia/autoresearch/results/case"),
        )

    def test_parse_runpod_ssh_info(self):
        module = load_module_globals()

        ssh_info = module["parse_runpod_ssh_info"](
            {
                "id": "pod123",
                "ip": "157.157.221.29",
                "port": 34557,
                "ssh_key": {"path": "/tmp/key"},
            },
            user="root",
        )

        self.assertEqual(ssh_info.target, "root@157.157.221.29")
        self.assertEqual(ssh_info.port, 34557)
        self.assertEqual(ssh_info.pod_id, "pod123")

    def test_resolve_remote_jax_profile_dir_relative_to_run_root(self):
        module = load_module_globals()

        resolved = module["resolve_remote_jax_profile_dir"](
            remote_run_root=PurePosixPath("/workspace/continuation-runs/continuation-run-001"),
            requested_profile_dir="xprof",
        )

        self.assertEqual(
            resolved,
            PurePosixPath(
                "/workspace/continuation-runs/continuation-run-001/xprof"
            ),
        )

    def test_build_repo_tar_command_has_expected_excludes(self):
        module = load_module_globals()

        command = module["build_repo_tar_command"](
            local_columbia_root=Path("/tmp/columbia"),
            repo_relative_path=Path("simsopt-jax"),
        )

        self.assertEqual(command[:1], ["tar"])
        self.assertIn(".git", command)
        self.assertIn(".DS_Store", command)
        self.assertIn("._*", command)
        self.assertIn(".venv-simsopt-jax", command)
        self.assertEqual(command[-1], "simsopt-jax")

    def test_build_repo_tar_env_disables_macos_copyfile_metadata(self):
        module = load_module_globals()

        env = module["build_repo_tar_env"]()

        self.assertEqual(env["COPYFILE_DISABLE"], "1")
        self.assertEqual(env["COPY_EXTENDED_ATTRIBUTES_DISABLE"], "1")

    def test_build_remote_repo_extract_command_wraps_remote_shell_payload(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir="/workspace/continuation-runs/continuation-run-001/xprof",
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=None,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )
        ssh_info = module["SshInfo"](
            host="157.157.221.29",
            port=34557,
            key_path=Path("/tmp/key"),
            user="root",
            pod_id="pod123",
        )

        command = module["build_remote_repo_extract_command"](plan, ssh_info)

        self.assertEqual(command[-2], "root@157.157.221.29")
        self.assertTrue(command[-1].startswith("bash -lc "))
        self.assertIn("mkdir -p /workspace/columbia", command[-1])
        self.assertIn(
            "mv /workspace/columbia/simsopt-jax/build "
            "/workspace/columbia/.runpod-build-cache-simsopt-jax",
            command[-1],
        )
        self.assertIn("rm -rf /workspace/columbia/simsopt-jax", command[-1])
        self.assertIn(
            "mv /workspace/columbia/.runpod-build-cache-simsopt-jax "
            "/workspace/columbia/simsopt-jax/build",
            command[-1],
        )
        self.assertIn("tar -xzf - -C /workspace/columbia", command[-1])

    def test_build_remote_execution_script_threads_thresholds_and_paths(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir="/workspace/continuation-runs/continuation-run-001/xprof",
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )

        script = module["build_remote_execution_script"](plan)

        self.assertIn(
            "RUN_LOCK_DIR=/workspace/continuation-runs/continuation-run-001.lock",
            script,
        )
        self.assertIn("trap cleanup_run_lock EXIT", script)
        self.assertIn('if ! mkdir "${RUN_LOCK_DIR}" 2>/dev/null; then', script)
        self.assertIn('printf "%s\\n" "$$" > "${RUN_LOCK_DIR}/pid"', script)
        self.assertIn("Miniforge3-Linux-x86_64.sh", script)
        self.assertIn('python -m pip install -e ".[JAX_GPU,dev]"', script)
        self.assertIn(
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT=0.0.dev0+gdeadbeef",
            script,
        )
        self.assertIn('ENV_SPEC_HASH="$(sha256sum "${ENV_SPEC_PATH}"', script)
        self.assertIn(
            'echo "Conda env ${ENV_NAME} already matches ${ENV_SPEC_PATH}; '
            'skipping env update."',
            script,
        )
        self.assertIn('printf "%s\\n" "${ENV_SPEC_HASH}" > "${ENV_HASH_PATH}"', script)
        self.assertLess(
            script.index("SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SIMSOPT"),
            script.index("conda env"),
        )
        self.assertIn("--initial-warm-start-run-dir", script)
        self.assertIn("/workspace/columbia/autoresearch/run", script)
        self.assertIn("--max-final-field-error 0.0005", script)
        self.assertIn("--max-final-abs-iota-error 0.02", script)
        self.assertIn("--max-final-non-qs 0.05", script)
        self.assertIn(
            "--jax-profile-dir /workspace/continuation-runs/continuation-run-001/xprof",
            script,
        )
        self.assertIn("--strict-validation", script)

    def test_build_remote_execution_script_supports_resume_mode(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir="/workspace/continuation-runs/continuation-run-001/xprof",
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="resume",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=None,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )

        script = module["build_remote_execution_script"](plan)

        self.assertIn("--resume-run-root /workspace/continuation-runs/continuation-run-001", script)
        self.assertIn(
            "--jax-profile-dir /workspace/continuation-runs/continuation-run-001/xprof",
            script,
        )
        self.assertNotIn("--output-root /workspace/continuation-runs --run-id run-001", script)

    def test_resolve_remote_profiling_report_path_uses_single_run_name(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=None,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )

        resolved = module["resolve_remote_profiling_report_path"](plan)

        self.assertEqual(
            resolved,
            PurePosixPath(
                "/workspace/continuation-runs/continuation-run-001/continuation_profiling_report.md"
            ),
        )

    def test_resolve_remote_profiling_report_path_uses_campaign_name(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run-a"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run-a",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/campaign-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/campaign-run-001/campaign_summary.json",
            remote_validation_path=None,
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
            local_donor_run_dirs=(
                Path("/tmp/columbia/autoresearch/run-a"),
                Path("/tmp/columbia/autoresearch/run-b"),
            ),
            remote_donor_run_dirs=(
                "/workspace/columbia/autoresearch/run-a",
                "/workspace/columbia/autoresearch/run-b",
            ),
        )

        resolved = module["resolve_remote_profiling_report_path"](plan)

        self.assertEqual(
            resolved,
            PurePosixPath(
                "/workspace/continuation-runs/campaign-run-001/campaign_profiling_report.md"
            ),
        )

    def test_build_remote_execution_script_uses_campaign_mode_for_multiple_donors(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run-a"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run-a",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/campaign-run-001",
            remote_jax_profile_dir="/workspace/continuation-runs/campaign-run-001/xprof",
            remote_summary_path="/workspace/continuation-runs/campaign-run-001/campaign_summary.json",
            remote_validation_path=None,
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
            local_donor_run_dirs=(
                Path("/tmp/columbia/autoresearch/run-a"),
                Path("/tmp/columbia/autoresearch/run-b"),
            ),
            remote_donor_run_dirs=(
                "/workspace/columbia/autoresearch/run-a",
                "/workspace/columbia/autoresearch/run-b",
            ),
        )

        script = module["build_remote_execution_script"](plan)

        self.assertIn("--campaign-donor-run-dir /workspace/columbia/autoresearch/run-a", script)
        self.assertIn("--campaign-donor-run-dir /workspace/columbia/autoresearch/run-b", script)
        self.assertNotIn("--initial-warm-start-run-dir", script)
        self.assertIn("--output-root /workspace/continuation-runs --run-id run-001", script)

    def test_resolve_launch_plan_maps_paths_and_run_root(self):
        module = load_module_globals()

        with tempfile.TemporaryDirectory() as tmpdir:
            columbia_root = Path(tmpdir)
            repo_root = columbia_root / "simsopt-jax"
            repo_root.mkdir()
            donor_dir = (
                columbia_root
                / "autoresearch"
                / "single_stage_results"
                / "outputs-wout_nfp5ginsburg_desc_iota21.nc"
                / "seed-run"
            )
            donor_dir.mkdir(parents=True)
            for filename in ("results.json", "biot_savart_opt.json", "surf_opt.json"):
                (donor_dir / filename).write_text("{}", encoding="utf-8")
            equilibrium_path = (
                columbia_root / "DATABASE" / "EQUILIBRIA" / "wout_nfp5ginsburg_desc_iota21.nc"
            )
            equilibrium_path.parent.mkdir(parents=True)
            equilibrium_path.write_text("", encoding="utf-8")

            args = argparse.Namespace(
                pod_id="pod123",
                run_id="run-xyz",
                pretend_version="0.0.dev0+gdeadbeef",
                local_columbia_root=str(columbia_root),
                local_repo_root=str(repo_root),
                equilibrium_path=str(equilibrium_path),
                donor_run_dir=str(donor_dir),
                plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
                remote_columbia_root="/workspace/columbia",
                remote_output_root="/workspace/continuation-runs",
                remote_cache_dir="/workspace/.jax-cache/cont",
                jax_profile_dir="xprof",
                remote_conda_root="/opt/conda",
                system_deps_mode="auto",
                backend_mode="jax_gpu_parity",
                transfer_guard="disallow",
                trial_policy="validated-fast",
                resume_existing_run=False,
                summarize_existing_run=False,
                max_final_field_error=5e-4,
                max_final_abs_iota_error=None,
                max_final_non_qs=0.05,
                mpol=8,
                ntor=6,
                nphi=255,
                ntheta=64,
                maxiter=300,
                coarse_maxiter=1,
                medium_maxiter=1,
                prefinal_maxiter=2,
                fetch_full_run_root=False,
                local_output_root=str(columbia_root / ".artifacts"),
            )

            plan = module["resolve_launch_plan"](args)

        self.assertEqual(plan.run_id, "run-xyz")
        self.assertEqual(plan.pretend_version, "0.0.dev0+gdeadbeef")
        self.assertEqual(
            plan.remote_repo_root,
            "/workspace/columbia/simsopt-jax",
        )
        self.assertEqual(
            plan.remote_equilibrium_path,
            "/workspace/columbia/DATABASE/EQUILIBRIA/wout_nfp5ginsburg_desc_iota21.nc",
        )
        self.assertEqual(
            plan.remote_run_root,
            "/workspace/continuation-runs/continuation-run-xyz",
        )
        self.assertEqual(
            plan.remote_jax_profile_dir,
            "/workspace/continuation-runs/continuation-run-xyz/xprof",
        )
        self.assertEqual(plan.run_mode, "new")

    def test_resolve_launch_plan_supports_multiple_donors(self):
        module = load_module_globals()

        with tempfile.TemporaryDirectory() as tmpdir:
            columbia_root = Path(tmpdir)
            repo_root = columbia_root / "simsopt-jax"
            repo_root.mkdir()
            donor_dirs = []
            for name in ("seed-a", "seed-b"):
                donor_dir = (
                    columbia_root
                    / "autoresearch"
                    / "single_stage_results"
                    / "outputs-wout_nfp5ginsburg_desc_iota21.nc"
                    / name
                )
                donor_dir.mkdir(parents=True)
                for filename in ("results.json", "biot_savart_opt.json", "surf_opt.json"):
                    (donor_dir / filename).write_text("{}", encoding="utf-8")
                donor_dirs.append(str(donor_dir))
            equilibrium_path = (
                columbia_root / "DATABASE" / "EQUILIBRIA" / "wout_nfp5ginsburg_desc_iota21.nc"
            )
            equilibrium_path.parent.mkdir(parents=True)
            equilibrium_path.write_text("", encoding="utf-8")

            args = argparse.Namespace(
                pod_id="pod123",
                run_id="run-xyz",
                pretend_version="0.0.dev0+gdeadbeef",
                local_columbia_root=str(columbia_root),
                local_repo_root=str(repo_root),
                equilibrium_path=str(equilibrium_path),
                donor_run_dir=donor_dirs,
                plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
                remote_columbia_root="/workspace/columbia",
                remote_output_root="/workspace/continuation-runs",
                remote_cache_dir="/workspace/.jax-cache/cont",
                jax_profile_dir="xprof",
                remote_conda_root="/opt/conda",
                system_deps_mode="auto",
                backend_mode="jax_gpu_parity",
                transfer_guard="disallow",
                trial_policy="validated-fast",
                resume_existing_run=False,
                summarize_existing_run=False,
                max_final_field_error=5e-4,
                max_final_abs_iota_error=None,
                max_final_non_qs=0.05,
                mpol=8,
                ntor=6,
                nphi=255,
                ntheta=64,
                maxiter=300,
                coarse_maxiter=1,
                medium_maxiter=1,
                prefinal_maxiter=2,
                fetch_full_run_root=False,
                local_output_root=str(columbia_root / ".artifacts"),
            )

            plan = module["resolve_launch_plan"](args)

        self.assertEqual(plan.remote_run_root, "/workspace/continuation-runs/campaign-run-xyz")
        self.assertEqual(
            plan.remote_summary_path,
            "/workspace/continuation-runs/campaign-run-xyz/campaign_summary.json",
        )
        self.assertIsNone(plan.remote_validation_path)
        self.assertEqual(len(plan.local_donor_run_dirs), 2)

    def test_resolve_local_single_stage_scan_root_prefers_fetched_run_root(self):
        module = load_module_globals()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "out"
            fetched_run_root = output_dir / "continuation-run-001"
            fetched_run_root.mkdir(parents=True)
            plan = module["LaunchPlan"](
                pod_id="pod123",
                run_id="run-001",
                pretend_version="0.0.dev0+gdeadbeef",
                local_columbia_root=Path("/tmp/columbia"),
                local_repo_root=Path("/tmp/columbia/simsopt-jax"),
                local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
                local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
                local_output_dir=output_dir,
                remote_columbia_root="/workspace/columbia",
                remote_repo_root="/workspace/columbia/simsopt-jax",
                remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
                plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
                system_deps_mode="auto",
                backend_mode="jax_gpu_parity",
                transfer_guard="disallow",
                trial_policy="validated-fast",
                run_mode="new",
                max_final_field_error=5e-4,
                max_final_abs_iota_error=0.02,
                max_final_non_qs=0.05,
                mpol=8,
                ntor=6,
                nphi=255,
                ntheta=64,
                maxiter=300,
                coarse_maxiter=1,
                medium_maxiter=1,
                prefinal_maxiter=2,
                fetch_full_run_root=True,
            )

            scan_root = module["resolve_local_single_stage_scan_root"](plan)

        self.assertEqual(scan_root, fetched_run_root)

    def test_build_candidate_ledger_command_threads_thresholds_and_paths(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path(
                "/tmp/columbia/autoresearch/single_stage_results/seed-run"
            ),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/single_stage_results/seed-run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="resume",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )

        command = module["build_candidate_ledger_command"](plan)

        self.assertIn("candidate_ledger.py", command[1])
        self.assertIn("/tmp/columbia/autoresearch/single_stage_results", command)
        self.assertIn("/tmp/out", command)
        self.assertIn("--single-stage-max-final-field-error", command)
        self.assertIn("--single-stage-max-final-abs-iota-error", command)
        self.assertIn("--single-stage-max-final-non-qs", command)
        self.assertIn("/tmp/out/candidate_ledger.json", command)

    def test_fetch_results_salvages_later_artifacts_after_summary_fetch_failure(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path="/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="new",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=True,
        )
        ssh_info = module["SshInfo"](
            host="157.157.221.29",
            port=34557,
            key_path=Path("/tmp/key"),
            user="root",
            pod_id="pod123",
        )
        fetch_globals = module["fetch_results"].__globals__
        calls: list[tuple[str, bool]] = []

        def fake_scp_from_remote(*, ssh_info, remote_source, local_destination, recursive):
            del ssh_info, local_destination
            calls.append((str(remote_source), recursive))
            if PurePosixPath(remote_source).name == "continuation_summary.json":
                raise subprocess.CalledProcessError(1, ["scp"])

        original_scp_from_remote = fetch_globals["scp_from_remote"]
        try:
            fetch_globals["scp_from_remote"] = fake_scp_from_remote
            report = module["fetch_results"](plan, ssh_info)
        finally:
            fetch_globals["scp_from_remote"] = original_scp_from_remote

        self.assertEqual(
            calls,
            [
                (
                    "/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
                    False,
                ),
                (
                    "/workspace/continuation-runs/continuation-run-001/continuation_validation.json",
                    False,
                ),
                (
                    "/workspace/continuation-runs/continuation-run-001/continuation_profiling_report.md",
                    False,
                ),
                ("/workspace/continuation-runs/continuation-run-001", True),
            ],
        )
        self.assertEqual(
            report.required_failures,
            (
                "summary:/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            ),
        )
        self.assertEqual(report.optional_failures, ())
        self.assertEqual(
            report.fetched_paths,
            (
                "/tmp/out/continuation_validation.json",
                "/tmp/out/continuation_profiling_report.md",
                "/tmp/out",
            ),
        )

    def test_fetch_results_marks_profile_fetch_optional(self):
        module = load_module_globals()
        plan = module["LaunchPlan"](
            pod_id="pod123",
            run_id="run-001",
            pretend_version="0.0.dev0+gdeadbeef",
            local_columbia_root=Path("/tmp/columbia"),
            local_repo_root=Path("/tmp/columbia/simsopt-jax"),
            local_equilibrium_path=Path("/tmp/columbia/DATABASE/EQUILIBRIA/wout.nc"),
            local_donor_run_dir=Path("/tmp/columbia/autoresearch/run"),
            local_output_dir=Path("/tmp/out"),
            remote_columbia_root="/workspace/columbia",
            remote_repo_root="/workspace/columbia/simsopt-jax",
            remote_equilibrium_path="/workspace/columbia/DATABASE/EQUILIBRIA/wout.nc",
            remote_donor_run_dir="/workspace/columbia/autoresearch/run",
            remote_output_root="/workspace/continuation-runs",
            remote_run_root="/workspace/continuation-runs/continuation-run-001",
            remote_jax_profile_dir=None,
            remote_summary_path="/workspace/continuation-runs/continuation-run-001/continuation_summary.json",
            remote_validation_path=None,
            remote_cache_dir="/workspace/.jax-cache/cont",
            remote_conda_root="/opt/conda",
            plasma_surf_filename="wout_nfp5ginsburg_desc_iota21.nc",
            system_deps_mode="auto",
            backend_mode="jax_gpu_parity",
            transfer_guard="disallow",
            trial_policy="validated-fast",
            run_mode="resume",
            max_final_field_error=5e-4,
            max_final_abs_iota_error=0.02,
            max_final_non_qs=0.05,
            mpol=8,
            ntor=6,
            nphi=255,
            ntheta=64,
            maxiter=300,
            coarse_maxiter=1,
            medium_maxiter=1,
            prefinal_maxiter=2,
            fetch_full_run_root=False,
        )
        ssh_info = module["SshInfo"](
            host="157.157.221.29",
            port=34557,
            key_path=Path("/tmp/key"),
            user="root",
            pod_id="pod123",
        )
        fetch_globals = module["fetch_results"].__globals__

        def fake_scp_from_remote(*, ssh_info, remote_source, local_destination, recursive):
            del ssh_info, local_destination, recursive
            if PurePosixPath(remote_source).name == "continuation_profiling_report.md":
                raise subprocess.CalledProcessError(1, ["scp"])

        original_scp_from_remote = fetch_globals["scp_from_remote"]
        try:
            fetch_globals["scp_from_remote"] = fake_scp_from_remote
            report = module["fetch_results"](plan, ssh_info)
        finally:
            fetch_globals["scp_from_remote"] = original_scp_from_remote

        self.assertEqual(report.required_failures, ())
        self.assertEqual(
            report.optional_failures,
            (
                "profiling_report:/workspace/continuation-runs/continuation-run-001/continuation_profiling_report.md",
            ),
        )


if __name__ == "__main__":
    unittest.main()
