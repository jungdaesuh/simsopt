"""CI-safe tests for the non-banana example CPU C++/JAX parity harness.

Covers:
  * Fixture registry completeness (all plan IDs are present).
  * Phase 0 baseline gates (JAX x64 + no GPU devices required).
  * Phase 1 ``minimal_stage2_flux_length_gap`` full value/gradient parity.
  * Phase 2 ``cws_saved_local_flux_nfp{2,3}`` local-flux fixtures.
  * Phase 3/4 full and planar Stage-II pass parity.
  * Supported/pass/partial fixture verdict semantics plus explicit
    unsupported-component rows.

Tests require simsoptpp for the CPU oracle.  Run with::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 conda run -n jax-0.9.2 \\
        python -m pytest tests/integration/test_non_banana_example_cpp_jax_cpu_parity.py -v
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

from pathlib import Path

import numpy as np
import pytest

import jax
import jaxlib

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

pytest.importorskip(
    "simsoptpp",
    reason="Non-banana parity tests require simsoptpp for the CPU oracle.",
)

from benchmarks import non_banana_example_cpp_jax_cpu_parity as harness  # noqa: E402
from benchmarks import non_banana_example_parity_fixtures as fixtures  # noqa: E402
from benchmarks.validation_ladder_contract import (  # noqa: E402
    PARITY_LADDER_TOLERANCES,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _subprocess_env(*, jax_platforms: str, jax_enable_x64: str) -> dict[str, str]:
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = jax_platforms
    env["JAX_ENABLE_X64"] = jax_enable_x64
    return env


# ---------------------------------------------------------------------------
# Registry coverage.


EXPECTED_FIXTURE_IDS = {
    "minimal_stage2_flux_length_gap",
    "surface_area_volume_simple",
    "pm_simple_fixed_state_gpmo_baseline",
    "pm_qa_fixed_state_gpmo_arbvec_or_multi",
    "pm_muse_famus",
    "pm_pm4stell_backtracking",
    "wireframe_rcls_basic_fixed_state",
    "wireframe_rcls_ports_constraint_gate",
    "wireframe_gsco_modular_fixed_state",
    "wireframe_gsco_sector_saddle_fixed_state",
    "wireframe_gsco_multistep_reduced_diagnostic",
    "tracing_fieldlines_qa_reduced_endpoint",
    "tracing_fieldlines_ncsx_reduced_endpoint",
    "tracing_particle_gc_vac_reduced_endpoint",
    "tracing_boozer_gc_reduced_endpoint",
    "strain_optimization_support_gate",
    "coil_forces_support_gate",
    "cws_saved_local_flux_nfp2",
    "cws_saved_local_flux_nfp3",
    "full_stage2_composite",
    "planar_stage2_composite",
    "position_orientation_flux_support_gate",
    "boozer_surface_basic",
    "boozer_qa_wrappers",
    "finite_beta_target_flux",
    "finitebuild_multifilament_support_gate",
    "qfm_surface",
}


def test_fixture_registry_covers_plan_ids():
    actual_ids = set(fixtures.fixture_ids())
    assert actual_ids == EXPECTED_FIXTURE_IDS, (
        f"Unexpected fixture IDs: {actual_ids ^ EXPECTED_FIXTURE_IDS}"
    )


def test_phase0_gates_require_x64_and_cpu():
    metadata = harness.build_run_metadata(git_sha_override=None)
    assert metadata["jax_enable_x64"] is True
    assert metadata["jax_platform"] == "cpu"
    for device in metadata["jax_devices"]:
        assert device["platform"] == "cpu", f"Expected CPU-only devices; saw {device!r}"


def test_harness_forces_cpu_platform_before_jax_backend_init():
    code = """
import json
import os

from benchmarks import non_banana_example_cpp_jax_cpu_parity as harness

metadata = harness.build_run_metadata(git_sha_override="test-suite-fixed-sha")
print(json.dumps({
    "env_jax_platforms": os.environ["JAX_PLATFORMS"],
    "env_jax_enable_x64": os.environ["JAX_ENABLE_X64"],
    "metadata_jax_enable_x64": metadata["jax_enable_x64"],
    "metadata_requested_jax_platform": metadata["requested_jax_platform"],
    "device_platforms": [device["platform"] for device in metadata["jax_devices"]],
}))
"""
    env = _subprocess_env(jax_platforms="cuda,cpu", jax_enable_x64="0")
    env["SIMSOPT_JAX_PLATFORM"] = "cuda"
    env["SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM"] = "cuda"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    observed = json.loads(completed.stdout.strip().splitlines()[-1])
    assert observed["env_jax_platforms"] == "cpu"
    assert observed["env_jax_enable_x64"] == "1"
    assert observed["metadata_jax_enable_x64"] is True
    assert observed["metadata_requested_jax_platform"] == "cpu"
    assert observed["device_platforms"]
    assert set(observed["device_platforms"]) == {"cpu"}


def test_cli_help_describes_cuda_followup_without_cpu_only_description():
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks/non_banana_example_cpp_jax_cpu_parity.py"),
            "--help",
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cpu", jax_enable_x64="1"),
        text=True,
        capture_output=True,
        check=True,
    )

    normalized_help = " ".join(completed.stdout.split())
    assert "Run the non-banana example CPU C++/JAX parity harness." in normalized_help
    assert "CUDA follow-up runs use cpu_cpp,jax_gpu" in normalized_help


def test_cli_minimal_fixture_uses_current_source_tree(tmp_path):
    output_path = tmp_path / "minimal.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks/non_banana_example_cpp_jax_cpu_parity.py"),
            "--fixtures",
            "minimal_stage2_flux_length_gap",
            "--git-sha",
            "test-suite-fixed-sha",
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cpu", jax_enable_x64="1"),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "Wrote parity artifact:" in completed.stdout

    payload = json.loads(output_path.read_text())
    entry = _select_fixture(payload, "minimal_stage2_flux_length_gap")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass"


def test_cli_rejects_jax_gpu_lane_on_cpu_backend(tmp_path):
    output_path = tmp_path / "gpu.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks/non_banana_example_cpp_jax_cpu_parity.py"),
            "--fixtures",
            "surface_area_volume_simple",
            "--lanes",
            "jax_gpu",
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cpu", jax_enable_x64="1"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "jax_gpu lane requires --baseline-json" in completed.stderr


def test_jax_gpu_lane_with_baseline_rejects_cpu_runtime(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps({"schema_version": fixtures.SCHEMA_VERSION, "fixtures": []}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="explicit CUDA parity environment"):
        harness.run_fixtures(
            ("surface_area_volume_simple",),
            lanes=("cpu_cpp", "jax_gpu"),
            baseline_json=baseline_path,
        )


def test_gpu_runtime_metadata_emits_declared_transfer_guard(monkeypatch):
    monkeypatch.setattr(
        harness,
        "query_nvidia_smi_facts",
        lambda: {
            "nvidia_smi_gpus": [
                {
                    "name": "unit-test-gpu",
                    "driver_version": "580.0",
                    "memory_total_mb": 1.0,
                }
            ],
            "cuda_driver_version": "580.0",
            "cuda_runtime_version": "13.0",
        },
    )
    monkeypatch.setattr(
        harness,
        "_query_nvidia_compute_capabilities",
        lambda: ("9.0",),
    )
    monkeypatch.setattr(
        harness,
        "_jax_platform_versions",
        lambda: ("CUDA 13.0",),
    )
    monkeypatch.setattr(
        harness,
        "_gpu_transfer_guard_probe",
        lambda: {"status": "pass", "mode": "disallow"},
    )

    runtime = harness._gpu_runtime_metadata()

    assert runtime["transfer_guard"] == "disallow"
    assert runtime["transfer_guard_probe"]["status"] == "pass"
    assert runtime["compute_capability"] == "9.0"


def test_jax_gpu_lane_jsonable_is_not_proven_by_name_only():
    lane = fixtures.LaneArtifact(
        lane="jax_cpu",
        objective_total=0.0,
        objective_native_subtotal=0.0,
        components={},
        gradient=None,
        gradient_norm=None,
        active_dof_names=(),
        active_dof_hash="active",
        fixed_free_mask_hash="mask",
        native_curve_spec_hashes=(),
        surface_point_hash="surface",
        unit_normal_hash="normal",
        field_B_hash="field",
        field_B_max=0.0,
        field_B_mean=0.0,
        Bdotn_array_hash="bdotn",
        Bdotn_max=0.0,
        Bdotn_mean=0.0,
        raw_arrays={},
        timing={},
    )

    payload = harness._lane_to_jsonable(lane, lane_name="jax_gpu")

    assert payload["lane"] == "jax_gpu"
    assert payload["gpu_readiness"]["gpu_proven"] is False


def test_lane_selection_filters_emitted_lanes_and_pairwise_comparisons(tmp_path):
    output_path = tmp_path / "cpu-only.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "benchmarks/non_banana_example_cpp_jax_cpu_parity.py"),
            "--fixtures",
            "surface_area_volume_simple",
            "--lanes",
            "cpu_cpp",
            "--output-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=_subprocess_env(jax_platforms="cpu", jax_enable_x64="1"),
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "Wrote parity artifact:" in completed.stdout

    payload = json.loads(output_path.read_text())
    entry = _select_fixture(payload, "surface_area_volume_simple")
    assert payload["metadata"]["selected_lanes"] == ["cpu_cpp"]
    assert set(entry["lanes"]) == {"cpu_cpp"}
    assert entry["comparisons"]["cpu_cpp_vs_jax_cpu"] == []
    assert entry["verdict"] == "fail", entry
    assert entry["passed"] is False
    assert list(entry["failures"]) == [
        "selected lanes omit the required parity pair for verdict"
    ]


def test_jax_cpu_vs_jax_gpu_comparisons_reuse_cpu_baseline_right_values():
    baseline_entry = {
        "fixture_id": "surface_area_volume_simple",
        "dof_contract": {"fixture_input_hash": "fixture-hash"},
        "comparisons": {
            "cpu_cpp_vs_jax_cpu": [
                {
                    "quantity": "area",
                    "component": "surface_scalar",
                    "jax_cpu_value": 1.0,
                }
            ]
        },
    }
    gpu_result = harness.FixtureResult(
        fixture_id="surface_area_volume_simple",
        source_example="examples/1_Simple/surf_vol_area.py",
        classification=fixtures.SUPPORTED,
        classification_reason="",
        fixture_inputs={},
        dof_contract={"fixture_input_hash": "fixture-hash"},
        native_spec_contract={},
        lanes={},
        comparisons={
            "cpu_cpp_vs_jax_gpu": [
                {
                    "quantity": "area",
                    "component": "surface_scalar",
                    "jax_gpu_value": 1.0,
                }
            ]
        },
        unsupported_components=(),
        mixed_lane_diagnostics=(),
        perturbation_diagnostics=None,
        verdict="pass",
        passed=True,
        failures=(),
    )

    comparisons = harness._jax_cpu_vs_jax_gpu_comparisons(
        baseline_entry=baseline_entry,
        gpu_result=gpu_result,
    )

    assert len(comparisons) == 1
    entry = comparisons[0]
    assert entry["left_lane"] == "jax_cpu"
    assert entry["right_lane"] == "jax_gpu"
    assert entry["jax_cpu_value"] == 1.0
    assert entry["jax_gpu_value"] == 1.0
    assert entry["verdict"] == "pass"


# ---------------------------------------------------------------------------
# Phase 1 — P0 minimal Stage-II fixture parity.


@pytest.fixture(scope="module")
def minimal_fixture_payload():
    payload = harness.run_fixtures(
        ["minimal_stage2_flux_length_gap"],
        git_sha_override="test-suite-fixed-sha",
    )
    return payload


def _select_fixture(payload, fixture_id):
    for entry in payload["fixtures"]:
        if entry["fixture_id"] == fixture_id:
            return entry
    raise KeyError(f"Fixture {fixture_id!r} not present in payload")


def test_minimal_stage2_passes_native_supported_parity(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Native-supported comparisons failed: {failing}"


def test_minimal_stage2_lists_unsupported_length_penalty(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    assert entry["unsupported_components"] == []
    assert entry["verdict"] == "pass", entry["verdict"]
    assert (
        "QuadraticPenalty_over_sum_CurveLength_max"
        in entry["lanes"]["cpu_cpp"]["components"]
    )
    assert (
        "QuadraticPenalty_over_sum_CurveLength_max"
        in entry["lanes"]["jax_cpu"]["components"]
    )


def test_minimal_stage2_dof_basis_matches(minimal_fixture_payload):
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    cpu_hash = entry["dof_contract"]["active_dof_hash_cpu"]
    jax_hash = entry["dof_contract"]["active_dof_hash_jax"]
    assert cpu_hash == jax_hash, (
        "CPU and JAX lanes must build from identical active-DOF vectors."
    )
    # The harness must report basis alignment explicitly: positional
    # gradient comparison is only valid when the structural names match.
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    cpu_struct = entry["dof_contract"]["active_dof_structural_names_cpu"]
    jax_struct = entry["dof_contract"]["active_dof_structural_names_jax"]
    assert cpu_struct == jax_struct, (cpu_struct[:3], jax_struct[:3])
    # Native curve spec hashes must be non-empty for the JAX lane.
    spec_hashes = entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert spec_hashes, "JAX lane must have non-empty native curve spec hashes"


def test_minimal_stage2_perturbation_diagnostic_is_live(minimal_fixture_payload):
    """The seed=1 Taylor diagnostic must run real CPU/JAX evaluations."""
    entry = _select_fixture(minimal_fixture_payload, "minimal_stage2_flux_length_gap")
    perturbation = entry["perturbation_diagnostics"]
    assert perturbation is not None, "perturbation diagnostic must be present"
    assert perturbation["seed"] == 1
    assert perturbation["direction_hash"] is not None
    samples = perturbation["samples"]
    assert len(samples) >= 5, samples
    # Cross-lane central differences must agree to within a tight tolerance
    # at every recorded eps for the full fixed-state flux+length objective.
    for sample in samples:
        assert sample["abs_diff"] < 1e-6, sample
    grad_jax_dir = perturbation["directional_derivative_grad_jax"]
    grad_cpu_dir = perturbation["directional_derivative_grad_cpu"]
    assert np.isfinite(grad_jax_dir) and np.isfinite(grad_cpu_dir)
    assert abs(grad_jax_dir - grad_cpu_dir) < 1e-6, (grad_jax_dir, grad_cpu_dir)


# ---------------------------------------------------------------------------
# Phase 2 — P1 CWS saved local-flux fixture parity.


def _cws_artifacts_present(fixture_id: str) -> bool:
    record = fixtures.get_fixture(fixture_id)
    inputs = record.spec.inputs
    case_dir = REPO_ROOT / "examples" / "3_Advanced" / inputs["case_dir"]
    coils_path = case_dir / "coils" / inputs["coils_file"]
    vmec_path = case_dir / inputs["vmec_input"]
    return coils_path.exists() and vmec_path.exists()


@pytest.mark.parametrize(
    "fixture_id",
    [
        "cws_saved_local_flux_nfp2",
        "cws_saved_local_flux_nfp3",
    ],
)
def test_cws_saved_local_flux_parity(fixture_id):
    if not _cws_artifacts_present(fixture_id):
        pytest.skip(f"{fixture_id} requires saved CWS artifacts in examples/3_Advanced")
    payload = harness.run_fixtures([fixture_id])
    entry = _select_fixture(payload, fixture_id)
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] in ("pass", "partial"), entry
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"CWS parity comparisons failed: {failing}"


def test_cws_saved_local_flux_rows_are_advertised_as_supported():
    assert "cws_saved_local_flux_nfp2" in fixtures.supported_fixture_ids()
    assert "cws_saved_local_flux_nfp3" in fixtures.supported_fixture_ids()
    for fixture_id in ("cws_saved_local_flux_nfp2", "cws_saved_local_flux_nfp3"):
        spec = fixtures.get_fixture(fixture_id).spec
        assert spec.classification == fixtures.SUPPORTED
        assert (
            "legacy CurveCWSFourier JSON reconstruction" in spec.classification_reason
        )


def test_file_backed_fixture_inputs_include_content_hashes():
    for fixture_id in (
        "finite_beta_target_flux",
        "pm_simple_fixed_state_gpmo_baseline",
        "pm_qa_fixed_state_gpmo_arbvec_or_multi",
        "pm_muse_famus",
        "pm_pm4stell_backtracking",
        "wireframe_rcls_basic_fixed_state",
        "wireframe_rcls_ports_constraint_gate",
        "wireframe_gsco_sector_saddle_fixed_state",
        "wireframe_gsco_multistep_reduced_diagnostic",
        "tracing_fieldlines_qa_reduced_endpoint",
        "tracing_boozer_gc_reduced_endpoint",
    ):
        inputs = fixtures.get_fixture(fixture_id).spec.inputs
        file_hashes = inputs["input_file_hashes"]
        assert file_hashes
        for metadata in file_hashes.values():
            assert metadata["path"]
            assert len(metadata["sha256"]) == 64
            assert metadata["size_bytes"] > 0


# ---------------------------------------------------------------------------
# Support-gate / partial classification paths.


def test_wireframe_rcls_ports_constraint_gate_reaches_partial_parity():
    payload = harness.run_fixtures(["wireframe_rcls_ports_constraint_gate"])
    entry = _select_fixture(payload, "wireframe_rcls_ports_constraint_gate")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == [
        "RCLS_current_vector_nonunique_nullspace"
    ]
    expected = {
        "surface_gamma",
        "surface_unit_normal",
        "wireframe_matrix",
        "wireframe_objective",
        "wireframe_constraints",
        "wireframe_field_B",
        "wireframe_field_dB_by_dX",
        "wireframe_Bnormal",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert any(cmp["component"] == "constraint_matrix_shape" for cmp in comparisons)
    assert all(cmp["verdict"] == "pass" for cmp in comparisons)
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["port_count"] == 16.0
    assert components["constraint_rows"] > 0.0
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True


def test_finite_beta_target_flux_reaches_pass_parity_with_cached_target():
    payload = harness.run_fixtures(["finite_beta_target_flux"])
    entry = _select_fixture(payload, "finite_beta_target_flux")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []

    target_metadata = entry["fixture_inputs"]["virtual_casing_target"]
    assert target_metadata["path"] == (
        "tests/test_files/finite_beta_w7x_B_external_normal_nphi32_ntheta32.npy"
    )
    assert target_metadata["shape"] == [32, 32]
    assert target_metadata["array_sha256"] == (
        "ae4f35b773e2db9b2feb566d7fdbea7545f63e06cbbf1872ca7ec7ce46b7d658"
    )

    target_array = np.load(REPO_ROOT / target_metadata["path"], allow_pickle=False)
    observed_hash = hashlib.sha256(
        np.ascontiguousarray(target_array, dtype=np.float64).tobytes()
    ).hexdigest()
    assert observed_hash == target_metadata["array_sha256"]

    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, f"Finite-beta target-flux comparisons failed: {failing}"
    quantities = {cmp["quantity"] for cmp in comparisons}
    assert {"SquaredFlux", "objective_native_subtotal", "gradient"} <= quantities
    assert (
        "sum_QuadraticPenalty_CurveLength_identity"
        in entry["lanes"]["cpu_cpp"]["components"]
    )
    assert (
        "sum_QuadraticPenalty_CurveLength_identity"
        in entry["lanes"]["jax_cpu"]["components"]
    )


def test_strain_optimization_support_gate_reaches_pass_parity():
    payload = harness.run_fixtures(["strain_optimization_support_gate"])
    entry = _select_fixture(payload, "strain_optimization_support_gate")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "torsional_strain",
        "binormal_curvature_strain",
        "torsional_penalty",
        "binormal_curvature_penalty",
        "objective_native_subtotal",
        "gradient",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert all(
        cmp["source_example"]
        == fixtures.STRAIN_OPTIMIZATION_SUPPORT_GATE_SPEC.source_example
        for cmp in comparisons
    )
    assert all(cmp["verdict"] == "pass" for cmp in comparisons)
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    perturbation = entry["perturbation_diagnostics"]
    assert perturbation is not None
    assert perturbation["samples"]
    assert all(sample["abs_diff"] < 1e-12 for sample in perturbation["samples"])


def test_coil_forces_support_gate_reaches_pass_public_wrapper_parity():
    payload = harness.run_fixtures(["coil_forces_support_gate"])
    entry = _select_fixture(payload, "coil_forces_support_gate")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "LpCurveForce",
        "LpCurveForce_independent_oracle",
        "B2Energy",
        "B2Energy_independent_oracle",
        "objective_native_subtotal",
        "lp_curve_force_gradient",
        "b2_energy_gradient",
        "gradient",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert all(
        cmp["source_example"] == fixtures.COIL_FORCES_SUPPORT_GATE_SPEC.source_example
        for cmp in comparisons
    )
    assert all(cmp["verdict"] == "pass" for cmp in comparisons)
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    perturbation = entry["perturbation_diagnostics"]
    assert perturbation is not None
    assert perturbation["samples"]
    assert all(sample["abs_diff"] < 1e-12 for sample in perturbation["samples"])


def test_finitebuild_multifilament_support_gate_reaches_pass_parity():
    payload = harness.run_fixtures(["finitebuild_multifilament_support_gate"])
    entry = _select_fixture(payload, "finitebuild_multifilament_support_gate")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "Bdotn",
        "SquaredFlux",
        "objective_native_subtotal",
        "gradient",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert all(cmp["verdict"] == "pass" for cmp in comparisons)
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    assert (
        "sum_QuadraticPenalty_CurveLength_max"
        in entry["lanes"]["cpu_cpp"]["components"]
    )
    assert (
        "sum_QuadraticPenalty_CurveLength_max"
        in entry["lanes"]["jax_cpu"]["components"]
    )
    assert "CurveCurveDistance" in entry["lanes"]["cpu_cpp"]["components"]
    assert "CurveCurveDistance" in entry["lanes"]["jax_cpu"]["components"]
    perturbation = entry["perturbation_diagnostics"]
    assert perturbation is not None
    assert perturbation["samples"]
    assert all(sample["abs_diff"] < 1e-10 for sample in perturbation["samples"])


def test_position_orientation_flux_support_gate_reaches_pass_parity():
    payload = harness.run_fixtures(["position_orientation_flux_support_gate"])
    entry = _select_fixture(payload, "position_orientation_flux_support_gate")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "Bdotn",
        "SquaredFlux",
        "objective_native_subtotal",
        "gradient",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]
    structural_names = entry["dof_contract"]["active_dof_structural_names_cpu"]
    assert any(":x0" in name for name in structural_names)
    assert any(":yaw" in name for name in structural_names)


def test_qfm_surface_reports_fixed_state_residual_label_parity():
    payload = harness.run_fixtures(["qfm_surface"])
    entry = _select_fixture(payload, "qfm_surface")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["unsupported_components"] == ["QfmSurface_host_solver"]
    assert (
        entry["fixture_inputs"]["post_constraint_target_state"]
        == "not_reconstructable_without_host_scipy_QfmSurface"
    )

    expected = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "Bdotn",
        "qfm_residual",
        "qfm_gradient",
        "area",
        "volume",
        "toroidal_flux",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]


def test_pm_simple_fixed_state_gpmo_baseline_passes_payload_and_field_parity():
    payload = harness.run_fixtures(["pm_simple_fixed_state_gpmo_baseline"])
    entry = _select_fixture(payload, "pm_simple_fixed_state_gpmo_baseline")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "pm_grid_payload",
        "pm_moments",
        "pm_residual",
        "pm_objective",
        "pm_history",
        "pm_dipole_field_B",
        "pm_dipole_Bdotn",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    assert entry["lanes"]["cpu_cpp"]["components"]["K"] == 4.0
    assert entry["lanes"]["jax_cpu"]["components"]["K"] == 4.0


def test_pm_qa_relax_and_split_reaches_partial_parity():
    payload = harness.run_fixtures(["pm_qa_fixed_state_gpmo_arbvec_or_multi"])
    entry = _select_fixture(payload, "pm_qa_fixed_state_gpmo_arbvec_or_multi")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["unsupported_components"] == [
        "qa_coil_current_optimization",
        "qa_plot_and_famus_outputs",
    ]
    expected = {
        "pm_grid_payload",
        "pm_moments",
        "pm_residual",
        "pm_proxy_residual",
        "pm_objective",
        "pm_proxy_objective",
        "pm_history",
        "pm_dipole_field_B",
        "pm_proxy_dipole_field_B",
        "pm_dipole_Bdotn",
        "pm_proxy_dipole_Bdotn",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert "relax_and_split_RS_history" in {
        cmp["component"] for cmp in comparisons if cmp["quantity"] == "pm_history"
    }
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing

    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    assert cpu_components["max_iter"] == jax_components["max_iter"] == 2.0
    assert cpu_components["max_iter_RS"] == jax_components["max_iter_RS"] == 2.0
    assert (
        cpu_components["algorithm_variant"]
        == jax_components["algorithm_variant"]
        == 4.0
    )


def test_pm_muse_famus_arbvec_backtracking_passes_history_parity():
    payload = harness.run_fixtures(["pm_muse_famus"])
    entry = _select_fixture(payload, "pm_muse_famus")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "pm_grid_payload",
        "pm_moments",
        "pm_residual",
        "pm_objective",
        "pm_history",
        "pm_dipole_field_B",
        "pm_dipole_Bdotn",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing

    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    assert cpu_components["K"] == jax_components["K"] == 5.0
    assert (
        cpu_components["algorithm_variant"]
        == jax_components["algorithm_variant"]
        == 3.0
    )
    assert {
        cmp["component"] for cmp in comparisons if cmp["quantity"] == "pm_moments"
    } == {"GPMO_ArbVec_backtracking"}
    assert "GPMO_ArbVec_backtracking_m_history" in {
        cmp["component"] for cmp in comparisons if cmp["quantity"] == "pm_history"
    }


def test_pm_pm4stell_arbvec_backtracking_passes_history_parity():
    payload = harness.run_fixtures(["pm_pm4stell_backtracking"])
    entry = _select_fixture(payload, "pm_pm4stell_backtracking")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "pm_grid_payload",
        "pm_moments",
        "pm_residual",
        "pm_objective",
        "pm_history",
        "pm_dipole_field_B",
        "pm_dipole_Bdotn",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing

    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    assert cpu_components["K"] == jax_components["K"] == 5.0
    assert (
        cpu_components["algorithm_variant"]
        == jax_components["algorithm_variant"]
        == 3.0
    )
    assert {
        cmp["component"] for cmp in comparisons if cmp["quantity"] == "pm_moments"
    } == {"GPMO_ArbVec_backtracking"}
    assert "GPMO_ArbVec_backtracking_m_history" in {
        cmp["component"] for cmp in comparisons if cmp["quantity"] == "pm_history"
    }


def test_wireframe_rcls_basic_fixed_state_passes_matrix_solve_and_field_parity():
    payload = harness.run_fixtures(["wireframe_rcls_basic_fixed_state"])
    entry = _select_fixture(payload, "wireframe_rcls_basic_fixed_state")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["unsupported_components"] == [
        "RCLS_current_vector_nonunique_nullspace"
    ]
    expected = {
        "wireframe_matrix",
        "wireframe_objective",
        "wireframe_constraints",
        "wireframe_field_B",
        "wireframe_field_dB_by_dX",
        "wireframe_Bnormal",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    assert (
        entry["lanes"]["cpu_cpp"]["components"]["constraints_satisfied"]
        == entry["lanes"]["jax_cpu"]["components"]["constraints_satisfied"]
        == 1.0
    )


def test_wireframe_gsco_modular_fixed_state_passes_history_parity():
    payload = harness.run_fixtures(["wireframe_gsco_modular_fixed_state"])
    entry = _select_fixture(payload, "wireframe_gsco_modular_fixed_state")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "wireframe_matrix",
        "wireframe_gsco_flags",
        "wireframe_gsco_solution",
        "wireframe_gsco_history",
        "objective_native_subtotal",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    assert entry["lanes"]["cpu_cpp"]["components"]["no_crossing"] == 0.0
    assert entry["lanes"]["cpu_cpp"]["components"]["no_new_coils"] == 0.0
    assert entry["lanes"]["cpu_cpp"]["components"]["match_current"] == 0.0
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True


def test_wireframe_gsco_sector_saddle_fixed_state_passes_constraint_parity():
    payload = harness.run_fixtures(["wireframe_gsco_sector_saddle_fixed_state"])
    entry = _select_fixture(payload, "wireframe_gsco_sector_saddle_fixed_state")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "wireframe_matrix",
        "wireframe_gsco_flags",
        "wireframe_gsco_solution",
        "wireframe_gsco_history",
        "wireframe_gsco_constraints",
        "wireframe_field_B",
        "wireframe_Bnormal",
        "objective_native_subtotal",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    assert any(cmp["component"] == "free_cell_mask" for cmp in comparisons)
    assert any(cmp["component"] == "initial_currents" for cmp in comparisons)
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["no_crossing"] == 1.0
    assert components["max_loop_count"] == 0.0
    assert components["history_length"] == 6.0
    assert components["constraints_satisfied"] == 1.0
    assert 0.0 < components["free_cell_count"] < components["total_cell_count"]
    assert components["initial_current_nonzero_count"] > 0.0
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True


def test_wireframe_gsco_multistep_first_step_reaches_partial_parity():
    payload = harness.run_fixtures(["wireframe_gsco_multistep_reduced_diagnostic"])
    entry = _select_fixture(payload, "wireframe_gsco_multistep_reduced_diagnostic")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == [
        "wireframe_multistep_mutation_loop",
        "wireframe_small_coil_pruning",
        "wireframe_final_adjustment_step",
        "wireframe_plot_and_vtk_outputs",
    ]
    expected = {
        "wireframe_matrix",
        "wireframe_gsco_flags",
        "wireframe_gsco_solution",
        "wireframe_gsco_history",
        "objective_native_subtotal",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["no_crossing"] == 1.0
    assert components["max_loop_count"] == 1.0
    assert components["history_length"] == 6.0
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True


def test_tracing_fieldlines_qa_reduced_endpoint_passes_endpoint_parity():
    payload = harness.run_fixtures(["tracing_fieldlines_qa_reduced_endpoint"])
    entry = _select_fixture(payload, "tracing_fieldlines_qa_reduced_endpoint")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    expected = {
        "field_B",
        "trajectory_endpoint",
        "trajectory_t_final",
        "trajectory_status_code",
        "phi_hit_xyz",
        "phi_hit_count",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing


def test_tracing_fieldlines_ncsx_reduced_endpoint_reaches_pass_parity():
    payload = harness.run_fixtures(["tracing_fieldlines_ncsx_reduced_endpoint"])
    entry = _select_fixture(payload, "tracing_fieldlines_ncsx_reduced_endpoint")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "field_B",
        "trajectory_endpoint",
        "trajectory_t_final",
        "trajectory_status_code",
        "phi_hit_xyz",
        "phi_hit_count",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["fieldline_count"] == 1.0
    assert components["phi_hit_count"] == 4.0
    assert components["tmax"] == 20.0


def test_tracing_particle_gc_vac_reduced_endpoint_reaches_pass_parity():
    payload = harness.run_fixtures(["tracing_particle_gc_vac_reduced_endpoint"])
    entry = _select_fixture(payload, "tracing_particle_gc_vac_reduced_endpoint")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == []
    expected = {
        "field_B",
        "field_GradAbsB",
        "trajectory_endpoint",
        "trajectory_t_final",
        "trajectory_status_code",
        "phi_hit_xyz",
        "phi_hit_count",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["particle_count"] == 1.0
    assert components["Ekin_eV"] == 5000.0
    assert components["tmax"] == 1e-7


def test_tracing_boozer_gc_reduced_endpoint_reaches_partial_parity():
    payload = harness.run_fixtures(["tracing_boozer_gc_reduced_endpoint"])
    entry = _select_fixture(payload, "tracing_boozer_gc_reduced_endpoint")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "partial", entry
    assert entry["classification"] == fixtures.SUPPORTED
    assert entry["unsupported_components"] == ["VMEC_input_external_solver"]
    expected = {
        "field_modB",
        "trajectory_endpoint",
        "trajectory_t_final",
        "trajectory_status_code",
        "phi_hit_xyz",
        "phi_hit_count",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    assert expected <= {cmp["quantity"] for cmp in comparisons}
    failing = [cmp for cmp in comparisons if cmp["verdict"] != "pass"]
    assert not failing, failing
    components = entry["lanes"]["cpu_cpp"]["components"]
    assert components["particle_count"] == 1.0
    assert components["Ekin_eV"] == 1000.0
    assert components["interpolation_degree"] == 2.0


def test_surface_area_volume_simple_passes_value_and_gradient_parity():
    payload = harness.run_fixtures(["surface_area_volume_simple"])
    entry = _select_fixture(payload, "surface_area_volume_simple")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []

    expected = {
        "surface_gamma",
        "surface_unit_normal",
        "area",
        "volume",
        "area_gradient",
        "volume_gradient",
        "area_perturbed_values",
        "volume_perturbed_values",
    }
    comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    recorded = {cmp["quantity"] for cmp in comparisons}
    assert expected <= recorded
    assert all(
        cmp["source_example"] == "examples/1_Simple/surf_vol_area.py"
        for cmp in comparisons
    )
    assert all(cmp["verdict"] == "pass" for cmp in comparisons)
    assert all("cpu_cpp_value" in cmp and "jax_cpu_value" in cmp for cmp in comparisons)

    cpu_lane = entry["lanes"]["cpu_cpp"]
    jax_lane = entry["lanes"]["jax_cpu"]
    assert cpu_lane["active_dof_hash"] == jax_lane["active_dof_hash"]
    assert cpu_lane["gpu_readiness"]["gpu_ready"] is False
    assert jax_lane["gpu_readiness"]["gpu_ready"] is False
    assert entry["dof_contract"]["active_dof_basis_aligned"] is True
    assert entry["dof_contract"]["fixture_input_hash"]


def test_supported_fixtures_have_source_rationale_and_acceptance_criteria():
    for fixture_id in fixtures.supported_fixture_ids():
        spec = fixtures.get_fixture(fixture_id).spec
        if fixture_id in EXPECTED_FIXTURE_IDS - {
            "pm_simple_fixed_state_gpmo_baseline",
            "position_orientation_flux_support_gate",
            "qfm_surface",
            "surface_area_volume_simple",
            "finitebuild_multifilament_support_gate",
            "strain_optimization_support_gate",
            "coil_forces_support_gate",
            "tracing_fieldlines_qa_reduced_endpoint",
            "pm_qa_fixed_state_gpmo_arbvec_or_multi",
            "pm_muse_famus",
            "pm_pm4stell_backtracking",
            "wireframe_gsco_modular_fixed_state",
            "wireframe_gsco_sector_saddle_fixed_state",
            "wireframe_gsco_multistep_reduced_diagnostic",
            "wireframe_rcls_basic_fixed_state",
            "wireframe_rcls_ports_constraint_gate",
            "tracing_fieldlines_ncsx_reduced_endpoint",
            "tracing_particle_gc_vac_reduced_endpoint",
            "tracing_boozer_gc_reduced_endpoint",
        }:
            continue
        assert spec.source_example
        assert spec.rationale
        assert spec.acceptance_criteria


def test_pass_or_partial_rows_have_numeric_cpu_oracle_comparisons():
    payload = harness.run_fixtures(["surface_area_volume_simple", "qfm_surface"])
    forbidden_markers = {"cpu_fallback", "host_fallback", "jax_self_reference"}
    for entry in payload["fixtures"]:
        assert entry["verdict"] in {"pass", "partial"}
        comparisons = entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        assert comparisons
        assert all(cmp["cpu_cpp_value"] is not None for cmp in comparisons)
        assert all(cmp["jax_cpu_value"] is not None for cmp in comparisons)
        assert not forbidden_markers & set(entry["unsupported_components"])
        assert not forbidden_markers & set(entry["mixed_lane_diagnostics"])


# ---------------------------------------------------------------------------
# Phase 6 — P2 Boozer surface fixed-state residual + label fixture parity.


def test_boozer_surface_basic_passes_residual_and_label_parity():
    """Fixed-state Boozer parity must pass for residual + labels.

    The fixture rebuilds the NCSX initial state on both CPU and JAX lanes
    (independent coil trees) and compares the pre-solve Boozer residual
    vector plus Area, Volume, and ToroidalFlux labels. The verdict must
    be 'pass' (no unsupported components) and every recorded comparison
    must individually pass.
    """
    payload = harness.run_fixtures(["boozer_surface_basic"])
    entry = _select_fixture(payload, "boozer_surface_basic")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Boozer fixed-state comparisons failed: {failing}"

    # Sanity-check that the expected quantities were exercised.
    recorded_quantities = {
        cmp["quantity"] for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    }
    expected_quantities = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "boozer_residual",
        "area",
        "volume",
        "toroidal_flux",
    }
    assert expected_quantities <= recorded_quantities, (
        f"Missing expected comparisons: {expected_quantities - recorded_quantities}"
    )

    # Label and residual values must agree at the direct_kernel bucket.
    for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]:
        if cmp["quantity"] in {
            "boozer_residual",
            "area",
            "volume",
            "toroidal_flux",
        }:
            assert cmp["tolerance_bucket"] == "direct_kernel", cmp
            expected_tolerance = PARITY_LADDER_TOLERANCES["direct_kernel"]
            assert cmp["tolerance_rtol"] == expected_tolerance["rtol"], cmp
            assert cmp["tolerance_atol"] == expected_tolerance["atol"], cmp


# ---------------------------------------------------------------------------
# Phase 6 — P2 BoozerQA wrappers fixed-solved-state parity.


def test_boozer_qa_wrappers_passes_native_supported_parity():
    """Solved-state QA wrapper parity must pass for Iotas/MajorRadius/NQS/length.

    The fixture solves the NCSX Boozer surface on the CPU side via
    ``BoozerSurface.solve_residual_equation_exactly_newton``, then
    evaluates the upstream QA wrappers (Iotas, MajorRadius,
    NonQuasiSymmetricRatio, and sum(CurveLength)) on both lanes. The JAX
    side recomputes the wrappers as pure JAX functions
    (``surface_major_radius_jax_from_dofs``, ``_qs_ratio_pure``) at the
    byte-equal solved surface DOFs and using a fresh ``BiotSavartJAX``
    coil_set_spec, plus ``CurveLengthJAX`` over an independently loaded curve
    tree.

    Every recorded native comparison must individually pass at the
    direct_kernel bucket.
    """
    payload = harness.run_fixtures(["boozer_qa_wrappers"])
    entry = _select_fixture(payload, "boozer_qa_wrappers")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry
    assert entry["unsupported_components"] == []

    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"BoozerQA wrapper comparisons failed: {failing}"

    # Sanity-check that the expected quantities were exercised.
    recorded_quantities = {
        cmp["quantity"] for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
    }
    expected_quantities = {
        "surface_gamma",
        "surface_unit_normal",
        "field_B",
        "iota",
        "major_radius",
        "nq_symmetric_ratio",
        "sum_CurveLength",
    }
    assert expected_quantities <= recorded_quantities, (
        f"Missing expected comparisons: {expected_quantities - recorded_quantities}"
    )

    # The wrapper scalars must use the direct_kernel bucket: the JAX side
    # is a pure-JAX recomputation at the same surface DOFs, with no LS
    # solver / adjoint involved in this fixture's parity claim.
    expected_tolerance = PARITY_LADDER_TOLERANCES["direct_kernel"]
    for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]:
        if cmp["quantity"] in {
            "iota",
            "major_radius",
            "nq_symmetric_ratio",
            "sum_CurveLength",
        }:
            assert cmp["tolerance_bucket"] == "direct_kernel", cmp
            assert cmp["tolerance_rtol"] == expected_tolerance["rtol"], cmp
            assert cmp["tolerance_atol"] == expected_tolerance["atol"], cmp

    # Both lanes must record the length penalty now that CurveLengthJAX covers it.
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    assert "sum_CurveLength" in cpu_components
    assert "sum_CurveLength" in jax_components
    assert "iota" in cpu_components
    assert "major_radius" in cpu_components
    assert "nq_symmetric_ratio" in cpu_components

    # Native curve spec hashes must be present on the JAX lane.
    spec_hashes = entry["native_spec_contract"]["native_curve_spec_hashes"]
    assert spec_hashes, "JAX lane must have non-empty native curve spec hashes"


# ---------------------------------------------------------------------------
# Phase 3 — P1 full Stage-II composite fixture parity (pass verdict).


def test_full_stage2_composite_passes_with_jax_curve_objectives():
    """Full composite must pass with JAX curve-objective components included.

    Also enforces the plan §"Math, Physics, And Computation Gates"
    requirement that composite objectives record BOTH raw and weighted
    component values plus the composite total.
    """
    payload = harness.run_fixtures(["full_stage2_composite"])
    entry = _select_fixture(payload, "full_stage2_composite")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry["verdict"]
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Native-supported comparisons failed: {failing}"

    assert entry["unsupported_components"] == []
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    assert "JF_total_cpu" in cpu_components
    assert "JF_total_jax" in jax_components
    assert "SquaredFlux" in cpu_components
    assert "SquaredFluxJAX" in jax_components

    # Plan §"Math, Physics, And Computation Gates": composite objectives
    # must record raw component values (before weights) and weighted
    # component values for every native curve-objective component.
    for component in (
        "sum_CurveLength",
        "CurveCurveDistance",
        "CurveSurfaceDistance",
        "sum_LpCurveCurvature",
        "sum_QuadraticPenalty_MeanSquaredCurvature_max",
    ):
        for lane_components in (cpu_components, jax_components):
            assert f"{component}_raw" in lane_components, (
                f"Missing _raw entry for {component}: {sorted(lane_components)}"
            )
            assert f"{component}_weighted" in lane_components, (
                f"Missing _weighted entry for {component}: {sorted(lane_components)}"
            )


# ---------------------------------------------------------------------------
# Phase 4 — P2 planar Stage-II composite fixture parity (pass verdict).


def test_planar_stage2_composite_passes_with_jax_curve_objectives():
    payload = harness.run_fixtures(["planar_stage2_composite"])
    entry = _select_fixture(payload, "planar_stage2_composite")
    assert entry["error"] is None, entry["error"]
    assert entry["verdict"] == "pass", entry["verdict"]
    failing = [
        cmp
        for cmp in entry["comparisons"]["cpu_cpp_vs_jax_cpu"]
        if cmp["verdict"] != "pass"
    ]
    assert not failing, f"Planar native-supported comparisons failed: {failing}"
    assert entry["unsupported_components"] == []

    # Native spec hashes must be present (CurvePlanarFourier exposes to_spec).
    assert entry["native_spec_contract"]["native_curve_spec_hashes"]

    # Plan §"Math, Physics, And Computation Gates": raw + weighted entries
    # for every curve-objective component. LinkingNumber enters with weight 1,
    # so _raw and _weighted are equal but both must be recorded on both lanes.
    cpu_components = entry["lanes"]["cpu_cpp"]["components"]
    jax_components = entry["lanes"]["jax_cpu"]["components"]
    for component in (
        "QuadraticPenalty_over_sum_CurveLength_identity",
        "CurveCurveDistance",
        "CurveSurfaceDistance",
        "sum_LpCurveCurvature",
        "sum_QuadraticPenalty_MeanSquaredCurvature_identity",
        "LinkingNumber",
    ):
        for lane_components in (cpu_components, jax_components):
            assert f"{component}_raw" in lane_components, (
                f"Missing _raw entry for {component}: {sorted(lane_components)}"
            )
            assert f"{component}_weighted" in lane_components, (
                f"Missing _weighted entry for {component}: {sorted(lane_components)}"
            )


# ---------------------------------------------------------------------------
# Schema sanity.


def test_payload_schema_versioned(minimal_fixture_payload):
    assert fixtures.SCHEMA_VERSION == "1.0"
    assert minimal_fixture_payload["schema_version"] == "1.0"
    assert minimal_fixture_payload["harness"] == "non_banana_example_cpp_jax_cpu_parity"
    metadata = minimal_fixture_payload["metadata"]
    assert metadata["jax_platform"] == "cpu"
    assert metadata["jax_enable_x64"] is True
    assert metadata["jax_version"] == jax.__version__
    assert metadata["jaxlib_version"] == jaxlib.__version__
    assert "jax.__version__" in metadata["version_probe_command"]
    assert "jaxlib.__version__" in metadata["version_probe_command"]
    lane_schema = metadata["lane_schema"]
    assert lane_schema["cpu_cpp"]["artifact_kind"] == "cpu_oracle"
    assert lane_schema["jax_cpu"]["artifact_kind"] == "jax_cpu_candidate"
    assert lane_schema["jax_gpu"]["status"] == "runtime_required"
    assert lane_schema["jax_gpu"]["required_environment"] == {
        "SIMSOPT_BACKEND_MODE": "jax_gpu_parity",
        "SIMSOPT_JAX_PLATFORM": "cuda",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "1",
        "SIMSOPT_EXAMPLE_PARITY_JAX_PLATFORM": "cuda",
    }
    assert lane_schema["jax_gpu"]["must_reuse_fixture_input_hash"] is True
    assert lane_schema["jax_gpu"]["cannot_upgrade_cpu_unsupported"] is True
    assert lane_schema["jax_gpu"]["separate_artifact_required"] is True
    assert lane_schema["jax_gpu"]["disallowed_first_proof_lane"] == "jax_gpu_fast"


def test_no_gpu_required_anywhere(minimal_fixture_payload):
    metadata = minimal_fixture_payload["metadata"]
    for device in metadata["jax_devices"]:
        assert device["platform"] == "cpu"
