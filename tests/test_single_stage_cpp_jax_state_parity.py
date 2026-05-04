import numpy as np

from benchmarks.single_stage_cpp_jax_state_parity import (
    DETERMINISTIC_FIXTURE_SCOPE,
    LANE_CPP_CPU,
    LANE_JAX_CPU,
    LANE_JAX_GPU,
    _comparison_status,
    build_fixed_state_artifact,
    merge_fixed_state_artifacts,
)
from benchmarks.single_stage_parity_matrix import (
    REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS,
    REQUIRED_FIXED_STATE_HASH_KEYS,
    REQUIRED_OPERATOR_LANE_OUTPUT_KEYS,
    _fixed_state_bucket,
)


def _build_deterministic_artifact():
    return build_fixed_state_artifact("cpu", DETERMINISTIC_FIXTURE_SCOPE)


def test_fixed_state_cpu_artifact_passes_cpu_lanes_and_blocks_gpu_lane():
    artifact = _build_deterministic_artifact()

    assert artifact["schema_version"] == 1
    assert artifact["passed"] is False
    assert artifact["lanes"][LANE_CPP_CPU]["status"] == "pass"
    assert artifact["lanes"][LANE_JAX_CPU]["status"] == "pass"
    assert artifact["lanes"][LANE_JAX_GPU]["status"] == "blocked"
    assert artifact["comparisons"]["cpp_cpu_vs_jax_cpu"]["status"] == "pass"
    assert artifact["comparisons"]["cpp_cpu_vs_jax_gpu"]["status"] == "blocked"


def test_fixed_state_cpu_lanes_emit_release_matrix_required_keys():
    artifact = _build_deterministic_artifact()

    for lane_name in (LANE_CPP_CPU, LANE_JAX_CPU):
        lane = artifact["lanes"][lane_name]
        for key in REQUIRED_FIXED_STATE_HASH_KEYS:
            assert key in lane["hashes"]
        for key in REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS:
            assert key in lane["assembled_outputs"]
        for key in REQUIRED_OPERATOR_LANE_OUTPUT_KEYS:
            assert key in lane["operator_outputs"]


def test_fixed_state_matrix_keeps_cpu_only_artifact_blocked_until_gpu_lane():
    artifact = _build_deterministic_artifact()

    bucket = _fixed_state_bucket(artifact)

    assert bucket["status"] == "blocked"
    assert "cpp_cpu_vs_jax_cpu_fixed_state" in bucket["comparisons"]
    assert "jax_gpu: missing assembled output total_objective" in bucket["failures"]


def test_fixed_state_comparison_detects_gradient_drift():
    artifact = _build_deterministic_artifact()
    lhs = artifact["lanes"][LANE_CPP_CPU]
    rhs = dict(artifact["lanes"][LANE_JAX_CPU])
    rhs_assembled = dict(rhs["assembled_outputs"])
    gradient = np.asarray(
        rhs_assembled["full_optimizer_basis_gradient"],
        dtype=np.float64,
    )
    gradient[0] += 1.0
    rhs_assembled["full_optimizer_basis_gradient"] = gradient.tolist()
    rhs["assembled_outputs"] = rhs_assembled

    comparison = _comparison_status(lhs, rhs)

    assert comparison["status"] == "drift"
    assert comparison["grad_max_abs_delta"] == 1.0


def _cpu_artifact_with_fake_cuda_lane():
    cpu_artifact = _build_deterministic_artifact()
    cuda_artifact = _build_deterministic_artifact()
    cuda_artifact["provenance"]["platform_request"] = "cuda"
    cuda_artifact["lanes"][LANE_JAX_GPU] = dict(cuda_artifact["lanes"][LANE_JAX_CPU])
    cuda_artifact["lanes"][LANE_JAX_GPU]["provenance"] = {
        **cuda_artifact["lanes"][LANE_JAX_CPU]["provenance"],
        "lane": LANE_JAX_GPU,
        "platform_request": "cuda",
        "backend": "cuda",
        "devices": ["cuda:0"],
        "nvidia_smi_gpus": [
            {
                "name": "H100",
                "driver_version": "0.0",
                "memory_total_mb": 81920.0,
            }
        ],
        "gpu_memory_mb": 1024.0,
        "peak_gpu_memory_mb": 1024.0,
    }
    return cpu_artifact, cuda_artifact


def test_fixed_state_merge_builds_complete_fixed_state_artifact():
    cpu_artifact, cuda_artifact = _cpu_artifact_with_fake_cuda_lane()

    merged = merge_fixed_state_artifacts([cpu_artifact, cuda_artifact])
    bucket = _fixed_state_bucket(merged)

    assert merged["passed"] is True
    assert merged["comparisons"]["cpp_cpu_vs_jax_cpu"]["status"] == "pass"
    assert merged["comparisons"]["cpp_cpu_vs_jax_gpu"]["status"] == "pass"
    assert merged["comparisons"]["jax_cpu_vs_jax_gpu"]["status"] == "pass"
    assert bucket["status"] == "pass"


def test_fixed_state_merge_fails_on_cross_platform_hash_mismatch():
    cpu_artifact, cuda_artifact = _cpu_artifact_with_fake_cuda_lane()
    cuda_artifact["lanes"][LANE_JAX_GPU]["hashes"] = {
        **cuda_artifact["lanes"][LANE_JAX_GPU]["hashes"],
        "active_dof_mask_hash": "different",
    }

    merged = merge_fixed_state_artifacts([cpu_artifact, cuda_artifact])

    assert merged["passed"] is False
    assert any("active_dof_mask_hash" in failure for failure in merged["failures"])
