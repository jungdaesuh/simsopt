import numpy as np

from benchmarks.single_stage_dof_mapping_proof import (
    build_coordinate_mapping_proof,
)
from benchmarks.single_stage_parity_matrix import _coordinate_mapping_bucket


REQUIRED_SECTIONS = (
    "inputs",
    "mapping",
    "active_indices",
    "frozen_indices",
    "state_reconstruction",
    "gradient_projection",
    "finite_difference_checks",
)


def test_coordinate_mapping_proof_passes_required_sections():
    artifact = build_coordinate_mapping_proof()

    assert artifact["schema_version"] == 1
    assert artifact["status"] == "pass"
    assert artifact["failures"] == []
    for section in REQUIRED_SECTIONS:
        assert artifact[section]["status"] == "pass"

    mapping = artifact["mapping"]
    assert mapping["entry_count"] == mapping["legacy_x_size"]
    assert mapping["entry_count"] == mapping["target_bs_x_size"]
    assert mapping["max_abs_initial_value_delta"] == 0.0
    assert artifact["frozen_indices"]["target_hits"] == []
    assert len(artifact["finite_difference_checks"]["directions"]) == 3


def test_coordinate_mapping_proof_satisfies_release_matrix_bucket():
    artifact = build_coordinate_mapping_proof()

    bucket = _coordinate_mapping_bucket(artifact)

    assert bucket["status"] == "pass"
    assert bucket["missing_sections"] == []
    assert bucket["failures"] == []


def test_coordinate_mapping_gradient_shape_mismatch_reports_lane_sizes():
    artifact = build_coordinate_mapping_proof(
        target_gradient_override=np.zeros(3, dtype=np.float64),
    )

    gradient_projection = artifact["gradient_projection"]

    assert artifact["status"] == "blocked"
    assert gradient_projection["status"] == "blocked"
    assert gradient_projection["projected_gradient_size"] == 10
    assert gradient_projection["target_gradient_size"] == 3
    assert "cpp_cpu JF.x -> jax_cpu bs.x" in gradient_projection["reason"]
