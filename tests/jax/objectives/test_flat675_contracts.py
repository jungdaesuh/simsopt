"""Artifact-free contract coverage for the flat-675 port.

The parity gates in ``test_flat675_objective.py`` are anchored on a frozen
input bundle that only exists on the campaign host, so on every other machine
they skip and the port's validation surface goes unexercised.  This module
holds that surface from synthetic payloads: minimal hand-written JSON records
and two-column arrays, no bundle, no device-sized work.

Each rejection test names the exact input a real bundle could carry and the
guard that must refuse it.  The happy-path tests beside them are the controls
— without one, a guard that rejects everything would look identical.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_COIL_DOF_COUNT,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_DOF_COUNT,
    FLAT675_VESSEL_DOF_COUNT,
    Flat675BoozerLabelType,
    Flat675BoozerMaterial,
    Flat675BoozerSystemPolicy,
    Flat675Candidate,
    Flat675ContractError,
    Flat675HardwareSoftPenaltyPolicy,
    Flat675Material,
    Flat675ObjectivePolicy,
    load_flat675_input_manifest,
    load_flat675_runtime_spec,
    load_flat675_vessel_template,
    solve_flat675_y_qr,
)
from simsopt_jax_adapters.geo.flat675._boozer_arrays import (
    build_flat675_boozer_system_arrays,
)
from simsopt_jax_adapters.geo.flat675.policy import flat675_outer_objective_config

# ---------------------------------------------------------------------------
# Synthetic payload builders.  Every test starts from one of these and mutates
# exactly the field under test, so a failure names one input.
# ---------------------------------------------------------------------------


def _candidate_payload() -> dict[str, Any]:
    return {
        "coil_coordinates": [0.1 * index for index in range(FLAT675_COIL_DOF_COUNT)],
        "vessel_coordinates": [0.9, 0.2, 0.2],
        "surface_coordinates": [
            0.001 * index for index in range(FLAT675_SURFACE_DOF_COUNT)
        ],
    }


def _soft_penalty_payload() -> dict[str, Any]:
    return {
        "schema": "simsopt.single_stage.hardware_soft_penalty_policy.v1",
        "common_scale": 10.0,
        "base_weights": {"coil_coil_distance": 100.0, "curvature": 0.1},
        "effective_weights": {"coil_coil_distance": 1000.0, "curvature": 1.0},
        "thresholds": {"coil_coil_distance_m": 0.05, "curvature_inverse_m": 40.0},
        "penalty_exponent": 2.0,
    }


def _objective_policy_payload() -> dict[str, Any]:
    return {
        "schema": ("simsopt.single_stage.fixed_state_genuine_675.objective_policy.v2"),
        "iota_target": 0.15,
        "boozer_label_constraint": {
            "weight": 1.0,
            "target": 0.1,
            "type": "volume",
            "phi_index": 0,
        },
        "weights": {
            "non_qs": 1.0,
            "boozer_residual": 1000.0,
            "iota_penalty": 100.0,
            "curve_length": 1.0,
            "curve_curve_distance": 1000.0,
            "curve_surface_distance": 1.0,
            "surface_vessel_distance": 1000.0,
            "curve_curvature": 1.0,
        },
        "targets": {
            "curve_length_m": 1.7,
            "curve_curve_distance_m": 0.05,
            "curve_surface_distance_m": 0.02,
            "surface_vessel_distance_m": 0.04,
            "curve_curvature_inverse_m": 40.0,
        },
        "banana_curve_index": 20,
        "non_qs_grid_size": 40,
        "curvature_p_norm": 2.0,
        "hardware_soft_penalty_policy": _soft_penalty_payload(),
    }


def _manifest_payload() -> dict[str, Any]:
    return {
        "candidate": _candidate_payload(),
        "objective_policy": _objective_policy_payload(),
        "boozer_construction_policy": {
            "schema": (
                "simsopt.single_stage.experimental_fullspace_675."
                "boozer_system_policy.v1"
            ),
            "weight_by_inverse_field_magnitude": True,
        },
    }


def _vessel_material_payload() -> dict[str, Any]:
    def array(values: list[float], shape: list[int]) -> dict[str, Any]:
        return {"shape": shape, "values": values}

    return {
        "schema": "simsopt.active_surface_vessel_material.v1",
        "surface": {"nfp": 5, "stellsym": True},
        "arrays": {
            "rc": array([0.9, 0.2], [2, 1]),
            "zs": array([0.0, 0.2], [2, 1]),
            "rs": array([0.0, 0.0], [2, 1]),
            "zc": array([0.0, 0.0], [2, 1]),
            "quadpoints_phi": array([0.0, 0.5], [2]),
            "quadpoints_theta": array([0.0, 0.5], [2]),
        },
    }


def _spec_array(values: list[float], shape: list[int]) -> dict[str, Any]:
    return {"dtype": "float64", "shape": shape, "data": values}


def _runtime_spec_payload() -> dict[str, Any]:
    return {
        "schema": "simsopt.single_stage.jax_runtime_spec",
        "schema_version": 1,
        "surface": {
            "surface_class": "SurfaceXYZTensorFourier",
            "dofs": _spec_array([0.0], [1]),
            "mpol": 1,
            "ntor": 0,
            "nfp": 1,
            "stellsym": True,
            "quadpoints_phi": _spec_array([0.0], [1]),
            "quadpoints_theta": _spec_array([0.0], [1]),
        },
        "field": {
            "coil_dof_extraction": {
                "kind": "dataclass",
                "class": "CoilSetDofExtractionSpec",
                "fields": {"coils": {"kind": "tuple", "items": []}},
            },
            "coil_set": {
                "kind": "dataclass",
                "class": "GroupedCoilSetSpec",
                "fields": {"groups": {"kind": "tuple", "items": []}},
            },
            "coil_dofs": _spec_array([1.0], [1]),
            "num_tf_coils": 20,
            "banana_curve_index": 20,
            "tf_current_A": 80000.0,
            "banana_current_A": 15000.0,
        },
        "boozer_init": {"iota": 0.15, "G": 2.0},
        "quadrature": {"nphi": 1, "ntheta": 1},
        "target_labels": ["qs_error"],
        "hardware_constants": {"coil_length_target_m": 1.7},
        "self_intersection_mode": "supported-surface-jax",
    }


def _write_bundle(
    root: Path,
    *,
    manifest: dict[str, Any] | None = None,
    vessel_material: dict[str, Any] | None = None,
    runtime_spec: dict[str, Any] | None = None,
) -> Path:
    """Write the named records of a synthetic bundle directory."""
    if manifest is not None:
        (root / "manifest.json").write_text(json.dumps(manifest))
    if vessel_material is not None:
        (root / "vessel_material.json").write_text(json.dumps(vessel_material))
    if runtime_spec is not None:
        (root / "single_stage_jax_runtime_spec.json").write_text(
            json.dumps(runtime_spec)
        )
    return root


def _objective_policy() -> Flat675ObjectivePolicy:
    """The valid policy the rejection tests below perturb one field of."""
    return Flat675ObjectivePolicy(
        iota_target=0.15,
        boozer_constraint_weight=1.0,
        boozer_target_label=0.1,
        boozer_label_type=Flat675BoozerLabelType.VOLUME,
        boozer_label_phi_index=0,
        non_qs_weight=1.0,
        non_qs_grid_size=40,
        residual_weight=1000.0,
        iota_weight=100.0,
        length_weight=1.0,
        length_target_m=1.7,
        curve_curve_weight=1000.0,
        curve_curve_threshold_m=0.05,
        curve_surface_weight=1.0,
        curve_surface_threshold_m=0.02,
        surface_vessel_weight=1000.0,
        surface_vessel_threshold_m=0.04,
        curvature_weight=1.0,
        curvature_threshold_inverse_m=40.0,
        optimized_coil_index=20,
        hardware_soft_penalty_policy=Flat675HardwareSoftPenaltyPolicy(
            common_scale=10.0,
            curve_curve_base_weight=100.0,
            curvature_base_weight=0.1,
            curve_curve_threshold_m=0.05,
            curvature_threshold_inverse_m=40.0,
            penalty_exponent=2.0,
        ),
    )


# ---------------------------------------------------------------------------
# formulation: the candidate's block sizes and finiteness
# ---------------------------------------------------------------------------


def test_candidate_payload_builds_the_flat_outer_vector_in_owner_order() -> None:
    candidate = Flat675Candidate.from_payload(_candidate_payload())

    vector = candidate.outer_vector()
    assert vector.shape == (FLAT675_OUTER_DOF_COUNT,)
    assert vector.dtype == np.float64
    assert vector[0] == candidate.coil_coordinates[0]
    assert vector[FLAT675_COIL_DOF_COUNT] == candidate.vessel_coordinates[0]
    assert (
        vector[FLAT675_COIL_DOF_COUNT + FLAT675_VESSEL_DOF_COUNT]
        == candidate.surface_coordinates[0]
    )


def test_candidate_payload_rejects_a_missing_owner_block() -> None:
    payload = _candidate_payload()
    del payload["vessel_coordinates"]

    with pytest.raises(Flat675ContractError, match="missing"):
        Flat675Candidate.from_payload(payload)


def test_candidate_payload_rejects_a_wrong_sized_owner_block() -> None:
    payload = _candidate_payload()
    payload["coil_coordinates"] = payload["coil_coordinates"][:-1]

    with pytest.raises(
        Flat675ContractError,
        match=f"exactly {FLAT675_COIL_DOF_COUNT} coordinates",
    ):
        Flat675Candidate.from_payload(payload)


def test_candidate_payload_rejects_a_nonfinite_coordinate() -> None:
    payload = _candidate_payload()
    payload["surface_coordinates"][7] = float("nan")

    with pytest.raises(Flat675ContractError, match="must be finite"):
        Flat675Candidate.from_payload(payload)


def test_candidate_rejects_a_wrong_sized_block_on_direct_construction() -> None:
    """The block-size invariant holds even when no payload was parsed."""
    valid = Flat675Candidate.from_payload(_candidate_payload())

    with pytest.raises(Flat675ContractError, match="must be a tuple of exactly"):
        dataclasses.replace(valid, vessel_coordinates=(0.9, 0.2))


# ---------------------------------------------------------------------------
# policy: the two-source hardware-penalty agreement and the scalar domains
# ---------------------------------------------------------------------------


def test_objective_policy_accepts_agreeing_hardware_penalty_material() -> None:
    policy = _objective_policy()

    assert policy.curvature_p_norm == 2.0
    assert policy.hardware_soft_penalty_policy.curve_curve_effective_weight == 1000.0


def test_objective_policy_rejects_a_curve_curve_weight_off_the_soft_penalty() -> None:
    """A weight that is not common_scale times its base is a second source."""
    with pytest.raises(Flat675ContractError, match="hardware penalties differ"):
        dataclasses.replace(_objective_policy(), curve_curve_weight=999.0)


def test_objective_policy_rejects_a_curvature_threshold_off_the_soft_penalty() -> None:
    with pytest.raises(Flat675ContractError, match="hardware penalties differ"):
        dataclasses.replace(_objective_policy(), curvature_threshold_inverse_m=39.0)


def test_objective_policy_rejects_a_negative_weight() -> None:
    with pytest.raises(Flat675ContractError, match="must be nonnegative"):
        dataclasses.replace(_objective_policy(), non_qs_weight=-1.0)


def test_objective_policy_rejects_a_zero_separation_threshold() -> None:
    with pytest.raises(Flat675ContractError, match="must be positive"):
        dataclasses.replace(_objective_policy(), surface_vessel_threshold_m=0.0)


def test_objective_policy_rejects_an_empty_non_qs_grid() -> None:
    with pytest.raises(Flat675ContractError, match="must be a positive integer"):
        dataclasses.replace(_objective_policy(), non_qs_grid_size=0)


def test_outer_objective_config_renders_the_policy_onto_the_shared_keys() -> None:
    """The config mapping is the port's only translation of policy to math."""
    vessel_gamma = jnp.zeros((2, 2, 3))
    config = flat675_outer_objective_config(
        _objective_policy(),
        nfp=5,
        vessel_gamma=vessel_gamma,
    )

    assert config["surface_vessel_threshold"] == 0.04
    assert config["curvature_p_norm"] == 2.0
    assert config["optimized_coil_index"] == 20
    # The vessel geometry must arrive untouched: converting it here would cut
    # the three vessel coordinates out of the gradient.
    assert config["vessel_gamma"] is vessel_gamma
    phi = np.asarray(config["non_qs_quadpoints_phi"])
    theta = np.asarray(config["non_qs_quadpoints_theta"])
    assert phi.shape == theta.shape == (40,)
    assert phi[0] == 0.0
    assert phi[-1] == pytest.approx((1.0 / 5.0) * (39.0 / 40.0))


# ---------------------------------------------------------------------------
# manifest: the frozen bundle's plain-JSON physics records
# ---------------------------------------------------------------------------


def test_manifest_reads_the_candidate_and_both_policies(tmp_path: Path) -> None:
    _write_bundle(tmp_path, manifest=_manifest_payload())

    manifest = load_flat675_input_manifest(tmp_path)

    assert manifest.candidate.vessel_coordinates == (0.9, 0.2, 0.2)
    assert manifest.objective_policy.optimized_coil_index == 20
    assert manifest.boozer_construction_policy == Flat675BoozerSystemPolicy(
        weight_by_inverse_field_magnitude=True
    )


def test_manifest_rejects_a_curvature_exponent_that_disagrees_with_its_twin(
    tmp_path: Path,
) -> None:
    """The manifest states the curvature exponent twice; both must agree."""
    payload = _manifest_payload()
    payload["objective_policy"]["curvature_p_norm"] = 3.0
    _write_bundle(tmp_path, manifest=payload)

    with pytest.raises(
        Flat675ContractError,
        match="curvature_p_norm differs from the hardware",
    ):
        load_flat675_input_manifest(tmp_path)


def test_manifest_rejects_an_unknown_objective_policy_schema(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["objective_policy"]["schema"] = "some.other.schema.v1"
    _write_bundle(tmp_path, manifest=payload)

    with pytest.raises(Flat675ContractError, match="objective_policy.schema"):
        load_flat675_input_manifest(tmp_path)


def test_manifest_rejects_an_unknown_boozer_policy_schema(tmp_path: Path) -> None:
    payload = _manifest_payload()
    payload["boozer_construction_policy"]["schema"] = "some.other.schema.v1"
    _write_bundle(tmp_path, manifest=payload)

    with pytest.raises(Flat675ContractError, match="boozer_construction_policy.schema"):
        load_flat675_input_manifest(tmp_path)


def test_manifest_rejects_a_missing_weight(tmp_path: Path) -> None:
    payload = _manifest_payload()
    del payload["objective_policy"]["weights"]["non_qs"]
    _write_bundle(tmp_path, manifest=payload)

    with pytest.raises(Flat675ContractError, match="objective_policy.weights"):
        load_flat675_input_manifest(tmp_path)


def test_vessel_material_reads_the_frozen_surface(tmp_path: Path) -> None:
    _write_bundle(tmp_path, vessel_material=_vessel_material_payload())

    template = load_flat675_vessel_template(tmp_path)

    assert template.nfp == 5
    assert template.stellsym is True
    assert template.mpol == 1
    assert template.ntor == 0


def test_vessel_material_rejects_a_nonfinite_coefficient(tmp_path: Path) -> None:
    """A NaN vessel coefficient must not reach the surface-to-vessel term."""
    payload = _vessel_material_payload()
    payload["arrays"]["rc"]["values"][1] = float("nan")
    _write_bundle(tmp_path, vessel_material=payload)

    with pytest.raises(
        Flat675ContractError, match=r"vessel_material\.arrays\.rc must be finite"
    ):
        load_flat675_vessel_template(tmp_path)


def test_vessel_material_rejects_an_unknown_schema(tmp_path: Path) -> None:
    payload = _vessel_material_payload()
    payload["schema"] = "some.other.schema.v1"
    _write_bundle(tmp_path, vessel_material=payload)

    with pytest.raises(Flat675ContractError, match="vessel_material.schema"):
        load_flat675_vessel_template(tmp_path)


# ---------------------------------------------------------------------------
# runtime_spec_loader: the tagged spec-tree wire format
# ---------------------------------------------------------------------------


def test_runtime_spec_decodes_a_minimal_seed(tmp_path: Path) -> None:
    _write_bundle(tmp_path, runtime_spec=_runtime_spec_payload())

    runtime_spec = load_flat675_runtime_spec(tmp_path)

    assert (runtime_spec.mpol, runtime_spec.ntor, runtime_spec.nfp) == (1, 0, 1)
    assert (runtime_spec.nphi, runtime_spec.ntheta) == (1, 1)
    # The wire format still spells the optimized coil the campaign's way.
    assert runtime_spec.seed.optimized_coil_index == 20
    assert runtime_spec.seed.optimized_coil_current_A == 15000.0


def test_runtime_spec_rejects_a_class_outside_the_spec_allowlist(
    tmp_path: Path,
) -> None:
    """A payload may only name immutable spec types this repo publishes."""
    payload = _runtime_spec_payload()
    payload["field"]["coil_set"]["class"] = "Popen"
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(Flat675ContractError, match="unsupported runtime-spec class"):
        load_flat675_runtime_spec(tmp_path)


def test_runtime_spec_rejects_a_decoded_type_that_is_not_the_declared_one(
    tmp_path: Path,
) -> None:
    payload = _runtime_spec_payload()
    payload["field"]["coil_set"] = payload["field"]["coil_dof_extraction"]
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(Flat675ContractError, match="must decode to GroupedCoilSetSpec"):
        load_flat675_runtime_spec(tmp_path)


def test_runtime_spec_rejects_an_unknown_node_kind(tmp_path: Path) -> None:
    payload = _runtime_spec_payload()
    payload["field"]["coil_set"] = {"kind": "callable", "name": "os.system"}
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(Flat675ContractError, match="unsupported node kind"):
        load_flat675_runtime_spec(tmp_path)


def test_runtime_spec_rejects_a_nonfinite_array(tmp_path: Path) -> None:
    payload = _runtime_spec_payload()
    payload["field"]["coil_dofs"]["data"] = [float("inf")]
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(Flat675ContractError, match="must be finite"):
        load_flat675_runtime_spec(tmp_path)


def test_runtime_spec_rejects_an_unknown_schema(tmp_path: Path) -> None:
    payload = _runtime_spec_payload()
    payload["schema"] = "some.other.schema"
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(Flat675ContractError, match="schema"):
        load_flat675_runtime_spec(tmp_path)


def test_runtime_spec_rejects_a_non_tensor_fourier_surface(tmp_path: Path) -> None:
    payload = _runtime_spec_payload()
    payload["surface"]["surface_class"] = "SurfaceRZFourier"
    _write_bundle(tmp_path, runtime_spec=payload)

    with pytest.raises(
        Flat675ContractError, match="SurfaceXYZTensorFourier runtime surface"
    ):
        load_flat675_runtime_spec(tmp_path)


# ---------------------------------------------------------------------------
# boozer material: the layout contract a runtime seed must satisfy
# ---------------------------------------------------------------------------


def _runtime_spec_with(**field_overrides: Any) -> dict[str, Any]:
    """The minimal runtime spec with named surface/field payloads replaced."""
    payload = _runtime_spec_payload()
    for name, value in field_overrides.items():
        section, _, key = name.partition("__")
        payload[section][key] = value
    return payload


def test_boozer_material_rejects_a_seed_without_eleven_coil_dofs(
    tmp_path: Path,
) -> None:
    """The minimal seed is well-formed but is not a flat-675 problem."""
    _write_bundle(tmp_path, runtime_spec=_runtime_spec_payload())
    runtime_spec = load_flat675_runtime_spec(tmp_path)

    with pytest.raises(
        Flat675ContractError,
        match=f"exactly {FLAT675_COIL_DOF_COUNT} coil DOFs",
    ):
        Flat675BoozerMaterial.from_runtime_spec(runtime_spec)


def test_boozer_material_rejects_a_surface_without_661_dofs(tmp_path: Path) -> None:
    """Eleven coil DOFs get past the seed check; the surface must also fit."""
    payload = _runtime_spec_with(
        field__coil_dofs=_spec_array([0.0] * FLAT675_COIL_DOF_COUNT, [11]),
    )
    _write_bundle(tmp_path, runtime_spec=payload)
    runtime_spec = load_flat675_runtime_spec(tmp_path)

    with pytest.raises(
        Flat675ContractError,
        match=f"exactly {FLAT675_SURFACE_DOF_COUNT} DOFs",
    ):
        Flat675BoozerMaterial.from_runtime_spec(runtime_spec)


def test_boozer_material_rejects_an_extraction_that_leaves_coil_dofs_unread(
    tmp_path: Path,
) -> None:
    """Both block sizes fit, but no coil consumes the eleven owner DOFs."""
    payload = _runtime_spec_with(
        field__coil_dofs=_spec_array([0.0] * FLAT675_COIL_DOF_COUNT, [11]),
        surface__dofs=_spec_array([0.0] * FLAT675_SURFACE_DOF_COUNT, [661]),
    )
    _write_bundle(tmp_path, runtime_spec=payload)
    runtime_spec = load_flat675_runtime_spec(tmp_path)

    with pytest.raises(
        Flat675ContractError,
        match=f"every one of {FLAT675_COIL_DOF_COUNT} owner DOFs",
    ):
        Flat675BoozerMaterial.from_runtime_spec(runtime_spec)


def test_material_rejects_a_boozer_record_of_the_wrong_type(
    tmp_path: Path,
) -> None:
    """The guard is deliberately called with the type it exists to refuse."""
    _write_bundle(tmp_path, vessel_material=_vessel_material_payload())
    template = load_flat675_vessel_template(tmp_path)

    with pytest.raises(Flat675ContractError, match="must be Flat675BoozerMaterial"):
        Flat675Material(
            boozer=template,  # type: ignore[arg-type]
            vessel_template=template,
        )


# ---------------------------------------------------------------------------
# y_solve: the two-column least-squares contract
# ---------------------------------------------------------------------------


def test_y_solve_closes_a_small_overdetermined_system() -> None:
    """A/b chosen so the exact least-squares answer is (1, 2)."""
    matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    solved = solve_flat675_y_qr(matrix, rhs)

    np.testing.assert_allclose(
        np.asarray(solved.solution), np.array([1.0, 2.0]), rtol=1.0e-14, atol=0.0
    )
    np.testing.assert_allclose(
        np.sort(np.asarray(solved.singular_values)),
        np.array([1.0, np.sqrt(3.0)]),
        rtol=1.0e-14,
        atol=0.0,
    )
    assert int(solved.numerical_rank) == 2
    assert bool(solved.numerics_finite) is True


def test_y_solve_reports_a_rank_deficient_system() -> None:
    """Both columns parallel: the rank must drop, not the solve."""
    matrix = jnp.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    solved = solve_flat675_y_qr(matrix, rhs)

    assert int(solved.numerical_rank) == 1


def test_y_solve_reports_nonfinite_input() -> None:
    matrix = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 2.0, np.nan], dtype=jnp.float64)

    assert bool(solve_flat675_y_qr(matrix, rhs).numerics_finite) is False


@pytest.mark.parametrize(
    ("matrix", "rhs", "expected_error", "message"),
    [
        (
            jnp.zeros((3, 3), dtype=jnp.float64),
            jnp.zeros(3, dtype=jnp.float64),
            ValueError,
            "shape .m, 2.",
        ),
        (
            jnp.zeros((1, 2), dtype=jnp.float64),
            jnp.zeros(1, dtype=jnp.float64),
            ValueError,
            "at least two rows",
        ),
        (
            jnp.zeros((3, 2), dtype=jnp.float64),
            jnp.zeros(4, dtype=jnp.float64),
            ValueError,
            "right-hand side must have shape",
        ),
        (
            jnp.zeros((3, 2), dtype=jnp.float32),
            jnp.zeros(3, dtype=jnp.float64),
            TypeError,
            "requires float64 inputs",
        ),
    ],
)
def test_y_solve_rejects_a_malformed_system(
    matrix: jnp.ndarray,
    rhs: jnp.ndarray,
    expected_error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(expected_error, match=message):
        solve_flat675_y_qr(matrix, rhs)


# ---------------------------------------------------------------------------
# _boozer_arrays: the A/b assembly kernel
# ---------------------------------------------------------------------------


def test_boozer_arrays_weight_each_point_by_its_inverse_field_magnitude() -> None:
    """|B| = 2 at the single point, so the weighting branch is visible."""
    magnetic_field = jnp.asarray([[[0.0, 0.0, 2.0]]], dtype=jnp.float64)
    toroidal_tangent = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64)
    poloidal_tangent = jnp.asarray([[[0.0, 1.0, 0.0]]], dtype=jnp.float64)
    normalization = np.sqrt(3.0)

    design_matrix, right_hand_side = build_flat675_boozer_system_arrays(
        magnetic_field,
        toroidal_tangent,
        poloidal_tangent,
        weight_by_inverse_field_magnitude=True,
    )

    np.testing.assert_allclose(
        np.asarray(design_matrix),
        np.array([[0.0, 0.0], [-2.0, 0.0], [0.0, 1.0]]) / normalization,
        rtol=1.0e-14,
        atol=1.0e-300,
    )
    np.testing.assert_allclose(
        np.asarray(right_hand_side),
        np.array([2.0, 0.0, 0.0]) / normalization,
        rtol=1.0e-14,
        atol=1.0e-300,
    )


def test_boozer_arrays_leave_points_unweighted_when_the_policy_says_so() -> None:
    magnetic_field = jnp.asarray([[[0.0, 0.0, 2.0]]], dtype=jnp.float64)
    toroidal_tangent = jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64)
    poloidal_tangent = jnp.asarray([[[0.0, 1.0, 0.0]]], dtype=jnp.float64)
    normalization = np.sqrt(3.0)

    design_matrix, right_hand_side = build_flat675_boozer_system_arrays(
        magnetic_field,
        toroidal_tangent,
        poloidal_tangent,
        weight_by_inverse_field_magnitude=False,
    )

    np.testing.assert_allclose(
        np.asarray(design_matrix),
        np.array([[0.0, 0.0], [-4.0, 0.0], [0.0, 2.0]]) / normalization,
        rtol=1.0e-14,
        atol=1.0e-300,
    )
    np.testing.assert_allclose(
        np.asarray(right_hand_side),
        np.array([4.0, 0.0, 0.0]) / normalization,
        rtol=1.0e-14,
        atol=1.0e-300,
    )
