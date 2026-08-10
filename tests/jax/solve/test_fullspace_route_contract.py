from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, asdict

import pytest
from simsopt_jax.solve.fullspace import (
    CFS_FTR1_POLICY,
    CFS_SQP1_POLICY,
    LEGACY_V1_ROUTE_CONTRACT_SHA256,
    LEGACY_V1_ROUTE_CONTRACT_SIZE_BYTES,
    LEGACY_V1_ROUTES,
    ROUTE_POLICIES,
    ROUTE_SCHEMA_VERSION_V2,
    ROUTE_SCHEMA_VERSION_V3,
    ROUTE_V2_CONTRACT_SHA256,
    ROUTE_V2_CONTRACT_SIZE_BYTES,
    ROUTE_V3_CONTRACT_SHA256,
    ROUTE_V3_CONTRACT_SIZE_BYTES,
    FullSpaceRoute,
    GlobalizationPolicy,
    PromotionStatus,
    frozen_route_contract_payload,
    frozen_route_contract_payload_v1,
    frozen_route_contract_payload_v2,
    frozen_route_contract_payload_v3,
    ftr_route_policy,
    route_policy,
    sqp_route_policy,
)


def test_legacy_v1_route_matrix_contains_exactly_the_four_frozen_routes() -> None:
    assert LEGACY_V1_ROUTES == (
        FullSpaceRoute.CFS_P0,
        FullSpaceRoute.CFS_AL1,
        FullSpaceRoute.CFS_AL2,
        FullSpaceRoute.CFS_AL1_B,
    )
    assert tuple(policy.route for policy in ROUTE_POLICIES) == LEGACY_V1_ROUTES
    assert len({policy.route for policy in ROUTE_POLICIES}) == 4
    assert tuple(policy.promotion_status for policy in ROUTE_POLICIES) == (
        PromotionStatus.DIAGNOSTIC_ONLY,
        PromotionStatus.PROMOTING,
        PromotionStatus.PROMOTING,
        PromotionStatus.CONDITIONAL_PROMOTING,
    )


@pytest.mark.parametrize("route", LEGACY_V1_ROUTES)
def test_route_lookup_returns_the_unique_frozen_policy(route: FullSpaceRoute) -> None:
    policy = route_policy(route)

    assert policy.route is route
    assert policy is next(item for item in ROUTE_POLICIES if item.route is route)


def test_route_enum_appends_new_routes_without_changing_legacy_v1_order() -> None:
    assert tuple(FullSpaceRoute) == (
        *LEGACY_V1_ROUTES,
        FullSpaceRoute.CFS_SQP1,
        FullSpaceRoute.CFS_FTR1,
    )


def test_route_lookups_reject_the_wrong_policy_family_clearly() -> None:
    assert sqp_route_policy(FullSpaceRoute.CFS_SQP1) is CFS_SQP1_POLICY
    with pytest.raises(ValueError, match="uses sqp_route_policy"):
        route_policy(FullSpaceRoute.CFS_SQP1)
    with pytest.raises(ValueError, match="not an SQP route"):
        sqp_route_policy(FullSpaceRoute.CFS_AL1)
    assert ftr_route_policy(FullSpaceRoute.CFS_FTR1) is CFS_FTR1_POLICY
    with pytest.raises(ValueError, match="uses ftr_route_policy"):
        route_policy(FullSpaceRoute.CFS_FTR1)
    with pytest.raises(ValueError, match="not an FTR route"):
        ftr_route_policy(FullSpaceRoute.CFS_SQP1)


def test_all_routes_pin_shared_scaling_convergence_and_resource_budgets() -> None:
    for policy in ROUTE_POLICIES:
        assert policy.maximum_total_inner_iterations == (
            10000 if policy.route is FullSpaceRoute.CFS_AL2 else 1000
        )
        assert policy.lbfgs_memory == 10
        assert policy.function_tolerance == 0.0
        assert policy.maximum_function_evaluations_per_inner_solve == 15000
        assert policy.scaling.variable_scaling == (
            "elementwise bootstrap scale: coil=max(abs(c0),1); "
            "surface=max(abs(surface0),1e-2); iota=max(abs(iota0),1e-1); "
            "G=max(abs(G0),1)"
        )
        assert policy.scaling.boozer_constraint_scaling == (
            "divide each raw component by sqrt(254)"
        )
        assert policy.scaling.volume_constraint_scaling == (
            "divide signed volume residual by abs(volume_target)"
        )
        assert policy.convergence.stationarity_norm == "infinity"
        assert policy.convergence.stationarity_tolerance == 1.0e-7
        assert policy.convergence.constraint_norm == "infinity"
        assert policy.convergence.constraint_tolerance == 1.0e-10
        assert policy.convergence.nonfinite_is_failure is True
        assert policy.resources.first_eval_process_timeout_s == 900.0
        assert policy.resources.canary_10_process_timeout_s == 180.0
        assert policy.resources.canary_100_process_timeout_s == 360.0
        assert policy.resources.complete_process_timeout_s == 900.0
        assert policy.resources.warm_solve_timeout_s == 360.0
        assert policy.resources.maximum_device_memory_fraction == 0.8
        assert policy.resources.initial_h2d_transfers == 1
        assert policy.resources.maximum_hot_h2d_transfers == 0
        assert policy.resources.maximum_hot_d2h_transfers == 0
        assert policy.resources.final_d2h_transfers == 1


def test_p0_is_fixed_penalty_diagnostic_and_al1_is_device_resident_alm() -> None:
    p0 = route_policy(FullSpaceRoute.CFS_P0)
    al1 = route_policy(FullSpaceRoute.CFS_AL1)

    assert p0.maximum_outer_stages == 1
    assert p0.inner_iterations_per_stage == 1000
    assert p0.initial_multiplier == 0.0
    assert p0.initial_penalty == p0.maximum_penalty == 10.0
    assert p0.penalty_growth == 1.0
    assert p0.globalization is GlobalizationPolicy.SEQUENTIAL_ZOOM
    assert p0.candidate_step_sizes == (1.0,)
    assert p0.globalization_batch_width == 1

    assert al1.maximum_outer_stages == 10
    assert al1.inner_iterations_per_stage == 100
    assert al1.initial_multiplier == 0.0
    assert al1.initial_penalty == 10.0
    assert al1.penalty_growth == 10.0
    assert al1.maximum_penalty == 250.0
    assert al1.globalization is GlobalizationPolicy.SEQUENTIAL_ZOOM
    assert al1.candidate_step_sizes == (1.0,)
    assert al1.globalization_batch_width == 1


def test_batched_route_pins_static_candidate_batch() -> None:
    policy = route_policy(FullSpaceRoute.CFS_AL1_B)

    assert policy.maximum_line_search_steps == 8
    assert policy.globalization is GlobalizationPolicy.STATIC_BATCHED_MASKED
    assert policy.candidate_step_sizes == (
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
    )
    assert policy.globalization_batch_width == len(policy.candidate_step_sizes) == 8


def test_al2_changes_only_al1_inner_accuracy_budget_and_route_identity() -> None:
    al1 = route_policy(FullSpaceRoute.CFS_AL1)
    al2 = route_policy(FullSpaceRoute.CFS_AL2)

    assert al2.promotion_status is PromotionStatus.PROMOTING
    assert al2.maximum_outer_stages == al1.maximum_outer_stages == 10
    assert al1.inner_iterations_per_stage == 100
    assert al1.maximum_total_inner_iterations == 1000
    assert al2.inner_iterations_per_stage == 1000
    assert al2.maximum_total_inner_iterations == 10000
    assert al2.lbfgs_memory == al1.lbfgs_memory
    assert al2.function_tolerance == al1.function_tolerance
    assert (
        al2.maximum_function_evaluations_per_inner_solve
        == al1.maximum_function_evaluations_per_inner_solve
    )
    assert al2.maximum_line_search_steps == al1.maximum_line_search_steps
    assert al2.globalization is al1.globalization
    assert al2.candidate_step_sizes == al1.candidate_step_sizes
    assert al2.globalization_batch_width == al1.globalization_batch_width
    assert al2.initial_multiplier == al1.initial_multiplier
    assert al2.initial_penalty == al1.initial_penalty
    assert al2.penalty_growth == al1.penalty_growth
    assert al2.maximum_penalty == al1.maximum_penalty
    assert al2.scaling == al1.scaling
    assert al2.convergence == al1.convergence
    assert al2.resources == al1.resources


def test_route_contract_is_immutable_and_payload_is_detached() -> None:
    with pytest.raises(FrozenInstanceError):
        ROUTE_POLICIES[0].initial_penalty = 20.0  # type: ignore[misc]

    first = frozen_route_contract_payload()
    second = frozen_route_contract_payload()
    assert first == second
    assert first is not second
    assert first["routes"] is not second["routes"]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def test_legacy_v1_route_contract_bytes_are_golden() -> None:
    payload = _canonical_json_bytes(frozen_route_contract_payload())

    assert frozen_route_contract_payload() == frozen_route_contract_payload_v1()
    assert len(payload) == LEGACY_V1_ROUTE_CONTRACT_SIZE_BYTES == 5599
    assert (
        hashlib.sha256(payload).hexdigest()
        == LEGACY_V1_ROUTE_CONTRACT_SHA256
        == "1cac4bd571dac722ae188693b26ab6cc86d2c5ca64f274f2a5b962a625a7b01b"
    )


def test_sqp1_policy_matches_the_reviewed_phase0_contract() -> None:
    policy = sqp_route_policy(FullSpaceRoute.CFS_SQP1)

    with pytest.raises(FrozenInstanceError):
        policy.maximum_iterations = 101  # type: ignore[misc]
    assert policy.promotion_status is PromotionStatus.PROMOTING
    assert policy.maximum_iterations == 100
    assert policy.maximum_joint_evaluations == 1200
    assert policy.reverse_row_batch_width == 8
    assert policy.initial_multiplier == 0.0
    assert policy.initial_bfgs_identity_scale == 1.0
    assert policy.powell_curvature_fraction == 0.2
    assert policy.maximum_consecutive_bfgs_resets == 2
    assert policy.regularization_ladder == (0.0, 1e-12, 1e-10, 1e-8, 1e-6)
    assert policy.kkt_forward_error_tolerance == 1.0e-7
    assert policy.kkt_solution_scaled_residual_tolerance == 1e-10
    assert policy.kkt_relative_residual_tolerance == 1e-10
    assert policy.schur_relative_residual_tolerance == 1e-10
    assert policy.rank_relative_threshold == 1e-12
    assert policy.initial_merit_penalty == 1.0
    assert policy.merit_multiplier_margin == 1.0
    assert policy.armijo_coefficient == 1e-4
    assert policy.candidate_step_sizes == tuple(0.5**index for index in range(11))
    assert policy.maximum_identity_retries == 1
    assert policy.objective_maximum == 4.4822247e-8
    assert policy.scaled_feasibility_tolerance == 1e-10
    assert policy.raw_kkt_stationarity_tolerance == 1e-7
    assert policy.derivative_one_step_process_timeout_s == 300.0
    assert policy.ten_step_process_timeout_s == 600.0
    assert policy.complete_process_timeout_s == 900.0
    assert policy.maximum_device_memory_fraction == 0.8
    assert policy.maximum_hot_h2d_transfers == 0
    assert policy.maximum_hot_d2h_transfers == 0
    assert policy.warm_synchronized_solve_max_s == 287.30421751597896


def test_route_v2_payload_embeds_frozen_v1_and_detached_sqp_contract() -> None:
    first = frozen_route_contract_payload_v2()
    second = frozen_route_contract_payload_v2()

    assert first == second
    assert first is not second
    assert first["schema_version"] == ROUTE_SCHEMA_VERSION_V2
    assert first["legacy_v1"] == frozen_route_contract_payload()
    assert first["sqp_routes"] == [asdict(CFS_SQP1_POLICY)]
    assert first["legacy_v1"] is not second["legacy_v1"]
    assert first["sqp_routes"] is not second["sqp_routes"]
    encoded = _canonical_json_bytes(first)
    assert len(encoded) == ROUTE_V2_CONTRACT_SIZE_BYTES == 6891
    assert (
        hashlib.sha256(encoded).hexdigest()
        == ROUTE_V2_CONTRACT_SHA256
        == "b3b36797924331721d36221c29f94a9d464c6f72812f47fad360085be0b37287"
    )


def test_route_v3_payload_adds_ftr_without_changing_v2_bytes() -> None:
    payload = frozen_route_contract_payload_v3()

    assert payload["schema_version"] == ROUTE_SCHEMA_VERSION_V3
    assert payload["legacy_v2"] == frozen_route_contract_payload_v2()
    assert payload["ftr_routes"] == [asdict(CFS_FTR1_POLICY)]
    assert (
        hashlib.sha256(_canonical_json_bytes(payload["legacy_v2"])).hexdigest()
        == ROUTE_V2_CONTRACT_SHA256
    )
    encoded = _canonical_json_bytes(payload)
    assert len(encoded) == ROUTE_V3_CONTRACT_SIZE_BYTES == 8179
    assert (
        hashlib.sha256(encoded).hexdigest()
        == ROUTE_V3_CONTRACT_SHA256
        == "c33782b484822441ddbfe60939fd7bfc7794e5c0a77a8e39b82f37c470fedbe6"
    )
