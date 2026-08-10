"""Typed route contract for device-resident single-stage full-space solves.

The numerical state machine is added behind these frozen policies. Keeping the
policy declarative makes route selection and receipts independently auditable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Final, NamedTuple

import jax
import jax.numpy as jnp

from simsopt_jax.geo.optimizers.fused_lbfgs import (
    FusedLBFGSOptions,
    FusedLBFGSResult,
    PreparedFusedLBFGS,
    PreparedParametricFusedLBFGS,
    prepare_fused_lbfgs,
    prepare_parametric_fused_lbfgs,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    FullSpaceProblem,
    FullSpaceState,
    evaluate_fullspace,
    flatten_fullspace_constraints,
    fullspace_kkt_primitives,
)

SCHEMA_VERSION = "single-stage-fullspace-routes-v1"
ROUTE_SCHEMA_VERSION_V2: Final = "single-stage-fullspace-routes-v2"
ROUTE_SCHEMA_VERSION_V3: Final = "single-stage-fullspace-routes-v3"
LEGACY_V1_ROUTE_CONTRACT_SIZE_BYTES: Final = 5599
LEGACY_V1_ROUTE_CONTRACT_SHA256: Final = (
    "1cac4bd571dac722ae188693b26ab6cc86d2c5ca64f274f2a5b962a625a7b01b"
)
ROUTE_V2_CONTRACT_SIZE_BYTES: Final = 6891
ROUTE_V2_CONTRACT_SHA256: Final = (
    "b3b36797924331721d36221c29f94a9d464c6f72812f47fad360085be0b37287"
)


class FullSpaceRoute(StrEnum):
    CFS_P0 = "CFS-P0"
    CFS_AL1 = "CFS-AL1"
    CFS_AL2 = "CFS-AL2"
    CFS_AL1_B = "CFS-AL1-B"
    CFS_SQP1 = "CFS-SQP1"
    CFS_FTR1 = "CFS-FTR1"


LEGACY_V1_ROUTES: Final = (
    FullSpaceRoute.CFS_P0,
    FullSpaceRoute.CFS_AL1,
    FullSpaceRoute.CFS_AL2,
    FullSpaceRoute.CFS_AL1_B,
)


class PromotionStatus(StrEnum):
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
    PROMOTING = "PROMOTING"
    CONDITIONAL_PROMOTING = "CONDITIONAL_PROMOTING"


class GlobalizationPolicy(StrEnum):
    SEQUENTIAL_ZOOM = "SEQUENTIAL_ZOOM"
    STATIC_BATCHED_MASKED = "STATIC_BATCHED_MASKED"


@dataclass(frozen=True, slots=True)
class ConvergencePolicy:
    stationarity_norm: str
    stationarity_tolerance: float
    constraint_norm: str
    constraint_tolerance: float
    nonfinite_is_failure: bool


@dataclass(frozen=True, slots=True)
class ScalingPolicy:
    variable_scaling: str
    boozer_constraint_scaling: str
    volume_constraint_scaling: str


@dataclass(frozen=True, slots=True)
class ResourceBudget:
    first_eval_process_timeout_s: float
    canary_10_process_timeout_s: float
    canary_100_process_timeout_s: float
    complete_process_timeout_s: float
    warm_solve_timeout_s: float
    maximum_device_memory_fraction: float
    initial_h2d_transfers: int
    maximum_hot_h2d_transfers: int
    maximum_hot_d2h_transfers: int
    final_d2h_transfers: int


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    route: FullSpaceRoute
    promotion_status: PromotionStatus
    maximum_total_inner_iterations: int
    maximum_outer_stages: int
    inner_iterations_per_stage: int
    lbfgs_memory: int
    function_tolerance: float
    maximum_function_evaluations_per_inner_solve: int
    maximum_line_search_steps: int
    globalization: GlobalizationPolicy
    candidate_step_sizes: tuple[float, ...]
    globalization_batch_width: int
    initial_multiplier: float
    initial_penalty: float
    penalty_growth: float
    maximum_penalty: float
    scaling: ScalingPolicy
    convergence: ConvergencePolicy
    resources: ResourceBudget
    selection_rule: str


@dataclass(frozen=True, slots=True)
class CfsSqp1Policy:
    """Frozen controls for the dense-Schur primal-dual SQP route."""

    route: FullSpaceRoute
    promotion_status: PromotionStatus
    maximum_iterations: int
    maximum_joint_evaluations: int
    reverse_row_batch_width: int
    initial_multiplier: float
    initial_bfgs_identity_scale: float
    powell_curvature_fraction: float
    maximum_consecutive_bfgs_resets: int
    regularization_ladder: tuple[float, ...]
    kkt_forward_error_tolerance: float
    kkt_solution_scaled_residual_tolerance: float
    kkt_relative_residual_tolerance: float
    schur_relative_residual_tolerance: float
    rank_relative_threshold: float
    initial_merit_penalty: float
    merit_multiplier_margin: float
    armijo_coefficient: float
    candidate_step_sizes: tuple[float, ...]
    maximum_identity_retries: int
    objective_maximum: float
    scaled_feasibility_tolerance: float
    raw_kkt_stationarity_tolerance: float
    derivative_one_step_process_timeout_s: float
    ten_step_process_timeout_s: float
    complete_process_timeout_s: float
    maximum_device_memory_fraction: float
    maximum_hot_h2d_transfers: int
    maximum_hot_d2h_transfers: int
    warm_synchronized_solve_max_s: float
    selection_rule: str


@dataclass(frozen=True, slots=True)
class CfsFtr1Policy:
    """Frozen controls for the filter/trust-region fullspace route."""

    route: FullSpaceRoute
    promotion_status: PromotionStatus
    maximum_iterations: int
    maximum_joint_evaluations: int
    reverse_row_batch_width: int
    initial_bfgs_identity_scale: float
    powell_curvature_fraction: float
    maximum_consecutive_bfgs_resets: int
    initial_trust_radius: float
    minimum_trust_radius: float
    maximum_trust_radius: float
    normal_radius_fraction: float
    maximum_tangential_cg_iterations: int
    filter_gamma_feasibility: float
    filter_gamma_objective: float
    objective_step_threshold: float
    acceptance_ratio: float
    radius_shrink_ratio: float
    expansion_ratio: float
    radius_contraction: float
    radius_expansion: float
    boundary_fraction: float
    linear_solve_relative_residual_tolerance: float
    linear_solve_forward_error_tolerance: float
    tangency_relative_residual_tolerance: float
    objective_maximum: float
    scaled_feasibility_tolerance: float
    raw_kkt_stationarity_tolerance: float
    ten_step_process_timeout_s: float
    complete_process_timeout_s: float
    maximum_device_memory_fraction: float
    maximum_hot_h2d_transfers: int
    maximum_hot_d2h_transfers: int
    warm_synchronized_solve_max_s: float
    selection_rule: str


@dataclass(frozen=True, slots=True)
class FullSpaceScaling:
    """Frozen diagonal maps between optimizer, physical, and equality spaces."""

    bootstrap_anchor: jax.Array
    variable_scale: jax.Array
    constraint_inverse_scale: jax.Array


jax.tree_util.register_dataclass(
    FullSpaceScaling,
    data_fields=[
        "bootstrap_anchor",
        "variable_scale",
        "constraint_inverse_scale",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsP0Diagnostics:
    """Device-side merit and feasibility values at one CFS-P0 coordinate."""

    physical_state: jax.Array
    physical_objective: jax.Array
    scaled_penalty_value: jax.Array
    raw_constraint_infinity_norm: jax.Array
    scaled_constraint_infinity_norm: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsP0Diagnostics,
    data_fields=[
        "physical_state",
        "physical_objective",
        "scaled_penalty_value",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsAl1Diagnostics:
    """Device-side augmented-Lagrangian quantities at one joint state."""

    physical_state: jax.Array
    physical_objective: jax.Array
    augmented_lagrangian_value: jax.Array
    raw_constraints: jax.Array
    scaled_constraints: jax.Array
    raw_constraint_infinity_norm: jax.Array
    scaled_constraint_infinity_norm: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsAl1Diagnostics,
    data_fields=[
        "physical_state",
        "physical_objective",
        "augmented_lagrangian_value",
        "raw_constraints",
        "scaled_constraints",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsAl1StageHistory:
    """Fixed-shape on-device history for the ten frozen AL stages."""

    penalty: jax.Array
    physical_objective: jax.Array
    scaled_feasibility_infinity_norm: jax.Array
    augmented_stationarity_infinity_norm: jax.Array
    optimizer_status: jax.Array
    inner_iterations: jax.Array
    function_evaluations: jax.Array
    stage_completed: jax.Array


jax.tree_util.register_dataclass(
    CfsAl1StageHistory,
    data_fields=[
        "penalty",
        "physical_objective",
        "scaled_feasibility_infinity_norm",
        "augmented_stationarity_infinity_norm",
        "optimizer_status",
        "inner_iterations",
        "function_evaluations",
        "stage_completed",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class CfsAl1Result:
    """Terminal device-array evidence for the promoting base AL route."""

    optimizer_coordinates: jax.Array
    physical_state: jax.Array
    scaled_multipliers: jax.Array
    raw_multipliers: jax.Array
    next_penalty: jax.Array
    physical_objective: jax.Array
    raw_constraint_infinity_norm: jax.Array
    scaled_constraint_infinity_norm: jax.Array
    raw_kkt_stationarity_infinity_norm: jax.Array
    completed_outer_stages: jax.Array
    total_inner_iterations: jax.Array
    total_function_evaluations: jax.Array
    total_gradient_evaluations: jax.Array
    nonfinite_evaluation_count: jax.Array
    all_accepted_states_finite: jax.Array
    converged: jax.Array
    fatal: jax.Array
    all_finite: jax.Array
    stage_history: CfsAl1StageHistory


jax.tree_util.register_dataclass(
    CfsAl1Result,
    data_fields=[
        "optimizer_coordinates",
        "physical_state",
        "scaled_multipliers",
        "raw_multipliers",
        "next_penalty",
        "physical_objective",
        "raw_constraint_infinity_norm",
        "scaled_constraint_infinity_norm",
        "raw_kkt_stationarity_infinity_norm",
        "completed_outer_stages",
        "total_inner_iterations",
        "total_function_evaluations",
        "total_gradient_evaluations",
        "nonfinite_evaluation_count",
        "all_accepted_states_finite",
        "converged",
        "fatal",
        "all_finite",
        "stage_history",
    ],
    meta_fields=[],
)


_CONVERGENCE = ConvergencePolicy(
    stationarity_norm="infinity",
    stationarity_tolerance=1.0e-7,
    constraint_norm="infinity",
    constraint_tolerance=1.0e-10,
    nonfinite_is_failure=True,
)
_SCALING = ScalingPolicy(
    variable_scaling=(
        "elementwise bootstrap scale: coil=max(abs(c0),1); "
        "surface=max(abs(surface0),1e-2); iota=max(abs(iota0),1e-1); "
        "G=max(abs(G0),1)"
    ),
    boozer_constraint_scaling="divide each raw component by sqrt(254)",
    volume_constraint_scaling="divide signed volume residual by abs(volume_target)",
)
_RESOURCES = ResourceBudget(
    first_eval_process_timeout_s=900.0,
    canary_10_process_timeout_s=180.0,
    canary_100_process_timeout_s=360.0,
    complete_process_timeout_s=900.0,
    warm_solve_timeout_s=360.0,
    maximum_device_memory_fraction=0.8,
    initial_h2d_transfers=1,
    maximum_hot_h2d_transfers=0,
    maximum_hot_d2h_transfers=0,
    final_d2h_transfers=1,
)
_SEQUENTIAL_STEPS = (1.0,)
_BATCHED_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125)


ROUTE_POLICIES = (
    RoutePolicy(
        route=FullSpaceRoute.CFS_P0,
        promotion_status=PromotionStatus.DIAGNOSTIC_ONLY,
        maximum_total_inner_iterations=1000,
        maximum_outer_stages=1,
        inner_iterations_per_stage=1000,
        lbfgs_memory=10,
        function_tolerance=0.0,
        maximum_function_evaluations_per_inner_solve=15000,
        maximum_line_search_steps=30,
        globalization=GlobalizationPolicy.SEQUENTIAL_ZOOM,
        candidate_step_sizes=_SEQUENTIAL_STEPS,
        globalization_batch_width=1,
        initial_multiplier=0.0,
        initial_penalty=10.0,
        penalty_growth=1.0,
        maximum_penalty=10.0,
        scaling=_SCALING,
        convergence=_CONVERGENCE,
        resources=_RESOURCES,
        selection_rule="diagnostic gate; never promotion eligible",
    ),
    RoutePolicy(
        route=FullSpaceRoute.CFS_AL1,
        promotion_status=PromotionStatus.PROMOTING,
        maximum_total_inner_iterations=1000,
        maximum_outer_stages=10,
        inner_iterations_per_stage=100,
        lbfgs_memory=10,
        function_tolerance=0.0,
        maximum_function_evaluations_per_inner_solve=15000,
        maximum_line_search_steps=30,
        globalization=GlobalizationPolicy.SEQUENTIAL_ZOOM,
        candidate_step_sizes=_SEQUENTIAL_STEPS,
        globalization_batch_width=1,
        initial_multiplier=0.0,
        initial_penalty=10.0,
        penalty_growth=10.0,
        maximum_penalty=250.0,
        scaling=_SCALING,
        convergence=_CONVERGENCE,
        resources=_RESOURCES,
        selection_rule="only after CFS-P0 shared correctness and transfer gates pass",
    ),
    RoutePolicy(
        route=FullSpaceRoute.CFS_AL2,
        promotion_status=PromotionStatus.PROMOTING,
        maximum_total_inner_iterations=10000,
        maximum_outer_stages=10,
        inner_iterations_per_stage=1000,
        lbfgs_memory=10,
        function_tolerance=0.0,
        maximum_function_evaluations_per_inner_solve=15000,
        maximum_line_search_steps=30,
        globalization=GlobalizationPolicy.SEQUENTIAL_ZOOM,
        candidate_step_sizes=_SEQUENTIAL_STEPS,
        globalization_batch_width=1,
        initial_multiplier=0.0,
        initial_penalty=10.0,
        penalty_growth=10.0,
        maximum_penalty=250.0,
        scaling=_SCALING,
        convergence=_CONVERGENCE,
        resources=_RESOURCES,
        selection_rule=("accuracy-budget escalation only after sealed CFS-AL1 failure"),
    ),
    RoutePolicy(
        route=FullSpaceRoute.CFS_AL1_B,
        promotion_status=PromotionStatus.CONDITIONAL_PROMOTING,
        maximum_total_inner_iterations=1000,
        maximum_outer_stages=10,
        inner_iterations_per_stage=100,
        lbfgs_memory=10,
        function_tolerance=0.0,
        maximum_function_evaluations_per_inner_solve=15000,
        maximum_line_search_steps=8,
        globalization=GlobalizationPolicy.STATIC_BATCHED_MASKED,
        candidate_step_sizes=_BATCHED_STEPS,
        globalization_batch_width=8,
        initial_multiplier=0.0,
        initial_penalty=10.0,
        penalty_growth=10.0,
        maximum_penalty=250.0,
        scaling=_SCALING,
        convergence=_CONVERGENCE,
        resources=_RESOURCES,
        selection_rule=(
            "only when CFS-AL1 passes correctness and feasibility progress, "
            "globalization fragmentation is at least 20 percent, and memory passes"
        ),
    ),
)

_SQP1_CANDIDATE_STEPS = tuple(0.5**index for index in range(11))

CFS_SQP1_POLICY = CfsSqp1Policy(
    route=FullSpaceRoute.CFS_SQP1,
    promotion_status=PromotionStatus.PROMOTING,
    maximum_iterations=100,
    maximum_joint_evaluations=1200,
    reverse_row_batch_width=8,
    initial_multiplier=0.0,
    initial_bfgs_identity_scale=1.0,
    powell_curvature_fraction=0.2,
    maximum_consecutive_bfgs_resets=2,
    regularization_ladder=(0.0, 1.0e-12, 1.0e-10, 1.0e-8, 1.0e-6),
    kkt_forward_error_tolerance=1.0e-7,
    kkt_solution_scaled_residual_tolerance=1.0e-10,
    kkt_relative_residual_tolerance=1.0e-10,
    schur_relative_residual_tolerance=1.0e-10,
    rank_relative_threshold=1.0e-12,
    initial_merit_penalty=1.0,
    merit_multiplier_margin=1.0,
    armijo_coefficient=1.0e-4,
    candidate_step_sizes=_SQP1_CANDIDATE_STEPS,
    maximum_identity_retries=1,
    objective_maximum=4.4822247e-8,
    scaled_feasibility_tolerance=1.0e-10,
    raw_kkt_stationarity_tolerance=1.0e-7,
    derivative_one_step_process_timeout_s=300.0,
    ten_step_process_timeout_s=600.0,
    complete_process_timeout_s=900.0,
    maximum_device_memory_fraction=0.8,
    maximum_hot_h2d_transfers=0,
    maximum_hot_d2h_transfers=0,
    warm_synchronized_solve_max_s=287.30421751597896,
    selection_rule=(
        "new GPU-native dense-Schur SQP tranche after bounded-negative AL closure"
    ),
)

CFS_FTR1_POLICY = CfsFtr1Policy(
    route=FullSpaceRoute.CFS_FTR1,
    promotion_status=PromotionStatus.PROMOTING,
    maximum_iterations=100,
    maximum_joint_evaluations=1200,
    reverse_row_batch_width=8,
    initial_bfgs_identity_scale=1.0,
    powell_curvature_fraction=0.2,
    maximum_consecutive_bfgs_resets=2,
    initial_trust_radius=1.0,
    minimum_trust_radius=2.0**-20,
    maximum_trust_radius=8.0,
    normal_radius_fraction=0.8,
    maximum_tangential_cg_iterations=64,
    filter_gamma_feasibility=1.0e-4,
    filter_gamma_objective=1.0e-4,
    objective_step_threshold=1.0e-4,
    acceptance_ratio=0.1,
    radius_shrink_ratio=0.25,
    expansion_ratio=0.75,
    radius_contraction=0.25,
    radius_expansion=2.0,
    boundary_fraction=0.8,
    linear_solve_relative_residual_tolerance=1.0e-10,
    linear_solve_forward_error_tolerance=1.0e-7,
    tangency_relative_residual_tolerance=1.0e-10,
    objective_maximum=4.4822247e-8,
    scaled_feasibility_tolerance=1.0e-10,
    raw_kkt_stationarity_tolerance=1.0e-7,
    ten_step_process_timeout_s=600.0,
    complete_process_timeout_s=900.0,
    maximum_device_memory_fraction=0.8,
    maximum_hot_h2d_transfers=0,
    maximum_hot_d2h_transfers=0,
    warm_synchronized_solve_max_s=287.30421751597896,
    selection_rule=(
        "filter/trust-region successor after bounded-negative AL and SQP routes"
    ),
)


def route_policy(route: FullSpaceRoute) -> RoutePolicy:
    """Return the one frozen policy for ``route``."""

    for policy in ROUTE_POLICIES:
        if policy.route is route:
            return policy
    if route is FullSpaceRoute.CFS_SQP1:
        raise ValueError("CFS-SQP1 uses sqp_route_policy(), not the legacy AL policy")
    if route is FullSpaceRoute.CFS_FTR1:
        raise ValueError("CFS-FTR1 uses ftr_route_policy(), not the legacy AL policy")
    raise ValueError(f"unsupported full-space route: {route!r}")


def sqp_route_policy(route: FullSpaceRoute) -> CfsSqp1Policy:
    """Return the frozen SQP policy, rejecting non-SQP route identities."""

    if route is not FullSpaceRoute.CFS_SQP1:
        raise ValueError(f"route is not an SQP route: {route.value}")
    return CFS_SQP1_POLICY


def ftr_route_policy(route: FullSpaceRoute) -> CfsFtr1Policy:
    """Return the frozen FTR policy, rejecting other route identities."""

    if route is not FullSpaceRoute.CFS_FTR1:
        raise ValueError(f"route is not an FTR route: {route.value}")
    return CFS_FTR1_POLICY


def fullspace_scaling_from_bootstrap(
    bootstrap_state: jax.Array,
    problem: FullSpaceProblem,
) -> FullSpaceScaling:
    """Construct the frozen diagonal scaling maps from the feasible bootstrap."""

    state = problem.layout.unpack(bootstrap_state)
    dtype = bootstrap_state.dtype
    variable_scale = problem.layout.pack(
        FullSpaceState(
            coil_dofs=jnp.maximum(
                jnp.abs(state.coil_dofs), jnp.asarray(1.0, dtype=dtype)
            ),
            surface_dofs=jnp.maximum(
                jnp.abs(state.surface_dofs), jnp.asarray(1.0e-2, dtype=dtype)
            ),
            iota=jnp.maximum(jnp.abs(state.iota), jnp.asarray(1.0e-1, dtype=dtype)),
            G=jnp.maximum(jnp.abs(state.G), jnp.asarray(1.0, dtype=dtype)),
        )
    )
    boozer_count = problem.exact_mask_indices.shape[0]
    if boozer_count != 254:
        raise ValueError(
            "CFS routes require the frozen 254-component Boozer equality mask"
        )
    constraint_inverse_scale = jnp.concatenate(
        (
            jnp.full(
                (boozer_count,),
                1.0 / jnp.sqrt(jnp.asarray(boozer_count, dtype=dtype)),
                dtype=dtype,
            ),
            jnp.reshape(1.0 / jnp.abs(problem.config.volume_target), (1,)),
        )
    )
    return FullSpaceScaling(
        bootstrap_anchor=bootstrap_state,
        variable_scale=variable_scale,
        constraint_inverse_scale=constraint_inverse_scale,
    )


def fullspace_optimizer_coordinates(
    physical_state: jax.Array,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Map physical joint coordinates to dimensionless optimizer coordinates."""

    return (physical_state - scaling.bootstrap_anchor) / scaling.variable_scale


def fullspace_physical_coordinates(
    optimizer_coordinates: jax.Array,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Map dimensionless optimizer coordinates to physical joint coordinates."""

    return scaling.bootstrap_anchor + optimizer_coordinates * scaling.variable_scale


def fullspace_scaled_constraints(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Return Boozer and volume equalities in their frozen scaled norm."""

    physical_state = fullspace_physical_coordinates(optimizer_coordinates, scaling)
    evaluation = evaluate_fullspace(physical_state, problem)
    return (
        flatten_fullspace_constraints(evaluation.constraints)
        * scaling.constraint_inverse_scale
    )


def _cfs_p0_value_and_diagnostics(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[jax.Array, CfsP0Diagnostics]:
    physical_state = fullspace_physical_coordinates(optimizer_coordinates, scaling)
    evaluation = evaluate_fullspace(physical_state, problem)
    raw_constraints = flatten_fullspace_constraints(evaluation.constraints)
    scaled_constraints = raw_constraints * scaling.constraint_inverse_scale
    penalty = jnp.asarray(
        route_policy(FullSpaceRoute.CFS_P0).initial_penalty,
        dtype=optimizer_coordinates.dtype,
    )
    scaled_penalty_value = evaluation.weighted_total + (
        jnp.asarray(0.5, dtype=optimizer_coordinates.dtype)
        * penalty
        * jnp.vdot(scaled_constraints, scaled_constraints)
    )
    all_finite = (
        jnp.all(jnp.isfinite(physical_state))
        & jnp.isfinite(evaluation.weighted_total)
        & jnp.isfinite(scaled_penalty_value)
        & jnp.all(jnp.isfinite(raw_constraints))
        & jnp.all(jnp.isfinite(scaled_constraints))
    )
    diagnostics = CfsP0Diagnostics(
        physical_state=physical_state,
        physical_objective=evaluation.weighted_total,
        scaled_penalty_value=scaled_penalty_value,
        raw_constraint_infinity_norm=jnp.max(jnp.abs(raw_constraints)),
        scaled_constraint_infinity_norm=jnp.max(jnp.abs(scaled_constraints)),
        all_finite=all_finite,
    )
    return scaled_penalty_value, diagnostics


def cfs_p0_penalty_value(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Evaluate the frozen diagnostic weighted-penalty objective for CFS-P0."""

    value, _ = _cfs_p0_value_and_diagnostics(
        optimizer_coordinates,
        problem,
        scaling,
    )
    return value


def cfs_p0_value_and_grad(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[tuple[jax.Array, CfsP0Diagnostics], jax.Array]:
    """Evaluate CFS-P0 merit, diagnostics, and gradient in one primal traversal."""

    return jax.value_and_grad(_cfs_p0_value_and_diagnostics, has_aux=True)(
        optimizer_coordinates,
        problem,
        scaling,
    )


def cfs_p0_diagnostics(
    optimizer_coordinates: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> CfsP0Diagnostics:
    """Compute transfer-free merit and feasibility evidence for one CFS-P0 state."""

    _, diagnostics = _cfs_p0_value_and_diagnostics(
        optimizer_coordinates,
        problem,
        scaling,
    )
    return diagnostics


_cfs_p0_diagnostics_kernel = jax.jit(cfs_p0_diagnostics)


def pack_cfs_al1_parameters(
    scaled_multipliers: jax.Array,
    penalty: jax.Array,
) -> jax.Array:
    """Pack scaled equality multipliers and penalty into one dynamic operand."""

    return jnp.concatenate((scaled_multipliers, jnp.reshape(penalty, (1,))))


def unpack_cfs_al1_parameters(
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Unpack the fixed-shape CFS-AL1 dynamic objective operand."""

    return parameters[:-1], parameters[-1]


def _cfs_al1_value_and_diagnostics(
    optimizer_coordinates: jax.Array,
    parameters: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[jax.Array, CfsAl1Diagnostics]:
    scaled_multipliers, penalty = unpack_cfs_al1_parameters(parameters)
    physical_state = fullspace_physical_coordinates(optimizer_coordinates, scaling)
    evaluation = evaluate_fullspace(physical_state, problem)
    raw_constraints = flatten_fullspace_constraints(evaluation.constraints)
    scaled_constraints = raw_constraints * scaling.constraint_inverse_scale
    augmented_value = (
        evaluation.weighted_total
        + jnp.vdot(scaled_multipliers, scaled_constraints)
        + jnp.asarray(0.5, dtype=optimizer_coordinates.dtype)
        * penalty
        * jnp.vdot(scaled_constraints, scaled_constraints)
    )
    all_finite = (
        jnp.all(jnp.isfinite(physical_state))
        & jnp.all(jnp.isfinite(parameters))
        & jnp.isfinite(evaluation.weighted_total)
        & jnp.isfinite(augmented_value)
        & jnp.all(jnp.isfinite(raw_constraints))
        & jnp.all(jnp.isfinite(scaled_constraints))
    )
    diagnostics = CfsAl1Diagnostics(
        physical_state=physical_state,
        physical_objective=evaluation.weighted_total,
        augmented_lagrangian_value=augmented_value,
        raw_constraints=raw_constraints,
        scaled_constraints=scaled_constraints,
        raw_constraint_infinity_norm=jnp.max(jnp.abs(raw_constraints)),
        scaled_constraint_infinity_norm=jnp.max(jnp.abs(scaled_constraints)),
        all_finite=all_finite,
    )
    return augmented_value, diagnostics


def cfs_al1_value(
    optimizer_coordinates: jax.Array,
    parameters: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Evaluate the frozen scaled CFS-AL1 augmented Lagrangian."""

    value, _ = _cfs_al1_value_and_diagnostics(
        optimizer_coordinates,
        parameters,
        problem,
        scaling,
    )
    return value


def cfs_al1_value_and_grad(
    optimizer_coordinates: jax.Array,
    parameters: jax.Array,
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
) -> tuple[tuple[jax.Array, CfsAl1Diagnostics], jax.Array]:
    """Evaluate one CFS-AL1 stage value, diagnostics, and scaled gradient."""

    return jax.value_and_grad(_cfs_al1_value_and_diagnostics, has_aux=True)(
        optimizer_coordinates,
        parameters,
        problem,
        scaling,
    )


def update_cfs_al1_parameters(
    parameters: jax.Array,
    scaled_constraints: jax.Array,
) -> jax.Array:
    """Apply the frozen post-stage multiplier and capped-penalty update."""

    return _update_cfs_al_parameters(
        parameters,
        scaled_constraints,
        route_policy(FullSpaceRoute.CFS_AL1),
    )


def _update_cfs_al_parameters(
    parameters: jax.Array,
    scaled_constraints: jax.Array,
    policy: RoutePolicy,
) -> jax.Array:
    scaled_multipliers, penalty = unpack_cfs_al1_parameters(parameters)
    updated_multipliers = scaled_multipliers + penalty * scaled_constraints
    maximum_penalty = jnp.asarray(policy.maximum_penalty, dtype=parameters.dtype)
    growth = jnp.asarray(policy.penalty_growth, dtype=parameters.dtype)
    updated_penalty = jnp.minimum(growth * penalty, maximum_penalty)
    return pack_cfs_al1_parameters(updated_multipliers, updated_penalty)


def cfs_al1_raw_multipliers(
    scaled_multipliers: jax.Array,
    scaling: FullSpaceScaling,
) -> jax.Array:
    """Convert scaled-equality multipliers for the raw-q KKT convention."""

    return scaling.constraint_inverse_scale * scaled_multipliers


class _CfsAl1Carry(NamedTuple):
    optimizer_coordinates: jax.Array
    parameters: jax.Array
    completed_stages: jax.Array
    total_inner_iterations: jax.Array
    total_function_evaluations: jax.Array
    total_gradient_evaluations: jax.Array
    nonfinite_evaluation_count: jax.Array
    all_accepted_states_finite: jax.Array
    fatal: jax.Array


class _CfsAl1StageRecord(NamedTuple):
    penalty: jax.Array
    physical_objective: jax.Array
    scaled_feasibility_infinity_norm: jax.Array
    augmented_stationarity_infinity_norm: jax.Array
    optimizer_status: jax.Array
    inner_iterations: jax.Array
    function_evaluations: jax.Array
    stage_completed: jax.Array


_CfsAlCompiledRun = Callable[[jax.Array, jax.Array], CfsAl1Result]


class _PreparedCfsAlComponents(NamedTuple):
    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_parameters: jax.Array
    initial_value: jax.Array
    initial_gradient: jax.Array
    run_compiled: _CfsAlCompiledRun


@dataclass(frozen=True, slots=True)
class PreparedCfsAl1:
    """One compiled ten-stage augmented-Lagrangian device program."""

    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_parameters: jax.Array
    initial_value: jax.Array
    initial_gradient: jax.Array
    _run_compiled: _CfsAlCompiledRun

    def run(self) -> CfsAl1Result:
        """Dispatch the complete frozen CFS-AL1 state machine once."""

        return self._run_compiled(
            self.initial_optimizer_coordinates,
            self.initial_parameters,
        )


CfsAl2Result = CfsAl1Result


@dataclass(frozen=True, slots=True)
class PreparedCfsAl2:
    """One compiled ten-stage high-accuracy augmented-Lagrangian program."""

    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_parameters: jax.Array
    initial_value: jax.Array
    initial_gradient: jax.Array
    _run_compiled: _CfsAlCompiledRun

    def run(self) -> CfsAl2Result:
        """Dispatch the complete frozen CFS-AL2 state machine once."""

        return self._run_compiled(
            self.initial_optimizer_coordinates,
            self.initial_parameters,
        )


def _build_cfs_al_program(
    problem: FullSpaceProblem,
    scaling: FullSpaceScaling,
    optimizer: PreparedParametricFusedLBFGS,
    policy: RoutePolicy,
) -> Callable[[jax.Array, jax.Array], CfsAl1Result]:
    dtype = scaling.variable_scale.dtype
    zero_count = jnp.asarray(0, dtype=jnp.int32)
    false = jnp.asarray(False)
    true = jnp.asarray(True)
    nan = jnp.asarray(jnp.nan, dtype=dtype)
    skipped_status = jnp.asarray(-1, dtype=jnp.int32)

    def run_stage(
        carry: _CfsAl1Carry,
    ) -> tuple[_CfsAl1Carry, _CfsAl1StageRecord]:
        scaled_multipliers, penalty = unpack_cfs_al1_parameters(carry.parameters)
        del scaled_multipliers
        result = optimizer.run_staged(
            carry.optimizer_coordinates,
            carry.parameters,
        )
        endpoint = result.state.parameters
        _, diagnostics = _cfs_al1_value_and_diagnostics(
            endpoint,
            carry.parameters,
            problem,
            scaling,
        )
        status_ok = (result.status == 0) | (result.status == 1)
        stage_fatal = (
            (~status_ok)
            | (result.evaluated_nonfinite_count != 0)
            | (~result.all_accepted_states_finite)
            | (~diagnostics.all_finite)
            | (
                result.function_evaluations
                > policy.maximum_function_evaluations_per_inner_solve
            )
        )
        updated_parameters = _update_cfs_al_parameters(
            carry.parameters,
            diagnostics.scaled_constraints,
            policy,
        )
        next_optimizer_coordinates = jnp.where(
            stage_fatal,
            carry.optimizer_coordinates,
            endpoint,
        )
        next_parameters = jnp.where(
            stage_fatal,
            carry.parameters,
            updated_parameters,
        )
        stage_completed = ~stage_fatal
        next_carry = _CfsAl1Carry(
            optimizer_coordinates=next_optimizer_coordinates,
            parameters=next_parameters,
            completed_stages=(
                carry.completed_stages + stage_completed.astype(jnp.int32)
            ),
            total_inner_iterations=(carry.total_inner_iterations + result.iterations),
            total_function_evaluations=(
                carry.total_function_evaluations + result.function_evaluations
            ),
            total_gradient_evaluations=(
                carry.total_gradient_evaluations + result.gradient_evaluations
            ),
            nonfinite_evaluation_count=(
                carry.nonfinite_evaluation_count + result.evaluated_nonfinite_count
            ),
            all_accepted_states_finite=(
                carry.all_accepted_states_finite & result.all_accepted_states_finite
            ),
            fatal=carry.fatal | stage_fatal,
        )
        record = _CfsAl1StageRecord(
            penalty=penalty,
            physical_objective=diagnostics.physical_objective,
            scaled_feasibility_infinity_norm=(
                diagnostics.scaled_constraint_infinity_norm
            ),
            augmented_stationarity_infinity_norm=jnp.max(
                jnp.abs(result.state.gradient)
            ),
            optimizer_status=result.status,
            inner_iterations=result.iterations,
            function_evaluations=result.function_evaluations,
            stage_completed=stage_completed,
        )
        return next_carry, record

    def skip_stage(
        carry: _CfsAl1Carry,
    ) -> tuple[_CfsAl1Carry, _CfsAl1StageRecord]:
        return carry, _CfsAl1StageRecord(
            penalty=nan,
            physical_objective=nan,
            scaled_feasibility_infinity_norm=nan,
            augmented_stationarity_infinity_norm=nan,
            optimizer_status=skipped_status,
            inner_iterations=zero_count,
            function_evaluations=zero_count,
            stage_completed=false,
        )

    def stage_body(
        carry: _CfsAl1Carry,
        _stage_index: jax.Array,
    ) -> tuple[_CfsAl1Carry, _CfsAl1StageRecord]:
        return jax.lax.cond(carry.fatal, skip_stage, run_stage, carry)

    def run(
        initial_optimizer_coordinates: jax.Array,
        initial_parameters: jax.Array,
    ) -> CfsAl1Result:
        initial_carry = _CfsAl1Carry(
            optimizer_coordinates=initial_optimizer_coordinates,
            parameters=initial_parameters,
            completed_stages=zero_count,
            total_inner_iterations=zero_count,
            total_function_evaluations=zero_count,
            total_gradient_evaluations=zero_count,
            nonfinite_evaluation_count=zero_count,
            all_accepted_states_finite=true,
            fatal=false,
        )
        final_carry, records = jax.lax.scan(
            stage_body,
            initial_carry,
            jnp.arange(policy.maximum_outer_stages, dtype=jnp.int32),
        )
        scaled_multipliers, next_penalty = unpack_cfs_al1_parameters(
            final_carry.parameters
        )
        _, final_diagnostics = _cfs_al1_value_and_diagnostics(
            final_carry.optimizer_coordinates,
            final_carry.parameters,
            problem,
            scaling,
        )
        raw_multipliers = cfs_al1_raw_multipliers(
            scaled_multipliers,
            scaling,
        )
        kkt = fullspace_kkt_primitives(
            final_diagnostics.physical_state,
            raw_multipliers,
            problem,
        )
        all_finite = (
            (~final_carry.fatal)
            & final_carry.all_accepted_states_finite
            & (final_carry.nonfinite_evaluation_count == 0)
            & final_diagnostics.all_finite
            & kkt.all_finite
        )
        converged = (
            all_finite
            & (
                final_diagnostics.scaled_constraint_infinity_norm
                <= policy.convergence.constraint_tolerance
            )
            & (kkt.stationarity_inf <= policy.convergence.stationarity_tolerance)
        )
        stage_history = CfsAl1StageHistory(*records)
        return CfsAl1Result(
            optimizer_coordinates=final_carry.optimizer_coordinates,
            physical_state=final_diagnostics.physical_state,
            scaled_multipliers=scaled_multipliers,
            raw_multipliers=raw_multipliers,
            next_penalty=next_penalty,
            physical_objective=final_diagnostics.physical_objective,
            raw_constraint_infinity_norm=(
                final_diagnostics.raw_constraint_infinity_norm
            ),
            scaled_constraint_infinity_norm=(
                final_diagnostics.scaled_constraint_infinity_norm
            ),
            raw_kkt_stationarity_infinity_norm=kkt.stationarity_inf,
            completed_outer_stages=final_carry.completed_stages,
            total_inner_iterations=final_carry.total_inner_iterations,
            total_function_evaluations=final_carry.total_function_evaluations,
            total_gradient_evaluations=final_carry.total_gradient_evaluations,
            nonfinite_evaluation_count=(final_carry.nonfinite_evaluation_count),
            all_accepted_states_finite=(final_carry.all_accepted_states_finite),
            converged=converged,
            fatal=final_carry.fatal,
            all_finite=all_finite,
            stage_history=stage_history,
        )

    return run


def _prepare_cfs_al_program(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
    policy: RoutePolicy,
) -> _PreparedCfsAlComponents:
    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_optimizer_coordinates = fullspace_optimizer_coordinates(
        initial_physical_state,
        scaling,
    )
    initial_parameters = pack_cfs_al1_parameters(
        jnp.full(
            scaling.constraint_inverse_scale.shape,
            policy.initial_multiplier,
            dtype=initial_optimizer_coordinates.dtype,
        ),
        jnp.asarray(policy.initial_penalty, dtype=initial_optimizer_coordinates.dtype),
    )

    def objective(
        optimizer_coordinates: jax.Array,
        parameters: jax.Array,
    ) -> jax.Array:
        return cfs_al1_value(
            optimizer_coordinates,
            parameters,
            problem,
            scaling,
        )

    optimizer = prepare_parametric_fused_lbfgs(
        objective,
        initial_optimizer_coordinates,
        initial_parameters,
        maximum_iterations=policy.inner_iterations_per_stage,
        maximum_function_evaluations=(
            policy.maximum_function_evaluations_per_inner_solve
        ),
        options=FusedLBFGSOptions(
            history_size=policy.lbfgs_memory,
            function_tolerance=policy.function_tolerance,
            gradient_tolerance=policy.convergence.stationarity_tolerance,
            maximum_line_search_steps=policy.maximum_line_search_steps,
        ),
        x_dtype=initial_optimizer_coordinates.dtype,
    )
    staged = jax.jit(_build_cfs_al_program(problem, scaling, optimizer, policy))
    compiled = staged.lower(
        initial_optimizer_coordinates,
        initial_parameters,
    ).compile()
    return _PreparedCfsAlComponents(
        scaling=scaling,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        initial_parameters=initial_parameters,
        initial_value=optimizer.initial_value,
        initial_gradient=optimizer.initial_gradient,
        run_compiled=compiled,
    )


def prepare_cfs_al1(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
) -> PreparedCfsAl1:
    """Compile the complete frozen CFS-AL1 route as one device program."""

    components = _prepare_cfs_al_program(
        problem,
        bootstrap_state,
        initial_physical_state,
        route_policy(FullSpaceRoute.CFS_AL1),
    )
    return PreparedCfsAl1(
        scaling=components.scaling,
        initial_optimizer_coordinates=components.initial_optimizer_coordinates,
        initial_parameters=components.initial_parameters,
        initial_value=components.initial_value,
        initial_gradient=components.initial_gradient,
        _run_compiled=components.run_compiled,
    )


def prepare_cfs_al2(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
) -> PreparedCfsAl2:
    """Compile the complete frozen CFS-AL2 route as one device program."""

    components = _prepare_cfs_al_program(
        problem,
        bootstrap_state,
        initial_physical_state,
        route_policy(FullSpaceRoute.CFS_AL2),
    )
    return PreparedCfsAl2(
        scaling=components.scaling,
        initial_optimizer_coordinates=components.initial_optimizer_coordinates,
        initial_parameters=components.initial_parameters,
        initial_value=components.initial_value,
        initial_gradient=components.initial_gradient,
        _run_compiled=components.run_compiled,
    )


@dataclass(frozen=True, slots=True)
class CfsP0Result:
    """Device-array result and progress evidence for diagnostic route CFS-P0."""

    optimizer: FusedLBFGSResult
    initial_diagnostics: CfsP0Diagnostics
    final_diagnostics: CfsP0Diagnostics
    initial_stationarity_infinity_norm: jax.Array
    final_stationarity_infinity_norm: jax.Array
    nonfinite_evaluation_count: jax.Array
    made_progress: jax.Array
    all_finite: jax.Array


jax.tree_util.register_dataclass(
    CfsP0Result,
    data_fields=[
        "optimizer",
        "initial_diagnostics",
        "final_diagnostics",
        "initial_stationarity_infinity_norm",
        "final_stationarity_infinity_norm",
        "nonfinite_evaluation_count",
        "made_progress",
        "all_finite",
    ],
    meta_fields=[],
)


@dataclass(frozen=True, slots=True)
class PreparedCfsP0:
    """Prepared callback-free CFS-P0 solve with a frozen changed state."""

    scaling: FullSpaceScaling
    initial_optimizer_coordinates: jax.Array
    initial_merit: jax.Array
    initial_gradient: jax.Array
    initial_diagnostics: CfsP0Diagnostics
    _problem: FullSpaceProblem
    _optimizer: PreparedFusedLBFGS

    def run(self, *, maximum_iterations: int) -> CfsP0Result:
        """Execute one fixed-shape device-resident diagnostic solve."""

        initial_diagnostics = self.initial_diagnostics
        optimizer_result = self._optimizer.run(
            self.initial_optimizer_coordinates,
            maxiter=maximum_iterations,
            maxfun=route_policy(
                FullSpaceRoute.CFS_P0
            ).maximum_function_evaluations_per_inner_solve,
        )
        final_diagnostics = _cfs_p0_diagnostics_kernel(
            optimizer_result.state.parameters,
            self._problem,
            self.scaling,
        )
        initial_stationarity = jnp.max(jnp.abs(self.initial_gradient))
        final_stationarity = jnp.max(jnp.abs(optimizer_result.state.gradient))
        nonfinite_evaluation_count = optimizer_result.evaluated_nonfinite_count
        made_progress = (
            final_diagnostics.scaled_penalty_value
            < initial_diagnostics.scaled_penalty_value
        ) & (
            final_diagnostics.scaled_constraint_infinity_norm
            < initial_diagnostics.scaled_constraint_infinity_norm
        )
        all_finite = (
            initial_diagnostics.all_finite
            & final_diagnostics.all_finite
            & jnp.all(jnp.isfinite(self.initial_gradient))
            & jnp.all(jnp.isfinite(optimizer_result.state.parameters))
            & jnp.isfinite(optimizer_result.state.objective_value)
            & jnp.all(jnp.isfinite(optimizer_result.state.gradient))
            & optimizer_result.all_accepted_states_finite
        )
        return CfsP0Result(
            optimizer=optimizer_result,
            initial_diagnostics=initial_diagnostics,
            final_diagnostics=final_diagnostics,
            initial_stationarity_infinity_norm=initial_stationarity,
            final_stationarity_infinity_norm=final_stationarity,
            nonfinite_evaluation_count=nonfinite_evaluation_count,
            made_progress=made_progress,
            all_finite=all_finite,
        )


def prepare_cfs_p0(
    problem: FullSpaceProblem,
    bootstrap_state: jax.Array,
    initial_physical_state: jax.Array,
) -> PreparedCfsP0:
    """Prepare the frozen centered CFS-P0 program for one changed state."""

    scaling = fullspace_scaling_from_bootstrap(bootstrap_state, problem)
    initial_optimizer_coordinates = fullspace_optimizer_coordinates(
        initial_physical_state,
        scaling,
    )
    policy = route_policy(FullSpaceRoute.CFS_P0)

    def objective(optimizer_coordinates: jax.Array) -> jax.Array:
        return cfs_p0_penalty_value(optimizer_coordinates, problem, scaling)

    optimizer = prepare_fused_lbfgs(
        objective,
        initial_optimizer_coordinates,
        options=FusedLBFGSOptions(
            history_size=policy.lbfgs_memory,
            function_tolerance=policy.function_tolerance,
            gradient_tolerance=policy.convergence.stationarity_tolerance,
            maximum_line_search_steps=policy.maximum_line_search_steps,
        ),
        x_dtype=initial_optimizer_coordinates.dtype,
    )
    initial_diagnostics = _cfs_p0_diagnostics_kernel(
        initial_optimizer_coordinates,
        problem,
        scaling,
    )
    return PreparedCfsP0(
        scaling=scaling,
        initial_optimizer_coordinates=initial_optimizer_coordinates,
        initial_merit=optimizer.initial_value,
        initial_gradient=optimizer.initial_gradient,
        initial_diagnostics=initial_diagnostics,
        _problem=problem,
        _optimizer=optimizer,
    )


def frozen_route_contract_payload_v1() -> dict[str, object]:
    """Return the byte-frozen legacy-v1 route matrix."""

    return {
        "schema_version": SCHEMA_VERSION,
        "routes": [asdict(policy) for policy in ROUTE_POLICIES],
    }


def frozen_route_contract_payload() -> dict[str, object]:
    """Return the legacy-v1 route contract for backward compatibility."""

    return frozen_route_contract_payload_v1()


def frozen_route_contract_payload_v2() -> dict[str, object]:
    """Return the additive route-v2 contract with typed SQP controls."""

    return {
        "legacy_v1": frozen_route_contract_payload_v1(),
        "schema_version": ROUTE_SCHEMA_VERSION_V2,
        "sqp_routes": [asdict(CFS_SQP1_POLICY)],
    }


def frozen_route_contract_payload_v3() -> dict[str, object]:
    """Return the additive route-v3 contract without mutating v1/v2 bytes."""

    return {
        "legacy_v2": frozen_route_contract_payload_v2(),
        "schema_version": ROUTE_SCHEMA_VERSION_V3,
        "ftr_routes": [asdict(CFS_FTR1_POLICY)],
    }
