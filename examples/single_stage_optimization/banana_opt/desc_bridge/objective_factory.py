"""Objective-stack planning for DESC joint banana runs.

This module is intentionally a preflight contract boundary. It records the DESC
objective classes that must be assembled at runtime and enforces that
``QuadraticFlux`` is fixed-equilibrium only.
"""

from __future__ import annotations

import inspect
import json
import os
import resource
import time
from collections.abc import Mapping, Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from banana_opt.desc_bridge.runtime_imports import activate_desc_source_root
from banana_opt.desc_joint_result_schema import (
    DescJointRunMode,
    validate_desc_joint_mode,
)
from banana_opt.hardware_contracts import (
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M,
)
from banana_opt.hardware_keepout import load_hardware_sdf

ObjectiveRole = Literal["physics", "hardware", "regularization"]
DescObjectiveDerivMode = Literal["auto", "batched", "blocked"]
DescObjectiveRuntimeStatus = Literal["passed", "failed"]
DescJointConstraintPolicy = Literal[
    "hard-hardware-and-force-balance",
    "hard-linking-current-and-force-balance",
    "hard-volume-and-force-balance",
    "proximal-force-balance",
]
DescBoundaryFidelityPolicy = Literal["off", "fix-high-modes"]
DescObjectiveAblationPolicy = Literal[
    "full",
    "physics-only",
    "no-coil-curvature",
    "no-plasma-coil-distance",
    "no-coil-set-distance",
    "no-coil-geometry",
    "no-linking-current",
]

FIXED_EQUILIBRIUM_POLISH = "fixed_equilibrium_polish"
VACUUM_JOINT = "vacuum_joint"
FINITE_BETA_JOINT = "finite_beta_joint"
QUADRATIC_FLUX_OBJECTIVE = "QuadraticFlux"
COIL_SET_MIN_DISTANCE_OBJECTIVE = "CoilSetMinDistance"
HARDWARE_SDF_KEEPOUT_OBJECTIVE = "HardwareSdfKeepout"
VOLUME_OBJECTIVE = "Volume"
FORCE_BALANCE_CONSTRAINT = "ForceBalance"
FIX_COIL_CURRENT_CONSTRAINT = "FixCoilCurrent"
FIX_BOUNDARY_R_CONSTRAINT = "FixBoundaryR"
FIX_BOUNDARY_Z_CONSTRAINT = "FixBoundaryZ"
HARD_HARDWARE_AND_FORCE_BALANCE_POLICY = "hard-hardware-and-force-balance"
HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY = (
    "hard-linking-current-and-force-balance"
)
HARD_VOLUME_AND_FORCE_BALANCE_POLICY = "hard-volume-and-force-balance"
PROXIMAL_FORCE_BALANCE_POLICY = "proximal-force-balance"
DESC_JOINT_CONSTRAINT_POLICIES: tuple[DescJointConstraintPolicy, ...] = (
    HARD_HARDWARE_AND_FORCE_BALANCE_POLICY,
    HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY,
    HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
    PROXIMAL_FORCE_BALANCE_POLICY,
)
BOUNDARY_FIDELITY_OFF = "off"
BOUNDARY_FIDELITY_FIX_HIGH_MODES = "fix-high-modes"
DESC_BOUNDARY_FIDELITY_POLICIES: tuple[DescBoundaryFidelityPolicy, ...] = (
    BOUNDARY_FIDELITY_OFF,
    BOUNDARY_FIDELITY_FIX_HIGH_MODES,
)
DEFAULT_BOUNDARY_FIDELITY_FREE_MODE_SUM = 1
FULL_DESC_OBJECTIVE_ABLATION_POLICY = "full"
PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY = "physics-only"
NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY = "no-coil-curvature"
NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY = "no-plasma-coil-distance"
NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY = "no-coil-set-distance"
NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY = "no-coil-geometry"
NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY = "no-linking-current"
DESC_OBJECTIVE_ABLATION_POLICIES: tuple[DescObjectiveAblationPolicy, ...] = (
    FULL_DESC_OBJECTIVE_ABLATION_POLICY,
    PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY,
    NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY,
    NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY,
    NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY,
    NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY,
    NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY,
)
COIL_GEOMETRY_WEIGHTED_OBJECTIVES = frozenset(
    {
        COIL_SET_MIN_DISTANCE_OBJECTIVE,
        HARDWARE_SDF_KEEPOUT_OBJECTIVE,
        "PlasmaCoilSetMinDistance",
        "CoilLength",
        "CoilCurvature",
    },
)
DESC_OBJECTIVE_ABLATION_OMITTED_OBJECTIVES: dict[
    DescObjectiveAblationPolicy,
    frozenset[str],
] = {
    FULL_DESC_OBJECTIVE_ABLATION_POLICY: frozenset(),
    PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY: frozenset(
        {
            "LinkingCurrentConsistency",
            COIL_SET_MIN_DISTANCE_OBJECTIVE,
            HARDWARE_SDF_KEEPOUT_OBJECTIVE,
            "PlasmaCoilSetMinDistance",
            "CoilLength",
            "CoilCurvature",
        }
    ),
    NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY: frozenset({"CoilCurvature"}),
    NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY: frozenset(
        {"PlasmaCoilSetMinDistance"}
    ),
    NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY: frozenset(
        {COIL_SET_MIN_DISTANCE_OBJECTIVE}
    ),
    NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY: COIL_GEOMETRY_WEIGHTED_OBJECTIVES,
    NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY: frozenset(
        {"LinkingCurrentConsistency"}
    ),
}
DEFAULT_DESC_OBJECTIVE_WEIGHTS: dict[str, float] = {
    QUADRATIC_FLUX_OBJECTIVE: 1.0,
    "VacuumBoundaryError": 1.0,
    "BoundaryError": 1.0,
    "LinkingCurrentConsistency": 1.0,
    FIX_COIL_CURRENT_CONSTRAINT: 1.0,
    FIX_BOUNDARY_R_CONSTRAINT: 1.0,
    FIX_BOUNDARY_Z_CONSTRAINT: 1.0,
    VOLUME_OBJECTIVE: 1.0,
    FORCE_BALANCE_CONSTRAINT: 1.0,
    COIL_SET_MIN_DISTANCE_OBJECTIVE: 1.0,
    HARDWARE_SDF_KEEPOUT_OBJECTIVE: 1.0,
    "PlasmaCoilSetMinDistance": 1.0,
    "CoilLength": 1.0,
    "CoilCurvature": 1.0,
}
FINITE_NONBINDING_HARDWARE_UPPER_BOUND = 1.0e30
LINKING_CURRENT_GRID_N_CAP = 20
DEFAULT_DESC_OBJECTIVE_USE_JIT = False
DEFAULT_DESC_OBJECTIVE_DERIV_MODE: DescObjectiveDerivMode = "blocked"
DESC_OBJECTIVE_DERIV_MODES: tuple[DescObjectiveDerivMode, ...] = (
    "auto",
    "batched",
    "blocked",
)


@dataclass(frozen=True, slots=True)
class DescObjectiveStackEntry:
    name: str
    role: ObjectiveRole
    eq_fixed: bool | None
    source: str
    runtime_constraint: bool = False

    def to_json_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role,
            "eq_fixed": self.eq_fixed,
            "source": self.source,
            "runtime_constraint": self.runtime_constraint,
        }


@dataclass(frozen=True, slots=True)
class DescObjectiveRuntimeAssemblyReport:
    mode: DescJointRunMode
    status: DescObjectiveRuntimeStatus
    reason: str
    objective_names: tuple[str, ...]
    constraint_names: tuple[str, ...]
    desc_source_root: Path | None
    desc_version: str | None
    objective_function_type: str | None
    grid_n: int
    linking_current_grid_n: int
    bs_chunk_size: int
    dist_chunk_size: int
    jac_chunk_size: int
    objective_use_jit: bool
    objective_deriv_mode: DescObjectiveDerivMode
    joint_constraint_policy: DescJointConstraintPolicy
    objective_ablation_policy: DescObjectiveAblationPolicy
    hardware_thresholds_m: dict[str, float]
    hardware_keepout: dict[str, object] | None
    volume_target_m3: float | None
    boundary_fidelity: dict[str, object] | None
    linking_current_normalization: dict[str, object] | None
    coil_geometry_weighting: dict[str, object] | None
    weights: dict[str, float]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_objective_runtime_assembly_report_v2",
            "mode": self.mode,
            "status": self.status,
            "reason": self.reason,
            "objective_names": list(self.objective_names),
            "constraint_names": list(self.constraint_names),
            "desc_source_root": (
                None
                if self.desc_source_root is None
                else os.fspath(self.desc_source_root)
            ),
            "desc_version": self.desc_version,
            "objective_function_type": self.objective_function_type,
            "grid_n": self.grid_n,
            "linking_current_grid_n": self.linking_current_grid_n,
            "bs_chunk_size": self.bs_chunk_size,
            "dist_chunk_size": self.dist_chunk_size,
            "jac_chunk_size": self.jac_chunk_size,
            "objective_use_jit": self.objective_use_jit,
            "objective_deriv_mode": self.objective_deriv_mode,
            "joint_constraint_policy": self.joint_constraint_policy,
            "objective_ablation_policy": self.objective_ablation_policy,
            "hardware_thresholds_m": dict(self.hardware_thresholds_m),
            "hardware_keepout": (
                None if self.hardware_keepout is None else dict(self.hardware_keepout)
            ),
            "volume_target_m3": self.volume_target_m3,
            "boundary_fidelity": (
                None
                if self.boundary_fidelity is None
                else dict(self.boundary_fidelity)
            ),
            "linking_current_normalization": (
                None
                if self.linking_current_normalization is None
                else dict(self.linking_current_normalization)
            ),
            "coil_geometry_weighting": (
                None
                if self.coil_geometry_weighting is None
                else dict(self.coil_geometry_weighting)
            ),
            "weights": dict(self.weights),
        }


@dataclass(frozen=True, slots=True)
class DescObjectiveRuntimeAssembly:
    objective_function: object
    constraints: tuple[object, ...]
    report: DescObjectiveRuntimeAssemblyReport


class DescObjectiveRuntimeAssemblyError(RuntimeError):
    def __init__(self, report: DescObjectiveRuntimeAssemblyReport) -> None:
        super().__init__(report.reason)
        self.report = report


@dataclass(frozen=True, slots=True)
class DescObjectiveRuntimeEvaluationReport:
    status: DescObjectiveRuntimeStatus
    reason: str
    objective_function_type: str
    use_jit: bool
    evaluation_mode: str
    dim_x: int
    dim_f: int
    scaled_error_l2: float | None
    jacobian_shape: tuple[int, int] | None
    scaled_error_all_finite: bool
    jacobian_all_finite: bool | None
    gradient_all_finite: bool | None
    build_seconds: float | None
    value_seconds: float | None
    jacobian_seconds: float | None
    gradient_seconds: float | None
    gradient_progress_path: Path | None
    objective_term_reports: tuple[dict[str, object], ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": "desc_objective_runtime_evaluation_report_v3",
            "status": self.status,
            "reason": self.reason,
            "objective_function_type": self.objective_function_type,
            "use_jit": self.use_jit,
            "evaluation_mode": self.evaluation_mode,
            "dim_x": self.dim_x,
            "dim_f": self.dim_f,
            "scaled_error_l2": self.scaled_error_l2,
            "jacobian_shape": (
                None if self.jacobian_shape is None else list(self.jacobian_shape)
            ),
            "scaled_error_all_finite": self.scaled_error_all_finite,
            "jacobian_all_finite": self.jacobian_all_finite,
            "gradient_all_finite": self.gradient_all_finite,
            "build_seconds": self.build_seconds,
            "value_seconds": self.value_seconds,
            "jacobian_seconds": self.jacobian_seconds,
            "gradient_seconds": self.gradient_seconds,
            "gradient_progress_path": (
                None
                if self.gradient_progress_path is None
                else os.fspath(self.gradient_progress_path)
            ),
            "objective_term_reports": [
                dict(term_report) for term_report in self.objective_term_reports
            ],
        }


class DescObjectiveRuntimeEvaluationError(RuntimeError):
    def __init__(self, report: DescObjectiveRuntimeEvaluationReport) -> None:
        super().__init__(report.reason)
        self.report = report


def build_desc_objective_stack_plan(
    mode: DescJointRunMode,
    *,
    include_hardware_keepout: bool,
    joint_constraint_policy: DescJointConstraintPolicy = (
        HARD_VOLUME_AND_FORCE_BALANCE_POLICY
    ),
    boundary_fidelity_policy: DescBoundaryFidelityPolicy = BOUNDARY_FIDELITY_OFF,
    objective_ablation_policy: DescObjectiveAblationPolicy = (
        FULL_DESC_OBJECTIVE_ABLATION_POLICY
    ),
) -> tuple[DescObjectiveStackEntry, ...]:
    validate_desc_joint_mode(mode)
    _validate_joint_constraint_policy(joint_constraint_policy)
    _validate_boundary_fidelity_policy(boundary_fidelity_policy)
    _validate_objective_ablation_policy(objective_ablation_policy)
    if mode == FIXED_EQUILIBRIUM_POLISH:
        if boundary_fidelity_policy != BOUNDARY_FIDELITY_OFF:
            raise ValueError(
                "boundary_fidelity_policy is only supported for DESC joint modes."
            )
        if objective_ablation_policy != FULL_DESC_OBJECTIVE_ABLATION_POLICY:
            raise ValueError(
                "objective_ablation_policy is only supported for DESC joint modes."
            )
        stack = (
            DescObjectiveStackEntry(
                QUADRATIC_FLUX_OBJECTIVE,
                "physics",
                True,
                "DESC fixed-equilibrium coil objective",
            ),
            DescObjectiveStackEntry(
                "LinkingCurrentConsistency",
                "physics",
                True,
                "DESC coil/plasma linking-current consistency",
            ),
            DescObjectiveStackEntry(
                FIX_COIL_CURRENT_CONSTRAINT,
                "hardware",
                None,
                "DESC hard optimized-coil current anchor",
                runtime_constraint=True,
            ),
            DescObjectiveStackEntry(
                "CoilLength",
                "regularization",
                None,
                "DESC coil length regularization",
            ),
            DescObjectiveStackEntry(
                "CoilCurvature",
                "regularization",
                None,
                "DESC coil curvature regularization",
            ),
            DescObjectiveStackEntry(
                COIL_SET_MIN_DISTANCE_OBJECTIVE,
                "hardware",
                None,
                "DESC coil/coil clearance objective",
            ),
            DescObjectiveStackEntry(
                "PlasmaCoilSetMinDistance",
                "hardware",
                True,
                "DESC plasma/coil clearance objective",
            ),
        )
    elif mode == VACUUM_JOINT:
        stack = _joint_objective_stack(
            boundary_objective="VacuumBoundaryError",
            include_hardware_keepout=include_hardware_keepout,
            joint_constraint_policy=joint_constraint_policy,
            boundary_fidelity_policy=boundary_fidelity_policy,
            objective_ablation_policy=objective_ablation_policy,
        )
    else:
        stack = _joint_objective_stack(
            boundary_objective="BoundaryError",
            include_hardware_keepout=include_hardware_keepout,
            joint_constraint_policy=joint_constraint_policy,
            boundary_fidelity_policy=boundary_fidelity_policy,
            objective_ablation_policy=objective_ablation_policy,
        )
    validate_objective_stack_for_mode(mode, stack)
    return stack


def validate_objective_stack_for_mode(
    mode: DescJointRunMode,
    stack: Sequence[DescObjectiveStackEntry],
) -> None:
    validate_desc_joint_mode(mode)
    names = [entry.name for entry in stack]
    if mode in {VACUUM_JOINT, FINITE_BETA_JOINT} and QUADRATIC_FLUX_OBJECTIVE in names:
        raise ValueError("DESC joint modes must not include QuadraticFlux.")
    if mode == VACUUM_JOINT and "VacuumBoundaryError" not in names:
        raise ValueError("vacuum_joint mode requires VacuumBoundaryError.")
    if mode == FINITE_BETA_JOINT and "BoundaryError" not in names:
        raise ValueError("finite_beta_joint mode requires BoundaryError.")
    if (
        mode in {VACUUM_JOINT, FINITE_BETA_JOINT}
        and FORCE_BALANCE_CONSTRAINT not in names
    ):
        raise ValueError("DESC joint modes require ForceBalance as a hard constraint.")
    if not names:
        raise ValueError("DESC objective stack must not be empty.")


def _joint_objective_stack(
    *,
    boundary_objective: str,
    include_hardware_keepout: bool,
    joint_constraint_policy: DescJointConstraintPolicy,
    boundary_fidelity_policy: DescBoundaryFidelityPolicy,
    objective_ablation_policy: DescObjectiveAblationPolicy,
) -> tuple[DescObjectiveStackEntry, ...]:
    hard_volume_constraint = joint_constraint_policy in {
        HARD_HARDWARE_AND_FORCE_BALANCE_POLICY,
        HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
    }
    hard_hardware_constraints = (
        joint_constraint_policy == HARD_HARDWARE_AND_FORCE_BALANCE_POLICY
    )
    hard_linking_current_constraint = (
        joint_constraint_policy == HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY
    )
    volume_eq_fixed = (
        False if hard_volume_constraint else None
    )
    volume_source = (
        "DESC moving-equilibrium hard plasma-volume anchor"
        if hard_volume_constraint
        else "DESC moving-equilibrium staged plasma-volume objective"
    )
    coil_clearance_source = (
        "DESC hard coil/coil clearance constraint"
        if hard_hardware_constraints
        else "DESC coil/coil clearance objective"
    )
    plasma_clearance_source = (
        "DESC hard moving-equilibrium plasma/coil clearance constraint"
        if hard_hardware_constraints
        else "DESC moving-equilibrium plasma/coil clearance objective"
    )
    coil_length_source = (
        "DESC hard coil length constraint"
        if hard_hardware_constraints
        else "DESC coil length regularization"
    )
    coil_curvature_source = (
        "DESC hard coil curvature constraint"
        if hard_hardware_constraints
        else "DESC coil curvature regularization"
    )
    linking_current_source = (
        "DESC hard coil/plasma linking-current consistency constraint"
        if hard_linking_current_constraint
        else "DESC coil/plasma linking-current consistency"
    )
    entries = [
        DescObjectiveStackEntry(
            boundary_objective,
            "physics",
            False,
            "DESC free-boundary joint equilibrium/coil objective",
        ),
        DescObjectiveStackEntry(
            "LinkingCurrentConsistency",
            "physics",
            False,
            linking_current_source,
            runtime_constraint=hard_linking_current_constraint,
        ),
        DescObjectiveStackEntry(
            VOLUME_OBJECTIVE,
            "physics",
            volume_eq_fixed,
            volume_source,
            runtime_constraint=hard_volume_constraint,
        ),
        DescObjectiveStackEntry(
            FORCE_BALANCE_CONSTRAINT,
            "physics",
            False,
            "DESC moving-equilibrium force-balance constraint",
            runtime_constraint=True,
        ),
        DescObjectiveStackEntry(
            COIL_SET_MIN_DISTANCE_OBJECTIVE,
            "hardware",
            None,
            coil_clearance_source,
            runtime_constraint=hard_hardware_constraints,
        ),
        DescObjectiveStackEntry(
            "PlasmaCoilSetMinDistance",
            "hardware",
            False,
            plasma_clearance_source,
            runtime_constraint=hard_hardware_constraints,
        ),
        DescObjectiveStackEntry(
            "CoilLength",
            "hardware" if hard_hardware_constraints else "regularization",
            None,
            coil_length_source,
            runtime_constraint=hard_hardware_constraints,
        ),
        DescObjectiveStackEntry(
            "CoilCurvature",
            "hardware" if hard_hardware_constraints else "regularization",
            None,
            coil_curvature_source,
            runtime_constraint=hard_hardware_constraints,
        ),
    ]
    if include_hardware_keepout:
        hardware_keepout_source = (
            "generic hard DESC signed-distance keepout constraint"
            if hard_hardware_constraints
            else "generic DESC signed-distance keepout objective"
        )
        entries.append(
            DescObjectiveStackEntry(
                HARDWARE_SDF_KEEPOUT_OBJECTIVE,
                "hardware",
                None,
                hardware_keepout_source,
                runtime_constraint=hard_hardware_constraints,
            )
        )
    if boundary_fidelity_policy == BOUNDARY_FIDELITY_FIX_HIGH_MODES:
        entries.extend(
            (
                DescObjectiveStackEntry(
                    FIX_BOUNDARY_R_CONSTRAINT,
                    "physics",
                    False,
                    "DESC hard high-mode LCFS R boundary fidelity anchor",
                    runtime_constraint=True,
                ),
                DescObjectiveStackEntry(
                    FIX_BOUNDARY_Z_CONSTRAINT,
                    "physics",
                    False,
                    "DESC hard high-mode LCFS Z boundary fidelity anchor",
                    runtime_constraint=True,
                ),
            )
        )
    ablated_entries = _apply_joint_objective_ablation_policy(
        entries,
        objective_ablation_policy=objective_ablation_policy,
    )
    return (
        *ablated_entries,
        DescObjectiveStackEntry(
            FIX_COIL_CURRENT_CONSTRAINT,
            "hardware",
            None,
            "DESC hard optimized-coil current anchor",
            runtime_constraint=True,
        ),
    )


def _apply_joint_objective_ablation_policy(
    entries: Sequence[DescObjectiveStackEntry],
    *,
    objective_ablation_policy: DescObjectiveAblationPolicy,
) -> tuple[DescObjectiveStackEntry, ...]:
    _validate_objective_ablation_policy(objective_ablation_policy)
    omitted_objectives = DESC_OBJECTIVE_ABLATION_OMITTED_OBJECTIVES[
        objective_ablation_policy
    ]
    return tuple(entry for entry in entries if entry.name not in omitted_objectives)


def assemble_desc_objective_stack_runtime(
    *,
    mode: DescJointRunMode,
    equilibrium: object,
    coilset: object,
    include_hardware_keepout: bool,
    hardware_sdf_manifest_path: Path | None = None,
    hardware_glb_path: Path | None = None,
    desc_source_root: Path | None = None,
    grid_n: int = 50,
    weights: dict[str, float] | None = None,
    bs_chunk_size: int = 10,
    dist_chunk_size: int = 2,
    jac_chunk_size: int = 5,
    objective_use_jit: bool = DEFAULT_DESC_OBJECTIVE_USE_JIT,
    objective_deriv_mode: DescObjectiveDerivMode = DEFAULT_DESC_OBJECTIVE_DERIV_MODE,
    joint_constraint_policy: DescJointConstraintPolicy = (
        HARD_VOLUME_AND_FORCE_BALANCE_POLICY
    ),
    boundary_fidelity_policy: DescBoundaryFidelityPolicy = BOUNDARY_FIDELITY_OFF,
    boundary_fidelity_free_mode_sum: int = (
        DEFAULT_BOUNDARY_FIDELITY_FREE_MODE_SUM
    ),
    objective_ablation_policy: DescObjectiveAblationPolicy = (
        FULL_DESC_OBJECTIVE_ABLATION_POLICY
    ),
    volume_target_m3: float | None = None,
) -> DescObjectiveRuntimeAssembly:
    plan = build_desc_objective_stack_plan(
        mode,
        include_hardware_keepout=include_hardware_keepout,
        joint_constraint_policy=joint_constraint_policy,
        boundary_fidelity_policy=boundary_fidelity_policy,
        objective_ablation_policy=objective_ablation_policy,
    )
    objective_entries = tuple(
        entry for entry in plan if not _is_runtime_constraint_entry(entry)
    )
    constraint_entries = tuple(
        entry for entry in plan if _is_runtime_constraint_entry(entry)
    )
    objective_names = tuple(entry.name for entry in objective_entries)
    constraint_names = tuple(entry.name for entry in constraint_entries)
    planned_names = tuple(entry.name for entry in plan)
    resolved_weights = _resolve_objective_weights(weights)
    _validate_runtime_positive_int(grid_n, field_name="grid_n")
    _validate_runtime_positive_int(bs_chunk_size, field_name="bs_chunk_size")
    _validate_runtime_positive_int(dist_chunk_size, field_name="dist_chunk_size")
    _validate_runtime_positive_int(jac_chunk_size, field_name="jac_chunk_size")
    if not isinstance(objective_use_jit, bool):
        raise ValueError("objective_use_jit must be a boolean.")
    if objective_deriv_mode not in DESC_OBJECTIVE_DERIV_MODES:
        allowed = ", ".join(DESC_OBJECTIVE_DERIV_MODES)
        raise ValueError(f"objective_deriv_mode must be one of {allowed}.")
    _validate_joint_constraint_policy(joint_constraint_policy)
    _validate_boundary_fidelity_policy(boundary_fidelity_policy)
    _validate_boundary_fidelity_free_mode_sum(boundary_fidelity_free_mode_sum)
    _validate_objective_ablation_policy(objective_ablation_policy)
    hardware_keepout_config = None
    resolved_volume_target_m3 = volume_target_m3
    boundary_fidelity_config = None
    linking_current_normalization = None
    coil_geometry_weighting = None
    try:
        coil_geometry_weighting = _coil_geometry_weighting_report(
            coilset,
            planned_names=planned_names,
        )
        hardware_keepout_config = (
            _load_desc_hardware_sdf_config(
                hardware_sdf_manifest_path=hardware_sdf_manifest_path,
                hardware_glb_path=hardware_glb_path,
            )
            if HARDWARE_SDF_KEEPOUT_OBJECTIVE in planned_names
            else None
        )
        with activate_desc_source_root(desc_source_root):
            import desc
            from desc.grid import LinearGrid
            from desc.objectives import (
                BoundaryError,
                CoilCurvature,
                CoilLength,
                CoilSetMinDistance,
                CoilSetSDFDistance,
                FixBoundaryR,
                FixBoundaryZ,
                FixCoilCurrent,
                ForceBalance,
                LinkingCurrentConsistency,
                ObjectiveFunction,
                PlasmaCoilSetMinDistance,
                QuadraticFlux,
                Volume,
                VacuumBoundaryError,
            )

            if "LinkingCurrentConsistency" in planned_names:
                _require_linking_current_consistency_linking_grid(
                    LinkingCurrentConsistency
                )
            plasma_grid = LinearGrid(N=grid_n)
            coil_grid = LinearGrid(N=grid_n)
            linking_current_grid_n = min(grid_n, LINKING_CURRENT_GRID_N_CAP)
            linking_current_grid = LinearGrid(N=linking_current_grid_n)
            objective_classes = {
                "BoundaryError": BoundaryError,
                "CoilCurvature": CoilCurvature,
                "CoilLength": CoilLength,
                COIL_SET_MIN_DISTANCE_OBJECTIVE: CoilSetMinDistance,
                FIX_BOUNDARY_R_CONSTRAINT: FixBoundaryR,
                FIX_BOUNDARY_Z_CONSTRAINT: FixBoundaryZ,
                FIX_COIL_CURRENT_CONSTRAINT: FixCoilCurrent,
                FORCE_BALANCE_CONSTRAINT: ForceBalance,
                HARDWARE_SDF_KEEPOUT_OBJECTIVE: CoilSetSDFDistance,
                "LinkingCurrentConsistency": LinkingCurrentConsistency,
                "PlasmaCoilSetMinDistance": PlasmaCoilSetMinDistance,
                QUADRATIC_FLUX_OBJECTIVE: QuadraticFlux,
                VOLUME_OBJECTIVE: Volume,
                "VacuumBoundaryError": VacuumBoundaryError,
            }
            resolved_volume_target_m3 = _resolve_volume_target_m3(
                equilibrium,
                objective_names=planned_names,
                volume_target_m3=volume_target_m3,
            )
            boundary_fidelity_config = _boundary_fidelity_config(
                equilibrium,
                planned_names=planned_names,
                policy=boundary_fidelity_policy,
                free_mode_sum=boundary_fidelity_free_mode_sum,
            )
            linking_current_normalization = _linking_current_normalization_report(
                coilset,
                planned_names=planned_names,
                configured_weight=resolved_weights["LinkingCurrentConsistency"],
            )
            objective_terms = tuple(
                _instantiate_desc_objective(
                    entry,
                    equilibrium=equilibrium,
                    coilset=coilset,
                    plasma_grid=plasma_grid,
                    coil_grid=coil_grid,
                    weights=resolved_weights,
                    bs_chunk_size=bs_chunk_size,
                    dist_chunk_size=dist_chunk_size,
                    jac_chunk_size=jac_chunk_size,
                    linking_current_grid=linking_current_grid,
                    linking_current_normalization=linking_current_normalization,
                    coil_geometry_weighting=coil_geometry_weighting,
                    boundary_fidelity_config=boundary_fidelity_config,
                    hardware_keepout_config=hardware_keepout_config,
                    volume_target_m3=resolved_volume_target_m3,
                    objective_classes=objective_classes,
                )
                for entry in objective_entries
            )
            constraint_terms = tuple(
                _instantiate_desc_objective(
                    entry,
                    equilibrium=equilibrium,
                    coilset=coilset,
                    plasma_grid=plasma_grid,
                    coil_grid=coil_grid,
                    weights=resolved_weights,
                    bs_chunk_size=bs_chunk_size,
                    dist_chunk_size=dist_chunk_size,
                    jac_chunk_size=jac_chunk_size,
                    linking_current_grid=linking_current_grid,
                    linking_current_normalization=linking_current_normalization,
                    coil_geometry_weighting=coil_geometry_weighting,
                    boundary_fidelity_config=boundary_fidelity_config,
                    hardware_keepout_config=hardware_keepout_config,
                    volume_target_m3=resolved_volume_target_m3,
                    objective_classes=objective_classes,
                )
                for entry in constraint_entries
            )
            objective_function = ObjectiveFunction(
                objective_terms,
                use_jit=objective_use_jit,
                deriv_mode=objective_deriv_mode,
            )
            report = _runtime_assembly_report(
                mode=mode,
                status="passed",
                reason="DESC objective runtime stack assembled.",
                objective_names=objective_names,
                constraint_names=constraint_names,
                desc_source_root=desc_source_root,
                desc_version=_desc_version(desc),
                objective_function=objective_function,
                grid_n=grid_n,
                linking_current_grid_n=linking_current_grid_n,
                bs_chunk_size=bs_chunk_size,
                dist_chunk_size=dist_chunk_size,
                jac_chunk_size=jac_chunk_size,
                objective_use_jit=objective_use_jit,
                objective_deriv_mode=objective_deriv_mode,
                joint_constraint_policy=joint_constraint_policy,
                objective_ablation_policy=objective_ablation_policy,
                hardware_keepout=(
                    None
                    if hardware_keepout_config is None
                    else hardware_keepout_config["report"]
                ),
                volume_target_m3=resolved_volume_target_m3,
                boundary_fidelity=(
                    None
                    if boundary_fidelity_config is None
                    else boundary_fidelity_config["report"]
                ),
                linking_current_normalization=linking_current_normalization,
                coil_geometry_weighting=coil_geometry_weighting,
                weights=resolved_weights,
            )
            return DescObjectiveRuntimeAssembly(
                objective_function=objective_function,
                constraints=constraint_terms,
                report=report,
            )
    except Exception as exc:
        report = _runtime_assembly_report(
            mode=mode,
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            objective_names=objective_names,
            constraint_names=constraint_names,
            desc_source_root=desc_source_root,
            desc_version=None,
            objective_function=None,
            grid_n=grid_n,
            linking_current_grid_n=min(grid_n, LINKING_CURRENT_GRID_N_CAP),
            bs_chunk_size=bs_chunk_size,
            dist_chunk_size=dist_chunk_size,
            jac_chunk_size=jac_chunk_size,
            objective_use_jit=objective_use_jit,
            objective_deriv_mode=objective_deriv_mode,
            joint_constraint_policy=joint_constraint_policy,
            objective_ablation_policy=objective_ablation_policy,
            hardware_keepout=(
                None
                if hardware_keepout_config is None
                else hardware_keepout_config["report"]
            ),
            volume_target_m3=resolved_volume_target_m3,
            boundary_fidelity=(
                None
                if boundary_fidelity_config is None
                else boundary_fidelity_config["report"]
            ),
            linking_current_normalization=linking_current_normalization,
            coil_geometry_weighting=coil_geometry_weighting,
            weights=resolved_weights,
        )
        raise DescObjectiveRuntimeAssemblyError(report) from exc


def _is_runtime_constraint_entry(entry: DescObjectiveStackEntry) -> bool:
    if entry.runtime_constraint:
        return True
    if entry.name == FIX_COIL_CURRENT_CONSTRAINT:
        return True
    return entry.eq_fixed is False and entry.name in {
        FORCE_BALANCE_CONSTRAINT,
        FIX_BOUNDARY_R_CONSTRAINT,
        FIX_BOUNDARY_Z_CONSTRAINT,
        VOLUME_OBJECTIVE,
    }


def _validate_joint_constraint_policy(value: object) -> None:
    if value not in DESC_JOINT_CONSTRAINT_POLICIES:
        allowed = ", ".join(DESC_JOINT_CONSTRAINT_POLICIES)
        raise ValueError(f"joint_constraint_policy must be one of {allowed}.")


def _validate_boundary_fidelity_policy(value: object) -> None:
    if value not in DESC_BOUNDARY_FIDELITY_POLICIES:
        allowed = ", ".join(DESC_BOUNDARY_FIDELITY_POLICIES)
        raise ValueError(f"boundary_fidelity_policy must be one of {allowed}.")


def _validate_boundary_fidelity_free_mode_sum(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            "boundary_fidelity_free_mode_sum must be a nonnegative integer."
        )


def _boundary_fidelity_config(
    equilibrium: object,
    *,
    planned_names: tuple[str, ...],
    policy: DescBoundaryFidelityPolicy,
    free_mode_sum: int,
) -> dict[str, object] | None:
    if (
        FIX_BOUNDARY_R_CONSTRAINT not in planned_names
        and FIX_BOUNDARY_Z_CONSTRAINT not in planned_names
    ):
        return None
    if policy != BOUNDARY_FIDELITY_FIX_HIGH_MODES:
        return None
    surface = getattr(equilibrium, "surface", None)
    if surface is None:
        raise ValueError("boundary fidelity requires equilibrium.surface.")
    R_modes = _boundary_basis_modes(
        surface,
        basis_name="R_basis",
        field_name="equilibrium.surface.R_basis.modes",
    )
    Z_modes = _boundary_basis_modes(
        surface,
        basis_name="Z_basis",
        field_name="equilibrium.surface.Z_basis.modes",
    )
    fixed_R_modes = _high_boundary_modes(R_modes, free_mode_sum=free_mode_sum)
    fixed_Z_modes = _high_boundary_modes(Z_modes, free_mode_sum=free_mode_sum)
    if fixed_R_modes.size == 0 or fixed_Z_modes.size == 0:
        raise ValueError(
            "boundary fidelity fix-high-modes policy requires at least one "
            "fixed R mode and one fixed Z mode."
        )
    return {
        "fixed_R_modes": fixed_R_modes.tolist(),
        "fixed_Z_modes": fixed_Z_modes.tolist(),
        "report": {
            "policy": policy,
            "free_mode_sum": int(free_mode_sum),
            "free_selector": "|m| + |n| <= free_mode_sum",
            "fixed_selector": "|m| + |n| > free_mode_sum",
            "R_mode_count": int(R_modes.shape[0]),
            "Z_mode_count": int(Z_modes.shape[0]),
            "fixed_R_mode_count": int(fixed_R_modes.shape[0]),
            "fixed_Z_mode_count": int(fixed_Z_modes.shape[0]),
            "free_R_mode_count": int(R_modes.shape[0] - fixed_R_modes.shape[0]),
            "free_Z_mode_count": int(Z_modes.shape[0] - fixed_Z_modes.shape[0]),
            "fixed_R_modes": fixed_R_modes.tolist(),
            "fixed_Z_modes": fixed_Z_modes.tolist(),
        },
    }


def _boundary_basis_modes(
    surface: object,
    *,
    basis_name: str,
    field_name: str,
) -> np.ndarray:
    basis = getattr(surface, basis_name, None)
    modes = np.asarray(getattr(basis, "modes", None), dtype=int)
    if modes.ndim != 2 or modes.shape[1] != 3:
        raise ValueError(f"{field_name} must be an array of [l, m, n] modes.")
    return np.ascontiguousarray(modes)


def _high_boundary_modes(modes: np.ndarray, *, free_mode_sum: int) -> np.ndarray:
    mode_sum = np.abs(modes[:, 1]) + np.abs(modes[:, 2])
    return np.ascontiguousarray(modes[mode_sum > free_mode_sum])


def _validate_objective_ablation_policy(value: object) -> None:
    if value not in DESC_OBJECTIVE_ABLATION_POLICIES:
        allowed = ", ".join(DESC_OBJECTIVE_ABLATION_POLICIES)
        raise ValueError(f"objective_ablation_policy must be one of {allowed}.")


def _linking_current_normalization_report(
    coilset: object,
    *,
    planned_names: tuple[str, ...],
    configured_weight: float,
) -> dict[str, object] | None:
    if "LinkingCurrentConsistency" not in planned_names:
        return None
    optimization_scope = getattr(coilset, "desc_joint_optimization_scope", None)
    if optimization_scope is None:
        return None
    if not hasattr(coilset, "_all_currents"):
        raise ValueError(
            "scoped DESC coilset must expose _all_currents() for "
            "LinkingCurrentConsistency normalization."
        )
    all_currents = np.asarray(coilset._all_currents(), dtype=float).reshape(-1)
    full_current_abs_sum_A = _finite_positive_sum(
        np.abs(all_currents),
        field_name="full linked-current normalization",
    )
    unique_currents = np.asarray(coilset.current, dtype=float).reshape(-1)
    optimized_indices = tuple(
        int(index) for index in optimization_scope.optimized_unique_coil_indices
    )
    fixed_indices = tuple(
        int(index) for index in optimization_scope.fixed_unique_coil_indices
    )
    optimized_current_abs_sum_A = _indexed_current_abs_sum(
        unique_currents,
        optimized_indices,
        field_name="optimized linked-current normalization",
    )
    fixed_current_abs_sum_A = _indexed_current_abs_sum(
        unique_currents,
        fixed_indices,
        field_name="fixed linked-current normalization",
    )
    optimized_current_vector = unique_currents[list(optimized_indices)]
    merged_currents = np.asarray(
        coilset._all_currents(optimized_current_vector),
        dtype=float,
    ).reshape(-1)
    merged_current_abs_sum_A = _finite_positive_sum(
        np.abs(merged_currents),
        field_name="scoped linked-current merge normalization",
    )
    if not np.isclose(
        merged_current_abs_sum_A,
        full_current_abs_sum_A,
        rtol=1e-12,
        atol=1e-9,
    ):
        raise ValueError(
            "scoped DESC coilset _all_currents(currents) must reproduce the "
            "full current normalization at the current optimizer point."
        )
    return {
        "source": "scoped_full_current_abs_sum",
        "reason": (
            "LinkingCurrentConsistency residuals merge fixed and optimized "
            "coil currents, so scoped banana-only optimizer views must be "
            "normalized by the same full current vector instead of by "
            "optimized banana params only."
        ),
        "configured_weight": float(configured_weight),
        "effective_weight": float(configured_weight) / full_current_abs_sum_A,
        "normalize": False,
        "full_current_abs_sum_A": full_current_abs_sum_A,
        "full_current_abs_sum_from_scoped_current_merge_A": (
            merged_current_abs_sum_A
        ),
        "optimized_unique_current_abs_sum_A": optimized_current_abs_sum_A,
        "fixed_unique_current_abs_sum_A": fixed_current_abs_sum_A,
        "optimized_unique_coil_count": len(optimized_indices),
        "fixed_unique_coil_count": len(fixed_indices),
    }


def _coil_geometry_weighting_report(
    coilset: object,
    *,
    planned_names: tuple[str, ...],
) -> dict[str, object] | None:
    weighted_objective_names = tuple(
        name for name in planned_names if name in COIL_GEOMETRY_WEIGHTED_OBJECTIVES
    )
    if not weighted_objective_names:
        return None
    optimization_scope = getattr(coilset, "desc_joint_optimization_scope", None)
    if optimization_scope is None:
        return None
    unique_coil_count = _positive_int_value(
        getattr(optimization_scope, "unique_coil_count", None),
        field_name="scoped coil geometry unique_coil_count",
    )
    if isinstance(coilset, Sized) and len(coilset) != unique_coil_count:
        raise ValueError(
            "scoped DESC coil geometry weighting requires the CoilSet length to "
            "match the optimization-scope unique coil count."
        )
    optimized_indices = tuple(
        int(index) for index in optimization_scope.optimized_unique_coil_indices
    )
    fixed_indices = tuple(
        int(index) for index in optimization_scope.fixed_unique_coil_indices
    )
    _validate_scope_indices(
        optimized_indices,
        unique_coil_count=unique_coil_count,
        field_name="optimized coil geometry indices",
    )
    _validate_scope_indices(
        fixed_indices,
        unique_coil_count=unique_coil_count,
        field_name="fixed coil geometry indices",
    )
    unit_weights = np.zeros(unique_coil_count, dtype=float)
    unit_weights[list(optimized_indices)] = 1.0
    return {
        "source": "scoped_optimized_coil_groups",
        "reason": (
            "DESC joint coil-geometry objectives are evaluated on the full "
            "expanded coilset, but fixed coil groups have zero objective weight "
            "so unreducible fixed-coil residual floors do not dominate the "
            "banana-only optimizer."
        ),
        "weighted_objective_names": list(weighted_objective_names),
        "unit_weight_vector_by_unique_coil": unit_weights.tolist(),
        "optimized_group_names": list(optimization_scope.optimized_group_names),
        "fixed_group_names": list(optimization_scope.fixed_group_names),
        "optimized_unique_coil_indices": list(optimized_indices),
        "fixed_unique_coil_indices": list(fixed_indices),
        "unique_coil_count": unique_coil_count,
        "optimized_unique_coil_count": len(optimized_indices),
        "fixed_unique_coil_count": len(fixed_indices),
    }


def _coil_geometry_objective_weight(
    name: str,
    *,
    weights: dict[str, float],
    coil_geometry_weighting: dict[str, object] | None,
) -> float | list[float]:
    configured_weight = weights[name]
    if (
        coil_geometry_weighting is None
        or name not in coil_geometry_weighting["weighted_objective_names"]
    ):
        return configured_weight
    unit_weights = np.asarray(
        coil_geometry_weighting["unit_weight_vector_by_unique_coil"],
        dtype=float,
    )
    return (configured_weight * unit_weights).tolist()


def _validate_scope_indices(
    indices: tuple[int, ...],
    *,
    unique_coil_count: int,
    field_name: str,
) -> None:
    if not indices:
        return
    if min(indices) < 0 or max(indices) >= unique_coil_count:
        raise ValueError(
            f"{field_name} must be within the {unique_coil_count} unique coils."
        )


def _positive_int_value(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return int(value)


def _finite_positive_sum(values: np.ndarray, *, field_name: str) -> float:
    total = float(np.sum(values))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive.")
    return total


def _indexed_current_abs_sum(
    unique_currents: np.ndarray,
    indices: tuple[int, ...],
    *,
    field_name: str,
) -> float:
    if not indices:
        return 0.0
    current_count = int(unique_currents.size)
    if min(indices) < 0 or max(indices) >= current_count:
        raise ValueError(
            f"{field_name} indices must be within the {current_count} unique "
            "coil currents."
        )
    total = float(np.sum(np.abs(unique_currents[list(indices)])))
    if not np.isfinite(total):
        raise ValueError(f"{field_name} must be finite.")
    return total


def evaluate_desc_objective_stack_runtime(
    objective_function: object,
    *,
    use_jit: bool = False,
    compute_jacobian: bool = False,
    compute_gradient: bool = False,
    gradient_progress_path: Path | None = None,
) -> DescObjectiveRuntimeEvaluationReport:
    objective_function_type = (
        f"{type(objective_function).__module__}.{type(objective_function).__qualname__}"
    )
    evaluation_mode = "sequential_terms"
    try:
        build_start = time.perf_counter()
        objective_function.build(use_jit=use_jit, verbose=0)
        build_seconds = time.perf_counter() - build_start

        dim_x = getattr(objective_function, "dim_x", None)
        state_vector = None
        if dim_x is None or compute_jacobian:
            things = tuple(objective_function.things)
            state_vector = np.asarray(objective_function.x(*things), dtype=float)
            dim_x = int(state_vector.size)

        value_start = time.perf_counter()
        scaled_error, objective_term_reports = _compute_scaled_error_by_term(
            objective_function
        )
        value_seconds = time.perf_counter() - value_start
        jacobian = None
        jacobian_seconds = None
        gradient_all_finite = None
        gradient_seconds = None
        if compute_jacobian:
            if state_vector is None:
                things = tuple(objective_function.things)
                state_vector = np.asarray(objective_function.x(*things), dtype=float)
            jacobian_start = time.perf_counter()
            jacobian = np.asarray(
                objective_function.jac_scaled_error(state_vector),
                dtype=float,
            )
            jacobian_seconds = time.perf_counter() - jacobian_start
        if compute_gradient:
            gradient_start = time.perf_counter()
            objective_term_reports = _compute_scalar_gradient_by_term(
                objective_function,
                objective_term_reports,
                progress_path=gradient_progress_path,
            )
            gradient_seconds = time.perf_counter() - gradient_start
            gradient_all_finite = all(
                bool(term_report["gradient_all_finite"])
                for term_report in objective_term_reports
            )
        scaled_error_all_finite = bool(np.all(np.isfinite(scaled_error)))
        jacobian_all_finite = (
            None if jacobian is None else bool(np.all(np.isfinite(jacobian)))
        )
        scaled_error_l2 = float(np.linalg.norm(scaled_error))
        if not np.isfinite(scaled_error_l2):
            scaled_error_l2 = None
        report = DescObjectiveRuntimeEvaluationReport(
            status="passed",
            reason="DESC objective value smoke evaluation passed.",
            objective_function_type=objective_function_type,
            use_jit=use_jit,
            evaluation_mode=evaluation_mode,
            dim_x=int(dim_x),
            dim_f=int(scaled_error.size),
            scaled_error_l2=scaled_error_l2,
            jacobian_shape=(
                None
                if jacobian is None
                else (int(jacobian.shape[0]), int(jacobian.shape[1]))
            ),
            scaled_error_all_finite=scaled_error_all_finite,
            jacobian_all_finite=jacobian_all_finite,
            gradient_all_finite=gradient_all_finite,
            build_seconds=build_seconds,
            value_seconds=value_seconds,
            jacobian_seconds=jacobian_seconds,
            gradient_seconds=gradient_seconds,
            gradient_progress_path=gradient_progress_path if compute_gradient else None,
            objective_term_reports=objective_term_reports,
        )
        if (
            not scaled_error_all_finite
            or jacobian_all_finite is False
            or gradient_all_finite is False
        ):
            failed_report = DescObjectiveRuntimeEvaluationReport(
                status="failed",
                reason=(
                    "DESC objective value/Jacobian/gradient smoke produced "
                    "non-finite values."
                ),
                objective_function_type=report.objective_function_type,
                use_jit=report.use_jit,
                evaluation_mode=report.evaluation_mode,
                dim_x=report.dim_x,
                dim_f=report.dim_f,
                scaled_error_l2=report.scaled_error_l2,
                jacobian_shape=report.jacobian_shape,
                scaled_error_all_finite=report.scaled_error_all_finite,
                jacobian_all_finite=report.jacobian_all_finite,
                gradient_all_finite=report.gradient_all_finite,
                build_seconds=report.build_seconds,
                value_seconds=report.value_seconds,
                jacobian_seconds=report.jacobian_seconds,
                gradient_seconds=report.gradient_seconds,
                gradient_progress_path=report.gradient_progress_path,
                objective_term_reports=report.objective_term_reports,
            )
            raise DescObjectiveRuntimeEvaluationError(failed_report)
        return report
    except DescObjectiveRuntimeEvaluationError:
        raise
    except Exception as exc:
        report = DescObjectiveRuntimeEvaluationReport(
            status="failed",
            reason=f"{type(exc).__name__}: {exc}",
            objective_function_type=objective_function_type,
            use_jit=use_jit,
            evaluation_mode=evaluation_mode,
            dim_x=0,
            dim_f=0,
            scaled_error_l2=None,
            jacobian_shape=None,
            scaled_error_all_finite=False,
            jacobian_all_finite=None,
            gradient_all_finite=None,
            build_seconds=None,
            value_seconds=None,
            jacobian_seconds=None,
            gradient_seconds=None,
            gradient_progress_path=gradient_progress_path if compute_gradient else None,
            objective_term_reports=(),
        )
        raise DescObjectiveRuntimeEvaluationError(report) from exc


def _compute_scaled_error_by_term(
    objective_function: object,
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    scaled_error_blocks: list[np.ndarray] = []
    term_reports: list[dict[str, object]] = []
    for objective in objective_function.objectives:
        objective_start = time.perf_counter()
        args = objective.xs(*objective.things)
        scaled_error = np.asarray(objective.compute_scaled_error(*args), dtype=float)
        objective_seconds = time.perf_counter() - objective_start
        scaled_error_l2 = float(np.linalg.norm(scaled_error))
        if not np.isfinite(scaled_error_l2):
            scaled_error_l2 = None
        scaled_error_blocks.append(scaled_error)
        term_reports.append(
            {
                "name": type(objective).__name__,
                "dim_f": int(scaled_error.size),
                "scaled_error_l2": scaled_error_l2,
                "scaled_error_all_finite": bool(np.all(np.isfinite(scaled_error))),
                "value_seconds": objective_seconds,
            }
        )
    return np.concatenate(scaled_error_blocks), tuple(term_reports)


def _compute_scalar_gradient_by_term(
    objective_function: object,
    term_reports: tuple[dict[str, object], ...],
    *,
    progress_path: Path | None,
) -> tuple[dict[str, object], ...]:
    """Evaluate scalar objective gradients one DESC term at a time."""
    from desc.derivatives import Derivative

    updated_term_reports: list[dict[str, object]] = []
    for term_index, (objective, term_report) in enumerate(
        zip(
            objective_function.objectives,
            term_reports,
            strict=True,
        )
    ):
        args = objective.xs(*objective.things)
        if progress_path is not None:
            _append_jsonl(
                progress_path,
                {
                    "event": "start",
                    "term_index": term_index,
                    "name": type(objective).__name__,
                    "process_peak_rss_ru_maxrss": _process_peak_rss_ru_maxrss(),
                },
            )

        def scalar_objective(*objective_args: object) -> object:
            return objective.compute_quadratic_scalar(*objective_args)

        gradient_start = time.perf_counter()
        gradient_blocks = Derivative(
            scalar_objective,
            tuple(range(len(args))),
            mode="grad",
        )(*args)
        gradient_seconds = time.perf_counter() - gradient_start
        gradient_arrays, gradient_block_shapes = _flatten_gradient_blocks(
            _normalize_gradient_blocks(
                gradient_blocks,
                expected_count=len(args),
            )
        )
        gradient_vector = (
            np.concatenate(gradient_arrays)
            if gradient_arrays
            else np.asarray([], dtype=float)
        )
        gradient_l2 = float(np.linalg.norm(gradient_vector))
        if not np.isfinite(gradient_l2):
            gradient_l2 = None

        updated_report = dict(term_report)
        updated_report.update(
            {
                "gradient_l2": gradient_l2,
                "gradient_all_finite": bool(np.all(np.isfinite(gradient_vector))),
                "gradient_seconds": gradient_seconds,
                "gradient_size": int(gradient_vector.size),
                "gradient_block_shapes": gradient_block_shapes,
            }
        )
        if progress_path is not None:
            _append_jsonl(
                progress_path,
                {
                    "event": "finish",
                    "term_index": term_index,
                    "name": type(objective).__name__,
                    "gradient_all_finite": updated_report["gradient_all_finite"],
                    "gradient_l2": gradient_l2,
                    "gradient_seconds": gradient_seconds,
                    "gradient_size": int(gradient_vector.size),
                    "process_peak_rss_ru_maxrss": _process_peak_rss_ru_maxrss(),
                },
            )
        updated_term_reports.append(updated_report)
    return tuple(updated_term_reports)


def _append_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True)
        stream.write("\n")


def _process_peak_rss_ru_maxrss() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _normalize_gradient_blocks(
    gradient_blocks: object,
    *,
    expected_count: int,
) -> tuple[object, ...]:
    if expected_count == 1:
        if isinstance(gradient_blocks, tuple) and len(gradient_blocks) == 1:
            return gradient_blocks
        return (gradient_blocks,)
    if not isinstance(gradient_blocks, tuple | list):
        raise TypeError(
            "DESC gradient evaluation returned a single block for a multi-argument "
            f"objective term: expected {expected_count} blocks."
        )
    if len(gradient_blocks) != expected_count:
        raise ValueError(
            "DESC gradient evaluation returned the wrong number of blocks: "
            f"expected {expected_count}, got {len(gradient_blocks)}."
        )
    return tuple(gradient_blocks)


def _flatten_gradient_blocks(
    gradient_blocks: tuple[object, ...],
) -> tuple[tuple[np.ndarray, ...], list[list[int]]]:
    gradient_arrays: list[np.ndarray] = []
    gradient_block_shapes: list[list[int]] = []
    for gradient_block in gradient_blocks:
        _append_gradient_leaves(
            gradient_block,
            gradient_arrays=gradient_arrays,
            gradient_block_shapes=gradient_block_shapes,
        )
    return tuple(gradient_arrays), gradient_block_shapes


def _append_gradient_leaves(
    gradient_block: object,
    *,
    gradient_arrays: list[np.ndarray],
    gradient_block_shapes: list[list[int]],
) -> None:
    if isinstance(gradient_block, dict):
        for key in sorted(gradient_block):
            _append_gradient_leaves(
                gradient_block[key],
                gradient_arrays=gradient_arrays,
                gradient_block_shapes=gradient_block_shapes,
            )
        return
    if isinstance(gradient_block, tuple | list):
        for item in gradient_block:
            _append_gradient_leaves(
                item,
                gradient_arrays=gradient_arrays,
                gradient_block_shapes=gradient_block_shapes,
            )
        return
    gradient_array = np.asarray(gradient_block, dtype=float)
    gradient_arrays.append(np.ravel(gradient_array))
    gradient_block_shapes.append([int(axis) for axis in gradient_array.shape])


def _instantiate_desc_objective(
    entry: DescObjectiveStackEntry,
    *,
    equilibrium: object,
    coilset: object,
    plasma_grid: object,
    coil_grid: object,
    weights: dict[str, float],
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    linking_current_grid: object,
    linking_current_normalization: dict[str, object] | None,
    coil_geometry_weighting: dict[str, object] | None,
    boundary_fidelity_config: dict[str, object] | None,
    hardware_keepout_config: dict[str, object] | None,
    volume_target_m3: float | None,
    objective_classes: dict[str, type],
) -> object:
    name = entry.name
    if name == QUADRATIC_FLUX_OBJECTIVE:
        return objective_classes[name](
            equilibrium,
            field=coilset,
            eval_grid=plasma_grid,
            field_grid=coil_grid,
            vacuum=True,
            weight=weights[name],
            bs_chunk_size=bs_chunk_size,
        )
    if name in {"VacuumBoundaryError", "BoundaryError"}:
        return objective_classes[name](
            eq=equilibrium,
            field=coilset,
            field_fixed=False,
            weight=weights[name],
        )
    if name == "LinkingCurrentConsistency":
        effective_weight = weights[name]
        normalize = True
        if linking_current_normalization is not None:
            effective_weight = float(
                linking_current_normalization["effective_weight"]
            )
            normalize = False
        return objective_classes[name](
            equilibrium,
            coilset,
            eq_fixed=bool(entry.eq_fixed),
            linking_grid=linking_current_grid,
            weight=effective_weight,
            normalize=normalize,
            jac_chunk_size=jac_chunk_size,
        )
    if name in {FIX_BOUNDARY_R_CONSTRAINT, FIX_BOUNDARY_Z_CONSTRAINT}:
        if boundary_fidelity_config is None:
            raise ValueError(
                f"{name} requires boundary_fidelity_config from an active "
                "boundary fidelity policy."
            )
        mode_key = (
            "fixed_R_modes"
            if name == FIX_BOUNDARY_R_CONSTRAINT
            else "fixed_Z_modes"
        )
        modes = np.asarray(boundary_fidelity_config[mode_key], dtype=int)
        if modes.size == 0:
            raise ValueError(f"{name} requires at least one boundary mode.")
        return objective_classes[name](
            equilibrium,
            modes=modes,
            weight=weights[name],
        )
    if name == FIX_COIL_CURRENT_CONSTRAINT:
        return objective_classes[name](
            coilset,
            weight=weights[name],
            indices=True,
        )
    if name == VOLUME_OBJECTIVE:
        if volume_target_m3 is None:
            raise ValueError("Volume objective requires a finite volume target.")
        return objective_classes[name](
            equilibrium,
            target=volume_target_m3,
            normalize=True,
            weight=weights[name],
            jac_chunk_size=jac_chunk_size,
        )
    if name == FORCE_BALANCE_CONSTRAINT:
        return objective_classes[name](
            equilibrium,
            target=0,
            weight=weights[name],
            jac_chunk_size=jac_chunk_size,
        )
    if name == COIL_SET_MIN_DISTANCE_OBJECTIVE:
        return objective_classes[name](
            coilset,
            bounds=(
                COIL_COIL_MIN_DIST_M,
                FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
            ),
            normalize_target=False,
            grid=coil_grid,
            weight=_coil_geometry_objective_weight(
                name,
                weights=weights,
                coil_geometry_weighting=coil_geometry_weighting,
            ),
            dist_chunk_size=dist_chunk_size,
        )
    if name == HARDWARE_SDF_KEEPOUT_OBJECTIVE:
        if hardware_keepout_config is None:
            raise ValueError(
                "HardwareSdfKeepout objective requires a hardware SDF manifest."
            )
        return objective_classes[name](
            coilset,
            sdf_gridsets=hardware_keepout_config["sdf_gridsets"],
            minimum_clearance=hardware_keepout_config["minimum_clearance_m"],
            outside_value=hardware_keepout_config["outside_value_m"],
            bounds=(
                hardware_keepout_config["minimum_clearance_m"],
                FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
            ),
            normalize_target=False,
            grid=coil_grid,
            weight=_coil_geometry_objective_weight(
                name,
                weights=weights,
                coil_geometry_weighting=coil_geometry_weighting,
            ),
            dist_chunk_size=dist_chunk_size,
        )
    if name == "PlasmaCoilSetMinDistance":
        return objective_classes[name](
            equilibrium,
            coilset,
            bounds=(
                COIL_PLASMA_MIN_DIST_M,
                FINITE_NONBINDING_HARDWARE_UPPER_BOUND,
            ),
            normalize_target=False,
            plasma_grid=plasma_grid,
            coil_grid=coil_grid,
            eq_fixed=bool(entry.eq_fixed),
            weight=_coil_geometry_objective_weight(
                name,
                weights=weights,
                coil_geometry_weighting=coil_geometry_weighting,
            ),
            dist_chunk_size=dist_chunk_size,
        )
    if name == "CoilLength":
        return objective_classes[name](
            coilset,
            bounds=(0.0, COIL_LENGTH_HARD_LIMIT_M),
            normalize_target=True,
            grid=coil_grid,
            weight=_coil_geometry_objective_weight(
                name,
                weights=weights,
                coil_geometry_weighting=coil_geometry_weighting,
            ),
        )
    if name == "CoilCurvature":
        if entry.runtime_constraint:
            return objective_classes[name](
                coilset,
                bounds=(0.0, MAX_CURVATURE_INV_M),
                normalize_target=True,
                grid=coil_grid,
                weight=_coil_geometry_objective_weight(
                    name,
                    weights=weights,
                    coil_geometry_weighting=coil_geometry_weighting,
                ),
            )
        return objective_classes[name](
            coilset,
            target=0.0,
            normalize_target=True,
            grid=coil_grid,
            weight=_coil_geometry_objective_weight(
                name,
                weights=weights,
                coil_geometry_weighting=coil_geometry_weighting,
            ),
        )
    raise ValueError(f"Unsupported DESC objective entry {name!r}.")


def _load_desc_hardware_sdf_config(
    *,
    hardware_sdf_manifest_path: Path | None,
    hardware_glb_path: Path | None,
) -> dict[str, object]:
    if hardware_sdf_manifest_path is None:
        raise ValueError("HardwareSdfKeepout objective requires hardware_sdf path.")
    sdf_data = load_hardware_sdf(
        hardware_sdf_manifest_path,
        glb_path=hardware_glb_path,
    )
    sdf_gridsets = tuple(
        (
            (group.grid, group.origin_m, group.spacing_m),
            *(
                (patch.grid, patch.origin_m, patch.spacing_m)
                for patch in group.patches
            ),
        )
        for group in sdf_data.groups
    )
    sdf_effective_margin_m = float(sdf_data.effective_margin_m)
    centerline_padding_m = float(TYPE_KK_OUTER_CHANNEL_CORNER_REACH_M)
    minimum_clearance_m = sdf_effective_margin_m + centerline_padding_m
    outside_value_m = minimum_clearance_m + 1.0
    return {
        "sdf_gridsets": sdf_gridsets,
        "minimum_clearance_m": minimum_clearance_m,
        "outside_value_m": outside_value_m,
        "report": {
            "source": "hardware_sdf_manifest",
            "manifest_path": sdf_data.manifest_path,
            "data_path": sdf_data.data_path,
            "manifest_sha256": sdf_data.manifest_sha256,
            "data_sha256": sdf_data.data_sha256,
            "group_labels": list(sdf_data.group_labels),
            "patch_count": sdf_data.patch_count,
            "minimum_clearance_m": minimum_clearance_m,
            "outside_value_m": outside_value_m,
            "sdf_effective_margin_m": sdf_effective_margin_m,
            "type_kk_centerline_padding_m": centerline_padding_m,
            "sampling_policy": (
                "DESC CoilSetSDFDistance samples coil centerlines; the banana "
                "bridge pads by Type-KK outer-channel corner reach. Final "
                "promotion remains bound to the SIMSOPT/CAD swept-solid oracle."
            ),
            "safety_margin_m": sdf_data.safety_margin_m,
            "effective_margin_m": sdf_data.effective_margin_m,
            "error_budget_m": dict(sdf_data.error_budget_m),
            "documented_gate_only_groups": sorted(
                str(key) for key in sdf_data.documented_gate_only
            ),
            "covered_by_other_in_loop_groups": sorted(
                str(key) for key in sdf_data.covered_by_other_in_loop
            ),
        },
    }


def _resolve_objective_weights(
    weights: dict[str, float] | None,
) -> dict[str, float]:
    resolved = dict(DEFAULT_DESC_OBJECTIVE_WEIGHTS)
    if weights is None:
        return resolved
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"DESC objective weight {name!r} must be numeric.")
        weight = float(value)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"DESC objective weight {name!r} must be finite and nonnegative."
            )
        resolved[name] = weight
    return resolved


def _resolve_volume_target_m3(
    equilibrium: object,
    *,
    objective_names: tuple[str, ...],
    volume_target_m3: float | None,
) -> float | None:
    if VOLUME_OBJECTIVE not in objective_names:
        return None
    if volume_target_m3 is not None:
        return _finite_nonzero_float(volume_target_m3, field_name="volume_target_m3")
    if not hasattr(equilibrium, "compute"):
        raise ValueError(
            "joint-mode Volume objective requires an equilibrium with compute('V')."
        )
    volume_payload = equilibrium.compute("V")
    if not isinstance(volume_payload, Mapping):
        raise ValueError("equilibrium.compute('V') must return a mapping.")
    if "V" not in volume_payload:
        raise ValueError("equilibrium.compute('V') must include key 'V'.")
    return _finite_nonzero_float(
        volume_payload["V"],
        field_name="equilibrium.compute('V')['V']",
    )


def _finite_nonzero_float(value: object, *, field_name: str) -> float:
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != 1:
        raise ValueError(f"{field_name} must be a scalar.")
    scalar = float(array[0])
    if not np.isfinite(scalar) or scalar == 0.0:
        raise ValueError(f"{field_name} must be a finite nonzero number.")
    return scalar


def _validate_runtime_positive_int(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")


def _runtime_assembly_report(
    *,
    mode: DescJointRunMode,
    status: DescObjectiveRuntimeStatus,
    reason: str,
    objective_names: tuple[str, ...],
    constraint_names: tuple[str, ...],
    desc_source_root: Path | None,
    desc_version: str | None,
    objective_function: object | None,
    grid_n: int,
    linking_current_grid_n: int,
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    objective_use_jit: bool,
    objective_deriv_mode: DescObjectiveDerivMode,
    joint_constraint_policy: DescJointConstraintPolicy,
    objective_ablation_policy: DescObjectiveAblationPolicy,
    hardware_keepout: dict[str, object] | None,
    volume_target_m3: float | None,
    boundary_fidelity: dict[str, object] | None,
    linking_current_normalization: dict[str, object] | None,
    coil_geometry_weighting: dict[str, object] | None,
    weights: dict[str, float],
) -> DescObjectiveRuntimeAssemblyReport:
    return DescObjectiveRuntimeAssemblyReport(
        mode=mode,
        status=status,
        reason=reason,
        objective_names=objective_names,
        constraint_names=constraint_names,
        desc_source_root=(
            None if desc_source_root is None else desc_source_root.resolve()
        ),
        desc_version=desc_version,
        objective_function_type=(
            None
            if objective_function is None
            else (
                f"{type(objective_function).__module__}."
                f"{type(objective_function).__qualname__}"
            )
        ),
        grid_n=grid_n,
        linking_current_grid_n=linking_current_grid_n,
        bs_chunk_size=bs_chunk_size,
        dist_chunk_size=dist_chunk_size,
        jac_chunk_size=jac_chunk_size,
        objective_use_jit=objective_use_jit,
        objective_deriv_mode=objective_deriv_mode,
        joint_constraint_policy=joint_constraint_policy,
        objective_ablation_policy=objective_ablation_policy,
        hardware_thresholds_m={
            "coil_coil_min_dist_m": COIL_COIL_MIN_DIST_M,
            "coil_length_hard_limit_m": COIL_LENGTH_HARD_LIMIT_M,
            "coil_plasma_min_dist_m": COIL_PLASMA_MIN_DIST_M,
            "max_curvature_inv_m": MAX_CURVATURE_INV_M,
        },
        hardware_keepout=hardware_keepout,
        volume_target_m3=volume_target_m3,
        boundary_fidelity=boundary_fidelity,
        linking_current_normalization=linking_current_normalization,
        coil_geometry_weighting=coil_geometry_weighting,
        weights=weights,
    )


def _desc_version(desc_module: object) -> str | None:
    version = getattr(desc_module, "__version__", None)
    if isinstance(version, str) and version != "":
        return version
    return None


def _require_linking_current_consistency_linking_grid(
    linking_current_consistency_class: type,
) -> None:
    signature = inspect.signature(linking_current_consistency_class)
    if "linking_grid" not in signature.parameters:
        raise RuntimeError(
            "DESC LinkingCurrentConsistency must accept the linking_grid keyword "
            "for this runner. Use the paired DESC checkout with the capped "
            "linking-grid patch instead of the unbounded upstream constructor."
        )


__all__ = [
    "COIL_SET_MIN_DISTANCE_OBJECTIVE",
    "BOUNDARY_FIDELITY_FIX_HIGH_MODES",
    "BOUNDARY_FIDELITY_OFF",
    "DEFAULT_DESC_OBJECTIVE_WEIGHTS",
    "DEFAULT_BOUNDARY_FIDELITY_FREE_MODE_SUM",
    "DESC_BOUNDARY_FIDELITY_POLICIES",
    "DESC_JOINT_CONSTRAINT_POLICIES",
    "DESC_OBJECTIVE_ABLATION_POLICIES",
    "DescBoundaryFidelityPolicy",
    "DescObjectiveAblationPolicy",
    "DescObjectiveStackEntry",
    "DescJointConstraintPolicy",
    "DescObjectiveRuntimeAssembly",
    "DescObjectiveRuntimeAssemblyError",
    "DescObjectiveRuntimeAssemblyReport",
    "DescObjectiveRuntimeEvaluationError",
    "DescObjectiveRuntimeEvaluationReport",
    "FINITE_BETA_JOINT",
    "FIXED_EQUILIBRIUM_POLISH",
    "FIX_BOUNDARY_R_CONSTRAINT",
    "FIX_BOUNDARY_Z_CONSTRAINT",
    "FIX_COIL_CURRENT_CONSTRAINT",
    "FORCE_BALANCE_CONSTRAINT",
    "FULL_DESC_OBJECTIVE_ABLATION_POLICY",
    "HARD_HARDWARE_AND_FORCE_BALANCE_POLICY",
    "HARD_LINKING_CURRENT_AND_FORCE_BALANCE_POLICY",
    "HARD_VOLUME_AND_FORCE_BALANCE_POLICY",
    "HARDWARE_SDF_KEEPOUT_OBJECTIVE",
    "LINKING_CURRENT_GRID_N_CAP",
    "NO_COIL_CURVATURE_OBJECTIVE_ABLATION_POLICY",
    "NO_COIL_GEOMETRY_OBJECTIVE_ABLATION_POLICY",
    "NO_COIL_SET_DISTANCE_OBJECTIVE_ABLATION_POLICY",
    "NO_LINKING_CURRENT_OBJECTIVE_ABLATION_POLICY",
    "NO_PLASMA_COIL_DISTANCE_OBJECTIVE_ABLATION_POLICY",
    "PHYSICS_ONLY_OBJECTIVE_ABLATION_POLICY",
    "PROXIMAL_FORCE_BALANCE_POLICY",
    "QUADRATIC_FLUX_OBJECTIVE",
    "VACUUM_JOINT",
    "VOLUME_OBJECTIVE",
    "assemble_desc_objective_stack_runtime",
    "build_desc_objective_stack_plan",
    "evaluate_desc_objective_stack_runtime",
    "validate_objective_stack_for_mode",
]
