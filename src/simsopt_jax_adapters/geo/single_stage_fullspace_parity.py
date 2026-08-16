"""Same-state parity gate for the coupled single-stage objective.

The authoritative side delegates raw objective evaluation and exact equality
construction to the existing adapter programs.  The candidate side evaluates
the pure full-space core at the identical packed state.  This module owns only
comparison/reporting policy; it does not define any physics formula.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.core.curve_geometry import curve_length_from_spec
from simsopt_jax.core.field import (
    coil_set_spec_from_dof_extraction_spec,
    coil_specs_from_dof_extraction_spec,
    grouped_biot_savart_B_from_spec,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs
from simsopt_jax.geo.boozer_residual import boozer_residual_vector
from simsopt_jax.objectives.single_stage_fullspace import (
    TERM_LEDGER,
    FullSpaceEvaluation,
    FullSpaceProblem,
    TermTolerance,
    evaluate_fullspace,
)
from simsopt_jax.runtime.host_boundary import host_value

from simsopt_jax_adapters.geo import boozer_surface, surface_objectives

_RAW_TERM_NAMES = ("non_qs", "residual", "iota", "major_radius", "length")
_ADAPTER_SURFACE_KIND = "xyztensorfourier"
_OBSERVABLE_NAMES = (
    "iota",
    "G",
    "volume",
    "major_radius",
    "total_length",
    "non_qs_ratio",
    "boozer_residual_scalar",
    "boozer_residual_rms",
)
_TERM_ID_BY_FIELD = {
    "raw_terms.non_qs": "non_qs",
    "raw_terms.residual": "boozer_residual_objective",
    "raw_terms.iota": "iota_penalty",
    "raw_terms.major_radius": "major_radius_penalty",
    "raw_terms.length": "length_penalty",
    "weighted_total": "non_qs",
    "full_boozer_residual": "boozer_equality",
    "masked_boozer_residual": "boozer_equality",
    "volume_constraint": "volume_equality",
    "observables.iota": "iota_penalty",
    "observables.G": "boozer_equality",
    "observables.volume": "volume_equality",
    "observables.major_radius": "major_radius_penalty",
    "observables.total_length": "length_penalty",
    "observables.non_qs_ratio": "non_qs",
    "observables.boozer_residual_scalar": "boozer_residual_objective",
    "observables.boozer_residual_rms": "boozer_equality",
}


@dataclass(frozen=True, slots=True)
class SameStateEvaluation:
    """Complete parity payload evaluated at one packed binary64 state."""

    raw_terms: Mapping[str, jax.Array]
    weighted_total: jax.Array
    full_boozer_residual: jax.Array
    masked_boozer_residual: jax.Array
    volume_constraint: jax.Array
    observables: Mapping[str, jax.Array]


@dataclass(frozen=True, slots=True)
class SameStateFieldComparison:
    field: str
    tolerance: TermTolerance
    max_absolute_error: float
    max_tolerance_ratio: float
    passed: bool


@dataclass(frozen=True, slots=True)
class SameStateParityReport:
    """Reusable real-bootstrap report with ledger-owned field tolerances."""

    state_little_endian_sha256: str
    comparisons: tuple[SameStateFieldComparison, ...]
    passed: bool


AuthoritativeEvaluator = Callable[[jax.Array, FullSpaceProblem], SameStateEvaluation]


def _state_sha256(z: jax.Array) -> str:
    values = np.ascontiguousarray(np.asarray(host_value(z)), dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _full_boozer_residual(z: jax.Array, problem: FullSpaceProblem) -> jax.Array:
    state = problem.layout.unpack(z)
    coil_set = coil_set_spec_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    gamma, xphi, xtheta = surface_gamma_tangents_from_dofs(
        problem.exact_surface_template,
        state.surface_dofs,
    )
    magnetic_field = grouped_biot_savart_B_from_spec(
        gamma.reshape((-1, 3)), coil_set
    ).reshape(gamma.shape)
    return boozer_residual_vector(
        state.G,
        state.iota,
        magnetic_field,
        xphi,
        xtheta,
        weight_inv_modB=problem.config.weight_inv_modB,
    )


def _snapshot_from_fullspace_evaluation(
    z: jax.Array,
    problem: FullSpaceProblem,
    evaluation: FullSpaceEvaluation,
) -> SameStateEvaluation:
    full_residual = _full_boozer_residual(z, problem)
    return SameStateEvaluation(
        raw_terms={
            name: getattr(evaluation.raw_terms, name) for name in _RAW_TERM_NAMES
        },
        weighted_total=evaluation.weighted_total,
        full_boozer_residual=full_residual,
        masked_boozer_residual=evaluation.constraints.boozer,
        volume_constraint=evaluation.constraints.volume,
        observables={
            name: getattr(evaluation.observables, name) for name in _OBSERVABLE_NAMES
        },
    )


def evaluate_fullspace_same_state(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> SameStateEvaluation:
    """Capture every parity field from the pure coupled evaluator."""

    return _snapshot_from_fullspace_evaluation(
        z,
        problem,
        evaluate_fullspace(z, problem),
    )


def _authoritative_outer_config(problem: FullSpaceProblem, vessel_gamma: jax.Array):
    config = problem.config
    return {
        "non_qs_quadpoints_phi": problem.non_qs_surface_template.quadpoints_phi,
        "non_qs_quadpoints_theta": problem.non_qs_surface_template.quadpoints_theta,
        "non_qs_axis": config.non_qs_axis,
        "optimized_coil_index": config.length_coil_indices[0],
        "length_coil_indices": config.length_coil_indices,
        "length_target": config.length_target,
        "major_radius_target": config.major_radius_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "surface_vessel_threshold": 0.0,
        "vessel_gamma": vessel_gamma,
        "non_qs_weight": config.non_qs_weight,
        "residual_weight": config.residual_weight,
        "iota_weight": config.iota_weight,
        "major_radius_weight": config.major_radius_weight,
        "length_weight": config.length_weight,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
    }


def evaluate_authoritative_same_state(
    z: jax.Array,
    problem: FullSpaceProblem,
) -> SameStateEvaluation:
    """Evaluate the existing adapter objective and exact equations at ``z``."""

    state = problem.layout.unpack(z)
    exact = problem.exact_surface_template
    label = problem.label_surface_template
    x_inner = jnp.concatenate(
        (
            state.surface_dofs,
            jnp.reshape(state.iota, (1,)),
            jnp.reshape(state.G, (1,)),
        )
    )
    coil_set = coil_set_spec_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    label_gamma, _label_xphi, _label_xtheta = surface_gamma_tangents_from_dofs(
        label,
        state.surface_dofs,
    )
    outer_config = _authoritative_outer_config(problem, label_gamma)
    raw_terms = surface_objectives._traceable_single_stage_outer_term_values(
        x_inner,
        state.coil_dofs,
        coil_set,
        quadpoints_phi=exact.quadpoints_phi,
        quadpoints_theta=exact.quadpoints_theta,
        mpol=exact.mpol,
        ntor=exact.ntor,
        nfp=exact.nfp,
        stellsym=exact.stellsym,
        scatter_indices=exact.scatter_indices,
        surface_kind=_ADAPTER_SURFACE_KIND,
        label_quadpoints_phi=label.quadpoints_phi,
        label_quadpoints_theta=label.quadpoints_theta,
        label_mpol=label.mpol,
        label_ntor=label.ntor,
        label_nfp=label.nfp,
        label_stellsym=label.stellsym,
        label_scatter_indices=label.scatter_indices,
        label_surface_kind=_ADAPTER_SURFACE_KIND,
        optimize_G=True,
        weight_inv_modB=problem.config.weight_inv_modB,
        constraint_weight=None,
        targetlabel=problem.config.volume_target,
        label_type="volume",
        phi_idx=0,
        iota_target=problem.config.iota_target,
        surface_quadpoints_phi=exact.quadpoints_phi,
        surface_quadpoints_theta=exact.quadpoints_theta,
        coil_dof_extraction_spec=problem.coil_dof_extraction,
        outer_objective_config=outer_config,
    )
    weighted_terms = (
        surface_objectives._traceable_weighted_single_stage_outer_term_values(
            raw_terms,
            outer_objective_config=outer_config,
        )
    )
    weighted_total = sum(
        (
            weighted_terms[name]
            for name, _weight in surface_objectives._TRACEABLE_SINGLE_STAGE_OUTER_TERM_SPECS
        ),
        start=jnp.asarray(0.0, dtype=jnp.float64),
    )
    exact_residual = boozer_surface._boozer_exact_residual(
        x_inner,
        coil_set_spec=coil_set,
        quadpoints_phi=exact.quadpoints_phi,
        quadpoints_theta=exact.quadpoints_theta,
        mpol=exact.mpol,
        ntor=exact.ntor,
        nfp=exact.nfp,
        stellsym=exact.stellsym,
        scatter_indices=exact.scatter_indices,
        surface_kind=_ADAPTER_SURFACE_KIND,
        clamped_dims=exact.clamped_dims,
        label_quadpoints_phi=label.quadpoints_phi,
        label_quadpoints_theta=label.quadpoints_theta,
        label_mpol=label.mpol,
        label_ntor=label.ntor,
        label_nfp=label.nfp,
        label_stellsym=label.stellsym,
        label_scatter_indices=label.scatter_indices,
        label_surface_kind=_ADAPTER_SURFACE_KIND,
        label_clamped_dims=label.clamped_dims,
        targetlabel=problem.config.volume_target,
        label_type="volume",
        phi_idx=0,
        mask_indices=problem.exact_mask_indices,
        stellsym_surface=exact.stellsym,
        weight_inv_modB=problem.config.weight_inv_modB,
    )
    full_residual = _full_boozer_residual(z, problem)
    coil_specs = coil_specs_from_dof_extraction_spec(
        problem.coil_dof_extraction,
        state.coil_dofs,
    )
    selected_lengths = tuple(
        curve_length_from_spec(coil_specs[index].curve)
        for index in problem.config.length_coil_indices
    )
    total_length = sum(selected_lengths[1:], start=selected_lengths[0])
    volume_constraint = exact_residual[-1]
    volume = volume_constraint + problem.config.volume_target
    major_radius = surface_objectives.surface_major_radius_jax_from_dofs(
        exact,
        state.surface_dofs,
    )
    return SameStateEvaluation(
        raw_terms={name: raw_terms[name] for name in _RAW_TERM_NAMES},
        weighted_total=weighted_total,
        full_boozer_residual=full_residual,
        masked_boozer_residual=exact_residual[:-1],
        volume_constraint=volume_constraint,
        observables={
            "iota": state.iota,
            "G": state.G,
            "volume": volume,
            "major_radius": major_radius,
            "total_length": total_length,
            "non_qs_ratio": raw_terms["non_qs"],
            "boozer_residual_scalar": raw_terms["residual"],
            "boozer_residual_rms": jnp.sqrt(jnp.mean(full_residual**2)),
        },
    )


def _tolerances_by_term_id() -> dict[str, TermTolerance]:
    return {row.term_id: row.tolerance for row in TERM_LEDGER}


def same_state_field_tolerances() -> dict[str, TermTolerance]:
    """Return the ledger-derived tolerance for every serialized parity field."""

    tolerances = _tolerances_by_term_id()
    return {field: tolerances[term_id] for field, term_id in _TERM_ID_BY_FIELD.items()}


def _flatten_fields(snapshot: SameStateEvaluation) -> dict[str, jax.Array]:
    fields = {f"raw_terms.{name}": value for name, value in snapshot.raw_terms.items()}
    fields["weighted_total"] = snapshot.weighted_total
    fields["full_boozer_residual"] = snapshot.full_boozer_residual
    fields["masked_boozer_residual"] = snapshot.masked_boozer_residual
    fields["volume_constraint"] = snapshot.volume_constraint
    fields.update(
        {f"observables.{name}": value for name, value in snapshot.observables.items()}
    )
    return fields


def _compare_field(
    field: str,
    reference: jax.Array,
    candidate: jax.Array,
    tolerance: TermTolerance,
) -> SameStateFieldComparison:
    left = np.asarray(host_value(reference), dtype=np.float64)
    right = np.asarray(host_value(candidate), dtype=np.float64)
    if (
        left.shape != right.shape
        or not np.all(np.isfinite(left))
        or not np.all(np.isfinite(right))
    ):
        return SameStateFieldComparison(field, tolerance, np.inf, np.inf, False)
    absolute_error = np.abs(right - left)
    allowed = tolerance.atol + tolerance.rtol * np.abs(left)
    ratio = np.divide(
        absolute_error,
        allowed,
        out=np.zeros_like(absolute_error),
        where=allowed != 0.0,
    )
    zero_allowed_failure = (allowed == 0.0) & (absolute_error != 0.0)
    max_absolute_error = float(np.max(absolute_error, initial=0.0))
    max_tolerance_ratio = (
        np.inf if np.any(zero_allowed_failure) else float(np.max(ratio, initial=0.0))
    )
    return SameStateFieldComparison(
        field=field,
        tolerance=tolerance,
        max_absolute_error=max_absolute_error,
        max_tolerance_ratio=max_tolerance_ratio,
        passed=max_tolerance_ratio <= 1.0,
    )


def build_same_state_parity_report(
    z: jax.Array,
    problem: FullSpaceProblem,
    *,
    authoritative_evaluator: AuthoritativeEvaluator = evaluate_authoritative_same_state,
) -> SameStateParityReport:
    """Compare adapter and coupled evaluators with the frozen ledger policy."""

    reference_fields = _flatten_fields(authoritative_evaluator(z, problem))
    candidate_fields = _flatten_fields(evaluate_fullspace_same_state(z, problem))
    if tuple(reference_fields) != tuple(candidate_fields):
        raise ValueError("same-state parity evaluator fields do not match")
    tolerances = _tolerances_by_term_id()
    comparisons = tuple(
        _compare_field(
            field,
            reference_fields[field],
            candidate_fields[field],
            tolerances[_TERM_ID_BY_FIELD[field]],
        )
        for field in reference_fields
    )
    return SameStateParityReport(
        state_little_endian_sha256=_state_sha256(z),
        comparisons=comparisons,
        passed=all(comparison.passed for comparison in comparisons),
    )


__all__ = [
    "AuthoritativeEvaluator",
    "SameStateEvaluation",
    "SameStateFieldComparison",
    "SameStateParityReport",
    "build_same_state_parity_report",
    "evaluate_authoritative_same_state",
    "evaluate_fullspace_same_state",
    "same_state_field_tolerances",
]
