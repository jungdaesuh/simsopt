"""One optional final nested-LS correction of a flat-675 outer point.

The correction freezes coil and vessel coordinates. Acceptance is a numerical
screen over explicitly supplied design limits, not physical validation.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from time import perf_counter
from types import MappingProxyType
from typing import Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_JAX_INNER_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_WEIGHT_INV_MODB,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_reduced_closures,
    run_reduced_nested_ls_schur_newton,
)

from .boozer_material import build_flat675_boozer_system, flat675_candidate_geometry
from .construction import Flat675Problem
from .nested_bridge import nested_view_from_flat675
from .objective import flat675_weighted_terms
from .policy import FLAT675_OBJECTIVE_TERM_KEYS
from .y_solve import solve_flat675_y_qr

_SCHUR_LINEAR_SOLVER = "dense_lu"


@dataclass(frozen=True, slots=True)
class Flat675AcceptanceLimits:
    """External numerical caps; objective increase is absolute weighted units."""

    max_boozer_weighted_rms: float
    max_absolute_label_error: float
    max_surface_displacement_m: float
    max_objective_increase: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field.name} must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class Flat675PolishResult:
    """Owned, read-only corrected coordinates and measured numerical diagnostics."""

    outer_vector: NDArray[np.float64]
    acceptance_status: Literal["accepted", "rejected", "not_assessed"]
    rejection_reasons: tuple[str, ...]
    solver_success: bool
    persisted: bool
    finite: bool
    same_branch_as_incoming: bool
    iteration_count: int
    elapsed_seconds: float
    full_gradient_l2_before: float
    full_gradient_l2_after: float
    reduced_gradient_l2_after: float
    timing_tolerance: float
    physics_tolerance: float
    timing_tolerance_ratio_before: float
    timing_tolerance_ratio: float
    physics_tolerance_ratio_before: float
    physics_tolerance_ratio: float
    boozer_weighted_rms_before: float
    boozer_weighted_rms_after: float
    absolute_label_error_before: float
    absolute_label_error_after: float
    weighted_terms_before: Mapping[str, float]
    weighted_terms_after: Mapping[str, float]
    objective_before: float
    objective_after: float
    objective_increase: float
    surface_displacement_max_m: float
    surface_displacement_rms_m: float
    iota_before: float
    iota_after: float
    G_before: float
    G_after: float
    delta_iota: float
    delta_G: float

    def as_dict(self) -> dict[str, object]:
        """Plain JSON-compatible diagnostics, including corrected coordinates."""
        diagnostics = {
            field.name: (
                list(self.rejection_reasons)
                if field.name == "rejection_reasons"
                else dict(getattr(self, field.name))
                if field.name in ("weighted_terms_before", "weighted_terms_after")
                else getattr(self, field.name)
            )
            for field in fields(self)
            if field.name != "outer_vector"
        }
        diagnostics["corrected_outer_vector"] = self.outer_vector.tolist()
        return {key: _json_finite(value) for key, value in diagnostics.items()}


def _json_finite(value: object) -> object:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_finite(item) for item in value]
    return value


def _owned_readonly(values: object) -> NDArray[np.float64]:
    owned = np.array(jax.device_get(values), dtype=np.float64, copy=True)
    owned.setflags(write=False)
    return owned


def _boozer_system(problem: Flat675Problem, outer: NDArray[np.float64]):
    layout = problem.material.layout
    geometry = flat675_candidate_geometry(
        problem.material.boozer,
        jnp.asarray(outer[layout.coil_slice], dtype=jnp.float64),
        jnp.asarray(outer[layout.surface_slice], dtype=jnp.float64),
    )
    return build_flat675_boozer_system(geometry, problem.boozer_policy)


def _boozer_rms(system, iota: float, G: float) -> float:
    """Component RMS of [G B - |B|² (x_phi + iota x_theta)] / |B|."""
    y = jnp.asarray((iota, G), dtype=jnp.float64)
    residual = np.asarray(system.design_matrix @ y - system.right_hand_side)
    return float(np.linalg.norm(residual))


def _weighted_terms(
    problem: Flat675Problem, outer: NDArray[np.float64]
) -> dict[str, float]:
    values = flat675_weighted_terms(
        jnp.asarray(outer, dtype=jnp.float64),
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    return dict(
        zip(FLAT675_OBJECTIVE_TERM_KEYS, map(float, np.asarray(values)), strict=True)
    )


def polish_flat675(
    problem: Flat675Problem,
    outer_vector: object,
    *,
    limits: Flat675AcceptanceLimits | None = None,
) -> Flat675PolishResult:
    """Correct only the surface and assess explicit limits against the same point.

    The Schur dense-LU Newton solve uses the established 1e-13 physics bar.
    Missing design limits leave acceptance unassessed after a valid solve.
    """
    started = perf_counter()
    incoming = _owned_readonly(outer_vector)
    if not np.all(np.isfinite(incoming)):
        raise ValueError("flat-675 polish requires a finite outer vector")
    if (
        problem.boozer_policy.weight_by_inverse_field_magnitude
        is not NESTED_LS_WEIGHT_INV_MODB
    ):
        raise ValueError(
            "flat-675 polish requires the nested inverse-|B| weighting policy"
        )
    view = nested_view_from_flat675(problem, incoming)
    jax_boozer = view.jax_inputs.new_boozer_surface_jax()
    residual_fn, objective_fn, _ = nested_ls_reduced_closures(
        jax_boozer,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    packed_before = jnp.asarray(
        np.concatenate((view.surface_dofs, (view.iota, view.G)))
    )
    full_before = float(
        np.linalg.norm(np.asarray(jax.grad(objective_fn)(packed_before)))
    )
    solved = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=view.iota,
        G=view.G,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
        stab=NESTED_LS_JAX_INNER_STAB,
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_BANANA_NEWTON_MAXITER,
        linear_solver=_SCHUR_LINEAR_SOLVER,
        max_dense_linearization_bytes=None,
        residual_fn=residual_fn,
        objective_fn=objective_fn,
    )
    corrected = np.array(incoming, copy=True)
    corrected[view.layout.surface_slice] = solved.surface_dofs
    corrected.setflags(write=False)
    corrected_system = _boozer_system(problem, corrected)
    closed_y = np.asarray(
        solve_flat675_y_qr(
            corrected_system.design_matrix, corrected_system.right_hand_side
        ).solution,
        dtype=np.float64,
    )
    iota_after, G_after = map(float, closed_y)
    packed_after = jnp.asarray(np.concatenate((solved.surface_dofs, closed_y)))
    full_after = float(np.linalg.norm(np.asarray(jax.grad(objective_fn)(packed_after))))
    reduced_after = float(np.linalg.norm(solved.reduced_gradient))
    before_gamma = np.array(view.surface_native.gamma(), dtype=np.float64, copy=True)
    label_before = float(view.label_native.J())
    view.surface_native.set_dofs(solved.surface_dofs)
    after_gamma = np.array(view.surface_native.gamma(), dtype=np.float64, copy=True)
    label_after = float(view.label_native.J())
    point_moves = np.linalg.norm(after_gamma - before_gamma, axis=-1)
    terms_before = _weighted_terms(problem, incoming)
    terms_after = _weighted_terms(problem, corrected)
    boozer_before = _boozer_rms(_boozer_system(problem, incoming), view.iota, view.G)
    boozer_after = _boozer_rms(corrected_system, iota_after, G_after)
    branch = (
        abs(solved.iota - view.iota) <= NESTED_LS_OUTER_IOTA_BRANCH_GUARD
        and abs(iota_after - view.iota) <= NESTED_LS_OUTER_IOTA_BRANCH_GUARD
    )
    max_move = float(np.max(point_moves))
    rms_move = float(np.sqrt(np.mean(point_moves**2)))
    finite = bool(np.all(np.isfinite(corrected))) and all(
        np.isfinite(value)
        for value in (
            full_before,
            full_after,
            reduced_after,
            label_before,
            label_after,
            boozer_before,
            boozer_after,
            max_move,
            rms_move,
            iota_after,
            G_after,
            solved.iota,
            solved.G,
            *terms_before.values(),
            *terms_after.values(),
        )
    )
    reasons = []
    if not solved.success:
        reasons.append("solver_failed")
    if not solved.persisted:
        reasons.append("not_persisted")
    if not finite:
        reasons.append("nonfinite")
    if not branch:
        reasons.append("branch_changed")
    if full_after > NESTED_LS_NEWTON_TOL:
        reasons.append("physics_stationarity")
    if reduced_after > NESTED_LS_NEWTON_TOL:
        reasons.append("reduced_stationarity")
    label_target = float(problem.objective_policy.boozer_target_label)
    label_error_after = abs(label_after - label_target)
    objective_increase = sum(terms_after.values()) - sum(terms_before.values())
    if limits is not None:
        if boozer_after > limits.max_boozer_weighted_rms:
            reasons.append("boozer_weighted_rms")
        if label_error_after > limits.max_absolute_label_error:
            reasons.append("label_error")
        if max_move > limits.max_surface_displacement_m:
            reasons.append("surface_displacement")
        if objective_increase > limits.max_objective_increase:
            reasons.append("objective_increase")
    status: Literal["accepted", "rejected", "not_assessed"] = (
        "rejected" if reasons else "not_assessed" if limits is None else "accepted"
    )
    return Flat675PolishResult(
        outer_vector=corrected,
        acceptance_status=status,
        rejection_reasons=tuple(reasons),
        solver_success=bool(solved.success),
        persisted=bool(solved.persisted),
        finite=finite,
        same_branch_as_incoming=branch,
        iteration_count=solved.iteration_count,
        elapsed_seconds=perf_counter() - started,
        full_gradient_l2_before=full_before,
        full_gradient_l2_after=full_after,
        reduced_gradient_l2_after=reduced_after,
        timing_tolerance=NESTED_LS_BANANA_NEWTON_TOL,
        physics_tolerance=NESTED_LS_NEWTON_TOL,
        timing_tolerance_ratio_before=full_before / NESTED_LS_BANANA_NEWTON_TOL,
        timing_tolerance_ratio=full_after / NESTED_LS_BANANA_NEWTON_TOL,
        physics_tolerance_ratio_before=full_before / NESTED_LS_NEWTON_TOL,
        physics_tolerance_ratio=full_after / NESTED_LS_NEWTON_TOL,
        boozer_weighted_rms_before=boozer_before,
        boozer_weighted_rms_after=boozer_after,
        absolute_label_error_before=abs(label_before - label_target),
        absolute_label_error_after=label_error_after,
        weighted_terms_before=MappingProxyType(terms_before),
        weighted_terms_after=MappingProxyType(terms_after),
        objective_before=sum(terms_before.values()),
        objective_after=sum(terms_after.values()),
        objective_increase=objective_increase,
        surface_displacement_max_m=max_move,
        surface_displacement_rms_m=rms_move,
        iota_before=view.iota,
        iota_after=iota_after,
        G_before=view.G,
        G_after=G_after,
        delta_iota=iota_after - view.iota,
        delta_G=G_after - view.G,
    )


__all__ = ["Flat675AcceptanceLimits", "Flat675PolishResult", "polish_flat675"]
