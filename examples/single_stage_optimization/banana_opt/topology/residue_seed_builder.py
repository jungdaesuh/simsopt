"""Automatic resonance targeting for the Greene residue objective.

This module closes the manual gap between the read-only residue probe
(``residue_diagnostics.run_residue_probe``) and the heavily gated residue
objective (``residue_objective.BiotSavartGreeneResidueObjective``). Given a coil
``BiotSavart`` field and a set of candidate rational targets it:

1. discovers the O/X periodic orbits with the adaptive DOP853 integrator and
   ranks them by an island-width proxy ``sqrt(|R|)`` (wider islands first), and
2. for the worst islands, runs the *real* validation kernels already shipped in
   ``residue_sensitivity`` -- the VJP/RK4 second-order Taylor gate, the
   direct(RK4)-vs-proxy(DOP853) residue consistency check, and (for nonzero
   winding) the real-field convergence proof -- and emits ``validation_status=
   "passed"`` targets + seeds JSON payloads.

The emitted payloads are designed to round-trip through
``residue_objective.load_residue_objective_targets`` and
``load_residue_objective_seeds`` and to construct a weighted
``BiotSavartGreeneResidueObjective`` without bypassing any of its
anti-self-attestation gates: every validation here is produced by running the
same kernels the loader re-checks, never hand-asserted.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite, sqrt
from pathlib import Path

import numpy as np

from simsopt.field.biotsavart import BiotSavart

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    DifferentiableMagneticFieldLike,
    FieldlineIntegratorOptions,
)
from .periodic_orbit import (
    BRANCH_STATUS_CONVERGED,
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitDiscoveryError,
    PeriodicOrbitSolverOptions,
    discover_periodic_orbit,
)
from .poincare_chart import PoincareChart
from .rational_target import RationalTarget
from .residue_diagnostics import (
    DEFAULT_BRANCH_PHASE_ANGLES,
    target_winding_ranked_initial_guesses,
)
from .residue_objective import (
    GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_ORDER,
    GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED,
    residue_objective_target_manifest_id,
)
from .residue_sensitivity import (
    DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    branch_resolved_biot_savart_residue_vjp_taylor_diagnostic,
)


SectionState = tuple[float, float]

RESIDUE_SEED_BUILDER_SCHEMA_VERSION = "greene_residue_seed_builder_v1"
DEFAULT_TAYLOR_PROBE_STEPS = (1.0e-5, 5.0e-6, 2.5e-6)
DEFAULT_RESIDUE_DIFFERENCE_TOLERANCE = 1.0e-3
DEFAULT_REAL_FIELD_WINDING_TOLERANCE = 1.0e-4


@dataclass(frozen=True, slots=True)
class IslandCandidate:
    """A discovered O/X fixed point ranked by island-width proxy ``sqrt(|R|)``.

    ``proxy_residue`` is the DOP853 discovery residue; it doubles as the
    discretisation-independent reference for the later direct-vs-proxy gate.
    """

    target: RationalTarget
    branch: str
    section_state: SectionState
    proxy_residue: float
    residue_classification: str
    winding: float
    island_width_proxy: float


@dataclass(frozen=True, slots=True)
class ValidatedBranchSeed:
    """One island candidate that passed every residue-objective gate."""

    target: RationalTarget
    branch: str
    seed_section_state: SectionState
    direct_residue: float
    proxy_residue: float
    residue_difference: float
    residue_difference_tolerance: float
    taylor_min_order: float
    taylor_diagnostic: dict[str, object]
    real_field: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class SkippedBranch:
    """An island candidate that could not be validated, with the reason."""

    target_id: str
    branch: str
    reason: str


@dataclass(frozen=True, slots=True)
class ResidueSeedManifests:
    """Consumer-ready payloads plus the audit trail of accepted/skipped work."""

    targets_payload: dict[str, object]
    seeds_payload: dict[str, object]
    validated: tuple[ValidatedBranchSeed, ...]
    skipped: tuple[SkippedBranch, ...]


def rank_island_candidates(
    field: DifferentiableMagneticFieldLike,
    *,
    targets: Sequence[RationalTarget],
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
    min_island_width: float = 0.0,
) -> tuple[IslandCandidate, ...]:
    """Discover converged O/X orbits and rank them widest-island-first.

    Non-converged branches (no such island in this field) are dropped, as are
    islands whose width proxy ``sqrt(|R|)`` is below ``min_island_width``.
    """
    floor = float(min_island_width)
    if floor < 0.0 or not isfinite(floor):
        raise ValueError("min_island_width must be finite and nonnegative")
    candidates: list[IslandCandidate] = []
    for target in targets:
        initial_guesses = target_winding_ranked_initial_guesses(
            field,
            target,
            chart,
            integrator_options=integrator_options,
            phase_angles=phase_angles,
        )
        for branch in target.branches:
            try:
                orbit = discover_periodic_orbit(
                    field,
                    initial_guesses,
                    target=target,
                    chart=chart,
                    branch=branch,
                    integrator_options=integrator_options,
                    solver_options=solver_options,
                )
            except PeriodicOrbitDiscoveryError:
                continue
            if not orbit.converged:
                continue
            residue = float(orbit.residue_diagnostic.residue)
            width = sqrt(abs(residue))
            if width < floor:
                continue
            candidates.append(
                IslandCandidate(
                    target=target,
                    branch=branch,
                    section_state=orbit.state,
                    proxy_residue=residue,
                    residue_classification=orbit.residue_diagnostic.classification,
                    winding=float(orbit.winding),
                    island_width_proxy=width,
                )
            )
    candidates.sort(key=lambda candidate: candidate.island_width_proxy, reverse=True)
    return tuple(candidates)


def build_validated_residue_seeds(
    biot_savart: BiotSavart,
    *,
    candidates: Sequence[IslandCandidate],
    chart: PoincareChart,
    validation_artifact_id: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    taylor_probe_steps: Sequence[float] = DEFAULT_TAYLOR_PROBE_STEPS,
    taylor_min_order: float = GREENE_RESIDUE_OBJECTIVE_TAYLOR_GATE_MIN_ORDER,
    residue_difference_tolerance: float = DEFAULT_RESIDUE_DIFFERENCE_TOLERANCE,
    winding_tolerance: float = DEFAULT_REAL_FIELD_WINDING_TOLERANCE,
    max_candidates: int | None = None,
    direction: Sequence[float] | None = None,
    direction_seed: int = 0,
    r_satisfied: float = 0.0,
    local_difference_step: float = 1.0e-6,
) -> ResidueSeedManifests:
    """Run the real validation kernels and emit consumer-ready manifests.

    Each candidate (highest-ranked first, capped at ``max_candidates``) must pass
    the second-order Taylor gate, the direct-vs-proxy residue consistency check,
    and -- for nonzero winding -- a real-field convergence proof. Candidates that
    fail any gate are recorded in ``skipped`` rather than silently dropped, and a
    build that validates nothing raises loudly.
    """
    if not isinstance(biot_savart, BiotSavart):
        raise TypeError("Residue seed builder requires a direct BiotSavart field")
    artifact_id = str(validation_artifact_id)
    if artifact_id == "":
        raise ValueError("Residue seed builder validation_artifact_id must be nonempty")
    difference_tolerance = float(residue_difference_tolerance)
    if difference_tolerance < 0.0 or not isfinite(difference_tolerance):
        raise ValueError("residue_difference_tolerance must be finite and nonnegative")
    winding_tol = float(winding_tolerance)
    if winding_tol < 0.0 or not isfinite(winding_tol):
        raise ValueError("winding_tolerance must be finite and nonnegative")
    order_floor = float(taylor_min_order)
    direction_array = (
        _unit_dof_direction(biot_savart, seed=direction_seed)
        if direction is None
        else _checked_direction(biot_savart, direction)
    )
    selected = (
        tuple(candidates)
        if max_candidates is None
        else tuple(candidates)[: max(int(max_candidates), 0)]
    )

    validated: list[ValidatedBranchSeed] = []
    skipped: list[SkippedBranch] = []
    for candidate in selected:
        seed, reason = _validate_candidate(
            biot_savart,
            candidate,
            chart=chart,
            direction=direction_array,
            taylor_probe_steps=taylor_probe_steps,
            order_floor=order_floor,
            difference_tolerance=difference_tolerance,
            winding_tolerance=winding_tol,
            integrator_options=integrator_options,
            solver_options=solver_options,
            r_satisfied=r_satisfied,
            local_difference_step=local_difference_step,
        )
        if seed is None:
            skipped.append(
                SkippedBranch(
                    target_id=candidate.target.manifest_key(),
                    branch=candidate.branch,
                    reason=reason,
                )
            )
            continue
        validated.append(seed)

    if not validated:
        raise ValueError(
            "Residue seed builder produced no passing seeds; skipped "
            f"{[ (s.target_id, s.branch, s.reason) for s in skipped ]}"
        )

    targets_payload, seeds_payload = _assemble_manifests(
        validated,
        validation_artifact_id=artifact_id,
    )
    return ResidueSeedManifests(
        targets_payload=targets_payload,
        seeds_payload=seeds_payload,
        validated=tuple(validated),
        skipped=tuple(skipped),
    )


def autogenerate_residue_manifests(
    biot_savart: BiotSavart,
    *,
    targets: Sequence[RationalTarget],
    chart: PoincareChart,
    validation_artifact_id: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
    min_island_width: float = 0.0,
    max_candidates: int | None = None,
    **build_kwargs: object,
) -> ResidueSeedManifests:
    """Rank islands then build validated manifests in one call."""
    candidates = rank_island_candidates(
        biot_savart,
        targets=targets,
        chart=chart,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=phase_angles,
        min_island_width=min_island_width,
    )
    return build_validated_residue_seeds(
        biot_savart,
        candidates=candidates,
        chart=chart,
        validation_artifact_id=validation_artifact_id,
        integrator_options=integrator_options,
        solver_options=solver_options,
        max_candidates=max_candidates,
        **build_kwargs,
    )


def write_residue_manifests(
    manifests: ResidueSeedManifests,
    *,
    targets_path: str | Path,
    seeds_path: str | Path,
) -> tuple[Path, Path]:
    """Write the targets + seeds payloads as indented JSON, returning the paths."""
    resolved_targets_path = Path(targets_path)
    resolved_seeds_path = Path(seeds_path)
    resolved_targets_path.write_text(
        json.dumps(manifests.targets_payload, indent=2),
        encoding="utf-8",
    )
    resolved_seeds_path.write_text(
        json.dumps(manifests.seeds_payload, indent=2),
        encoding="utf-8",
    )
    return resolved_targets_path, resolved_seeds_path


def generate_residue_seed_files(
    biot_savart: BiotSavart,
    *,
    targets: Sequence[RationalTarget],
    chart: PoincareChart,
    validation_artifact_id: str,
    targets_path: str | Path,
    seeds_path: str | Path,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
    min_island_width: float = 0.0,
    max_candidates: int | None = None,
    **build_kwargs: object,
) -> ResidueSeedManifests:
    """One-call CLI core: rank islands, validate, and write the JSON manifests."""
    manifests = autogenerate_residue_manifests(
        biot_savart,
        targets=targets,
        chart=chart,
        validation_artifact_id=validation_artifact_id,
        integrator_options=integrator_options,
        solver_options=solver_options,
        phase_angles=phase_angles,
        min_island_width=min_island_width,
        max_candidates=max_candidates,
        **build_kwargs,
    )
    write_residue_manifests(
        manifests,
        targets_path=targets_path,
        seeds_path=seeds_path,
    )
    return manifests


def _validate_candidate(
    biot_savart: BiotSavart,
    candidate: IslandCandidate,
    *,
    chart: PoincareChart,
    direction: np.ndarray,
    taylor_probe_steps: Sequence[float],
    order_floor: float,
    difference_tolerance: float,
    winding_tolerance: float,
    integrator_options: FieldlineIntegratorOptions,
    solver_options: PeriodicOrbitSolverOptions,
    r_satisfied: float,
    local_difference_step: float,
) -> tuple[ValidatedBranchSeed | None, str]:
    taylor = branch_resolved_biot_savart_residue_vjp_taylor_diagnostic(
        biot_savart,
        candidate.section_state,
        direction=direction,
        probe_steps=taylor_probe_steps,
        target=candidate.target,
        chart=chart,
        branch=candidate.branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
        r_satisfied=r_satisfied,
        local_difference_step=local_difference_step,
    )
    if taylor.base_status != BRANCH_STATUS_CONVERGED:
        return None, f"RK4 base orbit status {taylor.base_status}"
    if len(taylor.observed_orders) == 0:
        return None, "Taylor gate produced no observed orders"
    min_order = float(min(taylor.observed_orders))
    if not (min_order >= order_floor):
        return None, f"Taylor order {min_order:.4g} < {order_floor:.4g}"

    direct_residue = float(taylor.base_residue)
    proxy_residue = float(candidate.proxy_residue)
    residue_difference = abs(direct_residue - proxy_residue)
    if residue_difference > difference_tolerance:
        return None, (
            f"direct/proxy residue disagree by {residue_difference:.3g} "
            f"> {difference_tolerance:.3g}"
        )

    real_field: dict[str, object] | None = None
    if candidate.target.expected_winding() != 0:
        expected_winding = float(candidate.target.expected_winding())
        observed_winding = float(taylor.base_winding)
        winding_residual = observed_winding - expected_winding
        if abs(winding_residual) > winding_tolerance:
            return None, (
                f"winding residual {winding_residual:.3g} exceeds "
                f"tolerance {winding_tolerance:.3g}"
            )
        real_field = {
            "expected_winding": expected_winding,
            "observed_winding": observed_winding,
            "winding_residual": winding_residual,
            "winding_tolerance": winding_tolerance,
        }

    return (
        ValidatedBranchSeed(
            target=candidate.target,
            branch=candidate.branch,
            seed_section_state=_normalize_state(taylor.base_state),
            direct_residue=direct_residue,
            proxy_residue=proxy_residue,
            residue_difference=residue_difference,
            residue_difference_tolerance=difference_tolerance,
            taylor_min_order=min_order,
            taylor_diagnostic=taylor.to_json_dict(),
            real_field=real_field,
        ),
        "",
    )


def _assemble_manifests(
    validated: Sequence[ValidatedBranchSeed],
    *,
    validation_artifact_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    # Group validated branches by their originating target, preserving the
    # widest-island-first order in which they were accepted. The emitted target
    # declares exactly the branches that passed, so the objective's seed/target
    # key sets match (no "missing"/"unexpected" branch seed errors at load).
    ordered_targets: list[RationalTarget] = []
    branches_by_target: dict[RationalTarget, list[str]] = {}
    for seed in validated:
        if seed.target not in branches_by_target:
            branches_by_target[seed.target] = []
            ordered_targets.append(seed.target)
        branches_by_target[seed.target].append(seed.branch)

    limited_targets: list[RationalTarget] = []
    target_id_by_source: dict[RationalTarget, str] = {}
    for source in ordered_targets:
        validated_branches = tuple(
            branch
            for branch in source.branches
            if branch in set(branches_by_target[source])
        )
        limited = replace(source, branches=validated_branches)
        limited_targets.append(limited)
        target_id_by_source[source] = limited.manifest_key()

    target_manifest_id = residue_objective_target_manifest_id(limited_targets)

    branch_seeds: list[dict[str, object]] = []
    optimizer_taylor_validations: list[dict[str, object]] = []
    direct_proxy_validations: list[dict[str, object]] = []
    real_field_validations: list[dict[str, object]] = []
    for seed in validated:
        target_id = target_id_by_source[seed.target]
        section_state = list(seed.seed_section_state)
        branch_seeds.append(
            {
                "target_id": target_id,
                "branch": seed.branch,
                "section_state": section_state,
            }
        )
        optimizer_taylor_validations.append(
            {
                "target_id": target_id,
                "branch": seed.branch,
                "diagnostic": seed.taylor_diagnostic,
            }
        )
        direct_proxy_validations.append(
            {
                "target_id": target_id,
                "branch": seed.branch,
                "section_state": section_state,
                "validation_status": GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED,
                "direct_residue": seed.direct_residue,
                "proxy_residue": seed.proxy_residue,
                "absolute_residue_difference": seed.residue_difference,
                "residue_difference_tolerance": seed.residue_difference_tolerance,
            }
        )
        if seed.real_field is not None:
            real_field_validations.append(
                {
                    "target_id": target_id,
                    "branch": seed.branch,
                    "section_state": section_state,
                    "validation_status": (
                        GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED
                    ),
                    "branch_status": BRANCH_STATUS_CONVERGED,
                    **seed.real_field,
                }
            )

    seeds_payload: dict[str, object] = {
        "schema_version": RESIDUE_SEED_BUILDER_SCHEMA_VERSION,
        "target_manifest_id": target_manifest_id,
        "validation_status": GREENE_RESIDUE_OBJECTIVE_VALIDATION_STATUS_PASSED,
        "validation_artifact_id": validation_artifact_id,
        "branch_seeds": branch_seeds,
        "optimizer_taylor_validations": optimizer_taylor_validations,
        "direct_proxy_consistency_validations": direct_proxy_validations,
        "real_field_nonzero_winding_validations": real_field_validations,
    }
    targets_payload: dict[str, object] = {
        "targets": [_target_payload(target) for target in limited_targets]
    }
    return targets_payload, seeds_payload


def _target_payload(target: RationalTarget) -> dict[str, object]:
    return {
        "p": int(target.p),
        "q": int(target.q),
        "weight": float(target.weight),
        "radial_label": target.radial_label,
        "radial_window": (
            None if target.radial_window is None else list(target.radial_window)
        ),
        "branches": list(target.branches),
        "phi0": float(target.phi0),
        "nfp": int(target.nfp),
        "convention": target.convention,
        "map_convention": target.map_convention,
        "fourier_m": target.fourier_m,
        "fourier_n": target.fourier_n,
    }


def _unit_dof_direction(field: BiotSavart, *, seed: int) -> np.ndarray:
    free_dofs = np.asarray(field.x, dtype=float)
    if free_dofs.size == 0:
        raise ValueError("BiotSavart has no free DOFs to build a Taylor direction")
    generator = np.random.default_rng(int(seed))
    direction = generator.normal(size=free_dofs.shape)
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("Generated residue Taylor direction has zero norm")
    return direction / norm


def _checked_direction(field: BiotSavart, direction: Sequence[float]) -> np.ndarray:
    direction_array = np.asarray(direction, dtype=float)
    expected_shape = np.asarray(field.x, dtype=float).shape
    if direction_array.shape != expected_shape:
        raise ValueError(
            "Residue Taylor direction must match the BiotSavart free-DOF shape "
            f"{expected_shape}, got {direction_array.shape}"
        )
    norm = float(np.linalg.norm(direction_array))
    if norm <= 0.0 or not np.all(np.isfinite(direction_array)):
        raise ValueError("Residue Taylor direction must be finite and nonzero")
    return direction_array / norm


def _normalize_state(state: Sequence[float]) -> SectionState:
    if len(state) != 2:
        raise ValueError("Residue seed section_state must be [R, Z]")
    radius = float(state[0])
    height = float(state[1])
    if radius <= 0.0 or not isfinite(radius) or not isfinite(height):
        raise ValueError("Residue seed section_state requires finite R > 0")
    return (radius, height)
