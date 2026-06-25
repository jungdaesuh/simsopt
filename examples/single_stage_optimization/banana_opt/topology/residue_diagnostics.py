from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict
from math import cos, isfinite, pi, sin

import numpy as np

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    DifferentiableMagneticFieldLike,
    FieldlineIntegratorOptions,
    integrate_target_return_map,
    target_winding_residual,
)
from .periodic_orbit import (
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitDiscoveryError,
    PeriodicOrbitResult,
    PeriodicOrbitSolverOptions,
    discover_periodic_orbit,
)
from .poincare_chart import PoincareChart
from .rational_target import RationalTarget


GREENE_RESIDUE_PROBE_SCHEMA_VERSION = "greene_residue_probe_v1"
DEFAULT_BRANCH_PHASE_COUNT = 8
DEFAULT_RADIAL_WINDOW_SAMPLE_COUNT = 3
DEFAULT_BRANCH_PHASE_ANGLES = tuple(
    2.0 * pi * float(index) / float(DEFAULT_BRANCH_PHASE_COUNT)
    for index in range(DEFAULT_BRANCH_PHASE_COUNT)
)


def uniform_branch_phase_angles(phase_count: int) -> tuple[float, ...]:
    count = int(phase_count)
    if count <= 0:
        raise ValueError("Greene residue branch phase count must be positive")
    return tuple(2.0 * pi * float(index) / float(count) for index in range(count))


def section_state_from_chart(
    chart: PoincareChart,
    *,
    radial_label: float,
    theta: float,
) -> tuple[float, float]:
    label = float(radial_label)
    angle = float(theta) / float(chart.poloidal_orientation)
    if label < 0.0 or not isfinite(label) or not isfinite(angle):
        raise ValueError(
            "Greene residue initial state requires finite radial label >= 0"
        )
    radius = chart.radial_label_scale * label
    return (
        float(chart.axis_r + radius * cos(angle)),
        float(chart.axis_z + radius * sin(angle)),
    )


def radial_multistart_initial_guesses(
    target: RationalTarget,
    chart: PoincareChart,
    *,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
) -> tuple[tuple[float, float], ...]:
    if target.radial_label is None:
        raise ValueError(
            "Greene residue probe requires target.radial_label for radial multistart"
        )
    if len(phase_angles) == 0:
        raise ValueError("Greene residue probe requires at least one phase angle")
    return tuple(
        section_state_from_chart(
            chart,
            radial_label=radial_label,
            theta=float(theta),
        )
        for radial_label in radial_multistart_labels(target)
        for theta in phase_angles
    )


def target_winding_ranked_initial_guesses(
    field: DifferentiableMagneticFieldLike,
    target: RationalTarget,
    chart: PoincareChart,
    *,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
) -> tuple[tuple[float, float], ...]:
    guesses = radial_multistart_initial_guesses(
        target,
        chart,
        phase_angles=phase_angles,
    )
    ranked_guesses: list[tuple[tuple[float, float, float, int], tuple[float, float]]] = []
    unranked_guesses: list[tuple[float, float]] = []
    for index, guess in enumerate(guesses):
        try:
            return_result = integrate_target_return_map(
                field,
                guess,
                target=target,
                chart=chart,
                options=integrator_options,
            )
        except RuntimeError:
            unranked_guesses.append(guess)
            continue
        closure_residual = (
            float(return_result.final_state[0]) - float(guess[0]),
            float(return_result.final_state[1]) - float(guess[1]),
        )
        winding_residual = abs(target_winding_residual(return_result, target))
        closure_norm = float(
            (closure_residual[0] * closure_residual[0])
            + (closure_residual[1] * closure_residual[1])
        )
        target_label = (
            0.0 if target.radial_label is None else float(target.radial_label)
        )
        radial_distance = abs(chart.radial_label(guess) - target_label)
        ranked_guesses.append(
            (
                (
                    winding_residual,
                    closure_norm,
                    radial_distance,
                    int(index),
                ),
                guess,
            )
        )
    return tuple(
        [guess for _rank, guess in sorted(ranked_guesses, key=lambda item: item[0])]
        + unranked_guesses
    )


def radial_multistart_labels(target: RationalTarget) -> tuple[float, ...]:
    if target.radial_label is None:
        raise ValueError(
            "Greene residue probe requires target.radial_label for radial multistart"
        )
    labels = [float(target.radial_label)]
    if target.radial_window is not None:
        lower, upper = target.radial_window
        labels.extend(
            float(label)
            for label in (
                np.linspace(
                    float(lower),
                    float(upper),
                    DEFAULT_RADIAL_WINDOW_SAMPLE_COUNT,
                )
            )
        )
    unique_labels: list[float] = []
    for label in labels:
        if not any(abs(label - previous) <= 1.0e-15 for previous in unique_labels):
            unique_labels.append(label)
    return tuple(unique_labels)


def rational_target_payload(target: RationalTarget) -> dict[str, object]:
    return {
        "id": target.manifest_key(),
        "p": int(target.p),
        "q": int(target.q),
        "iota": f"{target.p}/{target.q}",
        "iota_float": float(target.iota_float),
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


def poincare_chart_payload(chart: PoincareChart) -> dict[str, object]:
    return {
        "axis_r": float(chart.axis_r),
        "axis_z": float(chart.axis_z),
        "poloidal_orientation": int(chart.poloidal_orientation),
        "radial_label_scale": float(chart.radial_label_scale),
    }


def periodic_orbit_result_payload(result: PeriodicOrbitResult) -> dict[str, object]:
    """Serialize one branch solve, withholding the residue unless it converged.

    The Greene residue ``R = (2 - trM)/4`` and its O/X classification are only
    meaningful when the period-``q`` fixed point was actually found:
    ``solve_periodic_orbit`` computes them from the monodromy at the *last
    iterate* (periodic_orbit.py:543,554), so a ``newton_stalled`` /
    ``max_iterations`` result -- which ``discover_periodic_orbit`` still returns
    as the least-bad candidate (periodic_orbit.py:359-368) -- carries a residue
    of a point that is not a fixed point. Reporting it would let a non-converged
    solve emit an O/X verdict (a fail-open this certificate cannot have).

    So ``residue``, ``residue_classification``, and ``traceM`` (the residue's
    direct input) are emitted only when ``result.converged``; otherwise they are
    ``None``, exactly as ``periodic_orbit_failure_payload`` reports a discovery
    failure. The non-residue iterate diagnostics (``detM``, ``winding``,
    ``section_state``, ``newton_residual``, ``solver_iterations``, ...) are kept
    so a non-converged branch is still auditable -- the withheld fields are the
    ones whose value is *defined* only at a converged fixed point.
    """

    converged = bool(result.converged)
    return {
        "target_id": result.target.manifest_key(),
        "target": rational_target_payload(result.target),
        "branch": result.branch,
        "branch_status": result.status,
        "converged": converged,
        "residue": (
            float(result.residue_diagnostic.residue) if converged else None
        ),
        "residue_classification": (
            result.residue_diagnostic.classification if converged else None
        ),
        "traceM": (
            float(result.residue_diagnostic.trace_m) if converged else None
        ),
        "detM": float(result.tangent_result.det_m),
        "winding": float(result.winding),
        "raw_return_section_winding": float(
            result.tangent_result.return_map.raw_return_section_winding
        ),
        "radial_label": float(result.radial_label),
        "min_Bphi_over_B": float(result.min_bphi_over_b),
        "solver_iterations": int(result.iterations),
        "newton_residual": float(result.closure_residual_norm),
        "initial_state": list(result.initial_state),
        "section_state": list(result.state),
        "closure_residual": list(result.closure_residual),
    }


def periodic_orbit_failure_payload(
    *,
    target: RationalTarget,
    branch: str,
    status: str,
    message: str,
    initial_guesses: Sequence[Sequence[float]],
) -> dict[str, object]:
    return {
        "target_id": target.manifest_key(),
        "target": rational_target_payload(target),
        "branch": branch,
        "branch_status": status,
        "converged": False,
        "residue": None,
        "residue_classification": None,
        "traceM": None,
        "detM": None,
        "winding": None,
        "raw_return_section_winding": None,
        "radial_label": None,
        "min_Bphi_over_B": None,
        "solver_iterations": 0,
        "newton_residual": None,
        "initial_state": None,
        "section_state": None,
        "closure_residual": None,
        "initial_guess_count": len(tuple(initial_guesses)),
        "failure_message": message,
    }


def residue_probe_status_counts(
    diagnostics: Sequence[dict[str, object]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for diagnostic in diagnostics:
        status = str(diagnostic["branch_status"])
        counts[status] = counts.get(status, 0) + 1
    return counts


def run_residue_probe(
    field: DifferentiableMagneticFieldLike,
    *,
    targets: Sequence[RationalTarget],
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    phase_angles: Sequence[float] = DEFAULT_BRANCH_PHASE_ANGLES,
) -> dict[str, object]:
    diagnostics: list[dict[str, object]] = []
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
                result = discover_periodic_orbit(
                    field,
                    initial_guesses,
                    target=target,
                    chart=chart,
                    branch=branch,
                    integrator_options=integrator_options,
                    solver_options=solver_options,
                )
            except PeriodicOrbitDiscoveryError as error:
                diagnostics.append(
                    periodic_orbit_failure_payload(
                        target=target,
                        branch=branch,
                        status=error.status,
                        message=str(error),
                        initial_guesses=initial_guesses,
                    )
                )
                continue
            diagnostics.append(periodic_orbit_result_payload(result))

    return {
        "schema_version": GREENE_RESIDUE_PROBE_SCHEMA_VERSION,
        "chart": poincare_chart_payload(chart),
        "integrator_options": asdict(integrator_options),
        "solver_options": asdict(solver_options),
        "targets": [rational_target_payload(target) for target in targets],
        "phase_angles": [float(theta) for theta in phase_angles],
        "diagnostics": diagnostics,
        "branch_status_counts": residue_probe_status_counts(diagnostics),
    }
