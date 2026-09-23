"""Exact matched workflow for ``1_Simple/surf_vol_area.py``."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from examples.jax.official_tiny_least_squares import (
    DRIVER_SURFACE,
    TrfOutcome,
    combine_trf_outcomes,
    guard_finite_endpoint,
    solve_jax_residual,
    surface_area_volume_residual,
    trf_outcome,
    value_and_jacobian,
)
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._official_least_squares import (
    declared_work_budget,
    official_stopping_keywords,
    solve_official_least_squares,
)
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from examples.jax.parity.symmetry import (
    global_column_swap_jacobian_invariants,
    parameter_invariants,
)
from simsopt import load
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import LeastSquaresProblem
from simsopt_jax.examples import ExecutionScale

import jax

WORKFLOW_STAGES = (
    "construct_first_area_volume_problem",
    "evaluate_first_initial_state",
    "solve_first_area_volume_problem",
    "materialize_accepted_surface_state",
    "construct_second_area_volume_problem",
    "evaluate_second_initial_state",
    "solve_second_area_volume_problem",
    "evaluate_both_final_states",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize the native surface constants for every execution lane."""
    native_surface = (
        SurfaceRZFourier(mpol=1, ntor=0) if scale == "native_default" else None
    )
    quadrature = (
        np.asarray(native_surface.quadpoints_phi)
        if native_surface is not None
        else np.linspace(0.0, 1.0, 32, endpoint=False)
    )
    native_quadrature = (
        {"quadrature_theta": np.asarray(native_surface.quadpoints_theta)}
        if native_surface is not None
        else {}
    )
    return create_input_bundle(
        root,
        case_id="native-surf-vol-area",
        random_seed=0,
        arrays={
            "initial_parameters": np.asarray((0.1, 0.1), dtype=np.float64),
            "quadrature": quadrature,
            **native_quadrature,
            "stage_targets": np.asarray(
                ((8.0, 0.6), (9.0, 0.8)),
                dtype=np.float64,
            ),
        },
        configuration={
            "major_radius": 1.0,
            "mpol": 1,
            "ntor": 0,
            "nfp": 1,
            "stellsym": True,
            "max_steps": 64 if scale == "bounded" else 256,
        },
        scale=scale,
    )


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _build_surface(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    quadrature = arrays["quadrature"]
    surface = SurfaceRZFourier(
        mpol=_configuration_int(bundle, "mpol"),
        ntor=_configuration_int(bundle, "ntor"),
        nfp=_configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        quadpoints_phi=quadrature,
        quadpoints_theta=(
            arrays["quadrature_theta"]
            if bundle.scale == "native_default"
            else quadrature
        ),
    )
    surface.set_rc(0, 0, _configuration_float(bundle, "major_radius"))
    surface.set_rc(1, 0, float(arrays["initial_parameters"][0]))
    surface.set_zs(1, 0, float(arrays["initial_parameters"][1]))
    surface.fix("rc(0,0)")
    return surface


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    surface,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "full_dofs": np.asarray(surface.local_full_x).tolist(),
            "free_positions": np.flatnonzero(surface.local_dofs_free_status).tolist(),
            "quadpoints_phi": np.asarray(surface.quadpoints_phi).tolist(),
            "quadpoints_theta": np.asarray(surface.quadpoints_theta).tolist(),
            "stage_targets": arrays["stage_targets"].tolist(),
            "mpol": surface.mpol,
            "ntor": surface.ntor,
            "nfp": surface.nfp,
            "stellsym": surface.stellsym,
            "max_steps": bundle.configuration["max_steps"],
        },
    )


def _roundtrip_surface(surface: SurfaceRZFourier) -> SurfaceRZFourier:
    """Save/load through an isolated directory preserving upstream roundtrip."""
    with TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "surf_fw.json")
        surface.save(path, indent=2)
        return load(path)


def _state(
    prefix: str,
    parameters: np.ndarray,
    area: float,
    volume: float,
    residual: np.ndarray,
    jacobian: np.ndarray,
) -> dict[str, np.ndarray]:
    column_sum, column_product, column_association = (
        global_column_swap_jacobian_invariants(
            jacobian[:, 0],
            jacobian[:, 1],
        )
    )
    jacobian_invariants = np.concatenate(
        (column_sum, column_product, column_association.reshape(-1))
    )
    objective_gradient = 2.0 * jacobian.T @ residual
    return {
        f"{prefix}:parameters": parameters,
        f"{prefix}:parameter_invariants": parameter_invariants(parameters),
        f"{prefix}:area": np.asarray(area, dtype=np.float64),
        f"{prefix}:volume": np.asarray(volume, dtype=np.float64),
        f"{prefix}:residual": residual,
        f"{prefix}:residual_jacobian": jacobian,
        f"{prefix}:residual_jacobian_invariants": jacobian_invariants,
        f"{prefix}:objective_sum_squares": np.asarray(
            np.vdot(residual, residual),
            dtype=np.float64,
        ),
        f"{prefix}:objective_gradient": objective_gradient,
        f"{prefix}:objective_gradient_invariants": parameter_invariants(
            objective_gradient
        ),
    }


def _native_stage(
    *,
    surface,
    targets: np.ndarray,
    bundle: InputBundle,
    prefix: str,
    centered: bool,
) -> tuple[dict[str, np.ndarray], TrfOutcome]:

    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    problem = LeastSquaresProblem.from_tuples(
        (
            (surface.area, float(targets[0]), 1.0),
            (surface.volume, float(targets[1]), 1.0),
        )
    )

    def state(phase: str) -> dict[str, np.ndarray]:
        parameters = np.asarray(problem.x, dtype=np.float64)
        area = float(surface.area())
        volume = float(surface.volume())
        residual = np.asarray((area, volume), dtype=np.float64) - targets
        jacobian = np.stack(
            (
                np.asarray(surface.darea(), dtype=np.float64)[free_positions],
                np.asarray(surface.dvolume(), dtype=np.float64)[free_positions],
            )
        )
        return _state(
            f"{prefix}:{phase}",
            parameters,
            area,
            volume,
            residual,
            jacobian,
        )

    initial = state("initial")
    budget = declared_work_budget(bundle.scale, _configuration_int(bundle, "max_steps"))
    # The official second solve passes the inert ``diff_method="centered"``; the
    # keyword is only read in the wrapper's ``grad`` branch (serial.py:159-164).
    method = {"diff_method": "centered"} if centered else {}
    result = solve_official_least_squares(
        problem, **method, **official_stopping_keywords(budget)
    )
    final = state("final")
    values = {**initial, **final}
    return values, guard_finite_endpoint(trf_outcome(result), values.values())


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:

    surface = _build_surface(bundle, arrays)
    fingerprint = _effective_fingerprint(bundle, arrays, surface)
    first_values, first_outcome = _native_stage(
        surface=surface,
        targets=arrays["stage_targets"][0],
        bundle=bundle,
        prefix="first",
        centered=False,
    )
    second_surface = _roundtrip_surface(surface)
    second_values, second_outcome = _native_stage(
        surface=second_surface,
        targets=arrays["stage_targets"][1],
        bundle=bundle,
        prefix="second",
        centered=True,
    )
    # The official wrapper discards SciPy's result; the recorder keeps it, so this
    # lane publishes the provider's own outcome for both solves. Official capture
    # A/reference-simple/runs/native-surf-vol-area/captured-natural-omp1:
    # status 1 `gtol` nfev 9 njev 8, then status 1 `gtol` nfev 5 njev 5.
    outcome = combine_trf_outcomes(first_outcome, second_outcome)
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="simsopt_least_squares_serial_solve",
        normalized_status=outcome.normalized_status,
        raw_status=outcome.raw_status,
        success=outcome.success,
        nit=None,
        nfev=outcome.nfev,
        njev=outcome.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**first_values, **second_values},
        applicability={
            "first:final:parameters": False,
            "first:final:residual_jacobian": False,
            "first:final:objective_gradient": False,
            "second:initial:parameters": False,
            "second:initial:residual_jacobian": False,
            "second:initial:objective_gradient": False,
            "second:final:parameters": False,
            "second:final:residual_jacobian": False,
            "second:final:objective_gradient": False,
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:

    surface = _build_surface(bundle, arrays)
    fingerprint = _effective_fingerprint(bundle, arrays, surface)

    def stage_values(prefix: str, current, targets: np.ndarray):
        residual = surface_area_volume_residual(
            np.asarray(current.local_full_x, dtype=np.float64),
            np.asarray(current.quadpoints_phi, dtype=np.float64),
            np.asarray(current.quadpoints_theta, dtype=np.float64),
            np.flatnonzero(current.local_dofs_free_status),
            targets,
            mpol=current.mpol,
            ntor=current.ntor,
            nfp=current.nfp,
            stellsym=current.stellsym,
        )
        initial_parameters = np.asarray(current.x, dtype=np.float64)
        initial_residual, initial_jacobian = value_and_jacobian(
            residual, initial_parameters
        )
        optimizer = solve_jax_residual(
            residual,
            initial_parameters,
            max_nfev=declared_work_budget(
                bundle.scale, _configuration_int(bundle, "max_steps")
            ),
        )
        final_parameters = np.asarray(optimizer.x, dtype=np.float64)
        final_residual, final_jacobian = value_and_jacobian(residual, final_parameters)
        current.x = final_parameters
        initial_state = _state(
            f"{prefix}:initial",
            initial_parameters,
            float(initial_residual[0] + targets[0]),
            float(initial_residual[1] + targets[1]),
            initial_residual,
            initial_jacobian,
        )
        final_state = _state(
            f"{prefix}:final",
            final_parameters,
            float(final_residual[0] + targets[0]),
            float(final_residual[1] + targets[1]),
            final_residual,
            final_jacobian,
        )
        values = {**initial_state, **final_state}
        return values, guard_finite_endpoint(trf_outcome(optimizer), values.values())

    first_values, first_outcome = stage_values(
        "first", surface, arrays["stage_targets"][0]
    )
    second_surface = _roundtrip_surface(surface)
    second_values, second_outcome = stage_values(
        "second", second_surface, arrays["stage_targets"][1]
    )
    outcome = combine_trf_outcomes(first_outcome, second_outcome)
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver=DRIVER_SURFACE,
        normalized_status=outcome.normalized_status,
        raw_status=outcome.raw_status,
        success=outcome.success,
        nit=None,
        nfev=outcome.nfev,
        njev=outcome.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **first_values,
            **second_values,
        },
        applicability={
            "first:final:parameters": False,
            "first:final:residual_jacobian": False,
            "first:final:objective_gradient": False,
            "second:initial:parameters": False,
            "second:initial:residual_jacobian": False,
            "second:initial:objective_gradient": False,
            "second:final:parameters": False,
            "second:final:residual_jacobian": False,
            "second:final:objective_gradient": False,
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact two-stage native or JAX surface workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
