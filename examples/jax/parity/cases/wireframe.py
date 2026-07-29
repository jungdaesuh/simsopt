"""Matched native/JAX bounded regularized wireframe-current workflow."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.parity_tolerances import parity_ladder_tolerances

WORKFLOW_STAGES = (
    "construct_bounded_wireframe_system",
    "evaluate_initial_system_and_constraints",
    "solve_wireframe_current_problem",
    "evaluate_final_currents_objective_and_feasibility",
)


def _wireframe_from_configuration(configuration: Mapping[str, object]):
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

    surface = SurfaceRZFourier(nfp=2, mpol=1, ntor=0)
    surface.set_rc(0, 0, float(configuration["major_radius"]))
    surface.set_rc(1, 0, float(configuration["minor_radius"]))
    surface.set_zs(1, 0, float(configuration["minor_radius"]))
    return ToroidalWireframe(
        surface,
        int(configuration["wireframe_nphi"]),
        int(configuration["wireframe_ntheta"]),
    )


def _build_wireframe(bundle: InputBundle):
    return _wireframe_from_configuration(bundle.configuration)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Persist a deterministic response matrix instead of regenerating it."""
    configuration = {
        "major_radius": 2.0,
        "minor_radius": 0.7,
        "wireframe_nphi": 4,
        "wireframe_ntheta": 6,
        "regularization": 0.1,
        "assume_no_crossings": False,
    }
    wireframe = _wireframe_from_configuration(configuration)
    rng = np.random.default_rng(20260726)
    response = rng.standard_normal((wireframe.n_segments + 10, wireframe.n_segments))
    target = rng.standard_normal((response.shape[0], 1))
    return create_input_bundle(
        root,
        case_id="wireframe-optimization",
        random_seed=20260726,
        arrays={
            "initial_currents": np.zeros((wireframe.n_segments, 1), dtype=np.float64),
            "response": np.asarray(response, dtype=np.float64),
            "target": np.asarray(target, dtype=np.float64),
        },
        configuration=configuration,
        scale=scale,
    )


def _construction(
    bundle: InputBundle, arrays: dict[str, np.ndarray], wireframe
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    constraint_matrix, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=bool(bundle.configuration["assume_no_crossings"]),
        remove_constrained_segments=True,
    )
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    matrix = np.asarray(constraint_matrix, dtype=np.float64)
    target = np.asarray(constraint_target, dtype=np.float64)
    payload = {
        "response": arrays["response"].tolist(),
        "target": arrays["target"].tolist(),
        "initial_currents": arrays["initial_currents"].tolist(),
        "constraint_matrix": matrix.tolist(),
        "constraint_target": target.tolist(),
        "free_segments": free_segments.tolist(),
        "regularization": bundle.configuration["regularization"],
        "assume_no_crossings": bundle.configuration["assume_no_crossings"],
    }
    fingerprint = effective_construction_fingerprint(bundle, payload)
    return matrix, target, free_segments, fingerprint


def _state(
    prefix: str,
    arrays: dict[str, np.ndarray],
    currents: np.ndarray,
    regularization: float,
    constraint_matrix: np.ndarray,
    constraint_target: np.ndarray,
    free_segments: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = arrays["response"] @ currents - arrays["target"]
    gradient = arrays["response"].T @ residual + regularization**2 * currents
    constraint_residual = (
        constraint_matrix @ currents[free_segments] - constraint_target
    )
    objective = 0.5 * (
        np.vdot(residual, residual) + regularization**2 * np.vdot(currents, currents)
    )
    return {
        f"{prefix}:currents": currents,
        f"{prefix}:normal_field_residual": residual,
        f"{prefix}:objective": np.asarray(objective, dtype=np.float64),
        f"{prefix}:objective_gradient": gradient,
        f"{prefix}:constraint_residual": constraint_residual,
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.solve.wireframe_optimization import rcls_wireframe

    wireframe = _build_wireframe(bundle)
    constraint_matrix, constraint_target, free_segments, fingerprint = _construction(
        bundle, arrays, wireframe
    )
    regularization = float(bundle.configuration["regularization"])
    initial = arrays["initial_currents"]
    initial_state = _state(
        "initial",
        arrays,
        initial,
        regularization,
        constraint_matrix,
        constraint_target,
        free_segments,
    )
    final, _, _, _ = rcls_wireframe(
        wireframe,
        arrays["response"],
        arrays["target"],
        regularization,
        bool(bundle.configuration["assume_no_crossings"]),
        False,
    )
    final = np.asarray(final, dtype=np.float64)
    final_state = _state(
        "final",
        arrays,
        final,
        regularization,
        constraint_matrix,
        constraint_target,
        free_segments,
    )
    tolerance = parity_ladder_tolerances("native_workflow")
    success = bool(
        np.linalg.norm(final_state["final:constraint_residual"])
        <= float(tolerance["terminal_constraint_norm_atol"])
        and np.isfinite(final_state["final:objective"]).all()
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="native_regularized_constrained_least_squares",
        normalized_status="converged" if success else "failed",
        raw_status="direct_kkt_solve_complete",
        success=success,
        nit=1,
        nfev=1,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**initial_state, **final_state},
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax_adapters.solve.wireframe import rcls_wireframe_jax

    import jax

    wireframe = _build_wireframe(bundle)
    constraint_matrix, constraint_target, free_segments, fingerprint = _construction(
        bundle, arrays, wireframe
    )
    regularization = float(bundle.configuration["regularization"])
    initial = arrays["initial_currents"]
    initial_state = _state(
        "initial",
        arrays,
        initial,
        regularization,
        constraint_matrix,
        constraint_target,
        free_segments,
    )
    result = rcls_wireframe_jax(
        wireframe,
        arrays["response"],
        arrays["target"],
        regularization,
        assume_no_crossings=bool(bundle.configuration["assume_no_crossings"]),
    )
    final = np.asarray(jax.device_get(jax.block_until_ready(result.x)))
    final_state = _state(
        "final",
        arrays,
        final,
        regularization,
        constraint_matrix,
        constraint_target,
        free_segments,
    )
    tolerance = parity_ladder_tolerances("native_workflow")
    success = bool(
        np.linalg.norm(final_state["final:constraint_residual"])
        <= float(tolerance["terminal_constraint_norm_atol"])
        and np.isfinite(final_state["final:objective"]).all()
    )
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="jax_regularized_constrained_least_squares",
        normalized_status="converged" if success else "failed",
        raw_status="direct_kkt_solve_complete",
        success=success,
        nit=1,
        nfev=1,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**initial_state, **final_state},
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the bounded constrained wireframe solve in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
