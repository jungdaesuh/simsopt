"""Shared exact native/JAX GSCO parity execution."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _effective_fingerprint,
)
from examples.jax.parity.input_bundle import InputBundle, create_input_bundle
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExecutionScale

GeometryBuilder = Callable[
    [Mapping[str, object]],
    tuple[ToroidalWireframe, SurfaceRZFourier, float],
]

WORKFLOW_STAGES = (
    "construct_qa_plasma_and_nescoil_wireframe",
    "seed_toroidal_field_coils_and_apply_constraints",
    "construct_area_weighted_normal_field_response",
    "evaluate_initial_currents_and_objectives",
    "run_fixed_iteration_gsco",
    "evaluate_final_currents_objectives_and_constraints",
)


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def create_gsco_input(
    root: Path,
    scale: ExecutionScale,
    *,
    case_id: str,
    configuration: dict[str, object],
    build_geometry: GeometryBuilder,
) -> InputBundle:
    """Freeze exact source geometry, topology, response, and GSCO state."""
    from simsopt.solve.wireframe_optimization import bnorm_obj_matrices

    wireframe, plasma, _ = build_geometry(configuration)
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma,
        area_weighted=True,
        verbose=False,
    )
    arrays = {
        "response_matrix": np.array(response, dtype=np.float64, copy=True),
        "target": np.array(target, dtype=np.float64, copy=True),
        "initial_currents": np.array(
            wireframe.currents,
            dtype=np.float64,
            copy=True,
        ).reshape((-1, 1)),
        "initial_loop_count": np.zeros(
            len(wireframe.get_free_cells(form="logical")),
            dtype=np.int64,
        ),
        "loops": np.array(wireframe.get_cell_key(), dtype=np.int32, copy=True),
        "free_loops": np.array(
            wireframe.get_free_cells(form="logical"),
            dtype=np.int32,
            copy=True,
        ),
        "segments": np.array(wireframe.segments, dtype=np.int32, copy=True),
        "connections": np.array(
            wireframe.connected_segments,
            dtype=np.int32,
            copy=True,
        ),
        "constrained_segments": np.array(
            wireframe.constrained_segments(),
            dtype=np.int64,
            copy=True,
        ),
    }
    return create_input_bundle(
        root,
        case_id=case_id,
        random_seed=0,
        arrays=arrays,
        configuration={
            **configuration,
            "n_segments": int(wireframe.n_segments),
            "n_free_loops": int(arrays["free_loops"].shape[0]),
        },
        scale=scale,
    )


def _construction_values(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"construction:{name}": value
        for name, value in arrays.items()
        if name not in {"initial_currents", "initial_loop_count"}
    }


def _published_values(
    arrays: dict[str, np.ndarray],
    *,
    x: np.ndarray,
    loop_count: np.ndarray,
    iter_history: np.ndarray,
    current_history: np.ndarray,
    loop_history: np.ndarray,
    normal_history: np.ndarray,
    sparsity_history: np.ndarray,
    total_history: np.ndarray,
) -> dict[str, np.ndarray]:
    initial_currents = arrays["initial_currents"]
    initial_residual = arrays["response_matrix"] @ initial_currents - arrays["target"]
    final_residual = arrays["response_matrix"] @ x - arrays["target"]
    constrained = arrays["constrained_segments"]
    constraint_satisfied = np.all(x.reshape(-1)[constrained] == 0.0)
    return {
        **_construction_values(arrays),
        "initial:currents": initial_currents,
        "initial:normal_field_residual": initial_residual,
        "initial:normal_objective": np.asarray(
            normal_history[0],
            dtype=np.float64,
        ),
        "initial:sparsity_objective": np.asarray(
            sparsity_history[0],
            dtype=np.float64,
        ),
        "initial:total_objective": np.asarray(
            total_history[0],
            dtype=np.float64,
        ),
        "final:currents": x,
        "final:loop_count": loop_count,
        "final:normal_field_residual": final_residual,
        "final:normal_objective": np.asarray(
            normal_history[-1],
            dtype=np.float64,
        ),
        "final:sparsity_objective": np.asarray(
            sparsity_history[-1],
            dtype=np.float64,
        ),
        "final:total_objective": np.asarray(
            total_history[-1],
            dtype=np.float64,
        ),
        "final:iterations": np.asarray(iter_history[-1], dtype=np.int64),
        "final:maximum_current": np.asarray(
            np.max(np.abs(x)),
            dtype=np.float64,
        ),
        "final:active_segments": np.asarray(
            np.count_nonzero(x),
            dtype=np.int64,
        ),
        "final:constraints_satisfied": np.asarray(constraint_satisfied),
        "history:iterations": iter_history,
        "history:currents": current_history,
        "history:loops": loop_history,
        "history:normal_objective": normal_history,
        "history:sparsity_objective": sparsity_history,
        "history:total_objective": total_history,
    }


def _observation(
    lane: str,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    *,
    backend_mode: str,
    platform: str,
    driver: str,
    values: dict[str, np.ndarray],
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["final:currents"]))
        and values["final:normal_objective"] < values["initial:normal_objective"]
        and bool(values["final:constraints_satisfied"])
    )
    iterations = int(values["final:iterations"])
    return LaneObservation(
        lane=lane,
        backend_mode=backend_mode,
        platform=platform,
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_gsco_complete",
        success=success,
        nit=iterations,
        nfev=iterations,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry: GeometryBuilder,
) -> LaneObservation:
    from simsopt.solve.wireframe_optimization import gsco_wireframe

    wireframe, _, _ = build_geometry(bundle.configuration)
    max_iterations = _configuration_int(bundle, "max_iterations")
    result = gsco_wireframe(
        wireframe,
        arrays["response_matrix"],
        arrays["target"],
        _configuration_float(bundle, "lambda_s"),
        bool(bundle.configuration["no_crossing"]),
        bool(bundle.configuration["match_current"]),
        _configuration_float(bundle, "default_current"),
        _configuration_float(bundle, "max_current"),
        max_iterations,
        max_iterations,
        no_new_coils=bool(bundle.configuration["no_new_coils"]),
        max_loop_count=_configuration_int(bundle, "max_loop_count"),
        x_init=arrays["initial_currents"],
        loop_count_init=arrays["initial_loop_count"],
        verbose=False,
    )
    x, loop_count, iterations, currents, loops, normal, sparsity, total, _ = result
    values = _published_values(
        arrays,
        x=np.asarray(x, dtype=np.float64),
        loop_count=np.asarray(loop_count, dtype=np.int64),
        iter_history=np.asarray(iterations, dtype=np.int64),
        current_history=np.asarray(currents, dtype=np.float64),
        loop_history=np.asarray(loops, dtype=np.int64),
        normal_history=np.asarray(normal, dtype=np.float64),
        sparsity_history=np.asarray(sparsity, dtype=np.float64),
        total_history=np.asarray(total, dtype=np.float64),
    )
    return _observation(
        "native-cpu",
        bundle,
        arrays,
        backend_mode="native_cpu",
        platform="cpu",
        driver="simsopt_cpp_gsco",
        values=values,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry: GeometryBuilder,
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.solve.wireframe import gsco_wireframe_jax

    import jax

    wireframe, _, _ = build_geometry(bundle.configuration)
    max_iterations = _configuration_int(bundle, "max_iterations")
    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    result = jax.device_get(
        gsco_wireframe_jax(
            wireframe,
            put("response_matrix"),
            put("target"),
            _configuration_float(bundle, "lambda_s"),
            bool(bundle.configuration["no_crossing"]),
            bool(bundle.configuration["match_current"]),
            _configuration_float(bundle, "default_current"),
            _configuration_float(bundle, "max_current"),
            max_iterations,
            max_iterations,
            no_new_coils=bool(bundle.configuration["no_new_coils"]),
            max_loop_count=_configuration_int(bundle, "max_loop_count"),
            x_init=put("initial_currents"),
            loop_count_init=put("initial_loop_count"),
            record_every=1,
            verbose=False,
        )
    )
    history_length = int(result.history_length)
    history = slice(0, history_length)
    values = _published_values(
        arrays,
        x=np.asarray(result.x, dtype=np.float64),
        loop_count=np.asarray(result.loop_count, dtype=np.int64),
        iter_history=np.asarray(result.iter_history[history], dtype=np.int64),
        current_history=np.asarray(result.curr_history[history], dtype=np.float64),
        loop_history=np.asarray(result.loop_history[history], dtype=np.int64),
        normal_history=np.asarray(result.f_B_history[history], dtype=np.float64),
        sparsity_history=np.asarray(result.f_S_history[history], dtype=np.float64),
        total_history=np.asarray(result.f_history[history], dtype=np.float64),
    )
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        arrays,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        driver="simsopt_jax_gsco",
        values=values,
    )


def execute_gsco(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry: GeometryBuilder,
) -> LaneObservation:
    """Execute an exact GSCO source workflow in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays, build_geometry)
    return _jax(lane, bundle, arrays, build_geometry)
