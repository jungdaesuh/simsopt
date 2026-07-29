"""Matched native-C++/JAX bounded greedy permanent-magnet workflow."""

from __future__ import annotations

import os
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

WORKFLOW_STAGES = (
    "construct_bounded_permanent_magnet_problem",
    "evaluate_initial_normal_field_residual",
    "run_deterministic_greedy_magnet_solve",
    "evaluate_final_moments_and_normal_field_residual",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Create the fixed two-dipole response and deterministic solve policy."""
    return create_input_bundle(
        root,
        case_id="permanent-magnet-optimization",
        random_seed=0,
        arrays={
            "response": np.eye(6, dtype=np.float64),
            "target": np.asarray((0.8, 0.0, 0.0, -0.6, 0.0, 0.0)),
            "initial_moments": np.zeros((2, 3), dtype=np.float64),
            "moment_maxima": np.ones(2, dtype=np.float64),
            "normal_norms": np.ones(6, dtype=np.float64),
            "dipole_grid_xyz": np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64
            ),
        },
        configuration={
            "iterations": 2,
            "regularization_l2": 0.0,
            "single_direction": -1,
            "coordinate_flag": "cartesian",
        },
        scale=scale,
    )


def _effective_fingerprint(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> str:
    payload = {
        "response": arrays["response"].tolist(),
        "target": arrays["target"].tolist(),
        "initial_moments": arrays["initial_moments"].tolist(),
        "moment_maxima": arrays["moment_maxima"].tolist(),
        "normal_norms": arrays["normal_norms"].tolist(),
        "dipole_grid_xyz": arrays["dipole_grid_xyz"].tolist(),
        "iterations": bundle.configuration["iterations"],
        "regularization_l2": bundle.configuration["regularization_l2"],
        "single_direction": bundle.configuration["single_direction"],
        "coordinate_flag": bundle.configuration["coordinate_flag"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _state(
    prefix: str,
    response: np.ndarray,
    target: np.ndarray,
    moments: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = response @ moments.reshape(-1) - target
    return {
        f"{prefix}:moments": moments,
        f"{prefix}:residual": residual,
        f"{prefix}:objective_sum_squares": np.asarray(
            np.vdot(residual, residual), dtype=np.float64
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    import simsoptpp

    if bundle.configuration["coordinate_flag"] != "cartesian":
        raise ValueError("native permanent-magnet case requires cartesian coordinates")

    response = arrays["response"]
    target = arrays["target"]
    initial = arrays["initial_moments"]
    maxima_vector = np.repeat(arrays["moment_maxima"], 3)
    _, _, _, final = simsoptpp.GPMO_baseline(
        np.ascontiguousarray(response.T),
        np.ascontiguousarray(target),
        np.sqrt(float(bundle.configuration["regularization_l2"])) * maxima_vector,
        np.ascontiguousarray(arrays["normal_norms"]),
        K=int(bundle.configuration["iterations"]),
        verbose=False,
        nhistory=int(bundle.configuration["iterations"]),
        single_direction=int(bundle.configuration["single_direction"]),
    )
    final = np.asarray(final, dtype=np.float64)
    selected = np.flatnonzero(np.linalg.norm(final, axis=1)).astype(np.int64)
    initial_state = _state("initial", response, target, initial)
    final_state = _state("final", response, target, final)
    success = bool(
        selected.size == int(bundle.configuration["iterations"])
        and final_state["final:objective_sum_squares"]
        < initial_state["initial:objective_sum_squares"]
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="simsoptpp_gpmo_baseline",
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=int(bundle.configuration["iterations"]),
        nfev=int(bundle.configuration["iterations"]),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_state,
            "initial:selected_dipoles": np.empty(0, dtype=np.int64),
            **final_state,
            "final:selected_dipoles": selected,
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
    from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

    import jax
    import jax.numpy as jnp

    response = jnp.asarray(arrays["response"])
    target = jnp.asarray(arrays["target"])
    initial = jnp.asarray(arrays["initial_moments"])
    grid = PermanentMagnetGridJAX(
        A_obj=response,
        b_obj=target,
        ATb=jnp.reshape(response.T @ target, (2, 3)),
        ATA_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
        m0=initial,
        m=initial,
        m_proxy=initial,
        m_maxima=jnp.asarray(arrays["moment_maxima"]),
        dipole_grid_xyz=jnp.asarray(arrays["dipole_grid_xyz"]),
        coordinate_flag=str(bundle.configuration["coordinate_flag"]),
        R0=0.0,
        nfp=1,
        stellsym=False,
        nphi=1,
        ntheta=target.size,
        ndipoles=initial.shape[0],
    )
    result = GPMO_baseline_jax(
        grid,
        K=int(bundle.configuration["iterations"]),
        reg_l2=float(bundle.configuration["regularization_l2"]),
        single_direction=int(bundle.configuration["single_direction"]),
    )
    final = np.asarray(jax.device_get(jax.block_until_ready(result.m)))
    selected = np.asarray(
        jax.device_get(jax.block_until_ready(result.selected_dipoles)),
        dtype=np.int64,
    )
    initial_state = _state(
        "initial", arrays["response"], arrays["target"], arrays["initial_moments"]
    )
    final_state = _state("final", arrays["response"], arrays["target"], final)
    success = bool(
        selected.size == int(bundle.configuration["iterations"])
        and final_state["final:objective_sum_squares"]
        < initial_state["initial:objective_sum_squares"]
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
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="simsopt_jax_gpmo_baseline",
        normalized_status="converged" if success else "failed",
        raw_status="fixed_iteration_budget_complete",
        success=success,
        nit=int(bundle.configuration["iterations"]),
        nfev=int(bundle.configuration["iterations"]),
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_state,
            "initial:selected_dipoles": np.empty(0, dtype=np.int64),
            **final_state,
            "final:selected_dipoles": selected,
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched two-dipole workflow in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
