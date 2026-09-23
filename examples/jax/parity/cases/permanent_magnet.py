"""Matched native-C++/JAX bounded greedy permanent-magnet workflow."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._fixed_work_status import (
    LaneTerminalLabel,
    fixed_work_label,
    gpmo_stop_reason,
)
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax_adapters.examples.gpmo_rules import gpmo_baseline_outputs_usable
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


def _label(
    bundle: InputBundle,
    final_moments: np.ndarray,
    selected: np.ndarray,
) -> LaneTerminalLabel:
    """Label one GPMO_baseline lane from the magnets it actually returned.

    ``GPMO_baseline`` (``permanent_magnet_optimization.cpp:1280-1325``) has no
    cap and no early exit: it always runs its ``K`` iterations, so the stop
    reason is ``iteration_count_completed`` unless every dipole in the grid is
    populated. A selected site is blanked in ``R2s`` and in
    ``Gamma_complement`` immediately, so the ``K`` sites are distinct and
    exactly ``K`` rows are nonzero at the end -- the equality is the invariant,
    and a short count means the lane did not run this algorithm. It reports no
    convergence status, so the category is ``not_applicable`` and the optimizer
    counters are null.
    """
    return fixed_work_label(
        raw_status=gpmo_stop_reason(
            nonzero_count=int(selected.size),
            grid_size=None,
            magnet_cap=None,
        ),
        outputs_usable=gpmo_baseline_outputs_usable(
            moments=final_moments,
            nonzero_count=int(selected.size),
            iterations=int(bundle.configuration["iterations"]),
        ),
    )


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
    label = _label(bundle, final, selected)
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
        normalized_status=label.normalized_status,
        raw_status=label.raw_status,
        success=label.success,
        nit=None,
        nfev=None,
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
    label = _label(bundle, final, selected)
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
        normalized_status=label.normalized_status,
        raw_status=label.raw_status,
        success=label.success,
        nit=None,
        nfev=None,
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
