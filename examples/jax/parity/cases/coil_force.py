"""Matched native/JAX bounded fixed-state force and frame workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import numpy as np
from simsopt_jax.examples import ExecutionScale
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane

WORKFLOW_STAGES = (
    "construct_bounded_fixed_coil_state",
    "evaluate_force_and_finite_build_objectives",
    "evaluate_force_and_frame_derivatives",
    "publish_fixed_state_diagnostics",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Create two concentric regularized coils and a zero-rotation frame."""
    unit_circle_dofs = np.asarray(
        (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        dtype=np.float64,
    )
    return create_input_bundle(
        root,
        case_id="coil-force-and-finite-build",
        random_seed=0,
        arrays={
            "target_curve_dofs": unit_circle_dofs,
            "source_curve_dofs": unit_circle_dofs,
        },
        configuration={
            "target_radius": 1.0,
            "source_radius": 1.3,
            "quadrature_points": 32,
            "current": 1.7e4,
            "regularization": 0.05**2 / np.sqrt(np.e),
            "force_power": 2.0,
            "force_threshold": 0.0,
        },
        scale=scale,
    )


def _build_coils(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.field import Current, RegularizedCoil
    from simsopt.geo import CurveXYZFourier

    quadrature_points = int(bundle.configuration["quadrature_points"])
    target_curve = CurveXYZFourier(quadrature_points, order=1)
    target_curve.x = (
        float(bundle.configuration["target_radius"]) * arrays["target_curve_dofs"]
    )
    source_curve = CurveXYZFourier(quadrature_points, order=1)
    source_curve.x = (
        float(bundle.configuration["source_radius"]) * arrays["source_curve_dofs"]
    )
    current = float(bundle.configuration["current"])
    regularization = float(bundle.configuration["regularization"])
    target = RegularizedCoil(target_curve, Current(current), regularization)
    source = RegularizedCoil(source_curve, Current(-current), regularization)
    return target, source


def _effective_fingerprint(bundle: InputBundle, target, source) -> str:
    payload = {
        "target_curve_dofs": np.asarray(target.curve.local_full_x).tolist(),
        "source_curve_dofs": np.asarray(source.curve.local_full_x).tolist(),
        "target_current": np.asarray(target.current.local_full_x).tolist(),
        "source_current": np.asarray(source.current.local_full_x).tolist(),
        "target_quadpoints": np.asarray(target.curve.quadpoints).tolist(),
        "source_quadpoints": np.asarray(source.curve.quadpoints).tolist(),
        "regularization": bundle.configuration["regularization"],
        "force_power": bundle.configuration["force_power"],
        "force_threshold": bundle.configuration["force_threshold"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _frame_values(
    frame,
    host_array: Callable[[object], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tangent, normal, binormal = frame.rotated_frame()
    frame_array = np.stack(
        (
            host_array(tangent),
            host_array(normal),
            host_array(binormal),
        ),
        axis=1,
    )
    gram = frame_array @ np.swapaxes(frame_array, 1, 2)
    residual = gram - np.broadcast_to(np.eye(3), gram.shape)
    torsion = host_array(frame.frame_torsion())
    return frame_array, residual, torsion


def _observation(
    *,
    lane: ParityLane,
    bundle: InputBundle,
    fingerprint: str,
    objective,
    frame,
    backend_mode: str,
    platform: str,
    precision: str,
    driver: str,
    host_array: Callable[[object], np.ndarray],
) -> LaneObservation:
    parameters = host_array(objective.x)
    frame_array, orthonormality_residual, torsion = _frame_values(frame, host_array)
    state = {
        "parameters": parameters,
        "force_objective": host_array(objective.J()),
        "force_gradient": host_array(objective.dJ()),
        "frame": frame_array,
        "frame_orthonormality_residual": orthonormality_residual,
        "torsion": torsion,
    }
    values = {
        f"{phase}:{name}": value
        for phase in ("initial", "final")
        for name, value in state.items()
    }
    tolerance = parity_ladder_tolerances("native_workflow")
    success = bool(
        np.isfinite(state["force_objective"]).all()
        and np.isfinite(state["force_gradient"]).all()
        and np.max(np.abs(orthonormality_residual))
        <= float(tolerance["terminal_orthonormality_atol"])
        and np.max(np.abs(torsion)) <= float(tolerance["terminal_orthonormality_atol"])
    )
    return LaneObservation(
        lane=lane,
        backend_mode=backend_mode,
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver=driver,
        normalized_status="not_applicable" if success else "failed",
        raw_status="fixed_state_evaluation_complete",
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.field import LpCurveForce
    from simsopt.geo import FramedCurveFrenet, ZeroRotation

    target, source = _build_coils(bundle, arrays)
    objective = LpCurveForce(
        target,
        [target, source],
        p=float(bundle.configuration["force_power"]),
        threshold=float(bundle.configuration["force_threshold"]),
    )
    frame = FramedCurveFrenet(target.curve, ZeroRotation(target.curve.quadpoints))
    return _observation(
        lane="native-cpu",
        bundle=bundle,
        fingerprint=_effective_fingerprint(bundle, target, source),
        objective=objective,
        frame=frame,
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        driver="native_fixed_state_force_and_frenet",
        host_array=lambda value: np.asarray(value, dtype=np.float64),
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax_adapters.field.force import LpCurveForce
    from simsopt_jax_adapters.geo.framed_curve import (
        FramedCurveFrenetJAX,
        ZeroRotationJAX,
    )

    import jax

    target, source = _build_coils(bundle, arrays)
    objective = LpCurveForce(
        target,
        [target, source],
        p=float(bundle.configuration["force_power"]),
        threshold=float(bundle.configuration["force_threshold"]),
    )
    frame = FramedCurveFrenetJAX(target.curve, ZeroRotationJAX(target.curve.quadpoints))

    def publish_array(value: object) -> np.ndarray:
        with jax.transfer_guard("allow"):
            ready = jax.block_until_ready(value)
            return np.asarray(jax.device_get(ready), dtype=np.float64)

    platform = jax.devices()[0].platform
    return _observation(
        lane=lane,
        bundle=bundle,
        fingerprint=_effective_fingerprint(bundle, target, source),
        objective=objective,
        frame=frame,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
        driver="jax_fixed_state_force_and_frenet",
        host_array=publish_array,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Evaluate the matched fixed coil state in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
