"""Exact matched workflow for ``2_Intermediate/strain_optimization.py``."""

from __future__ import annotations

import hashlib
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
    "construct_scaled_fixed_hsx_coil",
    "construct_centroid_frame_rotation",
    "evaluate_initial_strain_objective_and_gradient",
    "optimize_rotation_dofs_with_lbfgs",
    "evaluate_final_strains_objective_and_gradient",
)
SCIENTIFIC_GRADIENT_TOLERANCE = 1.0e-8


def _configuration(scale: ExecutionScale) -> dict[str, object]:
    return {
        "coil_configuration": "hsx",
        "coil_index": 1,
        "coil_order": 10,
        "points_per_period": 10,
        "scale_factor": 0.1,
        "rotation_order": 10,
        "objective_width": 1.0e-3,
        "reporting_width": 3.0e-3,
        "torsional_threshold": 2.0e-3,
        "curvature_threshold": 2.0e-3,
        "maxiter": 400 if scale == "native_default" else 50,
        "maxfun": 15000,
        "maxcor": 10,
        "maxls": 20,
        "gtol": 1.0e-20,
        "ftol": 1.0e-20,
    }


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


def _fixed_geometry(configuration: dict[str, object]):
    from simsopt.configs import get_data
    from simsopt.geo import CurveXYZFourier

    source_curves, _currents, _axis, _nfp, _field = get_data(
        str(configuration["coil_configuration"]),
        coil_order=int(configuration["coil_order"]),
        points_per_period=int(configuration["points_per_period"]),
    )
    source_curve = source_curves[int(configuration["coil_index"])]
    curve = CurveXYZFourier(source_curve.quadpoints, source_curve.order)
    curve.x = (
        np.asarray(source_curve.x, dtype=np.float64)
        * float(configuration["scale_factor"])
    )
    curve.fix_all()
    return curve


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the scaled HSX curve and initial rotation state."""
    configuration = _configuration(scale)
    curve = _fixed_geometry(configuration)
    rotation_size = 2 * int(configuration["rotation_order"]) + 1
    return create_input_bundle(
        root,
        case_id="native-strain-optimization",
        random_seed=0,
        arrays={
            "quadpoints": np.array(curve.quadpoints, dtype=np.float64, copy=True),
            "gamma": np.array(curve.gamma(), dtype=np.float64, copy=True),
            "gammadash": np.array(curve.gammadash(), dtype=np.float64, copy=True),
            "gammadashdash": np.array(
                curve.gammadashdash(),
                dtype=np.float64,
                copy=True,
            ),
            "initial_parameters": np.zeros(rotation_size, dtype=np.float64),
        },
        configuration=configuration,
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "arrays": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "sha256": _array_digest(value),
                }
                for name, value in sorted(arrays.items())
            },
            **bundle.configuration,
        },
    )


def _construction_values(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"construction:{name}": arrays[name]
        for name in ("quadpoints", "gamma", "gammadash", "gammadashdash")
    }


def _scientific_success(
    initial_state: dict[str, np.ndarray],
    final_state: dict[str, np.ndarray],
) -> bool:
    final_gradient = final_state["final:gradient"]
    return bool(
        np.isfinite(final_state["final:objective"])
        and final_state["final:objective"] < initial_state["initial:objective"]
        and np.all(np.isfinite(final_gradient))
        and np.linalg.norm(final_gradient, ord=np.inf)
        <= SCIENTIFIC_GRADIENT_TOLERANCE
    )


def _native_problem(bundle: InputBundle):
    from simsopt.geo import (
        CoilStrain,
        FrameRotation,
        FramedCurveCentroid,
        LPBinormalCurvatureStrainPenalty,
        LPTorsionalStrainPenalty,
    )

    curve = _fixed_geometry(dict(bundle.configuration))
    rotation = FrameRotation(
        curve.quadpoints,
        _configuration_int(bundle, "rotation_order"),
    )
    framed_curve = FramedCurveCentroid(curve, rotation)
    strain = CoilStrain(
        framed_curve,
        _configuration_float(bundle, "reporting_width"),
    )
    objective = LPTorsionalStrainPenalty(
        framed_curve,
        width=_configuration_float(bundle, "objective_width"),
        p=2,
        threshold=_configuration_float(bundle, "torsional_threshold"),
    ) + LPBinormalCurvatureStrainPenalty(
        framed_curve,
        width=_configuration_float(bundle, "objective_width"),
        p=2,
        threshold=_configuration_float(bundle, "curvature_threshold"),
    )
    return objective, strain


def _native_state(
    prefix: str,
    *,
    parameters: np.ndarray,
    objective,
    strain,
) -> dict[str, np.ndarray]:
    objective.x = parameters
    torsional = np.asarray(strain.torsional_strain(), dtype=np.float64)
    binormal = np.asarray(
        strain.binormal_curvature_strain(),
        dtype=np.float64,
    )
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:objective": np.asarray(objective.J(), dtype=np.float64),
        f"{prefix}:gradient": np.asarray(objective.dJ(), dtype=np.float64),
        f"{prefix}:torsional_strain": torsional,
        f"{prefix}:binormal_curvature_strain": binormal,
        f"{prefix}:maximum_torsional_strain": np.asarray(
            np.max(torsional),
            dtype=np.float64,
        ),
        f"{prefix}:maximum_binormal_curvature_strain": np.asarray(
            np.max(binormal),
            dtype=np.float64,
        ),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    import jax
    from scipy.optimize import minimize

    jax.config.update("jax_enable_x64", True)
    objective, strain = _native_problem(bundle)
    initial = arrays["initial_parameters"]
    initial_state = _native_state(
        "initial",
        parameters=initial,
        objective=objective,
        strain=strain,
    )

    def value_and_gradient(parameters: np.ndarray):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = minimize(
        value_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        tol=_configuration_float(bundle, "gtol"),
        options={
            "maxiter": _configuration_int(bundle, "maxiter"),
            "maxfun": _configuration_int(bundle, "maxfun"),
            "maxcor": _configuration_int(bundle, "maxcor"),
            "maxls": _configuration_int(bundle, "maxls"),
            "gtol": _configuration_float(bundle, "gtol"),
            "ftol": _configuration_float(bundle, "ftol"),
        },
    )
    final_state = _native_state(
        "final",
        parameters=np.asarray(result.x, dtype=np.float64),
        objective=objective,
        strain=strain,
    )
    success = _scientific_success(initial_state, final_state)
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="scipy_lbfgsb_native_strain_objective",
        normalized_status="converged" if success else "failed",
        raw_status=str(result.message),
        success=success,
        nit=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **_construction_values(arrays),
            **initial_state,
            **final_state,
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax

    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.examples import solve_strain_rotation

    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    device_result = solve_strain_rotation(
        quadpoints=put("quadpoints"),
        gamma=put("gamma"),
        gammadash=put("gammadash"),
        gammadashdash=put("gammadashdash"),
        initial_parameters=put("initial_parameters"),
        rotation_order=_configuration_int(bundle, "rotation_order"),
        objective_width=_configuration_float(bundle, "objective_width"),
        reporting_width=_configuration_float(bundle, "reporting_width"),
        torsional_threshold=_configuration_float(bundle, "torsional_threshold"),
        curvature_threshold=_configuration_float(bundle, "curvature_threshold"),
        maxiter=_configuration_int(bundle, "maxiter"),
        maxfun=_configuration_int(bundle, "maxfun"),
        gtol=_configuration_float(bundle, "gtol"),
        ftol=_configuration_float(bundle, "ftol"),
        maxcor=_configuration_int(bundle, "maxcor"),
        maxls=_configuration_int(bundle, "maxls"),
    )
    result = jax.device_get(device_result)

    def state_values(prefix: str, state) -> dict[str, np.ndarray]:
        return {
            f"{prefix}:parameters": np.asarray(state.parameters, dtype=np.float64),
            f"{prefix}:objective": np.asarray(state.objective, dtype=np.float64),
            f"{prefix}:gradient": np.asarray(state.gradient, dtype=np.float64),
            f"{prefix}:torsional_strain": np.asarray(
                state.torsional_strain,
                dtype=np.float64,
            ),
            f"{prefix}:binormal_curvature_strain": np.asarray(
                state.binormal_curvature_strain,
                dtype=np.float64,
            ),
            f"{prefix}:maximum_torsional_strain": np.asarray(
                state.maximum_torsional_strain,
                dtype=np.float64,
            ),
            f"{prefix}:maximum_binormal_curvature_strain": np.asarray(
                state.maximum_binormal_curvature_strain,
                dtype=np.float64,
            ),
        }

    values = {
        **_construction_values(arrays),
        **state_values("initial", result.initial),
        **state_values("final", result.final),
    }
    success = _scientific_success(
        {
            name: value
            for name, value in values.items()
            if name.startswith("initial:")
        },
        {
            name: value
            for name, value in values.items()
            if name.startswith("final:")
        },
    )
    platform = "cpu" if device is None else device.platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="simsopt_jax_lbfgsb_strain_objective",
        normalized_status="converged" if success else "failed",
        raw_status=f"status_{int(result.status)}",
        success=success,
        nit=int(result.iterations),
        nfev=int(result.function_evaluations),
        njev=int(result.gradient_evaluations),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact strain workflow in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
