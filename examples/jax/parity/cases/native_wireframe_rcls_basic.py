"""Exact matched workflow for ``2_Intermediate/wireframe_rcls_basic.py``."""

from __future__ import annotations

import hashlib
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

TEST_DATA = Path(__file__).resolve().parents[4] / "tests" / "test_files"
SURFACE_INPUT = TEST_DATA / "input.LandremanPaul2021_QA"

WORKFLOW_STAGES = (
    "construct_landreman_paul_qa_plasma_and_offset_wireframe",
    "apply_poloidal_current_constraint",
    "construct_area_weighted_normal_field_response",
    "evaluate_initial_feasible_currents_and_objectives",
    "solve_regularized_constrained_least_squares",
    "evaluate_final_field_objectives_constraints_and_current",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "plasma_nphi": 32 if native_scale else 16,
        "plasma_ntheta": 32 if native_scale else 16,
        "wireframe_nphi": 8 if native_scale else 4,
        "wireframe_ntheta": 12 if native_scale else 6,
        "wireframe_surface_distance": 0.3,
        "field_on_axis": 1.0,
        "regularization_weight": 1.0e-10,
        "assume_no_crossings": False,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
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


def _build_geometry(configuration: Mapping[str, object]):
    from simsopt.geo import SurfaceRZFourier, ToroidalWireframe

    plasma_nphi = configuration["plasma_nphi"]
    plasma_ntheta = configuration["plasma_ntheta"]
    wireframe_nphi = configuration["wireframe_nphi"]
    wireframe_ntheta = configuration["wireframe_ntheta"]
    assert isinstance(plasma_nphi, int)
    assert isinstance(plasma_ntheta, int)
    assert isinstance(wireframe_nphi, int)
    assert isinstance(wireframe_ntheta, int)
    plasma_surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        nphi=plasma_nphi,
        ntheta=plasma_ntheta,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(str(SURFACE_INPUT))
    wireframe_surface.extend_via_projected_normal(
        float(configuration["wireframe_surface_distance"])
    )
    wireframe = ToroidalWireframe(
        wireframe_surface,
        wireframe_nphi,
        wireframe_ntheta,
    )
    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = (
        -2.0
        * np.pi
        * plasma_surface.get_rc(0, 0)
        * float(configuration["field_on_axis"])
        / mu0
    )
    wireframe.set_poloidal_current(poloidal_current)
    return plasma_surface, wireframe


def _minimum_norm_feasible_currents(
    wireframe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    constraint, target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    constraint_array = np.asarray(constraint, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64).reshape((-1, 1))
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    gram = constraint_array @ constraint_array.T
    free_currents = constraint_array.T @ np.linalg.solve(gram, target_array)
    currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    currents[free_segments] = free_currents
    return currents, constraint_array, target_array, free_segments


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the exact geometry, response, constraints, and initial state."""
    from simsopt.solve.wireframe_optimization import bnorm_obj_matrices

    configuration = _scale_configuration(scale)
    plasma_surface, wireframe = _build_geometry(configuration)
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma_surface,
        area_weighted=True,
        verbose=False,
    )
    initial_currents, constraint, constraint_target, free_segments = (
        _minimum_norm_feasible_currents(wireframe)
    )
    normal = np.asarray(plasma_surface.normal(), dtype=np.float64)
    return create_input_bundle(
        root,
        case_id="native-wireframe-rcls-basic",
        random_seed=0,
        arrays={
            "response_matrix": np.array(response, dtype=np.float64, copy=True),
            "target": np.array(target, dtype=np.float64, copy=True),
            "constraint_matrix": np.array(constraint, dtype=np.float64, copy=True),
            "constraint_target": np.array(
                constraint_target, dtype=np.float64, copy=True
            ),
            "free_segments": np.array(free_segments, dtype=np.int64, copy=True),
            "initial_currents": np.array(initial_currents, dtype=np.float64, copy=True),
            "plasma_points": np.array(
                plasma_surface.gamma().reshape((-1, 3)),
                dtype=np.float64,
                copy=True,
            ),
            "plasma_unit_normal": np.array(
                plasma_surface.unitnormal().reshape((-1, 3)),
                dtype=np.float64,
                copy=True,
            ),
            "plasma_area_weights": np.array(
                np.linalg.norm(normal, axis=2).reshape(-1)
                / normal.shape[0]
                / normal.shape[1],
                dtype=np.float64,
                copy=True,
            ),
            "wireframe_nodes": np.array(
                np.stack(wireframe.nodes), dtype=np.float64, copy=True
            ),
            "wireframe_segments": np.array(
                wireframe.segments, dtype=np.int32, copy=True
            ),
            "wireframe_segment_signs": np.array(
                wireframe.seg_signs, dtype=np.float64, copy=True
            ),
        },
        configuration={
            **configuration,
            "poloidal_current": float(
                -2.0
                * np.pi
                * plasma_surface.get_rc(0, 0)
                * float(configuration["field_on_axis"])
                / (4.0 * np.pi * 1.0e-7)
            ),
            "n_segments": int(wireframe.n_segments),
            "degrees_of_freedom": int(wireframe.n_segments - constraint.shape[0]),
        },
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


def _state(
    prefix: str,
    *,
    arrays: dict[str, np.ndarray],
    currents: np.ndarray,
    regularization: float,
) -> dict[str, np.ndarray]:
    residual = arrays["response_matrix"] @ currents - arrays["target"]
    constraint_residual = (
        arrays["constraint_matrix"] @ currents[arrays["free_segments"]]
        - arrays["constraint_target"]
    )
    normal_objective = 0.5 * np.vdot(residual, residual)
    regularization_objective = 0.5 * regularization**2 * np.vdot(currents, currents)
    return {
        f"{prefix}:currents": currents,
        f"{prefix}:normal_field_residual": residual,
        f"{prefix}:normal_objective": np.asarray(normal_objective, dtype=np.float64),
        f"{prefix}:regularization_objective": np.asarray(
            regularization_objective, dtype=np.float64
        ),
        f"{prefix}:total_objective": np.asarray(
            normal_objective + regularization_objective,
            dtype=np.float64,
        ),
        f"{prefix}:constraint_residual": constraint_residual,
    }


def _construction_values(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        f"construction:{name}": arrays[name]
        for name in (
            "response_matrix",
            "target",
            "constraint_matrix",
            "constraint_target",
            "free_segments",
            "plasma_points",
            "wireframe_nodes",
            "wireframe_segments",
            "wireframe_segment_signs",
            "plasma_unit_normal",
            "plasma_area_weights",
        )
    }


def _field_values(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    currents: np.ndarray,
    magnetic_field: np.ndarray,
) -> dict[str, np.ndarray]:
    normal_field = np.sum(
        magnetic_field * arrays["plasma_unit_normal"],
        axis=1,
    )
    field_magnitude = np.linalg.norm(magnetic_field, axis=1)
    area = arrays["plasma_area_weights"]
    mean_relative_normal_field = np.sum(
        np.abs(normal_field / field_magnitude) * area
    ) / np.sum(area)
    return {
        "final:magnetic_field": magnetic_field,
        "final:normal_field": normal_field,
        "final:mean_relative_normal_field": np.asarray(
            mean_relative_normal_field,
            dtype=np.float64,
        ),
        "final:maximum_current": np.asarray(
            np.max(np.abs(currents)),
            dtype=np.float64,
        ),
        "final:degrees_of_freedom": np.asarray(
            _configuration_int(bundle, "degrees_of_freedom"),
            dtype=np.int64,
        ),
    }


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry=_build_geometry,
) -> LaneObservation:
    from simsopt.field import WireframeField
    from simsopt.solve.wireframe_optimization import rcls_wireframe

    _, wireframe = build_geometry(bundle.configuration)
    regularization = _configuration_float(bundle, "regularization_weight")
    final, _, _, _ = rcls_wireframe(
        wireframe,
        arrays["response_matrix"],
        arrays["target"],
        regularization,
        bool(bundle.configuration["assume_no_crossings"]),
        False,
    )
    final_currents = np.asarray(final, dtype=np.float64)
    wireframe.currents[:] = final_currents.reshape(-1)
    field = WireframeField(wireframe)
    field.set_points(arrays["plasma_points"])
    magnetic_field = np.asarray(field.B(), dtype=np.float64)
    values = {
        **_construction_values(arrays),
        **_state(
            "initial",
            arrays=arrays,
            currents=arrays["initial_currents"],
            regularization=regularization,
        ),
        **_state(
            "final",
            arrays=arrays,
            currents=final_currents,
            regularization=regularization,
        ),
        **_field_values(bundle, arrays, final_currents, magnetic_field),
    }
    constraint_scale = max(
        1.0,
        float(np.linalg.norm(arrays["constraint_target"], ord=np.inf)),
    )
    success = bool(
        np.all(np.isfinite(final_currents))
        and np.linalg.norm(values["final:constraint_residual"], ord=np.inf)
        <= 1.0e-11 * constraint_scale
        and values["final:normal_objective"] < values["initial:normal_objective"]
        and wireframe.check_constraints()
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
        driver="simsopt_wireframe_rcls_and_field_postprocessing",
        normalized_status="converged" if success else "failed",
        raw_status="direct_rcls_complete",
        success=success,
        nit=1,
        nfev=1,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry=_build_geometry,
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.examples import solve_wireframe_rcls

    import jax

    _, wireframe = build_geometry(bundle.configuration)
    regularization = _configuration_float(bundle, "regularization_weight")
    device = get_runtime_jax_device()

    def put(name: str):
        return jax.device_put(arrays[name], device)

    device_result = solve_wireframe_rcls(
        wireframe=wireframe,
        response=put("response_matrix"),
        target=put("target"),
        regularization=regularization,
        initial_currents=put("initial_currents"),
        plasma_points=put("plasma_points"),
        plasma_unit_normal=put("plasma_unit_normal"),
        plasma_area_weights=put("plasma_area_weights"),
        wireframe_nodes=put("wireframe_nodes"),
        wireframe_segments=put("wireframe_segments"),
        wireframe_segment_signs=put("wireframe_segment_signs"),
        assume_no_crossings=bool(bundle.configuration["assume_no_crossings"]),
    )
    result = jax.device_get(device_result)

    def published_state(prefix: str, state) -> dict[str, np.ndarray]:
        return {
            f"{prefix}:currents": np.asarray(state.currents, dtype=np.float64),
            f"{prefix}:normal_field_residual": np.asarray(
                state.normal_field_residual,
                dtype=np.float64,
            ),
            f"{prefix}:normal_objective": np.asarray(
                state.normal_objective,
                dtype=np.float64,
            ),
            f"{prefix}:regularization_objective": np.asarray(
                state.regularization_objective,
                dtype=np.float64,
            ),
            f"{prefix}:total_objective": np.asarray(
                state.total_objective,
                dtype=np.float64,
            ),
            f"{prefix}:constraint_residual": np.asarray(
                state.constraint_residual,
                dtype=np.float64,
            ),
        }

    values = {
        **_construction_values(arrays),
        **published_state("initial", result.initial),
        **published_state("final", result.final),
        "final:magnetic_field": np.asarray(
            result.magnetic_field,
            dtype=np.float64,
        ),
        "final:normal_field": np.asarray(result.normal_field, dtype=np.float64),
        "final:mean_relative_normal_field": np.asarray(
            result.mean_relative_normal_field,
            dtype=np.float64,
        ),
        "final:maximum_current": np.asarray(
            result.maximum_current,
            dtype=np.float64,
        ),
        "final:degrees_of_freedom": np.asarray(
            _configuration_int(bundle, "degrees_of_freedom"),
            dtype=np.int64,
        ),
    }
    constraint_scale = max(
        1.0,
        float(np.linalg.norm(arrays["constraint_target"], ord=np.inf)),
    )
    success = bool(
        result.finite_currents
        and np.linalg.norm(values["final:constraint_residual"], ord=np.inf)
        <= 1.0e-11 * constraint_scale
        and values["final:normal_objective"] < values["initial:normal_objective"]
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
        driver="simsopt_jax_wireframe_rcls_and_field_postprocessing",
        normalized_status="converged" if success else "failed",
        raw_status="direct_rcls_complete",
        success=success,
        nit=1,
        nfev=1,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact wireframe RCLS workflow in the selected lane."""
    return execute_problem(lane, bundle, arrays, _build_geometry)


def execute_problem(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    build_geometry,
) -> LaneObservation:
    """Execute an RCLS source workflow with its exact geometry constructor."""
    if lane == "native-cpu":
        return _native(bundle, arrays, build_geometry)
    return _jax(lane, bundle, arrays, build_geometry)
