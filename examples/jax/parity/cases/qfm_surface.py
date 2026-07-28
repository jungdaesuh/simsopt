"""Matched native/JAX bounded NCSX QFM penalty workflow."""

from __future__ import annotations

import os
from pathlib import Path

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
    "construct_bounded_qfm_volume_problem",
    "evaluate_initial_qfm_residual_constraint_and_derivatives",
    "solve_qfm_penalty_problem",
    "evaluate_final_qfm_residual_and_feasibility",
)


def _ragged_snapshot(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flattened = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    offsets = np.cumsum([0, *(value.size for value in flattened)], dtype=np.int64)
    return np.concatenate(flattened), offsets


def _surface_from_state(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.geo import SurfaceRZFourier

    surface = SurfaceRZFourier(
        mpol=int(bundle.configuration["mpol"]),
        ntor=int(bundle.configuration["ntor"]),
        stellsym=bool(bundle.configuration["stellsym"]),
        nfp=int(bundle.configuration["nfp"]),
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
    )
    surface.x = arrays["initial_parameters"]
    return surface


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Snapshot NCSX coil state and one fitted bounded surface."""
    from simsopt.configs.zoo import get_data
    from simsopt.geo import Area, SurfaceRZFourier

    _, _, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    quadrature_phi = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    quadrature_theta = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    label = Area(surface)
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    return create_input_bundle(
        root,
        case_id="qfm-surface-optimization",
        random_seed=0,
        arrays={
            "initial_parameters": np.asarray(surface.x, dtype=np.float64),
            "quadrature_phi": quadrature_phi,
            "quadrature_theta": quadrature_theta,
            "coil_curve_dofs": coil_curve_dofs,
            "coil_curve_dof_offsets": coil_curve_dof_offsets,
            "coil_currents": coil_currents,
            "coil_current_offsets": coil_current_offsets,
        },
        configuration={
            "nfp": nfp,
            "mpol": 1,
            "ntor": 1,
            "stellsym": True,
            "label": "area",
            "target": 0.98 * float(label.J()),
            "constraint_weight": 12.0,
            "relative_objective_tol": 0.0,
            "gradient_tol": 5.0e-8,
            "max_steps": 200 if scale == "bounded" else 300,
        },
        scale=scale,
    )


def _problem_components(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.configs.zoo import get_data
    from simsopt.geo import Area

    _, _, _, _, biotsavart = get_data("ncsx")
    surface = _surface_from_state(bundle, arrays)
    label = Area(surface)
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    expected_snapshots = (
        ("coil_curve_dofs", coil_curve_dofs),
        ("coil_curve_dof_offsets", coil_curve_dof_offsets),
        ("coil_currents", coil_currents),
        ("coil_current_offsets", coil_current_offsets),
    )
    for name, actual in expected_snapshots:
        if not np.array_equal(arrays[name], actual):
            raise ValueError(f"effective construction did not consume {name}")
    payload = {
        "surface_dofs": np.asarray(surface.x).tolist(),
        "quadrature_phi": np.asarray(surface.quadpoints_phi).tolist(),
        "quadrature_theta": np.asarray(surface.quadpoints_theta).tolist(),
        "coil_curve_dofs": coil_curve_dofs.tolist(),
        "coil_curve_dof_offsets": coil_curve_dof_offsets.tolist(),
        "coil_currents": coil_currents.tolist(),
        "coil_current_offsets": coil_current_offsets.tolist(),
        "nfp": surface.nfp,
        "mpol": surface.mpol,
        "ntor": surface.ntor,
        "stellsym": surface.stellsym,
        "target": bundle.configuration["target"],
        "constraint_weight": bundle.configuration["constraint_weight"],
        "relative_objective_tol": bundle.configuration["relative_objective_tol"],
        "gradient_tol": bundle.configuration["gradient_tol"],
        "max_steps": bundle.configuration["max_steps"],
    }
    fingerprint = effective_construction_fingerprint(bundle, payload)
    return biotsavart, surface, label, fingerprint


def _state(
    qfm,
    prefix: str,
    parameters: np.ndarray,
    constraint_weight: float,
) -> dict[str, np.ndarray]:
    qfm_value, qfm_gradient = qfm.qfm_objective(parameters, derivatives=1)
    constraint_value, constraint_gradient = qfm.qfm_label_constraint(
        parameters, derivatives=1
    )
    penalty_value, penalty_gradient = qfm.qfm_penalty_constraints(
        parameters,
        derivatives=1,
        constraint_weight=constraint_weight,
    )
    return {
        f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
        f"{prefix}:qfm_objective": np.asarray(qfm_value, dtype=np.float64),
        f"{prefix}:qfm_gradient": np.asarray(qfm_gradient, dtype=np.float64),
        f"{prefix}:constraint_value": np.asarray(constraint_value, dtype=np.float64),
        f"{prefix}:constraint_gradient": np.asarray(
            constraint_gradient, dtype=np.float64
        ),
        f"{prefix}:penalty_objective": np.asarray(penalty_value, dtype=np.float64),
        f"{prefix}:penalty_gradient": np.asarray(penalty_gradient, dtype=np.float64),
    }


def _terminal_success(
    initial_state: dict[str, np.ndarray],
    final_state: dict[str, np.ndarray],
) -> bool:
    tolerance = parity_ladder_tolerances("native_workflow")
    return bool(
        final_state["final:penalty_objective"]
        < initial_state["initial:penalty_objective"]
        and np.max(np.abs(final_state["final:constraint_value"]))
        <= float(tolerance["terminal_constraint_norm_atol"])
        and np.linalg.norm(final_state["final:penalty_gradient"], ord=np.inf)
        <= float(tolerance["terminal_stationarity_atol"])
        and np.isfinite(final_state["final:qfm_objective"]).all()
        and np.isfinite(final_state["final:constraint_value"]).all()
    )


def _normalized_driver_status(*, driver_success: bool) -> str:
    """Normalize the optimizer termination without rewriting scientific status."""
    return "converged" if driver_success else "failed"


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.field import BiotSavart
    from simsopt.geo import QfmSurface
    from scipy.optimize import minimize

    biotsavart, surface, label, fingerprint = _problem_components(bundle, arrays)
    qfm = QfmSurface(
        BiotSavart(biotsavart.coils),
        surface,
        label,
        float(bundle.configuration["target"]),
    )
    initial = arrays["initial_parameters"]
    constraint_weight = float(bundle.configuration["constraint_weight"])
    initial_state = _state(qfm, "initial", initial, constraint_weight)

    def objective(parameters: np.ndarray):
        return qfm.qfm_penalty_constraints(
            parameters,
            derivatives=1,
            constraint_weight=constraint_weight,
        )

    result = minimize(
        objective,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": int(bundle.configuration["max_steps"]),
            "ftol": float(bundle.configuration["relative_objective_tol"]),
            "gtol": float(bundle.configuration["gradient_tol"]),
            "maxcor": 200,
        },
    )
    surface.x = result.x
    final = np.asarray(surface.x, dtype=np.float64)
    final_state = _state(qfm, "final", final, constraint_weight)
    success = _terminal_success(initial_state, final_state)
    driver_success = bool(result.success)
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="scipy_lbfgsb_qfm_penalty",
        normalized_status=_normalized_driver_status(driver_success=driver_success),
        raw_status=str(result.message),
        success=success,
        nit=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**initial_state, **final_state},
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.qfm_surface import QfmSurfaceJAX

    import jax

    biotsavart, surface, label, fingerprint = _problem_components(bundle, arrays)
    qfm = QfmSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        surface,
        label,
        float(bundle.configuration["target"]),
    )
    initial = arrays["initial_parameters"]
    constraint_weight = float(bundle.configuration["constraint_weight"])
    initial_state = _state(qfm, "initial", initial, constraint_weight)
    result = qfm.minimize_qfm_penalty_jax(
        tol=float(bundle.configuration["gradient_tol"]),
        maxiter=int(bundle.configuration["max_steps"]),
        constraint_weight=constraint_weight,
    )
    final = np.asarray(surface.x, dtype=np.float64)
    final_state = _state(qfm, "final", final, constraint_weight)
    success = _terminal_success(initial_state, final_state)
    driver_success = bool(result["success"])
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
        driver="simsopt_jax_bfgs_qfm_penalty",
        normalized_status=_normalized_driver_status(driver_success=driver_success),
        raw_status=str(result["status"]),
        success=success,
        nit=int(result["iter"]),
        nfev=int(result["nfev"]),
        njev=int(result["njev"]),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**initial_state, **final_state},
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the bounded QFM penalty workflow in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
