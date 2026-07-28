"""Exact matched workflow for ``1_Simple/qfm.py``."""

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
    "construct_fitted_ncsx_qfm_surface",
    "evaluate_initial_qfm_state",
    "capture_volume_target",
    "solve_volume_penalty",
    "solve_volume_exact_constraint",
    "capture_toroidal_flux_target",
    "solve_toroidal_flux_penalty",
    "solve_toroidal_flux_exact_constraint",
    "check_volume_after_toroidal_flux",
    "capture_area_target",
    "solve_area_penalty",
    "solve_area_exact_constraint",
    "check_volume_after_area",
    "publish_final_qfm_state",
)

_LABELS = ("volume", "toroidal_flux", "area")


def _ragged_snapshot(values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flattened = [np.asarray(value, dtype=np.float64).reshape(-1) for value in values]
    offsets = np.cumsum([0, *(value.size for value in flattened)], dtype=np.int64)
    return np.concatenate(flattened), offsets


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Snapshot the fitted bounded NCSX surface and fixed coil construction."""
    from simsopt.configs.zoo import get_data
    from simsopt.geo import SurfaceRZFourier

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
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    return create_input_bundle(
        root,
        case_id="native-qfm",
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
            "constraint_weight": 1.0,
            "penalty_tolerance": 1.0e-12,
            "native_exact_tolerance": 1.0e-14,
            "jax_exact_tolerance": 1.0e-8,
            "max_steps": 80 if scale == "bounded" else 1000,
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


def _surface_from_state(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.geo import SurfaceRZFourier

    surface = SurfaceRZFourier(
        mpol=_configuration_int(bundle, "mpol"),
        ntor=_configuration_int(bundle, "ntor"),
        stellsym=bool(bundle.configuration["stellsym"]),
        nfp=_configuration_int(bundle, "nfp"),
        quadpoints_phi=arrays["quadrature_phi"],
        quadpoints_theta=arrays["quadrature_theta"],
    )
    surface.x = arrays["initial_parameters"]
    return surface


def _problem_components(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.configs.zoo import get_data

    _, _, _, _, biotsavart = get_data("ncsx")
    surface = _surface_from_state(bundle, arrays)
    coil_curve_dofs, coil_curve_dof_offsets = _ragged_snapshot(
        [np.asarray(coil.curve.local_full_x) for coil in biotsavart.coils]
    )
    coil_currents, coil_current_offsets = _ragged_snapshot(
        [np.asarray(coil.current.local_full_x) for coil in biotsavart.coils]
    )
    snapshots = (
        ("coil_curve_dofs", coil_curve_dofs),
        ("coil_curve_dof_offsets", coil_curve_dof_offsets),
        ("coil_currents", coil_currents),
        ("coil_current_offsets", coil_current_offsets),
    )
    for name, actual in snapshots:
        if not np.array_equal(arrays[name], actual):
            raise ValueError(f"effective construction did not consume {name}")
    fingerprint = effective_construction_fingerprint(
        bundle,
        {
            "surface_dofs": np.asarray(surface.x).tolist(),
            "quadrature_phi": np.asarray(surface.quadpoints_phi).tolist(),
            "quadrature_theta": np.asarray(surface.quadpoints_theta).tolist(),
            "coil_curve_dofs": coil_curve_dofs.tolist(),
            "coil_curve_dof_offsets": coil_curve_dof_offsets.tolist(),
            "coil_currents": coil_currents.tolist(),
            "coil_current_offsets": coil_current_offsets.tolist(),
            **dict(bundle.configuration),
        },
    )
    return biotsavart, surface, fingerprint


def _native_state(qfm_surface, label, target: float) -> dict[str, np.ndarray]:
    parameters = np.array(qfm_surface.surface.x, dtype=np.float64, copy=True)
    qfm_value, qfm_gradient = qfm_surface.qfm_objective(parameters, derivatives=1)
    label_value = float(label.J())
    return {
        "parameters": parameters,
        "qfm_value": np.asarray(qfm_value, dtype=np.float64),
        "qfm_gradient": np.array(qfm_gradient, dtype=np.float64, copy=True),
        "label_value": np.asarray(label_value, dtype=np.float64),
        "label_gradient": np.array(
            label.dJ_by_dsurfacecoefficients(),
            dtype=np.float64,
            copy=True,
        ),
        "label_residual_abs": np.asarray(
            abs(label_value - target),
            dtype=np.float64,
        ),
    }


def _prefixed(
    stage: str,
    phase: str,
    state: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {f"{stage}:{phase}:{name}": value for name, value in state.items()}


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.field import BiotSavart
    from simsopt.geo import Area, QfmSurface, ToroidalFlux, Volume

    biotsavart, surface, fingerprint = _problem_components(bundle, arrays)
    toroidal_field = BiotSavart(biotsavart.coils)
    initial_volume = float(Volume(surface).J())
    values: dict[str, np.ndarray] = {}
    success = True
    total_nit = 0
    total_nfev = 0
    total_njev = 0
    raw_statuses: list[str] = []

    for stage_name in _LABELS:
        if stage_name == "volume":
            label = Volume(surface)
        elif stage_name == "toroidal_flux":
            label = ToroidalFlux(surface, toroidal_field)
        else:
            label = Area(surface)
        target = float(label.J())
        qfm_surface = QfmSurface(biotsavart, surface, label, target)
        initial = _native_state(qfm_surface, label, target)
        if stage_name == "volume":
            values.update(
                {
                    "initial:parameters": initial["parameters"],
                    "initial:qfm_value": initial["qfm_value"],
                    "initial:qfm_gradient": initial["qfm_gradient"],
                }
            )
        values[f"{stage_name}:target"] = np.asarray(target, dtype=np.float64)
        values.update(_prefixed(stage_name, "initial", initial))

        penalty = qfm_surface.minimize_qfm_penalty_constraints_LBFGS(
            tol=_configuration_float(bundle, "penalty_tolerance"),
            maxiter=_configuration_int(bundle, "max_steps"),
            constraint_weight=_configuration_float(bundle, "constraint_weight"),
        )
        values.update(
            _prefixed(
                stage_name,
                "penalty",
                _native_state(qfm_surface, label, target),
            )
        )
        exact = qfm_surface.minimize_qfm_exact_constraints_SLSQP(
            tol=_configuration_float(bundle, "native_exact_tolerance"),
            maxiter=_configuration_int(bundle, "max_steps"),
        )
        exact_state = _native_state(qfm_surface, label, target)
        values.update(_prefixed(stage_name, "exact", exact_state))
        volume_residual = float(Volume(surface).J()) - initial_volume
        values[f"{stage_name}:volume_persistence_objective"] = np.asarray(
            0.5 * volume_residual * volume_residual,
            dtype=np.float64,
        )

        penalty_info = penalty["info"]
        exact_info = exact["info"]
        total_nit += int(penalty["iter"]) + int(exact["iter"])
        total_nfev += int(penalty_info.nfev) + int(exact_info.nfev)
        total_njev += int(penalty_info.njev) + int(exact_info.njev)
        raw_statuses.extend((str(penalty_info.message), str(exact_info.message)))
        success = bool(
            success
            and penalty["success"]
            and exact["success"]
            and float(exact_state["label_residual_abs"]) <= 1.0e-8
        )

    success = bool(
        success
        and float(values["area:exact:qfm_value"]) < float(values["initial:qfm_value"])
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
        driver="simsopt_lbfgsb_then_slsqp_qfm_sequence",
        normalized_status="converged" if success else "failed",
        raw_status="; ".join(raw_statuses),
        success=success,
        nit=total_nit,
        nfev=total_nfev,
        njev=total_njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _jax_state(state) -> dict[str, np.ndarray]:
    return {
        "parameters": np.asarray(state.parameters, dtype=np.float64),
        "qfm_value": np.asarray(state.qfm_value, dtype=np.float64),
        "qfm_gradient": np.asarray(state.qfm_gradient, dtype=np.float64),
        "label_value": np.asarray(state.label_value, dtype=np.float64),
        "label_gradient": np.asarray(state.label_gradient, dtype=np.float64),
        "label_residual_abs": np.asarray(
            state.label_residual_abs,
            dtype=np.float64,
        ),
    }


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.examples import solve_qfm_sequence
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

    import jax

    biotsavart, surface, fingerprint = _problem_components(bundle, arrays)
    device = get_runtime_jax_device()
    field = BiotSavartJAX(biotsavart.coils)
    coil_set_spec = field.coil_set_spec_from_dofs(
        jax.device_put(np.asarray(field.x, dtype=np.float64), device)
    )
    device_result = solve_qfm_sequence(
        initial_parameters=jax.device_put(arrays["initial_parameters"], device),
        quadpoints_phi=jax.device_put(arrays["quadrature_phi"], device),
        quadpoints_theta=jax.device_put(arrays["quadrature_theta"], device),
        coil_set_spec=coil_set_spec,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        max_steps=_configuration_int(bundle, "max_steps"),
        tolerance=_configuration_float(bundle, "jax_exact_tolerance"),
        constraint_weight=_configuration_float(bundle, "constraint_weight"),
    )
    result = jax.device_get(device_result)

    values: dict[str, np.ndarray] = {
        "initial:parameters": np.asarray(
            result.volume.initial.parameters,
            dtype=np.float64,
        ),
        "initial:qfm_value": np.asarray(
            result.volume.initial.qfm_value,
            dtype=np.float64,
        ),
        "initial:qfm_gradient": np.asarray(
            result.volume.initial.qfm_gradient,
            dtype=np.float64,
        ),
    }
    success = True
    total_nit = 0
    total_nfev = 0
    total_njev = 0
    raw_statuses: list[str] = []
    for stage_name in _LABELS:
        stage = getattr(result, stage_name)
        values[f"{stage_name}:target"] = np.asarray(stage.target, dtype=np.float64)
        values.update(_prefixed(stage_name, "initial", _jax_state(stage.initial)))
        values.update(_prefixed(stage_name, "penalty", _jax_state(stage.penalty)))
        values.update(_prefixed(stage_name, "exact", _jax_state(stage.exact)))
        values[f"{stage_name}:volume_persistence_objective"] = np.asarray(
            stage.volume_persistence_objective,
            dtype=np.float64,
        )
        total_nit += int(stage.penalty_optimizer.nit) + int(stage.exact_optimizer.nit)
        total_nfev += int(stage.penalty_optimizer.nfev) + int(
            stage.exact_optimizer.nfev
        )
        total_njev += int(stage.penalty_optimizer.njev) + int(
            stage.exact_optimizer.njev
        )
        raw_statuses.extend(
            (
                str(int(stage.penalty_optimizer.status)),
                str(int(stage.exact_optimizer.status)),
            )
        )
        success = bool(
            success
            and stage.penalty_optimizer.success
            and stage.exact_optimizer.success
            and float(stage.exact.label_residual_abs) <= 1.0e-8
        )
    success = bool(
        success
        and float(values["area:exact:qfm_value"]) < float(values["initial:qfm_value"])
    )
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
        driver="simsopt_bfgs_augmented_lagrangian_qfm_sequence",
        normalized_status="converged" if success else "failed",
        raw_status=",".join(raw_statuses),
        success=success,
        nit=total_nit,
        nfev=total_nfev,
        njev=total_njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact QFM sequence in the selected isolated lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
