"""Exact matched workflow for ``2_Intermediate/boozer.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import cast

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
    "construct_ncsx_coils_and_tensor_fourier_surface",
    "evaluate_initial_boozer_residual_and_jacobian",
    "reduce_area_constrained_residual_with_lbfgs",
    "polish_area_constrained_residual_with_least_squares",
    "triple_toroidal_flux_label_and_resolve_surface",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "mpol": 5 if native_scale else 2,
        "ntor": 5 if native_scale else 2,
        "native_bfgs_maxiter": 300,
        "native_ls_maxiter": 100,
        "jax_bfgs_maxiter": 300 if native_scale else 60,
        "jax_ls_maxiter": 100,
        "solver_tolerance": 1.0e-10,
        "constraint_weight": 100.0,
        "initial_iota": -0.4,
        "surface_distance": 0.10,
        "flux_multiplier": 3.0,
    }


def _configuration_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _problem(configuration: Mapping[str, object]):
    from simsopt.configs import get_data
    from simsopt.field import BiotSavart
    from simsopt.geo import SurfaceXYZTensorFourier

    _, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    field = BiotSavart(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    mpol = _configuration_int(configuration, "mpol")
    ntor = _configuration_int(configuration, "ntor")
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * ntor + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * mpol + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(
        magnetic_axis,
        _configuration_float(configuration, "surface_distance"),
        flip_theta=True,
    )
    return magnetic_axis, native_field, field, surface, float(G0)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze NCSX and initial-surface state for every solver lane."""
    configuration = _scale_configuration(scale)
    magnetic_axis, native_field, _, surface, G0 = _problem(configuration)
    return create_input_bundle(
        root,
        case_id="native-boozer",
        random_seed=0,
        arrays={
            "axis_dofs": np.asarray(
                magnetic_axis.local_full_x,
                dtype=np.float64,
            ),
            "field_dofs": np.asarray(native_field.x, dtype=np.float64),
            "surface_dofs": np.asarray(surface.get_dofs(), dtype=np.float64),
        },
        configuration={**configuration, "initial_G": G0},
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
    surface_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "axis_dofs": _array_digest(axis_dofs),
            "field_dofs": _array_digest(field_dofs),
            "surface_dofs": _array_digest(surface_dofs),
            **bundle.configuration,
        },
    )


def _residual_norm(surface, iota: float, G: float, field) -> float:
    from simsopt.geo import boozer_surface_residual

    residual = boozer_surface_residual(
        surface,
        iota,
        G,
        field,
        derivatives=0,
    )[0]
    return float(np.linalg.norm(np.asarray(residual, dtype=np.float64)))


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.geo import Area, BoozerSurface, ToroidalFlux

    magnetic_axis, native_field, field, surface, G0 = _problem(bundle.configuration)
    initial_iota = _configuration_float(bundle.configuration, "initial_iota")
    area = Area(surface)
    solver = BoozerSurface(
        native_field,
        surface,
        area,
        float(area.J()),
    )
    initial_x = np.concatenate(
        (arrays["surface_dofs"], np.asarray([initial_iota, G0], dtype=np.float64))
    )
    initial_residual, initial_jacobian = solver._get_residual_vector_and_jacobian(
        initial_x,
        _configuration_float(bundle.configuration, "constraint_weight"),
        True,
        True,
    )
    initial_residual_norm = _residual_norm(
        surface,
        initial_iota,
        G0,
        native_field,
    )
    rough = solver.minimize_boozer_penalty_constraints_LBFGS(
        tol=_configuration_float(bundle.configuration, "solver_tolerance"),
        maxiter=_configuration_int(bundle.configuration, "native_bfgs_maxiter"),
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
        iota=initial_iota,
        G=G0,
    )
    solver.need_to_run_code = True
    polished = solver.minimize_boozer_penalty_constraints_ls(
        tol=_configuration_float(bundle.configuration, "solver_tolerance"),
        maxiter=_configuration_int(bundle.configuration, "native_ls_maxiter"),
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
        iota=float(rough["iota"]),
        G=float(rough["G"]),
        method="manual",
    )
    area_iota = float(polished["iota"])
    area_G = float(polished["G"])
    area_label = float(area.J())
    area_residual_norm = _residual_norm(
        surface,
        area_iota,
        area_G,
        native_field,
    )

    toroidal_flux = ToroidalFlux(surface, field)
    flux_target = _configuration_float(
        bundle.configuration, "flux_multiplier"
    ) * float(toroidal_flux.J())
    flux_solver = BoozerSurface(
        native_field,
        surface,
        toroidal_flux,
        flux_target,
    )
    expanded = flux_solver.minimize_boozer_penalty_constraints_ls(
        tol=_configuration_float(bundle.configuration, "solver_tolerance"),
        maxiter=_configuration_int(bundle.configuration, "native_ls_maxiter"),
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
        iota=area_iota,
        G=area_G,
        method="manual",
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_surface_dofs=arrays["surface_dofs"],
        initial_residual=np.asarray(initial_residual, dtype=np.float64),
        initial_jacobian=np.asarray(initial_jacobian, dtype=np.float64),
        initial_residual_norm=initial_residual_norm,
        area_iota=area_iota,
        area_G=area_G,
        area_label=area_label,
        area_residual_norm=area_residual_norm,
        flux_target=flux_target,
        flux_iota=float(expanded["iota"]),
        flux_G=float(expanded["G"]),
        flux_label=float(toroidal_flux.J()),
        flux_residual_norm=_residual_norm(
            surface,
            float(expanded["iota"]),
            float(expanded["G"]),
            native_field,
        ),
        flux_surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
        area_solver_success=bool(polished["success"]),
        flux_solver_success=bool(expanded["success"]),
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsopt_scipy_lbfgsb_manual_lm",
    )


def _host_float(value: object) -> float:
    import jax

    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _host_bool(value: object) -> bool:
    import jax

    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax
    import jax.numpy as jnp
    from simsopt.geo import Area, ToroidalFlux
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

    magnetic_axis, native_field, field, surface, G0 = _problem(bundle.configuration)
    initial_iota = _configuration_float(bundle.configuration, "initial_iota")
    jax_field = BiotSavartJAX(native_field.coils)
    options = {
        "bfgs_maxiter": _configuration_int(
            bundle.configuration, "jax_bfgs_maxiter"
        ),
        "bfgs_tol": _configuration_float(
            bundle.configuration, "solver_tolerance"
        ),
        "newton_maxiter": _configuration_int(
            bundle.configuration, "jax_ls_maxiter"
        ),
        "newton_tol": _configuration_float(
            bundle.configuration, "solver_tolerance"
        ),
        "verbose": False,
    }
    area = Area(surface)
    solver = BoozerSurfaceJAX(
        jax_field,
        surface,
        area,
        float(area.J()),
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
        options=options,
    )
    initial_x = jnp.concatenate(
        (
            jax.device_put(
                np.asarray(arrays["surface_dofs"], dtype=np.float64)
            ),
            jax.device_put(
                np.asarray([initial_iota, G0], dtype=np.float64)
            ),
        )
    )
    kernels = solver._get_penalty_kernel_bundle(
        optimize_G=True,
        weight_inv_modB=True,
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
    )
    coil_spec = jax_field.coil_set_spec()
    initial_residual_device, initial_jacobian_device = (
        kernels.residual(initial_x, coil_spec),
        kernels.jacobian(initial_x, coil_spec),
    )
    initial_residual, initial_jacobian = jax.device_get(
        (initial_residual_device, initial_jacobian_device)
    )
    initial_residual_norm = _residual_norm(
        surface,
        initial_iota,
        G0,
        native_field,
    )

    rough = cast(
        Mapping[str, object],
        solver.minimize_boozer_penalty_constraints_LBFGS(
            tol=_configuration_float(bundle.configuration, "solver_tolerance"),
            maxiter=_configuration_int(bundle.configuration, "jax_bfgs_maxiter"),
            constraint_weight=_configuration_float(
                bundle.configuration, "constraint_weight"
            ),
            iota=initial_iota,
            G=G0,
        ),
    )
    solver.need_to_run_code = True
    polished = cast(
        Mapping[str, object],
        solver.minimize_boozer_penalty_constraints_ls(
            tol=_configuration_float(bundle.configuration, "solver_tolerance"),
            maxiter=_configuration_int(bundle.configuration, "jax_ls_maxiter"),
            constraint_weight=_configuration_float(
                bundle.configuration, "constraint_weight"
            ),
            iota=_host_float(rough["iota"]),
            G=_host_float(rough["G"]),
            method="manual",
        ),
    )
    area_iota = _host_float(polished["iota"])
    area_G = _host_float(polished["G"])
    area_label = float(area.J())
    area_residual_norm = _residual_norm(
        surface,
        area_iota,
        area_G,
        native_field,
    )

    toroidal_flux = ToroidalFlux(surface, field)
    flux_target = _configuration_float(
        bundle.configuration, "flux_multiplier"
    ) * float(toroidal_flux.J())
    flux_field = BiotSavartJAX(native_field.coils)
    flux_solver = BoozerSurfaceJAX(
        flux_field,
        surface,
        toroidal_flux,
        flux_target,
        constraint_weight=_configuration_float(
            bundle.configuration, "constraint_weight"
        ),
        options=options,
        surface_runtime_state=solver.surface_runtime_state,
    )
    expanded = cast(
        Mapping[str, object],
        flux_solver.minimize_boozer_penalty_constraints_ls(
            tol=_configuration_float(bundle.configuration, "solver_tolerance"),
            maxiter=_configuration_int(bundle.configuration, "jax_ls_maxiter"),
            constraint_weight=_configuration_float(
                bundle.configuration, "constraint_weight"
            ),
            iota=area_iota,
            G=area_G,
            method="manual",
        ),
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_surface_dofs=arrays["surface_dofs"],
        initial_residual=np.asarray(initial_residual, dtype=np.float64),
        initial_jacobian=np.asarray(initial_jacobian, dtype=np.float64),
        initial_residual_norm=initial_residual_norm,
        area_iota=area_iota,
        area_G=area_G,
        area_label=area_label,
        area_residual_norm=area_residual_norm,
        flux_target=flux_target,
        flux_iota=_host_float(expanded["iota"]),
        flux_G=_host_float(expanded["G"]),
        flux_label=float(toroidal_flux.J()),
        flux_residual_norm=_residual_norm(
            surface,
            _host_float(expanded["iota"]),
            _host_float(expanded["G"]),
            native_field,
        ),
        flux_surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
        area_solver_success=_host_bool(polished["success"]),
        flux_solver_success=_host_bool(expanded["success"]),
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver="simsopt_jax_bfgs_lm",
    )


def _values(
    *,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
    initial_surface_dofs: np.ndarray,
    initial_residual: np.ndarray,
    initial_jacobian: np.ndarray,
    initial_residual_norm: float,
    area_iota: float,
    area_G: float,
    area_label: float,
    area_residual_norm: float,
    flux_target: float,
    flux_iota: float,
    flux_G: float,
    flux_label: float,
    flux_residual_norm: float,
    flux_surface_dofs: np.ndarray,
    area_solver_success: bool,
    flux_solver_success: bool,
) -> dict[str, np.ndarray]:
    return {
        "construction:axis_dofs": axis_dofs,
        "construction:field_dofs": field_dofs,
        "initial:surface_dofs": initial_surface_dofs,
        "initial:residual": initial_residual,
        "initial:jacobian": initial_jacobian,
        "initial:residual_norm": np.asarray(initial_residual_norm, dtype=np.float64),
        "area:iota": np.asarray(area_iota, dtype=np.float64),
        "area:G": np.asarray(area_G, dtype=np.float64),
        "area:label": np.asarray(area_label, dtype=np.float64),
        "area:residual_norm": np.asarray(area_residual_norm, dtype=np.float64),
        "area:solver_success": np.asarray(area_solver_success, dtype=np.bool_),
        "flux:target": np.asarray(flux_target, dtype=np.float64),
        "flux:iota": np.asarray(flux_iota, dtype=np.float64),
        "flux:G": np.asarray(flux_G, dtype=np.float64),
        "flux:label": np.asarray(flux_label, dtype=np.float64),
        "flux:residual_norm": np.asarray(flux_residual_norm, dtype=np.float64),
        "flux:surface_dofs": flux_surface_dofs,
        "flux:solver_success": np.asarray(flux_solver_success, dtype=np.bool_),
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["flux:surface_dofs"]))
        and np.isfinite(float(values["flux:residual_norm"]))
        and float(values["flux:residual_norm"])
        < float(values["initial:residual_norm"])
        and np.isfinite(float(values["flux:iota"]))
        and np.isfinite(float(values["flux:G"]))
    )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle,
            values["construction:axis_dofs"],
            values["construction:field_dofs"],
            values["initial:surface_dofs"],
        ),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status=(
            f"area={bool(values['area:solver_success'])};"
            f"flux={bool(values['flux:solver_success'])}"
        ),
        success=success,
        nit=None,
        nfev=None,
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
    """Execute the exact three-stage Boozer-surface workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
