"""Exact matched workflow for ``1_Simple/surf_vol_area.py``."""

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
from examples.jax.parity.symmetry import (
    global_column_swap_jacobian_invariants,
    parameter_invariants,
)
from simsopt_jax.examples import ExecutionScale

WORKFLOW_STAGES = (
    "construct_first_area_volume_problem",
    "evaluate_first_initial_state",
    "solve_first_area_volume_problem",
    "materialize_accepted_surface_state",
    "construct_second_area_volume_problem",
    "evaluate_second_initial_state",
    "solve_second_area_volume_problem",
    "evaluate_both_final_states",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize the native surface constants for every execution lane."""
    return create_input_bundle(
        root,
        case_id="native-surf-vol-area",
        random_seed=0,
        arrays={
            "initial_parameters": np.asarray((0.1, 0.1), dtype=np.float64),
            "quadrature": np.linspace(0.0, 1.0, 32, endpoint=False),
            "stage_targets": np.asarray(
                ((8.0, 0.6), (9.0, 0.8)),
                dtype=np.float64,
            ),
        },
        configuration={
            "major_radius": 1.0,
            "mpol": 1,
            "ntor": 0,
            "nfp": 1,
            "stellsym": True,
            "rtol": 1.0e-12,
            "atol": 1.0e-12,
            "max_steps": 64 if scale == "bounded" else 256,
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


def _build_surface(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.geo import SurfaceRZFourier

    quadrature = arrays["quadrature"]
    surface = SurfaceRZFourier(
        mpol=_configuration_int(bundle, "mpol"),
        ntor=_configuration_int(bundle, "ntor"),
        nfp=_configuration_int(bundle, "nfp"),
        stellsym=bool(bundle.configuration["stellsym"]),
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, _configuration_float(bundle, "major_radius"))
    surface.set_rc(1, 0, float(arrays["initial_parameters"][0]))
    surface.set_zs(1, 0, float(arrays["initial_parameters"][1]))
    surface.fix_all()
    surface.unfix("rc(1,0)")
    surface.unfix("zs(1,0)")
    return surface


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    surface,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "full_dofs": np.asarray(surface.local_full_x).tolist(),
            "free_positions": np.flatnonzero(surface.local_dofs_free_status).tolist(),
            "quadpoints_phi": np.asarray(surface.quadpoints_phi).tolist(),
            "quadpoints_theta": np.asarray(surface.quadpoints_theta).tolist(),
            "stage_targets": arrays["stage_targets"].tolist(),
            "mpol": surface.mpol,
            "ntor": surface.ntor,
            "nfp": surface.nfp,
            "stellsym": surface.stellsym,
            "rtol": bundle.configuration["rtol"],
            "atol": bundle.configuration["atol"],
            "max_steps": bundle.configuration["max_steps"],
        },
    )


def _state(
    prefix: str,
    parameters: np.ndarray,
    area: float,
    volume: float,
    residual: np.ndarray,
    jacobian: np.ndarray,
) -> dict[str, np.ndarray]:
    column_sum, column_product, column_association = (
        global_column_swap_jacobian_invariants(
            jacobian[:, 0],
            jacobian[:, 1],
        )
    )
    jacobian_invariants = np.concatenate(
        (column_sum, column_product, column_association.reshape(-1))
    )
    objective_gradient = 2.0 * jacobian.T @ residual
    return {
        f"{prefix}:parameters": parameters,
        f"{prefix}:parameter_invariants": parameter_invariants(parameters),
        f"{prefix}:area": np.asarray(area, dtype=np.float64),
        f"{prefix}:volume": np.asarray(volume, dtype=np.float64),
        f"{prefix}:residual": residual,
        f"{prefix}:residual_jacobian": jacobian,
        f"{prefix}:residual_jacobian_invariants": jacobian_invariants,
        f"{prefix}:objective_sum_squares": np.asarray(
            np.vdot(residual, residual),
            dtype=np.float64,
        ),
        f"{prefix}:objective_gradient": objective_gradient,
        f"{prefix}:objective_gradient_invariants": parameter_invariants(
            objective_gradient
        ),
    }


def _native_stage(
    *,
    surface,
    targets: np.ndarray,
    bundle: InputBundle,
    prefix: str,
) -> tuple[dict[str, np.ndarray], bool]:
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.solve import least_squares_serial_solve

    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    problem = LeastSquaresProblem.from_tuples(
        (
            (surface.area, float(targets[0]), 1.0),
            (surface.volume, float(targets[1]), 1.0),
        )
    )

    def state(phase: str) -> dict[str, np.ndarray]:
        parameters = np.asarray(problem.x, dtype=np.float64)
        area = float(surface.area())
        volume = float(surface.volume())
        residual = np.asarray((area, volume), dtype=np.float64) - targets
        jacobian = np.stack(
            (
                np.asarray(surface.darea(), dtype=np.float64)[free_positions],
                np.asarray(surface.dvolume(), dtype=np.float64)[free_positions],
            )
        )
        return _state(
            f"{prefix}:{phase}",
            parameters,
            area,
            volume,
            residual,
            jacobian,
        )

    initial = state("initial")
    least_squares_serial_solve(
        problem,
        ftol=_configuration_float(bundle, "rtol"),
        xtol=_configuration_float(bundle, "atol"),
        gtol=_configuration_float(bundle, "atol"),
        max_nfev=_configuration_int(bundle, "max_steps"),
    )
    final = state("final")
    return {**initial, **final}, bool(
        np.linalg.norm(final[f"{prefix}:final:residual"]) <= 1.0e-8
    )


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt import load

    surface = _build_surface(bundle, arrays)
    fingerprint = _effective_fingerprint(bundle, arrays, surface)
    first_values, first_success = _native_stage(
        surface=surface,
        targets=arrays["stage_targets"][0],
        bundle=bundle,
        prefix="first",
    )
    surface.save("surf_fw.json", indent=2)
    second_surface = load("surf_fw.json")
    second_values, second_success = _native_stage(
        surface=second_surface,
        targets=arrays["stage_targets"][1],
        bundle=bundle,
        prefix="second",
    )
    success = first_success and second_success
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=fingerprint,
        driver="simsopt_least_squares_serial_solve",
        normalized_status="converged" if success else "failed",
        raw_status="two_stage_residual_threshold",
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**first_values, **second_values},
        applicability={
            "first:final:parameters": False,
            "first:final:residual_jacobian": False,
            "first:final:objective_gradient": False,
            "second:initial:parameters": False,
            "second:initial:residual_jacobian": False,
            "second:initial:objective_gradient": False,
            "second:final:parameters": False,
            "second:final:residual_jacobian": False,
            "second:final:objective_gradient": False,
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.examples import solve_rz_surface_area_volume_sequence

    import jax

    surface = _build_surface(bundle, arrays)
    fingerprint = _effective_fingerprint(bundle, arrays, surface)
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    result = solve_rz_surface_area_volume_sequence(
        full_dofs=jax.device_put(np.asarray(surface.local_full_x, dtype=np.float64)),
        quadpoints_phi=jax.device_put(
            np.asarray(surface.quadpoints_phi, dtype=np.float64)
        ),
        quadpoints_theta=jax.device_put(
            np.asarray(surface.quadpoints_theta, dtype=np.float64)
        ),
        free_positions=jax.device_put(free_positions),
        first_targets=jax.device_put(arrays["stage_targets"][0]),
        second_targets=jax.device_put(arrays["stage_targets"][1]),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        max_steps=_configuration_int(bundle, "max_steps"),
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
    )

    def host(value: jax.Array) -> np.ndarray:
        return np.asarray(jax.device_get(value), dtype=np.float64)

    def stage_values(prefix: str, stage) -> dict[str, np.ndarray]:
        return {
            **_state(
                f"{prefix}:initial",
                host(stage.initial_parameters),
                float(host(stage.initial_area)),
                float(host(stage.initial_volume)),
                host(stage.initial_residuals),
                host(stage.initial_jacobian),
            ),
            **_state(
                f"{prefix}:final",
                host(stage.final_parameters),
                float(host(stage.final_area)),
                float(host(stage.final_volume)),
                host(stage.final_residuals),
                host(stage.final_jacobian),
            ),
        }

    success = result.first.optimizer.success and result.second.optimizer.success
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
        driver=result.first.optimizer.driver.value,
        normalized_status="converged" if success else "failed",
        raw_status=(
            f"{result.first.optimizer.status},{result.second.optimizer.status}"
        ),
        success=success,
        nit=result.first.optimizer.nit + result.second.optimizer.nit,
        nfev=result.first.optimizer.nfev + result.second.optimizer.nfev,
        njev=result.first.optimizer.njev + result.second.optimizer.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **stage_values("first", result.first),
            **stage_values("second", result.second),
        },
        applicability={
            "first:final:parameters": False,
            "first:final:residual_jacobian": False,
            "first:final:objective_gradient": False,
            "second:initial:parameters": False,
            "second:initial:residual_jacobian": False,
            "second:initial:objective_gradient": False,
            "second:final:parameters": False,
            "second:final:residual_jacobian": False,
            "second:final:objective_gradient": False,
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact two-stage native or JAX surface workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
