"""JAX port of ``examples/2_Intermediate/boozerQA.py``.

The host constructs the NCSX coils and the initial volume-labelled
``SurfaceXYZTensorFourier``. ``BoozerSurfaceJAX`` solves the initial surface,
then a traceable outer objective optimizes the coil state for quasi-axisymmetry
while retaining the native iota, major-radius, and total base-coil-length
penalties. Each value-and-gradient evaluation executes on the selected JAX
device. The outer optimization remains host-driven so the nested surface solves
are compiled as bounded kernels instead of one memory-heavy optimization graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import Curve
from simsopt_jax.examples import ExampleResult, run_example, scalar_example_driver
from simsopt_jax.geo.optimizer_host_lbfgs import (
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
    minimize_lbfgs_host_core,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.surface_objectives import (
    make_traceable_objective_runtime_bundle,
    traceable_forward_result_outer_raw_terms,
)

EXAMPLE_ID = "native-boozerqa"
NATIVE_OUTER_ITERATIONS = 1000


def _host_float(value: object) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _host_bool(value: object) -> bool:
    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _surface_resolution(max_steps: int) -> tuple[int, int]:
    return (6, 6) if max_steps >= NATIVE_OUTER_ITERATIONS else (2, 2)


def _ncsx_configuration_options(max_steps: int) -> dict[str, int]:
    if max_steps >= NATIVE_OUTER_ITERATIONS:
        return {}
    return {
        "coil_order": 3,
        "magnetic_axis_order": 3,
        "points_per_period": 8,
    }


def _boozer_options() -> dict[str, object]:
    return {
        "newton_maxiter": 20,
        "newton_tol": 1.0e-10,
        "verbose": False,
    }


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        **_ncsx_configuration_options(max_steps),
    )
    magnetic_axis = cast(Curve, magnetic_axis)
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    mpol, ntor = _surface_resolution(max_steps)
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    volume_label = Volume(surface)
    volume_target = float(volume_label.J())
    boozer_surface = BoozerSurfaceJAX(
        field,
        surface,
        volume_label,
        volume_target,
        options=_boozer_options(),
    )
    initial_solve = cast(
        Mapping[str, object],
        boozer_surface.run_code_traceable(
            field.coil_set_spec(),
            jax.device_put(np.asarray(surface.get_dofs(), dtype=np.float64)),
            jax.device_put(np.asarray(-0.406, dtype=np.float64)),
            jax.device_put(np.asarray(G0, dtype=np.float64)),
        ),
    )
    boozer_surface.install_traceable_solved_runtime_state(initial_solve)
    initial_solver_success = _host_bool(initial_solve["success"])
    initial_residual = _host_float(
        jnp.sqrt(
            jnp.mean(
                jnp.square(jnp.asarray(initial_solve["residual"], dtype=jnp.float64))
            )
        )
    )
    iota_target = _host_float(initial_solve["iota"])
    major_radius_target = float(surface.major_radius())
    total_base_coil_length_target = float(
        sum(CurveLength(curve).J() for curve in base_curves)
    )

    qs_resolution = 20 if max_steps >= NATIVE_OUTER_ITERATIONS else 4
    objective_config: dict[str, object] = {
        "non_qs_weight": 1.0,
        "residual_weight": 0.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / nfp,
            2 * qs_resolution,
            endpoint=False,
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0,
            1.0,
            2 * qs_resolution,
            endpoint=False,
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": total_base_coil_length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
    runtime_bundle = make_traceable_objective_runtime_bundle(
        boozer_surface,
        field,
        iota_target,
        outer_objective_config=objective_config,
    )
    value_and_gradient = cast(
        Callable[[jax.Array], tuple[jax.Array, jax.Array]],
        runtime_bundle["value_and_grad"],
    )
    forward_result = cast(
        Callable[[jax.Array], Mapping[str, object]],
        runtime_bundle["forward_result"],
    )
    reporting_metrics_from_solution = cast(
        Callable[..., Mapping[str, object]],
        runtime_bundle["reporting_metrics_from_solution"],
    )

    initial_parameters = np.asarray(field.x, dtype=np.float64)
    initial_device = jax.device_put(initial_parameters)
    initial_value_device, initial_gradient_device = value_and_gradient(initial_device)
    jax.block_until_ready((initial_value_device, initial_gradient_device))
    initial_value_and_gradient = (
        _host_float(initial_value_device),
        np.asarray(jax.device_get(initial_gradient_device), dtype=np.float64),
    )

    def host_value_and_gradient(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        parameters_device = jax.device_put(np.asarray(parameters, dtype=np.float64))
        value_device, gradient_device = value_and_gradient(parameters_device)
        jax.block_until_ready((value_device, gradient_device))
        return (
            _host_float(value_device),
            np.asarray(jax.device_get(gradient_device), dtype=np.float64),
        )

    driver = scalar_example_driver()
    if driver == Driver.SIMSOPT_LBFGSB:
        optimizer_result = minimize_lbfgs_host_core(
            host_value_and_gradient,
            initial_parameters,
            maxiter=max_steps,
            maxcor=min(max_steps, 200),
            ftol=0.0,
            gtol=1.0e-8,
            maxls=20,
            initial_value_and_grad=initial_value_and_gradient,
        )
    else:
        optimizer_result = minimize_bfgs_host_core(
            host_value_and_gradient,
            initial_parameters,
            maxiter=max_steps,
            gtol=1.0e-8,
            maxls=20,
            initial_value_and_grad=initial_value_and_gradient,
            line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
        )
    solution = np.asarray(optimizer_result.x_k, dtype=np.float64)
    solution_device = jax.device_put(solution)
    final_objective = float(optimizer_result.f_k)
    final_forward = forward_result(solution_device)
    final_metrics = reporting_metrics_from_solution(
        solution_device,
        final_forward["x"],
        final_forward["success"],
        include_distance_metrics=False,
        outer_raw_terms=traceable_forward_result_outer_raw_terms(final_forward),
    )
    jax.block_until_ready((final_forward, final_metrics))
    final_boozer_objective = _host_float(final_metrics["final_boozer_residual"])
    final_residual = float(np.sqrt(2.0 * final_boozer_objective))
    final_iota = _host_float(final_metrics["final_iota"])
    final_volume = _host_float(final_metrics["final_volume"])
    inner_solver_success = _host_bool(final_metrics["solver_success"])
    outer_solver_success = bool(optimizer_result.converged)
    solver_success = bool(
        initial_solver_success and inner_solver_success and outer_solver_success
    )
    scientific_success = bool(
        initial_solver_success
        and inner_solver_success
        and np.isfinite(initial_residual)
        and np.isfinite(final_residual)
        and final_residual <= 1.0e-7
        and np.isfinite(initial_value_and_gradient[0])
        and np.isfinite(final_objective)
        and final_objective <= initial_value_and_gradient[0]
        and np.all(np.isfinite(optimizer_result.g_k))
        and np.isfinite(final_iota)
        and np.isfinite(final_volume)
        and abs(final_volume) > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_residual": initial_residual,
            "final_residual": final_residual,
            "initial_objective": initial_value_and_gradient[0],
            "final_objective": final_objective,
            "non_qs_ratio": _host_float(final_metrics["final_non_qs"]),
            "iota": final_iota,
            "volume": final_volume,
            "major_radius_penalty": _host_float(
                final_metrics["final_major_radius_penalty"]
            ),
            "total_base_coil_length_penalty": _host_float(
                final_metrics["final_length_penalty"]
            ),
            "solver_success": solver_success,
            "solver_status": int(optimizer_result.status),
            "solver_iterations": int(optimizer_result.k),
            "solver_evaluations": int(optimizer_result.nfev),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-boozerqa-",
        bounded_steps=2,
        native_default_steps=NATIVE_OUTER_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
