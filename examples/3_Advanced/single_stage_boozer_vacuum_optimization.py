#!/usr/bin/env python3
"""Native SIMSOPT reference for VMEC-free Boozer single-stage optimization.

This bounded reference is derived from the implicit-Boozer vacuum formulation
used by the SIMSOPT single-stage research workflow.  Coil changes trigger a
native ``BoozerSurface`` solve, and the surface adjoint maps iota, volume,
quasisymmetry, and residual sensitivities back to coil degrees of freedom.  It
does not invoke VMEC or SPEC.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from simsopt.configs import get_data
from simsopt.field import BiotSavart
from simsopt.geo import (
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
    boozer_surface_residual,
)
from simsopt.objectives import QuadraticPenalty
from simsopt.optimization_endpoint import (
    accepted_step_contract,
    certify_optimization_endpoint,
)
from simsopt.single_stage_boozer_vacuum import (
    NATIVE_ITERATIONS,
    OUTER_GRADIENT_TOLERANCE,
    scientific_success_for_scale,
)

EXAMPLE_ID = "native-single-stage-boozer-vacuum-optimization"


def _configuration_options(native_scale: bool) -> dict[str, int]:
    if native_scale:
        return {}
    return {
        "coil_order": 3,
        "magnetic_axis_order": 3,
        "points_per_period": 8,
    }


def solve(max_steps: int, *, native_scale: bool) -> dict[str, object]:
    base_curves, base_currents, magnetic_axis, nfp, field = get_data(
        "ncsx",
        **_configuration_options(native_scale),
    )
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    resolution = 6 if native_scale else 1
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * resolution + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * resolution + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    volume = Volume(surface)
    boozer_surface = BoozerSurface(
        field,
        surface,
        volume,
        volume.J(),
        options={
            "newton_maxiter": 20,
            "newton_tol": 1.0e-13,
            "verbose": False,
        },
    )
    inner_iterations = 20
    initial_solve = boozer_surface.solve_residual_equation_exactly_newton(
        tol=1.0e-13,
        maxiter=inner_iterations,
        iota=-0.406,
        G=G0,
    )
    initial_inner_success = bool(initial_solve["success"])
    iota_target = float(initial_solve["iota"])
    major_radius = MajorRadius(boozer_surface)
    lengths = [CurveLength(curve) for curve in base_curves]
    total_length = sum(lengths)
    non_qs = NonQuasiSymmetricRatio(
        boozer_surface,
        BiotSavart(field.coils),
        sDIM=20 if native_scale else 4,
    )
    residual = BoozerResidual(boozer_surface, field)
    objective = (
        non_qs
        + residual
        + QuadraticPenalty(Iotas(boozer_surface), iota_target, "identity")
        + QuadraticPenalty(major_radius, major_radius.J(), "identity")
        + QuadraticPenalty(total_length, float(total_length.J()), "max")
    )
    base_currents[0].fix_all()
    initial_parameters = np.asarray(objective.x, dtype=np.float64)
    initial_objective = float(objective.J())
    initial_gradient = np.asarray(objective.dJ(), dtype=np.float64)

    def value_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        previous_surface = np.asarray(boozer_surface.surface.x, dtype=np.float64)
        previous_iota = float(boozer_surface.res["iota"])
        previous_G = float(boozer_surface.res["G"])
        objective.x = parameters
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=np.float64)
        if not bool(boozer_surface.res["success"]):
            boozer_surface.surface.x = previous_surface
            boozer_surface.res["iota"] = previous_iota
            boozer_surface.res["G"] = previous_G
            return 1.0e3, gradient
        return value, gradient

    optimizer_result = minimize(
        value_and_gradient,
        initial_parameters,
        jac=True,
        method="BFGS",
        options={"maxiter": max_steps, "gtol": OUTER_GRADIENT_TOLERANCE},
    )
    objective.x = np.asarray(optimizer_result.x, dtype=np.float64)
    final_objective = float(objective.J())
    gradient = np.asarray(objective.dJ(), dtype=np.float64)
    final_iota = float(Iotas(boozer_surface).J())
    final_volume = float(Volume(surface).J())
    final_non_qs = float(non_qs.J())
    final_residual = float(residual.J())
    residual_vector = boozer_surface_residual(
        surface,
        boozer_surface.res["iota"],
        boozer_surface.res["G"],
        field,
        derivatives=0,
    )[0]
    residual_rms = float(np.sqrt(np.mean(np.square(residual_vector))))
    inner_solver_success = bool(boozer_surface.res["success"])
    parameters_finite = bool(
        np.all(np.isfinite(optimizer_result.x)) and np.all(np.isfinite(gradient))
    )
    observables_finite = bool(
        np.isfinite(final_objective)
        and np.isfinite(final_iota)
        and np.isfinite(final_volume)
        and np.isfinite(final_non_qs)
        and np.isfinite(final_residual)
        and np.isfinite(residual_rms)
    )
    inner_success = bool(initial_inner_success and inner_solver_success)
    endpoint_certificate = certify_optimization_endpoint(
        status_convention="scipy-bfgs",
        provider_success=bool(optimizer_result.success),
        provider_status=int(optimizer_result.status),
        iterations=int(optimizer_result.nit),
        max_iterations=max_steps,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(gradient))),
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        inner_success=inner_success,
    )
    scientific_success = scientific_success_for_scale(
        native_scale=native_scale,
        certificate=endpoint_certificate,
        accepted_step=accepted_step_contract(
            iterations=int(optimizer_result.nit),
            initial_stationary=endpoint_certificate.initial_stationary,
        ),
        inner_success=inner_success,
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        monotone_descent=final_objective <= initial_objective,
    )
    return {
        "example_id": EXAMPLE_ID,
        "backend_mode": "native_cpu",
        "platform": "cpu",
        "precision": "fp64",
        "status": "ok" if scientific_success else "failed",
        "observables": {
            "initial_objective": initial_objective,
            "initial_gradient": tuple(float(value) for value in initial_gradient),
            "final_objective": final_objective,
            "solution": tuple(float(value) for value in optimizer_result.x),
            "gradient": tuple(float(value) for value in gradient),
            "inner_solver_success": inner_solver_success,
            "outer_solver_success": bool(optimizer_result.success),
            "outer_stopping_reason": endpoint_certificate.stopping_reason,
            "initial_stationary": endpoint_certificate.initial_stationary,
            "terminal_stationary": endpoint_certificate.terminal_stationary,
            "solver_status": int(optimizer_result.status),
            "solver_iterations": int(optimizer_result.nit),
            "solver_evaluations": int(optimizer_result.nfev),
            "iota": final_iota,
            "volume": final_volume,
            "non_qs_ratio": final_non_qs,
            "boozer_residual": final_residual,
            "boozer_residual_rms": residual_rms,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    max_steps = (
        options.max_steps
        if options.max_steps is not None
        else (2 if options.smoke else NATIVE_ITERATIONS)
    )
    output_directory = options.output_dir
    if output_directory is None:
        output_directory = Path(
            tempfile.mkdtemp(prefix="simsopt-native-single-stage-boozer-")
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    result = solve(max_steps, native_scale=not options.smoke)
    if options.json:
        print(json.dumps(result, sort_keys=True))
    else:
        observables = result["observables"]
        print(f"initial objective={observables['initial_objective']:.8e}")
        print(f"final objective={observables['final_objective']:.8e}")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
