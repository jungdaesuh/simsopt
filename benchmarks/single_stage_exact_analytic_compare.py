"""Matched-policy exact-Boozer single-stage comparison: native C++ vs JAX analytic dense Newton.

Both lanes build the problem of ``examples/3_Advanced/single_stage_boozer_vacuum_optimization.py``
(NCSX coils, exact Boozer surface at ``--resolution``, non-QS + residual + iota + major-radius
+ length objective, SciPy BFGS with gtol 1e-15) and record every outer evaluation.
Diagnostic receipt only; ``sealed_claim`` is false.

Usage:
  python benchmarks/single_stage_exact_analytic_compare.py --lane native --max-steps 100 --out A.json
  SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 \\
      python benchmarks/single_stage_exact_analytic_compare.py --lane jax --max-steps 100 --out B.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from functools import reduce
from operator import add
from pathlib import Path
from typing import cast

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from repo_bootstrap import bootstrap_local_simsopt  # noqa: E402

# Native's coil terms evaluate through JAX on CPU and are float32 unless x64 is
# enabled; float64 is a physics requirement of this comparison on BOTH lanes, so
# an inherited JAX_ENABLE_X64=0 is overridden rather than honoured. The pin must
# precede bootstrap_local_simsopt: that call imports simsopt, which imports jax
# (simsopt/_core/json.py), and jax reads JAX_ENABLE_X64 only when imported.
os.environ["JAX_ENABLE_X64"] = "1"

bootstrap_local_simsopt(REPO / "src")

import numpy as np  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from simsopt.single_stage_boozer_vacuum import OUTER_GRADIENT_TOLERANCE  # noqa: E402
import jax  # noqa: E402
from simsopt.configs import get_data  # noqa: E402
from simsopt.field import BiotSavart  # noqa: E402
from simsopt.geo import (  # noqa: E402
    BoozerResidual,
    BoozerSurface,
    CurveLength,
    Iotas,
    MajorRadius,
    NonQuasiSymmetricRatio,
    SurfaceXYZTensorFourier,
    Volume,
)
from simsopt.objectives import QuadraticPenalty  # noqa: E402
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX  # noqa: E402
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX  # noqa: E402
from simsopt_jax_adapters.geo.single_stage_exact_analytic import ExactAnalyticSingleStage  # noqa: E402

DEFAULT_OUT = REPO / ".artifacts" / "single_stage_exact_analytic_compare.json"
INITIAL_IOTA = -0.406


def _surface_and_G(resolution: int):
    base_curves, base_currents, magnetic_axis, nfp, field = get_data("ncsx")
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    nq = 2 * resolution + 1
    surface = SurfaceXYZTensorFourier(
        mpol=resolution,
        ntor=resolution,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, nq, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, nq, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    return base_curves, base_currents, nfp, field, surface, G0


def _outer_objective_config(*, nfp, surface, major_radius_target, length_target):
    qs_resolution = 20
    return {
        "non_qs_weight": 1.0,
        "residual_weight": 1.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(0.0, 1.0 / nfp, 2 * qs_resolution, endpoint=False),
        "non_qs_quadpoints_theta": np.linspace(0.0, 1.0, 2 * qs_resolution, endpoint=False),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }


def _run_outer(value_and_gradient, x0, *, max_steps, trace):
    started = time.perf_counter()
    result = minimize(
        value_and_gradient,
        x0,
        jac=True,
        method="BFGS",
        options={"maxiter": int(max_steps), "gtol": OUTER_GRADIENT_TOLERANCE},
    )
    seconds = time.perf_counter() - started
    evaluations = trace
    return {
        "outer_seconds": seconds,
        "nit": int(result.nit),
        "nfev": int(result.nfev),
        "njev": int(result.njev),
        "status": int(result.status),
        "message": str(result.message),
        "final_objective": float(result.fun),
        "final_gradient_inf_norm": float(np.max(np.abs(result.jac))),
        "evaluations": evaluations,
        "eval_seconds_mean": float(np.mean([row["seconds"] for row in evaluations])),
        "eval_seconds_median": float(np.median([row["seconds"] for row in evaluations])),
        "inner_iterations_mean": float(np.mean([row["inner_iterations"] for row in evaluations])),
        "inner_failures": int(sum(1 for row in evaluations if not row["inner_success"])),
    }


def lane_native(resolution: int, max_steps: int):
    build_started = time.perf_counter()
    base_curves, base_currents, nfp, field, surface, G0 = _surface_and_G(resolution)
    volume = Volume(surface)
    boozer_surface = BoozerSurface(
        field, surface, volume, volume.J(),
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )
    initial_solve = boozer_surface.solve_residual_equation_exactly_newton(
        tol=1.0e-13, maxiter=20, iota=INITIAL_IOTA, G=G0
    )
    iota_target = float(initial_solve["iota"])
    major_radius = MajorRadius(boozer_surface)
    total_length = reduce(add, (CurveLength(curve) for curve in base_curves))
    non_qs = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(field.coils), sDIM=20)
    residual = BoozerResidual(boozer_surface, field)
    objective = (
        non_qs + residual
        + QuadraticPenalty(Iotas(boozer_surface), iota_target, "identity")
        + QuadraticPenalty(major_radius, cast(float, major_radius.J()), "identity")
        + QuadraticPenalty(total_length, float(total_length.J()), "max")
    )
    base_currents[0].fix_all()
    x0 = np.asarray(objective.x, dtype=np.float64)
    build_seconds = time.perf_counter() - build_started
    trace: list[dict[str, object]] = []

    def value_and_gradient(parameters):
        started = time.perf_counter()
        previous_surface = np.asarray(boozer_surface.surface.x, dtype=np.float64)
        previous_iota = float(boozer_surface.res["iota"])
        previous_G = float(boozer_surface.res["G"])
        objective.x = parameters
        value = float(objective.J())
        gradient = np.asarray(objective.dJ(), dtype=np.float64)
        success = bool(boozer_surface.res["success"])
        if not success:
            boozer_surface.surface.x = previous_surface
            boozer_surface.res["iota"] = previous_iota
            boozer_surface.res["G"] = previous_G
            value = 1.0e3
        trace.append({
            "seconds": time.perf_counter() - started,
            "value": value,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "inner_success": success,
            "inner_iterations": int(boozer_surface.res["iter"]),
        })
        return value, gradient

    payload = _run_outer(value_and_gradient, x0, max_steps=max_steps, trace=trace)
    if max_steps == 0:
        payload["initial_gradient"] = np.asarray(objective.dJ(), dtype=np.float64).tolist()
    payload.update({
        "lane": "native", "build_seconds": build_seconds, "coil_dofs": int(x0.size),
        "surface_dofs": int(surface.get_dofs().size), "iota_target": iota_target,
        "initial_inner_iterations": int(initial_solve["iter"]),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    })
    return payload, x0


def lane_jax(resolution: int, max_steps: int):
    build_started = time.perf_counter()
    base_curves, base_currents, nfp, native_field, surface, G0 = _surface_and_G(resolution)
    base_currents[0].fix_all()
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    boozer_surface = BoozerSurfaceJAX(
        field, surface, volume, float(volume.J()),
        options={"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False},
    )

    length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    evaluator = ExactAnalyticSingleStage(
        boozer_surface, field, iota=INITIAL_IOTA, G=G0,
        outer_objective_config=lambda: _outer_objective_config(
            nfp=nfp, surface=surface,
            major_radius_target=float(surface.major_radius()),
            length_target=length_target,
        ),
    )
    x0 = np.array(evaluator.coil_dofs, copy=True)
    compile_started = time.perf_counter()
    first = evaluator.evaluate(x0)
    first_eval_seconds = time.perf_counter() - compile_started
    build_seconds = time.perf_counter() - build_started
    trace: list[dict[str, object]] = []

    def value_and_gradient(parameters):
        started = time.perf_counter()
        evaluation = evaluator.evaluate(parameters)
        trace.append({
            "seconds": time.perf_counter() - started,
            "value": evaluation.value,
            "gradient_l2": float(np.linalg.norm(evaluation.gradient)),
            "inner_success": evaluation.inner_success,
            "inner_iterations": evaluation.inner_iterations,
        })
        return evaluation.value, evaluation.gradient

    payload = _run_outer(value_and_gradient, x0, max_steps=max_steps, trace=trace)
    payload.update({
        "lane": "jax", "build_seconds": build_seconds,
        "first_eval_seconds_incl_compile": first_eval_seconds,
        "initial_value": first.value, "initial_gradient_l2": float(np.linalg.norm(first.gradient)),
        "coil_dofs": int(x0.size), "surface_dofs": int(surface.get_dofs().size),
        "iota_target": evaluator.iota_target,
        "initial_inner_iterations": evaluator.initial_inner_iterations,
        "backend": str(jax.default_backend()), "devices": [str(d) for d in jax.devices()],
        "compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
    })
    return payload, x0, first


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("native", "jax", "parity"), required=True)
    parser.add_argument("--resolution", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    process_started = time.perf_counter()
    payload: dict[str, object] = {
        "resolution": int(args.resolution), "max_steps": int(args.max_steps),
        "diagnostic": True, "sealed_claim": False,
    }
    if args.lane == "native":
        lane, _x0 = lane_native(args.resolution, args.max_steps)
        payload["native"] = lane
    elif args.lane == "jax":
        lane, _x0, _first = lane_jax(args.resolution, args.max_steps)
        payload["jax"] = lane
    else:
        # Same process, one evaluation each at the same coils: value and gradient parity.
        jax_lane, x0, first = lane_jax(args.resolution, 0)
        native_lane, x0_native = lane_native(args.resolution, 0)
        native_first = native_lane["evaluations"][0]
        native_gradient = np.asarray(native_lane.pop("initial_gradient"), dtype=np.float64)
        payload["parity"] = {
            "x0_max_abs_diff": float(np.max(np.abs(x0 - x0_native))),
            "value_jax": first.value, "value_native": native_first["value"],
            "value_rel_diff": abs(first.value - native_first["value"]) / abs(native_first["value"]),
            "gradient_l2_jax": float(np.linalg.norm(first.gradient)),
            "gradient_l2_native": native_first["gradient_l2"],
            "gradient_max_abs_diff": float(np.max(np.abs(first.gradient - native_gradient))),
            "gradient_rel_l2_diff": float(
                np.linalg.norm(first.gradient - native_gradient) / np.linalg.norm(native_gradient)
            ),
        }
        payload["jax"] = jax_lane
        payload["native"] = native_lane
    payload["process_seconds"] = time.perf_counter() - process_started
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    summary: dict[str, object] = {
        k: v for k, v in payload.items() if k not in ("jax", "native")
    }
    for key in ("native", "jax"):
        lane = payload.get(key)
        if isinstance(lane, dict):
            summary[key] = {k: v for k, v in lane.items() if k != "evaluations"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
