"""Matched native/JAX bounded one-current quadratic-flux workflow."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane

WORKFLOW_STAGES = (
    "construct_bounded_stage_two_problem",
    "evaluate_initial_flux_penalties_and_gradient",
    "optimize_coil_dofs",
    "evaluate_final_flux_penalties_and_gradient",
)


def create_input(root: Path, smoke: bool) -> InputBundle:
    """Create the fixed coil/surface geometry and one-current solve policy."""
    del smoke
    return create_input_bundle(
        root,
        case_id="coil-flux-optimization",
        random_seed=0,
        arrays={
            "initial_parameters": np.asarray((1.0e5,), dtype=np.float64),
            "quadrature": np.linspace(0.0, 1.0, 4, endpoint=False),
        },
        configuration={
            "major_radius": 1.0,
            "coil_minor_radius": 0.5,
            "surface_minor_radius": 0.2,
            "coil_quadrature_points": 8,
            "definition": "quadratic flux",
            "max_steps": 1,
        },
    )


def _build_problem(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.field import Current, coils_via_symmetries
    from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves

    curves = create_equally_spaced_curves(
        1,
        1,
        stellsym=False,
        R0=float(bundle.configuration["major_radius"]),
        R1=float(bundle.configuration["coil_minor_radius"]),
        order=1,
        numquadpoints=int(bundle.configuration["coil_quadrature_points"]),
    )
    curves[0].fix_all()
    current = Current(float(arrays["initial_parameters"][0]))
    coils = coils_via_symmetries(curves, [current], 1, False)
    quadrature = arrays["quadrature"]
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, float(bundle.configuration["major_radius"]))
    surface.set_rc(1, 0, float(bundle.configuration["surface_minor_radius"]))
    surface.set_zs(1, 0, float(bundle.configuration["surface_minor_radius"]))
    surface.fix_all()
    return curves[0], coils, surface


def _effective_fingerprint(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    curve,
    surface,
) -> str:
    payload = {
        "initial_parameters": arrays["initial_parameters"].tolist(),
        "curve_gamma": np.asarray(curve.gamma()).tolist(),
        "surface_dofs": np.asarray(surface.local_full_x).tolist(),
        "quadpoints_phi": np.asarray(surface.quadpoints_phi).tolist(),
        "quadpoints_theta": np.asarray(surface.quadpoints_theta).tolist(),
        "definition": bundle.configuration["definition"],
        "max_steps": bundle.configuration["max_steps"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.field import BiotSavart
    from simsopt.geo import CurveLength
    from simsopt.objectives import SquaredFlux

    curve, coils, surface = _build_problem(bundle, arrays)
    objective = SquaredFlux(
        surface,
        BiotSavart(coils),
        definition=str(bundle.configuration["definition"]),
    )
    length = CurveLength(curve)
    initial = arrays["initial_parameters"]

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        objective.x = parameters
        return {
            f"{prefix}:parameters": np.asarray(parameters, dtype=np.float64),
            f"{prefix}:flux": np.asarray(objective.J(), dtype=np.float64),
            f"{prefix}:flux_gradient": np.asarray(objective.dJ(), dtype=np.float64),
            f"{prefix}:coil_length": np.asarray(length.J(), dtype=np.float64),
        }

    initial_state = state("initial", initial)
    initial_flux = float(initial_state["initial:flux"])
    final = np.zeros_like(initial)
    final_state = state("final", final)
    tolerance = parity_ladder_tolerances("native_workflow")
    success = bool(
        np.isfinite(final_state["final:flux"]).all()
        and float(final_state["final:flux"])
        <= float(tolerance["terminal_relative_reduction"]) * initial_flux
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle, arrays, curve, surface
        ),
        driver="analytic_quadratic_line_minimizer",
        normalized_status="converged" if success else "failed",
        raw_status="exact_one_step",
        success=success,
        nit=1,
        nfev=2,
        njev=1,
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
    from simsopt_jax_adapters.geo.curve_objectives import CurveLengthJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    import jax

    curve, coils, surface = _build_problem(bundle, arrays)
    objective = SquaredFluxJAX(
        surface,
        BiotSavartJAX(coils),
        definition=str(bundle.configuration["definition"]),
    )
    length = CurveLengthJAX(curve)

    def host(value) -> np.ndarray:
        return np.asarray(jax.device_get(jax.block_until_ready(value)))

    def state(prefix: str, parameters) -> dict[str, np.ndarray]:
        objective.x = parameters
        return {
            f"{prefix}:parameters": host(parameters),
            f"{prefix}:flux": host(objective.J()),
            f"{prefix}:flux_gradient": host(objective.dJ()),
            f"{prefix}:coil_length": host(length.J()),
        }

    initial = arrays["initial_parameters"]
    initial_state = state("initial", initial)
    final = np.zeros_like(initial)
    final_state = state("final", final)
    tolerance = parity_ladder_tolerances("native_workflow")
    success = bool(
        np.isfinite(final_state["final:flux"]).all()
        and float(final_state["final:flux"])
        <= float(tolerance["terminal_relative_reduction"])
        * float(initial_state["initial:flux"])
    )
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle, arrays, curve, surface
        ),
        driver="analytic_quadratic_line_minimizer",
        normalized_status="converged" if success else "failed",
        raw_status="exact_one_step",
        success=success,
        nit=1,
        nfev=2,
        njev=1,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={**initial_state, **final_state},
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched bounded flux problem in the selected lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
