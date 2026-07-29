"""JAX port of ``examples/1_Simple/tracing_fieldlines_QA.py``.

The host loads the same optimized stage-two coils and QA boundary, then
materializes the cylindrical interpolation tables.  Field evaluation,
surface-exit events, and Poincare-plane field-line tracing execute on the
selected JAX device.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import simsopt
from simsopt.field import LevelsetStoppingCriterion, SurfaceClassifier
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.tracing import compute_fieldlines

EXAMPLE_ID = "native-tracing-fieldlines-qa"
NATIVE_TMAX = 20_000
REPO_ROOT = Path(__file__).resolve().parents[3]


def _problem(scale: ExecutionScale):
    native_scale = scale == "native_default"
    surface = SurfaceRZFourier.from_vmec_input(
        str(REPO_ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"),
        nphi=200 if native_scale else 24,
        ntheta=30 if native_scale else 10,
        range="full torus",
    )
    native_field = simsopt.load(
        REPO_ROOT / "examples" / "1_Simple" / "inputs" / "biot_savart_opt.json"
    )
    source_field = BiotSavartJAX(native_field.coils)
    grid_size = 20 if native_scale else 5
    degree = 4 if native_scale else 2
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    classifier = SurfaceClassifier(
        surface,
        h=0.03 if native_scale else 0.08,
        p=2,
    )

    def skip(
        radial_values: np.ndarray,
        phi_values: np.ndarray,
        height_values: np.ndarray,
    ) -> np.ndarray:
        cylindrical_points = np.column_stack((radial_values, phi_values, height_values))
        return (classifier.evaluate_rphiz(cylindrical_points) < -0.05).reshape(-1)

    interpolated = InterpolatedFieldJAX(
        source_field,
        degree,
        (float(np.min(radii)), float(np.max(radii)), grid_size),
        (0.0, 2.0 * np.pi / surface.nfp, 2 * grid_size),
        (0.0, float(np.max(heights)), grid_size // 2),
        True,
        nfp=surface.nfp,
        stellsym=True,
        skip=skip,
    )
    return surface, source_field, interpolated, classifier


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    surface, source_field, interpolated, classifier = _problem(scale)
    surface_points = np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    source_field.set_points(surface_points)
    interpolated.set_points(surface_points)
    source_values = np.asarray(jax.device_get(source_field.B()), dtype=np.float64)
    interpolated_values = np.asarray(interpolated.B(), dtype=np.float64)
    interpolation_error = float(
        np.linalg.norm(interpolated_values - source_values)
        / np.linalg.norm(source_values)
    )

    nfieldlines = 10 if scale == "native_default" else 3
    radial_initial = np.linspace(1.2125346, 1.295, nfieldlines)
    vertical_initial = np.zeros(nfieldlines, dtype=np.float64)
    phis = tuple(index * 0.5 * np.pi / surface.nfp for index in range(4))
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        radial_initial,
        vertical_initial,
        tmax=max_steps,
        tol=1.0e-12,
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
    )
    initial_states = np.column_stack(
        (radial_initial, np.zeros(nfieldlines), vertical_initial)
    )
    final_states = np.stack([trajectory[-1, 1:4] for trajectory in trajectories])
    final_times = np.asarray(
        [trajectory[-1, 0] for trajectory in trajectories],
        dtype=np.float64,
    )
    statuses = tuple(
        0 if np.isclose(time, max_steps, rtol=0.0, atol=1.0e-10) else -1
        for time in final_times
    )
    scientific_success = bool(
        len(trajectories) == nfieldlines
        and np.all(np.isfinite(final_states))
        and np.isfinite(interpolation_error)
        and interpolation_error < 0.5
        and all(status <= 0 for status in statuses)
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_states": tuple(
                tuple(float(component) for component in state)
                for state in initial_states
            ),
            "final_states": tuple(
                tuple(float(component) for component in state) for state in final_states
            ),
            "poincare_hits": tuple(
                tuple(tuple(float(value) for value in hit) for hit in line_hits)
                for line_hits in phi_hits
            ),
            "integrator_status": statuses,
            "interpolation_error": interpolation_error,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-tracing-fieldlines-qa-",
        bounded_steps=50,
        native_default_steps=NATIVE_TMAX,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
