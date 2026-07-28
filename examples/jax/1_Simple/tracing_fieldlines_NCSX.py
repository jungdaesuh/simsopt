"""JAX port of ``examples/1_Simple/tracing_fieldlines_NCSX.py``.

The host loads the NCSX coils and magnetic axis, fits the same enclosing
surface, and samples the Biot-Savart field on a cylindrical interpolation
grid.  Interpolated field evaluation and all field-line integration then run
on the selected JAX device; plotting and VTK output remain optional reporting.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import jax
import numpy as np
from simsopt.configs import get_data
from simsopt.field import LevelsetStoppingCriterion, SurfaceClassifier
from simsopt.geo import SurfaceRZFourier
from simsopt.geo.curve import Curve
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.tracing import compute_fieldlines

EXAMPLE_ID = "native-tracing-fieldlines-ncsx"
NATIVE_TMAX = 40_000


def _problem(max_steps: int):
    _base_curves, _currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    magnetic_axis = cast(Curve, magnetic_axis)
    source_field = BiotSavartJAX(native_field.coils)
    native_scale = max_steps >= NATIVE_TMAX
    nphi, ntheta = (64, 24) if native_scale else (16, 8)
    grid_size = 20 if native_scale else 5
    degree = 4 if native_scale else 2
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface.fit_to_curve(magnetic_axis, 0.70, flip_theta=False)
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    classifier = SurfaceClassifier(surface, h=0.08 if not native_scale else 0.03, p=2)

    def skip(
        radii_values: np.ndarray,
        phi_values: np.ndarray,
        height_values: np.ndarray,
    ) -> np.ndarray:
        cylindrical_points = np.column_stack((radii_values, phi_values, height_values))
        return (classifier.evaluate_rphiz(cylindrical_points) < -0.05).reshape(-1)

    interpolated = InterpolatedFieldJAX(
        source_field,
        degree,
        (float(np.min(radii)), float(np.max(radii)), grid_size),
        (0.0, 2.0 * np.pi / nfp, 2 * grid_size),
        (0.0, float(np.max(heights)), grid_size // 2),
        True,
        nfp=nfp,
        stellsym=True,
        skip=skip,
    )
    return magnetic_axis, source_field, interpolated, classifier, nfp


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    magnetic_axis, source_field, interpolated, classifier, nfp = _problem(max_steps)
    axis_points = np.asarray(magnetic_axis.gamma(), dtype=np.float64)
    source_field.set_points(axis_points)
    interpolated.set_points(axis_points)
    source_values = np.asarray(jax.device_get(source_field.B()), dtype=np.float64)
    interpolated_values = np.asarray(interpolated.B(), dtype=np.float64)
    interpolation_error = float(
        np.linalg.norm(interpolated_values - source_values)
        / np.linalg.norm(source_values)
    )

    nfieldlines = 3 if max_steps < NATIVE_TMAX else 30
    radial_initial = np.linspace(
        axis_points[0, 0],
        axis_points[0, 0] + 0.14,
        nfieldlines,
    )
    vertical_initial = np.full(nfieldlines, axis_points[0, 2], dtype=np.float64)
    phis = tuple(index * 0.5 * np.pi / nfp for index in range(4))
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        radial_initial,
        vertical_initial,
        tmax=max_steps,
        tol=1.0e-7,
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
            "interpolation_error": interpolation_error,
            "integrator_status": statuses,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-tracing-fieldlines-ncsx-",
        bounded_steps=50,
        native_default_steps=NATIVE_TMAX,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
