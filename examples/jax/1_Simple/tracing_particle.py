"""JAX port of ``examples/1_Simple/tracing_particle.py``.

The host loads the NCSX coils, samples the magnetic field onto the same kind
of cylindrical interpolation grid, and draws proton guiding centres from the
magnetic axis with the native example's deterministic seed.  Interpolated
field evaluation and guiding-centre integration run on the selected JAX
device.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simsopt.configs import get_data
from simsopt.field import LevelsetStoppingCriterion, SurfaceClassifier
from simsopt.field.sampling import draw_uniform_on_curve
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.tracing import trace_particles

EXAMPLE_ID = "native-tracing-particle"
NATIVE_PARTICLE_COUNT = 100
NATIVE_TRACE_TIME = 1.0e-2
BOUNDED_TRACE_TIME = 1.0e-4
KINETIC_ENERGY = 5_000.0 * ONE_EV


def _problem(particle_count: int):
    _base_curves, _currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    source_field = BiotSavartJAX(native_field.coils)
    native_scale = particle_count >= NATIVE_PARTICLE_COUNT
    nphi, ntheta = (64, 24) if native_scale else (16, 8)
    grid_size = 16 if native_scale else 6
    degree = 3 if native_scale else 2
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=nphi,
        ntheta=ntheta,
    )
    surface.fit_to_curve(magnetic_axis, 0.20, flip_theta=False)
    radii = np.linalg.norm(surface.gamma()[:, :, :2], axis=2)
    heights = surface.gamma()[:, :, 2]
    interpolated = InterpolatedFieldJAX(
        source_field,
        degree,
        (float(np.min(radii)), float(np.max(radii)), grid_size),
        (0.0, 2.0 * np.pi / nfp, 2 * grid_size),
        (0.0, float(np.max(heights)), grid_size // 2),
        True,
        nfp=nfp,
        stellsym=True,
    )
    return magnetic_axis, surface, interpolated, nfp


def _field_strength(field: InterpolatedFieldJAX, points: np.ndarray) -> np.ndarray:
    field.set_points(np.ascontiguousarray(points, dtype=np.float64))
    return np.linalg.norm(np.asarray(field.B(), dtype=np.float64), axis=1)


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    particle_count = int(max_steps)
    magnetic_axis, surface, interpolated, nfp = _problem(particle_count)
    trace_time = (
        NATIVE_TRACE_TIME
        if particle_count >= NATIVE_PARTICLE_COUNT
        else BOUNDED_TRACE_TIME
    )

    speed_total = np.sqrt(2.0 * KINETIC_ENERGY / PROTON_MASS)
    random_generator = np.random.RandomState(1)
    pitch = random_generator.uniform(-1.0, 1.0, size=particle_count)
    parallel_speeds = pitch * speed_total
    initial_points, _ = draw_uniform_on_curve(
        magnetic_axis,
        particle_count,
        safetyfactor=10,
        randomgen=random_generator,
    )
    initial_points = np.asarray(initial_points, dtype=np.float64)
    initial_field_strength = _field_strength(interpolated, initial_points)
    magnetic_moments = (speed_total**2 - parallel_speeds**2) / (
        2.0 * initial_field_strength
    )

    classifier = SurfaceClassifier(
        surface,
        h=0.1,
        p=2,
    )
    phis = tuple(index * 0.5 * np.pi / nfp for index in range(4))
    trajectories, _phi_hits = trace_particles(
        interpolated,
        initial_points,
        parallel_speeds,
        tmax=trace_time,
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
        Ekin=KINETIC_ENERGY,
        tol=1.0e-9,
        phis=phis,
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
        mode="gc_vac",
        forget_exact_path=True,
    )

    final_rows = np.stack([trajectory[-1] for trajectory in trajectories])
    final_points = final_rows[:, 1:4]
    final_parallel_speeds = final_rows[:, 4]
    final_field_strength = _field_strength(interpolated, final_points)
    initial_energy_per_mass = 0.5 * speed_total**2
    final_energy_per_mass = (
        0.5 * final_parallel_speeds**2 + magnetic_moments * final_field_strength
    )
    energy_relative_error = float(
        np.max(
            np.abs(final_energy_per_mass - initial_energy_per_mass)
            / initial_energy_per_mass
        )
    )
    final_times = final_rows[:, 0]
    statuses = tuple(
        0 if np.isclose(time, trace_time, rtol=0.0, atol=1.0e-14) else -1
        for time in final_times
    )
    scientific_success = bool(
        len(trajectories) == particle_count
        and np.all(np.isfinite(final_rows))
        and np.isfinite(energy_relative_error)
        and energy_relative_error < 1.0e-3
        and all(status <= 0 for status in statuses)
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_state": tuple(
                (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(parallel_speed),
                )
                for point, parallel_speed in zip(
                    initial_points,
                    parallel_speeds,
                    strict=True,
                )
            ),
            "final_state": tuple(
                tuple(float(value) for value in row[1:]) for row in final_rows
            ),
            "energy_relative_error": energy_relative_error,
            "integrator_status": statuses,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-tracing-particle-",
        bounded_steps=3,
        native_default_steps=NATIVE_PARTICLE_COUNT,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
