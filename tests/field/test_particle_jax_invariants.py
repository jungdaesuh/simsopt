"""JAX particle-tracing invariants mirrored from legacy particle tests."""

from __future__ import annotations

import numpy as np

from simsopt.field.tracing import compute_gc_radius
from simsopt_jax_adapters.field.poloidal_field import PoloidalFieldJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX
from simsopt_jax_adapters.field.tracing import trace_particles


def test_guiding_center_and_fullorbit_remain_within_gyroradius_bound():
    """Mirror legacy GC-vs-full-orbit proximity on public JAX routes."""

    major_radius = 1.3
    field_strength = 0.8
    mass = 0.01
    charge = 1.0
    speed_total = 1.0
    kinetic_energy = 0.5 * mass * speed_total * speed_total
    initial_xyz = np.array(
        [
            [1.4, 0.0, 0.0],
            [1.45, 0.0, 0.01],
        ],
        dtype=np.float64,
    )
    parallel_speeds = np.array([0.45, 0.25], dtype=np.float64) * speed_total
    field = ToroidalFieldJAX(major_radius, field_strength)

    guiding_center_trajectories, _gc_phi_hits = trace_particles(
        field,
        initial_xyz,
        parallel_speeds,
        tmax=0.02,
        mass=mass,
        charge=charge,
        Ekin=kinetic_energy,
        tol=1.0e-10,
        mode="gc_vac",
    )
    fullorbit_trajectories, _fo_phi_hits = trace_particles(
        field,
        initial_xyz,
        parallel_speeds,
        tmax=0.02,
        mass=mass,
        charge=charge,
        Ekin=kinetic_energy,
        tol=1.0e-10,
        mode="full",
        phase_angle=0.0,
    )

    max_radius_ratio = 0.0
    for guiding_center, fullorbit in zip(
        guiding_center_trajectories, fullorbit_trajectories, strict=True
    ):
        assert guiding_center.shape[0] >= 2
        assert fullorbit.shape[0] > guiding_center.shape[0]
        for guiding_center_row in guiding_center:
            fullorbit_index = np.argmin(
                np.abs(guiding_center_row[0] - fullorbit[:, 0])
            )
            guiding_center_xyz = guiding_center_row[1:4]
            fullorbit_xyz = fullorbit[fullorbit_index, 1:4]
            fullorbit_velocity = fullorbit[fullorbit_index, 4:7]

            field.set_points_cart(np.ascontiguousarray(guiding_center_xyz.reshape(1, 3)))
            B = field.B()[0]
            abs_B = field.AbsB()[0, 0]
            B_unit = B / abs_B
            perpendicular_velocity = np.linalg.norm(
                fullorbit_velocity
                - np.sum(fullorbit_velocity * B_unit) * B_unit
            )
            guiding_center_radius = compute_gc_radius(
                mass, perpendicular_velocity, charge, abs_B
            )
            distance = np.linalg.norm(guiding_center_xyz - fullorbit_xyz)
            max_radius_ratio = max(max_radius_ratio, distance / guiding_center_radius)

    assert max_radius_ratio < 1.01


def test_guiding_center_conserves_energy_and_magnetic_moment():
    """Mirror the legacy guiding-center energy and moment invariant."""

    major_radius = 1.3
    field_strength = 0.8
    mass = 1.0
    charge = 1.0
    speed_total = 1.0
    kinetic_energy = 0.5 * mass * speed_total * speed_total
    initial_xyz = np.array(
        [
            [1.4, 0.0, 0.0],
            [1.45, 0.0, 0.02],
        ],
        dtype=np.float64,
    )
    parallel_speeds = np.array([-0.45, -0.25], dtype=np.float64) * speed_total
    field = ToroidalFieldJAX(major_radius, field_strength)

    trajectories, _phi_hits = trace_particles(
        field,
        initial_xyz,
        parallel_speeds,
        tmax=0.2,
        mass=mass,
        charge=charge,
        Ekin=kinetic_energy,
        tol=1.0e-10,
        mode="gc_vac",
    )

    max_energy_error = 0.0
    max_moment_error = 0.0
    for trajectory, initial_parallel_speed in zip(
        trajectories, parallel_speeds, strict=True
    ):
        assert trajectory.shape[0] >= 2
        xyz = np.ascontiguousarray(trajectory[:, 1:4])
        field.set_points_cart(xyz)
        abs_B = field.AbsB().reshape(-1)
        parallel_velocity = trajectory[:, 4]
        initial_moment = (
            speed_total * speed_total - initial_parallel_speed * initial_parallel_speed
        ) / (2.0 * abs_B[0])

        energy = mass * (0.5 * parallel_velocity * parallel_velocity + initial_moment * abs_B)
        moment_from_energy = (
            energy[0] / (mass * abs_B)
            - 0.5 * parallel_velocity * parallel_velocity / abs_B
        )
        max_energy_error = max(
            max_energy_error,
            np.max(np.abs(energy - energy[0])) / np.abs(energy[0]),
        )
        max_moment_error = max(
            max_moment_error,
            np.max(np.abs(moment_from_energy - initial_moment))
            / np.abs(initial_moment),
        )

    assert max_energy_error < 1.0e-10
    assert max_moment_error < 1.0e-10


def test_guiding_center_angular_momentum_conservation_axisymmetric_field():
    """Mirror legacy canonical angular-momentum conservation on the JAX route."""

    major_radius = 1.0
    field_strength = 2.0
    safety_factor = 3.1
    mass = 1.0
    charge = 1.0
    speed_total = 1.2
    kinetic_energy = 0.5 * mass * speed_total * speed_total

    phis = np.linspace(0.0, 2.0 * np.pi, 4, endpoint=False)
    initial_radius = major_radius + 0.08
    initial_z = 0.03
    initial_xyz = np.column_stack(
        [
            initial_radius * np.cos(phis),
            initial_radius * np.sin(phis),
            np.full(phis.shape, initial_z),
        ]
    )
    parallel_speeds = np.linspace(-0.6, -0.3, initial_xyz.shape[0]) * speed_total
    field = ToroidalFieldJAX(
        major_radius, field_strength
    ) + PoloidalFieldJAX(major_radius, field_strength, safety_factor)

    trajectories, _phi_hits = trace_particles(
        field,
        initial_xyz,
        parallel_speeds,
        tmax=0.1,
        mass=mass,
        charge=charge,
        Ekin=kinetic_energy,
        tol=1.0e-10,
        mode="gc_vac",
    )

    max_relative_pphi_error = 0.0
    max_relative_flux_term_error = 0.0
    for trajectory in trajectories:
        assert trajectory.shape[0] >= 2
        xyz = np.ascontiguousarray(trajectory[:, 1:4])
        field.set_points_cart(xyz)
        abs_B = field.AbsB().reshape(-1)
        parallel_velocity = trajectory[:, 4]

        cylindrical_radius = np.linalg.norm(xyz[:, :2], axis=1)
        minor_radius_squared = (
            (cylindrical_radius - major_radius) ** 2 + xyz[:, 2] ** 2
        )
        poloidal_flux = field_strength * minor_radius_squared / safety_factor / 2.0
        flux_term = charge * poloidal_flux
        parallel_term = mass * parallel_velocity * field_strength * major_radius / abs_B
        canonical_pphi = flux_term + parallel_term

        relative_pphi_error = np.max(
            np.abs(canonical_pphi - canonical_pphi[0])
        ) / np.abs(canonical_pphi[0])
        relative_flux_term_error = np.max(
            np.abs(flux_term - flux_term[0])
        ) / np.abs(flux_term[0])
        max_relative_pphi_error = max(max_relative_pphi_error, relative_pphi_error)
        max_relative_flux_term_error = max(
            max_relative_flux_term_error, relative_flux_term_error
        )

    assert max_relative_flux_term_error > 0.1
    assert max_relative_pphi_error < 1.0e-3
