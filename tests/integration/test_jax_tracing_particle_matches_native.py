"""The ``1_Simple/tracing_particle.py`` workflow: JAX against live native SIMSOPT.

Both lanes trace the same three seeded vacuum guiding centres through the
interpolant of the NCSX coils, with a surface fitted around the magnetic axis
as a level-set stop, at a small scale: natively with
``simsopt.field.tracing.trace_particles`` over ``InterpolatedField``, and with
``trace_particles_with_status`` over ``InterpolatedFieldJAX(BiotSavartJAX)``.
The native run is the reference.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import inspect
from collections.abc import Callable
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.field import (
    InterpolatedField,
    LevelsetStoppingCriterion,
    SurfaceClassifier,
)
from simsopt.field.sampling import draw_uniform_on_curve
from simsopt.field.tracing import trace_particles
from simsopt.geo import SurfaceRZFourier
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from simsopt_jax_adapters.field import tracing as tracing_adapter
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX

SURFACE_NPHI = 16
SURFACE_NTHETA = 8
GRID_SIZE = 6
INTERPOLATION_DEGREE = 2
PARTICLE_COUNT = 3
TMAX = 1.0e-5
#: An explicit trial budget keeps the small trace short.
JAX_MAX_STEPS = 512
TOLERANCE = 1.0e-9
SURFACE_DISTANCE = 0.20
CLASSIFIER_H = 0.1
CLASSIFIER_ORDER = 2
KINETIC_ENERGY = 5_000.0 * ONE_EV
SPEED_TOTAL = float(np.sqrt(2.0 * KINETIC_ENERGY / PROTON_MASS))

Values = dict[str, np.ndarray]


def _geometry():
    _, _, magnetic_axis, nfp, native_field = get_data("ncsx")
    nfp = int(nfp)
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=SURFACE_NPHI,
        ntheta=SURFACE_NTHETA,
    )
    surface.fit_to_curve(magnetic_axis, SURFACE_DISTANCE, flip_theta=False)
    classifier = SurfaceClassifier(surface, h=CLASSIFIER_H, p=CLASSIFIER_ORDER)
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    interpolation_arguments = (
        INTERPOLATION_DEGREE,
        (float(np.min(radii)), float(np.max(radii)), GRID_SIZE),
        (0.0, 2.0 * np.pi / nfp, 2 * GRID_SIZE),
        (0.0, float(np.max(heights)), GRID_SIZE // 2),
        True,
    )
    return magnetic_axis, nfp, native_field, classifier, interpolation_arguments


def _seeded_particles(magnetic_axis) -> tuple[np.ndarray, np.ndarray]:
    """Pitch first, then positions on the axis, from one ``RandomState(1)`` stream."""
    random_generator = np.random.RandomState(1)
    pitch = random_generator.uniform(-1.0, 1.0, size=PARTICLE_COUNT)
    initial_points, _ = draw_uniform_on_curve(
        magnetic_axis, PARTICLE_COUNT, safetyfactor=10, randomgen=random_generator
    )
    return np.asarray(initial_points, dtype=np.float64), pitch * SPEED_TOTAL


def _phi_planes(nfp: int) -> tuple[float, ...]:
    return tuple(index * 0.5 * np.pi / nfp for index in range(4))


def _values(
    *,
    magnetic_axis,
    native_field,
    initial_points: np.ndarray,
    parallel_speeds: np.ndarray,
    initial_field: np.ndarray,
    final_field_at: Callable[[np.ndarray], np.ndarray],
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    statuses: np.ndarray,
) -> Values:
    final_rows = np.stack([trajectory[-1] for trajectory in trajectories])
    final_positions = final_rows[:, 1:4]
    final_parallel_speeds = final_rows[:, 4]
    magnetic_moments = (SPEED_TOTAL**2 - parallel_speeds**2) / (
        2.0 * np.linalg.norm(initial_field, axis=1)
    )
    final_abs_field = np.linalg.norm(
        final_field_at(np.ascontiguousarray(final_positions, dtype=np.float64)), axis=1
    )
    final_energy_per_mass = (
        0.5 * final_parallel_speeds**2 + magnetic_moments * final_abs_field
    )
    initial_energy_per_mass = 0.5 * SPEED_TOTAL**2
    return {
        "construction:axis_dofs": np.asarray(
            magnetic_axis.local_full_x, dtype=np.float64
        ),
        "construction:field_dofs": np.asarray(native_field.x, dtype=np.float64),
        "initial:states": np.column_stack((initial_points, parallel_speeds)),
        "interpolation:initial_field": initial_field,
        "final:positions": final_positions,
        "final:parallel_speed_fraction": final_parallel_speeds / SPEED_TOTAL,
        "final:times": final_rows[:, 0],
        "final:status": np.asarray(statuses, dtype=np.int64),
        "poincare:counts": np.asarray(
            [hits.shape[0] for hits in phi_hits], dtype=np.int64
        ),
        "poincare:positions": np.concatenate(
            [np.asarray(hits[:, 2:5], dtype=np.float64) for hits in phi_hits], axis=0
        ),
        "conservation:magnetic_moments": magnetic_moments,
        "conservation:energy_relative_error": np.asarray(
            np.max(
                np.abs(final_energy_per_mass - initial_energy_per_mass)
                / initial_energy_per_mass
            ),
            dtype=np.float64,
        ),
    }


def _native_statuses(
    trajectories: list[np.ndarray], phi_hits: list[np.ndarray]
) -> np.ndarray:
    """Native ``trace_particles`` reports no status; derive the core's from its rows.

    ``0`` reached ``tmax``, ``-1 - i`` criterion ``i`` fired (its event row),
    ``1`` neither.
    """
    return np.asarray(
        [
            0
            if np.isclose(trajectory[-1, 0], TMAX, rtol=0.0, atol=1.0e-12)
            else int(hits[hits[:, 1] < 0][-1, 1])
            if hits.ndim == 2 and np.any(hits[:, 1] < 0)
            else 1
            for trajectory, hits in zip(trajectories, phi_hits, strict=True)
        ],
        dtype=np.int64,
    )


def _native_values() -> Values:
    magnetic_axis, nfp, native_field, classifier, interpolation_arguments = _geometry()
    initial_points, parallel_speeds = _seeded_particles(magnetic_axis)
    interpolated = InterpolatedField(
        native_field, *interpolation_arguments, nfp=nfp, stellsym=True
    )
    interpolated.set_points(initial_points)
    initial_field = np.asarray(interpolated.B(), dtype=np.float64)
    trajectories, phi_hits = trace_particles(
        interpolated,
        initial_points,
        parallel_speeds,
        tmax=TMAX,
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
        Ekin=KINETIC_ENERGY,
        tol=TOLERANCE,
        phis=_phi_planes(nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
        mode="gc_vac",
        forget_exact_path=True,
    )

    def final_field_at(points: np.ndarray) -> np.ndarray:
        interpolated.set_points(points)
        return np.asarray(interpolated.B(), dtype=np.float64)

    return _values(
        magnetic_axis=magnetic_axis,
        native_field=native_field,
        initial_points=initial_points,
        parallel_speeds=parallel_speeds,
        initial_field=initial_field,
        final_field_at=final_field_at,
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=_native_statuses(trajectories, phi_hits),
    )


def _jax_values() -> Values:
    magnetic_axis, nfp, native_field, classifier, interpolation_arguments = _geometry()
    initial_points, parallel_speeds = _seeded_particles(magnetic_axis)
    interpolated = InterpolatedFieldJAX(
        BiotSavartJAX(native_field.coils),
        *interpolation_arguments,
        nfp=nfp,
        stellsym=True,
    )
    interpolated.set_points(initial_points)
    initial_field = np.asarray(jax.device_get(interpolated.B()), dtype=np.float64)
    trajectories, phi_hits, statuses = tracing_adapter.trace_particles_with_status(
        interpolated,
        initial_points,
        parallel_speeds,
        tmax=TMAX,
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
        Ekin=KINETIC_ENERGY,
        tol=TOLERANCE,
        phis=_phi_planes(nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
        mode="gc_vac",
        comm=None,
        forget_exact_path=True,
        max_steps=JAX_MAX_STEPS,
    )

    def final_field_at(points: np.ndarray) -> np.ndarray:
        interpolated.set_points(points)
        return np.asarray(jax.device_get(interpolated.B()), dtype=np.float64)

    return _values(
        magnetic_axis=magnetic_axis,
        native_field=native_field,
        initial_points=initial_points,
        parallel_speeds=parallel_speeds,
        initial_field=initial_field,
        final_field_at=final_field_at,
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=statuses,
    )


def _jax_parity_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")


def _lane_is_healthy(values: Values) -> bool:
    """Every published array finite and no failed or capped particle."""
    return bool(
        all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for value in values.values()
        )
        and np.all(values["final:status"] <= 0)
    )


def test_particle_tracer_asks_for_no_trial_budget_by_default() -> None:
    """Upstream bounds neither trials nor steps, so the public tracer may not.

    ``simsoptpp/tracing.cpp`` loops ``do { ... } while(t < tmax && !stop);`` and
    ``simsopt.field.tracing.trace_particles`` takes no step parameter, so the
    JAX status entry point defaults to no budget at all.
    """
    assert not hasattr(tracing_adapter, "_DEFAULT_TRACING_TOTAL_TRIALS")
    assert (
        inspect.signature(tracing_adapter.trace_particles_with_status)
        .parameters["max_steps"]
        .default
        is None
    )


def test_exact_tracing_particle_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_values()
    _jax_parity_environment(monkeypatch)
    jax_values = _jax_values()

    assert _lane_is_healthy(native) is True
    assert _lane_is_healthy(jax_values) is True
    assert set(native) == set(jax_values)

    for observable in (
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:initial_field",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax_values[observable], native[observable], rtol=1.0e-12, atol=1.0e-14
        )

    np.testing.assert_allclose(
        jax_values["final:times"], native["final:times"], rtol=0.0, atol=2.0e-6
    )
    np.testing.assert_allclose(
        jax_values["final:positions"],
        native["final:positions"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax_values["final:parallel_speed_fraction"],
        native["final:parallel_speed_fraction"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax_values["poincare:positions"],
        native["poincare:positions"],
        rtol=0.0,
        atol=2.0e-3,
    )
    assert float(native["conservation:energy_relative_error"]) < 1.0e-3
    assert float(jax_values["conservation:energy_relative_error"]) < 1.0e-3


@pytest.mark.parametrize("positive_status", [1, 2])
def test_particle_tracer_publishes_a_positive_core_status_as_reported(
    monkeypatch: pytest.MonkeyPatch,
    positive_status: int,
) -> None:
    """A budget stop (1) or a step-control failure (2) reaches the caller as-is.

    The chunked core is replaced by one that reports the status directly, so the
    status cannot be re-derived from the end time.
    """
    magnetic_axis, *_ = _geometry()
    initial_points, parallel_speeds = _seeded_particles(magnetic_axis)
    initial = np.column_stack((initial_points, parallel_speeds))
    trajectory = np.stack(
        [
            np.stack((np.r_[0.0, point], np.r_[time, point]))
            for point, time in zip(initial, (TMAX / 2, TMAX / 2, TMAX), strict=True)
        ]
    )
    hits = np.zeros((3, 1, 6), dtype=np.float64)
    hits[1, 0, 1] = -1.0
    core_result = SimpleNamespace(
        trajectory=trajectory,
        mask=np.ones((3, 2), dtype=bool),
        phi_hits=hits,
        phi_hits_count=np.asarray([0, 1, 0]),
        status=np.asarray([positive_status, -1, 0]),
        t_final=np.asarray([TMAX / 2, TMAX / 2, TMAX]),
        steps_taken=np.ones(3, dtype=np.int32),
    )
    monkeypatch.setattr(
        tracing_adapter,
        "_trace_cartesian_chunks",
        lambda *_args, **_kwargs: (
            list(trajectory),
            [hits[0, :0], hits[1, :1], hits[2, :0]],
            core_result,
        ),
    )
    _jax_parity_environment(monkeypatch)

    _trajectories, _phi_hits, statuses = tracing_adapter.trace_particles_with_status(
        ToroidalFieldJAX(1.5, 1.0),
        initial_points,
        parallel_speeds,
        tmax=TMAX,
        mass=PROTON_MASS,
        charge=ELEMENTARY_CHARGE,
        Ekin=KINETIC_ENERGY,
        tol=TOLERANCE,
        mode="gc_vac",
        forget_exact_path=True,
        max_steps=JAX_MAX_STEPS,
    )

    assert statuses.tolist() == [positive_status, -1, 0]
