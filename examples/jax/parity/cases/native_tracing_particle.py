"""Exact matched workflow for ``1_Simple/tracing_particle.py``."""

from __future__ import annotations

import hashlib
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
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV, PROTON_MASS
from simsopt_jax.examples import ExecutionScale

KINETIC_ENERGY = 5_000.0 * ONE_EV
WORKFLOW_STAGES = (
    "construct_ncsx_coils_axis_and_particle_boundary",
    "sample_cylindrical_interpolated_field",
    "draw_seeded_particle_positions_and_pitch",
    "trace_vacuum_guiding_centres_with_levelset_stop",
    "record_endpoints_losses_energy_and_poincare_crossings",
)


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "surface_nphi": 64 if native_scale else 16,
        "surface_ntheta": 24 if native_scale else 8,
        "grid_size": 16 if native_scale else 6,
        "interpolation_degree": 3 if native_scale else 2,
        "particle_count": 100 if native_scale else 3,
        "tmax": 1.0e-2 if native_scale else 1.0e-4,
        "integrator_tolerance": 1.0e-9,
        "surface_distance": 0.20,
        "classifier_h": 0.1,
        "classifier_order": 2,
        "kinetic_energy": KINETIC_ENERGY,
        "mass": PROTON_MASS,
        "charge": ELEMENTARY_CHARGE,
    }


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _ncsx_objects():
    from simsopt.configs import get_data

    _, _, magnetic_axis, nfp, native_field = get_data("ncsx")
    return magnetic_axis, int(nfp), native_field


def _geometry(bundle: InputBundle):
    from simsopt.field import SurfaceClassifier
    from simsopt.geo import SurfaceRZFourier

    magnetic_axis, nfp, native_field = _ncsx_objects()
    surface = SurfaceRZFourier.from_nphi_ntheta(
        mpol=5,
        ntor=5,
        stellsym=True,
        nfp=nfp,
        range="full torus",
        nphi=_configuration_int(bundle, "surface_nphi"),
        ntheta=_configuration_int(bundle, "surface_ntheta"),
    )
    surface.fit_to_curve(
        magnetic_axis,
        _configuration_float(bundle, "surface_distance"),
        flip_theta=False,
    )
    classifier = SurfaceClassifier(
        surface,
        h=_configuration_float(bundle, "classifier_h"),
        p=_configuration_int(bundle, "classifier_order"),
    )
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    grid_size = _configuration_int(bundle, "grid_size")
    interpolation_arguments = (
        _configuration_int(bundle, "interpolation_degree"),
        (float(np.min(radii)), float(np.max(radii)), grid_size),
        (0.0, 2.0 * np.pi / nfp, 2 * grid_size),
        (0.0, float(np.max(heights)), grid_size // 2),
        True,
    )
    return (
        magnetic_axis,
        nfp,
        native_field,
        classifier,
        interpolation_arguments,
    )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze source DOFs and seeded particle initial states for every lane."""
    from simsopt.field.sampling import draw_uniform_on_curve

    configuration = _scale_configuration(scale)
    magnetic_axis, nfp, native_field = _ncsx_objects()
    particle_count = int(configuration["particle_count"])
    speed_total = np.sqrt(
        2.0 * float(configuration["kinetic_energy"])
        / float(configuration["mass"])
    )
    random_generator = np.random.RandomState(1)
    pitch = random_generator.uniform(-1.0, 1.0, size=particle_count)
    parallel_speeds = pitch * speed_total
    initial_points, _ = draw_uniform_on_curve(
        magnetic_axis,
        particle_count,
        safetyfactor=10,
        randomgen=random_generator,
    )
    return create_input_bundle(
        root,
        case_id="native-tracing-particle",
        random_seed=1,
        arrays={
            "axis_dofs": np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
            "field_dofs": np.asarray(native_field.x, dtype=np.float64),
            "initial_points": np.asarray(initial_points, dtype=np.float64),
            "parallel_speeds": np.asarray(parallel_speeds, dtype=np.float64),
            "phi_planes": np.asarray(
                tuple(index * 0.5 * np.pi / nfp for index in range(4)),
                dtype=np.float64,
            ),
        },
        configuration={**configuration, "nfp": nfp},
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "axis_dofs": _array_digest(axis_dofs),
            "field_dofs": _array_digest(field_dofs),
            **bundle.configuration,
        },
    )


def _values(
    *,
    axis_dofs: np.ndarray,
    field_dofs: np.ndarray,
    initial_points: np.ndarray,
    parallel_speeds: np.ndarray,
    initial_field: np.ndarray,
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    tmax: float,
    speed_total: float,
) -> dict[str, np.ndarray]:
    final_rows = np.stack([trajectory[-1] for trajectory in trajectories])
    final_times = final_rows[:, 0]
    final_positions = final_rows[:, 1:4]
    final_parallel_speeds = final_rows[:, 4]
    statuses = np.where(np.isclose(final_times, tmax, rtol=0.0, atol=1.0e-12), 0, -1)
    hit_counts = np.asarray([hits.shape[0] for hits in phi_hits], dtype=np.int64)
    hit_positions = np.concatenate(
        [np.asarray(hits[:, 2:5], dtype=np.float64) for hits in phi_hits],
        axis=0,
    )
    initial_abs_field = np.linalg.norm(initial_field, axis=1)
    magnetic_moments = (speed_total**2 - parallel_speeds**2) / (
        2.0 * initial_abs_field
    )
    return {
        "construction:axis_dofs": axis_dofs,
        "construction:field_dofs": field_dofs,
        "initial:states": np.column_stack((initial_points, parallel_speeds)),
        "interpolation:initial_field": initial_field,
        "final:positions": final_positions,
        "final:parallel_speed_fraction": final_parallel_speeds / speed_total,
        "final:times": final_times,
        "final:status": statuses,
        "poincare:counts": hit_counts,
        "poincare:positions": hit_positions,
        "conservation:magnetic_moments": magnetic_moments,
    }


def _with_energy_error(
    values: dict[str, np.ndarray],
    final_field: np.ndarray,
    speed_total: float,
) -> dict[str, np.ndarray]:
    final_parallel_fraction = values["final:parallel_speed_fraction"]
    final_parallel_speeds = final_parallel_fraction * speed_total
    final_abs_field = np.linalg.norm(final_field, axis=1)
    final_energy_per_mass = (
        0.5 * final_parallel_speeds**2
        + values["conservation:magnetic_moments"] * final_abs_field
    )
    initial_energy_per_mass = 0.5 * speed_total**2
    energy_relative_error = np.asarray(
        np.max(
            np.abs(final_energy_per_mass - initial_energy_per_mass)
            / initial_energy_per_mass
        ),
        dtype=np.float64,
    )
    return {
        **values,
        "conservation:energy_relative_error": energy_relative_error,
    }


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
) -> LaneObservation:
    success = bool(
        np.all(np.isfinite(values["final:positions"]))
        and np.all(np.isfinite(values["final:parallel_speed_fraction"]))
        and np.all(values["final:status"] <= 0)
        and float(values["conservation:energy_relative_error"]) < 1.0e-3
    )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle,
            values["construction:axis_dofs"],
            values["construction:field_dofs"],
        ),
        driver=driver,
        normalized_status="converged" if success else "failed",
        raw_status="integration_complete_or_levelset_stop",
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=values,
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.field import InterpolatedField, LevelsetStoppingCriterion
    from simsopt.field.tracing import trace_particles

    (
        magnetic_axis,
        nfp,
        native_field,
        classifier,
        interpolation_arguments,
    ) = _geometry(bundle)
    interpolated = InterpolatedField(
        native_field,
        *interpolation_arguments,
        nfp=nfp,
        stellsym=True,
    )
    initial_points = arrays["initial_points"]
    interpolated.set_points(initial_points)
    initial_field = np.asarray(interpolated.B(), dtype=np.float64)
    trajectories, phi_hits = trace_particles(
        interpolated,
        initial_points,
        arrays["parallel_speeds"],
        tmax=_configuration_float(bundle, "tmax"),
        mass=_configuration_float(bundle, "mass"),
        charge=_configuration_float(bundle, "charge"),
        Ekin=_configuration_float(bundle, "kinetic_energy"),
        tol=_configuration_float(bundle, "integrator_tolerance"),
        phis=tuple(arrays["phi_planes"]),
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
        mode="gc_vac",
        forget_exact_path=True,
    )
    speed_total = np.sqrt(
        2.0 * _configuration_float(bundle, "kinetic_energy")
        / _configuration_float(bundle, "mass")
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_points=initial_points,
        parallel_speeds=arrays["parallel_speeds"],
        initial_field=initial_field,
        trajectories=trajectories,
        phi_hits=phi_hits,
        tmax=_configuration_float(bundle, "tmax"),
        speed_total=speed_total,
    )
    interpolated.set_points(
        np.ascontiguousarray(values["final:positions"], dtype=np.float64)
    )
    values = _with_energy_error(
        values,
        np.asarray(interpolated.B(), dtype=np.float64),
        speed_total,
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsoptpp_lsoda_guiding_center",
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    import jax
    from simsopt.field import LevelsetStoppingCriterion
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
    from simsopt_jax_adapters.field.tracing import trace_particles

    (
        magnetic_axis,
        nfp,
        native_field,
        classifier,
        interpolation_arguments,
    ) = _geometry(bundle)
    source_field = BiotSavartJAX(native_field.coils)
    interpolated = InterpolatedFieldJAX(
        source_field,
        *interpolation_arguments,
        nfp=nfp,
        stellsym=True,
    )
    initial_points = arrays["initial_points"]
    interpolated.set_points(initial_points)
    initial_field = np.asarray(jax.device_get(interpolated.B()), dtype=np.float64)
    trajectories, phi_hits = trace_particles(
        interpolated,
        initial_points,
        arrays["parallel_speeds"],
        tmax=_configuration_float(bundle, "tmax"),
        mass=_configuration_float(bundle, "mass"),
        charge=_configuration_float(bundle, "charge"),
        Ekin=_configuration_float(bundle, "kinetic_energy"),
        tol=_configuration_float(bundle, "integrator_tolerance"),
        phis=tuple(arrays["phi_planes"]),
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
        mode="gc_vac",
        forget_exact_path=True,
    )
    speed_total = np.sqrt(
        2.0 * _configuration_float(bundle, "kinetic_energy")
        / _configuration_float(bundle, "mass")
    )
    values = _values(
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_points=initial_points,
        parallel_speeds=arrays["parallel_speeds"],
        initial_field=initial_field,
        trajectories=trajectories,
        phi_hits=phi_hits,
        tmax=_configuration_float(bundle, "tmax"),
        speed_total=speed_total,
    )
    interpolated.set_points(
        np.ascontiguousarray(values["final:positions"], dtype=np.float64)
    )
    final_field = np.asarray(jax.device_get(interpolated.B()), dtype=np.float64)
    values = _with_energy_error(values, final_field, speed_total)
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver="simsopt_jax_dopri5_guiding_center",
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact particle-tracing workflow in one solver lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
