"""Exact matched workflow for ``1_Simple/tracing_fieldlines_QA.py``."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Protocol

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

REPO_ROOT = Path(__file__).resolve().parents[4]
SURFACE_INPUT = REPO_ROOT / "tests/test_files/input.LandremanPaul2021_QA"
FIELD_INPUT = REPO_ROOT / "examples/1_Simple/inputs/biot_savart_opt.json"

WORKFLOW_STAGES = (
    "load_qa_boundary_and_optimized_stage_two_coils",
    "sample_cylindrical_interpolated_field_with_skip_domain",
    "evaluate_interpolation_error_on_qa_surface",
    "trace_three_fieldlines_with_levelset_stop",
    "record_endpoints_statuses_and_poincare_crossings",
)


class _LevelsetClassifier(Protocol):
    def evaluate_xyz(self, xyz: np.ndarray) -> np.ndarray:
        """Evaluate signed distance at Cartesian points."""


def _scale_configuration(scale: ExecutionScale) -> dict[str, object]:
    native_scale = scale == "native_default"
    return {
        "surface_nphi": 200 if native_scale else 24,
        "surface_ntheta": 30 if native_scale else 10,
        "grid_size": 20 if native_scale else 5,
        "interpolation_degree": 4 if native_scale else 2,
        "fieldline_count": 10 if native_scale else 3,
        "tmax": 20_000.0 if native_scale else 50.0,
        "native_integrator_tolerance": 1.0e-16,
        "jax_integrator_tolerance": 1.0e-13,
        "classifier_h": 0.03 if native_scale else 0.08,
        "classifier_order": 2,
        "skip_distance": -0.05,
        "radial_minimum": 1.2125346,
        "radial_maximum": 1.295,
        "surface_input_sha256": hashlib.sha256(SURFACE_INPUT.read_bytes()).hexdigest(),
        "field_input_sha256": hashlib.sha256(FIELD_INPUT.read_bytes()).hexdigest(),
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


def _qa_objects(bundle: InputBundle):
    import simsopt
    from simsopt.geo import SurfaceRZFourier

    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        nphi=_configuration_int(bundle, "surface_nphi"),
        ntheta=_configuration_int(bundle, "surface_ntheta"),
        range="full torus",
    )
    native_field = simsopt.load(FIELD_INPUT)
    return surface, native_field


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze QA surface/coil DOFs and initial states for every lane."""
    import simsopt
    from simsopt.geo import SurfaceRZFourier

    configuration = _scale_configuration(scale)
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        nphi=int(configuration["surface_nphi"]),
        ntheta=int(configuration["surface_ntheta"]),
        range="full torus",
    )
    native_field = simsopt.load(FIELD_INPUT)
    fieldline_count = int(configuration["fieldline_count"])
    radial_initial = np.linspace(
        float(configuration["radial_minimum"]),
        float(configuration["radial_maximum"]),
        fieldline_count,
    )
    return create_input_bundle(
        root,
        case_id="native-tracing-fieldlines-qa",
        random_seed=0,
        arrays={
            "surface_dofs": np.asarray(surface.local_full_x, dtype=np.float64),
            "field_dofs": np.asarray(native_field.x, dtype=np.float64),
            "surface_points": np.asarray(
                surface.gamma(),
                dtype=np.float64,
            ).reshape((-1, 3)),
            "initial_states": np.column_stack(
                (
                    radial_initial,
                    np.zeros(fieldline_count),
                    np.zeros(fieldline_count),
                )
            ),
            "phi_planes": np.asarray(
                tuple(index * 0.5 * np.pi / surface.nfp for index in range(4)),
                dtype=np.float64,
            ),
        },
        configuration={**configuration, "nfp": int(surface.nfp)},
        scale=scale,
    )


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _effective_fingerprint(
    bundle: InputBundle,
    surface_dofs: np.ndarray,
    field_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "surface_dofs": _array_digest(surface_dofs),
            "field_dofs": _array_digest(field_dofs),
            **bundle.configuration,
        },
    )


def _geometry(bundle: InputBundle):
    from simsopt.field import SurfaceClassifier

    surface, native_field = _qa_objects(bundle)
    classifier = SurfaceClassifier(
        surface,
        h=_configuration_float(bundle, "classifier_h"),
        p=_configuration_int(bundle, "classifier_order"),
    )
    surface_points = surface.gamma()
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    grid_size = _configuration_int(bundle, "grid_size")

    def skip(
        radial_values: np.ndarray,
        phi_values: np.ndarray,
        height_values: np.ndarray,
    ) -> np.ndarray:
        points = np.column_stack((radial_values, phi_values, height_values))
        return (
            classifier.evaluate_rphiz(points)
            < _configuration_float(bundle, "skip_distance")
        ).reshape(-1)

    interpolation_arguments = (
        _configuration_int(bundle, "interpolation_degree"),
        (float(np.min(radii)), float(np.max(radii)), grid_size),
        (0.0, 2.0 * np.pi / surface.nfp, 2 * grid_size),
        (0.0, float(np.max(heights)), grid_size // 2),
        True,
    )
    return surface, native_field, classifier, skip, interpolation_arguments


def _values(
    *,
    surface_dofs: np.ndarray,
    field_dofs: np.ndarray,
    initial_states: np.ndarray,
    surface_field: np.ndarray,
    interpolation_error: float,
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    tmax: float,
    classifier: _LevelsetClassifier,
) -> dict[str, np.ndarray]:
    final_times = np.asarray(
        [trajectory[-1, 0] for trajectory in trajectories],
        dtype=np.float64,
    )
    final_states = np.stack([trajectory[-1, 1:4] for trajectory in trajectories])
    statuses = np.where(np.isclose(final_times, tmax, rtol=0.0, atol=1.0e-10), 0, -1)
    hit_counts = np.asarray([hits.shape[0] for hits in phi_hits], dtype=np.int64)
    hit_positions = np.concatenate(
        [np.asarray(hits[:, 2:5], dtype=np.float64) for hits in phi_hits],
        axis=0,
    )
    return {
        "construction:surface_dofs": surface_dofs,
        "construction:field_dofs": field_dofs,
        "initial:states": initial_states,
        "interpolation:surface_field": surface_field,
        "interpolation:relative_error": np.asarray(
            interpolation_error,
            dtype=np.float64,
        ),
        "final:states": final_states,
        "final:times": final_times,
        "final:status": statuses,
        "final:levelset_distance": np.asarray(
            classifier.evaluate_xyz(final_states),
            dtype=np.float64,
        ).reshape(-1),
        "poincare:counts": hit_counts,
        "poincare:positions": hit_positions,
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
        np.all(np.isfinite(values["final:states"]))
        and np.all(np.isfinite(values["poincare:positions"]))
        and float(values["interpolation:relative_error"]) < 0.5
        and np.all(values["final:status"] <= 0)
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
            values["construction:surface_dofs"],
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
    from simsopt.field.tracing import compute_fieldlines

    surface, native_field, classifier, skip, interpolation_arguments = _geometry(bundle)
    interpolated = InterpolatedField(
        native_field,
        *interpolation_arguments,
        nfp=surface.nfp,
        stellsym=True,
        skip=skip,
    )
    surface_points = arrays["surface_points"]
    native_field.set_points(surface_points)
    interpolated.set_points(surface_points)
    direct_surface_field = np.asarray(native_field.B(), dtype=np.float64)
    surface_field = np.asarray(interpolated.B(), dtype=np.float64)
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        arrays["initial_states"][:, 0],
        arrays["initial_states"][:, 2],
        tmax=_configuration_float(bundle, "tmax"),
        tol=_configuration_float(bundle, "native_integrator_tolerance"),
        phis=tuple(arrays["phi_planes"]),
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
    )
    values = _values(
        surface_dofs=np.asarray(surface.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_states=arrays["initial_states"],
        surface_field=surface_field,
        interpolation_error=float(
            np.linalg.norm(surface_field - direct_surface_field)
            / np.linalg.norm(direct_surface_field)
        ),
        trajectories=trajectories,
        phi_hits=phi_hits,
        tmax=_configuration_float(bundle, "tmax"),
        classifier=classifier,
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsoptpp_lsoda_fieldline",
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt.field import LevelsetStoppingCriterion
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
    from simsopt_jax_adapters.field.tracing import compute_fieldlines

    import jax

    surface, native_field, classifier, skip, interpolation_arguments = _geometry(bundle)
    source_field = BiotSavartJAX(native_field.coils)
    interpolated = InterpolatedFieldJAX(
        source_field,
        *interpolation_arguments,
        nfp=surface.nfp,
        stellsym=True,
        skip=skip,
    )
    surface_points = arrays["surface_points"]
    source_field.set_points(surface_points)
    interpolated.set_points(surface_points)
    direct_surface_field, surface_field = jax.device_get(
        (source_field.B(), interpolated.B())
    )
    direct_surface_field = np.asarray(direct_surface_field, dtype=np.float64)
    surface_field = np.asarray(surface_field, dtype=np.float64)
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        arrays["initial_states"][:, 0],
        arrays["initial_states"][:, 2],
        tmax=_configuration_float(bundle, "tmax"),
        tol=_configuration_float(bundle, "jax_integrator_tolerance"),
        phis=tuple(arrays["phi_planes"]),
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
    )
    values = _values(
        surface_dofs=np.asarray(surface.local_full_x, dtype=np.float64),
        field_dofs=np.asarray(native_field.x, dtype=np.float64),
        initial_states=arrays["initial_states"],
        surface_field=surface_field,
        interpolation_error=float(
            np.linalg.norm(surface_field - direct_surface_field)
            / np.linalg.norm(direct_surface_field)
        ),
        trajectories=trajectories,
        phi_hits=phi_hits,
        tmax=_configuration_float(bundle, "tmax"),
        classifier=classifier,
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver="simsopt_jax_dopri5_fieldline",
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact QA field-line workflow in one solver lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
