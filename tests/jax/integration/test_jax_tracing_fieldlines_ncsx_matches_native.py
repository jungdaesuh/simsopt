"""The ``1_Simple/tracing_fieldlines_NCSX.py`` workflow: JAX against live native SIMSOPT.

Both lanes trace the same three field lines through the interpolant of the NCSX
coils, with a surface fitted around the magnetic axis as a level-set stop, at a
small scale: natively with ``simsopt.field.tracing.compute_fieldlines`` over
``InterpolatedField``, and with ``compute_fieldlines_with_status`` over
``InterpolatedFieldJAX(BiotSavartJAX)``. The native run is the reference.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import jax
import numpy as np
import pytest
from simsopt.configs import get_data
from simsopt.field import (
    InterpolatedField,
    LevelsetStoppingCriterion,
    SurfaceClassifier,
)
from simsopt.field.tracing import compute_fieldlines
from simsopt.geo import SurfaceRZFourier
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.tracing import compute_fieldlines_with_status

SURFACE_NPHI = 16
SURFACE_NTHETA = 8
GRID_SIZE = 5
INTERPOLATION_DEGREE = 2
FIELDLINE_COUNT = 3
TMAX = 50.0
TOLERANCE = 1.0e-7
SURFACE_DISTANCE = 0.70
CLASSIFIER_H = 0.08
CLASSIFIER_ORDER = 2
SKIP_DISTANCE = -0.05
RADIAL_SPAN = 0.14

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

    def skip(
        radial_values: np.ndarray, phi_values: np.ndarray, height_values: np.ndarray
    ) -> np.ndarray:
        points = np.column_stack((radial_values, phi_values, height_values))
        return (classifier.evaluate_rphiz(points) < SKIP_DISTANCE).reshape(-1)

    interpolation_arguments = (
        INTERPOLATION_DEGREE,
        (float(np.min(radii)), float(np.max(radii)), GRID_SIZE),
        (0.0, 2.0 * np.pi / nfp, 2 * GRID_SIZE),
        (0.0, float(np.max(heights)), GRID_SIZE // 2),
        True,
    )
    return magnetic_axis, native_field, classifier, skip, interpolation_arguments, nfp


def _initial_states(axis_points: np.ndarray) -> np.ndarray:
    radial = np.linspace(
        axis_points[0, 0], axis_points[0, 0] + RADIAL_SPAN, FIELDLINE_COUNT
    )
    return np.column_stack(
        (
            radial,
            np.zeros(FIELDLINE_COUNT),
            np.full(FIELDLINE_COUNT, axis_points[0, 2], dtype=np.float64),
        )
    )


def _phi_planes(nfp: int) -> tuple[float, ...]:
    return tuple(index * 0.5 * np.pi / nfp for index in range(4))


def _values(
    *,
    magnetic_axis,
    native_field,
    initial_states: np.ndarray,
    axis_field: np.ndarray,
    direct_axis_field: np.ndarray,
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    statuses: np.ndarray,
) -> Values:
    return {
        "construction:axis_dofs": np.asarray(
            magnetic_axis.local_full_x, dtype=np.float64
        ),
        "construction:field_dofs": np.asarray(native_field.x, dtype=np.float64),
        "initial:states": initial_states,
        "interpolation:axis_field": axis_field,
        "interpolation:relative_error": np.asarray(
            np.linalg.norm(axis_field - direct_axis_field)
            / np.linalg.norm(direct_axis_field),
            dtype=np.float64,
        ),
        "final:states": np.stack([trajectory[-1, 1:4] for trajectory in trajectories]),
        "final:times": np.asarray(
            [trajectory[-1, 0] for trajectory in trajectories], dtype=np.float64
        ),
        "final:status": np.asarray(statuses, dtype=np.int64),
        "poincare:counts": np.asarray(
            [hits.shape[0] for hits in phi_hits], dtype=np.int64
        ),
        "poincare:positions": np.concatenate(
            [np.asarray(hits[:, 2:5], dtype=np.float64) for hits in phi_hits], axis=0
        ),
    }


def _native_statuses(
    trajectories: list[np.ndarray], phi_hits: list[np.ndarray]
) -> np.ndarray:
    """Native ``compute_fieldlines`` reports no status; derive the core's from its rows.

    ``0`` reached ``tmax``, ``-1 - i`` criterion ``i`` fired (its event row),
    ``1`` neither.
    """
    return np.asarray(
        [
            0
            if np.isclose(trajectory[-1, 0], TMAX, rtol=0.0, atol=1.0e-10)
            else int(hits[hits[:, 1] < 0][-1, 1])
            if hits.ndim == 2 and np.any(hits[:, 1] < 0)
            else 1
            for trajectory, hits in zip(trajectories, phi_hits, strict=True)
        ],
        dtype=np.int64,
    )


def _native_values() -> Values:
    magnetic_axis, native_field, classifier, skip, interpolation_arguments, nfp = (
        _geometry()
    )
    interpolated = InterpolatedField(
        native_field, *interpolation_arguments, nfp=nfp, stellsym=True, skip=skip
    )
    axis_points = np.asarray(magnetic_axis.gamma(), dtype=np.float64)
    initial_states = _initial_states(axis_points)
    native_field.set_points(axis_points)
    interpolated.set_points(axis_points)
    direct_axis_field = np.asarray(native_field.B(), dtype=np.float64)
    axis_field = np.asarray(interpolated.B(), dtype=np.float64)
    trajectories, phi_hits = compute_fieldlines(
        interpolated,
        initial_states[:, 0],
        initial_states[:, 2],
        tmax=TMAX,
        tol=TOLERANCE,
        phis=_phi_planes(nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier.dist)],
    )
    return _values(
        magnetic_axis=magnetic_axis,
        native_field=native_field,
        initial_states=initial_states,
        axis_field=axis_field,
        direct_axis_field=direct_axis_field,
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=_native_statuses(trajectories, phi_hits),
    )


def _jax_values() -> Values:
    magnetic_axis, native_field, classifier, skip, interpolation_arguments, nfp = (
        _geometry()
    )
    source_field = BiotSavartJAX(native_field.coils)
    interpolated = InterpolatedFieldJAX(
        source_field, *interpolation_arguments, nfp=nfp, stellsym=True, skip=skip
    )
    axis_points = np.asarray(magnetic_axis.gamma(), dtype=np.float64)
    initial_states = _initial_states(axis_points)
    source_field.set_points(axis_points)
    interpolated.set_points(axis_points)
    direct_axis_field, axis_field = jax.device_get((source_field.B(), interpolated.B()))
    trajectories, phi_hits, statuses = compute_fieldlines_with_status(
        interpolated,
        initial_states[:, 0],
        initial_states[:, 2],
        tmax=TMAX,
        tol=TOLERANCE,
        phis=_phi_planes(nfp),
        stopping_criteria=[LevelsetStoppingCriterion(classifier)],
    )
    return _values(
        magnetic_axis=magnetic_axis,
        native_field=native_field,
        initial_states=initial_states,
        axis_field=np.asarray(axis_field, dtype=np.float64),
        direct_axis_field=np.asarray(direct_axis_field, dtype=np.float64),
        trajectories=trajectories,
        phi_hits=phi_hits,
        statuses=statuses,
    )


def _lane_is_healthy(values: Values) -> bool:
    """Every published array finite, a usable interpolant, no failed or capped line."""
    return bool(
        all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for value in values.values()
        )
        and float(values["interpolation:relative_error"]) < 0.5
        and np.all(values["final:status"] <= 0)
    )


def test_exact_tracing_fieldlines_ncsx_matches_native_and_jax_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = _native_values()

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax_values = _jax_values()

    assert _lane_is_healthy(native) is True
    assert _lane_is_healthy(jax_values) is True
    assert set(native) == set(jax_values)

    for observable in (
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:axis_field",
        "interpolation:relative_error",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax_values[observable], native[observable], rtol=1.0e-13, atol=1.0e-14
        )

    np.testing.assert_allclose(
        jax_values["final:times"], native["final:times"], rtol=0.0, atol=2.0e-2
    )
    np.testing.assert_allclose(
        jax_values["final:states"], native["final:states"], rtol=0.0, atol=3.0e-2
    )
    np.testing.assert_allclose(
        jax_values["poincare:positions"],
        native["poincare:positions"],
        rtol=0.0,
        atol=3.0e-2,
    )

    assert native["final:status"].tolist() == [0, 0, -1]
    assert native["poincare:counts"].tolist() == [31, 31, 4]
