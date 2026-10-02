"""Shared post-optimization diagnostics for the MUSE permanent-magnet example.

Official source: ``2_Intermediate/permanent_magnet_MUSE.py`` at upstream
``9e027eac38028d57aa23777be52a781aa860e347``. After GPMO the official script
evaluates the dipole squared flux and the total magnet volume on the separate
plotting surface (lines 243-255) and then traces field lines of the **TF coils
alone** (``bs = BiotSavart(coils)``, line 79) through an interpolant built at
lines 316-345, with 30 starts, ``tmax_fl = 20000``, ``tol = 1e-16``, degree 2,
``n = 20`` and ``SurfaceClassifier(s, h=.03, p=2)``.

The metric half is pure ``simsopt`` and is byte-identical in every lane. The
tracing half is the lane's own integrator: ``run_native_muse_post_workflow``
runs ``simsopt.field.compute_fieldlines`` -- the function the official capture
records as ``trace:function`` -- and ``run_jax_muse_post_workflow`` runs the
ported tracer, so a native lane never reports a JAX result as its own.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from simsopt.field import (
    BiotSavart,
    DipoleField,
    InterpolatedField,
    LevelsetStoppingCriterion,
    SurfaceClassifier,
)
from simsopt.field.tracing import compute_fieldlines
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils_for_pm_optimization,
)
from simsopt_jax.examples import ExecutionScale

from simsopt_jax_adapters.field.interpolated import InterpolatedFieldJAX
from simsopt_jax_adapters.field.tracing import compute_fieldlines_with_status

MUSE_MAGNETIC_FIELD_LIMIT = 1.465
MUSE_VACUUM_PERMEABILITY = 4.0 * np.pi * 1.0e-7

#: Workflow stages the post-GPMO half of the official script performs. Every
#: lane that runs them appends both, so the arbiter's one stage contract holds.
MUSE_POST_WORKFLOW_STAGES: tuple[str, ...] = (
    "evaluate_post_optimization_dipole_squared_flux_and_total_volume",
    "trace_interpolated_tf_coil_fieldlines",
)


@dataclass(frozen=True)
class MusePostWorkflowPolicy:
    """Resolution and tracing policy for one MUSE post-workflow scale."""

    fieldline_count: int
    fieldline_tmax: int
    interpolation_grid_size: int
    interpolation_degree: int = 2
    fieldline_tolerance: float = 1.0e-16
    classifier_h: float = 0.03
    classifier_p: int = 2
    skip_distance: float = -0.05


@dataclass(frozen=True)
class MusePostGeometry:
    """MUSE surfaces and TF-coil field shared by post diagnostics."""

    optimization_surface: SurfaceRZFourier
    plotting_surface: SurfaceRZFourier
    coil_field: BiotSavart


@dataclass(frozen=True)
class MusePostDiagnostics:
    """Host diagnostics from the post-GPMO MUSE workflow."""

    dipole_squared_flux: float
    total_magnet_volume: float
    trace_statuses: tuple[int, ...]
    trace_final_times: tuple[float, ...]
    trace_hit_counts: tuple[int, ...]
    #: Size of each line's LAST accepted step. Upstream fires a stopping
    #: criterion on the accepted post-step state and keeps the pre-crossing row
    #: (``simsoptpp/tracing.cpp:387-443``), so a criterion stop time is only
    #: defined to within one accepted step; this is that scale, per line, and
    #: it is the bound a cross-lane comparison of ``trace_final_times`` may use.
    #: It is a diagnostic of the lane's own integrator, not a published
    #: observable.
    trace_final_steps: tuple[float, ...]

    @property
    def trace_success(self) -> bool:
        """Every line ended at the level set or ran its full horizon.

        Official MUSE asserts nothing about the trace, so this is a reported
        diagnostic, never a completion gate for the official workflow. The
        official run ends all 30 lines at the level set (status -1).
        """

        return all(status <= 0 for status in self.trace_statuses)


@dataclass(frozen=True)
class _MuseTracingSetup:
    """Official tracing settings resolved once, shared by both integrators."""

    classifier: SurfaceClassifier
    degree: int
    radial_range: tuple[float, float, int]
    toroidal_range: tuple[float, float, int]
    vertical_range: tuple[float, float, int]
    extrapolate: bool
    skip: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    nfp: int
    radial_initial: np.ndarray
    vertical_initial: np.ndarray
    phis: tuple[float, ...]


def muse_post_workflow_policy(scale: ExecutionScale) -> MusePostWorkflowPolicy:
    """Return official or explicitly reduced MUSE post-workflow settings."""
    if scale == "native_default":
        return MusePostWorkflowPolicy(
            fieldline_count=30,
            fieldline_tmax=20_000,
            interpolation_grid_size=20,
        )
    return MusePostWorkflowPolicy(
        fieldline_count=3,
        fieldline_tmax=50,
        interpolation_grid_size=5,
    )


def build_muse_post_geometry(
    surface_path: Path,
    test_data: Path,
    *,
    nphi: int,
    ntheta: int,
) -> MusePostGeometry:
    """Build the official MUSE surface pair and TF-coil field once per workflow."""
    optimization_surface = SurfaceRZFourier.from_focus(
        surface_path,
        range="half period",
        nphi=nphi,
        ntheta=ntheta,
    )
    qphi = 2 * nphi
    plotting_surface = SurfaceRZFourier.from_focus(
        surface_path,
        quadpoints_phi=np.linspace(0.0, 1.0, qphi, endpoint=True),
        quadpoints_theta=np.linspace(0.0, 1.0, ntheta, endpoint=True),
    )
    _, _, coils = initialize_coils_for_pm_optimization(
        "muse_famus",
        test_data,
        optimization_surface,
    )
    return MusePostGeometry(
        optimization_surface=optimization_surface,
        plotting_surface=plotting_surface,
        coil_field=BiotSavart(coils),
    )


def _post_metrics(
    geometry: MusePostGeometry,
    final_moments: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    moment_maxima: np.ndarray,
    coordinate_flag: str,
) -> tuple[float, float]:
    """Official ``f_B_sf`` on the plotting surface and the total magnet volume."""
    plotting_surface = geometry.plotting_surface
    plotting_points = np.asarray(plotting_surface.gamma(), dtype=np.float64).reshape(
        (-1, 3)
    )
    geometry.coil_field.set_points(plotting_points)
    coil_normal_field = np.sum(
        np.asarray(geometry.coil_field.B(), dtype=np.float64).reshape(
            (*plotting_surface.unitnormal().shape,)
        )
        * plotting_surface.unitnormal(),
        axis=2,
    )
    dipole_field = DipoleField(
        dipole_grid_xyz,
        final_moments,
        nfp=geometry.optimization_surface.nfp,
        coordinate_flag=coordinate_flag,
        m_maxima=moment_maxima,
    )
    dipole_squared_flux = float(
        SquaredFlux(plotting_surface, dipole_field, -coil_normal_field).J()
    )
    total_magnet_volume = float(
        np.sum(np.linalg.norm(final_moments, axis=1))
        * geometry.optimization_surface.nfp
        * 2.0
        * MUSE_VACUUM_PERMEABILITY
        / MUSE_MAGNETIC_FIELD_LIMIT
    )
    return dipole_squared_flux, total_magnet_volume


def _tracing_setup(
    geometry: MusePostGeometry,
    policy: MusePostWorkflowPolicy,
) -> _MuseTracingSetup:
    """Resolve the official interpolant grid, classifier, skip rule and starts."""
    classifier = SurfaceClassifier(
        geometry.optimization_surface,
        h=policy.classifier_h,
        p=policy.classifier_p,
    )
    surface_points = np.asarray(geometry.optimization_surface.gamma(), dtype=np.float64)
    radii = np.linalg.norm(surface_points[:, :, :2], axis=2)
    heights = surface_points[:, :, 2]
    nfp = int(geometry.optimization_surface.nfp)

    def skip(
        radial_values: np.ndarray,
        phi_values: np.ndarray,
        height_values: np.ndarray,
    ) -> np.ndarray:
        cylindrical_points = np.column_stack((radial_values, phi_values, height_values))
        return (
            classifier.evaluate_rphiz(cylindrical_points) < policy.skip_distance
        ).reshape(-1)

    return _MuseTracingSetup(
        classifier=classifier,
        degree=policy.interpolation_degree,
        radial_range=(
            float(np.min(radii)),
            float(np.max(radii)),
            policy.interpolation_grid_size,
        ),
        toroidal_range=(
            0.0,
            2.0 * np.pi / nfp,
            2 * policy.interpolation_grid_size,
        ),
        vertical_range=(
            0.0,
            float(np.max(heights)),
            policy.interpolation_grid_size // 2,
        ),
        extrapolate=True,
        skip=skip,
        nfp=nfp,
        radial_initial=np.linspace(0.32, 0.36, policy.fieldline_count),
        vertical_initial=np.zeros(policy.fieldline_count, dtype=np.float64),
        # Official ``phis = [(i/4)*(2*pi/nfp) for i in range(4)]``.
        phis=tuple(index * 0.5 * np.pi / nfp for index in range(4)),
    )


def _core_statuses_from_hits(
    final_times: np.ndarray,
    phi_hits: list[np.ndarray],
    tmax: float,
) -> tuple[int, ...]:
    """Core terminal status per line, for an integrator that returns none.

    ``simsopt.field.compute_fieldlines`` returns only trajectories and hits, so
    the status is read from the returned data: a line that reached ``tmax`` is
    ``0``, a line whose last recorded hit carries a
    negative index stopped on that criterion, anything else is ``1``.
    """

    return tuple(
        0
        if np.isclose(time, tmax, rtol=0.0, atol=1.0e-10)
        else int(hits[hits[:, 1] < 0][-1, 1])
        if hits.ndim == 2 and np.any(hits[:, 1] < 0)
        else 1
        for time, hits in zip(final_times, phi_hits, strict=True)
    )


def _diagnostics(
    dipole_squared_flux: float,
    total_magnet_volume: float,
    trajectories: list[np.ndarray],
    phi_hits: list[np.ndarray],
    statuses: tuple[int, ...],
) -> MusePostDiagnostics:
    return MusePostDiagnostics(
        dipole_squared_flux=dipole_squared_flux,
        total_magnet_volume=total_magnet_volume,
        trace_statuses=statuses,
        trace_final_times=tuple(
            float(trajectory[-1, 0]) for trajectory in trajectories
        ),
        trace_hit_counts=tuple(int(hits.shape[0]) for hits in phi_hits),
        trace_final_steps=tuple(
            float(trajectory[-1, 0] - trajectory[-2, 0])
            if trajectory.shape[0] > 1
            else float(trajectory[-1, 0])
            for trajectory in trajectories
        ),
    )


def run_native_muse_post_workflow(
    geometry: MusePostGeometry,
    policy: MusePostWorkflowPolicy,
    final_moments: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    moment_maxima: np.ndarray,
    *,
    coordinate_flag: str,
) -> MusePostDiagnostics:
    """Official MUSE post metrics and coil-only trace through native simsopt."""
    dipole_squared_flux, total_magnet_volume = _post_metrics(
        geometry,
        final_moments,
        dipole_grid_xyz,
        moment_maxima,
        coordinate_flag,
    )
    setup = _tracing_setup(geometry, policy)
    interpolated_coil_field = InterpolatedField(
        geometry.coil_field,
        setup.degree,
        setup.radial_range,
        setup.toroidal_range,
        setup.vertical_range,
        setup.extrapolate,
        nfp=setup.nfp,
        stellsym=True,
        skip=setup.skip,
    )
    trajectories, phi_hits = compute_fieldlines(
        interpolated_coil_field,
        setup.radial_initial,
        setup.vertical_initial,
        tmax=policy.fieldline_tmax,
        tol=policy.fieldline_tolerance,
        phis=setup.phis,
        # Official line 302 passes the classifier's distance function.
        stopping_criteria=[LevelsetStoppingCriterion(setup.classifier.dist)],
    )
    final_times = np.asarray(
        [trajectory[-1, 0] for trajectory in trajectories],
        dtype=np.float64,
    )
    return _diagnostics(
        dipole_squared_flux,
        total_magnet_volume,
        trajectories,
        phi_hits,
        _core_statuses_from_hits(final_times, phi_hits, float(policy.fieldline_tmax)),
    )


def run_jax_muse_post_workflow(
    geometry: MusePostGeometry,
    policy: MusePostWorkflowPolicy,
    final_moments: np.ndarray,
    dipole_grid_xyz: np.ndarray,
    moment_maxima: np.ndarray,
    *,
    coordinate_flag: str,
) -> MusePostDiagnostics:
    """Official MUSE post metrics and coil-only trace through the JAX tracer."""
    dipole_squared_flux, total_magnet_volume = _post_metrics(
        geometry,
        final_moments,
        dipole_grid_xyz,
        moment_maxima,
        coordinate_flag,
    )
    setup = _tracing_setup(geometry, policy)
    interpolated_coil_field = InterpolatedFieldJAX(
        geometry.coil_field,
        setup.degree,
        setup.radial_range,
        setup.toroidal_range,
        setup.vertical_range,
        setup.extrapolate,
        nfp=setup.nfp,
        stellsym=True,
        skip=setup.skip,
    )
    trajectories, phi_hits, core_statuses = compute_fieldlines_with_status(
        interpolated_coil_field,
        setup.radial_initial,
        setup.vertical_initial,
        tmax=policy.fieldline_tmax,
        tol=policy.fieldline_tolerance,
        phis=setup.phis,
        stopping_criteria=[LevelsetStoppingCriterion(setup.classifier)],
    )
    return _diagnostics(
        dipole_squared_flux,
        total_magnet_volume,
        trajectories,
        phi_hits,
        tuple(int(status) for status in core_statuses),
    )
