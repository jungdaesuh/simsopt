"""Host bootstrap for the NCSX VMEC-free single-stage Boozer vacuum problem.

``examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`` fits a
volume-labelled exact Boozer surface to the NCSX magnetic axis and minimizes
non-quasisymmetry + Boozer residual + iota, major-radius and coil-length
penalties over the coil degrees of freedom.  This module owns that construction
and one outer evaluation of it on the exact analytic route
(:mod:`simsopt_jax_adapters.geo.single_stage_exact_analytic`); the outer
optimizer belongs to the caller.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, cast

import numpy as np
from numpy.typing import NDArray
from simsopt.configs import get_data
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import Curve
from simsopt_jax.runtime.host_boundary import host_float

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    ExactAnalyticSingleStage,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    make_traceable_objective_runtime_bundle,
)

__all__ = [
    "BOUNDED_SCALE",
    "NATIVE_SCALE",
    "NCSX_INITIAL_IOTA",
    "SingleStageVacuumEndpoint",
    "SingleStageVacuumProblem",
    "SingleStageVacuumScale",
]

# The rotational transform the native example seeds its first exact solve with.
NCSX_INITIAL_IOTA: Final = -0.406
# Native's ``BoozerSurface`` solver options, shared by both scales.
_INNER_SOLVER_OPTIONS: Final = MappingProxyType(
    {"newton_maxiter": 20, "newton_tol": 1.0e-13, "verbose": False}
)
# Native fits the seed surface this far from the magnetic axis.
_SURFACE_DISTANCE: Final = 0.1


@dataclass(frozen=True, slots=True)
class SingleStageVacuumScale:
    """One execution scale of the native example.

    ``surface_resolution`` is the Boozer surface ``mpol``/``ntor``, which also
    sets its ``2 * surface_resolution + 1`` quadrature points per direction;
    ``non_qs_resolution`` is native's ``NonQuasiSymmetricRatio(sDIM=...)``;
    ``ncsx_options`` are the ``get_data("ncsx", ...)`` coil and axis
    truncations (empty for the full shipped configuration).
    """

    surface_resolution: int
    non_qs_resolution: int
    ncsx_options: Mapping[str, int]


NATIVE_SCALE: Final = SingleStageVacuumScale(
    surface_resolution=6,
    non_qs_resolution=20,
    ncsx_options=MappingProxyType({}),
)
BOUNDED_SCALE: Final = SingleStageVacuumScale(
    surface_resolution=1,
    non_qs_resolution=4,
    ncsx_options=MappingProxyType(
        {"coil_order": 3, "magnetic_axis_order": 3, "points_per_period": 8}
    ),
)


@dataclass(frozen=True, slots=True)
class SingleStageVacuumEndpoint:
    """One evaluated coil state: reported objective, gradient and physics.

    ``value`` and ``gradient`` carry native's failed-inner-solve policy, so a
    ``value`` of ``INNER_FAILURE_VALUE`` with ``inner_success`` false is the
    sentinel, not a physical objective.  The physics fields describe the Boozer
    state the evaluation left in place.
    """

    value: float
    gradient: NDArray[np.float64]
    inner_success: bool
    iota: float
    volume: float
    non_qs_ratio: float
    boozer_residual: float

    @property
    def boozer_residual_rms(self) -> float:
        """Root mean square of the Boozer residual vector.

        ``boozer_residual`` is half the mean square of that vector, which is
        what native's ``BoozerResidual.J()`` returns.
        """
        return float(np.sqrt(2.0 * self.boozer_residual))


def _outer_objective_config(
    *,
    nfp: int,
    surface: SurfaceXYZTensorFourier,
    non_qs_resolution: int,
    major_radius_target: float,
    length_target: float,
) -> dict[str, object]:
    """The native objective: non-QS + residual + iota, radius and length penalties."""
    return {
        "non_qs_weight": 1.0,
        "residual_weight": 1.0,
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.linspace(
            0.0, 1.0 / nfp, 2 * non_qs_resolution, endpoint=False
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0, 1.0, 2 * non_qs_resolution, endpoint=False
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }


class SingleStageVacuumProblem:
    """The native single-stage vacuum problem: coil dofs in, objective out.

    Construction builds the NCSX coil graph and the volume-labelled surface at
    ``scale``, solves the seed Boozer surface, and freezes the iota,
    major-radius and coil-length targets at that solution, exactly as the native
    example does.  ``value_and_gradient`` is the ``jac=True`` outer callable and
    carries native's failed-inner-solve sentinel and warm-start restore;
    ``endpoint`` evaluates one coil state and adds the published physics.

    Construction is this problem's host boundary: it is where NCSX host data
    becomes device arrays.  Every crossing there is an explicit
    ``jax.device_put``, so construction runs under
    ``jax.transfer_guard_host_to_device("disallow")`` and the steady-state
    evaluation and reporting paths run under the full
    ``jax.transfer_guard("disallow")``.
    ``tests/jax/examples/test_single_stage_boozer_vacuum_problem.py`` pins both.
    """

    def __init__(self, scale: SingleStageVacuumScale) -> None:
        base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
            "ncsx", **scale.ncsx_options
        )
        magnetic_axis = cast(Curve, magnetic_axis)
        base_currents[0].fix_all()
        field = BiotSavartJAX(native_field.coils)
        current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
        initial_G = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
        resolution = scale.surface_resolution
        quadrature_points = 2 * resolution + 1
        surface = SurfaceXYZTensorFourier(
            mpol=resolution,
            ntor=resolution,
            stellsym=True,
            nfp=nfp,
            quadpoints_phi=np.linspace(
                0.0, 1.0 / nfp, quadrature_points, endpoint=False
            ),
            quadpoints_theta=np.linspace(0.0, 1.0, quadrature_points, endpoint=False),
        )
        surface.fit_to_curve(magnetic_axis, _SURFACE_DISTANCE, flip_theta=True)
        volume_label = Volume(surface)
        boozer_surface = BoozerSurfaceJAX(
            field,
            surface,
            volume_label,
            float(volume_label.J()),
            options=dict(_INNER_SOLVER_OPTIONS),
        )
        length_target = float(sum(CurveLength(curve).J() for curve in base_curves))

        def outer_objective_config() -> dict[str, object]:
            # The evaluator calls this once, after its initial exact solve has
            # been published as the surface's state: native likewise reads the
            # major-radius target and the vessel geometry off the solved surface.
            return _outer_objective_config(
                nfp=int(nfp),
                surface=surface,
                non_qs_resolution=scale.non_qs_resolution,
                major_radius_target=float(surface.major_radius()),
                length_target=length_target,
            )

        evaluator = ExactAnalyticSingleStage(
            boozer_surface,
            field,
            iota=NCSX_INITIAL_IOTA,
            G=initial_G,
            outer_objective_config=outer_objective_config,
        )
        self._evaluator = evaluator
        self.initial_coil_dofs: NDArray[np.float64] = np.array(
            evaluator.coil_dofs, copy=True
        )
        self.iota_target: float = evaluator.iota_target
        self.value_and_gradient = evaluator.scipy_value_and_gradient
        # The reporting program reads the published physics off an explicit
        # solved state without repeating the solve.  Its configuration is a
        # second call of the closure above, which is a pure read of the same
        # solved surface, so both the objective and the report share one
        # configuration.
        self._reporting_metrics = make_traceable_objective_runtime_bundle(
            boozer_surface,
            field,
            evaluator.iota_target,
            outer_objective_config=outer_objective_config(),
        )["reporting_metrics_from_solution"]

    def endpoint(self, coil_dofs: NDArray[np.float64]) -> SingleStageVacuumEndpoint:
        """Evaluate at ``coil_dofs`` and report the physics of the warm-start state.

        Native publishes its endpoint the same way: one objective evaluation at
        the returned coils, then iota, volume, non-QS ratio and Boozer residual
        read off the resulting Boozer state.  The two agree whenever the inner
        solve succeeded; if it did not, the physics here describes the restored
        warm start while native's describes the failed iterate it kept.  Either
        way ``inner_success`` is false and the endpoint fails its gate.
        """
        evaluation = self._evaluator.evaluate(coil_dofs)
        metrics = self._reporting_metrics(
            coil_dofs,
            self._evaluator.x_inner,
            evaluation.inner_success,
            include_distance_metrics=False,
        )
        return SingleStageVacuumEndpoint(
            value=evaluation.value,
            gradient=evaluation.gradient,
            inner_success=evaluation.inner_success,
            iota=host_float(metrics["final_iota"]),
            volume=host_float(metrics["final_volume"]),
            non_qs_ratio=host_float(metrics["final_non_qs"]),
            boozer_residual=host_float(metrics["final_boozer_residual"]),
        )
