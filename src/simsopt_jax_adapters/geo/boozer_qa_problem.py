"""The official BoozerQA outer problem on the analytic exact Boozer route.

``examples/2_Intermediate/boozerQA.py`` at upstream 9e027eac3 optimizes the NCSX
coil degrees of freedom for quasi-axisymmetry while holding the rotational
transform, the major radius and the total base-coil length at their seed values.
Every objective evaluation there re-solves the volume-labelled Boozer surface
with ``solve_residual_equation_exactly_newton`` -- the dense analytic Jacobian,
a direct solve and the undamped full Newton step -- and returns ``J = 1e3`` with
the previous surface restored when that solve fails (upstream boozerQA.py:102-109).

This module owns that workflow for the JAX mirror: it wires the official options
(:mod:`simsopt_jax.examples.boozer_official`, which stays their owner) onto the
certified analytic evaluator
(:class:`~simsopt_jax_adapters.geo.single_stage_exact_analytic.ExactAnalyticSingleStage`
over :class:`~simsopt_jax_adapters.geo.single_stage_exact_analytic.HostConstructionBoozerSurfaceJAX`),
which is upstream's algorithm: dense analytic Newton at tolerance 1e-13 with a
cap of 20, the last *successful* state as the warm start, the ``1e3`` sentinel on
failure, and the implicit-function gradient through a transpose solve of the same
Jacobian.  The outer optimizer belongs to the caller.

The shipped script ``examples/jax/2_Intermediate/boozerQA.py`` runs this
module, so the workflow its tests verify is the one a user executes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from simsopt.field import BiotSavart
from simsopt.geo import CurveLength, SurfaceXYZTensorFourier, Volume
from simsopt.geo.curve import Curve
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_RESIDUAL_WEIGHT,
    boozer_qa_outer_objective_config,
)
from simsopt_jax.runtime.host_boundary import host_array, host_bool, host_float

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    _reporting_metrics_with_explicit_staging,
)
from simsopt_jax_adapters.geo.single_stage_exact_analytic import (
    ExactAnalyticSingleStage,
    HostConstructionBoozerSurfaceJAX,
)
from simsopt_jax_adapters.geo.surface_objectives_traceable import (
    _make_traceable_reporting_metrics_from_solution_bundle,
)

__all__ = [
    "BoozerQAEndpoint",
    "BoozerQAProblem",
]


@dataclass(frozen=True, slots=True)
class BoozerQAEndpoint:
    """One evaluated coil state: reported objective, gradient and physics.

    ``value`` and ``gradient`` carry upstream's failed-inner-solve policy, so a
    ``value`` of
    :data:`~simsopt_jax_adapters.geo.single_stage_exact_analytic.INNER_FAILURE_VALUE`
    with ``inner_success`` false is the sentinel, not a physical objective.  The
    physics fields describe the Boozer state the evaluation left in place.
    """

    value: float
    gradient: NDArray[np.float64]
    inner_success: bool
    iota: float
    volume: float
    non_qs_ratio: float
    boozer_residual: float
    major_radius_penalty: float
    length_penalty: float

    @property
    def boozer_residual_rms(self) -> float:
        """Root mean square of the Boozer residual vector.

        ``boozer_residual`` is half the mean square of that vector, which is what
        native's ``BoozerResidual.J()`` returns.
        """
        return float(np.sqrt(2.0 * self.boozer_residual))


class BoozerQAProblem:
    """The official BoozerQA problem: coil dofs in, objective and physics out.

    The caller supplies the already-constructed NCSX pieces -- the base curves
    whose lengths set the length target, the native ``BiotSavart`` whose coils
    define the field, the seed surface, ``nfp`` and the seed ``G`` -- so the
    shipped script keeps upstream's inline construction as the construction
    source of truth.

    Construction solves the seed Boozer surface and freezes the iota, major-radius
    and coil-length targets at that solution, exactly as upstream does: upstream
    builds ``MajorRadius(boozer_surface)`` and ``Iotas(boozer_surface)`` after
    ``boozer_surface.solve_residual_equation_exactly_newton`` has run, so both
    targets are read off the SOLVED surface.  ``value_and_gradient`` is the
    ``jac=True`` outer callable; ``endpoint`` evaluates one coil state and adds
    the published physics.

    A seed solve that does not converge is an error at construction: the
    certified evaluator raises ``RuntimeError`` before this object exists, so
    ``initial_inner_success`` is True whenever a problem exists (the session
    route reported False and let the lane finish as failed).  The published
    ``final:inner_solver_success`` is the endpoint solve's success, as on every
    lane; the endpoint CERTIFICATE takes seed success AND endpoint success, the
    definition every lane shares, so the attribute stays even though it can
    only be True here.

    Construction is this problem's host boundary.  The Boozer adapter is the
    NumPy-baking :class:`HostConstructionBoozerSurfaceJAX`, so construction runs
    under ``jax.transfer_guard("disallow")`` and the compiled evaluate program
    captures host constants instead of device arrays.
    """

    def __init__(
        self,
        *,
        base_curves: Sequence[Curve],
        native_field: BiotSavart,
        surface: SurfaceXYZTensorFourier,
        nfp: int,
        initial_G: float,
        initial_iota: float,
        boozer_options: Mapping[str, object],
        non_qs_resolution: int,
        residual_weight: float = OFFICIAL_QA_RESIDUAL_WEIGHT,
    ) -> None:
        field = BiotSavartJAX(native_field.coils)
        volume_label = Volume(surface)
        boozer_surface = HostConstructionBoozerSurfaceJAX(
            field,
            surface,
            volume_label,
            float(volume_label.J()),
            options=dict(boozer_options),
        )
        length_target = float(sum(CurveLength(curve).J() for curve in base_curves))

        def outer_objective_config() -> dict[str, object]:
            # The evaluator calls this once, after its initial exact solve has
            # been published as the surface's state, so the major-radius target
            # and the vessel geometry are read off the solved surface.
            return boozer_qa_outer_objective_config(
                nfp=int(nfp),
                non_qs_resolution=int(non_qs_resolution),
                length_target=length_target,
                major_radius_target=float(surface.major_radius()),
                vessel_gamma=surface.gamma(),
                residual_weight=float(residual_weight),
            )

        evaluator = ExactAnalyticSingleStage(
            boozer_surface,
            field,
            iota=float(initial_iota),
            G=float(initial_G),
            outer_objective_config=outer_objective_config,
        )
        self._evaluator = evaluator
        self._boozer_surface = boozer_surface
        self.initial_coil_dofs: NDArray[np.float64] = np.array(
            evaluator.coil_dofs, copy=True
        )
        self.iota_target: float = evaluator.iota_target
        self.initial_inner_iterations: int = evaluator.initial_inner_iterations
        # The seed solve the evaluator ran and published on the surface; its
        # success and residual are read off that installed state rather than
        # recomputed, so they describe the solve the run actually started from.
        seed_solution = boozer_surface.res
        self.initial_inner_success: bool = host_bool(seed_solution["success"])
        self.initial_residual: float = float(
            np.sqrt(
                np.mean(
                    np.square(host_array(seed_solution["residual"], dtype=np.float64))
                )
            )
        )
        self.initial_volume: float = float(volume_label.J())
        self.value_and_gradient = evaluator.scipy_value_and_gradient
        # The reporting program reads the published physics off an explicit
        # solved state without repeating the solve.  It shares the evaluator's
        # own objective configuration, so the report and the objective can never
        # describe two different problems.
        self._reporting_metrics = _reporting_metrics_with_explicit_staging(
            _make_traceable_reporting_metrics_from_solution_bundle(
                {"state": evaluator._objective_cache_state}
            )
        )

    def endpoint(self, coil_dofs: NDArray[np.float64]) -> BoozerQAEndpoint:
        """Evaluate at ``coil_dofs`` and report the physics of the warm-start state.

        Upstream publishes its endpoint the same way: one objective evaluation at
        the returned coils, then iota, volume, the non-QS ratio and the Boozer
        residual read off the resulting Boozer state.  The two agree whenever the
        inner solve succeeded; if it did not, the physics here describes the
        restored warm start while upstream's describes the failed iterate it kept.
        Either way ``inner_success`` is false and the endpoint fails its gate.
        """
        evaluation = self._evaluator.evaluate(coil_dofs)
        metrics = self._reporting_metrics(
            coil_dofs,
            self._evaluator.x_inner,
            evaluation.inner_success,
            include_distance_metrics=False,
        )
        return BoozerQAEndpoint(
            value=evaluation.value,
            gradient=evaluation.gradient,
            inner_success=evaluation.inner_success,
            iota=host_float(metrics["final_iota"]),
            volume=host_float(metrics["final_volume"]),
            non_qs_ratio=host_float(metrics["final_non_qs"]),
            boozer_residual=host_float(metrics["final_boozer_residual"]),
            major_radius_penalty=host_float(metrics["final_major_radius_penalty"]),
            length_penalty=host_float(metrics["final_length_penalty"]),
        )
