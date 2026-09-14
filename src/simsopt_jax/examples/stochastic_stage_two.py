"""Single source of truth for the stochastic stage-two example scales."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionScale
from .scalar_stage import (
    STAGE_OPTIMIZER_OBSERVABLES,
    solve_scalar_stage,
    stage_optimizer_observables,
)


@dataclass(frozen=True, slots=True)
class StochasticStageTwoConfiguration:
    """Immutable scientific and numerical configuration for one execution scale."""

    surface_nphi: int
    surface_ntheta: int
    curve_order: int
    curve_quadrature: int
    num_base_curves: int
    major_radius: float
    minor_radius: float
    initial_current: float
    length_weight: float
    curve_curve_threshold: float
    curve_curve_weight: float
    curvature_threshold: float
    curvature_weight: float
    mean_squared_curvature_threshold: float
    mean_squared_curvature_weight: float
    arclength_variation_weight: float
    perturbation_sigma: float
    perturbation_length_scale: float
    training_sample_count: int
    out_of_sample_count: int
    training_seed: int
    out_of_sample_seed: int
    max_steps: int
    #: L-BFGS history size, the native example's ``maxcor=400``
    #: (``examples/2_Intermediate/stage_two_optimization_stochastic.py:198``).
    #: One value for both routes and both scales: a history that tracked
    #: ``max_steps`` would be a different policy at every scale.
    lbfgs_history_size: int
    #: Convergence tolerances, in the ``serial_solve_jax`` vocabulary that maps
    #: ``rtol -> ftol`` and ``atol -> gtol``.  Both are pinned to the native
    #: example's ``tol=1e-15``: SciPy's ``minimize(..., tol=)`` sets *both*
    #: ``ftol`` and ``gtol`` for L-BFGS-B, so
    #: ``examples/2_Intermediate/stage_two_optimization_stochastic.py:198``
    #: runs at ``ftol = gtol = 1e-15`` and a mirror at ``gtol = 1e-8`` would
    #: stop on a gradient test seven orders looser than the lane it mirrors.
    rtol: float
    atol: float


def stochastic_stage_two_configuration(
    scale: ExecutionScale,
) -> StochasticStageTwoConfiguration:
    """Return the canonical stochastic stage-two configuration."""
    native_scale = scale == "native_default"
    return StochasticStageTwoConfiguration(
        surface_nphi=64 if native_scale else 4,
        surface_ntheta=16 if native_scale else 4,
        curve_order=24 if native_scale else 2,
        curve_quadrature=360 if native_scale else 16,
        num_base_curves=4,
        major_radius=1.0,
        minor_radius=0.5,
        initial_current=1.0e5,
        length_weight=1.0e-6,
        curve_curve_threshold=0.1,
        curve_curve_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
        arclength_variation_weight=1.0e-2,
        perturbation_sigma=1.0e-3,
        perturbation_length_scale=0.5,
        training_sample_count=16 if native_scale else 2,
        out_of_sample_count=256 if native_scale else 4,
        training_seed=0,
        out_of_sample_seed=1,
        max_steps=400 if native_scale else 20,
        lbfgs_history_size=400,
        rtol=1.0e-15,
        atol=1.0e-15,
    )


#: This family's spelling of the shared single-stage route.  The stochastic
#: mirror and its parity twin call the solve under this name; the route, and
#: the drivers it accepts, are :func:`.scalar_stage.solve_scalar_stage`'s.
solve_stochastic_stage_two = solve_scalar_stage

#: This family's spelling of the shared single-stage name tuple and its
#: producer, kept importable under the names the mirror already reads.
STOCHASTIC_STAGE_TWO_OPTIMIZER_OBSERVABLES = STAGE_OPTIMIZER_OBSERVABLES
stochastic_stage_two_optimizer_observables = stage_optimizer_observables
