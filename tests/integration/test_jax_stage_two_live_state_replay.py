"""Both stage-two mirrors evaluate the native objective at the native run's states.

``1_Simple/stage_two_optimization_minimal.py`` and
``3_Advanced/stage_two_optimization_finitebuild.py`` each pass through two
states this file can name exactly: the optimizer's start vector and its end
vector.  Both are produced live here by a native SIMSOPT run of the workflow at
the reduced (bounded) scale, and the JAX lane is evaluated at both of them and
compared with the native objective's value, gradient AND term decomposition.
No reference data is stored.

This is same-state evidence.  It does not depend on the JAX lane's optimizer
path, so it stays informative where an end-point verdict is a band rather
than an equality: these tests say the function and the gradient being walked
are the native ones.

The native finite-build provider call is ``fun -> (1e-4 * J, 1e-4 * dJ)``, so
its start state is compared in that scaling, while its end state is ``J`` and
``dJ`` unscaled.  The minimal script has no scale.

What each bound is derived from
-------------------------------
Two float64 rounding models, both doubled (two independently rounded
evaluations are compared, each with its own error) and both RELATIVE (the
caller multiplies by the ``S`` of the reduction it is bounding): the
worst-case ``2 (n - 1) u`` (Higham, *Accuracy and Stability of Numerical
Algorithms*, 2nd ed., section 4.2) and the probabilistic ``2 lambda sqrt(n) u``
(Hoeffding at ``EXCEEDANCE_PROBABILITY`` on the ``n`` roundings modelled as
independent, bounded and mean-zero; Higham and Mary, *SIAM J. Sci. Comput.*
41(5), 2019).  Which one is used is decided by whether the ``S`` of the sum
being bounded is measured:

* the flux, the penalty remainder and the objective value are bounded by the
  PROBABILISTIC model, because each is paired with the ``S`` of its own sum,
  measured at the state (:class:`FluxReduction` for the field reduction);
* the GRADIENT is bounded by the WORST-CASE model, because its ``S`` is not
  computed here: ``(n - 1)`` stands in for ``lambda sqrt(n)`` times an
  amplification this file cannot measure, against the ambient gradient scale
  (the start state's gradient inf-norm, see :class:`StateExpectation`).

Every bound pairs ``n`` with the ``S`` of the SAME sum:

* squared flux: the surface quadrature reduces ``n_s`` NON-NEGATIVE
  contributions whose sum is the flux itself, and the field at each of those
  points is a second reduction over ``n_c = coils * curve quadrature``
  contributions whose own absolute sum enters through
  :attr:`FluxReduction.amplification`, computed from the geometry at that state.
* coil lengths: a positive quadrature over the curve's quadrature points.
* penalty remainder ``R = J - F``: propagated from the length budgets through
  ``dR/dL = weight * (L - L0)_+``, plus one ulp of ``J``.
* objective value: the flux budget plus the remainder budget plus the rounding
  of the objective's own ``k``-term outer sum.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
from scipy.optimize import minimize
from simsopt._core.optimizable import Optimizable
from simsopt.field import (
    BiotSavart,
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
    coils_via_symmetries,
)
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.examples.stage_two_finitebuild import (
    FINITE_BUILD_LBFGS_HISTORY,
    FINITE_BUILD_MAX_FUNCTION_EVALUATIONS,
    FINITE_BUILD_TOLERANCE,
    PreparedFiniteBuildStageTwo,
    prepare_finite_build_stage_two,
)
from simsopt_jax.examples.stage_two_minimal import (
    MINIMAL_STAGE_TWO_LBFGS_HISTORY,
    MINIMAL_STAGE_TWO_NATIVE_ITERATIONS,
    MINIMAL_STAGE_TWO_TOLERANCE,
    minimal_stage_two_state,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.finite_build_stage_two import (
    FINITE_BUILD_DIAGNOSTIC_FIELDS,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)

#: float64 unit roundoff (half the machine epsilon).
UNIT_ROUNDOFF = float(np.finfo(np.float64).eps) / 2.0
#: Two independently rounded evaluations are compared, each with its own error.
COMPARED_EVALUATIONS = 2.0
#: Probability the probabilistic model is allowed to exceed, fixed once.
EXCEEDANCE_PROBABILITY = 1.0e-9
#: ``lambda`` of ``lambda sqrt(n) u S``: Hoeffding, ``sqrt(2 ln(2 / p))``.
ROUNDING_FACTOR = math.sqrt(2.0 * math.log(2.0 / EXCEEDANCE_PROBABILITY))

#: ``mu0 / (4 pi)`` in SI, the Biot-Savart kernel's prefactor.
VACUUM_PERMEABILITY_OVER_4PI = 1.0e-7

#: Surface points per block of the pairwise contribution sum, to keep the
#: (points x sources x 3) intermediate small.
FIELD_SUM_BLOCK = 256


def worst_case_summation_budget(term_count: int) -> float:
    """Relative budget ``2 (n - 1) u`` for two float64 reductions."""
    if term_count < 2:
        raise ValueError("a summation budget needs at least two terms")
    return COMPARED_EVALUATIONS * float(term_count - 1) * UNIT_ROUNDOFF


def probabilistic_summation_budget(term_count: int) -> float:
    """Relative budget ``2 lambda sqrt(n) u`` for two float64 reductions."""
    if term_count < 1:
        raise ValueError("a reduction has at least one contribution")
    return (
        COMPARED_EVALUATIONS
        * ROUNDING_FACTOR
        * math.sqrt(float(term_count))
        * UNIT_ROUNDOFF
    )


def biot_savart_term_count(
    *, surface_points: int, coils: int, curve_quadrature: int
) -> int:
    """Float64 contributions ONE coil-field reduction runs over at one state."""
    if min(surface_points, coils, curve_quadrature) < 1:
        raise ValueError("a term count needs at least one term per factor")
    return surface_points * coils * curve_quadrature


def reduction_budget(*, term_count: int, absolute_sum: float) -> float:
    """Allowance for TWO float64 reductions of the same ``term_count`` terms.

    ``absolute_sum`` is the ``S`` of that reduction: the sum of the absolute
    values of the contributions it adds.
    """
    return probabilistic_summation_budget(term_count) * absolute_sum


def field_absolute_sum(surface: SurfaceRZFourier, coils: Sequence[Coil]) -> np.ndarray:
    """Sum of the absolute Biot-Savart contributions, per surface point.

    One quadrature contribution to ``B(x)`` has magnitude at most
    ``mu0 |I| |gamma'| / (4 pi n_q |x - gamma|^2)``, because
    ``|gamma' x (x - gamma)| <= |gamma'| |x - gamma|``.
    """
    points = np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    contributions = np.zeros(points.shape[0], dtype=np.float64)
    for coil in coils:
        gamma = np.asarray(coil.curve.gamma(), dtype=np.float64)
        speed = np.linalg.norm(
            np.asarray(coil.curve.gammadash(), dtype=np.float64), axis=1
        )
        weight = (
            VACUUM_PERMEABILITY_OVER_4PI
            * abs(float(coil.current.get_value()))
            * speed
            / gamma.shape[0]
        )
        for start in range(0, points.shape[0], FIELD_SUM_BLOCK):
            block = points[start : start + FIELD_SUM_BLOCK]
            offset = block[:, None, :] - gamma[None, :, :]
            contributions[start : start + FIELD_SUM_BLOCK] += np.sum(
                weight[None, :] / np.sum(offset * offset, axis=2), axis=1
            )
    return contributions


@dataclass(frozen=True)
class FluxReduction:
    """The two reductions ``SquaredFlux`` performs at one state.

    ``amplification`` is how an absolute error in the field enters the RELATIVE
    error of the flux: ``2 sum(|n| |B.n| S) / sum(|n| (B.n)^2)``, with ``S`` the
    field reduction's own absolute contribution sum.
    """

    surface_points: int
    source_terms: int
    amplification: float

    def budget(self, flux: float) -> float:
        """Allowance for two independent evaluations of the flux term."""
        return reduction_budget(
            term_count=self.surface_points, absolute_sum=flux
        ) + reduction_budget(
            term_count=self.source_terms, absolute_sum=self.amplification * flux
        )


def flux_reduction(
    surface: SurfaceRZFourier,
    flux: SquaredFlux,
    coils: Sequence[Coil],
    *,
    curve_quadrature: int,
) -> FluxReduction:
    """Measure both flux reductions at whatever state the graph currently holds."""
    normal = np.asarray(surface.normal(), dtype=np.float64)
    area = np.linalg.norm(normal, axis=2)
    unit_normal = normal / area[:, :, None]
    field_values = np.asarray(flux.field.B(), dtype=np.float64).reshape(normal.shape)
    normal_field = np.sum(field_values * unit_normal, axis=2)
    contributions = field_absolute_sum(surface, coils).reshape(area.shape)
    return FluxReduction(
        surface_points=int(area.size),
        source_terms=len(coils) * curve_quadrature,
        amplification=float(
            np.sum(np.abs(normal_field) * area * contributions)
            / (0.5 * np.sum(normal_field * normal_field * area))
        ),
    )


@dataclass(frozen=True)
class LengthPenaltyReduction:
    """The coil-length quadratures the objective's penalty consumes."""

    quadrature_terms: int
    lengths: np.ndarray
    targets: np.ndarray
    weight: float

    @property
    def length_budgets(self) -> np.ndarray:
        """Allowance for two evaluations of each published length."""
        return np.asarray(
            [
                reduction_budget(
                    term_count=self.quadrature_terms, absolute_sum=float(length)
                )
                for length in self.lengths
            ],
            dtype=np.float64,
        )

    def remainder_budget(self, *, scale: float, value: float) -> float:
        """Allowance for the non-flux remainder ``R = J - F``.

        ``dR = sum_i weight (L_i - L0_i)_+ dL_i`` because the penalty is
        ``0.5 (L - L0)_+^2``, plus one ulp of ``J``.
        """
        excess = np.maximum(self.lengths - self.targets, 0.0)
        return scale * float(
            np.sum(excess * self.length_budgets)
        ) * self.weight + UNIT_ROUNDOFF * abs(value)


@dataclass(frozen=True)
class StateReductions:
    """Every reduction one case's objective performs at one state."""

    flux: FluxReduction
    penalty: LengthPenaltyReduction
    objective_terms: int
    gradient_term_count: int


@dataclass(frozen=True)
class StateExpectation:
    """One native state, and the magnitudes its budgets are derived from."""

    label: str
    parameters: np.ndarray
    scale: float
    value: float
    flux: float
    remainder: float
    gradient: np.ndarray
    #: ``|native gradient at the START state|_inf``, in this state's scaling.
    #: A gradient component is a signed sum and at the end point it is a
    #: near-cancelling one, so its own reduced magnitude does not bound the
    #: sum of absolute contributions; the problem's ambient gradient scale does.
    ambient_gradient_norm: float


@dataclass(frozen=True)
class LaneState:
    """What one lane publishes at one state, in that state's scaling."""

    value: float
    flux: float
    remainder: float
    gradient: np.ndarray


def assert_state(
    expectation: StateExpectation,
    lane: LaneState,
    reductions: StateReductions,
) -> None:
    """Compare the JAX lane's value, terms and gradient against one native state."""
    flux_allowance = reductions.flux.budget(abs(expectation.flux))
    remainder_allowance = reductions.penalty.remainder_budget(
        scale=expectation.scale, value=expectation.value
    )
    value_allowance = (
        flux_allowance
        + remainder_allowance
        + worst_case_summation_budget(reductions.objective_terms)
        * abs(expectation.value)
    )
    gradient_allowance = worst_case_summation_budget(
        reductions.gradient_term_count
    ) * abs(expectation.ambient_gradient_norm)

    flux_gap = abs(lane.flux - expectation.flux)
    assert flux_gap <= flux_allowance, (
        f"{expectation.label}: squared flux differs from the native one by "
        f"{flux_gap:.6e}, above the derived budget {flux_allowance:.6e} "
        f"(field-reduction amplification {reductions.flux.amplification:.4g})"
    )
    remainder_gap = abs(lane.remainder - expectation.remainder)
    assert remainder_gap <= remainder_allowance, (
        f"{expectation.label}: the objective's non-flux remainder differs from "
        f"the native one by {remainder_gap:.6e}, above the derived budget "
        f"{remainder_allowance:.6e}"
    )
    value_gap = abs(lane.value - expectation.value)
    assert value_gap <= value_allowance, (
        f"{expectation.label}: objective differs from the native value by "
        f"{value_gap:.6e}, above the derived budget {value_allowance:.6e}"
    )
    gradient_gap = float(np.max(np.abs(lane.gradient - expectation.gradient)))
    assert gradient_gap <= gradient_allowance, (
        f"{expectation.label}: gradient differs from the native gradient by "
        f"{gradient_gap:.6e}, above the derived budget {gradient_allowance:.6e}"
    )


def assert_native_lengths(
    label: str,
    lengths: np.ndarray,
    native_lengths: np.ndarray,
    reductions: StateReductions,
) -> None:
    """Compare the lengths the penalty consumes against the native ones."""
    budgets = reductions.penalty.length_budgets
    gaps = np.abs(np.asarray(lengths, dtype=np.float64) - native_lengths)
    assert gaps.shape == budgets.shape
    assert np.all(gaps <= budgets), (
        f"{label}: coil lengths differ from the native ones by "
        f"{np.array2string(gaps, precision=3)}, above the derived budgets "
        f"{np.array2string(budgets, precision=3)}"
    )


def _surface(resolution: int) -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(
        SURFACE_INPUT, range="half period", nphi=resolution, ntheta=resolution
    )


# --------------------------------------------------------------------------
# stage_two_optimization_finitebuild, bounded scale
# --------------------------------------------------------------------------

FB_SURFACE_RESOLUTION = 4
FB_CURVE_ORDER = 2
FB_CURVE_QUADRATURE = 8
FB_NUM_BASE_CURVES = 4
FB_FILAMENTS = (2, 3)
FB_GAPS = (0.02, 0.04)
FB_LENGTH_WEIGHT = 1.0e-2
FB_CURVE_CURVE_THRESHOLD = 0.1
FB_CURVE_CURVE_WEIGHT = 10.0
FB_OBJECTIVE_SCALE = 1.0e-4
FB_MAX_STEPS = 3


def _finitebuild_geometry():
    surface = _surface(FB_SURFACE_RESOLUTION)
    base_curves = create_equally_spaced_curves(
        FB_NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.7,
        order=FB_CURVE_ORDER,
        numquadpoints=FB_CURVE_QUADRATURE,
        use_jax_curve=False,
    )
    filament_count = FB_FILAMENTS[0] * FB_FILAMENTS[1]
    base_currents = []
    for index in range(FB_NUM_BASE_CURVES):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (1.0e5 / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(curve, *FB_FILAMENTS, *FB_GAPS, rotation_order=1)
            for curve in base_curves
        )
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(
            apply_symmetries_to_curves(base_filaments, surface.nfp, True),
            apply_symmetries_to_currents(
                list(
                    itertools.chain.from_iterable(
                        [current] * filament_count for current in base_currents
                    )
                ),
                surface.nfp,
                True,
            ),
            strict=True,
        )
    ]
    config = FiniteBuildStageTwoConfig(
        num_base_curves=FB_NUM_BASE_CURVES,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=FB_FILAMENTS[0],
            numfilaments_b=FB_FILAMENTS[1],
            gapsize_n=FB_GAPS[0],
            gapsize_b=FB_GAPS[1],
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=tuple(float(CurveLength(curve).J()) for curve in base_curves),
        length_weight=FB_LENGTH_WEIGHT,
        curve_curve_minimum_distance=FB_CURVE_CURVE_THRESHOLD,
        curve_curve_weight=FB_CURVE_CURVE_WEIGHT,
    )
    symmetric_base_curves = apply_symmetries_to_curves(base_curves, surface.nfp, True)
    return surface, base_curves, symmetric_base_curves, coils, config


@dataclass(frozen=True)
class _NativeFiniteBuild:
    surface: SurfaceRZFourier
    coils: tuple[Coil, ...]
    flux: SquaredFlux
    lengths: tuple[CurveLength, ...]
    length_targets: tuple[float, ...]
    length_term: Optimizable
    distance_term: Optimizable
    objective: Optimizable


def _native_finitebuild() -> _NativeFiniteBuild:
    surface, base_curves, symmetric_base_curves, coils, config = _finitebuild_geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    lengths = tuple(CurveLength(curve) for curve in base_curves)
    length_term = FB_LENGTH_WEIGHT * sum(
        QuadraticPenalty(length, target, "max")
        for length, target in zip(lengths, config.length_targets, strict=True)
    )
    distance_term = FB_CURVE_CURVE_WEIGHT * CurveCurveDistance(
        symmetric_base_curves, FB_CURVE_CURVE_THRESHOLD
    )
    return _NativeFiniteBuild(
        surface=surface,
        coils=tuple(coils),
        flux=flux,
        lengths=lengths,
        length_targets=config.length_targets,
        length_term=length_term,
        distance_term=distance_term,
        objective=flux + length_term + distance_term,
    )


def _finitebuild_native_end(evaluator: _NativeFiniteBuild, start: np.ndarray):
    """The native lane's scaled L-BFGS-B, as the official provider call runs it."""

    def value_and_gradient(parameters: np.ndarray):
        evaluator.objective.x = parameters
        return (
            FB_OBJECTIVE_SCALE * float(evaluator.objective.J()),
            FB_OBJECTIVE_SCALE * np.asarray(evaluator.objective.dJ(), dtype=np.float64),
        )

    result = minimize(
        value_and_gradient,
        start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": FB_MAX_STEPS,
            "maxcor": FINITE_BUILD_LBFGS_HISTORY,
            "maxfun": FINITE_BUILD_MAX_FUNCTION_EVALUATIONS,
            "gtol": FINITE_BUILD_TOLERANCE,
            "ftol": FINITE_BUILD_TOLERANCE,
        },
        tol=FINITE_BUILD_TOLERANCE,
    )
    return np.asarray(result.x, dtype=np.float64)


def _finitebuild_reductions_at(
    evaluator: _NativeFiniteBuild, parameters: np.ndarray
) -> StateReductions:
    evaluator.objective.x = parameters
    return StateReductions(
        flux=flux_reduction(
            evaluator.surface,
            evaluator.flux,
            evaluator.coils,
            curve_quadrature=FB_CURVE_QUADRATURE,
        ),
        penalty=LengthPenaltyReduction(
            quadrature_terms=FB_CURVE_QUADRATURE,
            lengths=np.asarray([length.J() for length in evaluator.lengths]),
            targets=np.asarray(evaluator.length_targets, dtype=np.float64),
            weight=FB_LENGTH_WEIGHT,
        ),
        # flux + length penalty + distance penalty
        objective_terms=3,
        gradient_term_count=biot_savart_term_count(
            surface_points=FB_SURFACE_RESOLUTION**2,
            coils=len(evaluator.coils),
            curve_quadrature=FB_CURVE_QUADRATURE,
        ),
    )


def _finitebuild_native_expectation(
    evaluator: _NativeFiniteBuild,
    label: str,
    parameters: np.ndarray,
    scale: float,
    ambient_gradient_norm: float,
) -> tuple[StateExpectation, np.ndarray]:
    evaluator.objective.x = parameters
    expectation = StateExpectation(
        label=label,
        parameters=parameters,
        scale=scale,
        value=scale * float(evaluator.objective.J()),
        flux=scale * float(evaluator.flux.J()),
        remainder=scale
        * (float(evaluator.length_term.J()) + float(evaluator.distance_term.J())),
        gradient=scale * np.asarray(evaluator.objective.dJ(), dtype=np.float64),
        ambient_gradient_norm=ambient_gradient_norm,
    )
    return expectation, np.asarray([length.J() for length in evaluator.lengths])


def _finitebuild_jax_state(
    prepared: PreparedFiniteBuildStageTwo, device, parameters: np.ndarray, scale: float
):
    problem = prepared.problem
    problem.set_objective_parameter(
        jax.device_put(np.asarray(scale, dtype=np.float64), device)
    )
    value, gradient = jax.device_get(
        problem.value_and_grad(jax.device_put(parameters, device))
    )
    packed = np.asarray(
        jax.device_get(prepared.diagnostics(jax.device_put(parameters, device))),
        dtype=np.float64,
    )
    index = {name: i for i, name in enumerate(FINITE_BUILD_DIAGNOSTIC_FIELDS)}
    return (
        LaneState(
            value=float(value),
            flux=scale * float(packed[index["squared_flux"]]),
            remainder=scale
            * (
                float(packed[index["length_penalty"]])
                + float(packed[index["distance_penalty"]])
            ),
            gradient=np.asarray(gradient, dtype=np.float64),
        ),
        packed[len(FINITE_BUILD_DIAGNOSTIC_FIELDS) :],
    )


def test_finitebuild_jax_lane_matches_the_native_states() -> None:
    evaluator = _native_finitebuild()
    start = np.asarray(evaluator.objective.x, dtype=np.float64)
    end = _finitebuild_native_end(evaluator, start)

    evaluator.objective.x = start
    scaled_ambient = FB_OBJECTIVE_SCALE * float(
        np.max(np.abs(evaluator.objective.dJ()))
    )
    states = (
        _finitebuild_native_expectation(
            evaluator,
            "finitebuild start state (the provider's 1e-4 scaling)",
            start,
            FB_OBJECTIVE_SCALE,
            scaled_ambient,
        ),
        _finitebuild_native_expectation(
            evaluator,
            "finitebuild end state (unscaled)",
            end,
            1.0,
            scaled_ambient / FB_OBJECTIVE_SCALE,
        ),
    )
    reductions = tuple(
        _finitebuild_reductions_at(evaluator, parameters) for parameters in (start, end)
    )

    device = get_runtime_jax_device()
    surface, _base_curves, _symmetric, coils, config = _finitebuild_geometry()
    field = BiotSavartJAX(coils)
    flux_spec = SquaredFluxJAX(surface, field).fixed_surface_flux_spec()
    prepared = prepare_finite_build_stage_two(
        objective_fn=make_finite_build_stage_two_objective(field, flux_spec, config),
        diagnostics_fn=finite_build_stage_two_diagnostics(field, flux_spec, config),
        initial_parameters=jax.device_put(start, device),
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64), device),
    )

    for (expectation, native_lengths), state_reductions in zip(
        states, reductions, strict=True
    ):
        lane, lane_lengths = _finitebuild_jax_state(
            prepared, device, expectation.parameters, expectation.scale
        )
        assert_state(expectation, lane, state_reductions)
        if expectation.label.startswith("finitebuild end"):
            assert_native_lengths(
                expectation.label, lane_lengths, native_lengths, state_reductions
            )


# --------------------------------------------------------------------------
# stage_two_optimization_minimal, bounded scale
# --------------------------------------------------------------------------

MIN_SURFACE_RESOLUTION = 4
MIN_CURVE_ORDER = 2
MIN_CURVE_QUADRATURE = 16
MIN_NUM_BASE_CURVES = 4
MIN_LENGTH_WEIGHT = 1.0
MIN_LENGTH_TARGET = 18.0


def _minimal_geometry():
    surface = _surface(MIN_SURFACE_RESOLUTION)
    base_curves = create_equally_spaced_curves(
        MIN_NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=MIN_CURVE_ORDER,
        numquadpoints=MIN_CURVE_QUADRATURE,
    )
    # Official: ``Current(1.0) * 1e5``, so every free current dof is 1.0.
    base_currents = [Current(1.0) * 1.0e5 for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    return surface, base_curves, coils


@dataclass(frozen=True)
class _NativeMinimal:
    surface: SurfaceRZFourier
    coils: tuple[Coil, ...]
    flux: SquaredFlux
    total_length: Optimizable
    length_penalty: Optimizable
    objective: Optimizable


def _native_minimal() -> _NativeMinimal:
    surface, base_curves, coils = _minimal_geometry()
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    flux = SquaredFlux(surface, field)
    total_length = sum(CurveLength(curve) for curve in base_curves)
    length_penalty = QuadraticPenalty(total_length, MIN_LENGTH_TARGET, "max")
    return _NativeMinimal(
        surface=surface,
        coils=tuple(coils),
        flux=flux,
        total_length=total_length,
        length_penalty=length_penalty,
        objective=flux + MIN_LENGTH_WEIGHT * length_penalty,
    )


def _minimal_native_end(evaluator: _NativeMinimal, start: np.ndarray) -> np.ndarray:
    def value_and_gradient(parameters: np.ndarray):
        evaluator.objective.x = parameters
        return float(evaluator.objective.J()), np.asarray(
            evaluator.objective.dJ(), dtype=np.float64
        )

    result = minimize(
        value_and_gradient,
        start,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": MINIMAL_STAGE_TWO_NATIVE_ITERATIONS,
            "maxcor": MINIMAL_STAGE_TWO_LBFGS_HISTORY,
        },
        tol=MINIMAL_STAGE_TWO_TOLERANCE,
    )
    return np.asarray(result.x, dtype=np.float64)


def _minimal_reductions_at(
    evaluator: _NativeMinimal, parameters: np.ndarray
) -> StateReductions:
    evaluator.objective.x = parameters
    return StateReductions(
        flux=flux_reduction(
            evaluator.surface,
            evaluator.flux,
            evaluator.coils,
            curve_quadrature=MIN_CURVE_QUADRATURE,
        ),
        penalty=LengthPenaltyReduction(
            # The penalty consumes ONE reduced length: each base curve's
            # quadrature, plus the additions that total them.
            quadrature_terms=MIN_NUM_BASE_CURVES * MIN_CURVE_QUADRATURE
            + MIN_NUM_BASE_CURVES
            - 1,
            lengths=np.asarray([evaluator.total_length.J()], dtype=np.float64),
            targets=np.asarray([MIN_LENGTH_TARGET], dtype=np.float64),
            weight=MIN_LENGTH_WEIGHT,
        ),
        # flux + length penalty
        objective_terms=2,
        gradient_term_count=biot_savart_term_count(
            surface_points=MIN_SURFACE_RESOLUTION**2,
            coils=len(evaluator.coils),
            curve_quadrature=MIN_CURVE_QUADRATURE,
        ),
    )


def test_minimal_jax_lane_matches_the_native_states() -> None:
    evaluator = _native_minimal()
    start = np.asarray(evaluator.objective.x, dtype=np.float64)
    end = _minimal_native_end(evaluator, start)

    evaluator.objective.x = start
    ambient = float(np.max(np.abs(evaluator.objective.dJ())))
    expectations = []
    native_lengths = []
    for label, parameters in (
        ("minimal start state", start),
        ("minimal end state", end),
    ):
        evaluator.objective.x = parameters
        expectations.append(
            StateExpectation(
                label=label,
                parameters=parameters,
                scale=1.0,
                value=float(evaluator.objective.J()),
                flux=float(evaluator.flux.J()),
                remainder=MIN_LENGTH_WEIGHT * float(evaluator.length_penalty.J()),
                gradient=np.asarray(evaluator.objective.dJ(), dtype=np.float64),
                ambient_gradient_norm=ambient,
            )
        )
        native_lengths.append(float(evaluator.total_length.J()))
    reductions = tuple(
        _minimal_reductions_at(evaluator, parameters) for parameters in (start, end)
    )

    surface, _base_curves, coils = _minimal_geometry()
    field = BiotSavartJAX(coils)
    device = get_runtime_jax_device()
    state = minimal_stage_two_state(
        field=field,
        flux_spec=SquaredFluxJAX(surface, field).fixed_surface_flux_spec(),
        surface_gamma=jax.device_put(
            np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)), device
        ),
        surface_normal=jax.device_put(
            np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)), device
        ),
        num_base_curves=MIN_NUM_BASE_CURVES,
        length_weight=MIN_LENGTH_WEIGHT,
        length_target=MIN_LENGTH_TARGET,
    )

    for expectation, length, state_reductions in zip(
        expectations, native_lengths, reductions, strict=True
    ):
        published = jax.device_get(state(jax.device_put(expectation.parameters)))
        lane = LaneState(
            value=float(published.objective),
            flux=float(published.squared_flux),
            # The lane publishes this term already weighted.
            remainder=float(published.length_penalty),
            gradient=np.asarray(published.objective_gradient, dtype=np.float64),
        )
        assert_state(expectation, lane, state_reductions)
        if expectation.label == "minimal end state":
            assert_native_lengths(
                expectation.label,
                np.asarray([published.total_curve_length], dtype=np.float64),
                np.asarray([length], dtype=np.float64),
                state_reductions,
            )
