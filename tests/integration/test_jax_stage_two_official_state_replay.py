"""Both stage-two mirrors evaluate upstream's objective at upstream's states.

The official runs of ``examples/1_Simple/stage_two_optimization_minimal.py``
and ``examples/3_Advanced/stage_two_optimization_finitebuild.py`` at upstream
``9e027eac3`` each passed through two states this repository can name exactly:
the optimizer's start vector and its end vector.  This file evaluates the
NATIVE lane and the JAX lane at both of them and compares the value, the
gradient AND the objective's own term decomposition against the tracked
official fixture (``examples/jax/parity/official_reference``).

This is same-state evidence.  It does not depend on either lane's optimizer
path, so it stays informative where the end-point verdict is an endpoint
quality band rather than an equality: a band says the two paths landed in
upstream's own scatter, and these tests say the function and the gradient
being walked are upstream's.

Fixture keys, read from the capture instrumentation rather than guessed:
upstream's finite-build provider call is ``fun -> (1e-4 * J, 1e-4 * dJ)``
(``examples/3_Advanced/stage_two_optimization_finitebuild.py:148``), so its
``taylor:*`` entries carry that scale, while its ``final:*`` entries are
``JF.J()`` and ``JF.dJ()`` and do not.  The minimal script has no scale.

What each bound is derived from
-------------------------------
Both rounding models come from ``tests/official_state_budget.py``, which owns
them for every replay: the worst-case ``2 (n - 1) u`` and the probabilistic
``2 lambda sqrt(n) u`` (Hoeffding at ``EXCEEDANCE_PROBABILITY``), each stated
there with its assumption.  This file uses BOTH, and which one it uses is
decided by whether it has measured the ``S`` of the sum it is bounding:

* the flux, the penalty remainder and the objective value are bounded by the
  PROBABILISTIC model, because each is paired with the ``S`` of its own sum,
  measured at the state (:class:`FluxReduction` for the field reduction).
  The worst-case model over those same reductions is 6.2e-10 relative at the
  minimal end state and resolves no weight at all, which is why it is not used
  where a measured ``S`` is available;
* the GRADIENT is bounded by the WORST-CASE model, because its ``S`` is not
  computed here: ``(n - 1)`` stands in for ``lambda sqrt(n)`` times an
  amplification this file cannot measure.  That is not slack -- see below.

Every bound pairs ``n`` with the ``S`` of the SAME sum:

* squared flux: the surface quadrature reduces ``n_s`` NON-NEGATIVE
  contributions whose sum is the flux itself, and the field at each of those
  points is a second reduction over ``n_c = coils * curve quadrature``
  contributions whose own absolute sum enters through
  :attr:`FluxReduction.amplification`, computed from the case's geometry at that state.
  That amplification is about 11 at the official start states and 2200 to 2600
  at the official end states, where ``B . n`` is near-cancelling.  This is why
  no rounding bound on the objective VALUE can resolve a mis-weighted term at
  an end state, and why the terms are compared one by one below.
* coil lengths: a positive quadrature over the curve's quadrature points.
* penalty remainder ``R = J - F``: propagated from the length budgets through
  ``dR/dL = weight * (L - L0)_+``.  Each lane computes ``R`` from its own
  penalty term, so nothing cancels on the lane side, and the official ``R`` is
  ``final:objective - final:squared_flux``, a subtraction that costs one ulp
  of ``J``.
* objective value: the flux budget plus the remainder budget plus the rounding
  of the objective's own ``k``-term outer sum.
* gradient: the one bound this file does not fully derive.  Its reduction runs
  over every (surface point, source point) pair, and the sum of absolute
  contributions of one gradient component is not computed here, so the
  DETERMINISTIC ``(n - 1)`` factor stands in for ``lambda sqrt(n)`` times an
  amplification this file cannot measure, against the ambient gradient scale
  (see :class:`StateExpectation`).  Replacing it by ``lambda sqrt(n)`` makes
  this file FAIL, which is the evidence that the amplification is real and
  that the deterministic factor is not slack.

What this file cannot see
-------------------------
At both official finite-build states the curve-curve distance penalty is
exactly 0.0 -- the clearance (0.117 and 0.145) exceeds the 0.1 threshold, which
:func:`test_finitebuild_distance_penalty_is_inactive_at_the_official_states`
asserts -- so the curve-curve WEIGHT is invisible to any tolerance here.  So is
the minimal length weight at the START state, where the total length 12.57 is
below the 18.0 target.  Where a term is active, the smallest relative error in
its constant that fails this file is 9.5e-10 (finite-build length weight),
3.2e-8 (minimal length weight) and 2.5e-14 (minimal length target).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from examples.jax.parity.cases import (
    native_stage_two_optimization_finitebuild as finitebuild_case,
)
from examples.jax.parity.cases import (
    native_stage_two_optimization_minimal as minimal_case,
)
from examples.jax.parity.input_bundle import InputBundle, load_input_bundle
from examples.jax.parity.official_reference import (
    OfficialReference,
    load_official_reference,
)
from simsopt.field import Coil
from simsopt.geo import SurfaceRZFourier
from simsopt.objectives import SquaredFlux
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples.stage_two_minimal import minimal_stage_two_state

# venv site-packages/tests shadows the repo tests package, so the helper is
# imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from official_state_budget import (  # noqa: E402
    UNIT_ROUNDOFF,
    biot_savart_term_count,
    probabilistic_summation_budget,
    worst_case_summation_budget,
)

#: ``mu0 / (4 pi)`` in SI, the Biot-Savart kernel's prefactor.
VACUUM_PERMEABILITY_OVER_4PI = 1.0e-7

#: Surface points per block of the pairwise contribution sum, to keep the
#: (points x sources x 3) intermediate small.
FIELD_SUM_BLOCK = 256


def reduction_budget(*, term_count: int, absolute_sum: float) -> float:
    """Allowance for TWO float64 reductions of the same ``term_count`` terms.

    ``absolute_sum`` is the ``S`` of that reduction: the sum of the absolute
    values of the contributions it adds, never the magnitude of some other
    quantity that happens to be nearby.
    """
    return probabilistic_summation_budget(term_count) * absolute_sum


def field_absolute_sum(surface: SurfaceRZFourier, coils: Sequence[Coil]) -> np.ndarray:
    """Sum of the absolute Biot-Savart contributions, per surface point.

    One quadrature contribution to ``B(x)`` has magnitude at most
    ``mu0 |I| |gamma'| / (4 pi n_q |x - gamma|^2)``, because
    ``|gamma' x (x - gamma)| <= |gamma'| |x - gamma|``.  Summing that over every
    source point of every coil bounds the ``S`` that the field reduction's
    ``(n - 1) u S`` needs -- the quantity the objective's own value does not
    provide, and the reason a value-scaled budget is not the derived one.
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
    field reduction's own absolute contribution sum.  It is the cancellation of
    ``B . n`` made explicit, and it is measured from the geometry, not chosen.
    """

    surface_points: int
    source_terms: int
    amplification: float

    def budget(self, flux: float) -> float:
        """Allowance for two independent evaluations of the flux term."""
        return reduction_budget(
            term_count=self.surface_points, absolute_sum=flux
        ) + reduction_budget(
            term_count=self.source_terms,
            absolute_sum=self.amplification * flux,
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
    """The coil-length quadratures the objective's penalty consumes.

    ``lengths`` and ``targets`` are the case's own at this state; they enter a
    rounding bound as scale factors only.
    """

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
        ``0.5 (L - L0)_+^2``; ``UNIT_ROUNDOFF * |J|`` is what the official ``R``
        loses by being recovered as ``final:objective - final:squared_flux``.
        """
        excess = np.maximum(self.lengths - self.targets, 0.0)
        return scale * float(
            np.sum(excess * self.length_budgets)
        ) * self.weight + UNIT_ROUNDOFF * abs(value)


@dataclass(frozen=True)
class StateReductions:
    """Every reduction one case's objective performs at one official state."""

    flux: FluxReduction
    penalty: LengthPenaltyReduction
    objective_terms: int
    gradient_term_count: int


@dataclass(frozen=True)
class StateExpectation:
    """One official state, and the magnitudes its budgets are derived from."""

    label: str
    parameters: np.ndarray
    scale: float
    value: float
    flux: float
    remainder: float
    gradient: np.ndarray
    #: ``|official gradient at the START state|_inf``, in this state's scaling.
    #: A gradient component is a signed sum and at the official end point it is
    #: a near-cancelling one, so its own reduced magnitude does not bound the
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
    """Compare one lane's value, terms and gradient against one official state."""
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
        f"{expectation.label}: squared flux differs from the official one by "
        f"{flux_gap:.6e}, above the derived budget {flux_allowance:.6e} "
        f"(field-reduction amplification {reductions.flux.amplification:.4g})"
    )
    remainder_gap = abs(lane.remainder - expectation.remainder)
    assert remainder_gap <= remainder_allowance, (
        f"{expectation.label}: the objective's non-flux remainder differs from "
        f"the official one by {remainder_gap:.6e}, above the derived budget "
        f"{remainder_allowance:.6e}"
    )
    value_gap = abs(lane.value - expectation.value)
    assert value_gap <= value_allowance, (
        f"{expectation.label}: objective differs from the official value by "
        f"{value_gap:.6e}, above the derived budget {value_allowance:.6e}"
    )
    gradient_gap = float(np.max(np.abs(lane.gradient - expectation.gradient)))
    assert gradient_gap <= gradient_allowance, (
        f"{expectation.label}: gradient differs from the official gradient by "
        f"{gradient_gap:.6e}, above the derived budget {gradient_allowance:.6e}"
    )


def assert_official_lengths(
    label: str,
    lengths: np.ndarray,
    official_lengths: np.ndarray,
    reductions: StateReductions,
) -> None:
    """Compare the lengths the penalty consumes against upstream's own."""
    budgets = reductions.penalty.length_budgets
    gaps = np.abs(np.asarray(lengths, dtype=np.float64) - official_lengths)
    assert gaps.shape == budgets.shape
    assert np.all(gaps <= budgets), (
        f"{label}: coil lengths differ from the official ones by "
        f"{np.array2string(gaps, precision=3)}, above the derived budgets "
        f"{np.array2string(budgets, precision=3)}"
    )


def _native_default_bundle(case, root: Path) -> tuple[InputBundle, dict]:
    bundle = case.create_input(root, "native_default")
    _, arrays = load_input_bundle(root, bundle)
    return bundle, arrays


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    assert isinstance(value, int) and not isinstance(value, bool)
    return value


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    assert isinstance(value, float)
    return value


# --------------------------------------------------------------------------
# native-stage-two-optimization-finitebuild
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def finitebuild_reference() -> OfficialReference:
    return load_official_reference("native-stage-two-optimization-finitebuild")


@pytest.fixture(scope="module")
def finitebuild_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InputBundle, dict]:
    root = Path(tmp_path_factory.mktemp("finitebuild-official")) / "inputs"
    return _native_default_bundle(finitebuild_case, root)


def _finitebuild_reductions_at(
    bundle: InputBundle,
    evaluator,
    parameters: np.ndarray,
) -> StateReductions:
    evaluator.objective.x = parameters
    curve_quadrature = _configuration_int(bundle, "curve_quadrature")
    targets = np.asarray(bundle.configuration["length_targets"], dtype=np.float64)
    return StateReductions(
        flux=flux_reduction(
            evaluator.surface,
            evaluator.flux,
            evaluator.coils,
            curve_quadrature=curve_quadrature,
        ),
        penalty=LengthPenaltyReduction(
            quadrature_terms=curve_quadrature,
            lengths=np.asarray(
                [length.J() for length in evaluator.lengths], dtype=np.float64
            ),
            targets=targets,
            weight=_configuration_float(bundle, "length_weight"),
        ),
        # flux + length penalty + distance penalty
        objective_terms=3,
        gradient_term_count=biot_savart_term_count(
            surface_points=_configuration_int(bundle, "surface_resolution") ** 2,
            coils=len(evaluator.coils),
            curve_quadrature=curve_quadrature,
        ),
    )


@pytest.fixture(scope="module")
def finitebuild_reductions(
    finitebuild_bundle: tuple[InputBundle, dict],
    finitebuild_reference: OfficialReference,
) -> tuple[StateReductions, StateReductions]:
    bundle, _ = finitebuild_bundle
    evaluator = finitebuild_case.build_native_evaluator(bundle)
    return tuple(
        _finitebuild_reductions_at(bundle, evaluator, finitebuild_reference.array(key))
        for key in ("initial:parameters", "final:parameters")
    )


def _finitebuild_states(
    bundle: InputBundle,
    reference: OfficialReference,
) -> tuple[StateExpectation, StateExpectation]:
    scale = _configuration_float(bundle, "objective_scale")
    scaled_ambient = float(np.max(np.abs(reference.array("taylor:gradient_scaled"))))
    start_value = reference.scalar("taylor:objective_scaled")
    end_value = reference.scalar("final:objective")
    end_flux = reference.scalar("final:squared_flux")
    assert isinstance(start_value, float) and isinstance(end_value, float)
    assert isinstance(end_flux, float)
    return (
        StateExpectation(
            label="finitebuild start state (upstream's 1e-4 scaling)",
            parameters=reference.array("initial:parameters"),
            scale=scale,
            value=start_value,
            # Upstream's targets ARE the start lengths (`Jls[i].J()` at
            # construction) and the start clearance 0.117 exceeds the 0.1
            # threshold, so both penalties are exactly inactive here and the
            # objective is the flux alone.
            flux=start_value,
            remainder=0.0,
            gradient=reference.array("taylor:gradient_scaled"),
            ambient_gradient_norm=scaled_ambient,
        ),
        StateExpectation(
            label="finitebuild end state (unscaled)",
            parameters=reference.array("final:parameters"),
            scale=1.0,
            value=end_value,
            flux=end_flux,
            remainder=end_value - end_flux,
            gradient=reference.array("final:objective_gradient"),
            ambient_gradient_norm=scaled_ambient / scale,
        ),
    )


def _finitebuild_native_state(evaluator, parameters: np.ndarray, scale: float):
    evaluator.objective.x = parameters
    return (
        LaneState(
            value=scale * float(evaluator.objective.J()),
            flux=scale * float(evaluator.flux.J()),
            remainder=scale
            * (float(evaluator.length_term.J()) + float(evaluator.distance_term.J())),
            gradient=scale * np.asarray(evaluator.objective.dJ(), dtype=np.float64),
        ),
        np.asarray([length.J() for length in evaluator.lengths], dtype=np.float64),
    )


def _finitebuild_jax_state(construction, device, parameters: np.ndarray, scale: float):
    problem = construction.prepared.problem
    problem.set_objective_parameter(
        jax.device_put(np.asarray(scale, dtype=np.float64), device)
    )
    value, gradient = jax.device_get(
        problem.value_and_grad(jax.device_put(parameters, device))
    )
    diagnostics = finitebuild_case.finite_build_diagnostics(
        jax.device_get(
            construction.prepared.diagnostics(jax.device_put(parameters, device))
        )
    )
    return (
        LaneState(
            value=float(value),
            flux=scale * diagnostics.squared_flux,
            remainder=scale
            * (diagnostics.length_penalty + diagnostics.distance_penalty),
            gradient=np.asarray(gradient, dtype=np.float64),
        ),
        diagnostics,
    )


def test_finitebuild_case_starts_from_upstreams_own_start_vector(
    finitebuild_bundle: tuple[InputBundle, dict],
    finitebuild_reference: OfficialReference,
) -> None:
    """The states below are upstream's because the case's start vector is."""
    _, arrays = finitebuild_bundle

    np.testing.assert_array_equal(
        arrays["initial_parameters"],
        finitebuild_reference.array("initial:parameters"),
    )


def test_finitebuild_native_lane_matches_the_official_states(
    finitebuild_bundle: tuple[InputBundle, dict],
    finitebuild_reference: OfficialReference,
    finitebuild_reductions: tuple[StateReductions, StateReductions],
) -> None:
    bundle, _ = finitebuild_bundle
    evaluator = finitebuild_case.build_native_evaluator(bundle)
    states = _finitebuild_states(bundle, finitebuild_reference)

    for expectation, reductions in zip(states, finitebuild_reductions, strict=True):
        lane, lengths = _finitebuild_native_state(
            evaluator, expectation.parameters, expectation.scale
        )
        assert_state(expectation, lane, reductions)
        if expectation.label.startswith("finitebuild end"):
            assert_official_lengths(
                expectation.label,
                lengths,
                finitebuild_reference.array("final:curve_lengths"),
                reductions,
            )


def test_finitebuild_jax_lane_matches_the_official_states(
    finitebuild_bundle: tuple[InputBundle, dict],
    finitebuild_reference: OfficialReference,
    finitebuild_reductions: tuple[StateReductions, StateReductions],
) -> None:
    bundle, _ = finitebuild_bundle
    device = get_runtime_jax_device()
    states = _finitebuild_states(bundle, finitebuild_reference)
    construction = finitebuild_case.build_jax_construction(
        bundle,
        initial_parameters=jax.device_put(states[0].parameters, device),
        device=device,
    )

    for expectation, reductions in zip(states, finitebuild_reductions, strict=True):
        lane, diagnostics = _finitebuild_jax_state(
            construction, device, expectation.parameters, expectation.scale
        )
        assert_state(expectation, lane, reductions)
        if expectation.label.startswith("finitebuild end"):
            assert_official_lengths(
                expectation.label,
                diagnostics.coil_lengths,
                finitebuild_reference.array("final:curve_lengths"),
                reductions,
            )


def test_finitebuild_distance_penalty_is_inactive_at_the_official_states(
    finitebuild_bundle: tuple[InputBundle, dict],
    finitebuild_reference: OfficialReference,
) -> None:
    """The curve-curve weight is invisible here, and this says so in numbers.

    ``max(0, threshold - d)`` is exactly zero for every pair, so the penalty is
    exactly 0.0 on both lanes and no tolerance can resolve the weight in front
    of it.  The fixture's own end-state clearance is asserted against the
    case's threshold so that the claim cannot go stale silently.
    """
    bundle, _ = finitebuild_bundle
    threshold = _configuration_float(bundle, "curve_curve_threshold")
    evaluator = finitebuild_case.build_native_evaluator(bundle)
    device = get_runtime_jax_device()
    states = _finitebuild_states(bundle, finitebuild_reference)
    construction = finitebuild_case.build_jax_construction(
        bundle,
        initial_parameters=jax.device_put(states[0].parameters, device),
        device=device,
    )

    assert float(finitebuild_reference.scalar("final:curve_curve_minimum_distance")) > (
        threshold
    )
    for expectation in states:
        evaluator.objective.x = expectation.parameters
        assert float(evaluator.distance_term.J()) == 0.0
        _, diagnostics = _finitebuild_jax_state(
            construction, device, expectation.parameters, expectation.scale
        )
        assert diagnostics.distance_penalty == 0.0
        assert diagnostics.minimum_clearance > threshold


# --------------------------------------------------------------------------
# native-stage-two-optimization-minimal
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minimal_reference() -> OfficialReference:
    return load_official_reference("native-stage-two-optimization-minimal")


@pytest.fixture(scope="module")
def minimal_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[InputBundle, dict]:
    root = Path(tmp_path_factory.mktemp("minimal-official")) / "inputs"
    return _native_default_bundle(minimal_case, root)


def _minimal_reductions_at(
    bundle: InputBundle,
    evaluator,
    parameters: np.ndarray,
) -> StateReductions:
    evaluator.objective.x = parameters
    curve_quadrature = _configuration_int(bundle, "curve_quadrature")
    num_base_curves = _configuration_int(bundle, "num_base_curves")
    return StateReductions(
        flux=flux_reduction(
            evaluator.surface,
            evaluator.flux,
            evaluator.coils,
            curve_quadrature=curve_quadrature,
        ),
        penalty=LengthPenaltyReduction(
            # The penalty consumes ONE reduced length: each base curve's
            # quadrature, plus the additions that total them.
            quadrature_terms=num_base_curves * curve_quadrature + num_base_curves - 1,
            lengths=np.asarray([evaluator.total_length.J()], dtype=np.float64),
            targets=np.asarray(
                [_configuration_float(bundle, "length_target")], dtype=np.float64
            ),
            weight=_configuration_float(bundle, "length_weight"),
        ),
        # flux + length penalty
        objective_terms=2,
        gradient_term_count=biot_savart_term_count(
            surface_points=_configuration_int(bundle, "surface_resolution") ** 2,
            coils=len(evaluator.coils),
            curve_quadrature=curve_quadrature,
        ),
    )


@pytest.fixture(scope="module")
def minimal_reductions(
    minimal_bundle: tuple[InputBundle, dict],
    minimal_reference: OfficialReference,
) -> tuple[StateReductions, StateReductions]:
    bundle, _ = minimal_bundle
    evaluator = minimal_case.build_native_evaluator(bundle)
    return tuple(
        _minimal_reductions_at(bundle, evaluator, minimal_reference.array(key))
        for key in ("initial:parameters", "final:parameters")
    )


def _minimal_states(
    reference: OfficialReference,
) -> tuple[StateExpectation, StateExpectation]:
    ambient = float(np.max(np.abs(reference.array("taylor:gradient"))))
    start_value = reference.scalar("taylor:objective")
    end_value = reference.scalar("final:objective")
    end_flux = reference.scalar("final:squared_flux")
    assert isinstance(start_value, float) and isinstance(end_value, float)
    assert isinstance(end_flux, float)
    return (
        StateExpectation(
            label="minimal start state",
            parameters=reference.array("initial:parameters"),
            scale=1.0,
            value=start_value,
            # The start total length, 12.57, is below upstream's 18.0 target and
            # the penalty is one-sided, so it is exactly inactive here and the
            # objective is the flux alone.
            flux=start_value,
            remainder=0.0,
            gradient=reference.array("taylor:gradient"),
            ambient_gradient_norm=ambient,
        ),
        StateExpectation(
            label="minimal end state",
            parameters=reference.array("final:parameters"),
            scale=1.0,
            value=end_value,
            flux=end_flux,
            remainder=end_value - end_flux,
            gradient=reference.array("final:objective_gradient"),
            ambient_gradient_norm=ambient,
        ),
    )


def test_minimal_case_starts_from_upstreams_own_start_vector(
    minimal_bundle: tuple[InputBundle, dict],
    minimal_reference: OfficialReference,
) -> None:
    """The states below are upstream's because the case's start vector is."""
    _, arrays = minimal_bundle

    np.testing.assert_array_equal(
        arrays["initial_parameters"],
        minimal_reference.array("initial:parameters"),
    )


def test_minimal_native_lane_matches_the_official_states(
    minimal_bundle: tuple[InputBundle, dict],
    minimal_reference: OfficialReference,
    minimal_reductions: tuple[StateReductions, StateReductions],
) -> None:
    bundle, _ = minimal_bundle
    evaluator = minimal_case.build_native_evaluator(bundle)
    weight = _configuration_float(bundle, "length_weight")
    states = _minimal_states(minimal_reference)

    for expectation, reductions in zip(states, minimal_reductions, strict=True):
        evaluator.objective.x = expectation.parameters
        lane = LaneState(
            value=float(evaluator.objective.J()),
            flux=float(evaluator.flux.J()),
            remainder=weight * float(evaluator.length_penalty.J()),
            gradient=np.asarray(evaluator.objective.dJ(), dtype=np.float64),
        )
        assert_state(expectation, lane, reductions)
        if expectation.label == "minimal end state":
            assert_official_lengths(
                expectation.label,
                np.asarray([evaluator.total_length.J()], dtype=np.float64),
                np.asarray(
                    [minimal_reference.scalar("final:total_curve_length")],
                    dtype=np.float64,
                ),
                reductions,
            )


def test_minimal_jax_lane_matches_the_official_states(
    minimal_bundle: tuple[InputBundle, dict],
    minimal_reference: OfficialReference,
    minimal_reductions: tuple[StateReductions, StateReductions],
) -> None:
    bundle, _ = minimal_bundle
    geometry = minimal_case.build_jax_geometry(bundle)
    state = minimal_stage_two_state(
        field=geometry.field,
        flux_spec=geometry.flux_spec,
        surface_gamma=geometry.surface_gamma,
        surface_normal=geometry.surface_normal,
        num_base_curves=_configuration_int(bundle, "num_base_curves"),
        length_weight=_configuration_float(bundle, "length_weight"),
        length_target=_configuration_float(bundle, "length_target"),
    )
    states = _minimal_states(minimal_reference)

    for expectation, reductions in zip(states, minimal_reductions, strict=True):
        published = jax.device_get(state(jax.device_put(expectation.parameters)))
        lane = LaneState(
            value=float(published.objective),
            flux=float(published.squared_flux),
            # The lane publishes this term already weighted.
            remainder=float(published.length_penalty),
            gradient=np.asarray(published.objective_gradient, dtype=np.float64),
        )
        assert_state(expectation, lane, reductions)
        if expectation.label == "minimal end state":
            assert_official_lengths(
                expectation.label,
                np.asarray([published.total_curve_length], dtype=np.float64),
                np.asarray(
                    [minimal_reference.scalar("final:total_curve_length")],
                    dtype=np.float64,
                ),
                reductions,
            )
