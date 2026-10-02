"""Matched-state evaluator parity for the stochastic stage-two lanes.

Endpoint coordinates of an iteration-capped L-BFGS-B run are not a parity
criterion — the native lane's own endpoint moves under a change of
``OMP_NUM_THREADS`` alone.  What *is* a criterion, and what this file pins, is
the evaluator: handed the same samples and the same coordinates, the native
SIMSOPT objective of ``examples/2_Intermediate/stage_two_optimization_stochastic.py``
and the JAX mirror's ``make_stochastic_stage_two_objective`` must return the
same value and the same gradient to fp64 round-off.

Both lanes are built here from one shared sample bundle and starting state,
constructed as the shipped mirror constructs them.  The scale is ``bounded``
(2 samples, 4x4 surface, order-2 curves) and every solve is one iteration: this
is an evaluator test, not a trajectory test.

Run one file per process (JAX x64 is process-global)::

    PYTHONPATH=src:build/cp311-cp311-linux_x86_64 JAX_ENABLE_X64=1 \
    JAX_PLATFORMS=cpu MPI4PY_RC_INITIALIZE=false OMP_NUM_THREADS=4 \
    .venv/bin/python -m pytest tests/jax/examples/test_stochastic_matched_state_parity.py
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import jax
import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.optimize import minimize as scipy_minimize
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.geo import (
    ArclengthVariation,
    Curve,
    CurveCurveDistance,
    CurveLength,
    CurvePerturbed,
    GaussianSampler,
    LpCurveCurvature,
    MeanSquaredCurvature,
    PerturbationSample,
    RotatedCurve,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    StochasticPerturbationBundle,
    StochasticStageTwoConfiguration,
    materialize_stochastic_coil_perturbations,
    stochastic_stage_two_configuration,
)
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
)
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import TraceableScalarProblem, _scalar_options
from simsopt_jax.solve.simsopt.contracts import SimsoptLBFGSBOptions
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

ROOT = Path(__file__).resolve().parents[3]
#: The lane the mirror mirrors; its optimizer call is the policy SSOT.
NATIVE_EXAMPLE = (
    ROOT / "examples" / "2_Intermediate" / "stage_two_optimization_stochastic.py"
)
SURFACE_INPUT = ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"


#: fp64 round-off over a 2-sample bounded objective, not a physics tolerance:
#: the observed agreement at this scale is ~5e-17 absolute on both quantities,
#: and at the shipped native_default scale ~5e-16 on the gradient.
EVALUATOR_ATOL = 1.0e-12
EVALUATOR_RTOL = 1.0e-9

#: SciPy's ``minimize(..., tol=)`` sets ``ftol`` and ``gtol`` for L-BFGS-B, so
#: the native example's ``tol=1e-15`` is the policy both lanes are pinned to.
MATCHED_TOLERANCE = 1.0e-15
#: One iteration and a short history: an evaluator test, not a trajectory test.
SOLVE_BUDGET = 1
SOLVE_MAXCOR = 10

Evaluations = dict[str, dict[str, object]]


@dataclass(frozen=True, slots=True)
class _SharedInputs:
    """The one problem construction both lanes consume, plus its samples."""

    configuration: StochasticStageTwoConfiguration
    surface: SurfaceRZFourier
    base_curves: list[Curve]
    coils: list[Coil]
    sampler: GaussianSampler
    training: StochasticPerturbationBundle


@dataclass(frozen=True, slots=True)
class _Leg:
    """What one lane hands the assertions: its states and its evaluations."""

    dof_count: int
    initial_parameters: NDArray[np.float64]
    final_parameters: NDArray[np.float64]
    evaluations: Evaluations


def _symmetry_layout(
    coils: list[Coil], base_curves: list[Curve]
) -> tuple[tuple[int, ...], np.ndarray]:
    """Source base curve and rotation of every symmetry-expanded coil."""
    source_indices = tuple(
        final_index % len(base_curves) for final_index in range(len(coils))
    )
    rotations: list[np.ndarray] = []
    for coil, source_index in zip(coils, source_indices, strict=True):
        curve = coil.curve
        if curve is base_curves[source_index]:
            rotations.append(np.eye(3, dtype=np.float64))
        else:
            assert isinstance(curve, RotatedCurve)
            assert curve.curve is base_curves[source_index]
            rotations.append(np.asarray(curve.rotmat, dtype=np.float64))
    return source_indices, np.stack(rotations)


def _build_shared_inputs() -> _SharedInputs:
    """Build the bounded geometry and materialize the training samples once."""
    configuration = stochastic_stage_two_configuration("bounded")
    surface = SurfaceRZFourier.from_vmec_input(
        str(SURFACE_INPUT),
        range="full torus",
        nphi=configuration.surface_nphi,
        ntheta=configuration.surface_ntheta,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        configuration.num_base_curves,
        surface.nfp,
        stellsym=True,
        R0=configuration.major_radius,
        R1=configuration.minor_radius,
        order=configuration.curve_order,
        numquadpoints=configuration.curve_quadrature,
    )
    base_currents = [Current(configuration.initial_current) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    source_indices, rotations = _symmetry_layout(coils, base_curves)
    sampler = GaussianSampler(
        base_curves[0].quadpoints,
        sigma=configuration.perturbation_sigma,
        length_scale=configuration.perturbation_length_scale,
        n_derivs=1,
    )
    training = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(base_curves),
        sample_count=configuration.training_sample_count,
        seed=configuration.training_seed,
    )
    return _SharedInputs(
        configuration=configuration,
        surface=surface,
        base_curves=base_curves,
        coils=coils,
        sampler=sampler,
        training=training,
    )


def _evaluate_states(
    value_and_gradient: Callable[
        [NDArray[np.float64]], tuple[float, NDArray[np.float64]]
    ],
    *,
    initial_parameters: NDArray[np.float64],
    final_parameters: NDArray[np.float64],
    probe_parameters: NDArray[np.float64] | None,
) -> Evaluations:
    """Value and gradient at every state this leg can be pinned to."""
    states = {
        "initial": initial_parameters,
        "probe": probe_parameters,
        "final": final_parameters,
    }
    evaluated: Evaluations = {}
    for name, parameters in states.items():
        if parameters is None:
            continue
        value, gradient = value_and_gradient(np.asarray(parameters, dtype=np.float64))
        evaluated[name] = {
            "objective": float(value),
            "gradient": np.ascontiguousarray(np.asarray(gradient, dtype=np.float64)),
        }
    return evaluated


def _run_native_leg(shared: _SharedInputs) -> _Leg:
    """SciPy L-BFGS-B over the native SIMSOPT stochastic objective.

    ``shared``'s live geometry is left at the DOFs it arrived with, so the two
    lanes may be run against one ``_SharedInputs``.
    """
    configuration = shared.configuration
    base_curves = shared.base_curves
    coils = shared.coils
    training_fluxes = []
    for gamma_sample, gammadash_sample in zip(
        shared.training.gamma,
        shared.training.gammadash,
        strict=True,
    ):
        perturbed_coils = [
            Coil(
                CurvePerturbed(
                    coil.curve,
                    PerturbationSample(
                        shared.sampler,
                        sample=[gamma_perturbation, gammadash_perturbation],
                    ),
                ),
                coil.current,
            )
            for coil, gamma_perturbation, gammadash_perturbation in zip(
                coils,
                gamma_sample,
                gammadash_sample,
                strict=True,
            )
        ]
        training_fluxes.append(SquaredFlux(shared.surface, BiotSavart(perturbed_coils)))
    training_flux = sum(training_fluxes) * (1.0 / len(training_fluxes))
    objective = (
        training_flux
        + configuration.length_weight * sum(CurveLength(curve) for curve in base_curves)
        + configuration.curve_curve_weight
        * CurveCurveDistance(
            [coil.curve for coil in coils],
            configuration.curve_curve_threshold,
            num_basecurves=configuration.num_base_curves,
        )
        + configuration.curvature_weight
        * sum(
            LpCurveCurvature(curve, 2, configuration.curvature_threshold)
            for curve in base_curves
        )
        + configuration.mean_squared_curvature_weight
        * sum(
            QuadraticPenalty(
                MeanSquaredCurvature(curve),
                configuration.mean_squared_curvature_threshold,
                "max",
            )
            for curve in base_curves
        )
        + configuration.arclength_variation_weight
        * sum(ArclengthVariation(curve) for curve in base_curves)
    )
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)

    def value_and_gradient(parameters: NDArray[np.float64]):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    result = scipy_minimize(
        value_and_gradient,
        initial_parameters,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": SOLVE_BUDGET, "maxcor": SOLVE_MAXCOR},
        tol=MATCHED_TOLERANCE,
    )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    evaluations = _evaluate_states(
        value_and_gradient,
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        probe_parameters=None,
    )
    # ``shared`` holds one live simsopt geometry, and every ``objective.x =``
    # above mutated it.  Hand it back as found so the JAX lane starts from the
    # coordinates this one did.
    objective.x = initial_parameters
    return _Leg(
        dof_count=int(initial_parameters.size),
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        evaluations=evaluations,
    )


def _run_jax_leg(
    shared: _SharedInputs, *, probe_parameters: NDArray[np.float64]
) -> _Leg:
    """The mirror's objective under the device L-BFGS-B lane."""
    configuration = shared.configuration
    field = BiotSavartJAX(shared.coils)
    flux_spec = SquaredFluxJAX(shared.surface, field).fixed_surface_flux_spec()
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(shared.surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    surface_normal = jax.device_put(
        np.asarray(shared.surface.normal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    training = StochasticCoilPerturbations(
        gamma=jax.device_put(shared.training.gamma, device),
        gammadash=jax.device_put(shared.training.gammadash, device),
    )
    objective = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        StageTwoObjectiveConfig(
            num_base_curves=configuration.num_base_curves,
            length_weight=configuration.length_weight,
            curve_curve_minimum_distance=configuration.curve_curve_threshold,
            curve_curve_weight=configuration.curve_curve_weight,
            curvature_threshold=configuration.curvature_threshold,
            curvature_weight=configuration.curvature_weight,
            mean_squared_curvature_threshold=(
                configuration.mean_squared_curvature_threshold
            ),
            mean_squared_curvature_weight=configuration.mean_squared_curvature_weight,
            arclength_variation_weight=configuration.arclength_variation_weight,
        ),
    )
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    initial_device = jax.device_put(initial_parameters, device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    # The cache-marked solver callable, exactly as ``serial_solve_jax`` passes it.
    result = minimize(
        problem._solver_value_and_grad_fn,
        initial_device,
        driver=Driver.SIMSOPT_LBFGSB,
        options=SimsoptLBFGSBOptions(
            maxiter=SOLVE_BUDGET,
            maxcor=SOLVE_MAXCOR,
            ftol=MATCHED_TOLERANCE,
            gtol=MATCHED_TOLERANCE,
        ),
    )

    # The public bound method: the evaluator ``serial_solve_jax`` reports from.
    def value_and_gradient(parameters: NDArray[np.float64]):
        value_device, gradient_device = problem.value_and_grad(
            jax.device_put(np.asarray(parameters, dtype=np.float64), device)
        )
        value, gradient = jax.device_get(
            jax.block_until_ready((value_device, gradient_device))
        )
        return float(value), np.asarray(gradient, dtype=np.float64)

    final_parameters = np.asarray(result.x, dtype=np.float64)
    return _Leg(
        dof_count=int(initial_parameters.size),
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        evaluations=_evaluate_states(
            value_and_gradient,
            initial_parameters=initial_parameters,
            final_parameters=final_parameters,
            probe_parameters=probe_parameters,
        ),
    )


@pytest.fixture(scope="module")
def matched_lanes() -> tuple[_SharedInputs, _Leg, _Leg]:
    """Both lanes at ``bounded`` scale, the JAX lane probed at native's endpoint.

    The native leg's ``final`` state and the JAX leg's ``probe`` state are the
    same 96 coordinates by construction: the JAX leg is handed the array the
    native leg ended on.  That, plus the shared ``initial`` state, is two
    matched states for the price of two legs.
    """
    shared = _build_shared_inputs()
    native = _run_native_leg(shared)
    jax_leg = _run_jax_leg(shared, probe_parameters=native.final_parameters)
    return shared, native, jax_leg


def test_mirror_convergence_tolerances_equal_the_native_example_call() -> None:
    """``tol=1e-15`` in the native call is *two* tolerances, and both must mirror.

    ``scipy.optimize.minimize(..., method="L-BFGS-B", tol=x)`` sets ``ftol`` and
    ``gtol`` to ``x``, so the native example runs at ``ftol = gtol = 1e-15``.
    ``serial_solve_jax`` spells the same pair ``rtol -> ftol``, ``atol -> gtol``
    (``src/simsopt_jax/solve/serial.py::_scalar_options``); a mirror that left
    ``atol`` at the 1e-8 library default would stop on a gradient test seven
    orders looser than the lane it mirrors.
    """
    native_source = NATIVE_EXAMPLE.read_text(encoding="utf-8")
    assert "tol=1e-15" in native_source
    assert "method='L-BFGS-B'" in native_source

    for scale in ("bounded", "native_default"):
        configuration = stochastic_stage_two_configuration(scale)
        assert configuration.rtol == 1.0e-15
        assert configuration.atol == 1.0e-15, (
            f"{scale} mirror would pass gtol={configuration.atol!r} where the "
            "native example passes 1e-15"
        )

    options = _scalar_options(
        Driver.SIMSOPT_LBFGSB,
        rtol=1.0e-15,
        atol=1.0e-15,
        max_steps=400,
        maxcor=10,
        line_search_max_steps=None,
    )
    assert (options.ftol, options.gtol) == (1.0e-15, 1.0e-15)


def test_training_samples_are_reproducible_from_seed_rng_and_ordering(
    matched_lanes: tuple[_SharedInputs, _Leg, _Leg],
) -> None:
    shared, _, _ = matched_lanes
    sampler = GaussianSampler(
        shared.base_curves[0].quadpoints,
        sigma=shared.configuration.perturbation_sigma,
        length_scale=shared.configuration.perturbation_length_scale,
        n_derivs=1,
    )
    source_indices, rotations = _symmetry_layout(shared.coils, shared.base_curves)
    redrawn = materialize_stochastic_coil_perturbations(
        sampler,
        source_indices=source_indices,
        rotations=rotations,
        base_curve_count=len(shared.base_curves),
        sample_count=shared.configuration.training_sample_count,
        seed=shared.configuration.training_seed,
    )

    assert redrawn.sha256 == shared.training.sha256, (
        "redrawing the training bundle from the same seed produced a different "
        "fingerprint: the PCG64DXSM stream, the draw ordering "
        "(systematic-per-base-curve then statistical-per-coil) or the dtype "
        "moved, and the two lanes are no longer perturbed by the same bytes"
    )
    np.testing.assert_array_equal(redrawn.gamma, shared.training.gamma)
    np.testing.assert_array_equal(redrawn.gammadash, shared.training.gammadash)


def test_both_lanes_start_from_one_initial_state(
    matched_lanes: tuple[_SharedInputs, _Leg, _Leg],
) -> None:
    _, native, jax_leg = matched_lanes

    # The initial DOFs are read independently per lane (simsopt BiotSavart vs
    # BiotSavartJAX) and must still agree bit for bit.
    assert native.dof_count == jax_leg.dof_count
    np.testing.assert_array_equal(
        native.initial_parameters,
        jax_leg.initial_parameters,
        err_msg=(
            "the two lanes started from different coordinates, so nothing "
            "downstream of this is a parity measurement"
        ),
    )


def test_evaluator_agrees_at_the_matched_initial_state(
    matched_lanes: tuple[_SharedInputs, _Leg, _Leg],
) -> None:
    _, native, jax_leg = matched_lanes
    native_state = native.evaluations["initial"]
    jax_state = jax_leg.evaluations["initial"]

    assert native_state["objective"] == pytest.approx(
        jax_state["objective"], rel=EVALUATOR_RTOL, abs=EVALUATOR_ATOL
    ), (
        "stochastic objective value disagrees at the shared initial state: "
        f"native {native_state['objective']!r} vs jax {jax_state['objective']!r}"
    )
    gradient_gap = float(
        np.max(np.abs(native_state["gradient"] - jax_state["gradient"]))
    )
    assert gradient_gap <= EVALUATOR_ATOL, (
        "stochastic gradient disagrees at the shared initial state by "
        f"{gradient_gap:.3e}, above fp64 round-off; an evaluator defect, not "
        "an optimizer trajectory difference"
    )


def test_evaluator_agrees_at_the_native_endpoint_handed_to_the_jax_lane(
    matched_lanes: tuple[_SharedInputs, _Leg, _Leg],
) -> None:
    _, native, jax_leg = matched_lanes
    native_state = native.evaluations["final"]
    jax_state = jax_leg.evaluations["probe"]

    assert native_state["objective"] == pytest.approx(
        jax_state["objective"], rel=EVALUATOR_RTOL, abs=EVALUATOR_ATOL
    ), (
        "stochastic objective value disagrees at the native endpoint: "
        f"native {native_state['objective']!r} vs jax {jax_state['objective']!r}"
    )
    gradient_gap = float(
        np.max(np.abs(native_state["gradient"] - jax_state["gradient"]))
    )
    assert gradient_gap <= EVALUATOR_ATOL, (
        "stochastic gradient disagrees at the native endpoint by "
        f"{gradient_gap:.3e}, above fp64 round-off"
    )


def test_every_leg_retains_an_endpoint_gradient(
    matched_lanes: tuple[_SharedInputs, _Leg, _Leg],
) -> None:
    """Endpoints are retained with their gradients — and compared by nobody.

    A capped run's endpoint is not reproducible even native-to-native across
    thread counts, so no test here gates on ``final_parameters`` agreement; what
    the driver owes a reader is the gradient at the state it stopped on, which
    says how far from stationary that state is.
    """
    _, native, jax_leg = matched_lanes

    for lane, leg in (("native", native), ("jax", jax_leg)):
        gradient = leg.evaluations["final"]["gradient"]
        assert gradient.shape == (leg.dof_count,), (
            f"{lane} leg retained no endpoint gradient of the right shape: "
            f"{gradient.shape}"
        )
        assert np.all(np.isfinite(gradient))
