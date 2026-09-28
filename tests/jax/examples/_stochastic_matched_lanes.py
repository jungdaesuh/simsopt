"""Matched native and JAX stochastic stage-two lanes for the evaluator parity test.

``build_shared_inputs`` builds one sample bundle and starting state; ``run_native_leg``
and ``run_jax_leg`` solve from it (native SIMSOPT objective and the JAX mirror's
objective) and evaluate both at the matched states.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray
    from simsopt.geo import Curve, GaussianSampler, SurfaceRZFourier
    from simsopt_jax.examples import (
        ExecutionScale,
        StochasticPerturbationBundle,
        StochasticStageTwoConfiguration,
    )


class ProbeConventionError(RuntimeError):
    """A probe cannot produce, or cannot publish, trustworthy evidence."""


#: SciPy's ``minimize(..., tol=)`` sets ``ftol`` and ``gtol`` for L-BFGS-B, so
#: the native example's ``tol=1e-15`` is the policy both lanes are pinned to.
MATCHED_TOLERANCE = 1.0e-15


@dataclass(frozen=True, slots=True)
class SharedInputs:
    """The one problem construction both lanes consume, plus its samples."""

    scale: ExecutionScale
    configuration: StochasticStageTwoConfiguration
    configuration_mapping: dict[str, object]
    surface: SurfaceRZFourier
    base_curves: list[Curve]
    coils: list[object]
    sampler: GaussianSampler
    training: StochasticPerturbationBundle
    construction_seconds: float


def build_shared_inputs(scale: ExecutionScale) -> SharedInputs:
    """Build geometry and materialize the training perturbations once."""
    from examples.jax.parity.cases.native_stage_two_optimization_stochastic import (
        _build_geometry,
        _scale_configuration,
        _symmetry_layout,
    )
    from simsopt.geo import GaussianSampler
    from simsopt_jax.examples import (
        materialize_stochastic_coil_perturbations,
        stochastic_stage_two_configuration,
    )

    started = time.perf_counter()
    configuration = stochastic_stage_two_configuration(scale)
    mapping = _scale_configuration(scale)
    surface, base_curves, coils = _build_geometry(mapping)
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
    return SharedInputs(
        scale=scale,
        configuration=configuration,
        configuration_mapping=mapping,
        surface=surface,
        base_curves=base_curves,
        coils=coils,
        sampler=sampler,
        training=training,
        construction_seconds=time.perf_counter() - started,
    )


@dataclass(frozen=True, slots=True)
class SolveRow:
    """One timed solve: the published row, and the unit the publication gate reads.

    ``kind`` splits first-touch from steady state; ``success``/``status``/``nit``
    are the solver's own verdict, which :func:`refused_solve_rows` reads before
    anything is published.
    """

    index: int
    kind: str
    seconds: float
    nit: int
    nfev: int
    njev: int
    objective: float
    status: int
    success: bool
    message: str


def _solve_row(
    index: int,
    *,
    seconds: float,
    nit: int,
    nfev: int,
    njev: int,
    objective: float,
    status: int,
    success: bool,
    message: str,
) -> SolveRow:
    return SolveRow(
        index=index,
        # Index 0 is the first solve in this process: it carries first-touch
        # cost on both lanes (BLAS warmup natively, trace+lowering+compile on
        # the JAX lane).  Warm and cold never fold into one number.
        kind="cold_in_process" if index == 0 else "warm",
        seconds=seconds,
        nit=nit,
        nfev=nfev,
        njev=njev,
        objective=objective,
        status=status,
        success=success,
        message=message,
    )


def evaluate_states(
    value_and_gradient: Callable[
        [NDArray[np.float64]], tuple[float, NDArray[np.float64]]
    ],
    *,
    initial_parameters: NDArray[np.float64],
    final_parameters: NDArray[np.float64],
    probe_parameters: NDArray[np.float64] | None,
) -> dict[str, dict[str, object]]:
    """Value and gradient at every state this leg can be pinned to.

    One owner for *which* states are evaluated and what is kept: the caller
    supplies only its lane's evaluator. Keyed by :data:`EVALUATION_STATES`;
    ``probe`` is absent when ``--evaluate-at`` was omitted. Full fp64 values and
    gradient arrays are returned — the artifact stamps scalars, the endpoint
    archive stores the arrays.
    """
    import numpy as np

    if (
        probe_parameters is not None
        and probe_parameters.shape != initial_parameters.shape
    ):
        raise ProbeConventionError(
            f"--evaluate-at supplied a {probe_parameters.shape} state but this "
            f"leg's DOF vector is {initial_parameters.shape}: the state belongs "
            "to a different configuration or scale, so evaluating there would "
            "compare two different problems"
        )
    states = {
        "initial": initial_parameters,
        "probe": probe_parameters,
        "final": final_parameters,
    }
    evaluated: dict[str, dict[str, object]] = {}
    for name, parameters in states.items():
        if parameters is None:
            continue
        value, gradient = value_and_gradient(np.asarray(parameters, dtype=np.float64))
        evaluated[name] = {
            "objective": float(value),
            "gradient": np.ascontiguousarray(np.asarray(gradient, dtype=np.float64)),
        }
    return evaluated


def run_native_leg(
    shared: SharedInputs,
    *,
    budget: int,
    maxcor: int,
    repeat: int,
    probe_parameters: NDArray[np.float64] | None,
) -> tuple[list[SolveRow], dict[str, object]]:
    """SciPy L-BFGS-B over the simsopt objective; timed window = ``minimize``.

    Returns the timed rows — which the caller gates and serializes — and the
    rest of the leg payload. ``shared``'s live geometry is left at the DOFs it
    arrived with, so the two lanes may be run against one ``SharedInputs``.
    """
    import numpy as np
    from scipy.optimize import minimize
    from simsopt.field import BiotSavart, Coil
    from simsopt.geo import (
        ArclengthVariation,
        CurveCurveDistance,
        CurveLength,
        CurvePerturbed,
        LpCurveCurvature,
        MeanSquaredCurvature,
        PerturbationSample,
    )
    from simsopt.objectives import QuadraticPenalty, SquaredFlux

    configuration = shared.configuration
    surface = shared.surface
    base_curves = shared.base_curves
    coils = shared.coils

    construction_start = time.perf_counter()
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
        training_fluxes.append(SquaredFlux(surface, BiotSavart(perturbed_coils)))
    training_flux = sum(training_fluxes) * (1.0 / len(training_fluxes))
    lengths = [CurveLength(curve) for curve in base_curves]
    curve_curve = CurveCurveDistance(
        [coil.curve for coil in coils],
        configuration.curve_curve_threshold,
        num_basecurves=configuration.num_base_curves,
    )
    curvatures = [
        LpCurveCurvature(curve, 2, configuration.curvature_threshold)
        for curve in base_curves
    ]
    mean_squared_curvatures = [MeanSquaredCurvature(curve) for curve in base_curves]
    arclength_variations = [ArclengthVariation(curve) for curve in base_curves]
    objective = (
        training_flux
        + configuration.length_weight * sum(lengths)
        + configuration.curve_curve_weight * curve_curve
        + configuration.curvature_weight * sum(curvatures)
        + configuration.mean_squared_curvature_weight
        * sum(
            QuadraticPenalty(
                value,
                configuration.mean_squared_curvature_threshold,
                "max",
            )
            for value in mean_squared_curvatures
        )
        + configuration.arclength_variation_weight * sum(arclength_variations)
    )
    initial_parameters = np.asarray(BiotSavart(coils).x, dtype=np.float64)
    construction_seconds = time.perf_counter() - construction_start

    def value_and_gradient(parameters: NDArray[np.float64]):
        objective.x = parameters
        return float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)

    options = {"maxiter": budget, "maxcor": maxcor}
    rows: list[SolveRow] = []
    result = None
    for index in range(repeat):
        started = time.perf_counter()
        result = minimize(
            value_and_gradient,
            initial_parameters,
            jac=True,
            method="L-BFGS-B",
            options=options,
            tol=MATCHED_TOLERANCE,
        )
        seconds = time.perf_counter() - started
        rows.append(
            _solve_row(
                index,
                seconds=seconds,
                nit=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                objective=float(result.fun),
                status=int(result.status),
                success=bool(result.success),
                message=str(result.message),
            )
        )
        print(
            f"native solve {index} seconds={seconds:.6f} nit={result.nit}", flush=True
        )
    final_parameters = np.asarray(result.x, dtype=np.float64)
    evaluations = evaluate_states(
        value_and_gradient,
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        probe_parameters=probe_parameters,
    )
    # ``shared`` holds one live simsopt geometry, and every ``objective.x =``
    # above — the solve's and the evaluations' — mutated it.  Hand it back as
    # found so a caller that runs both lanes in one process (the matched-state
    # parity test) starts the second lane from the coordinates the first did.
    objective.x = initial_parameters
    return rows, {
        "driver": "scipy_lbfgsb",
        "policy": {
            "maxiter": budget,
            "maxcor": maxcor,
            "tol": MATCHED_TOLERANCE,
            "scipy_options": dict(options),
            "scipy_tol_sets": ["ftol", "gtol"],
        },
        "construction_seconds": construction_seconds,
        "dof_count": int(initial_parameters.size),
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "final_objective": float(result.fun),
        "evaluations": evaluations,
    }


def run_jax_leg(
    shared: SharedInputs,
    *,
    budget: int,
    maxcor: int,
    repeat: int,
    sample_tile: int | None,
    probe_parameters: NDArray[np.float64] | None,
    jax_driver: str = "simsopt_lbfgsb",
) -> tuple[list[SolveRow], dict[str, object]]:
    """The mirror's device L-BFGS-B lane; timed window = ``minimize``.

    Returns the timed rows — which the caller gates and serializes — and the
    rest of the leg payload.
    """
    import jax
    import numpy as np
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.objectives import (
        StageTwoObjectiveConfig,
        StochasticCoilPerturbations,
        make_stochastic_stage_two_objective,
    )
    from simsopt_jax.solve.dispatch import minimize
    from simsopt_jax.solve.driver import Driver
    from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions
    from simsopt_jax.solve.serial import TraceableScalarProblem
    from simsopt_jax.solve.simsopt.contracts import SimsoptLBFGSBOptions
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

    configuration = shared.configuration
    construction_start = time.perf_counter()
    field = BiotSavartJAX(shared.coils)
    flux = SquaredFluxJAX(shared.surface, field)
    flux_spec = flux.fixed_surface_flux_spec()
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
    objective_config = StageTwoObjectiveConfig(
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
    )
    # ``sample_tile=None`` is the factory's own default (the sequential scan
    # oracle), and the factory validates the lever against this bundle's sample
    # count at build time, typed — so a bad tile fails here, before the clock.
    objective = make_stochastic_stage_two_objective(
        field,
        flux_spec,
        training,
        surface_gamma,
        surface_normal,
        objective_config,
        sample_tile=sample_tile,
    )
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    initial_device = jax.device_put(initial_parameters, device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    jax.block_until_ready(initial_device)
    construction_seconds = time.perf_counter() - construction_start

    # One matched policy, spelled in each route's own options class.  The SciPy
    # route builds it through ``ScipyLBFGSBOptions.native_matched``, the one
    # owner of "what a native script that names only maxiter/maxcor/tol means",
    # so this lane and the shipped mirror cannot drift apart silently.
    if jax_driver == "scipy_lbfgsb":
        driver_choice = Driver.SCIPY_LBFGSB
        options = ScipyLBFGSBOptions.native_matched(
            maxiter=budget, maxcor=maxcor, tol=MATCHED_TOLERANCE
        )
    else:
        driver_choice = Driver.SIMSOPT_LBFGSB
        options = SimsoptLBFGSBOptions(
            maxiter=budget,
            maxcor=maxcor,
            ftol=MATCHED_TOLERANCE,
            gtol=MATCHED_TOLERANCE,
        )
    # The cache-marked solver callable, exactly as ``serial_solve_jax`` passes
    # it: the public bound method is unmarked, so the fused L-BFGS executable cache would
    # re-trace every solve and "warm" would silently include lowering.  No step
    # observer is attached, which is what selects ``fused_stepwise`` in
    # ``simsopt_jax.solve.dispatch._legacy_lbfgsb_options``.
    solver_value_and_grad = problem._solver_value_and_grad_fn

    def solve():
        return minimize(
            solver_value_and_grad,
            initial_device,
            driver=driver_choice,
            options=options,
        )

    rows: list[SolveRow] = []
    result = None
    for index in range(repeat):
        started = time.perf_counter()
        # ``minimize`` returns ``result.x`` on the host, so the device queue is
        # already drained when the clock stops; the explicit block above covers
        # the staging that precedes it.
        result = solve()
        seconds = time.perf_counter() - started
        rows.append(
            _solve_row(
                index,
                seconds=seconds,
                nit=int(result.nit),
                nfev=int(result.nfev),
                njev=int(result.njev),
                objective=float(result.fun),
                status=int(result.status),
                success=bool(result.success),
                message=str(result.message),
            )
        )
        print(f"jax solve {index} seconds={seconds:.6f} nit={result.nit}", flush=True)

    # Outside every timed window: the public bound method, so this is the same
    # evaluator ``serial_solve_jax`` reports its bounded objective from, and the
    # comparable half of a cross-lane check that endpoint coordinates are not.
    def value_and_gradient(parameters: NDArray[np.float64]):
        value_device, gradient_device = problem.value_and_grad(
            jax.device_put(np.asarray(parameters, dtype=np.float64), device)
        )
        value, gradient = jax.device_get(
            jax.block_until_ready((value_device, gradient_device))
        )
        return float(value), np.asarray(gradient, dtype=np.float64)

    final_parameters = np.asarray(result.x, dtype=np.float64)
    evaluations = evaluate_states(
        value_and_gradient,
        initial_parameters=initial_parameters,
        final_parameters=final_parameters,
        probe_parameters=probe_parameters,
    )
    devices = [
        {"platform": str(item.platform), "kind": str(item.device_kind)}
        for item in jax.local_devices()
    ]
    return rows, {
        "driver": jax_driver,
        "policy": {
            "options_type": type(options).__name__,
            **asdict(options),
            "step_observer_attached": False,
            "sample_tile": sample_tile,
        },
        "construction_seconds": construction_seconds,
        "dof_count": int(initial_parameters.size),
        "initial_parameters": initial_parameters,
        "final_parameters": final_parameters,
        "final_objective": float(result.fun),
        "evaluations": evaluations,
        "jax_devices": devices,
        "solve_device": None if device is None else str(device.platform),
    }
