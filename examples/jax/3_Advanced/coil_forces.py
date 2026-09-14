"""JAX port of ``examples/3_Advanced/coil_forces.py``.

The host builds the native Landreman-Paul surface, regularized coil graph, and
symmetry-expanded currents.  Quadratic flux, engineering penalties, Lorentz
force, vacuum energy, gradients, the directional Taylor test, and both
optimization stages then execute on the selected JAX CPU or GPU.  The stage
length weight is an explicit device parameter, so one compiled graph serves both
stages.  ``STAGE_DRIVER`` selects the L-BFGS-B implementation: the default is
native's own SciPy routine driving the device objective, and the fused device
ports remain reachable through the same constant.  Both stages start from
native's post-Taylor-test state, because the native script reuses ``JF.x`` --
the last Taylor evaluation -- as its optimizer guess.  Only accepted endpoint
diagnostics return to the host.

Splitting the length penalty out of the stage objective is bitwise invariant
against the pre-change formulation (and so is the aligned parity mirror) only
while the total base-curve length stays under the 17.4 m target and the penalty
is therefore exactly zero; once it is active the two spellings reassociate the
same sum and their values differ at the ~1 ULP level.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    solve_scalar_stage,
    two_stage_optimizer_observables,
)
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableArrayFunction,
    TraceableParametricScalarProblem,
    serial_solve_jax,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    ForceStageTwoConfig,
    force_stage_two_diagnostics,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-coil-forces"
NATIVE_ITERATIONS = 400
FIRST_LENGTH_WEIGHT = 1.0e-3
SECOND_LENGTH_WEIGHT = 1.0e-4
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"

#: ``tol`` the native script hands ``scipy.optimize.minimize``. SciPy's
#: ``minimize`` forwards one ``tol`` to L-BFGS-B as *both* ``ftol`` and
#: ``gtol``, so a single constant carries native's whole stopping policy.
NATIVE_TOLERANCE = 1.0e-15
#: Native's ``options={'maxcor': 300}``.
NATIVE_HISTORY_SIZE = 300
#: Native's Taylor-test step sizes, in the order it evaluates them.
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)
#: L-BFGS-B implementation both stages use. ``Driver.SCIPY_LBFGSB`` is native's
#: own routine driving the device objective program; ``Driver.SIMSOPT_LBFGSB``
#: and ``Driver.SIMSOPT_BFGS`` select the fused device ports instead.
STAGE_DRIVER = Driver.SCIPY_LBFGSB


def _stage_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=3,
        length_target=17.4,
        length_target_mode="max",
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def _force_config() -> ForceStageTwoConfig:
    return ForceStageTwoConfig(
        num_force_coils=3,
        force_weight=1.0e-2,
        vacuum_energy_weight=1.0e-4,
        force_power=4.0,
        force_threshold=0.0,
        downsample=1,
    )


def _build_problem(
    scale: ExecutionScale,
) -> tuple[
    BiotSavartJAX,
    SquaredFluxJAX,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    native_scale = scale == "native_default"
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=32 if native_scale else 4,
        ntheta=32 if native_scale else 4,
    )
    base_curves = create_equally_spaced_curves(
        3,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=5 if native_scale else 2,
        numquadpoints=75 if native_scale else 8,
        use_jax_curve=False,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    regularization = 0.05**2 / np.sqrt(np.e)
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [regularization for _ in base_curves],
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    # Place on the runtime policy's device, never on the process default: a
    # process whose platform nobody pinned defaults to the CPU once native
    # simsopt has been imported (``src/simsopt/geo/jit.py``).
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)), device
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)), device
    )
    target_quadpoints = jax.device_put(
        np.stack(
            tuple(
                np.asarray(curve.quadpoints, dtype=np.float64) for curve in base_curves
            )
        ),
        device,
    )
    regularizations = jax.device_put(
        np.full(len(coils), regularization, dtype=np.float64), device
    )
    return (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    )


def _native_taylor_direction(dof_size: int) -> np.ndarray:
    """Native's ``np.random.seed(1); np.random.uniform(size=dofs.shape)``."""
    return np.random.RandomState(1).uniform(size=dof_size)


def _native_optimization_start(
    initial_parameters: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """The state native minimizes from: its last Taylor-test evaluation.

    ``examples/3_Advanced/coil_forces.py`` evaluates the objective at
    ``dofs - eps*h`` last, leaves that state on ``JF``, and then reuses
    ``JF.x`` as the optimizer guess, so the optimization never starts from
    the unperturbed coil set.
    """
    return initial_parameters - TAYLOR_EPSILONS[-1] * direction


def _taylor_errors(
    problem: TraceableParametricScalarProblem,
    initial: jax.Array,
    direction: jax.Array,
) -> jax.Array:
    """Native's central-difference gradient check, one error per epsilon."""
    _value, gradient = problem.value_and_grad(initial)
    directional_derivative = jnp.vdot(gradient, direction)
    # Native's step sizes are host constants, so placing them is a real
    # host-to-device crossing and is spelled as one: ``jnp.asarray`` would make
    # the same transfer implicitly and a strict transfer guard would refuse it.
    epsilons = jax.device_put(
        np.asarray(TAYLOR_EPSILONS, dtype=initial.dtype), initial.device
    )

    def error(epsilon: jax.Array) -> jax.Array:
        forward = problem.objective(initial + epsilon * direction)
        backward = problem.objective(initial - epsilon * direction)
        return (forward - backward) / (epsilon + epsilon) - directional_derivative

    return jax.vmap(error)(epsilons)


def _run_stage(
    problem: TraceableParametricScalarProblem,
    *,
    driver: Driver,
    max_steps: int,
) -> OptimizerResult:
    """One stage from ``problem.x``, at this script's native-matched policy.

    ``Driver.SIMSOPT_BFGS`` is not an L-BFGS-B route -- it keeps no curvature
    history and bounds its own line search instead -- so it is driven here.
    Both L-BFGS-B routes, SciPy's and the fused device port's, are
    :func:`solve_scalar_stage`'s, which is where their shared policy lives.
    """
    if driver == Driver.SIMSOPT_BFGS:
        return serial_solve_jax(
            problem,
            driver=driver,
            max_steps=max_steps,
            line_search_max_steps=40,
            rtol=NATIVE_TOLERANCE,
            atol=NATIVE_TOLERANCE,
            require_success=False,
        )
    return solve_scalar_stage(
        problem,
        driver=driver,
        max_steps=max_steps,
        maxcor=NATIVE_HISTORY_SIZE,
        tol=NATIVE_TOLERANCE,
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    solve_started = perf_counter()
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = _build_problem(scale)
    flux_objective = flux.traceable_objective()
    force_config = _force_config()
    stage_config = _stage_config()
    objective = make_force_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
        stage_config,
        force_config,
    )
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)

    def weighted_objective(
        parameters: jax.Array,
        length_weight: jax.Array,
    ) -> jax.Array:
        return objective(parameters) + length_penalty(parameters, length_weight)

    initial_parameters = np.asarray(field.x, dtype=np.float64)
    taylor_direction = _native_taylor_direction(initial_parameters.size)
    start_parameters = _native_optimization_start(initial_parameters, taylor_direction)
    device = get_runtime_jax_device()
    initial_device = jax.device_put(initial_parameters, device)
    direction_device = jax.device_put(taylor_direction, device)
    problem = TraceableParametricScalarProblem(
        objective_fn=weighted_objective,
        objective_parameter=jax.device_put(
            np.asarray(FIRST_LENGTH_WEIGHT, dtype=np.float64), device
        ),
        x=jax.device_put(start_parameters, device),
    )
    initial_objective_device = problem.objective(initial_device)
    start_objective_device, start_gradient_device = problem.value_and_grad(problem.x)
    taylor_errors_device = _taylor_errors(problem, initial_device, direction_device)
    minimize_region_started = perf_counter()
    first_result = _run_stage(problem, driver=STAGE_DRIVER, max_steps=max_steps)
    problem.set_objective_parameter(
        jax.device_put(np.asarray(SECOND_LENGTH_WEIGHT, dtype=np.float64), device)
    )
    second_result = _run_stage(problem, driver=STAGE_DRIVER, max_steps=max_steps)
    two_stage_minimize_seconds = perf_counter() - minimize_region_started
    solution_device = jax.block_until_ready(problem.x)
    final_objective_device, gradient_device = problem.value_and_grad(solution_device)
    # Read before the host boundary: the endpoint objective is an output of the
    # compiled program, so its device is where the optimization actually ran.
    execution_device = str(final_objective_device.device)
    force_diagnostics = force_stage_two_diagnostics(
        field,
        target_quadpoints,
        regularizations,
        force_config,
    )
    diagnostics = TraceableArrayFunction(
        function_fn=lambda parameters: jnp.concatenate(
            (
                force_diagnostics(parameters),
                flux_objective(parameters)[None],
            )
        ),
        x=solution_device,
    )
    force_values_device = jax.block_until_ready(diagnostics(solution_device))
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    final_gradient = np.asarray(jax.device_get(gradient_device), dtype=np.float64)
    start_gradient = np.asarray(
        jax.device_get(start_gradient_device),
        dtype=np.float64,
    )
    scalar_values = np.asarray(
        jax.device_get(
            jnp.concatenate(
                (
                    jnp.stack(
                        (
                            initial_objective_device,
                            start_objective_device,
                            final_objective_device,
                        )
                    ),
                    force_values_device,
                    taylor_errors_device,
                )
            )
        ),
        dtype=np.float64,
    )
    (
        initial_objective,
        start_objective,
        final_objective,
        force_objective,
        maximum_force,
        vacuum_energy,
        squared_flux,
    ) = (float(value) for value in scalar_values[:7])
    taylor_errors = scalar_values[7:]
    solver_accepted = bool(
        first_result.status in (0, 1) and second_result.status in (0, 1)
    )
    scientific_success = bool(
        solver_accepted
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(final_gradient))
        and np.all(np.isfinite(scalar_values))
        and final_objective < initial_objective
        and force_objective >= 0.0
        and maximum_force >= 0.0
        and vacuum_energy >= 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(
                float(value) for value in initial_parameters
            ),
            "initial_objective": initial_objective,
            "start_parameters": tuple(float(value) for value in start_parameters),
            "start_objective": start_objective,
            "start_gradient": tuple(float(value) for value in start_gradient),
            "taylor_errors": tuple(float(value) for value in taylor_errors),
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "squared_flux": squared_flux,
            "force_objective": force_objective,
            "maximum_force": maximum_force,
            "vacuum_energy": vacuum_energy,
            "final_gradient": tuple(float(value) for value in final_gradient),
            # Solver outcome, policy and clocks under the one spelling every
            # two-stage mirror publishes, so one reader reads them all the same
            # way.  The per-stage clocks bracket each ``minimize`` call alone.
            **two_stage_optimizer_observables(
                first_result,
                second_result,
                execution_device=execution_device,
                two_stage_minimize_seconds=two_stage_minimize_seconds,
                whole_call_seconds=perf_counter() - solve_started,
            ),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-coil-forces-",
        bounded_steps=3,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
