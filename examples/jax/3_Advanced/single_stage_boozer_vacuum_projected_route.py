"""VMEC-free single-stage optimization by the projected full-space route.

This is the same physics as ``3_Advanced/single_stage_boozer_vacuum_optimization``
-- the NCSX coil set, an implicit Boozer surface at fixed volume, and the
non-quasisymmetry / Boozer-residual / iota / major-radius / length objective --
solved by a different formulation.

The mirror example nests the two problems: an outer optimizer moves coil degrees
of freedom, and every outer evaluation runs an inner Newton solve to put the
surface back on the Boozer manifold.  This script COUPLES them.  Coil dofs,
surface dofs, the rotational transform and the net poloidal current are one
state vector, the Boozer residual and the volume label are exact equality
CONSTRAINTS on that vector, and the route walks the constraint manifold
directly:

* a certified Gram factorization projects the objective gradient into the
  tangent space and supplies the least-squares multipliers,
* a truncated Newton-CG solve on the projected LAGRANGIAN curvature
  ``P (grad2 Phi + sum_i lambda_i grad2 c_i) P`` gives the step -- the
  constraint-curvature term is exactly what an objective-only model omits,
* an Armijo line search on the retracted objective decides how far to go, and
* an exact retraction restores feasibility at every accepted iterate, so the
  raw equality residuals never leave 1e-10.

There is no trust region and no penalty: the constraints are enforced, not
priced.  The loop runs on the host, one jitted kernel per phase, so nothing in
the compiled program depends on the iteration budget and a two-step run compiles
the same executables a seven-hundred-step run does.

``--smoke`` runs the SAME problem for two iterations; it is a bounded diagnostic
lane, and its ``ok`` status means the route stayed feasible and descended, never
that it converged.  The route's configuration below is the one certified in
``docs/single_stage_jax_gpu_projected_route_certification_plan.md``; the
certification chain owns the frozen copy and
``tests/jax/examples/test_single_stage_boozer_vacuum_projected_route_example.py``
pins this one to it so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsOptions,
    ProjectedLbfgsStatus,
    build_projected_lbfgs_kernels,
    run_projected_lbfgs,
)
from simsopt_jax.objectives.single_stage_fullspace import (
    evaluate_fullspace,
    flatten_fullspace_constraints,
)
from simsopt_jax.runtime.host_boundary import host_array, host_float
from simsopt_jax.solve.fullspace import (
    fullspace_optimizer_coordinates,
    fullspace_physical_coordinates,
    fullspace_scaling_from_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    build_single_stage_fullspace_bootstrap,
)

EXAMPLE_ID = "projected-route-single-stage-boozer-vacuum-optimization"

# The budget the certified route is configured with.  It is smaller than the
# native BFGS reference's because the coupled formulation reaches the reference's
# endpoint objective in roughly four hundred iterations, not a thousand.
PROJECTED_NATIVE_ITERATIONS = 700

ROUTE_OPTIONS = ProjectedLbfgsOptions(
    maximum_iterations=PROJECTED_NATIVE_ITERATIONS,
    feasibility_tolerance=1.0e-10,
    objective_target=4.4822246533126125e-08,
    lagrangian_newton=True,
    frozen_projector_line_search=True,
    newton_tangent_fraction_threshold=0.25,
    projector_refresh_period=4,
)

# Terminal states that are not a defect.  Everything else -- a nonfinite state,
# a non-descent direction, an infeasible start -- means the route did not run.
_SOUND_TERMINAL_STATES = (
    ProjectedLbfgsStatus.ITERATION_LIMIT,
    ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED,
    ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE,
)


def selected_kernel_names(joint_value_constraints, options: ProjectedLbfgsOptions):
    """Name the executables this configuration selects, compiling none of them.

    ``build_projected_lbfgs_kernels`` is the same construction the loop drives,
    so what this reports is what a run compiles rather than a re-spelled list
    that can drift away from it.
    """

    kernels = build_projected_lbfgs_kernels(joint_value_constraints, options=options)
    return tuple(
        name
        for name in kernels._fields
        if name != "frozen_projector_line_search"
        and getattr(kernels, name) is not None
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    bootstrap = build_single_stage_fullspace_bootstrap()
    problem = bootstrap.problem
    scaling = fullspace_scaling_from_bootstrap(bootstrap.z0, problem)

    def raw_joint(coordinates: jax.Array) -> tuple[jax.Array, jax.Array]:
        """The weighted objective and the RAW equality residuals at one point.

        Raw, not scaled: the optimizer's own feasibility tolerance is then the
        physical 1e-10 on the Boozer residual and the volume label rather than a
        proxy for it.
        """

        physical = fullspace_physical_coordinates(coordinates, scaling)
        evaluation = evaluate_fullspace(physical, problem)
        return (
            evaluation.weighted_total,
            flatten_fullspace_constraints(evaluation.constraints),
        )

    options = replace(ROUTE_OPTIONS, maximum_iterations=max_steps)
    start = fullspace_optimizer_coordinates(bootstrap.z0, scaling)
    kernels = selected_kernel_names(raw_joint, options)

    initial_evaluation = evaluate_fullspace(
        fullspace_physical_coordinates(start, scaling), problem
    )
    initial_objective = host_float(initial_evaluation.weighted_total)

    run = run_projected_lbfgs(raw_joint, start, options=options)

    terminal_physical = fullspace_physical_coordinates(run.coordinates, scaling)
    terminal = evaluate_fullspace(terminal_physical, problem)
    state = host_array(terminal_physical, dtype=np.float64)
    objectives = [record.objective for record in run.iterations]
    worst_feasibility = max(
        (record.feasibility_inf for record in run.iterations),
        default=run.feasibility_inf,
    )
    status = ProjectedLbfgsStatus(int(run.status))
    monotone = all(
        later <= earlier
        for earlier, later in zip(objectives, objectives[1:], strict=False)
    )
    feasible = worst_feasibility <= options.feasibility_tolerance
    sound = status in _SOUND_TERMINAL_STATES and feasible and monotone

    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "scale": scale,
            "route": "projected-lagrangian-newton-cg",
            "selected_kernels": kernels,
            "joint_dof_count": int(start.shape[0]),
            "equality_count": int(problem.exact_mask_indices.shape[0] + 1),
            "maximum_iterations": options.maximum_iterations,
            "initial_objective": initial_objective,
            "final_objective": run.objective,
            "objective_target": options.objective_target,
            "iterations_run": len(run.iterations),
            "terminal_status": status.name,
            "monotone_descent": monotone,
            "feasibility_inf": run.feasibility_inf,
            "maximum_feasibility_inf": float(worst_feasibility),
            "feasibility_tolerance": options.feasibility_tolerance,
            "projected_gradient_inf": run.projected_gradient_inf,
            "projector_materializations": run.projector_materializations,
            "engine_compile_seconds": run.compile_seconds,
            "engine_solve_seconds": run.solve_seconds,
            "iota": host_float(terminal.observables.iota),
            "G": host_float(terminal.observables.G),
            "volume": host_float(terminal.observables.volume),
            "major_radius": host_float(terminal.observables.major_radius),
            "total_length": host_float(terminal.observables.total_length),
            "non_qs_ratio": host_float(terminal.observables.non_qs_ratio),
            "boozer_residual": host_float(terminal.observables.boozer_residual_scalar),
            "boozer_residual_rms": host_float(terminal.observables.boozer_residual_rms),
            "solution": tuple(float(value) for value in state),
        },
        status="ok" if sound else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-projected-single-stage-",
        bounded_steps=2,
        native_default_steps=PROJECTED_NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
