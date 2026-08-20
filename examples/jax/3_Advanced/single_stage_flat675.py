"""VMEC-free single-stage optimization on the flat coupled formulation.

This is the official example for the flat coupled single-stage problem
statement: coil degrees of freedom, vessel degrees of freedom and Boozer
surface degrees of freedom are ONE state vector, and there is no nested
equilibrium solve anywhere inside an objective evaluation.

The mirror lesson ``3_Advanced/single_stage_boozer_vacuum_optimization`` nests
two problems -- an outer optimizer moves coils, and every outer evaluation
runs an inner Newton solve to put the surface back on the Boozer manifold.
The flat formulation removes that inner solve entirely.  The rotational
transform and the net poloidal current are not solved for; they are closed in
closed form by a two-column least-squares solve of the Boozer system, which is
differentiable like everything around it.  What is left is a single scalar
objective over the whole state vector.

WHY THAT MATTERS FOR EXECUTION
------------------------------
Because there is no inner solve, the entire objective is one device program,
and the whole optimization is the fused on-device L-BFGS-B lane end to end --
no host round trip per accepted step, no host-side rejection or anchor
protocol.  This script publishes its own proof of that: it solves inside
``jax.transfer_guard("disallow")`` with ``host_transfer_audit()`` open and
reports the transfer ledger as an observable.  ``host_step_transfers`` and
``host_callback_transfers`` are zero on a fused run; a stepwise fallback would
make one of them nonzero, and the endpoint read-back after the guarded region
is the positive control that the audit saw anything at all.

THE LAYOUT IS FIXED AT 11 + 3 + 661
-----------------------------------
Eleven coil owner degrees of freedom (one free current plus a
curve-on-winding-surface family), three vessel degrees of freedom, and 661
boundary degrees of freedom -- a stellarator-symmetric
``SurfaceXYZTensorFourier`` at ``mpol = ntor = 10``.  The constructor
:func:`~simsopt_jax_adapters.geo.flat675.build_flat675_problem` fits any
compatible simsopt boundary onto that layout and refuses, rather than
silently reshapes, a boundary the layout cannot represent.  This example is
built from repository test-file geometry so it runs from a clean clone.

WHERE TO RUN IT, AND WHAT IS ACTUALLY CERTIFIED
-----------------------------------------------
The sealed campaign receipt ``docs/receipts/flat675_fused_campaign.md``
measures this production lane on a GPU against a native CPU denominator and
reports 1.67x at equal budget 3, 7.70x at the headline equal budget 37, and
7.36x quality-matched, all on process wall.  That certification is scoped to
the FROZEN-BUNDLE configuration at the one archived start candidate the
campaign measured -- reachable here with ``--bundle`` when that host-local
bundle is present.  The configuration this script ships by default runs the
same production lane on repository geometry and makes NO timing claim of its
own.

Cold start is disclosed, not claimed: the first solve in a process pays the
full XLA compile of the fused program (~150 s in the receipt's own cold
measurement, N=1, reported and never claimed).  The win regime is repeated or
warm work in a process that has already compiled -- which is also why
``--smoke`` here is dominated by compilation rather than by arithmetic.

``--smoke`` runs the same production lane on a deliberately small problem for
a couple of iterations.  Its ``ok`` status means the fused lane executed and
stayed finite, never that anything converged.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.examples.single_stage_flat675 import (
    FLAT675_LBFGS_HISTORY,
    FLAT675_LBFGS_MAXLS,
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.runtime.host_boundary import host_array, host_transfer_audit
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    Flat675ContractError,
    Flat675Problem,
    bind_flat675_programs,
    build_flat675_problem,
    load_flat675_bundle,
)

EXAMPLE_ID = "flat675-single-stage-coupled-optimization"

TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"
BOUNDARY_INPUT = TEST_DATA / "input.LandremanPaul2021_QA_lowres"

# The frozen campaign input, which only exists on the campaign host.  ``--bundle``
# selects it; nothing else in this script reads it.
BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
BUNDLE_FLAG = "--bundle"

# Iteration budgets.  Both are small: this example demonstrates the fused lane's
# execution shape, and the sealed receipt -- not this script -- carries the
# speed claim.
BOUNDED_STEPS = 2
NATIVE_DEFAULT_STEPS = 20

# Repository-geometry problem size.  The surface layout is fixed at 661 DOFs by
# the formulation; only the quadrature and the coil discretization vary here.
BOUNDED_GRID = 6
NATIVE_DEFAULT_GRID = 16
BOUNDED_CURVE_QUADPOINTS = 24
NATIVE_DEFAULT_CURVE_QUADPOINTS = 64

# The winding surface sits this many plasma minor radii out, and the planar TF
# coils a little beyond it.
WINDING_SURFACE_FACTOR = 2.2
TF_COIL_RADIUS_FACTOR = 2.6
TF_BASE_COIL_COUNT = 3
TF_COIL_CURRENT_A = 1.0e5
WINDING_COIL_CURRENT_A = 1.5e5
CURVE_ORDER = 2

# The certified campaign's own winding-surface coil shape: a closed saddle loop
# in ``(phi, theta)``, transcribed from the archived bundle's base curve so the
# repository-geometry problem starts from the same kind of coil the receipt
# measured rather than from an arbitrary one.
WINDING_COIL_DOFS = (
    0.119251,
    0.012469,
    -0.017700,
    -0.068677,
    -0.014250,
    0.430936,
    0.082708,
    -0.015905,
    -0.113852,
    -0.025142,
)

SOLVE_OBJECTIVE_SCALE = 1.0


def _repository_problem(scale: ExecutionScale) -> Flat675Problem:
    """Build the flat-675 problem from repository test-file geometry."""
    native = scale == "native_default"
    grid = NATIVE_DEFAULT_GRID if native else BOUNDED_GRID
    quadpoints = NATIVE_DEFAULT_CURVE_QUADPOINTS if native else BOUNDED_CURVE_QUADPOINTS
    boundary = SurfaceRZFourier.from_vmec_input(
        str(BOUNDARY_INPUT), range="half period", nphi=grid, ntheta=grid
    )

    points = np.asarray(boundary.gamma(), dtype=np.float64).reshape((-1, 3))
    radius = np.hypot(points[:, 0], points[:, 1])
    major_radius = 0.5 * (float(radius.max()) + float(radius.min()))
    minor_radius = float(np.max(np.hypot(radius - major_radius, points[:, 2])))

    winding_surface = SurfaceRZFourier(
        nfp=boundary.nfp,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    winding_surface.set_rc(0, 0, major_radius)
    winding_surface.set_rc(1, 0, minor_radius * WINDING_SURFACE_FACTOR)
    winding_surface.set_zs(1, 0, minor_radius * WINDING_SURFACE_FACTOR)

    base_curve = CurveCWSFourier(
        quadpoints=quadpoints, order=CURVE_ORDER, surf=winding_surface
    )
    base_curve.x = np.asarray(WINDING_COIL_DOFS, dtype=np.float64)
    winding_coils = coils_via_symmetries(
        [base_curve], [Current(WINDING_COIL_CURRENT_A)], boundary.nfp, True
    )

    # The TF coils carry the background field and are fixed: the certified coil
    # layout gives them empty owner maps, so the eleven owner DOFs all belong to
    # the winding-surface family.
    tf_curves = create_equally_spaced_curves(
        TF_BASE_COIL_COUNT,
        boundary.nfp,
        stellsym=True,
        R0=major_radius,
        R1=minor_radius * TF_COIL_RADIUS_FACTOR,
        order=CURVE_ORDER,
        numquadpoints=32,
        use_jax_curve=False,
    )
    tf_currents = []
    for curve in tf_curves:
        curve.fix_all()
        current = Current(TF_COIL_CURRENT_A)
        current.fix_all()
        tf_currents.append(current)
    tf_coils = coils_via_symmetries(tf_curves, tf_currents, boundary.nfp, True)

    field = BiotSavartJAX(list(tf_coils) + list(winding_coils))
    return build_flat675_problem(boundary=boundary, field=field, nphi=grid, ntheta=grid)


def _bundle_problem() -> Flat675Problem:
    """Read the certified frozen-bundle configuration, or refuse by name."""
    if not BUNDLE_ROOT.is_dir():
        raise Flat675ContractError(
            f"{BUNDLE_FLAG} runs the certified frozen-bundle configuration, "
            f"whose input bundle is host-local and was not found at "
            f"{BUNDLE_ROOT}. Run without {BUNDLE_FLAG} to use the repository "
            "geometry this example ships with."
        )
    return load_flat675_bundle(BUNDLE_ROOT)


def _solve(
    problem: Flat675Problem,
    *,
    max_steps: int,
    scale: ExecutionScale,
    configuration: str,
) -> ExampleResult:
    """Run the production fused lane once and publish what it did."""
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    start = jax.device_put(problem.start_candidate.outer_vector())
    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=start,
        objective_scale=jax.device_put(
            np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)
        ),
    )

    # The lesson is inside this block: the solve crosses no host boundary, so a
    # strict guard is the right place to run it, and the audit is what turns
    # "no host boundary" from a claim into an observable.
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=max_steps,
            rtol=1.0e-15,
            atol=1.0e-12,
        )
    ledger = {entry.phase: entry.calls for entry in audit.summary()}

    solution = host_array(prepared.problem.x, dtype=np.float64)
    terms = host_array(prepared.diagnostics(start), dtype=np.float64)
    objective = float(result.fun)
    finite = bool(np.all(np.isfinite(solution)) and np.isfinite(objective))

    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "scale": scale,
            "configuration": configuration,
            "formulation": "flat-coupled-single-stage",
            "outer_dof_count": int(solution.shape[0]),
            "lbfgs_history": FLAT675_LBFGS_HISTORY,
            "lbfgs_max_line_search_steps": FLAT675_LBFGS_MAXLS,
            "max_steps": max_steps,
            "iterations_run": int(result.nit),
            "objective_evaluations": int(result.nfev),
            "final_objective": objective,
            "endpoint_finite": finite,
            "host_step_transfers": int(ledger.get("advance", 0)),
            "host_callback_transfers": int(ledger.get("callback", 0)),
            "host_unclassified_transfers": int(ledger.get("unclassified", 0)),
            "host_endpoint_transfers": int(ledger.get("final_result", 0)),
            "initial_weighted_terms": {
                name: float(value)
                for name, value in zip(FLAT675_OBJECTIVE_TERM_KEYS, terms)
            },
        },
        status="ok"
        if (
            finite
            and solution.shape[0] == FLAT675_OUTER_DOF_COUNT
            and ledger.get("advance", 0) == 0
            and ledger.get("callback", 0) == 0
            and ledger.get("unclassified", 0) == 0
            and ledger.get("final_result", 0) > 0
        )
        else "failed",
    )


def solve(_output_dir: Path, max_steps: int, scale: ExecutionScale) -> ExampleResult:
    """Repository-geometry entry point used when ``--bundle`` is absent."""
    return _solve(
        _repository_problem(scale),
        max_steps=max_steps,
        scale=scale,
        configuration="repository-geometry",
    )


def solve_bundle(
    _output_dir: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    """Certified frozen-bundle entry point used when ``--bundle`` is given."""
    return _solve(
        _bundle_problem(),
        max_steps=max_steps,
        scale=scale,
        configuration="certified-frozen-bundle",
    )


def main(arguments: list[str] | None = None) -> int:
    import sys

    argv = list(sys.argv[1:] if arguments is None else arguments)
    selected = solve_bundle if BUNDLE_FLAG in argv else solve
    return run_example(
        [argument for argument in argv if argument != BUNDLE_FLAG],
        description=__doc__,
        temporary_prefix="simsopt-jax-flat675-single-stage-",
        bounded_steps=BOUNDED_STEPS,
        native_default_steps=NATIVE_DEFAULT_STEPS,
        solve=selected,
    )


if __name__ == "__main__":
    raise SystemExit(main())
