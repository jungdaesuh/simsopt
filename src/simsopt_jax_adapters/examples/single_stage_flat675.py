"""Flat coupled single-stage optimization on the 11+3+661 layout.

``examples/jax/3_Advanced/single_stage_flat675.py`` is the published entry
point and owns the lesson prose; this module owns the two shipped problem
configurations, the fused solve that publishes its own transfer ledger, and
the CLI that runs them.
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.backend.dtypes import explicit_device_array
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.examples.single_stage_flat675 import (
    FLAT675_LBFGS_HISTORY,
    FLAT675_LBFGS_MAXLS,
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.runtime.host_boundary import (
    disallow_host_transfers,
    host_array,
    host_transfer_audit,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.examples import REPOSITORY_ROOT
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo import CurveCWSFourier
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    Flat675AcceptanceLimits,
    Flat675ContractError,
    Flat675Problem,
    bind_flat675_programs,
    build_flat675_problem,
    load_flat675_bundle,
    polish_flat675,
)

EXAMPLE_ID = "flat675-single-stage-coupled-optimization"

TEST_DATA = REPOSITORY_ROOT / "tests" / "test_files"
BOUNDARY_INPUT = TEST_DATA / "input.LandremanPaul2021_QA_lowres"

# The frozen campaign input, which only exists on the campaign host.  ``--bundle``
# selects it; nothing else in this module reads it.
BUNDLE_ROOT = (
    Path.home() / "simsopt_mixed_artifacts" / "genuine675-r3-input-1c23f6c5-20260721-r1"
)
BUNDLE_FLAG = "--bundle"

# Iteration budgets.  Both are small: this example demonstrates the fused lane's
# execution shape; the GPU-vs-native numbers the entry-point script discloses
# are not a claim of this default run.
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
# repository-geometry problem starts from the same kind of coil the certified
# configuration used rather than from an arbitrary one.
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


def repository_problem(scale: ExecutionScale) -> Flat675Problem:
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


def bundle_problem() -> Flat675Problem:
    """Read the certified frozen-bundle configuration, or refuse by name."""
    if not BUNDLE_ROOT.is_dir():
        raise Flat675ContractError(
            f"{BUNDLE_FLAG} runs the certified frozen-bundle configuration, "
            f"whose input bundle is host-local and was not found at "
            f"{BUNDLE_ROOT}. Run without {BUNDLE_FLAG} to use the repository "
            "geometry this example ships with."
        )
    return load_flat675_bundle(BUNDLE_ROOT)


def solve_problem(
    problem: Flat675Problem,
    *,
    max_steps: int,
    scale: ExecutionScale,
    configuration: str,
    polish: bool = False,
    acceptance_limits: Flat675AcceptanceLimits | None = None,
) -> ExampleResult:
    """Run the production fused lane once and publish what it did."""
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    start = explicit_device_array(
        problem.start_candidate.outer_vector(), dtype=np.float64
    )
    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=start,
        objective_scale=explicit_device_array(SOLVE_OBJECTIVE_SCALE, dtype=np.float64),
    )

    # The lesson is inside this block: the solve crosses no host boundary, so a
    # strict guard is the right place to run it, and the audit is what turns
    # "no host boundary" from a claim into an observable.  Both contexts come
    # from the boundary SSOT; the guard is ``jax.transfer_guard("disallow")``.
    with host_transfer_audit() as audit, disallow_host_transfers():
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
    correction = (
        polish_flat675(problem, solution, limits=acceptance_limits)
        if polish and finite
        else None
    )

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
            "flat_solver_success": bool(result.success),
            "polish": correction.as_dict()
            if correction is not None
            else {
                "acceptance_status": "rejected" if polish else "not_assessed",
                "reason": "nonfinite_flat_endpoint"
                if polish
                else "polish_not_requested",
            },
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
            and (correction is None or correction.acceptance_status != "rejected")
        )
        else "failed",
    )


def solve(
    _output_dir: Path,
    max_steps: int,
    scale: ExecutionScale,
    *,
    polish: bool = False,
    acceptance_limits: Flat675AcceptanceLimits | None = None,
) -> ExampleResult:
    """Repository-geometry entry point used when ``--bundle`` is absent."""
    return solve_problem(
        repository_problem(scale),
        max_steps=max_steps,
        scale=scale,
        configuration="repository-geometry",
        polish=polish,
        acceptance_limits=acceptance_limits,
    )


def solve_bundle(
    _output_dir: Path,
    max_steps: int,
    scale: ExecutionScale,
    *,
    polish: bool = False,
    acceptance_limits: Flat675AcceptanceLimits | None = None,
) -> ExampleResult:
    """Certified frozen-bundle entry point used when ``--bundle`` is given."""
    return solve_problem(
        bundle_problem(),
        max_steps=max_steps,
        scale=scale,
        configuration="certified-frozen-bundle",
        polish=polish,
        acceptance_limits=acceptance_limits,
    )


def main(arguments: list[str] | None = None, *, description: str | None = None) -> int:
    """Run the example; ``description`` is the entry-point script's own prose."""

    argv = list(sys.argv[1:] if arguments is None else arguments)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        BUNDLE_FLAG, action="store_true", help="use the frozen local bundle"
    )
    parser.add_argument(
        "--polish", action="store_true", help="check and correct the final surface"
    )
    parser.add_argument(
        "--max-boozer-rms",
        type=float,
        help="cap on per-component |B|-weighted Boozer RMS",
    )
    parser.add_argument(
        "--max-label-error", type=float, help="absolute volume-label error cap (m^3)"
    )
    parser.add_argument(
        "--max-surface-movement", type=float, help="maximum correction displacement (m)"
    )
    parser.add_argument(
        "--max-objective-increase",
        type=float,
        help="absolute increase cap in the original weighted objective",
    )
    options, remaining = parser.parse_known_args(argv)
    values = (
        options.max_boozer_rms,
        options.max_label_error,
        options.max_surface_movement,
        options.max_objective_increase,
    )
    limits = None
    if any(value is not None for value in values):
        if not options.polish or any(value is None for value in values):
            parser.error("acceptance requires --polish and all four --max-* limits")
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            parser.error("acceptance limits must be finite and nonnegative")
        limits = Flat675AcceptanceLimits(
            max_boozer_weighted_rms=options.max_boozer_rms,
            max_absolute_label_error=options.max_label_error,
            max_surface_displacement_m=options.max_surface_movement,
            max_objective_increase=options.max_objective_increase,
        )
    selected = solve_bundle if options.bundle else solve
    if "--help" in remaining or "-h" in remaining:
        print("Optional final surface check:")
        print(parser.format_help())
    return run_example(
        remaining,
        description=description,
        temporary_prefix="simsopt-jax-flat675-single-stage-",
        bounded_steps=BOUNDED_STEPS,
        native_default_steps=NATIVE_DEFAULT_STEPS,
        solve=partial(selected, polish=True, acceptance_limits=limits)
        if options.polish
        else selected,
    )
