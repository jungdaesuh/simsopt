"""What rung-2 generality buys, proven end to end on small problems.

Four claims, each with its own gate:

* a **generic coil set** — plain curves with a current each, the case the
  rung-1 validator refused — constructs and solves;
* a **non-certified resolution** does the same;
* a **stellarator-asymmetric** boundary does too, and its quadrature range is
  *proven* rather than assumed: the volume label on the range the constructor
  selects must match the same boundary evaluated on a full-torus grid;
* the **new blocks carry gradient**, evaluated at anchors where the relevant
  terms are active, with exact-zero controls where a term is legitimately
  inactive.

That last one is the F2 lesson: a liveness test at an anchor where the term is
switched off proves nothing, because zero is what a disconnected block returns
too.  Every liveness assertion here is paired with a control.

Artifact-free and CPU-only.  Any seconds these take are incidental; nothing
here is a measurement.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import (
    Surface,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    create_equally_spaced_curves,
)
from simsopt_jax.core.surface_dofs import surface_gamma_tangents_from_dofs
from simsopt_jax.core.surface_integrals import surface_volume
from simsopt_jax.examples.single_stage_flat675 import (
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    Flat675Problem,
    bind_flat675_programs,
    build_flat675_problem,
    default_flat675_objective_policy,
    fit_flat675_boundary,
    surface_quadrature_range,
)

NFP = 2
GRID = 8
STEPS = 5
SMALL_MPOL = SMALL_NTOR = 4

MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.18


def _boundary(*, stellsym: bool = True, asymmetry: float = 0.0) -> SurfaceRZFourier:
    """A small boundary, optionally carrying genuine rs/zc content."""
    quadpoints = np.linspace(0.0, 1.0, 32, endpoint=False)
    surface = SurfaceRZFourier(
        nfp=NFP,
        stellsym=stellsym,
        mpol=2,
        ntor=2,
        quadpoints_phi=quadpoints,
        quadpoints_theta=quadpoints,
    )
    surface.set_rc(0, 0, MAJOR_RADIUS)
    surface.set_rc(1, 0, MINOR_RADIUS)
    surface.set_zs(1, 0, MINOR_RADIUS)
    surface.set_rc(1, 1, 0.01)
    if asymmetry:
        surface.set_rs(1, 0, asymmetry)
        surface.set_zc(1, 0, 0.75 * asymmetry)
    return surface


def _generic_field(count: int = 3, order: int = 2) -> BiotSavartJAX:
    """Plain curves with a current each — the rung-1 refusal case."""
    curves = create_equally_spaced_curves(
        count, NFP, stellsym=True, R0=MAJOR_RADIUS, R1=0.5, order=order
    )
    return BiotSavartJAX(
        coils_via_symmetries(curves, [Current(1.0e5) for _ in curves], NFP, True)
    )


def _solve(problem: Flat675Problem) -> tuple[float, float, float, float, dict]:
    """Run the fused lane once; return start/end objective and gradient norms."""
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    gradient_of = jax.grad(programs.objective_fn)
    start = jax.device_put(problem.start_candidate.outer_vector())
    start_objective = float(programs.objective_fn(start))
    start_gradient = float(
        np.max(np.abs(np.asarray(jax.device_get(gradient_of(start)))))
    )

    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=start,
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
    )
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=STEPS,
            rtol=0.0,
            atol=1.0e-12,
        )
    endpoint = np.asarray(jax.device_get(result.x), dtype=np.float64)
    end_gradient = float(
        np.max(
            np.abs(np.asarray(jax.device_get(gradient_of(jax.device_put(endpoint)))))
        )
    )
    ledger = {entry.phase: entry.calls for entry in audit.summary()}
    assert np.all(np.isfinite(endpoint))
    return start_objective, float(result.fun), start_gradient, end_gradient, ledger


# --- acceptance gate 2: a generic coil set ----------------------------------


def test_generic_coil_set_constructs_and_solves() -> None:
    """Plain curves + per-coil currents, on a small layout, end to end."""
    field = _generic_field()
    problem = build_flat675_problem(
        boundary=_boundary(),
        field=field,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nphi=GRID,
        ntheta=GRID,
    )

    layout = problem.material.layout
    assert layout.coil_dof_count == len(np.asarray(field.x))
    assert layout.coil_dof_count != 11
    assert problem.start_candidate.outer_vector().shape == (layout.outer_dof_count,)

    start, end, grad_start, grad_end, ledger = _solve(problem)
    assert end < start
    assert grad_end < grad_start
    assert ledger.get("advance", 0) == 0
    assert ledger.get("callback", 0) == 0


# --- acceptance gate 3: a non-certified resolution --------------------------


@pytest.mark.parametrize(("mpol", "ntor"), [(4, 4), (5, 2)])
def test_non_certified_resolution_constructs_and_solves(mpol: int, ntor: int) -> None:
    """The layout is the caller's request, and it solves like any other."""
    problem = build_flat675_problem(
        boundary=_boundary(),
        field=_generic_field(),
        mpol=mpol,
        ntor=ntor,
        nphi=GRID,
        ntheta=GRID,
    )

    assert (
        problem.material.layout.surface_mpol,
        problem.material.layout.surface_ntor,
    ) == (mpol, ntor)

    start, end, grad_start, grad_end, _ledger = _solve(problem)
    assert end < start
    assert grad_end < grad_start


# --- requirement 3: stellsym=False, with the range PROVEN -------------------


def _volume_of(spec) -> float:
    gamma, toroidal, poloidal = surface_gamma_tangents_from_dofs(spec, spec.dofs)
    return float(surface_volume(gamma, np.cross(toroidal, poloidal)))


def test_asymmetric_range_selection_is_proven_by_the_volume_label() -> None:
    """The selected range must give the SAME volume as a full-torus grid.

    The formulation's integrals are grid means, so a range is only correct if
    it tiles the torus.  This does not assume that of ``field period``: it
    fits the same asymmetric boundary twice — once on the range the
    constructor selects, once on an explicit full-torus grid — and compares
    the volume label the objective would use.
    """
    boundary = _boundary(stellsym=False, asymmetry=0.02)
    assert surface_quadrature_range(stellsym=False) == "field period"

    selected = fit_flat675_boundary(
        boundary,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        stellsym=False,
        nphi=32,
        ntheta=32,
    )

    # The same boundary, fitted on a full-torus grid built here rather than by
    # the constructor, so the comparison is against an independent sampling.
    quadpoints = Surface.get_quadpoints(
        nfp=NFP, range="full torus", nphi=32 * NFP, ntheta=32
    )
    reference = SurfaceXYZTensorFourier(
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nfp=NFP,
        stellsym=False,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    source = SurfaceRZFourier(
        nfp=NFP,
        stellsym=False,
        mpol=2,
        ntor=2,
        quadpoints_phi=quadpoints[0],
        quadpoints_theta=quadpoints[1],
    )
    source.x = boundary.x
    reference.least_squares_fit(source.gamma())

    from simsopt_jax.core.specs import make_surface_xyz_tensor_fourier_spec

    reference_spec = make_surface_xyz_tensor_fourier_spec(
        dofs=np.asarray(reference.get_dofs(), dtype=np.float64),
        quadpoints_phi=np.asarray(reference.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=np.asarray(reference.quadpoints_theta, dtype=np.float64),
        nfp=NFP,
        stellsym=False,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
    )

    selected_volume = _volume_of(selected)
    reference_volume = _volume_of(reference_spec)
    assert reference_volume != 0.0
    relative = abs(selected_volume - reference_volume) / abs(reference_volume)
    assert relative < 1.0e-12, (
        f"the selected range gives volume {selected_volume!r} against the "
        f"full-torus {reference_volume!r} (relative {relative:.3e}); the "
        "range does not tile the torus"
    )


def test_asymmetric_problem_constructs_and_solves() -> None:
    """stellsym=False is a layout the formulation carries end to end."""
    problem = build_flat675_problem(
        boundary=_boundary(stellsym=False, asymmetry=0.02),
        field=_generic_field(),
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        stellsym=False,
        nphi=GRID,
        ntheta=GRID,
    )

    layout = problem.material.layout
    assert layout.surface_stellsym is False
    assert layout.surface_dof_count == 3 * (2 * SMALL_MPOL + 1) * (2 * SMALL_NTOR + 1)

    start, end, grad_start, grad_end, _ledger = _solve(problem)
    assert end < start
    assert grad_end < grad_start


# --- requirement 4: gradient liveness at ACTIVE anchors ---------------------


def _gradient_blocks(problem: Flat675Problem) -> tuple[np.ndarray, np.ndarray]:
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    start = jax.device_put(problem.start_candidate.outer_vector())
    gradient = np.asarray(
        jax.device_get(jax.grad(programs.objective_fn)(start)), dtype=np.float64
    )
    layout = problem.material.layout
    return gradient[layout.coil_slice], gradient[layout.surface_slice]


def _vessel_term_and_gradient(problem: Flat675Problem) -> tuple[float, np.ndarray]:
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    start = jax.device_put(problem.start_candidate.outer_vector())
    terms = np.asarray(jax.device_get(programs.diagnostics_fn(start)), dtype=np.float64)
    gradient = np.asarray(
        jax.device_get(jax.grad(programs.objective_fn)(start)), dtype=np.float64
    )
    index = FLAT675_OBJECTIVE_TERM_KEYS.index("surface_vessel")
    return float(terms[index]), gradient[problem.material.layout.vessel_slice]


@pytest.mark.parametrize(
    ("mpol", "ntor", "stellsym", "coil_count"),
    [
        pytest.param(SMALL_MPOL, SMALL_NTOR, True, 3, id="generic-coils-small"),
        pytest.param(5, 2, True, 2, id="non-certified-resolution"),
        pytest.param(SMALL_MPOL, SMALL_NTOR, False, 3, id="asymmetric-layout"),
    ],
)
def test_new_blocks_carry_gradient(
    mpol: int, ntor: int, stellsym: bool, coil_count: int
) -> None:
    """Every coil and surface coordinate of a generalized layout is live.

    The anchor matters: these are read at the constructor's own start, where
    the residual, iota and non-QS terms are all active, so a zero here would
    mean a disconnected block rather than a switched-off term.
    """
    problem = build_flat675_problem(
        boundary=_boundary(stellsym=stellsym, asymmetry=0.0 if stellsym else 0.02),
        field=_generic_field(count=coil_count),
        mpol=mpol,
        ntor=ntor,
        stellsym=stellsym,
        nphi=GRID,
        ntheta=GRID,
    )

    coil_gradient, surface_gradient = _gradient_blocks(problem)
    layout = problem.material.layout

    assert coil_gradient.shape == (layout.coil_dof_count,)
    assert surface_gradient.shape == (layout.surface_dof_count,)
    assert np.all(np.isfinite(coil_gradient))
    assert np.all(np.isfinite(surface_gradient))
    # Live, not merely finite: a disconnected block returns exact zeros.
    assert np.count_nonzero(coil_gradient) == layout.coil_dof_count
    assert np.count_nonzero(surface_gradient) > 0


def test_vessel_block_is_exactly_zero_when_its_term_is_inactive() -> None:
    """The control the liveness assertions need to mean anything.

    The synthesized vessel is placed outside the hinge, so this term is
    switched off — and the block's gradient is exactly zero.  That is what a
    zero looks like when it is legitimate.
    """
    problem = build_flat675_problem(
        boundary=_boundary(),
        field=_generic_field(),
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nphi=GRID,
        ntheta=GRID,
    )

    term, vessel_gradient = _vessel_term_and_gradient(problem)

    assert term == 0.0
    assert np.all(vessel_gradient == 0.0)


def test_vessel_block_is_live_once_its_term_switches_on() -> None:
    """The active anchor: raise the hinge until it bites, and the block moves.

    Same problem, same start — only the threshold changes, so a nonzero here
    cannot be attributed to anything but the term becoming active.
    """
    boundary = _boundary()
    field = _generic_field()
    inactive = build_flat675_problem(
        boundary=boundary,
        field=field,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nphi=GRID,
        ntheta=GRID,
    )
    base_policy = inactive.objective_policy
    import dataclasses

    active_policy = dataclasses.replace(base_policy, surface_vessel_threshold_m=1.0)
    active = build_flat675_problem(
        boundary=boundary,
        field=field,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nphi=GRID,
        ntheta=GRID,
        objective_policy=active_policy,
        vessel=inactive.material.vessel_template,
    )

    inactive_term, inactive_gradient = _vessel_term_and_gradient(inactive)
    active_term, active_gradient = _vessel_term_and_gradient(active)

    assert inactive_term == 0.0
    assert np.all(inactive_gradient == 0.0)
    assert active_term > 0.0
    assert np.count_nonzero(active_gradient) == active.material.layout.vessel_dof_count


def test_the_default_policy_points_at_a_free_coil_of_the_generic_set() -> None:
    """A generic set has no fixed TF coils, so the first coil is the one."""
    field = _generic_field()
    problem = build_flat675_problem(
        boundary=_boundary(),
        field=field,
        mpol=SMALL_MPOL,
        ntor=SMALL_NTOR,
        nphi=GRID,
        ntheta=GRID,
    )

    assert problem.objective_policy.optimized_coil_index == 0
    assert problem.objective_policy == default_flat675_objective_policy(
        optimized_coil_index=0
    )
