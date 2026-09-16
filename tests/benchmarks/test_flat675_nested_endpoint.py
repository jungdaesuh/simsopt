"""The nested-LS endpoint correction, on a problem small enough to test.

The flat-675 layout itself is 661 surface DOFs and a frozen bundle; nothing
here touches it.  These tests build their own NCSX 7x7 / mpol=ntor=2 Boozer
problem with real simsopt coils, converge it once with the nested route's own
native solver, and then perturb the converged surface by a known amount.  That
is the state the measurement has to be right about: a point that is *near* the
Boozer manifold but not on it, which is exactly the claim a flat endpoint
makes about itself.

One tiny real ``Flat675Problem`` is built as well, at the same reduced
resolution, so the cross-reference term is checked against the production
bridge rather than against a hand-assembled view.
"""

from __future__ import annotations

import dataclasses
import inspect

import numpy as np
import pytest
import benchmarks.flat675_nested_endpoint as endpoint_module
from benchmarks.flat675_nested_endpoint import (
    EXIT_STATUS_FROM_FULL_DECISION_GRADIENT,
    EXIT_STATUS_FROM_REDUCED_GRADIENT,
    NESTED_CORRECTION_PHYSICS_TOLERANCE,
    NESTED_CORRECTION_TOLERANCE,
    NESTED_LS_JAX_INNER_LINEAR_SOLVER,
    NativeConstraintWeightMismatch,
    NestedCorrection,
    correct_with_nested_ls_jax,
    correct_with_nested_ls_native,
    flat675_boozer_term,
    jax_penalty_evaluation,
    native_boozer_at,
    native_penalty_evaluation,
    nested_ls_residual_norm,
    stays_on_incoming_branch,
)
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, Volume
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.flat675 import (
    Flat675Problem,
    build_flat675_problem,
)
from simsopt_jax_adapters.geo.flat675.layout import (
    FlatSingleStageLayout,
    surface_block_dof_count,
)
from simsopt_jax_adapters.geo.flat675.nested_bridge import (
    NESTED_BRIDGE_NEWTON_OPTIONS,
    NestedJaxInputs,
    NestedView,
    nested_view_from_flat675,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_JAX_INNER_POLICY_NAME,
    NESTED_LS_JAX_INNER_STAB,
    NESTED_LS_NATIVE_INNER_POLICY_NAME,
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_EXIT_STATUSES,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_WEIGHT_INV_MODB,
)
from simsopt_jax_adapters.geo.nested_ls_ncsx import clone_surface_xyz_tensor_fourier
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_reduced_closures,
    run_reduced_nested_ls_schur_newton,
)
from simsopt_jax_adapters.geo.surface_specs import surface_spec_from_surface

_MPOL = 2
_NTOR = 2
_GRID = 7
_IOTA_SEED = -0.406
_VESSEL_DOF_COUNT = 3

# Large enough that the incoming residual is ~10 orders above the nested
# tolerance (1.5e10x measured on the seed; the test below asserts 1e9x), small enough that the corrected surface is the same Boozer branch.
_PERTURBATION_M = 1.0e-4
_PERTURBATION_SEED = 20260915

# The view below is built by hand rather than through
# ``nested_view_from_flat675``, so it takes the bridge's OWN exported option
# set.  A restated copy is how this fixture came to carry the physics-bar
# tolerances while the bridge had already moved to the banana (timing) ones,
# i.e. how the test stopped exercising the policy production ships.


def _small_surface(nfp: int) -> SurfaceXYZTensorFourier:
    return SurfaceXYZTensorFourier(
        mpol=_MPOL,
        ntor=_NTOR,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, _GRID, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, _GRID, endpoint=False),
    )


def _g_from_currents(base_currents, nfp: int, *, stellsym: bool) -> float:
    # coils_via_symmetries replicates every base coil nfp * (1 + stellsym) times.
    replicas = nfp * (2 if stellsym else 1)
    current_sum = replicas * sum(abs(current.get_value()) for current in base_currents)
    return 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))


@pytest.fixture(scope="module")
def perturbed_view() -> NestedView:
    """A converged NCSX Boozer surface, moved off the manifold by a known step.

    The seed uses the same banana ``run_code`` policy the native lane is
    judged by, so "converged" means converged in the sense the measurement
    uses, not in some other solver's sense.
    """
    base_curves, base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    del base_curves
    surface = _small_surface(nfp)
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    label_target = float(Volume(surface).J())
    seed_boozer = native_boozer_at(
        biotsavart,
        surface,
        label_target=label_target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
    )
    seed = seed_boozer.run_code(
        _IOTA_SEED, G=_g_from_currents(base_currents, nfp, stellsym=True)
    )
    assert bool(seed["success"]), "the NCSX 7x7 seed solve did not converge"

    generator = np.random.default_rng(_PERTURBATION_SEED)
    converged_dofs = np.array(surface.get_dofs(), dtype=np.float64, copy=True)
    surface_dofs = converged_dofs + _PERTURBATION_M * generator.standard_normal(
        converged_dofs.size
    )
    surface_native = clone_surface_xyz_tensor_fourier(surface)
    surface_native.set_dofs(surface_dofs)
    coil_dofs = np.array(biotsavart.x, dtype=np.float64, copy=True)
    return NestedView(
        coil_dofs=coil_dofs,
        vessel_dofs=np.zeros((_VESSEL_DOF_COUNT,), dtype=np.float64),
        surface_dofs=surface_dofs,
        iota=float(seed["iota"]),
        G=float(seed["G"]),
        coils_native=tuple(biotsavart.coils),
        biotsavart_native=biotsavart,
        surface_native=surface_native,
        label_native=Volume(surface_native),
        jax_inputs=NestedJaxInputs(
            biotsavart_jax=BiotSavartJAX(biotsavart.coils),
            surface_template=surface_spec_from_surface(surface_native),
            surface_dofs=surface_dofs,
            label_target=label_target,
            constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
            weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
            newton_options=NESTED_BRIDGE_NEWTON_OPTIONS,
        ),
        layout=FlatSingleStageLayout(
            coil_dof_count=int(coil_dofs.size),
            surface_mpol=_MPOL,
            surface_ntor=_NTOR,
            surface_stellsym=True,
        ),
        vector=np.concatenate(
            (coil_dofs, np.zeros((_VESSEL_DOF_COUNT,), dtype=np.float64), surface_dofs)
        ),
    )


@pytest.fixture(scope="module")
def jax_correction(perturbed_view: NestedView) -> NestedCorrection:
    return correct_with_nested_ls_jax(perturbed_view)


@pytest.fixture(scope="module")
def native_correction(perturbed_view: NestedView) -> NestedCorrection:
    return correct_with_nested_ls_native(perturbed_view)


@pytest.mark.boozer
def test_view_layout_agrees_with_the_constructed_vector(
    perturbed_view: NestedView,
) -> None:
    """The hand-built view is self-consistent, so nothing below reads a lie."""
    layout = perturbed_view.layout
    assert layout.surface_dof_count == perturbed_view.surface_dofs.size
    assert layout.surface_dof_count == surface_block_dof_count(
        mpol=_MPOL, ntor=_NTOR, stellsym=True
    )
    assert perturbed_view.vector.size == layout.outer_dof_count
    np.testing.assert_array_equal(
        perturbed_view.surface_native.get_dofs(), perturbed_view.surface_dofs
    )


@pytest.mark.boozer
def test_both_lanes_measure_the_same_incoming_residual(
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> None:
    """One residual definition, two kernels, on the same view.

    This is the claim that licenses every other comparison in this file: if
    the lanes disagreed here they would be answering different questions.
    """
    np.testing.assert_allclose(
        jax_correction.residual_norm_before,
        native_correction.residual_norm_before,
        rtol=1.0e-12,
        atol=0.0,
    )


@pytest.mark.boozer
def test_the_incoming_point_is_off_the_boozer_manifold(
    native_correction: NestedCorrection,
) -> None:
    """The perturbation really does leave the manifold, by many decades."""
    assert native_correction.residual_norm_before > 1.0e9 * (
        NESTED_CORRECTION_TOLERANCE
    )


@pytest.mark.boozer
@pytest.mark.parametrize("lane", ["jax", "native"])
def test_each_lane_corrects_to_the_nested_tolerance(
    lane: str,
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> None:
    correction = jax_correction if lane == "jax" else native_correction
    assert correction.lane == lane
    assert correction.tolerance == NESTED_CORRECTION_TOLERANCE
    assert correction.residual_norm_after <= NESTED_CORRECTION_TOLERANCE
    assert correction.residual_norm_after < correction.residual_norm_before
    assert correction.converged is True
    assert correction.exit_status == NESTED_LS_NEWTON_EXIT_CONVERGED
    assert correction.persisted is True
    # Some inner stage did real work; which one is the provenance test below.
    assert int(correction.newton_iterations) + int(correction.bfgs_iterations or 0) >= 1


@pytest.mark.boozer
def test_each_lane_names_its_own_inner_policy_and_per_stage_counts(
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> None:
    """Two materially different inner solvers, two records that say so.

    The contract forbids a lane-agnostic inner record ("a shared record that
    omits that distinction is false provenance").  The JAX lane is a pure
    Newton with no BFGS stage, so ``bfgs_iterations`` is ``None`` -- the
    statement "no such stage", which is not the statement "zero steps".  The
    native lane is the banana BFGS-then-Newton sequence, and both of its
    stages are counted because its ``wall_s`` pays for both.
    """
    assert jax_correction.inner_policy == NESTED_LS_JAX_INNER_POLICY_NAME
    assert jax_correction.bfgs_iterations is None
    assert jax_correction.newton_iterations >= 1
    assert jax_correction.reduced_gradient_l2 is not None
    assert jax_correction.reduced_gradient_l2 <= NESTED_CORRECTION_TOLERANCE

    assert native_correction.inner_policy == NESTED_LS_NATIVE_INNER_POLICY_NAME
    assert native_correction.bfgs_iterations is not None
    # The BFGS pre-stage is real work that a single "iterations" number would
    # have attributed to the Newton stage or lost entirely.
    assert native_correction.bfgs_iterations >= 1
    assert native_correction.newton_iterations >= 1
    # The C++ LS lane has no projected-y reduced gradient to publish.
    assert native_correction.reduced_gradient_l2 is None

    for correction in (jax_correction, native_correction):
        assert correction.exit_status in NESTED_LS_NEWTON_EXIT_STATUSES
        # Coils are frozen on both lanes: the correction is surface-only.
        assert correction.coil_delta_inf == 0.0

    # The two exit statuses are classified from DIFFERENT norms, and the
    # record has to say which: the JAX Newton drives the reduced gradient at
    # the projected ``y*``, the C++ LS lane has no such quantity and is judged
    # on the full-decision gradient.  One undifferentiated column would invite
    # a reader to compare two convergence tests as if they were one.
    assert jax_correction.exit_status_quantity == EXIT_STATUS_FROM_REDUCED_GRADIENT
    assert (
        native_correction.exit_status_quantity
        == EXIT_STATUS_FROM_FULL_DECISION_GRADIENT
    )
    assert jax_correction.exit_status_quantity != native_correction.exit_status_quantity
    # The quantity named on each lane is the one that lane can actually
    # publish: exactly the lane with a reduced gradient is the lane whose
    # status was classified from one.
    assert (jax_correction.reduced_gradient_l2 is not None) is (
        jax_correction.exit_status_quantity == EXIT_STATUS_FROM_REDUCED_GRADIENT
    )
    assert (native_correction.reduced_gradient_l2 is not None) is (
        native_correction.exit_status_quantity == EXIT_STATUS_FROM_REDUCED_GRADIENT
    )


@pytest.mark.boozer
def test_the_native_lane_records_the_thread_count_it_actually_ran_at(
    native_correction: NestedCorrection,
    jax_correction: NestedCorrection,
) -> None:
    """H8: the lane's own OMP pin, read in the process that ran the solve.

    The native lane is pinned to one thread for determinism, which is NOT the
    thread count the certified native banana lane runs at
    (``NESTED_LS_GATE6_NATIVE_OMP_THREADS``), so the record has to name it or
    its walls get read as that lane's.
    """
    assert native_correction.omp_num_threads == "1"
    # The JAX lane's number is deliberately NOT compared against
    # ``os.environ.get("OMP_NUM_THREADS")``: that is the same expression the
    # lane itself evaluates, so the comparison would pass whatever the lane
    # recorded.  Only the contract of the field is pinned here -- the variable
    # as a string, or ``None`` when the parent did not set it -- because the
    # value is provenance about the host, not about the solve: the JAX lane is
    # not thread-dependent the way the C++ kernel's warm start is.
    assert jax_correction.omp_num_threads is None or (
        jax_correction.omp_num_threads.isdigit()
    )


@pytest.mark.boozer
def test_both_lanes_stay_on_the_incoming_boozer_branch(
    perturbed_view: NestedView,
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> None:
    """The branch guard is recorded, and it passes for this perturbation."""
    for correction in (jax_correction, native_correction):
        assert correction.iota_before == perturbed_view.iota
        assert correction.iota_branch_guard == NESTED_LS_OUTER_IOTA_BRANCH_GUARD
        assert abs(correction.iota_after - correction.iota_before) <= (
            NESTED_LS_OUTER_IOTA_BRANCH_GUARD
        )
        assert correction.same_branch_as_incoming is True


@pytest.mark.boozer
def test_the_native_lane_refuses_a_view_with_a_foreign_constraint_weight(
    perturbed_view: NestedView,
) -> None:
    """The native lane's two halves must be built at the SAME weight.

    ``_run_native_banana_bfgs_then_newton`` passes
    ``NESTED_LS_CONSTRAINT_WEIGHT`` to the kernel itself, while
    ``native_boozer_at`` builds the ``BoozerSurface`` from the view's own
    ``constraint_weight``.  They agree today only because the bridge always
    uses the constant, so a view that carries anything else must stop the lane
    rather than minimise one penalty and report the residual of another.  The
    check fires before the child is spawned, so this costs no solve.
    """
    foreign = dataclasses.replace(
        perturbed_view,
        jax_inputs=dataclasses.replace(
            perturbed_view.jax_inputs,
            constraint_weight=2.0 * float(NESTED_LS_CONSTRAINT_WEIGHT),
        ),
    )
    with pytest.raises(NativeConstraintWeightMismatch):
        correct_with_nested_ls_native(foreign)


def test_the_native_lane_refuses_a_view_with_a_foreign_inverse_modb_weighting(
    perturbed_view: NestedView,
) -> None:
    """The same fail-closed rule for the other hardwired knob, weight_inv_modB.

    ``_run_native_banana_bfgs_then_newton`` passes
    ``NESTED_LS_CONSTRAINT_WEIGHT`` to the kernel itself, while
    ``native_boozer_at`` builds the ``BoozerSurface`` from the view's own
    ``constraint_weight``.  They agree today only because the bridge always
    uses the constant, so a view that carries anything else must stop the lane
    rather than minimise one penalty and report the residual of another.  The
    check fires before the child is spawned, so this costs no solve.
    """
    foreign = dataclasses.replace(
        perturbed_view,
        jax_inputs=dataclasses.replace(
            perturbed_view.jax_inputs,
            weight_inv_modB=not NESTED_LS_WEIGHT_INV_MODB,
        ),
    )
    with pytest.raises(NativeConstraintWeightMismatch):
        correct_with_nested_ls_native(foreign)


@pytest.mark.boozer
def test_the_branch_guard_predicate_rejects_a_branch_change() -> None:
    """Both sides of the guard, on the predicate both lanes share.

    ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` was set by a measured branch jump
    (``iota 0.1409 -> -0.0024``), so that jump is the case driven here.
    """
    assert stays_on_incoming_branch(iota_before=0.1409, iota_after=0.1409) is True
    assert (
        stays_on_incoming_branch(
            iota_before=0.1409,
            iota_after=0.1409 - NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
        )
        is True
    )
    assert stays_on_incoming_branch(iota_before=0.1409, iota_after=-0.0024) is False
    assert (
        stays_on_incoming_branch(
            iota_before=0.1409,
            iota_after=0.1409 + 1.0000001 * NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
        )
        is False
    )


@pytest.mark.boozer
def test_the_two_bars_are_both_available_and_ordered(
    native_correction: NestedCorrection,
) -> None:
    """The timing bar is what the lanes stop at; the physics bar is tighter.

    Publishing one ratio without naming its bar is how "1.6e9x the nested
    tolerance" becomes ambiguous by two orders of magnitude.

    The two bars are pinned to the contract constants they claim to be, and
    the measured point is placed relative to BOTH.  Asserting instead that
    ``r / physics > r / timing`` would assert nothing about this measurement:
    it follows from ``physics < timing`` for every positive ``r``.
    """
    assert NESTED_CORRECTION_TOLERANCE == NESTED_LS_BANANA_NEWTON_TOL
    assert NESTED_CORRECTION_PHYSICS_TOLERANCE == NESTED_LS_NEWTON_TOL
    assert NESTED_CORRECTION_PHYSICS_TOLERANCE < NESTED_CORRECTION_TOLERANCE
    assert native_correction.tolerance == NESTED_CORRECTION_TOLERANCE
    # The measurement: this point comes in over both bars and leaves under the
    # timing one, which is the bar the lane stopped at.
    assert native_correction.residual_norm_before > NESTED_CORRECTION_TOLERANCE
    assert native_correction.residual_norm_before > NESTED_CORRECTION_PHYSICS_TOLERANCE
    assert native_correction.residual_norm_after <= NESTED_CORRECTION_TOLERANCE


@pytest.mark.boozer
def test_the_two_lanes_land_on_the_same_surface(
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> None:
    """Different solvers, same Boozer branch: the correction is well posed."""
    np.testing.assert_allclose(
        jax_correction.surface_dofs_after,
        native_correction.surface_dofs_after,
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax_correction.iota_after, native_correction.iota_after, rtol=1.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        jax_correction.G_after, native_correction.G_after, rtol=1.0e-12, atol=0.0
    )


@pytest.mark.boozer
def test_displacements_are_metres_on_the_surface_quadrature(
    perturbed_view: NestedView,
    jax_correction: NestedCorrection,
) -> None:
    """The reported displacement is the geometry's, recomputed independently.

    The minor radius is the *incoming* surface's own, so a reader dividing by
    it gets a displacement relative to the surface under judgement.
    """
    surface = clone_surface_xyz_tensor_fourier(perturbed_view.surface_native)
    surface.set_dofs(perturbed_view.surface_dofs)
    gamma_before = np.array(surface.gamma(), dtype=np.float64, copy=True)
    minor_radius = float(surface.minor_radius())
    surface.set_dofs(jax_correction.surface_dofs_after)
    gamma_after = np.array(surface.gamma(), dtype=np.float64, copy=True)
    point_delta = np.linalg.norm(gamma_after - gamma_before, axis=-1).reshape(-1)

    assert jax_correction.minor_radius_m == minor_radius
    np.testing.assert_allclose(
        jax_correction.point_displacement_max_m,
        float(np.max(point_delta)),
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_correction.point_displacement_rms_m,
        float(np.sqrt(np.mean(point_delta * point_delta))),
        rtol=1.0e-15,
        atol=0.0,
    )
    dof_delta = jax_correction.surface_dofs_after - perturbed_view.surface_dofs
    np.testing.assert_allclose(
        jax_correction.dof_displacement_l2,
        float(np.linalg.norm(dof_delta)),
        rtol=1.0e-15,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_correction.dof_displacement_max,
        float(np.max(np.abs(dof_delta))),
        rtol=1.0e-15,
        atol=0.0,
    )
    # The correction is a real move in metres, not a no-op the metrics round
    # to zero, and it is bounded by the perturbation's own scale.
    assert 0.0 < jax_correction.point_displacement_max_m < 1.0e-2 * minor_radius


@pytest.mark.boozer
def test_the_jax_residual_definition_is_the_native_one(
    perturbed_view: NestedView,
) -> None:
    """Both penalty evaluators agree on value and gradient at one state.

    ``residual_norm_before`` is a norm, so two different gradients could in
    principle produce the same number; this pins the vectors themselves.
    """
    jax_boozer = perturbed_view.jax_inputs.new_boozer_surface_jax()
    _residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(
        jax_boozer,
        constraint_weight=perturbed_view.jax_inputs.constraint_weight,
        weight_inv_modB=perturbed_view.jax_inputs.weight_inv_modB,
    )
    surface = clone_surface_xyz_tensor_fourier(perturbed_view.surface_native)
    native = native_boozer_at(
        perturbed_view.biotsavart_native,
        surface,
        label_target=perturbed_view.jax_inputs.label_target,
        constraint_weight=perturbed_view.jax_inputs.constraint_weight,
    )
    arguments = {
        "surface_dofs": perturbed_view.surface_dofs,
        "iota": perturbed_view.iota,
        "G": perturbed_view.G,
    }
    jax_evaluation = jax_penalty_evaluation(objective_fn, **arguments)
    native_evaluation = native_penalty_evaluation(native, **arguments)
    np.testing.assert_allclose(
        jax_evaluation.objective,
        native_evaluation.objective,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_evaluation.gradient,
        native_evaluation.gradient,
        rtol=1.0e-9,
        atol=1.0e-12 * nested_ls_residual_norm(native_evaluation),
    )


@pytest.mark.boozer
def test_native_lane_ignores_the_parent_thread_count(
    perturbed_view: NestedView,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one-thread child makes the parent's OpenMP team invisible.

    On this kernel a native Boozer solve from a non-converged warm start is
    thread-dependent, so a lane that inherited the parent's team would not be
    reproducible from a different harness.
    """
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    from_one = correct_with_nested_ls_native(perturbed_view)
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    from_eight = correct_with_nested_ls_native(perturbed_view)

    np.testing.assert_array_equal(
        from_one.surface_dofs_after, from_eight.surface_dofs_after
    )
    assert from_one.residual_norm_before == from_eight.residual_norm_before
    assert from_one.residual_norm_after == from_eight.residual_norm_after
    assert from_one.iota_after == from_eight.iota_after
    assert from_one.G_after == from_eight.G_after
    assert from_one.bfgs_iterations == from_eight.bfgs_iterations
    assert from_one.newton_iterations == from_eight.newton_iterations
    assert from_one.converged == from_eight.converged
    assert from_one.exit_status == from_eight.exit_status
    assert from_one.persisted == from_eight.persisted
    assert from_one.omp_num_threads == from_eight.omp_num_threads == "1"
    assert from_one.point_displacement_max_m == from_eight.point_displacement_max_m
    assert from_one.minor_radius_m == from_eight.minor_radius_m


class _SchurCallRecorded(RuntimeError):
    """Sentinel: the inner solve was reached, with the arguments recorded."""


@pytest.mark.boozer
def test_jax_lane_runs_the_memory_feasible_inner_solve(
    perturbed_view: NestedView,
) -> None:
    """The JAX lane must take the Schur path, at the certified knobs.

    This is a code-path assertion, not a numerical one, because the defect it
    guards is invisible at this size: the generic reduced Newton polish
    converges fine on 37 DOFs and only blows past 27 GiB of XLA
    rematerialization at the certified 661.  The recorder binds against the
    real signature of ``run_reduced_nested_ls_schur_newton``, so a wrong arity
    or a renamed keyword fails here rather than passing silently.
    """
    signature = inspect.signature(run_reduced_nested_ls_schur_newton)
    recorded: dict[str, object] = {}

    def recorder(jax_boozer, **kwargs):
        bound = signature.bind(jax_boozer, **kwargs)
        recorded.update(bound.arguments)
        raise _SchurCallRecorded

    original = endpoint_module.run_reduced_nested_ls_schur_newton
    endpoint_module.run_reduced_nested_ls_schur_newton = recorder
    try:
        with pytest.raises(_SchurCallRecorded):
            correct_with_nested_ls_jax(perturbed_view)
    finally:
        endpoint_module.run_reduced_nested_ls_schur_newton = original

    assert recorded["linear_solver"] == NESTED_LS_JAX_INNER_LINEAR_SOLVER
    assert recorded["max_dense_linearization_bytes"] is None
    assert recorded["tol"] == NESTED_CORRECTION_TOLERANCE
    assert recorded["maxiter"] == NESTED_LS_BANANA_NEWTON_MAXITER
    assert recorded["stab"] == NESTED_LS_JAX_INNER_STAB
    assert recorded["iota"] == perturbed_view.iota
    assert recorded["G"] == perturbed_view.G
    # Closures are handed in, so the solve differentiates the same program the
    # residual is measured with rather than rebuilding a second one.
    assert recorded["residual_fn"] is not None
    assert recorded["objective_fn"] is not None


@pytest.fixture(scope="module")
def tiny_flat675_problem() -> Flat675Problem:
    """A real ``Flat675Problem`` at the reduced resolution used here.

    Not the certified 11/3/661 layout: the point is that the flat objective's
    Boozer term and the nested residual are two functionals of the same
    penalty, which is a statement about the code path, not about 661.
    """
    _base_curves, _base_currents, _axis, nfp, biotsavart = get_data("ncsx")
    boundary = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=_MPOL,
        ntor=_NTOR,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    boundary.set_rc(0, 0, 1.5)
    boundary.set_rc(1, 0, 0.3)
    boundary.set_zs(1, 0, 0.3)
    return build_flat675_problem(
        boundary=boundary,
        field=BiotSavartJAX(biotsavart.coils),
        mpol=_MPOL,
        ntor=_NTOR,
        nphi=_GRID,
        ntheta=_GRID,
    )


@pytest.mark.boozer
def test_flat675_boozer_term_is_the_weighted_penalty_value(
    tiny_flat675_problem: Flat675Problem,
) -> None:
    """The cross-reference term is ``residual_weight * J_LS`` at the view.

    The flat objective's Boozer term and the nested residual are the value and
    the stationarity of one penalty; publishing them side by side is only
    honest if that relation holds, so it is measured rather than asserted in
    prose.  The tolerance is the bridge's own native-vs-JAX field agreement
    (1e-12) propagated through one penalty evaluation.
    """
    vector = np.asarray(
        tiny_flat675_problem.start_candidate.outer_vector(), dtype=np.float64
    )
    view = nested_view_from_flat675(tiny_flat675_problem, vector)
    surface = clone_surface_xyz_tensor_fourier(view.surface_native)
    native = native_boozer_at(
        view.biotsavart_native,
        surface,
        label_target=view.jax_inputs.label_target,
        constraint_weight=view.jax_inputs.constraint_weight,
    )
    evaluation = native_penalty_evaluation(
        native,
        surface_dofs=view.surface_dofs,
        iota=view.iota,
        G=view.G,
    )
    weight = float(tiny_flat675_problem.objective_policy.residual_weight)
    np.testing.assert_allclose(
        flat675_boozer_term(tiny_flat675_problem, vector),
        weight * evaluation.objective,
        rtol=1.0e-9,
        atol=0.0,
    )
