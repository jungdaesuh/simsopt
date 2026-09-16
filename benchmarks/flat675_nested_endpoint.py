"""How far a flat-675 point sits from the nested route's Boozer manifold.

The flat coupled single-stage solve carries the Boozer residual as one
weighted term of an eight-term objective and closes ``(iota, G)`` by a
two-column least-squares solve inside the forward pass.  The nested route
instead drives an inner Newton to a stopping tolerance.  Asking whether a
flat endpoint "is a Boozer surface" is therefore only meaningful against the
nested route's own stopping policy, applied to the nested route's own
residual -- which is what this module measures.

**One residual, two lanes.**  The nested contract's stationarity object is
``||grad J_LS||_2`` over the full decision ``[surface_dofs, iota, G]``, with
the contract's knobs (``constraint_weight=1``, free ``G``,
``weight_inv_modB``); it is what the inner Newton drives to
``NESTED_LS_BANANA_NEWTON_TOL``.  Both lanes evaluate exactly that functional
on ``view.surface_native``'s quadrature -- the native lane through the C++
``boozer_penalty_constraints_vectorized`` kernel, the JAX lane through the
packed penalty closure the reduced Newton itself differentiates.  The two
agree to a relative 1e-12 on the same view; the lanes may then walk different
trajectories, and the point of the measurement is that they land at the same
place.

**Two lanes, two inner solvers.**  They are NOT the same step and NOT the same
stopping policy, and every record says which one it is.  The JAX lane is the
reduced Schur Newton (``NESTED_LS_JAX_INNER_POLICY_NAME``): pure Newton on the
surface block, no pre-stage.  The native lane is the contract's banana
sequence (``NESTED_LS_NATIVE_INNER_POLICY_NAME``,
``NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON``): BFGS to
``NESTED_LS_BANANA_BFGS_TOL`` and then Newton to
``NESTED_LS_BANANA_NEWTON_TOL``.  ``BoozerSurface.run_code`` returns only the
Newton stage's result, so a single ``iterations`` number would report the
Newton count while ``wall_s`` paid for both stages -- the native lane's
"1 step" would be the BFGS stage's uncounted work.  :class:`NestedCorrection`
therefore carries ``bfgs_iterations`` and ``newton_iterations`` separately,
taken from ``nested_ls_reduced_scale._run_native_banana_bfgs_then_newton``,
the repository's own native banana driver, which is the only caller shape that
publishes both.

**The branch guard.**  ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` is the contract's
rule that convergence alone rejects nothing: an inner solve whose ``iota``
moves more than the guard from the incoming anchor has landed on a different
Boozer branch and is a failed evaluation, not a correction, identically in both
lanes.  Every record carries ``same_branch_as_incoming``; a consumer that reads
``converged`` without it is reading a solve of a different problem.

**Two bars.**  ``NESTED_CORRECTION_TOLERANCE`` is the contract's TIMING bar
(``NESTED_LS_TIMING_BAR``, banana ``run_code``);
``NESTED_CORRECTION_PHYSICS_TOLERANCE`` is its PHYSICS bar
(``NESTED_LS_PHYSICS_BAR``, the reconstruct/rejudge Newton), two orders
tighter.  Both lanes STOP at the timing bar, so that is the ``tolerance`` on
the record and the bar ``converged`` and ``exit_status`` name; the physics bar
is what ``benchmarks/nested_ls_outer_claim.py`` judges its certified rows
with, and any statement comparable to that gate must name it.

Note that ``residual_norm_after`` is the *full-decision* gradient norm even on
the JAX lane, whose Newton drives the *reduced* gradient ``grad phi_hat``.  The
two coincide at a projected ``y*`` up to the ``y``-block that the projected
solve leaves at round-off, so reporting the full norm never flatters the
reduced lane and keeps one number comparable across both.

**Why the JAX lane is the Schur path.**  At the certified 661-DOF layout the
generic reduced Newton polish differentiates its Hessian-vector products
through the projected-``y`` QR solve, and XLA cannot rematerialize that
program below 27.13 GiB -- it is RESOURCE_EXHAUSTED on a 32 GB device.  The
Schur walk carries the 2x2 ``Phi_yy`` block explicitly instead, so its largest
live array is the dense 661x661 operator.  See
:func:`correct_with_nested_ls_jax`.

**Why the native lane is a child process.**  The C++ Boozer kernel's OpenMP
reductions make a non-converged warm start thread-dependent, so a native
correction launched from an ``OMP_NUM_THREADS=8`` pytest process is not the
same solve as one launched from a single-threaded one.  ``OMP_NUM_THREADS`` is
read when libgomp starts, so no in-process pin can fix that.  The native lane
therefore runs in a subprocess whose environment comes from
``build_parity_lane_environment("native-cpu", ...)`` -- the repository's SSOT
for that pin -- ahead of the extension load, and the parent's thread count
becomes invisible to the result.
"""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Final, Literal

import jax
import jax.numpy as jnp
import numpy as np
import simsopt
from examples.jax.parity.runtime import build_parity_lane_environment
from numpy.typing import NDArray
from simsopt.field import BiotSavart
from simsopt.geo import BoozerSurface, SurfaceXYZTensorFourier, Volume

from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    Flat675Problem,
    flat675_weighted_terms,
)
from simsopt_jax_adapters.geo.flat675.nested_bridge import NestedView
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_JAX_INNER_POLICY_NAME,
    NESTED_LS_JAX_INNER_STAB,
    NESTED_LS_NATIVE_INNER_POLICY_NAME,
    NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED,
    NESTED_LS_NEWTON_EXIT_CONVERGED,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
    NESTED_LS_PHYSICS_BAR,
    NESTED_LS_TIMING_BAR,
    NESTED_LS_WEIGHT_INV_MODB,
    nested_ls_banana_run_code_options,
    nested_ls_newton_exit_status,
)
from simsopt_jax_adapters.geo.nested_ls_ncsx import clone_surface_xyz_tensor_fourier
from simsopt_jax_adapters.geo.nested_ls_newton_parity import (
    NestedLsPenaltyEvaluation,
    pack_nested_ls_decision,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_reduced_closures,
    run_reduced_nested_ls_schur_newton,
)

# ``_run_native_banana_bfgs_then_newton`` is the repository's own native
# banana driver and is imported deliberately, private name and all: it is the
# only entry point that publishes the BFGS stage's iteration count (see
# :func:`run_native_correction_child`).  It carries ONE coupling this module
# must not inherit silently -- it passes ``NESTED_LS_CONSTRAINT_WEIGHT`` to the
# kernel itself, while :func:`native_boozer_at` builds the ``BoozerSurface``
# from the view's own ``constraint_weight``.  The two agree only as long as the
# bridge keeps using the contract constant, so
# :func:`correct_with_nested_ls_native` fails closed when they do not.
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    _run_native_banana_bfgs_then_newton,
    nested_ls_threading_env,
)
from simsopt_jax_adapters.isolated_kernel import pythonpath_with_loaded_kernel

PRODUCTION_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: The stopping tolerance both lanes are judged against.  The nested route's
#: timing bar (``run_code``: BFGS then Newton) is the policy the flat endpoint
#: is being compared to, so its Newton tolerance is the bar, not the tighter
#: reconstruct/physics one.
NESTED_CORRECTION_TOLERANCE: Final[float] = NESTED_LS_BANANA_NEWTON_TOL

#: The name of the bar above, so a consumer never has to infer it from a float.
NESTED_CORRECTION_TOLERANCE_BAR: Final[str] = NESTED_LS_TIMING_BAR

#: The contract's PHYSICS bar (``NESTED_LS_NEWTON_TOL``, two orders tighter).
#: Neither lane stops here -- both stop at the timing bar above -- but this is
#: the bar ``benchmarks/nested_ls_outer_claim.py`` rejudges its certified rows
#: against, so any statement that claims comparability with that gate must be
#: computed against this number and say so.
NESTED_CORRECTION_PHYSICS_TOLERANCE: Final[float] = NESTED_LS_NEWTON_TOL

#: The name of the physics bar.
NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR: Final[str] = NESTED_LS_PHYSICS_BAR

#: The flat-675 objective term that carries the Boozer penalty.
FLAT675_BOOZER_TERM_KEY: Final[str] = "residual"

#: The inner linear solve for the JAX lane.  Dense LU on the Schur complement
#: is what the certified 661-DOF nested-LS track runs
#: (``nested_ls_reduced_scale._solve_nested_inner_leg``,
#: ``nested_ls_ncsx.run_ncsx_schur_inner``); the 661x661 operator is 3.5 MB, so
#: there is nothing for GMRES to save and no Krylov forcing schedule to tune.
NESTED_LS_JAX_INNER_LINEAR_SOLVER: Final[str] = "dense_lu"

#: The quantity a lane's ``exit_status`` was classified FROM.  The two lanes
#: do not classify the same norm, and one undifferentiated ``exit_status``
#: column would invite a reader to compare two different convergence tests:
#: the JAX reduced Schur Newton judges the reduced gradient
#: ``||grad phi_hat||_2`` at the projected ``y*``, while the C++ LS solver has
#: no projected-``y`` reduced gradient at all and is judged on the
#: full-decision gradient ``||grad J_LS||_2`` -- the same norm
#: :func:`nested_ls_residual_norm` returns.
EXIT_STATUS_FROM_REDUCED_GRADIENT: Final[str] = "reduced_gradient"
EXIT_STATUS_FROM_FULL_DECISION_GRADIENT: Final[str] = "full_decision_gradient"

_GEOMETRY_FILE: Final[str] = "geometry.json"
_SCALARS_FILE: Final[str] = "scalars.pkl"
_CORRECTION_FILE: Final[str] = "correction.pkl"

# ``python -S -c`` so the OpenMP pin is in the environment before the child
# imports the compiled extension.  The child re-enters this module, which is
# what keeps the native lane's residual, displacement and record construction
# the same code as the JAX lane's rather than a second implementation of it.
_NATIVE_CHILD_SOURCE: Final[str] = """\
import sys

from benchmarks.flat675_nested_endpoint import run_native_correction_child

run_native_correction_child(sys.argv[1])
"""


class NestedCorrectionChildFailed(RuntimeError):
    """Raised when the one-thread native correction child does not finish."""


class NativeConstraintWeightMismatch(ValueError):
    """Raised when the view's constraint weight is not the one the solver uses.

    The native lane's inner solve is driven by
    ``_run_native_banana_bfgs_then_newton``, which passes
    ``NESTED_LS_CONSTRAINT_WEIGHT`` to the kernel, while the ``BoozerSurface``
    the walk mutates is constructed from the view's ``constraint_weight``.
    If a view ever carries a different weight the lane would minimise one
    penalty and report the residual of another, so it stops here instead.
    """


@dataclass(frozen=True, slots=True)
class NestedCorrection:
    """One lane's nested-LS correction of one incoming flat-675 point.

    The record is provenance-complete in the sense the contract requires
    (``nested_ls_contract.py``: "Inner records name solver families,
    sequences, and per-stage options rather than a lane-agnostic policy"):

    ``inner_policy``
        ``NESTED_LS_JAX_INNER_POLICY_NAME`` or
        ``NESTED_LS_NATIVE_INNER_POLICY_NAME``.  The two lanes run materially
        different inner solvers; a shared record that omitted this would be
        false provenance.
    ``bfgs_iterations`` / ``newton_iterations``
        Per stage, not summed.  ``None`` for ``bfgs_iterations`` on the JAX
        lane means the stage does not exist there, not that it took zero
        steps.  ``wall_s`` on the native lane pays for BOTH stages.
    ``converged`` / ``exit_status`` / ``exit_status_quantity`` / ``persisted``
        ``converged`` is the two-valued bit (the solver's own ``success`` AND
        the residual under ``tolerance``); ``exit_status`` is the contract's
        three-valued refinement (``converged`` / ``coarse_converged`` /
        ``failed``) classified against ``tolerance`` with the coarse band
        ``NESTED_LS_NEWTON_COARSE_TOL`` and then floored by ``success``
        (:func:`exit_status_with_solver_success`), so the two fields cannot
        contradict each other.  ``exit_status_quantity`` names the NORM that
        classification read, which is not the same norm on the two lanes
        (:data:`EXIT_STATUS_FROM_REDUCED_GRADIENT` /
        :data:`EXIT_STATUS_FROM_FULL_DECISION_GRADIENT`); without it the
        column reads as one convergence test when it is two.  ``persisted``
        says whether the walk committed its iterate at all.  A non-persisted
        walk returns the INCOMING surface dofs, so a consumer reading
        ``surface_dofs_after`` without ``persisted`` can read its own input
        back as a solve outcome.
    ``reduced_gradient_l2``
        The reduced gradient ``||grad phi_hat||_2`` at the returned point,
        which is what the JAX Schur Newton actually drives.  ``None`` on the
        native lane: the C++ LS solver has no projected-``y`` reduced
        gradient to publish, so the reduced-gradient clause of the certified
        rejudge gate cannot be evaluated for it here.  Also ``None`` on ANY
        non-persisted walk: that norm was measured at the start point the
        walk refused to leave, so publishing it would describe the incoming
        point as a solve outcome.
    ``same_branch_as_incoming``
        ``|iota_after - iota_before| <= iota_branch_guard``
        (``NESTED_LS_OUTER_IOTA_BRANCH_GUARD``).  False means the inner solve
        landed on a different Boozer branch; per the contract that is a failed
        evaluation, not a correction, however well it converged.
    ``omp_num_threads``
        The ``OMP_NUM_THREADS`` environment variable exactly as seen by the
        process that ran THIS lane's solve (the parent for the JAX lane, the
        one-thread child for the native lane), read there rather than assumed
        here; ``None`` means the variable was not set in that process, which
        the Markdown renders as ``unpinned`` rather than as the string
        ``"None"``.  It is load-bearing on the native lane only -- the C++
        kernel's OpenMP reductions make a non-converged warm start
        thread-dependent -- and on the JAX lane it is provenance about the
        host, not about the solve.

    ``wall_s`` times that lane's inner solve in its own process: for the JAX
    lane that includes the XLA compile of the reduced Newton, for the native
    lane it covers both the BFGS and the Newton stage but excludes the
    child's process start and imports.  It is a progress number, not a timing
    claim.
    """

    lane: Literal["jax", "native"]
    inner_policy: str
    tolerance: float
    residual_norm_before: float
    residual_norm_after: float
    bfgs_iterations: int | None
    newton_iterations: int
    converged: bool
    exit_status: str
    exit_status_quantity: str
    persisted: bool
    reduced_gradient_l2: float | None
    coil_delta_inf: float
    surface_dofs_after: NDArray[np.float64]
    iota_before: float
    iota_after: float
    G_after: float
    same_branch_as_incoming: bool
    iota_branch_guard: float
    dof_displacement_l2: float
    dof_displacement_max: float
    point_displacement_max_m: float
    point_displacement_rms_m: float
    minor_radius_m: float
    omp_num_threads: str | None
    wall_s: float


def native_penalty_evaluation(
    native_boozer: BoozerSurface,
    *,
    surface_dofs: NDArray[np.float64],
    iota: float,
    G: float,
) -> NestedLsPenaltyEvaluation:
    """``J_LS`` and ``grad J_LS`` from the C++ kernel at one packed decision.

    ``evaluate_nested_ls_penalty_pair`` is the pair form of this call and
    cannot be used here: it evaluates both lanes at once, and the native lane
    runs in a child that must not pay a JAX compile to learn its own residual.
    The knobs come from the same contract constants that function reads and
    the record is its record, so the two produce the identical numbers at the
    identical state.
    """
    packed = pack_nested_ls_decision(surface_dofs, iota, G)
    value, gradient = native_boozer.boozer_penalty_constraints_vectorized(
        packed,
        derivatives=1,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        optimize_G=True,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    return NestedLsPenaltyEvaluation(
        objective=float(value),
        gradient=np.array(gradient, dtype=np.float64, copy=True).reshape(-1),
    )


def jax_penalty_evaluation(
    objective_fn,
    *,
    surface_dofs: NDArray[np.float64],
    iota: float,
    G: float,
) -> NestedLsPenaltyEvaluation:
    """``J_LS`` and ``grad J_LS`` from the packed penalty closure.

    ``objective_fn`` is the second closure of
    :func:`~simsopt_jax_adapters.geo.nested_ls_reduced.nested_ls_reduced_closures`,
    i.e. the very function the reduced Newton differentiates, so the JAX lane's
    residual is measured by the solver's own program.
    """
    packed = pack_nested_ls_decision(surface_dofs, iota, G)
    value, gradient = jax.value_and_grad(objective_fn)(
        jnp.asarray(packed, dtype=jnp.float64)
    )
    return NestedLsPenaltyEvaluation(
        objective=float(jax.device_get(value)),
        gradient=np.array(
            jax.device_get(gradient), dtype=np.float64, copy=True
        ).reshape(-1),
    )


def nested_ls_residual_norm(evaluation: NestedLsPenaltyEvaluation) -> float:
    """The nested contract's residual: the 2-norm of ``grad J_LS``."""
    return float(np.linalg.norm(evaluation.gradient))


def stays_on_incoming_branch(*, iota_before: float, iota_after: float) -> bool:
    """The contract's Boozer branch guard, as one predicate for both lanes.

    ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD`` exists because a converged inner
    solve can converge onto a DIFFERENT Boozer branch (the B3 shakedown
    measured ``iota 0.1409 -> -0.0024`` with every inner solve converging), so
    convergence alone rejects nothing.  One function, one constant, both
    lanes: the guard cannot drift between them.

    The contract defines this guard on ``iota`` and on nothing else, and this
    predicate is deliberately exactly that.  It therefore says nothing about a
    walk that diverged without changing ``iota`` much -- a solve whose surface
    displacement is astronomical can still pass here.  That case is rejected
    one level up, by :func:`correction_evaluation_status` in the comparison
    harness, which refuses to call any non-converged, non-persisted or
    non-finite walk a correction BEFORE this guard is consulted.
    """
    return bool(
        abs(float(iota_after) - float(iota_before)) <= NESTED_LS_OUTER_IOTA_BRANCH_GUARD
    )


def exit_status_with_solver_success(status: str, *, solver_success: bool) -> str:
    """The classified exit status, floored by the solver's own success bit.

    ``nested_ls_newton_exit_status`` reads one gradient norm and nothing else,
    so a walk that exhausted ``newton_maxiter`` while sitting just under the
    tolerance classifies as ``converged`` in the same record whose
    ``converged`` bit (which carries ``success``) is False.  A solver that did
    not declare success did not pass its own stopping test, so ``converged``
    is demoted to ``coarse_converged`` -- the status is still allowed to say
    the point is close, never that the solve finished.  Any weaker status is
    already honest and passes through unchanged.
    """
    if solver_success or status != NESTED_LS_NEWTON_EXIT_CONVERGED:
        return status
    return NESTED_LS_NEWTON_EXIT_COARSE_CONVERGED


def native_boozer_at(
    biotsavart: BiotSavart,
    surface: SurfaceXYZTensorFourier,
    *,
    label_target: float,
    constraint_weight: float,
) -> BoozerSurface:
    """A C++ ``BoozerSurface`` on the nested route's banana ``run_code`` policy.

    The label is ``Volume``: that is ``NESTED_LS_LABEL``, and the bridge
    refuses any flat-675 problem whose Boozer label is something else, so the
    two sides cannot disagree about what is being constrained.
    """
    return BoozerSurface(
        biotsavart,
        surface,
        Volume(surface),
        float(label_target),
        constraint_weight=float(constraint_weight),
        options=dict(nested_ls_banana_run_code_options()),
    )


def _correction_record(
    *,
    lane: Literal["jax", "native"],
    inner_policy: str,
    geometry_surface: SurfaceXYZTensorFourier,
    surface_before: NDArray[np.float64],
    surface_after: NDArray[np.float64],
    residual_norm_before: float,
    residual_norm_after: float,
    iota_before: float,
    iota_after: float,
    G_after: float,
    bfgs_iterations: int | None,
    newton_iterations: int,
    solver_success: bool,
    exit_status: str,
    exit_status_quantity: str,
    persisted: bool,
    reduced_gradient_l2: float | None,
    coil_delta_inf: float,
    omp_num_threads: str | None,
    wall_s: float,
) -> NestedCorrection:
    """Assemble one lane's record, including the metric-space displacement.

    ``geometry_surface`` is mutated here and must be a scratch clone: the
    displacement is read off the surface's own quadrature grid by evaluating
    ``gamma`` at both DOF vectors, and ``minor_radius`` is the *incoming*
    surface's own, so the relative displacement a reader computes is relative
    to the surface being judged rather than to the corrected one.

    The branch guard is applied here rather than in either lane so the two
    cannot drift: one predicate, one constant, both lanes.  So are the two
    consistency rules that would otherwise have to be repeated per lane: the
    exit status is floored by ``solver_success``
    (:func:`exit_status_with_solver_success`), and ``reduced_gradient_l2`` is
    dropped on a non-persisted walk, whose returned point is the caller's own
    input rather than a solve outcome.
    """
    before = np.asarray(surface_before, dtype=np.float64).reshape(-1)
    after = np.asarray(surface_after, dtype=np.float64).reshape(-1)
    geometry_surface.set_dofs(before)
    gamma_before = np.array(geometry_surface.gamma(), dtype=np.float64, copy=True)
    minor_radius = float(geometry_surface.minor_radius())
    geometry_surface.set_dofs(after)
    gamma_after = np.array(geometry_surface.gamma(), dtype=np.float64, copy=True)
    point_delta = np.linalg.norm(gamma_after - gamma_before, axis=-1).reshape(-1)
    dof_delta = after - before
    return NestedCorrection(
        lane=lane,
        inner_policy=str(inner_policy),
        tolerance=NESTED_CORRECTION_TOLERANCE,
        residual_norm_before=float(residual_norm_before),
        residual_norm_after=float(residual_norm_after),
        bfgs_iterations=None if bfgs_iterations is None else int(bfgs_iterations),
        newton_iterations=int(newton_iterations),
        converged=bool(solver_success)
        and float(residual_norm_after) <= NESTED_CORRECTION_TOLERANCE,
        exit_status=exit_status_with_solver_success(
            str(exit_status), solver_success=bool(solver_success)
        ),
        exit_status_quantity=str(exit_status_quantity),
        persisted=bool(persisted),
        reduced_gradient_l2=(
            None
            if reduced_gradient_l2 is None or not persisted
            else float(reduced_gradient_l2)
        ),
        coil_delta_inf=float(coil_delta_inf),
        surface_dofs_after=np.array(after, dtype=np.float64, copy=True),
        iota_before=float(iota_before),
        iota_after=float(iota_after),
        G_after=float(G_after),
        same_branch_as_incoming=stays_on_incoming_branch(
            iota_before=iota_before, iota_after=iota_after
        ),
        iota_branch_guard=float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
        dof_displacement_l2=float(np.linalg.norm(dof_delta)),
        dof_displacement_max=float(np.max(np.abs(dof_delta))),
        point_displacement_max_m=float(np.max(point_delta)),
        point_displacement_rms_m=float(np.sqrt(np.mean(point_delta * point_delta))),
        minor_radius_m=minor_radius,
        omp_num_threads=omp_num_threads,
        wall_s=float(wall_s),
    )


def correct_with_nested_ls_jax(view: NestedView) -> NestedCorrection:
    """Reduced nested-LS Schur Newton on the surface block, coils frozen.

    The decision is the surface block alone; ``(iota, G)`` come from the
    projected two-column solve at every iterate, which is the nested route's
    own inner formulation.  The walk starts at the incoming ``(iota, G)`` so
    the flat route's inner state seeds the nested one.

    The linear solve is the certified 661-DOF one: the Schur complement
    ``H_ss`` materialized from packed penalty HVPs and factored by dense LU
    (:data:`NESTED_LS_JAX_INNER_LINEAR_SOLVER`), which is what
    ``_solve_nested_inner_leg`` and ``run_ncsx_schur_inner`` run at this size.
    ``run_reduced_nested_ls_newton`` is NOT usable here even though it is the
    same mathematical step: its generic Newton polish builds a GMRES operator
    over the *reduced* objective ``phi_hat``, whose HVP differentiates through
    the projected-``y`` QR, and at 661 DOFs XLA cannot rematerialize that
    program below 27.13 GiB -- it dies with RESOURCE_EXHAUSTED on a 32 GB
    device.  The Schur path never differentiates through QR (it carries the
    2x2 ``Phi_yy`` block explicitly), so its live memory is the dense 661x661
    operator, 3.5 MB.  Same residual, and the same stopping tolerance
    ``NESTED_CORRECTION_TOLERANCE`` -- but NOT the same step as the native
    lane and NOT the same stopping policy: this walk is pure Newton on the
    reduced objective and has no BFGS pre-stage, which is why the record
    names ``NESTED_LS_JAX_INNER_POLICY_NAME`` and reports
    ``bfgs_iterations=None``.
    """
    jax_boozer = view.jax_inputs.new_boozer_surface_jax()
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(
        jax_boozer,
        constraint_weight=view.jax_inputs.constraint_weight,
        weight_inv_modB=view.jax_inputs.weight_inv_modB,
    )
    surface_before = np.array(view.surface_dofs, dtype=np.float64, copy=True)
    residual_before = nested_ls_residual_norm(
        jax_penalty_evaluation(
            objective_fn,
            surface_dofs=surface_before,
            iota=view.iota,
            G=view.G,
        )
    )
    started = perf_counter()
    result = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(view.iota),
        G=float(view.G),
        constraint_weight=view.jax_inputs.constraint_weight,
        weight_inv_modB=view.jax_inputs.weight_inv_modB,
        stab=NESTED_LS_JAX_INNER_STAB,
        tol=NESTED_CORRECTION_TOLERANCE,
        maxiter=NESTED_LS_BANANA_NEWTON_MAXITER,
        linear_solver=NESTED_LS_JAX_INNER_LINEAR_SOLVER,
        max_dense_linearization_bytes=None,
        residual_fn=residual_fn,
        objective_fn=objective_fn,
    )
    wall_s = perf_counter() - started
    residual_after = nested_ls_residual_norm(
        jax_penalty_evaluation(
            objective_fn,
            surface_dofs=result.surface_dofs,
            iota=result.iota,
            G=result.G,
        )
    )
    return _correction_record(
        lane="jax",
        inner_policy=NESTED_LS_JAX_INNER_POLICY_NAME,
        geometry_surface=clone_surface_xyz_tensor_fourier(view.surface_native),
        surface_before=surface_before,
        surface_after=result.surface_dofs,
        residual_norm_before=residual_before,
        residual_norm_after=residual_after,
        iota_before=float(view.iota),
        iota_after=result.iota,
        G_after=result.G,
        # There is no BFGS stage on this lane; ``None`` says "no such stage",
        # which is not the same statement as "zero steps".
        bfgs_iterations=None,
        newton_iterations=result.iteration_count,
        solver_success=result.success,
        exit_status=result.exit_status,
        # This lane's Newton drives the reduced gradient and its exit status
        # is classified from that norm; the native lane's is not.
        exit_status_quantity=EXIT_STATUS_FROM_REDUCED_GRADIENT,
        persisted=result.persisted,
        reduced_gradient_l2=float(np.linalg.norm(result.reduced_gradient)),
        coil_delta_inf=result.coil_delta_inf,
        omp_num_threads=nested_ls_threading_env()["OMP_NUM_THREADS"],
        wall_s=wall_s,
    )


def run_native_correction_child(payload_root: str) -> None:
    """Child entry point: the whole native correction, in a one-thread process.

    Geometry crosses the process boundary as simsopt's own serialization, so
    the child solves against the same coil objects the bridge reconstructed
    rather than against the archived ``native_biot_savart.json`` coil set,
    which is a different field.
    """
    root = Path(payload_root)
    scalars = pickle.loads((root / _SCALARS_FILE).read_bytes())
    _biotsavart, surface = simsopt.load(str(root / _GEOMETRY_FILE))
    surface_before = np.asarray(scalars["surface_dofs"], dtype=np.float64)
    surface.set_dofs(surface_before)
    native_boozer = native_boozer_at(
        _biotsavart,
        surface,
        label_target=scalars["label_target"],
        constraint_weight=scalars["constraint_weight"],
    )
    residual_before = nested_ls_residual_norm(
        native_penalty_evaluation(
            native_boozer,
            surface_dofs=surface_before,
            iota=scalars["iota"],
            G=scalars["G"],
        )
    )
    started = perf_counter()
    # ``BoozerSurface.run_code``'s BoozerLS branch IS this sequence, but it
    # returns only the Newton stage's dict, so the BFGS stage's iteration
    # count -- work ``wall_s`` below has already paid for -- is unrecoverable
    # from it.  ``_run_native_banana_bfgs_then_newton`` is the repository's
    # own native banana driver and runs the identical two stages at the
    # identical options (the overlay it applies is the banana option set this
    # ``BoozerSurface`` was already constructed with), publishing both counts,
    # the coil displacement and the process's observed threading.
    result = _run_native_banana_bfgs_then_newton(
        native_boozer,
        iota=float(scalars["iota"]),
        G=float(scalars["G"]),
    )
    wall_s = perf_counter() - started
    surface_after = np.array(surface.get_dofs(), dtype=np.float64, copy=True)
    residual_after = nested_ls_residual_norm(
        native_penalty_evaluation(
            native_boozer,
            surface_dofs=surface_after,
            iota=result["iota"],
            G=result["G"],
        )
    )
    correction = _correction_record(
        lane="native",
        inner_policy=NESTED_LS_NATIVE_INNER_POLICY_NAME,
        geometry_surface=clone_surface_xyz_tensor_fourier(surface),
        surface_before=surface_before,
        surface_after=surface_after,
        residual_norm_before=residual_before,
        residual_norm_after=residual_after,
        iota_before=float(scalars["iota"]),
        iota_after=float(result["iota"]),
        G_after=float(result["G"]),
        bfgs_iterations=int(result["bfgs_iter"]),
        newton_iterations=int(result["newton_iter"]),
        solver_success=bool(result["success"]),
        # The C++ solver writes its iterate into ``surface`` unconditionally,
        # so the record always carries the point the solver left and there is
        # no non-persisting branch to classify: ``persisted`` is the constant
        # True on this lane, whatever that point turned out to be.  Finiteness
        # is a separate question, and it is asked of the classifier below
        # rather than of ``persisted``.  The status is the contract's own
        # classifier applied to the quantity this lane HAS -- the full-decision
        # gradient norm the native Newton's tolerance judges -- in place of the
        # reduced gradient it cannot compute, which is why the record publishes
        # the quantity next to the status.
        exit_status=nested_ls_newton_exit_status(
            persisted=True,
            finite_iterate=bool(
                np.all(np.isfinite(surface_after))
                and np.isfinite(float(result["iota"]))
                and np.isfinite(float(result["G"]))
            ),
            reduced_gradient_l2=residual_after,
            tol=NESTED_CORRECTION_TOLERANCE,
        ),
        exit_status_quantity=EXIT_STATUS_FROM_FULL_DECISION_GRADIENT,
        persisted=True,
        reduced_gradient_l2=None,
        coil_delta_inf=float(result["coil_delta_inf"]),
        omp_num_threads=result["omp_num_threads"],
        wall_s=wall_s,
    )
    (root / _CORRECTION_FILE).write_bytes(pickle.dumps(correction))


def correct_with_nested_ls_native(view: NestedView) -> NestedCorrection:
    """C++ BoozerSurface LS/Newton on the same point, in a one-thread child.

    The child's environment is built by the repository's native-cpu parity
    SSOT, which pins ``OMP_NUM_THREADS=1`` and puts the parent's already
    loaded compiled kernel ahead of ``src/simsoptpp`` on ``PYTHONPATH``.  The
    parent's own thread count therefore does not reach the solve.
    """
    # The native inner driver hardwires BOTH penalty knobs; a view that carries
    # anything else would minimise one penalty and report the residual of
    # another, so the lane fails closed before the child is spawned.
    hardwired = (
        (
            "constraint_weight",
            float(view.jax_inputs.constraint_weight),
            float(NESTED_LS_CONSTRAINT_WEIGHT),
        ),
        (
            "weight_inv_modB",
            bool(view.jax_inputs.weight_inv_modB),
            bool(NESTED_LS_WEIGHT_INV_MODB),
        ),
    )
    for name, carried, driver_value in hardwired:
        if carried != driver_value:
            raise NativeConstraintWeightMismatch(
                f"the native lane's inner driver is hardwired to {name}="
                f"{driver_value!r}, but this view carries {name}={carried!r}: "
                "the walk would minimise one penalty and the record would report "
                "the residual of another"
            )
    with tempfile.TemporaryDirectory(prefix="flat675-nested-native-") as directory:
        root = Path(directory)
        simsopt.save(
            [view.biotsavart_native, view.surface_native],
            str(root / _GEOMETRY_FILE),
        )
        (root / _SCALARS_FILE).write_bytes(
            pickle.dumps(
                {
                    "surface_dofs": np.array(
                        view.surface_dofs, dtype=np.float64, copy=True
                    ),
                    "iota": float(view.iota),
                    "G": float(view.G),
                    "label_target": float(view.jax_inputs.label_target),
                    "constraint_weight": float(view.jax_inputs.constraint_weight),
                }
            )
        )
        environment = build_parity_lane_environment(
            "native-cpu", dict(os.environ), repo_root=PRODUCTION_ROOT
        )
        environment["PYTHONPATH"] = pythonpath_with_loaded_kernel(
            *environment["PYTHONPATH"].split(os.pathsep)
        )
        completed = subprocess.run(
            (sys.executable, "-S", "-c", _NATIVE_CHILD_SOURCE, str(root)),
            cwd=PRODUCTION_ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise NestedCorrectionChildFailed(
                "the one-thread native nested-LS child exited with "
                f"{completed.returncode}:\n{completed.stderr}"
            )
        return pickle.loads((root / _CORRECTION_FILE).read_bytes())


def flat675_boozer_term(
    problem: Flat675Problem,
    vector: NDArray[np.float64],
) -> float:
    """The flat-675 objective's own weighted Boozer term at one outer vector.

    This is the *weighted* term the flat objective sums, i.e.
    ``problem.objective_policy.residual_weight`` times the BoozerLS penalty
    ``J_LS = J_boozer + 0.5 * constraint_weight * (label - target)^2`` at the
    vector's own ``(iota, G)``.  It is the same functional ``J_LS`` whose
    gradient norm the nested residual reports -- a value, not a stationarity
    measure -- so the two numbers answer different questions about the same
    penalty and must be published side by side with that weight named.
    """
    weighted_terms = flat675_weighted_terms(
        jnp.asarray(np.asarray(vector, dtype=np.float64), dtype=jnp.float64),
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    term_index = FLAT675_OBJECTIVE_TERM_KEYS.index(FLAT675_BOOZER_TERM_KEY)
    return float(jax.device_get(weighted_terms[term_index]))


__all__ = [
    "FLAT675_BOOZER_TERM_KEY",
    "NESTED_CORRECTION_PHYSICS_TOLERANCE",
    "NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR",
    "NESTED_CORRECTION_TOLERANCE",
    "NESTED_CORRECTION_TOLERANCE_BAR",
    "NESTED_LS_JAX_INNER_LINEAR_SOLVER",
    "NestedCorrection",
    "NestedCorrectionChildFailed",
    "correct_with_nested_ls_jax",
    "correct_with_nested_ls_native",
    "flat675_boozer_term",
    "jax_penalty_evaluation",
    "native_boozer_at",
    "native_penalty_evaluation",
    "nested_ls_residual_norm",
    "run_native_correction_child",
    "stays_on_incoming_branch",
]
