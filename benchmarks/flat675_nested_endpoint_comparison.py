"""Is the flat-675 endpoint a Boozer surface to the nested route's tolerance?

The flat coupled single-stage formulation (``examples/jax/3_Advanced/
single_stage_flat675.py``) puts 11 coil, 3 vessel and 661 surface coordinates
in ONE vector and carries the Boozer residual as a weighted objective term
instead of solving it.  The nested route
(``src/simsopt_jax_adapters/geo/nested_ls_contract.py``) instead drives that
residual to ``NESTED_LS_BANANA_NEWTON_TOL`` with a Newton inner solve at every
outer step, so its surface is a Boozer surface by construction.

This program measures the gap between the two, at the start point and at the
solve endpoint, with no adjectives:

* the nested-contract residual before and after a nested correction, under ONE
  shared definition for both lanes (the shared-definition check is a gate here,
  not an assumption);
* how far the nested correction moves the surface -- in DOFs, in metres on the
  quadrature grid, and relative to the minor radius;
* what happens to ``(iota, G)``, which the flat formulation closes with a
  two-column least-squares y-solve rather than solving for;
* the eight-term flat-675 objective and every one of its terms at the four
  points ``start``, ``endpoint``, ``endpoint + jax correction`` and
  ``endpoint + native correction``;
* whether the JAX and native (C++ ``BoozerSurface`` LS/Newton) corrections
  agree with each other, which is what makes a small correction believable.

**Two bars, both published.**  The nested contract carries a TIMING bar
(``NESTED_LS_TIMING_BAR``, banana ``run_code``, ``NESTED_LS_BANANA_NEWTON_TOL
= 1e-11``) and a PHYSICS bar (``NESTED_LS_PHYSICS_BAR``, the
reconstruct/rejudge Newton, ``NESTED_LS_NEWTON_TOL = 1e-13``).  Both lanes
STOP at the timing bar, so that is what ``converged`` means; the certified
rows in ``benchmarks/nested_ls_outer_claim.py`` are rejudged at the PHYSICS
bar.  Every residual here is therefore reported as a ratio against BOTH, and
every gate names the bar it uses.

**The bounded repository fixture is underresolved.** Its 661 surface
coefficients are sampled on a 6-by-6 grid, so the nested-LS residual the
inner solve differentiates has a 110-by-663 Jacobian against the decision
``[surface_dofs, iota, G]``: 553 unknowns more than equations, and therefore
a Gauss-Newton normal matrix that is singular at every iterate (counted from
the solver's own residual closure in
``tests/benchmarks/test_flat675_nested_endpoint_comparison.py``). This
configuration exercises failure reporting; it cannot certify a unique nested
correction. Which way the singular inner solve terminates is decided by the
last bits of the incoming endpoint, and THREE outcomes have been recorded at
the same fixture: a failed Newton return, a ``numpy.linalg.LinAlgError``
before the solver returns, and a converged solve that landed on a different
Boozer branch (``|Delta iota| = 5.21e-2`` against the ``5e-2`` guard). None of
them is a correction, all three exit nonzero, and unavailable post-state stays
null. Use a resolved fixture for numerical certification.

``nested_ls_outer_claim.py``'s "endpoint C++ LS Newton rejudge no-op"
(``benchmarks/nested_ls_outer_jax_child.py``) is: the endpoint Newton at
``nested_ls_physics_newton_kwargs()`` succeeds, takes ``iter == 0``, moves
neither coils nor surface, leaves ``grad_l2 <= NESTED_LS_NEWTON_TOL``, AND
leaves the reduced gradient at the projected ``y*`` under the same tolerance.
:func:`nested_correction_is_noop` here is that predicate at that bar, with the
reduced-gradient clause evaluated on the JAX lane -- whose Schur result
publishes ``reduced_gradient`` -- and NOT evaluable on the native lane, which
has no projected-``y`` reduced gradient to publish.  The payload says so per
lane (``reduced_gradient_clause_available``), so the native lane's no-op is
the weaker statement and is never read as the stronger one.

**The branch guard.**  ``NESTED_LS_OUTER_IOTA_BRANCH_GUARD = 0.05`` is the
contract's rule that convergence alone rejects nothing: an inner solve whose
``iota`` moves further than the guard from the incoming anchor has landed on a
different Boozer branch and is a FAILED EVALUATION, not a correction.  Every
row carries ``same_branch_as_incoming`` and an ``evaluation_status`` of
``correction`` or ``rejected_branch_change``, and the verdict block states it
per lane.

Its "endpoint eight-term J parity" is the one-sided band
``jax_j <= native_j * (1 + rtol)``; the same band is computed here against the
native C++ twin (``examples/3_Advanced/single_stage_flat675.py``) at
``OBJECTIVE_RTOL``.

Run it (clone-runnable; the ``bundle`` configuration additionally needs the
host-local frozen input bundle)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 MPI4PY_RC_INITIALIZE=false \\
    MPLBACKEND=Agg PYTHONPATH=<repo>/src:<repo>/build/<tag> \\
    python benchmarks/flat675_nested_endpoint_comparison.py \\
        --configuration repository-geometry --max-steps 3 \\
        --out-json runs/repo_b3_cpu.json --jax-platform cpu

A GPU run must additionally carry ``--xla_gpu_autotune_level=0`` in
``XLA_FLAGS``; this program refuses to run on GPU without it, because fresh
compiles otherwise differ at 3e-15 and the endpoint would not be reproducible.

The Markdown table is written next to ``--out-json`` (same stem, ``.md``) and
echoed on stdout.  Nothing here adjudicates the campaign: it reports numbers,
walls and gate booleans, and the report page states them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Final, Literal

import jax
import numpy as np
from numpy.typing import NDArray

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simsopt_jax.examples.single_stage_flat675 import (
    FLAT675_LBFGS_HISTORY,
    FLAT675_LBFGS_MAXLS,
    prepare_single_stage_flat675,
    solve_single_stage_flat675,
)
from simsopt_jax.runtime.host_boundary import host_transfer_audit
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.geo.flat675 import (
    FLAT675_OBJECTIVE_TERM_KEYS,
    FLAT675_OUTER_DOF_COUNT,
    FLAT675_SURFACE_SLICE,
    Flat675Problem,
    bind_flat675_programs,
    load_flat675_bundle,
)
from simsopt_jax_adapters.geo.flat675.nested_bridge import (
    NestedView,
    nested_view_from_flat675,
)
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_OUTER_IOTA_BRANCH_GUARD,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    dump_strict_json,
    nested_ls_runtime_identity,
    sha256_float64,
)

from benchmarks.flat675_nested_endpoint import (
    NESTED_CORRECTION_PHYSICS_TOLERANCE,
    NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR,
    NESTED_CORRECTION_TOLERANCE,
    NESTED_CORRECTION_TOLERANCE_BAR,
    NestedCorrection,
    correct_with_nested_ls_jax,
    correct_with_nested_ls_native,
    flat675_boozer_term,
)

# v1 -> v2: ``NestedCorrection`` became true per-lane provenance (inner policy,
# per-stage iteration counts, three-valued exit status, persistence, the
# reduced gradient where it exists, the lane's OMP thread count), every
# residual is now published against BOTH contract bars, and the no-op gate
# moved from the timing bar to the physics bar and gained the branch-guard and
# reduced-gradient clauses.  Fields changed meaning, not only count -- a v1
# consumer reading a v2 ``nested_correction_is_noop`` would read a
# physics-bar verdict as a timing-bar one -- so the id has to move.  v3:
# ``evaluation_status`` is decided by convergence and persistence BEFORE the
# branch guard (``failed_solve`` is never a ``correction``) and every record
# names the gradient its ``exit_status`` was classified on.  v4 makes absent
# native failure post-state values null and makes the CLI nonzero on any
# record the contract does not call a ``correction`` -- ``failed_solve`` and
# ``rejected_branch_change`` alike, since the branch guard makes the second a
# failed EVALUATION.  That rule is about the exit code, not the payload, so it
# does not move the id.  The id also stays at v4 for the lane-agreement
# pre-state fill-in: ``residual_before_*`` and
# ``shared_residual_definition_ok`` are published on a failed solve too,
# because they are measured before either inner solve runs.  No field changed
# meaning -- a value a v4 consumer would have read as null now carries the
# number that field has always been defined to hold -- so only availability
# widened, and null still means "nothing to compare".
SCHEMA: Final[str] = "flat675-nested-endpoint-comparison-v4"

# The shipped lessons this program runs.  The tier directories are not
# importable packages, so the two scripts are loaded by file location -- the
# same route ``_load_script`` in ``tests/jax/examples/test_single_stage_flat675_native_twin.py``
# and ``benchmarks/flat675_promotion_robustness_child.py`` take, and for the
# same reason: rebuilding either configuration here would put a second copy of
# the certified geometry in the tree.
JAX_EXAMPLE_PATH: Final[Path] = (
    REPO_ROOT / "examples" / "jax" / "3_Advanced" / "single_stage_flat675.py"
)
NATIVE_EXAMPLE_PATH: Final[Path] = (
    REPO_ROOT / "examples" / "3_Advanced" / "single_stage_flat675.py"
)

# Transcribed from ``OBJECTIVE_RTOL`` in ``tests/jax/examples/test_single_stage_flat675_native_twin.py``
# (``OBJECTIVE_RTOL``), which is the file that owns the JAX-vs-native
# same-point objective bar.  ``tests`` is not a package, so the value cannot be
# imported; ``tests/benchmarks/test_flat675_nested_endpoint_comparison.py``
# parses that file and fails if the two ever drift apart.
OBJECTIVE_RTOL: Final[float] = 1.0e-10

# The repository-geometry configuration runs at the example's bounded scale
# (grid 6, 24 curve quadrature points).  The frozen CLI has no scale flag, so
# the choice is recorded in the payload rather than left implicit.
REPOSITORY_EXECUTION_SCALE: Final[str] = "bounded"

# The solve tolerances the shipped JAX lesson passes to the fused lane
# (the ``rtol``/``atol`` of the solve call in ``examples/jax/3_Advanced/single_stage_flat675.py``).
SOLVE_RTOL: Final[float] = 1.0e-15
SOLVE_ATOL: Final[float] = 1.0e-12
SOLVE_OBJECTIVE_SCALE: Final[float] = 1.0

# Memory gist (GPU autotuner nondeterminism): fresh compiles differ at 3e-15
# unless the fusion autotuner is off, and it is speed-neutral here.
REQUIRED_GPU_XLA_FLAG: Final[str] = "--xla_gpu_autotune_level=0"

# Both lanes' ``residual_norm_before`` come from the same definition evaluated
# on the same view, so they must agree to round-off, not to a physics band.
SHARED_RESIDUAL_RTOL: Final[float] = 1.0e-12
# Below the nested solver's own stopping tolerance two residual norms are
# indistinguishable by construction, so the definition gate carries that
# absolute floor: |jax - native| <= rtol * max(|jax|, |native|) + atol.
#
# Read the START rows with that in mind: there the two lanes sit at ~1.4e-14,
# their RELATIVE gap is ~4.8e-3, and the gate passes only through this
# absolute floor.  That is the floor doing its job rather than the gate being
# vacuous -- two norms three orders under the solver tolerance cannot be
# compared relatively, because neither one's low-order digits mean anything.
# The definition equality itself is proven where the numbers are large enough
# to carry it: at the ENDPOINT rows the relative gap is ~1e-15 at a residual
# of ~1.6e-2, entirely inside ``SHARED_RESIDUAL_RTOL`` with the floor
# contributing nothing.
SHARED_RESIDUAL_ATOL: Final[float] = NESTED_CORRECTION_TOLERANCE

#: What one lane's run at one point actually is.  Only a walk that CONVERGED,
#: PERSISTED its iterate and produced a finite displacement corrected this
#: point at all; anything else is ``failed_solve``, whatever its ``iota``
#: happened to do.  The repository-geometry B3 CPU row is the case that
#: forced this ordering: a walk with ``converged False``, ``exit_status
#: failed`` and ``dof_displacement_max 2.95e14`` sits within the iota guard
#: and was published as a "correction".
#:
#: Among walks that did converge, one that left the incoming Boozer branch is
#: still not a correction OF THIS POINT: the contract's branch guard
#: (``NESTED_LS_OUTER_IOTA_BRANCH_GUARD``) makes it a rejected evaluation,
#: because calling it a correction would report a solve of a different
#: problem as an answer to this one.
EVALUATION_CORRECTION: Final[str] = "correction"
EVALUATION_REJECTED_BRANCH_CHANGE: Final[str] = "rejected_branch_change"
EVALUATION_FAILED_SOLVE: Final[str] = "failed_solve"

Configuration = Literal["bundle", "repository-geometry"]
JaxPlatform = Literal["cpu", "gpu"]
PointName = Literal["start", "endpoint"]

POINT_NAMES: Final[tuple[PointName, PointName]] = ("start", "endpoint")
LANE_NAMES: Final[tuple[str, str]] = ("jax", "native")


# --------------------------------------------------------------------------
# Injected collaborators (production wiring at the bottom of this section)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NestedLanes:
    """The unit-A bridge and the two unit-B correction lanes, as collaborators.

    Passing them in rather than calling them by name is what lets the contract
    tests drive this program with stand-ins that carry the real signatures.
    """

    view_fn: Callable[[Flat675Problem, NDArray[np.float64]], NestedView]
    jax_fn: Callable[[NestedView], NestedCorrection]
    native_fn: Callable[[NestedView], NestedCorrection]
    boozer_term_fn: Callable[[Flat675Problem, NDArray[np.float64]], float]


PRODUCTION_LANES: Final[NestedLanes] = NestedLanes(
    view_fn=nested_view_from_flat675,
    jax_fn=correct_with_nested_ls_jax,
    native_fn=correct_with_nested_ls_native,
    boozer_term_fn=flat675_boozer_term,
)


def _load_script(path: Path, module_name: str) -> ModuleType:
    """Load a shipped example script by file location.

    The example tiers (``3_Advanced``) are directories, not packages, and their
    names are not identifiers, so there is no import path to them.  This is the
    in-tree route to the shipped configuration and the native twin.
    """
    specification = spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise SystemExit(f"cannot load the shipped example at {path}")
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@dataclass(frozen=True, slots=True)
class ObjectiveLane:
    """One lane's eight-term objective over the 675-vector."""

    value: Callable[[NDArray[np.float64]], float]
    weighted_terms: Callable[[NDArray[np.float64]], dict[str, float]]


def jax_objective_lane(problem: Flat675Problem) -> ObjectiveLane:
    """The certified JAX eight-term objective bound to one problem."""
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )

    def value(vector: NDArray[np.float64]) -> float:
        return float(np.asarray(jax.device_get(programs.objective_fn(vector))))

    def weighted_terms(vector: NDArray[np.float64]) -> dict[str, float]:
        terms = np.asarray(
            jax.device_get(programs.diagnostics_fn(vector)), dtype=np.float64
        )
        return {
            str(key): float(term)
            for key, term in zip(FLAT675_OBJECTIVE_TERM_KEYS, terms, strict=True)
        }

    return ObjectiveLane(value=value, weighted_terms=weighted_terms)


def native_twin_value(
    *, configuration: Configuration
) -> Callable[[NDArray[np.float64]], float]:
    """The native C++ twin's eight-term objective over the same 675-vector.

    ``examples/3_Advanced/single_stage_flat675.py`` is the twin the JAX example
    is certified against; its ``value_and_gradient`` unpacks the vector into
    real simsopt objects, so it is an independent evaluation of the same
    objective, not a second call into the JAX program.
    """
    twin_module = _load_script(
        NATIVE_EXAMPLE_PATH, "native_flat675_twin_for_comparison"
    )
    twin = (
        twin_module._bundle_problem()
        if configuration == "bundle"
        else twin_module._repository_problem(native_scale=False)
    )

    def value(vector: NDArray[np.float64]) -> float:
        objective, _gradient = twin.value_and_gradient(
            np.asarray(vector, dtype=np.float64)
        )
        return float(objective)

    return value


def build_problem(configuration: Configuration) -> Flat675Problem:
    """The flat-675 problem for one configuration, from the shipped lesson."""
    jax_example = _load_script(JAX_EXAMPLE_PATH, "jax_flat675_for_comparison")
    if configuration == "bundle":
        return load_flat675_bundle(jax_example.BUNDLE_ROOT)
    return jax_example._repository_problem(REPOSITORY_EXECUTION_SCALE)


# --------------------------------------------------------------------------
# The fused solve, and the four points
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    """The fused lane's endpoint plus the observables the report quotes."""

    start_vector: NDArray[np.float64]
    endpoint_vector: NDArray[np.float64]
    iterations: int
    objective_evaluations: int
    final_objective: float
    success: bool
    endpoint_is_optimizer_x: bool
    transfer_ledger: dict[str, int]
    wall_s: float


def run_fused_solve(problem: Flat675Problem, *, max_steps: int) -> SolveOutcome:
    """Run the shipped fused flat-675 lane once and keep its endpoint."""
    programs = bind_flat675_programs(
        material=problem.material,
        objective_policy=problem.objective_policy,
        boozer_policy=problem.boozer_policy,
    )
    start = np.asarray(problem.start_candidate.outer_vector(), dtype=np.float64)
    prepared = prepare_single_stage_flat675(
        objective_fn=programs.objective_fn,
        diagnostics_fn=programs.diagnostics_fn,
        initial_parameters=jax.device_put(start),
        objective_scale=jax.device_put(
            np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)
        ),
    )
    started = perf_counter()
    with host_transfer_audit() as audit, jax.transfer_guard("disallow"):
        result = solve_single_stage_flat675(
            prepared,
            driver=Driver.SIMSOPT_LBFGSB,
            max_steps=max_steps,
            rtol=SOLVE_RTOL,
            atol=SOLVE_ATOL,
        )
    endpoint = np.asarray(jax.device_get(result.x), dtype=np.float64)
    wall = perf_counter() - started
    optimizer_x = np.asarray(jax.device_get(prepared.problem.x), dtype=np.float64)
    return SolveOutcome(
        start_vector=start,
        endpoint_vector=endpoint,
        iterations=int(result.nit),
        objective_evaluations=int(result.nfev),
        final_objective=float(result.fun),
        success=bool(result.success),
        endpoint_is_optimizer_x=bool(np.array_equal(endpoint, optimizer_x)),
        transfer_ledger={
            str(entry.phase): int(entry.calls) for entry in audit.summary()
        },
        wall_s=wall,
    )


def corrected_vector(
    vector: NDArray[np.float64], correction: NestedCorrection
) -> NDArray[np.float64]:
    """The same coil and vessel blocks with the nested lane's surface block."""
    updated = np.array(vector, dtype=np.float64, copy=True)
    if correction.surface_dofs_after is None:
        raise SystemExit(
            f"{correction.lane} correction has no persisted post-state surface."
        )
    surface_after = np.asarray(correction.surface_dofs_after, dtype=np.float64)
    if surface_after.shape != updated[FLAT675_SURFACE_SLICE].shape:
        raise SystemExit(
            f"{correction.lane} correction returned a surface block of shape "
            f"{surface_after.shape}; the layout needs "
            f"{updated[FLAT675_SURFACE_SLICE].shape}."
        )
    updated[FLAT675_SURFACE_SLICE] = surface_after
    return updated


def _relative_gap(left: float, right: float) -> float:
    """Relative difference on the scale the native-twin objective bar uses.

    The floor of 1.0 in the denominator is the twin test's own construction
    (``start_relative`` in ``tests/jax/examples/test_single_stage_flat675_native_twin.py``), kept
    here so this program's twin numbers are read against the same bar.
    """
    return abs(left - right) / max(abs(right), 1.0)


def _relative_difference(left: float, right: float) -> float:
    """Symmetric relative difference, with no floor in the denominator.

    Residual norms are far below 1, so the twin bar's floored denominator would
    turn a relative gate into an absolute one and pass anything small.
    """
    scale = max(abs(left), abs(right))
    if scale == 0.0:
        return 0.0
    return abs(left - right) / scale


def _finite_or_none(value: float | None) -> float | None:
    """A displacement as written, or ``None`` when the walk diverged.

    Same reason as ``_bar_ratio``: ``dump_strict_json(allow_nan=False)`` would
    otherwise abort the run at write time on exactly the ``failed_solve``
    record the evaluation status exists to publish.
    """
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _bar_ratio(residual: float | None, bar: float | None) -> float | None:
    """One residual against one contract bar, or ``None`` when it has none.

    The payload is written by ``dump_strict_json(allow_nan=False)``, so an
    unguarded ``residual / bar`` aborts the ENTIRE run at write time on
    exactly the diverged solve this page exists to report.  ``None`` (JSON
    ``null``, Markdown ``n/a``) is the honest statement: there is no ratio,
    not a ratio of zero.

    The QUOTIENT is what is tested, not the residual: dividing by the physics
    bar multiplies by 1e13, so a residual that is finite but large overflows
    to infinity here while passing any test applied to the input.
    """
    if residual is None or bar is None:
        return None
    value = float(residual)
    if not math.isfinite(value):
        return None
    ratio = value / float(bar)
    return ratio if math.isfinite(ratio) else None


def _term_deltas(
    after: Mapping[str, float], before: Mapping[str, float]
) -> dict[str, float]:
    return {key: float(after[key] - before[key]) for key in before}


def correction_evaluation_status(correction: NestedCorrection) -> str:
    """``correction``, ``rejected_branch_change`` or ``failed_solve``.

    The clauses are ORDERED, and the order is the whole point.  The branch
    guard is a statement about WHICH Boozer branch a solve landed on, so it is
    only meaningful once there is a solve to speak of: a diverged walk that
    never converged, never persisted, or moved the surface by a non-finite
    amount has no branch to be on, and reading its ``same_branch_as_incoming``
    first is how a failure with a quiet ``iota`` gets published as a
    correction.

    ``converged`` here is the timing-bar bit the lanes stop at (the solver's
    own ``success`` AND the residual under ``NESTED_CORRECTION_TOLERANCE``);
    this status is about whether the row describes a correction at all, not
    about which bar it clears, which is what ``nested_correction_is_noop``
    and the ``residual_before_under_*_bar`` flags are for.
    """
    if not (correction.converged and correction.persisted):
        return EVALUATION_FAILED_SOLVE
    if (
        correction.dof_displacement_max is None
        or correction.point_displacement_max_m is None
        or correction.same_branch_as_incoming is None
        or correction.surface_dofs_after is None
        or correction.residual_norm_after is None
        or correction.iota_after is None
        or correction.G_after is None
        or correction.coil_delta_inf is None
    ):
        return EVALUATION_FAILED_SOLVE
    finite_displacement = math.isfinite(
        float(correction.dof_displacement_max)
    ) and math.isfinite(float(correction.point_displacement_max_m))
    if not finite_displacement:
        return EVALUATION_FAILED_SOLVE
    if correction.same_branch_as_incoming is False:
        return EVALUATION_REJECTED_BRANCH_CHANGE
    return EVALUATION_CORRECTION


def nested_correction_is_noop(correction: NestedCorrection) -> bool:
    """The certified rejudge no-op, at the PHYSICS bar, for one lane.

    Clause by clause against ``benchmarks/nested_ls_outer_jax_child.py``'s
    ``rejudge_noop`` (plus its separate ``reduced_grad_ok``):

    ``rejudge_success``          -> ``converged`` and ``persisted``  [TIMING]
    ``rejudge_iter == 0``        -> every inner stage took zero steps
    ``coil_delta_inf == 0.0``    -> ``coil_delta_inf == 0.0``
    ``surface_delta_inf == 0.0`` -> ``dof_displacement_max == 0.0``
    ``grad_l2 <= grad_tol``      -> ``residual_norm_before`` under
                                    ``NESTED_CORRECTION_PHYSICS_TOLERANCE``
    ``reduced_grad_l2 <= tol``   -> ``reduced_gradient_l2`` under the same
                                    tolerance, WHERE THE LANE HAS ONE

    plus the branch guard, which that gate does not carry because its lane
    never moves: a solve that left the branch is not a no-op under any bar.

    Two of these clauses are not physics-bar quantities and are marked
    ``[TIMING]``: ``converged`` is the solver's ``success`` bit AND the
    residual under the TIMING bar ``NESTED_CORRECTION_TOLERANCE``, because
    that is the bar both lanes actually stop at, and ``persisted`` is not a
    bar at all.  Nothing is lost by mixing them in here: a residual under the
    physics bar is two orders under the timing bar, so the residual half of
    the ``converged`` clause is IMPLIED by the physics clause below it, and
    the only independent content ``[TIMING]`` adds is the solver's own
    ``success`` bit and the commit.

    The reduced-gradient clause is the one asymmetry between the lanes, and
    it runs the OPPOSITE way to the rest of the record: the native lane has
    no projected-``y`` reduced gradient, so that clause is vacuous there and
    a native ``True`` is weaker than a JAX ``True`` by exactly that clause.
    Every OTHER clause is at least as hard for the native lane, and
    ``rejudge_iter == 0`` is strictly harder: the native lane must take zero
    BFGS steps as well as zero Newton steps, a stage the JAX lane does not
    have.  The payload records ``reduced_gradient_clause_available`` next to
    every verdict so a reader can tell which of the two statements they are
    looking at.
    """
    if not (
        correction.converged
        and correction.persisted
        and correction.same_branch_as_incoming is True
        and correction.newton_iterations is not None
        and correction.coil_delta_inf is not None
        and correction.dof_displacement_max is not None
    ):
        return False
    residual_before = float(correction.residual_norm_before)
    reduced = correction.reduced_gradient_l2
    return bool(
        int(correction.newton_iterations) == 0
        and int(correction.bfgs_iterations or 0) == 0
        and float(correction.coil_delta_inf) == 0.0
        and float(correction.dof_displacement_max) == 0.0
        and math.isfinite(residual_before)
        and residual_before <= NESTED_CORRECTION_PHYSICS_TOLERANCE
        and (
            reduced is None
            or (
                math.isfinite(float(reduced))
                and float(reduced) <= NESTED_CORRECTION_PHYSICS_TOLERANCE
            )
        )
    )


def _correction_record(
    *,
    correction: NestedCorrection,
    base_vector: NDArray[np.float64],
    base_objective: float,
    base_terms: Mapping[str, float],
    problem: Flat675Problem,
    lanes: NestedLanes,
    objective: ObjectiveLane,
    twin_value: Callable[[NDArray[np.float64]], float],
) -> dict[str, object]:
    """One lane's correction, and the objective at the point it corrected to."""
    evaluation_status = correction_evaluation_status(correction)
    has_reportable_post_state = evaluation_status != EVALUATION_FAILED_SOLVE
    if has_reportable_post_state:
        point = corrected_vector(base_vector, correction)
        view = lanes.view_fn(problem, point)
        corrected_objective: float | None = objective.value(point)
        corrected_terms: dict[str, float] | None = objective.weighted_terms(point)
        twin_objective: float | None = twin_value(point)
        surface_dofs_after_sha256: str | None = sha256_float64(
            correction.surface_dofs_after
        )
        iota_after = float(correction.iota_after)
        G_after = float(correction.G_after)
        iota_delta: float | None = iota_after - float(correction.iota_before)
        y_solve_iota: float | None = float(view.iota)
        y_solve_G: float | None = float(view.G)
        y_solve_minus_lane_iota: float | None = y_solve_iota - iota_after
        y_solve_minus_lane_G: float | None = y_solve_G - G_after
        flat675_boozer_term: float | None = float(lanes.boozer_term_fn(problem, point))
        objective_minus_base: float | None = corrected_objective - base_objective
        weighted_term_deltas: dict[str, float] | None = _term_deltas(
            corrected_terms, base_terms
        )
        native_twin_relative_gap: float | None = _relative_gap(
            corrected_objective, twin_objective
        )
        native_twin_within_objective_rtol: bool | None = (
            native_twin_relative_gap <= OBJECTIVE_RTOL
        )
    else:
        corrected_objective = None
        corrected_terms = None
        twin_objective = None
        surface_dofs_after_sha256 = None
        iota_after = None
        G_after = None
        iota_delta = None
        y_solve_iota = None
        y_solve_G = None
        y_solve_minus_lane_iota = None
        y_solve_minus_lane_G = None
        flat675_boozer_term = None
        objective_minus_base = None
        weighted_term_deltas = None
        native_twin_relative_gap = None
        native_twin_within_objective_rtol = None
    reduced_gradient_l2 = correction.reduced_gradient_l2
    return {
        "lane": str(correction.lane),
        # Provenance: which inner solver produced this record, in how many
        # steps of which stage, how it exited, and under which thread count.
        # One lane-agnostic "iterations" would compare two different solvers.
        "inner_policy": str(correction.inner_policy),
        "bfgs_iterations": (
            None
            if correction.bfgs_iterations is None
            else int(correction.bfgs_iterations)
        ),
        "newton_iterations": (
            None
            if correction.newton_iterations is None
            else int(correction.newton_iterations)
        ),
        "exit_status": str(correction.exit_status),
        # WHICH norm that status was classified from.  The two lanes do not
        # classify the same one, so the column is only readable next to this.
        "exit_status_quantity": str(correction.exit_status_quantity),
        "persisted": bool(correction.persisted),
        "failure_reason": correction.failure_reason,
        "reduced_gradient_l2": (
            None if reduced_gradient_l2 is None else float(reduced_gradient_l2)
        ),
        "reduced_gradient_clause_available": bool(reduced_gradient_l2 is not None),
        "coil_delta_inf": _finite_or_none(correction.coil_delta_inf),
        "omp_num_threads": correction.omp_num_threads,
        "tolerance": float(correction.tolerance),
        "tolerance_bar": NESTED_CORRECTION_TOLERANCE_BAR,
        # The corrected surface itself is identified by its sha rather than
        # inlined four times over; the point it defines is what the numbers
        # below are measured at.
        "surface_dofs_after_sha256": surface_dofs_after_sha256,
        "residual_norm_before": float(correction.residual_norm_before),
        "residual_norm_after": _finite_or_none(correction.residual_norm_after),
        # The same residual against both contract bars, so no reader has to
        # guess which one a ratio was taken against.
        "residual_before_over_timing_bar": _bar_ratio(
            correction.residual_norm_before, NESTED_CORRECTION_TOLERANCE
        ),
        "residual_before_over_physics_bar": _bar_ratio(
            correction.residual_norm_before, NESTED_CORRECTION_PHYSICS_TOLERANCE
        ),
        "residual_after_over_timing_bar": _bar_ratio(
            correction.residual_norm_after, NESTED_CORRECTION_TOLERANCE
        ),
        "residual_after_over_physics_bar": _bar_ratio(
            correction.residual_norm_after, NESTED_CORRECTION_PHYSICS_TOLERANCE
        ),
        # ``converged`` is the timing bar (both lanes stop there).
        "converged": bool(correction.converged),
        "iota_before": float(correction.iota_before),
        "iota_after": iota_after,
        "iota_delta": iota_delta,
        "same_branch_as_incoming": (
            correction.same_branch_as_incoming if has_reportable_post_state else None
        ),
        "iota_branch_guard": float(correction.iota_branch_guard),
        "evaluation_status": evaluation_status,
        "G_after": G_after,
        "dof_displacement_l2": _finite_or_none(correction.dof_displacement_l2),
        "dof_displacement_max": _finite_or_none(correction.dof_displacement_max),
        "point_displacement_max_m": _finite_or_none(
            correction.point_displacement_max_m
        ),
        "point_displacement_rms_m": _finite_or_none(
            correction.point_displacement_rms_m
        ),
        "minor_radius_m": _finite_or_none(correction.minor_radius_m),
        "point_displacement_max_over_minor_radius": _bar_ratio(
            correction.point_displacement_max_m, correction.minor_radius_m
        ),
        "point_displacement_rms_over_minor_radius": _bar_ratio(
            correction.point_displacement_rms_m, correction.minor_radius_m
        ),
        "wall_s": float(correction.wall_s),
        # The nested lane reports its own (iota, G); the flat formulation
        # closes them from the surface by a y-solve.  Both are recorded, and
        # their difference is a statement about the corrected surface.
        "y_solve_iota_at_corrected": y_solve_iota,
        "y_solve_G_at_corrected": y_solve_G,
        "y_solve_minus_lane_iota": y_solve_minus_lane_iota,
        "y_solve_minus_lane_G": y_solve_minus_lane_G,
        "flat675_boozer_term_at_corrected": flat675_boozer_term,
        "objective": corrected_objective,
        "objective_minus_base": objective_minus_base,
        "weighted_terms": corrected_terms,
        "weighted_term_deltas_vs_base": weighted_term_deltas,
        "native_twin_objective": twin_objective,
        "native_twin_relative_gap": native_twin_relative_gap,
        "native_twin_within_objective_rtol": native_twin_within_objective_rtol,
        # The comparable statement to nested_ls_outer_claim.py's "endpoint C++
        # LS Newton rejudge no-op", at that gate's own PHYSICS bar and with
        # its reduced-gradient clause where the lane can supply one.
        "nested_correction_is_noop": nested_correction_is_noop(correction),
        "nested_correction_noop_bar": NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR,
        "residual_before_under_timing_bar": bool(
            math.isfinite(float(correction.residual_norm_before))
            and float(correction.residual_norm_before) <= NESTED_CORRECTION_TOLERANCE
        ),
        "residual_before_under_physics_bar": bool(
            math.isfinite(float(correction.residual_norm_before))
            and float(correction.residual_norm_before)
            <= NESTED_CORRECTION_PHYSICS_TOLERANCE
        ),
    }


def _lane_agreement(
    *,
    jax_correction: NestedCorrection,
    native_correction: NestedCorrection,
) -> dict[str, object]:
    """Do the two independent nested corrections land on the same surface?

    The pre-state block (``residual_before_*``,
    ``shared_residual_definition_ok``) compares the two lanes' INCOMING
    residuals.  Both lanes evaluate that number on the same view before either
    inner solve starts, so it exists on every record and is published whatever
    the corrections do afterwards -- it is the check that the two lanes were
    handed the same problem, which is exactly the thing worth knowing when one
    of them then fails.  Only the post-state block is null for a failed solve,
    which has no post-state to compare.
    """
    jax_before = float(jax_correction.residual_norm_before)
    native_before = float(native_correction.residual_norm_before)
    before_absolute_gap = abs(jax_before - native_before)
    definition_bound = (
        SHARED_RESIDUAL_RTOL * max(abs(jax_before), abs(native_before))
        + SHARED_RESIDUAL_ATOL
    )
    post_state: dict[str, float | None] = dict.fromkeys(
        (
            "surface_dofs_l2",
            "surface_dofs_max",
            "iota_after_delta",
            "G_after_delta",
            "residual_after_jax_minus_native",
        )
    )
    if (
        correction_evaluation_status(jax_correction) != EVALUATION_FAILED_SOLVE
        and correction_evaluation_status(native_correction) != EVALUATION_FAILED_SOLVE
    ):
        jax_surface = np.asarray(jax_correction.surface_dofs_after, dtype=np.float64)
        native_surface = np.asarray(
            native_correction.surface_dofs_after, dtype=np.float64
        )
        difference = jax_surface - native_surface
        post_state = {
            "surface_dofs_l2": float(np.linalg.norm(difference)),
            "surface_dofs_max": float(np.max(np.abs(difference)))
            if difference.size
            else 0.0,
            "iota_after_delta": float(jax_correction.iota_after)
            - float(native_correction.iota_after),
            "G_after_delta": float(jax_correction.G_after)
            - float(native_correction.G_after),
            "residual_after_jax_minus_native": float(jax_correction.residual_norm_after)
            - float(native_correction.residual_norm_after),
        }
    return {
        "surface_dofs_l2": post_state["surface_dofs_l2"],
        "surface_dofs_max": post_state["surface_dofs_max"],
        "iota_after_delta": post_state["iota_after_delta"],
        "G_after_delta": post_state["G_after_delta"],
        "residual_before_relative_gap": _relative_difference(jax_before, native_before),
        "residual_before_absolute_gap": before_absolute_gap,
        # Both lanes must evaluate the SAME residual at the SAME view, so this
        # is a definition gate, not a physics band; the absolute floor is the
        # solver tolerance, below which two norms cannot be told apart.
        "shared_residual_definition_ok": bool(before_absolute_gap <= definition_bound),
        "residual_after_jax_minus_native": post_state[
            "residual_after_jax_minus_native"
        ],
    }


def _point_record(
    *,
    name: PointName,
    vector: NDArray[np.float64],
    problem: Flat675Problem,
    lanes: NestedLanes,
    objective: ObjectiveLane,
    twin_value: Callable[[NDArray[np.float64]], float],
) -> dict[str, object]:
    """Everything the report quotes about one point of the flat solve."""
    started = perf_counter()
    view = lanes.view_fn(problem, vector)
    base_objective = objective.value(vector)
    base_terms = objective.weighted_terms(vector)
    twin_objective = twin_value(vector)
    jax_correction = lanes.jax_fn(view)
    native_correction = lanes.native_fn(view)
    corrections = {
        "jax": _correction_record(
            correction=jax_correction,
            base_vector=vector,
            base_objective=base_objective,
            base_terms=base_terms,
            problem=problem,
            lanes=lanes,
            objective=objective,
            twin_value=twin_value,
        ),
        "native": _correction_record(
            correction=native_correction,
            base_vector=vector,
            base_objective=base_objective,
            base_terms=base_terms,
            problem=problem,
            lanes=lanes,
            objective=objective,
            twin_value=twin_value,
        ),
    }
    return {
        "name": str(name),
        "iota": float(view.iota),
        "G": float(view.G),
        "objective": base_objective,
        "weighted_terms": base_terms,
        "flat675_boozer_term": float(lanes.boozer_term_fn(problem, vector)),
        "native_twin_objective": twin_objective,
        "native_twin_relative_gap": _relative_gap(base_objective, twin_objective),
        "native_twin_within_objective_rtol": (
            _relative_gap(base_objective, twin_objective) <= OBJECTIVE_RTOL
        ),
        "corrections": corrections,
        "lane_agreement": _lane_agreement(
            jax_correction=jax_correction,
            native_correction=native_correction,
        ),
        "wall_s": perf_counter() - started,
    }


# --------------------------------------------------------------------------
# Runtime identity and the top-level comparison
# --------------------------------------------------------------------------


def runtime_identity(*, jax_platform: JaxPlatform) -> dict[str, object]:
    """Kernel sha, JAX identity, XLA flags and threading, for the payload."""
    identity = dict(nested_ls_runtime_identity())
    identity["jax_version"] = str(jax.__version__)
    identity["xla_flags"] = os.environ.get("XLA_FLAGS")
    identity["requested_jax_platform"] = str(jax_platform)
    identity["repo_root"] = str(REPO_ROOT)
    return identity


def require_platform(jax_platform: JaxPlatform) -> None:
    """Refuse a run whose backend or XLA flags are not what the row claims."""
    backend = str(jax.default_backend())
    if jax_platform == "gpu":
        if backend not in ("gpu", "cuda", "rocm"):
            raise SystemExit(
                f"--jax-platform gpu but jax.default_backend() is {backend!r}; "
                "set JAX_PLATFORMS=cuda before starting the process."
            )
        flags = os.environ.get("XLA_FLAGS", "")
        if REQUIRED_GPU_XLA_FLAG not in flags:
            raise SystemExit(
                f"a GPU row must carry {REQUIRED_GPU_XLA_FLAG} in XLA_FLAGS "
                "(fresh compiles differ at 3e-15 without it); "
                f"XLA_FLAGS is {flags!r}."
            )
        return
    if backend != "cpu":
        raise SystemExit(
            f"--jax-platform cpu but jax.default_backend() is {backend!r}; "
            "set JAX_PLATFORMS=cpu before starting the process."
        )


def compare(
    *,
    problem: Flat675Problem,
    objective: ObjectiveLane,
    twin_value: Callable[[NDArray[np.float64]], float],
    configuration: Configuration,
    jax_platform: JaxPlatform,
    max_steps: int,
    lanes: NestedLanes = PRODUCTION_LANES,
    solve_fn: Callable[..., SolveOutcome] = run_fused_solve,
) -> dict[str, object]:
    """Run the fused solve and measure both points; return the JSON payload."""
    started = perf_counter()
    solve = solve_fn(problem, max_steps=max_steps)
    vectors: dict[PointName, NDArray[np.float64]] = {
        "start": solve.start_vector,
        "endpoint": solve.endpoint_vector,
    }
    points = {
        name: _point_record(
            name=name,
            vector=vectors[name],
            problem=problem,
            lanes=lanes,
            objective=objective,
            twin_value=twin_value,
        )
        for name in POINT_NAMES
    }
    endpoint = points["endpoint"]
    endpoint_corrections = endpoint["corrections"]
    start_point = points["start"]
    branch_claims = [
        points[name]["corrections"][lane]["same_branch_as_incoming"]
        for name in POINT_NAMES
        for lane in LANE_NAMES
    ]
    every_correction_stayed_on_branch: bool | None
    if any(claim is None for claim in branch_claims):
        every_correction_stayed_on_branch = None
    else:
        every_correction_stayed_on_branch = all(bool(claim) for claim in branch_claims)
    shared_residual_definition_ok: bool | None
    if (
        endpoint["lane_agreement"]["shared_residual_definition_ok"] is None
        or start_point["lane_agreement"]["shared_residual_definition_ok"] is None
    ):
        shared_residual_definition_ok = None
    else:
        shared_residual_definition_ok = bool(
            endpoint["lane_agreement"]["shared_residual_definition_ok"]
            and start_point["lane_agreement"]["shared_residual_definition_ok"]
        )
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "question": (
            "At the endpoint of the flat coupled single-stage solve, is the "
            "surface a Boozer surface to the tolerance the nested route "
            "guarantees?"
        ),
        "configuration": str(configuration),
        "execution_scale": (
            "frozen-bundle" if configuration == "bundle" else REPOSITORY_EXECUTION_SCALE
        ),
        "max_steps": int(max_steps),
        "jax_platform": str(jax_platform),
        "outer_dof_count": int(FLAT675_OUTER_DOF_COUNT),
        "objective_term_keys": [str(key) for key in FLAT675_OBJECTIVE_TERM_KEYS],
        "nested_contract": {
            # The bar both lanes STOP at.
            "tolerance": float(NESTED_LS_BANANA_NEWTON_TOL),
            "tolerance_bar": NESTED_CORRECTION_TOLERANCE_BAR,
            "newton_maxiter": int(NESTED_LS_BANANA_NEWTON_MAXITER),
            # The bar nested_ls_outer_claim.py rejudges its certified rows at.
            # Neither lane here stops at it; every residual is published
            # against both so the two are never silently interchanged.
            "physics_tolerance": float(NESTED_LS_NEWTON_TOL),
            "physics_tolerance_bar": NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR,
            "physics_newton_maxiter": int(NESTED_LS_NEWTON_MAXITER),
            "physics_newton_stab": float(NESTED_LS_NEWTON_STAB),
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            # H8: this harness pins the native lane to one thread for
            # determinism (build_parity_lane_environment("native-cpu")), while
            # the certified native banana lane runs at the contract's
            # best-of-sweep thread count.  Both numbers are published so the
            # walls here are never read as that lane's.
            "certified_native_omp_threads": int(NESTED_LS_GATE6_NATIVE_OMP_THREADS),
            "residual_definition": (
                "the nested contract's residual -- what the reduced nested-LS "
                "Newton inner solve drives to NESTED_LS_BANANA_NEWTON_TOL -- "
                "evaluated on the layout surface's quadrature, identical for "
                "the jax and native lanes"
            ),
            "source": "src/simsopt_jax_adapters/geo/nested_ls_contract.py",
        },
        "objective_rtol": float(OBJECTIVE_RTOL),
        "shared_residual_rtol": float(SHARED_RESIDUAL_RTOL),
        "shared_residual_atol": float(SHARED_RESIDUAL_ATOL),
        "solve": {
            "driver": "SIMSOPT_LBFGSB",
            "lbfgs_history": int(FLAT675_LBFGS_HISTORY),
            "lbfgs_max_line_search_steps": int(FLAT675_LBFGS_MAXLS),
            "rtol": float(SOLVE_RTOL),
            "atol": float(SOLVE_ATOL),
            "iterations": solve.iterations,
            "objective_evaluations": solve.objective_evaluations,
            "final_objective": solve.final_objective,
            "success": solve.success,
            "endpoint_is_optimizer_x": solve.endpoint_is_optimizer_x,
            "host_transfer_ledger": solve.transfer_ledger,
            "fused_lane_host_transfers_zero": bool(
                solve.transfer_ledger.get("advance", 0) == 0
                and solve.transfer_ledger.get("callback", 0) == 0
                and solve.transfer_ledger.get("unclassified", 0) == 0
            ),
            "wall_s": solve.wall_s,
        },
        "points": points,
        "verdict": {
            # The headline: is the endpoint already a Boozer surface?  The
            # no-op gate is the PHYSICS bar (nested_ls_outer_claim.py's), the
            # ratios are published against both, and every key names its bar.
            "noop_bar": NESTED_CORRECTION_PHYSICS_TOLERANCE_BAR,
            "endpoint_jax_correction_is_noop_at_physics_bar": bool(
                endpoint_corrections["jax"]["nested_correction_is_noop"]
            ),
            "endpoint_native_correction_is_noop_at_physics_bar": bool(
                endpoint_corrections["native"]["nested_correction_is_noop"]
            ),
            # ``None`` where the residual was not finite: see ``_bar_ratio``.
            "endpoint_residual_over_timing_bar_jax": endpoint_corrections["jax"][
                "residual_before_over_timing_bar"
            ],
            "endpoint_residual_over_timing_bar_native": endpoint_corrections["native"][
                "residual_before_over_timing_bar"
            ],
            "endpoint_residual_over_physics_bar_jax": endpoint_corrections["jax"][
                "residual_before_over_physics_bar"
            ],
            "endpoint_residual_over_physics_bar_native": endpoint_corrections["native"][
                "residual_before_over_physics_bar"
            ],
            # The contract's branch guard, per lane and per point.  A lane
            # that left the branch produced a rejected evaluation, not a
            # correction, and none of its displacement numbers describe a
            # correction of THIS point.
            "iota_branch_guard": float(NESTED_LS_OUTER_IOTA_BRANCH_GUARD),
            "endpoint_jax_correction_stayed_on_branch": endpoint_corrections["jax"][
                "same_branch_as_incoming"
            ],
            "endpoint_native_correction_stayed_on_branch": endpoint_corrections[
                "native"
            ]["same_branch_as_incoming"],
            "start_jax_correction_stayed_on_branch": start_point["corrections"]["jax"][
                "same_branch_as_incoming"
            ],
            "start_native_correction_stayed_on_branch": start_point["corrections"][
                "native"
            ]["same_branch_as_incoming"],
            "every_correction_stayed_on_branch": every_correction_stayed_on_branch,
            "shared_residual_definition_ok": shared_residual_definition_ok,
            # Control: if the twin disagrees at the START point the twin is a
            # different problem, and every twin number below is uninterpretable.
            "native_twin_control_ok": bool(
                start_point["native_twin_within_objective_rtol"]
            ),
            # nested_ls_outer_claim.py's J-parity gate is one-sided and refuses
            # a non-positive denominator ("native_endpoint_j_invalid"); the same
            # guard is carried here so a sign flip cannot mint a verdict.
            "native_twin_objective_valid_for_band": bool(
                math.isfinite(float(endpoint["native_twin_objective"]))
                and float(endpoint["native_twin_objective"]) > 0.0
            ),
            "endpoint_j_parity_within_band": bool(
                math.isfinite(float(endpoint["native_twin_objective"]))
                and float(endpoint["native_twin_objective"]) > 0.0
                and float(endpoint["objective"])
                <= float(endpoint["native_twin_objective"]) * (1.0 + OBJECTIVE_RTOL)
            ),
        },
        "walls": {
            "solve_s": solve.wall_s,
            "start_point_s": float(start_point["wall_s"]),
            "endpoint_point_s": float(endpoint["wall_s"]),
            "total_s": perf_counter() - started,
        },
        "runtime": runtime_identity(jax_platform=jax_platform),
    }
    return payload


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _number(value: object) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = float(value)
    if number == 0.0:
        return "0"
    if abs(number) >= 1.0e-4 and abs(number) < 1.0e6:
        return f"{number:.12g}"
    return f"{number:.6e}"


def _omp_cell(value: object) -> str:
    """The lane's ``OMP_NUM_THREADS`` cell.

    ``None`` means the variable was not set in the process that ran that
    lane's solve, and ``str(None)`` renders that as the string ``"None"``,
    which reads on the page as a value rather than as an absence.
    """
    return "unpinned" if value is None else str(value)


def _optional(value: object) -> str:
    """A cell for a quantity one lane does not have.

    ``n/a`` is not zero and not a small number: it is the statement that this
    lane cannot publish this quantity at all, which is the whole point of the
    reduced-gradient column.
    """
    return "n/a" if value is None else _number(value)


def _difference_or_none(left: object, right: object) -> float | None:
    """Difference only where both post-state values exist."""
    if left is None or right is None:
        return None
    return float(left) - float(right)


def render_markdown(payload: Mapping[str, object]) -> str:
    """One Markdown section per run: the numbers, with no adjectives."""
    points = payload["points"]
    nested = payload["nested_contract"]
    solve = payload["solve"]
    verdict = payload["verdict"]
    walls = payload["walls"]
    term_keys = [str(key) for key in payload["objective_term_keys"]]

    lines: list[str] = []
    lines.append(
        f"### {payload['configuration']} / max-steps {payload['max_steps']} / "
        f"{payload['jax_platform']}"
    )
    lines.append("")
    lines.append(
        f"Timing bar {_number(nested['tolerance'])} "
        f"({nested['tolerance_bar']}, NESTED_LS_BANANA_NEWTON_TOL, maxiter "
        f"{nested['newton_maxiter']}) -- both lanes stop here. Physics bar "
        f"{_number(nested['physics_tolerance'])} "
        f"({nested['physics_tolerance_bar']}, NESTED_LS_NEWTON_TOL, maxiter "
        f"{nested['physics_newton_maxiter']}) -- the bar "
        f"`nested_ls_outer_claim.py` rejudges at, and the bar the no-op gate "
        f"below uses. iota branch guard "
        f"{_number(nested['iota_branch_guard'])}; objective rtol "
        f"{_number(payload['objective_rtol'])}."
    )
    lines.append("")

    lines.append("#### Nested correction per lane, per point")
    lines.append("")
    header = (
        "point",
        "lane",
        "residual before",
        "residual after",
        "before / timing bar",
        "before / physics bar",
        "‖Δdof‖₂",
        "‖Δdof‖∞",
        "max ‖Δx‖ (m)",
        "rms ‖Δx‖ (m)",
        "max ‖Δx‖ / a",
        "Δiota",
        "ΔG",
        "wall (s)",
    )
    lines.append(_row(header))
    lines.append(_row(["---"] * len(header)))
    for name in POINT_NAMES:
        point = points[name]
        for lane in LANE_NAMES:
            record = point["corrections"][lane]
            lines.append(
                _row(
                    (
                        str(name),
                        lane,
                        _number(record["residual_norm_before"]),
                        _optional(record["residual_norm_after"]),
                        _optional(record["residual_before_over_timing_bar"]),
                        _optional(record["residual_before_over_physics_bar"]),
                        _optional(record["dof_displacement_l2"]),
                        _optional(record["dof_displacement_max"]),
                        _optional(record["point_displacement_max_m"]),
                        _optional(record["point_displacement_rms_m"]),
                        _optional(record["point_displacement_max_over_minor_radius"]),
                        _optional(record["iota_delta"]),
                        _optional(_difference_or_none(record["G_after"], point["G"])),
                        _number(record["wall_s"]),
                    )
                )
            )
    lines.append("")

    lines.append("#### Inner-solver provenance, exit and the branch guard")
    lines.append("")
    provenance_header = (
        "point",
        "lane",
        "inner policy",
        "bfgs it",
        "newton it",
        "exit status",
        "exit classified on",
        "persisted",
        "converged (timing bar)",
        "reduced ‖g‖₂",
        "coil Δ∞",
        "Δiota",
        "same branch",
        "evaluation",
        "no-op (physics bar)",
        "OMP_NUM_THREADS",
    )
    lines.append(_row(provenance_header))
    lines.append(_row(["---"] * len(provenance_header)))
    for name in POINT_NAMES:
        for lane in LANE_NAMES:
            record = points[name]["corrections"][lane]
            bfgs = record["bfgs_iterations"]
            lines.append(
                _row(
                    (
                        str(name),
                        lane,
                        str(record["inner_policy"]),
                        "n/a" if bfgs is None else str(bfgs),
                        str(record["newton_iterations"]),
                        str(record["exit_status"]),
                        str(record["exit_status_quantity"]),
                        _number(record["persisted"]),
                        _number(record["converged"]),
                        _optional(record["reduced_gradient_l2"]),
                        _optional(record["coil_delta_inf"]),
                        _optional(record["iota_delta"]),
                        _optional(record["same_branch_as_incoming"]),
                        str(record["evaluation_status"]),
                        _number(record["nested_correction_is_noop"]),
                        _omp_cell(record["omp_num_threads"]),
                    )
                )
            )
    lines.append("")

    lines.append("#### Eight-term objective at the four points")
    lines.append("")
    endpoint = points["endpoint"]
    columns = (
        ("start", points["start"]["weighted_terms"], points["start"]["objective"]),
        ("endpoint", endpoint["weighted_terms"], endpoint["objective"]),
        (
            "endpoint + jax",
            endpoint["corrections"]["jax"]["weighted_terms"],
            endpoint["corrections"]["jax"]["objective"],
        ),
        (
            "endpoint + native",
            endpoint["corrections"]["native"]["weighted_terms"],
            endpoint["corrections"]["native"]["objective"],
        ),
    )
    term_header = (
        "term",
        *(name for name, _terms, _total in columns),
        "Δ jax−endpoint",
        "Δ native−endpoint",
    )
    lines.append(_row(term_header))
    lines.append(_row(["---"] * len(term_header)))
    for key in term_keys:
        endpoint_term = float(endpoint["weighted_terms"][key])
        lines.append(
            _row(
                (
                    key,
                    *(
                        _optional(None if terms is None else terms[key])
                        for _name, terms, _total in columns
                    ),
                    _optional(
                        _difference_or_none(
                            None
                            if endpoint["corrections"]["jax"]["weighted_terms"] is None
                            else endpoint["corrections"]["jax"]["weighted_terms"][key],
                            endpoint_term,
                        )
                    ),
                    _optional(
                        _difference_or_none(
                            None
                            if endpoint["corrections"]["native"]["weighted_terms"]
                            is None
                            else endpoint["corrections"]["native"]["weighted_terms"][
                                key
                            ],
                            endpoint_term,
                        )
                    ),
                )
            )
        )
    endpoint_total = float(endpoint["objective"])
    lines.append(
        _row(
            (
                "**J (sum)**",
                *(_optional(total) for _name, _terms, total in columns),
                _optional(
                    _difference_or_none(
                        endpoint["corrections"]["jax"]["objective"], endpoint_total
                    )
                ),
                _optional(
                    _difference_or_none(
                        endpoint["corrections"]["native"]["objective"], endpoint_total
                    )
                ),
            )
        )
    )
    lines.append(
        _row(
            (
                "native twin J",
                _number(points["start"]["native_twin_objective"]),
                _number(endpoint["native_twin_objective"]),
                _optional(endpoint["corrections"]["jax"]["native_twin_objective"]),
                _optional(endpoint["corrections"]["native"]["native_twin_objective"]),
                "",
                "",
            )
        )
    )
    lines.append(
        _row(
            (
                "native twin rel. gap",
                _number(points["start"]["native_twin_relative_gap"]),
                _number(endpoint["native_twin_relative_gap"]),
                _optional(endpoint["corrections"]["jax"]["native_twin_relative_gap"]),
                _optional(
                    endpoint["corrections"]["native"]["native_twin_relative_gap"]
                ),
                "",
                "",
            )
        )
    )
    lines.append("")

    lines.append("#### iota, G and lane agreement")
    lines.append("")
    ig_header = (
        "point",
        "iota (y-solve)",
        "G (y-solve)",
        "iota jax lane",
        "iota native lane",
        "G jax lane",
        "G native lane",
        "‖Δsurface dof‖₂ jax−native",
        "‖Δsurface dof‖∞ jax−native",
        "shared residual definition",
    )
    lines.append(_row(ig_header))
    lines.append(_row(["---"] * len(ig_header)))
    for name in POINT_NAMES:
        point = points[name]
        agreement = point["lane_agreement"]
        lines.append(
            _row(
                (
                    str(name),
                    _number(point["iota"]),
                    _number(point["G"]),
                    _optional(point["corrections"]["jax"]["iota_after"]),
                    _optional(point["corrections"]["native"]["iota_after"]),
                    _optional(point["corrections"]["jax"]["G_after"]),
                    _optional(point["corrections"]["native"]["G_after"]),
                    _optional(agreement["surface_dofs_l2"]),
                    _optional(agreement["surface_dofs_max"]),
                    _optional(agreement["shared_residual_definition_ok"]),
                )
            )
        )
    lines.append("")

    lines.append("#### Solve, gates and wall clocks")
    lines.append("")
    lines.append(_row(("quantity", "value")))
    lines.append(_row(("---", "---")))
    for label, value in (
        ("outer iterations run", str(solve["iterations"])),
        ("objective evaluations", str(solve["objective_evaluations"])),
        ("solve success", _number(solve["success"])),
        ("endpoint is optimizer x", _number(solve["endpoint_is_optimizer_x"])),
        (
            "fused-lane host transfers zero",
            _number(solve["fused_lane_host_transfers_zero"]),
        ),
        (
            f"endpoint jax correction is a no-op ({verdict['noop_bar']})",
            _number(verdict["endpoint_jax_correction_is_noop_at_physics_bar"]),
        ),
        (
            f"endpoint native correction is a no-op ({verdict['noop_bar']})",
            _number(verdict["endpoint_native_correction_is_noop_at_physics_bar"]),
        ),
        (
            "endpoint residual / timing bar (jax)",
            _optional(verdict["endpoint_residual_over_timing_bar_jax"]),
        ),
        (
            "endpoint residual / timing bar (native)",
            _optional(verdict["endpoint_residual_over_timing_bar_native"]),
        ),
        (
            "endpoint residual / physics bar (jax)",
            _optional(verdict["endpoint_residual_over_physics_bar_jax"]),
        ),
        (
            "endpoint residual / physics bar (native)",
            _optional(verdict["endpoint_residual_over_physics_bar_native"]),
        ),
        (
            "endpoint jax correction stayed on the incoming branch",
            _optional(verdict["endpoint_jax_correction_stayed_on_branch"]),
        ),
        (
            "endpoint native correction stayed on the incoming branch",
            _optional(verdict["endpoint_native_correction_stayed_on_branch"]),
        ),
        (
            "every correction stayed on the incoming branch",
            _optional(verdict["every_correction_stayed_on_branch"]),
        ),
        (
            "shared residual definition ok",
            _optional(verdict["shared_residual_definition_ok"]),
        ),
        (
            "native twin control ok (start point)",
            _number(verdict["native_twin_control_ok"]),
        ),
        (
            "native twin J is positive (band is meaningful)",
            _number(verdict["native_twin_objective_valid_for_band"]),
        ),
        (
            "endpoint J within the twin band",
            _number(verdict["endpoint_j_parity_within_band"]),
        ),
        ("solve wall (s)", _number(walls["solve_s"])),
        ("start-point measurement wall (s)", _number(walls["start_point_s"])),
        ("endpoint measurement wall (s)", _number(walls["endpoint_point_s"])),
        ("total wall (s)", _number(walls["total_s"])),
    ):
        lines.append(_row((label, value)))
    lines.append("")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--configuration", required=True, choices=("bundle", "repository-geometry")
    )
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--jax-platform", required=True, choices=("cpu", "gpu"))
    return parser.parse_args(argv)


def non_correction_evaluations(payload: Mapping[str, object]) -> tuple[str, ...]:
    """Every ``point/lane=status`` the contract does not accept as a correction.

    Both non-correction statuses land here, not ``failed_solve`` alone: the
    branch guard makes a converged inner solve on a DIFFERENT Boozer branch a
    failed EVALUATION, so a run carrying one must not exit 0 and read as a
    clean comparison.  Empty means every lane published a correction.
    """
    return tuple(
        f"{name}/{lane}="
        f"{payload['points'][name]['corrections'][lane]['evaluation_status']}"
        for name in POINT_NAMES
        for lane in LANE_NAMES
        if payload["points"][name]["corrections"][lane]["evaluation_status"]
        != EVALUATION_CORRECTION
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_steps < 1:
        raise SystemExit(f"--max-steps must be at least 1; got {args.max_steps}")
    require_platform(args.jax_platform)
    problem = build_problem(args.configuration)
    payload = compare(
        problem=problem,
        objective=jax_objective_lane(problem),
        twin_value=native_twin_value(configuration=args.configuration),
        configuration=args.configuration,
        jax_platform=args.jax_platform,
        max_steps=int(args.max_steps),
    )
    markdown = render_markdown(payload)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(dump_strict_json(payload))
    out_json.with_suffix(".md").write_text(markdown)
    print(markdown, flush=True)
    print(
        json.dumps(
            {
                "out_json": str(out_json),
                "out_markdown": str(out_json.with_suffix(".md")),
                "verdict": payload["verdict"],
                "walls": payload["walls"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    non_corrections = non_correction_evaluations(payload)
    if non_corrections:
        print(
            "nested endpoint correction not accepted: " + ", ".join(non_corrections),
            file=sys.stderr,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
