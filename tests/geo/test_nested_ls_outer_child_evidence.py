"""Evidence contracts at the nested-LS outer children's own surface.

Tier-0 hardening of ``docs/nested_ls_upgrade_implementation_plan.md``:

* every emitted per-evaluation row declares whether its value is the
  eight-term outer objective or the containment barrier
  (``value_is_valid``), identically on both lanes, and a rejected row
  names no solved surface of its own;
* every emitted restart-attempt row carries the same declaration about
  scipy's ``result.fun``, decided by provenance through the shared
  contract rule: ``result.fun`` is the last evaluation's value, so a
  wholly rejected line search leaves a containment barrier there while
  ``result.x`` is restored to the incumbent;
* an accepted-step callback naming a point the candidate store never
  staged publishes the run's accumulated evidence — including counters
  derived from that evidence rather than zeroed — and *then* exits
  nonzero, instead of dying inside scipy with the ledger unwritten;
* both lanes publish a ``runtime`` block, so each binds the compiled
  extension its own numbers came out of;
* the rejudge gate refuses a fault receipt, which is otherwise a
  schema-valid endpoint document.

Both children reach physics only through a run context, so these tests
drive the real optimizer loop, the real transaction and the real payload
builder with a fake inner solve. That is the fake-collaborator idiom
``tests/geo/test_nested_ls_outer_transaction.py`` already uses to drive
``NativeOuterRun`` without a real Boozer solve; the JAX lane is driven the
same way rather than through a second idiom. The stand-ins themselves live
in ``tests/geo/_nested_ls_outer_fakes.py`` so the two files cannot drift.

Only the loaded world is injected. Each child's own decisions — the typed
inner-solve refusals, the receipt hash, the fault name, the process
identity — are module attributes the production path reads directly, and a
test that needs a different one patches that attribute. A test that passed
those in as context fields would be asserting the value it injected itself.

The fake objective is ``J(c) = c.c``, which is what lets these tests check
the bit against something other than the code that sets it: a row claiming
``value_is_valid=True`` must carry ``c.c`` at that row's own coils, and a
row claiming ``False`` must carry the shared contract barrier priced off
the committed anchor. Row values are exactly reproducible float64, so they
are compared exactly: a relative band would admit a barrier priced off a
different but nearby anchor, which is what the bit exists to catch.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys

# CPU by construction, pinned BEFORE the child import below. The JAX child
# ``setdefault``s ``JAX_PLATFORMS=cuda,cpu`` and ``SIMSOPT_BACKEND_MODE=
# jax_gpu_fast`` at import, and every jax-lane test here reaches
# ``_cache_meta`` -> ``jax.default_backend()``, which is first backend use:
# under those defaults it would initialize CUDA and preallocate device memory
# inside the pytest process. Nothing in this file needs a device, and a
# developer running ``pytest tests/geo/…`` during a certification run must not
# contend for the GPU, so the file refuses one rather than trusting the caller
# to have exported the right environment.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["SIMSOPT_BACKEND_MODE"] = "jax_cpu_fast"
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from benchmarks import nested_ls_outer_jax_child as jax_child
from benchmarks import nested_ls_outer_native_child as native_child
from numpy.typing import NDArray
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    nested_ls_outer_rejection_barrier,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    NestedLsInnerSolveFailed,
    nested_ls_runtime_identity,
)

from ._nested_ls_outer_fakes import (
    _FakeBoozer,
    _FakeJaxBoozer,
    _FakeJaxOuterState,
    _FakeObjective,
    objective_at,
)

REPO = Path(__file__).resolve().parents[2]

# One transcript vocabulary, replayed on both lanes. ``START`` is x0 and
# primes the transaction, ``FEASIBLE_TRIAL`` is a second solvable point that
# stays pending until accepted, ``FAILING_TRIAL`` is the point whose inner
# solve refuses and therefore takes the containment barrier, and ``UNSTAGED``
# is the point scipy announces in the accept-without-candidate corner.
START = (0.25, -0.5)
FEASIBLE_TRIAL = (0.5, -0.25)
FAILING_TRIAL = (9.0, 9.0)
UNSTAGED = (1.5, 1.5)
# A refused trial 1e-9 from ``FEASIBLE_TRIAL``: close enough that the barrier
# priced off it rounds to bitwise ``J(FEASIBLE_TRIAL)``. dcsrch contracts
# toward exactly this regime, which is why a value comparison cannot decide
# whether scipy's ``fun`` is an objective.
NEAR_ANCHOR_TRIAL = (FEASIBLE_TRIAL[0] + 1.0e-9, FEASIBLE_TRIAL[1])

REJECTION_DISTANCE_SCALE = 1.0
BUDGET = 3
SEED_IOTA = 0.14
SEED_G = 2.0
LANES = ("jax", "native")

# The two real scipy 1.17.1 stop lines this file replays. ``ABNORMAL`` is the
# one the shared classifier restarts.
MAXITER_STOP = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
ABNORMAL_STOP = "ABNORMAL: LINE SEARCH FAILED"

# The measured wholly-rejected pair, reproduced from the shipped fixture by
# ``test_wholly_rejected_attempt_does_not_publish_its_barrier_as_an_objective``
# and quoted by both children. ``START`` gives an anchor objective of exactly
# 0.3125, and real scipy's final rejected trial prices this barrier off it.
ANCHOR_OBJECTIVE = 0.3125
WHOLLY_REJECTED_BARRIER = 0.31250004043721513

FAKE_RUNTIME_IDENTITY: dict[str, object] = {
    "hostname": "fake-host",
    "simsoptpp_sha256": "fake-extension-sha",
    "timestamp_utc": "fake-write-time",
}


def barrier_at(coil_dofs: object, *, anchor: object) -> float:
    """The containment barrier the shared contract prices at one trial."""

    value, _gradient = nested_ls_outer_rejection_barrier(
        anchor_value=objective_at(anchor),
        anchor_parameters=np.asarray(anchor, dtype=np.float64),
        trial_parameters=np.asarray(coil_dofs, dtype=np.float64),
        scale=REJECTION_DISTANCE_SCALE,
    )
    return float(value)


def _fake_runtime_identity() -> dict[str, object]:
    """Stand in for the real identity, which hashes the loaded extension."""

    return dict(FAKE_RUNTIME_IDENTITY)


@dataclass(frozen=True, slots=True)
class _FakeOptimizeResult:
    x: NDArray[np.float64]
    fun: float
    jac: NDArray[np.float64]
    nit: int
    nfev: int
    njev: int
    success: bool
    status: int
    message: str


@dataclass(frozen=True, slots=True)
class _Step:
    """One thing L-BFGS-B does at one point before it returns.

    Either it evaluates the objective there, or it announces the point to the
    accepted-step callback. Keeping the two in one ordered script is what lets
    a transcript put a rejected trial *after* an accepted step, which is the
    order that decides which anchor prices the barrier.
    """

    point: tuple[float, ...]
    is_accept: bool


def evaluate(point: tuple[float, ...]) -> _Step:
    """A line-search evaluation at ``point``."""

    return _Step(point, False)


def accept(point: tuple[float, ...]) -> _Step:
    """An accepted-step callback announcing ``point``."""

    return _Step(point, True)


@dataclass(frozen=True, slots=True)
class _Attempt:
    """One scipy attempt a transcript replays.

    ``steps`` are what happens *after* ``x0``; real L-BFGS-B always evaluates
    the start point first, and the stand-in takes that point from the ``x0``
    it is handed rather than from a constant, so a restart that hands back the
    committed incumbent is visible in the ledger. ``status`` and ``message``
    are the scipy verdict.
    """

    steps: tuple[_Step, ...]
    status: int
    message: str


class _FakeMinimize:
    """A scipy stand-in replaying one exact per-attempt transcript.

    It honours the two things the child controls: the ``x0`` it is handed
    (evaluated first, as scipy does) and ``options["maxiter"]`` (a transcript
    announcing more accepted iterates than the budget allows is refused, not
    silently replayed). Both are recorded in :attr:`calls` so a test can
    assert the restart arithmetic instead of assuming it.

    ``fun`` and ``jac`` come from the LAST evaluation while ``x`` is the last
    accepted iterate — which is exactly why a wholly rejected line search
    leaves a containment barrier in ``fun`` beside a restored ``x``. Nothing
    here interprets the numbers: the transcript is the input and the child's
    ledger and payload are what the tests read. When an announced point was
    never staged the callback raises and this stand-in never returns, which is
    the accept-without-candidate corner.
    """

    def __init__(self, *attempts: _Attempt) -> None:
        self.attempts = attempts
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        fun: object,
        x0: object,
        *,
        jac: bool,
        method: str,
        options: dict[str, object],
        callback: object,
    ) -> _FakeOptimizeResult:
        attempt = self.attempts[len(self.calls)]
        maxiter = int(options["maxiter"])
        start = np.array(x0, dtype=np.float64)
        self.calls.append({"x0": start, "maxiter": maxiter})
        iterate_steps = sum(1 for step in attempt.steps if step.is_accept)
        if iterate_steps > maxiter:
            raise AssertionError(
                f"transcript attempt {len(self.calls)} announces {iterate_steps} "
                f"accepted iterates, which the child's own maxiter={maxiter} "
                "does not allow"
            )
        value, gradient = fun(start)
        evaluations = 1
        endpoint = start
        for step in attempt.steps:
            point = np.array(step.point, dtype=np.float64)
            if step.is_accept:
                callback(point)
                endpoint = point
            else:
                value, gradient = fun(point)
                evaluations += 1
        return _FakeOptimizeResult(
            x=endpoint,
            fun=float(value),
            jac=np.asarray(gradient, dtype=np.float64),
            nit=iterate_steps,
            nfev=evaluations,
            njev=evaluations,
            success=attempt.status == 0,
            status=attempt.status,
            message=attempt.message,
        )


def completing_transcript() -> _FakeMinimize:
    """One attempt that evaluates a refused trial and then accepts a step.

    The refusal comes first, so its barrier is priced off the start point,
    which is still the committed anchor when it happens.
    """

    return _FakeMinimize(
        _Attempt(
            steps=(
                evaluate(FEASIBLE_TRIAL),
                evaluate(FAILING_TRIAL),
                accept(FEASIBLE_TRIAL),
            ),
            status=1,
            message=MAXITER_STOP,
        )
    )


def barrier_rounds_to_the_anchor_transcript() -> _FakeMinimize:
    """A step is accepted, then a trial so close to it that the barrier hides.

    ``NEAR_ANCHOR_TRIAL`` sits ``1e-9`` from the accepted incumbent, so
    ``J_a + 0.5*mu*||d||^2`` rounds to bitwise ``J_a``: scipy's surviving
    ``fun`` is numerically the incumbent's objective while being a barrier.
    """

    return _FakeMinimize(
        _Attempt(
            steps=(
                evaluate(FEASIBLE_TRIAL),
                accept(FEASIBLE_TRIAL),
                evaluate(NEAR_ANCHOR_TRIAL),
            ),
            status=1,
            message=MAXITER_STOP,
        )
    )


def fault_on_first_attempt_transcript() -> _FakeMinimize:
    """The accept-without-candidate corner on the very first attempt."""

    return _FakeMinimize(
        _Attempt(
            steps=(evaluate(FAILING_TRIAL), accept(UNSTAGED)),
            status=1,
            message=MAXITER_STOP,
        )
    )


def fault_on_second_attempt_transcript() -> _FakeMinimize:
    """A restart, then the accept-without-candidate corner on attempt two.

    Attempt one accepts one iterate and stops ``ABNORMAL``, which the shared
    classifier restarts. Attempt two is handed the committed incumbent and the
    remaining budget, and faults before completing any iterate of its own.
    """

    return _FakeMinimize(
        _Attempt(
            steps=(evaluate(FEASIBLE_TRIAL), accept(FEASIBLE_TRIAL)),
            status=2,
            message=ABNORMAL_STOP,
        ),
        _Attempt(
            steps=(evaluate(FAILING_TRIAL), accept(UNSTAGED)),
            status=1,
            message=MAXITER_STOP,
        ),
    )


def _outer_policy() -> native_child.OuterOptimizerPolicy:
    """The real sealed-policy dataclass, with this transcript's knobs."""

    return native_child.OuterOptimizerPolicy(
        source="fake-lane",
        method="L-BFGS-B",
        ftol=0.0,
        gtol=0.0,
        maxls=8,
        maxiter=BUDGET,
        maxcor=3,
        rejection_distance_scale=REJECTION_DISTANCE_SCALE,
        lane_policy_names=dict(native_child.IMPLEMENTED_LANE_POLICY_NAMES),
    )


def refuses_only_the_failing_trial(coil_dofs: NDArray[np.float64]) -> bool:
    """The ordinary case: one trial point the inner solve cannot reach."""

    return bool(np.array_equal(coil_dofs, np.asarray(FAILING_TRIAL, np.float64)))


def refuses_every_point_but_the_start(coil_dofs: NDArray[np.float64]) -> bool:
    """The wholly rejected line search: only the start point ever solves."""

    return not bool(np.array_equal(coil_dofs, np.asarray(START, np.float64)))


def refuses_the_near_anchor_trial(coil_dofs: NDArray[np.float64]) -> bool:
    """Only the trial that sits 1e-9 from the accepted incumbent is refused."""

    return bool(np.array_equal(coil_dofs, np.asarray(NEAR_ANCHOR_TRIAL, np.float64)))


def refuses_nothing(coil_dofs: NDArray[np.float64]) -> bool:
    """Every point solves, so the optimizer runs to its own stop."""

    return False


def _jax_inner_solve(inner_refuses: object) -> object:
    """One JAX-lane inner solve that refuses whatever ``inner_refuses`` names.

    It raises the real ``NestedLsInnerSolveFailed``, not a stand-in: which
    exception types the lane answers with the sealed sentinel is the child's
    own decision, so the test exercises the production ``except`` clause.
    """

    def solve(
        state: _FakeJaxOuterState,
        coil_dofs: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64]]:
        coils = np.asarray(coil_dofs, dtype=np.float64)
        if inner_refuses(coils):
            raise NestedLsInnerSolveFailed(iteration_count=40, grad_l2=1.0, tol=1.0e-11)
        state.set_anchor(np.array([float(np.sum(coils))]), SEED_IOTA, SEED_G)
        state.inner_iterations = 3
        state.inner_grad_l2 = 1.0e-14
        state.adjoint_live_eta = 2.0e-14
        return objective_at(coils), 2.0 * coils

    return solve


def _native_inner_solve(inner_refuses: object) -> object:
    """One native-lane nested solve refusing whatever ``inner_refuses`` names."""

    def solve(boozer: _FakeBoozer, *, iota: float, G: float) -> dict[str, object]:
        coils = np.asarray(boozer.biotsavart.x, dtype=np.float64)
        boozer.surface.set_dofs(np.array([float(np.sum(coils))]))
        return {
            "success": not inner_refuses(coils),
            "bfgs_iter": 1,
            "newton_iter": 1,
            "bfgs_seconds": 0.0,
            "newton_seconds": 0.0,
            "seconds": 0.0,
            "coil_delta_inf": 0.0,
            "iota": iota,
            "G": G,
        }

    return solve


def _jax_context(inner_refuses: object) -> jax_child._OuterRunContext:
    """A JAX run context whose heavy world is the fake inner solve."""

    start_coils = np.asarray(START, dtype=np.float64)
    start_surface = np.array([float(np.sum(start_coils))], dtype=np.float64)
    state = _FakeJaxOuterState()
    state.set_anchor(start_surface, SEED_IOTA, SEED_G)
    return jax_child._OuterRunContext(
        state=state,
        jax_boozer=_FakeJaxBoozer(start_coils, start_surface),
        outer_policy=_outer_policy(),
        start_coils=start_coils,
        start_surface=start_surface,
        start_iota=SEED_IOTA,
        start_g=SEED_G,
        lane_meta={"lane": "fake"},
        import_init_seconds=0.0,
        problem_load_seconds=0.0,
        prepare_seconds=0.0,
        inner_value_and_grad=_jax_inner_solve(inner_refuses),
    )


def _native_context(
    set_module_attr: object,
    inner_refuses: object,
) -> native_child.NativeOuterRunContext:
    """A native run context whose nested solve is the fake inner solve.

    ``set_module_attr`` installs the fake on the child module: pytest's
    ``monkeypatch.setattr`` under a test, plain ``setattr`` in the
    subprocess driver that has no fixtures.
    """

    set_module_attr(
        native_child,
        "_run_native_banana_bfgs_then_newton",
        _native_inner_solve(inner_refuses),
    )
    start_coils = np.asarray(START, dtype=np.float64)
    start_surface = np.array([float(np.sum(start_coils))], dtype=np.float64)
    boozer = _FakeBoozer(start_coils, start_surface)
    run = native_child.NativeOuterRun(
        _FakeObjective(boozer),
        seed=native_child.InnerWarmStart(
            surface_dofs=start_surface,
            iota=SEED_IOTA,
            G=SEED_G,
        ),
        rejection_distance_scale=REJECTION_DISTANCE_SCALE,
        start_coil_dofs=start_coils,
    )
    return native_child.NativeOuterRunContext(
        run=run,
        outer_policy=_outer_policy(),
        objective_weights=tuple(
            1.0 for _key in native_child.FLAT675_OBJECTIVE_TERM_KEYS
        ),
        vessel_dofs=np.array([0.0], dtype=np.float64),
        lane_meta={"lane": "fake"},
        seed_iota=SEED_IOTA,
        seed_G=SEED_G,
        start_coil_dofs=start_coils,
        threading={"OMP_NUM_THREADS": "4", "OMP_PROC_BIND": None, "OMP_PLACES": None},
        module_import_seconds=0.0,
        build_seconds=0.0,
        child_started=0.0,
    )


def drive_lane(
    lane: str,
    set_module_attr: object,
    *,
    transcript: _FakeMinimize | None = None,
    inner_refuses: object = refuses_only_the_failing_trial,
) -> dict[str, object]:
    """Drive one lane's real optimizer loop and return that child's payload.

    With ``transcript`` the scipy stand-in replays one exact per-attempt
    script; without it the child runs against the real installed scipy, which
    is the only way to observe what scipy itself leaves in ``result.fun``.

    The process identity is substituted on the module attribute the payload
    builder reads, because the real one hashes the loaded native extension.
    """

    module = jax_child if lane == "jax" else native_child
    set_module_attr(module, "nested_ls_runtime_identity", _fake_runtime_identity)
    if transcript is not None:
        set_module_attr(module, "minimize", transcript)
    if lane == "jax":
        return jax_child._drive_outer_run(_jax_context(inner_refuses))
    return native_child._drive_native_run(
        _native_context(set_module_attr, inner_refuses)
    )


def ledger_rows(lane: str, payload: dict[str, object]) -> list[dict[str, object]]:
    """The emitted per-evaluation rows of either lane, in one shape.

    Only the keys the two lanes must agree on are normalized; the values
    come straight out of the payload the child publishes.
    """

    if lane == "jax":
        return [
            {
                "value": row["j"],
                "coil_dofs": row["coil_dofs"],
                "value_is_valid": row["value_is_valid"],
                "inner_feasible": row["inner_feasible"],
                "rejection_reason": row["rejection_reason"],
            }
            for row in payload["outer_evals"]
        ]
    return [
        {
            "value": row["objective"],
            "coil_dofs": row["coil_dofs"],
            "value_is_valid": row["value_is_valid"],
            "inner_feasible": row["inner_feasible"],
            "rejection_reason": row["rejection_reason"],
        }
        for row in payload["evaluations"]
    ]


def run_child_main_with_accept_fault(lane: str, out_json: str) -> int:
    """Run one child's real ``main`` with the accept fault installed.

    Called from a fresh interpreter by
    :func:`test_child_writes_its_failure_receipt_then_exits_nonzero`: the
    exit status a faulting ``main`` produces is the contract under test,
    and only a real process can show it.
    """

    def _set(module: object, name: str, value: object) -> None:
        setattr(module, name, value)

    transcript = fault_on_first_attempt_transcript()
    argv = [out_json, "--budget", str(BUDGET), "--maxcor", "3"]
    if lane == "jax":
        _set(jax_child, "nested_ls_runtime_identity", _fake_runtime_identity)
        context = _jax_context(refuses_only_the_failing_trial)
        _set(jax_child, "_prepare_outer_run", lambda **_kwargs: context)
        _set(jax_child, "minimize", transcript)
        return jax_child.main(argv)
    _set(native_child, "nested_ls_runtime_identity", _fake_runtime_identity)
    native_context = _native_context(_set, refuses_only_the_failing_trial)
    _set(native_child, "_prepare_native_run", lambda **_kwargs: native_context)
    _set(native_child, "minimize", transcript)
    return native_child.main(argv)


@pytest.mark.parametrize("lane", LANES)
def test_barrier_rows_publish_invalid_values_and_solved_rows_publish_valid(
    lane: str, monkeypatch
):
    """A rejected evaluation's row says its value is not the objective.

    Pre-change this fails on both lanes for the plainest reason: neither
    child emitted the key at all, so every row read here raises ``KeyError``.
    """

    payload = drive_lane(lane, monkeypatch.setattr, transcript=completing_transcript())
    rows = ledger_rows(lane, payload)

    assert len(rows) == 3, f"{lane} lane published {len(rows)} rows, expected 3"
    barrier_rows = [row for row in rows if row["rejection_reason"] is not None]
    solved_rows = [row for row in rows if row["rejection_reason"] is None]
    assert len(barrier_rows) == 1, (
        f"{lane} lane published {len(barrier_rows)} rejected rows, expected 1"
    )
    assert barrier_rows[0]["value_is_valid"] is False, (
        f"{lane} lane published a rejected row at coils "
        f"{barrier_rows[0]['coil_dofs']} with value_is_valid="
        f"{barrier_rows[0]['value_is_valid']!r}; a barrier value is not the "
        "objective and must be flagged False"
    )
    for row in solved_rows:
        assert row["value_is_valid"] is True, (
            f"{lane} lane published a solved row at coils {row['coil_dofs']} "
            f"with value_is_valid={row['value_is_valid']!r}; that row's value "
            "is the eight-term objective and must be flagged True"
        )


@pytest.mark.parametrize("lane", LANES)
def test_no_emitted_row_reports_a_barrier_value_as_the_objective(
    lane: str, monkeypatch
):
    """Every ``value_is_valid=True`` row really is the objective at its coils.

    This is the contract that makes the bit worth having, checked against
    the fake objective rather than against the code that sets the bit: a
    valid row must carry ``J(c)`` at its own coils, and an invalid row must
    carry the shared contract barrier priced off the committed anchor.

    Compared exactly, not within a relative band. Both sides are float64
    computed from the same published coils, so equality is reproducible, and
    a 1e-6 band would accept a barrier priced off a *different but nearby*
    anchor — precisely the confusion the bit exists to prevent.
    """

    payload = drive_lane(lane, monkeypatch.setattr, transcript=completing_transcript())

    for row in ledger_rows(lane, payload):
        coils = tuple(float(entry) for entry in row["coil_dofs"])
        if row["value_is_valid"]:
            assert row["value"] == objective_at(coils), (
                f"{lane} lane row at coils {coils} claims value_is_valid=True "
                f"but publishes {row['value']!r}, not the objective "
                f"{objective_at(coils)!r}; a consumer aggregating these rows "
                "would average a surrogate into a physics figure"
            )
        else:
            assert row["value"] == barrier_at(coils, anchor=START), (
                f"{lane} lane row at coils {coils} claims value_is_valid=False "
                f"but publishes {row['value']!r}, which is not the containment "
                f"barrier {barrier_at(coils, anchor=START)!r} off the committed "
                "anchor"
            )


def _surface_hash(dofs: object) -> str:
    """The child's own hash of one surface block, recomputed independently."""

    return jax_child.sha256_float64(np.asarray(dofs, dtype=np.float64))


@pytest.mark.parametrize("lane", LANES)
def test_a_rejected_row_names_no_solved_surface_of_its_own(lane: str, monkeypatch):
    """A refused evaluation publishes no *solved* surface, only its anchor's.

    Each lane's rejection path Nones out every other piece of inner telemetry
    because the solve raised before producing it, and the surface hash obeys
    the same rule: publishing the committed anchor's hash under
    ``inner_surface_sha256`` would claim this evaluation produced a surface it
    never produced, on a row whose coils are the trial's. ``value_is_valid``
    does not cover that — the bit is scoped to the value and its gradient
    norms.

    The anchor's surface is still a fact about the row, so it stays, under a
    name that says whose it is. Trajectory forensics read exactly that: a
    rejected row's anchor hash is how a poisoned anchor is caught persisting
    across evaluations.
    """

    payload = drive_lane(lane, monkeypatch.setattr, transcript=completing_transcript())

    rows = payload["outer_evals"] if lane == "jax" else payload["evaluations"]
    rejected = [row for row in rows if row["rejection_reason"] is not None]
    solved = [row for row in rows if row["rejection_reason"] is None]
    assert len(rejected) == 1 and len(solved) == 2

    start_surface_hash = _surface_hash([sum(START)])
    feasible_surface_hash = _surface_hash([sum(FEASIBLE_TRIAL)])

    assert rejected[0]["inner_surface_sha256"] is None, (
        f"{lane} lane published inner_surface_sha256="
        f"{rejected[0]['inner_surface_sha256']!r} on a rejected row at coils "
        f"{rejected[0]['coil_dofs']}; that is the committed anchor's surface, "
        "not one this evaluation solved for"
    )
    assert rejected[0]["anchor_surface_sha256"] == start_surface_hash, (
        f"{lane} lane published anchor_surface_sha256="
        f"{rejected[0]['anchor_surface_sha256']!r} on a row the start point "
        "was still the committed anchor for; the forensic content the rename "
        "was supposed to preserve is gone"
    )
    assert [row["inner_surface_sha256"] for row in solved] == [
        start_surface_hash,
        feasible_surface_hash,
    ], (
        f"{lane} lane's solved rows name surfaces "
        f"{[row['inner_surface_sha256'] for row in solved]}, not the ones "
        "their own inner solves produced"
    )
    assert [row["anchor_surface_sha256"] for row in solved] == [
        start_surface_hash,
        start_surface_hash,
    ], (
        f"{lane} lane's solved rows name anchors "
        f"{[row['anchor_surface_sha256'] for row in solved]}; both solved from "
        "the start point, which was the committed anchor throughout"
    )
    assert solved[1]["inner_surface_sha256"] != solved[1]["anchor_surface_sha256"], (
        f"{lane} lane published the same hash for the surface an evaluation "
        "solved and the anchor it started from, so this test cannot tell the "
        "two fields apart"
    )


@pytest.mark.parametrize("lane", LANES)
def test_a_rejected_row_carries_only_its_own_coils(lane: str, monkeypatch):
    """No rejected row on either lane republishes the anchor's coils.

    The other half of the rule above: a barrier row's ``coil_dofs`` are the
    trial's, and its distance to the anchor is published as a measurement
    rather than by silently reusing the anchor's own numbers.
    """

    payload = drive_lane(lane, monkeypatch.setattr, transcript=completing_transcript())

    rows = ledger_rows(lane, payload)
    barrier_row = next(row for row in rows if row["rejection_reason"] is not None)
    assert tuple(float(entry) for entry in barrier_row["coil_dofs"]) == FAILING_TRIAL, (
        f"{lane} lane published a barrier row at coils "
        f"{barrier_row['coil_dofs']}, not the trial's {FAILING_TRIAL}"
    )
    assert barrier_row["inner_feasible"] is False


@pytest.mark.parametrize("lane", LANES)
def test_wholly_rejected_attempt_does_not_publish_its_barrier_as_an_objective(
    lane: str, monkeypatch
):
    """scipy's surviving ``result.fun`` is published, and flagged not-objective.

    Real scipy 1.17.1, not a stand-in: when every line-search step is
    rejected, L-BFGS-B restores ``result.x`` and ``result.jac`` to the
    anchor but leaves ``result.fun`` holding the last rejected trial's
    containment barrier. Both children copy that field straight into their
    attempt ledger, so without the bit a barrier sentinel sits there
    labelled as an objective — the evaluation-level failure ``value_is_valid``
    prevents, one level up.

    The exact pair is asserted, not just its inequality, because both children
    quote it in a source comment: this is the measurement that keeps the quote
    from drifting away from the fixture that ships.
    """

    payload = drive_lane(
        lane,
        monkeypatch.setattr,
        inner_refuses=refuses_every_point_but_the_start,
    )

    attempts = payload["restart_attempts"]
    assert len(attempts) == 1, (
        f"{lane} lane recorded {len(attempts)} attempts, expected the single "
        "wholly rejected one"
    )
    attempt = attempts[0]
    assert objective_at(START) == ANCHOR_OBJECTIVE, (
        "the shipped fixture no longer starts at the anchor objective both "
        f"children quote: {objective_at(START)!r} != {ANCHOR_OBJECTIVE!r}"
    )
    assert attempt["fun"] == WHOLLY_REJECTED_BARRIER, (
        f"{lane} lane's wholly rejected attempt reported fun={attempt['fun']!r}; "
        f"both children quote {WHOLLY_REJECTED_BARRIER!r} as the barrier this "
        "fixture reproduces, so either the quote or the fixture has moved"
    )
    assert attempt["fun"] != ANCHOR_OBJECTIVE, (
        f"{lane} lane's wholly rejected attempt reported fun={attempt['fun']!r}, "
        f"which already equals the anchor objective {ANCHOR_OBJECTIVE!r}; this "
        "test no longer reproduces the scipy corner it exists to guard"
    )
    assert attempt["value_is_valid"] is False, (
        f"{lane} lane published attempt fun={attempt['fun']!r} with "
        f"value_is_valid={attempt['value_is_valid']!r} while the committed "
        f"incumbent's objective is {ANCHOR_OBJECTIVE!r}; that number is a "
        "containment barrier and must not be readable as an objective"
    )
    if lane == "jax":
        assert payload["result_fun_is_valid"] is False, (
            "jax lane published result_fun="
            f"{payload['result_fun']!r} as a valid objective after a wholly "
            "rejected line search"
        )


@pytest.mark.parametrize("lane", LANES)
def test_completed_attempt_publishes_its_objective_as_valid(lane: str, monkeypatch):
    """The same bit says True when ``fun`` really is the objective at ``x``.

    Real scipy again, with an inner solve that refuses nothing, so the run
    ends on an accepted step and scipy's last evaluation stands at the point
    the attempt reports. The guard has to be able to say yes, or it is not a
    discriminator.

    Nothing here is true by construction. The bit is derived by the child from
    its own ledger, not from anything this test hands it, and the assertions
    below check the provenance the bit claims: that the last emitted row is a
    valid row, that its coils are exactly the reported endpoint, and that
    ``fun`` is bitwise that row's value. A child that flagged the attempt
    valid without that provenance, or that sourced the last evaluation
    wrongly, fails here.
    """

    payload = drive_lane(lane, monkeypatch.setattr, inner_refuses=refuses_nothing)

    attempts = payload["restart_attempts"]
    assert len(attempts) == 1, (
        f"{lane} lane recorded {len(attempts)} attempts, expected one"
    )
    attempt = attempts[0]
    rows = ledger_rows(lane, payload)
    last_row = rows[-1]
    assert last_row["value_is_valid"] is True, (
        f"{lane} lane's last evaluation was not a valid row, so this test is "
        "not exercising the True direction it exists for"
    )
    assert tuple(float(entry) for entry in last_row["coil_dofs"]) == tuple(
        float(entry) for entry in payload["optimizer_x"]
    ), (
        f"{lane} lane ended at optimizer_x={payload['optimizer_x']} while its "
        f"last evaluation stood at {last_row['coil_dofs']}; the True direction "
        "requires those to be the same point"
    )
    assert attempt["fun"] == last_row["value"], (
        f"{lane} lane published attempt fun={attempt['fun']!r} beside a last "
        f"evaluation of {last_row['value']!r}; scipy's fun is the last "
        "evaluation's value, so a difference means the child read the wrong row"
    )
    assert attempt["value_is_valid"] is True, (
        f"{lane} lane flagged a completed attempt's fun={attempt['fun']!r} as "
        "not the objective, although its last evaluation published the "
        f"objective at the reported endpoint {payload['optimizer_x']}"
    )
    if lane == "jax":
        assert payload["result_fun_is_valid"] is True, (
            "jax lane flagged a completed run's result_fun as not the objective"
        )


@pytest.mark.parametrize("lane", LANES)
def test_a_barrier_that_rounds_to_the_incumbent_objective_is_still_a_barrier(
    lane: str, monkeypatch
):
    """The bit is provenance, not a value coincidence.

    This transcript accepts a step and then evaluates a trial ``1e-9`` away
    that the inner solve refuses, so ``result.x`` is the incumbent and
    ``result.fun`` is a containment barrier whose value rounds to bitwise the
    incumbent's own objective. A rule that decided the bit by comparing
    ``result.fun`` against the incumbent's stored ``J`` stamps this attempt
    valid — and dcsrch contracts toward exactly this regime near convergence,
    so it is the case that matters, not a contrived one. The provenance rule
    reads the last emitted row instead and says False.
    """

    coincidence = barrier_at(NEAR_ANCHOR_TRIAL, anchor=FEASIBLE_TRIAL)
    assert coincidence == objective_at(FEASIBLE_TRIAL), (
        "the fixture no longer reproduces the rounding coincidence this test "
        f"exists for: barrier {coincidence!r} against incumbent objective "
        f"{objective_at(FEASIBLE_TRIAL)!r}"
    )

    payload = drive_lane(
        lane,
        monkeypatch.setattr,
        transcript=barrier_rounds_to_the_anchor_transcript(),
        inner_refuses=refuses_the_near_anchor_trial,
    )

    attempt = payload["restart_attempts"][0]
    rows = ledger_rows(lane, payload)
    assert tuple(float(entry) for entry in payload["optimizer_x"]) == FEASIBLE_TRIAL
    assert rows[-1]["value_is_valid"] is False, (
        "this transcript is supposed to end on a refused trial; it ended on "
        f"{rows[-1]['coil_dofs']} with value_is_valid=True"
    )
    assert attempt["fun"] == coincidence, (
        f"{lane} lane published attempt fun={attempt['fun']!r}, not the "
        f"barrier {coincidence!r} its last evaluation returned"
    )
    assert attempt["value_is_valid"] is False, (
        f"{lane} lane flagged fun={attempt['fun']!r} as the objective because "
        "it equals the committed incumbent's J; that number is the last "
        "rejected trial's containment barrier and no consumer may read it as "
        "a measurement of the objective"
    )
    if lane == "jax":
        assert payload["result_fun_is_valid"] is False, (
            "jax lane published result_fun="
            f"{payload['result_fun']!r} as a valid objective although it is a "
            "barrier that merely rounds to the incumbent's J"
        )


@pytest.mark.parametrize("lane", LANES)
def test_accept_without_candidate_publishes_the_ledger_before_failing(
    lane: str, monkeypatch
):
    """A callback with no matching candidate fails closed with evidence intact.

    Pre-change the store's ``RuntimeError`` propagated out of scipy and out
    of the child, so there was no payload at all to assert on: every
    assertion below failed at the ``drive_lane`` call with that
    ``RuntimeError``.

    The counters are part of the evidence. scipy returned no verdict for the
    faulting attempt, so the child derives them from what it does hold — one
    evaluation row per call into its own objective (``jac=True``, so nfev and
    njev are the same count) and one completed accepted-step callback per
    iterate. Publishing zeros beside a two-row ledger, as it did before, is a
    receipt that contradicts itself.
    """

    payload = drive_lane(
        lane,
        monkeypatch.setattr,
        transcript=fault_on_first_attempt_transcript(),
    )

    expected_schema = (
        NESTED_LS_OUTER_JAX_CHILD_SCHEMA
        if lane == "jax"
        else NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA
    )
    assert payload["schema"] == expected_schema, (
        f"{lane} lane failure payload declares schema {payload['schema']!r}; a "
        "consumer keyed on the child contract cannot read it"
    )
    assert payload["child_fault_reason"] == (
        NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    ), (
        f"{lane} lane failure payload names the fault "
        f"{payload['child_fault_reason']!r}, not the contract's "
        f"{NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON!r}"
    )
    assert payload["success"] is False, (
        f"{lane} lane reported success={payload['success']!r} on a run that "
        "failed at the accepted-step callback"
    )
    rows = ledger_rows(lane, payload)
    assert [tuple(float(entry) for entry in row["coil_dofs"]) for row in rows] == [
        START,
        FAILING_TRIAL,
    ], (
        f"{lane} lane lost ledger rows accumulated before the fault; published "
        f"{[row['coil_dofs'] for row in rows]}"
    )
    assert [row["value_is_valid"] for row in rows] == [True, False], (
        f"{lane} lane preserved rows but not their value semantics: "
        f"{[row['value_is_valid'] for row in rows]}"
    )
    counters = {key: payload[key] for key in ("nit", "nfev", "njev", "restart_count")}
    assert counters == {"nit": 0, "nfev": 2, "njev": 2, "restart_count": 0}, (
        f"{lane} lane published {counters} beside a two-row ledger with no "
        "completed iterate; the fault path must count what the child actually "
        "has, not what scipy never returned"
    )
    assert payload["restart_nits"] == [] and payload["restart_attempts"] == [], (
        f"{lane} lane invented a scipy verdict for the faulting attempt: "
        f"restart_nits={payload['restart_nits']!r} "
        f"restart_attempts={payload['restart_attempts']!r}"
    )
    if lane == "jax":
        endpoint_j = payload["endpoint_j"]
    else:
        endpoint_j = payload["endpoint"]["objective"]
    assert endpoint_j == objective_at(START), (
        f"{lane} lane failure payload reports endpoint J={endpoint_j!r}; the "
        f"committed incumbent was the start point with J={objective_at(START)!r}"
    )
    assert json.loads(json.dumps(payload, allow_nan=False))["child_fault_reason"] == (
        NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    ), f"{lane} lane failure payload does not round-trip through strict JSON"


@pytest.mark.parametrize("lane", LANES)
def test_fault_on_a_restarted_attempt_counts_the_restart_that_happened(
    lane: str, monkeypatch
):
    """The restart arithmetic survives a fault on the second attempt.

    ``restart_nits`` only grows when scipy returns, so counting its entries
    under-reports by one on every faulted attempt — and the fault path is
    exactly where a reader is trying to reconstruct how far the run got. The
    count is taken when an attempt begins instead.

    The same transcript pins what the child hands scipy on the restart: the
    committed incumbent as ``x0`` and the unspent share of the budget as
    ``maxiter``. The stand-in honours both, so a child that restarted from the
    original start point or re-spent the whole budget fails here.
    """

    transcript = fault_on_second_attempt_transcript()
    payload = drive_lane(lane, monkeypatch.setattr, transcript=transcript)

    assert len(transcript.calls) == 2, (
        f"{lane} lane made {len(transcript.calls)} scipy attempts, expected a "
        "restart followed by the faulting attempt"
    )
    first, second = transcript.calls
    assert tuple(float(entry) for entry in first["x0"]) == START
    assert first["maxiter"] == BUDGET
    assert tuple(float(entry) for entry in second["x0"]) == FEASIBLE_TRIAL, (
        f"{lane} lane restarted from {second['x0']}, not the committed "
        f"incumbent {FEASIBLE_TRIAL}"
    )
    assert second["maxiter"] == BUDGET - 1, (
        f"{lane} lane handed the restart maxiter={second['maxiter']}, not the "
        f"{BUDGET - 1} iterations the first attempt left unspent"
    )
    counters = {key: payload[key] for key in ("nit", "nfev", "njev", "restart_count")}
    assert counters == {"nit": 1, "nfev": 4, "njev": 4, "restart_count": 1}, (
        f"{lane} lane published {counters}; attempt one reported nit=1 nfev=2 "
        "and attempt two added two evaluations and no completed iterate before "
        "faulting, and one restart really happened"
    )
    assert payload["restart_nits"] == [1], (
        f"{lane} lane published restart_nits={payload['restart_nits']!r}; only "
        "the attempt scipy returned from may contribute one"
    )
    assert len(payload["restart_attempts"]) == 1, (
        f"{lane} lane published {len(payload['restart_attempts'])} attempt rows; "
        "the faulting attempt produced no scipy verdict to record"
    )
    assert len(ledger_rows(lane, payload)) == 4, (
        f"{lane} lane published {len(ledger_rows(lane, payload))} ledger rows; "
        "both attempts' evaluations, including each attempt's own x0, belong "
        "in the receipt"
    )


@pytest.mark.parametrize("lane", LANES)
def test_child_writes_its_failure_receipt_then_exits_nonzero(lane: str, tmp_path: Path):
    """The child process leaves its evidence on disk and still exits nonzero.

    Pre-change the callback's ``RuntimeError`` killed the process: the exit
    status was nonzero for the wrong reason and ``out_json`` was never
    written, so the ``is_file`` assertion below failed.

    The status is asserted exactly, and the absence of a traceback with it.
    Both children ``return 1`` on this path; a bare ``!= 0`` would also accept
    a segfault or a signal death, and status 1 alone would still accept an
    uncaught exception, which CPython also exits 1 on. Those are the failure
    modes this test exists to distinguish from a clean fail-closed exit.
    """

    out_json = tmp_path / f"{lane}_accept_fault.json"
    program = (
        f"import sys; sys.path.insert(0, {str(REPO)!r}); "
        f"sys.path.insert(0, {str(REPO / 'tests')!r}); "
        "import geo.test_nested_ls_outer_child_evidence as fixtures; "
        "raise SystemExit("
        f"fixtures.run_child_main_with_accept_fault({lane!r}, sys.argv[1]))"
    )
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", program, str(out_json)],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, (
        f"{lane} child exited {completed.returncode} after failing at the "
        "accepted-step callback; the contract is a deliberate `return 1` after "
        f"the receipt is written. stdout={completed.stdout[-2000:]} "
        f"stderr={completed.stderr[-2000:]}"
    )
    assert "Traceback" not in completed.stderr, (
        f"{lane} child exited 1 by raising rather than by returning; an "
        "uncaught exception shares the status of the deliberate fail-closed "
        f"exit, so the traceback is what separates them. "
        f"stderr={completed.stderr[-2000:]}"
    )
    assert out_json.is_file(), (
        f"{lane} child exited {completed.returncode} without writing its "
        f"receipt to {out_json}; the run's evidence was destroyed. "
        f"stderr={completed.stderr[-2000:]}"
    )
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["child_fault_reason"] == (
        NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON
    ), (
        f"{lane} child wrote a receipt naming {payload['child_fault_reason']!r} "
        "instead of the accept-without-candidate fault"
    )
    assert payload["success"] is False, (
        f"{lane} child wrote success={payload['success']!r} on a failed run"
    )
    assert len(ledger_rows(lane, payload)) == 2, (
        f"{lane} child wrote a receipt missing the rows it had already "
        "accumulated when the fault fired"
    )


@pytest.mark.parametrize("lane", LANES)
def test_both_lanes_publish_a_runtime_block(lane: str, monkeypatch):
    """Each child binds the compiled extension its own numbers came out of.

    The parent's record cannot stand in: the children are separate processes
    launched with rewritten environments, so only the child that loaded a
    binary can witness which one it loaded. The native lane is the lane whose
    numbers come out of ``simsoptpp``, and it published no ``runtime`` block
    at all.
    """

    payload = drive_lane(lane, monkeypatch.setattr, transcript=completing_transcript())

    assert payload["runtime"] == FAKE_RUNTIME_IDENTITY, (
        f"{lane} lane published runtime={payload.get('runtime')!r}; both lanes "
        "must stamp the identity through the module attribute the other lane "
        "stamps"
    )


def test_both_lanes_bind_their_binary_through_one_identity_function():
    """One function, both lanes, so the two receipts compare field for field."""

    assert (
        jax_child.nested_ls_runtime_identity is native_child.nested_ls_runtime_identity
    ), "the lanes reach two different runtime-identity functions"
    identity = nested_ls_runtime_identity()
    for key in ("simsoptpp_path", "simsoptpp_sha256", "timestamp_utc"):
        assert key in identity, (
            f"the shared runtime identity publishes no {key!r}, so a child "
            "receipt cannot bind the binary its numbers came out of"
        )


@pytest.mark.parametrize("lane", LANES)
def test_rejudge_refuses_an_endpoint_written_by_a_faulted_child(
    lane: str, monkeypatch, tmp_path: Path
):
    """A fault receipt is schema-valid, and the physics gate must still say no.

    The fault payload publishes a complete endpoint block copied from the
    committed incumbent, precisely so the evidence survives, and the receipts
    are deliberately not unlinked. Nothing in the layout distinguishes one
    from a finished run, so ``_load_endpoint_record`` refuses it by name
    instead of certifying physics at a point whose producing run never
    finished standing on it.
    """

    faulted = drive_lane(
        lane,
        monkeypatch.setattr,
        transcript=fault_on_first_attempt_transcript(),
    )
    completed = drive_lane(
        lane, monkeypatch.setattr, transcript=completing_transcript()
    )
    fault_path = tmp_path / f"{lane}_fault.json"
    fault_path.write_text(json.dumps(faulted, allow_nan=False), encoding="utf-8")
    completed_path = tmp_path / f"{lane}_completed.json"
    completed_path.write_text(json.dumps(completed, allow_nan=False), encoding="utf-8")

    with pytest.raises(SystemExit) as refusal:
        jax_child._load_endpoint_record(fault_path)
    assert NESTED_LS_OUTER_ACCEPT_WITHOUT_CANDIDATE_REASON in str(refusal.value), (
        f"the rejudge gate refused the {lane} fault receipt without naming the "
        f"fault: {refusal.value}"
    )

    record = jax_child._load_endpoint_record(completed_path)
    assert record.lane == lane and record.j == objective_at(FEASIBLE_TRIAL), (
        "the gate must still read a completed receipt of the same schema; it "
        f"read lane={record.lane!r} j={record.j!r}"
    )
