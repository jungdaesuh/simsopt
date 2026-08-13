"""Gates on the projected route's GPU attempt-protocol launcher.

Everything here runs on CPU, which is deliberate: the expensive facts this file
protects are not numerical, they are protocol facts -- the verdict vocabulary is
closed, the pinned-term gate does not manufacture a false reject, a published
receipt re-derives its own verdict, and the launcher refuses a process that
resolved to a backend other than the GPU.  The one thing a CPU process cannot
check is that the chain runs end to end on the GPU; the real entry path is
executed here as launched (in a subprocess, with nothing monkeypatched) as far
as the backend gate, and beyond it by the supervised GPU smoke.

That shape is the predecessor route's lesson: a fifty-nine-test suite was green
while its launcher raised ``NameError`` in its first phase, because every test
imported the module and monkeypatched the constant that was missing.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import benchmarks.rehearse_single_stage_projected_route_cpu as rehearsal
import benchmarks.run_single_stage_projected_route_gpu_root as launcher
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.process_gpu_monitor import (
    GPU_MEMORY_UNAVAILABLE_REASONS,
    ProcessGpuMemoryMonitorError,
)
from benchmarks.single_stage_fullspace_snapshot import canonical_json_bytes
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

REPOSITORY = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------ frozen protocol


def test_the_claim_contract_literals_are_the_ones_the_plan_states() -> None:
    """The three numbers the claim is discharged by, spelled out.

    Every other test in the chain reaches these through their symbols, so a
    symbol quietly reassigned would move the bar and leave the suite green.
    The literals are the plan document's section 1 table: the native
    reference's endpoint objective, the route's own feasibility tolerance, and
    the wall native spent -- in SECONDS, which is the unit
    ``ProjectedLbfgsRun.compile_seconds`` and ``.solve_seconds`` report.
    """

    assert rehearsal.NATIVE_TARGET_OBJECTIVE == 4.4822246533126125e-08
    assert rehearsal.CERTIFIED_ROUTE_OPTIONS.objective_target == 4.4822246533126125e-08
    assert rehearsal.CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance == 1.0e-10
    assert rehearsal.NATIVE_WALL_SECONDS_BAR == 287.30421751597896


def test_the_certified_wall_is_derived_from_engine_compile_plus_engine_solve() -> None:
    """The timed boundary, DERIVED from its halves exactly as the verdict is.

    Bootstrap and the identity gate are setup native's 287.30 s does not
    contain; naming the boundary is what makes the comparison checkable.  The
    bar is strict: a wall equal to it does not discharge the claim.
    """

    attempt = _attempt("LATCHED")
    attempt["evidence"]["timing_seconds"] = {
        "engine_compile": 12.5,
        "engine_solve": 100.25,
        "engine_wall": 112.75,
        "attempt_wall": 140.0,
    }
    assert launcher.attempt_engine_wall_seconds(attempt) == 112.75 == (12.5 + 100.25)

    bar = rehearsal.NATIVE_WALL_SECONDS_BAR
    at_the_bar = _attempt("LATCHED", engine_wall=bar)
    assert (
        launcher.derive_verdict(
            [at_the_bar],
            wall_seconds_bar=bar,
            conformance=launcher.CONFORMANCE_PREREGISTERED,
        )
        == launcher.VERDICT_QUALITY_ONLY
    )


def test_a_restated_engine_wall_is_refused_rather_than_believed() -> None:
    """The quantity the claim is judged on may not be published by assertion.

    ``engine_wall`` used to be read back; both halves are published beside it
    and their sum is an IEEE addition of the same two doubles, so a receipt
    whose wall is not its own compile plus its own solve has restated the
    number rather than measured it.
    """

    attempt = _attempt("LATCHED")
    attempt["evidence"]["timing_seconds"] = {
        "engine_compile": 12.5,
        "engine_solve": 100.25,
        "engine_wall": 12.75,
    }
    with pytest.raises(launcher.ProjectedRootError, match="is not its own"):
        launcher.attempt_engine_wall_seconds(attempt)


def test_the_attempt_protocol_is_the_one_the_plan_pre_registered() -> None:
    """N = 3, stop at the first latch, and every attempt publishes."""

    assert launcher.PREREGISTERED_ATTEMPTS == 3
    assert launcher.GPU_REQUIRED_ENVIRONMENT == {
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    # Compile is inside the claim, so both cache knobs are pinned: the defaults
    # skip small and fast entries, which is most of this route's bundle.
    assert launcher.PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES == -1
    assert launcher.PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS == 0.0


def test_the_launcher_certifies_the_configuration_the_latches_used() -> None:
    """The route under protocol is the rehearsal's frozen one, unmodified."""

    options = rehearsal.rehearsal_options(rehearsal.CERTIFIED_MAXIMUM_ITERATIONS)
    assert options == rehearsal.CERTIFIED_ROUTE_OPTIONS
    assert options.lagrangian_newton is True
    assert options.frozen_projector_line_search is True
    assert options.newton_tangent_fraction_threshold == 0.25
    assert options.projector_refresh_period == 4


def _attempt(
    outcome: str,
    *,
    index: int = 1,
    engine_wall: float = 100.0,
    gate: str | None = None,
) -> dict:
    """One supervised attempt in the shape the supervisor publishes it.

    The compile half is 0.0 so that ``engine_compile + engine_solve`` reproduces
    the requested wall to the bit -- these fixtures probe the verdict boundary,
    which a rounded split would blur by a ULP at exactly the value that matters.
    """

    evidence: dict | None = {
        "gate_refused": gate,
        "solve": {"latched": outcome == "LATCHED"},
        "timing_seconds": {
            "engine_compile": 0.0,
            "engine_solve": engine_wall,
            "engine_wall": engine_wall,
        },
    }
    if outcome == "PROTOCOL_FAILURE":
        evidence = None
    return {
        "attempt_index": index,
        "artifact_relative_path": f"attempts/attempt-{index}",
        "outcome": outcome,
        "return_code": 2 if gate is not None else 0,
        "timed_out": outcome == "TIMEOUT",
        "evidence": evidence,
    }


def _derive(attempts: list[dict], *, conformance: str | None = None) -> str:
    return launcher.derive_verdict(
        attempts,
        wall_seconds_bar=rehearsal.NATIVE_WALL_SECONDS_BAR,
        conformance=launcher.CONFORMANCE_PREREGISTERED
        if conformance is None
        else conformance,
    )


def test_every_protocol_outcome_maps_to_one_of_exactly_four_verdicts() -> None:
    """There is no undefined outcome: roots 1-4 all died in unwritten ones."""

    bar = rehearsal.NATIVE_WALL_SECONDS_BAR
    assert _derive([_attempt("LATCHED", engine_wall=bar - 1.0)]) == (
        launcher.VERDICT_CLAIM_DISCHARGED
    )
    assert _derive([_attempt("LATCHED", engine_wall=bar + 1.0)]) == (
        launcher.VERDICT_QUALITY_ONLY
    )
    assert _derive(
        [_attempt("COMPLETED_WITHOUT_LATCH", index=index) for index in (1, 2, 3)]
    ) == launcher.VERDICT_NO_LATCH
    assert _derive(
        [_attempt("GATE_REFUSED", gate="problem_identity")]
    ) == launcher.verdict_of_gate("problem_identity")
    assert _derive([_attempt("TIMEOUT")]).startswith(
        launcher.VERDICT_GATE_REFUSED_PREFIX
    )
    assert _derive([]).startswith(launcher.VERDICT_GATE_REFUSED_PREFIX)


def test_the_claim_is_discharged_by_the_first_latching_attempt_not_the_first() -> None:
    """A no-latch draw indicts nothing, and the wall that counts is the latch's."""

    bar = rehearsal.NATIVE_WALL_SECONDS_BAR
    attempts = [
        _attempt("COMPLETED_WITHOUT_LATCH", index=1, engine_wall=bar - 50.0),
        _attempt("LATCHED", index=2, engine_wall=bar + 10.0),
    ]
    assert _derive(attempts) == launcher.VERDICT_QUALITY_ONLY


def test_a_latch_under_the_bar_at_a_bounded_budget_is_capped_at_quality_only() -> None:
    """``CLAIM_DISCHARGED`` requires the pre-registered protocol, not just a latch.

    A bounded run used to mint the campaign's headline verdict beside
    ``quality_claim: NOT_CLAIMED_AT_BOUNDED_BUDGET`` and a per-term physics gate
    that never ran, with a zero exit code -- a self-contradicting artifact whose
    only honest field was the one no downstream reader keys on.  The underlying
    measurement is real, so it is published as ``QUALITY_ONLY`` rather than
    thrown away.
    """

    fast_latch = [_attempt("LATCHED", engine_wall=1.0)]
    assert _derive(fast_latch) == launcher.VERDICT_CLAIM_DISCHARGED
    assert (
        _derive(fast_latch, conformance=launcher.CONFORMANCE_BOUNDED_SMOKE)
        == launcher.VERDICT_QUALITY_ONLY
    )
    # A bounded run that misses is still a miss, under the name section 4 gives
    # it: capping applies to the discharge, not to the outcome space.
    assert (
        _derive(
            [_attempt("COMPLETED_WITHOUT_LATCH")],
            conformance=launcher.CONFORMANCE_BOUNDED_SMOKE,
        )
        == launcher.VERDICT_NO_LATCH
    )


def test_conformance_is_one_label_derived_from_the_three_frozen_facts() -> None:
    """N, the certified budget and the cold lane decide it, in one place."""

    assert (
        launcher.attempt_protocol_conformance(
            authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
            iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            cold_lane=True,
        )
        == launcher.CONFORMANCE_PREREGISTERED
    )
    for authorized, iterations, cold_lane in (
        (10, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True),
        (launcher.PREREGISTERED_ATTEMPTS, 400, True),
        (launcher.PREREGISTERED_ATTEMPTS, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, False),
    ):
        assert (
            launcher.attempt_protocol_conformance(
                authorized_attempts=authorized,
                iterations=iterations,
                cold_lane=cold_lane,
            )
            == launcher.CONFORMANCE_BOUNDED_SMOKE
        )


def test_a_no_latch_draw_at_the_certified_budget_keeps_the_protocol_running() -> None:
    """``NO_LATCH_IN_PROTOCOL`` must be reachable at a root, or N=3 buys nothing.

    The pinned-term ledger gate binds the attempt that DISCHARGES the claim.
    Gated on every certified-budget attempt instead, a non-latching draw fails
    ``weighted_total``'s ``not_worse`` band with certainty -- its objective is
    above the target by definition -- so the first miss, roughly a one-in-five
    event by the campaign's own measured rate, would publish
    ``GATE_REFUSED:endpoint_ledger``, break the attempt loop after one of three
    attempts and make ``COMPLETED_WITHOUT_LATCH`` unreachable.
    """

    for latched in (True, False):
        assert rehearsal.endpoint_ledger_is_gated(
            iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, latched=latched
        ) is latched
    assert not rehearsal.endpoint_ledger_is_gated(
        iterations=rehearsal.REHEARSAL_MAXIMUM_ITERATIONS, latched=True
    )

    misses = [_attempt("COMPLETED_WITHOUT_LATCH", index=index) for index in (1, 2, 3)]
    assert _derive(misses) == launcher.VERDICT_NO_LATCH


def test_the_supervisor_reads_exactly_the_bytes_the_child_wrote(
    tmp_path: Path,
) -> None:
    """The canonical encoding is newline-terminated, and a split removes it.

    Found by the first real GPU launch: the child produced a correct canonical
    document, the supervisor split the stream, and the missing terminator made
    every attempt unparseable after a complete solve.  The two halves are
    exercised together here so they cannot drift apart again.
    """

    payload = {"attempt_index": 1, "gate_refused": None, "value": 1.5}
    emitted = canonical_json_bytes(payload)
    assert emitted.endswith(b"\n")
    assert launcher._parse_attempt_stdout(b"noise on an earlier line\n" + emitted) == (
        payload
    )
    assert launcher._parse_attempt_stdout(b"") is None


def test_a_child_that_printed_past_its_payload_is_a_named_protocol_failure() -> None:
    """Unparseable bytes end in the protocol's own outcome, not a traceback.

    The parse used to raise, and the raise propagated out of the supervisor:
    a one-shot root spent with no artifact published at all.  Noncanonical
    bytes are still refused -- they are simply refused INTO
    ``PROTOCOL_FAILURE``, the outcome an empty stream already produced.
    """

    emitted = canonical_json_bytes({"attempt_index": 1, "gate_refused": None})
    trailing = emitted + b"a stray line after the payload\n"
    assert launcher._parse_attempt_stdout(trailing) is None
    assert launcher._parse_attempt_stdout(b'{ "spaced" : 1 }\n') is None
    assert (
        launcher._attempt_outcome(
            launcher._parse_attempt_stdout(trailing), return_code=0, timed_out=False
        )
        == "PROTOCOL_FAILURE"
    )
    assert _derive([_attempt("PROTOCOL_FAILURE")]).startswith(
        launcher.VERDICT_GATE_REFUSED_PREFIX
    )


def test_attempt_outcomes_classify_without_inventing_a_fifth() -> None:
    latched = {"gate_refused": None, "solve": {"latched": True}}
    missed = {"gate_refused": None, "solve": {"latched": False}}
    assert launcher._attempt_outcome(latched, return_code=0, timed_out=False) == (
        "LATCHED"
    )
    assert launcher._attempt_outcome(missed, return_code=0, timed_out=False) == (
        "COMPLETED_WITHOUT_LATCH"
    )
    assert launcher._attempt_outcome(
        {"gate_refused": "solve"}, return_code=2, timed_out=False
    ) == "GATE_REFUSED"
    assert launcher._attempt_outcome(None, return_code=1, timed_out=False) == (
        "PROTOCOL_FAILURE"
    )
    assert launcher._attempt_outcome(latched, return_code=0, timed_out=True) == (
        "TIMEOUT"
    )


# ------------------------------------------------------- pinned-term ledger gate


def _ledger(**overrides: float) -> dict:
    """A ledger whose terminal side equals native, then perturbed by name."""

    native = {
        "constraint.boozer|inf": 3.0e-15,
        "constraint.volume": 0.0,
        "observable.G": 13.0,
        "observable.iota": -0.42,
        "observable.major_radius": 1.44,
        "observable.non_qs_ratio": 1.9e-2,
        "observable.total_length": 14.9,
        "observable.volume": -0.178,
        "raw.non_qs": 3.6e-4,
        "raw.residual": 1.0e-20,
        "state.G": 13.0,
        "weighted_total": 4.48e-8,
    }
    terminal = {**native, **overrides}
    return {"terminal": terminal, "native": native}


def test_an_endpoint_that_beat_native_on_non_qs_is_not_refused() -> None:
    """The banked Q1 latch came out 0.9% BETTER than native on non-QS.

    A two-sided relative band on the quality terms would refuse the very
    evidence this campaign banked -- the false-reject class the V260 shell gate
    and the SQP rho floor already cost two verdicts.
    """

    verdict = rehearsal.gate_endpoint_ledger(
        _ledger(**{"raw.non_qs": 3.6e-4 * 0.991, "observable.non_qs_ratio": 1.9e-2 * 0.991})
    )
    assert verdict["passed"] is True
    assert verdict["terms"]["raw.non_qs"]["comparison"] == "not_worse"
    assert verdict["terms"]["raw.non_qs"]["measured"] < 0.0


def test_an_endpoint_materially_worse_than_native_is_refused() -> None:
    """``not_worse`` is one-sided, not absent."""

    verdict = rehearsal.gate_endpoint_ledger(_ledger(**{"raw.non_qs": 3.6e-4 * 1.01}))
    assert verdict["passed"] is False
    assert verdict["failed_terms"] == ["raw.non_qs"]


def test_geometry_terms_are_gated_relatively_and_residuals_absolutely() -> None:
    """Both sides of a machine-zero residual sit below any meaningful ratio."""

    assert rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.iota": -0.42 * (1.0 + 1.0e-5)})
    )["passed"]
    assert not rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.iota": -0.42 * (1.0 + 1.0e-2)})
    )["passed"]
    # A residual moving from 1e-20 to 1e-18 is a 99x relative change and a
    # 1e-18 absolute one; only the absolute reading means anything here.
    assert rehearsal.gate_endpoint_ledger(_ledger(**{"raw.residual": 1.0e-18}))["passed"]
    assert not rehearsal.gate_endpoint_ledger(
        _ledger(**{"constraint.volume": 1.0e-9})
    )["passed"]


def test_the_hinged_length_term_is_free_below_native_and_gated_above_it() -> None:
    """The length penalty is ``0.5 * max(L - target, 0)**2``, flat underneath.

    Nothing in the shared objective pins a SHORTER coil set, and the banked Q1
    latch is in that flat region -- its ``raw.length`` is exactly 0.0 while
    native's is 1.278e-11.  A two-sided band on total length therefore gates a
    free direction, which is the reason G is informational; a one-sided band
    gates the direction the objective actually penalizes.
    """

    assert rehearsal.PINNED_ENDPOINT_QUALITY_GATES["observable.total_length"] == (
        ("not_worse", 1.0e-4)
    )
    shorter = rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.total_length": 14.9 * (1.0 - 1.0e-2)})
    )
    assert shorter["passed"] is True
    longer = rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.total_length": 14.9 * (1.0 + 1.0e-2)})
    )
    assert longer["failed_terms"] == ["observable.total_length"]


def test_the_free_direction_G_is_never_gated() -> None:
    """Nothing in the shared objective pins the net poloidal current."""

    verdict = rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.G": 13.0 * 0.99, "state.G": 13.0 * 0.99})
    )
    assert verdict["passed"] is True
    assert not set(verdict["terms"]) & set(rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES)
    assert set(verdict["terms"]) == set(rehearsal.PINNED_ENDPOINT_QUALITY_TERMS)


def test_every_pinned_term_carries_exactly_one_comparison_class() -> None:
    """The gate map and the pinned set have one membership between them."""

    assert set(rehearsal.PINNED_ENDPOINT_QUALITY_GATES) == set(
        rehearsal.PINNED_ENDPOINT_QUALITY_TERMS
    )
    assert {
        comparison for comparison, _ in rehearsal.PINNED_ENDPOINT_QUALITY_GATES.values()
    } <= {"absolute", "relative", "not_worse"}


def test_a_bounded_run_does_not_read_as_a_root() -> None:
    """Three attempts cannot reach the target, and the receipt says so.

    The endpoint ledger is gated at the certified budget only -- a bounded
    attempt sits four orders of magnitude away, so a gate there would fail on
    every term and prove nothing -- and the same budget rule decides whether
    the receipt claims quality at all.  A bounded smoke that read as a spent
    pre-registered protocol would drag in the successor-root rule of plan
    section 12.1, which applies to a root and to nothing else.
    """

    bounded = launcher.build_root_evidence(
        attempts=[],
        cold_lane=None,
        snapshot={},
        supervisor={},
        authorized_attempts=1,
        iterations=3,
        cold_lane_authorized=True,
        cache={},
        verdict=launcher.VERDICT_NO_LATCH,
        chain_seconds=1.0,
    )
    assert bounded["quality_claim"] == "NOT_CLAIMED_AT_BOUNDED_BUDGET"
    assert bounded["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_BOUNDED_SMOKE
    )

    root = launcher.build_root_evidence(
        attempts=[_attempt("LATCHED", engine_wall=1.0)],
        cold_lane=None,
        snapshot={},
        supervisor={},
        authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
        iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        cold_lane_authorized=True,
        cache={},
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        chain_seconds=1.0,
    )
    assert root["quality_claim"] == "CERTIFIED_BUDGET"
    assert root["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_PREREGISTERED
    )
    assert root["claim"]["target_objective"] == rehearsal.NATIVE_TARGET_OBJECTIVE
    assert root["claim"]["wall_seconds_bar"] == rehearsal.NATIVE_WALL_SECONDS_BAR
    assert root["claim"]["feasibility_tolerance"] == (
        rehearsal.CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
    )
    # k over N, the denominator section 4 names -- the attempts AUTHORIZED, not
    # the attempts the stop rule got to.
    assert root["attempt_protocol"]["latch_rate"] == (
        f"1/{launcher.PREREGISTERED_ATTEMPTS}"
    )
    assert root["attempt_protocol"]["attempts_run"] == 1


# ----------------------------------------------------------------- environment


def test_the_environment_gate_names_the_variable_that_differs() -> None:
    complete = dict(launcher.GPU_REQUIRED_ENVIRONMENT)
    assert (
        rehearsal.validate_environment(
            complete, required=launcher.GPU_REQUIRED_ENVIRONMENT
        )
        == complete
    )
    with pytest.raises(rehearsal.RehearsalError, match="JAX_PLATFORMS"):
        rehearsal.validate_environment(
            {**complete, "JAX_PLATFORMS": "cpu"},
            required=launcher.GPU_REQUIRED_ENVIRONMENT,
        )


def test_a_process_that_resolved_to_the_cpu_is_refused() -> None:
    """JAX resolves its platform lazily, so a missing variable runs on CPU.

    This test executes in a CPU process on purpose: the refusal it asserts is
    the one that stands between a silent CPU fallback and a wall reported
    against a GPU bar.
    """

    with pytest.raises(launcher.ProjectedRootError, match="resolved backend"):
        launcher.bind_gpu_backend()


def test_the_attempt_child_is_launched_with_the_cache_it_will_be_timed_on() -> None:
    """The priming process and the timed process share one cache, provably."""

    argv, environment = launcher.attempt_invocation(
        Path("/tmp/attempt"),
        attempt_index=2,
        iterations=700,
        cache_directory=Path("/tmp/cache"),
        environment={"JAX_PLATFORMS": "cuda"},
    )
    assert "--attempt-child" in argv
    assert argv[2].endswith("run_single_stage_projected_route_gpu_root.py")
    assert environment[launcher.COMPILATION_CACHE_ENVIRONMENT_VARIABLE] == "/tmp/cache"
    assert environment["JAX_PLATFORMS"] == "cuda"


# ---------------------------------------------------------------------- cache


def test_cache_evidence_tells_a_warm_run_from_a_cold_one(tmp_path: Path) -> None:
    empty = launcher.compilation_cache_state(tmp_path)
    assert empty["entry_count"] == 0
    (tmp_path / "entry-a").write_bytes(b"x" * 16)
    populated = launcher.compilation_cache_state(tmp_path)
    assert populated["entry_count"] == 1
    assert populated["total_bytes"] == 16
    assert populated["entries_digest"] != empty["entries_digest"]


# --------------------------------------------------------- publish and revalidate


def _synthetic_ledger(*, gated: bool) -> dict:
    """The ledger scope an attempt must publish, optionally with its verdicts."""

    ledger = {
        **_ledger(),
        "pinned_quality_terms": list(rehearsal.PINNED_ENDPOINT_QUALITY_TERMS),
        "informational_observables": list(
            rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES
        ),
        "gated_at_this_budget": gated,
    }
    if gated:
        ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)
    return ledger


def _root_evidence(
    *,
    verdict: str,
    attempts: list[dict],
    cold_lane: dict | None = None,
    authorized_attempts: int = launcher.PREREGISTERED_ATTEMPTS,
    iterations: int = rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
    cold_lane_authorized: bool = True,
    claim: dict | None = None,
) -> dict:
    """The root receipt shape, with every field re-validation re-derives."""

    return {
        "schema_version": launcher.GPU_ROOT_SCHEMA_VERSION,
        "route": launcher.PROJECTED_ROUTE,
        "verdict": verdict,
        "claim": claim
        if claim is not None
        else {
            "target_objective": rehearsal.NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": rehearsal.NATIVE_WALL_SECONDS_BAR,
            "feasibility_tolerance": (
                rehearsal.CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
            ),
        },
        "attempt_protocol": {
            "authorized_attempts": authorized_attempts,
            "maximum_iterations": iterations,
            "cold_lane_authorized": cold_lane_authorized,
            "conformance": launcher.attempt_protocol_conformance(
                authorized_attempts=authorized_attempts,
                iterations=iterations,
                cold_lane=cold_lane_authorized,
            ),
        },
        "attempts": attempts,
        "cold_lane": cold_lane,
        "timing_boundary": "engine_compile_plus_solve",
    }


def _synthetic_attempt(
    attempt_directory: Path,
    *,
    engine_wall: float,
    terminal_objective: float,
    maximum_feasibility_inf: float | None,
    ledger: dict,
    options_delta: dict | None = None,
) -> dict:
    """One LATCHED attempt with a real terminal-state array behind it."""

    attempt_directory.mkdir(parents=True)
    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    with (attempt_directory / rehearsal.TERMINAL_COORDINATES_FILENAME).open(
        "wb"
    ) as stream:
        np.save(stream, np.asarray(coordinates, dtype=np.float64), allow_pickle=False)
    attempt = _attempt("LATCHED", engine_wall=engine_wall)
    attempt["evidence"] = {
        **attempt["evidence"],
        "problem_identity": {"bound": True, "sha_is_binding": False},
        "lowering_pre_gate": {"budget_independent": True},
        "options": {"objective_target": rehearsal.NATIVE_TARGET_OBJECTIVE},
        "certified_options_delta": {} if options_delta is None else options_delta,
        "endpoint_ledger": ledger,
        "solve": {
            "latched": True,
            "maximum_feasibility_inf": maximum_feasibility_inf,
            "terminal_objective": terminal_objective,
        },
        "endpoint_agreement": {
            "loop_terminal_objective": 4.48e-8,
            "standalone_terminal_objective": 4.48e-8 * (1.0 + 5.0e-16),
            "relative_tolerance": (
                launcher.DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE
            ),
            "absolute_floor": launcher.DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
            "terminal_state_sha256": exact_numeric_tree_sha256(coordinates),
        },
    }
    return attempt


def _publish_synthetic_root(
    root: Path,
    *,
    verdict: str,
    engine_wall: float,
    terminal_objective: float = 4.48e-8,
    maximum_feasibility_inf: float | None = 1.0e-14,
    ledger: dict | None = None,
    options_delta: dict | None = None,
    authorized_attempts: int = launcher.PREREGISTERED_ATTEMPTS,
    iterations: int = rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
) -> Path:
    """A structurally complete root receipt with no GPU behind it.

    Published through the REAL publication path, which now re-validates before
    it seals: a receipt this helper cannot get past ``validate_root_artifact``
    never becomes a sealed artifact at all.
    """

    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=engine_wall,
        terminal_objective=terminal_objective,
        maximum_feasibility_inf=maximum_feasibility_inf,
        ledger=_synthetic_ledger(gated=False) if ledger is None else ledger,
        options_delta=options_delta,
    )
    return launcher.publish_root(
        staging,
        root / "final",
        _root_evidence(
            verdict=verdict,
            attempts=[attempt],
            authorized_attempts=authorized_attempts,
            iterations=iterations,
        ),
    )


def _refusal(root: Path) -> dict:
    return json.loads((root / "staging" / launcher.REFUSAL_FILENAME).read_bytes())


def test_a_published_root_revalidates_from_its_sealed_bytes(tmp_path: Path) -> None:
    published = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=rehearsal.NATIVE_WALL_SECONDS_BAR - 100.0,
    )
    assert stat.S_IMODE(published.stat().st_mode) == 0o555
    for path in published.rglob("*"):
        expected = 0o555 if path.is_dir() else 0o444
        assert stat.S_IMODE(path.stat().st_mode) == expected
    evidence = launcher.validate_root_artifact(published)
    assert evidence["verdict"] == launcher.VERDICT_CLAIM_DISCHARGED


def test_revalidation_gates_publication_and_leaves_the_refusal_unsealed(
    tmp_path: Path,
) -> None:
    """Step 10 is a GATE, not an annotation, and its refusal is recorded.

    Re-validation used to run after ``seal_and_sync`` and ``renameat2``, so a
    refusal left a sealed, immutable 0444 artifact whose ``verdict`` field the
    launcher's own validator had rejected, with the only record on a stderr the
    plan calls volatile.  Judged before the seal, a refusal instead leaves the
    staging tree writable and carrying the refusal that stopped it, and no
    artifact appears at the published name at all.
    """

    with pytest.raises(launcher.ProjectedRootError, match="not the one the"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=rehearsal.NATIVE_WALL_SECONDS_BAR + 100.0,
        )
    assert not (tmp_path / "final").exists()
    staging = tmp_path / "staging"
    # Unsealed: the refusal has to be writable into the tree it describes, so
    # nothing here may carry the 0555/0444 modes a published artifact carries.
    assert stat.S_IMODE(staging.stat().st_mode) & stat.S_IWUSR
    assert stat.S_IMODE(
        (staging / launcher.EVIDENCE_FILENAME).stat().st_mode
    ) & stat.S_IWUSR
    refusal = _refusal(tmp_path)
    assert refusal["schema_version"] == launcher.REFUSAL_SCHEMA_VERSION
    assert refusal["refused_at"] == "pre_seal_revalidation"
    assert refusal["published"] is False
    assert "ProjectedRootError" in refusal["error"]


def test_publication_refuses_a_latch_above_the_native_endpoint_objective(
    tmp_path: Path,
) -> None:
    """``LATCHED`` is a status code; the claim's quality quantity is a number.

    The optimizer sets the status from its OWN configured target, so the
    published objective and the published target are both re-derived against
    the plan's literal rather than trusted through the enum.
    """

    with pytest.raises(launcher.ProjectedRootError, match="above the native endpoint"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=rehearsal.NATIVE_WALL_SECONDS_BAR - 100.0,
            terminal_objective=rehearsal.NATIVE_TARGET_OBJECTIVE * 1.0001,
        )
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("recorded", [None, 1.0e-9])
def test_publication_refuses_a_feasibility_that_is_not_within_the_tolerance(
    tmp_path: Path, recorded: float | None
) -> None:
    """Every comparison against a NaN is false, so ``> tolerance`` fails open.

    A nonfinite worst iterate reaches the receipt as ``null`` -- canonical JSON
    refuses NaN and ``json_scalar`` writes null instead -- and a null is not a
    number under the bound any more than 1e-9 is.  Both readings are refused by
    the contract's own ``<= tolerance``.
    """

    assert rehearsal.json_scalar(float("nan")) is None
    with pytest.raises(launcher.ProjectedRootError, match="infeasible iterate"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            maximum_feasibility_inf=recorded,
        )


def test_publication_refuses_an_attempt_that_narrowed_the_pinned_term_set(
    tmp_path: Path,
) -> None:
    """Quality parity is defined by the campaign, never by the run reporting it."""

    narrowed = _synthetic_ledger(gated=False)
    narrowed["pinned_quality_terms"] = ["observable.iota"]
    with pytest.raises(launcher.ProjectedRootError, match="ledger scope differs"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            ledger=narrowed,
        )


def test_publication_refuses_a_restated_certified_options_delta(
    tmp_path: Path,
) -> None:
    """Same route, asserted: the delta is re-derived from the published options.

    Substitution soundness rests on every budget being the frozen configuration
    with one field replaced.  The artifact published that delta and nothing
    re-derived it, so an attempt could assert a clean delta beside options that
    were not the certified ones.
    """

    with pytest.raises(launcher.ProjectedRootError, match="options delta"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            options_delta={"maximum_iterations": 3},
        )


def test_publication_refuses_a_conformance_label_the_protocol_does_not_derive(
    tmp_path: Path,
) -> None:
    """The label the verdict is conditioned on is recomputed, never read back."""

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        ledger=_synthetic_ledger(gated=False),
    )
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[attempt],
        authorized_attempts=1,
        iterations=3,
    )
    evidence["attempt_protocol"]["conformance"] = launcher.CONFORMANCE_PREREGISTERED
    with pytest.raises(launcher.ProjectedRootError, match="published conformance"):
        launcher.publish_root(staging, tmp_path / "final", evidence)


def test_publication_refuses_an_outcome_its_own_evidence_does_not_derive(
    tmp_path: Path,
) -> None:
    """"Recompute rather than believe" has to reach the classifier, not stop above it.

    The verdict is a function of the outcome strings, and those were read back.
    All three facts ``_attempt_outcome`` consumes -- the evidence document, the
    child's return code and whether it timed out -- are published beside them.
    """

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        ledger=_synthetic_ledger(gated=False),
    )
    attempt["evidence"]["solve"]["latched"] = False
    with pytest.raises(launcher.ProjectedRootError, match="attempt outcome"):
        launcher.publish_root(
            staging,
            tmp_path / "final",
            _root_evidence(
                verdict=launcher.VERDICT_CLAIM_DISCHARGED, attempts=[attempt]
            ),
        )


def test_a_bounded_budget_cannot_publish_a_discharged_claim(tmp_path: Path) -> None:
    """A latch under the bar at ``--iterations 3`` reads ``QUALITY_ONLY``.

    The suite used to ratify the opposite shape: a synthetic root whose
    evidence carried neither ``attempt_protocol`` nor ``quality_claim`` was
    accepted with ``verdict: CLAIM_DISCHARGED``, documenting that the campaign's
    headline verdict needed neither the certified budget nor the per-term
    physics gate.
    """

    with pytest.raises(launcher.ProjectedRootError, match="not the one the"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            authorized_attempts=1,
            iterations=3,
        )
    published = _publish_synthetic_root(
        tmp_path / "second",
        verdict=launcher.VERDICT_QUALITY_ONLY,
        engine_wall=1.0,
        authorized_attempts=1,
        iterations=3,
    )
    evidence = launcher.validate_root_artifact(published)
    assert evidence["verdict"] == launcher.VERDICT_QUALITY_ONLY
    assert evidence["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_BOUNDED_SMOKE
    )


def test_a_gated_ledger_survives_its_own_canonical_round_trip(tmp_path: Path) -> None:
    """The recompute must not manufacture a false reject on an honest root.

    The verdicts are derived once from Python floats at publication and again
    from the JSON floats a reader loads.  Proving those agree is the whole
    licence for recomputing instead of reading back: a band that refused a
    valid endpoint over a round-trip ULP would be the V260/rho-floor class of
    false reject, which this campaign has already paid for three times.
    """

    published = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
        ledger=_synthetic_ledger(gated=True),
    )
    evidence = launcher.validate_root_artifact(published)
    ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
    assert ledger["gated_at_this_budget"] is True
    assert ledger["pinned_term_gate"]["passed"] is True
    assert ledger["pinned_term_gate"]["failed_terms"] == []


def test_publication_recomputes_a_gated_ledgers_verdicts_from_its_terms(
    tmp_path: Path,
) -> None:
    """A published pass on a term whose numbers do not pass is refused."""

    gated = _synthetic_ledger(gated=True)
    assert gated["pinned_term_gate"]["passed"] is True
    gated["terminal"] = {**gated["terminal"], "observable.iota": -0.42 * (1.0 + 1.0e-2)}
    with pytest.raises(launcher.ProjectedRootError, match="not the one its ledger"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            ledger=gated,
        )


def test_validation_refuses_a_root_whose_seal_was_reopened(tmp_path: Path) -> None:
    """0444/0555 is a property of the published bytes, not of the publisher."""

    published = _publish_synthetic_root(
        tmp_path, verdict=launcher.VERDICT_CLAIM_DISCHARGED, engine_wall=1.0
    )
    (published / launcher.EVIDENCE_FILENAME).chmod(0o644)
    with pytest.raises(rehearsal.RehearsalError, match="sealed artifact mode differs"):
        launcher.validate_root_artifact(published)


def test_publication_rejects_a_cold_lane_that_ran_warm(tmp_path: Path) -> None:
    """A cold lane against a populated cache measures nothing it claims to.

    Found by the first real GPU launch: the cache was sampled after the
    identity gate, which pays the point-evaluation compile, so a genuinely cold
    process published ``warm: true``.  The sample now happens before this
    process has traced anything, and a cold lane that still reads warm is a
    defect rather than a documented number.
    """

    staging = tmp_path / "staging"
    (staging / "cold-lane").mkdir(parents=True)
    cold = _attempt("COMPLETED_WITHOUT_LATCH", index=0)
    cold["artifact_relative_path"] = "cold-lane"
    cold["timed_against_bar"] = False
    cold["evidence"] = {
        **cold["evidence"],
        "compilation_cache": {"warm": True},
    }
    evidence = _root_evidence(
        verdict=launcher.verdict_of_gate("solve"),
        attempts=[_attempt("GATE_REFUSED", gate="solve")],
        cold_lane=cold,
    )
    with pytest.raises(launcher.ProjectedRootError, match="populated cache"):
        launcher.publish_root(staging, tmp_path / "final", evidence)


def test_validation_rejects_a_tampered_artifact_tree(tmp_path: Path) -> None:
    published = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
    )
    tampered = published / launcher.EVIDENCE_FILENAME
    tampered.chmod(0o644)
    tampered.write_bytes(tampered.read_bytes() + b" ")
    with pytest.raises(launcher.ProjectedRootError, match="manifest differs"):
        launcher.validate_root_artifact(published)


@pytest.mark.parametrize(
    "field", ["wall_seconds_bar", "target_objective", "feasibility_tolerance"]
)
def test_publication_rejects_a_restated_native_reference(
    tmp_path: Path, field: str
) -> None:
    """An artifact may not move any of the three numbers it is judged against."""

    staging = tmp_path / "staging"
    staging.mkdir()
    claim = {
        "target_objective": rehearsal.NATIVE_TARGET_OBJECTIVE,
        "wall_seconds_bar": rehearsal.NATIVE_WALL_SECONDS_BAR,
        "feasibility_tolerance": (
            rehearsal.CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
        ),
    }
    claim[field] = claim[field] * 2.0
    evidence = _root_evidence(
        verdict=launcher.VERDICT_NO_LATCH, attempts=[], claim=claim
    )
    with pytest.raises(launcher.ProjectedRootError, match="restates the native"):
        launcher.publish_root(staging, tmp_path / "final", evidence)


def test_publication_refuses_to_replace_an_existing_root(tmp_path: Path) -> None:
    _publish_synthetic_root(
        tmp_path, verdict=launcher.VERDICT_CLAIM_DISCHARGED, engine_wall=1.0
    )
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(FileExistsError):
        launcher.publish_root(
            second,
            tmp_path / "final",
            _root_evidence(
                verdict=launcher.verdict_of_gate("attempt_protocol"), attempts=[]
            ),
        )


def test_the_protocol_claims_its_output_root_before_spending_compute(
    tmp_path: Path,
) -> None:
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(launcher.ProjectedRootError, match="already exists"):
        launcher.run_attempt_protocol(occupied, cache_directory=tmp_path / "cache")


def test_the_protocol_refuses_a_cache_that_is_not_cold(tmp_path: Path) -> None:
    """A cache with entries in it makes the published cold lane a fiction."""

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "stale-entry").write_bytes(b"x")
    with pytest.raises(launcher.ProjectedRootError, match="must start empty"):
        launcher.run_attempt_protocol(tmp_path / "root", cache_directory=cache)


# ------------------------------------------------------------------ entry path


def test_the_launcher_entry_path_runs_as_launched_and_refuses_the_cpu(
    tmp_path: Path,
) -> None:
    """Executed as a subprocess with nothing monkeypatched.

    The predecessor route spent a one-shot root on a ``NameError`` in its
    launcher's first phase while its suite was green, because every test in that
    suite imported the module instead of launching it.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(REPOSITORY / "benchmarks/run_single_stage_projected_route_gpu_root.py"),
            "--output-root",
            str(tmp_path / "root"),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--attempts",
            "1",
            "--iterations",
            "1",
        ],
        capture_output=True,
        check=False,
        cwd=REPOSITORY,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "PYTHONPATH": os.pathsep.join(
                (str(REPOSITORY / "src"), str(REPOSITORY))
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
    )
    assert completed.returncode != 0
    assert "resolved backend" in completed.stderr, completed.stderr[-2000:]


def test_the_supervisor_does_not_claim_a_gpu_zero_it_does_not_have() -> None:
    """The supervisor resolves the backend in-process, so it is not GPU-zero.

    Recorded rather than asserted away: an artifact that claimed a GPU-free
    supervisor while holding a CUDA context would be describing a different
    process than the one that ran.
    """

    payload = launcher.supervisor_payload(
        {"backend": "gpu", "device_kind": "NVIDIA GeForce RTX 5090"},
        gpu_uuid="GPU-0000",
        timeout_seconds=3600.0,
        preflight={"visible_gpu_uuids": ["GPU-0000"]},
    )
    assert payload["gpu_zero_asserted"] is False
    assert payload["gpu_uuid"] == "GPU-0000"
    assert payload["attempt_timeout_seconds"] == 3600.0
    assert payload["runtime_identity"]["backend"] == "gpu"
    assert payload["preflight"]["visible_gpu_uuids"] == ["GPU-0000"]
    assert json.loads(canonical_json_bytes(payload)) == payload


def test_the_runtime_identity_records_the_interpreter_that_produced_it() -> None:
    """Plan section 3 pins the warm-cache behaviour to a named venv.

    The artifact recorded jax, jaxlib and the native extension but not which
    interpreter ran, and the child's argv -- the only other carrier -- is
    published as a digest, so the environment was not recoverable from the
    sealed bytes at all.
    """

    identity = launcher.gpu_runtime_identity()
    assert identity["python_executable"] == str(Path(sys.executable).resolve())
    assert identity["python_prefix"] == str(Path(sys.prefix).resolve())


# ----------------------------------------------------- telemetry never vetoes


class _FailingMonitor:
    """A bound monitor whose sampler stored a failure, as the real one does."""

    def __init__(self, *, gpu_uuid: str, pid: int, argv: tuple[str, ...]) -> None:
        self.gpu_uuid = gpu_uuid
        self.identity = SimpleNamespace(pid=pid, start_ticks=4242, argv=argv)

    def finish(self) -> object:
        raise ProcessGpuMemoryMonitorError("nvidia-smi query failed")


def test_a_sampler_failure_is_published_as_evidence_not_raised() -> None:
    """A pure-telemetry failure may not spend the one-shot root.

    ``ProcessGpuMemoryMonitor.finish`` re-raises whatever its polling thread
    stored, and the supervisor called it with no handler between the raise and
    the interpreter: an ``nvidia-smi`` query timed out among the ~10^4 this
    protocol performs, or an unrelated process contributing an ``[N/A]`` row,
    would discard up to four completed GPU runs -- possibly including the latch
    -- with nothing sealed and nothing published.  Process GPU-memory sampling
    is named nowhere in the section 6 gate order; it is absorbed into the
    monitor module's own unavailability union instead.
    """

    argv = ("python", "--attempt-child")
    payload = launcher._gpu_memory_payload(
        _FailingMonitor(gpu_uuid="GPU-0000", pid=4321, argv=argv),
        gpu_uuid="GPU-0000",
        provider_pid=4321,
        argv=argv,
    )
    assert payload["availability"] == "unavailable"
    assert payload["unavailable_reason"] == "sampler-failed"
    assert payload["device_uuid"] == "GPU-0000"
    assert payload["child_pid"] == 4321
    assert payload["child_start_time_ticks"] == 4242
    assert payload["sample_count"] == 0
    assert payload["peak_used_memory_mib"] is None
    assert json.loads(canonical_json_bytes(payload)) == payload


def test_a_monitor_that_cannot_bind_leaves_the_attempt_unobserved() -> None:
    """The same absorption covers the bind and the start, not only the finish.

    ``BoundProcessGpuMemoryMonitor`` reads procfs and refuses an argv mismatch
    or a child that already reached zombie, and those raises sat in the same
    unguarded window.  A PID that cannot exist is the cheapest way to reach it.
    """

    argv = ("python", "--attempt-child")
    assert (
        launcher._start_gpu_memory_monitor(
            gpu_uuid="GPU-0000", provider_pid=2**30, expected_argv=argv
        )
        is None
    )
    payload = launcher._gpu_memory_payload(
        None, gpu_uuid="GPU-0000", provider_pid=2**30, argv=argv
    )
    assert payload["availability"] == "unavailable"
    assert payload["unavailable_reason"] == "sampler-failed"
    assert payload["child_start_time_ticks"] is None


def test_the_unavailable_reason_vocabulary_is_derived_from_its_own_type() -> None:
    """One listing of the reasons; a second would be a twin, and twins drift."""

    assert "sampler-failed" in GPU_MEMORY_UNAVAILABLE_REASONS
    assert GPU_MEMORY_UNAVAILABLE_REASONS == frozenset(
        {"cpu-device", "provider-pid-not-observed", "sampler-failed"}
    )


# ------------------------------------------------------------------- preflight


def test_the_preflight_refuses_a_device_that_is_not_the_one_the_claim_names(
    tmp_path: Path,
) -> None:
    """Three external resources are checked before the first child is spawned.

    Plan section 11: enumerate the failure class, not the instance.  A missing
    data file already spent one root of the predecessor route, and neither the
    device UUID the receipt names nor the NVIDIA tooling the telemetry drives
    was checked at all before this.
    """

    preflight = launcher.preflight_external_resources(gpu_uuid=launcher.GPU_UUID)
    assert launcher.GPU_UUID in preflight["visible_gpu_uuids"]
    assert preflight["native_endpoint_state_content_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
    )
    assert preflight["native_endpoint_state_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256
    )
    with pytest.raises(launcher.ProjectedRootError, match="is not among the visible"):
        launcher.preflight_external_resources(gpu_uuid="GPU-not-this-box")


def test_the_launcher_import_closure_binds_in_a_fresh_interpreter(
    tmp_path: Path,
) -> None:
    """The launcher's own modules, hashed under the root's import topology.

    ``tests/conftest.py`` strips the scikit-build meta-path finder, so every
    in-pytest exercise of the source binding runs under an import topology the
    root's children never have -- and that finder is exactly what spent the
    predecessor's second root.  The launcher's only fresh-interpreter test dies
    one line earlier, at the backend gate, and it imports strictly more
    repository modules than the rehearsal does.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            (
                "import benchmarks.run_single_stage_projected_route_gpu_root as m;"
                "print(len(m.bind_execution_sources(m.REPOSITORY)['bound_modules']))"
            ),
        ],
        capture_output=True,
        check=False,
        cwd=REPOSITORY,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert int(completed.stdout.strip()) > 100


def test_the_artifact_manifest_shape_is_the_rehearsals_enumeration(
    tmp_path: Path,
) -> None:
    """One enumeration rule serves both lanes; only the schema name differs."""

    (tmp_path / "a-file").write_bytes(b"payload")
    payload = rehearsal.artifact_manifest_payload(
        tmp_path, schema_version=launcher.GPU_ROOT_MANIFEST_SCHEMA_VERSION
    )
    assert payload["schema_version"] == launcher.GPU_ROOT_MANIFEST_SCHEMA_VERSION
    assert [entry["relative_path"] for entry in payload["files"]] == ["a-file"]
    assert json.loads(canonical_json_bytes(payload)) == payload
