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
import shutil
import stat
import subprocess
import sys
import tempfile
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


# ------------------------------------------------------- off-tmpfs test storage
#
# Ruling 9's three tests are the only machine evidence for the rule, and under
# this box's DEFAULT environment they were the three that failed: pytest derives
# ``tmp_path`` from ``$TMPDIR``, ``/tmp`` here is tmpfs, and the filesystem-type
# refusal fired before the assertion each test makes -- so an operator running
# the pre-root suite the obvious way saw the newest ruling's tests in red, and
# the EDQUOT-class write probe the ruling is built around was never exercised at
# all.  The suite provisions its own storage instead of depending on the
# runner's shell.


def _off_tmpfs_base() -> Path:
    """The first candidate directory on this box that is not a RAM filesystem."""

    declared = os.environ.get("TMPDIR", "")
    candidates = [
        *([Path(declared)] if declared.strip() else []),
        Path("/var/tmp"),
        Path.home() / ".cache",
    ]
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if launcher.filesystem_type(candidate) not in (
            launcher.REFUSED_STORAGE_FILESYSTEM_TYPES
        ):
            return candidate
    raise RuntimeError(f"no candidate directory off tmpfs among {candidates}")


@pytest.fixture(scope="session")
def off_tmpfs_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A session directory off tmpfs, whatever the runner's ``TMPDIR`` says."""

    base = _off_tmpfs_base() / "projected-route-suite"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(dir=base))
    yield root
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def off_tmpfs_path(off_tmpfs_root: Path, request: pytest.FixtureRequest) -> Path:
    """One test's own directory, guaranteed off tmpfs."""

    path = off_tmpfs_root / request.node.name.replace("/", "_")[:80]
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    attempt = _attempt("LATCHED", engine_wall=112.75)
    attempt["evidence"]["timing_seconds"] = {
        **attempt["evidence"]["timing_seconds"],
        "engine_compile": 12.5,
        "engine_solve": 100.25,
        "engine_wall": 112.75,
    }
    assert launcher.attempt_engine_wall_seconds(attempt) == 112.75 == (12.5 + 100.25)

    # Neither half is free: a receipt deriving the right sum out of a negative
    # compile and an inflated solve has published a wall neither phase cost, and
    # the engine's wall is a strict part of the wall the supervisor observed.
    signed = _attempt("LATCHED", engine_wall=100.0)
    signed["evidence"]["timing_seconds"] = {
        **signed["evidence"]["timing_seconds"],
        "engine_compile": -1.0e6,
        "engine_solve": 1.0e6 + 100.0,
        "engine_wall": 100.0,
    }
    with pytest.raises(launcher.ProjectedRootError, match="not a pair of durations"):
        launcher.attempt_engine_wall_seconds(signed)
    unsupervised = _attempt("LATCHED", engine_wall=100.0)
    unsupervised["supervised_seconds"] = 1.0
    with pytest.raises(launcher.ProjectedRootError, match="supervised wall"):
        launcher.attempt_engine_wall_seconds(unsupervised)

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


def _options_payload(iterations: int) -> dict:
    """Every field of the certified configuration at one budget, as published.

    The whole dataclass, because re-validation checks the KEY SET against
    ``CERTIFIED_ROUTE_OPTIONS.__dataclass_fields__`` before deriving the delta:
    over whatever fields an attempt happened to publish, a truncated options
    block derives an empty delta and passes the substitution-soundness check.
    """

    options = rehearsal.rehearsal_options(iterations)
    return {
        field: rehearsal.json_scalar(getattr(options, field))
        for field in sorted(options.__dataclass_fields__)
    }


def _options_delta(iterations: int) -> dict:
    options = rehearsal.rehearsal_options(iterations)
    return {
        field: rehearsal.json_scalar(getattr(options, field))
        for field in sorted(options.__dataclass_fields__)
        if getattr(options, field) != getattr(rehearsal.CERTIFIED_ROUTE_OPTIONS, field)
    }


def _runtime_identity() -> dict:
    """The identity ``gpu_runtime_identity`` publishes, key for key."""

    return {
        "backend": "gpu",
        "device_count": 1,
        "device_kind": "NVIDIA GeForce RTX 5090",
        "device_platform": "cuda",
        "jax_version": "0.0.0",
        "jaxlib_version": "0.0.0",
        "native_extension_path": "/simsoptpp.so",
        "process_id": 1,
        "python_executable": "/python",
        "python_prefix": "/venv",
    }


def _endpoint_agreement(terminal_state_sha256: str = "0" * 64) -> dict:
    """The whole agreement block ``certify_endpoint_agreement`` publishes."""

    return {
        "loop_terminal_objective": 4.48e-8,
        "standalone_terminal_objective": 4.48e-8 * (1.0 + 5.0e-16),
        "relative_tolerance": launcher.DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        "absolute_floor": launcher.DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
        "terminal_feasibility_inf": 1.0e-14,
        "feasibility_absolute_tolerance": (
            rehearsal.CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
        ),
        "terminal_state_sha256": terminal_state_sha256,
    }


def _cache_state(entry_count: int = 1) -> dict:
    """One cache-state block, in the shape ``compilation_cache_state`` writes."""

    return {
        "entry_count": entry_count,
        "total_bytes": 16 * entry_count,
        "entries_digest": "0" * 64,
    }


def _attempt_cache(*, warm: bool) -> dict:
    """The child's whole cache accounting, not just the ``warm`` flag.

    Re-validation walks the receipt's nested blocks, so a fixture publishing one
    key of five is a fixture asserting a shape the protocol cannot produce --
    which is how a hollow ``compilation_cache`` came to re-validate clean.
    """

    return {
        "configuration": {
            "directory": "/cache",
            "enabled": True,
            "min_entry_size_bytes": launcher.PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
            "min_compile_time_seconds": (
                launcher.PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS
            ),
        },
        "at_entry": _cache_state(1 if warm else 0),
        "before_engine": _cache_state(1 if warm else 0),
        "after": _cache_state(2),
        "warm": warm,
    }


def _solve_payload(
    *, latched: bool, terminal_objective: float, maximum_feasibility_inf: float | None
) -> dict:
    """Every host-side scalar ``_solve_payload`` publishes, in its shape."""

    return {
        "status": 0,
        "status_name": "OBJECTIVE_TARGET_REACHED" if latched else "MAXIMUM_ITERATIONS",
        "latched": latched,
        "iterations_run": 7,
        "terminal_objective": terminal_objective,
        "terminal_feasibility_inf": 1.0e-14,
        "terminal_projected_gradient_inf": 1.0e-7,
        "stored_pairs": 5,
        "projector_materializations": 2,
        "tangency_forced_refreshes": 0,
        "line_search_forced_refreshes": 0,
        "monotone_descent": True,
        "maximum_feasibility_inf": maximum_feasibility_inf,
        "collapse_proximity_margin": 1.0,
        "rows": [],
    }


def _gpu_memory() -> dict:
    """The observation ``_gpu_memory_payload`` normalizes, whole."""

    return {
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "availability": "unavailable",
        "unavailable_reason": "sampler-failed",
        "device_uuid": launcher.GPU_UUID,
        "parent_pid": 1,
        "child_pid": 2,
        "child_start_time_ticks": None,
        "child_argv_sha256": "0" * 64,
        "sample_count": 0,
        "peak_used_memory_mib": None,
    }


def _attempt_evidence(
    *,
    index: int,
    engine_wall: float,
    latched: bool,
    iterations: int,
) -> dict:
    """The complete child document, in the shape ``run_attempt`` returns it."""

    return {
        "schema_version": launcher.GPU_ATTEMPT_SCHEMA_VERSION,
        "route": launcher.PROJECTED_ROUTE,
        "attempt_index": index,
        "environment": {
            **launcher.GPU_REQUIRED_ENVIRONMENT,
            launcher.COMPILATION_CACHE_ENVIRONMENT_VARIABLE: "/cache",
        },
        "runtime_identity": _runtime_identity(),
        "execution_sources": {"bound_modules": []},
        "problem_identity": {"bound": True, "sha_is_binding": False},
        "lowering_pre_gate": {"budget_independent": True},
        "options": _options_payload(iterations),
        "certified_options_delta": _options_delta(iterations),
        "compilation_cache": _attempt_cache(warm=True),
        "solve": _solve_payload(
            latched=latched,
            terminal_objective=4.48e-8,
            maximum_feasibility_inf=1.0e-14,
        ),
        "endpoint_agreement": _endpoint_agreement(),
        "endpoint_ledger": _synthetic_ledger(gated=False),
        "timing_seconds": {
            "bootstrap": 1.0,
            "problem_identity": 1.0,
            "lowering_pre_gate": 1.0,
            "engine_compile": 0.0,
            "engine_solve": engine_wall,
            "engine_wall": engine_wall,
            "attempt_wall": engine_wall + 3.0,
        },
        "timing_boundary": "engine_compile_plus_solve",
        "quality_claim": (
            "CERTIFIED_BUDGET"
            if iterations == rehearsal.CERTIFIED_MAXIMUM_ITERATIONS
            else "NOT_CLAIMED_AT_BOUNDED_BUDGET"
        ),
        "gate_refused": None,
    }


def _attempt(
    outcome: str,
    *,
    index: int = 1,
    engine_wall: float = 100.0,
    gate: str | None = None,
    iterations: int = rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
) -> dict:
    """One supervised attempt in the shape the supervisor publishes it.

    COMPLETE, in both shapes the child has.  Re-validation freezes the key sets
    every block of the receipt is built from, because a truncated document -- a
    producer that raised part-way through ``build_root_evidence``, a partial
    write, a hand-assembled tree -- could not be told from a whole one by a
    validator that indexes into the fields it happens to need.  A fixture that
    publishes less than the supervisor does is a fixture asserting a shape the
    protocol cannot produce.

    The compile half is 0.0 so that ``engine_compile + engine_solve`` reproduces
    the requested wall to the bit -- these fixtures probe the verdict boundary,
    which a rounded split would blur by a ULP at exactly the value that matters.
    """

    evidence: dict | None
    if gate is not None:
        evidence = {
            "schema_version": launcher.GPU_ATTEMPT_SCHEMA_VERSION,
            "route": launcher.PROJECTED_ROUTE,
            "attempt_index": index,
            "gate_refused": gate,
            "error": "ProjectedRootError: the gate refused",
        }
    elif outcome == "PROTOCOL_FAILURE":
        evidence = None
    else:
        evidence = _attempt_evidence(
            index=index,
            engine_wall=engine_wall,
            latched=outcome == "LATCHED",
            iterations=iterations,
        )
    return {
        "attempt_index": index,
        "artifact_relative_path": f"attempts/attempt-{index}",
        "outcome": outcome,
        "return_code": 2 if gate is not None else 0,
        "timed_out": outcome == "TIMEOUT",
        "supervised_seconds": engine_wall + 1.0,
        "argv_sha256": "0" * 64,
        "gpu_memory": _gpu_memory(),
        "stderr_tail": "",
        "stdout_tail": None,
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
    """N, the certified budget and whether the cold lane RAN decide it.

    The third fact is the lane's AUTHORIZATION, never its outcome (plan section
    12.9).  Feeding the outcome in charged a fully conforming run for an
    infrastructure fault on a draw the protocol does not contain -- see
    ``test_an_anomalous_cold_lane_is_published_and_does_not_dispose_the_root``.
    """

    assert (
        launcher.attempt_protocol_conformance(
            authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
            iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            cold_lane_authorized=True,
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
                cold_lane_authorized=cold_lane,
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


def test_a_canonical_document_of_another_shape_is_a_named_protocol_failure() -> None:
    """``latched`` is required to BE a boolean, not indexed for.

    The document reaching the classifier has passed canonical-JSON validity and
    nothing else, and the call sits inside the supervisor's unguarded window: a
    canonical document of a different shape as the child's last stdout line
    raised ``KeyError`` out of ``supervise_attempt``, discarding every completed
    attempt unpublished -- the same hazard ``_parse_attempt_stdout`` repaired
    one level up.
    """

    for solve in ({}, {"latched": None}, {"latched": "yes"}, {"latched": 1}):
        assert (
            launcher._attempt_outcome(
                {"gate_refused": None, "solve": solve},
                return_code=0,
                timed_out=False,
            )
            == "PROTOCOL_FAILURE"
        )


# ------------------------------------------------------- pinned-term ledger gate


def _ledger(**overrides: float) -> dict:
    """A ledger whose terminal side equals native, then perturbed by name.

    The native side is the CAMPAIGN's, not an invented one: re-validation now
    judges a receipt's native side against ``NATIVE_ENDPOINT_PINNED_TERMS``, so
    a fixture with a made-up reference would assert a shape no honest artifact
    can carry.  The perturbations below are relative to the real values and the
    band arithmetic is unchanged by the substitution.
    """

    native = {
        **rehearsal.NATIVE_ENDPOINT_PINNED_TERMS,
        "observable.G": 13.887472087505376,
        "state.G": 13.887472087505376,
    }
    terminal = {**native, **overrides}
    return {"terminal": terminal, "native": native}


def _scaled(name: str, factor: float) -> float:
    """One pinned term's native value, moved by a relative factor."""

    return rehearsal.NATIVE_ENDPOINT_PINNED_TERMS[name] * factor


def test_an_endpoint_that_beat_native_on_non_qs_is_not_refused() -> None:
    """The banked Q1 latch came out 0.9% BETTER than native on non-QS.

    A two-sided relative band on the quality terms would refuse the very
    evidence this campaign banked -- the false-reject class the V260 shell gate
    and the SQP rho floor already cost two verdicts.
    """

    verdict = rehearsal.gate_endpoint_ledger(
        _ledger(
            **{
                "raw.non_qs": _scaled("raw.non_qs", 0.991),
                "observable.non_qs_ratio": _scaled("observable.non_qs_ratio", 0.991),
            }
        )
    )
    assert verdict["passed"] is True
    assert verdict["terms"]["raw.non_qs"]["comparison"] == "not_worse"
    assert verdict["terms"]["raw.non_qs"]["measured"] < 0.0


def test_an_endpoint_materially_worse_than_native_is_refused() -> None:
    """``not_worse`` is one-sided, not absent."""

    verdict = rehearsal.gate_endpoint_ledger(
        _ledger(**{"raw.non_qs": _scaled("raw.non_qs", 1.01)})
    )
    assert verdict["passed"] is False
    assert verdict["failed_terms"] == ["raw.non_qs"]


def test_geometry_terms_are_gated_relatively_and_residuals_absolutely() -> None:
    """Both sides of a machine-zero residual sit below any meaningful ratio."""

    assert rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.iota": _scaled("observable.iota", 1.0 + 1.0e-5)})
    )["passed"]
    assert not rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.iota": _scaled("observable.iota", 1.0 + 1.0e-2)})
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
        _ledger(
            **{"observable.total_length": _scaled("observable.total_length", 0.99)}
        )
    )
    assert shorter["passed"] is True
    longer = rehearsal.gate_endpoint_ledger(
        _ledger(
            **{"observable.total_length": _scaled("observable.total_length", 1.01)}
        )
    )
    assert longer["failed_terms"] == ["observable.total_length"]


def test_the_free_direction_G_is_never_gated() -> None:
    """Nothing in the shared objective pins the net poloidal current."""

    verdict = rehearsal.gate_endpoint_ledger(
        _ledger(
            **{
                "observable.G": 13.887472087505376 * 0.99,
                "state.G": 13.887472087505376 * 0.99,
            }
        )
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
        cold_lane=_attempt("COMPLETED_WITHOUT_LATCH", index=0),
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
        Path("/var/tmp/attempt"),
        attempt_index=2,
        iterations=700,
        cache_directory=Path("/var/tmp/cache"),
        environment={"JAX_PLATFORMS": "cuda", "TMPDIR": "/an/inherited/place"},
        temporary_directory=Path("/var/tmp/spill"),
    )
    assert "--attempt-child" in argv
    assert argv[2].endswith("run_single_stage_projected_route_gpu_root.py")
    assert environment[launcher.COMPILATION_CACHE_ENVIRONMENT_VARIABLE] == (
        "/var/tmp/cache"
    )
    assert environment["JAX_PLATFORMS"] == "cuda"
    # SET, not forwarded: the supervisor preflighted one directory, and a child
    # spilling somewhere else would enforce plan section 11's rule against a
    # directory nobody used.
    assert environment[launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE] == (
        "/var/tmp/spill"
    )


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


def _synthetic_ledger(*, gated: bool, **overrides: float) -> dict:
    """The COMPLETE ledger an attempt publishes, optionally with its verdicts.

    Whole, because re-validation walks it: the two digests of ruling 6 and the
    reference's filename are compared to the frozen constants, and the three
    term maps must carry the campaign's terms.
    """

    rows = _ledger(**overrides)
    ledger = {
        **rows,
        "relative_difference": {
            name: (
                abs(rows["terminal"][name] - rows["native"][name])
                / abs(rows["native"][name])
                if rows["native"][name] != 0.0
                else None
            )
            for name in sorted(rows["terminal"])
        },
        "native_state_relative_path": rehearsal.NATIVE_ENDPOINT_STATE_PATH.name,
        "native_state_sha256": rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256,
        "native_state_content_sha256": (
            rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
        ),
        "pinned_quality_terms": list(rehearsal.PINNED_ENDPOINT_QUALITY_TERMS),
        "informational_observables": list(
            rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES
        ),
        "gated_at_this_budget": gated,
    }
    if gated:
        ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)
    return ledger


def _storage_probe(role: str) -> dict:
    """One probe record, in the shape ``probe_writable_storage`` returns it."""

    return {
        "role": role,
        "directory": f"/var/tmp/{role}",
        "resolved_directory": f"/var/tmp/{role}",
        "filesystem_type": "ext4",
        "device_id": 66306,
        "one_byte_write": "ok",
        "advisory_available_bytes": 1 << 40,
    }


def _preflight() -> dict:
    """The WHOLE preflight, including ruling 6's digests and ruling 9's record.

    The fixture used to publish ``preflight: {}`` and every published-root test
    flowed through it, so the suite asserted that a ``CLAIM_DISCHARGED`` root
    needs no native-reference digests, no device inventory and no storage
    evidence at all -- one level below the frozen key set that was supposed to
    prevent exactly that.
    """

    return {
        "gpu_inventory_executable": "/usr/bin/nvidia-smi",
        "visible_gpu_uuids": [launcher.GPU_UUID],
        "native_endpoint_state_path": str(rehearsal.NATIVE_ENDPOINT_STATE_PATH),
        "native_endpoint_state_sha256": (
            rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256
        ),
        "native_endpoint_state_content_sha256": (
            rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
        ),
        "temporary_directory": "/var/tmp/temporary",
        "resolved_temporary_directory": "/var/tmp/temporary",
        "storage": [
            _storage_probe("temporary"),
            _storage_probe("compilation_cache"),
            _storage_probe("output"),
        ],
    }


def _supervisor() -> dict:
    """The supervisor block ``supervisor_payload`` writes, key for key."""

    return {
        "runtime_identity": _runtime_identity(),
        "gpu_uuid": launcher.GPU_UUID,
        "attempt_timeout_seconds": launcher.ATTEMPT_TIMEOUT_SECONDS,
        "gpu_zero_asserted": False,
        "preflight": _preflight(),
    }


def _root_evidence(
    *,
    verdict: str,
    attempts: list[dict],
    cold_lane: dict | None = None,
    authorized_attempts: int = launcher.PREREGISTERED_ATTEMPTS,
    iterations: int = rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
    cold_lane_authorized: bool | None = None,
    claim: dict | None = None,
) -> dict:
    """The root receipt shape, with every field re-validation re-derives.

    The COMPLETE shape, matching ``build_root_evidence`` key for key.  The
    eight-key document this helper used to build re-validated clean while
    missing its source snapshot, its supervisor block (and therefore the whole
    preflight, including the native-reference digests), its cache accounting,
    its quality claim and every draw statistic of section 4 -- so the suite
    documented that the campaign's headline verdict needed none of them.

    ``cold_lane_authorized`` defaults to whether a lane was published, because
    the flag and the lane are one fact told twice, and conformance is derived
    from what the lane MEASURED rather than from the flag.
    """

    authorized = cold_lane is not None if cold_lane_authorized is None else (
        cold_lane_authorized
    )
    latched = [attempt for attempt in attempts if attempt["outcome"] == "LATCHED"]
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
            "preregistered_attempts": launcher.PREREGISTERED_ATTEMPTS,
            "authorized_attempts": authorized_attempts,
            "attempts_run": len(attempts),
            "stop_rule": launcher.ATTEMPT_STOP_RULE,
            "latch_count": len(latched),
            "latch_rate": f"{len(latched)}/{authorized_attempts}",
            "cold_lane_authorized": authorized,
            "conformance": launcher.attempt_protocol_conformance(
                authorized_attempts=authorized_attempts,
                iterations=iterations,
                cold_lane_authorized=authorized,
            ),
            "maximum_iterations": iterations,
            "certified_maximum_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        },
        "attempts": attempts,
        "cold_lane": cold_lane,
        "cold_lane_anomaly": launcher.cold_lane_anomaly(cold_lane),
        "compilation_cache": _cache_state(),
        "source_snapshot": {
            "relative_path": "source-snapshot",
            "manifest_sha256": "0" * 64,
            "entry_count": 1,
            "worktree": {
                "git_head": "0" * 40,
                "tracked_diff_sha256": "0" * 64,
                "untracked_bytes_manifest_sha256": "0" * 64,
                "repo_root": str(REPOSITORY),
            },
        },
        "supervisor": _supervisor(),
        "quality_claim": (
            "CERTIFIED_BUDGET"
            if iterations == rehearsal.CERTIFIED_MAXIMUM_ITERATIONS
            else "NOT_CLAIMED_AT_BOUNDED_BUDGET"
        ),
        "timing_boundary": "engine_compile_plus_solve",
        "timing_seconds": {"chain_wall": 1.0},
    }


def _synthetic_attempt(
    attempt_directory: Path,
    *,
    engine_wall: float,
    terminal_objective: float,
    maximum_feasibility_inf: float | None,
    ledger: dict | None = None,
    options_delta: dict | None = None,
    outcome: str = "LATCHED",
    index: int = 1,
    relative_path: str | None = None,
    iterations: int = rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
    warm: bool = True,
) -> dict:
    """One attempt with a real terminal-state array behind it.

    The ledger defaults to the one ``endpoint_ledger_is_gated`` says this
    attempt must carry, because that is now RE-DERIVED at re-validation rather
    than read: a latch at the certified budget publishes per-term verdicts, and
    anything else publishes the ledger ungated.
    """

    attempt_directory.mkdir(parents=True)
    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    with (attempt_directory / rehearsal.TERMINAL_COORDINATES_FILENAME).open(
        "wb"
    ) as stream:
        np.save(stream, np.asarray(coordinates, dtype=np.float64), allow_pickle=False)
    attempt = _attempt(
        outcome, index=index, engine_wall=engine_wall, iterations=iterations
    )
    if relative_path is not None:
        attempt["artifact_relative_path"] = relative_path
    gated = rehearsal.endpoint_ledger_is_gated(
        iterations=iterations, latched=outcome == "LATCHED"
    )
    attempt["evidence"] = {
        **attempt["evidence"],
        "certified_options_delta": (
            _options_delta(iterations) if options_delta is None else options_delta
        ),
        "compilation_cache": _attempt_cache(warm=warm),
        "endpoint_ledger": _synthetic_ledger(gated=gated) if ledger is None else ledger,
        "solve": _solve_payload(
            latched=outcome == "LATCHED",
            terminal_objective=terminal_objective,
            maximum_feasibility_inf=maximum_feasibility_inf,
        ),
        "endpoint_agreement": _endpoint_agreement(
            exact_numeric_tree_sha256(coordinates)
        ),
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
    cold_lane: bool = True,
) -> Path:
    """A structurally complete root receipt with no GPU behind it.

    Published through the REAL publication path, which now re-validates before
    it seals: a receipt this helper cannot get past ``validate_root_artifact``
    never becomes a sealed artifact at all.

    A cold lane is published by default because a PREREGISTERED protocol has
    one -- section 3 pre-registers it beside N and the budget, and conformance
    is now derived from the measurement the lane produced.
    """

    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=engine_wall,
        terminal_objective=terminal_objective,
        maximum_feasibility_inf=maximum_feasibility_inf,
        ledger=ledger,
        options_delta=options_delta,
        iterations=iterations,
    )
    cold: dict | None = None
    if cold_lane:
        cold = _synthetic_attempt(
            staging / launcher.COLD_LANE_DIRECTORY,
            engine_wall=engine_wall,
            terminal_objective=terminal_objective,
            maximum_feasibility_inf=1.0e-14,
            outcome="COMPLETED_WITHOUT_LATCH",
            index=0,
            relative_path=launcher.COLD_LANE_DIRECTORY,
            iterations=iterations,
            warm=False,
        )
        cold["timed_against_bar"] = False
    return launcher.publish_root(
        staging,
        root / "final",
        _root_evidence(
            verdict=verdict,
            attempts=[attempt],
            cold_lane=cold,
            authorized_attempts=authorized_attempts,
            iterations=iterations,
        ),
    )


def _refusal(root: Path) -> dict:
    return json.loads((root / "staging" / launcher.REFUSAL_FILENAME).read_bytes())


def test_a_published_root_revalidates_from_its_sealed_bytes(tmp_path: Path) -> None:
    """A COMPLETE, gated, passing root round-trips through the real seal.

    This test used to publish and accept a ``CLAIM_DISCHARGED`` root at
    ``iterations = 700`` whose ledger was UNGATED -- ``PREREGISTERED``
    conformance with no per-term physics gate anywhere in it -- and asserted
    that re-validation accepted it.  That is the pathology the previous
    remediation named as the reason its own defect went unseen ("the suite
    ratified that shape rather than refusing it"), reproduced inside the fix for
    it.  The shape is refused below; what is accepted here is the honest one.
    """

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
    ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
    assert ledger["gated_at_this_budget"] is True
    assert ledger["pinned_term_gate"]["passed"] is True


def test_a_discharged_root_whose_physics_gate_never_ran_is_refused(
    tmp_path: Path,
) -> None:
    """Section 1.1's gate is not optional on the attempt that discharges.

    Ruling 1 made a runtime boolean the switch that decides whether the per-term
    physics gate runs at all, and re-validation READ that boolean: a sealed
    ``CLAIM_DISCHARGED`` root at ``PREREGISTERED`` conformance whose latching
    attempt carried ``gated_at_this_budget: false`` and no ``pinned_term_gate``
    at all published, sealed 0444, ``renameat2``-ed, and re-validated clean.
    The boolean is a pure function of the budget and the outcome, and it is now
    asked of the owner both lanes ask.
    """

    with pytest.raises(launcher.ProjectedRootError, match="not what its budget"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            ledger=_synthetic_ledger(gated=False),
        )
    assert not (tmp_path / "final").exists()


def test_a_discharged_root_whose_physics_gate_failed_is_refused(
    tmp_path: Path,
) -> None:
    """A receipt may not certify its own contradiction.

    The only ledger check was that a published gate EQUALS its own
    recomputation, which a faithfully recorded FAILURE satisfies.  So a root
    could discharge the campaign's headline claim while carrying, in the same
    sealed tree, the per-term verdict that the endpoint did not reach native's
    physics.
    """

    failed = _synthetic_ledger(gated=False)
    failed["terminal"] = {**failed["terminal"], "raw.non_qs": 3.6e-4 * 1.01}
    failed["gated_at_this_budget"] = True
    failed["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(failed)
    assert failed["pinned_term_gate"]["passed"] is False
    with pytest.raises(launcher.ProjectedRootError, match="failed pinned-term gate"):
        _publish_synthetic_root(
            tmp_path,
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            engine_wall=1.0,
            ledger=failed,
        )
    assert not (tmp_path / "final").exists()


def test_a_discharged_root_must_have_run_the_budget_its_label_claims(
    tmp_path: Path,
) -> None:
    """Conformance is derived from a budget nothing tied to the attempts.

    ``attempt_protocol.maximum_iterations: 700`` forced ``PREREGISTERED`` while
    the attempt's own ``options.maximum_iterations`` said 400, so the exact
    defect the previous remediation closed in the launcher -- a bounded run
    minting the headline verdict -- stayed reachable through the validator the
    plan then promoted to the gate on publication.
    """

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        iterations=400,
    )
    cold = _synthetic_attempt(
        staging / launcher.COLD_LANE_DIRECTORY,
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        outcome="COMPLETED_WITHOUT_LATCH",
        index=0,
        relative_path=launcher.COLD_LANE_DIRECTORY,
        iterations=400,
        warm=False,
    )
    cold["timed_against_bar"] = False
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED, attempts=[attempt], cold_lane=cold
    )
    with pytest.raises(launcher.ProjectedRootError, match="iterations, not the"):
        launcher.publish_root(staging, tmp_path / "final", evidence)
    assert not (tmp_path / "final").exists()


def _refuse_published(root: Path, evidence: dict, *, match: str) -> None:
    """Publish one receipt through the real path and require it to be refused.

    Either refusal type counts: the campaign's shared owners live in the
    rehearsal module and raise its error, which the validator has always been
    able to surface (``gate_endpoint_ledger`` does), and the refusal record
    names the type it caught either way.
    """

    with pytest.raises((launcher.ProjectedRootError, rehearsal.RehearsalError), match=match):
        launcher.publish_root(root / "staging", root / "final", evidence)
    assert not (root / "final").exists()


def _mutated_root(tmp_path: Path, name: str, mutate) -> tuple[Path, dict]:
    """A complete, otherwise-honest ``CLAIM_DISCHARGED`` receipt, then mutated."""

    root = tmp_path / name
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    cold = _synthetic_attempt(
        staging / launcher.COLD_LANE_DIRECTORY,
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        outcome="COMPLETED_WITHOUT_LATCH",
        index=0,
        relative_path=launcher.COLD_LANE_DIRECTORY,
        warm=False,
    )
    cold["timed_against_bar"] = False
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED, attempts=[attempt], cold_lane=cold
    )
    mutate(evidence)
    return root, evidence


def test_a_hollow_custody_block_cannot_pass_for_a_published_one(
    tmp_path: Path,
) -> None:
    """Completeness reaches BELOW the top level, or it reaches nothing.

    Six mutations published through the real ``publish_root`` and re-validated
    clean at the previous revision, including a ``CLAIM_DISCHARGED`` root with
    no preflight block at all -- so ruling 7's promise that "a receipt missing
    its source snapshot, supervisor block, preflight, cache accounting or
    telemetry cannot pass for a whole one" was true of the names and false of
    everything under them.  PRESENT-BUT-NULL was equivalent to absent.
    """

    def drop_preflight(evidence: dict) -> None:
        del evidence["supervisor"]["preflight"]

    def hollow_supervisor(evidence: dict) -> None:
        evidence["supervisor"] = {"gpu_uuid": launcher.GPU_UUID}

    def empty_preflight(evidence: dict) -> None:
        evidence["supervisor"]["preflight"] = {}

    def null_snapshot(evidence: dict) -> None:
        evidence["source_snapshot"] = {"relative_path": "source-snapshot"}

    def null_cache(evidence: dict) -> None:
        evidence["compilation_cache"] = None

    def empty_timing(evidence: dict) -> None:
        evidence["timing_seconds"] = {}

    def extra_claim_key(evidence: dict) -> None:
        evidence["claim"]["A_KEY_NO_PRODUCER_EMITS"] = 1.0

    def null_telemetry(evidence: dict) -> None:
        evidence["attempts"][0]["gpu_memory"] = None

    def hollow_child_cache(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["compilation_cache"] = {"warm": True}

    def hollow_child_solve(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"] = {
            "latched": True,
            "terminal_objective": 4.48e-8,
            "maximum_feasibility_inf": 1.0e-14,
        }

    for name, mutate in (
        ("drop_preflight", drop_preflight),
        ("hollow_supervisor", hollow_supervisor),
        ("empty_preflight", empty_preflight),
        ("null_snapshot", null_snapshot),
        ("null_cache", null_cache),
        ("empty_timing", empty_timing),
        ("extra_claim_key", extra_claim_key),
        ("null_telemetry", null_telemetry),
        ("hollow_child_cache", hollow_child_cache),
        ("hollow_child_solve", hollow_child_solve),
    ):
        root, evidence = _mutated_root(tmp_path, name, mutate)
        _refuse_published(root, evidence, match="incomplete|not a document")


def test_a_receipt_may_not_supply_the_reference_its_physics_gate_is_judged_against(
    tmp_path: Path,
) -> None:
    """The forged-native escape: ``terminal == native`` on all ten pinned terms.

    Ruling 7 bound whether the per-term gate RAN and whether it PASSED, and not
    what it ran against.  ``gate_endpoint_ledger`` takes both sides from the
    document it is handed, so a ledger publishing ``1.0`` for every pinned term
    on both sides recomputed each verdict to ``measured = 0.0, passed = true``
    beside ``solve.terminal_objective = 4.48e-8`` -- an internal contradiction
    nothing noticed -- and sealed as ``CLAIM_DISCHARGED``.
    """

    def forged_native(evidence: dict) -> None:
        forged = {name: 1.0 for name in rehearsal.PINNED_ENDPOINT_QUALITY_TERMS}
        forged.update(
            {name: 1.0 for name in rehearsal.INFORMATIONAL_ENDPOINT_OBSERVABLES}
        )
        ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
        ledger["terminal"] = dict(forged)
        ledger["native"] = dict(forged)
        ledger["relative_difference"] = {name: 0.0 for name in forged}
        ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)

    root, evidence = _mutated_root(tmp_path, "forged_native", forged_native)
    # The gate the receipt published is self-consistent and passes; what refuses
    # it is the reference.
    assert evidence["attempts"][0]["evidence"]["endpoint_ledger"]["pinned_term_gate"][
        "passed"
    ]
    _refuse_published(root, evidence, match="not the campaign's frozen native")

    def drifted_native(evidence: dict) -> None:
        ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
        ledger["native"] = {
            **ledger["native"],
            "observable.iota": ledger["native"]["observable.iota"] * (1.0 + 1.0e-5),
        }
        ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)

    root, evidence = _mutated_root(tmp_path, "drifted_native", drifted_native)
    _refuse_published(root, evidence, match="not the campaign's frozen native")

    def restated_ledger_digest(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["endpoint_ledger"][
            "native_state_sha256"
        ] = "f" * 64

    root, evidence = _mutated_root(
        tmp_path, "restated_ledger_digest", restated_ledger_digest
    )
    _refuse_published(root, evidence, match="another native endpoint reference")

    def restated_preflight_digest(evidence: dict) -> None:
        evidence["supervisor"]["preflight"]["native_endpoint_state_content_sha256"] = (
            "e" * 64
        )

    root, evidence = _mutated_root(
        tmp_path, "restated_preflight_digest", restated_preflight_digest
    )
    _refuse_published(root, evidence, match="other than the campaign's pinned one")

    def restated_storage(evidence: dict) -> None:
        evidence["supervisor"]["preflight"]["storage"][0]["filesystem_type"] = "tmpfs"

    root, evidence = _mutated_root(tmp_path, "restated_storage", restated_storage)
    _refuse_published(root, evidence, match="which plan section 11 refuses")

    def unseen_device(evidence: dict) -> None:
        evidence["supervisor"]["preflight"]["visible_gpu_uuids"] = ["GPU-0000dead"]

    root, evidence = _mutated_root(tmp_path, "unseen_device", unseen_device)
    _refuse_published(root, evidence, match="did not see the device")


def test_the_published_stop_rule_is_enforced_against_the_attempt_sequence(
    tmp_path: Path,
) -> None:
    """``latch_rate: 3/3`` on a loop that breaks after the first latch.

    ``ATTEMPT_STOP_RULE`` was re-derived as a STRING LITERAL and the statistics
    were re-derived from an attempt list nothing constrained, so a receipt could
    publish three latching attempts on a protocol that can produce at most one,
    or four draws under three authorized with the rate reported over the
    denominator section 4 names.
    """

    def three_latches(evidence: dict) -> None:
        first = evidence["attempts"][0]
        evidence["attempts"] = [
            first,
            {**first, "attempt_index": 2, "artifact_relative_path": "attempts/attempt-2"},
            {**first, "attempt_index": 3, "artifact_relative_path": "attempts/attempt-3"},
        ]
        evidence["attempt_protocol"]["attempts_run"] = 3
        evidence["attempt_protocol"]["latch_count"] = 3
        evidence["attempt_protocol"]["latch_rate"] = "3/3"

    root, evidence = _mutated_root(tmp_path, "three_latches", three_latches)
    _refuse_published(root, evidence, match="does not obey the published stop rule")

    def four_draws(evidence: dict) -> None:
        first = evidence["attempts"][0]
        miss = {
            **first,
            "outcome": "COMPLETED_WITHOUT_LATCH",
            "evidence": {
                **first["evidence"],
                "solve": {**first["evidence"]["solve"], "latched": False},
                "endpoint_ledger": _synthetic_ledger(gated=False),
            },
        }
        evidence["attempts"] = [
            {**miss, "attempt_index": index, "artifact_relative_path": f"attempts/attempt-{index}"}
            for index in (1, 2, 3)
        ] + [{**first, "attempt_index": 4, "artifact_relative_path": "attempts/attempt-4"}]
        evidence["attempt_protocol"]["attempts_run"] = 4

    root, evidence = _mutated_root(tmp_path, "four_draws", four_draws)
    _refuse_published(root, evidence, match="attempts under")

    # A draw on disk the receipt does not publish is a suppressed draw.
    root = tmp_path / "suppressed"
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    (staging / "attempts" / "attempt-2").mkdir(parents=True)
    evidence = _root_evidence(
        verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt]
    )
    _refuse_published(root, evidence, match="the receipt does not publish")


def test_an_attempt_may_not_declare_an_execution_context_it_did_not_run_in(
    tmp_path: Path,
) -> None:
    """The certified wall belongs to a GPU child, and the receipt has to say so.

    The ROOT's schema, route and timing boundary were re-derived; the attempt's
    were not, so a ``CLAIM_DISCHARGED`` receipt could name the CPU as the backend
    that produced the wall, declare another timing boundary, or carry another
    route's schema on the document the verdict is derived from.
    """

    def cpu_backend(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["runtime_identity"]["backend"] = "cpu"

    root, evidence = _mutated_root(tmp_path, "cpu_backend", cpu_backend)
    _refuse_published(root, evidence, match="not the 'gpu' the wall is claimed on")

    def other_boundary(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["timing_boundary"] = "attempt_wall"

    root, evidence = _mutated_root(tmp_path, "other_boundary", other_boundary)
    _refuse_published(root, evidence, match="a different run than the record")

    def other_index(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["attempt_index"] = 7

    root, evidence = _mutated_root(tmp_path, "other_index", other_index)
    _refuse_published(root, evidence, match="a different run than the record")

    def forbidden_environment(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["environment"]["JAX_PLATFORMS"] = "cpu"

    root, evidence = _mutated_root(
        tmp_path, "forbidden_environment", forbidden_environment
    )
    _refuse_published(root, evidence, match="environment the route forbids")


def test_revalidation_asserts_the_precision_it_re_derives_in(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A third party without x64 gets a named refusal, not a false one.

    The terminal-state digest is re-derived through ``jnp.asarray(..., float64)``,
    which with x64 disabled downcasts to float32 WITHOUT RAISING -- so a valid
    sealed root was refused with "published terminal state differs from its
    hash", a message indicting the artifact rather than the reader's shell.
    """

    published = _publish_synthetic_root(
        tmp_path, verdict=launcher.VERDICT_CLAIM_DISCHARGED, engine_wall=1.0
    )
    monkeypatch.setattr(
        launcher, "jax", SimpleNamespace(config=SimpleNamespace(jax_enable_x64=False))
    )
    with pytest.raises(launcher.ProjectedRootError, match="JAX_ENABLE_X64"):
        launcher.validate_root_artifact(published)


def test_the_frozen_nested_shapes_are_the_ones_the_producers_write() -> None:
    """Every nested shape is bound to the function that writes it, not a fixture.

    A frozen key set asserted against the suite's own helper is a twin, and
    twins drift -- which is exactly how the fixture publishing ``preflight: {}``
    came to certify that a discharged root needs no preflight at all.
    """

    assert frozenset(launcher.CACHE_STATE_SHAPE) == frozenset(
        launcher.compilation_cache_state(REPOSITORY / "does-not-exist")
    )
    assert frozenset(launcher.CACHE_CONFIGURATION_SHAPE) == frozenset(
        _attempt_cache(warm=False)["configuration"]
    )
    assert frozenset(launcher.ROOT_CLAIM_SHAPE) == frozenset(
        launcher.build_root_evidence(
            attempts=[],
            cold_lane=None,
            snapshot={},
            supervisor={},
            authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
            iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            cold_lane_authorized=True,
            cache={},
            verdict=launcher.VERDICT_NO_LATCH,
            chain_seconds=1.0,
        )["claim"]
    )
    assert frozenset(launcher.SUPERVISOR_SHAPE) == frozenset(
        launcher.supervisor_payload(
            {}, gpu_uuid=launcher.GPU_UUID, timeout_seconds=1.0, preflight={}
        )
    )
    assert frozenset(launcher.WORKTREE_IDENTITY_SHAPE) == frozenset(
        launcher.capture_worktree_identity(REPOSITORY).to_payload()
    )
    assert frozenset(launcher.COLD_LANE_ANOMALY_SHAPE) == frozenset(
        launcher.cold_lane_anomaly(_attempt("TIMEOUT", index=0))
    )
    # The three that are only reachable with a device stay bound by the shapes
    # the bounded GPU smoke publishes through, which is what makes the smoke a
    # producer test rather than a liveness check.
    assert frozenset(launcher.RUNTIME_IDENTITY_SHAPE) == frozenset(_runtime_identity())
    assert frozenset(launcher.GPU_MEMORY_SHAPE) == frozenset(_gpu_memory())
    assert frozenset(launcher.PREFLIGHT_SHAPE) == frozenset(_preflight())
    assert frozenset(launcher.STORAGE_PROBE_SHAPE) == frozenset(
        _storage_probe("temporary")
    )


def test_a_truncated_receipt_cannot_pass_for_a_complete_one(tmp_path: Path) -> None:
    """Every block section 6 builds has a frozen key set.

    ``validate_root_artifact`` checked a schema string and then indexed into
    whatever fields it needed, so a ``CLAIM_DISCHARGED`` root with no source
    snapshot, no supervisor block (and therefore no preflight and no native
    reference digests), no cache accounting, no quality claim and no draw
    statistics re-validated clean and was indistinguishable from a whole one.
    """

    for dropped in (
        "source_snapshot",
        "supervisor",
        "quality_claim",
        "compilation_cache",
        "timing_seconds",
    ):
        root = tmp_path / dropped
        staging = root / "staging"
        attempt = _synthetic_attempt(
            staging / "attempts" / "attempt-1",
            engine_wall=1.0,
            terminal_objective=4.48e-8,
            maximum_feasibility_inf=1.0e-14,
        )
        evidence = _root_evidence(
            verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt]
        )
        del evidence[dropped]
        with pytest.raises(launcher.ProjectedRootError, match="not the complete receipt"):
            launcher.publish_root(staging, root / "final", evidence)

    root = tmp_path / "protocol"
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    evidence = _root_evidence(verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt])
    del evidence["attempt_protocol"]["latch_rate"]
    with pytest.raises(launcher.ProjectedRootError, match="protocol block is incomplete"):
        launcher.publish_root(staging, root / "final", evidence)

    root = tmp_path / "attempt"
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    del attempt["gpu_memory"]
    evidence = _root_evidence(verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt])
    with pytest.raises(launcher.ProjectedRootError, match="record is incomplete"):
        launcher.publish_root(staging, root / "final", evidence)

    root = tmp_path / "child"
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    del attempt["evidence"]["execution_sources"]
    evidence = _root_evidence(verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt])
    with pytest.raises(launcher.ProjectedRootError, match="document is incomplete"):
        launcher.publish_root(staging, root / "final", evidence)


def test_the_frozen_key_sets_are_the_ones_the_receipt_builder_writes() -> None:
    """The completeness check is bound to the producer, not to a second listing.

    A frozen key set that drifts from ``build_root_evidence`` is a twin, and
    twins drift -- this one would drift into refusing every honest root.
    """

    built = launcher.build_root_evidence(
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
    assert frozenset(built) == launcher.ROOT_EVIDENCE_REQUIRED_KEYS
    assert frozenset(built["attempt_protocol"]) == (
        launcher.ATTEMPT_PROTOCOL_REQUIRED_KEYS
    )
    assert frozenset(_attempt("LATCHED")) == launcher.SUPERVISED_ATTEMPT_REQUIRED_KEYS
    assert frozenset(_attempt("LATCHED")["evidence"]) == (
        launcher.ATTEMPT_EVIDENCE_REQUIRED_KEYS
    )
    assert frozenset(_attempt("GATE_REFUSED", gate="solve")["evidence"]) == (
        launcher.REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS
    )


def test_the_draw_statistics_are_derived_from_the_attempts(tmp_path: Path) -> None:
    """Section 4's k/N was a read-back beside a conformance label that was not.

    A one-attempt root could publish ``latch_rate: 3/3``, ``latch_count: 99``
    and ``attempts_run: 7`` and re-validate clean, because the validator read
    only ``authorized_attempts``, ``maximum_iterations``, ``cold_lane_authorized``
    and ``conformance``.
    """

    for field, value in (
        ("latch_rate", "3/3"),
        ("latch_count", 99),
        ("attempts_run", 7),
        ("preregistered_attempts", 11),
        ("stop_rule", "stop at the first OBJECTIVE_TARGET_REACHED"),
        ("certified_maximum_iterations", 400),
    ):
        root = tmp_path / field
        staging = root / "staging"
        attempt = _synthetic_attempt(
            staging / "attempts" / "attempt-1",
            engine_wall=1.0,
            terminal_objective=4.48e-8,
            maximum_feasibility_inf=1.0e-14,
        )
        evidence = _root_evidence(
            verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt]
        )
        evidence["attempt_protocol"][field] = value
        with pytest.raises(
            launcher.ProjectedRootError, match="published attempt protocol"
        ):
            launcher.publish_root(staging, root / "final", evidence)


def test_a_root_may_not_name_a_device_the_claim_is_not_stated_for(
    tmp_path: Path,
) -> None:
    """Section 1.2 makes the speed result RTX 5090 specific.

    ``--gpu-uuid`` is operator-supplied, the preflight only checked that the
    device it names is VISIBLE, and no validator ever compared it to the frozen
    constant -- so on a multi-GPU host the receipt could name a device other
    than the one the claim is stated for.
    """

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    evidence = _root_evidence(
        verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt]
    )
    evidence["supervisor"] = {**evidence["supervisor"], "gpu_uuid": "GPU-some-other"}
    with pytest.raises(launcher.ProjectedRootError, match="not the device the claim"):
        launcher.publish_root(staging, tmp_path / "final", evidence)
    with pytest.raises(launcher.ProjectedRootError, match="this plan certifies"):
        launcher.run_attempt_protocol(
            tmp_path / "root",
            cache_directory=tmp_path / "cache",
            gpu_uuid="GPU-some-other",
        )


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


def test_a_truncated_options_block_cannot_derive_an_empty_delta(
    tmp_path: Path,
) -> None:
    """The delta was derived over the keys the ATTEMPT published.

    An attempt publishing only ``objective_target`` therefore derived
    ``delta == {}`` and passed the substitution-soundness check the whole
    "same route" argument rests on, and an unknown published field reached
    ``getattr`` on the frozen dataclass as an ``AttributeError`` rather than a
    named refusal.  The key set is the frozen configuration's.
    """

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    attempt["evidence"]["options"] = {
        "objective_target": rehearsal.NATIVE_TARGET_OBJECTIVE,
        "maximum_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
    }
    attempt["evidence"]["certified_options_delta"] = {}
    evidence = _root_evidence(
        verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt]
    )
    with pytest.raises(
        launcher.ProjectedRootError, match="not the certified configuration's fields"
    ):
        launcher.publish_root(staging, tmp_path / "final", evidence)

    unknown = tmp_path / "unknown"
    unknown_staging = unknown / "staging"
    other = _synthetic_attempt(
        unknown_staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    other["evidence"]["options"] = {**other["evidence"]["options"], "invented": 1.0}
    with pytest.raises(
        launcher.ProjectedRootError, match="not the certified configuration's fields"
    ):
        launcher.publish_root(
            unknown_staging,
            unknown / "final",
            _root_evidence(verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[other]),
        )


def test_every_attempts_wall_is_derived_not_only_the_first_latching_one(
    tmp_path: Path,
) -> None:
    """Section 4 makes the wall of EVERY attempt part of the artifact.

    ``attempt_engine_wall_seconds`` was reached from ``derive_verdict``'s
    ``LATCHED`` branch alone, so a non-latching attempt could publish
    ``engine_compile: 10, engine_solve: 20, engine_wall: 1`` and re-validate.
    """

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        outcome="COMPLETED_WITHOUT_LATCH",
    )
    attempt["evidence"]["timing_seconds"] = {
        **attempt["evidence"]["timing_seconds"],
        "engine_compile": 10.0,
        "engine_solve": 20.0,
        "engine_wall": 1.0,
    }
    evidence = _root_evidence(verdict=launcher.VERDICT_NO_LATCH, attempts=[attempt])
    with pytest.raises(launcher.ProjectedRootError, match="is not its own"):
        launcher.publish_root(staging, tmp_path / "final", evidence)


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
        iterations=3,
    )
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[attempt],
        authorized_attempts=1,
        iterations=3,
    )
    evidence["attempt_protocol"]["conformance"] = launcher.CONFORMANCE_PREREGISTERED
    with pytest.raises(
        launcher.ProjectedRootError, match="attempt protocol conformance"
    ):
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
    (staging / launcher.COLD_LANE_DIRECTORY).mkdir(parents=True)
    cold = _attempt("COMPLETED_WITHOUT_LATCH", index=0)
    cold["artifact_relative_path"] = launcher.COLD_LANE_DIRECTORY
    cold["timed_against_bar"] = False
    cold["evidence"] = {
        **cold["evidence"],
        "compilation_cache": _attempt_cache(warm=True),
    }
    evidence = _root_evidence(
        verdict=launcher.verdict_of_gate("solve"),
        attempts=[_attempt("GATE_REFUSED", gate="solve")],
        cold_lane=cold,
    )
    with pytest.raises(launcher.ProjectedRootError, match="populated cache"):
        launcher.publish_root(staging, tmp_path / "final", evidence)


def test_an_anomalous_cold_lane_is_published_and_does_not_dispose_the_root(
    tmp_path: Path,
) -> None:
    """Plan section 12.9: the cold lane is DIAGNOSTICS, never disposition.

    Ruling 8 fed the lane's outcome into the conformance label, on a predicate
    that could not tell counter-evidence from infrastructure: a lane that failed
    the per-term quality gate and one that died on ``GATE_REFUSED:bootstrap``
    both returned the same ``False``.  Either way a run that ran the
    pre-registered N, the certified budget and the lane was labelled
    ``BOUNDED_SMOKE``, its verdict capped at ``QUALITY_ONLY`` -- which section
    4's table disposes as ROOT SPENT -- and the pair it minted beside
    ``quality_claim: CERTIFIED_BUDGET`` is one ``derive_verdict``'s own contract
    says cannot occur.  The lane is the FIRST GPU process of the session against
    an empty cache, which is exactly where a first-compile timeout or an OOM
    lands.

    So the anomaly publishes in full and the disposition comes from the three
    timed attempts.  ``--no-cold-lane`` still demotes: that is a pre-registration
    fact, not an outcome.
    """

    assert launcher.cold_lane_measured(None) is False
    for outcome in ("LATCHED", "COMPLETED_WITHOUT_LATCH"):
        assert launcher.cold_lane_measured({"outcome": outcome}) is True
    for outcome in ("GATE_REFUSED", "TIMEOUT", "PROTOCOL_FAILURE"):
        assert launcher.cold_lane_measured({"outcome": outcome}) is False
    assert launcher.cold_lane_anomaly(None) is None

    staging = tmp_path / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    (staging / launcher.COLD_LANE_DIRECTORY).mkdir(parents=True)
    cold = _attempt("GATE_REFUSED", index=0, gate="endpoint_ledger")
    cold["artifact_relative_path"] = launcher.COLD_LANE_DIRECTORY
    cold["timed_against_bar"] = False
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[attempt],
        cold_lane=cold,
    )
    # The run conformed; the lane's fault is stated, not charged.
    assert evidence["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_PREREGISTERED
    )
    assert evidence["cold_lane_anomaly"] == {
        "outcome": "GATE_REFUSED",
        "gate_refused": "endpoint_ledger",
        "return_code": 2,
        "timed_out": False,
        "supervised_seconds": cold["supervised_seconds"],
        "artifact_relative_path": launcher.COLD_LANE_DIRECTORY,
    }
    published = launcher.publish_root(staging, tmp_path / "final", evidence)
    sealed = launcher.validate_root_artifact(published)
    assert sealed["verdict"] == launcher.VERDICT_CLAIM_DISCHARGED
    assert sealed["cold_lane_anomaly"]["gate_refused"] == "endpoint_ledger"

    # A receipt may not hide the anomaly behind a null, and a latching or
    # missing lane has none to publish.
    hidden = tmp_path / "hidden"
    hidden_staging = hidden / "staging"
    hidden_attempt = _synthetic_attempt(
        hidden_staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    (hidden_staging / launcher.COLD_LANE_DIRECTORY).mkdir(parents=True)
    hidden_evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[hidden_attempt],
        cold_lane=cold,
    )
    hidden_evidence["cold_lane_anomaly"] = None
    with pytest.raises(launcher.ProjectedRootError, match="cold-lane anomaly"):
        launcher.publish_root(hidden_staging, hidden / "final", hidden_evidence)

    # And ``--no-cold-lane`` still caps, because the lane was never run.
    assert launcher.attempt_protocol_conformance(
        authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
        iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        cold_lane_authorized=False,
    ) == launcher.CONFORMANCE_BOUNDED_SMOKE


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
    off_tmpfs_path: Path,
) -> None:
    """Three external resources are checked before the first child is spawned.

    Plan section 11: enumerate the failure class, not the instance.  A missing
    data file already spent one root of the predecessor route, and neither the
    device UUID the receipt names nor the NVIDIA tooling the telemetry drives
    was checked at all before this.
    """

    cache = off_tmpfs_path / "cache"
    cache.mkdir()
    environment = {launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE: str(off_tmpfs_path)}
    preflight = launcher.preflight_external_resources(
        gpu_uuid=launcher.GPU_UUID,
        cache_directory=cache,
        output_root=off_tmpfs_path / "root",
        environment=environment,
    )
    assert launcher.GPU_UUID in preflight["visible_gpu_uuids"]
    assert preflight["native_endpoint_state_content_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_CONTENT_SHA256
    )
    assert preflight["native_endpoint_state_sha256"] == (
        rehearsal.NATIVE_ENDPOINT_STATE_FILE_SHA256
    )
    # The temporary, cache and output storage every child writes through, each
    # proven by a write rather than asserted from a capacity number.
    assert preflight["temporary_directory"] == str(off_tmpfs_path)
    assert [probe["role"] for probe in preflight["storage"]] == [
        "temporary",
        "compilation_cache",
        "output",
    ]
    assert all(probe["one_byte_write"] == "ok" for probe in preflight["storage"])
    assert all(
        probe["filesystem_type"] not in launcher.REFUSED_STORAGE_FILESYSTEM_TYPES
        for probe in preflight["storage"]
    )
    assert json.loads(canonical_json_bytes(preflight)) == preflight
    with pytest.raises(launcher.ProjectedRootError, match="is not among the visible"):
        launcher.preflight_external_resources(
            gpu_uuid="GPU-not-this-box",
            cache_directory=cache,
            output_root=off_tmpfs_path / "root",
            environment=environment,
        )


# ------------------------------------------------------- temporary storage


def test_the_temporary_directory_is_resolved_by_xlas_rule_not_pythons() -> None:
    """XLA spills from C++: ``TMPDIR`` or ``/tmp``, with no fallthrough.

    Python's ``tempfile`` probes its candidate and falls through to ``/var/tmp``
    when the probe fails, which is exactly why a quota-exhausted ``/tmp`` left
    every Python path on the box working while the spill died inside the
    bootstrap gate.  Resolving this the Python way would preflight a directory
    the C++ runtime never uses.
    """

    assert launcher.DEFAULT_TEMPORARY_DIRECTORY == Path("/tmp")
    assert launcher.resolve_temporary_directory({}) == Path("/tmp")
    assert launcher.resolve_temporary_directory(
        {launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE: "  "}
    ) == Path("/tmp")
    assert launcher.resolve_temporary_directory(
        {launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE: "/var/tmp"}
    ) == Path("/var/tmp")
    # The rule is a CANDIDATE LIST in TSL's order, not one name.  Both shipped
    # binaries carry all three, and ``TEST_TMPDIR`` is tried FIRST -- so reading
    # only ``TMPDIR`` preflighted one directory while the children spilled
    # through another under a shell holding a Bazel-ism.
    assert launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES == (
        "TEST_TMPDIR",
        "TMPDIR",
        "TMP",
    )
    assert launcher.resolve_temporary_directory(
        {"TEST_TMPDIR": "/var/tmp/first", "TMPDIR": "/var/tmp/second", "TMP": "/x"}
    ) == Path("/var/tmp/first")
    assert launcher.resolve_temporary_directory({"TMP": "/var/tmp/third"}) == (
        Path("/var/tmp/third")
    )
    # And all three are OVERRIDDEN in the child, or the rule is enforced against
    # a directory nobody used.
    _, child_environment = launcher.attempt_invocation(
        Path("/var/tmp/attempt"),
        attempt_index=1,
        iterations=3,
        cache_directory=Path("/var/tmp/cache"),
        environment={"TEST_TMPDIR": "/tmp", "TMP": "/tmp", "TMPDIR": "/tmp"},
        temporary_directory=Path("/var/tmp/safe"),
    )
    assert [
        child_environment[name]
        for name in launcher.TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES
    ] == ["/var/tmp/safe"] * 3


def test_a_temporary_directory_on_tmpfs_is_refused_before_any_compute(
    off_tmpfs_path: Path,
) -> None:
    """Plan section 11's rule, enforced by code rather than stated in prose.

    ``TMPDIR`` appeared nowhere in this repository except two sentences of the
    plan, and the check those sentences named -- free space -- is structurally
    blind to the binding limit: measured on this box while the condition was
    live, ``/tmp`` reported 12.29 GiB available and 571769 free inodes while a
    one-byte write returned ``EDQUOT``.  An empty tmpfs passes a write probe and
    then fills during the run, so the FILESYSTEM TYPE is refused too.
    """

    assert launcher.filesystem_type(Path("/dev/shm")) == "tmpfs"
    assert launcher.filesystem_type(off_tmpfs_path) not in (
        launcher.REFUSED_STORAGE_FILESYSTEM_TYPES
    )
    with pytest.raises(launcher.ProjectedRootError, match="is on tmpfs"):
        launcher.probe_writable_storage(Path("/dev/shm"), role="temporary")


def test_a_directory_that_refuses_a_write_is_refused_however_much_it_reports(
    off_tmpfs_path: Path,
) -> None:
    """A write is the only check that sees a quota.

    The refusal is on the write, not on ``statvfs`` -- which reports the same
    free space either way, as it did on the box where a one-byte write returned
    ``EDQUOT`` and left a zero-length file behind.  A read-only directory is the
    portable way to reach the same errno class.
    """

    unwritable = off_tmpfs_path / "unwritable"
    unwritable.mkdir()
    unwritable.chmod(0o500)
    try:
        assert shutil.disk_usage(unwritable).free > 0
        with pytest.raises(
            launcher.ProjectedRootError, match="refused a one-byte write"
        ):
            launcher.probe_writable_storage(unwritable, role="output")
    finally:
        unwritable.chmod(0o700)
    with pytest.raises(launcher.ProjectedRootError, match="does not exist"):
        launcher.probe_writable_storage(off_tmpfs_path / "absent", role="temporary")
    # A path that exists and is not a directory said "does not exist", which is
    # a five-minute diagnosis at the moment an operator is launching a root.
    a_file = off_tmpfs_path / "a-file"
    a_file.write_bytes(b"")
    with pytest.raises(launcher.ProjectedRootError, match="is not a directory"):
        launcher.probe_writable_storage(a_file, role="temporary")
    # A RELATIVE declaration is refused rather than probed: the supervisor
    # resolves it against its own working directory while every child resolves
    # it against the repository, so the rule would be enforced against a
    # directory nobody uses and the spill would fall through to ``/tmp``.
    with pytest.raises(launcher.ProjectedRootError, match="is relative"):
        launcher.probe_writable_storage(Path("relative-tmp"), role="temporary")
    probe = launcher.probe_writable_storage(off_tmpfs_path, role="temporary")
    assert probe["one_byte_write"] == "ok"
    assert probe["device_id"] == os.stat(off_tmpfs_path).st_dev
    # The declared path and the one the write landed in are both published, so a
    # symlinked temporary directory can be re-identified from the sealed bytes.
    assert probe["directory"] == str(off_tmpfs_path)
    assert probe["resolved_directory"] == str(off_tmpfs_path.resolve())
    assert not list(off_tmpfs_path.glob(f"{launcher.STORAGE_PROBE_PREFIX}*"))


def test_a_stacked_mount_reports_the_filesystem_the_kernel_resolves(
    off_tmpfs_path: Path,
) -> None:
    """Mountinfo is in MOUNT order; the effective mount is the LAST at a path.

    Skipping ties reported the SHADOWED filesystem, so ``mount -t tmpfs tmpfs
    /var/tmp/scratch`` -- the ordinary way an operator hands a compile a RAM
    scratch directory, and what several container and systemd-hardening configs
    do -- published the ext4 underneath it, passed the tmpfs refusal, passed the
    write probe because an empty tmpfs takes a byte, and then filled during the
    run.  This box carries one real stacked mount, and it is a stable one.
    """

    stacked = Path("/proc/sys/fs/binfmt_misc")
    orders = [
        line.split()
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if len(line.split()) > 4 and line.split()[4] == str(stacked)
    ]
    if len(orders) < 2:
        pytest.skip("this namespace carries no stacked mount to read")
    effective = orders[-1][orders[-1].index("-") + 1]
    assert launcher.filesystem_type(stacked) == effective
    assert launcher.filesystem_type(Path("/dev/shm")) == "tmpfs"
    assert launcher.filesystem_type(off_tmpfs_path) not in (
        launcher.REFUSED_STORAGE_FILESYSTEM_TYPES
    )


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
