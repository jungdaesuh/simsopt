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

import benchmarks.rehearse_single_stage_projected_route_cpu as rehearsal
import benchmarks.run_single_stage_projected_route_gpu_root as launcher
import jax.numpy as jnp
import numpy as np
import pytest
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


def test_the_certified_wall_is_engine_compile_plus_engine_solve() -> None:
    """The timed boundary, read off an attempt exactly as the verdict reads it.

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
        launcher.derive_verdict([at_the_bar], wall_seconds_bar=bar)
        == launcher.VERDICT_QUALITY_ONLY
    )


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
    return {
        "attempt_index": index,
        "artifact_relative_path": f"attempts/attempt-{index}",
        "outcome": outcome,
        "evidence": {
            "gate_refused": gate,
            "timing_seconds": {"engine_wall": engine_wall},
        },
    }


def test_every_protocol_outcome_maps_to_one_of_exactly_four_verdicts() -> None:
    """There is no undefined outcome: roots 1-4 all died in unwritten ones."""

    bar = rehearsal.NATIVE_WALL_SECONDS_BAR
    assert (
        launcher.derive_verdict([_attempt("LATCHED", engine_wall=bar - 1.0)],
                                wall_seconds_bar=bar)
        == launcher.VERDICT_CLAIM_DISCHARGED
    )
    assert (
        launcher.derive_verdict([_attempt("LATCHED", engine_wall=bar + 1.0)],
                                wall_seconds_bar=bar)
        == launcher.VERDICT_QUALITY_ONLY
    )
    assert (
        launcher.derive_verdict(
            [_attempt("COMPLETED_WITHOUT_LATCH", index=index) for index in (1, 2, 3)],
            wall_seconds_bar=bar,
        )
        == launcher.VERDICT_NO_LATCH
    )
    assert launcher.derive_verdict(
        [_attempt("GATE_REFUSED", gate="problem_identity")], wall_seconds_bar=bar
    ) == launcher.verdict_of_gate("problem_identity")
    assert launcher.derive_verdict(
        [_attempt("TIMEOUT")], wall_seconds_bar=bar
    ).startswith(launcher.VERDICT_GATE_REFUSED_PREFIX)
    assert launcher.derive_verdict([], wall_seconds_bar=bar).startswith(
        launcher.VERDICT_GATE_REFUSED_PREFIX
    )


def test_the_claim_is_discharged_by_the_first_latching_attempt_not_the_first() -> None:
    """A no-latch draw indicts nothing, and the wall that counts is the latch's."""

    bar = rehearsal.NATIVE_WALL_SECONDS_BAR
    attempts = [
        _attempt("COMPLETED_WITHOUT_LATCH", index=1, engine_wall=bar - 50.0),
        _attempt("LATCHED", index=2, engine_wall=bar + 10.0),
    ]
    assert (
        launcher.derive_verdict(attempts, wall_seconds_bar=bar)
        == launcher.VERDICT_QUALITY_ONLY
    )


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
    assert launcher.derive_verdict(
        [
            {
                "attempt_index": 1,
                "artifact_relative_path": "attempts/attempt-1",
                "outcome": "PROTOCOL_FAILURE",
                "evidence": None,
            }
        ],
        wall_seconds_bar=rehearsal.NATIVE_WALL_SECONDS_BAR,
    ).startswith(launcher.VERDICT_GATE_REFUSED_PREFIX)


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
        _ledger(**{"observable.iota": -0.42 * (1.0 + 1.0e-7)})
    )["passed"]
    assert not rehearsal.gate_endpoint_ledger(
        _ledger(**{"observable.iota": -0.42 * (1.0 + 1.0e-4)})
    )["passed"]
    # A residual moving from 1e-20 to 1e-18 is a 99x relative change and a
    # 1e-18 absolute one; only the absolute reading means anything here.
    assert rehearsal.gate_endpoint_ledger(_ledger(**{"raw.residual": 1.0e-18}))["passed"]
    assert not rehearsal.gate_endpoint_ledger(
        _ledger(**{"constraint.volume": 1.0e-9})
    )["passed"]


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
        cache={},
        verdict=launcher.VERDICT_NO_LATCH,
        chain_seconds=1.0,
    )
    assert bounded["quality_claim"] == "NOT_CLAIMED_AT_BOUNDED_BUDGET"
    assert bounded["attempt_protocol"]["conformance"] == "BOUNDED_SMOKE"

    root = launcher.build_root_evidence(
        attempts=[],
        cold_lane=None,
        snapshot={},
        supervisor={},
        authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
        iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        cache={},
        verdict=launcher.VERDICT_NO_LATCH,
        chain_seconds=1.0,
    )
    assert root["quality_claim"] == "CERTIFIED_BUDGET"
    assert root["attempt_protocol"]["conformance"] == "PREREGISTERED"
    assert root["claim"]["target_objective"] == rehearsal.NATIVE_TARGET_OBJECTIVE
    assert root["claim"]["wall_seconds_bar"] == rehearsal.NATIVE_WALL_SECONDS_BAR


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


def _publish_synthetic_root(
    root: Path,
    *,
    verdict: str,
    engine_wall: float,
    terminal_objective: float = 4.48e-8,
    maximum_feasibility_inf: float | None = 1.0e-14,
    ledger: dict | None = None,
) -> tuple[Path, dict]:
    """A structurally complete root receipt with no GPU behind it."""

    staging = root / "staging"
    attempt_directory = staging / "attempts" / "attempt-1"
    attempt_directory.mkdir(parents=True)
    coordinates = jnp.asarray([0.25, -0.5, 1.0], dtype=jnp.float64)
    with (attempt_directory / rehearsal.TERMINAL_COORDINATES_FILENAME).open(
        "wb"
    ) as stream:
        np.save(stream, np.asarray(coordinates, dtype=np.float64), allow_pickle=False)
    attempt = {
        "attempt_index": 1,
        "artifact_relative_path": "attempts/attempt-1",
        "outcome": "LATCHED",
        "evidence": {
            "gate_refused": None,
            "problem_identity": {"bound": True, "sha_is_binding": False},
            "lowering_pre_gate": {"budget_independent": True},
            "options": {"objective_target": rehearsal.NATIVE_TARGET_OBJECTIVE},
            "endpoint_ledger": (
                _synthetic_ledger(gated=False) if ledger is None else ledger
            ),
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
            "timing_seconds": {"engine_wall": engine_wall},
        },
    }
    evidence = {
        "schema_version": launcher.GPU_ROOT_SCHEMA_VERSION,
        "route": launcher.PROJECTED_ROUTE,
        "verdict": verdict,
        "claim": {
            "target_objective": rehearsal.NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": rehearsal.NATIVE_WALL_SECONDS_BAR,
        },
        "attempts": [attempt],
        "cold_lane": None,
        "timing_boundary": "engine_compile_plus_solve",
    }
    published = launcher.publish_root(staging, root / "final", evidence)
    return published, evidence


def test_a_published_root_revalidates_from_its_sealed_bytes(tmp_path: Path) -> None:
    published, _ = _publish_synthetic_root(
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


def test_validation_recomputes_the_verdict_instead_of_believing_it(
    tmp_path: Path,
) -> None:
    """A receipt that names a verdict its own attempts do not derive is refused."""

    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=rehearsal.NATIVE_WALL_SECONDS_BAR + 100.0,
    )
    with pytest.raises(launcher.ProjectedRootError, match="not the one the"):
        launcher.validate_root_artifact(published)


def test_validation_refuses_a_latch_above_the_native_endpoint_objective(
    tmp_path: Path,
) -> None:
    """``LATCHED`` is a status code; the claim's quality quantity is a number.

    The optimizer sets the status from its OWN configured target, so the
    published objective and the published target are both re-derived against
    the plan's literal rather than trusted through the enum.
    """

    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=rehearsal.NATIVE_WALL_SECONDS_BAR - 100.0,
        terminal_objective=rehearsal.NATIVE_TARGET_OBJECTIVE * 1.0001,
    )
    with pytest.raises(launcher.ProjectedRootError, match="above the native endpoint"):
        launcher.validate_root_artifact(published)


@pytest.mark.parametrize("recorded", [None, 1.0e-9])
def test_validation_refuses_a_feasibility_that_is_not_within_the_tolerance(
    tmp_path: Path, recorded: float | None
) -> None:
    """Every comparison against a NaN is false, so ``> tolerance`` fails open.

    A nonfinite worst iterate reaches the receipt as ``null`` -- canonical JSON
    refuses NaN and ``json_scalar`` writes null instead -- and a null is not a
    number under the bound any more than 1e-9 is.  Both readings are refused by
    the contract's own ``<= tolerance``.
    """

    assert rehearsal.json_scalar(float("nan")) is None
    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
        maximum_feasibility_inf=recorded,
    )
    with pytest.raises(launcher.ProjectedRootError, match="infeasible iterate"):
        launcher.validate_root_artifact(published)


def test_validation_refuses_an_attempt_that_narrowed_the_pinned_term_set(
    tmp_path: Path,
) -> None:
    """Quality parity is defined by the campaign, never by the run reporting it."""

    narrowed = _synthetic_ledger(gated=False)
    narrowed["pinned_quality_terms"] = ["observable.iota"]
    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
        ledger=narrowed,
    )
    with pytest.raises(launcher.ProjectedRootError, match="ledger scope differs"):
        launcher.validate_root_artifact(published)


def test_a_gated_ledger_survives_its_own_canonical_round_trip(tmp_path: Path) -> None:
    """The recompute must not manufacture a false reject on an honest root.

    The verdicts are derived once from Python floats at publication and again
    from the JSON floats a reader loads.  Proving those agree is the whole
    licence for recomputing instead of reading back: a band that refused a
    valid endpoint over a round-trip ULP would be the V260/rho-floor class of
    false reject, which this campaign has already paid for three times.
    """

    published, _ = _publish_synthetic_root(
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


def test_validation_recomputes_a_gated_ledgers_verdicts_from_its_terms(
    tmp_path: Path,
) -> None:
    """A published pass on a term whose numbers do not pass is refused."""

    gated = _synthetic_ledger(gated=True)
    assert gated["pinned_term_gate"]["passed"] is True
    gated["terminal"] = {**gated["terminal"], "observable.iota": -0.42 * (1.0 + 1.0e-3)}
    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
        ledger=gated,
    )
    with pytest.raises(launcher.ProjectedRootError, match="not the one its ledger"):
        launcher.validate_root_artifact(published)


def test_validation_refuses_a_root_whose_seal_was_reopened(tmp_path: Path) -> None:
    """0444/0555 is a property of the published bytes, not of the publisher."""

    published, _ = _publish_synthetic_root(
        tmp_path, verdict=launcher.VERDICT_CLAIM_DISCHARGED, engine_wall=1.0
    )
    (published / launcher.EVIDENCE_FILENAME).chmod(0o644)
    with pytest.raises(rehearsal.RehearsalError, match="sealed artifact mode differs"):
        launcher.validate_root_artifact(published)


def test_validation_rejects_a_cold_lane_that_ran_warm(tmp_path: Path) -> None:
    """A cold lane against a populated cache measures nothing it claims to.

    Found by the first real GPU launch: the cache was sampled after the
    identity gate, which pays the point-evaluation compile, so a genuinely cold
    process published ``warm: true``.  The sample now happens before this
    process has traced anything, and a cold lane that still reads warm is a
    defect rather than a documented number.
    """

    staging = tmp_path / "staging"
    cold_directory = staging / "cold-lane"
    cold_directory.mkdir(parents=True)
    evidence = {
        "schema_version": launcher.GPU_ROOT_SCHEMA_VERSION,
        "route": launcher.PROJECTED_ROUTE,
        "verdict": launcher.verdict_of_gate("solve"),
        "claim": {
            "target_objective": rehearsal.NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": rehearsal.NATIVE_WALL_SECONDS_BAR,
        },
        "attempts": [_attempt("GATE_REFUSED", gate="solve")],
        "cold_lane": {
            "attempt_index": 0,
            "artifact_relative_path": "cold-lane",
            "outcome": "COMPLETED_WITHOUT_LATCH",
            "timed_against_bar": False,
            "evidence": {
                "gate_refused": None,
                "compilation_cache": {"warm": True},
            },
        },
        "timing_boundary": "engine_compile_plus_solve",
    }
    published = launcher.publish_root(staging, tmp_path / "final", evidence)
    with pytest.raises(launcher.ProjectedRootError, match="populated cache"):
        launcher.validate_root_artifact(published)


def test_validation_rejects_a_tampered_artifact_tree(tmp_path: Path) -> None:
    published, _ = _publish_synthetic_root(
        tmp_path,
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        engine_wall=1.0,
    )
    tampered = published / launcher.EVIDENCE_FILENAME
    tampered.chmod(0o644)
    tampered.write_bytes(tampered.read_bytes() + b" ")
    with pytest.raises(launcher.ProjectedRootError, match="manifest differs"):
        launcher.validate_root_artifact(published)


def test_validation_rejects_a_restated_native_reference(tmp_path: Path) -> None:
    """An artifact may not move the bar it is then judged against."""

    staging = tmp_path / "staging"
    staging.mkdir()
    evidence = {
        "schema_version": launcher.GPU_ROOT_SCHEMA_VERSION,
        "route": launcher.PROJECTED_ROUTE,
        "verdict": launcher.VERDICT_NO_LATCH,
        "claim": {
            "target_objective": rehearsal.NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": 1.0e6,
        },
        "attempts": [],
        "cold_lane": None,
        "timing_boundary": "engine_compile_plus_solve",
    }
    published = launcher.publish_root(staging, tmp_path / "final", evidence)
    with pytest.raises(launcher.ProjectedRootError, match="restates the native"):
        launcher.validate_root_artifact(published)


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
            {
                "schema_version": launcher.GPU_ROOT_SCHEMA_VERSION,
                "route": launcher.PROJECTED_ROUTE,
                "verdict": launcher.VERDICT_NO_LATCH,
                "claim": {},
                "attempts": [],
                "cold_lane": None,
                "timing_boundary": "engine_compile_plus_solve",
            },
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
    )
    assert payload["gpu_zero_asserted"] is False
    assert payload["gpu_uuid"] == "GPU-0000"
    assert payload["attempt_timeout_seconds"] == 3600.0
    assert payload["runtime_identity"]["backend"] == "gpu"
    assert json.loads(canonical_json_bytes(payload)) == payload


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
