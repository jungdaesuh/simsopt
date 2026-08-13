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

import ast
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from types import SimpleNamespace

import benchmarks.rehearse_single_stage_projected_route_cpu as rehearsal
import benchmarks.run_single_stage_projected_route_gpu_root as launcher
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks.process_gpu_monitor import (
    GPU_MEMORY_UNAVAILABLE_REASONS,
    ProcessGpuMemoryMonitorError,
)
from benchmarks.single_stage_fullspace_snapshot import canonical_json_bytes
from simsopt_jax.geo.optimizers.projected_lbfgs import KernelLowering
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
    """The first candidate directory on this box that takes the suite's writes.

    Selected by the launcher's OWN probe rather than by a type check of its
    own: a relative ``TMPDIR`` resolves against pytest's rootdir, which is this
    frozen repository, and a ``/var/tmp`` that is off tmpfs but read-only or
    quota-exhausted passes a filesystem-type check and then fails at ``mkdir``
    as an unhandled ``OSError`` mid-suite.  ``probe_writable_storage`` refuses a
    non-absolute path, refuses a RAM filesystem and then WRITES A BYTE, which is
    the half of ruling 9 the EDQUOT lesson calls binding.
    """

    declared = os.environ.get("TMPDIR", "")
    candidates = [
        *([Path(declared)] if declared.strip() else []),
        Path("/var/tmp"),
        Path.home() / ".cache",
    ]
    for candidate in candidates:
        try:
            launcher.probe_writable_storage(candidate, role="suite temporary")
        except launcher.ProjectedRootError:
            continue
        return candidate
    raise RuntimeError(f"no candidate directory the suite may write to among {candidates}")


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


def _endpoint_agreement(
    terminal_state_sha256: str = "0" * 64,
    *,
    terminal_objective: float = 4.48e-8,
    terminal_feasibility_inf: float = 1.0e-14,
    standalone_terminal_objective: float | None = None,
) -> dict:
    """The whole agreement block ``certify_endpoint_agreement`` publishes.

    Its two objective halves are not free of the rest of the receipt.
    ``loop_terminal_objective`` IS ``solve.terminal_objective`` -- one float
    through two writers -- and ``standalone_terminal_objective`` IS the endpoint
    ledger's terminal ``weighted_total``, because both are
    ``case.standalone_evaluation(run.coordinates)`` evaluated in one process on
    one input.  A fixture that lets them drift is a fixture asserting a receipt
    the child cannot write, which is how an agreement block whose two halves
    agreed with each other to 5e-16 and with nothing else in the run came to
    seal.  The feasibility is the same fact told twice the same way.
    """

    return {
        "loop_terminal_objective": terminal_objective,
        "standalone_terminal_objective": (
            terminal_objective
            if standalone_terminal_objective is None
            else standalone_terminal_objective
        ),
        "relative_tolerance": launcher.DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        "absolute_floor": launcher.DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
        "terminal_feasibility_inf": terminal_feasibility_inf,
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


def _iterates(
    *,
    terminal_objective: float,
    maximum_feasibility_inf: float | None,
    terminal_feasibility_inf: float = 1.0e-14,
) -> list[dict]:
    """The recorded iterates the solve summary is derived against.

    The summary the claim's feasibility gate reads is ``max`` over these rows,
    and a fixture publishing ``rows: []`` beside ``iterations_run: 7`` is a
    fixture asserting a shape the producer cannot write -- which is how a
    receipt carrying two iterates nine decades outside the tolerance came to
    seal beside a passing summary.  The last iterate carries the worst
    feasibility and the objectives descend, as a monotone run's do.

    TWO further properties of a real trajectory this fixture used to deny.  A
    row OPENS at its objective, and the engine breaks at the top of the loop
    when the point it opens at has reached the target -- so no recorded iterate
    is ever at or below ``objective_target``, and the fixture's opening
    objectives stay above it.  And the terminal point is the CANDIDATE the last
    recorded iteration accepted, not the point it opened at, which is why the
    engine's own row carries both and why ``terminal_objective`` here is the
    last row's ``candidate_objective`` rather than its ``objective``.  A fixture
    that made the terminal one of the opening objectives asserted an identity
    two banked 5090 latches refute.
    """

    smaller = (
        None if maximum_feasibility_inf is None else maximum_feasibility_inf / 10.0
    )
    target = rehearsal.CERTIFIED_ROUTE_OPTIONS.objective_target
    opening = max(terminal_objective, target) * 10.0
    objectives = [opening * 100.0, opening * 10.0, opening]
    candidates = [objectives[1], objectives[2], terminal_objective]
    feasibilities = [smaller, smaller, maximum_feasibility_inf]
    return [
        {
            "index": index,
            "objective": objectives[index],
            "candidate_objective": candidates[index],
            "feasibility_inf": feasibilities[index],
            "candidate_feasibility_inf": (
                terminal_feasibility_inf if index == 2 else feasibilities[index]
            ),
        }
        for index in range(3)
    ]


def _solve_payload(
    *,
    latched: bool,
    terminal_objective: float,
    maximum_feasibility_inf: float | None,
    terminal_feasibility_inf: float = 1.0e-14,
) -> dict:
    """Every host-side scalar ``_solve_payload`` publishes, in its shape.

    The status and its name come from the engine's own enumeration: the fixture
    used to publish ``status: 0`` (``RUNNING``) under the name
    ``MAXIMUM_ITERATIONS``, which the engine calls ``ITERATION_LIMIT`` and never
    reports at the end of a run.
    """

    status = (
        launcher.ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED
        if latched
        else launcher.ProjectedLbfgsStatus.ITERATION_LIMIT
    )
    rows = _iterates(
        terminal_objective=terminal_objective,
        maximum_feasibility_inf=maximum_feasibility_inf,
        terminal_feasibility_inf=terminal_feasibility_inf,
    )
    return {
        "status": int(status),
        "status_name": status.name,
        "latched": latched,
        "iterations_run": len(rows),
        "terminal_objective": terminal_objective,
        "terminal_feasibility_inf": terminal_feasibility_inf,
        "terminal_projected_gradient_inf": 1.0e-7,
        "stored_pairs": 5,
        "projector_materializations": 2,
        "tangency_forced_refreshes": 0,
        "line_search_forced_refreshes": 0,
        "monotone_descent": True,
        "maximum_feasibility_inf": maximum_feasibility_inf,
        "collapse_proximity_margin": 1.0,
        "rows": rows,
    }


def _invocation_sha256(index: int) -> str:
    """One draw's argv digest, distinct per draw as the supervisor's are.

    The supervisor launches every child with its own attempt root and index, so
    no two draws of one root share an invocation -- which is what tells a cold
    lane from a copy of a timed attempt.
    """

    return rehearsal.sha256_hex(f"attempt-{index}".encode())


def _gpu_memory(index: int = 1) -> dict:
    """The observation ``_gpu_memory_payload`` normalizes, whole.

    ``child_argv_sha256`` is the sampler's digest of the argv it observed on the
    device, which for an honest record is the argv the supervisor launched.
    """

    return {
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "availability": "unavailable",
        "unavailable_reason": "sampler-failed",
        "device_uuid": launcher.GPU_UUID,
        "parent_pid": 1,
        "child_pid": 2 + index,
        "child_start_time_ticks": None,
        "child_argv_sha256": _invocation_sha256(index),
        "sample_count": 0,
        "peak_used_memory_mib": None,
    }


def _execution_sources() -> dict:
    """The custody block ``bind_execution_sources`` publishes, from the manifest.

    The fixture used to publish ``{"bound_modules": []}`` -- a receipt asserting
    that ZERO source modules executed -- against a producer that publishes four
    keys and 297 bound modules, and the suite's only coverage of the field was
    its deletion, which the outer key set already caught.  That is how a nulled
    ``execution_sources`` shipped inside the remediation that named it.  Built
    here from the campaign's own manifest and the launcher's own list of the
    modules the chain runs through, so it is derived rather than transcribed.
    """

    manifest, entries = rehearsal.load_execution_source_manifest(REPOSITORY)
    return {
        "manifest": manifest,
        "bound_modules": [
            {
                "module": path.replace("/", ".").removesuffix(".py"),
                "relative_path": path,
                "sha256": entries[path]["sha256"],
                "size_bytes": entries[path]["size_bytes"],
            }
            for path in sorted(launcher.CHAIN_EXECUTION_SOURCE_PATHS)
        ],
        "unmanifested_repository_modules": [],
        "interpreter_installation_modules": {"count": 0, "roots": []},
    }


def _problem_identity() -> dict:
    """The identity binding ``bind_problem_identity`` publishes, whole.

    Derived through the producer's own owner at the campaign's reference
    observables, so the fixture is a document the child can actually write.
    """

    return rehearsal.problem_identity_evidence(
        dict(rehearsal.CPU_BOOTSTRAP_OBSERVABLES),
        problem_sha256="0" * 64,
        bootstrap_sha256="1" * 64,
    )


def _lowering_pre_gate(iterations: int) -> dict:
    """The whole pre-gate record ``measure_lowering_pre_gate`` publishes.

    The kernels are the campaign's own list -- the six the CERTIFIED
    configuration selects, bound to the real producer by execution in the
    rehearsal suite.  This fixture used to invent two kernel names this
    repository never lowers (``projected_lbfgs_step``, ``projected_lbfgs_loop``)
    totalling 12 288 bytes against a producer that lowers six totalling
    65 million, and the validator accepted it -- four reviewers found the same
    hole in one round.  The SIZES stay fixture values, because the producer's
    differ between two processes on one box.
    """

    kernels = [
        {"name": name, "ir_bytes": 4096 * (index + 1), "while_operations": index}
        for index, name in enumerate(rehearsal.CERTIFIED_LOWERED_KERNEL_NAMES)
    ]
    return {
        "rehearsal_iterations": iterations,
        "certified_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
        "budget_independent": True,
        "kernels": kernels,
        "total_ir_bytes": sum(kernel["ir_bytes"] for kernel in kernels),
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
        "execution_sources": _execution_sources(),
        "problem_identity": _problem_identity(),
        "lowering_pre_gate": _lowering_pre_gate(iterations),
        "options": _options_payload(iterations),
        "certified_options_delta": _options_delta(iterations),
        "compilation_cache": _attempt_cache(warm=True),
        "solve": _solve_payload(
            latched=latched,
            terminal_objective=4.48e-8,
            maximum_feasibility_inf=1.0e-14,
        ),
        "endpoint_agreement": _endpoint_agreement(terminal_objective=4.48e-8),
        "endpoint_ledger": _synthetic_ledger(gated=False, weighted_total=4.48e-8),
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
        # The three measurements NEST: the engine's compile plus solve sits
        # inside the attempt's own wall, which sits inside the wall the
        # supervisor observed.  The fixture used to publish an attempt wall
        # outside the supervised one, which no supervisor can observe.
        "supervised_seconds": engine_wall + 4.0,
        "argv_sha256": _invocation_sha256(index),
        "gpu_memory": _gpu_memory(index),
        "stderr_tail": "",
        "stdout_tail": None,
        "evidence": evidence,
    }


def _relaunched(record: dict, index: int, *, directory: str | None = None) -> dict:
    """One record re-stamped as the draw the protocol launched at ``index``.

    A forgery about the attempt SEQUENCE must differ from an honest receipt in
    the sequence and in nothing else, or it is refused for the wrong reason and
    the test proves a narrower property than its name -- the finding class the
    round-5 review filed against two of the round-4 forgery tests.  Every draw
    is launched into its own directory at its own index, so its invocation
    digest and the sampler's observation of it are its own.
    """

    evidence = record["evidence"]
    return {
        **record,
        "attempt_index": index,
        "artifact_relative_path": (
            f"{launcher.ATTEMPTS_DIRECTORY}/attempt-{index}"
            if directory is None
            else directory
        ),
        "argv_sha256": _invocation_sha256(index),
        "gpu_memory": _gpu_memory(index),
        "evidence": (
            {**evidence, "attempt_index": index}
            if isinstance(evidence, dict)
            else evidence
        ),
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


def test_conformance_is_one_label_derived_from_the_four_frozen_facts() -> None:
    """N, the certified budget, whether the cold lane RAN, and the timeout.

    The third fact is the lane's AUTHORIZATION, never its outcome (plan section
    12.9).  Feeding the outcome in charged a fully conforming run for an
    infrastructure fault on a draw the protocol does not contain -- see
    ``test_an_anomalous_cold_lane_is_published_and_does_not_dispose_the_root``.

    The fourth is the supervision timeout, and it is here because it was bound
    to nothing at all: the gate that requires a claimed timeout to have been
    waited took BOTH sides out of the receipt, so a root publishing
    ``attempt_timeout_seconds: 1e-9`` sealed ``CLAIM_DISCHARGED`` beside a lane
    that "timed out" in half a second.  A moved timeout is a real experiment and
    demotes rather than refusing, exactly as a moved budget does.
    """

    assert (
        launcher.attempt_protocol_conformance(
            authorized_attempts=launcher.PREREGISTERED_ATTEMPTS,
            iterations=rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            cold_lane_authorized=True,
            attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
        )
        == launcher.CONFORMANCE_PREREGISTERED
    )
    for authorized, iterations, cold_lane, timeout in (
        (10, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True, launcher.ATTEMPT_TIMEOUT_SECONDS),
        (launcher.PREREGISTERED_ATTEMPTS, 400, True, launcher.ATTEMPT_TIMEOUT_SECONDS),
        (
            launcher.PREREGISTERED_ATTEMPTS,
            rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            False,
            launcher.ATTEMPT_TIMEOUT_SECONDS,
        ),
        # Every value the field admitted while nothing read it.
        (launcher.PREREGISTERED_ATTEMPTS, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True, 1e-9),
        (launcher.PREREGISTERED_ATTEMPTS, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True, 0.0),
        (launcher.PREREGISTERED_ATTEMPTS, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True, -1.0),
        (launcher.PREREGISTERED_ATTEMPTS, rehearsal.CERTIFIED_MAXIMUM_ITERATIONS, True, 1e12),
    ):
        assert (
            launcher.attempt_protocol_conformance(
                authorized_attempts=authorized,
                iterations=iterations,
                cold_lane_authorized=cold_lane,
                attempt_timeout_seconds=timeout,
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
        attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
        attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
        # From the producer's own owner: the column is now RE-DERIVED at
        # re-validation, so a second spelling here would be a twin that drifts
        # into refusing every honest receipt.
        "relative_difference": rehearsal.endpoint_relative_differences(
            rows["terminal"], rows["native"]
        ),
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
                attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
        # The chain contains every draw it published: the lane and the timed
        # attempts run sequentially inside one supervised session, so a fixture
        # publishing a chain wall shorter than their sum is one no supervisor
        # can observe.
        "timing_seconds": {
            "chain_wall": 1.0
            + sum(
                float(draw["supervised_seconds"])
                for draw in (*attempts, *(() if cold_lane is None else (cold_lane,)))
            )
        },
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
    published_ledger = (
        _synthetic_ledger(gated=gated, weighted_total=terminal_objective)
        if ledger is None
        else ledger
    )
    attempt["evidence"] = {
        **attempt["evidence"],
        "certified_options_delta": (
            _options_delta(iterations) if options_delta is None else options_delta
        ),
        "compilation_cache": _attempt_cache(warm=warm),
        "endpoint_ledger": published_ledger,
        "solve": _solve_payload(
            latched=outcome == "LATCHED",
            terminal_objective=terminal_objective,
            maximum_feasibility_inf=maximum_feasibility_inf,
        ),
        # The standalone half is the ledger's own terminal weighted total,
        # because in the producer both are one evaluation of one state.
        "endpoint_agreement": _endpoint_agreement(
            exact_numeric_tree_sha256(coordinates),
            terminal_objective=terminal_objective,
            standalone_terminal_objective=published_ledger["terminal"]["weighted_total"],
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

    failed = _synthetic_ledger(gated=True, **{"raw.non_qs": 3.6e-4 * 1.01})
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


def _rebind_chain_wall(evidence: dict) -> dict:
    """Restate the root's chain wall around the draws the receipt now carries.

    The lane and the timed attempts run sequentially inside one supervised
    session, so the root's own wall is at least their sum -- and a test that
    ADDS or REPLACES a draw after ``_root_evidence`` built the receipt would
    otherwise be publishing a chain wall no supervisor could have observed, and
    would be refused for the timing rather than for the thing it is about.
    """

    draws = [
        *evidence["attempts"],
        *(() if evidence["cold_lane"] is None else (evidence["cold_lane"],)),
    ]
    evidence["timing_seconds"]["chain_wall"] = 1.0 + sum(
        float(draw["supervised_seconds"]) for draw in draws
    )
    return evidence


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
    published_wall = evidence["timing_seconds"]["chain_wall"]
    mutate(evidence)
    # A mutation that ADDS or REPLACES a draw leaves the chain wall describing
    # the draws the receipt used to carry, and the forgery would then be refused
    # for its timing rather than for the thing it is about -- the "refused for a
    # narrower reason than the test claims" class.  A mutation that forges the
    # chain wall ITSELF is left exactly as it wrote it.
    if evidence.get("timing_seconds", {}).get("chain_wall") == published_wall:
        _rebind_chain_wall(evidence)
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


# -------------------------------------------------- the round-5 forged receipts
#
# The round-5 review published twenty-two receipts through the REAL
# ``publish_root`` that sealed as ``CLAIM_DISCHARGED`` and re-validated clean
# from their sealed bytes.  Two of them forged a fact that DECIDES the claim --
# the route the attempt ran, and the worst iterate the feasibility gate reads --
# and one forged the pre-registration fact the headline verdict rests on.  Each
# is re-published here and must be refused before the seal, by a reason that
# names the gate that refuses it.


def test_publication_refuses_an_attempt_that_ran_a_different_route(
    tmp_path: Path,
) -> None:
    """A5-1/E5-1/P5-1: the receipt is bound to the certified route's VALUES.

    The options block was checked for its KEY SET and its published delta was
    re-derived from the published options -- and then the delta was constrained
    by nothing at all.  Twenty-one of the twenty-four fields were free, so a
    ``CLAIM_DISCHARGED`` receipt could declare ``lagrangian_newton: false``
    (the arm that IS the route under certification), ``gauss_newton: true``, a
    disabled line search, a backtracker that never contracts, or
    ``feasibility_tolerance: 1e-3`` beside ``claim.feasibility_tolerance:
    1e-10`` -- each with a truthful delta -- and re-validate clean.  Six roots
    of that shape sealed in one review.

    At the certified budget the honest delta is EMPTY, which is what makes the
    subset rule and the plan's sentence the same rule.
    """

    assert _options_delta(rehearsal.CERTIFIED_MAXIMUM_ITERATIONS) == {}
    assert _options_delta(3) == {"maximum_iterations": 3}

    for field, value in (
        ("lagrangian_newton", False),
        ("gauss_newton", True),
        ("frozen_projector_line_search", False),
        ("backtracking_factor", 1.0),
        ("feasibility_tolerance", 1.0e-3),
        ("newton_tangent_fraction_threshold", 0.99),
    ):

        def substitute(evidence: dict, field: str = field, value: object = value) -> None:
            for record in (evidence["attempts"][0], evidence["cold_lane"]):
                options = {**record["evidence"]["options"], field: value}
                record["evidence"]["options"] = options
                # The delta a forger publishes is TRUTHFUL: what the previous
                # revision could not tell is a substituted route from a bounded
                # budget, not a consistent receipt from an inconsistent one.
                record["evidence"]["certified_options_delta"] = {
                    name: published
                    for name, published in options.items()
                    if published
                    != rehearsal.json_scalar(
                        getattr(rehearsal.CERTIFIED_ROUTE_OPTIONS, name)
                    )
                }

        root, evidence = _mutated_root(tmp_path, f"route_{field}", substitute)
        _refuse_published(
            root, evidence, match="ran a route other than the certified one"
        )


def test_the_solve_summary_is_the_one_its_published_iterates_derive(
    tmp_path: Path,
) -> None:
    """A5-2: the feasibility gate may not read a scalar its own rows contradict.

    ``maximum_feasibility_inf`` is ``max`` over the iterates the same receipt
    publishes as ``solve.rows``, and nothing compared them: a receipt carrying
    recorded iterates at 0.005 and 0.027 -- nine decades outside the tolerance
    the claim is stated at -- sealed ``CLAIM_DISCHARGED`` beside a summary of
    1e-14, so a reader who did the arithmetic the receipt invites got a
    different answer from the validator that accepted it.  The same receipt
    published ``iterations_run: 700`` with zero rows, ``latched: true`` beside
    ``status_name: LINE_SEARCH_COLLAPSE`` and ``stored_pairs: -5``.
    """

    def infeasible_iterates(evidence: dict) -> None:
        solve = evidence["attempts"][0]["evidence"]["solve"]
        solve["rows"] = [
            {"index": 0, "objective": 4.48e-6, "feasibility_inf": 0.005},
            {"index": 1, "objective": 4.48e-8, "feasibility_inf": 0.027},
        ]
        solve["iterations_run"] = 2

    root, evidence = _mutated_root(tmp_path, "iterates", infeasible_iterates)
    _refuse_published(
        root, evidence, match="its own recorded iterates do not carry"
    )

    def budget_without_iterates(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"]["iterations_run"] = 700

    root, evidence = _mutated_root(tmp_path, "no_iterates", budget_without_iterates)
    _refuse_published(root, evidence, match="against 700 iterations run")

    def ascending_objectives(evidence: dict) -> None:
        solve = evidence["attempts"][0]["evidence"]["solve"]
        solve["rows"] = list(reversed(solve["rows"]))

    root, evidence = _mutated_root(tmp_path, "ascending", ascending_objectives)
    _refuse_published(
        root, evidence, match="not what its recorded objectives derive"
    )

    def latch_without_the_status(evidence: dict) -> None:
        solve = evidence["attempts"][0]["evidence"]["solve"]
        solve["status"] = int(launcher.ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE)
        solve["status_name"] = launcher.ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE.name

    root, evidence = _mutated_root(tmp_path, "latch_status", latch_without_the_status)
    _refuse_published(root, evidence, match="publishes latched=True under status")

    def invented_status(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"]["status"] = 99

    root, evidence = _mutated_root(tmp_path, "status", invented_status)
    _refuse_published(root, evidence, match="not one the engine reports")

    def negative_counter(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"]["stored_pairs"] = -5

    root, evidence = _mutated_root(tmp_path, "counter", negative_counter)
    _refuse_published(root, evidence, match="which is not a count")

    def an_iterate_without_its_feasibility(evidence: dict) -> None:
        rows = evidence["attempts"][0]["evidence"]["solve"]["rows"]
        rows[0] = {"index": 0, "objective": rows[0]["objective"]}

    root, evidence = _mutated_root(
        tmp_path, "row_without_feasibility", an_iterate_without_its_feasibility
    )
    _refuse_published(root, evidence, match="iterate 0 publishes no feasibility_inf")


def test_a_published_cold_lane_must_be_a_draw_of_its_own(tmp_path: Path) -> None:
    """A5-3/P5-11: ruling 18 bound the lane's DIRECTORY, not a cold DRAW.

    ``cold_lane_authorized`` is the lane's only channel to ``PREREGISTERED`` and
    therefore to the headline verdict, and the three forms round 4 published are
    refused -- but a forger paid one ``mkdir``: an EMPTY ``cold-lane/`` beside a
    record that produced nothing, and a ``cold-lane/`` holding a byte-copy of
    ``attempts/attempt-1`` beside a copy of attempt 1's own record, both still
    minted ``CLAIM_DISCHARGED``.

    What separates a draw from a retelling is not its endpoint -- two honest
    draws of the same problem at the same budget produced BITWISE IDENTICAL
    worst iterates on the 5090 -- but its INVOCATION, its cache and the wall the
    supervisor actually waited.  Ruling 17 is preserved in the other direction
    below: an honest lane that really timed out publishes.
    """

    def the_lanes_invocation_is_attempt_ones(evidence: dict) -> None:
        cold = evidence["cold_lane"]
        cold["argv_sha256"] = evidence["attempts"][0]["argv_sha256"]
        cold["gpu_memory"]["child_argv_sha256"] = cold["argv_sha256"]

    root, evidence = _mutated_root(
        tmp_path, "lane_copies_a_draw", the_lanes_invocation_is_attempt_ones
    )
    _refuse_published(root, evidence, match="copy of a draw rather than a draw")

    # The whole published form: the lane's record IS attempt 1's, re-stamped.
    copied = tmp_path / "lane_is_attempt_one"
    staging = copied / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    (staging / launcher.COLD_LANE_DIRECTORY).mkdir(parents=True)
    shutil.copyfile(
        staging / "attempts" / "attempt-1" / rehearsal.TERMINAL_COORDINATES_FILENAME,
        staging
        / launcher.COLD_LANE_DIRECTORY
        / rehearsal.TERMINAL_COORDINATES_FILENAME,
    )
    lane = json.loads(json.dumps(attempt))
    lane["attempt_index"] = 0
    lane["artifact_relative_path"] = launcher.COLD_LANE_DIRECTORY
    lane["outcome"] = "COMPLETED_WITHOUT_LATCH"
    lane["timed_against_bar"] = False
    lane["evidence"]["attempt_index"] = 0
    lane["evidence"]["solve"] = _solve_payload(
        latched=False, terminal_objective=4.48e-8, maximum_feasibility_inf=1.0e-14
    )
    lane["evidence"]["endpoint_ledger"] = _synthetic_ledger(gated=False)
    lane["evidence"]["compilation_cache"] = _attempt_cache(warm=False)
    _refuse_published(
        copied,
        _root_evidence(
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            attempts=[attempt],
            cold_lane=lane,
        ),
        match="copy of a draw rather than a draw",
    )

    # An EMPTY directory beside a lane that claims a timeout it did not wait
    # for.  ``communicate(timeout=...)`` cannot raise before its timeout
    # elapses, so a lane supervised for 105 s under a 3600 s timeout was never
    # launched.
    def a_timeout_nobody_waited_for(evidence: dict) -> None:
        cold = _attempt("TIMEOUT", index=0)
        cold["artifact_relative_path"] = launcher.COLD_LANE_DIRECTORY
        cold["timed_against_bar"] = False
        cold["evidence"] = None
        evidence["cold_lane"] = cold
        evidence["cold_lane_anomaly"] = launcher.cold_lane_anomaly(cold)

    root, evidence = _mutated_root(
        tmp_path, "unwaited_timeout", a_timeout_nobody_waited_for
    )
    _refuse_published(root, evidence, match="claims a timeout after")

    # And the honest form of exactly that lane publishes: an infrastructure
    # failure is not evidence about the claim (ruling 17), so it is recorded as
    # an anomaly beside a discharged root rather than destroying it.
    def an_honest_timeout(evidence: dict) -> None:
        cold = _attempt("TIMEOUT", index=0)
        cold["artifact_relative_path"] = launcher.COLD_LANE_DIRECTORY
        cold["timed_against_bar"] = False
        cold["evidence"] = None
        cold["supervised_seconds"] = launcher.ATTEMPT_TIMEOUT_SECONDS + 1.0
        evidence["cold_lane"] = cold
        evidence["cold_lane_anomaly"] = launcher.cold_lane_anomaly(cold)

    root, evidence = _mutated_root(tmp_path, "honest_timeout", an_honest_timeout)
    published = launcher.publish_root(root / "staging", root / "final", evidence)
    sealed = launcher.validate_root_artifact(published)
    assert sealed["verdict"] == launcher.VERDICT_CLAIM_DISCHARGED
    assert sealed["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_PREREGISTERED
    )
    assert sealed["cold_lane_anomaly"]["outcome"] == "TIMEOUT"


def test_publication_refuses_a_lowering_the_certified_route_does_not_select(
    tmp_path: Path,
) -> None:
    """Ruling 16's kernel list, re-derived rather than asserted.

    The list was checked for non-emptiness and for its own internal sum, so a
    receipt publishing one invented kernel of one IR byte re-validated clean --
    and so did the suite's own fixture, which published two kernel names this
    repository never lowers.  Four reviewers refuted the word "re-derived" for
    it in one round.  WHICH kernels are lowered is a function of the
    configuration, so the list is now the campaign's own.
    """

    def invented_kernels(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["lowering_pre_gate"] = {
            "rehearsal_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            "certified_iterations": rehearsal.CERTIFIED_MAXIMUM_ITERATIONS,
            "budget_independent": True,
            "kernels": [
                {"name": "not_a_real_kernel", "ir_bytes": 1, "while_operations": 0}
            ],
            "total_ir_bytes": 1,
        }

    root, evidence = _mutated_root(tmp_path, "invented_kernels", invented_kernels)
    _refuse_published(
        root, evidence, match="not the kernel set the certified configuration selects"
    )

    def a_kernel_dropped(evidence: dict) -> None:
        lowering = evidence["attempts"][0]["evidence"]["lowering_pre_gate"]
        kernels = lowering["kernels"][:-1]
        lowering["kernels"] = kernels
        lowering["total_ir_bytes"] = sum(kernel["ir_bytes"] for kernel in kernels)

    root, evidence = _mutated_root(tmp_path, "kernel_dropped", a_kernel_dropped)
    _refuse_published(
        root, evidence, match="not the kernel set the certified configuration selects"
    )

    def a_kernel_of_no_ir(evidence: dict) -> None:
        lowering = evidence["attempts"][0]["evidence"]["lowering_pre_gate"]
        lowering["kernels"][0]["ir_bytes"] = 0
        lowering["total_ir_bytes"] = sum(
            kernel["ir_bytes"] for kernel in lowering["kernels"]
        )

    root, evidence = _mutated_root(tmp_path, "kernel_no_ir", a_kernel_of_no_ir)
    _refuse_published(root, evidence, match="which is not a lowering")


def test_publication_refuses_a_module_the_manifest_does_not_cover(
    tmp_path: Path,
) -> None:
    """E5-4: the escape half of the custody block had no reader.

    ``unmanifested_repository_modules`` is where a repository module that
    resolved outside the manifest's roots LANDS -- the scikit-build-core
    editable-finder class the whole block exists to catch -- and it was
    shape-checked and read by nothing.  A certified launch imports nothing from
    the tree but the three manifested roots: measured ``[]`` on both lanes of
    the bounded 5090 smoke.
    """

    def an_unmanifested_module(evidence: dict) -> None:
        sources = evidence["attempts"][0]["evidence"]["execution_sources"]
        sources["unmanifested_repository_modules"] = [
            {"module": "somewhere.else", "relative_path": "elsewhere/module.py"}
        ]

    root, evidence = _mutated_root(tmp_path, "unmanifested", an_unmanifested_module)
    _refuse_published(
        root, evidence, match="repository modules the manifest does not describe"
    )


def test_every_draw_is_a_child_this_supervisor_launched(tmp_path: Path) -> None:
    """A5-8/A5-9: the telemetry, the device and the wall chain, all unread.

    Each draw carries three independent traces of its own launch -- the digest
    of the argv the supervisor used, the sampler's digest of the argv it
    OBSERVED on the device, and the nested walls -- and none was read, so an
    attempt could name another GPU, publish telemetry for another child, or
    report a 1e-9 s attempt wall around a 187 s engine.
    """

    def another_device(evidence: dict) -> None:
        evidence["attempts"][0]["gpu_memory"]["device_uuid"] = "GPU-0000dead"

    root, evidence = _mutated_root(tmp_path, "another_device", another_device)
    _refuse_published(root, evidence, match="was observed on GPU")

    def telemetry_of_another_child(evidence: dict) -> None:
        evidence["attempts"][0]["gpu_memory"]["child_argv_sha256"] = "0" * 64

    root, evidence = _mutated_root(tmp_path, "another_child", telemetry_of_another_child)
    _refuse_published(root, evidence, match="telemetry for a child other than")

    def a_wall_inside_its_engine(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["timing_seconds"]["attempt_wall"] = 1.0e-9

    root, evidence = _mutated_root(tmp_path, "wall_nesting", a_wall_inside_its_engine)
    _refuse_published(root, evidence, match="timings do not nest")

    def a_negative_phase(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["timing_seconds"]["bootstrap"] = -50.0

    root, evidence = _mutated_root(tmp_path, "negative_phase", a_negative_phase)
    _refuse_published(root, evidence, match="which is not a duration")

    def a_cache_it_did_not_enter(evidence: dict) -> None:
        cache = evidence["attempts"][0]["evidence"]["compilation_cache"]
        cache["at_entry"] = _cache_state(0)

    root, evidence = _mutated_root(tmp_path, "warm_lie", a_cache_it_did_not_enter)
    _refuse_published(root, evidence, match="against a cache holding")


# ------------------------------------------------- the round-4 forged receipts
#
# Six receipts were published through the REAL ``publish_root`` by the round-4
# adversarial review and re-validated clean from their sealed bytes, four of
# them as ``CLAIM_DISCHARGED``.  Each is re-published here, by the scenario
# letter it carried in that review, and each must now be refused before the
# seal.  They are kept as named tests rather than folded into the tables above
# because a forgery that once worked is the only evidence that a fix works.


def test_round4_forgery_i_hollowed_custody_blocks_are_refused(tmp_path: Path) -> None:
    """Scenario I: the three blocks below the frozen floor, reduced to nothing.

    ``execution_sources`` to ``{}``, ``problem_identity`` to the two keys the
    validator read, ``lowering_pre_gate`` to the one -- on the discharging
    attempt AND on the cold lane.  Published and re-validated clean as
    ``CLAIM_DISCHARGED``: the shape tree stopped at the blocks it enumerated,
    and these three were not among them.

    Each block is hollowed INDIVIDUALLY and refused by ITS OWN name.  The
    previous form of this test hollowed all three and asserted one message, so
    the walk raised at the first block and the other two hollowings were never
    evaluated: the test proved one third of its docstring, which is the
    finding class it exists to close.
    """

    hollowed = {
        "execution_sources": {},
        "problem_identity": {"bound": True, "sha_is_binding": False},
        "lowering_pre_gate": {"budget_independent": True},
    }
    for block, hollow_value in hollowed.items():

        def hollow(evidence: dict, block: str = block, value: object = hollow_value) -> None:
            for record in (evidence["attempts"][0], evidence["cold_lane"]):
                record["evidence"][block] = value

        root, evidence = _mutated_root(tmp_path, f"hollow_{block}", hollow)
        _refuse_published(root, evidence, match=f"{block} is incomplete")

    def hollow_all_three(evidence: dict) -> None:
        for record in (evidence["attempts"][0], evidence["cold_lane"]):
            record["evidence"].update(hollowed)

    root, evidence = _mutated_root(tmp_path, "hollow_custody", hollow_all_three)
    _refuse_published(root, evidence, match="execution_sources is incomplete")


def test_round4_forgery_i2_a_null_execution_sources_block_is_refused(
    tmp_path: Path,
) -> None:
    """Scenario I2: ``execution_sources: null`` on the discharging attempt.

    The one key of ``ATTEMPT_EVIDENCE_REQUIRED_KEYS`` that no code in the module
    read.  ``null``, ``{}``, ``"a string"`` and ``{"bound_modules": []}`` all
    sealed as ``CLAIM_DISCHARGED`` and re-validated clean, on both lanes, so the
    receipt asserted nothing at all about which bytes executed.
    """

    for name, value in (
        ("null", None),
        ("empty", {}),
        ("string", "a string"),
        ("no_modules", {"bound_modules": []}),
    ):

        def forge(evidence: dict, value: object = value) -> None:
            evidence["attempts"][0]["evidence"]["execution_sources"] = value

        root, evidence = _mutated_root(tmp_path, f"execution_sources_{name}", forge)
        _refuse_published(
            root, evidence, match="execution_sources is (not a document|incomplete)"
        )

    def other_repository(evidence: dict) -> None:
        sources = evidence["attempts"][0]["evidence"]["execution_sources"]
        sources["bound_modules"] = [
            {
                "module": "somewhere.else",
                "relative_path": "src/simsopt_jax/geo/optimizers/projected_lbfgs.py",
                "sha256": "0" * 64,
                "size_bytes": 1,
            }
        ]

    root, evidence = _mutated_root(tmp_path, "other_repository", other_repository)
    _refuse_published(root, evidence, match="bytes the manifest does not describe")

    def missing_engine(evidence: dict) -> None:
        sources = evidence["attempts"][0]["evidence"]["execution_sources"]
        sources["bound_modules"] = [
            module
            for module in sources["bound_modules"]
            if not module["relative_path"].startswith("src/")
        ]

    root, evidence = _mutated_root(tmp_path, "missing_engine", missing_engine)
    _refuse_published(root, evidence, match="modules the certified chain runs through")

    # The two gates of ``_validate_execution_sources`` the four shape forms
    # above never reach: the four legs are refused by the outer key-set check
    # before the re-derivation runs at all, so without these the block's own
    # manifest comparison and its empty-bound-set refusal are exercised by
    # nothing in the suite.
    def a_complete_block_binding_nothing(evidence: dict) -> None:
        sources = evidence["attempts"][0]["evidence"]["execution_sources"]
        sources["bound_modules"] = []

    root, evidence = _mutated_root(tmp_path, "no_bound_modules", a_complete_block_binding_nothing)
    _refuse_published(root, evidence, match="binds no manifest module")

    def another_manifest(evidence: dict) -> None:
        sources = evidence["attempts"][0]["evidence"]["execution_sources"]
        sources["manifest"] = {**sources["manifest"], "entries_sha256": "0" * 64}

    root, evidence = _mutated_root(tmp_path, "another_manifest", another_manifest)
    _refuse_published(
        root, evidence, match="execution-source manifest other than the campaign's"
    )


def test_round4_forgery_h_a_nulled_leaf_is_refused(tmp_path: Path) -> None:
    """Scenario H: every leaf the validator did not read, set to null.

    The shape tree refused a non-mapping where a block was declared, which is
    what closed the round-3 finding, and then skipped every leaf.  A
    ``CLAIM_DISCHARGED`` receipt whose cache accounting was three nulls, whose
    device inventory was null and whose telemetry was entirely null published
    and re-validated clean -- "missing its cache accounting and telemetry" in
    any sense a reader cares about.
    """

    def null_runtime_identity(evidence: dict) -> None:
        identity = evidence["supervisor"]["runtime_identity"]
        evidence["supervisor"]["runtime_identity"] = dict.fromkeys(identity)

    def null_cache_accounting(evidence: dict) -> None:
        cache = evidence["compilation_cache"]
        evidence["compilation_cache"] = dict.fromkeys(cache)

    def null_chain_wall(evidence: dict) -> None:
        evidence["timing_seconds"]["chain_wall"] = None

    def null_snapshot_digest(evidence: dict) -> None:
        evidence["source_snapshot"]["manifest_sha256"] = None

    def null_worktree(evidence: dict) -> None:
        worktree = evidence["source_snapshot"]["worktree"]
        evidence["source_snapshot"]["worktree"] = dict.fromkeys(worktree)

    def null_probe_identity(evidence: dict) -> None:
        for probe in evidence["supervisor"]["preflight"]["storage"]:
            probe["directory"] = None
            probe["resolved_directory"] = None

    def null_device_inventory(evidence: dict) -> None:
        evidence["supervisor"]["preflight"]["visible_gpu_uuids"] = None

    def null_telemetry(evidence: dict) -> None:
        memory = evidence["attempts"][0]["gpu_memory"]
        evidence["attempts"][0]["gpu_memory"] = {
            **dict.fromkeys(memory),
            "device_uuid": launcher.GPU_UUID,
        }

    def null_attempt_cache(evidence: dict) -> None:
        cache = evidence["attempts"][0]["evidence"]["compilation_cache"]
        for state in ("at_entry", "before_engine", "after"):
            cache[state] = dict.fromkeys(cache[state])

    def null_solve_status(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"]["status_name"] = None

    def null_argv(evidence: dict) -> None:
        evidence["attempts"][0]["argv_sha256"] = None

    for name, mutate in (
        ("runtime_identity", null_runtime_identity),
        ("cache_accounting", null_cache_accounting),
        ("chain_wall", null_chain_wall),
        ("snapshot_digest", null_snapshot_digest),
        ("worktree", null_worktree),
        ("probe_identity", null_probe_identity),
        ("device_inventory", null_device_inventory),
        ("telemetry", null_telemetry),
        ("attempt_cache", null_attempt_cache),
        ("solve_status", null_solve_status),
        ("argv", null_argv),
    ):
        root, evidence = _mutated_root(tmp_path, f"null_{name}", mutate)
        _refuse_published(root, evidence, match="is null where the receipt publishes")


def test_round4_forgery_j_a_wrongly_typed_leaf_is_refused(tmp_path: Path) -> None:
    """Scenario J: published scalars of the wrong type, accepted as they were.

    ``chain_wall: "not a number"``, ``entry_count: true``, ``total_bytes:
    "1e9"``, ``sample_count: false``, ``iterations_run: "seven"`` and
    ``attempt_wall: "-inf"`` all published and re-validated clean.  A leaf now
    states what its producer writes there, and ``bool`` is excluded from the
    number leaves explicitly because it is a subclass of ``int``.
    """

    def string_chain_wall(evidence: dict) -> None:
        evidence["timing_seconds"]["chain_wall"] = "not a number"

    def boolean_entry_count(evidence: dict) -> None:
        evidence["compilation_cache"]["entry_count"] = True

    def string_total_bytes(evidence: dict) -> None:
        evidence["compilation_cache"]["total_bytes"] = "1e9"

    def boolean_sample_count(evidence: dict) -> None:
        evidence["attempts"][0]["gpu_memory"]["sample_count"] = False

    def string_iterations_run(evidence: dict) -> None:
        evidence["attempts"][0]["evidence"]["solve"]["iterations_run"] = "seven"

    def string_attempt_wall(evidence: dict) -> None:
        timing = evidence["attempts"][0]["evidence"]["timing_seconds"]
        timing["attempt_wall"] = "-inf"

    def mapping_for_a_list(evidence: dict) -> None:
        evidence["supervisor"]["preflight"]["storage"] = {"role": "temporary"}

    for name, mutate in (
        ("chain_wall", string_chain_wall),
        ("entry_count", boolean_entry_count),
        ("total_bytes", string_total_bytes),
        ("sample_count", boolean_sample_count),
        ("iterations_run", string_iterations_run),
        ("attempt_wall", string_attempt_wall),
        ("storage", mapping_for_a_list),
    ):
        root, evidence = _mutated_root(tmp_path, f"typed_{name}", mutate)
        # The boolean forms name their own defect: ``true`` is not a count, and
        # two refusal sites that read identically cannot be told apart by the
        # coverage census.
        _refuse_published(
            root, evidence, match="(is not a|is a boolean where the receipt publishes)"
        )


def test_round4_forgery_k_a_cold_lane_aliased_onto_an_attempt_is_refused(
    tmp_path: Path,
) -> None:
    """Scenario K: ``PREREGISTERED`` minted with no cold lane in the tree.

    After the lane was taken out of the verdict, ``cold_lane_authorized``
    became its ONLY channel to the conformance label and therefore to the
    headline verdict -- and the only check on it was that a record existed.  A
    The case this fix left open -- a ``cold-lane`` directory that EXISTS with
    fabricated or copied content -- is
    ``test_a_published_cold_lane_must_be_a_draw_of_its_own``.

    A lane whose ``artifact_relative_path`` pointed at ``attempts/attempt-1`` was
    validated against attempt 1's own sealed array and passed, so a root with no
    ``cold-lane`` directory at all published ``CLAIM_DISCHARGED``.
    """

    root = tmp_path / "aliased"
    staging = root / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    aliased = _synthetic_attempt(
        staging / "unpublished-lane",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        outcome="COMPLETED_WITHOUT_LATCH",
        index=0,
        relative_path=f"{launcher.ATTEMPTS_DIRECTORY}/attempt-1",
        warm=False,
    )
    aliased["timed_against_bar"] = False
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[attempt],
        cold_lane=aliased,
    )
    _refuse_published(root, evidence, match="directory the protocol runs it in")

    # The same fact from the other side: an honestly named lane the tree does
    # not carry, and a tree that carries a lane the receipt does not claim.
    absent = tmp_path / "absent"
    staging = absent / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    unpublished = _synthetic_attempt(
        staging / "unpublished-lane-2",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
        outcome="COMPLETED_WITHOUT_LATCH",
        index=0,
        relative_path=launcher.COLD_LANE_DIRECTORY,
        warm=False,
    )
    unpublished["timed_against_bar"] = False
    evidence = _root_evidence(
        verdict=launcher.VERDICT_CLAIM_DISCHARGED,
        attempts=[attempt],
        cold_lane=unpublished,
    )
    _refuse_published(absent, evidence, match="carries no 'cold-lane' directory")

    unclaimed = tmp_path / "unclaimed"
    staging = unclaimed / "staging"
    attempt = _synthetic_attempt(
        staging / "attempts" / "attempt-1",
        engine_wall=1.0,
        terminal_objective=4.48e-8,
        maximum_feasibility_inf=1.0e-14,
    )
    (staging / launcher.COLD_LANE_DIRECTORY).mkdir(parents=True)
    (staging / launcher.COLD_LANE_DIRECTORY / "stray.json").write_bytes(b"{}\n")
    evidence = _root_evidence(
        verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt], cold_lane=None
    )
    _refuse_published(unclaimed, evidence, match="carries 'cold-lane' directory")


def test_round4_forgery_m_a_restated_relative_difference_is_refused(
    tmp_path: Path,
) -> None:
    """Scenario M: the distance column replaced with zeros beside honest sides.

    Both sides of the ledger are re-gated and the column that reports the
    distance between them was checked for its key set and never for its
    arithmetic, so a published receipt could report perfect agreement on every
    term while carrying the numbers that disagree.
    """

    def restate_column(evidence: dict) -> None:
        ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
        ledger["terminal"] = {
            **ledger["terminal"],
            "observable.iota": _scaled("observable.iota", 1.0 + 1.0e-9),
        }
        ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)
        ledger["relative_difference"] = dict.fromkeys(
            ledger["relative_difference"], 0.0
        )

    root, evidence = _mutated_root(tmp_path, "restated_column", restate_column)
    _refuse_published(root, evidence, match="relative differences are not the ones")


# ------------------------------------------------------- the mutation kill set
#
# The round-5 review deleted ``_validate_lowering_pre_gate`` and
# ``_validate_problem_identity`` outright, re-ran this file, and got 83 green:
# ruling 16's two most complete re-derivations were protected by nothing, and 37
# of the 82 refusal sites in the re-validation path were never reached by any
# test.  A gate no test can kill is a gate the next revision can delete.
#
# So every named validator owns one forgery that ONLY it refuses.  Each case is
# published twice through the real ``publish_root``: once whole, where the
# refusal must name that validator's own words, and once with the validator
# replaced by a no-op, where the same receipt must PUBLISH -- which is what
# proves the refusal came from there and from nowhere else.  The meta-test below
# requires the table to name every ``_validate_*`` function in the module, so a
# validator cannot be added without one.


def _forge_extra_claim_key(evidence: dict) -> None:
    evidence["claim"]["A_KEY_NO_PRODUCER_EMITS"] = 1.0


def _forge_untyped_cache_digest(evidence: dict) -> None:
    """A leaf whose ONLY reader is the type check, so the kill's other half holds.

    ``chain_wall`` used to carry this kill and can no longer: it is now read as a
    duration around the draws, so the deleted-validator half of the case --
    "with ``_validate_leaf`` a no-op the same receipt must PUBLISH" -- would fail
    inside the chain-wall arithmetic instead, proving nothing about the leaf
    walker.  The root cache's aggregate digest is shape-checked and read by
    nothing, which is exactly the property this kill needs.
    """

    evidence["compilation_cache"]["entries_digest"] = 12345


def _forge_tmpfs_storage(evidence: dict) -> None:
    evidence["supervisor"]["preflight"]["storage"][0]["filesystem_type"] = "tmpfs"


def _forge_other_bytes(evidence: dict) -> None:
    sources = evidence["attempts"][0]["evidence"]["execution_sources"]
    sources["bound_modules"][0] = {**sources["bound_modules"][0], "sha256": "0" * 64}


def _forge_widened_identity(evidence: dict) -> None:
    identity = evidence["attempts"][0]["evidence"]["problem_identity"]
    identity["relative_tolerances"] = dict.fromkeys(identity["relative_tolerances"], 1.0)


def _forge_invented_kernel(evidence: dict) -> None:
    lowering = evidence["attempts"][0]["evidence"]["lowering_pre_gate"]
    lowering["kernels"] = [
        {**lowering["kernels"][0], "name": "not_a_real_kernel"},
        *lowering["kernels"][1:],
    ]


def _forge_another_route(evidence: dict) -> None:
    for record in (evidence["attempts"][0], evidence["cold_lane"]):
        options = {**record["evidence"]["options"], "lagrangian_newton": False}
        record["evidence"]["options"] = options
        record["evidence"]["certified_options_delta"] = {
            name: value
            for name, value in options.items()
            if value
            != rehearsal.json_scalar(getattr(rehearsal.CERTIFIED_ROUTE_OPTIONS, name))
        }


def _forge_contradicted_summary(evidence: dict) -> None:
    solve = evidence["attempts"][0]["evidence"]["solve"]
    solve["rows"] = [
        {"index": 0, "objective": 4.48e-6, "feasibility_inf": 0.005},
        {"index": 1, "objective": 4.48e-8, "feasibility_inf": 0.027},
    ]
    solve["iterations_run"] = 2


def _forge_zeroed_distance_column(evidence: dict) -> None:
    # The term has to MOVE for the column to be a lie: the fixture's two sides
    # agree, so a column of zeros is the honest one there.
    ledger = evidence["attempts"][0]["evidence"]["endpoint_ledger"]
    ledger["terminal"] = {
        **ledger["terminal"],
        "observable.iota": _scaled("observable.iota", 1.0 + 1.0e-9),
    }
    ledger["pinned_term_gate"] = rehearsal.gate_endpoint_ledger(ledger)
    ledger["relative_difference"] = dict.fromkeys(ledger["relative_difference"], 0.0)


def _forge_another_device(evidence: dict) -> None:
    evidence["attempts"][0]["gpu_memory"]["device_uuid"] = "GPU-0000dead"


def _forge_lane_copying_a_draw(evidence: dict) -> None:
    cold = evidence["cold_lane"]
    cold["argv_sha256"] = evidence["attempts"][0]["argv_sha256"]
    cold["gpu_memory"]["child_argv_sha256"] = cold["argv_sha256"]


def _forge_truncated_record(evidence: dict) -> None:
    del evidence["attempts"][0]["stdout_tail"]


def _forge_outcome_its_evidence_denies(evidence: dict) -> None:
    solve = evidence["attempts"][0]["evidence"]["solve"]
    solve["latched"] = False
    solve["status"] = int(launcher.ProjectedLbfgsStatus.ITERATION_LIMIT)
    solve["status_name"] = launcher.ProjectedLbfgsStatus.ITERATION_LIMIT.name


def _forge_another_timing_boundary(evidence: dict) -> None:
    evidence["attempts"][0]["evidence"]["timing_boundary"] = "wall_clock"


def _forge_infeasible_iterate(evidence: dict) -> None:
    solve = evidence["attempts"][0]["evidence"]["solve"]
    solve["maximum_feasibility_inf"] = 1.0e-9
    solve["rows"] = _iterates(
        terminal_objective=4.48e-8, maximum_feasibility_inf=1.0e-9
    )


def _forge_lane_on_the_cpu(evidence: dict) -> None:
    evidence["cold_lane"]["evidence"]["runtime_identity"]["backend"] = "cpu"


def _forge_agreement_untied_to_the_run(evidence: dict) -> None:
    """Both halves of the agreement moved together, away from the solve summary.

    Moved together so the pair still certifies against each other: with
    ``_validate_terminal_endpoint_column`` deleted, nothing else in the receipt
    compares the agreement block to the run it is supposed to be an agreement
    ABOUT, which is the whole finding -- a self-contained pair of numbers
    agreeing to 5e-16 beside a terminal objective they have never met.
    """

    endpoint = evidence["attempts"][0]["evidence"]["endpoint_agreement"]
    endpoint["loop_terminal_objective"] = 1.0
    endpoint["standalone_terminal_objective"] = 1.0


_VALIDATOR_KILLS: tuple[tuple[str, object, str], ...] = (
    ("_validate_document_shape", _forge_extra_claim_key, "root.claim is incomplete"),
    (
        "_validate_leaf",
        _forge_untyped_cache_digest,
        "root.compilation_cache.entries_digest is not a string",
    ),
    (
        "_validate_terminal_endpoint_column",
        _forge_agreement_untied_to_the_run,
        "one measurement told twice",
    ),
    (
        "_validate_preflight_record",
        _forge_tmpfs_storage,
        "which plan section 11 refuses",
    ),
    (
        "_validate_execution_sources",
        _forge_other_bytes,
        "bytes the manifest does not describe",
    ),
    (
        "_validate_problem_identity",
        _forge_widened_identity,
        "problem identity is not the one its measured observables derive",
    ),
    (
        "_validate_lowering_pre_gate",
        _forge_invented_kernel,
        "not the kernel set the certified configuration selects",
    ),
    (
        "_validate_certified_route_options",
        _forge_another_route,
        "ran a route other than the certified one",
    ),
    (
        "_validate_solve_telemetry",
        _forge_contradicted_summary,
        "its own recorded iterates do not carry",
    ),
    (
        "_validate_endpoint_ledger_arithmetic",
        _forge_zeroed_distance_column,
        "relative differences are not the ones",
    ),
    ("_validate_supervised_launch", _forge_another_device, "was observed on GPU"),
    (
        "_validate_cold_lane_draw",
        _forge_lane_copying_a_draw,
        "copy of a draw rather than a draw",
    ),
    (
        "_validate_attempt_shape",
        _forge_truncated_record,
        "supervised attempt record is incomplete",
    ),
    (
        "_validate_attempt_outcome",
        _forge_outcome_its_evidence_denies,
        "is not the one its evidence derives",
    ),
    (
        "_validate_attempt_record",
        _forge_another_timing_boundary,
        "describes a different run than the record carrying it",
    ),
    ("_validate_attempt", _forge_infeasible_iterate, "published an infeasible iterate"),
    (
        "_validate_cold_lane",
        _forge_lane_on_the_cpu,
        "not the 'gpu' the wall is claimed on",
    ),
)


@pytest.mark.parametrize(
    "validator, forge, match",
    _VALIDATOR_KILLS,
    ids=[case[0] for case in _VALIDATOR_KILLS],
)
def test_deleting_a_named_validator_turns_a_refusal_into_a_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validator: str,
    forge: object,
    match: str,
) -> None:
    """Each named validator is the SOLE owner of one refusal this suite makes."""

    root, evidence = _mutated_root(tmp_path, f"{validator}_whole", forge)
    _refuse_published(root, evidence, match=match)

    monkeypatch.setattr(
        launcher, validator, lambda *arguments, **keywords: None, raising=True
    )
    deleted, evidence = _mutated_root(tmp_path, f"{validator}_deleted", forge)
    published = launcher.publish_root(
        deleted / "staging", deleted / "final", evidence
    )
    assert published.is_dir()


def test_every_named_validator_is_covered_by_the_mutation_kill_set() -> None:
    """A validator cannot be added without a forgery that proves it necessary.

    The structural half of the fix: the table is asserted to be exactly the
    module's own ``_validate_*`` surface, so the next revision cannot add a gate
    the suite never reaches -- which is how two whole validators came to be
    deletable with 83 tests green.
    """

    named = frozenset(
        name
        for name, value in vars(launcher).items()
        if name.startswith("_validate_") and callable(value)
    )
    assert named == frozenset(case[0] for case in _VALIDATOR_KILLS)


# --------------------------------------------------------- refusal-site census
#
# The validator kill table is true at FUNCTION granularity and was false one
# level down.  Measured against the previous revision: 52 of 127 refusal sites
# were reached by no test, 30 of them INSIDE functions the kill table claims to
# protect, and seven individual checks were deleted one at a time -- including
# the re-hash of the receipt's only re-evaluatable artifact -- with the whole
# suite green each time.  A gate no test can kill is a gate the next revision
# can delete, and that is as true of a check as of the function around it.
#
# So the site census below is to refusal SITES what ``UNSHAPED_LEAVES`` is to
# unshaped blocks: the suite walks the launcher's own ``raise`` statements and
# requires this map to be exactly what it finds, so a check cannot be added
# without a disposition.  A site is either killed by a case in
# ``_CHECK_KILLS`` -- published through the real path, refused, and the refusal
# traced back to that exact line -- or carries the reason it is not.


def _refusal_sites() -> dict[str, int]:
    """Every ``raise`` of a refusal in the launcher, keyed by owner and words.

    The key is the owning function plus the message TEMPLATE with its
    interpolations blanked, which is what a reader of a refusal actually sees
    and is stable under a re-worded value.  Two sites that read identically
    would collide, so the launcher names them apart.
    """

    source = (
        REPOSITORY / "benchmarks" / "run_single_stage_projected_route_gpu_root.py"
    ).read_text(encoding="utf-8")

    def template(node: ast.AST) -> str:
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else "{}"
        if isinstance(node, ast.JoinedStr):
            return "".join(template(part) for part in node.values)
        if isinstance(node, ast.FormattedValue):
            return "{}"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return template(node.left) + template(node.right)
        if isinstance(node, ast.Call):
            return "".join(template(argument) for argument in node.args)
        return "{}"

    sites: dict[str, int] = {}

    class Walk(ast.NodeVisitor):
        owner = "<module>"

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            outer, self.owner = self.owner, node.name
            self.generic_visit(node)
            self.owner = outer

        def visit_Raise(self, node: ast.Raise) -> None:
            call = node.exc
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                return
            if call.func.id not in {"ProjectedRootError", "RehearsalError"}:
                return
            words = " ".join(template(call.args[0]).split()) if call.args else ""
            key = f"{self.owner}: {words}"
            assert key not in sites, f"two refusal sites read identically: {key}"
            sites[key] = node.lineno

    Walk().visit(ast.parse(source))
    return sites


# The dispositions.  Each says exactly what is guaranteed and nothing more --
# the census is worthless if its reasons are the "because I said so" prose the
# UNSHAPED_LEAVES reasons were caught being twice.  Three of the six are checked
# by the meta-test rather than asserted.
_CHECK_KILLED = (
    "a case in _CHECK_KILLS publishes a forgery refused at this exact line "
    "(checked: the site is in that table)"
)
_UNREACHABLE_BY_CONSTRUCTION = (
    "no receipt can reach it: _validate_attempt_outcome runs first and derives "
    "PROTOCOL_FAILURE for any draw whose evidence is not a document, while this "
    "line needs an outcome that is neither TIMEOUT nor PROTOCOL_FAILURE beside "
    "evidence that is not a document. Defensive, and dead -- named here rather "
    "than given a kill test that would have to forge the impossible"
)
_PRODUCER_ONLY = (
    "producer-side: raised while the chain RUNS and unreachable from "
    "validate_root_artifact, so no sealed receipt can drive it (checked: the "
    "owner is not in the re-validation call graph)"
)
_WALKER_COVERED = (
    "the shape walker itself, driven over the whole tree by the truncation, "
    "nulled-leaf and wrongly-typed-leaf forgeries"
)
_OWNER_KILLED = (
    "the owning validator is in _VALIDATOR_KILLS, so the FUNCTION cannot be "
    "deleted silently; this branch is not pinned to its own line (checked: the "
    "owner is in that table)"
)
_NO_CHECK_KILL = (
    "no check-granularity kill, and the owner is not a named validator either, "
    "so neither table sees it. This round's declared residue"
)


def _solve_of(evidence: dict, *, lane: bool = False) -> dict:
    draw = evidence["cold_lane"] if lane else evidence["attempts"][0]
    return draw["evidence"]["solve"]


def _kill_status_name_disagrees(root: Path, evidence: dict) -> None:
    _solve_of(evidence)["status_name"] = "ITERATION_LIMIT"


def _kill_worst_iterate_without_iterates(root: Path, evidence: dict) -> None:
    solve = _solve_of(evidence)
    solve["rows"] = []
    solve["iterations_run"] = 0


def _kill_latch_without_iterates(root: Path, evidence: dict) -> None:
    solve = _solve_of(evidence)
    solve["rows"] = []
    solve["iterations_run"] = 0
    solve["maximum_feasibility_inf"] = None


def _kill_no_kernel_at_all(root: Path, evidence: dict) -> None:
    lowering = evidence["attempts"][0]["evidence"]["lowering_pre_gate"]
    lowering["kernels"] = []
    lowering["total_ir_bytes"] = 0


def _kill_terminal_state_is_another_array(root: Path, evidence: dict) -> None:
    """The re-hash of the receipt's ONE re-evaluatable artifact.

    Deleted one line at a time, this check was free: the suite never opened the
    published array, so a sealed root could carry a terminal state that is not
    the state its own digest names.
    """

    path = (
        root
        / "staging"
        / "attempts"
        / "attempt-1"
        / rehearsal.TERMINAL_COORDINATES_FILENAME
    )
    with path.open("wb") as stream:
        np.save(
            stream, np.asarray([1.0, 2.0, 3.0], dtype=np.float64), allow_pickle=False
        )


def _kill_identity_that_derives_unbound(root: Path, evidence: dict) -> None:
    """A problem identity that IS what its measurements derive, and is unbound.

    The whole-block equality is checked first, so the forgery has to be the
    producer's own derivation over observables that miss the campaign's -- which
    is exactly the shape a run on a different problem publishes.
    """

    drifted = {
        name: value * 2.0 for name, value in rehearsal.CPU_BOOTSTRAP_OBSERVABLES.items()
    }
    evidence["attempts"][0]["evidence"]["problem_identity"] = (
        rehearsal.problem_identity_evidence(
            drifted, problem_sha256="0" * 64, bootstrap_sha256="1" * 64
        )
    )


def _kill_preflight_without_the_pinned_device(root: Path, evidence: dict) -> None:
    evidence["supervisor"]["preflight"]["visible_gpu_uuids"] = ["GPU-not-the-one"]


def _kill_inventory_that_is_not_one(root: Path, evidence: dict) -> None:
    evidence["supervisor"]["preflight"]["visible_gpu_uuids"] = [
        launcher.GPU_UUID,
        12345,
        None,
        {"not": "a uuid"},
    ]


def _kill_relative_temporary_directory(root: Path, evidence: dict) -> None:
    """The absoluteness rule, which lived only in the producer.

    ``probe_writable_storage`` refuses a relative directory by name while the
    chain RUNS, so the receipt could declare one on all four fields and seal: a
    directory the children spill through that no third party can resolve.
    """

    preflight = evidence["supervisor"]["preflight"]
    preflight["temporary_directory"] = "relative/tmp"
    preflight["resolved_temporary_directory"] = "relative/tmp"
    preflight["storage"][0]["directory"] = "relative/tmp"
    preflight["storage"][0]["resolved_directory"] = "relative/tmp"


def _kill_negative_chain_wall(root: Path, evidence: dict) -> None:
    evidence["timing_seconds"]["chain_wall"] = -1.0e9


def _kill_fractional_budget(root: Path, evidence: dict) -> None:
    """700.9 certified iterations, which every reader used to truncate to 700."""

    options = evidence["attempts"][0]["evidence"]["options"]
    options["maximum_iterations"] = 700.9
    evidence["attempts"][0]["evidence"]["certified_options_delta"] = {
        "maximum_iterations": 700.9
    }


def _kill_agreement_untied_to_the_run(root: Path, evidence: dict) -> None:
    _forge_agreement_untied_to_the_run(evidence)


def _kill_standalone_untied_to_the_ledger(root: Path, evidence: dict) -> None:
    endpoint = evidence["attempts"][0]["evidence"]["endpoint_agreement"]
    endpoint["standalone_terminal_objective"] = (
        endpoint["standalone_terminal_objective"] * (1.0 + 1.0e-13)
    )


def _kill_feasibility_stated_against_another_tolerance(
    root: Path, evidence: dict
) -> None:
    endpoint = evidence["attempts"][0]["evidence"]["endpoint_agreement"]
    endpoint["feasibility_absolute_tolerance"] = 1.0


def _kill_terminal_feasibility_outside_the_tolerance(root: Path, evidence: dict) -> None:
    """Both copies moved together, so what refuses is the TOLERANCE, not a twin."""

    attempt = evidence["attempts"][0]["evidence"]
    attempt["solve"]["terminal_feasibility_inf"] = 0.99
    attempt["endpoint_agreement"]["terminal_feasibility_inf"] = 0.99


def _kill_nonfinite_terminal_on_a_completed_chain(root: Path, evidence: dict) -> None:
    """On the LANE, whose completed chain the latch gate never reaches."""

    _solve_of(evidence, lane=True)["terminal_objective"] = None


def _kill_iterate_at_or_below_the_target(root: Path, evidence: dict) -> None:
    """A recorded iterate the engine would have stopped before recording."""

    target = rehearsal.CERTIFIED_ROUTE_OPTIONS.objective_target
    solve = _solve_of(evidence)
    for index, row in enumerate(solve["rows"]):
        row["objective"] = target / (2.0**index)


def _kill_terminal_objective_off_the_trajectory(root: Path, evidence: dict) -> None:
    """A terminal objective neither endpoint of the last recorded iteration."""

    _solve_of(evidence)["terminal_objective"] = 4.0e-8
    evidence["attempts"][0]["evidence"]["endpoint_agreement"][
        "loop_terminal_objective"
    ] = 4.0e-8


# One forgery per CHECK, refused at that exact ``raise``.  Deleting the check
# makes the case red twice over: the refusal disappears, or it comes from
# another line.  That is the property ruling 22 states and held only of the
# function around the check.
_CHECK_KILLS: tuple[tuple[str, object, str], ...] = (
    (
        "_validate_solve_telemetry: attempt publishes status {} under the name {}, which the engine calls {}",
        _kill_status_name_disagrees,
        "under the name",
    ),
    (
        "_validate_solve_telemetry: attempt publishes a worst iterate {} with no iterates",
        _kill_worst_iterate_without_iterates,
        "with no iterates",
    ),
    (
        "_validate_solve_telemetry: attempt publishes a latch with no recorded iterate, so nothing it recorded reached the target it claims",
        _kill_latch_without_iterates,
        "a latch with no recorded iterate",
    ),
    (
        "_validate_solve_telemetry: attempt records iterate {} at objective {}, at or below the target {} the engine stops before recording",
        _kill_iterate_at_or_below_the_target,
        "at or below the target",
    ),
    (
        "_validate_solve_telemetry: attempt publishes a terminal objective {} that is neither endpoint of its last recorded iteration ({})",
        _kill_terminal_objective_off_the_trajectory,
        "neither endpoint of its last recorded iteration",
    ),
    (
        "_validate_lowering_pre_gate: attempt lowered no kernel at all",
        _kill_no_kernel_at_all,
        "lowered no kernel at all",
    ),
    (
        "_validate_attempt_record: published terminal state differs from its hash",
        _kill_terminal_state_is_another_array,
        "published terminal state differs from its hash",
    ),
    (
        "_validate_problem_identity: attempt claims an unbound problem",
        _kill_identity_that_derives_unbound,
        "claims an unbound problem",
    ),
    (
        "_validate_preflight_record: root preflight did not see the device the claim names ({})",
        _kill_preflight_without_the_pinned_device,
        "did not see the device the claim names",
    ),
    (
        "_validate_preflight_record: root preflight publishes a device inventory that is not one: {}",
        _kill_inventory_that_is_not_one,
        "device inventory that is not one",
    ),
    (
        "_validate_preflight_record: root preflight publishes {} as the {} the children spill through, which no reader can resolve",
        _kill_relative_temporary_directory,
        "which no reader can resolve",
    ),
    (
        "validate_root_artifact: root publishes a chain wall of {} s around draws it supervised for {} s",
        _kill_negative_chain_wall,
        "publishes a chain wall of",
    ),
    (
        "_validate_certified_route_options: attempt options publish maximum_iterations as {}, which is not a budget",
        _kill_fractional_budget,
        "which is not a budget",
    ),
    (
        "_validate_terminal_endpoint_column: attempt publishes a {} of {} in its solve summary and {} in its endpoint agreement, which are one measurement told twice",
        _kill_agreement_untied_to_the_run,
        "one measurement told twice",
    ),
    (
        "_validate_terminal_endpoint_column: attempt publishes a standalone terminal objective {} beside an endpoint ledger whose terminal weighted total is {}, which is the same evaluation of the same state",
        _kill_standalone_untied_to_the_ledger,
        "the same evaluation of the same state",
    ),
    (
        "_validate_terminal_endpoint_column: attempt states its terminal feasibility against {}, not the certified route's {}",
        _kill_feasibility_stated_against_another_tolerance,
        "states its terminal feasibility against",
    ),
    (
        "_validate_terminal_endpoint_column: attempt publishes a terminal feasibility {} outside the certified route's {}",
        _kill_terminal_feasibility_outside_the_tolerance,
        "outside the certified route's",
    ),
    (
        "_validate_terminal_endpoint_column: attempt publishes a completed chain whose {} is {}, which no chain that cleared the endpoint agreement can carry",
        _kill_nonfinite_terminal_on_a_completed_chain,
        "which no chain that cleared the endpoint agreement can carry",
    ),
)

# Every refusal the launcher raises, and what the suite guarantees about it.
# Generated once by walking the launcher's own ``raise`` statements and then
# dispositioned by hand; the meta-test below requires it to stay exactly what
# the walker finds, so a check cannot be added without a disposition.
_REFUSAL_SITES: dict[str, str] = {
    'bind_gpu_backend: resolved backend is {}, not {}': _PRODUCER_ONLY,
    'run_attempt: {} must equal {}, observed {}': _PRODUCER_ONLY,
    'run_attempt: iterate feasibility {} is not within {}': _PRODUCER_ONLY,
    'run_attempt: pinned endpoint terms differ from native: {}': _PRODUCER_ONLY,
    'attempt_engine_wall_seconds: attempt publishes no timing block to derive from': _NO_CHECK_KILL,
    'attempt_engine_wall_seconds: attempt publishes an engine compile/solve pair that is not a pair of durations: {} + {}': _NO_CHECK_KILL,
    'attempt_engine_wall_seconds: attempt engine wall {} is not its own compile plus solve ({})': _NO_CHECK_KILL,
    'attempt_engine_wall_seconds: attempt engine wall {} is not within the supervised wall {} it is a part of': _NO_CHECK_KILL,
    'filesystem_type: no mount in this namespace carries {}': _PRODUCER_ONLY,
    'probe_writable_storage: {} directory {} is relative; the supervisor would probe it against its own working directory while the children resolve it against {}': _PRODUCER_ONLY,
    'probe_writable_storage: {}': _PRODUCER_ONLY,
    'probe_writable_storage: {} directory {} is on {}; plan section 11 requires every directory this root writes to off tmpfs (set {} and the paths accordingly)': _PRODUCER_ONLY,
    'probe_writable_storage: {} directory {} refused a one-byte write: errno {} ({})': _PRODUCER_ONLY,
    'preflight_external_resources: {} is not on PATH': _PRODUCER_ONLY,
    'preflight_external_resources: pinned GPU {} is not among the visible devices {}': _PRODUCER_ONLY,
    'publish_source_snapshot: source changed during snapshot publication': _PRODUCER_ONLY,
    '_validate_document_shape: {} is not a document': _WALKER_COVERED,
    '_validate_document_shape: {} is incomplete: missing {}, unexpected {}': _WALKER_COVERED,
    '_validate_document_shape: {}.{} is not a published list': _WALKER_COVERED,
    '_validate_leaf: {} is null where the receipt publishes {}': _WALKER_COVERED,
    '_validate_leaf: {} is a boolean where the receipt publishes {}: {}': _WALKER_COVERED,
    '_validate_leaf: {} is not {}: {}': _WALKER_COVERED,
    "_validate_preflight_record: root preflight names a native endpoint reference other than the campaign's pinned one": _OWNER_KILLED,
    '_validate_preflight_record: root preflight publishes a device inventory that is not one: {}': _CHECK_KILLED,
    '_validate_preflight_record: root preflight did not see the device the claim names ({})': _CHECK_KILLED,
    '_validate_preflight_record: root published a {} directory on {}, which plan section 11 refuses': _OWNER_KILLED,
    '_validate_preflight_record: root published a {} directory whose write probe did not succeed': _OWNER_KILLED,
    '_validate_preflight_record: root preflight did not probe the three directories the protocol writes': _OWNER_KILLED,
    '_validate_preflight_record: root preflight publishes {} as the {} the children spill through, which no reader can resolve': _CHECK_KILLED,
    '_validate_preflight_record: root preflight probed {} and published {} as the {} the children spill through': _OWNER_KILLED,
    '_validate_execution_sources: the execution-source manifest this receipt is judged against is not loadable: {}': _OWNER_KILLED,
    "_validate_execution_sources: attempt names an execution-source manifest other than the campaign's: {}": _OWNER_KILLED,
    '_validate_execution_sources: attempt binds no manifest module, so its receipt says nothing about which bytes executed': _OWNER_KILLED,
    '_validate_execution_sources: attempt binds {} to {} with bytes the manifest does not describe': _OWNER_KILLED,
    '_validate_execution_sources: attempt does not bind the modules the certified chain runs through: {}': _OWNER_KILLED,
    '_validate_execution_sources: attempt executed repository modules the manifest does not describe: {}': _OWNER_KILLED,
    '_validate_problem_identity: attempt binds identity to an unstable sha': _OWNER_KILLED,
    "_validate_problem_identity: attempt publishes bootstrap observables other than the campaign's": _OWNER_KILLED,
    '_validate_problem_identity: attempt publishes a bootstrap observable that is not a number': _OWNER_KILLED,
    '_validate_problem_identity: attempt problem identity is not the one its measured observables derive': _OWNER_KILLED,
    '_validate_problem_identity: attempt claims an unbound problem': _CHECK_KILLED,
    '_validate_lowering_pre_gate: attempt claims budget-dependent lowering': _OWNER_KILLED,
    '_validate_lowering_pre_gate: attempt lowered against {} certified iterations, not {}': _OWNER_KILLED,
    '_validate_lowering_pre_gate: attempt lowered at {} iterations, not the {} it ran': _OWNER_KILLED,
    '_validate_lowering_pre_gate: attempt lowered no kernel at all': _CHECK_KILLED,
    '_validate_lowering_pre_gate: attempt lowered {}, which is not the kernel set the certified configuration selects ({})': _OWNER_KILLED,
    '_validate_lowering_pre_gate: attempt publishes kernel {} with {} IR bytes and {} while operations, which is not a lowering': _OWNER_KILLED,
    '_validate_lowering_pre_gate: attempt lowering total {} is not the sum of its kernels ({})': _OWNER_KILLED,
    "_validate_certified_route_options: attempt options are not the certified configuration's fields": _OWNER_KILLED,
    '_validate_certified_route_options: attempt options publish maximum_iterations as {}, which is not a budget': _CHECK_KILLED,
    '_validate_certified_route_options: attempt options delta {} is not the one its options derive ({})': _OWNER_KILLED,
    '_validate_certified_route_options: attempt ran a route other than the certified one: {}': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes {} recorded iterates against {} iterations run': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes {} as {}, which is not a count': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes a worst iterate {} with no iterates': _CHECK_KILLED,
    '_validate_solve_telemetry: attempt publishes a worst iterate feasibility {} that its own recorded iterates do not carry (their maximum is {})': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes a worst iterate feasibility {} that its own recorded iterates do not carry': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes monotone_descent={}, which is not what its recorded objectives derive ({})': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes status {}, which is not one the engine reports': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt publishes status {} under the name {}, which the engine calls {}': _CHECK_KILLED,
    '_validate_solve_telemetry: attempt publishes latched={} under status {}': _OWNER_KILLED,
    '_validate_solve_telemetry: attempt records iterate {} at objective {}, at or below the target {} the engine stops before recording': _CHECK_KILLED,
    '_validate_solve_telemetry: attempt publishes a latch with no recorded iterate, so nothing it recorded reached the target it claims': _CHECK_KILLED,
    '_validate_solve_telemetry: attempt publishes a terminal objective {} that is neither endpoint of its last recorded iteration ({})': _CHECK_KILLED,
    '_iterate_column: attempt iterate {} is not a document': _NO_CHECK_KILL,
    '_iterate_column: attempt iterate {} publishes no {}': _NO_CHECK_KILL,
    '_iterate_column: attempt iterate {} publishes {} as {}, which is not a measurement': _NO_CHECK_KILL,
    '_validate_endpoint_ledger_arithmetic: attempt endpoint ledger {} side publishes {} as {}, which is not a physics measurement': _OWNER_KILLED,
    '_validate_endpoint_ledger_arithmetic: attempt endpoint ledger sides do not carry the same terms': _OWNER_KILLED,
    '_validate_endpoint_ledger_arithmetic: attempt endpoint ledger relative differences are not the ones its two sides derive': _OWNER_KILLED,
    "_validate_terminal_endpoint_column: attempt states its terminal feasibility against {}, not the certified route's {}": _CHECK_KILLED,
    '_validate_terminal_endpoint_column: attempt publishes a completed chain whose {} is {}, which no chain that cleared the endpoint agreement can carry': _CHECK_KILLED,
    '_validate_terminal_endpoint_column: attempt publishes a {} of {} in its solve summary and {} in its endpoint agreement, which are one measurement told twice': _CHECK_KILLED,
    "_validate_terminal_endpoint_column: attempt publishes a terminal feasibility {} outside the certified route's {}": _CHECK_KILLED,
    '_validate_terminal_endpoint_column: attempt publishes a standalone terminal objective {} beside an endpoint ledger whose terminal weighted total is {}, which is the same evaluation of the same state': _CHECK_KILLED,
    'validate_root_artifact: re-validation requires JAX_ENABLE_X64=true: the published terminal state is a float64 array, and with x64 disabled its digest is re-derived at float32 and disagrees with every honest receipt': _NO_CHECK_KILL,
    'validate_root_artifact: root manifest differs from the artifact tree': _NO_CHECK_KILL,
    'validate_root_artifact: root evidence schema differs': _NO_CHECK_KILL,
    'validate_root_artifact: root evidence is not the complete receipt: missing {}, unexpected {}': _NO_CHECK_KILL,
    'validate_root_artifact: root evidence restates the native reference': _NO_CHECK_KILL,
    'validate_root_artifact: root evidence states a different timing boundary': _NO_CHECK_KILL,
    'validate_root_artifact: root names GPU {}, not the device the claim is stated for ({})': _NO_CHECK_KILL,
    'validate_root_artifact: root cold-lane authorization does not match the lane it published': _NO_CHECK_KILL,
    'validate_root_artifact: root cold-lane record is not a document': _NO_CHECK_KILL,
    'validate_root_artifact: root publishes its cold lane at {}, not in the {} directory the protocol runs it in': _NO_CHECK_KILL,
    "validate_root_artifact: root publishes a cold lane indexed {} among the protocol's own draws": _NO_CHECK_KILL,
    'validate_root_artifact: root claims cold_lane_authorized={} against a tree that {} {} directory': _NO_CHECK_KILL,
    'validate_root_artifact: root publishes no list of attempts': _NO_CHECK_KILL,
    "validate_root_artifact: root publishes two attempts launched by the same invocation, which the protocol's own per-attempt directory and index make impossible": _NO_CHECK_KILL,
    'validate_root_artifact: root publishes a chain wall of {} s around draws it supervised for {} s': _CHECK_KILLED,
    'validate_root_artifact: published cold-lane anomaly {} is not the one its lane derives ({})': _NO_CHECK_KILL,
    'validate_root_artifact: root published {} attempts under {} authorized': _NO_CHECK_KILL,
    'validate_root_artifact: attempt {} outcome {} does not obey the published stop rule: {}': _NO_CHECK_KILL,
    "validate_root_artifact: root attempts are not the protocol's consecutive draws, each in its own directory": _NO_CHECK_KILL,
    'validate_root_artifact: the artifact tree carries attempt directories the receipt does not publish: {}': _NO_CHECK_KILL,
    'validate_root_artifact: published attempt protocol {} {} is not the one the attempts derive ({})': _NO_CHECK_KILL,
    'validate_root_artifact: published quality claim {} is not the one its budget derives ({})': _NO_CHECK_KILL,
    'validate_root_artifact: published verdict {} is not the one the attempts derive ({})': _NO_CHECK_KILL,
    'validate_root_artifact: the cold lane may not be timed against the bar': _NO_CHECK_KILL,
    '_validate_attempt_shape: supervised attempt record is not a document': _OWNER_KILLED,
    '_validate_attempt_shape: supervised attempt record is incomplete: missing {}, unexpected {}': _OWNER_KILLED,
    '_validate_attempt_shape: attempt evidence is not a document': _OWNER_KILLED,
    '_validate_attempt_shape: attempt evidence document is incomplete: missing {}, unexpected {}': _OWNER_KILLED,
    '_validate_attempt_outcome: attempt outcome {} is not the one its evidence derives ({})': _OWNER_KILLED,
    '_validate_supervised_launch: {} publishes device telemetry for a child other than the one it launched': _OWNER_KILLED,
    '_validate_supervised_launch: {} was observed on GPU {}, not the device the claim is stated for ({})': _OWNER_KILLED,
    '_validate_supervised_launch: {} names the supervisor as its own child process': _OWNER_KILLED,
    '_validate_supervised_launch: {} publishes a supervised wall of {}, which is not a duration': _OWNER_KILLED,
    '_validate_supervised_launch: {} claims a timeout after {} s under the {} s timeout it publishes': _OWNER_KILLED,
    '_validate_attempt_record: attempt carries no evidence document': _UNREACHABLE_BY_CONSTRUCTION,
    '_validate_attempt_record: attempt evidence describes a different run than the record carrying it': _OWNER_KILLED,
    '_validate_attempt_record: attempt names backend {}, not the {} the wall is claimed on': _OWNER_KILLED,
    '_validate_attempt_record: attempt ran under an environment the route forbids': _OWNER_KILLED,
    '_validate_attempt_record: attempt publishes {} as {}, which is not a duration': _OWNER_KILLED,
    '_validate_attempt_record: attempt timings do not nest: engine wall {}, attempt wall {}, supervised wall {}': _OWNER_KILLED,
    '_validate_attempt_record: attempt ran {} iterations, not the {} its protocol claims': _OWNER_KILLED,
    '_validate_attempt_record: attempt quality claim {} is not the one its budget derives ({})': _OWNER_KILLED,
    '_validate_attempt_record: attempt publishes warm={} against a cache holding {} entries at entry': _OWNER_KILLED,
    '_validate_attempt_record: attempt targets an objective other than the native endpoint': _OWNER_KILLED,
    '_validate_attempt_record: attempt published a latch above the native endpoint objective': _OWNER_KILLED,
    "_validate_attempt_record: attempt endpoint ledger scope differs from the campaign's": _OWNER_KILLED,
    '_validate_attempt_record: attempt endpoint ledger names another native endpoint reference': _OWNER_KILLED,
    "_validate_attempt_record: attempt endpoint ledger {} side does not carry the campaign's terms": _OWNER_KILLED,
    '_validate_attempt_record: attempt ledger claims gated={}, which is not what its budget and outcome derive ({})': _OWNER_KILLED,
    '_validate_attempt_record: attempt pinned-term gate is not the one its ledger derives': _OWNER_KILLED,
    "_validate_attempt_record: attempt endpoint tolerances differ from the campaign's": _OWNER_KILLED,
    '_validate_attempt_record: attempt published a completed chain whose directory {} carries no {}': _OWNER_KILLED,
    '_validate_attempt_record: published terminal state differs from its hash': _CHECK_KILLED,
    '_validate_attempt: attempt discharges the claim with a failed pinned-term gate: {}': _OWNER_KILLED,
    "_validate_attempt: attempt endpoint differs from the campaign's frozen native reference on {}": _OWNER_KILLED,
    '_validate_attempt: attempt published an infeasible iterate': _OWNER_KILLED,
    "_validate_cold_lane_draw: root publishes a cold lane whose invocation is a timed attempt's, so its record is a copy of a draw rather than a draw": _OWNER_KILLED,
    '_validate_cold_lane_draw: the cold lane ran against a populated cache ({} entries at entry)': _OWNER_KILLED,
    'run_attempt_protocol: attempt protocol must authorize at least one attempt': _PRODUCER_ONLY,
    'run_attempt_protocol: attempt budget must be positive': _PRODUCER_ONLY,
    'run_attempt_protocol: this plan certifies {}, not {}': _PRODUCER_ONLY,
    'run_attempt_protocol: root output already exists: {}': _PRODUCER_ONLY,
    'run_attempt_protocol: compilation cache directory must start empty: {}': _PRODUCER_ONLY,
    'main: --output-root is required outside an attempt child': _PRODUCER_ONLY,
}


_LAUNCHER_SOURCE_FILE = str(
    (REPOSITORY / "benchmarks" / "run_single_stage_projected_route_gpu_root.py").resolve()
)


@pytest.mark.parametrize(
    "site, forge, match",
    _CHECK_KILLS,
    ids=[case[0].split(": ")[1][:48] for case in _CHECK_KILLS],
)
def test_each_claim_bearing_check_owns_a_forgery_refused_at_that_exact_line(
    tmp_path: Path,
    site: str,
    forge: object,
    match: str,
) -> None:
    """One forgery per CHECK, and the refusal has to come from that check.

    The validator kill table proves a FUNCTION is load-bearing.  It cannot see a
    refusal site INSIDE one, and seven of those were deleted one at a time with
    the whole suite green -- including the re-hash of the published terminal
    state, the receipt's only re-evaluatable artifact.  Asserting the refusing
    LINE is what makes each check individually non-deletable: remove it and the
    refusal either stops happening or arrives from somewhere else.
    """

    sites = _refusal_sites()
    assert site in sites, f"the launcher no longer raises: {site}"
    name = f"check_{abs(hash(site))}"
    root = tmp_path / name
    _, evidence = _mutated_root(tmp_path, name, lambda document: forge(root, document))

    with pytest.raises(
        (launcher.ProjectedRootError, rehearsal.RehearsalError)
    ) as caught:
        launcher.publish_root(root / "staging", root / "final", evidence)
    assert re.search(match, str(caught.value)), str(caught.value)
    assert not (root / "final").exists()

    frames = [
        frame
        for frame in traceback.extract_tb(caught.value.__traceback__)
        if frame.filename == _LAUNCHER_SOURCE_FILE
    ]
    assert frames, "the refusal did not come out of the launcher"
    assert frames[-1].lineno == sites[site], (
        f"refused at line {frames[-1].lineno}, which is not the check this case "
        f"names (line {sites[site]})"
    )


def test_every_refusal_site_carries_a_disposition() -> None:
    """A check cannot be added without saying what the suite does about it.

    The structural half of the check-granularity fix, and the same shape as the
    ``UNSHAPED_LEAVES`` guard one level down: the census is required to be
    EXACTLY the launcher's own ``raise`` sites, so the next revision cannot ship
    a gate that no table sees.  It is also the honest record of what is NOT
    covered -- 30 sites this round -- because a census that only listed the
    covered ones would be the same overclaim it exists to retire.
    """

    walked = frozenset(_refusal_sites())
    declared = frozenset(_REFUSAL_SITES)
    assert walked == declared, (
        f"undeclared refusal sites: {sorted(walked - declared)}; "
        f"declared sites the launcher no longer raises: {sorted(declared - walked)}"
    )
    assert all(reason.strip() for reason in _REFUSAL_SITES.values())

    # The three dispositions that make a checkable claim are checked.
    killed = frozenset(case[0] for case in _CHECK_KILLS)
    assert killed == frozenset(
        site for site, reason in _REFUSAL_SITES.items() if reason is _CHECK_KILLED
    )
    named_validators = frozenset(
        name
        for name, value in vars(launcher).items()
        if name.startswith("_validate_") and callable(value)
    )
    covered_by_table = frozenset(case[0] for case in _VALIDATOR_KILLS)
    for site, reason in _REFUSAL_SITES.items():
        owner = site.split(":")[0]
        if reason is _OWNER_KILLED:
            assert owner in named_validators and owner in covered_by_table, site
        if reason is _PRODUCER_ONLY:
            assert owner not in named_validators, site
            assert owner not in _REVALIDATION_CALL_GRAPH, site


def _revalidation_call_graph() -> frozenset[str]:
    """Every launcher function reachable from ``validate_root_artifact``.

    What makes the ``_PRODUCER_ONLY`` disposition a measurement rather than a
    claim: a refusal in a function no sealed receipt can drive is genuinely
    outside the re-validation surface, and this is how the census says so.
    """

    source = (
        REPOSITORY / "benchmarks" / "run_single_stage_projected_route_gpu_root.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    bodies = {
        node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
    }
    reached: set[str] = set()
    pending = ["validate_root_artifact"]
    while pending:
        name = pending.pop()
        if name in reached or name not in bodies:
            continue
        reached.add(name)
        for node in ast.walk(bodies[name]):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                pending.append(node.func.id)
    return frozenset(reached)


_REVALIDATION_CALL_GRAPH = _revalidation_call_graph()


def test_every_claim_bearing_leaf_is_declared_bound_to_something() -> None:
    """The terminal generalisation: an unbound claim-bearing leaf is unrepresentable.

    Six rounds of this campaign have the same shape -- the remediation binds the
    leaf the last round found and leaves its neighbour free, and the next round
    finds the neighbour.  The shape tree made an ABSENT SHAPE unrepresentable by
    deriving the required key sets from it; ``LEAF_BINDINGS`` does the same for
    an UNBOUND VALUE.  Every typed leaf of ``RECEIPT_SHAPES`` is declared as a
    frozen-literal comparison, a re-derivation, a digest recomputation, or
    unbound WITH ITS REASON -- and a claim-bearing leaf may not be the last.

    What this does not prove is that the named anchor is reached on every path;
    that is what the refusal-site census and its kill tests are for, and the two
    are meant to be read together.
    """

    def walk(shape: dict, prefix: str, found: list[str]) -> None:
        for name, nested in shape.items():
            path = f"{prefix}.{name}"
            if isinstance(nested, launcher._Leaf):
                found.append(path)
            elif isinstance(nested, launcher._Dispatched):
                continue
            elif isinstance(nested, tuple):
                walk(nested[0], f"{path}[]", found)
            else:
                walk(nested, path, found)

    found: list[str] = []
    for document, shape in launcher.RECEIPT_SHAPES.items():
        walk(shape, document, found)

    declared = frozenset(launcher.LEAF_BINDINGS)
    assert frozenset(found) == declared, (
        f"leaves with no binding declared: {sorted(frozenset(found) - declared)}; "
        f"declared leaves the tree no longer carries: "
        f"{sorted(declared - frozenset(found))}"
    )
    for path, (kind, anchor) in launcher.LEAF_BINDINGS.items():
        assert kind in launcher.BINDING_KINDS, path
        assert anchor.strip(), path
        if kind != launcher.BINDING_NONE:
            # An anchor that does not resolve is a census entry naming a
            # comparison nothing in this module can perform.
            assert hasattr(launcher, anchor), f"{path} names a missing anchor: {anchor}"

    assert launcher.CLAIM_BEARING_LEAVES <= declared
    unbound = sorted(
        path
        for path in launcher.CLAIM_BEARING_LEAVES
        if launcher.LEAF_BINDINGS[path][0] == launcher.BINDING_NONE
    )
    assert not unbound, f"claim-bearing leaves bound to nothing: {unbound}"


def test_a_self_published_timeout_cannot_mint_the_headline_verdict(
    tmp_path: Path,
) -> None:
    """E6-1/P6-1/A6-2: the timeout gate took both sides out of the receipt.

    The supervised-launch gate requires a record claiming a timeout to have
    waited "the timeout it publishes", and the timeout it publishes was compared
    to NOTHING -- the frozen ``ATTEMPT_TIMEOUT_SECONDS`` reached the validator
    through no path at all.  So a root carrying ``attempt_timeout_seconds:
    1e-9`` sealed ``CLAIM_DISCHARGED`` / ``PREREGISTERED`` beside a lane that
    "timed out" in half a second, erasing the pre-registered cold measurement
    while keeping the pre-registered label -- and the suite's own test of that
    branch pinned the honest timeout in its fixture, so it could only ever prove
    the weaker property.

    The fix is the shape that made the certified-route gate hold: the frozen
    literal is the anchor, and the one permitted departure is folded into the
    conformance label rather than refused, because ``--attempt-timeout-seconds``
    is a real knob and an operator who moves it is running a real experiment.
    """

    def a_timeout_of_its_own_choosing(evidence: dict) -> None:
        evidence["supervisor"]["attempt_timeout_seconds"] = 1.0e-9
        cold = evidence["cold_lane"]
        cold["outcome"] = "TIMEOUT"
        cold["timed_out"] = True
        cold["evidence"] = None
        cold["return_code"] = -9
        cold["supervised_seconds"] = 0.5
        evidence["cold_lane_anomaly"] = launcher.cold_lane_anomaly(cold)

    root, evidence = _mutated_root(tmp_path, "own_timeout", a_timeout_of_its_own_choosing)
    # The lane's own wall no longer clears the timeout it publishes, and the
    # verdict the attempts derive is no longer the headline one.
    _refuse_published(root, evidence, match="is not the one the attempts derive")

    # The same receipt, restated honestly: a moved timeout is a BOUNDED_SMOKE,
    # and it publishes.
    honest = tmp_path / "moved_timeout"
    staging = honest / "staging"
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
    moved = _root_evidence(
        verdict=launcher.VERDICT_QUALITY_ONLY, attempts=[attempt], cold_lane=cold
    )
    moved["supervisor"]["attempt_timeout_seconds"] = 900.0
    moved["attempt_protocol"]["conformance"] = launcher.CONFORMANCE_BOUNDED_SMOKE
    published = launcher.publish_root(staging, honest / "final", moved)
    sealed = launcher.validate_root_artifact(published)
    assert sealed["verdict"] == launcher.VERDICT_QUALITY_ONLY
    assert sealed["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_BOUNDED_SMOKE
    )


def test_a_count_a_budget_and_a_size_are_whole_numbers(tmp_path: Path) -> None:
    """A6-3: every reader of these leaves truncated, so the checks lost to their own reader.

    ``int(-0.5) == 0`` passed the check whose words are "which is not a count";
    ``int(2.9) == 2`` passed "which is not one the engine reports" and minted
    ``latched: true``; ``ir_bytes: 1.5`` satisfied the internal-sum identity
    under truncation; and ``maximum_iterations: 700.9`` sealed as
    ``CERTIFIED_BUDGET`` / ``PREREGISTERED``.  The deferral that left this open
    reasoned that a receipt claiming 700.9 certified iterations describes
    nothing physical -- true, and beside the point, because the truncation was
    what let it seal.

    Measured on the real producers, every one of these is a Python ``int`` by
    construction, so refusing the fractional form refuses nothing honest.
    """

    def fractional_stored_pairs(evidence: dict) -> None:
        _solve_of(evidence)["stored_pairs"] = -0.5

    def fractional_status(evidence: dict) -> None:
        _solve_of(evidence)["status"] = 2.9

    def fractional_kernel_size(evidence: dict) -> None:
        kernels = evidence["attempts"][0]["evidence"]["lowering_pre_gate"]["kernels"]
        kernels[0]["ir_bytes"] = 1.5

    def fractional_entry_count(evidence: dict) -> None:
        cache = evidence["attempts"][0]["evidence"]["compilation_cache"]
        cache["at_entry"]["entry_count"] = 0.5

    def fractional_child_pid(evidence: dict) -> None:
        evidence["attempts"][0]["gpu_memory"]["child_pid"] = 7.5

    for name, mutate in (
        ("stored_pairs", fractional_stored_pairs),
        ("status", fractional_status),
        ("ir_bytes", fractional_kernel_size),
        ("entry_count", fractional_entry_count),
        ("child_pid", fractional_child_pid),
    ):
        root, evidence = _mutated_root(tmp_path, f"fractional_{name}", mutate)
        _refuse_published(root, evidence, match="is not a whole number")


def test_the_latch_a_receipt_discharges_cannot_be_denied_by_its_own_rows(
    tmp_path: Path,
) -> None:
    """A6-1: the feasibility column was bound and the objective endpoint was not.

    A receipt whose recorded iterates never fall to the target sealed
    ``CLAIM_DISCHARGED`` beside ``terminal_objective`` at the target,
    ``latched: true`` and ``OBJECTIVE_TARGET_REACHED`` -- the latch, which is
    the whole content of section 1's claim, denied by the receipt's own
    arithmetic by seven decades.  Three tellings of the same number sat beside
    it, two of them EXACT copies in the producer, and none was compared.

    The closure is a chain, not a scalar: the summary's terminal objective is
    the agreement block's loop half, whose standalone half is the endpoint
    ledger's terminal ``weighted_total``, which the frozen-native pinned-term
    gate judges against the campaign's own literal on the attempt that
    discharges the claim.  Nothing in that chain is free.
    """

    def rows_that_never_reach_the_target(evidence: dict) -> None:
        solve = _solve_of(evidence)
        # A descending trajectory that stops seven decades above the target,
        # beside a terminal objective at it -- self-consistent in every field
        # the previous revision read.
        solve["rows"] = [
            {
                "index": index,
                "objective": objective,
                "candidate_objective": objective / 10.0,
                "feasibility_inf": 1.0e-14,
                "candidate_feasibility_inf": 1.0e-14,
            }
            for index, objective in enumerate((1000.0, 100.0, 10.0, 1.0))
        ]
        solve["iterations_run"] = 4

    root, evidence = _mutated_root(
        tmp_path, "rows_deny_the_latch", rows_that_never_reach_the_target
    )
    _refuse_published(
        root, evidence, match="neither endpoint of its last recorded iteration"
    )

    def a_terminal_objective_its_ledger_denies(evidence: dict) -> None:
        attempt = evidence["attempts"][0]["evidence"]
        attempt["solve"]["terminal_objective"] = 4.48e-30
        attempt["endpoint_agreement"]["loop_terminal_objective"] = 4.48e-30

    root, evidence = _mutated_root(
        tmp_path, "terminal_denies_its_ledger", a_terminal_objective_its_ledger_denies
    )
    _refuse_published(
        root, evidence, match="neither endpoint of its last recorded iteration"
    )


def test_a_truncated_options_block_refuses_by_name_rather_than_by_key_error(
    tmp_path: Path,
) -> None:
    """E6-2: the objective-target read ran EIGHT LINES before the field-set gate.

    A reader of sealed bytes needs a sentence naming the defect, and a truncated
    options block answered with a bare ``KeyError: 'objective_target'`` -- the
    class the nullable-leaf guard closed one block over, recurring inside the
    remediation that closed it.  The two checks are simply in the wrong order.
    """

    def a_budget_and_nothing_else(evidence: dict) -> None:
        attempt = evidence["attempts"][0]["evidence"]
        attempt["options"] = {"maximum_iterations": 700}
        attempt["certified_options_delta"] = {}

    root, evidence = _mutated_root(tmp_path, "truncated_options", a_budget_and_nothing_else)
    _refuse_published(
        root, evidence, match="options are not the certified configuration's fields"
    )


def _root_with_cold_lane(
    root: Path,
    *,
    outcome: str,
    ledger: dict | None,
    maximum_feasibility_inf: float | None,
) -> Path:
    """One healthy timed latch beside a cold lane the caller shapes."""

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
        maximum_feasibility_inf=maximum_feasibility_inf,
        ledger=ledger,
        outcome=outcome,
        index=0,
        relative_path=launcher.COLD_LANE_DIRECTORY,
        warm=False,
    )
    cold["timed_against_bar"] = False
    return launcher.publish_root(
        staging,
        root / "final",
        _root_evidence(
            verdict=launcher.VERDICT_CLAIM_DISCHARGED,
            attempts=[attempt],
            cold_lane=cold,
        ),
    )


def test_an_unlucky_cold_lane_draw_cannot_refuse_the_whole_publication(
    tmp_path: Path,
) -> None:
    """The lane decides nothing, INCLUDING whether the artifact exists.

    Ruling 13 took the lane out of the conformance label and out of
    ``derive_verdict`` and left it running the full discharging-attempt
    validation inside ``publish_root``.  Executed, a cold lane that latched and
    missed one pinned band -- and a cold lane that merely MISSED with one
    infeasible recorded iterate, the ordinary one-in-five outcome the
    pre-registered N exists to absorb -- refused the entire root after the lane
    and all three timed attempts had been spent: a refusal record, no artifact
    and no verdict at all.  That is worse than the ``QUALITY_ONLY`` cap ruling
    13 reversed ruling 8 to avoid.
    """

    missed_band = _synthetic_ledger(
        gated=True, **{"observable.iota": _scaled("observable.iota", 1.0 + 1.0e-3)}
    )
    assert not missed_band["pinned_term_gate"]["passed"]
    published = _root_with_cold_lane(
        tmp_path / "missed_band",
        outcome="LATCHED",
        ledger=missed_band,
        maximum_feasibility_inf=1.0e-14,
    )
    evidence = launcher.validate_root_artifact(published)
    assert evidence["verdict"] == launcher.VERDICT_CLAIM_DISCHARGED
    assert evidence["attempt_protocol"]["conformance"] == (
        launcher.CONFORMANCE_PREREGISTERED
    )
    assert not evidence["cold_lane"]["evidence"]["endpoint_ledger"][
        "pinned_term_gate"
    ]["passed"]

    published = _root_with_cold_lane(
        tmp_path / "infeasible_iterate",
        outcome="COMPLETED_WITHOUT_LATCH",
        ledger=None,
        maximum_feasibility_inf=1.0e-9,
    )
    evidence = launcher.validate_root_artifact(published)
    assert evidence["verdict"] == launcher.VERDICT_CLAIM_DISCHARGED
    assert evidence["cold_lane"]["evidence"]["solve"]["maximum_feasibility_inf"] == (
        1.0e-9
    )

    # What the lane still cannot do is lie about what it was.  Shape and
    # honesty are validated on both lanes; only the claim-bearing gates are not.
    def cold_ran_on_the_cpu(evidence: dict) -> None:
        evidence["cold_lane"]["evidence"]["runtime_identity"]["backend"] = "cpu"

    root, evidence = _mutated_root(tmp_path, "cold_cpu", cold_ran_on_the_cpu)
    _refuse_published(root, evidence, match="not the 'gpu' the wall is claimed on")

    def cold_hollowed_custody(evidence: dict) -> None:
        evidence["cold_lane"]["evidence"]["execution_sources"] = None

    root, evidence = _mutated_root(tmp_path, "cold_custody", cold_hollowed_custody)
    _refuse_published(root, evidence, match="execution_sources is not a document")


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
        # ``weighted_total`` is left as the run's own terminal objective: it is
        # the ledger term the endpoint agreement is bound to, so forging it too
        # would make this receipt refuse on the agreement rather than on the
        # reference, and the test would prove a narrower property than its name.
        endpoint = evidence["attempts"][0]["evidence"]["endpoint_agreement"]
        forged["weighted_total"] = endpoint["standalone_terminal_objective"]
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
        # Internally coherent, so that what refuses it is the reference and not
        # the arithmetic of its own published columns.
        ledger["relative_difference"] = rehearsal.endpoint_relative_differences(
            ledger["terminal"], ledger["native"]
        )
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
            _relaunched(first, 2),
            _relaunched(first, 3),
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
            _relaunched(miss, index) for index in (1, 2, 3)
        ] + [_relaunched(first, 4)]
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


def test_the_frozen_nested_shapes_are_the_ones_the_producers_write(
    off_tmpfs_path: Path,
) -> None:
    """Every nested shape is bound to the function that writes it, not a fixture.

    A frozen key set asserted against the suite's own helper is a twin, and
    twins drift -- which is exactly how the fixture publishing ``preflight: {}``
    came to certify that a discharged root needs no preflight at all, and how a
    fixture publishing ``{"bound_modules": []}`` came to certify that a
    discharged root need not say which bytes executed.

    Everything a CPU process can produce is bound BY EXECUTION here, including
    the four that a previous revision described as "the three that are only
    reachable with a device": ``gpu_runtime_identity`` runs on any backend,
    ``probe_writable_storage`` and ``configure_persistent_compilation_cache``
    need no device at all, and ``_gpu_memory_payload`` normalizes an
    unobserved child without one.  The single genuinely device-gated producer
    is the preflight, which queries the pinned GPU's inventory.
    """

    assert frozenset(launcher.CACHE_STATE_SHAPE) == frozenset(
        launcher.compilation_cache_state(REPOSITORY / "does-not-exist")
    )
    cache_directory = jax.config.jax_compilation_cache_dir
    minimum_entry = jax.config.jax_persistent_cache_min_entry_size_bytes
    minimum_compile = jax.config.jax_persistent_cache_min_compile_time_secs
    try:
        configuration = launcher.configure_persistent_compilation_cache(
            off_tmpfs_path / "cache"
        )
    finally:
        jax.config.update("jax_compilation_cache_dir", cache_directory)
        jax.config.update(
            "jax_persistent_cache_min_entry_size_bytes", minimum_entry
        )
        jax.config.update(
            "jax_persistent_cache_min_compile_time_secs", minimum_compile
        )
    assert frozenset(launcher.CACHE_CONFIGURATION_SHAPE) == frozenset(configuration)
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
            attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
    assert frozenset(launcher.RUNTIME_IDENTITY_SHAPE) == frozenset(
        launcher.gpu_runtime_identity()
    )
    assert frozenset(launcher.GPU_MEMORY_SHAPE) == frozenset(
        launcher._gpu_memory_payload(
            None, gpu_uuid=launcher.GPU_UUID, provider_pid=1, argv=("python",)
        )
    )
    assert frozenset(launcher.STORAGE_PROBE_SHAPE) == frozenset(
        launcher.probe_writable_storage(off_tmpfs_path, role="temporary")
    )
    execution_sources = rehearsal.bind_execution_sources(REPOSITORY)
    assert frozenset(launcher.EXECUTION_SOURCES_SHAPE) == frozenset(execution_sources)
    assert frozenset(launcher.EXECUTION_SOURCE_MANIFEST_SHAPE) == frozenset(
        execution_sources["manifest"]
    )
    assert frozenset(launcher.BOUND_MODULE_SHAPE) == frozenset(
        execution_sources["bound_modules"][0]
    )
    assert frozenset(launcher.INTERPRETER_INSTALLATION_SHAPE) == frozenset(
        execution_sources["interpreter_installation_modules"]
    )
    assert frozenset(launcher.PROBLEM_IDENTITY_SHAPE) == frozenset(
        rehearsal.problem_identity_evidence(
            dict(rehearsal.CPU_BOOTSTRAP_OBSERVABLES),
            problem_sha256="0" * 64,
            bootstrap_sha256="1" * 64,
        )
    )
    # The lowered-kernel record, from the producer's own payload function.  The
    # OUTER pre-gate record needs a bootstrapped case and two lowerings of the
    # real objective, which is a 26 s job this file has no other reason to pay,
    # so it is bound by execution in the rehearsal suite -- against the record a
    # REAL rehearsal published -- by
    # ``test_the_lowering_pre_gate_record_is_the_one_its_producer_writes``.
    assert frozenset(launcher.LOWERED_KERNEL_SHAPE) == frozenset(
        rehearsal.lowering_payload(
            KernelLowering(name="evaluate", ir_bytes=1, while_operations=0)
        )
    )
    # The one producer that genuinely needs the pinned device: it queries the
    # GPU inventory.  It stays bound by the shape the bounded GPU smoke
    # publishes through, which is what makes the smoke a producer test rather
    # than a liveness check.
    assert frozenset(launcher.PREFLIGHT_SHAPE) == frozenset(_preflight())


def _unshaped_leaves_the_walker_finds() -> dict[str, str]:
    """Every place a shape admits a mapping or a list without an inner shape."""

    found: dict[str, str] = {}

    def walk(shape: dict, prefix: str) -> None:
        for name, nested in sorted(shape.items()):
            path = f"{prefix}.{name}"
            if isinstance(nested, launcher._Dispatched):
                found[path] = f"dispatched: {nested.owner}"
            elif isinstance(nested, launcher._Leaf):
                if nested in (launcher._LIST, launcher._MAPPING):
                    found[path] = "unshaped leaf"
            elif isinstance(nested, tuple):
                walk(nested[0], f"{path}[]")
            else:
                walk(nested, path)

    for root, shape in launcher.RECEIPT_SHAPES.items():
        walk(shape, root)
    return found


def test_the_shape_tree_covers_its_own_required_key_sets() -> None:
    """No required key without a shape, and no unshaped block without a reason.

    THIS is the test that had to exist.  Four consecutive review rounds found
    the same defect one level lower each time -- the top-level names frozen and
    the blocks unchecked, then the blocks frozen and one required key left with
    no entry in the parallel map of shapes, so ``execution_sources: null``
    sealed as ``CLAIM_DISCHARGED`` while the plan said it could not.  The suite
    could not see it, because a test that enumerates the shapes which EXIST is
    structurally blind to a shape that is ABSENT.

    So the property asserted here is COVERAGE, not enumeration: every required
    key set is exactly the key set of its shape (they are derived from it, and
    this refuses a future revision that reintroduces a second listing), no leaf
    of any shape is an untyped free pass, and the complete list of places where
    a mapping or list is admitted without an inner shape is declared in the
    module WITH ITS REASON and is exactly what a walk of the trees finds.
    """

    assert launcher.ROOT_EVIDENCE_REQUIRED_KEYS == frozenset(
        launcher.ROOT_EVIDENCE_SHAPE
    )
    assert launcher.ATTEMPT_PROTOCOL_REQUIRED_KEYS == frozenset(
        launcher.ATTEMPT_PROTOCOL_SHAPE
    )
    assert launcher.SUPERVISED_ATTEMPT_REQUIRED_KEYS == frozenset(
        launcher.SUPERVISED_ATTEMPT_SHAPE
    )
    assert launcher.ATTEMPT_EVIDENCE_REQUIRED_KEYS == frozenset(
        launcher.ATTEMPT_EVIDENCE_SHAPE
    )
    assert launcher.REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS == frozenset(
        launcher.REFUSED_ATTEMPT_EVIDENCE_SHAPE
    )
    # Every node of every tree is one of exactly four things, and none of them
    # is "unchecked".
    def every_node(shape: dict, prefix: str) -> None:
        for name, nested in shape.items():
            path = f"{prefix}.{name}"
            assert isinstance(
                nested, (launcher._Leaf, launcher._Dispatched, dict, tuple)
            ), path
            if isinstance(nested, tuple):
                every_node(nested[0], f"{path}[]")
            elif isinstance(nested, dict):
                every_node(nested, path)

    for root, shape in launcher.RECEIPT_SHAPES.items():
        every_node(shape, root)
    assert _unshaped_leaves_the_walker_finds().keys() == (
        launcher.UNSHAPED_LEAVES.keys()
    )
    assert all(reason.strip() for reason in launcher.UNSHAPED_LEAVES.values())
    # Both custody blocks the previous revision reached with nothing at all are
    # now in the tree with a shape of their own.
    assert launcher.ATTEMPT_EVIDENCE_SHAPE["execution_sources"] == (
        launcher.EXECUTION_SOURCES_SHAPE
    )
    assert launcher.ATTEMPT_EVIDENCE_SHAPE["problem_identity"] == (
        launcher.PROBLEM_IDENTITY_SHAPE
    )
    assert launcher.ATTEMPT_EVIDENCE_SHAPE["lowering_pre_gate"] == (
        launcher.LOWERING_PRE_GATE_SHAPE
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
    with pytest.raises(
        launcher.ProjectedRootError, match="attempt_protocol is incomplete"
    ):
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
        attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
        # ``weighted_total`` is the run's own terminal objective, because the
        # producer evaluates the ledger's terminal side and the agreement's
        # standalone half at one state in one process.
        ledger=_synthetic_ledger(gated=True, weighted_total=4.48e-8),
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
    # The distance column moves with the terms, so what is stale is the GATE.
    gated["relative_difference"] = rehearsal.endpoint_relative_differences(
        gated["terminal"], gated["native"]
    )
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
        attempt_timeout_seconds=launcher.ATTEMPT_TIMEOUT_SECONDS,
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
