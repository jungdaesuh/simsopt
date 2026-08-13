"""GPU attempt-protocol launcher for the projected route's certified claim.

The claim under certification is stated in
``docs/single_stage_jax_gpu_projected_route_certification_plan.md``: on the
audited full single-stage VMEC-free examples workload, this repository's JAX GPU
route reaches the native C++ reference's endpoint objective, at strictly better
feasibility, in less wall time than native spent -- compile included.

Three properties of that claim decide the shape of this module.

**The latch is a draw, not a deterministic outcome.**  The A100 replication
settled it: 4 of 5 arms across two boxes reached the target and one terminated
in line-search collapse at a carried projector's tangency.  A single-trajectory
one-shot root can therefore be burned by a draw that indicts nothing, so the
protocol is pre-registered before the root opens: N = 3 sequential attempts of
the frozen configuration, stopping at the first that latches, with EVERY
attempt's telemetry published whether it latched or not, and exactly four named
verdicts (``CLAIM_DISCHARGED``, ``NO_LATCH_IN_PROTOCOL``, ``QUALITY_ONLY``,
``GATE_REFUSED:<gate>``).  Roots one through four of the predecessor route all
died in stages whose semantics had never been written down; there is no
undefined outcome here.

**Compile is inside the claim.**  Native's 287.30 s bar excluded nothing, so a
gate that excludes compile is not a comparison.  Each attempt is therefore its
own PROCESS: a second attempt run in this process would inherit the first's
``jax.jit`` caches and report a compile of milliseconds, and the claim is
discharged by the FIRST LATCHING attempt's wall -- which is not necessarily the
first attempt.  The cold lane runs first against an empty persistent cache, so
its compile is the honest cold measurement AND it primes the cache the timed
attempts load from; the cache is an accounting device published with its entry
count and digest, not a hiding place.

**Identity is bound by observables, never by the problem sha.**  Two runs of the
same commit on the same GPU produced different exact-numeric problem shas.  The
bootstrap observables agree to ~1e-14 relative across every backend the route
has run on, so they carry the identity and the sha carries only provenance.

Everything this module shares with the bounded CPU rehearsal is IMPORTED from
it -- the frozen configuration, the identity gate, the lowering pre-gate, the
endpoint ledger and its pinned-term bands, the sealing and publication
primitives.  A second spelling of any of them is a twin, and twins drift.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, Self

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import simsoptpp
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsRun,
    ProjectedLbfgsStatus,
    run_projected_lbfgs,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

from benchmarks.process_gpu_monitor import process_gpu_memory_artifact
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    CERTIFIED_MAXIMUM_ITERATIONS,
    CERTIFIED_ROUTE_OPTIONS,
    INFORMATIONAL_ENDPOINT_OBSERVABLES,
    NATIVE_TARGET_OBJECTIVE,
    NATIVE_WALL_SECONDS_BAR,
    PINNED_ENDPOINT_QUALITY_TERMS,
    TERMINAL_COORDINATES_FILENAME,
    BoundCase,
    artifact_manifest_payload,
    bind_execution_sources,
    bind_problem_identity,
    build_endpoint_ledger,
    certify_endpoint_agreement,
    gate_endpoint_ledger,
    iteration_payload,
    json_scalar,
    measure_lowering_pre_gate,
    rehearsal_options,
    rename_noreplace,
    seal_and_sync,
    sha256_hex,
    validate_environment,
    validate_sealed_modes,
)
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    REHEARSAL_ROUTE as PROJECTED_ROUTE,
)
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    REPOSITORY_ROOT as REPOSITORY,
)

# The snapshot's source-role bindings are the campaign's, asked of the campaign.
# Restating which paths carry which role is exactly the twin-constant drift the
# mistake book records twice (P153), so the private enumeration is imported
# rather than re-spelled: plan section 12.4 requires the GPU root artifact to
# publish through the machinery the campaign already drives.
from benchmarks.run_single_stage_native_equivalent_quality_campaign import (
    SOURCE_SNAPSHOT_DIRECTORY,
    _enumerated_source_roots,
)
from benchmarks.single_stage_fullspace_process_gpu_monitor import (
    BoundProcessGpuMemoryMonitor,
)
from benchmarks.single_stage_fullspace_snapshot import (
    JsonValue,
    SnapshotValidationError,
    canonical_json_bytes,
    capture_worktree_identity,
    load_canonical_json_bytes,
    publish_immutable_snapshot,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
    certify_agreement,
)
from benchmarks.single_stage_native_equivalent_quality_receipt import GPU_UUID

GPU_ROOT_SCHEMA_VERSION: Final = "single-stage-projected-route-gpu-root-v1"
GPU_ROOT_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-projected-route-gpu-root-manifest-v1"
)
GPU_ATTEMPT_SCHEMA_VERSION: Final = "single-stage-projected-route-gpu-attempt-v1"

EVIDENCE_FILENAME: Final = "root-evidence.json"
MANIFEST_FILENAME: Final = "artifact-manifest.json"
ATTEMPTS_DIRECTORY: Final = "attempts"
COLD_LANE_DIRECTORY: Final = "cold-lane"

# Frozen by plan section 12.2 and never widened by an artifact: the observed
# latch rate of 4/5 across two boxes puts three consecutive misses near 1%, and
# a larger N costs GPU hours linearly.
PREREGISTERED_ATTEMPTS: Final = 3

# The four verdicts of plan section 4.  ``GATE_REFUSED`` is always published
# with the gate that refused appended, so a defect report names its defect.
VERDICT_CLAIM_DISCHARGED: Final = "CLAIM_DISCHARGED"
VERDICT_NO_LATCH: Final = "NO_LATCH_IN_PROTOCOL"
VERDICT_QUALITY_ONLY: Final = "QUALITY_ONLY"
VERDICT_GATE_REFUSED_PREFIX: Final = "GATE_REFUSED:"

GPU_REQUIRED_ENVIRONMENT: Final = {
    "JAX_PLATFORMS": "cuda",
    "JAX_ENABLE_X64": "true",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
REQUIRED_BACKEND: Final = "gpu"

# Both knobs are load-bearing.  The defaults skip cache entries that are small
# or that compiled quickly, which is most of this route's bundle, so a cache
# configured with either one alone stays cold across processes.  Measured on
# .venv-qn-gpu across separate processes: 0.031 s warm against 0.095 s cold.
PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES: Final = -1
PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS: Final = 0.0
COMPILATION_CACHE_ENVIRONMENT_VARIABLE: Final = "JAX_COMPILATION_CACHE_DIR"

ATTEMPT_TIMEOUT_SECONDS: Final = 3600.0
_FAILURE_TAIL_BYTES: Final = 8192


class ProjectedRootError(RuntimeError):
    """A launcher gate refused; the partial root is kept as the evidence."""


class GateRefusal(RuntimeError):
    """A named gate refused, so the artifact can publish which one."""

    def __init__(self, gate: str, cause: BaseException) -> None:
        super().__init__(f"{gate}: {cause}")
        self.gate = gate
        self.cause = cause


def verdict_of_gate(gate: str) -> str:
    """The published verdict for a refused gate."""

    return f"{VERDICT_GATE_REFUSED_PREFIX}{gate}"


# --------------------------------------------------------------- attempt child


def configure_persistent_compilation_cache(directory: Path) -> dict[str, JsonValue]:
    """Point this process at the persistent cache and record what it did.

    Called before any tracing so that the compile the certified wall contains is
    a cache load.  The directory itself arrives through the environment, which
    is what makes the priming process and the timed process demonstrably share
    one cache rather than two that happen to agree.
    """

    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update(
        "jax_persistent_cache_min_entry_size_bytes",
        PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
    )
    jax.config.update(
        "jax_persistent_cache_min_compile_time_secs",
        PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    )
    return {
        "directory": str(directory),
        "enabled": bool(jax.config.jax_enable_compilation_cache),
        "min_entry_size_bytes": PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES,
        "min_compile_time_seconds": PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS,
    }


def compilation_cache_state(directory: Path) -> dict[str, JsonValue]:
    """Entry count, total bytes and an aggregate digest of one cache tree.

    The digest covers names and sizes rather than contents: a reader needs to
    tell a warm run from a cold one and to see the cache grow, and hashing
    hundreds of megabytes of XLA blobs to say so would be theatre.
    """

    rows: list[list[JsonValue]] = []
    total = 0
    if directory.is_dir():
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            size = path.stat(follow_symlinks=False).st_size
            rows.append([path.relative_to(directory).as_posix(), size])
            total += size
    return {
        "entry_count": len(rows),
        "total_bytes": total,
        "entries_digest": sha256_hex(canonical_json_bytes(rows)),
    }


def gpu_runtime_identity() -> dict[str, JsonValue]:
    """Name the silicon and the toolchain the claim is being stated for.

    The speed result is RTX 5090 specific (plan section 1.2), so the artifact
    records the device it was taken on rather than leaving a reader to assume.
    """

    devices = jax.devices()
    device = devices[0]
    return {
        "backend": jax.default_backend(),
        "device_count": len(devices),
        "device_kind": device.device_kind,
        "device_platform": device.platform,
        "jax_version": jax.__version__,
        "jaxlib_version": jaxlib.__version__,
        "native_extension_path": str(Path(simsoptpp.__file__).resolve(strict=True)),
        "process_id": os.getpid(),
    }


def bind_gpu_backend() -> dict[str, JsonValue]:
    """Refuse a launch that silently resolved to a backend other than the GPU.

    JAX resolves its platform lazily, so a missing ``JAX_PLATFORMS`` does not
    fail: it runs the whole chain on the CPU and reports a wall against a GPU
    bar.  The environment gate catches the variable and this catches the
    resolution, because they are not the same fact.
    """

    identity = gpu_runtime_identity()
    if identity["backend"] != REQUIRED_BACKEND:
        raise ProjectedRootError(
            f"resolved backend is {identity['backend']!r}, not {REQUIRED_BACKEND!r}"
        )
    return identity


class _gate:
    """Re-raise whatever a phase raised under the gate name it belongs to.

    The verdict vocabulary names the refusing gate, so the chain cannot lose
    which phase refused between the raise and the receipt.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind: object, value: BaseException | None, traceback: object) -> bool:
        if value is None or isinstance(value, GateRefusal):
            return False
        raise GateRefusal(self.name, value) from value


def _solve_payload(run: ProjectedLbfgsRun) -> dict[str, JsonValue]:
    """Every host-side scalar the solve produced, in the rehearsal's shape."""

    objectives = [record.objective for record in run.iterations]
    feasibilities = [record.feasibility_inf for record in run.iterations]
    return {
        "status": int(run.status),
        "status_name": ProjectedLbfgsStatus(int(run.status)).name,
        "latched": bool(
            int(run.status) == int(ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED)
        ),
        "iterations_run": len(run.iterations),
        "terminal_objective": run.objective,
        "terminal_feasibility_inf": run.feasibility_inf,
        "terminal_projected_gradient_inf": run.projected_gradient_inf,
        "stored_pairs": run.stored_pairs,
        "projector_materializations": run.projector_materializations,
        "tangency_forced_refreshes": run.tangency_forced_refreshes,
        "line_search_forced_refreshes": run.line_search_forced_refreshes,
        "monotone_descent": all(
            later <= earlier
            for earlier, later in zip(objectives, objectives[1:], strict=False)
        ),
        "maximum_feasibility_inf": json_scalar(
            max(feasibilities, default=float("nan"))
        ),
        "rows": [iteration_payload(record) for record in run.iterations],
    }


def run_attempt(
    attempt_root: Path,
    *,
    attempt_index: int,
    iterations: int,
    cache_directory: Path,
    environment: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Execute one attempt's whole chain and return its canonical evidence.

    The phases run in the order plan section 6 fixes, each failing closed under
    the gate name the verdict will carry.  The endpoint ledger is GATED only at
    the certified budget: a bounded attempt sits orders of magnitude from the
    endpoint, so a gate there would fail on every term and prove nothing.
    """

    started = time.perf_counter()
    with _gate("environment"):
        environment_evidence = validate_environment(
            environment, required=GPU_REQUIRED_ENVIRONMENT
        )
        declared = environment.get(COMPILATION_CACHE_ENVIRONMENT_VARIABLE)
        if declared != str(cache_directory):
            raise ProjectedRootError(
                f"{COMPILATION_CACHE_ENVIRONMENT_VARIABLE} must equal "
                f"{str(cache_directory)!r}, observed {declared!r}"
            )
        environment_evidence[COMPILATION_CACHE_ENVIRONMENT_VARIABLE] = declared
        cache_configuration = configure_persistent_compilation_cache(cache_directory)
        # Sampled HERE, before anything in this process has traced or compiled.
        # A later sample is not the fact the reader needs: the identity gate
        # pays the point-evaluation compile, so a cold run measured after it
        # would report a populated cache and call itself warm.
        cache_at_entry = compilation_cache_state(cache_directory)
        runtime_identity = bind_gpu_backend()

    with _gate("execution_sources"):
        execution_sources = bind_execution_sources(REPOSITORY)

    with _gate("bootstrap"):
        bootstrap_started = time.perf_counter()
        case = BoundCase()
        bootstrap_seconds = time.perf_counter() - bootstrap_started

    with _gate("problem_identity"):
        identity_started = time.perf_counter()
        problem_identity = bind_problem_identity(case)
        identity_seconds = time.perf_counter() - identity_started

    with _gate("lowering_pre_gate"):
        lowering_started = time.perf_counter()
        lowering = measure_lowering_pre_gate(case, iterations)
        lowering_seconds = time.perf_counter() - lowering_started

    cache_before_engine = compilation_cache_state(cache_directory)
    with _gate("solve"):
        options = rehearsal_options(iterations)
        run = run_projected_lbfgs(case.raw_joint, case.start, options=options)
    cache_after = compilation_cache_state(cache_directory)

    with _gate("feasibility"):
        # Plan section 1: feasibility holds at EVERY recorded iterate, on the
        # RAW equalities, absolutely -- never relatively, because both sides sit
        # decades below the tolerance the route enforces.
        worst = max(
            (record.feasibility_inf for record in run.iterations),
            default=run.feasibility_inf,
        )
        # Negated rather than written as ``> tolerance``: every comparison
        # against a NaN is false, so the ``>`` form lets a nonfinite iterate
        # through the gate whose whole job is to refuse it.
        if not worst <= CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance:
            raise ProjectedRootError(
                f"iterate feasibility {worst!r} is not within "
                f"{CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance!r}"
            )

    with _gate("endpoint_agreement"):
        endpoint = certify_endpoint_agreement(case, run)

    with _gate("endpoint_ledger"):
        gated = iterations == CERTIFIED_MAXIMUM_ITERATIONS
        ledger = build_endpoint_ledger(case, run, gated=gated)
        if gated and not ledger["pinned_term_gate"]["passed"]:
            raise ProjectedRootError(
                "pinned endpoint terms differ from native: "
                f"{ledger['pinned_term_gate']['failed_terms']}"
            )

    with _gate("attempt_publication"):
        coordinates = np.asarray(jax.device_get(run.coordinates), dtype=np.float64)
        with (attempt_root / TERMINAL_COORDINATES_FILENAME).open("wb") as stream:
            np.save(stream, coordinates, allow_pickle=False)

    return {
        "schema_version": GPU_ATTEMPT_SCHEMA_VERSION,
        "route": PROJECTED_ROUTE,
        "attempt_index": attempt_index,
        "environment": environment_evidence,
        "runtime_identity": runtime_identity,
        "execution_sources": execution_sources,
        "problem_identity": problem_identity,
        "lowering_pre_gate": lowering,
        "options": {
            field: json_scalar(getattr(options, field))
            for field in sorted(options.__dataclass_fields__)
        },
        "certified_options_delta": {
            field: json_scalar(getattr(options, field))
            for field in sorted(options.__dataclass_fields__)
            if getattr(options, field) != getattr(CERTIFIED_ROUTE_OPTIONS, field)
        },
        "compilation_cache": {
            "configuration": cache_configuration,
            "at_entry": cache_at_entry,
            "before_engine": cache_before_engine,
            "after": cache_after,
            "warm": bool(cache_at_entry["entry_count"] > 0),
        },
        "solve": _solve_payload(run),
        "endpoint_agreement": endpoint,
        "endpoint_ledger": ledger,
        "timing_seconds": {
            "bootstrap": bootstrap_seconds,
            "problem_identity": identity_seconds,
            "lowering_pre_gate": lowering_seconds,
            "engine_compile": run.compile_seconds,
            "engine_solve": run.solve_seconds,
            "engine_wall": run.compile_seconds + run.solve_seconds,
            "attempt_wall": time.perf_counter() - started,
        },
        # The certified wall is the engine's compile plus its solve.  Building
        # the problem and binding it are setup native's 287.30 s does not
        # contain either; naming the boundary is what makes the comparison
        # checkable instead of assumed.
        "timing_boundary": "engine_compile_plus_solve",
        "quality_claim": (
            "CERTIFIED_BUDGET"
            if iterations == CERTIFIED_MAXIMUM_ITERATIONS
            else "NOT_CLAIMED_AT_BOUNDED_BUDGET"
        ),
        "gate_refused": None,
    }


def run_attempt_child(
    attempt_root: Path,
    *,
    attempt_index: int,
    iterations: int,
    cache_directory: Path,
) -> int:
    """Run one attempt as a child process and print its canonical evidence.

    A refused gate is EVIDENCE, not an unhandled exception: the protocol names
    four verdicts and one of them is a defect report, so the child publishes
    which gate refused and exits nonzero rather than leaving the supervisor to
    infer a cause from a traceback.
    """

    try:
        payload = run_attempt(
            attempt_root,
            attempt_index=attempt_index,
            iterations=iterations,
            cache_directory=cache_directory,
            environment=os.environ,
        )
        status = 0
    except GateRefusal as refusal:
        payload = {
            "schema_version": GPU_ATTEMPT_SCHEMA_VERSION,
            "route": PROJECTED_ROUTE,
            "attempt_index": attempt_index,
            "gate_refused": refusal.gate,
            "error": f"{type(refusal.cause).__name__}: {refusal.cause}",
        }
        status = 2
    # Written as bytes: the canonical encoding is newline-terminated already,
    # and it is the canonical BYTES the supervisor validates.
    sys.stdout.buffer.write(canonical_json_bytes(payload))
    sys.stdout.buffer.flush()
    return status


# ------------------------------------------------------------------ supervisor


def attempt_invocation(
    attempt_root: Path,
    *,
    attempt_index: int,
    iterations: int,
    cache_directory: Path,
    environment: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
    """The exact argv and environment one attempt child is launched with."""

    argv = (
        sys.executable,
        "-B",
        str(Path(__file__).resolve(strict=True)),
        "--attempt-child",
        "--attempt-root",
        str(attempt_root),
        "--attempt-index",
        str(attempt_index),
        "--iterations",
        str(iterations),
        "--cache-dir",
        str(cache_directory),
    )
    child_environment = {
        **environment,
        COMPILATION_CACHE_ENVIRONMENT_VARIABLE: str(cache_directory),
        "PYTHONPATH": os.pathsep.join((str(REPOSITORY / "src"), str(REPOSITORY))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    return argv, child_environment


def supervise_attempt(
    staging_root: Path,
    relative_path: str,
    *,
    attempt_index: int,
    iterations: int,
    cache_directory: Path,
    environment: Mapping[str, str],
    gpu_uuid: str,
    timeout_seconds: float,
) -> dict[str, JsonValue]:
    """Run one attempt in its own process and retain every outcome it has.

    The process boundary is the point: a second attempt inside this process
    would inherit the first's compiled executables and report a compile that
    never happened, and the claim is discharged by the first LATCHING attempt's
    wall, which is not necessarily the first attempt's.
    """

    attempt_root = staging_root / relative_path
    attempt_root.mkdir(parents=True)
    argv, child_environment = attempt_invocation(
        attempt_root,
        attempt_index=attempt_index,
        iterations=iterations,
        cache_directory=cache_directory,
        environment=environment,
    )
    started = time.perf_counter()
    child = subprocess.Popen(
        argv,
        cwd=REPOSITORY,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitor = BoundProcessGpuMemoryMonitor(
        gpu_uuid=gpu_uuid,
        provider_pid=child.pid,
        expected_argv=argv,
    )
    monitor.start()
    timed_out = False
    try:
        stdout, stderr = child.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        child.kill()
        stdout, stderr = child.communicate()
    supervised_seconds = time.perf_counter() - started
    evidence = _parse_attempt_stdout(stdout)
    outcome = _attempt_outcome(
        evidence, return_code=child.returncode, timed_out=timed_out
    )
    return {
        "attempt_index": attempt_index,
        "artifact_relative_path": relative_path,
        "outcome": outcome,
        "return_code": int(child.returncode),
        "timed_out": timed_out,
        "supervised_seconds": supervised_seconds,
        "argv_sha256": sha256_hex(canonical_json_bytes(list(argv))),
        "gpu_memory": _gpu_memory_payload(monitor),
        "stderr_tail": stderr[-_FAILURE_TAIL_BYTES:].decode("utf-8", "replace"),
        # Kept only when the stream did not carry the child's document, so a
        # PROTOCOL_FAILURE names what arrived instead of leaving a reader to
        # guess; a successful attempt's stdout IS the evidence beside it.
        "stdout_tail": (
            None
            if evidence is not None
            else stdout[-_FAILURE_TAIL_BYTES:].decode("utf-8", "replace")
        ),
        "evidence": evidence,
    }


def _gpu_memory_payload(monitor: BoundProcessGpuMemoryMonitor) -> dict[str, JsonValue]:
    """Serialize one PID-and-device-bound GPU-memory observation.

    Normalized through the monitor module's own union so that a child the
    sampler never caught is published as explicit unavailability rather than as
    an inferred zero -- the distinction the monitor was written to preserve.
    """

    artifact = process_gpu_memory_artifact(monitor.finish())
    return {
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "availability": artifact.availability,
        "unavailable_reason": artifact.unavailable_reason,
        "device_uuid": artifact.gpu_uuid,
        "parent_pid": os.getpid(),
        "child_pid": monitor.identity.pid,
        "child_start_time_ticks": monitor.identity.start_ticks,
        "child_argv_sha256": sha256_hex(
            canonical_json_bytes(list(monitor.identity.argv))
        ),
        "sample_count": len(artifact.samples),
        "peak_used_memory_mib": artifact.peak_used_memory_mib,
    }


def _parse_attempt_stdout(stdout: bytes) -> JsonValue:
    """Read the child's single canonical-JSON line, refusing anything else.

    The canonical encoding is newline-terminated, and splitting the stream
    removes that terminator, so it is restored before validation: the check is
    on the child's canonical bytes, not on a re-encoding of them.

    Bytes that are not that document yield ``None``, which ``_attempt_outcome``
    names ``PROTOCOL_FAILURE`` -- the same outcome an empty stream already
    produced.  This is the protocol's closed outcome space, not a rescue: a
    child that printed anything after its payload used to raise here, and the
    raise propagated out of the supervisor, spending the one-shot root with no
    artifact published at all.  The refused bytes are published as
    ``stdout_tail`` so the defect report names what the child actually wrote.
    """

    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        return load_canonical_json_bytes(lines[-1] + b"\n")
    except SnapshotValidationError:
        return None


def _attempt_outcome(
    evidence: JsonValue, *, return_code: int, timed_out: bool
) -> str:
    """Classify one attempt without inventing a fifth protocol outcome."""

    if timed_out:
        return "TIMEOUT"
    if not isinstance(evidence, dict):
        return "PROTOCOL_FAILURE"
    if evidence.get("gate_refused") is not None:
        return "GATE_REFUSED"
    if return_code != 0:
        return "PROTOCOL_FAILURE"
    solve = evidence.get("solve")
    if not isinstance(solve, dict):
        return "PROTOCOL_FAILURE"
    return "LATCHED" if solve["latched"] else "COMPLETED_WITHOUT_LATCH"


def attempt_engine_wall_seconds(attempt: Mapping[str, JsonValue]) -> float:
    """The certified wall of one attempt: engine compile plus engine solve."""

    evidence = attempt["evidence"]
    return float(evidence["timing_seconds"]["engine_wall"])


def derive_verdict(
    attempts: Sequence[Mapping[str, JsonValue]],
    *,
    wall_seconds_bar: float,
) -> str:
    """Derive the protocol's verdict from the attempts alone.

    Kept a pure function of the published attempts so that re-validation can
    recompute the verdict from the sealed bytes instead of believing the field
    the run wrote.  The outcome space is closed: there is no fifth answer.
    """

    if not attempts:
        return verdict_of_gate("attempt_protocol")
    for attempt in attempts:
        outcome = attempt["outcome"]
        if outcome == "LATCHED":
            wall = attempt_engine_wall_seconds(attempt)
            return (
                VERDICT_CLAIM_DISCHARGED
                if wall < wall_seconds_bar
                else VERDICT_QUALITY_ONLY
            )
        if outcome == "GATE_REFUSED":
            return verdict_of_gate(str(attempt["evidence"]["gate_refused"]))
        if outcome != "COMPLETED_WITHOUT_LATCH":
            return verdict_of_gate(f"attempt_process:{outcome}")
    return VERDICT_NO_LATCH


def supervisor_payload(
    runtime_identity: Mapping[str, JsonValue],
    *,
    gpu_uuid: str,
    timeout_seconds: float,
) -> dict[str, JsonValue]:
    """Describe the supervising process, including what it does NOT claim.

    The supervisor resolves the GPU backend in its own process to refuse a
    silent CPU fallback, so it holds a CUDA context and is not GPU-zero.  It
    builds no problem and compiles no kernel -- every measured second belongs
    to a child -- but an artifact that claimed a GPU-free supervisor would be
    describing a different process than the one that ran.
    """

    return {
        "runtime_identity": dict(runtime_identity),
        "gpu_uuid": gpu_uuid,
        "attempt_timeout_seconds": timeout_seconds,
        "gpu_zero_asserted": False,
    }


def publish_source_snapshot(staging_root: Path) -> dict[str, JsonValue]:
    """Seal the executing source into the artifact, per plan section 12.4.

    A one-shot root's evidence has to stay readable after the tree has moved
    on, and an import-hash list points at bytes that no longer exist.  The
    worktree identity is captured on both sides of the copy so a tree that
    changed underneath the publication cannot pass for one that did not.
    """

    worktree = capture_worktree_identity(REPOSITORY)
    publication = publish_immutable_snapshot(
        staging_root / SOURCE_SNAPSHOT_DIRECTORY,
        _enumerated_source_roots(
            REPOSITORY, Path(simsoptpp.__file__).resolve(strict=True)
        ),
        worktree=worktree,
    )
    if capture_worktree_identity(REPOSITORY) != worktree:
        raise ProjectedRootError("source changed during snapshot publication")
    return {
        "relative_path": SOURCE_SNAPSHOT_DIRECTORY,
        "manifest_sha256": publication.manifest_sha256,
        "entry_count": len(publication.entries),
        "worktree": publication.worktree.to_payload(),
    }


def build_root_evidence(
    *,
    attempts: Sequence[Mapping[str, JsonValue]],
    cold_lane: Mapping[str, JsonValue] | None,
    snapshot: Mapping[str, JsonValue],
    supervisor: Mapping[str, JsonValue],
    authorized_attempts: int,
    iterations: int,
    cache: Mapping[str, JsonValue],
    verdict: str,
    chain_seconds: float,
) -> dict[str, JsonValue]:
    """Assemble the root receipt, telemetry of every attempt included."""

    latched = [attempt for attempt in attempts if attempt["outcome"] == "LATCHED"]
    preregistered = (
        authorized_attempts == PREREGISTERED_ATTEMPTS
        and iterations == CERTIFIED_MAXIMUM_ITERATIONS
    )
    return {
        "schema_version": GPU_ROOT_SCHEMA_VERSION,
        "route": PROJECTED_ROUTE,
        "verdict": verdict,
        "claim": {
            "target_objective": NATIVE_TARGET_OBJECTIVE,
            "wall_seconds_bar": NATIVE_WALL_SECONDS_BAR,
            "feasibility_tolerance": CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance,
        },
        "attempt_protocol": {
            "preregistered_attempts": PREREGISTERED_ATTEMPTS,
            "authorized_attempts": authorized_attempts,
            "attempts_run": len(attempts),
            "stop_rule": "stop at the first OBJECTIVE_TARGET_REACHED",
            "latch_count": len(latched),
            "latch_rate": f"{len(latched)}/{len(attempts)}",
            # A bounded run is not a root and must not read as one.  The
            # successor-root rule of plan section 12.1 applies to a spent
            # PREREGISTERED protocol and to nothing else.
            "conformance": "PREREGISTERED" if preregistered else "BOUNDED_SMOKE",
            "maximum_iterations": iterations,
            "certified_maximum_iterations": CERTIFIED_MAXIMUM_ITERATIONS,
        },
        "attempts": [dict(attempt) for attempt in attempts],
        "cold_lane": None if cold_lane is None else dict(cold_lane),
        "compilation_cache": dict(cache),
        "source_snapshot": dict(snapshot),
        "supervisor": dict(supervisor),
        "quality_claim": (
            "CERTIFIED_BUDGET"
            if iterations == CERTIFIED_MAXIMUM_ITERATIONS
            else "NOT_CLAIMED_AT_BOUNDED_BUDGET"
        ),
        "timing_boundary": "engine_compile_plus_solve",
        "timing_seconds": {"chain_wall": chain_seconds},
    }


def publish_root(
    staging_root: Path, output_root: Path, evidence: Mapping[str, JsonValue]
) -> Path:
    """Write the receipt, seal the tree, and publish it without replacing."""

    (staging_root / EVIDENCE_FILENAME).write_bytes(canonical_json_bytes(evidence))
    (staging_root / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(
            artifact_manifest_payload(
                staging_root, schema_version=GPU_ROOT_MANIFEST_SCHEMA_VERSION
            )
        )
    )
    seal_and_sync(staging_root)
    rename_noreplace(staging_root, output_root)
    descriptor = os.open(output_root.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return output_root


def validate_root_artifact(artifact_root: Path) -> dict[str, JsonValue]:
    """Re-derive every claim the sealed root artifact makes about itself.

    Run against the bytes this process just published and runnable later by
    anyone.  The verdict is RECOMPUTED from the attempts rather than read, the
    endpoint agreements are re-certified against the campaign's frozen band
    after the recorded band is checked against it, and each published terminal
    state is re-hashed.  State hashes compare exactly -- they are same-source
    copies -- while cross-executable numbers are toleranced, because demanding
    bitwise equality between two independently compiled executables is what
    refused the predecessor route's fourth root after a complete solve.
    """

    manifest = load_canonical_json_bytes((artifact_root / MANIFEST_FILENAME).read_bytes())
    if manifest != artifact_manifest_payload(
        artifact_root, schema_version=GPU_ROOT_MANIFEST_SCHEMA_VERSION
    ):
        raise ProjectedRootError("root manifest differs from the artifact tree")
    validate_sealed_modes(artifact_root)

    evidence = load_canonical_json_bytes((artifact_root / EVIDENCE_FILENAME).read_bytes())
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != GPU_ROOT_SCHEMA_VERSION
        or evidence.get("route") != PROJECTED_ROUTE
    ):
        raise ProjectedRootError("root evidence schema differs")
    if evidence["claim"]["wall_seconds_bar"] != NATIVE_WALL_SECONDS_BAR or (
        evidence["claim"]["target_objective"] != NATIVE_TARGET_OBJECTIVE
    ):
        raise ProjectedRootError("root evidence restates the native reference")
    if evidence["timing_boundary"] != "engine_compile_plus_solve":
        raise ProjectedRootError("root evidence states a different timing boundary")

    attempts = evidence["attempts"]
    recomputed = derive_verdict(attempts, wall_seconds_bar=NATIVE_WALL_SECONDS_BAR)
    if recomputed != evidence["verdict"]:
        raise ProjectedRootError(
            f"published verdict {evidence['verdict']!r} is not the one the "
            f"attempts derive ({recomputed!r})"
        )
    for attempt in attempts:
        _validate_attempt(artifact_root, attempt)
    cold = evidence["cold_lane"]
    if cold is not None:
        if cold["timed_against_bar"]:
            raise ProjectedRootError("the cold lane may not be timed against the bar")
        # The cold lane is what makes the cache an accounting device rather
        # than a hiding place, and it can only say what compile costs without a
        # cache if it ran without one.
        if isinstance(cold["evidence"], dict) and (
            cold["evidence"]["gate_refused"] is None
            and cold["evidence"]["compilation_cache"]["warm"]
        ):
            raise ProjectedRootError("the cold lane ran against a populated cache")
        _validate_attempt(artifact_root, cold)
    return evidence


def _validate_attempt(artifact_root: Path, attempt: Mapping[str, JsonValue]) -> None:
    """Re-derive one attempt's endpoint agreement and published state hash."""

    if attempt["outcome"] in {"TIMEOUT", "PROTOCOL_FAILURE"}:
        return
    evidence = attempt["evidence"]
    if not isinstance(evidence, dict):
        raise ProjectedRootError("attempt carries no evidence document")
    if evidence["gate_refused"] is not None:
        return
    if evidence["problem_identity"]["sha_is_binding"]:
        raise ProjectedRootError("attempt binds identity to an unstable sha")
    if not evidence["problem_identity"]["bound"]:
        raise ProjectedRootError("attempt claims an unbound problem")
    if not evidence["lowering_pre_gate"]["budget_independent"]:
        raise ProjectedRootError("attempt claims budget-dependent lowering")

    # The claim's quality quantity is a NUMBER, not a status code.  ``LATCHED``
    # is the optimizer reporting that some iterate fell to its own configured
    # ``objective_target``, so an attempt configured against a different target
    # would publish a latch that discharges a different claim.  Both halves are
    # checked here: the target the run used, and the objective it reached.
    if evidence["options"]["objective_target"] != NATIVE_TARGET_OBJECTIVE:
        raise ProjectedRootError(
            "attempt targets an objective other than the native endpoint"
        )
    if (
        attempt["outcome"] == "LATCHED"
        and evidence["solve"]["terminal_objective"] > NATIVE_TARGET_OBJECTIVE
    ):
        raise ProjectedRootError(
            "attempt published a latch above the native endpoint objective"
        )

    # An artifact may not narrow what quality parity is measured on: the pinned
    # set and the informational set are the campaign's, not the run's, and a
    # gated ledger's verdicts are recomputed from the published terms.
    ledger = evidence["endpoint_ledger"]
    if ledger["pinned_quality_terms"] != list(PINNED_ENDPOINT_QUALITY_TERMS) or (
        ledger["informational_observables"] != list(INFORMATIONAL_ENDPOINT_OBSERVABLES)
    ):
        raise ProjectedRootError("attempt endpoint ledger scope differs from the campaign's")
    if ledger["gated_at_this_budget"] and (
        gate_endpoint_ledger(ledger) != ledger["pinned_term_gate"]
    ):
        raise ProjectedRootError("attempt pinned-term gate is not the one its ledger derives")

    endpoint = evidence["endpoint_agreement"]
    if (
        endpoint["relative_tolerance"] != DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE
        or endpoint["absolute_floor"] != DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR
    ):
        raise ProjectedRootError("attempt endpoint tolerances differ from the campaign's")
    certify_agreement(
        endpoint["standalone_terminal_objective"],
        endpoint["loop_terminal_objective"],
        "published projected route attempt terminal objective",
        relative_tolerance=DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
        absolute_floor=DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
    )
    # ``json_scalar`` writes null for a nonfinite feasibility, so an attempt
    # whose worst iterate was NaN publishes null here rather than a number
    # under the bound.  Both readings are refused by the same check.
    worst_feasibility = evidence["solve"]["maximum_feasibility_inf"]
    if not isinstance(worst_feasibility, float) or not (
        worst_feasibility <= CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
    ):
        raise ProjectedRootError("attempt published an infeasible iterate")

    coordinates_path = (
        artifact_root
        / str(attempt["artifact_relative_path"])
        / TERMINAL_COORDINATES_FILENAME
    )
    with coordinates_path.open("rb") as stream:
        coordinates = np.load(stream, allow_pickle=False)
    republished = exact_numeric_tree_sha256(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    if republished != endpoint["terminal_state_sha256"]:
        raise ProjectedRootError("published terminal state differs from its hash")


def run_attempt_protocol(
    output_root: Path,
    *,
    cache_directory: Path,
    attempts_authorized: int = PREREGISTERED_ATTEMPTS,
    iterations: int = CERTIFIED_MAXIMUM_ITERATIONS,
    gpu_uuid: str = GPU_UUID,
    cold_lane: bool = True,
    timeout_seconds: float = ATTEMPT_TIMEOUT_SECONDS,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Run the pre-registered protocol once and publish its sealed artifact.

    The output root is claimed before a second of compute is spent, the cold
    lane runs first against an empty cache so that its compile is the honest
    cold number and the cache it leaves behind is what the timed attempts load,
    and the artifact is published and re-validated whatever the verdict.
    """

    chain_started = time.perf_counter()
    resolved_environment = os.environ if environment is None else environment
    if attempts_authorized < 1:
        raise ProjectedRootError("attempt protocol must authorize at least one attempt")
    if iterations < 1:
        raise ProjectedRootError("attempt budget must be positive")
    if output_root.exists():
        raise ProjectedRootError(f"root output already exists: {output_root}")
    if cache_directory.exists() and any(cache_directory.iterdir()):
        raise ProjectedRootError(
            f"compilation cache directory must start empty: {cache_directory}"
        )
    cache_directory.mkdir(parents=True, exist_ok=True)

    staging_root = (
        output_root.parent / f"{output_root.name}.partial-{os.urandom(8).hex()}"
    )
    os.mkdir(staging_root, 0o700)

    supervisor = supervisor_payload(
        bind_gpu_backend(), gpu_uuid=gpu_uuid, timeout_seconds=timeout_seconds
    )
    snapshot = publish_source_snapshot(staging_root)

    cold: dict[str, JsonValue] | None = None
    if cold_lane:
        cold = supervise_attempt(
            staging_root,
            COLD_LANE_DIRECTORY,
            attempt_index=0,
            iterations=iterations,
            cache_directory=cache_directory,
            environment=resolved_environment,
            gpu_uuid=gpu_uuid,
            timeout_seconds=timeout_seconds,
        )
        # Documented, never timed against the bar: its compile is cold by
        # construction and it is the process that primes the cache.
        cold["timed_against_bar"] = False

    attempts: list[dict[str, JsonValue]] = []
    for index in range(1, attempts_authorized + 1):
        attempt = supervise_attempt(
            staging_root,
            f"{ATTEMPTS_DIRECTORY}/attempt-{index}",
            attempt_index=index,
            iterations=iterations,
            cache_directory=cache_directory,
            environment=resolved_environment,
            gpu_uuid=gpu_uuid,
            timeout_seconds=timeout_seconds,
        )
        attempts.append(attempt)
        if attempt["outcome"] != "COMPLETED_WITHOUT_LATCH":
            break

    verdict = derive_verdict(attempts, wall_seconds_bar=NATIVE_WALL_SECONDS_BAR)
    evidence = build_root_evidence(
        attempts=attempts,
        cold_lane=cold,
        snapshot=snapshot,
        supervisor=supervisor,
        authorized_attempts=attempts_authorized,
        iterations=iterations,
        cache=compilation_cache_state(cache_directory),
        verdict=verdict,
        chain_seconds=time.perf_counter() - chain_started,
    )
    published = publish_root(staging_root, output_root, evidence)
    validate_root_artifact(published)
    return published


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--attempts", type=int, default=PREREGISTERED_ATTEMPTS)
    parser.add_argument("--iterations", type=int, default=CERTIFIED_MAXIMUM_ITERATIONS)
    parser.add_argument("--gpu-uuid", default=GPU_UUID)
    parser.add_argument("--no-cold-lane", action="store_true")
    parser.add_argument(
        "--attempt-timeout-seconds", type=float, default=ATTEMPT_TIMEOUT_SECONDS
    )
    parser.add_argument("--attempt-child", action="store_true")
    parser.add_argument("--attempt-root", type=Path)
    parser.add_argument("--attempt-index", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.attempt_child:
        return run_attempt_child(
            arguments.attempt_root.resolve(strict=True),
            attempt_index=arguments.attempt_index,
            iterations=arguments.iterations,
            cache_directory=arguments.cache_dir.resolve(strict=True),
        )
    if arguments.output_root is None:
        raise ProjectedRootError("--output-root is required outside an attempt child")
    published = run_attempt_protocol(
        arguments.output_root.resolve(),
        cache_directory=arguments.cache_dir.resolve(),
        attempts_authorized=arguments.attempts,
        iterations=arguments.iterations,
        gpu_uuid=arguments.gpu_uuid,
        cold_lane=not arguments.no_cold_lane,
        timeout_seconds=arguments.attempt_timeout_seconds,
    )
    evidence = load_canonical_json_bytes((published / EVIDENCE_FILENAME).read_bytes())
    sys.stdout.buffer.write(
        canonical_json_bytes(
            {
                "published_root": str(published),
                "verdict": evidence["verdict"],
                "attempt_protocol": evidence["attempt_protocol"],
                "attempt_walls": [
                    attempt["evidence"]["timing_seconds"]["engine_wall"]
                    if isinstance(attempt["evidence"], dict)
                    and attempt["evidence"].get("timing_seconds") is not None
                    else None
                    for attempt in evidence["attempts"]
                ],
            }
        )
    )
    sys.stdout.buffer.flush()
    return 0 if evidence["verdict"] == VERDICT_CLAIM_DISCHARGED else 1


__all__ = (
    "ATTEMPTS_DIRECTORY",
    "ATTEMPT_TIMEOUT_SECONDS",
    "COLD_LANE_DIRECTORY",
    "COMPILATION_CACHE_ENVIRONMENT_VARIABLE",
    "EVIDENCE_FILENAME",
    "GPU_ATTEMPT_SCHEMA_VERSION",
    "GPU_REQUIRED_ENVIRONMENT",
    "GPU_ROOT_MANIFEST_SCHEMA_VERSION",
    "GPU_ROOT_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS",
    "PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "PREREGISTERED_ATTEMPTS",
    "PROJECTED_ROUTE",
    "VERDICT_CLAIM_DISCHARGED",
    "VERDICT_GATE_REFUSED_PREFIX",
    "VERDICT_NO_LATCH",
    "VERDICT_QUALITY_ONLY",
    "GateRefusal",
    "ProjectedRootError",
    "attempt_engine_wall_seconds",
    "attempt_invocation",
    "bind_gpu_backend",
    "build_root_evidence",
    "compilation_cache_state",
    "configure_persistent_compilation_cache",
    "derive_verdict",
    "gpu_runtime_identity",
    "main",
    "publish_root",
    "publish_source_snapshot",
    "run_attempt",
    "run_attempt_child",
    "run_attempt_protocol",
    "supervise_attempt",
    "supervisor_payload",
    "validate_root_artifact",
    "verdict_of_gate",
)


if __name__ == "__main__":
    raise SystemExit(main())
