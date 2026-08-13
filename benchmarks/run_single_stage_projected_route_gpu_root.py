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
import shutil
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
    ProjectedLbfgsOptions,
    ProjectedLbfgsRun,
    ProjectedLbfgsStatus,
    run_projected_lbfgs,
)
from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256

from benchmarks.process_gpu_monitor import (
    SUPERVISOR_GPU_INVENTORY_QUERY,
    ProcessGpuMemoryMeasurement,
    ProcessGpuMemoryMonitorError,
    process_gpu_memory_artifact,
    sampler_failure_unavailable,
)
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    CERTIFIED_MAXIMUM_ITERATIONS,
    CERTIFIED_ROUTE_OPTIONS,
    INFORMATIONAL_ENDPOINT_OBSERVABLES,
    NATIVE_ENDPOINT_STATE_PATH,
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
    collapse_proximity_margin,
    endpoint_ledger_is_gated,
    gate_endpoint_ledger,
    iteration_payload,
    json_scalar,
    load_native_endpoint_state,
    measure_lowering_pre_gate,
    rehearsal_options,
    rename_noreplace,
    run_latched,
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
REFUSAL_FILENAME: Final = "root-validation-refusal.json"
REFUSAL_SCHEMA_VERSION: Final = "single-stage-projected-route-gpu-root-refusal-v1"
ATTEMPTS_DIRECTORY: Final = "attempts"
COLD_LANE_DIRECTORY: Final = "cold-lane"

# Plan section 12.2 pre-registers N and the certified budget together, and
# section 3 pre-registers the cold lane beside them.  A run that differs in any
# of the three is a bounded smoke, and the verdict is conditioned on that label.
CONFORMANCE_PREREGISTERED: Final = "PREREGISTERED"
CONFORMANCE_BOUNDED_SMOKE: Final = "BOUNDED_SMOKE"

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

# Plan section 11: a GPU launch runs with ``TMPDIR`` off tmpfs.  XLA spills PTX
# through the system temporary directory from C++, where the resolution rule is
# ``TMPDIR`` or ``/tmp`` -- it is NOT Python's rule, which probes the candidate
# and falls through to ``/var/tmp`` when the probe fails.  That asymmetry is the
# whole failure: on a quota-exhausted ``/tmp`` every Python path on the box kept
# working while the spill failed inside the bootstrap gate with
# ``RESOURCE_EXHAUSTED: ... Disk quota exceeded``, publishing
# ``GATE_REFUSED:bootstrap`` -- a defect report about the route, for an operator
# environment condition.  The directory resolved here is therefore XLA's, and it
# is what the children are launched with rather than whatever they inherit.
TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE: Final = "TMPDIR"
DEFAULT_TEMPORARY_DIRECTORY: Final = Path("/tmp")

# Filesystem types whose capacity is RAM and whose per-user limit is a quota no
# capacity API reports.  Refused for every directory this protocol writes to.
REFUSED_STORAGE_FILESYSTEM_TYPES: Final = frozenset({"tmpfs", "ramfs", "devtmpfs"})

STORAGE_PROBE_PREFIX: Final = ".projected-route-storage-probe-"

# The stop rule the attempt loop actually implements.  Published and re-derived
# from one owner: the loop breaks on the first outcome that is not
# ``COMPLETED_WITHOUT_LATCH``, which is a latch AND every refused, timed-out or
# unparseable draw, not "the first latch" as the receipt used to say.
ATTEMPT_STOP_RULE: Final = (
    "stop at the first attempt whose outcome is not COMPLETED_WITHOUT_LATCH"
)

# A cold lane that produced the measurement plan section 3 pre-registers.  A
# refused, timed-out or unparseable lane produced no cold compile number and
# primed no cache it can account for, which is the same state ``--no-cold-lane``
# leaves the protocol in.
COLD_LANE_MEASURED_OUTCOMES: Final = frozenset(
    {"LATCHED", "COMPLETED_WITHOUT_LATCH"}
)

# The receipt's own shape, frozen.  ``validate_root_artifact`` used to index
# into whatever fields it needed, so a receipt missing its source snapshot, its
# supervisor block, its preflight, its cache accounting and its telemetry
# re-validated clean and could not be told from a complete one.  These are the
# key sets ``build_root_evidence``, ``supervise_attempt`` and ``run_attempt``
# write; the bounded GPU smoke publishes through all three and re-validates, so
# a producer that drifts from them fails there rather than at the root.
ROOT_EVIDENCE_REQUIRED_KEYS: Final = frozenset({
    "attempt_protocol",
    "attempts",
    "claim",
    "cold_lane",
    "compilation_cache",
    "quality_claim",
    "route",
    "schema_version",
    "source_snapshot",
    "supervisor",
    "timing_boundary",
    "timing_seconds",
    "verdict",
})
ATTEMPT_PROTOCOL_REQUIRED_KEYS: Final = frozenset({
    "attempts_run",
    "authorized_attempts",
    "certified_maximum_iterations",
    "cold_lane_authorized",
    "conformance",
    "latch_count",
    "latch_rate",
    "maximum_iterations",
    "preregistered_attempts",
    "stop_rule",
})
SUPERVISED_ATTEMPT_REQUIRED_KEYS: Final = frozenset({
    "argv_sha256",
    "artifact_relative_path",
    "attempt_index",
    "evidence",
    "gpu_memory",
    "outcome",
    "return_code",
    "stderr_tail",
    "stdout_tail",
    "supervised_seconds",
    "timed_out",
})
ATTEMPT_EVIDENCE_REQUIRED_KEYS: Final = frozenset({
    "attempt_index",
    "certified_options_delta",
    "compilation_cache",
    "endpoint_agreement",
    "endpoint_ledger",
    "environment",
    "execution_sources",
    "gate_refused",
    "lowering_pre_gate",
    "options",
    "problem_identity",
    "quality_claim",
    "route",
    "runtime_identity",
    "schema_version",
    "solve",
    "timing_boundary",
    "timing_seconds",
})
REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS: Final = frozenset({
    "attempt_index",
    "error",
    "gate_refused",
    "route",
    "schema_version",
})


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


def _quality_claim(iterations: int) -> str:
    """Whether a run at this budget claims section 1.1's quality parity.

    One owner for the attempt's field and the root's, so re-validation can
    re-derive both instead of reading the label a run wrote about itself.
    """

    return (
        "CERTIFIED_BUDGET"
        if iterations == CERTIFIED_MAXIMUM_ITERATIONS
        else "NOT_CLAIMED_AT_BOUNDED_BUDGET"
    )


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
    The interpreter is recorded for the same reason: plan section 3 pins the
    warm-cache behaviour to ``.venv-qn-gpu``, so which environment ran is
    contract-relevant provenance and must not have to be inferred from a
    hashed argv.
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
        "python_executable": str(Path(sys.executable).resolve(strict=True)),
        "python_prefix": str(Path(sys.prefix).resolve(strict=True)),
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


def _solve_payload(
    run: ProjectedLbfgsRun, options: ProjectedLbfgsOptions
) -> dict[str, JsonValue]:
    """Every host-side scalar the solve produced, in the rehearsal's shape.

    Every float goes through ``json_scalar``.  ``canonical_json_bytes`` refuses
    the nonfinite values, and the child's encoding call sits outside its
    ``GateRefusal`` handler, so a raw scalar left unsanitized turns a completed
    -- possibly latching -- solve into an empty stdout and a
    ``PROTOCOL_FAILURE`` over an encoding detail.  A nonfinite scalar is
    published as null, which every downstream gate refuses as a number.
    """

    objectives = [record.objective for record in run.iterations]
    feasibilities = [record.feasibility_inf for record in run.iterations]
    return {
        "status": int(run.status),
        "status_name": ProjectedLbfgsStatus(int(run.status)).name,
        "latched": run_latched(run),
        "iterations_run": len(run.iterations),
        "terminal_objective": json_scalar(run.objective),
        "terminal_feasibility_inf": json_scalar(run.feasibility_inf),
        "terminal_projected_gradient_inf": json_scalar(run.projected_gradient_inf),
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
        "collapse_proximity_margin": json_scalar(
            collapse_proximity_margin(run, options)
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
    the gate name the verdict will carry.  The endpoint ledger is GATED only on
    the attempt that discharges the claim -- see ``endpoint_ledger_is_gated``.
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
        gated = endpoint_ledger_is_gated(
            iterations=iterations, latched=run_latched(run)
        )
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
        "solve": _solve_payload(run, options),
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
        "quality_claim": _quality_claim(iterations),
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
    temporary_directory: Path,
) -> tuple[tuple[str, ...], dict[str, str]]:
    """The exact argv and environment one attempt child is launched with.

    ``TMPDIR`` is SET rather than inherited.  The supervisor preflighted one
    directory; forwarding whatever the operator's shell happened to hold would
    leave the four children spilling somewhere else, and plan section 11's rule
    would be enforced against a directory nobody used.
    """

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
        TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE: str(temporary_directory),
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
    temporary_directory: Path,
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
        temporary_directory=temporary_directory,
    )
    started = time.perf_counter()
    child = subprocess.Popen(
        argv,
        cwd=REPOSITORY,
        env=child_environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    monitor = _start_gpu_memory_monitor(
        gpu_uuid=gpu_uuid, provider_pid=child.pid, expected_argv=argv
    )
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
        "gpu_memory": _gpu_memory_payload(
            monitor, gpu_uuid=gpu_uuid, provider_pid=child.pid, argv=argv
        ),
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


def _start_gpu_memory_monitor(
    *, gpu_uuid: str, provider_pid: int, expected_argv: Sequence[str]
) -> BoundProcessGpuMemoryMonitor | None:
    """Attach the PID-and-device-bound sampler, or report that it did not.

    ``None`` means this child runs unobserved: procfs did not yield an identity
    (a child that raced to zombie reads an empty ``cmdline``), the argv did not
    match, or the sampling thread refused to start.  Process GPU-memory
    sampling is telemetry -- plan section 6 does not list it among the gates --
    so it may not hold veto power over the publication of a one-shot root.
    """

    try:
        monitor = BoundProcessGpuMemoryMonitor(
            gpu_uuid=gpu_uuid,
            provider_pid=provider_pid,
            expected_argv=expected_argv,
        )
        monitor.start()
    except (OSError, RuntimeError, ValueError):
        return None
    return monitor


def _gpu_memory_payload(
    monitor: BoundProcessGpuMemoryMonitor | None,
    *,
    gpu_uuid: str,
    provider_pid: int,
    argv: Sequence[str],
) -> dict[str, JsonValue]:
    """Serialize one PID-and-device-bound GPU-memory observation.

    Normalized through the monitor module's own union so that a child the
    sampler never caught is published as explicit unavailability rather than as
    an inferred zero -- the distinction the monitor was written to preserve --
    and so that a sampler which FAILED is published the same way instead of
    raised.  ``ProcessGpuMemoryMonitor.finish`` re-raises whatever its polling
    thread stored: an ``nvidia-smi`` query that timed out among the ~10^4 this
    protocol performs, a row some unrelated process rendered ``[N/A]``, an
    exhausted sample cap.  Raised from here, any of those would propagate
    through the supervisor and spend the root -- attempts, verdict and all --
    with nothing sealed and nothing published, which is precisely the undefined
    outcome plan section 4 exists to eliminate.
    """

    identity = None if monitor is None else monitor.identity
    if monitor is None:
        measurement: ProcessGpuMemoryMeasurement = sampler_failure_unavailable(
            gpu_uuid=gpu_uuid, provider_pid=provider_pid
        )
    else:
        try:
            measurement = monitor.finish()
        except ProcessGpuMemoryMonitorError:
            measurement = sampler_failure_unavailable(
                gpu_uuid=gpu_uuid, provider_pid=provider_pid
            )
    artifact = process_gpu_memory_artifact(measurement)
    observed_argv = list(argv if identity is None else identity.argv)
    return {
        "monitor_scope": "whole-child-exact-pid-exact-device",
        "availability": artifact.availability,
        "unavailable_reason": artifact.unavailable_reason,
        "device_uuid": artifact.gpu_uuid,
        "parent_pid": os.getpid(),
        "child_pid": provider_pid,
        "child_start_time_ticks": None if identity is None else identity.start_ticks,
        "child_argv_sha256": sha256_hex(canonical_json_bytes(observed_argv)),
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
    """Classify one attempt without inventing a fifth protocol outcome.

    ``latched`` is required to BE a boolean rather than indexed for.  The
    document reaching here has passed canonical-JSON validity and nothing else,
    and this call sits inside the supervisor's unguarded window: a canonical
    document of a different shape as the child's last stdout line used to raise
    ``KeyError`` out of ``supervise_attempt``, discarding every completed
    attempt unpublished -- the hazard ``_parse_attempt_stdout`` repaired one
    level up.  An unexpected shape is the ``PROTOCOL_FAILURE`` the closed
    outcome space already has for it.
    """

    if timed_out:
        return "TIMEOUT"
    if not isinstance(evidence, dict):
        return "PROTOCOL_FAILURE"
    if evidence.get("gate_refused") is not None:
        return "GATE_REFUSED"
    if return_code != 0:
        return "PROTOCOL_FAILURE"
    solve = evidence.get("solve")
    if not isinstance(solve, dict) or not isinstance(solve.get("latched"), bool):
        return "PROTOCOL_FAILURE"
    return "LATCHED" if solve["latched"] else "COMPLETED_WITHOUT_LATCH"


def attempt_engine_wall_seconds(attempt: Mapping[str, JsonValue]) -> float:
    """The certified wall of one attempt: engine compile plus engine solve.

    DERIVED from its two halves rather than read back, and refused if the
    published sum disagrees.  Both are IEEE additions of the same two published
    doubles, so agreement is exact; a receipt whose ``engine_wall`` is not its
    own compile plus its own solve has restated the quantity the claim is
    judged on.
    """

    timing = attempt["evidence"]["timing_seconds"]
    if not isinstance(timing, dict):
        raise ProjectedRootError("attempt publishes no timing block to derive from")
    derived = float(timing["engine_compile"]) + float(timing["engine_solve"])
    if derived != float(timing["engine_wall"]):
        raise ProjectedRootError(
            f"attempt engine wall {timing['engine_wall']!r} is not its own "
            f"compile plus solve ({derived!r})"
        )
    return derived


def cold_lane_measured(cold_lane: Mapping[str, JsonValue] | None) -> bool:
    """Whether the cold lane produced the measurement section 3 pre-registers.

    Ruling 1 carried to the lane it was not written for.  The cold lane is a
    fourth full-budget draw at the certified budget, run FIRST and outside the
    attempt loop, and its outcome was never inspected: a lane that latched and
    failed the per-term quality gate published ``GATE_REFUSED:endpoint_ledger``,
    primed the cache the timed attempts were then measured against, and left
    ``conformance: PREREGISTERED`` untouched -- the strongest available
    counter-evidence to the quality claim, sealed into the same tree with no
    effect on anything.

    A lane that MISSED still measured a cold compile and still primed the cache,
    so a miss conforms; ruling 1's whole point is that a stochastic miss indicts
    nothing.  Every other outcome leaves the protocol in the state
    ``--no-cold-lane`` leaves it in -- no cold number, nothing to account the
    warm walls against -- and is labelled the same way, which caps the verdict
    at ``QUALITY_ONLY`` instead of breaking the loop.  The lane's telemetry
    publishes either way.
    """

    if cold_lane is None:
        return False
    return cold_lane["outcome"] in COLD_LANE_MEASURED_OUTCOMES


def attempt_protocol_conformance(
    *, authorized_attempts: int, iterations: int, cold_lane: bool
) -> str:
    """Whether a run IS the pre-registered protocol or a bounded smoke.

    The three facts plan sections 3 and 12.2 freeze together -- N = 3, the
    certified budget, and the cold lane the warm numbers are accounted against
    -- decide one label, in one place, read by the verdict and re-derived at
    re-validation.  A bounded smoke that read as a spent pre-registered protocol
    would drag in the successor-root rule of section 12.1, which applies to a
    root and to nothing else.

    ``cold_lane`` is the lane's own OUTCOME through ``cold_lane_measured``, not
    the ``--no-cold-lane`` flag.  The flag says what was asked for; conformance
    is a statement about what the protocol produced, and a lane that was
    authorized and then refused produced exactly what an unauthorized one does.
    """

    preregistered = (
        authorized_attempts == PREREGISTERED_ATTEMPTS
        and iterations == CERTIFIED_MAXIMUM_ITERATIONS
        and cold_lane
    )
    return CONFORMANCE_PREREGISTERED if preregistered else CONFORMANCE_BOUNDED_SMOKE


def derive_verdict(
    attempts: Sequence[Mapping[str, JsonValue]],
    *,
    wall_seconds_bar: float,
    conformance: str,
) -> str:
    """Derive the protocol's verdict from the attempts and the budget label.

    Kept a pure function of the published attempts so that re-validation can
    recompute the verdict from the sealed bytes instead of believing the field
    the run wrote.  The outcome space is closed: there is no fifth answer.

    ``CLAIM_DISCHARGED`` additionally requires ``CONFORMANCE_PREREGISTERED``.
    The claim is a claim about the pre-registered protocol at the certified
    budget, and it is at that budget alone that the per-term ledger gate --
    section 1.1's definition of quality parity -- can run.  A bounded run that
    latches under the bar is a true measurement and reads ``QUALITY_ONLY``; it
    is not the campaign's headline verdict, and the launcher will not mint one
    beside ``quality_claim: NOT_CLAIMED_AT_BOUNDED_BUDGET``.
    """

    if not attempts:
        return verdict_of_gate("attempt_protocol")
    for attempt in attempts:
        outcome = attempt["outcome"]
        if outcome == "LATCHED":
            wall = attempt_engine_wall_seconds(attempt)
            discharged = (
                wall < wall_seconds_bar and conformance == CONFORMANCE_PREREGISTERED
            )
            return VERDICT_CLAIM_DISCHARGED if discharged else VERDICT_QUALITY_ONLY
        if outcome == "GATE_REFUSED":
            return verdict_of_gate(str(attempt["evidence"]["gate_refused"]))
        if outcome != "COMPLETED_WITHOUT_LATCH":
            return verdict_of_gate(f"attempt_process:{outcome}")
    return VERDICT_NO_LATCH


def _unescape_mount_field(field: str) -> str:
    """Undo the four octal escapes ``/proc/self/mountinfo`` emits in a path."""

    for escape, character in (
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\040", " "),
        ("\\134", "\\"),
    ):
        field = field.replace(escape, character)
    return field


def filesystem_type(path: Path) -> str:
    """The type of the filesystem the mount carrying ``path`` provides.

    Read from ``/proc/self/mountinfo`` because the property that decides this
    protocol's fate -- "capacity is RAM and the limit is a per-uid quota" -- is
    exactly the one no capacity API reports.  Mounts nest, so the longest mount
    point that is an ancestor of the resolved path wins.
    """

    resolved = path.resolve()
    ancestry = [str(resolved), *(str(parent) for parent in resolved.parents)]
    depth_of = {name: len(ancestry) - index for index, name in enumerate(ancestry)}
    deepest = 0
    kind = ""
    with open("/proc/self/mountinfo", encoding="utf-8") as stream:
        for line in stream:
            fields = line.split()
            depth = depth_of.get(_unescape_mount_field(fields[4]))
            if depth is None or depth <= deepest:
                continue
            deepest = depth
            kind = fields[fields.index("-") + 1]
    if not kind:
        raise ProjectedRootError(f"no mount in this namespace carries {resolved}")
    return kind


def probe_writable_storage(directory: Path, *, role: str) -> dict[str, JsonValue]:
    """Refuse a directory this run writes to that cannot take a single byte.

    ``statvfs`` is not this check and cannot be.  Measured on the certifying box
    while the condition was live: ``/tmp`` reported 12.29 GiB available and
    571769 free inodes while a one-byte write returned ``EDQUOT`` and left a
    zero-length file behind -- a redirected write that returns exit 0 with an
    empty result.  A per-uid tmpfs quota is invisible to every capacity API and
    visible to exactly one thing, which is a write.  The capacity number is
    published beside the probe, labelled for what it is worth.

    The filesystem type is refused first, because an empty tmpfs passes a
    one-byte probe and then fills during the run.  Plan section 11 enumerates
    the failure CLASS, not the instance.
    """

    if not directory.is_dir():
        raise ProjectedRootError(f"{role} directory does not exist: {directory}")
    kind = filesystem_type(directory)
    if kind in REFUSED_STORAGE_FILESYSTEM_TYPES:
        raise ProjectedRootError(
            f"{role} directory {directory} is on {kind}; plan section 11 requires "
            f"every directory this root writes to off tmpfs "
            f"(set {TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE} and the paths "
            f"accordingly)"
        )
    probe = directory / f"{STORAGE_PROBE_PREFIX}{os.urandom(8).hex()}"
    try:
        descriptor = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as failure:
        raise ProjectedRootError(
            f"{role} directory {directory} refused a one-byte write: errno "
            f"{failure.errno} ({failure.strerror})"
        ) from failure
    finally:
        probe.unlink(missing_ok=True)
    capacity = os.statvfs(directory)
    return {
        "role": role,
        "directory": str(directory),
        "filesystem_type": kind,
        "device_id": os.stat(directory).st_dev,
        "one_byte_write": "ok",
        # Advisory only: this is the number that said 12.29 GiB free on the box
        # where the write above returned EDQUOT.
        "advisory_available_bytes": capacity.f_bavail * capacity.f_frsize,
    }


def resolve_temporary_directory(environment: Mapping[str, str]) -> Path:
    """The directory XLA will spill through, by XLA's resolution rule."""

    declared = environment.get(TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE, "")
    return Path(declared) if declared.strip() else DEFAULT_TEMPORARY_DIRECTORY


def preflight_external_resources(
    *,
    gpu_uuid: str,
    cache_directory: Path,
    output_root: Path,
    environment: Mapping[str, str],
) -> dict[str, JsonValue]:
    """Check every resource outside this process BEFORE the first child runs.

    Plan section 11: enumerate the failure class, not the instance.  Four
    resources this protocol depends on live outside the repository and outside
    the gates -- the NVIDIA tooling the telemetry uses, the device the claim
    names, the sealed native endpoint that is the reference side of the quality
    gate, and the temporary, cache and output storage every child writes
    through.  Each is checkable in milliseconds; each otherwise spends the root
    at the first child.  A missing data file already spent one root of the
    predecessor route, and a quota-exhausted ``/tmp`` published
    ``GATE_REFUSED:bootstrap`` for a bounded smoke on this box.
    """

    executable = shutil.which(SUPERVISOR_GPU_INVENTORY_QUERY[0])
    if executable is None:
        raise ProjectedRootError(
            f"{SUPERVISOR_GPU_INVENTORY_QUERY[0]} is not on PATH"
        )
    completed = subprocess.run(
        SUPERVISOR_GPU_INVENTORY_QUERY,
        capture_output=True,
        check=True,
        text=True,
        timeout=30.0,
    )
    visible = tuple(
        line.split(",")[0].strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    if gpu_uuid not in visible:
        raise ProjectedRootError(
            f"pinned GPU {gpu_uuid} is not among the visible devices {visible}"
        )
    native_state = load_native_endpoint_state()
    temporary_directory = resolve_temporary_directory(environment)
    storage = [
        probe_writable_storage(temporary_directory, role="temporary"),
        probe_writable_storage(cache_directory, role="compilation_cache"),
        probe_writable_storage(output_root.parent, role="output"),
    ]
    return {
        "gpu_inventory_executable": executable,
        "visible_gpu_uuids": list(visible),
        "native_endpoint_state_path": str(NATIVE_ENDPOINT_STATE_PATH),
        "native_endpoint_state_sha256": native_state.file_sha256,
        "native_endpoint_state_content_sha256": native_state.content_sha256,
        # Recorded, not only checked: a reader of the sealed bytes could not
        # previously tell a root that ran under a safe temporary directory from
        # one that did not.
        "temporary_directory": str(temporary_directory),
        "storage": storage,
    }


def supervisor_payload(
    runtime_identity: Mapping[str, JsonValue],
    *,
    gpu_uuid: str,
    timeout_seconds: float,
    preflight: Mapping[str, JsonValue],
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
        "preflight": dict(preflight),
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
    cold_lane_authorized: bool,
    cache: Mapping[str, JsonValue],
    verdict: str,
    chain_seconds: float,
) -> dict[str, JsonValue]:
    """Assemble the root receipt, telemetry of every attempt included."""

    latched = [attempt for attempt in attempts if attempt["outcome"] == "LATCHED"]
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
            "stop_rule": ATTEMPT_STOP_RULE,
            "latch_count": len(latched),
            # k over N, the denominator plan section 4 names -- the attempts
            # AUTHORIZED, not the attempts the stop rule got to.  The cold lane
            # is a fourth full-budget draw and is deliberately outside it: it is
            # not part of the protocol and can only make the rate conservative.
            "latch_rate": f"{len(latched)}/{authorized_attempts}",
            "cold_lane_authorized": cold_lane_authorized,
            "conformance": attempt_protocol_conformance(
                authorized_attempts=authorized_attempts,
                iterations=iterations,
                cold_lane=cold_lane_measured(cold_lane),
            ),
            "maximum_iterations": iterations,
            "certified_maximum_iterations": CERTIFIED_MAXIMUM_ITERATIONS,
        },
        "attempts": [dict(attempt) for attempt in attempts],
        "cold_lane": None if cold_lane is None else dict(cold_lane),
        "compilation_cache": dict(cache),
        "source_snapshot": dict(snapshot),
        "supervisor": dict(supervisor),
        "quality_claim": _quality_claim(iterations),
        "timing_boundary": "engine_compile_plus_solve",
        "timing_seconds": {"chain_wall": chain_seconds},
    }


def write_root_receipt(
    staging_root: Path, evidence: Mapping[str, JsonValue]
) -> None:
    """Write the receipt and, last, the manifest that describes the tree."""

    (staging_root / EVIDENCE_FILENAME).write_bytes(canonical_json_bytes(evidence))
    (staging_root / MANIFEST_FILENAME).write_bytes(
        canonical_json_bytes(
            artifact_manifest_payload(
                staging_root, schema_version=GPU_ROOT_MANIFEST_SCHEMA_VERSION
            )
        )
    )


def _fsync(path: Path) -> None:
    """Make one file or directory durable before the next step depends on it."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_root(
    staging_root: Path, output_root: Path, evidence: Mapping[str, JsonValue]
) -> Path:
    """Write the receipt, re-validate it, then seal and publish it.

    Plan section 6 step 10 GATES publication rather than annotating it.  The
    receipt is re-derived from the bytes on disk while the staging tree is still
    writable, so a refusal leaves an UNSEALED tree carrying the refusal it
    raised -- not a sealed, ``renameat2``-published, 0444 artifact whose
    ``verdict`` field the launcher's own validator rejected and which a later
    reader cannot tell apart from one it accepted.  Only the sealed modes are
    checked after the fact, because they are the one property that does not
    exist yet at the moment the receipt is judged.

    The receipt and the refusal record are made DURABLE where they are written.
    ``seal_and_sync`` is what fsyncs a published tree, and it is exactly the
    step a refusal never reaches, so the one machine-readable record of why a
    root refused used to live only in the page cache -- the same "the only
    record is on a stderr section 11 calls volatile" failure this ruling exists
    to eliminate, displaced one step later.
    """

    write_root_receipt(staging_root, evidence)
    _fsync(staging_root / EVIDENCE_FILENAME)
    _fsync(staging_root / MANIFEST_FILENAME)
    _fsync(staging_root)
    try:
        validate_root_artifact(staging_root, sealed=False)
    except BaseException as refusal:
        refusal_path = staging_root / REFUSAL_FILENAME
        refusal_path.write_bytes(
            canonical_json_bytes(
                {
                    "schema_version": REFUSAL_SCHEMA_VERSION,
                    "refused_at": "pre_seal_revalidation",
                    "published": False,
                    "error": f"{type(refusal).__name__}: {refusal}",
                }
            )
        )
        _fsync(refusal_path)
        _fsync(staging_root)
        raise
    seal_and_sync(staging_root)
    rename_noreplace(staging_root, output_root)
    _fsync(output_root.parent)
    validate_sealed_modes(output_root)
    return output_root


def validate_root_artifact(
    artifact_root: Path, *, sealed: bool = True
) -> dict[str, JsonValue]:
    """Re-derive every claim the root artifact makes about itself.

    Run against the bytes this process just wrote -- before they are sealed, so
    that a refusal can still gate the publication -- and runnable later by
    anyone against the sealed tree, which is what ``sealed`` selects: the modes
    are the one property that does not exist yet at the moment the receipt is
    judged.  The verdict is RECOMPUTED from the attempts and the re-derived
    conformance label rather than read, each attempt's outcome is re-derived
    from its own evidence and exit status, the endpoint agreements are
    re-certified against the campaign's frozen band after the recorded band is
    checked against it, and each published terminal state is re-hashed.  State
    hashes compare exactly -- they are same-source copies -- while
    cross-executable numbers are toleranced, because demanding bitwise equality
    between two independently compiled executables is what refused the
    predecessor route's fourth root after a complete solve.

    Three facts the campaign's own history says a ``CLAIM_DISCHARGED`` receipt
    must carry are re-derived here rather than read: that the per-term physics
    gate of section 1.1 HAD to run on the discharging attempt
    (``endpoint_ledger_is_gated``), that it PASSED, and that the attempts ran at
    the budget the conformance label claims.  Without them this function was a
    consistency check over the fields it happened to find, and a sealed
    ``CLAIM_DISCHARGED`` root whose physics gate never ran -- or ran and failed
    -- published and re-validated clean.  It also asserts the receipt is
    COMPLETE, because a truncated document could not be told from a whole one:
    every block section 6 builds has a frozen key set.
    """

    manifest = load_canonical_json_bytes((artifact_root / MANIFEST_FILENAME).read_bytes())
    if manifest != artifact_manifest_payload(
        artifact_root, schema_version=GPU_ROOT_MANIFEST_SCHEMA_VERSION
    ):
        raise ProjectedRootError("root manifest differs from the artifact tree")
    if sealed:
        validate_sealed_modes(artifact_root)

    evidence = load_canonical_json_bytes((artifact_root / EVIDENCE_FILENAME).read_bytes())
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema_version") != GPU_ROOT_SCHEMA_VERSION
        or evidence.get("route") != PROJECTED_ROUTE
    ):
        raise ProjectedRootError("root evidence schema differs")
    if frozenset(evidence) != ROOT_EVIDENCE_REQUIRED_KEYS:
        raise ProjectedRootError(
            "root evidence is not the complete receipt: missing "
            f"{sorted(ROOT_EVIDENCE_REQUIRED_KEYS - frozenset(evidence))}, "
            f"unexpected {sorted(frozenset(evidence) - ROOT_EVIDENCE_REQUIRED_KEYS)}"
        )
    claim = evidence["claim"]
    if (
        claim["wall_seconds_bar"] != NATIVE_WALL_SECONDS_BAR
        or claim["target_objective"] != NATIVE_TARGET_OBJECTIVE
        or claim["feasibility_tolerance"] != CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
    ):
        raise ProjectedRootError("root evidence restates the native reference")
    if evidence["timing_boundary"] != "engine_compile_plus_solve":
        raise ProjectedRootError("root evidence states a different timing boundary")
    # The device the claim names is section 1.2's, not the run's: the speed
    # result is RTX 5090 specific and ``--gpu-uuid`` accepts any visible device.
    if evidence["supervisor"]["gpu_uuid"] != GPU_UUID:
        raise ProjectedRootError(
            f"root names GPU {evidence['supervisor']['gpu_uuid']!r}, not the "
            f"device the claim is stated for ({GPU_UUID!r})"
        )

    protocol = evidence["attempt_protocol"]
    if frozenset(protocol) != ATTEMPT_PROTOCOL_REQUIRED_KEYS:
        raise ProjectedRootError("root attempt protocol block is incomplete")
    cold = evidence["cold_lane"]
    if bool(protocol["cold_lane_authorized"]) != (cold is not None):
        raise ProjectedRootError(
            "root cold-lane authorization does not match the lane it published"
        )
    attempts = evidence["attempts"]
    for attempt in attempts:
        _validate_attempt_shape(attempt, cold_lane=False)
    if cold is not None:
        _validate_attempt_shape(cold, cold_lane=True)
    for attempt in attempts:
        _validate_attempt_outcome(attempt)
    if cold is not None:
        _validate_attempt_outcome(cold)

    # Section 4's draw statistics were pure read-backs beside a conformance
    # label that was re-derived, so a one-attempt root could publish
    # ``latch_rate: 3/3`` and ``attempts_run: 7`` and re-validate clean.  Every
    # one of them is a function of the attempts and the frozen constants.
    conformance = attempt_protocol_conformance(
        authorized_attempts=int(protocol["authorized_attempts"]),
        iterations=int(protocol["maximum_iterations"]),
        cold_lane=cold_lane_measured(cold),
    )
    latch_count = len([a for a in attempts if a["outcome"] == "LATCHED"])
    derived_protocol: dict[str, JsonValue] = {
        "attempts_run": len(attempts),
        "certified_maximum_iterations": CERTIFIED_MAXIMUM_ITERATIONS,
        "conformance": conformance,
        "latch_count": latch_count,
        "latch_rate": f"{latch_count}/{int(protocol['authorized_attempts'])}",
        "preregistered_attempts": PREREGISTERED_ATTEMPTS,
        "stop_rule": ATTEMPT_STOP_RULE,
    }
    for name, derived in derived_protocol.items():
        if protocol[name] != derived:
            raise ProjectedRootError(
                f"published attempt protocol {name} {protocol[name]!r} is not "
                f"the one the attempts derive ({derived!r})"
            )
    quality_claim = _quality_claim(int(protocol["maximum_iterations"]))
    if evidence["quality_claim"] != quality_claim:
        raise ProjectedRootError(
            f"published quality claim {evidence['quality_claim']!r} is not the "
            f"one its budget derives ({quality_claim!r})"
        )

    recomputed = derive_verdict(
        attempts,
        wall_seconds_bar=NATIVE_WALL_SECONDS_BAR,
        conformance=conformance,
    )
    if recomputed != evidence["verdict"]:
        raise ProjectedRootError(
            f"published verdict {evidence['verdict']!r} is not the one the "
            f"attempts derive ({recomputed!r})"
        )
    for attempt in attempts:
        _validate_attempt(artifact_root, attempt, protocol)
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
        _validate_attempt(artifact_root, cold, protocol)
    return evidence


def _validate_attempt_shape(
    attempt: Mapping[str, JsonValue], *, cold_lane: bool
) -> None:
    """Refuse a supervised record or child document that is not the whole one.

    A producer that raised part-way through, a partial write, or a hand-built
    document cannot be told from a complete attempt by any check that indexes
    into the fields it happens to need.  The child publishes two shapes and only
    two -- a completed chain, or a named gate refusal -- so both are frozen.
    """

    required = SUPERVISED_ATTEMPT_REQUIRED_KEYS | (
        frozenset({"timed_against_bar"}) if cold_lane else frozenset()
    )
    if frozenset(attempt) != required:
        raise ProjectedRootError(
            "supervised attempt record is incomplete: missing "
            f"{sorted(required - frozenset(attempt))}, unexpected "
            f"{sorted(frozenset(attempt) - required)}"
        )
    evidence = attempt["evidence"]
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        raise ProjectedRootError("attempt evidence is not a document")
    expected = (
        REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS
        if evidence.get("gate_refused") is not None
        else ATTEMPT_EVIDENCE_REQUIRED_KEYS
    )
    if frozenset(evidence) != expected:
        raise ProjectedRootError(
            "attempt evidence document is incomplete: missing "
            f"{sorted(expected - frozenset(evidence))}, unexpected "
            f"{sorted(frozenset(evidence) - expected)}"
        )


def _validate_attempt_outcome(attempt: Mapping[str, JsonValue]) -> None:
    """Re-derive one attempt's outcome from its evidence and its exit status.

    The verdict is a function of these strings, so recomputing the verdict from
    them and calling that "recomputed" would stop one level above where it
    matters.  The three facts the classifier consumes are all published.
    """

    recomputed = _attempt_outcome(
        attempt["evidence"],
        return_code=int(attempt["return_code"]),
        timed_out=bool(attempt["timed_out"]),
    )
    if recomputed != attempt["outcome"]:
        raise ProjectedRootError(
            f"attempt outcome {attempt['outcome']!r} is not the one its "
            f"evidence derives ({recomputed!r})"
        )


def _validate_attempt(
    artifact_root: Path,
    attempt: Mapping[str, JsonValue],
    protocol: Mapping[str, JsonValue],
) -> None:
    """Re-derive one attempt's endpoint agreement and published state hash."""

    if attempt["outcome"] in {"TIMEOUT", "PROTOCOL_FAILURE"}:
        return
    evidence = attempt["evidence"]
    if not isinstance(evidence, dict):
        raise ProjectedRootError("attempt carries no evidence document")
    if evidence["gate_refused"] is not None:
        return
    # The wall of EVERY attempt, not only the first latching one: section 4
    # makes each attempt's wall part of the artifact, and ``derive_verdict``
    # reaches ``attempt_engine_wall_seconds`` on exactly one of them.
    attempt_engine_wall_seconds(attempt)
    # The budget the conformance label is derived from has to be the budget the
    # attempt RAN.  A receipt declaring ``maximum_iterations: 700`` beside an
    # attempt whose own options say 400 minted the campaign's headline verdict
    # for a bounded run -- the defect closed in the launcher and left open in
    # the validator the plan then made the gate on publication.
    if int(evidence["options"]["maximum_iterations"]) != int(
        protocol["maximum_iterations"]
    ):
        raise ProjectedRootError(
            f"attempt ran {evidence['options']['maximum_iterations']!r} iterations, "
            f"not the {protocol['maximum_iterations']!r} its protocol claims"
        )
    quality_claim = _quality_claim(int(evidence["options"]["maximum_iterations"]))
    if evidence["quality_claim"] != quality_claim:
        raise ProjectedRootError(
            f"attempt quality claim {evidence['quality_claim']!r} is not the one "
            f"its budget derives ({quality_claim!r})"
        )
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
    # Substitution soundness rests on "same route": every budget is the frozen
    # configuration with one field replaced.  The published delta is therefore
    # RE-DERIVED from the published options against the frozen object, not read.
    # The KEY SET is checked first, against the frozen dataclass: derived over
    # whatever fields an attempt happened to publish, a truncated options block
    # yields an empty delta and passes, and an unknown field reaches ``getattr``
    # as an ``AttributeError`` rather than a named refusal.
    if frozenset(evidence["options"]) != frozenset(
        CERTIFIED_ROUTE_OPTIONS.__dataclass_fields__
    ):
        raise ProjectedRootError(
            "attempt options are not the certified configuration's fields"
        )
    delta = {
        field: value
        for field, value in evidence["options"].items()
        if value != json_scalar(getattr(CERTIFIED_ROUTE_OPTIONS, field))
    }
    if delta != evidence["certified_options_delta"]:
        raise ProjectedRootError(
            f"attempt options delta {evidence['certified_options_delta']!r} is "
            f"not the one its options derive ({delta!r})"
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
    # ``gated_at_this_budget`` was the one decision field re-validation READ,
    # and it is the field that switches section 1.1's physics gate on.  Read, a
    # ``CLAIM_DISCHARGED`` root could carry an ungated ledger on its latching
    # attempt -- no per-term verdicts at all -- and nothing objected.  It is a
    # pure function of two published facts, asked of the same owner both lanes
    # ask.
    gated = endpoint_ledger_is_gated(
        iterations=int(protocol["maximum_iterations"]),
        latched=attempt["outcome"] == "LATCHED",
    )
    if gated != bool(ledger["gated_at_this_budget"]):
        raise ProjectedRootError(
            f"attempt ledger claims gated={ledger['gated_at_this_budget']!r}, "
            f"which is not what its budget and outcome derive ({gated!r})"
        )
    if gated:
        if gate_endpoint_ledger(ledger) != ledger["pinned_term_gate"]:
            raise ProjectedRootError(
                "attempt pinned-term gate is not the one its ledger derives"
            )
        # Equality of a faithfully recorded FAILURE with its own recomputation
        # is a consistency check, not a quality gate.  Section 1.1 defines
        # quality parity as this verdict passing, so a discharging endpoint that
        # failed it may not be sealed as one that did.
        if not ledger["pinned_term_gate"]["passed"]:
            raise ProjectedRootError(
                "attempt discharges the claim with a failed pinned-term gate: "
                f"{ledger['pinned_term_gate']['failed_terms']}"
            )

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

    The output namespace is checked and every external resource is preflighted
    before a second of compute is spent, the cold lane runs first against an
    empty cache so that its compile is the honest cold number and the cache it
    leaves behind is what the timed attempts load, and the artifact is published
    whatever the verdict -- after the re-validation that gates it.

    The final name is claimed only at the ``renameat2``: what this reserves up
    front is the fact that nothing occupies it yet, plus a private
    ``.partial-<hex>`` staging tree.  Plan section 11 constrains the root to one
    supervised session, which is what makes that sufficient.
    """

    chain_started = time.perf_counter()
    resolved_environment = os.environ if environment is None else environment
    if attempts_authorized < 1:
        raise ProjectedRootError("attempt protocol must authorize at least one attempt")
    if iterations < 1:
        raise ProjectedRootError("attempt budget must be positive")
    # Section 1.2 states the speed result for one device.  ``--gpu-uuid`` exists
    # so a successor root under a revised plan can name its own hardware; under
    # THIS plan the device is frozen, and refusing here costs nothing while
    # refusing at re-validation would cost the root.
    if gpu_uuid != GPU_UUID:
        raise ProjectedRootError(
            f"this plan certifies {GPU_UUID}, not {gpu_uuid}"
        )
    if output_root.exists():
        raise ProjectedRootError(f"root output already exists: {output_root}")
    if cache_directory.exists() and any(cache_directory.iterdir()):
        raise ProjectedRootError(
            f"compilation cache directory must start empty: {cache_directory}"
        )
    cache_directory.mkdir(parents=True, exist_ok=True)

    # Both BEFORE the staging tree exists, so a refusal leaves the filesystem
    # exactly as it found it and costs zero compute -- which is what makes
    # refusing a temporary directory on tmpfs the cheap half of the trade
    # against a ``GATE_REFUSED:bootstrap`` on a spent root.  The backend stays
    # first: a process that resolved to the CPU is not a launch of this protocol
    # at all, and it is the cheapest fact to establish.
    runtime_identity = bind_gpu_backend()
    preflight = preflight_external_resources(
        gpu_uuid=gpu_uuid,
        cache_directory=cache_directory,
        output_root=output_root,
        environment=resolved_environment,
    )
    temporary_directory = resolve_temporary_directory(resolved_environment)

    staging_root = (
        output_root.parent / f"{output_root.name}.partial-{os.urandom(8).hex()}"
    )
    os.mkdir(staging_root, 0o700)

    supervisor = supervisor_payload(
        runtime_identity,
        gpu_uuid=gpu_uuid,
        timeout_seconds=timeout_seconds,
        preflight=preflight,
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
            temporary_directory=temporary_directory,
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
            temporary_directory=temporary_directory,
            gpu_uuid=gpu_uuid,
            timeout_seconds=timeout_seconds,
        )
        attempts.append(attempt)
        if attempt["outcome"] != "COMPLETED_WITHOUT_LATCH":
            break

    verdict = derive_verdict(
        attempts,
        wall_seconds_bar=NATIVE_WALL_SECONDS_BAR,
        conformance=attempt_protocol_conformance(
            authorized_attempts=attempts_authorized,
            iterations=iterations,
            cold_lane=cold_lane_measured(cold),
        ),
    )
    evidence = build_root_evidence(
        attempts=attempts,
        cold_lane=cold,
        snapshot=snapshot,
        supervisor=supervisor,
        authorized_attempts=attempts_authorized,
        iterations=iterations,
        cold_lane_authorized=cold_lane,
        cache=compilation_cache_state(cache_directory),
        verdict=verdict,
        chain_seconds=time.perf_counter() - chain_started,
    )
    return publish_root(staging_root, output_root, evidence)


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
    "ATTEMPT_EVIDENCE_REQUIRED_KEYS",
    "ATTEMPT_PROTOCOL_REQUIRED_KEYS",
    "ATTEMPT_STOP_RULE",
    "ATTEMPT_TIMEOUT_SECONDS",
    "COLD_LANE_DIRECTORY",
    "COLD_LANE_MEASURED_OUTCOMES",
    "COMPILATION_CACHE_ENVIRONMENT_VARIABLE",
    "CONFORMANCE_BOUNDED_SMOKE",
    "CONFORMANCE_PREREGISTERED",
    "DEFAULT_TEMPORARY_DIRECTORY",
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
    "REFUSAL_FILENAME",
    "REFUSAL_SCHEMA_VERSION",
    "REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS",
    "REFUSED_STORAGE_FILESYSTEM_TYPES",
    "ROOT_EVIDENCE_REQUIRED_KEYS",
    "SUPERVISED_ATTEMPT_REQUIRED_KEYS",
    "TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE",
    "VERDICT_CLAIM_DISCHARGED",
    "VERDICT_GATE_REFUSED_PREFIX",
    "VERDICT_NO_LATCH",
    "VERDICT_QUALITY_ONLY",
    "GateRefusal",
    "ProjectedRootError",
    "attempt_engine_wall_seconds",
    "attempt_invocation",
    "attempt_protocol_conformance",
    "bind_gpu_backend",
    "build_root_evidence",
    "cold_lane_measured",
    "compilation_cache_state",
    "configure_persistent_compilation_cache",
    "derive_verdict",
    "filesystem_type",
    "gpu_runtime_identity",
    "main",
    "preflight_external_resources",
    "probe_writable_storage",
    "publish_root",
    "publish_source_snapshot",
    "resolve_temporary_directory",
    "run_attempt",
    "run_attempt_child",
    "run_attempt_protocol",
    "supervise_attempt",
    "supervisor_payload",
    "validate_root_artifact",
    "verdict_of_gate",
    "write_root_receipt",
)


if __name__ == "__main__":
    raise SystemExit(main())
