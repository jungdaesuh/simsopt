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
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
    CERTIFIED_LOWERED_KERNEL_NAMES,
    CERTIFIED_MAXIMUM_ITERATIONS,
    CERTIFIED_ROUTE_OPTIONS,
    CPU_BOOTSTRAP_OBSERVABLES,
    INFORMATIONAL_ENDPOINT_OBSERVABLES,
    NATIVE_ENDPOINT_STATE_CONTENT_SHA256,
    NATIVE_ENDPOINT_STATE_FILE_SHA256,
    NATIVE_ENDPOINT_STATE_PATH,
    NATIVE_TARGET_OBJECTIVE,
    NATIVE_WALL_SECONDS_BAR,
    PINNED_ENDPOINT_QUALITY_TERMS,
    TERMINAL_COORDINATES_FILENAME,
    BoundCase,
    RehearsalError,
    artifact_manifest_payload,
    bind_execution_sources,
    bind_problem_identity,
    build_endpoint_ledger,
    certify_endpoint_agreement,
    certify_native_reference,
    collapse_proximity_margin,
    endpoint_ledger_is_gated,
    endpoint_relative_differences,
    gate_endpoint_ledger,
    gate_endpoint_ledger_against_frozen_native,
    iteration_payload,
    json_scalar,
    load_execution_source_manifest,
    load_native_endpoint_state,
    measure_lowering_pre_gate,
    problem_identity_evidence,
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
# TSL's candidate list, in its order.  Both shipped binaries carry all three
# names beside the resolver's terminal "We are not able to find a directory for
# temporary files." -- and ``TEST_TMPDIR`` is tried BEFORE ``TMPDIR``, so an
# operator shell holding it would send the spill somewhere the preflight never
# probed while the receipt recorded the probed one.
TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES: Final = ("TEST_TMPDIR", "TMPDIR", "TMP")
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

# The receipt's own shape, frozen -- ONE listing, walked recursively.
#
# ``validate_root_artifact`` used to index into whatever fields it needed, so a
# receipt missing its source snapshot, its supervisor block, its preflight, its
# cache accounting and its telemetry re-validated clean and could not be told
# from a complete one.  Freezing the top-level names closed that one level deep;
# freezing every block below them closed it two levels deep; and the round after
# that found the identical hole one level lower again -- ``execution_sources``,
# the module-byte custody binding, sat in a required KEY SET with no entry in
# the parallel map of nested SHAPES, so ``execution_sources: null`` published
# and re-validated clean as ``CLAIM_DISCHARGED`` beside a plan sentence saying
# it could not.  Two hand-written enumerations -- "the keys that must be there"
# and "the blocks that have shapes" -- reproduce that hole once per round, and
# a suite that enumerates the shapes which EXIST is structurally blind to a
# shape that is ABSENT.
#
# So there is one structure and the key sets are DERIVED from it.  Every value
# in a shape is a nested shape, an ``_each(...)`` list of them, a TYPED LEAF, or
# an explicitly ``_dispatched(...)`` node naming the function that validates it.
# There is no ``_ANY``: a leaf states what the producer writes there, so a
# receipt whose cache accounting is ``{entry_count: null, total_bytes: null,
# entries_digest: null}``, or whose ``chain_wall`` is the string ``"not a
# number"``, is refused by name instead of passing for one that states them.
# ``UNSHAPED_LEAVES`` is the complete list of places where a mapping or a list
# is admitted without an inner shape, each with its reason; the suite walks the
# trees and requires that list to be exactly what it finds, so a block cannot be
# added without a shape and without saying why.


@dataclass(frozen=True, slots=True)
class _Leaf:
    """What the producer writes at one leaf of the receipt."""

    description: str
    types: tuple[type, ...]
    nullable: bool


@dataclass(frozen=True, slots=True)
class _Dispatched:
    """A node whose shape is chosen by the named function, not fixed here."""

    owner: str


def _leaf(description: str, *types: type, nullable: bool = False) -> _Leaf:
    return _Leaf(description, types, nullable)


_STRING: Final = _leaf("a string", str)
_STRING_OR_NULL: Final = _leaf("a string or null", str, nullable=True)
_NUMBER: Final = _leaf("a number", int, float)
_NUMBER_OR_NULL: Final = _leaf("a number or null", int, float, nullable=True)
# A COUNT, an INDEX, a BUDGET or a SIZE IN BYTES is a whole number, and the
# ``_NUMBER`` leaf admitted a fractional one.  Every gate that reads such a leaf
# read it through ``int(...)``, which TRUNCATES, so the receipt's own words were
# defeated by their own reader: ``stored_pairs: -0.5`` passed the check that says
# "which is not a count" (``int(-0.5) == 0``), ``status: 2.9`` passed "which is
# not one the engine reports" and minted ``latched: true``, and
# ``maximum_iterations: 700.9`` truncated into ``CERTIFIED_BUDGET`` /
# ``PREREGISTERED``.  The deferral that left this open reasoned that a receipt
# claiming 700.9 certified iterations describes nothing physical -- true, and
# beside the point, because the truncation is what let it seal.  Measured on the
# real producers at these bytes: every one of these leaves is a Python ``int``
# by construction (``len(...)``, ``os.getpid()``, ``st_dev``, ``st_size``,
# ``int(run.status)``, ``text.count(...)``), so refusing the float form refuses
# nothing an honest chain writes.  ``bool`` is excluded here as it is for
# ``_NUMBER``: ``true`` is not a count either.
_INTEGER: Final = _leaf("a whole number", int)
_INTEGER_OR_NULL: Final = _leaf("a whole number or null", int, nullable=True)
_BOOL: Final = _leaf("a boolean", bool)
_LIST: Final = _leaf("a list", list)
_MAPPING: Final = _leaf("a mapping", dict)
_NULL: Final = _leaf("null", nullable=True)


def _each(shape: Mapping[str, object]) -> tuple[Mapping[str, object]]:
    """A published list whose every element carries ``shape``."""

    return (shape,)


CACHE_STATE_SHAPE: Final = {
    "entry_count": _INTEGER,
    "entries_digest": _STRING,
    "total_bytes": _INTEGER,
}
CACHE_CONFIGURATION_SHAPE: Final = {
    "directory": _STRING,
    "enabled": _BOOL,
    "min_compile_time_seconds": _NUMBER,
    "min_entry_size_bytes": _INTEGER,
}
RUNTIME_IDENTITY_SHAPE: Final = {
    "backend": _STRING,
    "device_count": _INTEGER,
    "device_kind": _STRING,
    "device_platform": _STRING,
    "jax_version": _STRING,
    "jaxlib_version": _STRING,
    "native_extension_path": _STRING,
    "process_id": _INTEGER,
    "python_executable": _STRING,
    "python_prefix": _STRING,
}
STORAGE_PROBE_SHAPE: Final = {
    "advisory_available_bytes": _INTEGER,
    "device_id": _INTEGER,
    "directory": _STRING,
    "filesystem_type": _STRING,
    "one_byte_write": _STRING,
    "resolved_directory": _STRING,
    "role": _STRING,
}
PREFLIGHT_SHAPE: Final = {
    "gpu_inventory_executable": _STRING,
    "native_endpoint_state_content_sha256": _STRING,
    "native_endpoint_state_path": _STRING,
    "native_endpoint_state_sha256": _STRING,
    "resolved_temporary_directory": _STRING,
    "storage": _each(STORAGE_PROBE_SHAPE),
    "temporary_directory": _STRING,
    "visible_gpu_uuids": _LIST,
}
SUPERVISOR_SHAPE: Final = {
    "attempt_timeout_seconds": _NUMBER,
    "gpu_uuid": _STRING,
    "gpu_zero_asserted": _BOOL,
    "preflight": PREFLIGHT_SHAPE,
    "runtime_identity": RUNTIME_IDENTITY_SHAPE,
}
WORKTREE_IDENTITY_SHAPE: Final = {
    "git_head": _STRING,
    "repo_root": _STRING,
    "tracked_diff_sha256": _STRING,
    "untracked_bytes_manifest_sha256": _STRING,
}
SOURCE_SNAPSHOT_SHAPE: Final = {
    "entry_count": _INTEGER,
    "manifest_sha256": _STRING,
    "relative_path": _STRING,
    "worktree": WORKTREE_IDENTITY_SHAPE,
}
ROOT_CLAIM_SHAPE: Final = {
    "feasibility_tolerance": _NUMBER,
    "target_objective": _NUMBER,
    "wall_seconds_bar": _NUMBER,
}
ROOT_TIMING_SHAPE: Final = {"chain_wall": _NUMBER}
COLD_LANE_ANOMALY_SHAPE: Final = {
    "artifact_relative_path": _STRING,
    "gate_refused": _STRING_OR_NULL,
    "outcome": _STRING,
    "return_code": _INTEGER,
    "supervised_seconds": _NUMBER,
    "timed_out": _BOOL,
}
GPU_MEMORY_SHAPE: Final = {
    "availability": _STRING,
    "child_argv_sha256": _STRING,
    "child_pid": _INTEGER,
    "child_start_time_ticks": _INTEGER_OR_NULL,
    "device_uuid": _STRING,
    "monitor_scope": _STRING,
    "parent_pid": _INTEGER,
    "peak_used_memory_mib": _NUMBER_OR_NULL,
    "sample_count": _INTEGER,
    "unavailable_reason": _STRING_OR_NULL,
}
ATTEMPT_TIMING_SHAPE: Final = {
    "attempt_wall": _NUMBER,
    "bootstrap": _NUMBER,
    "engine_compile": _NUMBER,
    "engine_solve": _NUMBER,
    "engine_wall": _NUMBER,
    "lowering_pre_gate": _NUMBER,
    "problem_identity": _NUMBER,
}
ATTEMPT_CACHE_SHAPE: Final = {
    "after": CACHE_STATE_SHAPE,
    "at_entry": CACHE_STATE_SHAPE,
    "before_engine": CACHE_STATE_SHAPE,
    "configuration": CACHE_CONFIGURATION_SHAPE,
    "warm": _BOOL,
}
ATTEMPT_SOLVE_SHAPE: Final = {
    # The five host scalars that go through ``json_scalar`` publish null for a
    # nonfinite value, which is the shape a NaN reaches the receipt in; the
    # gates that read them refuse null as a number rather than here.
    "collapse_proximity_margin": _NUMBER_OR_NULL,
    "iterations_run": _INTEGER,
    "latched": _BOOL,
    "line_search_forced_refreshes": _INTEGER,
    "maximum_feasibility_inf": _NUMBER_OR_NULL,
    "monotone_descent": _BOOL,
    "projector_materializations": _INTEGER,
    "rows": _LIST,
    "status": _INTEGER,
    "status_name": _STRING,
    "stored_pairs": _INTEGER,
    "tangency_forced_refreshes": _INTEGER,
    "terminal_feasibility_inf": _NUMBER_OR_NULL,
    "terminal_objective": _NUMBER_OR_NULL,
    "terminal_projected_gradient_inf": _NUMBER_OR_NULL,
}
ATTEMPT_ENVIRONMENT_SHAPE: Final = {
    COMPILATION_CACHE_ENVIRONMENT_VARIABLE: _STRING,
    **{name: _STRING for name in GPU_REQUIRED_ENVIRONMENT},
}
ENDPOINT_AGREEMENT_SHAPE: Final = {
    "absolute_floor": _NUMBER,
    "feasibility_absolute_tolerance": _NUMBER,
    "loop_terminal_objective": _NUMBER,
    "relative_tolerance": _NUMBER,
    "standalone_terminal_objective": _NUMBER,
    "terminal_feasibility_inf": _NUMBER,
    "terminal_state_sha256": _STRING,
}

# The custody blocks the previous revision reached with nothing at all: the
# module-byte binding, the observable identity binding and the lowering
# pre-gate.  All three are the child's answer to "which bytes ran, on which
# problem, through which kernels", all three are published on every attempt,
# and all three were unreachable by the walker and unread by every validator.
EXECUTION_SOURCE_MANIFEST_SHAPE: Final = {
    "entries_sha256": _STRING,
    "entry_count": _INTEGER,
    "manifest_sha256": _STRING,
    "relative_path": _STRING,
    "schema_version": _STRING,
}
BOUND_MODULE_SHAPE: Final = {
    "module": _STRING,
    "relative_path": _STRING,
    "sha256": _STRING,
    "size_bytes": _INTEGER,
}
UNMANIFESTED_MODULE_SHAPE: Final = {"module": _STRING, "relative_path": _STRING}
INTERPRETER_INSTALLATION_SHAPE: Final = {"count": _INTEGER, "roots": _LIST}
EXECUTION_SOURCES_SHAPE: Final = {
    "bound_modules": _each(BOUND_MODULE_SHAPE),
    "interpreter_installation_modules": INTERPRETER_INSTALLATION_SHAPE,
    "manifest": EXECUTION_SOURCE_MANIFEST_SHAPE,
    "unmanifested_repository_modules": _each(UNMANIFESTED_MODULE_SHAPE),
}
PROBLEM_IDENTITY_SHAPE: Final = {
    "bound": _BOOL,
    "checks": _MAPPING,
    "feasibility_absolute_tolerance": _NUMBER,
    "measured_observables": _MAPPING,
    "recorded_bootstrap_sha256": _STRING,
    "recorded_problem_sha256": _STRING,
    "reference_observables": _MAPPING,
    "relative_difference": _MAPPING,
    "relative_tolerances": _MAPPING,
    "sha_is_binding": _BOOL,
}
LOWERED_KERNEL_SHAPE: Final = {
    "ir_bytes": _INTEGER,
    "name": _STRING,
    "while_operations": _INTEGER,
}
LOWERING_PRE_GATE_SHAPE: Final = {
    "budget_independent": _BOOL,
    "certified_iterations": _INTEGER,
    "kernels": _each(LOWERED_KERNEL_SHAPE),
    "rehearsal_iterations": _INTEGER,
    "total_ir_bytes": _INTEGER,
}

_ENDPOINT_LEDGER_KEYS: Final = {
    "gated_at_this_budget": _BOOL,
    "informational_observables": _LIST,
    "native": _MAPPING,
    "native_state_content_sha256": _STRING,
    "native_state_relative_path": _STRING,
    "native_state_sha256": _STRING,
    "pinned_quality_terms": _LIST,
    "relative_difference": _MAPPING,
    "terminal": _MAPPING,
}
ENDPOINT_LEDGER_SHAPE: Final = dict(_ENDPOINT_LEDGER_KEYS)
GATED_ENDPOINT_LEDGER_SHAPE: Final = {
    **_ENDPOINT_LEDGER_KEYS,
    "pinned_term_gate": {
        "failed_terms": _LIST,
        "passed": _BOOL,
        "terms": _MAPPING,
    },
}

ATTEMPT_EVIDENCE_SHAPE: Final = {
    "attempt_index": _INTEGER,
    "certified_options_delta": _MAPPING,
    "compilation_cache": ATTEMPT_CACHE_SHAPE,
    "endpoint_agreement": ENDPOINT_AGREEMENT_SHAPE,
    "endpoint_ledger": _Dispatched("_validate_attempt_shape, gated or ungated"),
    "environment": ATTEMPT_ENVIRONMENT_SHAPE,
    "execution_sources": EXECUTION_SOURCES_SHAPE,
    # A completed chain publishes ``gate_refused: null`` and nothing else; the
    # refused shape below is the other document the child has.
    "gate_refused": _NULL,
    "lowering_pre_gate": LOWERING_PRE_GATE_SHAPE,
    "options": _MAPPING,
    "problem_identity": PROBLEM_IDENTITY_SHAPE,
    "quality_claim": _STRING,
    "route": _STRING,
    "runtime_identity": RUNTIME_IDENTITY_SHAPE,
    "schema_version": _STRING,
    "solve": ATTEMPT_SOLVE_SHAPE,
    "timing_boundary": _STRING,
    "timing_seconds": ATTEMPT_TIMING_SHAPE,
}
REFUSED_ATTEMPT_EVIDENCE_SHAPE: Final = {
    "attempt_index": _INTEGER,
    "error": _STRING,
    "gate_refused": _STRING,
    "route": _STRING,
    "schema_version": _STRING,
}
SUPERVISED_ATTEMPT_SHAPE: Final = {
    "argv_sha256": _STRING,
    "artifact_relative_path": _STRING,
    "attempt_index": _INTEGER,
    "evidence": _Dispatched("_validate_attempt_shape, one of the child's two"),
    "gpu_memory": GPU_MEMORY_SHAPE,
    "outcome": _STRING,
    "return_code": _INTEGER,
    "stderr_tail": _STRING,
    "stdout_tail": _STRING_OR_NULL,
    "supervised_seconds": _NUMBER,
    "timed_out": _BOOL,
}
COLD_LANE_SHAPE: Final = {
    **SUPERVISED_ATTEMPT_SHAPE,
    "timed_against_bar": _BOOL,
}
ATTEMPT_PROTOCOL_SHAPE: Final = {
    "attempts_run": _INTEGER,
    "authorized_attempts": _INTEGER,
    "certified_maximum_iterations": _INTEGER,
    "cold_lane_authorized": _BOOL,
    "conformance": _STRING,
    "latch_count": _INTEGER,
    "latch_rate": _STRING,
    "maximum_iterations": _INTEGER,
    "preregistered_attempts": _INTEGER,
    "stop_rule": _STRING,
}
ROOT_EVIDENCE_SHAPE: Final = {
    "attempt_protocol": ATTEMPT_PROTOCOL_SHAPE,
    "attempts": _Dispatched("_validate_attempt_shape, element by element"),
    "claim": ROOT_CLAIM_SHAPE,
    "cold_lane": _Dispatched("_validate_attempt_shape, or null"),
    "cold_lane_anomaly": _Dispatched("cold_lane_anomaly, re-derived and compared"),
    "compilation_cache": CACHE_STATE_SHAPE,
    "quality_claim": _STRING,
    "route": _STRING,
    "schema_version": _STRING,
    "source_snapshot": SOURCE_SNAPSHOT_SHAPE,
    "supervisor": SUPERVISOR_SHAPE,
    "timing_boundary": _STRING,
    "timing_seconds": ROOT_TIMING_SHAPE,
    "verdict": _STRING,
}

# DERIVED, never written twice: a required key with no shape is the defect this
# structure exists to make impossible, and it is impossible only while these
# are functions of the shapes rather than second listings beside them.
ROOT_EVIDENCE_REQUIRED_KEYS: Final = frozenset(ROOT_EVIDENCE_SHAPE)
ATTEMPT_PROTOCOL_REQUIRED_KEYS: Final = frozenset(ATTEMPT_PROTOCOL_SHAPE)
SUPERVISED_ATTEMPT_REQUIRED_KEYS: Final = frozenset(SUPERVISED_ATTEMPT_SHAPE)
ATTEMPT_EVIDENCE_REQUIRED_KEYS: Final = frozenset(ATTEMPT_EVIDENCE_SHAPE)
REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS: Final = frozenset(REFUSED_ATTEMPT_EVIDENCE_SHAPE)

# Every document the receipt is built from, by the name its refusals carry.
RECEIPT_SHAPES: Final = {
    "root": ROOT_EVIDENCE_SHAPE,
    "supervised attempt": SUPERVISED_ATTEMPT_SHAPE,
    "cold lane": COLD_LANE_SHAPE,
    "attempt evidence": ATTEMPT_EVIDENCE_SHAPE,
    "refused attempt evidence": REFUSED_ATTEMPT_EVIDENCE_SHAPE,
    "endpoint ledger": ENDPOINT_LEDGER_SHAPE,
    "gated endpoint ledger": GATED_ENDPOINT_LEDGER_SHAPE,
    "cold lane anomaly": COLD_LANE_ANOMALY_SHAPE,
}

# Where a mapping or a list is admitted without an inner shape, and why.  The
# suite walks ``RECEIPT_SHAPES`` and requires this to be exactly what it finds,
# which is what stops the next block from arriving unshaped: adding one costs a
# line here and a reason.
UNSHAPED_LEAVES: Final = {
    "root.attempts": "each element is a supervised attempt, walked by _validate_attempt_shape",
    "root.cold_lane": "null, or a cold-lane record walked by _validate_attempt_shape",
    "root.cold_lane_anomaly": "null, or COLD_LANE_ANOMALY_SHAPE; re-derived from the lane and compared",
    "root.supervisor.preflight.visible_gpu_uuids": "an inventory of device UUIDs; every element is required to be a string and the pinned one to be among them",
    "supervised attempt.evidence": "null, refused or complete; _validate_attempt_shape picks the shape",
    "cold lane.evidence": "null, refused or complete; _validate_attempt_shape picks the shape",
    "attempt evidence.certified_options_delta": "the fields this budget changed; re-derived from options against the frozen configuration and required to name maximum_iterations or nothing",
    "attempt evidence.endpoint_ledger": "gated or ungated; _validate_attempt_shape picks the shape from gated_at_this_budget",
    "attempt evidence.options": "the certified dataclass's fields; the key set is checked against it and every VALUE compared to CERTIFIED_ROUTE_OPTIONS, with only maximum_iterations permitted to differ",
    "attempt evidence.problem_identity.checks": "one boolean per bound observable; re-derived by problem_identity_evidence",
    "attempt evidence.problem_identity.measured_observables": "the four bootstrap observables; re-derived by problem_identity_evidence",
    "attempt evidence.problem_identity.reference_observables": "the campaign's frozen observables; compared to them",
    "attempt evidence.problem_identity.relative_difference": "re-derived by problem_identity_evidence from the measured side",
    "attempt evidence.problem_identity.relative_tolerances": "the campaign's frozen tolerances; compared to them",
    "attempt evidence.execution_sources.interpreter_installation_modules.roots": "the hidden top-level directories the interpreter installation lives under",
    "attempt evidence.solve.rows": "the recorded iterates; _validate_solve_telemetry re-derives iterations_run, maximum_feasibility_inf, monotone_descent and (through status) status_name and latched from them, and requires the terminal objective to be an endpoint of the last one. The terminal scalars are measured at a point the rows do not contain, so they are bound to the endpoint agreement rather than projected from a row",
    "endpoint ledger.informational_observables": "the campaign's frozen informational set; compared to it",
    "endpoint ledger.native": "one number per physics term; compared term by term to the campaign's frozen native reference on a PRE-REGISTERED attempt (ruling 17 leaves the cold lane's native side uncompared, by design)",
    "endpoint ledger.pinned_quality_terms": "the campaign's frozen pinned set; compared to it",
    "endpoint ledger.relative_difference": "re-derived from the two sides",
    "endpoint ledger.terminal": "one number per physics term; the relative-difference column is recomputed from it and weighted_total is required to be the standalone terminal objective (this node carries no gate: an UNGATED ledger is the shape a lane or a non-latching draw publishes, and by ruling 13 it decides nothing)",
    "gated endpoint ledger.informational_observables": "the campaign's frozen informational set; compared to it",
    "gated endpoint ledger.native": "one number per physics term; compared term by term to the campaign's frozen native reference on a PRE-REGISTERED attempt (ruling 17 leaves the cold lane's native side uncompared, by design)",
    "gated endpoint ledger.pinned_quality_terms": "the campaign's frozen pinned set; compared to it",
    "gated endpoint ledger.pinned_term_gate.failed_terms": "the terms the recomputed gate names; the whole block is recomputed and compared",
    "gated endpoint ledger.pinned_term_gate.terms": "one verdict per pinned term; the whole block is recomputed and compared",
    "gated endpoint ledger.relative_difference": "re-derived from the two sides",
    "gated endpoint ledger.terminal": "one number per physics term; the gate is recomputed from it, against the campaign's frozen native literals on the attempt that discharges the claim, and weighted_total is required to be the standalone terminal objective",
}

# WHAT EVERY TYPED LEAF IS BOUND TO, and for the ones bound to nothing, why.
#
# The campaign's whole history is one defect told six times: each remediation
# binds the leaf the last round found and leaves its neighbour free, and the
# next round finds the neighbour.  A shape tree made an ABSENT SHAPE
# unrepresentable; this makes an UNBOUND CLAIM-BEARING LEAF unrepresentable the
# same way.  Every ``_Leaf`` of ``RECEIPT_SHAPES`` is declared here as exactly
# one of:
#
#   ``BINDING_FROZEN``    compared to something outside the receipt -- a
#                         campaign constant, or a frozen owner that holds one.
#   ``BINDING_DERIVED``   re-derived from other published fields, or from the
#                         producer's own owner asked again.
#   ``BINDING_DIGEST``    recomputed by hashing an artifact the tree carries.
#   ``BINDING_NONE``      read by nothing, with the reason stated.
#
# and the second element is the ANCHOR: the module-level name the comparison is
# against, or the function that re-derives it, or -- for ``BINDING_NONE`` -- the
# reason.  The suite requires this map to be exactly the leaves the walker
# finds, requires every anchor to resolve in this module, and requires
# ``CLAIM_BEARING_LEAVES`` to carry no ``BINDING_NONE``.
#
# What it does NOT prove is that the named anchor is reached on every path; that
# is what the refusal-site census and its kill tests in the suite are for, and
# the two are meant to be read together.  What it does make impossible is the
# thing that has actually happened six times: a claim-bearing leaf shipped with
# nothing on the other side of it.
BINDING_FROZEN: Final = "frozen literal"
BINDING_DERIVED: Final = "re-derivation"
BINDING_DIGEST: Final = "digest"
BINDING_NONE: Final = "unbound"
BINDING_KINDS: Final = frozenset(
    {BINDING_FROZEN, BINDING_DERIVED, BINDING_DIGEST, BINDING_NONE}
)


def _prefixed(
    prefix: str, bindings: Mapping[str, tuple[str, str]]
) -> dict[str, tuple[str, str]]:
    """One block's bindings, under the path the walker reports it at."""

    return {f"{prefix}.{name}": binding for name, binding in bindings.items()}


_RUNTIME_IDENTITY_UNREAD: Final = (
    BINDING_NONE,
    (
        "the timed child's device and toolchain beyond its backend; deferred, and "
        "the same deferral covers the supervisor's own copy"
    ),
)
_CACHE_STATE_UNREAD: Final = (
    BINDING_NONE,
    (
        "cache accounting published for a reader; only at_entry.entry_count decides "
        "anything (warm), and the directory itself is a standing deferral"
    ),
)
_SUPERVISED_DRAW_BINDINGS: Final = {
    "argv_sha256": (BINDING_DERIVED, "_validate_supervised_launch"),
    "artifact_relative_path": (BINDING_DERIVED, "validate_root_artifact"),
    "attempt_index": (BINDING_DERIVED, "validate_root_artifact"),
    "gpu_memory.availability": (
        BINDING_NONE,
        (
            "the sampler's own verdict on itself; the producer's fallback publishes "
            "unavailability rather than an inferred zero and nothing gates on it"
        ),
    ),
    "gpu_memory.child_argv_sha256": (BINDING_DERIVED, "_validate_supervised_launch"),
    "gpu_memory.child_pid": (BINDING_DERIVED, "_validate_supervised_launch"),
    "gpu_memory.child_start_time_ticks": (
        BINDING_NONE,
        (
            "procfs start ticks, published so a reader can tell two children apart; "
            "unread, and null whenever the sampler bound nothing"
        ),
    ),
    "gpu_memory.device_uuid": (BINDING_FROZEN, "GPU_UUID"),
    "gpu_memory.monitor_scope": (
        BINDING_NONE,
        "a label for what the sampler watched; unread",
    ),
    "gpu_memory.parent_pid": (BINDING_DERIVED, "_validate_supervised_launch"),
    "gpu_memory.peak_used_memory_mib": (
        BINDING_NONE,
        (
            "the observation itself, reported and never gated; the claim is about "
            "wall time, not occupancy"
        ),
    ),
    "gpu_memory.sample_count": (
        BINDING_NONE,
        "how many samples the poller took; unread",
    ),
    "gpu_memory.unavailable_reason": (
        BINDING_NONE,
        "why the sampler bound nothing; unread",
    ),
    "outcome": (BINDING_DERIVED, "_validate_attempt_outcome"),
    "return_code": (BINDING_DERIVED, "_attempt_outcome"),
    "stderr_tail": (
        BINDING_NONE,
        (
            "the child's narrative tail, published so a defect report names what the "
            "child wrote; a free-form string by construction"
        ),
    ),
    "stdout_tail": (
        BINDING_NONE,
        "the bytes a protocol failure refused, published for the same reason",
    ),
    "supervised_seconds": (BINDING_DERIVED, "_validate_supervised_launch"),
    "timed_out": (BINDING_DERIVED, "_validate_supervised_launch"),
}
_ENDPOINT_LEDGER_BINDINGS: Final = {
    "gated_at_this_budget": (BINDING_DERIVED, "endpoint_ledger_is_gated"),
    "informational_observables": (BINDING_FROZEN, "INFORMATIONAL_ENDPOINT_OBSERVABLES"),
    "native": (BINDING_FROZEN, "certify_native_reference"),
    "native_state_content_sha256": (
        BINDING_FROZEN,
        "NATIVE_ENDPOINT_STATE_CONTENT_SHA256",
    ),
    "native_state_relative_path": (BINDING_FROZEN, "NATIVE_ENDPOINT_STATE_PATH"),
    "native_state_sha256": (BINDING_FROZEN, "NATIVE_ENDPOINT_STATE_FILE_SHA256"),
    "pinned_quality_terms": (BINDING_FROZEN, "PINNED_ENDPOINT_QUALITY_TERMS"),
    "relative_difference": (BINDING_DERIVED, "endpoint_relative_differences"),
    "terminal": (BINDING_DERIVED, "_validate_terminal_endpoint_column"),
}
LEAF_BINDINGS: Final = {
    **_prefixed(
        "root",
        {
            "attempt_protocol.attempts_run": (BINDING_DERIVED, "validate_root_artifact"),
            "attempt_protocol.authorized_attempts": (
                BINDING_FROZEN,
                "PREREGISTERED_ATTEMPTS",
            ),
            "attempt_protocol.certified_maximum_iterations": (
                BINDING_FROZEN,
                "CERTIFIED_MAXIMUM_ITERATIONS",
            ),
            "attempt_protocol.cold_lane_authorized": (
                BINDING_DERIVED,
                "validate_root_artifact",
            ),
            "attempt_protocol.conformance": (
                BINDING_DERIVED,
                "attempt_protocol_conformance",
            ),
            "attempt_protocol.latch_count": (BINDING_DERIVED, "validate_root_artifact"),
            "attempt_protocol.latch_rate": (BINDING_DERIVED, "validate_root_artifact"),
            "attempt_protocol.maximum_iterations": (
                BINDING_DERIVED,
                "_validate_attempt_record",
            ),
            "attempt_protocol.preregistered_attempts": (
                BINDING_FROZEN,
                "PREREGISTERED_ATTEMPTS",
            ),
            "attempt_protocol.stop_rule": (BINDING_FROZEN, "ATTEMPT_STOP_RULE"),
            "claim.feasibility_tolerance": (BINDING_FROZEN, "CERTIFIED_ROUTE_OPTIONS"),
            "claim.target_objective": (BINDING_FROZEN, "NATIVE_TARGET_OBJECTIVE"),
            "claim.wall_seconds_bar": (BINDING_FROZEN, "NATIVE_WALL_SECONDS_BAR"),
            "compilation_cache.entry_count": _CACHE_STATE_UNREAD,
            "compilation_cache.entries_digest": _CACHE_STATE_UNREAD,
            "compilation_cache.total_bytes": _CACHE_STATE_UNREAD,
            "quality_claim": (BINDING_DERIVED, "_quality_claim"),
            "route": (BINDING_FROZEN, "PROJECTED_ROUTE"),
            "schema_version": (BINDING_FROZEN, "GPU_ROOT_SCHEMA_VERSION"),
            "source_snapshot.entry_count": (
                BINDING_NONE,
                (
                    "the sealed source snapshot is re-derived by nothing; tying it "
                    "means recomputing the snapshot manifest inside the validator, "
                    "which is new code on the publication path, and the artifact "
                    "manifest already covers every byte after publication"
                ),
            ),
            "source_snapshot.manifest_sha256": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "source_snapshot.relative_path": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "source_snapshot.worktree.git_head": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "source_snapshot.worktree.repo_root": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "source_snapshot.worktree.tracked_diff_sha256": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "source_snapshot.worktree.untracked_bytes_manifest_sha256": (
                BINDING_NONE,
                "same deferral as source_snapshot.entry_count",
            ),
            "supervisor.attempt_timeout_seconds": (
                BINDING_FROZEN,
                "ATTEMPT_TIMEOUT_SECONDS",
            ),
            "supervisor.gpu_uuid": (BINDING_FROZEN, "GPU_UUID"),
            "supervisor.gpu_zero_asserted": (
                BINDING_NONE,
                (
                    "the supervisor states that it is NOT GPU-zero; a receipt saying "
                    "otherwise describes a different supervisor and nothing reads it"
                ),
            ),
            "supervisor.preflight.gpu_inventory_executable": (
                BINDING_NONE,
                "which binary answered the device query; provenance, unread",
            ),
            "supervisor.preflight.native_endpoint_state_content_sha256": (
                BINDING_FROZEN,
                "NATIVE_ENDPOINT_STATE_CONTENT_SHA256",
            ),
            "supervisor.preflight.native_endpoint_state_path": (
                BINDING_FROZEN,
                "NATIVE_ENDPOINT_STATE_PATH",
            ),
            "supervisor.preflight.native_endpoint_state_sha256": (
                BINDING_FROZEN,
                "NATIVE_ENDPOINT_STATE_FILE_SHA256",
            ),
            "supervisor.preflight.resolved_temporary_directory": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.storage[].advisory_available_bytes": (
                BINDING_NONE,
                (
                    "advisory by construction: this is the number that said 12.29 GiB "
                    "free on a filesystem whose one-byte write returned EDQUOT"
                ),
            ),
            "supervisor.preflight.storage[].device_id": (
                BINDING_NONE,
                "which device the probed directory sits on; provenance, unread",
            ),
            "supervisor.preflight.storage[].directory": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.storage[].filesystem_type": (
                BINDING_FROZEN,
                "REFUSED_STORAGE_FILESYSTEM_TYPES",
            ),
            "supervisor.preflight.storage[].one_byte_write": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.storage[].resolved_directory": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.storage[].role": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.temporary_directory": (
                BINDING_DERIVED,
                "_validate_preflight_record",
            ),
            "supervisor.preflight.visible_gpu_uuids": (BINDING_FROZEN, "GPU_UUID"),
            "supervisor.runtime_identity.backend": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.device_count": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.device_kind": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.device_platform": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.jax_version": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.jaxlib_version": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.native_extension_path": (
                _RUNTIME_IDENTITY_UNREAD
            ),
            "supervisor.runtime_identity.process_id": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.python_executable": _RUNTIME_IDENTITY_UNREAD,
            "supervisor.runtime_identity.python_prefix": _RUNTIME_IDENTITY_UNREAD,
            "timing_boundary": (BINDING_FROZEN, "validate_root_artifact"),
            "timing_seconds.chain_wall": (BINDING_DERIVED, "validate_root_artifact"),
            "verdict": (BINDING_DERIVED, "derive_verdict"),
        },
    ),
    **_prefixed("supervised attempt", _SUPERVISED_DRAW_BINDINGS),
    **_prefixed(
        "cold lane",
        {
            **_SUPERVISED_DRAW_BINDINGS,
            "timed_against_bar": (BINDING_DERIVED, "validate_root_artifact"),
        },
    ),
    **_prefixed(
        "attempt evidence",
        {
            "attempt_index": (BINDING_DERIVED, "_validate_attempt_record"),
            "certified_options_delta": (
                BINDING_DERIVED,
                "_validate_certified_route_options",
            ),
            "compilation_cache.after.entry_count": _CACHE_STATE_UNREAD,
            "compilation_cache.after.entries_digest": _CACHE_STATE_UNREAD,
            "compilation_cache.after.total_bytes": _CACHE_STATE_UNREAD,
            "compilation_cache.at_entry.entry_count": (
                BINDING_DERIVED,
                "_validate_attempt_record",
            ),
            "compilation_cache.at_entry.entries_digest": _CACHE_STATE_UNREAD,
            "compilation_cache.at_entry.total_bytes": _CACHE_STATE_UNREAD,
            "compilation_cache.before_engine.entry_count": _CACHE_STATE_UNREAD,
            "compilation_cache.before_engine.entries_digest": _CACHE_STATE_UNREAD,
            "compilation_cache.before_engine.total_bytes": _CACHE_STATE_UNREAD,
            "compilation_cache.configuration.directory": (
                BINDING_NONE,
                (
                    "the attempt's compilation-cache directory is a standing "
                    "deferral; the cold lane is what makes the cache an accounting "
                    "device, and it is bound through warm rather than through a path"
                ),
            ),
            "compilation_cache.configuration.enabled": (
                BINDING_NONE,
                "same deferral as the cache directory",
            ),
            "compilation_cache.configuration.min_compile_time_seconds": (
                BINDING_NONE,
                "same deferral as the cache directory",
            ),
            "compilation_cache.configuration.min_entry_size_bytes": (
                BINDING_NONE,
                "same deferral as the cache directory",
            ),
            "compilation_cache.warm": (BINDING_DERIVED, "_validate_attempt_record"),
            "endpoint_agreement.absolute_floor": (
                BINDING_FROZEN,
                "DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR",
            ),
            "endpoint_agreement.feasibility_absolute_tolerance": (
                BINDING_FROZEN,
                "CERTIFIED_ROUTE_OPTIONS",
            ),
            "endpoint_agreement.loop_terminal_objective": (
                BINDING_DERIVED,
                "_validate_terminal_endpoint_column",
            ),
            "endpoint_agreement.relative_tolerance": (
                BINDING_FROZEN,
                "DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE",
            ),
            "endpoint_agreement.standalone_terminal_objective": (
                BINDING_DERIVED,
                "_validate_terminal_endpoint_column",
            ),
            "endpoint_agreement.terminal_feasibility_inf": (
                BINDING_DERIVED,
                "_validate_terminal_endpoint_column",
            ),
            "endpoint_agreement.terminal_state_sha256": (
                BINDING_DIGEST,
                "exact_numeric_tree_sha256",
            ),
            "environment.JAX_COMPILATION_CACHE_DIR": (
                BINDING_NONE,
                "same deferral as the cache directory",
            ),
            "environment.JAX_ENABLE_X64": (BINDING_FROZEN, "GPU_REQUIRED_ENVIRONMENT"),
            "environment.JAX_PLATFORMS": (BINDING_FROZEN, "GPU_REQUIRED_ENVIRONMENT"),
            "environment.XLA_PYTHON_CLIENT_PREALLOCATE": (
                BINDING_FROZEN,
                "GPU_REQUIRED_ENVIRONMENT",
            ),
            "execution_sources.bound_modules[].module": (
                BINDING_NONE,
                (
                    "the importable NAME of a bound module, published so a refusal "
                    "can say which one; custody is decided by the three leaves below"
                ),
            ),
            "execution_sources.bound_modules[].relative_path": (
                BINDING_DERIVED,
                "_validate_execution_sources",
            ),
            "execution_sources.bound_modules[].sha256": (
                BINDING_DERIVED,
                "_validate_execution_sources",
            ),
            "execution_sources.bound_modules[].size_bytes": (
                BINDING_DERIVED,
                "_validate_execution_sources",
            ),
            "execution_sources.interpreter_installation_modules.count": (
                BINDING_NONE,
                (
                    "hidden top-level directories INSIDE the checkout that the "
                    "interpreter installation lives under; a venv outside the tree "
                    "publishes {count: 0, roots: []} and is equally honest, so the "
                    "field cannot discharge an interpreter pin"
                ),
            ),
            "execution_sources.interpreter_installation_modules.roots": (
                BINDING_NONE,
                "same deferral as its count",
            ),
            "execution_sources.manifest.entries_sha256": (
                BINDING_DERIVED,
                "load_execution_source_manifest",
            ),
            "execution_sources.manifest.entry_count": (
                BINDING_DERIVED,
                "load_execution_source_manifest",
            ),
            "execution_sources.manifest.manifest_sha256": (
                BINDING_DERIVED,
                "load_execution_source_manifest",
            ),
            "execution_sources.manifest.relative_path": (
                BINDING_DERIVED,
                "load_execution_source_manifest",
            ),
            "execution_sources.manifest.schema_version": (
                BINDING_DERIVED,
                "load_execution_source_manifest",
            ),
            "execution_sources.unmanifested_repository_modules[].module": (
                BINDING_DERIVED,
                "_validate_execution_sources",
            ),
            "execution_sources.unmanifested_repository_modules[].relative_path": (
                BINDING_DERIVED,
                "_validate_execution_sources",
            ),
            "gate_refused": (BINDING_DERIVED, "_validate_attempt_shape"),
            "lowering_pre_gate.budget_independent": (
                BINDING_DERIVED,
                "_validate_lowering_pre_gate",
            ),
            "lowering_pre_gate.certified_iterations": (
                BINDING_FROZEN,
                "CERTIFIED_MAXIMUM_ITERATIONS",
            ),
            "lowering_pre_gate.kernels[].ir_bytes": (
                BINDING_DERIVED,
                "_validate_lowering_pre_gate",
            ),
            "lowering_pre_gate.kernels[].name": (
                BINDING_FROZEN,
                "CERTIFIED_LOWERED_KERNEL_NAMES",
            ),
            "lowering_pre_gate.kernels[].while_operations": (
                BINDING_DERIVED,
                "_validate_lowering_pre_gate",
            ),
            "lowering_pre_gate.rehearsal_iterations": (
                BINDING_DERIVED,
                "_validate_lowering_pre_gate",
            ),
            "lowering_pre_gate.total_ir_bytes": (
                BINDING_DERIVED,
                "_validate_lowering_pre_gate",
            ),
            "options": (BINDING_FROZEN, "CERTIFIED_ROUTE_OPTIONS"),
            "problem_identity.bound": (BINDING_DERIVED, "_validate_problem_identity"),
            "problem_identity.checks": (BINDING_DERIVED, "problem_identity_evidence"),
            "problem_identity.feasibility_absolute_tolerance": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.measured_observables": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.recorded_bootstrap_sha256": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.recorded_problem_sha256": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.reference_observables": (
                BINDING_FROZEN,
                "CPU_BOOTSTRAP_OBSERVABLES",
            ),
            "problem_identity.relative_difference": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.relative_tolerances": (
                BINDING_DERIVED,
                "problem_identity_evidence",
            ),
            "problem_identity.sha_is_binding": (
                BINDING_DERIVED,
                "_validate_problem_identity",
            ),
            "quality_claim": (BINDING_DERIVED, "_quality_claim"),
            "route": (BINDING_FROZEN, "PROJECTED_ROUTE"),
            "runtime_identity.backend": (BINDING_FROZEN, "REQUIRED_BACKEND"),
            "runtime_identity.device_count": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.device_kind": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.device_platform": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.jax_version": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.jaxlib_version": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.native_extension_path": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.process_id": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.python_executable": _RUNTIME_IDENTITY_UNREAD,
            "runtime_identity.python_prefix": _RUNTIME_IDENTITY_UNREAD,
            "schema_version": (BINDING_FROZEN, "GPU_ATTEMPT_SCHEMA_VERSION"),
            "solve.collapse_proximity_margin": (
                BINDING_NONE,
                (
                    "reported and never acted on: the run's smallest recorded step "
                    "scale in units of the line-search floor, whose only separation "
                    "the banked evidence carries is unity"
                ),
            ),
            "solve.iterations_run": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.latched": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.line_search_forced_refreshes": (
                BINDING_DERIVED,
                "_validate_solve_telemetry",
            ),
            "solve.maximum_feasibility_inf": (
                BINDING_DERIVED,
                "_validate_solve_telemetry",
            ),
            "solve.monotone_descent": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.projector_materializations": (
                BINDING_DERIVED,
                "_validate_solve_telemetry",
            ),
            "solve.rows": (BINDING_DERIVED, "_iterate_column"),
            "solve.status": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.status_name": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.stored_pairs": (BINDING_DERIVED, "_validate_solve_telemetry"),
            "solve.tangency_forced_refreshes": (
                BINDING_DERIVED,
                "_validate_solve_telemetry",
            ),
            "solve.terminal_feasibility_inf": (
                BINDING_DERIVED,
                "_validate_terminal_endpoint_column",
            ),
            "solve.terminal_objective": (
                BINDING_DERIVED,
                "_validate_terminal_endpoint_column",
            ),
            "solve.terminal_projected_gradient_inf": (
                BINDING_NONE,
                (
                    "the terminal projected gradient, reported: section 1 states the "
                    "claim on the objective and on feasibility, and a stationarity "
                    "band is not part of it"
                ),
            ),
            "timing_boundary": (BINDING_FROZEN, "_validate_attempt_record"),
            "timing_seconds.attempt_wall": (BINDING_DERIVED, "_validate_attempt_record"),
            "timing_seconds.bootstrap": (BINDING_DERIVED, "_validate_attempt_record"),
            "timing_seconds.engine_compile": (
                BINDING_DERIVED,
                "attempt_engine_wall_seconds",
            ),
            "timing_seconds.engine_solve": (
                BINDING_DERIVED,
                "attempt_engine_wall_seconds",
            ),
            "timing_seconds.engine_wall": (
                BINDING_DERIVED,
                "attempt_engine_wall_seconds",
            ),
            "timing_seconds.lowering_pre_gate": (
                BINDING_DERIVED,
                "_validate_attempt_record",
            ),
            "timing_seconds.problem_identity": (
                BINDING_DERIVED,
                "_validate_attempt_record",
            ),
        },
    ),
    **_prefixed(
        "refused attempt evidence",
        {
            "attempt_index": (
                BINDING_NONE,
                (
                    "a refused draw decides nothing but its own outcome (ruling 17), "
                    "so its document is shape-checked and its gate name read"
                ),
            ),
            "error": (
                BINDING_NONE,
                "the refusing gate's own words; free-form by construction",
            ),
            "gate_refused": (BINDING_DERIVED, "_attempt_outcome"),
            "route": (
                BINDING_NONE,
                "same reason as the refused document's attempt_index",
            ),
            "schema_version": (
                BINDING_NONE,
                "same reason as the refused document's attempt_index",
            ),
        },
    ),
    **_prefixed("endpoint ledger", _ENDPOINT_LEDGER_BINDINGS),
    **_prefixed(
        "gated endpoint ledger",
        {
            **_ENDPOINT_LEDGER_BINDINGS,
            "pinned_term_gate.failed_terms": (BINDING_DERIVED, "gate_endpoint_ledger"),
            "pinned_term_gate.passed": (
                BINDING_DERIVED,
                "gate_endpoint_ledger_against_frozen_native",
            ),
            "pinned_term_gate.terms": (BINDING_DERIVED, "gate_endpoint_ledger"),
        },
    ),
    **_prefixed(
        "cold lane anomaly",
        {
            "artifact_relative_path": (BINDING_DERIVED, "cold_lane_anomaly"),
            "gate_refused": (BINDING_DERIVED, "cold_lane_anomaly"),
            "outcome": (BINDING_DERIVED, "cold_lane_anomaly"),
            "return_code": (BINDING_DERIVED, "cold_lane_anomaly"),
            "supervised_seconds": (BINDING_DERIVED, "cold_lane_anomaly"),
            "timed_out": (BINDING_DERIVED, "cold_lane_anomaly"),
        },
    ),
}

# The leaves section 1's claim is made of: the verdict and its pre-registered
# precondition, the five comparisons that decide whether a draw discharges the
# claim, the route the claim is about, the device it is stated for, the custody
# of the bytes that produced it, and the wall it is measured in.  None of these
# may be ``BINDING_NONE``, and the suite makes that unrepresentable.
CLAIM_BEARING_LEAVES: Final = frozenset(
    {
        "root.verdict",
        "root.quality_claim",
        "root.route",
        "root.schema_version",
        "root.claim.feasibility_tolerance",
        "root.claim.target_objective",
        "root.claim.wall_seconds_bar",
        "root.attempt_protocol.attempts_run",
        "root.attempt_protocol.authorized_attempts",
        "root.attempt_protocol.certified_maximum_iterations",
        "root.attempt_protocol.cold_lane_authorized",
        "root.attempt_protocol.conformance",
        "root.attempt_protocol.latch_count",
        "root.attempt_protocol.latch_rate",
        "root.attempt_protocol.maximum_iterations",
        "root.attempt_protocol.preregistered_attempts",
        "root.attempt_protocol.stop_rule",
        "root.supervisor.attempt_timeout_seconds",
        "root.supervisor.gpu_uuid",
        "root.supervisor.preflight.native_endpoint_state_content_sha256",
        "root.supervisor.preflight.native_endpoint_state_path",
        "root.supervisor.preflight.native_endpoint_state_sha256",
        "root.supervisor.preflight.visible_gpu_uuids",
        "root.supervisor.preflight.storage[].filesystem_type",
        "root.timing_seconds.chain_wall",
        "root.timing_boundary",
        "supervised attempt.outcome",
        "supervised attempt.return_code",
        "supervised attempt.supervised_seconds",
        "supervised attempt.timed_out",
        "supervised attempt.argv_sha256",
        "supervised attempt.attempt_index",
        "supervised attempt.artifact_relative_path",
        "supervised attempt.gpu_memory.device_uuid",
        "attempt evidence.options",
        "attempt evidence.certified_options_delta",
        "attempt evidence.route",
        "attempt evidence.schema_version",
        "attempt evidence.quality_claim",
        "attempt evidence.timing_boundary",
        "attempt evidence.timing_seconds.engine_compile",
        "attempt evidence.timing_seconds.engine_solve",
        "attempt evidence.timing_seconds.engine_wall",
        "attempt evidence.timing_seconds.attempt_wall",
        "attempt evidence.runtime_identity.backend",
        "attempt evidence.environment.JAX_ENABLE_X64",
        "attempt evidence.environment.JAX_PLATFORMS",
        "attempt evidence.environment.XLA_PYTHON_CLIENT_PREALLOCATE",
        "attempt evidence.execution_sources.bound_modules[].relative_path",
        "attempt evidence.execution_sources.bound_modules[].sha256",
        "attempt evidence.execution_sources.bound_modules[].size_bytes",
        "attempt evidence.execution_sources.manifest.entries_sha256",
        "attempt evidence.execution_sources.manifest.entry_count",
        "attempt evidence.execution_sources.manifest.manifest_sha256",
        "attempt evidence.execution_sources.unmanifested_repository_modules[].relative_path",
        "attempt evidence.problem_identity.bound",
        "attempt evidence.problem_identity.measured_observables",
        "attempt evidence.problem_identity.reference_observables",
        "attempt evidence.problem_identity.sha_is_binding",
        "attempt evidence.lowering_pre_gate.budget_independent",
        "attempt evidence.lowering_pre_gate.certified_iterations",
        "attempt evidence.lowering_pre_gate.rehearsal_iterations",
        "attempt evidence.lowering_pre_gate.kernels[].name",
        "attempt evidence.solve.rows",
        "attempt evidence.solve.latched",
        "attempt evidence.solve.status",
        "attempt evidence.solve.status_name",
        "attempt evidence.solve.iterations_run",
        "attempt evidence.solve.maximum_feasibility_inf",
        "attempt evidence.solve.terminal_objective",
        "attempt evidence.solve.terminal_feasibility_inf",
        "attempt evidence.compilation_cache.warm",
        "attempt evidence.compilation_cache.at_entry.entry_count",
        "attempt evidence.endpoint_agreement.loop_terminal_objective",
        "attempt evidence.endpoint_agreement.standalone_terminal_objective",
        "attempt evidence.endpoint_agreement.terminal_feasibility_inf",
        "attempt evidence.endpoint_agreement.feasibility_absolute_tolerance",
        "attempt evidence.endpoint_agreement.relative_tolerance",
        "attempt evidence.endpoint_agreement.absolute_floor",
        "attempt evidence.endpoint_agreement.terminal_state_sha256",
        "gated endpoint ledger.gated_at_this_budget",
        "gated endpoint ledger.native",
        "gated endpoint ledger.terminal",
        "gated endpoint ledger.pinned_quality_terms",
        "gated endpoint ledger.native_state_sha256",
        "gated endpoint ledger.native_state_content_sha256",
        "gated endpoint ledger.pinned_term_gate.passed",
        "gated endpoint ledger.pinned_term_gate.terms",
        "gated endpoint ledger.pinned_term_gate.failed_terms",
    }
)

# The three modules the certified chain cannot run without, named by the files
# this process imported rather than by a second spelling of their paths: the
# launcher itself, the rehearsal module every shared primitive comes from, and
# the engine under certification.  A receipt whose custody block does not bind
# all three is not a receipt of this chain.
CHAIN_EXECUTION_SOURCE_MODULES: Final = (
    Path(__file__),
    Path(str(sys.modules[bind_execution_sources.__module__].__file__)),
    Path(str(sys.modules[run_projected_lbfgs.__module__].__file__)),
)
CHAIN_EXECUTION_SOURCE_PATHS: Final = frozenset(
    path.resolve(strict=True).relative_to(REPOSITORY).as_posix()
    for path in CHAIN_EXECUTION_SOURCE_MODULES
)


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
        # ALL THREE of TSL's candidates, not just ``TMPDIR``: the resolver tries
        # ``TEST_TMPDIR`` first, so overriding one name while forwarding the
        # other two leaves the rule enforced against a directory nobody used.
        **{
            name: str(temporary_directory)
            for name in TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES
        },
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

    The SUM is the certified quantity and the only one a claim may quote.  The
    halves name where a compile happened rather than separating compile from
    execution: ``engine_compile`` is the first point evaluation, and every other
    kernel -- retraction, metric apply, pair admission, Newton solve -- compiles
    on its own first call inside ``engine_solve`` (see ``ProjectedLbfgsRun``).
    No second is dropped by that boundary, so the wall this returns is complete.
    """

    timing = attempt["evidence"]["timing_seconds"]
    if not isinstance(timing, dict):
        raise ProjectedRootError("attempt publishes no timing block to derive from")
    compile_seconds = float(timing["engine_compile"])
    solve_seconds = float(timing["engine_solve"])
    # Deriving the sum constrains neither half.  A receipt publishing
    # ``engine_compile = -1e6`` beside ``engine_solve = 1e6 + 100`` derives a
    # 100 s wall exactly, and 100 s is not what either phase cost.  Both halves
    # are durations: finite and nonnegative.
    if not all(
        math.isfinite(half) and half >= 0.0 for half in (compile_seconds, solve_seconds)
    ):
        raise ProjectedRootError(
            f"attempt publishes an engine compile/solve pair that is not a pair "
            f"of durations: {compile_seconds!r} + {solve_seconds!r}"
        )
    derived = compile_seconds + solve_seconds
    if derived != float(timing["engine_wall"]):
        raise ProjectedRootError(
            f"attempt engine wall {timing['engine_wall']!r} is not its own "
            f"compile plus solve ({derived!r})"
        )
    # The supervisor timed the whole child from ``Popen`` to ``communicate``.
    # The certified wall is a strict part of that, so a receipt whose engine
    # wall exceeds the wall the supervisor observed has restated the quantity
    # the claim is judged on.
    supervised = float(attempt["supervised_seconds"])
    if not math.isfinite(supervised) or derived > supervised:
        raise ProjectedRootError(
            f"attempt engine wall {derived!r} is not within the supervised "
            f"wall {supervised!r} it is a part of"
        )
    return derived


def cold_lane_measured(cold_lane: Mapping[str, JsonValue] | None) -> bool:
    """Whether the cold lane produced the measurement section 3 pre-registers.

    One owner for the PARTITION, and for nothing beyond it -- see
    ``cold_lane_anomaly`` for what each side costs.  A lane that latched or
    missed measured the cold compile and primed the cache; every other outcome
    produced neither, and a lane that was never authorized produced neither
    either.
    """

    if cold_lane is None:
        return False
    return cold_lane["outcome"] in COLD_LANE_MEASURED_OUTCOMES


def cold_lane_anomaly(
    cold_lane: Mapping[str, JsonValue] | None,
) -> dict[str, JsonValue] | None:
    """Publish, in full, a cold lane that produced neither a latch nor a miss.

    Plan section 12.9: THE COLD LANE IS DIAGNOSTICS, NEVER DISPOSITION.  It is
    "a fourth full-budget draw that is not part of the protocol" (section 12.8's
    own words) and it is the FIRST GPU process of the session, against a cache
    that must start empty -- precisely where a first-compile timeout, an OOM or
    a bootstrap fault lands.  None of those is evidence about the claim, and the
    predicate that used to feed the conformance label could not tell them from
    evidence: it returned the same ``False`` for a lane that failed the per-term
    quality gate and for one that died on ``GATE_REFUSED:bootstrap``.  Either
    way it forced ``BOUNDED_SMOKE`` on a run that ran the pre-registered N, the
    certified budget and the lane; section 4's table disposes the resulting
    ``QUALITY_ONLY`` as ROOT SPENT; and the pair it minted beside
    ``quality_claim: CERTIFIED_BUDGET`` is one ``derive_verdict``'s own contract
    says cannot occur.

    So an anomalous lane is PUBLISHED rather than charged.  A reader of the
    sealed bytes gets the lane's outcome, the gate that refused it, its exit
    status, whether it timed out and its supervised wall -- the whole diagnosis
    -- while the disposition rests on the three timed attempts, each of which
    runs the same physics gate on its own endpoint.  ``None`` means the lane
    latched, missed, or was never authorized.
    """

    if cold_lane is None or cold_lane_measured(cold_lane):
        return None
    evidence = cold_lane["evidence"]
    return {
        "outcome": cold_lane["outcome"],
        "gate_refused": (
            evidence["gate_refused"] if isinstance(evidence, dict) else None
        ),
        "return_code": cold_lane["return_code"],
        "timed_out": cold_lane["timed_out"],
        "supervised_seconds": cold_lane["supervised_seconds"],
        "artifact_relative_path": cold_lane["artifact_relative_path"],
    }


def attempt_protocol_conformance(
    *,
    authorized_attempts: int,
    iterations: int,
    cold_lane_authorized: bool,
    attempt_timeout_seconds: float,
) -> str:
    """Whether a run IS the pre-registered protocol or a bounded smoke.

    The facts plan sections 3 and 12.2 freeze together -- N = 3, the certified
    budget, the cold lane the warm numbers are accounted against, and the
    timeout every draw is supervised under -- decide one label, in one place,
    read by the verdict and re-derived at re-validation.  A bounded smoke that
    read as a spent pre-registered protocol would drag in the successor-root
    rule of section 12.1, which applies to a root and to nothing else.

    ``cold_lane_authorized`` is whether the lane RAN, never what it produced
    (plan section 12.9).  ``--no-cold-lane`` is a real departure from the
    pre-registered protocol and demotes.  A lane that ran and then refused a
    gate is a fact about a draw the protocol does not contain, and charging it
    here labelled a fully conforming run a smoke and spent the root on an
    outcome that says nothing about the claim; ``cold_lane_anomaly`` publishes
    it instead.

    ``attempt_timeout_seconds`` is here because it was bound to NOTHING.  The
    supervised-launch gate requires a record claiming a timeout to have waited
    the timeout it publishes -- and took BOTH sides of that comparison out of
    the document being judged, which is the exact defect the certified-route
    value gate was written to retire, reintroduced in the same commit.  Roots
    carrying ``1e-9``, ``0.0`` and ``-1.0`` in this field sealed
    ``CLAIM_DISCHARGED`` / ``PREREGISTERED`` beside a lane that "timed out"
    after half a second, which erases the pre-registered cold measurement while
    keeping the pre-registered label.  ``--attempt-timeout-seconds`` is a real
    knob and an operator who moves it is running a real experiment, so this
    demotes rather than refuses -- the same shape the certified-route gate uses
    for ``maximum_iterations``, and the anchor is the frozen literal OUTSIDE the
    receipt.
    """

    preregistered = (
        authorized_attempts == PREREGISTERED_ATTEMPTS
        and iterations == CERTIFIED_MAXIMUM_ITERATIONS
        and cold_lane_authorized
        and attempt_timeout_seconds == ATTEMPT_TIMEOUT_SECONDS
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

    THE COLD LANE DOES NOT REACH HERE, and neither does its outcome (plan
    section 12.9).  The attempts this function reads are the pre-registered
    ones; the lane is a fourth draw outside the protocol, run first and against
    an empty cache, and an infrastructure fault there is a diagnosis published
    as ``cold_lane_anomaly`` rather than a disposition charged to the root.  The
    conformance label the caller passes carries whether the lane was AUTHORIZED,
    which is a pre-registration fact and not an outcome.
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
            # ``<`` and not ``<=``: mountinfo is in MOUNT order, and among
            # several mounts sharing one mount point the kernel resolves the
            # LAST.  Skipping ties reported the SHADOWED filesystem, so
            # ``mount -t tmpfs tmpfs /var/tmp/scratch`` -- the ordinary way an
            # operator hands a compile a RAM scratch directory -- was published
            # as the ext4 underneath it and passed the tmpfs refusal.
            if depth is None or depth < deepest:
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

    if not directory.is_absolute():
        raise ProjectedRootError(
            f"{role} directory {directory} is relative; the supervisor would "
            f"probe it against its own working directory while the children "
            f"resolve it against {REPOSITORY}"
        )
    if not directory.is_dir():
        raise ProjectedRootError(
            f"{role} directory is not a directory: {directory}"
            if directory.exists()
            else f"{role} directory does not exist: {directory}"
        )
    resolved = directory.resolve()
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
        # The DECLARED path is what an operator set; the RESOLVED one is what
        # the write landed in.  A symlinked temporary directory publishes two
        # different strings here, and a reader of sealed bytes needs the second
        # to know which filesystem the root actually ran under.
        "resolved_directory": str(resolved),
        "filesystem_type": kind,
        "device_id": os.stat(directory).st_dev,
        "one_byte_write": "ok",
        # Advisory only: this is the number that said 12.29 GiB free on the box
        # where the write above returned EDQUOT.
        "advisory_available_bytes": capacity.f_bavail * capacity.f_frsize,
    }


def resolve_temporary_directory(environment: Mapping[str, str]) -> Path:
    """The directory XLA will spill through, by XLA's resolution rule.

    The rule is TSL's, and it is a CANDIDATE LIST, not one name: the resolver
    tries ``TEST_TMPDIR`` first, then ``TMPDIR``, then ``TMP``, then ``/tmp``,
    and dies with ``We are not able to find a directory for temporary files.``
    if none of them takes a write.  All three names are carried by both shipped
    binaries.  Reading only ``TMPDIR`` preflighted one directory while the
    children spilled through another, under an operator shell that happened to
    hold a Bazel-ism -- section 11 requires the failure CLASS, so all three are
    resolved here and all three are overridden in the child environment.
    """

    for name in TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES:
        declared = environment.get(name, "")
        if declared.strip():
            return Path(declared)
    return DEFAULT_TEMPORARY_DIRECTORY


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
        "resolved_temporary_directory": str(temporary_directory.resolve()),
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
    attempt_timeout_seconds: float,
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
                cold_lane_authorized=cold_lane_authorized,
                attempt_timeout_seconds=attempt_timeout_seconds,
            ),
            "maximum_iterations": iterations,
            "certified_maximum_iterations": CERTIFIED_MAXIMUM_ITERATIONS,
        },
        "attempts": [dict(attempt) for attempt in attempts],
        "cold_lane": None if cold_lane is None else dict(cold_lane),
        # Diagnostics, never disposition (section 12.9): a lane that produced
        # neither a latch nor a miss is stated in full, beside a verdict derived
        # without it.
        "cold_lane_anomaly": cold_lane_anomaly(cold_lane),
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


def _validate_document_shape(
    document: JsonValue, shape: Mapping[str, object], *, where: str
) -> None:
    """Refuse a receipt block that is not, recursively and by type, the whole one.

    The frozen key sets were exactly one level deep, so a ``supervisor`` holding
    nothing but ``gpu_uuid``, an empty ``source_snapshot``, a null
    ``compilation_cache`` and an emptied ``timing_seconds`` all passed for the
    complete document they replaced: the ``frozenset`` matched at the top and no
    reader followed.  PRESENT-BUT-NULL was equivalent to absent for every block
    nothing indexed into.  Freezing the blocks moved the same hole to the LEAVES:
    a receipt whose cache accounting was three nulls, or whose ``chain_wall`` was
    the string ``"not a number"``, still published.  This walks the whole tree
    from one listing and checks every leaf against what its producer writes.
    """

    if not isinstance(document, dict):
        raise ProjectedRootError(f"{where} is not a document")
    if frozenset(document) != frozenset(shape):
        raise ProjectedRootError(
            f"{where} is incomplete: missing "
            f"{sorted(frozenset(shape) - frozenset(document))}, unexpected "
            f"{sorted(frozenset(document) - frozenset(shape))}"
        )
    for name, nested in shape.items():
        if isinstance(nested, _Dispatched):
            continue
        value = document[name]
        if isinstance(nested, _Leaf):
            _validate_leaf(value, nested, where=f"{where}.{name}")
            continue
        if isinstance(nested, tuple):
            if not isinstance(value, list):
                raise ProjectedRootError(f"{where}.{name} is not a published list")
            for index, element in enumerate(value):
                _validate_document_shape(
                    element, nested[0], where=f"{where}.{name}[{index}]"
                )
            continue
        _validate_document_shape(value, nested, where=f"{where}.{name}")


def _validate_leaf(value: JsonValue, leaf: _Leaf, *, where: str) -> None:
    """Refuse a published leaf that is not what its producer writes there.

    Named refusals, not incidental ``TypeError``s: a third party re-validating
    sealed bytes needs a sentence naming the defect, and the previous revision
    answered a nulled device inventory with ``argument of type 'NoneType' is not
    iterable`` from the first reader that happened to touch it.
    """

    if value is None:
        if leaf.nullable:
            return
        raise ProjectedRootError(
            f"{where} is null where the receipt publishes {leaf.description}"
        )
    # ``bool`` is a subclass of ``int``, so a number leaf must exclude it
    # explicitly or ``true`` passes for a count.  Named apart from the ordinary
    # type refusal because it is a different defect and because two refusal
    # sites that read identically cannot be told apart by the coverage census.
    if isinstance(value, bool) and bool not in leaf.types:
        raise ProjectedRootError(
            f"{where} is a boolean where the receipt publishes "
            f"{leaf.description}: {value!r}"
        )
    if not isinstance(value, leaf.types):
        raise ProjectedRootError(f"{where} is not {leaf.description}: {value!r}")


def _validate_preflight_record(preflight: Mapping[str, JsonValue]) -> None:
    """Re-derive the supervisor's preflight against the constants it checked.

    Ruling 6 pinned the native reference AT LOAD, in the supervisor's process,
    and left no residue a reader of sealed bytes could check: a receipt could
    restate both digests, name a path that does not exist, and publish a device
    inventory and a storage record that never happened.  Ruling 9 acquired the
    same status the moment its evidence was published rather than re-derived.
    Both are constants here, so both are compared.
    """

    if (
        preflight["native_endpoint_state_sha256"] != NATIVE_ENDPOINT_STATE_FILE_SHA256
        or preflight["native_endpoint_state_content_sha256"]
        != NATIVE_ENDPOINT_STATE_CONTENT_SHA256
        or preflight["native_endpoint_state_path"] != str(NATIVE_ENDPOINT_STATE_PATH)
    ):
        raise ProjectedRootError(
            "root preflight names a native endpoint reference other than the "
            "campaign's pinned one"
        )
    visible = preflight["visible_gpu_uuids"]
    # An INVENTORY of device UUIDs, which is what its own reason calls it: the
    # producer builds it from one column of the supervisor's device query, so
    # every element is a string.  Only membership was enforced, so a receipt
    # could publish the pinned UUID beside integers, nulls and nested documents
    # and still read as an inventory.
    if any(not isinstance(entry, str) for entry in visible):
        raise ProjectedRootError(
            f"root preflight publishes a device inventory that is not one: "
            f"{visible!r}"
        )
    if GPU_UUID not in visible:
        raise ProjectedRootError(
            f"root preflight did not see the device the claim names ({GPU_UUID!r})"
        )
    for probe in preflight["storage"]:
        if probe["filesystem_type"] in REFUSED_STORAGE_FILESYSTEM_TYPES:
            raise ProjectedRootError(
                f"root published a {probe['role']} directory on "
                f"{probe['filesystem_type']}, which plan section 11 refuses"
            )
        if probe["one_byte_write"] != "ok":
            raise ProjectedRootError(
                f"root published a {probe['role']} directory whose write probe "
                f"did not succeed"
            )
    if [probe["role"] for probe in preflight["storage"]] != [
        "temporary",
        "compilation_cache",
        "output",
    ]:
        raise ProjectedRootError(
            "root preflight did not probe the three directories the protocol writes"
        )
    # The temporary directory is published TWICE -- once as the resolved XLA
    # spill path and once as the probe that cleared it -- and a receipt naming
    # one directory beside a probe of another states nothing about the storage
    # the run used.  The two are one fact, so they are compared.
    temporary = preflight["storage"][0]
    # And it is an ABSOLUTE directory.  ``probe_writable_storage`` refuses a
    # relative one by name, in the PRODUCER -- so the refusal never reached a
    # reader of sealed bytes, and a receipt declaring ``temporary/tmp`` on all
    # four fields sealed: a directory the children spill through that no third
    # party can resolve, which is the decoupling the absoluteness rule exists to
    # stop, surviving one level below where its closure landed.
    for name in ("temporary_directory", "resolved_temporary_directory"):
        if not Path(str(preflight[name])).is_absolute():
            raise ProjectedRootError(
                f"root preflight publishes {preflight[name]!r} as the {name} the "
                f"children spill through, which no reader can resolve"
            )
    for declared, probed in (
        ("temporary_directory", "directory"),
        ("resolved_temporary_directory", "resolved_directory"),
    ):
        # Named per PAIR: reporting the declared/probed pair of the OTHER field
        # produced a refusal that reads as self-contradictory to a third party
        # ("probed '/var/tmp/temporary' and published '/var/tmp/temporary'").
        if preflight[declared] != temporary[probed]:
            raise ProjectedRootError(
                f"root preflight probed {temporary[probed]!r} and published "
                f"{preflight[declared]!r} as the {declared} the children spill "
                f"through"
            )
    # And where the reference itself is still on the box, it is RE-LOADED, which
    # re-verifies both digests against the constants above.  A reader whose box
    # no longer carries it keeps the comparison against the frozen literals; a
    # reader whose box carries a DIFFERENT file learns so here, which is the
    # whole point of ruling 6.
    if NATIVE_ENDPOINT_STATE_PATH.exists():
        load_native_endpoint_state()


def _validate_execution_sources(execution_sources: Mapping[str, JsonValue]) -> None:
    """Re-derive the module-byte custody binding against the live manifest.

    This block answers the one question a sealed source snapshot cannot: did the
    bytes the snapshot contains actually EXECUTE?  The predecessor route lost a
    one-shot root to an editable install whose meta-path finder outranked
    ``PYTHONPATH``, so production modules resolved outside the tree the run
    believed it was executing, and this is the residue of the gate that catches
    that class.  It was published on every attempt, given no shape, and read by
    nothing -- so ``execution_sources: null``, ``{}``, ``"a string"`` and
    ``{"bound_modules": []}`` all sealed as ``CLAIM_DISCHARGED`` and re-validated
    clean, and the receipt asserted nothing at all about which bytes ran.

    Re-derived, not read: the manifest evidence must be this repository's own
    manifest recomputed from its bytes, every bound module must hash to the
    entry the manifest holds for it, and the three modules the chain cannot run
    without must be among them.  A reader on another box needs this repository
    at this commit for the frozen constants already; the manifest is one of
    them.
    """

    try:
        manifest_evidence, entries = load_execution_source_manifest(REPOSITORY)
    except RehearsalError as failure:
        raise ProjectedRootError(
            f"the execution-source manifest this receipt is judged against is "
            f"not loadable: {failure}"
        ) from failure
    if execution_sources["manifest"] != manifest_evidence:
        raise ProjectedRootError(
            f"attempt names an execution-source manifest other than the "
            f"campaign's: {execution_sources['manifest']!r}"
        )
    bound = execution_sources["bound_modules"]
    if not bound:
        raise ProjectedRootError(
            "attempt binds no manifest module, so its receipt says nothing "
            "about which bytes executed"
        )
    for module in bound:
        entry = entries.get(str(module["relative_path"]))
        if (
            entry is None
            or module["sha256"] != entry["sha256"]
            or int(module["size_bytes"]) != int(entry["size_bytes"])
        ):
            raise ProjectedRootError(
                f"attempt binds {module['module']!r} to "
                f"{module['relative_path']!r} with bytes the manifest does not "
                f"describe"
            )
    executed = {str(module["relative_path"]) for module in bound}
    missing = sorted(CHAIN_EXECUTION_SOURCE_PATHS - executed)
    if missing:
        raise ProjectedRootError(
            f"attempt does not bind the modules the certified chain runs "
            f"through: {missing}"
        )
    # The escape half.  A repository module that resolved outside the manifest's
    # roots lands HERE rather than in ``bound_modules``, which is exactly the
    # class the block exists to catch, and it was shape-checked and read by
    # nothing.  A certified launch imports nothing from the tree but the three
    # manifested roots -- measured on both lanes of the bounded 5090 smoke, and
    # on CPU: ``[]``.
    unmanifested = execution_sources["unmanifested_repository_modules"]
    if unmanifested:
        raise ProjectedRootError(
            f"attempt executed repository modules the manifest does not "
            f"describe: {sorted(str(module['relative_path']) for module in unmanifested)}"
        )


def _validate_problem_identity(identity: Mapping[str, JsonValue]) -> None:
    """Re-derive the observable identity binding from its measured side.

    Plan section 2: identity is bound by observables, never by the problem sha.
    The block that records it had no shape and two read fields, so a receipt
    could publish ``{"bound": true, "sha_is_binding": false}`` and nothing else
    and re-validate clean.  ``problem_identity_evidence`` is the one owner of
    the derivation the child ran, so it is asked again here, on the published
    measurements, and the whole block must be what it returns.
    """

    if identity["sha_is_binding"]:
        raise ProjectedRootError("attempt binds identity to an unstable sha")
    measured = identity["measured_observables"]
    if not isinstance(measured, dict) or frozenset(measured) != frozenset(
        CPU_BOOTSTRAP_OBSERVABLES
    ):
        raise ProjectedRootError(
            "attempt publishes bootstrap observables other than the campaign's"
        )
    if any(not isinstance(value, float) for value in measured.values()):
        raise ProjectedRootError(
            "attempt publishes a bootstrap observable that is not a number"
        )
    derived = problem_identity_evidence(
        measured,
        problem_sha256=str(identity["recorded_problem_sha256"]),
        bootstrap_sha256=str(identity["recorded_bootstrap_sha256"]),
    )
    if identity != derived:
        raise ProjectedRootError(
            "attempt problem identity is not the one its measured observables "
            "derive"
        )
    if not identity["bound"]:
        raise ProjectedRootError("attempt claims an unbound problem")


def _validate_lowering_pre_gate(
    lowering: Mapping[str, JsonValue], *, iterations: int
) -> None:
    """Re-derive the lowering pre-gate's own accounting.

    Section 6.1's gate is that the rehearsal budget and the certified budget
    lower to IDENTICAL IR, which is what makes a bounded rehearsal stand in for
    a certified compile.  The receipt's record of it was one boolean the
    validator read; the budgets it ran at, the kernels it lowered and the size
    it reports are re-derived here.

    The KERNEL LIST was the half four reviewers refuted in one round: it was
    checked for non-emptiness and for an internal sum, so a receipt publishing a
    single invented kernel of one IR byte -- and the suite's own fixture, which
    published two kernels this repository never lowers -- were both accepted.
    WHICH kernels a configuration lowers is a function of that configuration
    (``evaluate_carried`` exists only above a projector refresh period of one,
    ``frozen_retract`` only under the frozen-projector line search,
    ``lagrangian_newton_direction`` only under the reduced-Lagrangian arm), so
    the list is re-derived against the campaign's own
    ``CERTIFIED_LOWERED_KERNEL_NAMES``.  Their SIZES are not re-derivable by a
    reader, for a reason this module used to state wrongly: the totals do NOT
    differ between processes -- three independent CPU processes at one commit
    measured 65 204 569 bytes to the byte -- they differ between COMMITS with
    the engine file byte-identical (65 207 733 one commit earlier, 65 200 869 on
    the 5090).  The total is a function of the tree, not of the process, so
    freezing it would be a false reject waiting on the next edit to any
    manifested module; which is why the substantive section 6.1 gate (identical
    IR at both budgets) runs in the child, where both sides are lowered by one
    process.
    """

    if not lowering["budget_independent"]:
        raise ProjectedRootError("attempt claims budget-dependent lowering")
    if int(lowering["certified_iterations"]) != CERTIFIED_MAXIMUM_ITERATIONS:
        raise ProjectedRootError(
            f"attempt lowered against {lowering['certified_iterations']!r} "
            f"certified iterations, not {CERTIFIED_MAXIMUM_ITERATIONS!r}"
        )
    if int(lowering["rehearsal_iterations"]) != iterations:
        raise ProjectedRootError(
            f"attempt lowered at {lowering['rehearsal_iterations']!r} "
            f"iterations, not the {iterations!r} it ran"
        )
    kernels = lowering["kernels"]
    if not kernels:
        raise ProjectedRootError("attempt lowered no kernel at all")
    lowered = sorted(str(kernel["name"]) for kernel in kernels)
    if lowered != sorted(CERTIFIED_LOWERED_KERNEL_NAMES):
        raise ProjectedRootError(
            f"attempt lowered {lowered!r}, which is not the kernel set the "
            f"certified configuration selects "
            f"({sorted(CERTIFIED_LOWERED_KERNEL_NAMES)!r})"
        )
    for kernel in kernels:
        if int(kernel["ir_bytes"]) <= 0 or int(kernel["while_operations"]) < 0:
            raise ProjectedRootError(
                f"attempt publishes kernel {kernel['name']!r} with "
                f"{kernel['ir_bytes']!r} IR bytes and "
                f"{kernel['while_operations']!r} while operations, which is not "
                f"a lowering"
            )
    total = sum(int(kernel["ir_bytes"]) for kernel in kernels)
    if int(lowering["total_ir_bytes"]) != total:
        raise ProjectedRootError(
            f"attempt lowering total {lowering['total_ir_bytes']!r} is not the "
            f"sum of its kernels ({total!r})"
        )


def _validate_certified_route_options(
    options: JsonValue, delta: JsonValue
) -> None:
    """Bind the configuration the attempt RAN to the certified route's VALUES.

    Section 1's claim is a claim about ONE route, and the campaign's whole
    substitution argument is that a bounded CPU rehearsal and the certified GPU
    run are the same configuration with one field replaced
    (``rehearsal_options`` = ``replace(CERTIFIED_ROUTE_OPTIONS,
    maximum_iterations=...)``).  The previous revision checked the KEY SET
    against the frozen dataclass and re-derived the published DELTA from the
    published options -- and then constrained the delta to nothing at all.  Both
    sides of that comparison came out of the same document, so a
    ``CLAIM_DISCHARGED`` receipt could declare ``lagrangian_newton: false``,
    ``gauss_newton: true``, ``frozen_projector_line_search: false``,
    ``backtracking_factor: 1.0`` and ``feasibility_tolerance: 1e-3`` beside a
    self-consistent delta and re-validate clean: twenty-one of the twenty-four
    fields were free, including the reduced-Lagrangian Newton--CG arm that IS
    the route under certification.  Three roles reached it from three entry
    points, and the CPU rehearsal's own suite had enforced the stronger property
    since round 1.

    So every value is compared to the frozen configuration's, and the only field
    a budget may replace is the budget.  At the certified budget the permitted
    delta is therefore EMPTY; at a bounded one it is exactly
    ``{"maximum_iterations": n}``, which is what the published label
    ``NOT_CLAIMED_AT_BOUNDED_BUDGET`` already says out loud.
    """

    fields = frozenset(CERTIFIED_ROUTE_OPTIONS.__dataclass_fields__)
    if not isinstance(options, dict) or frozenset(options) != fields:
        raise ProjectedRootError(
            "attempt options are not the certified configuration's fields"
        )
    # The one field a budget may replace is the one field the shape tree cannot
    # type, because ``options`` is the dataclass's own mapping and carries no
    # inner shape.  Every reader of it truncates, so ``maximum_iterations:
    # 700.9`` minted ``CERTIFIED_BUDGET`` and ``PREREGISTERED`` for a budget
    # describing nothing physical.  A budget is a whole number of iterations.
    budget = options["maximum_iterations"]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ProjectedRootError(
            f"attempt options publish maximum_iterations as {budget!r}, which is "
            f"not a budget"
        )
    certified = {
        field: json_scalar(getattr(CERTIFIED_ROUTE_OPTIONS, field))
        for field in fields
    }
    derived = {
        field: value
        for field, value in options.items()
        if value != certified[field]
    }
    if delta != derived:
        raise ProjectedRootError(
            f"attempt options delta {delta!r} is not the one its options derive "
            f"({derived!r})"
        )
    substituted = sorted(frozenset(derived) - {"maximum_iterations"})
    if substituted:
        raise ProjectedRootError(
            "attempt ran a route other than the certified one: "
            + ", ".join(
                f"{field}={options[field]!r} where the certified configuration "
                f"is {certified[field]!r}"
                for field in substituted
            )
        )


def _validate_solve_telemetry(solve: Mapping[str, JsonValue]) -> None:
    """Re-derive the solve summary from the iterates the same receipt publishes.

    Section 6's feasibility gate -- one of the five comparisons that decide
    whether a draw discharges the claim -- reads the summary scalar
    ``maximum_feasibility_inf``.  In the producer that scalar is ``max`` over the
    very iterates published beside it as ``rows``, so the gated number and the
    recorded evidence are one measurement told twice, and nothing compared them:
    a receipt could publish iterates carrying 0.005 and 0.027 beside
    ``maximum_feasibility_inf: 1e-14`` and seal ``CLAIM_DISCHARGED`` -- a
    nine-decade contradiction in plain sight, with a reader who does the
    arithmetic the receipt invites getting a different answer from the validator
    that accepted it.

    The identities re-derived here hold EXACTLY for the real producer -- checked
    against both lanes of a real 5090 receipt and against a live CPU solve -- so
    binding them cannot burn an honest root.  Where a nonfinite scalar makes an
    identity un-re-derivable from the published bytes (``json_scalar`` writes
    null and the raw value is gone), the check admits every reading the producer
    could have written rather than guessing one.

    THE OBJECTIVE COLUMN was left free by the revision that bound the
    feasibility one, and it is the column section 1's claim is made of: a
    receipt whose 700 recorded iterates never fell below 1.0 sealed
    ``CLAIM_DISCHARGED`` beside ``terminal_objective: 4.48e-8``,
    ``latched: true`` and ``OBJECTIVE_TARGET_REACHED`` -- the latch denied by
    the receipt's own rows by seven decades.  What is NOT true, and was proposed
    as the closure, is that a latch implies ``min(objectives) <= target``:
    MEASURED on both banked 5090 latches and on a live CPU solve, the engine
    breaks at the TOP of the loop when the current point reaches the target, so
    no recorded iterate is ever at or below it (Q1 ``4.529e-8``, Q2
    ``4.517e-8``, target ``4.482e-8``).  Gating that implication would have
    refused the campaign's own banked evidence.  The true relations are the
    reverse implication -- no recorded iterate may be at or below the target,
    because such an iterate would have ended the loop before it was recorded --
    and ADJACENCY: the terminal point is an endpoint of the LAST recorded
    iteration, either the point it opened at (a line-search collapse records its
    opening point and then breaks without advancing) or the candidate it
    accepted.  Those two endpoints are measured through a different kernel from
    the terminal re-evaluation, so they agree with it to a few ULP and never
    bitwise (measured RELATIVE: 1.3e-16 on CPU, 2.8e-14 and 1.7e-14 on the two
    5090 latches -- re-measured at 2.7547e-14 and 1.6993e-14 relative against
    this gate's rel_tol=1e-11, margins of 363x and 588x, with the absolute
    deviations 1.22e-21 and 7.61e-22 also below its abs_tol=1e-19 floor), and
    the comparison uses the campaign's own cross-executable endpoint band rather
    than a new constant -- three hundred times the worst deviation this campaign
    has measured.  THE UNIT IS STATED because it decides the gate's meaning: an
    ABSOLUTE reading of those three numbers against a 4.48e-8 objective makes
    2.8e-14 a 6.2e-7 relative deviation and implies this gate refuses both
    banked latches, which is the opposite of what was measured.

    THE SCAN ASYMMETRY IS DELIBERATE AND MUST NOT BE WIDENED.  The reverse
    implication above reads the ``objective`` column only, while adjacency
    accepts EITHER endpoint.  Extending the reverse implication to
    ``candidate_objective`` would refuse both banked 5090 latches, on which
    ``count(candidate_objective <= target) == 1`` by construction -- the last
    ACCEPTED CANDIDATE is the latch point the loop breaks on at the top of the
    next iteration, so it is the one recorded value legitimately at or below the
    target.  What these row-side checks bind is the TERMINAL scalar, against a
    frozen campaign literal outside the receipt; the rows themselves remain
    unbound to an engine trace, which the plan defers with its reasons
    (section 12.12) and adjudicates as accepted residual in section 12.13.
    """

    rows = solve["rows"]
    if int(solve["iterations_run"]) != len(rows):
        raise ProjectedRootError(
            f"attempt publishes {len(rows)} recorded iterates against "
            f"{solve['iterations_run']!r} iterations run"
        )
    for name in (
        "iterations_run",
        "stored_pairs",
        "projector_materializations",
        "tangency_forced_refreshes",
        "line_search_forced_refreshes",
    ):
        if int(solve[name]) < 0:
            raise ProjectedRootError(
                f"attempt publishes {name} as {solve[name]!r}, which is not a count"
            )
    feasibilities = _iterate_column(rows, "feasibility_inf")
    objectives = _iterate_column(rows, "objective")
    worst = solve["maximum_feasibility_inf"]
    recorded = [value for value in feasibilities if value is not None]
    if not rows:
        # ``max(..., default=nan)`` through ``json_scalar``.
        if worst is not None:
            raise ProjectedRootError(
                f"attempt publishes a worst iterate {worst!r} with no iterates"
            )
    elif len(recorded) == len(feasibilities):
        if worst != max(feasibilities):
            raise ProjectedRootError(
                f"attempt publishes a worst iterate feasibility {worst!r} that "
                f"its own recorded iterates do not carry (their maximum is "
                f"{max(feasibilities)!r})"
            )
    elif worst is not None and worst != max(recorded, default=None):
        # A nonfinite iterate publishes null: ``max`` then yields either that
        # nonfinite value (null here) or the maximum of the finite ones,
        # depending on where in the sequence it fell.  Both are honest.
        raise ProjectedRootError(
            f"attempt publishes a worst iterate feasibility {worst!r} that its "
            f"own recorded iterates do not carry"
        )
    if all(value is not None for value in objectives):
        descent = all(
            later <= earlier
            for earlier, later in zip(objectives, objectives[1:], strict=False)
        )
        if bool(solve["monotone_descent"]) != descent:
            raise ProjectedRootError(
                f"attempt publishes monotone_descent="
                f"{solve['monotone_descent']!r}, which is not what its recorded "
                f"objectives derive ({descent!r})"
            )
    try:
        status_name = ProjectedLbfgsStatus(int(solve["status"])).name
    except ValueError as failure:
        raise ProjectedRootError(
            f"attempt publishes status {solve['status']!r}, which is not one "
            f"the engine reports"
        ) from failure
    if solve["status_name"] != status_name:
        raise ProjectedRootError(
            f"attempt publishes status {solve['status']!r} under the name "
            f"{solve['status_name']!r}, which the engine calls {status_name!r}"
        )
    latched = status_name == ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED.name
    if bool(solve["latched"]) != latched:
        raise ProjectedRootError(
            f"attempt publishes latched={solve['latched']!r} under status "
            f"{solve['status_name']!r}"
        )
    target = json_scalar(CERTIFIED_ROUTE_OPTIONS.objective_target)
    reached = [
        index
        for index, value in enumerate(objectives)
        if value is not None and value <= target
    ]
    if reached:
        raise ProjectedRootError(
            f"attempt records iterate {reached[0]} at objective "
            f"{objectives[reached[0]]!r}, at or below the target "
            f"{target!r} the engine stops before recording"
        )
    if latched and not rows:
        raise ProjectedRootError(
            "attempt publishes a latch with no recorded iterate, so nothing it "
            "recorded reached the target it claims"
        )
    terminal_objective = solve["terminal_objective"]
    if rows and isinstance(terminal_objective, float):
        candidates = _iterate_column(rows, "candidate_objective")
        endpoints = [
            value for value in (objectives[-1], candidates[-1]) if value is not None
        ]
        if endpoints and not any(
            math.isclose(
                terminal_objective,
                value,
                rel_tol=DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE,
                abs_tol=DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR,
            )
            for value in endpoints
        ):
            raise ProjectedRootError(
                f"attempt publishes a terminal objective {terminal_objective!r} "
                f"that is neither endpoint of its last recorded iteration "
                f"({endpoints!r})"
            )


def _iterate_column(rows: Sequence[JsonValue], name: str) -> list[float | None]:
    """One published quantity, taken from every recorded iterate.

    The rows are the child's own per-iteration records and carry every field of
    the engine's iteration tuple, which is a function of the configuration -- so
    they are not shaped here.  The three columns the solve summary is derived
    against -- the two it is a projection of, and the accepted candidate the
    terminal endpoint is adjacent to -- are required to be there and to be
    measurements.
    """

    column: list[float | None] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProjectedRootError(f"attempt iterate {index} is not a document")
        if name not in row:
            raise ProjectedRootError(
                f"attempt iterate {index} publishes no {name}"
            )
        value = row[name]
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float))
        ):
            raise ProjectedRootError(
                f"attempt iterate {index} publishes {name} as {value!r}, which "
                f"is not a measurement"
            )
        column.append(value)
    return column


def _validate_endpoint_ledger_arithmetic(ledger: Mapping[str, JsonValue]) -> None:
    """Recompute the ledger's relative-difference column from its two sides.

    Both sides are re-gated -- the terminal against the frozen native reference,
    the native against the campaign's literals -- while the column that reports
    the distance between them was checked for its key set and never for its
    arithmetic, so every entry could read ``0.0`` beside honest sides.  It is a
    published number in a receipt a reader reads, so it is derived from the same
    owner the producer derived it from.
    """

    terminal = ledger["terminal"]
    native = ledger["native"]
    for side, rows in (("terminal", terminal), ("native", native)):
        for name, value in rows.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ProjectedRootError(
                    f"attempt endpoint ledger {side} side publishes {name} as "
                    f"{value!r}, which is not a physics measurement"
                )
    if frozenset(terminal) != frozenset(native):
        raise ProjectedRootError(
            "attempt endpoint ledger sides do not carry the same terms"
        )
    derived = endpoint_relative_differences(terminal, native)
    if ledger["relative_difference"] != derived:
        raise ProjectedRootError(
            "attempt endpoint ledger relative differences are not the ones its "
            "two sides derive"
        )


def _validate_terminal_endpoint_column(evidence: Mapping[str, JsonValue]) -> None:
    """Make the receipt's four tellings of the terminal endpoint ONE fact.

    The scalar the latch gate reads -- ``solve.terminal_objective`` -- was
    re-derived from nothing, while the receipt states the same measurement three
    more times and two of those tellings are EXACT copies in the producer.  So a
    receipt could seal ``CLAIM_DISCHARGED`` with ``terminal_objective:
    4.48e-30`` beside an endpoint ledger holding the native weighted total, or
    with an agreement block whose two halves agreed with each other to 5e-16 and
    with nothing else in the run.

    The producer's identities, measured at these bytes through the canonical
    round trip the receipt takes:

    * ``solve.terminal_objective`` is ``json_scalar(run.objective)`` and
      ``endpoint_agreement.loop_terminal_objective`` is ``run.objective`` -- the
      same float, and finite on any completed chain, because
      ``certify_endpoint_agreement`` refuses a nonfinite pair by name before the
      chain can complete.
    * ``solve.terminal_feasibility_inf`` and
      ``endpoint_agreement.terminal_feasibility_inf`` are both
      ``run.feasibility_inf``, and the same refusal makes them finite.
    * ``endpoint_agreement.standalone_terminal_objective`` and the ledger's
      ``terminal.weighted_total`` are both
      ``float(case.standalone_evaluation(run.coordinates).weighted_total)``,
      evaluated twice in one process on one input -- bitwise equal, measured.

    That last one is the whole point: it is what puts an ANCHOR OUTSIDE THE
    RECEIPT under the claim's headline number.  ``weighted_total`` is a pinned
    quality term, so on the attempt that discharges the claim
    ``gate_endpoint_ledger_against_frozen_native`` judges it against the
    campaign's frozen native literal, and the chain
    ``terminal_objective -> loop -> (campaign band) -> standalone -> ledger
    terminal -> frozen literal`` leaves the forger nothing free to move.

    The feasibility half is closed the same way and then against the route's own
    frozen tolerance, which is the child's own gate re-derived
    (``certify_endpoint_agreement`` refuses a terminal feasibility outside it):
    a receipt publishing ``terminal_feasibility_inf: 0.99`` beside
    ``feasibility_absolute_tolerance: 1.0`` is refused rather than sealed.
    """

    solve = evidence["solve"]
    endpoint = evidence["endpoint_agreement"]
    tolerance = CERTIFIED_ROUTE_OPTIONS.feasibility_tolerance
    if endpoint["feasibility_absolute_tolerance"] != tolerance:
        raise ProjectedRootError(
            f"attempt states its terminal feasibility against "
            f"{endpoint['feasibility_absolute_tolerance']!r}, not the certified "
            f"route's {tolerance!r}"
        )
    for name, published, agreed in (
        (
            "terminal objective",
            solve["terminal_objective"],
            endpoint["loop_terminal_objective"],
        ),
        (
            "terminal feasibility",
            solve["terminal_feasibility_inf"],
            endpoint["terminal_feasibility_inf"],
        ),
    ):
        if not isinstance(published, float):
            raise ProjectedRootError(
                f"attempt publishes a completed chain whose {name} is "
                f"{published!r}, which no chain that cleared the endpoint "
                f"agreement can carry"
            )
        if published != agreed:
            raise ProjectedRootError(
                f"attempt publishes a {name} of {published!r} in its solve "
                f"summary and {agreed!r} in its endpoint agreement, which are "
                f"one measurement told twice"
            )
    if not endpoint["terminal_feasibility_inf"] <= tolerance:
        raise ProjectedRootError(
            f"attempt publishes a terminal feasibility "
            f"{endpoint['terminal_feasibility_inf']!r} outside the certified "
            f"route's {tolerance!r}"
        )
    ledger_terminal = evidence["endpoint_ledger"]["terminal"]["weighted_total"]
    if endpoint["standalone_terminal_objective"] != ledger_terminal:
        raise ProjectedRootError(
            f"attempt publishes a standalone terminal objective "
            f"{endpoint['standalone_terminal_objective']!r} beside an endpoint "
            f"ledger whose terminal weighted total is {ledger_terminal!r}, "
            f"which is the same evaluation of the same state"
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
    COMPLETE, RECURSIVELY AND BY TYPE, because a truncated document could not be
    told from a whole one: every block section 6 builds is a node of ONE frozen
    shape tree whose keys ARE the required key sets, and every leaf of that tree
    states what its producer writes there (``_validate_document_shape``).  The
    previous revision said this while ``execution_sources`` -- the module-byte
    custody binding -- had no shape and no reader at all, which is the sentence
    this revision had to make true rather than restate.

    A shape is not a value, and the revision that froze the whole tree left the
    values inside it free: the options block was key-checked and its delta
    re-derived from itself, so a receipt could discharge the claim for a
    DIFFERENT optimizer configuration; the feasibility gate read a summary
    scalar its own published iterates contradicted by nine decades; and the
    cold lane's pre-registration fact was bound to a directory entry.  Each of
    those is now a comparison against something outside the document being
    judged -- the campaign's frozen configuration, the receipt's own recorded
    iterates, and the invocation and cache of a draw nobody else took.

    The reference the physics gate is judged against is the CAMPAIGN's, never
    the artifact's.  The published native side is compared to
    ``NATIVE_ENDPOINT_PINNED_TERMS`` term by term, the gate is recomputed from
    those literals, the two digests of ruling 6 are compared to the frozen
    constants wherever the receipt states them, and the reference file itself is
    re-loaded -- digests verified -- when this box still has it.

    Re-validation is FLOATING-POINT WORK, so it asserts its own precision: the
    terminal-state digest is re-derived through ``jnp.asarray(..., float64)``,
    which silently yields float32 with x64 disabled and refuses a valid root
    with a message indicting the artifact rather than the reader's environment.
    """

    if not jax.config.jax_enable_x64:
        raise ProjectedRootError(
            "re-validation requires JAX_ENABLE_X64=true: the published terminal "
            "state is a float64 array, and with x64 disabled its digest is "
            "re-derived at float32 and disagrees with every honest receipt"
        )
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
    _validate_document_shape(evidence, ROOT_EVIDENCE_SHAPE, where="root")
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
    _validate_preflight_record(evidence["supervisor"]["preflight"])

    # The block's own key set and leaf types were walked above, with the rest of
    # the tree, from the one listing -- a second check here would be the second
    # enumeration this structure exists to remove.
    protocol = evidence["attempt_protocol"]
    cold = evidence["cold_lane"]
    if bool(protocol["cold_lane_authorized"]) != (cold is not None):
        raise ProjectedRootError(
            "root cold-lane authorization does not match the lane it published"
        )
    # ``cold_lane_authorized`` is the lane's ONLY channel to the conformance
    # label, and therefore to the headline verdict, and the only check on it was
    # that a record existed somewhere in the receipt.  A record whose path
    # pointed at ``attempts/attempt-1`` was validated against attempt 1's own
    # sealed array and passed, so a root with no cold lane in the tree at all
    # minted ``PREREGISTERED`` and ``CLAIM_DISCHARGED``.  The pre-registration
    # fact is bound to the directory the lane runs in: the supervisor creates it
    # before it launches the lane, so it exists for every outcome the lane has.
    cold_lane_directory = artifact_root / COLD_LANE_DIRECTORY
    if cold is not None and not isinstance(cold, dict):
        raise ProjectedRootError("root cold-lane record is not a document")
    if cold is not None and cold["artifact_relative_path"] != COLD_LANE_DIRECTORY:
        raise ProjectedRootError(
            f"root publishes its cold lane at "
            f"{cold['artifact_relative_path']!r}, not in the "
            f"{COLD_LANE_DIRECTORY!r} directory the protocol runs it in"
        )
    if cold is not None and int(cold["attempt_index"]) != 0:
        raise ProjectedRootError(
            f"root publishes a cold lane indexed {cold['attempt_index']!r} "
            f"among the protocol's own draws"
        )
    if cold_lane_directory.is_dir() != (cold is not None):
        raise ProjectedRootError(
            f"root claims cold_lane_authorized="
            f"{bool(protocol['cold_lane_authorized'])!r} against a tree that "
            f"{'carries' if cold_lane_directory.is_dir() else 'carries no'} "
            f"{COLD_LANE_DIRECTORY!r} directory"
        )
    anomaly = evidence["cold_lane_anomaly"]
    if anomaly is not None:
        _validate_document_shape(
            anomaly, COLD_LANE_ANOMALY_SHAPE, where="root.cold_lane_anomaly"
        )
    attempts = evidence["attempts"]
    if not isinstance(attempts, list):
        raise ProjectedRootError("root publishes no list of attempts")
    for attempt in attempts:
        _validate_attempt_shape(attempt, cold_lane=False)
    if cold is not None:
        _validate_attempt_shape(cold, cold_lane=True)
    for attempt in attempts:
        _validate_attempt_outcome(attempt)
    if cold is not None:
        _validate_attempt_outcome(cold)
    # Every draw the receipt publishes is one child this supervisor launched,
    # timed and sampled, and no two draws are the same launch.
    timeout_seconds = float(evidence["supervisor"]["attempt_timeout_seconds"])
    for index, attempt in enumerate(attempts):
        _validate_supervised_launch(
            attempt, timeout_seconds=timeout_seconds, where=f"attempt {index + 1}"
        )
    if cold is not None:
        _validate_supervised_launch(
            cold, timeout_seconds=timeout_seconds, where="the cold lane"
        )
    invocations = [str(attempt["argv_sha256"]) for attempt in attempts]
    if len(frozenset(invocations)) != len(invocations):
        raise ProjectedRootError(
            "root publishes two attempts launched by the same invocation, which "
            "the protocol's own per-attempt directory and index make impossible"
        )

    # Section 4's draw statistics were pure read-backs beside a conformance
    # label that was re-derived, so a one-attempt root could publish
    # ``latch_rate: 3/3`` and ``attempts_run: 7`` and re-validate clean.  Every
    # one of them is a function of the attempts and the frozen constants.
    conformance = attempt_protocol_conformance(
        authorized_attempts=int(protocol["authorized_attempts"]),
        iterations=int(protocol["maximum_iterations"]),
        cold_lane_authorized=bool(protocol["cold_lane_authorized"]),
        attempt_timeout_seconds=timeout_seconds,
    )
    # The whole chain contains every draw it published: the lane and the timed
    # attempts run sequentially inside one supervised session, so the root's own
    # wall is at least their sum.  It was read by nothing, so a receipt could
    # publish ``chain_wall: -1e9`` or ``1e-300`` beside walls that add to
    # minutes -- the residue of the attempt-level timing chain on the one
    # measurement that gate did not reach.
    chain_wall = float(evidence["timing_seconds"]["chain_wall"])
    supervised_total = sum(
        float(draw["supervised_seconds"])
        for draw in (*attempts, *(() if cold is None else (cold,)))
    )
    if not math.isfinite(chain_wall) or chain_wall < supervised_total:
        raise ProjectedRootError(
            f"root publishes a chain wall of "
            f"{evidence['timing_seconds']['chain_wall']!r} s around draws it "
            f"supervised for {supervised_total!r} s"
        )
    # Section 12.9: the lane's own outcome is published, in full, and reaches
    # nothing else.  Re-derived here so a receipt cannot hide an anomalous lane
    # behind a null.
    derived_anomaly = cold_lane_anomaly(cold)
    if evidence["cold_lane_anomaly"] != derived_anomaly:
        raise ProjectedRootError(
            f"published cold-lane anomaly {evidence['cold_lane_anomaly']!r} is "
            f"not the one its lane derives ({derived_anomaly!r})"
        )
    # The published stop rule was a STRING, re-derived from a constant, while
    # the attempt SEQUENCE it describes was unconstrained: a receipt could carry
    # three ``LATCHED`` attempts and ``latch_rate: 3/3`` on a loop that breaks
    # after the first.  The rule is enforced against the list it names.
    if len(attempts) > int(protocol["authorized_attempts"]):
        raise ProjectedRootError(
            f"root published {len(attempts)} attempts under "
            f"{protocol['authorized_attempts']!r} authorized"
        )
    for index, attempt in enumerate(attempts[:-1]):
        if attempt["outcome"] != "COMPLETED_WITHOUT_LATCH":
            raise ProjectedRootError(
                f"attempt {index + 1} outcome {attempt['outcome']!r} does not "
                f"obey the published stop rule: {ATTEMPT_STOP_RULE}"
            )
    expected_paths = [
        f"{ATTEMPTS_DIRECTORY}/attempt-{index}"
        for index in range(1, len(attempts) + 1)
    ]
    if [attempt["artifact_relative_path"] for attempt in attempts] != expected_paths or (
        [int(attempt["attempt_index"]) for attempt in attempts]
        != list(range(1, len(attempts) + 1))
    ):
        raise ProjectedRootError(
            "root attempts are not the protocol's consecutive draws, each in its "
            "own directory"
        )
    # Every draw ON DISK has to be a draw in the receipt.  The loop writes an
    # attempt directory before it launches the child, so a tree carrying one the
    # attempt list does not claim is a suppressed draw.
    attempts_directory = artifact_root / ATTEMPTS_DIRECTORY
    unclaimed = sorted(
        entry.name
        for entry in (
            attempts_directory.iterdir() if attempts_directory.is_dir() else ()
        )
        if entry.is_dir() and f"{ATTEMPTS_DIRECTORY}/{entry.name}" not in expected_paths
    )
    if unclaimed:
        raise ProjectedRootError(
            f"the artifact tree carries attempt directories the receipt does "
            f"not publish: {unclaimed}"
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
        _validate_cold_lane_draw(cold, attempts)
        _validate_cold_lane(artifact_root, cold, protocol)
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

    if not isinstance(attempt, dict):
        raise ProjectedRootError("supervised attempt record is not a document")
    record_shape = COLD_LANE_SHAPE if cold_lane else SUPERVISED_ATTEMPT_SHAPE
    required = frozenset(record_shape)
    if frozenset(attempt) != required:
        raise ProjectedRootError(
            "supervised attempt record is incomplete: missing "
            f"{sorted(required - frozenset(attempt))}, unexpected "
            f"{sorted(frozenset(attempt) - required)}"
        )
    _validate_document_shape(attempt, record_shape, where="supervised attempt")
    evidence = attempt["evidence"]
    if evidence is None:
        return
    if not isinstance(evidence, dict):
        raise ProjectedRootError("attempt evidence is not a document")
    expected_shape = (
        REFUSED_ATTEMPT_EVIDENCE_SHAPE
        if evidence.get("gate_refused") is not None
        else ATTEMPT_EVIDENCE_SHAPE
    )
    expected = frozenset(expected_shape)
    if frozenset(evidence) != expected:
        raise ProjectedRootError(
            "attempt evidence document is incomplete: missing "
            f"{sorted(expected - frozenset(evidence))}, unexpected "
            f"{sorted(frozenset(evidence) - expected)}"
        )
    _validate_document_shape(evidence, expected_shape, where="attempt evidence")
    if evidence.get("gate_refused") is not None:
        return
    _validate_document_shape(
        evidence["endpoint_ledger"],
        (
            GATED_ENDPOINT_LEDGER_SHAPE
            if evidence["endpoint_ledger"].get("gated_at_this_budget")
            else ENDPOINT_LEDGER_SHAPE
        )
        if isinstance(evidence["endpoint_ledger"], dict)
        else ENDPOINT_LEDGER_SHAPE,
        where="attempt evidence.endpoint_ledger",
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


def _validate_supervised_launch(
    record: Mapping[str, JsonValue], *, timeout_seconds: float, where: str
) -> None:
    """Re-derive that this record is one CHILD LAUNCH the supervisor observed.

    Every draw in the receipt -- the three timed attempts and the cold lane --
    is a process the supervisor started, timed and sampled, and the record
    carries three independent traces of that: the digest of the argv it was
    launched with, the sampler's digest of the argv it OBSERVED on the device,
    and the supervised wall.  None of the three was read, so a record could be
    a copy of another draw, could name a device the claim is not stated for, or
    could claim a timeout it did not wait for.  The identities are exact for the
    real producer (verified on both lanes of a real 5090 receipt): the sampler
    is handed the same argv the child was launched with and refuses to bind a
    child whose procfs argv differs, and ``communicate(timeout=...)`` cannot
    raise before its timeout elapses.
    """

    memory = record["gpu_memory"]
    if memory["child_argv_sha256"] != record["argv_sha256"]:
        raise ProjectedRootError(
            f"{where} publishes device telemetry for a child other than the one "
            f"it launched"
        )
    if memory["device_uuid"] != GPU_UUID:
        raise ProjectedRootError(
            f"{where} was observed on GPU {memory['device_uuid']!r}, not the "
            f"device the claim is stated for ({GPU_UUID!r})"
        )
    if int(memory["child_pid"]) == int(memory["parent_pid"]):
        raise ProjectedRootError(
            f"{where} names the supervisor as its own child process"
        )
    supervised = float(record["supervised_seconds"])
    if not math.isfinite(supervised) or supervised <= 0.0:
        raise ProjectedRootError(
            f"{where} publishes a supervised wall of {supervised!r}, which is "
            f"not a duration"
        )
    if bool(record["timed_out"]) and supervised < float(timeout_seconds):
        raise ProjectedRootError(
            f"{where} claims a timeout after {supervised!r} s under the "
            f"{float(timeout_seconds)!r} s timeout it publishes"
        )


def _validate_attempt_record(
    artifact_root: Path,
    attempt: Mapping[str, JsonValue],
    protocol: Mapping[str, JsonValue],
) -> Mapping[str, JsonValue] | None:
    """Re-derive WHICH RUN produced this record, and what its own bytes say.

    Everything here is a fact about the record: the context the child ran in,
    the custody of the bytes that ran, the ROUTE it ran (every option value
    against the campaign's frozen configuration), the budget it ran at, the
    warmth of the cache it entered, the nesting of its three walls, the
    arithmetic of its own published columns -- including the solve summary
    re-derived from the iterates published beside it -- and the array behind its
    terminal-state hash.
    None of it is a comparison against another executable, so an honest draw
    cannot fail any of it -- which is what makes it the part BOTH lanes run.
    The claim-bearing gates live in ``_validate_attempt``, and the cold lane
    (plan section 12.9: diagnostics, never disposition) does not run them.

    Returns the child's completed document, or ``None`` when the record carries
    no chain to check -- a timeout, a protocol failure, or a named gate refusal.
    """

    if attempt["outcome"] in {"TIMEOUT", "PROTOCOL_FAILURE"}:
        return None
    evidence = attempt["evidence"]
    if not isinstance(evidence, dict):
        raise ProjectedRootError("attempt carries no evidence document")
    if evidence["gate_refused"] is not None:
        return None
    # The EXECUTION CONTEXT the attempt declares, re-derived rather than merely
    # published.  The root's own schema, route and timing boundary were checked;
    # the attempt's were not, so a ``CLAIM_DISCHARGED`` receipt could name the
    # CPU as the backend that produced the certified wall, state another timing
    # boundary, or carry another route's schema.
    if (
        evidence["schema_version"] != GPU_ATTEMPT_SCHEMA_VERSION
        or evidence["route"] != PROJECTED_ROUTE
        or evidence["timing_boundary"] != "engine_compile_plus_solve"
        or int(evidence["attempt_index"]) != int(attempt["attempt_index"])
    ):
        raise ProjectedRootError(
            "attempt evidence describes a different run than the record carrying it"
        )
    if evidence["runtime_identity"]["backend"] != REQUIRED_BACKEND:
        raise ProjectedRootError(
            f"attempt names backend {evidence['runtime_identity']['backend']!r}, "
            f"not the {REQUIRED_BACKEND!r} the wall is claimed on"
        )
    if any(
        evidence["environment"][name] != value
        for name, value in GPU_REQUIRED_ENVIRONMENT.items()
    ):
        raise ProjectedRootError("attempt ran under an environment the route forbids")
    # The wall of EVERY attempt, not only the first latching one: section 4
    # makes each attempt's wall part of the artifact, and ``derive_verdict``
    # reaches ``attempt_engine_wall_seconds`` on exactly one of them.
    engine_wall = attempt_engine_wall_seconds(attempt)
    # And the wall is a CHAIN: the engine's compile plus solve sits inside the
    # attempt's own wall, which sits inside the wall the supervisor observed.
    # Only the outer relation was checked, so a receipt could publish
    # ``attempt_wall: 1e-9`` beside a 187 s engine and three negative phase
    # durations.  Each phase is a duration of this run and the three
    # measurements nest, by construction and by measurement on both lanes.
    timing = evidence["timing_seconds"]
    for phase in ATTEMPT_TIMING_SHAPE:
        seconds = float(timing[phase])
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ProjectedRootError(
                f"attempt publishes {phase} as {timing[phase]!r}, which is not "
                f"a duration"
            )
    if not engine_wall <= float(timing["attempt_wall"]) <= float(
        attempt["supervised_seconds"]
    ):
        raise ProjectedRootError(
            f"attempt timings do not nest: engine wall {engine_wall!r}, attempt "
            f"wall {timing['attempt_wall']!r}, supervised wall "
            f"{attempt['supervised_seconds']!r}"
        )
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
    # "Warm" is what the cold lane's whole accounting turns on, and it was a
    # published boolean beside the cache state it is a function of.  The
    # producer derives it from the entry count it sampled before this process
    # had traced anything, so it is derived from the same number here.
    cache = evidence["compilation_cache"]
    if bool(cache["warm"]) != (int(cache["at_entry"]["entry_count"]) > 0):
        raise ProjectedRootError(
            f"attempt publishes warm={cache['warm']!r} against a cache holding "
            f"{cache['at_entry']['entry_count']!r} entries at entry"
        )
    # The three custody blocks: which bytes ran, which problem they ran on, and
    # which kernels they lowered.  All three were published on every attempt and
    # reached by nothing -- ``execution_sources`` by no reader at all -- so each
    # is re-derived against the campaign's own authorities.
    _validate_execution_sources(evidence["execution_sources"])
    _validate_problem_identity(evidence["problem_identity"])
    _validate_lowering_pre_gate(
        evidence["lowering_pre_gate"],
        iterations=int(evidence["options"]["maximum_iterations"]),
    )

    # Substitution soundness rests on "same route": every budget is the frozen
    # configuration with one field replaced.  Both the delta and every VALUE it
    # is derived from are bound to the campaign's frozen object.  This runs
    # FIRST of the two: its opening act is the field-set check, and reading a
    # named option out of the block before that check answered a truncated
    # options block with a bare ``KeyError`` instead of the refusal that names
    # the defect.
    _validate_certified_route_options(
        evidence["options"], evidence["certified_options_delta"]
    )
    # The claim's quality quantity is a NUMBER, not a status code.  ``LATCHED``
    # is the optimizer reporting that some iterate fell to its own configured
    # ``objective_target``, so an attempt configured against a different target
    # would publish a latch that discharges a different claim.  Both halves are
    # checked here: the target the run used, and the objective it reached.
    if evidence["options"]["objective_target"] != NATIVE_TARGET_OBJECTIVE:
        raise ProjectedRootError(
            "attempt targets an objective other than the native endpoint"
        )
    # The solve summary, re-derived from the iterates published beside it.
    _validate_solve_telemetry(evidence["solve"])
    if attempt["outcome"] == "LATCHED":
        terminal_objective = evidence["solve"]["terminal_objective"]
        # ``json_scalar`` writes null for a nonfinite terminal objective, and a
        # null reached ``>`` as an unnamed ``TypeError`` from the one nullable
        # numeric leaf this function reads.
        if not isinstance(terminal_objective, float) or (
            terminal_objective > NATIVE_TARGET_OBJECTIVE
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
    # Nor may it supply the REFERENCE those terms are judged against.  Both
    # sides of the gate used to come out of the document being judged, so a
    # ledger publishing ``terminal == native == 1.0`` on all ten pinned terms
    # recomputed every verdict to ``measured = 0.0, passed = true`` and sealed
    # as ``CLAIM_DISCHARGED`` -- ruling 7 bound whether the gate RAN and whether
    # it PASSED, and not what it ran against.  The native side is the campaign's
    # frozen evaluation of the digest-pinned reference, and the digests the
    # ledger names are ruling 6's constants.
    if (
        ledger["native_state_sha256"] != NATIVE_ENDPOINT_STATE_FILE_SHA256
        or ledger["native_state_content_sha256"] != NATIVE_ENDPOINT_STATE_CONTENT_SHA256
        or ledger["native_state_relative_path"] != NATIVE_ENDPOINT_STATE_PATH.name
    ):
        raise ProjectedRootError(
            "attempt endpoint ledger names another native endpoint reference"
        )
    for side in ("terminal", "native", "relative_difference"):
        if not isinstance(ledger[side], dict) or not frozenset(ledger[side]) >= (
            frozenset(PINNED_ENDPOINT_QUALITY_TERMS)
            | frozenset(INFORMATIONAL_ENDPOINT_OBSERVABLES)
        ):
            raise ProjectedRootError(
                f"attempt endpoint ledger {side} side does not carry the "
                f"campaign's terms"
            )
    _validate_endpoint_ledger_arithmetic(ledger)
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
    if gated and gate_endpoint_ledger(ledger) != ledger["pinned_term_gate"]:
        raise ProjectedRootError(
            "attempt pinned-term gate is not the one its ledger derives"
        )
    endpoint = evidence["endpoint_agreement"]
    if (
        endpoint["relative_tolerance"] != DIAG4_ENDPOINT_AGREEMENT_RELATIVE_TOLERANCE
        or endpoint["absolute_floor"] != DIAG4_ENDPOINT_AGREEMENT_ABSOLUTE_FLOOR
    ):
        raise ProjectedRootError("attempt endpoint tolerances differ from the campaign's")
    # The four tellings of the terminal endpoint, made one fact.  Runs after the
    # ledger's own sides are known to be physics measurements carrying the
    # campaign's terms, because it reads one of them.
    _validate_terminal_endpoint_column(evidence)

    coordinates_path = (
        artifact_root
        / str(attempt["artifact_relative_path"])
        / TERMINAL_COORDINATES_FILENAME
    )
    if not coordinates_path.is_file():
        raise ProjectedRootError(
            f"attempt published a completed chain whose directory "
            f"{str(attempt['artifact_relative_path'])!r} carries no "
            f"{TERMINAL_COORDINATES_FILENAME}"
        )
    with coordinates_path.open("rb") as stream:
        coordinates = np.load(stream, allow_pickle=False)
    republished = exact_numeric_tree_sha256(
        jnp.asarray(coordinates, dtype=jnp.float64)
    )
    if republished != endpoint["terminal_state_sha256"]:
        raise ProjectedRootError("published terminal state differs from its hash")
    return evidence


def _validate_attempt(
    artifact_root: Path,
    attempt: Mapping[str, JsonValue],
    protocol: Mapping[str, JsonValue],
) -> None:
    """Re-derive one PRE-REGISTERED attempt, including the claim it bears.

    The record's own facts first, then the five comparisons that decide whether
    this draw discharges section 1's claim: its native side against the
    campaign's frozen reference, its per-term gate's verdict, that gate
    recomputed FROM the frozen literals, its two independently compiled
    endpoints against each other, and its worst iterate against the feasibility
    bound.  Each of these compares two independently compiled executables or
    judges the physics, so each is a gate on the CLAIM -- which is why the cold
    lane does not run them.
    """

    evidence = _validate_attempt_record(artifact_root, attempt, protocol)
    if evidence is None:
        return
    ledger = evidence["endpoint_ledger"]
    certify_native_reference(ledger["native"])
    if bool(ledger["gated_at_this_budget"]):
        # Equality of a faithfully recorded FAILURE with its own recomputation
        # is a consistency check, not a quality gate.  Section 1.1 defines
        # quality parity as this verdict passing, so a discharging endpoint that
        # failed it may not be sealed as one that did.
        if not ledger["pinned_term_gate"]["passed"]:
            raise ProjectedRootError(
                "attempt discharges the claim with a failed pinned-term gate: "
                f"{ledger['pinned_term_gate']['failed_terms']}"
            )
        # And the gate that DECIDES is recomputed against the frozen reference,
        # not against the ledger's own native side.  Bitwise agreement between
        # the two recomputations is neither required nor possible -- the
        # literals are this repository's CPU evaluation, the receipt's are the
        # GPU's -- so what is demanded is that the frozen-reference gate passes.
        frozen = gate_endpoint_ledger_against_frozen_native(ledger)
        if not frozen["passed"]:
            raise ProjectedRootError(
                "attempt endpoint differs from the campaign's frozen native "
                f"reference on {frozen['failed_terms']}"
            )
    endpoint = evidence["endpoint_agreement"]
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


def _validate_cold_lane(
    artifact_root: Path,
    cold: Mapping[str, JsonValue],
    protocol: Mapping[str, JsonValue],
) -> None:
    """Validate the cold lane for SHAPE and HONESTY, and for nothing else.

    Plan section 12.9 ruling 13 took the lane out of the conformance label and
    out of ``derive_verdict`` -- "what changes is that it decides nothing" --
    and left it running the FULL discharging-attempt validation inside
    ``publish_root``.  Executed, it still decided the largest thing there is: a
    lane that latched and missed one pinned band, or that merely missed the
    latch with one infeasible recorded iterate, raised inside the publication
    gate, so the run wrote a refusal record and NO ARTIFACT AT ALL -- after the
    cold lane and all three timed attempts had been spent.  That is strictly
    worse than the ``QUALITY_ONLY`` cap ruling 13 reversed ruling 8 to avoid,
    and it is reachable on the sanctioned path by an ordinary unlucky draw: the
    lane is a fourth full-budget draw at the same budget, run first, against an
    empty cache, and the campaign's own measured miss rate is one in five.

    So the lane is validated for what it IS -- a complete, honest record of a
    draw this protocol ran, whose bytes, budget, problem, arithmetic and sealed
    array are its own -- and never for what it SAYS about the claim.  A lane
    that fails a claim-bearing gate is not silently accepted: it either refuses
    in the child, which publishes it as ``cold_lane_anomaly`` with the gate
    named, or it publishes its whole ledger for a reader to recompute, beside
    three timed attempts each of which runs every gate on its own endpoint.
    """

    _validate_attempt_record(artifact_root, cold, protocol)


def _validate_cold_lane_draw(
    cold: Mapping[str, JsonValue], attempts: Sequence[Mapping[str, JsonValue]]
) -> None:
    """Re-derive that the published lane is a DRAW OF ITS OWN, not a retelling.

    Ruling 18 bound the lane's path, its index and the EXISTENCE of the
    directory it runs in.  A forger paid one ``mkdir``: an empty ``cold-lane/``
    beside a record that produced nothing, and a ``cold-lane/`` holding a
    byte-copy of ``attempts/attempt-1`` beside a deep copy of attempt 1's own
    record, both minted ``PREREGISTERED`` and therefore the headline verdict.

    What separates a draw from a copy is not its terminal state -- two honest
    draws of the same problem at the same budget produced BITWISE IDENTICAL
    endpoints on the 5090 (measured: both lanes of the bounded smoke report the
    same worst iterate to the last digit), so demanding difference there would
    burn an honest root.  It is the INVOCATION and the CACHE: the lane is
    launched at index 0 into its own directory, so its argv digest is not any
    timed attempt's, and it is the session's first process against a cache the
    protocol required to be empty, so its own accounting must say so.

    Ruling 17 is preserved: a lane that timed out, failed the protocol or
    refused a gate publishes with its anomaly recorded, and nothing here reads
    the lane's outcome.  What is refused is a lane that claims a draw it did not
    take.
    """

    invocations = {str(attempt["argv_sha256"]) for attempt in attempts}
    if str(cold["argv_sha256"]) in invocations:
        raise ProjectedRootError(
            "root publishes a cold lane whose invocation is a timed attempt's, "
            "so its record is a copy of a draw rather than a draw"
        )
    evidence = cold["evidence"]
    if not isinstance(evidence, dict) or evidence["gate_refused"] is not None:
        return
    # The cold lane is what makes the cache an accounting device rather than a
    # hiding place, and it can only say what a compile costs without a cache if
    # it ran without one.  ``warm`` is re-derived from the entry count the lane
    # sampled before it traced anything (``_validate_attempt_record``), so this
    # is a statement about the cache and not about a published boolean.
    cache = evidence["compilation_cache"]
    if cache["warm"]:
        raise ProjectedRootError(
            f"the cold lane ran against a populated cache "
            f"({cache['at_entry']['entry_count']!r} entries at entry)"
        )


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

    # Both BEFORE the staging tree exists, so a refusal costs zero compute --
    # which is what makes refusing a temporary directory on tmpfs the cheap half
    # of the trade against a ``GATE_REFUSED:bootstrap`` on a spent root.  The
    # backend stays first: a process that resolved to the CPU is not a launch of
    # this protocol at all, and it is the cheapest fact to establish.
    #
    # "Leaves the filesystem exactly as it found it" is APPROXIMATE, and stated
    # here rather than claimed: the ``mkdir`` above creates ``--cache-dir`` if
    # it is absent (a re-run finds it empty and passes the check above it), and
    # ``bind_gpu_backend`` initialises a CUDA context under whatever storage the
    # operator had, before that storage has been probed.  Neither costs the
    # root; neither is nothing.
    #
    # That context then lives for the whole protocol, alongside up to four
    # full-budget GPU children, and it is ACCEPTED rather than overlooked.  The
    # supervisor's own artifact says so in its bytes (``supervisor_payload``
    # publishes ``gpu_zero_asserted = False``).  It stays in the parent because
    # a silent CPU resolution has to be refused BEFORE any child spends compute
    # or any tree is staged, and a refusal delegated to a child would arrive
    # after the launch it was meant to prevent -- ``nvidia-smi`` cannot stand in
    # for it, since the fact being established is which backend THIS
    # interpreter resolved, not which device the box has.  What bounds the cost
    # is that the children inherit this process's environment and refuse unless
    # it carries ``XLA_PYTHON_CLIENT_PREALLOCATE=false``, so every launch that
    # reaches a receipt had the parent's own context non-preallocating too, and
    # every launch that did not is refused at the first child.  Moving the
    # binding into a disposable probe child would trade that bounded cost for a
    # process launch inside the refusal window, and is deferred to a successor
    # plan rather than taken on a frozen protocol.
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
    # Ruling 10 reaches the NAME as well as the contents.  ``root-validation-refusal.json``
    # is fsynced where it is written, but an unfsynced ``mkdir`` leaves the
    # directory entry that names its tree in the page cache, so a power loss in
    # the refusal window could take the freshly durable record with it.
    _fsync(output_root.parent)

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
            cold_lane_authorized=cold_lane,
            attempt_timeout_seconds=timeout_seconds,
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
        attempt_timeout_seconds=timeout_seconds,
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
    "ATTEMPT_CACHE_SHAPE",
    "ATTEMPT_ENVIRONMENT_SHAPE",
    "ATTEMPT_EVIDENCE_REQUIRED_KEYS",
    "ATTEMPT_EVIDENCE_SHAPE",
    "ATTEMPT_PROTOCOL_REQUIRED_KEYS",
    "ATTEMPT_PROTOCOL_SHAPE",
    "ATTEMPT_SOLVE_SHAPE",
    "ATTEMPT_STOP_RULE",
    "ATTEMPT_TIMEOUT_SECONDS",
    "ATTEMPT_TIMING_SHAPE",
    "BOUND_MODULE_SHAPE",
    "CACHE_CONFIGURATION_SHAPE",
    "CACHE_STATE_SHAPE",
    "CHAIN_EXECUTION_SOURCE_PATHS",
    "COLD_LANE_ANOMALY_SHAPE",
    "COLD_LANE_DIRECTORY",
    "COLD_LANE_MEASURED_OUTCOMES",
    "COLD_LANE_SHAPE",
    "COMPILATION_CACHE_ENVIRONMENT_VARIABLE",
    "CONFORMANCE_BOUNDED_SMOKE",
    "CONFORMANCE_PREREGISTERED",
    "DEFAULT_TEMPORARY_DIRECTORY",
    "ENDPOINT_AGREEMENT_SHAPE",
    "ENDPOINT_LEDGER_SHAPE",
    "EVIDENCE_FILENAME",
    "EXECUTION_SOURCES_SHAPE",
    "EXECUTION_SOURCE_MANIFEST_SHAPE",
    "GATED_ENDPOINT_LEDGER_SHAPE",
    "GPU_ATTEMPT_SCHEMA_VERSION",
    "GPU_MEMORY_SHAPE",
    "GPU_REQUIRED_ENVIRONMENT",
    "GPU_ROOT_MANIFEST_SCHEMA_VERSION",
    "GPU_ROOT_SCHEMA_VERSION",
    "INTERPRETER_INSTALLATION_SHAPE",
    "LOWERED_KERNEL_SHAPE",
    "LOWERING_PRE_GATE_SHAPE",
    "MANIFEST_FILENAME",
    "PERSISTENT_CACHE_MIN_COMPILE_TIME_SECONDS",
    "PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
    "PREFLIGHT_SHAPE",
    "PREREGISTERED_ATTEMPTS",
    "PROBLEM_IDENTITY_SHAPE",
    "PROJECTED_ROUTE",
    "RECEIPT_SHAPES",
    "REFUSAL_FILENAME",
    "REFUSAL_SCHEMA_VERSION",
    "REFUSED_ATTEMPT_EVIDENCE_REQUIRED_KEYS",
    "REFUSED_ATTEMPT_EVIDENCE_SHAPE",
    "REFUSED_STORAGE_FILESYSTEM_TYPES",
    "ROOT_CLAIM_SHAPE",
    "ROOT_EVIDENCE_REQUIRED_KEYS",
    "ROOT_EVIDENCE_SHAPE",
    "ROOT_TIMING_SHAPE",
    "RUNTIME_IDENTITY_SHAPE",
    "SOURCE_SNAPSHOT_SHAPE",
    "STORAGE_PROBE_SHAPE",
    "SUPERVISED_ATTEMPT_REQUIRED_KEYS",
    "SUPERVISED_ATTEMPT_SHAPE",
    "SUPERVISOR_SHAPE",
    "TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLE",
    "TEMPORARY_DIRECTORY_ENVIRONMENT_VARIABLES",
    "UNMANIFESTED_MODULE_SHAPE",
    "UNSHAPED_LEAVES",
    "VERDICT_CLAIM_DISCHARGED",
    "VERDICT_GATE_REFUSED_PREFIX",
    "VERDICT_NO_LATCH",
    "VERDICT_QUALITY_ONLY",
    "WORKTREE_IDENTITY_SHAPE",
    "GateRefusal",
    "ProjectedRootError",
    "attempt_engine_wall_seconds",
    "attempt_invocation",
    "attempt_protocol_conformance",
    "bind_gpu_backend",
    "build_root_evidence",
    "capture_worktree_identity",
    "cold_lane_anomaly",
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
