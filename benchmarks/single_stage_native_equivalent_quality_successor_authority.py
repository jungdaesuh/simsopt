"""Validate one-shot DIAG3--DIAG5 successor authorization evidence."""

from __future__ import annotations

import ctypes
import errno
import fcntl
import hashlib
import math
import os
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Final, Iterator

from simsopt_jax.runtime.exact_numeric_identity import exact_numeric_tree_sha256
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NEQ_GNTR3_SCHEMA_VERSION,
)

from benchmarks.single_stage_fullspace_snapshot import (
    DIAG4_CPU_SNAPSHOT_ROLES,
    DIAG5_CPU_SNAPSHOT_ROLES,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    SnapshotEntry,
    SnapshotIdentity,
    SnapshotPublication,
    WorktreeIdentity,
    build_snapshot_identity,
    canonical_json_bytes,
    load_canonical_json_bytes,
    load_snapshot,
    project_diag5_gpu_snapshot_identity,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG2_FROZEN_NUMERICAL_ENTRIES,
    DIAG2_SCHEMA_VERSION,
    DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
    DIAG4_NUMERICAL_ROUTE,
    DIAG4_PLAN_SHA256,
    DIAG4_ROUTE,
    DIAG4_SCHEMA_VERSION,
    DIAG5_EVIDENCE_SLOT_PATHS,
    DIAG5_SCHEMA_VERSION,
    DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    DiagnosticReceiptV5,
    FailureReasonCodeV5,
    FailureStageV5,
    NativeEquivalentNumericalIdentity,
    ScientificOutcome,
    StructuredFailureV5,
    load_and_validate_diag2_artifact,
    load_and_validate_diag5_artifact,
    validate_native_equivalent_scientific_evidence,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    DIAG5_SCHEMA_VERSION as DIAG5_SCIENTIFIC_EVIDENCE_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-command-buffer-recovery-authorization-v1"
)
ROUTE: Final = "NEQ-GNTR1-DIAG3-CB0"
PLAN_SHA256: Final = "3d46564297f6f18a04a69152eb71bfcb45796aa662f34d4b8b3e74a82a08a9b1"
GPU_UUID: Final = "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
NATIVE_REFERENCE_MANIFEST_SHA256: Final = (
    "5e2a68db43dd92d3287e33f827a055b9a5b2799ce464df4be19c0bfc5eef61db"
)
PLAN_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_plan.md"
AUTHORITY_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag3_command_buffer_recovery_authorization.json"
QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr1-command-buffer-recovery-qualification-v1"
)
CONTROLLING_CPU_COMMAND: Final = (
    "env JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src "
    ".venv-qn-cpu/bin/python -m pytest -q "
    "tests/benchmarks/test_process_gpu_monitor.py "
    "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py "
    "tests/benchmarks/test_single_stage_fullspace_snapshot.py "
    "tests/benchmarks/test_single_stage_compute_graph_attribution_control.py "
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py "
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py"
)
CONTROLLING_CPU_PASSED: Final = 880
R1_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag2-r1-20260811T123501Z"
)
R1_ARTIFACT_SHA256: Final = {
    "diagnostic.json": (
        "b671fea9294991ad3749fc115dab24ce3abe60c23b699da9dbe8904641bcdae7"
    ),
    "artifact-manifest.json": (
        "c6d244c9e82d31edc04036fff0722b75b59334d36aec599746c9e8d302500029"
    ),
    "cold/stderr.bin": (
        "debed91bf17f9eea2d0a56bd53df59ed4dfc1872065fee6a67d3cd7e1eb0e26a"
    ),
    "cold/raw-trace/plugins/profile/2026_08_11_08_50_43/jungdaesuh-playstation.trace.json.gz": (
        "4db22ea34a8092e3fc55e366a4ffe83055951bc2425322ed5f874c2bfe9b8c4e"
    ),
    "cold/raw-trace/plugins/profile/2026_08_11_08_50_43/jungdaesuh-playstation.xplane.pb": (
        "9983075bf5c00f2c1fc98be9492e08111f8f436081b6abf0e6c4f263d8f57fad"
    ),
}
QUALIFIED_FILE_PATHS: Final = (
    "benchmarks/process_gpu_monitor.py",
    "benchmarks/run_single_stage_native_equivalent_quality_campaign.py",
    "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
    "benchmarks/single_stage_fullspace_snapshot.py",
    "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
    "docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md",
    "docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md",
    "tests/benchmarks/_diag2_fixture.py",
    "tests/benchmarks/test_process_gpu_monitor.py",
    "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
    "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
)
DIAG3_SOURCE_DELTA_ALLOWLIST: Final = frozenset(
    {*QUALIFIED_FILE_PATHS, AUTHORITY_RELATIVE_PATH, PLAN_RELATIVE_PATH}
)

DIAG4_AUTHORITY_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-diag4-authorization-v1"
DIAG4_QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag4-qualification-v1"
)
DIAG4_CONSUMPTION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag4-authority-consumption-v1"
)
DIAG4_PLAN_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md"
DIAG4_PREQUALIFICATION_PLAN_SHA256: Final = (
    "5c27a900472917749558f1b86502bfeb0aec900c733f53d8a29c0dbe41a770"
)
DIAG4_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-prequalification-plan-control-v1"
)
DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH: Final = "control/prequalification-plan.md"
DIAG4_AUTHORITY_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_authorization.json"
DIAG4_GPU_UUID: Final = GPU_UUID
DIAG4_BASE_POLICY_SHA256: Final = (
    "6face7116d36d2eae954bb5b3bde465f37c990ceb9b319ba2c20bb62ad1a6f99"
)
DIAG4_CONSUMED_DIAG3_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr1-diag3-cb0-20260811T150010Z.partial-56a1ec6d730cc005db84f99e9965b868"
)
DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS: Final = frozenset(
    {
        "cold/arrays/accepted_physical_ledger.npy",
        "cold/history.json",
        "cold/policy.json",
        "cold/terminal-numerical.json",
        "frozen-numerical-subset.json",
        "native-reference/artifact-manifest.json",
        "policy-authority.json",
        "source-snapshot/source-manifest.json",
        "supervisor-terminal.json",
    }
)
DIAG4_IDENTITY_FIELDS: Final = frozenset(
    {
        "problem_sha256",
        "optimizer_options_sha256",
        "base_neq_gntr1_policy_sha256",
        "scaling_sha256",
        "bootstrap_state_sha256",
        "initial_physical_state_sha256",
        "identity_sha256",
    }
)
DIAG4_QUALIFIED_FILE_PATHS: Final = frozenset(
    {
        "benchmarks/process_gpu_monitor.py",
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "benchmarks/run_single_stage_native_equivalent_quality_campaign.py",
        "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json",
        "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
        "benchmarks/single_stage_fullspace_snapshot.py",
        "benchmarks/single_stage_native_equivalent_endpoint_audit.py",
        "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "benchmarks/single_stage_native_equivalent_quality_receipt.py",
        "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
        "benchmarks/single_stage_native_equivalent_reference.py",
        "tests/benchmarks/_diag2_fixture.py",
        "tests/benchmarks/test_process_gpu_monitor.py",
        "tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
        "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
        "tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_reference.py",
        "tests/geo/test_fullspace_native_equivalent_quality.py",
        "tests/geo/test_projected_gauss_newton_trust_region.py",
    }
)
DIAG4_EXECUTION_SOURCE_MANIFEST_PATH: Final = (
    "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json"
)
DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-execution-source-authority-v1"
)
DIAG4_EXECUTION_SOURCE_ENTRY_COUNT: Final = 603
DIAG4_EXECUTION_SOURCE_BROAD_ROOTS: Final = ("benchmarks", "examples", "src")
DIAG4_FROZEN_NUMERICAL_PATHS: Final = frozenset(
    {
        "benchmarks/single_stage_native_equivalent_reference.py",
        "examples/jax/parity/cases/native_boozerqa.py",
        "examples/jax/parity/input_bundle.py",
        "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py",
        "src/simsopt_jax/objectives/single_stage_fullspace.py",
        "src/simsopt_jax/runtime/trace_annotations.py",
        "src/simsopt_jax/solve/fullspace.py",
        "src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py",
        "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py",
        "src/simsopt_jax_adapters/geo/single_stage_fullspace.py",
        "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py",
    }
)
DIAG4_CPU20_RESULT_PATH: Final = Path("/tmp/diag4_safeguard_cpu20_result.json")
DIAG4_CPU20_RESULT_SHA256: Final = (
    "7f08eadfe17e3a18f4c8f480d482945ddff3df4ed5a6092f4252ef1d054846fe"
)
DIAG4_CPU20_HARNESS_PATH: Final = Path("/tmp/diag4_safeguard_cpu20.py")
DIAG4_CPU20_HARNESS_SHA256: Final = (
    "bb339b9968885f642af2d247c92ef01a26f9dc8a0e639f539a29cc7060897525"
)
DIAG4_CPU20_COMMAND: Final = (
    "/usr/bin/env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true "
    "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=src "
    "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/"
    ".venv-qn-cpu/bin/python /tmp/diag4_safeguard_cpu20.py"
)
DIAG4_CPU20_DURATION_SECONDS: Final = 1076.8481874999707
DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-v1"
)
DIAG4_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v1"
)
DIAG4_CPU_QUALIFICATION_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag4-cpu-qualification-20260811T214932Z"
)
DIAG4_CPU_QUALIFICATION_COMMAND: Final = (
    "env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true "
    "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
    "PYTHONPATH=src .venv-qn-cpu/bin/python "
    "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py "
    f"--output-root {DIAG4_CPU_QUALIFICATION_ROOT}"
)
_DIAG4_STATIC_PATHS: Final = (
    "benchmarks/process_gpu_monitor.py",
    "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
    "benchmarks/run_single_stage_native_equivalent_quality_campaign.py",
    "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
    "benchmarks/single_stage_fullspace_snapshot.py",
    "benchmarks/single_stage_native_equivalent_endpoint_audit.py",
    "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "benchmarks/single_stage_native_equivalent_quality_receipt.py",
    "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
    "benchmarks/single_stage_native_equivalent_reference.py",
    "examples/jax/parity/cases/native_boozerqa.py",
    "examples/jax/parity/input_bundle.py",
    "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py",
    "src/simsopt_jax/objectives/single_stage_fullspace.py",
    "src/simsopt_jax/runtime/trace_annotations.py",
    "src/simsopt_jax/solve/fullspace.py",
    "src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py",
    "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py",
    "src/simsopt_jax_adapters/geo/single_stage_fullspace.py",
    "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py",
    "tests/benchmarks/_diag2_fixture.py",
    "tests/benchmarks/test_process_gpu_monitor.py",
    "tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
    "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
    "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
    "tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
    "tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py",
    "tests/benchmarks/test_single_stage_native_equivalent_reference.py",
    "tests/geo/test_fullspace_native_equivalent_quality.py",
    "tests/geo/test_projected_gauss_newton_trust_region.py",
)
_DIAG4_TEST_PATHS: Final = _DIAG4_STATIC_PATHS[21:]
_DIAG4_GIT_DIFF_PATHS: Final = (
    *_DIAG4_STATIC_PATHS[:7],
    DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
    *_DIAG4_STATIC_PATHS[7:],
)
DIAG4_CONTROLLING_CPU_COMMAND: Final = (
    "env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true "
    "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
    "PYTHONPATH=src .venv-qn-cpu/bin/python -m pytest -q "
    "--basetemp /home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag4-pytest-qualification-20260811T223700Z "
    + " ".join(_DIAG4_TEST_PATHS)
)
DIAG4_STATIC_COMMANDS: Final = MappingProxyType(
    {
        "compileall": ".venv-qn-cpu/bin/python -m compileall -q "
        + " ".join(_DIAG4_STATIC_PATHS),
        "git_diff_check": "git diff --check -- "
        + DIAG4_PLAN_RELATIVE_PATH
        + " "
        + " ".join(_DIAG4_GIT_DIFF_PATHS),
        "ruff_check": ".venv-qn-cpu/bin/python -m ruff check "
        + " ".join(_DIAG4_STATIC_PATHS),
        "ruff_format_check": ".venv-qn-cpu/bin/python -m ruff format --check "
        + " ".join(_DIAG4_STATIC_PATHS),
    }
)
DIAG4_REVIEW_ROLES: Final = frozenset(
    {"atomic-lifecycle", "numerical-controller", "receipt-schema", "source-snapshot"}
)

DIAG5_ROUTE: Final = "NEQ-GNTR3-DIAG5"
DIAG5_PLAN_SHA256: Final = (
    "24300e9742bcbb14b3fc3e2cceab37dedc310410290a13370a11d21ef749ec7a"
)
DIAG5_BLANK_PLAN_SHA256: Final = (
    "ce244ac37bb437ea022a4b73e62bf49f8d4d5cf88b610ad2146e895f3471ce1c"
)
DIAG5_BLANK_PLAN_SIZE_BYTES: Final = 76153
DIAG5_PLAN_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag5_native_binding_recovery_plan.md"
DIAG5_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-prequalification-plan-control-v2"
)
DIAG5_QUALIFIED_FILE_PATHS: Final = frozenset(
    {
        "benchmarks/process_gpu_monitor.py",
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "benchmarks/run_single_stage_native_equivalent_quality_campaign.py",
        "benchmarks/single_stage_fullspace_process_gpu_monitor.py",
        "benchmarks/single_stage_fullspace_snapshot.py",
        "benchmarks/single_stage_native_equivalent_endpoint_audit.py",
        "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "benchmarks/single_stage_native_equivalent_quality_gntr3_execution_sources.json",
        "benchmarks/single_stage_native_equivalent_quality_receipt.py",
        "benchmarks/single_stage_native_equivalent_quality_successor_authority.py",
        "benchmarks/single_stage_native_equivalent_reference.py",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json",
        "src/simsopt/configs/NCSX.dat",
        "tests/benchmarks/_diag2_fixture.py",
        "tests/benchmarks/test_process_gpu_monitor.py",
        "tests/benchmarks/test_qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
        "tests/benchmarks/test_single_stage_fullspace_snapshot.py",
        "tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_reference.py",
        "tests/geo/test_fullspace_native_equivalent_quality.py",
        "tests/geo/test_projected_gauss_newton_trust_region.py",
    }
)
DIAG5_FROZEN_NUMERICAL_PATHS: Final = frozenset(
    {
        "benchmarks/single_stage_native_equivalent_reference.py",
        "examples/jax/parity/cases/native_boozerqa.py",
        "examples/jax/parity/input_bundle.py",
        "src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py",
        "src/simsopt_jax/objectives/single_stage_fullspace.py",
        "src/simsopt_jax/runtime/trace_annotations.py",
        "src/simsopt_jax/solve/fullspace.py",
        "src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py",
        "src/simsopt_jax/solve/fullspace_native_equivalent_quality.py",
        "src/simsopt_jax_adapters/geo/single_stage_fullspace.py",
        "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py",
    }
)
DIAG5_EXECUTION_SOURCE_ENTRY_COUNT: Final = 630
DIAG5_CPU_QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-v2"
)
DIAG5_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-cpu-trajectory-qualification-manifest-v2"
)
DIAG5_AUTHORITY_SCHEMA_VERSION: Final = "single-stage-neq-gntr3-diag5-authorization-v1"
DIAG5_QUALIFICATION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag5-qualification-v1"
)
DIAG5_CONSUMPTION_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag5-authority-consumption-v1"
)
DIAG5_CPU_QUALIFICATION_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag5-cpu-qualification-20260812T110000Z"
)
DIAG5_GPU_OUTPUT_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-rtx5090-20260812T030000Z"
)
DIAG5_GPU_STAGING_ROOT: Final = Path(f"{DIAG5_GPU_OUTPUT_ROOT}.partial-claim")
DIAG5_GPU_ROLLBACK_ROOT: Final = Path(f"{DIAG5_GPU_OUTPUT_ROOT}.partial-rollback")
DIAG5_PHYSICAL_FAILURE_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag5-physical-publication-failure-v1"
)
DIAG5_PHYSICAL_FAILURE_PATH: Final = DIAG5_GPU_OUTPUT_ROOT.parent / (
    f".{DIAG5_GPU_OUTPUT_ROOT.name}.diag5-physical-publication-failure.json"
)
DIAG5_REVIEW_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/neq-gntr3-diag5-reviews-20260812T110000Z"
)
DIAG5_NATIVE_REFERENCE_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/neq-gntr1-diag3-cb0-20260811T150010Z.partial-56a1ec6d730cc005db84f99e9965b868/native-reference"
)
DIAG5_INPUT_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/.single-stage-speed-20260804.partial-20260805T052535Z-2add24ec/inputs"
)
DIAG5_AUTHORITY_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag5_native_binding_recovery_authorization.json"
DIAG5_GPU_INTERPRETER: Final = Path(
    "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/"
    ".venv-qn-gpu/bin/python"
)
DIAG5_NATIVE_COPY_RELATIVE_PATH: Final = (
    "native/simsoptpp.cpython-311-x86_64-linux-gnu.so"
)
DIAG5_FAILED_DIAG4_PARTIAL_ROOT: Final = Path(
    "/home/jungdaesuh/simsopt-campaigns/"
    "neq-gntr3-diag4-cpu-qualification-20260811T214932Z.partial-claim"
)
DIAG5_FAILED_DIAG4_FINAL_ROOT: Final = DIAG4_CPU_QUALIFICATION_ROOT
DIAG5_FAILED_DIAG4_STAGE: Final = "NATIVE_EXTENSION_RUNTIME_BINDING"
DIAG5_FAILED_DIAG4_EXCEPTION_CLASS: Final = "QualificationError"
DIAG5_FAILED_DIAG4_EXCEPTION_MESSAGE: Final = "native extension runtime binding differs"
DIAG5_FAILED_DIAG4_QUALIFIER_RELATIVE_PATH: Final = (
    "execution-source/benchmarks/"
    "qualify_single_stage_native_equivalent_quality_gntr3_cpu.py"
)
DIAG5_FAILED_DIAG4_QUALIFIER_SHA256: Final = (
    "fbe302885c5b392958fb69ed5081edc0d69104573f19843c5be480c37af44c51"
)
DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_RELATIVE_PATH: Final = (
    "execution-source/benchmarks/"
    "single_stage_native_equivalent_quality_gntr3_execution_sources.json"
)
DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_SHA256: Final = (
    "386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604"
)
DIAG5_FAILED_DIAG4_EXECUTION_SOURCE_ENTRY_COUNT: Final = 603
DIAG5_FAILED_DIAG4_EXECUTION_ENTRIES_SHA256: Final = (
    "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2"
)
DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION: Final = (
    "single-stage-neq-gntr3-diag4-independent-postmortem-v1"
)
DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH: Final = "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_independent_postmortem.json"
DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH: Final = (
    "control/predecessor-postmortem.json"
)
DIAG5_PREDECESSOR_POSTMORTEM_SHA256: Final = (
    "377b3f2c4401ac9a9c002cf801294dabd2d0b92238bd44c3da5c1d37585d7791"
)
DIAG5_FAILED_DIAG4_NATIVE_PATH: Final = Path(
    "/home/jungdaesuh/code/columbia/simsopt-pr-jax-port-squashed/"
    ".venv-qn-cpu/lib/python3.11/site-packages/"
    "simsoptpp.cpython-311-x86_64-linux-gnu.so"
)
_DIAG5_RETRACTED_REVIEW_HASHES: Final = MappingProxyType(
    {
        "reviewed_qualified_files_sha256": (
            "e1938b81503c696bd5dc796045cdd8164e14453420b48fb38fb0f89b35ddbcc8"
        ),
        "reviewed_frozen_numerical_entries_sha256": (
            "57a3bf08fad41871812322b516f994a8e66abe2104c0e8ed0055688e3209f7e0"
        ),
        "reviewed_execution_source_manifest_sha256": (
            "386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604"
        ),
        "reviewed_execution_source_entries_sha256": (
            "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2"
        ),
        "reviewed_plan_full_sha256": (
            "5c27a90047291774955858f1b86502bfeb0aec900c733f53d8a29c0dbe41a770"
        ),
        "reviewed_plan_prefix_sha256": (
            "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c"
        ),
    }
)
_DIAG5_RETRACTED_REVIEW_IDENTITIES: Final = (
    (
        "numerical-controller",
        "codex-numerical-controller-current-manifest",
        "numerical-controller-20260811T220006-manifest386698c5",
    ),
    (
        "receipt-schema",
        "codex-receipt-schema-a55a4fac",
        "5c87cc42-3234-4b9f-bcd8-3eee3e0ea01d",
    ),
    (
        "source-snapshot",
        "/root/ftr_runner_receipt",
        "source-snapshot-final-20260811-ftr01",
    ),
    (
        "atomic-lifecycle",
        "codex-atomic-lifecycle-current-manifest",
        "/root/diag_runner_map/ssot_atomic_review@2026-08-12T02:01:09Z",
    ),
)
DIAG5_FAILED_DIAG4_FORBIDDEN_PATHS: Final = frozenset(
    {
        "arrays",
        "artifact-manifest.json",
        "endpoint-audit.json",
        "history.json",
        "policy.json",
        "safeguard-telemetry.json",
        "scientific-evidence.json",
        "source-snapshot",
        "terminal-numerical.json",
    }
)


@dataclass(frozen=True, slots=True)
class Diag5NativeExtensionBinding:
    """One descriptor-bound live native extension; link count is telemetry."""

    path: Path
    sha256: str
    size_bytes: int
    device: int
    inode: int
    link_count: int


@dataclass(frozen=True, slots=True)
class Diag5NativeExtensionClaim:
    """One lifetime-held live native binding; valid only inside its context."""

    binding: Diag5NativeExtensionBinding
    _leaf: _Diag4LockedLeaf = field(repr=False, compare=False)
    _directories: Mapping[Path, int] = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class Diag5PredecessorFailureEvidence:
    """Exact immutable DIAG4 precompile failure consumed by DIAG5."""

    partial_root: Path
    failed_stage: str
    exception_class: str
    exception_message: str
    qualifier_sha256: str
    execution_manifest_sha256: str
    execution_entries_sha256: str
    execution_source_entry_count: int
    copied_tree_entry_count: int
    predecessor_full_tree_sha256: str
    postmortem_path: Path
    postmortem_sha256: str


def _diag5_descriptor_bytes(descriptor: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return b"".join(chunks)


def _diag5_path_lexically_absent(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    return False


def _diag5_predecessor_tree_entries(execution_root: Path) -> dict[str, JsonValue]:
    """Read the exact sealed predecessor tree through no-follow dirfds."""

    entries: dict[str, JsonValue] = {}
    root_descriptor = os.open(
        execution_root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )

    def visit(directory_descriptor: int, prefix: PurePosixPath) -> None:
        directory_metadata = os.fstat(directory_descriptor)
        if (
            not stat.S_ISDIR(directory_metadata.st_mode)
            or stat.S_IMODE(directory_metadata.st_mode) != 0o555
        ):
            raise ValueError("DIAG5 predecessor execution directory is not sealed")
        for name in sorted(os.listdir(directory_descriptor)):
            metadata = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
            relative = prefix / name
            if stat.S_ISDIR(metadata.st_mode):
                child = os.open(
                    name,
                    os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                    dir_fd=directory_descriptor,
                )
                try:
                    visit(child, relative)
                    rebound = os.stat(
                        name, dir_fd=directory_descriptor, follow_symlinks=False
                    )
                    held = os.fstat(child)
                    if (rebound.st_dev, rebound.st_ino) != (
                        held.st_dev,
                        held.st_ino,
                    ):
                        raise ValueError(
                            "DIAG5 predecessor execution directory binding drifted"
                        )
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("DIAG5 predecessor execution leaf is not regular")
            descriptor = os.open(
                name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_descriptor,
            )
            try:
                held = os.fstat(descriptor)
                payload = _diag5_descriptor_bytes(descriptor)
                rebound = os.stat(
                    name, dir_fd=directory_descriptor, follow_symlinks=False
                )
                if (
                    (held.st_dev, held.st_ino, held.st_size)
                    != (rebound.st_dev, rebound.st_ino, rebound.st_size)
                    or held.st_size != len(payload)
                    or stat.S_IMODE(held.st_mode) != 0o444
                    or held.st_nlink != 1
                ):
                    raise ValueError("DIAG5 predecessor execution leaf binding differs")
                entries[relative.as_posix()] = {
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                    "mode": "0444",
                    "link_count": 1,
                }
            finally:
                os.close(descriptor)

    try:
        visit(root_descriptor, PurePosixPath())
    finally:
        try:
            held_root = os.fstat(root_descriptor)
            bound_root = execution_root.stat(follow_symlinks=False)
            if (held_root.st_dev, held_root.st_ino) != (
                bound_root.st_dev,
                bound_root.st_ino,
            ):
                raise ValueError("DIAG5 predecessor execution root binding drifted")
        finally:
            os.close(root_descriptor)
    return entries


def _diag5_read_predecessor_tree_leaf(
    execution_root: Path, relative_path: str
) -> bytes:
    parts = PurePosixPath(relative_path).parts
    directory = os.open(
        execution_root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        for part in parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        descriptor = os.open(
            parts[-1],
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=directory,
        )
        try:
            held = os.fstat(descriptor)
            payload = _diag5_descriptor_bytes(descriptor)
            rebound = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
            if (held.st_dev, held.st_ino, held.st_size) != (
                rebound.st_dev,
                rebound.st_ino,
                rebound.st_size,
            ):
                raise ValueError("DIAG5 predecessor leaf binding drifted")
            return payload
        finally:
            os.close(descriptor)
    finally:
        os.close(directory)


def _diag5_expected_retracted_reviews() -> list[JsonValue]:
    return [
        {
            "role": role,
            "reviewer": reviewer,
            "session": session,
            "verdict": "RETRACTED",
            **_DIAG5_RETRACTED_REVIEW_HASHES,
        }
        for role, reviewer, session in _DIAG5_RETRACTED_REVIEW_IDENTITIES
    ]


def validate_diag5_predecessor_postmortem_artifact(
    artifact_root: Path,
    reference: ArtifactRef,
) -> Mapping[str, JsonValue]:
    """Deep-load the one exact sealed DIAG4 postmortem copied into DIAG5."""

    root = artifact_root.resolve(strict=True)
    if (
        reference.relative_path != DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH
        or reference.schema_version != DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        or reference.sha256 != DIAG5_PREDECESSOR_POSTMORTEM_SHA256
    ):
        raise ValueError("DIAG5 predecessor postmortem reference differs")
    relative = PurePosixPath(reference.relative_path)
    path = root.joinpath(*relative.parts)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        payload = _diag5_descriptor_bytes(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or metadata.st_size != reference.size_bytes
            or (metadata.st_dev, metadata.st_ino, metadata.st_size)
            != (bound.st_dev, bound.st_ino, bound.st_size)
            or hashlib.sha256(payload).hexdigest() != reference.sha256
        ):
            raise ValueError("DIAG5 predecessor postmortem artifact differs")
    finally:
        os.close(descriptor)
    value = load_canonical_json_bytes(payload)
    document = _mapping(value, "DIAG5 predecessor postmortem artifact")
    if hashlib.sha256(canonical_json_bytes(document)).hexdigest() != (
        DIAG5_PREDECESSOR_POSTMORTEM_SHA256
    ):
        raise ValueError("DIAG5 predecessor postmortem semantics differ")
    return MappingProxyType(dict(document))


def observe_diag5_native_extension_binding(path: Path) -> Diag5NativeExtensionBinding:
    """Observe one live extension now; a lifetime claim must retain its descriptor."""

    canonical = path.resolve(strict=True)
    if canonical != path.absolute():
        raise ValueError("DIAG5 native extension path must be canonical")
    descriptor = os.open(canonical, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        observed = os.fstat(descriptor)
        bound = canonical.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink < 1
            or (observed.st_dev, observed.st_ino, observed.st_size)
            != (bound.st_dev, bound.st_ino, bound.st_size)
        ):
            raise ValueError("DIAG5 native extension descriptor binding differs")
        return Diag5NativeExtensionBinding(
            path=canonical,
            sha256=hashlib.sha256(_diag5_descriptor_bytes(descriptor)).hexdigest(),
            size_bytes=observed.st_size,
            device=observed.st_dev,
            inode=observed.st_ino,
            link_count=observed.st_nlink,
        )
    finally:
        os.close(descriptor)


@contextmanager
def claim_diag5_native_extension_binding(
    path: Path,
) -> Iterator[Diag5NativeExtensionClaim]:
    """Hold the complete path chain, inode lock, bytes, mode, and link topology."""

    canonical = path.resolve(strict=True)
    if canonical != path.absolute():
        raise ValueError("DIAG5 native extension path must be canonical")
    with _diag4_claim_directories((canonical.parent,)) as directories:
        leaf = _open_diag4_locked_leaf(canonical, directories, "DIAG5 native extension")
        try:
            observed = os.fstat(leaf.descriptor)
            binding = Diag5NativeExtensionBinding(
                path=canonical,
                sha256=leaf.initial_sha256,
                size_bytes=leaf.initial_size_bytes,
                device=observed.st_dev,
                inode=observed.st_ino,
                link_count=observed.st_nlink,
            )
            if binding.link_count < 1:
                raise ValueError("DIAG5 live native link count must be positive")
            yield Diag5NativeExtensionClaim(binding, leaf, directories)
            _assert_diag4_locked_leaf_binding(leaf, directories)
            if observe_diag5_native_extension_binding(canonical) != binding:
                raise ValueError("DIAG5 live native extension binding drifted")
        finally:
            fcntl.flock(leaf.descriptor, fcntl.LOCK_UN)
            os.close(leaf.descriptor)


def validate_diag5_cross_runtime_native_bindings(
    cpu: Diag5NativeExtensionBinding,
    gpu: Diag5NativeExtensionBinding,
) -> None:
    """Require byte identity across runtime paths without equating link topology."""

    if cpu.link_count < 1 or gpu.link_count < 1:
        raise ValueError("DIAG5 live native link count must be positive")
    if (cpu.sha256, cpu.size_bytes) != (gpu.sha256, gpu.size_bytes):
        raise ValueError("DIAG5 CPU/GPU native binary identity differs")


def revalidate_diag5_native_extension_binding(
    binding: Diag5NativeExtensionBinding,
) -> None:
    """Fail if one previously captured live path, inode, bytes, or links drifted."""

    if observe_diag5_native_extension_binding(binding.path) != binding:
        raise ValueError("DIAG5 live native extension binding drifted")


def validate_diag5_sealed_native_copy(
    path: Path,
    binding: Diag5NativeExtensionBinding,
) -> None:
    """Require the snapshot copy to be immutable, unique, and byte-identical."""

    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        bound = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino, metadata.st_size)
            != (bound.st_dev, bound.st_ino, bound.st_size)
            or metadata.st_size != binding.size_bytes
            or hashlib.sha256(_diag5_descriptor_bytes(descriptor)).hexdigest()
            != binding.sha256
        ):
            raise ValueError("DIAG5 sealed native snapshot copy differs")
    finally:
        os.close(descriptor)


def validate_diag5_predecessor_failure(
    evidence: Diag5PredecessorFailureEvidence,
    *,
    repository_root: Path,
) -> None:
    """Validate the preserved DIAG4 partial without treating it as science."""

    repository = repository_root.resolve(strict=True)
    root = evidence.partial_root
    root_metadata = root.lstat()
    if (
        root != DIAG5_FAILED_DIAG4_PARTIAL_ROOT
        or root.resolve(strict=True) != root
        or not stat.S_ISDIR(root_metadata.st_mode)
        or stat.S_IMODE(root_metadata.st_mode) != 0o755
    ):
        raise ValueError("DIAG5 predecessor partial root differs")
    if not _diag5_path_lexically_absent(DIAG5_FAILED_DIAG4_FINAL_ROOT):
        raise ValueError("DIAG5 predecessor final root must remain absent")
    if (
        evidence.failed_stage != DIAG5_FAILED_DIAG4_STAGE
        or evidence.exception_class != DIAG5_FAILED_DIAG4_EXCEPTION_CLASS
        or evidence.exception_message != DIAG5_FAILED_DIAG4_EXCEPTION_MESSAGE
        or evidence.qualifier_sha256 != DIAG5_FAILED_DIAG4_QUALIFIER_SHA256
        or evidence.execution_manifest_sha256
        != DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_SHA256
        or evidence.execution_source_entry_count
        != DIAG5_FAILED_DIAG4_EXECUTION_SOURCE_ENTRY_COUNT
        or evidence.execution_entries_sha256
        != DIAG5_FAILED_DIAG4_EXECUTION_ENTRIES_SHA256
        or evidence.copied_tree_entry_count
        != DIAG5_FAILED_DIAG4_EXECUTION_SOURCE_ENTRY_COUNT + 1
    ):
        raise ValueError("DIAG5 predecessor failure evidence differs")
    expected_postmortem = repository / DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH
    if evidence.postmortem_path != expected_postmortem:
        raise ValueError("DIAG5 predecessor postmortem path differs")
    postmortem_descriptor = os.open(
        expected_postmortem, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    )
    try:
        postmortem_metadata = os.fstat(postmortem_descriptor)
        postmortem_bound = expected_postmortem.stat(follow_symlinks=False)
        postmortem_bytes = _diag5_descriptor_bytes(postmortem_descriptor)
        if (
            not stat.S_ISREG(postmortem_metadata.st_mode)
            or stat.S_IMODE(postmortem_metadata.st_mode) != 0o444
            or postmortem_metadata.st_nlink != 1
            or (postmortem_metadata.st_dev, postmortem_metadata.st_ino)
            != (postmortem_bound.st_dev, postmortem_bound.st_ino)
        ):
            raise ValueError("DIAG5 predecessor postmortem is not sealed")
    finally:
        os.close(postmortem_descriptor)
    if hashlib.sha256(postmortem_bytes).hexdigest() != evidence.postmortem_sha256:
        raise ValueError("DIAG5 predecessor postmortem bytes differ")
    postmortem = _mapping(
        load_canonical_json_bytes(postmortem_bytes), "DIAG5 predecessor postmortem"
    )
    _exact_keys(
        postmortem,
        frozenset(
            {
                "schema_version",
                "session_reference",
                "original_stdout_retained",
                "original_stderr_retained",
                "original_process_receipt",
                "reconstruction",
            }
        ),
        "DIAG5 predecessor postmortem",
    )
    if (
        postmortem["schema_version"] != DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        or postmortem["session_reference"] != "74963"
        or postmortem["original_stdout_retained"] is not False
        or postmortem["original_stderr_retained"] is not False
        or postmortem["original_process_receipt"] != "NOT_PRODUCED"
    ):
        raise ValueError("DIAG5 predecessor postmortem provenance differs")
    reconstruction = _mapping(
        postmortem["reconstruction"], "DIAG5 predecessor reconstruction"
    )
    _exact_keys(
        reconstruction,
        frozenset(
            {
                "command_text",
                "partial_root",
                "failed_stage",
                "exception_class",
                "exception_message",
                "qualifier_sha256",
                "execution_manifest_sha256",
                "execution_entries_sha256",
                "execution_source_entry_count",
                "copied_tree_entry_count",
                "predecessor_full_tree_sha256",
                "copied_qualifier_predicate",
                "native_binding",
                "final_root_absent",
                "scientific_paths_absent",
                "prior_reviews_retracted",
                "retracted_reviews_sha256",
            }
        ),
        "DIAG5 predecessor reconstruction",
    )
    if (
        reconstruction["command_text"] != DIAG4_CPU_QUALIFICATION_COMMAND
        or reconstruction["partial_root"] != str(root)
        or reconstruction["failed_stage"] != evidence.failed_stage
        or reconstruction["exception_class"] != evidence.exception_class
        or reconstruction["exception_message"] != evidence.exception_message
        or reconstruction["qualifier_sha256"] != evidence.qualifier_sha256
        or reconstruction["execution_manifest_sha256"]
        != evidence.execution_manifest_sha256
        or reconstruction["execution_entries_sha256"]
        != evidence.execution_entries_sha256
        or reconstruction["execution_source_entry_count"]
        != evidence.execution_source_entry_count
        or reconstruction["copied_tree_entry_count"] != evidence.copied_tree_entry_count
        or reconstruction["predecessor_full_tree_sha256"]
        != evidence.predecessor_full_tree_sha256
        or reconstruction["copied_qualifier_predicate"] != "observed.st_nlink != 1"
        or reconstruction["final_root_absent"] is not True
        or reconstruction["scientific_paths_absent"] is not True
    ):
        raise ValueError("DIAG5 predecessor reconstruction differs")
    reviews = reconstruction["prior_reviews_retracted"]
    expected_reviews = _diag5_expected_retracted_reviews()
    if (
        reviews != expected_reviews
        or hashlib.sha256(canonical_json_bytes(reviews)).hexdigest()
        != reconstruction["retracted_reviews_sha256"]
    ):
        raise ValueError("DIAG5 predecessor review retraction differs")
    native_binding = _mapping(
        reconstruction["native_binding"], "DIAG5 predecessor native binding"
    )
    _exact_keys(
        native_binding,
        frozenset(
            {
                "path",
                "loader",
                "sha256",
                "size_bytes",
                "device",
                "inode",
                "link_count",
            }
        ),
        "DIAG5 predecessor native binding",
    )
    if (
        native_binding["path"] != str(DIAG5_FAILED_DIAG4_NATIVE_PATH)
        or native_binding["loader"] != "_ScikitBuildLoaderWrapper"
        or native_binding["sha256"]
        != "41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b"
        or native_binding["size_bytes"] != 2883776
        or native_binding["device"] != 66306
        or native_binding["inode"] != 50480769
        or native_binding["link_count"] != 2
    ):
        raise ValueError("DIAG5 predecessor native topology differs")
    root_descriptor = os.open(
        root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        if set(os.listdir(root_descriptor)) != {"execution-source"}:
            raise ValueError("DIAG5 predecessor partial root membership differs")
    finally:
        os.close(root_descriptor)
    execution_root = root / "execution-source"
    observed = _diag5_predecessor_tree_entries(execution_root)
    qualifier_relative = DIAG5_FAILED_DIAG4_QUALIFIER_RELATIVE_PATH.removeprefix(
        "execution-source/"
    )
    manifest_relative = (
        DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_RELATIVE_PATH.removeprefix(
            "execution-source/"
        )
    )
    if (
        observed[qualifier_relative]["sha256"] != evidence.qualifier_sha256
        or observed[manifest_relative]["sha256"] != evidence.execution_manifest_sha256
    ):
        raise ValueError("DIAG5 predecessor copied authority bytes differ")
    manifest_bytes = _diag5_read_predecessor_tree_leaf(
        execution_root, manifest_relative
    )
    manifest = _mapping(
        load_canonical_json_bytes(manifest_bytes),
        "DIAG5 predecessor execution-source manifest",
    )
    entries = _mapping(manifest["entries"], "DIAG5 predecessor execution entries")
    if (
        len(entries) != evidence.execution_source_entry_count
        or manifest["entries_sha256"] != evidence.execution_entries_sha256
        or hashlib.sha256(canonical_json_bytes(entries)).hexdigest()
        != evidence.execution_entries_sha256
    ):
        raise ValueError("DIAG5 predecessor execution-source membership differs")
    expected_paths = {*entries, Path(DIAG4_EXECUTION_SOURCE_MANIFEST_PATH).as_posix()}
    if (
        set(observed) != expected_paths
        or len(observed) != evidence.copied_tree_entry_count
    ):
        raise ValueError("DIAG5 predecessor copied execution tree differs")
    for relative, raw_entry in entries.items():
        entry = _mapping(raw_entry, f"DIAG5 predecessor manifest entry {relative}")
        _exact_keys(
            entry,
            frozenset({"sha256", "size_bytes"}),
            f"DIAG5 predecessor manifest entry {relative}",
        )
        if (
            observed[relative]["sha256"] != entry["sha256"]
            or observed[relative]["size_bytes"] != entry["size_bytes"]
        ):
            raise ValueError("DIAG5 predecessor execution leaf differs from manifest")
    if observed[manifest_relative]["sha256"] != evidence.execution_manifest_sha256:
        raise ValueError("DIAG5 predecessor manifest leaf differs")
    if hashlib.sha256(canonical_json_bytes(observed)).hexdigest() != (
        evidence.predecessor_full_tree_sha256
    ):
        raise ValueError("DIAG5 predecessor full-tree aggregate differs")


class Diag5AuthorityLifecycle(str, Enum):
    UNCONSUMED = "UNCONSUMED"
    CONSUMPTION_UNCERTAIN = "CONSUMPTION_UNCERTAIN"
    CONSUMED = "CONSUMED"


class Diag5ConsumptionMarkerInvalidError(ValueError):
    """The durable DIAG5 consumption marker is absent or differs."""


class Diag5PublishedOutputKind(str, Enum):
    FINAL = "FINAL"


class Diag5RollbackCause(str, Enum):
    NONE = "NONE"
    ROLLBACK_COLLISION = "ROLLBACK_COLLISION"
    ROLLBACK_RENAME_FAILED = "ROLLBACK_RENAME_FAILED"
    ROLLBACK_PARENT_FSYNC_FAILED = "ROLLBACK_PARENT_FSYNC_FAILED"
    ROLLBACK_DEEP_LOAD_FAILED = "ROLLBACK_DEEP_LOAD_FAILED"
    ROLLBACK_VISIBILITY_AMBIGUOUS = "ROLLBACK_VISIBILITY_AMBIGUOUS"


class Diag5RollbackState(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    AMBIGUOUS = "AMBIGUOUS"


class Diag5PhysicalPathState(str, Enum):
    ABSENT = "ABSENT"
    VISIBLE_VALIDATED = "VISIBLE_VALIDATED"
    VISIBLE_INVALID = "VISIBLE_INVALID"
    VISIBILITY_AMBIGUOUS = "VISIBILITY_AMBIGUOUS"


class Diag5EvidenceNamespaceState(str, Enum):
    PENDING_BOUND = "PENDING_BOUND"
    PENDING_UNLINKED = "PENDING_UNLINKED"
    PENDING_AMBIGUOUS = "PENDING_AMBIGUOUS"


class Diag5PhysicalCancellationCause(str, Enum):
    NONE = "NONE"
    CANCEL_UNLINK_FAILED = "CANCEL_UNLINK_FAILED"
    CANCEL_PARENT_FSYNC_FAILED = "CANCEL_PARENT_FSYNC_FAILED"
    CANCEL_REVALIDATION_FAILED = "CANCEL_REVALIDATION_FAILED"
    CANCEL_VISIBILITY_AMBIGUOUS = "CANCEL_VISIBILITY_AMBIGUOUS"


class Diag5PhysicalCancellationState(str, Enum):
    CANCELLED = "CANCELLED"
    SPENT = "SPENT"


@dataclass(frozen=True, slots=True)
class Diag5PhysicalCancellationObservation:
    cause: Diag5PhysicalCancellationCause
    state: Diag5PhysicalCancellationState
    evidence_namespace_state: Diag5EvidenceNamespaceState
    staging_path_state: Diag5PhysicalPathState
    final_path_state: Diag5PhysicalPathState
    rollback_path_state: Diag5PhysicalPathState


class Diag5PhysicalCancellationError(RuntimeError):
    """A spent pre-final evidence reservation with typed namespace evidence."""

    def __init__(
        self,
        observation: Diag5PhysicalCancellationObservation,
        cause: BaseException,
    ) -> None:
        super().__init__(observation.cause.value)
        self.observation = observation
        self.cause = cause


@dataclass(frozen=True, slots=True)
class Diag5RollbackObservation:
    rollback_cause: Diag5RollbackCause
    rollback_state: Diag5RollbackState
    final_path_state: Diag5PhysicalPathState
    rollback_path_state: Diag5PhysicalPathState
    evidence_namespace_state_at_seal: Diag5EvidenceNamespaceState


@dataclass(frozen=True, slots=True)
class Diag5ConsumedAuthority:
    path: Path
    sha256: str
    payload: Mapping[str, JsonValue]


@dataclass(slots=True)
class _Diag5AuthorityLease:
    repository: Path
    output_root: Path
    authority_path: Path
    authority_bytes: bytes
    locked_leaves: Mapping[Path, _Diag4LockedLeaf]
    directory_descriptors: Mapping[Path, int]
    cpu_native_claim: Diag5NativeExtensionClaim
    gpu_native_claim: Diag5NativeExtensionClaim
    staging_descriptor: int | None = None
    gpu_snapshot_root_descriptor: int | None = None
    consumption_marker_descriptor: int | None = None
    physical_evidence: _Diag5PhysicalEvidenceLease | None = None
    published_output: Diag5PublishedOutputKind | None = None
    rollback_attempted: bool = False
    consumed: Diag5ConsumedAuthority | None = None
    lifecycle: Diag5AuthorityLifecycle = Diag5AuthorityLifecycle.UNCONSUMED
    active: bool = True


@dataclass(slots=True)
class _Diag5PhysicalEvidenceLease:
    descriptor: int
    pending_name: str
    device: int
    inode: int
    active: bool = True


@dataclass(frozen=True, slots=True)
class Diag5PhysicalEvidenceReservation:
    """Opaque retained reservation for the sole DIAG5 physical-failure record."""

    _claim: Diag5SuccessorAuthorityClaim = field(repr=False, compare=False)
    _lease: _Diag5PhysicalEvidenceLease = field(repr=False, compare=False)


class Diag5FinalizerSourceKind(str, Enum):
    PUBLISHED_SNAPSHOT = "PUBLISHED_SNAPSHOT"
    PRE_SOURCE_FAILURE = "PRE_SOURCE_FAILURE"


class Diag5FinalizerFailureCategory(str, Enum):
    DEEP_LOAD = "DEEP_LOAD"
    REVALIDATION = "REVALIDATION"
    FINALIZATION = "FINALIZATION"


class Diag5FinalizerError(RuntimeError):
    """One classified finalizer failure that keeps rollback evidence usable."""

    def __init__(
        self, category: Diag5FinalizerFailureCategory, cause: BaseException
    ) -> None:
        super().__init__(category.value)
        self.category = category
        self.cause = cause


@dataclass(frozen=True, slots=True)
class PublishedSnapshot:
    kind: Diag5FinalizerSourceKind
    snapshot: SnapshotPublication

    def __post_init__(self) -> None:
        if self.kind is not Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT:
            raise ValueError("DIAG5 published-snapshot discriminant differs")


@dataclass(frozen=True, slots=True)
class PreSourceFailure:
    kind: Diag5FinalizerSourceKind
    outcome: StructuredFailureV5
    supervisor_terminal: ArtifactRef
    diagnostic_receipt: ArtifactRef

    def __post_init__(self) -> None:
        if self.kind is not Diag5FinalizerSourceKind.PRE_SOURCE_FAILURE:
            raise ValueError("DIAG5 pre-source-failure discriminant differs")


Diag5FinalizerSourceInput = PublishedSnapshot | PreSourceFailure


@dataclass(frozen=True, slots=True)
class Diag5SuccessorAuthorityClaim:
    payload: Mapping[str, JsonValue]
    authority_sha256: str
    plan_prefix_sha256: str
    completed_plan_sha256: str
    expected_gpu_uuid: str
    expected_numerical_identity: Mapping[str, str]
    expected_frozen_numerical_entries: Mapping[str, str]
    expected_gpu_output_root: Path
    expected_gpu_staging_root: Path
    expected_gpu_rollback_root: Path
    expected_cpu_qualification_root: Path
    expected_cpu_source_snapshot_entries: Mapping[str, tuple[str, int, str]]
    expected_gpu_source_snapshot_identity: SnapshotIdentity
    expected_native_copy_relative_path: str
    expected_copied_native_sha256: str
    expected_copied_native_size_bytes: int
    cpu_native_binding: Diag5NativeExtensionBinding
    gpu_native_binding: Diag5NativeExtensionBinding
    predecessor_postmortem: ArtifactRef
    expected_interpreter: Mapping[str, JsonValue]
    expected_native_reference: Mapping[str, JsonValue]
    expected_input_bundle: Mapping[str, JsonValue]
    _lease: _Diag5AuthorityLease = field(repr=False, compare=False)

    @property
    def expected_cpu_native_binding(self) -> Diag5NativeExtensionBinding:
        return self.cpu_native_binding

    @property
    def expected_gpu_native_binding(self) -> Diag5NativeExtensionBinding:
        return self.gpu_native_binding

    @property
    def expected_gpu_logical_source_root(self) -> Path:
        return self.expected_gpu_staging_root / "source-snapshot"


def diag5_consumption_marker_path(output_root: Path) -> Path:
    output = output_root.absolute()
    return output.parent / f".{output.name}.diag5-authority-consumed.json"


def _diag5_dirfd_name_absent(directory: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return True
    return False


def _diag5_artifact_ref(value: JsonValue, context: str) -> ArtifactRef:
    item = _mapping(value, context)
    _exact_keys(
        item,
        frozenset({"relative_path", "sha256", "size_bytes", "schema_version"}),
        context,
    )
    return ArtifactRef(
        _diag4_relative_path(item["relative_path"], f"{context}.relative_path"),
        _diag4_sha256(item["sha256"], f"{context}.sha256"),
        _integer(item["size_bytes"], f"{context}.size_bytes"),
        _string(item["schema_version"], f"{context}.schema_version"),
    )


def _diag5_native_binding(
    value: JsonValue,
    *,
    cpu: bool,
    locked_leaves: Mapping[Path, _Diag4LockedLeaf] | None = None,
) -> Diag5NativeExtensionBinding:
    item = _mapping(value, "DIAG5 native binding")
    prefix = "cpu" if cpu else "gpu"
    path_key = f"{prefix}_native_extension_path"
    link_key = f"{prefix}_native_extension_link_count"
    device_key = f"{prefix}_native_extension_device"
    inode_key = f"{prefix}_native_extension_inode"
    _exact_keys(
        item,
        frozenset(
            {
                path_key,
                "native_extension_sha256",
                "native_extension_size_bytes",
                link_key,
                device_key,
                inode_key,
            }
        ),
        "DIAG5 native binding",
    )
    path_text = _string(item[path_key], f"DIAG5 {prefix} native path")
    path = Path(path_text).resolve(strict=True)
    if path_text != str(path):
        raise ValueError(f"DIAG5 {prefix} native path is not canonical")
    binding = Diag5NativeExtensionBinding(
        path=path,
        sha256=_diag4_sha256(item["native_extension_sha256"], "DIAG5 native SHA"),
        size_bytes=_integer(item["native_extension_size_bytes"], "DIAG5 native size"),
        device=_integer(item[device_key], "DIAG5 native device"),
        inode=_integer(item[inode_key], "DIAG5 native inode"),
        link_count=_integer(item[link_key], "DIAG5 native link count"),
    )
    if locked_leaves is None:
        observed_binding = observe_diag5_native_extension_binding(path)
    else:
        leaf = locked_leaves[path]
        observed = os.fstat(leaf.descriptor)
        observed_binding = Diag5NativeExtensionBinding(
            path=path,
            sha256=hashlib.sha256(_diag4_descriptor_bytes(leaf.descriptor)).hexdigest(),
            size_bytes=observed.st_size,
            device=observed.st_dev,
            inode=observed.st_ino,
            link_count=observed.st_nlink,
        )
    if binding.link_count < 1 or observed_binding != binding:
        raise ValueError(f"DIAG5 {prefix} native binding differs")
    return binding


def _validate_diag5_cpu_qualification(
    reference: ArtifactRef,
    *,
    cpu_binding: Diag5NativeExtensionBinding,
    execution_manifest_sha256: str,
    execution_entries_sha256: str,
    expected_execution_entries: Mapping[str, tuple[str, int, str]],
    execution_manifest_size_bytes: int,
    expected_numerical_identity: Mapping[str, JsonValue],
    expected_input_bundle: Mapping[str, JsonValue],
    expected_native_reference_manifest_sha256: str,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
    locked_leaves: Mapping[Path, _Diag4LockedLeaf] | None,
) -> tuple[Mapping[str, tuple[str, int, str]], SnapshotIdentity]:
    if (
        reference.relative_path != "scientific-evidence.json"
        or reference.schema_version != DIAG5_CPU_QUALIFICATION_SCHEMA_VERSION
    ):
        raise ValueError("DIAG5 decisive CPU qualification reference differs")
    root = DIAG5_CPU_QUALIFICATION_ROOT
    manifest_bytes = _diag4_bound_file_bytes(
        root / "artifact-manifest.json",
        "DIAG5 CPU qualification manifest",
        locked_leaf_bytes,
    )
    directories, files, manifest_source_sha, manifest_entries_sha = (
        _diag4_cpu_manifest_entries(
            manifest_bytes,
            expected_schema=DIAG5_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION,
        )
    )
    observed_files: set[str] = set()
    observed_directories: set[str] = set()
    for path in root.rglob("*"):
        metadata = path.lstat()
        relative = path.relative_to(root).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("DIAG5 CPU qualification contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o555:
                raise ValueError("DIAG5 CPU qualification directory mode differs")
            observed_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o444 or metadata.st_nlink != 1:
                raise ValueError("DIAG5 CPU qualification file topology differs")
            observed_files.add(relative)
        else:
            raise ValueError("DIAG5 CPU qualification contains a special entry")
    expected_files = {relative for relative, _digest, _size in files} | {
        "artifact-manifest.json"
    }
    if observed_files != expected_files or observed_directories != set(directories):
        raise ValueError("DIAG5 CPU qualification tree closure differs")
    if (
        manifest_source_sha != execution_manifest_sha256
        or manifest_entries_sha != execution_entries_sha256
    ):
        raise ValueError("DIAG5 CPU qualification source identity differs")
    for relative, digest, size_bytes in files:
        payload = _diag4_bound_file_bytes(
            root / relative,
            f"DIAG5 CPU qualification file {relative}",
            locked_leaf_bytes,
        )
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"DIAG5 CPU qualification file differs: {relative}")
    scientific_bytes = _diag4_bound_file_bytes(
        root / reference.relative_path,
        "DIAG5 CPU scientific evidence",
        locked_leaf_bytes,
    )
    if (
        len(scientific_bytes) != reference.size_bytes
        or hashlib.sha256(scientific_bytes).hexdigest() != reference.sha256
    ):
        raise ValueError("DIAG5 CPU scientific evidence reference differs")
    scientific = _mapping(
        load_canonical_json_bytes(scientific_bytes), "DIAG5 CPU scientific evidence"
    )
    _exact_keys(
        scientific,
        frozenset(
            {
                "backend",
                "callback_count",
                "configuration_fingerprint",
                "execution_source_entries_sha256",
                "execution_source_manifest_sha256",
                "input_fingerprint",
                "cpu_native_binding",
                "native_reference_artifact_sha256",
                "numerical_identity",
                "output_root",
                "policy_sha256",
                "prequalification_plan_control",
                "predecessor_postmortem",
                "promotion_eligible",
                "qualification_passed",
                "route",
                "runtime",
                "schema_version",
                "scientific_outcome",
                "source_manifest_entries",
                "source_manifest_sha256",
                "speed",
                "synchronized_solve_seconds",
                "timings_monotonic_ns",
            }
        ),
        "DIAG5 CPU scientific evidence",
    )
    scientific_cpu = _diag5_native_binding(
        scientific["cpu_native_binding"], cpu=True, locked_leaves=locked_leaves
    )
    scientific_postmortem = _diag5_artifact_ref(
        scientific["predecessor_postmortem"], "DIAG5 CPU predecessor postmortem"
    )
    runtime = _mapping(scientific["runtime"], "DIAG5 CPU runtime")
    if (
        scientific.get("schema_version") != DIAG5_CPU_QUALIFICATION_SCHEMA_VERSION
        or scientific.get("route") != DIAG4_NUMERICAL_ROUTE
        or scientific.get("backend") != "cpu"
        or scientific.get("output_root") != str(root)
        or scientific.get("qualification_passed") is not True
        or scientific.get("promotion_eligible") is not False
        or scientific.get("scientific_outcome") != "QUALITY_HIT"
        or scientific.get("speed") != "NOT_PRODUCED"
        or scientific.get("execution_source_manifest_sha256")
        != execution_manifest_sha256
        or scientific.get("execution_source_entries_sha256") != execution_entries_sha256
        or scientific.get("numerical_identity") != expected_numerical_identity
        or scientific.get("input_fingerprint")
        != expected_input_bundle["input_fingerprint"]
        or scientific.get("configuration_fingerprint")
        != expected_input_bundle["configuration_fingerprint"]
        or scientific.get("native_reference_artifact_sha256")
        != expected_native_reference_manifest_sha256
        or scientific.get("callback_count") != 0
        or runtime.get("backend") != "cpu"
        or runtime.get("x64_enabled") is not True
        or runtime.get("native_extension_path") != str(cpu_binding.path)
        or runtime.get("native_extension_sha256") != cpu_binding.sha256
        or runtime.get("native_extension_size_bytes") != cpu_binding.size_bytes
        or runtime.get("native_extension_link_count") != cpu_binding.link_count
        or scientific_cpu != cpu_binding
        or scientific_postmortem.sha256 != DIAG5_PREDECESSOR_POSTMORTEM_SHA256
    ):
        raise ValueError("DIAG5 CPU scientific result differs")
    if locked_leaf_bytes is None:
        snapshot = load_snapshot(
            root / "source-snapshot", required_roles=DIAG5_CPU_SNAPSHOT_ROLES
        )
        cpu_snapshot_identity = snapshot.identity()
        source_entries = [
            {
                "relative_path": entry.relative_path,
                "role": entry.role,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in snapshot.entries
        ]
        source_manifest_sha256 = snapshot.manifest_sha256
    else:
        source_manifest_bytes = _diag4_bound_file_bytes(
            root / "source-snapshot/source-manifest.json",
            "DIAG5 CPU source manifest",
            locked_leaf_bytes,
        )
        source_manifest = _mapping(
            load_canonical_json_bytes(source_manifest_bytes),
            "DIAG5 CPU source manifest",
        )
        _exact_keys(
            source_manifest,
            frozenset({"entries", "schema_version", "worktree"}),
            "DIAG5 CPU source manifest",
        )
        if source_manifest["schema_version"] != SOURCE_MANIFEST_SCHEMA_VERSION:
            raise ValueError("DIAG5 CPU source-manifest schema differs")
        worktree = _mapping(
            source_manifest["worktree"], "DIAG5 CPU source-manifest worktree"
        )
        _exact_keys(
            worktree,
            frozenset(
                {
                    "git_head",
                    "repo_root",
                    "tracked_diff_sha256",
                    "untracked_bytes_manifest_sha256",
                }
            ),
            "DIAG5 CPU source-manifest worktree",
        )
        git_head = _string(worktree["git_head"], "DIAG5 CPU source git HEAD")
        repo_root = _string(worktree["repo_root"], "DIAG5 CPU source repository root")
        if (
            len(git_head) != 40
            or any(character not in "0123456789abcdef" for character in git_head)
            or not Path(repo_root).is_absolute()
        ):
            raise ValueError("DIAG5 CPU source worktree identity differs")
        _diag4_sha256(worktree["tracked_diff_sha256"], "DIAG5 CPU tracked diff SHA")
        _diag4_sha256(
            worktree["untracked_bytes_manifest_sha256"],
            "DIAG5 CPU untracked bytes manifest SHA",
        )
        raw_source_entries = source_manifest["entries"]
        if not isinstance(raw_source_entries, list):
            raise TypeError("DIAG5 CPU source entries must be an array")
        source_entries = []
        for raw_entry in raw_source_entries:
            entry = _mapping(raw_entry, "DIAG5 CPU source entry")
            _exact_keys(
                entry,
                frozenset({"relative_path", "role", "sha256", "size_bytes"}),
                "DIAG5 CPU source entry",
            )
            relative = _diag4_relative_path(
                entry["relative_path"], "DIAG5 CPU source relative path"
            )
            source_bytes = _diag4_bound_file_bytes(
                root / "source-snapshot" / relative,
                f"DIAG5 CPU source {relative}",
                locked_leaf_bytes,
            )
            if (
                len(source_bytes) != entry["size_bytes"]
                or hashlib.sha256(source_bytes).hexdigest() != entry["sha256"]
            ):
                raise ValueError("DIAG5 CPU held source entry differs")
            source_entries.append(dict(entry))
        source_paths = [
            _string(entry["relative_path"], "DIAG5 CPU source path")
            for entry in source_entries
        ]
        if source_paths != sorted(source_paths) or len(source_paths) != len(
            set(source_paths)
        ):
            raise ValueError("DIAG5 CPU source paths are not sorted and unique")
        source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
        cpu_snapshot_identity = build_snapshot_identity(
            tuple(
                SnapshotEntry(
                    _string(entry["role"], "DIAG5 CPU source role"),
                    _string(entry["relative_path"], "DIAG5 CPU source path"),
                    _integer(entry["size_bytes"], "DIAG5 CPU source size"),
                    _string(entry["sha256"], "DIAG5 CPU source SHA"),
                )
                for entry in source_entries
            ),
            WorktreeIdentity(
                git_head,
                _string(
                    worktree["tracked_diff_sha256"],
                    "DIAG5 CPU tracked diff SHA",
                ),
                _string(
                    worktree["untracked_bytes_manifest_sha256"],
                    "DIAG5 CPU untracked bytes manifest SHA",
                ),
                repo_root,
            ),
        )
        if cpu_snapshot_identity.manifest_sha256 != source_manifest_sha256:
            raise ValueError("DIAG5 held CPU snapshot identity differs")
    observed = {
        _string(entry["relative_path"], "DIAG5 CPU source path"): (
            _string(entry["sha256"], "DIAG5 CPU source SHA"),
            _integer(entry["size_bytes"], "DIAG5 CPU source size"),
            _string(entry["role"], "DIAG5 CPU source role"),
        )
        for entry in source_entries
    }
    if (
        scientific["source_manifest_sha256"] != source_manifest_sha256
        or scientific["source_manifest_entries"] != source_entries
    ):
        raise ValueError("DIAG5 CPU source-manifest summary differs")
    plan_control = _mapping(
        scientific["prequalification_plan_control"],
        "DIAG5 prequalification plan control",
    )
    if plan_control != {
        "schema_version": DIAG5_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION,
        "snapshot_relative_path": "control/prequalification-plan.md",
        "source_relative_path": DIAG5_PLAN_RELATIVE_PATH,
        "sha256": DIAG5_BLANK_PLAN_SHA256,
        "size_bytes": DIAG5_BLANK_PLAN_SIZE_BYTES,
        "plan_prefix_sha256": DIAG5_PLAN_SHA256,
    }:
        raise ValueError("DIAG5 prequalification plan control differs")
    if locked_leaf_bytes is None:
        validate_diag5_predecessor_postmortem_artifact(root, scientific_postmortem)
    else:
        postmortem_bytes = _diag4_bound_file_bytes(
            root / scientific_postmortem.relative_path,
            "DIAG5 held predecessor postmortem artifact",
            locked_leaf_bytes,
        )
        if (
            len(postmortem_bytes) != scientific_postmortem.size_bytes
            or hashlib.sha256(postmortem_bytes).hexdigest()
            != scientific_postmortem.sha256
            or _mapping(
                load_canonical_json_bytes(postmortem_bytes),
                "DIAG5 held predecessor postmortem",
            )["schema_version"]
            != DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        ):
            raise ValueError("DIAG5 held predecessor postmortem differs")
    expected = dict(expected_execution_entries)
    expected[DIAG4_EXECUTION_SOURCE_MANIFEST_PATH] = (
        execution_manifest_sha256,
        execution_manifest_size_bytes,
        "execution_source_manifest",
    )
    expected["control/prequalification-plan.md"] = (
        DIAG5_BLANK_PLAN_SHA256,
        DIAG5_BLANK_PLAN_SIZE_BYTES,
        "prequalification_plan",
    )
    expected[DIAG5_NATIVE_COPY_RELATIVE_PATH] = (
        cpu_binding.sha256,
        cpu_binding.size_bytes,
        "native_extension",
    )
    if (
        len(source_entries) != DIAG5_EXECUTION_SOURCE_ENTRY_COUNT + 3
        or observed != expected
    ):
        raise ValueError("DIAG5 CPU source snapshot cardinality differs")
    if not directories:
        raise ValueError("DIAG5 CPU qualification manifest has no directories")
    return MappingProxyType(observed), project_diag5_gpu_snapshot_identity(
        cpu_snapshot_identity
    )


def _validate_diag5_review_map(
    value: JsonValue,
    aggregate: JsonValue,
    *,
    phase: str,
    qualified_sha256: str,
    frozen_sha256: str,
    execution_manifest_sha256: str,
    execution_entries_sha256: str,
    cpu_qualification: ArtifactRef | None,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
    locked_leaves: Mapping[Path, _Diag4LockedLeaf] | None,
) -> Mapping[str, ArtifactRef]:
    raw = _mapping(value, f"DIAG5 {phase} reviews")
    roles = (
        "numerical-controller",
        "receipt-schema",
        "source-snapshot",
        "atomic-lifecycle",
    )
    if (
        frozenset(raw) != frozenset(roles)
        or aggregate != hashlib.sha256(canonical_json_bytes(raw)).hexdigest()
    ):
        raise ValueError(f"DIAG5 {phase} review map differs")
    schema = f"single-stage-neq-gntr3-diag5-{phase.lower().replace('_', '-')}-review-v1"
    references: dict[str, ArtifactRef] = {}
    identities: set[tuple[str, str]] = set()
    for role in roles:
        reference = _diag5_artifact_ref(raw[role], f"DIAG5 {phase} review {role}")
        expected_relative = f"{phase.lower().replace('_', '-')}/{role}.json"
        if (
            reference.relative_path != expected_relative
            or reference.schema_version != schema
        ):
            raise ValueError(f"DIAG5 {phase} review reference differs")
        payload_bytes = _diag4_bound_file_bytes(
            review_path := DIAG5_REVIEW_ROOT / reference.relative_path,
            f"DIAG5 {phase} review {role}",
            locked_leaf_bytes,
        )
        metadata = (
            os.fstat(locked_leaves[review_path].descriptor)
            if locked_leaves is not None
            else review_path.lstat()
        )
        if stat.S_IMODE(metadata.st_mode) != 0o444 or metadata.st_nlink != 1:
            raise ValueError(f"DIAG5 {phase} review topology differs")
        if (
            len(payload_bytes) != reference.size_bytes
            or hashlib.sha256(payload_bytes).hexdigest() != reference.sha256
        ):
            raise ValueError(f"DIAG5 {phase} review bytes differ")
        review = _mapping(
            load_canonical_json_bytes(payload_bytes), f"DIAG5 {phase} review"
        )
        common = {
            "schema_version",
            "phase",
            "role",
            "reviewer",
            "session",
            "verdict",
            "qualified_files_sha256",
            "frozen_numerical_entries_sha256",
            "execution_source_manifest_sha256",
            "execution_source_entries_sha256",
            "predecessor_postmortem_sha256",
            "predecessor_full_tree_sha256",
            "blank_plan_sha256",
            "plan_prefix_sha256",
        }
        expected_keys = common | (
            {"cpu_qualification_manifest_sha256", "cpu_scientific_evidence_sha256"}
            if phase == "POST_RUN"
            else set()
        )
        _exact_keys(review, frozenset(expected_keys), f"DIAG5 {phase} review")
        identity = (
            _string(review["reviewer"], "reviewer"),
            _string(review["session"], "session"),
        )
        if identity in identities:
            raise ValueError("DIAG5 reviewer identity is duplicated")
        identities.add(identity)
        if (
            review["schema_version"] != schema
            or review["phase"] != phase
            or review["role"] != role
            or review["verdict"] != "GO"
            or review["qualified_files_sha256"] != qualified_sha256
            or review["frozen_numerical_entries_sha256"] != frozen_sha256
            or review["execution_source_manifest_sha256"] != execution_manifest_sha256
            or review["execution_source_entries_sha256"] != execution_entries_sha256
            or review["predecessor_postmortem_sha256"]
            != DIAG5_PREDECESSOR_POSTMORTEM_SHA256
            or review["blank_plan_sha256"] != DIAG5_BLANK_PLAN_SHA256
            or review["plan_prefix_sha256"] != DIAG5_PLAN_SHA256
            or (
                phase == "POST_RUN"
                and review["cpu_scientific_evidence_sha256"] != cpu_qualification.sha256
            )
            or (
                phase == "POST_RUN"
                and review["cpu_qualification_manifest_sha256"]
                != hashlib.sha256(
                    _diag4_bound_file_bytes(
                        DIAG5_CPU_QUALIFICATION_ROOT / "artifact-manifest.json",
                        "DIAG5 CPU qualification manifest",
                        locked_leaf_bytes,
                    )
                ).hexdigest()
            )
        ):
            raise ValueError(f"DIAG5 {phase} review identity differs")
        references[role] = reference
    return MappingProxyType(references)


def _validate_diag5_execution_policy(value: JsonValue) -> None:
    execution = _mapping(value, "DIAG5 execution policy")
    expected = {
        "parent_platform": "cpu",
        "child_platform": "cuda",
        "jax_enable_x64": True,
        "compilation_cache_enabled": False,
        "child_preallocate": True,
        "command_buffer_enabled": False,
        "required_xla_flag": "--xla_gpu_enable_command_buffer=",
    }
    _exact_keys(execution, frozenset(expected), "DIAG5 execution policy")
    if execution != expected:
        raise ValueError("DIAG5 execution policy differs")


def _validate_diag5_input_bundle_fingerprints(
    expected: Mapping[str, JsonValue],
    *,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
) -> None:
    document = _mapping(
        load_canonical_json_bytes(
            _diag4_bound_file_bytes(
                DIAG5_INPUT_ROOT / "input_bundle.json",
                "DIAG5 input bundle document",
                locked_leaf_bytes,
            )
        ),
        "DIAG5 input bundle document",
    )
    configuration = _mapping(
        document["configuration"], "DIAG5 input bundle configuration"
    )
    configuration_sha = hashlib.sha256(canonical_json_bytes(configuration)).hexdigest()
    arrays = _mapping(document["arrays"], "DIAG5 input bundle arrays")
    normalized_arrays: dict[str, JsonValue] = {}
    for name, raw_reference in arrays.items():
        reference = _mapping(raw_reference, f"DIAG5 input array {name}")
        _exact_keys(
            reference,
            frozenset({"dtype", "order", "path", "sha256", "shape"}),
            f"DIAG5 input array {name}",
        )
        relative = _diag4_relative_path(
            reference["path"], f"DIAG5 input array {name} path"
        )
        payload = _diag4_bound_file_bytes(
            DIAG5_INPUT_ROOT / relative,
            f"DIAG5 input array {name}",
            locked_leaf_bytes,
        )
        if hashlib.sha256(payload).hexdigest() != reference["sha256"]:
            raise ValueError(f"DIAG5 input array differs: {name}")
        normalized_arrays[name] = dict(reference)
    fingerprint_payload: dict[str, JsonValue] = {
        "case_id": document["case_id"],
        "random_seed": document["random_seed"],
        "configuration_fingerprint": configuration_sha,
        "arrays": normalized_arrays,
    }
    if document["schema_version"] == 2:
        fingerprint_payload["scale"] = document["scale"]
    input_sha = hashlib.sha256(canonical_json_bytes(fingerprint_payload)).hexdigest()
    if (
        document["configuration_fingerprint"] != configuration_sha
        or document["input_fingerprint"] != input_sha
        or expected["configuration_fingerprint"] != configuration_sha
        or expected["input_fingerprint"] != input_sha
    ):
        raise ValueError("DIAG5 input bundle fingerprint differs")


_DIAG5_AUTHORITY_KEYS: Final = frozenset(
    {
        "schema_version",
        "route",
        "numerical_route",
        "scientific_evidence_schema",
        "plan_prefix_sha256",
        "completed_plan_sha256",
        "qualification_record_sha256",
        "qualified_files",
        "qualified_files_sha256",
        "frozen_numerical_entries",
        "frozen_numerical_entries_sha256",
        "execution_source_manifest_sha256",
        "execution_source_entries_sha256",
        "predecessor_postmortem",
        "predecessor_full_tree_sha256",
        "decisive_cpu_qualification",
        "pre_run_reviews",
        "pre_run_reviews_sha256",
        "post_run_reviews",
        "post_run_reviews_sha256",
        "cpu_native_binding",
        "gpu_native_binding",
        "native_reference",
        "input_bundle",
        "consumed_diag3",
        "numerical_identity",
        "interpreter",
        "roots",
        "gpu_uuid",
        "execution_policy",
        "launch",
    }
)


def _validate_diag5_authority_payload(
    payload: Mapping[str, JsonValue],
    *,
    repository: Path,
    output_root: Path,
    locked_leaves: Mapping[Path, _Diag4LockedLeaf] | None = None,
) -> tuple[
    Diag5NativeExtensionBinding,
    Diag5NativeExtensionBinding,
    ArtifactRef,
    ArtifactRef,
    Mapping[str, tuple[str, int, str]],
    SnapshotIdentity,
]:
    _exact_keys(payload, _DIAG5_AUTHORITY_KEYS, "DIAG5 authority")
    if (
        payload["schema_version"] != DIAG5_AUTHORITY_SCHEMA_VERSION
        or payload["route"] != DIAG5_ROUTE
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["scientific_evidence_schema"]
        != DIAG5_SCIENTIFIC_EVIDENCE_SCHEMA_VERSION
        or payload["plan_prefix_sha256"] != DIAG5_PLAN_SHA256
        or payload["gpu_uuid"] != DIAG4_GPU_UUID
    ):
        raise ValueError("DIAG5 authority identity differs")
    for name in (
        "completed_plan_sha256",
        "qualification_record_sha256",
        "qualified_files_sha256",
        "frozen_numerical_entries_sha256",
        "execution_source_manifest_sha256",
        "execution_source_entries_sha256",
        "predecessor_full_tree_sha256",
        "pre_run_reviews_sha256",
        "post_run_reviews_sha256",
    ):
        _diag4_sha256(payload[name], f"DIAG5 authority.{name}")
    _validate_diag5_execution_policy(payload["execution_policy"])
    _diag4_numerical_identity(payload["numerical_identity"])
    qualified = _diag4_qualified_files(payload["qualified_files"])
    frozen = _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
    if frozenset(qualified) != DIAG5_QUALIFIED_FILE_PATHS:
        raise ValueError("DIAG5 qualified membership differs")
    if frozenset(frozen) != DIAG5_FROZEN_NUMERICAL_PATHS:
        raise ValueError("DIAG5 frozen membership differs")
    for name, values in (
        ("qualified_files", qualified),
        ("frozen_numerical_entries", frozen),
    ):
        aggregate = hashlib.sha256(canonical_json_bytes(values)).hexdigest()
        if payload[f"{name}_sha256"] != aggregate:
            raise ValueError(f"DIAG5 {name} aggregate differs")
        for relative, digest in values.items():
            if (
                hashlib.sha256(
                    _diag4_bound_file_bytes(
                        repository / relative,
                        f"DIAG5 source {relative}",
                        None
                        if locked_leaves is None
                        else {
                            path: _diag4_descriptor_bytes(leaf.descriptor)
                            for path, leaf in locked_leaves.items()
                        },
                    )
                ).hexdigest()
                != digest
            ):
                raise ValueError(f"DIAG5 source differs: {relative}")
    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    held_bytes = (
        None
        if locked_leaves is None
        else {
            path: _diag4_descriptor_bytes(leaf.descriptor)
            for path, leaf in locked_leaves.items()
        }
    )
    manifest_bytes = _diag4_bound_file_bytes(
        manifest_path, "DIAG5 execution-source manifest", held_bytes
    )
    execution_entries, execution_sizes, execution_entries_sha256 = (
        _diag4_execution_source_entries(
            manifest_bytes,
            repository=repository,
            qualified=qualified,
            frozen=frozen,
            locked_leaf_bytes=held_bytes,
            expected_count=DIAG5_EXECUTION_SOURCE_ENTRY_COUNT,
        )
    )
    if (
        hashlib.sha256(manifest_bytes).hexdigest()
        != payload["execution_source_manifest_sha256"]
        or execution_entries_sha256 != payload["execution_source_entries_sha256"]
        or len(execution_entries) != DIAG5_EXECUTION_SOURCE_ENTRY_COUNT
    ):
        raise ValueError("DIAG5 execution-source authority differs")
    plan = repository / DIAG5_PLAN_RELATIVE_PATH
    plan_bytes = _diag4_bound_file_bytes(plan, "DIAG5 completed plan", held_bytes)
    prefix, marker, record = plan_bytes.partition(b"## Qualification Record\n")
    if (
        not marker
        or not record
        or hashlib.sha256(prefix).hexdigest() != DIAG5_PLAN_SHA256
        or hashlib.sha256(plan_bytes).hexdigest() != payload["completed_plan_sha256"]
        or hashlib.sha256(record).hexdigest() != payload["qualification_record_sha256"]
    ):
        raise ValueError("DIAG5 completed plan or qualification record differs")
    record_payload = _mapping(
        load_canonical_json_bytes(record), "DIAG5 qualification record"
    )
    _exact_keys(
        record_payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_prefix_sha256",
                "blank_plan_sha256",
                "qualified_files_sha256",
                "frozen_numerical_entries_sha256",
                "execution_source_manifest_sha256",
                "execution_source_entries_sha256",
                "predecessor_postmortem",
                "predecessor_full_tree_sha256",
                "cpu_qualification",
                "pre_run_reviews",
                "pre_run_reviews_sha256",
                "post_run_reviews",
                "post_run_reviews_sha256",
                "verdict",
            }
        ),
        "DIAG5 qualification record",
    )
    if canonical_json_bytes(record_payload) != record:
        raise ValueError("DIAG5 qualification record is not canonical")
    roots = _mapping(payload["roots"], "DIAG5 roots")
    _exact_keys(
        roots,
        frozenset(
            {
                "cpu_qualification_root",
                "gpu_output_root",
                "gpu_staging_root",
                "gpu_rollback_root",
                "consumption_marker",
            }
        ),
        "DIAG5 roots",
    )
    expected_roots = {
        "cpu_qualification_root": DIAG5_CPU_QUALIFICATION_ROOT,
        "gpu_output_root": output_root,
        "gpu_staging_root": DIAG5_GPU_STAGING_ROOT,
        "gpu_rollback_root": DIAG5_GPU_ROLLBACK_ROOT,
        "consumption_marker": diag5_consumption_marker_path(output_root),
    }
    if any(roots[name] != str(path) for name, path in expected_roots.items()):
        raise ValueError("DIAG5 roots differ")
    launch = _mapping(payload["launch"], "DIAG5 launch")
    if launch != {
        "preflight_exact": 1,
        "cold_max": 1,
        "warm_exact": 0,
        "retry_allowed": False,
    }:
        raise ValueError("DIAG5 launch cardinality differs")
    cpu = _diag5_native_binding(
        payload["cpu_native_binding"], cpu=True, locked_leaves=locked_leaves
    )
    gpu = _diag5_native_binding(
        payload["gpu_native_binding"], cpu=False, locked_leaves=locked_leaves
    )
    validate_diag5_cross_runtime_native_bindings(cpu, gpu)
    postmortem = _diag5_artifact_ref(
        payload["predecessor_postmortem"], "DIAG5 predecessor postmortem"
    )
    if postmortem.sha256 != DIAG5_PREDECESSOR_POSTMORTEM_SHA256:
        raise ValueError("DIAG5 predecessor postmortem differs")
    if locked_leaves is None:
        predecessor = validate_diag5_predecessor_failure(
            DIAG5_FAILED_DIAG4_PARTIAL_ROOT,
            repository_root=repository,
            postmortem_path=repository / DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH,
        )
        if (
            payload["predecessor_full_tree_sha256"]
            != predecessor.predecessor_full_tree_sha256
        ):
            raise ValueError("DIAG5 predecessor full-tree identity differs")
    else:
        predecessor_execution = DIAG5_FAILED_DIAG4_PARTIAL_ROOT / "execution-source"
        predecessor_manifest_path = (
            DIAG5_FAILED_DIAG4_PARTIAL_ROOT
            / DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_RELATIVE_PATH
        )
        predecessor_manifest_bytes = _diag4_descriptor_bytes(
            locked_leaves[predecessor_manifest_path].descriptor
        )
        predecessor_manifest = _mapping(
            load_canonical_json_bytes(predecessor_manifest_bytes),
            "DIAG5 held predecessor manifest",
        )
        predecessor_entries = _mapping(
            predecessor_manifest["entries"], "DIAG5 held predecessor entries"
        )
        observed_predecessor: dict[str, JsonValue] = {}
        for relative in (
            *predecessor_entries,
            DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
        ):
            leaf = locked_leaves[predecessor_execution / relative]
            metadata = os.fstat(leaf.descriptor)
            observed_predecessor[relative] = {
                "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
                "nlink": metadata.st_nlink,
                "sha256": hashlib.sha256(
                    _diag4_descriptor_bytes(leaf.descriptor)
                ).hexdigest(),
                "size_bytes": metadata.st_size,
            }
        if (
            len(predecessor_entries) != DIAG5_FAILED_DIAG4_EXECUTION_SOURCE_ENTRY_COUNT
            or predecessor_manifest["entries_sha256"]
            != DIAG5_FAILED_DIAG4_EXECUTION_ENTRIES_SHA256
            or hashlib.sha256(canonical_json_bytes(observed_predecessor)).hexdigest()
            != payload["predecessor_full_tree_sha256"]
        ):
            raise ValueError("DIAG5 held predecessor full-tree identity differs")
    decisive_cpu = _diag5_artifact_ref(
        payload["decisive_cpu_qualification"],
        "DIAG5 decisive CPU qualification",
    )
    cpu_source_snapshot, gpu_source_snapshot_identity = (
        _validate_diag5_cpu_qualification(
            decisive_cpu,
            cpu_binding=cpu,
            execution_manifest_sha256=_string(
                payload["execution_source_manifest_sha256"],
                "DIAG5 execution-source manifest SHA",
            ),
            execution_entries_sha256=_string(
                payload["execution_source_entries_sha256"],
                "DIAG5 execution-source entries SHA",
            ),
            expected_execution_entries=MappingProxyType(
                {
                    relative: (
                        digest,
                        execution_sizes[relative],
                        "test"
                        if relative.startswith("tests/")
                        else "benchmark"
                        if relative.startswith("benchmarks/")
                        else "execution_source",
                    )
                    for relative, digest in execution_entries.items()
                }
            ),
            execution_manifest_size_bytes=len(manifest_bytes),
            expected_numerical_identity=_mapping(
                payload["numerical_identity"], "DIAG5 numerical identity"
            ),
            expected_input_bundle=_mapping(
                payload["input_bundle"], "DIAG5 input bundle"
            ),
            expected_native_reference_manifest_sha256=_string(
                _mapping(payload["native_reference"], "DIAG5 native reference")[
                    "manifest_sha256"
                ],
                "DIAG5 native-reference manifest SHA",
            ),
            locked_leaf_bytes=held_bytes,
            locked_leaves=locked_leaves,
        )
    )
    review_files: set[str] = set()
    review_directories: set[str] = set()
    for path in DIAG5_REVIEW_ROOT.rglob("*"):
        metadata = path.lstat()
        relative = path.relative_to(DIAG5_REVIEW_ROOT).as_posix()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("DIAG5 review tree contains a symlink")
        if stat.S_ISDIR(metadata.st_mode):
            if stat.S_IMODE(metadata.st_mode) != 0o555:
                raise ValueError("DIAG5 review directory mode differs")
            review_directories.add(relative)
        elif stat.S_ISREG(metadata.st_mode):
            review_files.add(relative)
        else:
            raise ValueError("DIAG5 review tree contains a special entry")
    expected_review_files = {
        f"{phase}/{role}.json"
        for phase in ("pre-run", "post-run")
        for role in (
            "numerical-controller",
            "receipt-schema",
            "source-snapshot",
            "atomic-lifecycle",
        )
    }
    if review_files != expected_review_files or review_directories != {
        "pre-run",
        "post-run",
    }:
        raise ValueError("DIAG5 review tree closure differs")
    pre_reviews = _validate_diag5_review_map(
        payload["pre_run_reviews"],
        payload["pre_run_reviews_sha256"],
        phase="PRE_RUN",
        qualified_sha256=_string(payload["qualified_files_sha256"], "qualified SHA"),
        frozen_sha256=_string(payload["frozen_numerical_entries_sha256"], "frozen SHA"),
        execution_manifest_sha256=_string(
            payload["execution_source_manifest_sha256"], "manifest SHA"
        ),
        execution_entries_sha256=_string(
            payload["execution_source_entries_sha256"], "entries SHA"
        ),
        cpu_qualification=None,
        locked_leaf_bytes=held_bytes,
        locked_leaves=locked_leaves,
    )
    post_reviews = _validate_diag5_review_map(
        payload["post_run_reviews"],
        payload["post_run_reviews_sha256"],
        phase="POST_RUN",
        qualified_sha256=_string(payload["qualified_files_sha256"], "qualified SHA"),
        frozen_sha256=_string(payload["frozen_numerical_entries_sha256"], "frozen SHA"),
        execution_manifest_sha256=_string(
            payload["execution_source_manifest_sha256"], "manifest SHA"
        ),
        execution_entries_sha256=_string(
            payload["execution_source_entries_sha256"], "entries SHA"
        ),
        cpu_qualification=decisive_cpu,
        locked_leaf_bytes=held_bytes,
        locked_leaves=locked_leaves,
    )
    if set(pre_reviews) != set(post_reviews):
        raise ValueError("DIAG5 review role sets differ")
    record_expected = {
        "schema_version": DIAG5_QUALIFICATION_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "plan_prefix_sha256": DIAG5_PLAN_SHA256,
        "blank_plan_sha256": DIAG5_BLANK_PLAN_SHA256,
        "qualified_files_sha256": payload["qualified_files_sha256"],
        "frozen_numerical_entries_sha256": payload["frozen_numerical_entries_sha256"],
        "execution_source_manifest_sha256": payload["execution_source_manifest_sha256"],
        "execution_source_entries_sha256": payload["execution_source_entries_sha256"],
        "predecessor_postmortem": payload["predecessor_postmortem"],
        "predecessor_full_tree_sha256": payload["predecessor_full_tree_sha256"],
        "cpu_qualification": payload["decisive_cpu_qualification"],
        "pre_run_reviews": payload["pre_run_reviews"],
        "pre_run_reviews_sha256": payload["pre_run_reviews_sha256"],
        "post_run_reviews": payload["post_run_reviews"],
        "post_run_reviews_sha256": payload["post_run_reviews_sha256"],
        "verdict": "GO",
    }
    if record_payload != record_expected:
        raise ValueError("DIAG5 qualification record authority differs")
    interpreter = _mapping(payload["interpreter"], "DIAG5 interpreter")
    _exact_keys(
        interpreter,
        frozenset({"absolute_path", "sha256", "size_bytes"}),
        "DIAG5 interpreter",
    )
    interpreter_bytes = _diag4_bound_file_bytes(
        DIAG5_GPU_INTERPRETER,
        "DIAG5 interpreter",
        held_bytes,
    )
    if (
        interpreter["absolute_path"] != str(DIAG5_GPU_INTERPRETER)
        or interpreter["sha256"] != hashlib.sha256(interpreter_bytes).hexdigest()
        or interpreter["size_bytes"] != len(interpreter_bytes)
    ):
        raise ValueError("DIAG5 interpreter differs")
    native_reference = _mapping(payload["native_reference"], "DIAG5 native reference")
    native_manifest = DIAG5_NATIVE_REFERENCE_ROOT / "artifact-manifest.json"
    native_manifest_bytes = _diag4_bound_file_bytes(
        native_manifest, "DIAG5 native-reference manifest", held_bytes
    )
    if native_reference != {
        "absolute_root": str(DIAG5_NATIVE_REFERENCE_ROOT),
        "manifest_sha256": hashlib.sha256(native_manifest_bytes).hexdigest(),
    }:
        raise ValueError("DIAG5 native reference differs")
    _validate_diag4_native_tree(
        native_manifest_bytes,
        reference_root=DIAG5_NATIVE_REFERENCE_ROOT,
        locked_leaf_bytes=held_bytes,
    )
    input_bundle = _mapping(payload["input_bundle"], "DIAG5 input bundle")
    _exact_keys(
        input_bundle,
        frozenset({"absolute_root", "input_fingerprint", "configuration_fingerprint"}),
        "DIAG5 input bundle",
    )
    if input_bundle["absolute_root"] != str(DIAG5_INPUT_ROOT):
        raise ValueError("DIAG5 input bundle differs")
    _diag4_sha256(input_bundle["input_fingerprint"], "DIAG5 input fingerprint")
    _diag4_sha256(
        input_bundle["configuration_fingerprint"], "DIAG5 configuration fingerprint"
    )
    _validate_diag5_input_bundle_fingerprints(
        input_bundle, locked_leaf_bytes=held_bytes
    )
    _diag4_consumed_manifest(
        payload["consumed_diag3"],
        consumed_root=DIAG4_CONSUMED_DIAG3_ROOT,
        locked_leaf_bytes=held_bytes,
    )
    return (
        cpu,
        gpu,
        postmortem,
        decisive_cpu,
        cpu_source_snapshot,
        gpu_source_snapshot_identity,
    )


def validate_diag5_successor_authority(
    authority_path: Path, *, repository_root: Path, output_root: Path
) -> Mapping[str, JsonValue]:
    repository = repository_root.resolve(strict=True)
    expected = repository / DIAG5_AUTHORITY_RELATIVE_PATH
    if (
        authority_path.resolve(strict=True) != expected
        or output_root.absolute() != DIAG5_GPU_OUTPUT_ROOT
    ):
        raise ValueError("DIAG5 authority path or output root differs")
    payload = _mapping(
        load_canonical_json_bytes(expected.read_bytes()), "DIAG5 authority"
    )
    _validate_diag5_authority_payload(
        payload, repository=repository, output_root=output_root.absolute()
    )
    return MappingProxyType(dict(payload))


def _diag5_assert_active(claim: Diag5SuccessorAuthorityClaim) -> None:
    if not claim._lease.active:
        raise RuntimeError("DIAG5 authority claim is no longer active")


def _diag5_assert_output_absent(output: Path, *, allow_staging: bool) -> None:
    permitted = {DIAG5_GPU_STAGING_ROOT} if allow_staging else set()
    candidates = {
        output,
        DIAG5_GPU_STAGING_ROOT,
        DIAG5_GPU_ROLLBACK_ROOT,
        diag5_consumption_marker_path(output),
        DIAG5_PHYSICAL_FAILURE_PATH,
    }
    candidates.update(output.parent.glob(f"{output.name}.partial-*"))
    candidates.update(
        output.parent.glob(f".{output.name}.diag5-authority-consumed.json.pending-*")
    )
    candidates.update(
        output.parent.glob(
            f".{output.name}.diag5-physical-publication-failure.json.pending-*"
        )
    )
    present = {path for path in candidates if not _diag5_path_lexically_absent(path)}
    if present != permitted:
        raise FileExistsError("DIAG5 output lifecycle is not exact")


def _diag5_discovery_leaf_paths(
    payload: Mapping[str, JsonValue], *, repository: Path
) -> set[Path]:
    qualified = _diag4_qualified_files(payload["qualified_files"])
    frozen = _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    manifest = _mapping(
        load_canonical_json_bytes(manifest_path.read_bytes()),
        "DIAG5 execution-source manifest",
    )
    execution = _mapping(manifest["entries"], "DIAG5 execution-source entries")
    cpu_manifest = DIAG5_CPU_QUALIFICATION_ROOT / "artifact-manifest.json"
    _directories, cpu_files, _source_sha, _entries_sha = _diag4_cpu_manifest_entries(
        cpu_manifest.read_bytes(),
        expected_schema=DIAG5_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION,
    )
    review_paths = {
        DIAG5_REVIEW_ROOT
        / _diag5_artifact_ref(raw, "DIAG5 review reference").relative_path
        for field in ("pre_run_reviews", "post_run_reviews")
        for raw in _mapping(payload[field], f"DIAG5 {field}").values()
    }

    native = _mapping(payload["native_reference"], "DIAG5 native reference")
    _exact_keys(
        native,
        frozenset({"absolute_root", "manifest_sha256"}),
        "DIAG5 native reference",
    )
    if native["absolute_root"] != str(DIAG5_NATIVE_REFERENCE_ROOT):
        raise ValueError("DIAG5 native-reference root differs")
    native_manifest = DIAG5_NATIVE_REFERENCE_ROOT / "artifact-manifest.json"
    native_entries = {
        DIAG5_NATIVE_REFERENCE_ROOT / relative
        for relative, _digest, _size in _diag4_native_manifest_entries(
            native_manifest.read_bytes()
        )
    }
    input_bundle = _mapping(payload["input_bundle"], "DIAG5 input bundle")
    _exact_keys(
        input_bundle,
        frozenset({"absolute_root", "input_fingerprint", "configuration_fingerprint"}),
        "DIAG5 input bundle",
    )
    if input_bundle["absolute_root"] != str(DIAG5_INPUT_ROOT):
        raise ValueError("DIAG5 input root differs")
    input_files: set[Path] = set()
    for path in DIAG5_INPUT_ROOT.rglob("*"):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("DIAG5 input bundle contains a symlink")
        if stat.S_ISREG(metadata.st_mode):
            input_files.add(path)
    consumed = _mapping(payload["consumed_diag3"], "DIAG5 consumed DIAG3")
    consumed_root = Path(
        _string(consumed["root"], "DIAG5 consumed DIAG3 root")
    ).resolve(strict=True)
    consumed_entries = consumed["entries"]
    if not isinstance(consumed_entries, list):
        raise TypeError("DIAG5 consumed DIAG3 entries must be an array")
    consumed_paths = {
        consumed_root
        / _diag4_relative_path(
            _mapping(item, "DIAG5 consumed entry")["relative_path"],
            "DIAG5 consumed path",
        )
        for item in consumed_entries
    }
    predecessor_execution = DIAG5_FAILED_DIAG4_PARTIAL_ROOT / "execution-source"
    predecessor_manifest = (
        predecessor_execution
        / DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_RELATIVE_PATH.removeprefix(
            "execution-source/"
        )
    )
    predecessor_payload = _mapping(
        load_canonical_json_bytes(predecessor_manifest.read_bytes()),
        "DIAG5 predecessor execution manifest",
    )
    predecessor_paths = {
        predecessor_execution
        / _diag4_relative_path(relative, "DIAG5 predecessor source path")
        for relative in _mapping(
            predecessor_payload["entries"], "DIAG5 predecessor source entries"
        )
    }
    return {
        *(repository / relative for relative in qualified),
        *(repository / relative for relative in frozen),
        *(repository / relative for relative in execution),
        manifest_path,
        *(
            DIAG5_CPU_QUALIFICATION_ROOT / relative
            for relative, _digest, _size in cpu_files
        ),
        cpu_manifest,
        *review_paths,
        native_manifest,
        *native_entries,
        *input_files,
        *consumed_paths,
        predecessor_manifest,
        *predecessor_paths,
    }


def _open_diag5_shared_locked_leaf(
    path: Path,
    directory_descriptors: Mapping[Path, int],
    context: str,
) -> _Diag4LockedLeaf:
    absolute = Path(os.path.abspath(path))
    descriptor = os.open(
        absolute.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=directory_descriptors[absolute.parent],
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        _assert_diag4_locked_file_binding(
            directory_descriptors[absolute.parent],
            absolute.name,
            descriptor,
            context,
        )
        initial_bytes = _diag4_descriptor_bytes(descriptor)
        return _Diag4LockedLeaf(
            absolute,
            descriptor,
            hashlib.sha256(initial_bytes).hexdigest(),
            len(initial_bytes),
            stat.S_IMODE(os.fstat(descriptor).st_mode),
        )
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def claim_diag5_successor_authority(
    authority_path: Path, *, repository_root: Path, output_root: Path
) -> Iterator[Diag5SuccessorAuthorityClaim]:
    repository = repository_root.resolve(strict=True)
    output = output_root.absolute()
    expected_authority = repository / DIAG5_AUTHORITY_RELATIVE_PATH
    if (
        authority_path.resolve(strict=True) != expected_authority
        or output != DIAG5_GPU_OUTPUT_ROOT
    ):
        raise ValueError("DIAG5 authority path or output root differs")
    authority_bytes = expected_authority.read_bytes()
    payload = _mapping(load_canonical_json_bytes(authority_bytes), "DIAG5 authority")
    (
        cpu,
        gpu,
        postmortem,
        _decisive_cpu,
        _cpu_source_snapshot,
        _gpu_source_snapshot_identity,
    ) = _validate_diag5_authority_payload(
        payload, repository=repository, output_root=output
    )
    qualified = _diag4_qualified_files(payload["qualified_files"])
    frozen = _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    manifest_bytes = manifest_path.read_bytes()
    execution_entries, execution_sizes, _entries_sha256 = (
        _diag4_execution_source_entries(
            manifest_bytes,
            repository=repository,
            qualified=qualified,
            frozen=frozen,
            locked_leaf_bytes=None,
            expected_count=DIAG5_EXECUTION_SOURCE_ENTRY_COUNT,
        )
    )
    leaf_paths = _diag5_discovery_leaf_paths(payload, repository=repository) | {
        expected_authority,
        repository / DIAG5_PLAN_RELATIVE_PATH,
        manifest_path,
        DIAG5_GPU_INTERPRETER.resolve(strict=True),
        cpu.path,
        gpu.path,
    }
    directories = tuple({path.parent for path in leaf_paths} | {output.parent})
    locked: dict[Path, _Diag4LockedLeaf] = {}
    with _diag4_claim_directories(directories) as directory_descriptors:
        try:
            for path in sorted(leaf_paths, key=str):
                locked[path] = _open_diag5_shared_locked_leaf(
                    path, directory_descriptors, f"DIAG5 leaf {path}"
                )
            _diag5_assert_output_absent(output, allow_staging=False)
            locked_authority = locked[expected_authority]
            if _diag4_descriptor_bytes(locked_authority.descriptor) != authority_bytes:
                raise ValueError("DIAG5 authority changed during claim")
            held_payload = _mapping(
                load_canonical_json_bytes(
                    _diag4_descriptor_bytes(locked_authority.descriptor)
                ),
                "DIAG5 held authority",
            )
            (
                held_cpu,
                held_gpu,
                held_postmortem,
                _held_decisive,
                held_snapshot,
                held_gpu_snapshot_identity,
            ) = _validate_diag5_authority_payload(
                held_payload,
                repository=repository,
                output_root=output,
                locked_leaves=locked,
            )
            if (
                held_payload != payload
                or held_cpu != cpu
                or held_gpu != gpu
                or held_postmortem != postmortem
                or held_snapshot != _cpu_source_snapshot
                or held_gpu_snapshot_identity != _gpu_source_snapshot_identity
            ):
                raise ValueError("DIAG5 held authority identity differs")
            lease = _Diag5AuthorityLease(
                repository=repository,
                output_root=output,
                authority_path=expected_authority,
                authority_bytes=authority_bytes,
                locked_leaves=MappingProxyType(locked),
                directory_descriptors=directory_descriptors,
                cpu_native_claim=Diag5NativeExtensionClaim(
                    cpu, locked[cpu.path], directory_descriptors
                ),
                gpu_native_claim=Diag5NativeExtensionClaim(
                    gpu, locked[gpu.path], directory_descriptors
                ),
            )
            claim = Diag5SuccessorAuthorityClaim(
                payload=MappingProxyType(dict(payload)),
                authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
                plan_prefix_sha256=DIAG5_PLAN_SHA256,
                completed_plan_sha256=_string(
                    payload["completed_plan_sha256"], "DIAG5 completed plan SHA"
                ),
                expected_gpu_uuid=DIAG4_GPU_UUID,
                expected_numerical_identity=MappingProxyType(
                    dict(
                        _mapping(
                            payload["numerical_identity"], "DIAG5 numerical identity"
                        )
                    )
                ),
                expected_frozen_numerical_entries=MappingProxyType(frozen),
                expected_gpu_output_root=output,
                expected_gpu_staging_root=DIAG5_GPU_STAGING_ROOT,
                expected_gpu_rollback_root=DIAG5_GPU_ROLLBACK_ROOT,
                expected_cpu_qualification_root=DIAG5_CPU_QUALIFICATION_ROOT,
                expected_cpu_source_snapshot_entries=held_snapshot,
                expected_gpu_source_snapshot_identity=held_gpu_snapshot_identity,
                expected_native_copy_relative_path=DIAG5_NATIVE_COPY_RELATIVE_PATH,
                expected_copied_native_sha256=cpu.sha256,
                expected_copied_native_size_bytes=cpu.size_bytes,
                cpu_native_binding=cpu,
                gpu_native_binding=gpu,
                predecessor_postmortem=postmortem,
                expected_interpreter=MappingProxyType(
                    dict(_mapping(payload["interpreter"], "DIAG5 interpreter"))
                ),
                expected_native_reference=MappingProxyType(
                    dict(
                        _mapping(payload["native_reference"], "DIAG5 native reference")
                    )
                ),
                expected_input_bundle=MappingProxyType(
                    dict(_mapping(payload["input_bundle"], "DIAG5 input bundle"))
                ),
                _lease=lease,
            )
            try:
                yield claim
                revalidate_diag5_successor_authority(claim)
            finally:
                lease.active = False
        finally:
            if "lease" in locals() and lease.staging_descriptor is not None:
                os.close(lease.staging_descriptor)
            if "lease" in locals() and lease.gpu_snapshot_root_descriptor is not None:
                os.close(lease.gpu_snapshot_root_descriptor)
            if "lease" in locals() and lease.consumption_marker_descriptor is not None:
                os.close(lease.consumption_marker_descriptor)
            if "lease" in locals() and lease.physical_evidence is not None:
                os.close(lease.physical_evidence.descriptor)
            for leaf in reversed(tuple(locked.values())):
                fcntl.flock(leaf.descriptor, fcntl.LOCK_UN)
                os.close(leaf.descriptor)


def bind_diag5_staging_root(
    claim: Diag5SuccessorAuthorityClaim, staging_root: Path
) -> None:
    _diag5_assert_active(claim)
    lease = claim._lease
    if (
        lease.staging_descriptor is not None
        or staging_root.absolute() != DIAG5_GPU_STAGING_ROOT
    ):
        raise ValueError("DIAG5 staging root differs or is already bound")
    metadata = staging_root.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("DIAG5 staging root is not a directory")
    descriptor = os.open(
        staging_root, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    lease.staging_descriptor = descriptor
    _diag5_assert_output_absent(lease.output_root, allow_staging=True)


def _diag5_rename_noreplace_at(directory: int, source: str, destination: str) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if (
        renameat2(
            directory,
            os.fsencode(source),
            directory,
            os.fsencode(destination),
            1,
        )
        == 0
    ):
        return
    error_number = ctypes.get_errno()
    error_type = FileExistsError if error_number == errno.EEXIST else OSError
    raise error_type(error_number, os.strerror(error_number), destination)


def _diag5_physical_pending_name() -> str:
    return f"{DIAG5_PHYSICAL_FAILURE_PATH.name}.pending-{os.getpid()}"


def _diag5_assert_reservation(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
) -> _Diag5PhysicalEvidenceLease:
    _diag5_assert_active(claim)
    if reservation._claim is not claim or claim._lease.physical_evidence is not (
        reservation._lease
    ):
        raise ValueError("DIAG5 physical-evidence reservation differs")
    return reservation._lease


def _diag5_evidence_namespace_state(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
) -> Diag5EvidenceNamespaceState:
    physical = _diag5_assert_reservation(claim, reservation)
    directory = claim._lease.directory_descriptors[DIAG5_PHYSICAL_FAILURE_PATH.parent]
    held = os.fstat(physical.descriptor)
    try:
        visible = os.stat(
            physical.pending_name, dir_fd=directory, follow_symlinks=False
        )
    except FileNotFoundError:
        return Diag5EvidenceNamespaceState.PENDING_UNLINKED
    except OSError:
        return Diag5EvidenceNamespaceState.PENDING_AMBIGUOUS
    if (visible.st_dev, visible.st_ino) == (held.st_dev, held.st_ino):
        return Diag5EvidenceNamespaceState.PENDING_BOUND
    return Diag5EvidenceNamespaceState.PENDING_AMBIGUOUS


def prepare_diag5_physical_failure_evidence(
    claim: Diag5SuccessorAuthorityClaim,
) -> Diag5PhysicalEvidenceReservation:
    """Reserve and retain the sole physical-failure pending inode."""

    _diag5_assert_active(claim)
    lease = claim._lease
    if (
        lease.lifecycle is not Diag5AuthorityLifecycle.CONSUMED
        or lease.staging_descriptor is None
        or lease.physical_evidence is not None
        or lease.published_output is not None
    ):
        raise RuntimeError("DIAG5 physical evidence cannot be reserved in this state")
    directory = lease.directory_descriptors[DIAG5_PHYSICAL_FAILURE_PATH.parent]
    _assert_diag4_locked_directory_binding(
        DIAG5_PHYSICAL_FAILURE_PATH.parent,
        directory,
        lease.directory_descriptors,
    )
    pending_name = _diag5_physical_pending_name()
    prefix = f"{DIAG5_PHYSICAL_FAILURE_PATH.name}.pending-"
    occupied = {
        name
        for name in os.listdir(directory)
        if name == DIAG5_PHYSICAL_FAILURE_PATH.name or name.startswith(prefix)
    }
    if occupied:
        raise FileExistsError("DIAG5 physical-evidence namespace is occupied")
    descriptor = os.open(
        pending_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=directory,
    )
    try:
        held = os.fstat(descriptor)
        visible = os.stat(pending_name, dir_fd=directory, follow_symlinks=False)
        if (
            not stat.S_ISREG(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o600
            or held.st_nlink != 1
            or held.st_size != 0
            or (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino)
        ):
            raise ValueError("DIAG5 physical-evidence reservation differs")
        physical = _Diag5PhysicalEvidenceLease(
            descriptor, pending_name, held.st_dev, held.st_ino
        )
        lease.physical_evidence = physical
        return Diag5PhysicalEvidenceReservation(claim, physical)
    except BaseException:
        os.close(descriptor)
        raise


def publish_diag5_bound_staging(
    claim: Diag5SuccessorAuthorityClaim,
    kind: Diag5PublishedOutputKind,
) -> Path:
    """Rename the exact held staging inode to final without replacement."""

    _diag5_assert_active(claim)
    lease = claim._lease
    if (
        kind is not Diag5PublishedOutputKind.FINAL
        or lease.physical_evidence is None
        or not lease.physical_evidence.active
        or _diag5_evidence_namespace_state(
            claim, Diag5PhysicalEvidenceReservation(claim, lease.physical_evidence)
        )
        is not Diag5EvidenceNamespaceState.PENDING_BOUND
        or lease.published_output is not None
        or lease.staging_descriptor is None
    ):
        raise RuntimeError("DIAG5 staging cannot be published in this state")
    directory = lease.directory_descriptors[lease.output_root.parent]
    held = os.fstat(lease.staging_descriptor)
    staging = os.stat(
        DIAG5_GPU_STAGING_ROOT.name, dir_fd=directory, follow_symlinks=False
    )
    if (held.st_dev, held.st_ino) != (staging.st_dev, staging.st_ino):
        raise ValueError("DIAG5 staging inode differs before publication")
    _diag5_rename_noreplace_at(
        directory, DIAG5_GPU_STAGING_ROOT.name, DIAG5_GPU_OUTPUT_ROOT.name
    )
    lease.published_output = kind
    return DIAG5_GPU_OUTPUT_ROOT


def fsync_diag5_output_parent(claim: Diag5SuccessorAuthorityClaim) -> None:
    """Synchronize the retained output-parent descriptor."""

    _diag5_assert_active(claim)
    if claim._lease.published_output is None:
        raise RuntimeError("DIAG5 output has not been published")
    os.fsync(claim._lease.directory_descriptors[DIAG5_GPU_OUTPUT_ROOT.parent])


def revalidate_diag5_published_output(
    claim: Diag5SuccessorAuthorityClaim,
    kind: Diag5PublishedOutputKind,
) -> None:
    """Rebind the sole published output path to the retained staging inode."""

    _diag5_assert_active(claim)
    lease = claim._lease
    if (
        kind is not Diag5PublishedOutputKind.FINAL
        or lease.published_output is not kind
        or lease.staging_descriptor is None
    ):
        raise RuntimeError("DIAG5 published output state differs")
    directory = lease.directory_descriptors[DIAG5_GPU_OUTPUT_ROOT.parent]
    held = os.fstat(lease.staging_descriptor)
    final = os.stat(DIAG5_GPU_OUTPUT_ROOT.name, dir_fd=directory, follow_symlinks=False)
    if (
        (held.st_dev, held.st_ino) != (final.st_dev, final.st_ino)
        or not _diag5_dirfd_name_absent(directory, DIAG5_GPU_STAGING_ROOT.name)
        or not _diag5_dirfd_name_absent(directory, DIAG5_GPU_ROLLBACK_ROOT.name)
    ):
        raise ValueError("DIAG5 published output identity differs")
    revalidate_diag5_successor_authority(claim)


def _diag5_physical_path_state(
    claim: Diag5SuccessorAuthorityClaim,
    path: Path,
    *,
    validated: bool,
) -> Diag5PhysicalPathState:
    directory = claim._lease.directory_descriptors[path.parent]
    descriptor = claim._lease.staging_descriptor
    if descriptor is None:
        raise RuntimeError("DIAG5 staging descriptor is absent")
    try:
        visible = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        held = os.fstat(descriptor)
    except FileNotFoundError:
        return Diag5PhysicalPathState.ABSENT
    except OSError:
        return Diag5PhysicalPathState.VISIBILITY_AMBIGUOUS
    if (visible.st_dev, visible.st_ino) != (held.st_dev, held.st_ino):
        return Diag5PhysicalPathState.VISIBLE_INVALID
    return (
        Diag5PhysicalPathState.VISIBLE_VALIDATED
        if validated
        else Diag5PhysicalPathState.VISIBLE_INVALID
    )


def rollback_diag5_bound_final(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
    *,
    deep_load: Callable[[Path], object],
) -> Diag5RollbackObservation:
    """Attempt the sole final-to-rollback transition and classify its evidence."""

    physical = _diag5_assert_reservation(claim, reservation)
    lease = claim._lease
    if (
        not physical.active
        or lease.published_output is not Diag5PublishedOutputKind.FINAL
        or lease.rollback_attempted
        or lease.staging_descriptor is None
    ):
        raise RuntimeError("DIAG5 rollback cannot run in this state")
    lease.rollback_attempted = True
    directory = lease.directory_descriptors[DIAG5_GPU_OUTPUT_ROOT.parent]
    evidence_state = _diag5_evidence_namespace_state(claim, reservation)
    if not _diag5_dirfd_name_absent(directory, DIAG5_GPU_ROLLBACK_ROOT.name):
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_COLLISION,
            Diag5RollbackState.FAILED,
            _diag5_physical_path_state(claim, DIAG5_GPU_OUTPUT_ROOT, validated=False),
            _diag5_physical_path_state(claim, DIAG5_GPU_ROLLBACK_ROOT, validated=False),
            evidence_state,
        )
    try:
        _diag5_rename_noreplace_at(
            directory, DIAG5_GPU_OUTPUT_ROOT.name, DIAG5_GPU_ROLLBACK_ROOT.name
        )
    except FileExistsError:
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_COLLISION,
            Diag5RollbackState.FAILED,
            _diag5_physical_path_state(claim, DIAG5_GPU_OUTPUT_ROOT, validated=False),
            _diag5_physical_path_state(claim, DIAG5_GPU_ROLLBACK_ROOT, validated=False),
            evidence_state,
        )
    except OSError:
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_RENAME_FAILED,
            Diag5RollbackState.FAILED,
            _diag5_physical_path_state(claim, DIAG5_GPU_OUTPUT_ROOT, validated=False),
            _diag5_physical_path_state(claim, DIAG5_GPU_ROLLBACK_ROOT, validated=False),
            evidence_state,
        )
    try:
        os.fsync(directory)
    except OSError:
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_PARENT_FSYNC_FAILED,
            Diag5RollbackState.FAILED,
            _diag5_physical_path_state(claim, DIAG5_GPU_OUTPUT_ROOT, validated=False),
            _diag5_physical_path_state(claim, DIAG5_GPU_ROLLBACK_ROOT, validated=False),
            evidence_state,
        )
    final_state = _diag5_physical_path_state(
        claim, DIAG5_GPU_OUTPUT_ROOT, validated=False
    )
    rollback_state = _diag5_physical_path_state(
        claim, DIAG5_GPU_ROLLBACK_ROOT, validated=False
    )
    if (
        final_state is not Diag5PhysicalPathState.ABSENT
        or rollback_state is Diag5PhysicalPathState.VISIBILITY_AMBIGUOUS
    ):
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_VISIBILITY_AMBIGUOUS,
            Diag5RollbackState.AMBIGUOUS,
            final_state,
            rollback_state,
            evidence_state,
        )
    try:
        deep_load(DIAG5_GPU_ROLLBACK_ROOT)
    except Exception:  # noqa: BLE001 - callback failure is the typed deep-load result.
        return Diag5RollbackObservation(
            Diag5RollbackCause.ROLLBACK_DEEP_LOAD_FAILED,
            Diag5RollbackState.FAILED,
            final_state,
            rollback_state,
            evidence_state,
        )
    return Diag5RollbackObservation(
        Diag5RollbackCause.NONE,
        Diag5RollbackState.SUCCEEDED,
        final_state,
        _diag5_physical_path_state(claim, DIAG5_GPU_ROLLBACK_ROOT, validated=True),
        evidence_state,
    )


def _diag5_finalizer_native_bindings(
    claim: Diag5SuccessorAuthorityClaim,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for role, binding in (
        ("cpu", claim.cpu_native_binding),
        ("gpu", claim.gpu_native_binding),
    ):
        prefix = f"{role}_native_extension"
        result[role] = {
            f"{prefix}_path": str(binding.path),
            "native_extension_sha256": binding.sha256,
            "native_extension_size_bytes": binding.size_bytes,
            f"{prefix}_link_count": binding.link_count,
            f"{prefix}_device": binding.device,
            f"{prefix}_inode": binding.inode,
        }
    return result


def _load_diag5_finalizer_receipt(
    claim: Diag5SuccessorAuthorityClaim,
    *,
    physical_memory_bytes: int,
) -> DiagnosticReceiptV5:
    return load_and_validate_diag5_artifact(
        DIAG5_GPU_OUTPUT_ROOT,
        expected_native_bindings=_diag5_finalizer_native_bindings(claim),
        expected_authority_sha256=claim.authority_sha256,
        expected_predecessor_postmortem=claim.predecessor_postmortem,
        expected_source_snapshot_identity=claim.expected_gpu_source_snapshot_identity,
        expected_frozen_numerical_entries=claim.expected_frozen_numerical_entries,
        expected_gpu_uuid=claim.expected_gpu_uuid,
        expected_logical_snapshot_root=claim.expected_gpu_logical_source_root,
        physical_memory_bytes=physical_memory_bytes,
    )


def _diag5_validate_finalizer_artifact_ref(
    claim: Diag5SuccessorAuthorityClaim,
    reference: ArtifactRef,
    *,
    relative_path: str,
    schema_version: str,
) -> None:
    if (
        reference.relative_path != relative_path
        or reference.schema_version != schema_version
    ):
        raise ValueError("DIAG5 finalizer artifact reference identity differs")
    root = claim._lease.staging_descriptor
    if root is None:
        raise RuntimeError("DIAG5 final output descriptor is absent")
    descriptor = os.open(
        relative_path,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=root,
    )
    try:
        metadata = os.fstat(descriptor)
        rebound = os.stat(relative_path, dir_fd=root, follow_symlinks=False)
        payload = _diag4_descriptor_bytes(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
            or (metadata.st_dev, metadata.st_ino, metadata.st_size)
            != (rebound.st_dev, rebound.st_ino, rebound.st_size)
            or metadata.st_size != reference.size_bytes
            or hashlib.sha256(payload).hexdigest() != reference.sha256
        ):
            raise ValueError("DIAG5 finalizer artifact reference bytes differ")
    finally:
        os.close(descriptor)


def _validate_diag5_finalizer_source(
    claim: Diag5SuccessorAuthorityClaim,
    source: Diag5FinalizerSourceInput,
    *,
    physical_memory_bytes: int,
) -> None:
    if type(source) not in {PublishedSnapshot, PreSourceFailure}:
        raise TypeError("DIAG5 finalizer source input differs")
    receipt = _load_diag5_finalizer_receipt(
        claim, physical_memory_bytes=physical_memory_bytes
    )
    slots = dict(receipt.evidence_slots)
    terminal = slots["supervisor_terminal"]
    if terminal.artifact is None or terminal.reason is not None:
        raise ValueError("DIAG5 finalizer supervisor terminal slot differs")
    if type(source) is PublishedSnapshot:
        try:
            validate_diag5_successor_snapshot(source.snapshot, claim)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise Diag5FinalizerError(
                Diag5FinalizerFailureCategory.REVALIDATION, error
            ) from error
        source_manifest = slots["source_manifest"]
        expected_manifest = source.snapshot.source_identity(
            DIAG5_GPU_OUTPUT_ROOT
        ).snapshot_manifest
        if (
            source_manifest.artifact != expected_manifest
            or source_manifest.reason is not None
        ):
            raise ValueError("DIAG5 published snapshot manifest slot differs")
        return
    assert type(source) is PreSourceFailure
    allowed = {
        (
            FailureStageV5.AUTHORITY,
            FailureReasonCodeV5.IDENTITY_REVALIDATION_FAILED,
        ),
        (FailureStageV5.SETUP, FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED),
    }
    if (
        (source.outcome.stage, source.outcome.reason) not in allowed
        or receipt.failure != source.outcome
        or terminal.artifact != source.supervisor_terminal
    ):
        raise ValueError("DIAG5 pre-source outcome differs")
    _diag5_validate_finalizer_artifact_ref(
        claim,
        source.supervisor_terminal,
        relative_path="supervisor-terminal.json",
        schema_version=DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    _diag5_validate_finalizer_artifact_ref(
        claim,
        source.diagnostic_receipt,
        relative_path="diagnostic.json",
        schema_version=DIAG5_SCHEMA_VERSION,
    )
    for index, name in enumerate(DIAG5_EVIDENCE_SLOT_PATHS):
        slot = slots[name]
        if name == "supervisor_terminal":
            continue
        expected_reason = source.outcome.reason if index == 0 else None
        if slot.artifact is not None or slot.reason is not expected_reason:
            raise ValueError("DIAG5 pre-source evidence vector differs")
    final_root = claim._lease.staging_descriptor
    if final_root is None:
        raise RuntimeError("DIAG5 final output descriptor is absent")
    names = os.listdir(final_root)
    if "source-snapshot" in names or any(
        name.startswith(".source-snapshot.staging-") for name in names
    ):
        raise ValueError("DIAG5 pre-source namespace is occupied")


def _diag5_finalizer_deep_load(
    claim: Diag5SuccessorAuthorityClaim,
    source: Diag5FinalizerSourceInput,
    *,
    physical_memory_bytes: int,
) -> None:
    try:
        _validate_diag5_finalizer_source(
            claim, source, physical_memory_bytes=physical_memory_bytes
        )
    except Diag5FinalizerError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise Diag5FinalizerError(
            Diag5FinalizerFailureCategory.DEEP_LOAD, error
        ) from error


def finalize_diag5_physical_evidence_success(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
    source: Diag5FinalizerSourceInput,
    *,
    physical_memory_bytes: int,
) -> None:
    """Deep-validate the exact source branch, then remove the reservation."""

    physical = _diag5_assert_reservation(claim, reservation)
    if type(physical_memory_bytes) is not int or physical_memory_bytes <= 0:
        error = ValueError("DIAG5 physical memory must be a positive integer")
        raise Diag5FinalizerError(
            Diag5FinalizerFailureCategory.DEEP_LOAD, error
        ) from error
    if (
        not physical.active
        or claim._lease.rollback_attempted
        or claim._lease.published_output is not Diag5PublishedOutputKind.FINAL
    ):
        raise RuntimeError("DIAG5 physical evidence cannot be finalized")
    directory = claim._lease.directory_descriptors[DIAG5_PHYSICAL_FAILURE_PATH.parent]
    if (
        os.fstat(physical.descriptor).st_size != 0
        or _diag5_evidence_namespace_state(claim, reservation)
        is not Diag5EvidenceNamespaceState.PENDING_BOUND
    ):
        raise ValueError("DIAG5 physical-evidence reservation is not empty and bound")
    try:
        revalidate_diag5_published_output(claim, Diag5PublishedOutputKind.FINAL)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise Diag5FinalizerError(
            Diag5FinalizerFailureCategory.REVALIDATION, error
        ) from error
    _diag5_finalizer_deep_load(
        claim, source, physical_memory_bytes=physical_memory_bytes
    )
    try:
        revalidate_diag5_successor_authority(claim)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise Diag5FinalizerError(
            Diag5FinalizerFailureCategory.REVALIDATION, error
        ) from error
    try:
        _unlink_diag4_pending_marker(
            physical.descriptor, physical.pending_name, directory
        )
        os.fsync(directory)
        if (
            _diag5_evidence_namespace_state(claim, reservation)
            is not Diag5EvidenceNamespaceState.PENDING_UNLINKED
            or not _diag5_dirfd_name_absent(directory, DIAG5_PHYSICAL_FAILURE_PATH.name)
            or _diag5_physical_path_state(claim, DIAG5_GPU_OUTPUT_ROOT, validated=True)
            is not Diag5PhysicalPathState.VISIBLE_VALIDATED
        ):
            raise ValueError("DIAG5 clean physical-evidence namespace differs")
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise Diag5FinalizerError(
            Diag5FinalizerFailureCategory.FINALIZATION, error
        ) from error
    physical.active = False


def cancel_diag5_physical_failure_evidence(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
) -> Diag5PhysicalCancellationObservation:
    """Cancel the exact empty reservation after a pre-final rename failure."""

    physical = _diag5_assert_reservation(claim, reservation)
    lease = claim._lease
    if (
        not physical.active
        or lease.published_output is not None
        or lease.rollback_attempted
        or lease.staging_descriptor is None
    ):
        raise RuntimeError("DIAG5 physical evidence cannot be cancelled")
    directory = lease.directory_descriptors[DIAG5_PHYSICAL_FAILURE_PATH.parent]
    physical.active = False

    def evidence_state() -> Diag5EvidenceNamespaceState:
        try:
            return _diag5_evidence_namespace_state(claim, reservation)
        except Exception:  # noqa: BLE001 - ambiguity is the typed observation.
            return Diag5EvidenceNamespaceState.PENDING_AMBIGUOUS

    def path_state(path: Path, *, validated: bool) -> Diag5PhysicalPathState:
        try:
            return _diag5_physical_path_state(claim, path, validated=validated)
        except Exception:  # noqa: BLE001 - ambiguity is the typed observation.
            return Diag5PhysicalPathState.VISIBILITY_AMBIGUOUS

    def observation(
        cause: Diag5PhysicalCancellationCause,
        state: Diag5PhysicalCancellationState,
    ) -> Diag5PhysicalCancellationObservation:
        return Diag5PhysicalCancellationObservation(
            cause,
            state,
            evidence_state(),
            path_state(DIAG5_GPU_STAGING_ROOT, validated=True),
            path_state(DIAG5_GPU_OUTPUT_ROOT, validated=False),
            path_state(DIAG5_GPU_ROLLBACK_ROOT, validated=False),
        )

    initial = observation(
        Diag5PhysicalCancellationCause.NONE,
        Diag5PhysicalCancellationState.CANCELLED,
    )
    try:
        reservation_empty = os.fstat(physical.descriptor).st_size == 0
    except OSError:
        reservation_empty = False
    if (
        not reservation_empty
        or initial.evidence_namespace_state
        is not Diag5EvidenceNamespaceState.PENDING_BOUND
        or initial.staging_path_state is not Diag5PhysicalPathState.VISIBLE_VALIDATED
        or initial.final_path_state is not Diag5PhysicalPathState.ABSENT
        or initial.rollback_path_state is not Diag5PhysicalPathState.ABSENT
    ):
        raise Diag5PhysicalCancellationError(
            Diag5PhysicalCancellationObservation(
                Diag5PhysicalCancellationCause.CANCEL_VISIBILITY_AMBIGUOUS,
                Diag5PhysicalCancellationState.SPENT,
                initial.evidence_namespace_state,
                initial.staging_path_state,
                initial.final_path_state,
                initial.rollback_path_state,
            ),
            ValueError("DIAG5 physical-evidence cancellation entry state differs"),
        )

    try:
        _unlink_diag4_pending_marker(
            physical.descriptor, physical.pending_name, directory
        )
    except Exception as error:
        raise Diag5PhysicalCancellationError(
            observation(
                Diag5PhysicalCancellationCause.CANCEL_UNLINK_FAILED,
                Diag5PhysicalCancellationState.SPENT,
            ),
            error,
        ) from error
    try:
        os.fsync(directory)
    except Exception as error:
        raise Diag5PhysicalCancellationError(
            observation(
                Diag5PhysicalCancellationCause.CANCEL_PARENT_FSYNC_FAILED,
                Diag5PhysicalCancellationState.SPENT,
            ),
            error,
        ) from error
    cancelled = observation(
        Diag5PhysicalCancellationCause.NONE,
        Diag5PhysicalCancellationState.CANCELLED,
    )
    if (
        cancelled.evidence_namespace_state
        is not Diag5EvidenceNamespaceState.PENDING_UNLINKED
        or cancelled.staging_path_state is not Diag5PhysicalPathState.VISIBLE_VALIDATED
        or cancelled.final_path_state is not Diag5PhysicalPathState.ABSENT
        or cancelled.rollback_path_state is not Diag5PhysicalPathState.ABSENT
        or not _diag5_dirfd_name_absent(directory, DIAG5_PHYSICAL_FAILURE_PATH.name)
    ):
        raise Diag5PhysicalCancellationError(
            Diag5PhysicalCancellationObservation(
                Diag5PhysicalCancellationCause.CANCEL_VISIBILITY_AMBIGUOUS,
                Diag5PhysicalCancellationState.SPENT,
                cancelled.evidence_namespace_state,
                cancelled.staging_path_state,
                cancelled.final_path_state,
                cancelled.rollback_path_state,
            ),
            ValueError("DIAG5 cancelled physical-evidence namespace differs"),
        )
    try:
        revalidate_diag5_successor_authority(claim)
    except Exception as error:
        raise Diag5PhysicalCancellationError(
            Diag5PhysicalCancellationObservation(
                Diag5PhysicalCancellationCause.CANCEL_REVALIDATION_FAILED,
                Diag5PhysicalCancellationState.SPENT,
                cancelled.evidence_namespace_state,
                cancelled.staging_path_state,
                cancelled.final_path_state,
                cancelled.rollback_path_state,
            ),
            error,
        ) from error
    return cancelled


def _validate_diag5_physical_failure_payload(
    claim: Diag5SuccessorAuthorityClaim,
    payload: Mapping[str, JsonValue],
    namespace_state: Diag5EvidenceNamespaceState,
) -> bytes:
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "authority_sha256",
                "original_reason",
                "rollback_cause",
                "rollback_state",
                "final_path",
                "final_path_state",
                "rollback_path",
                "rollback_path_state",
                "evidence_namespace_state_at_seal",
                "sealed_artifact_manifest_sha256",
            }
        ),
        "DIAG5 physical-publication failure",
    )
    if (
        payload["schema_version"] != DIAG5_PHYSICAL_FAILURE_SCHEMA_VERSION
        or payload["route"] != DIAG5_ROUTE
        or payload["authority_sha256"] != claim.authority_sha256
        or payload["final_path"] != str(DIAG5_GPU_OUTPUT_ROOT)
        or payload["rollback_path"] != str(DIAG5_GPU_ROLLBACK_ROOT)
        or payload["evidence_namespace_state_at_seal"] != namespace_state.value
    ):
        raise ValueError("DIAG5 physical-publication failure identity differs")
    _diag4_sha256(
        payload["sealed_artifact_manifest_sha256"],
        "DIAG5 sealed artifact manifest SHA",
    )
    if payload["original_reason"] not in {
        "FINAL_FSYNC_FAILED",
        "FINAL_DEEP_LOAD_FAILED",
        "POST_FINAL_AUTHORITY_REVALIDATION_FAILED",
        "POST_FINAL_AUTHORITY_FINALIZATION_FAILED",
    }:
        raise ValueError("DIAG5 physical-publication original reason differs")
    state = _string(payload["rollback_state"], "DIAG5 rollback state")
    cause = _string(payload["rollback_cause"], "DIAG5 rollback cause")
    valid_pair = (
        (state == "SUCCEEDED" and cause == "NONE")
        or (
            state == "FAILED"
            and cause
            in {
                "ROLLBACK_COLLISION",
                "ROLLBACK_RENAME_FAILED",
                "ROLLBACK_PARENT_FSYNC_FAILED",
                "ROLLBACK_DEEP_LOAD_FAILED",
            }
        )
        or (state == "AMBIGUOUS" and cause == "ROLLBACK_VISIBILITY_AMBIGUOUS")
    )
    path_states = {
        "ABSENT",
        "VISIBLE_VALIDATED",
        "VISIBLE_INVALID",
        "VISIBILITY_AMBIGUOUS",
    }
    if (
        not valid_pair
        or payload["final_path_state"] not in path_states
        or payload["rollback_path_state"] not in path_states
        or (
            state == "SUCCEEDED"
            and (
                payload["final_path_state"] != "ABSENT"
                or payload["rollback_path_state"] != "VISIBLE_VALIDATED"
            )
        )
    ):
        raise ValueError("DIAG5 physical-publication state differs")
    return canonical_json_bytes(payload)


def publish_diag5_physical_failure_evidence(
    claim: Diag5SuccessorAuthorityClaim,
    reservation: Diag5PhysicalEvidenceReservation,
    payload: Mapping[str, JsonValue],
) -> Path | None:
    """Seal once through the retained inode and publish only when still bound."""

    physical = _diag5_assert_reservation(claim, reservation)
    if not physical.active or not claim._lease.rollback_attempted:
        raise RuntimeError("DIAG5 physical-failure evidence cannot be published")
    namespace_state = _diag5_evidence_namespace_state(claim, reservation)
    data = _validate_diag5_physical_failure_payload(claim, payload, namespace_state)
    physical.active = False
    os.ftruncate(physical.descriptor, 0)
    os.lseek(physical.descriptor, 0, os.SEEK_SET)
    offset = 0
    while offset < len(data):
        offset += os.write(physical.descriptor, data[offset:])
    os.fchmod(physical.descriptor, 0o444)
    os.fsync(physical.descriptor)
    if namespace_state is not Diag5EvidenceNamespaceState.PENDING_BOUND:
        return None
    directory = claim._lease.directory_descriptors[DIAG5_PHYSICAL_FAILURE_PATH.parent]
    _assert_diag4_pending_binding(physical.descriptor, physical.pending_name, directory)
    os.link(
        f"/proc/self/fd/{physical.descriptor}",
        DIAG5_PHYSICAL_FAILURE_PATH.name,
        dst_dir_fd=directory,
        follow_symlinks=True,
    )
    _unlink_diag4_pending_marker(physical.descriptor, physical.pending_name, directory)
    os.fsync(directory)
    held = os.fstat(physical.descriptor)
    visible = os.stat(
        DIAG5_PHYSICAL_FAILURE_PATH.name,
        dir_fd=directory,
        follow_symlinks=False,
    )
    if (
        _diag4_descriptor_bytes(physical.descriptor) != data
        or stat.S_IMODE(held.st_mode) != 0o444
        or held.st_nlink != 1
        or (held.st_dev, held.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        raise ValueError("DIAG5 physical-failure evidence publication differs")
    return DIAG5_PHYSICAL_FAILURE_PATH


def diag5_authority_lifecycle(
    claim: Diag5SuccessorAuthorityClaim,
) -> Diag5AuthorityLifecycle:
    _diag5_assert_active(claim)
    return claim._lease.lifecycle


def revalidate_diag5_successor_authority(
    claim: Diag5SuccessorAuthorityClaim,
) -> None:
    _diag5_assert_active(claim)
    lease = claim._lease
    for leaf in lease.locked_leaves.values():
        _assert_diag4_locked_leaf_binding(leaf, lease.directory_descriptors)
    cpu_held = claim._lease.cpu_native_claim._leaf
    gpu_held = claim._lease.gpu_native_claim._leaf
    if (
        _diag5_native_binding(
            claim.payload["cpu_native_binding"],
            cpu=True,
            locked_leaves={claim.cpu_native_binding.path: cpu_held},
        )
        != claim.cpu_native_binding
    ):
        raise ValueError("DIAG5 CPU native binding drifted")
    if (
        _diag5_native_binding(
            claim.payload["gpu_native_binding"],
            cpu=False,
            locked_leaves={claim.gpu_native_binding.path: gpu_held},
        )
        != claim.gpu_native_binding
    ):
        raise ValueError("DIAG5 GPU native binding drifted")
    if lease.staging_descriptor is not None:
        held = os.fstat(lease.staging_descriptor)
        authorized = (
            DIAG5_GPU_STAGING_ROOT,
            DIAG5_GPU_OUTPUT_ROOT,
            DIAG5_GPU_ROLLBACK_ROOT,
        )
        bound_paths: list[Path] = []
        for path in authorized:
            if _diag5_path_lexically_absent(path):
                continue
            bound = path.stat(follow_symlinks=False)
            if (held.st_dev, held.st_ino) != (bound.st_dev, bound.st_ino):
                raise ValueError(
                    "DIAG5 output path does not bind the held staging inode"
                )
            bound_paths.append(path)
        competing = {
            path
            for path in lease.output_root.parent.glob(
                f"{lease.output_root.name}.partial-*"
            )
            if path not in authorized
        }
        if len(bound_paths) != 1 or competing:
            raise ValueError("DIAG5 held output lifecycle differs")
    marker = diag5_consumption_marker_path(lease.output_root)
    if lease.lifecycle is Diag5AuthorityLifecycle.UNCONSUMED:
        directory = lease.directory_descriptors[marker.parent]
        _assert_diag4_locked_directory_binding(
            marker.parent, directory, lease.directory_descriptors
        )
        if not _diag5_dirfd_name_absent(directory, marker.name):
            raise Diag5ConsumptionMarkerInvalidError(
                "DIAG5 authority was unexpectedly consumed"
            )
    else:
        validate_diag5_consumption_marker(claim)
    validated, _cpu, _postmortem, _decisive, _snapshot, _gpu_snapshot = (
        _validate_diag5_authority_payload(
            _mapping(
                load_canonical_json_bytes(lease.authority_bytes),
                "DIAG5 held authority",
            ),
            repository=lease.repository,
            output_root=lease.output_root,
            locked_leaves=lease.locked_leaves,
        )
    )
    if validated != claim.cpu_native_binding:
        raise ValueError("DIAG5 held authority identity differs")


def _diag5_consumption_payload(
    claim: Diag5SuccessorAuthorityClaim,
) -> dict[str, JsonValue]:
    return {
        "schema_version": DIAG5_CONSUMPTION_SCHEMA_VERSION,
        "route": DIAG5_ROUTE,
        "authority_sha256": claim.authority_sha256,
        "plan_prefix_sha256": claim.plan_prefix_sha256,
        "completed_plan_sha256": claim.completed_plan_sha256,
        "output_root": str(claim.expected_gpu_output_root),
    }


def validate_diag5_consumption_marker(
    claim: Diag5SuccessorAuthorityClaim,
) -> Diag5ConsumedAuthority:
    _diag5_assert_active(claim)
    path = diag5_consumption_marker_path(claim.expected_gpu_output_root)
    directory = claim._lease.directory_descriptors[path.parent]
    descriptor = claim._lease.consumption_marker_descriptor
    close_descriptor = descriptor is None
    if descriptor is None:
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory,
            )
        except OSError as error:
            raise Diag5ConsumptionMarkerInvalidError(
                "DIAG5 consumption marker is absent"
            ) from error
    try:
        payload_bytes = _diag4_descriptor_bytes(descriptor)
        held = os.fstat(descriptor)
        bound = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        if (
            payload_bytes != canonical_json_bytes(_diag5_consumption_payload(claim))
            or not stat.S_ISREG(held.st_mode)
            or stat.S_IMODE(held.st_mode) != 0o444
            or held.st_nlink != 1
            or (held.st_dev, held.st_ino) != (bound.st_dev, bound.st_ino)
        ):
            raise Diag5ConsumptionMarkerInvalidError(
                "DIAG5 consumption marker differs or is not sealed"
            )
    finally:
        if close_descriptor:
            os.close(descriptor)
    return Diag5ConsumedAuthority(
        path,
        hashlib.sha256(payload_bytes).hexdigest(),
        MappingProxyType(_diag5_consumption_payload(claim)),
    )


def consume_diag5_successor_authority(
    claim: Diag5SuccessorAuthorityClaim,
) -> Diag5ConsumedAuthority:
    _diag5_assert_active(claim)
    lease = claim._lease
    if (
        lease.lifecycle is not Diag5AuthorityLifecycle.UNCONSUMED
        or lease.staging_descriptor is None
    ):
        raise RuntimeError("DIAG5 authority cannot be consumed in this state")
    revalidate_diag5_successor_authority(claim)
    marker = diag5_consumption_marker_path(lease.output_root)
    pending_name = f"{marker.name}.pending-{os.getpid()}"
    payload = canonical_json_bytes(_diag5_consumption_payload(claim))
    directory = lease.directory_descriptors[marker.parent]
    try:
        descriptor = os.open(
            pending_name,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory,
        )
    except OSError:
        lease.lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        _assert_diag4_locked_directory_binding(
            marker.parent, directory, lease.directory_descriptors
        )
        if _diag5_dirfd_name_absent(
            directory, pending_name
        ) and _diag5_dirfd_name_absent(directory, marker.name):
            os.fsync(directory)
            _assert_diag4_locked_directory_binding(
                marker.parent, directory, lease.directory_descriptors
            )
            if _diag5_dirfd_name_absent(
                directory, pending_name
            ) and _diag5_dirfd_name_absent(directory, marker.name):
                lease.lifecycle = Diag5AuthorityLifecycle.UNCONSUMED
        raise
    published = False
    try:
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        _assert_diag4_pending_binding(descriptor, pending_name, directory)
        os.link(
            f"/proc/self/fd/{descriptor}",
            marker.name,
            dst_dir_fd=directory,
            follow_symlinks=True,
        )
        published = True
        lease.lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        _assert_diag4_published_marker_binding(
            descriptor, marker.name, payload, directory
        )
        lease.consumption_marker_descriptor = descriptor
        _unlink_diag4_pending_marker(descriptor, pending_name, directory)
        os.fsync(directory)
        lease.lifecycle = Diag5AuthorityLifecycle.CONSUMED
        lease.consumed = validate_diag5_consumption_marker(claim)
        return lease.consumed
    except BaseException:
        lease.lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        _assert_diag4_locked_directory_binding(
            marker.parent, directory, lease.directory_descriptors
        )
        marker_exists = not _diag5_dirfd_name_absent(directory, marker.name)
        try:
            _unlink_diag4_pending_marker(descriptor, pending_name, directory)
            os.fsync(directory)
            _assert_diag4_locked_directory_binding(
                marker.parent, directory, lease.directory_descriptors
            )
            pending_absent = _diag5_dirfd_name_absent(directory, pending_name)
            marker_absent = _diag5_dirfd_name_absent(directory, marker.name)
            if not published and not marker_exists and pending_absent and marker_absent:
                lease.lifecycle = Diag5AuthorityLifecycle.UNCONSUMED
            else:
                lease.lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        except BaseException:
            lease.lifecycle = Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN
            raise
        raise
    finally:
        if lease.consumption_marker_descriptor != descriptor:
            os.close(descriptor)


def validate_diag5_successor_snapshot(
    snapshot: SnapshotPublication, claim: Diag5SuccessorAuthorityClaim
) -> None:
    _diag5_assert_active(claim)
    gpu_claim = claim._lease.gpu_native_claim
    _assert_diag4_locked_leaf_binding(gpu_claim._leaf, gpu_claim._directories)
    gpu_metadata = os.fstat(gpu_claim._leaf.descriptor)
    if (
        gpu_metadata.st_dev,
        gpu_metadata.st_ino,
        gpu_metadata.st_size,
        gpu_metadata.st_nlink,
        hashlib.sha256(_diag4_descriptor_bytes(gpu_claim._leaf.descriptor)).hexdigest(),
    ) != (
        gpu_claim.binding.device,
        gpu_claim.binding.inode,
        gpu_claim.binding.size_bytes,
        gpu_claim.binding.link_count,
        gpu_claim.binding.sha256,
    ):
        raise ValueError("DIAG5 held GPU native binding drifted")
    observed = {
        entry.relative_path: (entry.sha256, entry.size_bytes, entry.role)
        for entry in snapshot.entries
    }
    expected = {
        relative: identity
        for relative, identity in claim.expected_cpu_source_snapshot_entries.items()
        if identity[2] != "prequalification_plan"
    }
    if (
        len(expected) != DIAG5_EXECUTION_SOURCE_ENTRY_COUNT + 2
        or observed != expected
        or snapshot.identity() != claim.expected_gpu_source_snapshot_identity
        or snapshot.manifest_sha256
        != claim.expected_gpu_source_snapshot_identity.manifest_sha256
        or _load_diag5_bound_gpu_snapshot_identity(snapshot, claim)
        != claim.expected_gpu_source_snapshot_identity
        or expected[DIAG5_NATIVE_COPY_RELATIVE_PATH]
        != (
            claim.expected_copied_native_sha256,
            claim.expected_copied_native_size_bytes,
            "native_extension",
        )
    ):
        raise ValueError("DIAG5 GPU source snapshot differs from authority")


def _load_diag5_bound_gpu_snapshot_identity(
    snapshot: SnapshotPublication,
    claim: Diag5SuccessorAuthorityClaim,
) -> SnapshotIdentity:
    """Reconstruct the physical GPU snapshot through the held staging inode."""

    staging_descriptor = claim._lease.staging_descriptor
    if staging_descriptor is None:
        raise RuntimeError("DIAG5 staging root is not held")
    expected_root = Path("source-snapshot")
    root_descriptor = os.open(
        expected_root.as_posix(),
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=staging_descriptor,
    )
    directory_descriptors: dict[PurePosixPath, int] = {
        PurePosixPath("."): root_descriptor
    }
    leaf_descriptors: list[int] = []
    try:
        physical_root = os.fstat(root_descriptor)
        cached_root = snapshot.root.stat(follow_symlinks=False)
        retained_root_descriptor = claim._lease.gpu_snapshot_root_descriptor
        if retained_root_descriptor is None:
            claim._lease.gpu_snapshot_root_descriptor = os.dup(root_descriptor)
        else:
            retained_root = os.fstat(retained_root_descriptor)
            if (physical_root.st_dev, physical_root.st_ino) != (
                retained_root.st_dev,
                retained_root.st_ino,
            ):
                raise ValueError("DIAG5 GPU snapshot root was substituted")
        if (
            stat.S_IMODE(physical_root.st_mode) != 0o555
            or (physical_root.st_dev, physical_root.st_ino)
            != (cached_root.st_dev, cached_root.st_ino)
            or snapshot.manifest_path != snapshot.root / "source-manifest.json"
        ):
            raise ValueError("DIAG5 GPU snapshot root inode differs")
        pending = [PurePosixPath(".")]
        observed_files: set[str] = set()
        while pending:
            relative_directory = pending.pop()
            descriptor = directory_descriptors[relative_directory]
            for name in os.listdir(descriptor):
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                relative = (
                    PurePosixPath(name)
                    if relative_directory == PurePosixPath(".")
                    else relative_directory / name
                )
                if stat.S_ISDIR(metadata.st_mode):
                    if stat.S_IMODE(metadata.st_mode) != 0o555:
                        raise ValueError("DIAG5 GPU snapshot directory mode differs")
                    child = os.open(
                        name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
                        dir_fd=descriptor,
                    )
                    directory_descriptors[relative] = child
                    pending.append(relative)
                elif stat.S_ISREG(metadata.st_mode):
                    observed_files.add(relative.as_posix())
                else:
                    raise ValueError("DIAG5 GPU snapshot contains a special entry")
        manifest_relative = PurePosixPath("source-manifest.json")
        expected_files = {
            manifest_relative.as_posix(),
            *(
                entry.relative_path
                for entry in claim.expected_gpu_source_snapshot_identity.entries
            ),
        }
        if observed_files != expected_files:
            raise ValueError("DIAG5 GPU snapshot physical closure differs")

        def held_bytes(relative: PurePosixPath) -> bytes:
            parent = relative.parent
            descriptor = os.open(
                relative.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=directory_descriptors[parent],
            )
            leaf_descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
            held = os.fstat(descriptor)
            bound = os.stat(
                relative.name,
                dir_fd=directory_descriptors[parent],
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(held.st_mode)
                or stat.S_IMODE(held.st_mode) != 0o444
                or held.st_nlink != 1
                or (held.st_dev, held.st_ino) != (bound.st_dev, bound.st_ino)
            ):
                raise ValueError("DIAG5 GPU snapshot leaf topology differs")
            payload = _diag4_descriptor_bytes(descriptor)
            rebound_held = os.fstat(descriptor)
            rebound_path = os.stat(
                relative.name,
                dir_fd=directory_descriptors[parent],
                follow_symlinks=False,
            )
            if (rebound_held.st_dev, rebound_held.st_ino) != (
                rebound_path.st_dev,
                rebound_path.st_ino,
            ):
                raise ValueError("DIAG5 GPU snapshot leaf was substituted")
            return payload

        manifest_bytes = held_bytes(manifest_relative)
        manifest = _mapping(
            load_canonical_json_bytes(manifest_bytes), "DIAG5 GPU source manifest"
        )
        _exact_keys(
            manifest,
            frozenset({"entries", "schema_version", "worktree"}),
            "DIAG5 GPU source manifest",
        )
        raw_entries = manifest["entries"]
        if manifest[
            "schema_version"
        ] != SOURCE_MANIFEST_SCHEMA_VERSION or not isinstance(raw_entries, list):
            raise ValueError("DIAG5 GPU source-manifest schema differs")
        entries: list[SnapshotEntry] = []
        for raw_entry in raw_entries:
            entry = _mapping(raw_entry, "DIAG5 GPU source entry")
            _exact_keys(
                entry,
                frozenset({"relative_path", "role", "sha256", "size_bytes"}),
                "DIAG5 GPU source entry",
            )
            relative = PurePosixPath(
                _diag4_relative_path(
                    entry["relative_path"], "DIAG5 GPU source entry path"
                )
            )
            payload = held_bytes(relative)
            digest = _diag4_sha256(entry["sha256"], "DIAG5 GPU source entry SHA")
            size_bytes = _integer(entry["size_bytes"], "DIAG5 GPU source entry size")
            if (
                len(payload) != size_bytes
                or hashlib.sha256(payload).hexdigest() != digest
            ):
                raise ValueError("DIAG5 GPU snapshot leaf bytes differ")
            entries.append(
                SnapshotEntry(
                    _string(entry["role"], "DIAG5 GPU source entry role"),
                    relative.as_posix(),
                    size_bytes,
                    digest,
                )
            )
        worktree = _mapping(manifest["worktree"], "DIAG5 GPU source worktree")
        identity = build_snapshot_identity(
            tuple(entries),
            WorktreeIdentity(
                _string(worktree["git_head"], "DIAG5 GPU source git HEAD"),
                _string(worktree["tracked_diff_sha256"], "DIAG5 GPU tracked SHA"),
                _string(
                    worktree["untracked_bytes_manifest_sha256"],
                    "DIAG5 GPU untracked SHA",
                ),
                _string(worktree["repo_root"], "DIAG5 GPU repository root"),
            ),
        )
        if hashlib.sha256(manifest_bytes).hexdigest() != identity.manifest_sha256:
            raise ValueError("DIAG5 GPU source-manifest bytes differ")
        for relative, descriptor in directory_descriptors.items():
            if relative == PurePosixPath("."):
                rebound = os.stat(
                    expected_root.as_posix(),
                    dir_fd=staging_descriptor,
                    follow_symlinks=False,
                )
            else:
                rebound = os.stat(
                    relative.name,
                    dir_fd=directory_descriptors[relative.parent],
                    follow_symlinks=False,
                )
            held = os.fstat(descriptor)
            if (held.st_dev, held.st_ino) != (rebound.st_dev, rebound.st_ino):
                raise ValueError("DIAG5 GPU snapshot directory was substituted")
        return identity
    finally:
        for descriptor in reversed(leaf_descriptors):
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
        for descriptor in reversed(tuple(directory_descriptors.values())):
            os.close(descriptor)


@dataclass(frozen=True, slots=True)
class SuccessorAuthorityClaim:
    """One process-exclusive validated authority and its exact evidence bytes."""

    payload: Mapping[str, JsonValue]
    authority_sha256: str
    plan_sha256: str


def _mapping(value: JsonValue, context: str) -> Mapping[str, JsonValue]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be an object")
    return value


def _string(value: JsonValue, context: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string")
    return value


def _integer(value: JsonValue, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer")
    return value


def _boolean(value: JsonValue, context: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a boolean")
    return value


def _finite_duration(value: JsonValue, context: str) -> float:
    if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
        raise TypeError(f"{context} must be a finite positive JSON float")
    return value


def _exact_keys(
    value: Mapping[str, JsonValue], expected: frozenset[str], context: str
) -> None:
    if frozenset(value) != expected:
        raise ValueError(f"{context} keys differ")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_plan(repository_root: Path) -> None:
    plan = (repository_root / PLAN_RELATIVE_PATH).read_bytes()
    prefix, marker, _ = plan.partition(b"## Qualification Record\n")
    if not marker or hashlib.sha256(prefix).hexdigest() != PLAN_SHA256:
        raise ValueError("DIAG3 frozen plan prefix differs")


def _validate_qualification_record(
    plan: bytes,
    payload: Mapping[str, JsonValue],
    *,
    authority_sha256: str,
    output_root: Path,
) -> None:
    _, marker, record_bytes = plan.partition(b"## Qualification Record\n")
    if not marker or not record_bytes:
        raise ValueError("DIAG3 qualification record is missing")
    record = _mapping(
        load_canonical_json_bytes(record_bytes), "DIAG3 qualification record"
    )
    _exact_keys(
        record,
        frozenset(
            {
                "schema_version",
                "plan_sha256",
                "authority_sha256",
                "output_root",
                "controlling_cpu",
                "static_checks",
                "qualified_files",
                "frozen_numerical_entries",
                "native_reference_manifest_sha256",
                "no_gpu_used_for_qualification",
                "independent_reviews",
                "authorization",
            }
        ),
        "DIAG3 qualification record",
    )
    record_qualified = _mapping(record["qualified_files"], "qualified_files")
    _exact_keys(record_qualified, frozenset(QUALIFIED_FILE_PATHS), "qualified_files")
    typed_record_qualified = {
        relative: _string(record_qualified[relative], f"qualified_files.{relative}")
        for relative in QUALIFIED_FILE_PATHS
    }
    if (
        _string(record["schema_version"], "schema_version")
        != QUALIFICATION_SCHEMA_VERSION
        or _string(record["plan_sha256"], "plan_sha256") != PLAN_SHA256
        or _string(record["authority_sha256"], "authority_sha256") != authority_sha256
        or _string(record["output_root"], "output_root") != str(output_root.absolute())
        or typed_record_qualified != _qualified_files(payload)
        or record["frozen_numerical_entries"] != payload["frozen_numerical_entries"]
        or _string(
            record["native_reference_manifest_sha256"],
            "native_reference_manifest_sha256",
        )
        != NATIVE_REFERENCE_MANIFEST_SHA256
        or not _boolean(
            record["no_gpu_used_for_qualification"],
            "no_gpu_used_for_qualification",
        )
    ):
        raise ValueError("DIAG3 qualification authority differs")
    controlling = _mapping(record["controlling_cpu"], "controlling_cpu")
    _exact_keys(
        controlling,
        frozenset({"command", "passed", "duration_seconds"}),
        "controlling_cpu",
    )
    if (
        _string(controlling["command"], "controlling_cpu.command")
        != CONTROLLING_CPU_COMMAND
        or _integer(controlling["passed"], "controlling_cpu.passed")
        != CONTROLLING_CPU_PASSED
    ):
        raise ValueError("DIAG3 controlling CPU qualification differs")
    _finite_duration(
        controlling["duration_seconds"], "controlling_cpu.duration_seconds"
    )
    static_checks = _mapping(record["static_checks"], "static_checks")
    static_check_names = frozenset(
        {"ruff_check", "ruff_format_check", "compileall", "git_diff_check"}
    )
    _exact_keys(static_checks, static_check_names, "static_checks")
    if not all(
        _boolean(static_checks[name], f"static_checks.{name}")
        for name in static_check_names
    ):
        raise ValueError("DIAG3 static qualification differs")
    reviews = record["independent_reviews"]
    if not isinstance(reviews, list) or len(reviews) < 2:
        raise ValueError("DIAG3 independent reviews are incomplete")
    reviewers: set[str] = set()
    for index, raw_review in enumerate(reviews):
        review = _mapping(raw_review, f"independent_reviews[{index}]")
        _exact_keys(
            review,
            frozenset({"reviewer", "session", "verdict"}),
            f"independent_reviews[{index}]",
        )
        reviewer = _string(review["reviewer"], f"independent_reviews[{index}].reviewer")
        session = _string(review["session"], f"independent_reviews[{index}].session")
        verdict = _string(review["verdict"], f"independent_reviews[{index}].verdict")
        if not reviewer or not session or verdict != "GO":
            raise ValueError("DIAG3 independent review differs")
        reviewers.add(reviewer)
    if len(reviewers) < 2:
        raise ValueError("DIAG3 independent reviewers are not distinct")
    authorization = _mapping(record["authorization"], "authorization")
    _exact_keys(
        authorization,
        frozenset(
            {
                "preflight_launches",
                "maximum_cold_launches",
                "warm_allowed",
                "retry_allowed",
            }
        ),
        "authorization",
    )
    if (
        _integer(
            authorization["preflight_launches"], "authorization.preflight_launches"
        )
        != 1
        or _integer(
            authorization["maximum_cold_launches"],
            "authorization.maximum_cold_launches",
        )
        != 1
        or _boolean(authorization["warm_allowed"], "authorization.warm_allowed")
        is not False
        or _boolean(authorization["retry_allowed"], "authorization.retry_allowed")
        is not False
    ):
        raise ValueError("DIAG3 qualification launch cardinality differs")


def _qualified_files(payload: Mapping[str, JsonValue]) -> dict[str, str]:
    qualified = _mapping(payload["qualified_files"], "qualified_files")
    _exact_keys(qualified, frozenset(QUALIFIED_FILE_PATHS), "qualified_files")
    return {
        relative: _string(qualified[relative], f"qualified_files.{relative}")
        for relative in QUALIFIED_FILE_PATHS
    }


def _validate_r1() -> None:
    receipt = load_and_validate_diag2_artifact(R1_ROOT)
    if (
        receipt.verdict != "DIAGNOSTIC_INCOMPLETE"
        or receipt.next_route != "NOT_PRODUCED"
        or receipt.failure is None
        or receipt.failure.stage.value != "COLD_CRASH"
        or receipt.failure.reason.value != "CHILD_EXIT_NONZERO"
    ):
        raise ValueError("consumed R1 outcome differs")
    for relative, digest in R1_ARTIFACT_SHA256.items():
        if _sha256_file(R1_ROOT / relative) != digest:
            raise ValueError(f"consumed R1 artifact differs: {relative}")


def _validate_successor_authority_bytes(
    authority_bytes: bytes,
    *,
    repository: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
) -> Mapping[str, JsonValue]:
    """Validate one exact authority byte string against its bound live inputs."""

    payload = _mapping(load_canonical_json_bytes(authority_bytes), "DIAG3 authority")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "plan_sha256",
                "scientific_evidence_schema",
                "execution_policy",
                "launch",
                "consumed_r1",
                "native_reference_manifest_sha256",
                "frozen_numerical_entries",
                "qualified_files",
            }
        ),
        "DIAG3 authority",
    )
    if (
        payload["schema_version"] != SCHEMA_VERSION
        or payload["route"] != ROUTE
        or payload["plan_sha256"] != PLAN_SHA256
        or payload["scientific_evidence_schema"] != DIAG2_SCHEMA_VERSION
        or payload["native_reference_manifest_sha256"]
        != NATIVE_REFERENCE_MANIFEST_SHA256
    ):
        raise ValueError("DIAG3 authority identity differs")

    execution = _mapping(payload["execution_policy"], "execution_policy")
    _exact_keys(
        execution,
        frozenset(
            {
                "parent_platform",
                "child_platform",
                "jax_enable_x64",
                "compilation_cache_enabled",
                "child_preallocate",
                "command_buffer_enabled",
                "required_xla_flag",
            }
        ),
        "execution_policy",
    )
    if (
        _string(execution["parent_platform"], "execution_policy.parent_platform")
        != "cpu"
        or _string(execution["child_platform"], "execution_policy.child_platform")
        != "cuda"
        or _boolean(execution["jax_enable_x64"], "execution_policy.jax_enable_x64")
        is not True
        or _boolean(
            execution["compilation_cache_enabled"],
            "execution_policy.compilation_cache_enabled",
        )
        is not False
        or _boolean(
            execution["child_preallocate"], "execution_policy.child_preallocate"
        )
        is not True
        or _boolean(
            execution["command_buffer_enabled"],
            "execution_policy.command_buffer_enabled",
        )
        is not False
        or _string(execution["required_xla_flag"], "execution_policy.required_xla_flag")
        != "--xla_gpu_enable_command_buffer="
    ):
        raise ValueError("DIAG3 execution policy differs")

    launch = _mapping(payload["launch"], "launch")
    _exact_keys(
        launch,
        frozenset(
            {
                "output_root",
                "reference_root",
                "input_root",
                "interpreter",
                "gpu_uuid",
                "preflight_launches",
                "maximum_cold_launches",
                "warm_allowed",
                "retry_allowed",
            }
        ),
        "launch",
    )
    expected_paths = {
        "output_root": output_root.absolute(),
        "reference_root": reference_root.resolve(strict=True),
        "input_root": input_root.resolve(strict=True),
        "interpreter": interpreter.resolve(strict=True),
    }
    for name, expected in expected_paths.items():
        actual = Path(_string(launch[name], f"launch.{name}"))
        actual = (
            actual.absolute() if name == "output_root" else actual.resolve(strict=True)
        )
        if actual != expected:
            raise ValueError(f"DIAG3 launch {name} differs")
    if (
        _string(launch["gpu_uuid"], "launch.gpu_uuid") != GPU_UUID
        or _integer(launch["preflight_launches"], "launch.preflight_launches") != 1
        or _integer(launch["maximum_cold_launches"], "launch.maximum_cold_launches")
        != 1
        or launch["warm_allowed"] is not False
        or launch["retry_allowed"] is not False
    ):
        raise ValueError("DIAG3 launch policy differs")
    if os.path.lexists(output_root.absolute()) or tuple(
        output_root.parent.glob(f"{output_root.name}.partial-*")
    ):
        raise FileExistsError("DIAG3 output root or staging sibling already exists")

    consumed = _mapping(payload["consumed_r1"], "consumed_r1")
    _exact_keys(
        consumed,
        frozenset({"root", "diagnostic_sha256", "manifest_sha256"}),
        "consumed_r1",
    )
    if consumed != {
        "root": str(R1_ROOT),
        "diagnostic_sha256": R1_ARTIFACT_SHA256["diagnostic.json"],
        "manifest_sha256": R1_ARTIFACT_SHA256["artifact-manifest.json"],
    }:
        raise ValueError("DIAG3 consumed R1 authority differs")

    frozen = payload["frozen_numerical_entries"]
    expected_frozen = [
        {"relative_path": relative, "sha256": digest}
        for relative, digest in DIAG2_FROZEN_NUMERICAL_ENTRIES
    ]
    if frozen != expected_frozen:
        raise ValueError("DIAG3 frozen numerical authority differs")
    for relative, digest in DIAG2_FROZEN_NUMERICAL_ENTRIES:
        if _sha256_file(repository / relative) != digest:
            raise ValueError(f"DIAG3 frozen numerical bytes differ: {relative}")

    reference_manifest = reference_root.resolve(strict=True) / "artifact-manifest.json"
    if _sha256_file(reference_manifest) != NATIVE_REFERENCE_MANIFEST_SHA256:
        raise ValueError("DIAG3 native-reference manifest differs")

    qualified = _qualified_files(payload)
    for relative, digest in qualified.items():
        if digest != _sha256_file(repository / relative):
            raise ValueError(f"DIAG3 qualified file differs: {relative}")

    _validate_plan(repository)
    _validate_r1()
    return payload


def validate_successor_authority(
    authority_path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
) -> Mapping[str, JsonValue]:
    """Validate the exact successor authority before any output path is created."""

    repository = repository_root.resolve(strict=True)
    authority = authority_path.resolve(strict=True)
    if authority != repository / AUTHORITY_RELATIVE_PATH:
        raise ValueError("DIAG3 authority path differs")
    return _validate_successor_authority_bytes(
        authority.read_bytes(),
        repository=repository,
        output_root=output_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
    )


@contextmanager
def _claim_directories(
    directories: tuple[Path, ...],
) -> Iterator[Mapping[Path, int]]:
    """Lock and bind each requested directory's complete root-down inode chain."""

    resolved_directories = {path.resolve(strict=True) for path in directories}
    directory_chain = tuple(
        sorted(
            {
                ancestor
                for directory in resolved_directories
                for ancestor in (directory, *directory.parents)
            },
            key=lambda path: (len(path.parts), str(path)),
        )
    )
    descriptors: dict[Path, int] = {}
    try:
        for directory in directory_chain:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
            if directory.parent == directory:
                descriptor = os.open(directory, flags)
            else:
                descriptor = os.open(
                    directory.name,
                    flags,
                    dir_fd=descriptors[directory.parent],
                )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                os.close(descriptor)
                raise RuntimeError(
                    "DIAG3 successor authority or output is already claimed"
                ) from error
            descriptors[directory] = descriptor
            _assert_locked_directory_binding(directory, descriptor, descriptors)
        yield descriptors
    finally:
        try:
            for directory, descriptor in descriptors.items():
                _assert_locked_directory_binding(directory, descriptor, descriptors)
        finally:
            for descriptor in reversed(tuple(descriptors.values())):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _assert_locked_directory_binding(
    directory: Path,
    descriptor: int,
    descriptors: Mapping[Path, int],
) -> None:
    locked = os.fstat(descriptor)
    if directory.parent == directory:
        bound = os.stat(directory, follow_symlinks=False)
    else:
        bound = os.stat(
            directory.name,
            dir_fd=descriptors[directory.parent],
            follow_symlinks=False,
        )
    if not stat.S_ISDIR(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError(f"DIAG3 directory inode is not bound: {directory}")


def _assert_locked_authority_binding(
    authority_directory_descriptor: int,
    authority_name: str,
    authority_descriptor: int,
) -> None:
    locked = os.fstat(authority_descriptor)
    bound = os.stat(
        authority_name,
        dir_fd=authority_directory_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError("DIAG3 authority inode is not bound to its pathname")


@contextmanager
def claim_successor_authority(
    authority_path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
) -> Iterator[SuccessorAuthorityClaim]:
    """Atomically claim one authority and hold it through the supervised run."""

    repository = repository_root.resolve(strict=True)
    expected_authority = repository / AUTHORITY_RELATIVE_PATH
    if authority_path.resolve(strict=True) != expected_authority:
        raise ValueError("DIAG3 authority path differs")
    authority_directory = expected_authority.parent.resolve(strict=True)
    output_directory = output_root.parent.resolve(strict=True)
    with _claim_directories(
        (authority_directory, output_directory)
    ) as directory_descriptors:
        authority_directory_descriptor = directory_descriptors[authority_directory]
        authority_descriptor = os.open(
            expected_authority.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=authority_directory_descriptor,
        )
        with os.fdopen(authority_descriptor, "rb") as stream:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "DIAG3 successor authority is already claimed"
                ) from error
            _assert_locked_authority_binding(
                authority_directory_descriptor,
                expected_authority.name,
                stream.fileno(),
            )
            authority_bytes = stream.read()
            payload = _validate_successor_authority_bytes(
                authority_bytes,
                repository=repository,
                output_root=output_root,
                reference_root=reference_root,
                input_root=input_root,
                interpreter=interpreter,
            )
            plan_bytes = (repository / PLAN_RELATIVE_PATH).read_bytes()
            prefix, marker, _ = plan_bytes.partition(b"## Qualification Record\n")
            if not marker or hashlib.sha256(prefix).hexdigest() != PLAN_SHA256:
                raise ValueError("DIAG3 locked plan bytes differ")
            _validate_qualification_record(
                plan_bytes,
                payload,
                authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
                output_root=output_root,
            )
            _assert_locked_authority_binding(
                authority_directory_descriptor,
                expected_authority.name,
                stream.fileno(),
            )
            try:
                yield SuccessorAuthorityClaim(
                    payload=payload,
                    authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
                    plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                )
            finally:
                try:
                    _assert_locked_authority_binding(
                        authority_directory_descriptor,
                        expected_authority.name,
                        stream.fileno(),
                    )
                finally:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def validate_successor_snapshot(
    snapshot: SnapshotPublication,
    claim: SuccessorAuthorityClaim,
) -> None:
    """Bind every allowlisted source-snapshot byte to the held authority claim."""

    qualified = _qualified_files(claim.payload)
    special = {
        AUTHORITY_RELATIVE_PATH: claim.authority_sha256,
        PLAN_RELATIVE_PATH: claim.plan_sha256,
    }
    expected = {**qualified, **special}
    if frozenset(expected) != DIAG3_SOURCE_DELTA_ALLOWLIST:
        raise ValueError("DIAG3 authority does not close the source delta allowlist")
    observed = {entry.relative_path: entry.sha256 for entry in snapshot.entries}
    for relative, digest in expected.items():
        if observed.get(relative) != digest:
            raise ValueError(f"DIAG3 snapshot differs from authority: {relative}")


@dataclass(frozen=True, slots=True)
class Diag4NumericalIdentity:
    """Opaque hashes that identify the exact independently qualified GNTR3 graph."""

    problem_sha256: str
    optimizer_options_sha256: str
    base_neq_gntr1_policy_sha256: str
    scaling_sha256: str
    bootstrap_state_sha256: str
    initial_physical_state_sha256: str
    identity_sha256: str

    def to_payload(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in sorted(DIAG4_IDENTITY_FIELDS)}


@dataclass(frozen=True, slots=True)
class Diag4ConsumedAuthority:
    """Durable one-shot consumption evidence created before preflight launch."""

    path: Path
    sha256: str
    payload: Mapping[str, JsonValue]


class Diag4AuthorityLifecycle(str, Enum):
    CLAIMED = "CLAIMED"
    STAGING_BOUND = "STAGING_BOUND"
    CONSUMED = "CONSUMED"
    CONSUMPTION_UNCERTAIN = "CONSUMPTION_UNCERTAIN"
    PRELAUNCH_FAILURE_FINALIZED = "PRELAUNCH_FAILURE_FINALIZED"


class Diag4ConsumptionMarkerInvalidError(ValueError):
    """The durable DIAG4 consumption marker is absent or no longer exact."""


@dataclass(frozen=True, slots=True)
class _Diag4LockedLeaf:
    path: Path
    descriptor: int
    initial_sha256: str
    initial_size_bytes: int
    initial_mode: int


@dataclass(slots=True)
class _Diag4AuthorityLease:
    repository: Path
    output_root: Path
    reference_root: Path
    input_root: Path
    interpreter: Path
    authority_path: Path
    authority_bytes: bytes
    authority_descriptor: int
    plan_descriptor: int
    directory_descriptors: Mapping[Path, int]
    locked_leaves: Mapping[Path, _Diag4LockedLeaf]
    staging_path: Path | None = None
    staging_descriptor: int | None = None
    consumption_marker_descriptor: int | None = None
    consumed: Diag4ConsumedAuthority | None = None
    lifecycle_state: Diag4AuthorityLifecycle = Diag4AuthorityLifecycle.CLAIMED
    active: bool = True


@dataclass(frozen=True, slots=True)
class Diag4SuccessorAuthorityClaim:
    """One held DIAG4 authority with exact source and numerical identity locks."""

    payload: Mapping[str, JsonValue]
    authority_sha256: str
    plan_prefix_sha256: str
    completed_plan_sha256: str
    numerical_identity: Diag4NumericalIdentity
    expected_numerical_identity: Mapping[str, str]
    expected_frozen_numerical_entries: Mapping[str, str]
    expected_execution_source_entries: Mapping[str, tuple[str, int]]
    expected_execution_source_manifest_sha256: str
    expected_native_extension_path: Path
    expected_native_extension_sha256: str
    expected_native_extension_size_bytes: int
    _lease: _Diag4AuthorityLease = field(repr=False, compare=False)


def _diag4_sha256(value: JsonValue, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{context} must be a lowercase SHA-256")
    return digest


def _diag4_relative_path(value: JsonValue, context: str) -> str:
    relative = PurePosixPath(_string(value, context))
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != str(value)
    ):
        raise ValueError(f"{context} must be a canonical relative path")
    return relative.as_posix()


def _diag4_regular_file(path: Path, context: str) -> os.stat_result:
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{context} must be a regular non-symlink file")
    return metadata


def _diag4_bound_file_bytes(
    path: Path,
    context: str,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
) -> bytes:
    if locked_leaf_bytes is not None:
        try:
            return locked_leaf_bytes[path]
        except KeyError as error:
            raise ValueError(f"{context} is not held by the authority claim") from error
    _diag4_regular_file(path, context)
    return path.read_bytes()


def _diag4_numerical_identity(value: JsonValue) -> Diag4NumericalIdentity:
    payload = _mapping(value, "DIAG4 numerical identity")
    _exact_keys(payload, DIAG4_IDENTITY_FIELDS, "DIAG4 numerical identity")
    normalized = {
        name: _diag4_sha256(payload[name], f"DIAG4 numerical identity.{name}")
        for name in DIAG4_IDENTITY_FIELDS
    }
    if normalized["base_neq_gntr1_policy_sha256"] != DIAG4_BASE_POLICY_SHA256:
        raise ValueError("DIAG4 base NEQ-GNTR1 policy identity differs")
    expected_identity_sha256 = derive_diag4_numerical_identity_sha256(normalized)
    if normalized["identity_sha256"] != expected_identity_sha256:
        raise ValueError("DIAG4 aggregate numerical identity differs")
    return Diag4NumericalIdentity(**normalized)


def derive_diag4_numerical_identity_sha256(
    value: Mapping[str, str],
) -> str:
    """Derive the authoritative NEQ-GNTR3 aggregate from six component hashes."""

    component_fields = DIAG4_IDENTITY_FIELDS - {"identity_sha256"}
    if frozenset(value) not in {component_fields, DIAG4_IDENTITY_FIELDS}:
        raise ValueError("DIAG4 numerical identity component keys differ")
    normalized = {
        name: _diag4_sha256(value[name], f"DIAG4 identity component.{name}")
        for name in component_fields
    }
    identity_payload = (
        NEQ_GNTR3_SCHEMA_VERSION,
        DIAG4_NUMERICAL_ROUTE,
        normalized["base_neq_gntr1_policy_sha256"],
        normalized["problem_sha256"],
        normalized["optimizer_options_sha256"],
        normalized["scaling_sha256"],
        normalized["bootstrap_state_sha256"],
        normalized["initial_physical_state_sha256"],
    )
    return exact_numeric_tree_sha256(identity_payload)


def _diag4_qualified_files(value: JsonValue) -> dict[str, str]:
    payload = _mapping(value, "DIAG4 qualified files")
    qualified = {
        _diag4_relative_path(relative, "DIAG4 qualified file path"): _diag4_sha256(
            digest, f"DIAG4 qualified file {relative}"
        )
        for relative, digest in payload.items()
    }
    if frozenset(qualified) != DIAG4_QUALIFIED_FILE_PATHS:
        raise ValueError("DIAG4 qualified file membership differs")
    return qualified


def _diag4_frozen_numerical_entries(value: JsonValue) -> dict[str, str]:
    payload = _mapping(value, "DIAG4 frozen numerical entries")
    frozen = {
        _diag4_relative_path(relative, "DIAG4 frozen numerical path"): _diag4_sha256(
            digest, f"DIAG4 frozen numerical entry {relative}"
        )
        for relative, digest in payload.items()
    }
    if frozenset(frozen) != DIAG4_FROZEN_NUMERICAL_PATHS:
        raise ValueError("DIAG4 frozen numerical membership differs")
    return frozen


def _diag4_execution_source_membership(
    repository: Path,
    *,
    qualified: Mapping[str, str],
    frozen: Mapping[str, str],
) -> frozenset[str]:
    broad: set[str] = set()
    for root_name in DIAG4_EXECUTION_SOURCE_BROAD_ROOTS:
        root = repository / root_name
        for path in root.rglob("*.py"):
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode):
                broad.add(path.relative_to(repository).as_posix())
    return frozenset(
        broad | (set(qualified) - {DIAG4_EXECUTION_SOURCE_MANIFEST_PATH}) | set(frozen)
    )


def _diag4_execution_source_entries(
    manifest_bytes: bytes,
    *,
    repository: Path,
    qualified: Mapping[str, str],
    frozen: Mapping[str, str],
    locked_leaf_bytes: Mapping[Path, bytes] | None,
    expected_count: int | None = None,
) -> tuple[dict[str, str], dict[str, int], str]:
    manifest = _mapping(
        load_canonical_json_bytes(manifest_bytes),
        "DIAG4 execution-source manifest",
    )
    _exact_keys(
        manifest,
        frozenset({"schema_version", "entries", "entries_sha256"}),
        "DIAG4 execution-source manifest",
    )
    raw_entries = _mapping(
        manifest["entries"], "DIAG4 execution-source manifest entries"
    )
    entries_sha256 = _diag4_sha256(
        manifest["entries_sha256"], "DIAG4 execution-source entries SHA"
    )
    if (
        manifest["schema_version"] != DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION
        or hashlib.sha256(canonical_json_bytes(raw_entries)).hexdigest()
        != entries_sha256
    ):
        raise ValueError("DIAG4 execution-source manifest identity differs")
    entries: dict[str, str] = {}
    sizes: dict[str, int] = {}
    for raw_relative, raw_entry in raw_entries.items():
        relative = _diag4_relative_path(
            raw_relative, "DIAG4 execution-source entry path"
        )
        entry = _mapping(raw_entry, f"DIAG4 execution-source entry {relative}")
        _exact_keys(
            entry,
            frozenset({"sha256", "size_bytes"}),
            f"DIAG4 execution-source entry {relative}",
        )
        entries[relative] = _diag4_sha256(
            entry["sha256"], f"DIAG4 execution-source entry {relative} SHA"
        )
        size_bytes = _integer(
            entry["size_bytes"], f"DIAG4 execution-source entry {relative} size"
        )
        if size_bytes < 0:
            raise ValueError("DIAG4 execution-source entry size is negative")
        sizes[relative] = size_bytes
    if DIAG4_EXECUTION_SOURCE_MANIFEST_PATH in entries:
        raise ValueError("DIAG4 execution-source manifest contains itself")
    expected_membership = _diag4_execution_source_membership(
        repository,
        qualified=qualified,
        frozen=frozen,
    )
    required_count = (
        DIAG4_EXECUTION_SOURCE_ENTRY_COUNT if expected_count is None else expected_count
    )
    if frozenset(entries) != expected_membership or len(entries) != required_count:
        raise ValueError("DIAG4 execution-source membership differs")
    for relative, digest in entries.items():
        source_bytes = _diag4_bound_file_bytes(
            repository / relative,
            f"DIAG4 execution source {relative}",
            locked_leaf_bytes,
        )
        if (
            len(source_bytes) != sizes[relative]
            or hashlib.sha256(source_bytes).hexdigest() != digest
        ):
            raise ValueError(f"DIAG4 execution source differs: {relative}")
    for relative in frozenset(entries) & frozenset(qualified):
        if entries[relative] != qualified[relative]:
            raise ValueError("DIAG4 qualified execution-source identity conflicts")
    for relative in frozenset(entries) & frozenset(frozen):
        if entries[relative] != frozen[relative]:
            raise ValueError("DIAG4 frozen execution-source identity conflicts")
    return entries, sizes, entries_sha256


def _diag4_expected_cpu_source_entries(
    execution_entries: Mapping[str, str],
    execution_manifest_sha256: str,
) -> dict[str, str]:
    return {
        **execution_entries,
        DIAG4_EXECUTION_SOURCE_MANIFEST_PATH: execution_manifest_sha256,
    }


def _diag4_consumed_manifest(
    value: JsonValue,
    *,
    consumed_root: Path,
    locked_leaf_bytes: Mapping[Path, bytes] | None = None,
) -> tuple[Mapping[str, JsonValue], tuple[Mapping[str, JsonValue], ...]]:
    payload = _mapping(value, "DIAG4 consumed DIAG3")
    _exact_keys(
        payload,
        frozenset({"root", "evidence_manifest_sha256", "entries"}),
        "DIAG4 consumed DIAG3",
    )
    if (
        Path(_string(payload["root"], "DIAG4 consumed DIAG3 root")).resolve(strict=True)
        != consumed_root
    ):
        raise ValueError("DIAG4 consumed DIAG3 root differs")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise TypeError("DIAG4 consumed DIAG3 entries must be an array")
    entries: list[Mapping[str, JsonValue]] = []
    observed_paths: list[str] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"DIAG4 consumed entry {index}")
        _exact_keys(
            entry,
            frozenset({"relative_path", "sha256", "size_bytes"}),
            f"DIAG4 consumed entry {index}",
        )
        relative = _diag4_relative_path(
            entry["relative_path"], f"DIAG4 consumed entry {index} path"
        )
        digest = _diag4_sha256(entry["sha256"], f"DIAG4 consumed entry {index} SHA")
        size_bytes = _integer(entry["size_bytes"], f"DIAG4 consumed entry {index} size")
        if size_bytes < 0:
            raise ValueError("DIAG4 consumed entry size must be nonnegative")
        path = consumed_root / relative
        entry_bytes = _diag4_bound_file_bytes(
            path,
            f"DIAG4 consumed entry {relative}",
            locked_leaf_bytes,
        )
        if (
            len(entry_bytes) != size_bytes
            or hashlib.sha256(entry_bytes).hexdigest() != digest
        ):
            raise ValueError(f"DIAG4 consumed DIAG3 entry differs: {relative}")
        observed_paths.append(relative)
        entries.append(
            MappingProxyType(
                {
                    "relative_path": relative,
                    "sha256": digest,
                    "size_bytes": size_bytes,
                }
            )
        )
    if observed_paths != sorted(set(observed_paths)):
        raise ValueError("DIAG4 consumed DIAG3 entries are not sorted and unique")
    if not DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS.issubset(observed_paths):
        raise ValueError("DIAG4 consumed DIAG3 manifest omits required evidence")
    manifest_payload = [dict(entry) for entry in entries]
    if (
        _diag4_sha256(
            payload["evidence_manifest_sha256"],
            "DIAG4 consumed evidence manifest SHA",
        )
        != hashlib.sha256(canonical_json_bytes(manifest_payload)).hexdigest()
    ):
        raise ValueError("DIAG4 consumed DIAG3 evidence manifest differs")
    return payload, tuple(entries)


def _diag4_native_manifest_entries(
    manifest_bytes: bytes,
) -> tuple[tuple[str, str, int], ...]:
    payload = _mapping(
        load_canonical_json_bytes(manifest_bytes),
        "DIAG4 native-reference manifest",
    )
    _exact_keys(
        payload,
        frozenset({"schema_version", "entries"}),
        "DIAG4 native-reference manifest",
    )
    if not _string(payload["schema_version"], "DIAG4 native-reference manifest schema"):
        raise ValueError("DIAG4 native-reference manifest schema is empty")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("DIAG4 native-reference manifest entries are empty")
    entries: list[tuple[str, str, int]] = []
    for index, raw_entry in enumerate(raw_entries):
        entry = _mapping(raw_entry, f"DIAG4 native-reference entry {index}")
        _exact_keys(
            entry,
            frozenset({"relative_path", "sha256", "size_bytes"}),
            f"DIAG4 native-reference entry {index}",
        )
        relative = _diag4_relative_path(
            entry["relative_path"], f"DIAG4 native-reference entry {index} path"
        )
        digest = _diag4_sha256(
            entry["sha256"], f"DIAG4 native-reference entry {index} SHA"
        )
        size_bytes = _integer(
            entry["size_bytes"], f"DIAG4 native-reference entry {index} size"
        )
        if size_bytes < 0:
            raise ValueError("DIAG4 native-reference entry size must be nonnegative")
        entries.append((relative, digest, size_bytes))
    if [entry[0] for entry in entries] != sorted({entry[0] for entry in entries}):
        raise ValueError("DIAG4 native-reference entries are not sorted and unique")
    return tuple(entries)


def _validate_diag4_native_tree(
    manifest_bytes: bytes,
    *,
    reference_root: Path,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
) -> None:
    for relative, digest, size_bytes in _diag4_native_manifest_entries(manifest_bytes):
        entry_bytes = _diag4_bound_file_bytes(
            reference_root / relative,
            f"DIAG4 native-reference entry {relative}",
            locked_leaf_bytes,
        )
        if (
            len(entry_bytes) != size_bytes
            or hashlib.sha256(entry_bytes).hexdigest() != digest
        ):
            raise ValueError(f"DIAG4 native-reference entry differs: {relative}")


def _diag4_command_receipt(
    value: JsonValue,
    *,
    context: str,
    expected_command: str,
    passed_type: type[bool | int],
) -> Mapping[str, JsonValue]:
    receipt = _mapping(value, context)
    _exact_keys(
        receipt,
        frozenset({"command", "duration_seconds", "exit_code", "passed", "run_count"}),
        context,
    )
    passed = receipt["passed"]
    if passed_type is bool:
        passed_valid = _boolean(passed, f"{context}.passed") is True
    else:
        passed_valid = _integer(passed, f"{context}.passed") > 0
    if (
        _string(receipt["command"], f"{context}.command") != expected_command
        or _integer(receipt["run_count"], f"{context}.run_count") != 1
        or _integer(receipt["exit_code"], f"{context}.exit_code") != 0
        or not passed_valid
    ):
        raise ValueError(f"{context} differs")
    _finite_duration(receipt["duration_seconds"], f"{context}.duration_seconds")
    return receipt


def _diag4_historical_cpu20(
    value: JsonValue,
    *,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
) -> Mapping[str, JsonValue]:
    evidence = _mapping(value, "DIAG4 historical CPU20")
    _exact_keys(
        evidence,
        frozenset(
            {
                "accepted_steps",
                "attempts",
                "command",
                "duration_seconds",
                "exit_code",
                "git_head",
                "harness_path",
                "harness_sha256",
                "one_shot_no_retry",
                "promotion_eligible",
                "result_path",
                "result_sha256",
                "run_count",
                "use",
            }
        ),
        "DIAG4 historical CPU20",
    )
    expected = {
        "accepted_steps": 20,
        "attempts": 20,
        "command": DIAG4_CPU20_COMMAND,
        "exit_code": 0,
        "git_head": "52dea17ddf3012cf923fc92da78c0d73a17f4625",
        "harness_path": str(DIAG4_CPU20_HARNESS_PATH),
        "harness_sha256": DIAG4_CPU20_HARNESS_SHA256,
        "one_shot_no_retry": True,
        "promotion_eligible": False,
        "result_path": str(DIAG4_CPU20_RESULT_PATH),
        "result_sha256": DIAG4_CPU20_RESULT_SHA256,
        "run_count": 1,
        "use": "ROUTE_SELECTION_ONLY",
    }
    if any(
        evidence[name] != expected_value for name, expected_value in expected.items()
    ):
        raise ValueError("DIAG4 historical CPU20 identity differs")
    if (
        _finite_duration(
            evidence["duration_seconds"], "DIAG4 historical CPU20 duration"
        )
        != DIAG4_CPU20_DURATION_SECONDS
    ):
        raise ValueError("DIAG4 historical CPU20 duration differs")
    for path, digest, context in (
        (DIAG4_CPU20_RESULT_PATH, DIAG4_CPU20_RESULT_SHA256, "result"),
        (DIAG4_CPU20_HARNESS_PATH, DIAG4_CPU20_HARNESS_SHA256, "harness"),
    ):
        payload = _diag4_bound_file_bytes(
            path, f"DIAG4 historical CPU20 {context}", locked_leaf_bytes
        )
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"DIAG4 historical CPU20 {context} bytes differ")
    return evidence


def _diag4_cpu_manifest_entries(
    manifest_bytes: bytes,
    *,
    expected_schema: str = DIAG4_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, int], ...], str, str]:
    manifest = _mapping(
        load_canonical_json_bytes(manifest_bytes), "DIAG4 CPU qualification manifest"
    )
    _exact_keys(
        manifest,
        frozenset(
            {
                "directories",
                "execution_source_entries_sha256",
                "execution_source_manifest_sha256",
                "files",
                "schema_version",
            }
        ),
        "DIAG4 CPU qualification manifest",
    )
    if manifest["schema_version"] != expected_schema:
        raise ValueError("DIAG4 CPU qualification manifest schema differs")
    directories = manifest["directories"]
    if not isinstance(directories, list):
        raise TypeError("DIAG4 CPU qualification directories must be an array")
    directory_paths: list[str] = []
    for index, raw_directory in enumerate(directories):
        directory = _mapping(raw_directory, f"DIAG4 CPU directory {index}")
        _exact_keys(
            directory,
            frozenset({"mode", "relative_path"}),
            f"DIAG4 CPU directory {index}",
        )
        if directory["mode"] != "0555":
            raise ValueError("DIAG4 CPU qualification directory mode differs")
        directory_paths.append(
            _diag4_relative_path(
                directory["relative_path"], f"DIAG4 CPU directory {index} path"
            )
        )
    if directory_paths != sorted(set(directory_paths)):
        raise ValueError(
            "DIAG4 CPU qualification directories are not sorted and unique"
        )
    files = manifest["files"]
    if not isinstance(files, list):
        raise TypeError("DIAG4 CPU qualification files must be an array")
    entries: list[tuple[str, str, int]] = []
    for index, raw_file in enumerate(files):
        entry = _mapping(raw_file, f"DIAG4 CPU file {index}")
        _exact_keys(
            entry,
            frozenset({"mode", "relative_path", "sha256", "size_bytes"}),
            f"DIAG4 CPU file {index}",
        )
        if entry["mode"] != "0444":
            raise ValueError("DIAG4 CPU qualification file mode differs")
        relative = _diag4_relative_path(
            entry["relative_path"], f"DIAG4 CPU file {index} path"
        )
        size_bytes = _integer(entry["size_bytes"], f"DIAG4 CPU file {index} size")
        if size_bytes < 0:
            raise ValueError("DIAG4 CPU qualification file size is negative")
        entries.append(
            (
                relative,
                _diag4_sha256(entry["sha256"], f"DIAG4 CPU file {index} SHA"),
                size_bytes,
            )
        )
    paths = [entry[0] for entry in entries]
    required = {
        "endpoint-audit.json",
        "evidence-index.json",
        "history.json",
        "policy.json",
        "safeguard-telemetry.json",
        "scientific-evidence.json",
        "terminal-numerical.json",
        "source-snapshot/source-manifest.json",
    }
    if (
        paths != sorted(set(paths))
        or not required.issubset(paths)
        or not any(
            path.startswith("arrays/") and path.endswith(".npy") for path in paths
        )
    ):
        raise ValueError("DIAG4 CPU qualification file membership differs")
    return (
        tuple(directory_paths),
        tuple(entries),
        _diag4_sha256(
            manifest["execution_source_manifest_sha256"],
            "DIAG4 CPU manifest execution-source manifest SHA",
        ),
        _diag4_sha256(
            manifest["execution_source_entries_sha256"],
            "DIAG4 CPU manifest execution-source entries SHA",
        ),
    )


def _diag4_decisive_cpu_qualification(
    value: JsonValue,
    *,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
    expected_numerical_identity: JsonValue,
    expected_source_entries: Mapping[str, str],
    expected_execution_source_manifest_sha256: str,
    expected_execution_source_entries_sha256: str,
    expected_native_extension_path: Path,
    expected_native_extension_sha256: str,
    expected_native_extension_size_bytes: int,
) -> Mapping[str, JsonValue]:
    evidence = _mapping(value, "DIAG4 decisive CPU qualification")
    _exact_keys(
        evidence,
        frozenset(
            {
                "artifact_manifest_sha256",
                "command",
                "duration_seconds",
                "exit_code",
                "qualification_passed",
                "root",
                "run_count",
                "schema_version",
                "scientific_evidence_sha256",
                "scientific_outcome",
            }
        ),
        "DIAG4 decisive CPU qualification",
    )
    if (
        evidence["schema_version"] != DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION
        or evidence["root"] != str(DIAG4_CPU_QUALIFICATION_ROOT)
        or evidence["command"] != DIAG4_CPU_QUALIFICATION_COMMAND
        or _integer(evidence["run_count"], "DIAG4 decisive CPU run_count") != 1
        or _integer(evidence["exit_code"], "DIAG4 decisive CPU exit_code") != 0
        or _boolean(evidence["qualification_passed"], "DIAG4 decisive CPU pass flag")
        is not True
        or evidence["scientific_outcome"] != "QUALITY_HIT"
    ):
        raise ValueError("DIAG4 decisive CPU qualification identity differs")
    _finite_duration(
        evidence["duration_seconds"], "DIAG4 decisive CPU qualification duration"
    )
    manifest_path = DIAG4_CPU_QUALIFICATION_ROOT / "artifact-manifest.json"
    manifest_bytes = _diag4_bound_file_bytes(
        manifest_path, "DIAG4 decisive CPU manifest", locked_leaf_bytes
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != _diag4_sha256(
        evidence["artifact_manifest_sha256"], "DIAG4 decisive CPU manifest SHA"
    ):
        raise ValueError("DIAG4 decisive CPU manifest bytes differ")
    directories, entries, manifest_source_sha256, manifest_entries_sha256 = (
        _diag4_cpu_manifest_entries(manifest_bytes)
    )
    if (
        manifest_source_sha256 != expected_execution_source_manifest_sha256
        or manifest_entries_sha256 != expected_execution_source_entries_sha256
    ):
        raise ValueError("DIAG4 decisive CPU manifest source identity differs")
    tree_paths = tuple(DIAG4_CPU_QUALIFICATION_ROOT.rglob("*"))
    if any(
        path.is_symlink() or (not path.is_file() and not path.is_dir())
        for path in tree_paths
    ):
        raise ValueError("DIAG4 decisive CPU artifact contains a special path")
    actual_files = tuple(
        sorted(
            path.relative_to(DIAG4_CPU_QUALIFICATION_ROOT).as_posix()
            for path in DIAG4_CPU_QUALIFICATION_ROOT.rglob("*")
            if path.is_file()
        )
    )
    expected_files = tuple(
        sorted({"artifact-manifest.json", *(relative for relative, _, _ in entries)})
    )
    actual_directories = tuple(
        sorted(
            path.relative_to(DIAG4_CPU_QUALIFICATION_ROOT).as_posix()
            for path in DIAG4_CPU_QUALIFICATION_ROOT.rglob("*")
            if path.is_dir()
        )
    )
    if actual_files != expected_files or actual_directories != directories:
        raise ValueError("DIAG4 decisive CPU artifact tree is not closed")
    if (
        stat.S_IMODE(DIAG4_CPU_QUALIFICATION_ROOT.stat().st_mode) != 0o555
        or stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
        or manifest_path.stat().st_nlink != 1
        or any(
            stat.S_IMODE((DIAG4_CPU_QUALIFICATION_ROOT / relative).stat().st_mode)
            != 0o555
            for relative in directories
        )
    ):
        raise ValueError("DIAG4 decisive CPU artifact directory mode differs")
    for relative, digest, size_bytes in entries:
        if (
            stat.S_IMODE((DIAG4_CPU_QUALIFICATION_ROOT / relative).stat().st_mode)
            != 0o444
            or (DIAG4_CPU_QUALIFICATION_ROOT / relative).stat().st_nlink != 1
        ):
            raise ValueError("DIAG4 decisive CPU artifact file mode differs")
        payload = _diag4_bound_file_bytes(
            DIAG4_CPU_QUALIFICATION_ROOT / relative,
            f"DIAG4 decisive CPU file {relative}",
            locked_leaf_bytes,
        )
        if len(payload) != size_bytes or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError(f"DIAG4 decisive CPU file differs: {relative}")
    scientific_bytes = _diag4_bound_file_bytes(
        DIAG4_CPU_QUALIFICATION_ROOT / "scientific-evidence.json",
        "DIAG4 decisive CPU scientific evidence",
        locked_leaf_bytes,
    )
    if hashlib.sha256(scientific_bytes).hexdigest() != _diag4_sha256(
        evidence["scientific_evidence_sha256"],
        "DIAG4 decisive CPU scientific evidence SHA",
    ):
        raise ValueError("DIAG4 decisive CPU scientific evidence bytes differ")
    scientific = _mapping(
        load_canonical_json_bytes(scientific_bytes),
        "DIAG4 decisive CPU scientific evidence",
    )
    _exact_keys(
        scientific,
        frozenset(
            {
                "backend",
                "callback_count",
                "configuration_fingerprint",
                "input_fingerprint",
                "native_reference_artifact_sha256",
                "numerical_identity",
                "output_root",
                "policy_sha256",
                "promotion_eligible",
                "qualification_passed",
                "route",
                "runtime",
                "schema_version",
                "scientific_outcome",
                "execution_source_manifest_sha256",
                "execution_source_entries_sha256",
                "prequalification_plan_control",
                "native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "source_manifest_entries",
                "source_manifest_sha256",
                "speed",
                "synchronized_solve_seconds",
                "timings_monotonic_ns",
            }
        ),
        "DIAG4 decisive CPU scientific evidence",
    )
    runtime = _mapping(scientific["runtime"], "DIAG4 decisive CPU runtime")
    if (
        scientific.get("schema_version") != DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION
        or scientific.get("route") != DIAG4_NUMERICAL_ROUTE
        or scientific.get("backend") != "cpu"
        or scientific.get("output_root") != str(DIAG4_CPU_QUALIFICATION_ROOT)
        or scientific.get("qualification_passed") is not True
        or scientific.get("promotion_eligible") is not False
        or scientific.get("scientific_outcome") != "QUALITY_HIT"
        or scientific.get("speed") != "NOT_PRODUCED"
        or scientific.get("callback_count") != 0
        or scientific.get("execution_source_manifest_sha256")
        != expected_execution_source_manifest_sha256
        or scientific.get("execution_source_entries_sha256")
        != expected_execution_source_entries_sha256
        or scientific.get("native_extension_path")
        != str(expected_native_extension_path)
        or scientific.get("native_extension_sha256") != expected_native_extension_sha256
        or scientific.get("native_extension_size_bytes")
        != expected_native_extension_size_bytes
        or scientific.get("numerical_identity") != expected_numerical_identity
        or runtime.get("backend") != "cpu"
        or runtime.get("x64_enabled") is not True
        or runtime.get("native_extension_path") != str(expected_native_extension_path)
        or runtime.get("native_extension_sha256") != expected_native_extension_sha256
        or runtime.get("native_extension_size_bytes")
        != expected_native_extension_size_bytes
    ):
        raise ValueError("DIAG4 decisive CPU scientific result differs")
    snapshot = load_snapshot(
        DIAG4_CPU_QUALIFICATION_ROOT / "source-snapshot",
        required_roles=DIAG4_CPU_SNAPSHOT_ROLES,
    )
    source_entries = [
        {
            "relative_path": entry.relative_path,
            "role": entry.role,
            "sha256": entry.sha256,
            "size_bytes": entry.size_bytes,
        }
        for entry in snapshot.entries
    ]
    if (
        scientific["source_manifest_sha256"] != snapshot.manifest_sha256
        or scientific["source_manifest_entries"] != source_entries
    ):
        raise ValueError("DIAG4 decisive CPU source manifest summary differs")
    repository_entries = {
        entry.relative_path: entry.sha256
        for entry in snapshot.entries
        if entry.role not in {"native_extension", "prequalification_plan"}
    }
    native_entries = tuple(
        entry for entry in snapshot.entries if entry.role == "native_extension"
    )
    plan_entries = tuple(
        entry for entry in snapshot.entries if entry.role == "prequalification_plan"
    )
    plan_control = _mapping(
        scientific["prequalification_plan_control"],
        "DIAG4 prequalification plan control",
    )
    _exact_keys(
        plan_control,
        frozenset(
            {
                "schema_version",
                "snapshot_relative_path",
                "source_relative_path",
                "sha256",
                "size_bytes",
                "plan_prefix_sha256",
            }
        ),
        "DIAG4 prequalification plan control",
    )
    if (
        repository_entries != dict(expected_source_entries)
        or len(native_entries) != 1
        or len(plan_entries) != 1
        or plan_control["schema_version"]
        != DIAG4_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION
        or plan_control["snapshot_relative_path"]
        != DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH
        or plan_control["source_relative_path"] != DIAG4_PLAN_RELATIVE_PATH
        or plan_control["sha256"] != DIAG4_PREQUALIFICATION_PLAN_SHA256
        or plan_control["size_bytes"] != plan_entries[0].size_bytes
        or plan_control["plan_prefix_sha256"] != DIAG4_PLAN_SHA256
        or plan_entries[0].relative_path != DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH
        or plan_entries[0].sha256 != DIAG4_PREQUALIFICATION_PLAN_SHA256
        or native_entries[0].relative_path
        != f"native/{expected_native_extension_path.name}"
        or native_entries[0].sha256 != expected_native_extension_sha256
        or native_entries[0].size_bytes != expected_native_extension_size_bytes
    ):
        raise ValueError("DIAG4 decisive CPU source authority differs")
    evidence_index = _mapping(
        load_canonical_json_bytes(
            _diag4_bound_file_bytes(
                DIAG4_CPU_QUALIFICATION_ROOT / "evidence-index.json",
                "DIAG4 decisive CPU evidence index",
                locked_leaf_bytes,
            )
        ),
        "DIAG4 decisive CPU evidence index",
    )
    expected_evidence_names = frozenset(
        {
            "endpoint_audit",
            "history",
            "policy",
            "safeguard_telemetry",
            "terminal_numerical",
        }
    )
    _exact_keys(
        evidence_index, expected_evidence_names, "DIAG4 decisive CPU evidence index"
    )
    references: dict[str, ArtifactRef] = {}
    documents: dict[str, JsonValue] = {}
    for name in expected_evidence_names:
        reference_payload = _mapping(
            evidence_index[name], f"DIAG4 decisive CPU evidence reference {name}"
        )
        _exact_keys(
            reference_payload,
            frozenset({"relative_path", "schema_version", "sha256", "size_bytes"}),
            f"DIAG4 decisive CPU evidence reference {name}",
        )
        reference = ArtifactRef(
            relative_path=_diag4_relative_path(
                reference_payload["relative_path"], f"DIAG4 CPU evidence {name} path"
            ),
            sha256=_diag4_sha256(
                reference_payload["sha256"], f"DIAG4 CPU evidence {name} SHA"
            ),
            size_bytes=_integer(
                reference_payload["size_bytes"], f"DIAG4 CPU evidence {name} size"
            ),
            schema_version=_string(
                reference_payload["schema_version"],
                f"DIAG4 CPU evidence {name} schema",
            ),
        )
        document_bytes = _diag4_bound_file_bytes(
            DIAG4_CPU_QUALIFICATION_ROOT / reference.relative_path,
            f"DIAG4 decisive CPU evidence {name}",
            locked_leaf_bytes,
        )
        if (
            len(document_bytes) != reference.size_bytes
            or hashlib.sha256(document_bytes).hexdigest() != reference.sha256
        ):
            raise ValueError(f"DIAG4 decisive CPU evidence differs: {name}")
        references[name] = reference
        documents[name] = load_canonical_json_bytes(document_bytes)
    identity_payload = _mapping(
        expected_numerical_identity, "DIAG4 decisive CPU numerical identity"
    )
    scientific_evidence = validate_native_equivalent_scientific_evidence(
        artifact_root=DIAG4_CPU_QUALIFICATION_ROOT,
        history=documents["history"],
        safeguard_telemetry=documents["safeguard_telemetry"],
        terminal_numerical=documents["terminal_numerical"],
        policy=documents["policy"],
        endpoint_audit=documents["endpoint_audit"],
        expected_history_evidence=references["history"],
        expected_numerical_identity=NativeEquivalentNumericalIdentity(
            numerical_route=DIAG4_NUMERICAL_ROUTE,
            numerical_result_schema_version=DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
            **{name: identity_payload[name] for name in DIAG4_IDENTITY_FIELDS},
        ),
        backend="cpu",
    )
    if scientific_evidence.outcome is not ScientificOutcome.QUALITY_HIT:
        raise ValueError(
            "DIAG4 decisive CPU scientific reconstruction is not QUALITY_HIT"
        )
    return evidence


def _validate_diag4_qualification_record(
    plan_bytes: bytes,
    authority: Mapping[str, JsonValue],
    *,
    output_root: Path,
    execution_source_entries: Mapping[str, str],
    execution_source_manifest_sha256: str,
    execution_source_entries_sha256: str,
    native_extension_path: Path,
    native_extension_sha256: str,
    native_extension_size_bytes: int,
    locked_leaf_bytes: Mapping[Path, bytes] | None,
) -> None:
    prefix, marker, record_bytes = plan_bytes.partition(b"## Qualification Record\n")
    if not marker or not record_bytes:
        raise ValueError("DIAG4 qualification record is missing")
    prefix_sha256 = hashlib.sha256(prefix).hexdigest()
    if prefix_sha256 != DIAG4_PLAN_SHA256:
        raise ValueError("DIAG4 frozen plan prefix differs")
    record = _mapping(
        load_canonical_json_bytes(record_bytes), "DIAG4 qualification record"
    )
    expected_keys = frozenset(
        {
            "schema_version",
            "plan_prefix_sha256",
            "output_root",
            "controlling_cpu",
            "decisive_cpu_qualification",
            "static_checks",
            "static_commands",
            "qualified_files",
            "qualified_files_sha256",
            "frozen_numerical_entries",
            "frozen_numerical_entries_sha256",
            "execution_source_manifest_sha256",
            "execution_source_entries_sha256",
            "native_extension_path",
            "native_extension_sha256",
            "native_extension_size_bytes",
            "historical_cpu20",
            "native_reference_manifest_sha256",
            "consumed_diag3",
            "numerical_identity",
            "no_gpu_used_for_qualification",
            "independent_reviews",
            "authorization",
        }
    )
    _exact_keys(record, expected_keys, "DIAG4 qualification record")
    record_files = _diag4_qualified_files(record["qualified_files"])
    authority_files = _diag4_qualified_files(authority["qualified_files"])
    if (
        record["schema_version"] != DIAG4_QUALIFICATION_SCHEMA_VERSION
        or record["plan_prefix_sha256"] != DIAG4_PLAN_SHA256
        or record["output_root"] != str(output_root)
        or record_files != authority_files
        or _diag4_sha256(
            record["qualified_files_sha256"],
            "DIAG4 qualification qualified files SHA",
        )
        != authority["qualified_files_sha256"]
        or record["frozen_numerical_entries"] != authority["frozen_numerical_entries"]
        or record["frozen_numerical_entries_sha256"]
        != authority["frozen_numerical_entries_sha256"]
        or record["execution_source_manifest_sha256"]
        != execution_source_manifest_sha256
        or record["execution_source_entries_sha256"] != execution_source_entries_sha256
        or record["native_extension_path"] != str(native_extension_path)
        or record["native_extension_sha256"] != native_extension_sha256
        or record["native_extension_size_bytes"] != native_extension_size_bytes
        or record["historical_cpu20"] != authority["historical_cpu20"]
        or record["decisive_cpu_qualification"]
        != authority["decisive_cpu_qualification"]
        or _diag4_sha256(
            record["native_reference_manifest_sha256"],
            "DIAG4 qualification native-reference manifest SHA",
        )
        != authority["native_reference_manifest_sha256"]
        or record["consumed_diag3"] != authority["consumed_diag3"]
        or record["numerical_identity"] != authority["numerical_identity"]
        or record["no_gpu_used_for_qualification"] is not True
    ):
        raise ValueError("DIAG4 qualification authority differs")
    _diag4_historical_cpu20(
        record["historical_cpu20"], locked_leaf_bytes=locked_leaf_bytes
    )
    _diag4_decisive_cpu_qualification(
        record["decisive_cpu_qualification"],
        locked_leaf_bytes=locked_leaf_bytes,
        expected_numerical_identity=record["numerical_identity"],
        expected_source_entries=_diag4_expected_cpu_source_entries(
            execution_source_entries, execution_source_manifest_sha256
        ),
        expected_execution_source_manifest_sha256=execution_source_manifest_sha256,
        expected_execution_source_entries_sha256=execution_source_entries_sha256,
        expected_native_extension_path=native_extension_path,
        expected_native_extension_sha256=native_extension_sha256,
        expected_native_extension_size_bytes=native_extension_size_bytes,
    )
    _diag4_command_receipt(
        record["controlling_cpu"],
        context="DIAG4 controlling CPU",
        expected_command=DIAG4_CONTROLLING_CPU_COMMAND,
        passed_type=int,
    )
    static_checks = _mapping(record["static_checks"], "DIAG4 static checks")
    static_names = frozenset(
        {"ruff_check", "ruff_format_check", "compileall", "git_diff_check"}
    )
    _exact_keys(static_checks, static_names, "DIAG4 static checks")
    if not all(
        _boolean(static_checks[name], f"DIAG4 static checks.{name}")
        for name in static_names
    ):
        raise ValueError("DIAG4 static qualification differs")
    static_commands = _mapping(record["static_commands"], "DIAG4 static commands")
    _exact_keys(static_commands, static_names, "DIAG4 static commands")
    for name in static_names:
        _diag4_command_receipt(
            static_commands[name],
            context=f"DIAG4 static command {name}",
            expected_command=DIAG4_STATIC_COMMANDS[name],
            passed_type=bool,
        )
    reviews = record["independent_reviews"]
    if not isinstance(reviews, list) or len(reviews) != 4:
        raise ValueError("DIAG4 independent reviews are incomplete")
    reviewers: set[str] = set()
    sessions: set[str] = set()
    roles: set[str] = set()
    for index, raw_review in enumerate(reviews):
        review = _mapping(raw_review, f"DIAG4 independent review {index}")
        _exact_keys(
            review,
            frozenset(
                {
                    "reviewed_frozen_numerical_entries_sha256",
                    "reviewed_qualified_files_sha256",
                    "reviewed_execution_source_manifest_sha256",
                    "reviewed_execution_source_entries_sha256",
                    "reviewer",
                    "role",
                    "session",
                    "verdict",
                }
            ),
            f"DIAG4 independent review {index}",
        )
        reviewer = _string(review["reviewer"], f"DIAG4 reviewer {index}")
        role = _string(review["role"], f"DIAG4 review role {index}")
        session = _string(review["session"], f"DIAG4 review session {index}")
        verdict = _string(review["verdict"], f"DIAG4 review verdict {index}")
        if (
            not reviewer
            or not session
            or verdict != "GO"
            or review["reviewed_qualified_files_sha256"]
            != record["qualified_files_sha256"]
            or review["reviewed_frozen_numerical_entries_sha256"]
            != record["frozen_numerical_entries_sha256"]
            or review["reviewed_execution_source_manifest_sha256"]
            != execution_source_manifest_sha256
            or review["reviewed_execution_source_entries_sha256"]
            != execution_source_entries_sha256
        ):
            raise ValueError("DIAG4 independent review differs")
        reviewers.add(reviewer)
        sessions.add(session)
        roles.add(role)
    if len(reviewers) != 4 or len(sessions) != 4 or roles != DIAG4_REVIEW_ROLES:
        raise ValueError("DIAG4 independent reviewers are not distinct")
    authorization = _mapping(record["authorization"], "DIAG4 authorization")
    _exact_keys(
        authorization,
        frozenset(
            {
                "preflight_launches",
                "maximum_cold_launches",
                "warm_allowed",
                "retry_allowed",
            }
        ),
        "DIAG4 authorization",
    )
    if (
        _integer(
            authorization["preflight_launches"],
            "DIAG4 authorization.preflight_launches",
        )
        != 1
        or _integer(
            authorization["maximum_cold_launches"],
            "DIAG4 authorization.maximum_cold_launches",
        )
        != 1
        or _boolean(authorization["warm_allowed"], "DIAG4 authorization.warm_allowed")
        is not False
        or _boolean(authorization["retry_allowed"], "DIAG4 authorization.retry_allowed")
        is not False
    ):
        raise ValueError("DIAG4 qualification launch cardinality differs")
    if (
        authority["qualification_record_sha256"]
        != hashlib.sha256(record_bytes).hexdigest()
    ):
        raise ValueError("DIAG4 qualification record hash differs")


def _validate_diag4_authority_bytes(
    authority_bytes: bytes,
    *,
    repository: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
    require_output_absent: bool,
    locked_leaf_bytes: Mapping[Path, bytes] | None = None,
) -> tuple[Mapping[str, JsonValue], Diag4NumericalIdentity, bytes]:
    payload = _mapping(load_canonical_json_bytes(authority_bytes), "DIAG4 authority")
    _exact_keys(
        payload,
        frozenset(
            {
                "schema_version",
                "route",
                "numerical_route",
                "scientific_evidence_schema",
                "plan_prefix_sha256",
                "completed_plan_sha256",
                "qualification_record_sha256",
                "qualified_files",
                "qualified_files_sha256",
                "frozen_numerical_entries",
                "frozen_numerical_entries_sha256",
                "execution_source_manifest_sha256",
                "execution_source_entries_sha256",
                "historical_cpu20",
                "decisive_cpu_qualification",
                "native_extension_path",
                "native_extension_sha256",
                "native_extension_size_bytes",
                "native_reference_manifest_sha256",
                "consumed_diag3",
                "numerical_identity",
                "execution_policy",
                "launch",
            }
        ),
        "DIAG4 authority",
    )
    if (
        payload["schema_version"] != DIAG4_AUTHORITY_SCHEMA_VERSION
        or payload["route"] != DIAG4_ROUTE
        or payload["numerical_route"] != DIAG4_NUMERICAL_ROUTE
        or payload["scientific_evidence_schema"] != DIAG4_SCHEMA_VERSION
        or payload["plan_prefix_sha256"] != DIAG4_PLAN_SHA256
    ):
        raise ValueError("DIAG4 authority identity differs")
    for name in (
        "completed_plan_sha256",
        "qualification_record_sha256",
        "qualified_files_sha256",
        "frozen_numerical_entries_sha256",
        "execution_source_manifest_sha256",
        "execution_source_entries_sha256",
        "native_extension_sha256",
        "native_reference_manifest_sha256",
    ):
        _diag4_sha256(payload[name], f"DIAG4 authority.{name}")
    execution = _mapping(payload["execution_policy"], "DIAG4 execution policy")
    _exact_keys(
        execution,
        frozenset(
            {
                "parent_platform",
                "child_platform",
                "jax_enable_x64",
                "compilation_cache_enabled",
                "child_preallocate",
                "command_buffer_enabled",
                "required_xla_flag",
            }
        ),
        "DIAG4 execution policy",
    )
    if (
        _string(execution["parent_platform"], "DIAG4 execution parent platform")
        != "cpu"
        or _string(execution["child_platform"], "DIAG4 execution child platform")
        != "cuda"
        or _boolean(execution["jax_enable_x64"], "DIAG4 execution x64") is not True
        or _boolean(
            execution["compilation_cache_enabled"],
            "DIAG4 execution compilation cache",
        )
        is not False
        or _boolean(execution["child_preallocate"], "DIAG4 execution preallocate")
        is not True
        or _boolean(
            execution["command_buffer_enabled"],
            "DIAG4 execution command buffer",
        )
        is not False
        or _string(execution["required_xla_flag"], "DIAG4 execution required XLA flag")
        != "--xla_gpu_enable_command_buffer="
    ):
        raise ValueError("DIAG4 execution policy differs")
    launch = _mapping(payload["launch"], "DIAG4 launch")
    _exact_keys(
        launch,
        frozenset(
            {
                "output_root",
                "reference_root",
                "input_root",
                "interpreter",
                "gpu_uuid",
                "preflight_launches",
                "maximum_cold_launches",
                "warm_allowed",
                "retry_allowed",
            }
        ),
        "DIAG4 launch",
    )
    expected_paths = {
        "output_root": output_root,
        "reference_root": reference_root,
        "input_root": input_root,
        "interpreter": interpreter,
    }
    for name, expected in expected_paths.items():
        actual = Path(_string(launch[name], f"DIAG4 launch.{name}"))
        actual = (
            actual.absolute() if name == "output_root" else actual.resolve(strict=True)
        )
        if actual != expected:
            raise ValueError(f"DIAG4 launch {name} differs")
    if (
        _string(launch["gpu_uuid"], "DIAG4 launch.gpu_uuid") != DIAG4_GPU_UUID
        or _integer(launch["preflight_launches"], "DIAG4 launch.preflight_launches")
        != 1
        or _integer(
            launch["maximum_cold_launches"],
            "DIAG4 launch.maximum_cold_launches",
        )
        != 1
        or _boolean(launch["warm_allowed"], "DIAG4 launch.warm_allowed") is not False
        or _boolean(launch["retry_allowed"], "DIAG4 launch.retry_allowed") is not False
    ):
        raise ValueError("DIAG4 launch policy differs")
    if require_output_absent:
        _validate_diag4_output_state(output_root, before_consumption=True)
    qualified = _diag4_qualified_files(payload["qualified_files"])
    if (
        payload["qualified_files_sha256"]
        != hashlib.sha256(canonical_json_bytes(qualified)).hexdigest()
    ):
        raise ValueError("DIAG4 qualified source manifest differs")
    for relative, digest in qualified.items():
        path = repository / relative
        source_bytes = _diag4_bound_file_bytes(
            path,
            f"DIAG4 qualified source {relative}",
            locked_leaf_bytes,
        )
        if hashlib.sha256(source_bytes).hexdigest() != digest:
            raise ValueError(f"DIAG4 qualified source differs: {relative}")
    frozen_entries = _diag4_frozen_numerical_entries(
        payload["frozen_numerical_entries"]
    )
    if (
        payload["frozen_numerical_entries_sha256"]
        != hashlib.sha256(canonical_json_bytes(frozen_entries)).hexdigest()
    ):
        raise ValueError("DIAG4 frozen numerical manifest differs")
    for relative, digest in frozen_entries.items():
        path = repository / relative
        source_bytes = _diag4_bound_file_bytes(
            path,
            f"DIAG4 frozen numerical source {relative}",
            locked_leaf_bytes,
        )
        if hashlib.sha256(source_bytes).hexdigest() != digest:
            raise ValueError(f"DIAG4 frozen numerical source differs: {relative}")
    execution_manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    execution_manifest_bytes = _diag4_bound_file_bytes(
        execution_manifest_path,
        "DIAG4 execution-source manifest",
        locked_leaf_bytes,
    )
    execution_manifest_sha256 = hashlib.sha256(execution_manifest_bytes).hexdigest()
    if execution_manifest_sha256 != payload["execution_source_manifest_sha256"]:
        raise ValueError("DIAG4 execution-source manifest bytes differ")
    execution_entries, _execution_entry_sizes, execution_entries_sha256 = (
        _diag4_execution_source_entries(
            execution_manifest_bytes,
            repository=repository,
            qualified=qualified,
            frozen=frozen_entries,
            locked_leaf_bytes=locked_leaf_bytes,
        )
    )
    if execution_entries_sha256 != payload["execution_source_entries_sha256"]:
        raise ValueError("DIAG4 execution-source entry aggregate differs")
    native_extension_text = _string(
        payload["native_extension_path"], "DIAG4 native extension path"
    )
    native_extension_path = Path(native_extension_text).resolve(strict=True)
    if native_extension_text != str(native_extension_path):
        raise ValueError("DIAG4 native extension path is not canonical")
    native_extension_bytes = _diag4_bound_file_bytes(
        native_extension_path,
        "DIAG4 native extension",
        locked_leaf_bytes,
    )
    native_extension_size_bytes = _integer(
        payload["native_extension_size_bytes"], "DIAG4 native extension size"
    )
    if (
        native_extension_size_bytes < 0
        or len(native_extension_bytes) != native_extension_size_bytes
        or hashlib.sha256(native_extension_bytes).hexdigest()
        != payload["native_extension_sha256"]
    ):
        raise ValueError("DIAG4 native extension identity differs")
    native_manifest = reference_root / "artifact-manifest.json"
    native_manifest_bytes = _diag4_bound_file_bytes(
        native_manifest,
        "DIAG4 native reference manifest",
        locked_leaf_bytes,
    )
    if (
        hashlib.sha256(native_manifest_bytes).hexdigest()
        != payload["native_reference_manifest_sha256"]
    ):
        raise ValueError("DIAG4 native-reference manifest differs")
    _validate_diag4_native_tree(
        native_manifest_bytes,
        reference_root=reference_root,
        locked_leaf_bytes=locked_leaf_bytes,
    )
    consumed_root = DIAG4_CONSUMED_DIAG3_ROOT.resolve(strict=True)
    _diag4_consumed_manifest(
        payload["consumed_diag3"],
        consumed_root=consumed_root,
        locked_leaf_bytes=locked_leaf_bytes,
    )
    _diag4_historical_cpu20(
        payload["historical_cpu20"], locked_leaf_bytes=locked_leaf_bytes
    )
    _diag4_decisive_cpu_qualification(
        payload["decisive_cpu_qualification"],
        locked_leaf_bytes=locked_leaf_bytes,
        expected_numerical_identity=payload["numerical_identity"],
        expected_source_entries=_diag4_expected_cpu_source_entries(
            execution_entries, execution_manifest_sha256
        ),
        expected_execution_source_manifest_sha256=execution_manifest_sha256,
        expected_execution_source_entries_sha256=execution_entries_sha256,
        expected_native_extension_path=native_extension_path,
        expected_native_extension_sha256=payload["native_extension_sha256"],
        expected_native_extension_size_bytes=native_extension_size_bytes,
    )
    identity = _diag4_numerical_identity(payload["numerical_identity"])
    plan_path = repository / DIAG4_PLAN_RELATIVE_PATH
    plan_bytes = _diag4_bound_file_bytes(
        plan_path,
        "DIAG4 completed plan",
        locked_leaf_bytes,
    )
    if hashlib.sha256(plan_bytes).hexdigest() != payload["completed_plan_sha256"]:
        raise ValueError("DIAG4 completed plan hash differs")
    _validate_diag4_qualification_record(
        plan_bytes,
        payload,
        output_root=output_root,
        execution_source_entries=execution_entries,
        execution_source_manifest_sha256=execution_manifest_sha256,
        execution_source_entries_sha256=execution_entries_sha256,
        native_extension_path=native_extension_path,
        native_extension_sha256=payload["native_extension_sha256"],
        native_extension_size_bytes=native_extension_size_bytes,
        locked_leaf_bytes=locked_leaf_bytes,
    )
    return payload, identity, plan_bytes


def diag4_consumption_marker_path(output_root: Path) -> Path:
    """Return the durable sibling marker for one exact DIAG4 output claim."""

    output = output_root.absolute()
    return output.parent / f".{output.name}.diag4-authority-consumed.json"


def _validate_diag4_output_state(
    output_root: Path,
    *,
    before_consumption: bool,
    staging_path: Path | None = None,
) -> None:
    partials = tuple(output_root.parent.glob(f"{output_root.name}.partial-*"))
    final_exists = os.path.lexists(output_root)
    marker_exists = os.path.lexists(diag4_consumption_marker_path(output_root))
    pending_markers = tuple(
        output_root.parent.glob(
            f"{diag4_consumption_marker_path(output_root).name}.pending-*"
        )
    )
    if before_consumption:
        expected_partials = () if staging_path is None else (staging_path,)
        if (
            final_exists
            or partials != expected_partials
            or marker_exists
            or pending_markers
        ):
            raise FileExistsError(
                "DIAG4 output root, staging sibling, or consumption marker exists"
            )
        return
    if not marker_exists or pending_markers:
        raise FileNotFoundError("DIAG4 authority consumption marker is absent")
    if (final_exists and partials) or len(partials) > 1:
        raise FileExistsError("DIAG4 competing final or staging outputs exist")
    if staging_path is not None and partials not in ((), (staging_path,)):
        raise FileExistsError("DIAG4 unbound staging output exists")


def validate_diag4_successor_authority(
    authority_path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
) -> Mapping[str, JsonValue]:
    """Validate unconsumed DIAG4 authority under the full retained claim locks."""

    with claim_diag4_successor_authority(
        authority_path,
        repository_root=repository_root,
        output_root=output_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
    ) as claim:
        return claim.payload


@contextmanager
def _diag4_claim_directories(
    directories: tuple[Path, ...],
) -> Iterator[Mapping[Path, int]]:
    resolved = {Path(os.path.abspath(path)) for path in directories}
    chain = tuple(
        sorted(
            {
                ancestor
                for directory in resolved
                for ancestor in (directory, *directory.parents)
            },
            key=lambda path: (len(path.parts), str(path)),
        )
    )
    descriptors: dict[Path, int] = {}
    try:
        for directory in chain:
            flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
            descriptor = (
                os.open(directory, flags)
                if directory.parent == directory
                else os.open(
                    directory.name,
                    flags,
                    dir_fd=descriptors[directory.parent],
                )
            )
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                os.close(descriptor)
                raise RuntimeError(
                    "DIAG4 successor authority or output is already claimed"
                ) from error
            descriptors[directory] = descriptor
            _assert_diag4_locked_directory_binding(directory, descriptor, descriptors)
        yield MappingProxyType(descriptors)
    finally:
        try:
            for directory, descriptor in descriptors.items():
                _assert_diag4_locked_directory_binding(
                    directory, descriptor, descriptors
                )
        finally:
            for descriptor in reversed(tuple(descriptors.values())):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def _assert_diag4_locked_directory_binding(
    directory: Path,
    descriptor: int,
    descriptors: Mapping[Path, int],
) -> None:
    locked = os.fstat(descriptor)
    bound = (
        os.stat(directory, follow_symlinks=False)
        if directory.parent == directory
        else os.stat(
            directory.name,
            dir_fd=descriptors[directory.parent],
            follow_symlinks=False,
        )
    )
    if not stat.S_ISDIR(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError(f"DIAG4 directory inode is not bound: {directory}")


def _assert_diag4_locked_file_binding(
    directory_descriptor: int,
    name: str,
    descriptor: int,
    context: str,
) -> None:
    locked = os.fstat(descriptor)
    bound = os.stat(
        name,
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError(f"DIAG4 {context} inode is not bound to its pathname")


def _open_diag4_locked_leaf(
    path: Path,
    directory_descriptors: Mapping[Path, int],
    context: str,
) -> _Diag4LockedLeaf:
    absolute = Path(os.path.abspath(path))
    parent_descriptor = directory_descriptors[absolute.parent]
    descriptor = os.open(
        absolute.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        dir_fd=parent_descriptor,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _assert_diag4_locked_file_binding(
            parent_descriptor,
            absolute.name,
            descriptor,
            context,
        )
        initial_bytes = _diag4_descriptor_bytes(descriptor)
        return _Diag4LockedLeaf(
            path=absolute,
            descriptor=descriptor,
            initial_sha256=hashlib.sha256(initial_bytes).hexdigest(),
            initial_size_bytes=len(initial_bytes),
            initial_mode=stat.S_IMODE(os.fstat(descriptor).st_mode),
        )
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        raise


def _assert_diag4_locked_leaf_binding(
    leaf: _Diag4LockedLeaf,
    directory_descriptors: Mapping[Path, int],
) -> bytes:
    _assert_diag4_locked_file_binding(
        directory_descriptors[leaf.path.parent],
        leaf.path.name,
        leaf.descriptor,
        f"bound leaf {leaf.path}",
    )
    observed = _diag4_descriptor_bytes(leaf.descriptor)
    if (
        len(observed) != leaf.initial_size_bytes
        or hashlib.sha256(observed).hexdigest() != leaf.initial_sha256
        or stat.S_IMODE(os.fstat(leaf.descriptor).st_mode) != leaf.initial_mode
    ):
        raise ValueError(f"DIAG4 bound leaf bytes differ: {leaf.path}")
    return observed


def _diag4_descriptor_bytes(descriptor: int) -> bytes:
    size = os.fstat(descriptor).st_size
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(descriptor, size - offset, offset)
        if not chunk:
            raise OSError("DIAG4 locked file read was incomplete")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _diag4_marker_payload(claim: Diag4SuccessorAuthorityClaim) -> dict[str, JsonValue]:
    return {
        "schema_version": DIAG4_CONSUMPTION_SCHEMA_VERSION,
        "route": DIAG4_ROUTE,
        "authority_sha256": claim.authority_sha256,
        "plan_prefix_sha256": claim.plan_prefix_sha256,
        "completed_plan_sha256": claim.completed_plan_sha256,
        "output_root": str(claim._lease.output_root),
    }


def _diag4_assert_active(claim: Diag4SuccessorAuthorityClaim) -> _Diag4AuthorityLease:
    lease = claim._lease
    if not lease.active:
        raise RuntimeError("DIAG4 successor authority claim is no longer active")
    return lease


def diag4_authority_lifecycle(
    claim: Diag4SuccessorAuthorityClaim,
) -> Diag4AuthorityLifecycle:
    """Return the authoritative held-claim lifecycle after any operation failure."""

    return _diag4_assert_active(claim).lifecycle_state


def validate_diag4_consumption_marker(
    claim: Diag4SuccessorAuthorityClaim,
) -> None:
    """Validate only the durable marker bound to a consumed held claim."""

    lease = _diag4_assert_active(claim)
    if lease.lifecycle_state not in {
        Diag4AuthorityLifecycle.CONSUMED,
        Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
    }:
        raise Diag4ConsumptionMarkerInvalidError(
            "DIAG4 authority has no consumed marker state"
        )
    if lease.consumed is None or lease.consumption_marker_descriptor is None:
        raise Diag4ConsumptionMarkerInvalidError(
            "DIAG4 consumption marker evidence is unavailable"
        )
    expected_bytes = canonical_json_bytes(_diag4_marker_payload(claim))
    try:
        _assert_diag4_locked_file_binding(
            lease.directory_descriptors[lease.output_root.parent],
            lease.consumed.path.name,
            lease.consumption_marker_descriptor,
            "authority consumption marker",
        )
        if (
            _diag4_descriptor_bytes(lease.consumption_marker_descriptor)
            != expected_bytes
            or hashlib.sha256(expected_bytes).hexdigest() != lease.consumed.sha256
        ):
            raise ValueError("DIAG4 authority consumption marker bytes differ")
    except (OSError, ValueError) as error:
        raise Diag4ConsumptionMarkerInvalidError(
            "DIAG4 authority consumption marker differs"
        ) from error


def _assert_diag4_staging_binding(lease: _Diag4AuthorityLease) -> None:
    if lease.staging_path is None or lease.staging_descriptor is None:
        raise RuntimeError("DIAG4 staging root is not bound")
    bound_path = (
        lease.staging_path if os.path.lexists(lease.staging_path) else lease.output_root
    )
    output_descriptor = lease.directory_descriptors[lease.output_root.parent]
    locked = os.fstat(lease.staging_descriptor)
    bound = os.stat(
        bound_path.name,
        dir_fd=output_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISDIR(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError("DIAG4 staging directory inode is not bound")


def bind_diag4_staging_root(
    claim: Diag4SuccessorAuthorityClaim,
    staging_root: Path,
) -> None:
    """Bind the one runner-created staging inode before consuming authority."""

    lease = _diag4_assert_active(claim)
    if lease.lifecycle_state is not Diag4AuthorityLifecycle.CLAIMED:
        raise RuntimeError("DIAG4 staging root is already bound")
    staging = staging_root.absolute()
    partial_prefix = f"{lease.output_root.name}.partial-"
    if (
        staging.parent != lease.output_root.parent
        or not staging.name.startswith(partial_prefix)
        or staging.name == partial_prefix
    ):
        raise ValueError("DIAG4 staging root path differs")
    _validate_diag4_output_state(
        lease.output_root,
        before_consumption=True,
        staging_path=staging,
    )
    output_descriptor = lease.directory_descriptors[lease.output_root.parent]
    descriptor = os.open(
        staging.name,
        os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        dir_fd=output_descriptor,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lease.staging_path = staging
        lease.staging_descriptor = descriptor
        lease.lifecycle_state = Diag4AuthorityLifecycle.STAGING_BOUND
        _assert_diag4_staging_binding(lease)
    except BaseException:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        lease.staging_path = None
        lease.staging_descriptor = None
        lease.lifecycle_state = Diag4AuthorityLifecycle.CLAIMED
        raise


def _validate_diag4_prelaunch_failure_output_state(
    lease: _Diag4AuthorityLease,
) -> None:
    partials = tuple(
        lease.output_root.parent.glob(f"{lease.output_root.name}.partial-*")
    )
    if (
        not os.path.lexists(lease.output_root)
        or partials
        or os.path.lexists(diag4_consumption_marker_path(lease.output_root))
    ):
        raise FileExistsError("DIAG4 finalized prelaunch failure output state differs")
    _assert_diag4_staging_binding(lease)


def revalidate_diag4_successor_authority(
    claim: Diag4SuccessorAuthorityClaim,
    *,
    require_output_absent: bool,
) -> None:
    """Revalidate every locked identity while the DIAG4 authority remains held."""

    lease = _diag4_assert_active(claim)
    consumption_started = lease.lifecycle_state in {
        Diag4AuthorityLifecycle.CONSUMED,
        Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
    }
    if require_output_absent == consumption_started:
        raise ValueError("DIAG4 revalidation phase differs from consumption state")
    if lease.lifecycle_state is Diag4AuthorityLifecycle.PRELAUNCH_FAILURE_FINALIZED:
        _validate_diag4_prelaunch_failure_output_state(lease)
    else:
        _validate_diag4_output_state(
            lease.output_root,
            before_consumption=not consumption_started,
            staging_path=lease.staging_path,
        )
        if lease.staging_descriptor is not None:
            _assert_diag4_staging_binding(lease)
    for directory, descriptor in lease.directory_descriptors.items():
        _assert_diag4_locked_directory_binding(
            directory, descriptor, lease.directory_descriptors
        )
    authority_directory = lease.authority_path.parent
    directory_descriptor = lease.directory_descriptors[authority_directory]
    _assert_diag4_locked_file_binding(
        directory_descriptor,
        lease.authority_path.name,
        lease.authority_descriptor,
        "authority",
    )
    _assert_diag4_locked_file_binding(
        lease.directory_descriptors[
            (lease.repository / DIAG4_PLAN_RELATIVE_PATH).parent
        ],
        Path(DIAG4_PLAN_RELATIVE_PATH).name,
        lease.plan_descriptor,
        "completed plan",
    )
    if _diag4_descriptor_bytes(lease.authority_descriptor) != lease.authority_bytes:
        raise ValueError("DIAG4 locked authority bytes differ")
    plan_bytes = _diag4_descriptor_bytes(lease.plan_descriptor)
    locked_leaf_bytes = {
        path: _assert_diag4_locked_leaf_binding(leaf, lease.directory_descriptors)
        for path, leaf in lease.locked_leaves.items()
    }
    locked_leaf_bytes[lease.authority_path] = lease.authority_bytes
    locked_leaf_bytes[lease.repository / DIAG4_PLAN_RELATIVE_PATH] = plan_bytes
    payload, identity, observed_plan = _validate_diag4_authority_bytes(
        lease.authority_bytes,
        repository=lease.repository,
        output_root=lease.output_root,
        reference_root=lease.reference_root,
        input_root=lease.input_root,
        interpreter=lease.interpreter,
        require_output_absent=False,
        locked_leaf_bytes=locked_leaf_bytes,
    )
    if (
        observed_plan != plan_bytes
        or payload != claim.payload
        or identity != claim.numerical_identity
        or identity.to_payload() != dict(claim.expected_numerical_identity)
        or _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
        != dict(claim.expected_frozen_numerical_entries)
    ):
        raise ValueError("DIAG4 held authority identity differs")
    if consumption_started:
        validate_diag4_consumption_marker(claim)


def _assert_diag4_pending_binding(
    descriptor: int,
    pending_name: str,
    output_descriptor: int,
) -> None:
    locked = os.fstat(descriptor)
    bound = os.stat(
        pending_name,
        dir_fd=output_descriptor,
        follow_symlinks=False,
    )
    if not stat.S_ISREG(locked.st_mode) or (
        locked.st_dev,
        locked.st_ino,
    ) != (bound.st_dev, bound.st_ino):
        raise ValueError("DIAG4 pending marker inode is not bound")


def _assert_diag4_published_marker_binding(
    descriptor: int,
    marker_name: str,
    marker_bytes: bytes,
    output_descriptor: int,
) -> None:
    locked = os.fstat(descriptor)
    published = os.stat(
        marker_name,
        dir_fd=output_descriptor,
        follow_symlinks=False,
    )
    if (
        not stat.S_ISREG(published.st_mode)
        or (locked.st_dev, locked.st_ino) != (published.st_dev, published.st_ino)
        or _diag4_descriptor_bytes(descriptor) != marker_bytes
    ):
        raise ValueError("DIAG4 published marker inode or bytes differ")


def _unlink_diag4_pending_marker(
    descriptor: int,
    pending_name: str,
    output_descriptor: int,
) -> None:
    _assert_diag4_pending_binding(descriptor, pending_name, output_descriptor)
    os.unlink(pending_name, dir_fd=output_descriptor)


def consume_diag4_successor_authority(
    claim: Diag4SuccessorAuthorityClaim,
) -> Diag4ConsumedAuthority:
    """Atomically and durably consume the held authority before first preflight."""

    lease = _diag4_assert_active(claim)
    if lease.lifecycle_state is Diag4AuthorityLifecycle.CONSUMED:
        raise RuntimeError("DIAG4 successor authority is already consumed")
    if lease.lifecycle_state is Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN:
        raise RuntimeError("DIAG4 successor authority consumption is uncertain")
    if lease.lifecycle_state is not Diag4AuthorityLifecycle.STAGING_BOUND:
        raise RuntimeError("DIAG4 staging root must be bound before consumption")
    revalidate_diag4_successor_authority(claim, require_output_absent=True)
    marker_path = diag4_consumption_marker_path(lease.output_root)
    payload = _diag4_marker_payload(claim)
    marker_bytes = canonical_json_bytes(payload)
    output_directory = lease.output_root.parent.resolve(strict=True)
    output_descriptor = lease.directory_descriptors[output_directory]
    pending_name = f"{marker_path.name}.pending-{os.getpid()}"
    descriptor = os.open(
        pending_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
        dir_fd=output_descriptor,
    )
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    published = False
    try:
        offset = 0
        while offset < len(marker_bytes):
            offset += os.write(descriptor, marker_bytes[offset:])
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        _assert_diag4_pending_binding(descriptor, pending_name, output_descriptor)
        os.link(
            f"/proc/self/fd/{descriptor}",
            marker_path.name,
            dst_dir_fd=output_descriptor,
            follow_symlinks=True,
        )
        published = True
        lease.lifecycle_state = Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        _assert_diag4_published_marker_binding(
            descriptor,
            marker_path.name,
            marker_bytes,
            output_descriptor,
        )
        lease.consumption_marker_descriptor = descriptor
        consumed = Diag4ConsumedAuthority(
            path=marker_path,
            sha256=hashlib.sha256(marker_bytes).hexdigest(),
            payload=MappingProxyType(payload),
        )
        lease.consumed = consumed
        _unlink_diag4_pending_marker(descriptor, pending_name, output_descriptor)
        os.fsync(output_descriptor)
        lease.lifecycle_state = Diag4AuthorityLifecycle.CONSUMED
    except BaseException:
        marker_exists = os.path.lexists(marker_path)
        if published or marker_exists:
            lease.lifecycle_state = Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        try:
            _unlink_diag4_pending_marker(descriptor, pending_name, output_descriptor)
        except (FileNotFoundError, ValueError):
            if not published and not marker_exists:
                lease.lifecycle_state = Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        except BaseException:
            lease.lifecycle_state = Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
            raise
        raise
    finally:
        if lease.consumption_marker_descriptor != descriptor:
            os.close(descriptor)
    revalidate_diag4_successor_authority(claim, require_output_absent=False)
    return consumed


def finalize_diag4_prelaunch_failure(
    claim: Diag4SuccessorAuthorityClaim,
) -> None:
    """Close an unconsumed zero-child failure after staging becomes final."""

    lease = _diag4_assert_active(claim)
    if lease.lifecycle_state is Diag4AuthorityLifecycle.PRELAUNCH_FAILURE_FINALIZED:
        raise RuntimeError("DIAG4 prelaunch failure is already finalized")
    if lease.lifecycle_state in {
        Diag4AuthorityLifecycle.CONSUMED,
        Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
    }:
        raise RuntimeError("DIAG4 consumed authority cannot finalize prelaunch failure")
    if lease.lifecycle_state is not Diag4AuthorityLifecycle.STAGING_BOUND:
        raise RuntimeError(
            "DIAG4 staging root must be bound before failure finalization"
        )
    _validate_diag4_prelaunch_failure_output_state(lease)
    lease.lifecycle_state = Diag4AuthorityLifecycle.PRELAUNCH_FAILURE_FINALIZED
    revalidate_diag4_successor_authority(claim, require_output_absent=True)


def _diag4_discovery_leaf_paths(
    authority_bytes: bytes,
    *,
    repository: Path,
    reference_root: Path,
    consumed_root: Path,
    interpreter: Path,
) -> tuple[Path, ...]:
    payload = _mapping(
        load_canonical_json_bytes(authority_bytes),
        "DIAG4 discovery authority",
    )
    qualified = _diag4_qualified_files(payload["qualified_files"])
    frozen = _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
    execution_manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    execution_manifest = _mapping(
        load_canonical_json_bytes(execution_manifest_path.read_bytes()),
        "DIAG4 discovery execution-source manifest",
    )
    execution_entries = _mapping(
        execution_manifest["entries"],
        "DIAG4 discovery execution-source entries",
    )
    execution_paths = tuple(
        repository
        / _diag4_relative_path(relative, "DIAG4 discovery execution-source entry path")
        for relative in execution_entries
    )
    native_extension = Path(
        _string(payload["native_extension_path"], "DIAG4 native extension path")
    ).resolve(strict=True)
    consumed = _mapping(payload["consumed_diag3"], "DIAG4 discovery consumed DIAG3")
    raw_consumed_entries = consumed["entries"]
    if not isinstance(raw_consumed_entries, list):
        raise TypeError("DIAG4 discovery consumed entries must be an array")
    consumed_paths = tuple(
        consumed_root
        / _diag4_relative_path(
            _mapping(entry, "DIAG4 discovery consumed entry")["relative_path"],
            "DIAG4 discovery consumed entry path",
        )
        for entry in raw_consumed_entries
    )
    native_manifest = reference_root / "artifact-manifest.json"
    native_manifest_bytes = native_manifest.read_bytes()
    native_paths = tuple(
        reference_root / relative
        for relative, _, _ in _diag4_native_manifest_entries(native_manifest_bytes)
    )
    cpu_manifest = DIAG4_CPU_QUALIFICATION_ROOT / "artifact-manifest.json"
    cpu_manifest_bytes = cpu_manifest.read_bytes()
    cpu_paths = tuple(
        DIAG4_CPU_QUALIFICATION_ROOT / relative
        for relative, _, _ in _diag4_cpu_manifest_entries(cpu_manifest_bytes)[1]
    )
    return tuple(
        sorted(
            {
                *(repository / relative for relative in qualified),
                *(repository / relative for relative in frozen),
                *execution_paths,
                native_extension,
                native_manifest,
                *native_paths,
                *consumed_paths,
                DIAG4_CPU20_RESULT_PATH,
                DIAG4_CPU20_HARNESS_PATH,
                cpu_manifest,
                *cpu_paths,
                interpreter,
            },
            key=str,
        )
    )


@contextmanager
def claim_diag4_successor_authority(
    authority_path: Path,
    *,
    repository_root: Path,
    output_root: Path,
    reference_root: Path,
    input_root: Path,
    interpreter: Path,
) -> Iterator[Diag4SuccessorAuthorityClaim]:
    """Hold exact DIAG4 root/file locks until the caller's final deep-load exits."""

    repository = repository_root.resolve(strict=True)
    output = output_root.absolute()
    reference = reference_root.resolve(strict=True)
    input_directory = input_root.resolve(strict=True)
    executable = interpreter.resolve(strict=True)
    expected_authority = repository / DIAG4_AUTHORITY_RELATIVE_PATH
    if Path(os.path.abspath(authority_path)) != expected_authority:
        raise ValueError("DIAG4 authority path differs")
    authority_directory = expected_authority.parent
    plan_path = repository / DIAG4_PLAN_RELATIVE_PATH
    consumed_root = DIAG4_CONSUMED_DIAG3_ROOT.resolve(strict=True)
    cpu_qualification_root = DIAG4_CPU_QUALIFICATION_ROOT.resolve(strict=True)
    discovery_authority_bytes = expected_authority.read_bytes()
    discovery_leaves = tuple(
        path
        for path in _diag4_discovery_leaf_paths(
            discovery_authority_bytes,
            repository=repository,
            reference_root=reference,
            consumed_root=consumed_root,
            interpreter=executable,
        )
        if path not in {expected_authority, plan_path}
    )
    leaf_directories = tuple(path.parent for path in discovery_leaves)
    with _diag4_claim_directories(
        (
            repository,
            authority_directory,
            plan_path.parent,
            output.parent,
            reference,
            input_directory,
            executable.parent,
            consumed_root,
            cpu_qualification_root,
            *leaf_directories,
        )
    ) as descriptors:
        authority_directory_descriptor = descriptors[authority_directory]
        authority_descriptor = os.open(
            expected_authority.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=authority_directory_descriptor,
        )
        plan_descriptor = os.open(
            plan_path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=descriptors[plan_path.parent],
        )
        locked_leaves: dict[Path, _Diag4LockedLeaf] = {}
        try:
            for descriptor in (authority_descriptor, plan_descriptor):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _assert_diag4_locked_file_binding(
                authority_directory_descriptor,
                expected_authority.name,
                authority_descriptor,
                "authority",
            )
            _assert_diag4_locked_file_binding(
                descriptors[plan_path.parent],
                plan_path.name,
                plan_descriptor,
                "completed plan",
            )
            authority_bytes = _diag4_descriptor_bytes(authority_descriptor)
            if authority_bytes != discovery_authority_bytes:
                raise ValueError("DIAG4 authority changed during claim discovery")
            for leaf_path in discovery_leaves:
                locked_leaves[leaf_path] = _open_diag4_locked_leaf(
                    leaf_path,
                    descriptors,
                    f"authority-bound leaf {leaf_path}",
                )
            plan_bytes = _diag4_descriptor_bytes(plan_descriptor)
            locked_leaf_bytes = {
                path: _assert_diag4_locked_leaf_binding(leaf, descriptors)
                for path, leaf in locked_leaves.items()
            }
            locked_leaf_bytes[expected_authority] = authority_bytes
            locked_leaf_bytes[plan_path] = plan_bytes
            payload, identity, plan_bytes = _validate_diag4_authority_bytes(
                authority_bytes,
                repository=repository,
                output_root=output,
                reference_root=reference,
                input_root=input_directory,
                interpreter=executable,
                require_output_absent=True,
                locked_leaf_bytes=locked_leaf_bytes,
            )
            if _diag4_descriptor_bytes(plan_descriptor) != plan_bytes:
                raise ValueError("DIAG4 locked completed plan bytes differ")
            claim_qualified = _diag4_qualified_files(payload["qualified_files"])
            claim_frozen = _diag4_frozen_numerical_entries(
                payload["frozen_numerical_entries"]
            )
            execution_manifest_sha256 = _diag4_sha256(
                payload["execution_source_manifest_sha256"],
                "DIAG4 execution-source manifest SHA",
            )
            execution_entries, execution_entry_sizes, _ = (
                _diag4_execution_source_entries(
                    locked_leaf_bytes[
                        repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
                    ],
                    repository=repository,
                    qualified=claim_qualified,
                    frozen=claim_frozen,
                    locked_leaf_bytes=locked_leaf_bytes,
                )
            )
            native_extension_path = Path(
                _string(payload["native_extension_path"], "DIAG4 native extension path")
            ).resolve(strict=True)
            native_extension_size_bytes = _integer(
                payload["native_extension_size_bytes"],
                "DIAG4 native extension size",
            )
            lease = _Diag4AuthorityLease(
                repository=repository,
                output_root=output,
                reference_root=reference,
                input_root=input_directory,
                interpreter=executable,
                authority_path=expected_authority,
                authority_bytes=authority_bytes,
                authority_descriptor=authority_descriptor,
                plan_descriptor=plan_descriptor,
                directory_descriptors=descriptors,
                locked_leaves=MappingProxyType(locked_leaves),
            )
            claim = Diag4SuccessorAuthorityClaim(
                payload=MappingProxyType(dict(payload)),
                authority_sha256=hashlib.sha256(authority_bytes).hexdigest(),
                plan_prefix_sha256=DIAG4_PLAN_SHA256,
                completed_plan_sha256=hashlib.sha256(plan_bytes).hexdigest(),
                numerical_identity=identity,
                expected_numerical_identity=MappingProxyType(identity.to_payload()),
                expected_frozen_numerical_entries=MappingProxyType(
                    _diag4_frozen_numerical_entries(payload["frozen_numerical_entries"])
                ),
                expected_execution_source_entries=MappingProxyType(
                    {
                        relative: (digest, execution_entry_sizes[relative])
                        for relative, digest in execution_entries.items()
                    }
                ),
                expected_execution_source_manifest_sha256=(execution_manifest_sha256),
                expected_native_extension_path=native_extension_path,
                expected_native_extension_sha256=payload["native_extension_sha256"],
                expected_native_extension_size_bytes=native_extension_size_bytes,
                _lease=lease,
            )
            try:
                yield claim
            finally:
                try:
                    revalidate_diag4_successor_authority(
                        claim,
                        require_output_absent=(
                            lease.lifecycle_state
                            not in {
                                Diag4AuthorityLifecycle.CONSUMED,
                                Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
                            }
                        ),
                    )
                finally:
                    lease.active = False
        finally:
            if "lease" in locals() and lease.staging_descriptor is not None:
                fcntl.flock(lease.staging_descriptor, fcntl.LOCK_UN)
                os.close(lease.staging_descriptor)
            if "lease" in locals() and lease.consumption_marker_descriptor is not None:
                os.close(lease.consumption_marker_descriptor)
            for leaf in reversed(tuple(locked_leaves.values())):
                fcntl.flock(leaf.descriptor, fcntl.LOCK_UN)
                os.close(leaf.descriptor)
            for descriptor in (plan_descriptor, authority_descriptor):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


def validate_diag4_successor_snapshot(
    snapshot: SnapshotPublication,
    claim: Diag4SuccessorAuthorityClaim,
) -> None:
    """Bind the sealed DIAG4 snapshot to every qualified and authority byte."""

    _diag4_assert_active(claim)
    expected: dict[str, tuple[str, int, str]] = {}
    for relative, (
        digest,
        size_bytes,
    ) in claim.expected_execution_source_entries.items():
        role = (
            "test"
            if relative.startswith("tests/")
            else "benchmark"
            if relative.startswith("benchmarks/")
            else "execution_source"
        )
        expected[relative] = (digest, size_bytes, role)
    manifest_leaf = claim._lease.locked_leaves[
        claim._lease.repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    ]
    expected[DIAG4_EXECUTION_SOURCE_MANIFEST_PATH] = (
        claim.expected_execution_source_manifest_sha256,
        manifest_leaf.initial_size_bytes,
        "execution_source_manifest",
    )
    expected[f"native/{claim.expected_native_extension_path.name}"] = (
        claim.expected_native_extension_sha256,
        claim.expected_native_extension_size_bytes,
        "native_extension",
    )
    observed = {
        entry.relative_path: (entry.sha256, entry.size_bytes, entry.role)
        for entry in snapshot.entries
    }
    if observed != expected:
        raise ValueError("DIAG4 GPU source snapshot differs from authority")


__all__ = (
    "AUTHORITY_RELATIVE_PATH",
    "DIAG3_SOURCE_DELTA_ALLOWLIST",
    "DIAG4_AUTHORITY_RELATIVE_PATH",
    "DIAG4_AUTHORITY_SCHEMA_VERSION",
    "DIAG4_BASE_POLICY_SHA256",
    "DIAG4_CONSUMED_DIAG3_ROOT",
    "DIAG4_CONSUMPTION_SCHEMA_VERSION",
    "DIAG4_CONTROLLING_CPU_COMMAND",
    "DIAG4_CPU20_COMMAND",
    "DIAG4_CPU20_DURATION_SECONDS",
    "DIAG4_CPU20_HARNESS_PATH",
    "DIAG4_CPU20_HARNESS_SHA256",
    "DIAG4_CPU20_RESULT_PATH",
    "DIAG4_CPU20_RESULT_SHA256",
    "DIAG4_CPU_QUALIFICATION_COMMAND",
    "DIAG4_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION",
    "DIAG4_CPU_QUALIFICATION_ROOT",
    "DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION",
    "DIAG4_EXECUTION_SOURCE_BROAD_ROOTS",
    "DIAG4_EXECUTION_SOURCE_ENTRY_COUNT",
    "DIAG4_EXECUTION_SOURCE_MANIFEST_PATH",
    "DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION",
    "DIAG4_FROZEN_NUMERICAL_PATHS",
    "DIAG4_GPU_UUID",
    "DIAG4_IDENTITY_FIELDS",
    "DIAG4_PLAN_RELATIVE_PATH",
    "DIAG4_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION",
    "DIAG4_PREQUALIFICATION_PLAN_SHA256",
    "DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH",
    "DIAG4_QUALIFICATION_SCHEMA_VERSION",
    "DIAG4_QUALIFIED_FILE_PATHS",
    "DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS",
    "DIAG4_REVIEW_ROLES",
    "DIAG4_STATIC_COMMANDS",
    "DIAG5_AUTHORITY_RELATIVE_PATH",
    "DIAG5_AUTHORITY_SCHEMA_VERSION",
    "DIAG5_BLANK_PLAN_SHA256",
    "DIAG5_BLANK_PLAN_SIZE_BYTES",
    "DIAG5_CONSUMPTION_SCHEMA_VERSION",
    "DIAG5_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION",
    "DIAG5_CPU_QUALIFICATION_ROOT",
    "DIAG5_CPU_QUALIFICATION_SCHEMA_VERSION",
    "DIAG5_EXECUTION_SOURCE_ENTRY_COUNT",
    "DIAG5_FAILED_DIAG4_FINAL_ROOT",
    "DIAG5_FAILED_DIAG4_PARTIAL_ROOT",
    "DIAG5_FROZEN_NUMERICAL_PATHS",
    "DIAG5_GPU_INTERPRETER",
    "DIAG5_GPU_OUTPUT_ROOT",
    "DIAG5_GPU_ROLLBACK_ROOT",
    "DIAG5_GPU_STAGING_ROOT",
    "DIAG5_INPUT_ROOT",
    "DIAG5_NATIVE_COPY_RELATIVE_PATH",
    "DIAG5_NATIVE_REFERENCE_ROOT",
    "DIAG5_PHYSICAL_FAILURE_PATH",
    "DIAG5_PHYSICAL_FAILURE_SCHEMA_VERSION",
    "DIAG5_PLAN_RELATIVE_PATH",
    "DIAG5_PLAN_SHA256",
    "DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH",
    "DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH",
    "DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION",
    "DIAG5_PREDECESSOR_POSTMORTEM_SHA256",
    "DIAG5_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION",
    "DIAG5_QUALIFICATION_SCHEMA_VERSION",
    "DIAG5_QUALIFIED_FILE_PATHS",
    "DIAG5_REVIEW_ROOT",
    "DIAG5_ROUTE",
    "GPU_UUID",
    "PLAN_SHA256",
    "QUALIFIED_FILE_PATHS",
    "ROUTE",
    "SCHEMA_VERSION",
    "Diag4AuthorityLifecycle",
    "Diag4ConsumedAuthority",
    "Diag4ConsumptionMarkerInvalidError",
    "Diag4NumericalIdentity",
    "Diag4SuccessorAuthorityClaim",
    "Diag5AuthorityLifecycle",
    "Diag5ConsumedAuthority",
    "Diag5ConsumptionMarkerInvalidError",
    "Diag5EvidenceNamespaceState",
    "Diag5FinalizerError",
    "Diag5FinalizerFailureCategory",
    "Diag5FinalizerSourceInput",
    "Diag5FinalizerSourceKind",
    "Diag5NativeExtensionBinding",
    "Diag5NativeExtensionClaim",
    "Diag5PhysicalCancellationCause",
    "Diag5PhysicalCancellationError",
    "Diag5PhysicalCancellationObservation",
    "Diag5PhysicalCancellationState",
    "Diag5PhysicalEvidenceReservation",
    "Diag5PhysicalPathState",
    "Diag5PredecessorFailureEvidence",
    "Diag5PublishedOutputKind",
    "Diag5RollbackCause",
    "Diag5RollbackObservation",
    "Diag5RollbackState",
    "Diag5SuccessorAuthorityClaim",
    "PreSourceFailure",
    "PublishedSnapshot",
    "SuccessorAuthorityClaim",
    "bind_diag4_staging_root",
    "bind_diag5_staging_root",
    "cancel_diag5_physical_failure_evidence",
    "claim_diag4_successor_authority",
    "claim_diag5_native_extension_binding",
    "claim_diag5_successor_authority",
    "claim_successor_authority",
    "consume_diag4_successor_authority",
    "consume_diag5_successor_authority",
    "derive_diag4_numerical_identity_sha256",
    "diag4_authority_lifecycle",
    "diag4_consumption_marker_path",
    "diag5_authority_lifecycle",
    "diag5_consumption_marker_path",
    "finalize_diag4_prelaunch_failure",
    "finalize_diag5_physical_evidence_success",
    "fsync_diag5_output_parent",
    "observe_diag5_native_extension_binding",
    "prepare_diag5_physical_failure_evidence",
    "publish_diag5_bound_staging",
    "publish_diag5_physical_failure_evidence",
    "revalidate_diag4_successor_authority",
    "revalidate_diag5_native_extension_binding",
    "revalidate_diag5_published_output",
    "revalidate_diag5_successor_authority",
    "rollback_diag5_bound_final",
    "validate_diag4_consumption_marker",
    "validate_diag4_successor_authority",
    "validate_diag4_successor_snapshot",
    "validate_diag5_consumption_marker",
    "validate_diag5_cross_runtime_native_bindings",
    "validate_diag5_predecessor_failure",
    "validate_diag5_predecessor_postmortem_artifact",
    "validate_diag5_sealed_native_copy",
    "validate_diag5_successor_authority",
    "validate_diag5_successor_snapshot",
    "validate_successor_authority",
    "validate_successor_snapshot",
)
