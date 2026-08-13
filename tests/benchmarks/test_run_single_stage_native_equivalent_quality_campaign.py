from __future__ import annotations

import fcntl
import hashlib
import inspect
import multiprocessing
import os
import shutil
import signal
import stat
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Never, cast

import benchmarks.run_single_stage_native_equivalent_quality_campaign as runner
import benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt as diagnostic_receipt
import numpy as np
import pytest
from benchmarks import process_gpu_monitor
from benchmarks import (
    single_stage_native_equivalent_quality_successor_authority as successor_authority,
)
from benchmarks.single_stage_fullspace_snapshot import (
    XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG,
    SnapshotEntry,
    SnapshotPublication,
    WorktreeIdentity,
)
from benchmarks.single_stage_native_equivalent_quality_receipt import SampleName


def _claim_test_repository(repository: Path, authority_bytes: bytes) -> Path:
    documentation = repository / "docs"
    documentation.mkdir(parents=True)
    authority_path = repository / successor_authority.AUTHORITY_RELATIVE_PATH
    authority_path.write_bytes(authority_bytes)
    plan_prefix = b"claim-test-plan\n"
    (repository / successor_authority.PLAN_RELATIVE_PATH).write_bytes(
        plan_prefix + b"## Qualification Record\n{}\n"
    )
    return authority_path


def _claim_process(
    authority_path: Path,
    repository: Path,
    output_root: Path,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    expected_authority_sha256: str,
) -> None:
    with successor_authority.claim_successor_authority(
        authority_path,
        repository_root=repository,
        output_root=output_root,
        reference_root=repository,
        input_root=repository,
        interpreter=authority_path,
    ) as claim:
        assert claim.authority_sha256 == expected_authority_sha256
        ready.set()
        assert release.wait(timeout=10.0)


def _claim_process_expecting_replacement(
    authority_path: Path,
    repository: Path,
    output_root: Path,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
    expected_error: str,
) -> None:
    with pytest.raises(ValueError, match=expected_error):  # noqa: SIM117
        with successor_authority.claim_successor_authority(
            authority_path,
            repository_root=repository,
            output_root=output_root,
            reference_root=repository,
            input_root=repository,
            interpreter=authority_path,
        ):
            ready.set()
            assert release.wait(timeout=10.0)


def _blocked_claim_process(
    authority_path: Path,
    repository: Path,
    output_root: Path,
) -> None:
    with pytest.raises(RuntimeError, match="already claimed"):  # noqa: SIM117
        with successor_authority.claim_successor_authority(
            authority_path,
            repository_root=repository,
            output_root=output_root,
            reference_root=repository,
            input_root=repository,
            interpreter=authority_path,
        ):
            raise AssertionError("concurrent claim entered its critical section")


def _claim_test_authority_validator(
    authority_bytes: bytes,
    **_arguments: object,
) -> dict[str, successor_authority.JsonValue]:
    return {"authority_sha256": hashlib.sha256(authority_bytes).hexdigest()}


def _diag4_authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    repository_name: str = "diag4-repository",
    output_root: Path | None = None,
    consumed_root: Path | None = None,
) -> SimpleNamespace:
    repository = tmp_path / repository_name
    documentation = repository / "docs"
    documentation.mkdir(parents=True)
    output = output_root or tmp_path / "campaigns" / "diag4-output"
    output.parent.mkdir(parents=True, exist_ok=True)
    cpu20_result = tmp_path / "cpu20-result.json"
    cpu20_result.write_bytes(b"historical cpu20 result\n")
    cpu20_harness = tmp_path / "cpu20-harness.py"
    cpu20_harness.write_bytes(b"historical cpu20 harness\n")
    monkeypatch.setattr(successor_authority, "DIAG4_CPU20_RESULT_PATH", cpu20_result)
    monkeypatch.setattr(
        successor_authority,
        "DIAG4_CPU20_RESULT_SHA256",
        hashlib.sha256(cpu20_result.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(successor_authority, "DIAG4_CPU20_HARNESS_PATH", cpu20_harness)
    monkeypatch.setattr(
        successor_authority,
        "DIAG4_CPU20_HARNESS_SHA256",
        hashlib.sha256(cpu20_harness.read_bytes()).hexdigest(),
    )
    identity = {
        name: hashlib.sha256(f"identity:{name}".encode()).hexdigest()
        for name in successor_authority.DIAG4_IDENTITY_FIELDS
        if name != "identity_sha256"
    }
    identity["base_neq_gntr1_policy_sha256"] = (
        successor_authority.DIAG4_BASE_POLICY_SHA256
    )
    identity["identity_sha256"] = (
        successor_authority.derive_diag4_numerical_identity_sha256(identity)
    )
    cpu_qualification_root = tmp_path / "cpu-qualification"
    cpu_files = {
        "arrays/accepted_optimizer_coordinates.npy": b"array\n",
        "endpoint-audit.json": b"{}\n",
        "history.json": b"{}\n",
        "policy.json": b"{}\n",
        "safeguard-telemetry.json": b"{}\n",
        "scientific-evidence.json": runner.canonical_json_bytes(
            {
                "backend": "cpu",
                "callback_count": 0,
                "configuration_fingerprint": "a" * 64,
                "input_fingerprint": "b" * 64,
                "native_reference_artifact_sha256": "c" * 64,
                "numerical_identity": identity,
                "output_root": str(cpu_qualification_root),
                "policy_sha256": successor_authority.DIAG4_BASE_POLICY_SHA256,
                "promotion_eligible": False,
                "qualification_passed": True,
                "route": diagnostic_receipt.DIAG4_NUMERICAL_ROUTE,
                "runtime": {"backend": "cpu", "x64_enabled": True},
                "schema_version": (
                    successor_authority.DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION
                ),
                "scientific_outcome": "QUALITY_HIT",
                "source_manifest_entries": [],
                "source_manifest_sha256": "d" * 64,
                "speed": "NOT_PRODUCED",
                "synchronized_solve_seconds": 1.0,
                "timings_monotonic_ns": {},
            }
        ),
        "terminal-numerical.json": b"{}\n",
    }
    cpu_manifest = cpu_qualification_root / "artifact-manifest.json"
    monkeypatch.setattr(
        successor_authority, "DIAG4_CPU_QUALIFICATION_ROOT", cpu_qualification_root
    )
    cpu_qualification_command = (
        "env JAX_PLATFORMS=cpu JAX_ENABLE_X64=true "
        "XLA_PYTHON_CLIENT_PREALLOCATE=false PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
        "PYTHONPATH=src .venv-qn-cpu/bin/python "
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py "
        f"--output-root {cpu_qualification_root}"
    )
    monkeypatch.setattr(
        successor_authority,
        "DIAG4_CPU_QUALIFICATION_COMMAND",
        cpu_qualification_command,
    )
    reference = repository / "reference"
    reference.mkdir()
    native_entry_path = reference / "arrays" / "native.bin"
    native_entry_path.parent.mkdir()
    native_entry_path.write_bytes(b"native reference entry\n")
    native_manifest = reference / "artifact-manifest.json"
    native_manifest.write_bytes(
        runner.canonical_json_bytes(
            {
                "schema_version": "native-reference-test-v1",
                "entries": [
                    {
                        "relative_path": "arrays/native.bin",
                        "sha256": hashlib.sha256(
                            native_entry_path.read_bytes()
                        ).hexdigest(),
                        "size_bytes": native_entry_path.stat().st_size,
                    }
                ],
            }
        )
    )
    input_root = repository / "input"
    input_root.mkdir()
    interpreter = repository / "python"
    interpreter.write_bytes(b"interpreter\n")
    native_extension = repository / "native" / "simsoptpp.so"
    native_extension.parent.mkdir()
    native_extension.write_bytes(b"loaded native extension\n")
    consumed = consumed_root or tmp_path / "consumed-diag3"
    consumed.mkdir(parents=True, exist_ok=True)
    for relative in sorted(successor_authority.DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS):
        path = consumed / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(f"consumed:{relative}\n".encode())
    consumed_entries = [
        {
            "relative_path": relative,
            "sha256": hashlib.sha256((consumed / relative).read_bytes()).hexdigest(),
            "size_bytes": (consumed / relative).stat().st_size,
        }
        for relative in sorted(successor_authority.DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS)
    ]
    consumed_payload = {
        "root": str(consumed.resolve()),
        "evidence_manifest_sha256": hashlib.sha256(
            runner.canonical_json_bytes(consumed_entries)
        ).hexdigest(),
        "entries": consumed_entries,
    }
    source_paths = (
        successor_authority.DIAG4_QUALIFIED_FILE_PATHS
        | successor_authority.DIAG4_FROZEN_NUMERICAL_PATHS
    ) - {successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH}
    for relative in sorted(source_paths):
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source:{relative}\n".encode())
    ignored_python = repository / "src" / "simsopt" / "_version.py"
    ignored_python.parent.mkdir(parents=True, exist_ok=True)
    ignored_python.write_bytes(b"VERSION = 'fixture'\n")
    frozen = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in sorted(successor_authority.DIAG4_FROZEN_NUMERICAL_PATHS)
    }
    frozen_sha256 = hashlib.sha256(runner.canonical_json_bytes(frozen)).hexdigest()
    prefix = b"# Frozen DIAG4 plan\n\n"
    plan_prefix_sha256 = hashlib.sha256(prefix).hexdigest()
    prequalification_plan = prefix + b"## Qualification Record\n"
    prequalification_plan_sha256 = hashlib.sha256(prequalification_plan).hexdigest()
    monkeypatch.setattr(successor_authority, "DIAG4_PLAN_SHA256", plan_prefix_sha256)
    monkeypatch.setattr(
        successor_authority,
        "DIAG4_PREQUALIFICATION_PLAN_SHA256",
        prequalification_plan_sha256,
    )
    broad_python_paths = {
        path.relative_to(repository).as_posix()
        for root_name in successor_authority.DIAG4_EXECUTION_SOURCE_BROAD_ROOTS
        for path in (repository / root_name).rglob("*.py")
        if path.is_file() and not path.is_symlink()
    }
    execution_paths = (
        broad_python_paths
        | (
            set(successor_authority.DIAG4_QUALIFIED_FILE_PATHS)
            - {successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH}
        )
        | set(successor_authority.DIAG4_FROZEN_NUMERICAL_PATHS)
    )
    execution_entry_payload = {
        relative: {
            "sha256": hashlib.sha256((repository / relative).read_bytes()).hexdigest(),
            "size_bytes": (repository / relative).stat().st_size,
        }
        for relative in sorted(execution_paths)
    }
    execution_entries_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(execution_entry_payload)
    ).hexdigest()
    execution_manifest_path = (
        repository / successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    )
    execution_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    execution_manifest_path.write_bytes(
        runner.canonical_json_bytes(
            {
                "schema_version": (
                    successor_authority.DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION
                ),
                "entries": execution_entry_payload,
                "entries_sha256": execution_entries_sha256,
            }
        )
    )
    monkeypatch.setattr(
        successor_authority,
        "DIAG4_EXECUTION_SOURCE_ENTRY_COUNT",
        len(execution_entry_payload),
    )
    execution_manifest_sha256 = hashlib.sha256(
        execution_manifest_path.read_bytes()
    ).hexdigest()
    qualified = {
        relative: hashlib.sha256((repository / relative).read_bytes()).hexdigest()
        for relative in sorted(successor_authority.DIAG4_QUALIFIED_FILE_PATHS)
    }
    qualified_sha256 = hashlib.sha256(
        runner.canonical_json_bytes(qualified)
    ).hexdigest()
    expected_source_entries = {
        **{
            relative: str(entry["sha256"])
            for relative, entry in execution_entry_payload.items()
        },
        successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH: (
            execution_manifest_sha256
        ),
    }
    snapshot_entries: list[dict[str, successor_authority.JsonValue]] = []
    snapshot_files: dict[str, bytes] = {}
    for relative, digest in sorted(expected_source_entries.items()):
        payload = (repository / relative).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        role = (
            "execution_source_manifest"
            if relative == successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
            else "test"
            if relative.startswith("tests/")
            else "benchmark"
            if relative.startswith("benchmarks/")
            else "configuration"
            if relative.startswith("docs/")
            else "execution_source"
        )
        snapshot_entries.append(
            {
                "relative_path": relative,
                "role": role,
                "sha256": digest,
                "size_bytes": len(payload),
            }
        )
        snapshot_files[f"source-snapshot/{relative}"] = payload
    snapshot_entries.append(
        {
            "relative_path": (
                successor_authority.DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH
            ),
            "role": "prequalification_plan",
            "sha256": prequalification_plan_sha256,
            "size_bytes": len(prequalification_plan),
        }
    )
    snapshot_files[
        "source-snapshot/"
        + successor_authority.DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH
    ] = prequalification_plan
    native_snapshot_payload = native_extension.read_bytes()
    snapshot_entries.append(
        {
            "relative_path": "native/simsoptpp.so",
            "role": "native_extension",
            "sha256": hashlib.sha256(native_snapshot_payload).hexdigest(),
            "size_bytes": len(native_snapshot_payload),
        }
    )
    snapshot_files["source-snapshot/native/simsoptpp.so"] = native_snapshot_payload
    snapshot_entries.sort(key=lambda entry: str(entry["relative_path"]))
    snapshot_manifest_bytes = runner.canonical_json_bytes(
        {
            "entries": snapshot_entries,
            "schema_version": "single-stage-fullspace-source-manifest-v1",
            "worktree": {
                "git_head": "a" * 40,
                "repo_root": str(tmp_path / "source-repository"),
                "tracked_diff_sha256": "b" * 64,
                "untracked_bytes_manifest_sha256": "c" * 64,
            },
        }
    )
    snapshot_files["source-snapshot/source-manifest.json"] = snapshot_manifest_bytes
    scientific_payload = runner.load_canonical_json_bytes(
        cpu_files["scientific-evidence.json"]
    )
    assert isinstance(scientific_payload, dict)
    scientific_payload["source_manifest_entries"] = snapshot_entries
    scientific_payload["source_manifest_sha256"] = hashlib.sha256(
        snapshot_manifest_bytes
    ).hexdigest()
    scientific_payload["execution_source_manifest_sha256"] = execution_manifest_sha256
    scientific_payload["execution_source_entries_sha256"] = execution_entries_sha256
    scientific_payload["prequalification_plan_control"] = {
        "schema_version": (
            successor_authority.DIAG4_PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION
        ),
        "snapshot_relative_path": (
            successor_authority.DIAG4_PREQUALIFICATION_PLAN_SNAPSHOT_PATH
        ),
        "source_relative_path": successor_authority.DIAG4_PLAN_RELATIVE_PATH,
        "sha256": prequalification_plan_sha256,
        "size_bytes": len(prequalification_plan),
        "plan_prefix_sha256": plan_prefix_sha256,
    }
    scientific_payload["native_extension_path"] = str(native_extension.resolve())
    scientific_payload["native_extension_sha256"] = hashlib.sha256(
        native_extension.read_bytes()
    ).hexdigest()
    scientific_payload["native_extension_size_bytes"] = native_extension.stat().st_size
    runtime_payload = scientific_payload["runtime"]
    assert isinstance(runtime_payload, dict)
    runtime_payload.update(
        {
            "native_extension_path": str(native_extension.resolve()),
            "native_extension_sha256": scientific_payload["native_extension_sha256"],
            "native_extension_size_bytes": native_extension.stat().st_size,
        }
    )
    cpu_files["scientific-evidence.json"] = runner.canonical_json_bytes(
        scientific_payload
    )
    evidence_index: dict[str, successor_authority.JsonValue] = {}
    for name, relative in {
        "endpoint_audit": "endpoint-audit.json",
        "history": "history.json",
        "policy": "policy.json",
        "safeguard_telemetry": "safeguard-telemetry.json",
        "terminal_numerical": "terminal-numerical.json",
    }.items():
        payload = cpu_files[relative]
        evidence_index[name] = {
            "relative_path": relative,
            "schema_version": f"test-{name}-v1",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
    cpu_files["evidence-index.json"] = runner.canonical_json_bytes(evidence_index)
    cpu_files.update(snapshot_files)
    if not cpu_manifest.exists():
        for relative, payload in cpu_files.items():
            path = cpu_qualification_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            path.chmod(0o444)
        artifact_directories = sorted(
            {
                path.relative_to(cpu_qualification_root).as_posix()
                for path in cpu_qualification_root.rglob("*")
                if path.is_dir()
            }
        )
        cpu_manifest_payload = {
            "directories": [
                {"mode": "0555", "relative_path": relative}
                for relative in artifact_directories
            ],
            "files": [
                {
                    "mode": "0444",
                    "relative_path": relative,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "size_bytes": len(payload),
                }
                for relative, payload in sorted(cpu_files.items())
            ],
            "execution_source_manifest_sha256": execution_manifest_sha256,
            "execution_source_entries_sha256": execution_entries_sha256,
            "schema_version": (
                successor_authority.DIAG4_CPU_QUALIFICATION_MANIFEST_SCHEMA_VERSION
            ),
        }
        cpu_manifest.write_bytes(runner.canonical_json_bytes(cpu_manifest_payload))
        cpu_manifest.chmod(0o444)
        for relative in reversed(artifact_directories):
            (cpu_qualification_root / relative).chmod(0o555)
        cpu_qualification_root.chmod(0o555)
    monkeypatch.setattr(
        successor_authority,
        "validate_native_equivalent_scientific_evidence",
        lambda **_kwargs: SimpleNamespace(
            outcome=diagnostic_receipt.ScientificOutcome.QUALITY_HIT
        ),
    )
    historical_cpu20 = {
        "accepted_steps": 20,
        "attempts": 20,
        "command": successor_authority.DIAG4_CPU20_COMMAND,
        "duration_seconds": successor_authority.DIAG4_CPU20_DURATION_SECONDS,
        "exit_code": 0,
        "git_head": "52dea17ddf3012cf923fc92da78c0d73a17f4625",
        "harness_path": str(cpu20_harness),
        "harness_sha256": successor_authority.DIAG4_CPU20_HARNESS_SHA256,
        "one_shot_no_retry": True,
        "promotion_eligible": False,
        "result_path": str(cpu20_result),
        "result_sha256": successor_authority.DIAG4_CPU20_RESULT_SHA256,
        "run_count": 1,
        "use": "ROUTE_SELECTION_ONLY",
    }
    decisive_cpu_qualification = {
        "artifact_manifest_sha256": hashlib.sha256(
            cpu_manifest.read_bytes()
        ).hexdigest(),
        "command": cpu_qualification_command,
        "duration_seconds": 2.5,
        "exit_code": 0,
        "qualification_passed": True,
        "root": str(cpu_qualification_root),
        "run_count": 1,
        "schema_version": (successor_authority.DIAG4_CPU_QUALIFICATION_SCHEMA_VERSION),
        "scientific_evidence_sha256": hashlib.sha256(
            cpu_files["scientific-evidence.json"]
        ).hexdigest(),
        "scientific_outcome": "QUALITY_HIT",
    }
    qualified_relative = min(
        successor_authority.DIAG4_QUALIFIED_FILE_PATHS
        - successor_authority.DIAG4_FROZEN_NUMERICAL_PATHS
    )
    frozen_relative = min(successor_authority.DIAG4_FROZEN_NUMERICAL_PATHS)
    qualified_path = repository / qualified_relative
    frozen_path = repository / frozen_relative
    monkeypatch.setattr(successor_authority, "DIAG4_CONSUMED_DIAG3_ROOT", consumed)
    qualification = {
        "schema_version": successor_authority.DIAG4_QUALIFICATION_SCHEMA_VERSION,
        "plan_prefix_sha256": plan_prefix_sha256,
        "output_root": str(output.absolute()),
        "controlling_cpu": {
            "command": successor_authority.DIAG4_CONTROLLING_CPU_COMMAND,
            "exit_code": 0,
            "passed": 7,
            "duration_seconds": 1.25,
            "run_count": 1,
        },
        "static_checks": {
            "ruff_check": True,
            "ruff_format_check": True,
            "compileall": True,
            "git_diff_check": True,
        },
        "static_commands": {
            name: {
                "command": command,
                "duration_seconds": 0.5,
                "exit_code": 0,
                "passed": True,
                "run_count": 1,
            }
            for name, command in successor_authority.DIAG4_STATIC_COMMANDS.items()
        },
        "qualified_files": qualified,
        "qualified_files_sha256": qualified_sha256,
        "frozen_numerical_entries": frozen,
        "frozen_numerical_entries_sha256": frozen_sha256,
        "execution_source_manifest_sha256": execution_manifest_sha256,
        "execution_source_entries_sha256": execution_entries_sha256,
        "native_extension_path": str(native_extension.resolve()),
        "native_extension_sha256": hashlib.sha256(
            native_extension.read_bytes()
        ).hexdigest(),
        "native_extension_size_bytes": native_extension.stat().st_size,
        "historical_cpu20": historical_cpu20,
        "decisive_cpu_qualification": decisive_cpu_qualification,
        "native_reference_manifest_sha256": hashlib.sha256(
            native_manifest.read_bytes()
        ).hexdigest(),
        "consumed_diag3": consumed_payload,
        "numerical_identity": identity,
        "no_gpu_used_for_qualification": True,
        "independent_reviews": [
            {
                "reviewed_frozen_numerical_entries_sha256": frozen_sha256,
                "reviewed_qualified_files_sha256": qualified_sha256,
                "reviewed_execution_source_manifest_sha256": (
                    execution_manifest_sha256
                ),
                "reviewed_execution_source_entries_sha256": (execution_entries_sha256),
                "reviewer": f"reviewer-{index}",
                "role": role,
                "session": f"session-{index}",
                "verdict": "GO",
            }
            for index, role in enumerate(
                sorted(successor_authority.DIAG4_REVIEW_ROLES), start=1
            )
        ],
        "authorization": {
            "preflight_launches": 1,
            "maximum_cold_launches": 1,
            "warm_allowed": False,
            "retry_allowed": False,
        },
    }
    record_bytes = runner.canonical_json_bytes(qualification)
    plan_bytes = prefix + b"## Qualification Record\n" + record_bytes
    plan_path = repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH
    plan_path.write_bytes(plan_bytes)
    authority = {
        "schema_version": successor_authority.DIAG4_AUTHORITY_SCHEMA_VERSION,
        "route": diagnostic_receipt.DIAG4_ROUTE,
        "numerical_route": diagnostic_receipt.DIAG4_NUMERICAL_ROUTE,
        "scientific_evidence_schema": diagnostic_receipt.DIAG4_SCHEMA_VERSION,
        "plan_prefix_sha256": plan_prefix_sha256,
        "completed_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "qualification_record_sha256": hashlib.sha256(record_bytes).hexdigest(),
        "qualified_files": qualified,
        "qualified_files_sha256": qualification["qualified_files_sha256"],
        "frozen_numerical_entries": frozen,
        "frozen_numerical_entries_sha256": frozen_sha256,
        "execution_source_manifest_sha256": execution_manifest_sha256,
        "execution_source_entries_sha256": execution_entries_sha256,
        "native_extension_path": str(native_extension.resolve()),
        "native_extension_sha256": hashlib.sha256(
            native_extension.read_bytes()
        ).hexdigest(),
        "native_extension_size_bytes": native_extension.stat().st_size,
        "historical_cpu20": historical_cpu20,
        "decisive_cpu_qualification": decisive_cpu_qualification,
        "native_reference_manifest_sha256": qualification[
            "native_reference_manifest_sha256"
        ],
        "consumed_diag3": consumed_payload,
        "numerical_identity": identity,
        "execution_policy": {
            "parent_platform": "cpu",
            "child_platform": "cuda",
            "jax_enable_x64": True,
            "compilation_cache_enabled": False,
            "child_preallocate": True,
            "command_buffer_enabled": False,
            "required_xla_flag": "--xla_gpu_enable_command_buffer=",
        },
        "launch": {
            "output_root": str(output.absolute()),
            "reference_root": str(reference.resolve()),
            "input_root": str(input_root.resolve()),
            "interpreter": str(interpreter.resolve()),
            "gpu_uuid": successor_authority.DIAG4_GPU_UUID,
            "preflight_launches": 1,
            "maximum_cold_launches": 1,
            "warm_allowed": False,
            "retry_allowed": False,
        },
    }
    authority_path = repository / successor_authority.DIAG4_AUTHORITY_RELATIVE_PATH
    authority_path.write_bytes(runner.canonical_json_bytes(authority))
    return SimpleNamespace(
        repository=repository,
        authority_path=authority_path,
        output_root=output,
        reference_root=reference,
        input_root=input_root,
        interpreter=interpreter,
        consumed_root=consumed,
        cpu20_harness=cpu20_harness,
        cpu20_result=cpu20_result,
        cpu_qualification_root=cpu_qualification_root,
        cpu_manifest=cpu_manifest,
        execution_manifest_path=execution_manifest_path,
        execution_entries=execution_entry_payload,
        execution_entries_sha256=execution_entries_sha256,
        native_extension=native_extension,
        qualified_path=qualified_path,
        frozen_path=frozen_path,
        native_manifest=native_manifest,
        native_entry_path=native_entry_path,
        identity=identity,
        qualified=qualified,
        qualified_relative=qualified_relative,
        frozen=frozen,
        frozen_relative=frozen_relative,
    )


def _diag4_claim_arguments(fixture: SimpleNamespace) -> dict[str, Path]:
    return {
        "repository_root": fixture.repository,
        "output_root": fixture.output_root,
        "reference_root": fixture.reference_root,
        "input_root": fixture.input_root,
        "interpreter": fixture.interpreter,
    }


def _bind_diag4_bootstrap_fixture(
    fixture: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner, "_DIAG4_BOOTSTRAP_CPU_ROOT", fixture.cpu_qualification_root
    )
    monkeypatch.setattr(
        runner,
        "_DIAG4_BOOTSTRAP_EXECUTION_ENTRY_COUNT",
        len(fixture.execution_entries),
    )
    monkeypatch.setattr(
        runner,
        "_DIAG4_BOOTSTRAP_PLAN_PREFIX_SHA256",
        successor_authority.DIAG4_PLAN_SHA256,
    )


def test_diag4_preimport_bootstrap_reexecs_exact_sealed_supervisor_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    _bind_diag4_bootstrap_fixture(fixture, monkeypatch)
    live_entry = tmp_path / "live" / runner._DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    live_entry.parent.mkdir(parents=True)
    live_entry.write_bytes(b"live launcher\n")
    calls: dict[str, object] = {}

    class BootstrapExecCalled(RuntimeError):
        pass

    def execve(
        executable: str,
        argv: Sequence[str],
        environment: Mapping[str, str],
    ) -> Never:
        calls["executable"] = executable
        calls["argv"] = tuple(argv)
        calls["environment"] = dict(environment)
        raise BootstrapExecCalled

    monkeypatch.setattr(runner.os, "chdir", lambda path: calls.setdefault("cwd", path))
    original_argv = (
        str(live_entry),
        "--output",
        str(fixture.output_root),
        "--reference",
        str(fixture.reference_root),
        "--input-root",
        str(fixture.input_root),
        "--diagnostic-successor-authority",
        str(fixture.authority_path),
    )

    with pytest.raises(BootstrapExecCalled):
        runner._diag4_preimport_bootstrap(
            argv=original_argv,
            current_entry=live_entry,
            environment={"UNCHANGED": "yes", "PYTHONPATH": "live"},
            execve=execve,
        )

    sealed_root = fixture.cpu_qualification_root / "source-snapshot"
    sealed_entry = sealed_root / runner._DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    assert calls["cwd"] == sealed_root
    assert calls["executable"] == str(fixture.interpreter.resolve())
    assert calls["argv"] == (
        str(fixture.interpreter.resolve()),
        "-B",
        str(sealed_entry.resolve()),
        *original_argv[1:],
    )
    child_environment = calls["environment"]
    assert isinstance(child_environment, dict)
    assert child_environment["PYTHONPATH"] == os.pathsep.join(
        (str(sealed_root / "src"), str(sealed_root))
    )
    assert child_environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert child_environment["UNCHANGED"] == "yes"

    monkeypatch.setattr(
        runner.os,
        "chdir",
        lambda _path: pytest.fail("sealed supervisor must not chdir again"),
    )
    runner._diag4_preimport_bootstrap(
        argv=(str(sealed_entry), *original_argv[1:]),
        current_entry=sealed_entry,
        environment={},
        execve=lambda *_arguments: pytest.fail("sealed supervisor must not reexec"),
    )


def test_diag4_preimport_bootstrap_revalidates_held_source_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    _bind_diag4_bootstrap_fixture(fixture, monkeypatch)
    sealed_entry = (
        fixture.cpu_qualification_root
        / "source-snapshot"
        / runner._DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    )
    original_revalidate = runner._diag4_bootstrap_revalidate_bindings

    def mutate_then_revalidate(
        bindings: Sequence[runner._Diag4BootstrapBinding],
    ) -> None:
        sealed_entry.chmod(0o644)
        sealed_entry.write_bytes(b"mutated during bootstrap\n")
        original_revalidate(bindings)

    monkeypatch.setattr(
        runner, "_diag4_bootstrap_revalidate_bindings", mutate_then_revalidate
    )

    with pytest.raises(RuntimeError, match="bootstrap binding drifted"):
        runner._diag4_preimport_bootstrap(
            argv=(
                str(tmp_path / "live-runner.py"),
                "--diagnostic-successor-authority",
                str(fixture.authority_path),
            ),
            current_entry=fixture.authority_path,
            environment={},
            execve=lambda *_arguments: pytest.fail("drifted source must not execute"),
        )


def test_diag4_preimport_rejects_before_any_repository_or_jax_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path / "fixture", monkeypatch)
    isolated_repository = tmp_path / "isolated-repository"
    isolated_entry = isolated_repository / runner._DIAG4_BOOTSTRAP_ENTRY_RELATIVE_PATH
    isolated_entry.parent.mkdir(parents=True)
    isolated_entry.write_bytes(Path(runner.__file__).read_bytes())
    isolated_authority = (
        isolated_repository / runner._DIAG4_BOOTSTRAP_AUTHORITY_RELATIVE_PATH
    )
    isolated_authority.parent.mkdir(parents=True)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    authority["route"] = "WRONG-ROUTE"
    isolated_authority.write_bytes(runner.canonical_json_bytes(authority))

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(isolated_entry),
            "--diagnostic-successor-authority",
            str(isolated_authority),
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "DIAG4 authority identity differs" in completed.stderr
    assert "No module named 'benchmarks'" not in completed.stderr
    assert "No module named 'jax'" not in completed.stderr


def test_prepare_diag4_snapshot_copies_exact_cpu_snapshot_without_live_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        runner, "DIAG4_CPU_QUALIFICATION_ROOT", fixture.cpu_qualification_root
    )

    def forbidden(*_arguments: object, **_keywords: object) -> Never:
        pytest.fail("DIAG4 GPU snapshot must not inspect or publish live sources")

    monkeypatch.setattr(runner, "capture_worktree_identity", forbidden)
    monkeypatch.setattr(runner, "_enumerated_source_roots", forbidden)
    monkeypatch.setattr(runner, "publish_immutable_snapshot", forbidden)
    staging_root = tmp_path / "diag4-staging"
    staging_root.mkdir()

    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path,
        **_diag4_claim_arguments(fixture),
    ) as claim:
        publication = runner._prepare_diag4_snapshot(
            staging_root,
            native_extension_path=fixture.native_extension.resolve(),
            successor_claim=claim,
        )
        successor_authority.validate_diag4_successor_snapshot(publication, claim)

    cpu = runner.load_snapshot(
        fixture.cpu_qualification_root / runner.SOURCE_SNAPSHOT_DIRECTORY,
        required_roles=runner.DIAG4_CPU_SNAPSHOT_ROLES,
    )
    gpu = runner.load_snapshot(
        publication.root,
        required_roles=runner.DIAG4_GPU_SNAPSHOT_ROLES,
    )
    assert gpu.worktree == cpu.worktree
    assert gpu.entries == tuple(
        entry for entry in cpu.entries if entry.role != "prequalification_plan"
    )
    assert all(entry.role != "configuration" for entry in gpu.entries)


def test_main_dispatches_exact_diag4_authority_to_typed_claim_and_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    claim = object()
    observed: dict[str, object] = {}

    class ClaimContext:
        def __enter__(self) -> object:
            return claim

        def __exit__(
            self,
            _exception_type: object,
            _exception: object,
            _traceback: object,
        ) -> None:
            return None

    def claim_diag4(
        authority_path: Path,
        **arguments: Path,
    ) -> ClaimContext:
        observed["authority_path"] = authority_path
        observed["claim_arguments"] = arguments
        return ClaimContext()

    def run_diag4(final_root: Path, **arguments: object) -> dict[str, object]:
        observed["final_root"] = final_root
        observed["run_arguments"] = arguments
        return {"schema_version": "test-summary-v1", "verdict": "INCOMPLETE"}

    monkeypatch.setattr(runner, "claim_diag4_successor_authority", claim_diag4)
    monkeypatch.setattr(runner, "run_diag4", run_diag4)
    monkeypatch.setattr(
        runner,
        "claim_successor_authority",
        lambda *_arguments, **_keywords: pytest.fail(
            "DIAG4 authority must not enter the legacy claim"
        ),
    )

    result = runner.main(
        (
            "--output",
            str(fixture.output_root),
            "--reference",
            str(fixture.reference_root),
            "--input-root",
            str(fixture.input_root),
            "--interpreter",
            str(fixture.interpreter),
            "--diagnostic-successor-authority",
            str(fixture.authority_path),
        )
    )

    assert result == 0
    assert observed["authority_path"] == fixture.authority_path.absolute()
    claim_arguments = observed["claim_arguments"]
    assert isinstance(claim_arguments, dict)
    assert claim_arguments["repository_root"] == fixture.repository
    run_arguments = observed["run_arguments"]
    assert isinstance(run_arguments, dict)
    assert run_arguments["successor_claim"] is claim
    assert run_arguments["repo_root"] == fixture.repository
    assert runner.load_canonical_json_bytes(capsys.readouterr().out.encode()) == {
        "schema_version": "test-summary-v1",
        "verdict": "INCOMPLETE",
    }


def _diag4_reseal_qualification(
    fixture: SimpleNamespace,
    mutate: Callable[[dict[str, successor_authority.JsonValue]], None],
) -> None:
    plan_path = fixture.repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH
    prefix, marker, record_bytes = plan_path.read_bytes().partition(
        b"## Qualification Record\n"
    )
    assert marker
    record = runner.load_canonical_json_bytes(record_bytes)
    assert isinstance(record, dict)
    mutate(record)
    changed_record_bytes = runner.canonical_json_bytes(record)
    changed_plan_bytes = prefix + marker + changed_record_bytes
    plan_path.write_bytes(changed_plan_bytes)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    authority["completed_plan_sha256"] = hashlib.sha256(changed_plan_bytes).hexdigest()
    authority["qualification_record_sha256"] = hashlib.sha256(
        changed_record_bytes
    ).hexdigest()
    fixture.authority_path.write_bytes(runner.canonical_json_bytes(authority))


def _diag4_claim_process(
    fixture: SimpleNamespace,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ):
        ready.set()
        assert release.wait(timeout=10.0)


def _diag4_blocked_claim_process(fixture: SimpleNamespace) -> None:
    with pytest.raises(RuntimeError, match="already claimed"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            raise AssertionError("concurrent DIAG4 claim entered")


def _diag4_finalized_claim_process(
    fixture: SimpleNamespace,
    ready: multiprocessing.synchronize.Event,
    release: multiprocessing.synchronize.Event,
) -> None:
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-finalized"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        os.replace(staging, fixture.output_root)
        successor_authority.finalize_diag4_prelaunch_failure(claim)
        ready.set()
        assert release.wait(timeout=10.0)


def _diag4_replace_pending_process(
    pending_path: Path,
    displaced_path: Path,
    replace_now: multiprocessing.synchronize.Event,
    replaced: multiprocessing.synchronize.Event,
) -> None:
    assert replace_now.wait(timeout=10.0)
    os.replace(pending_path, displaced_path)
    pending_path.write_bytes(b"foreign pending bytes\n")
    replaced.set()


def _qualification_payload() -> dict[str, successor_authority.JsonValue]:
    return {
        "qualified_files": {
            relative: hashlib.sha256(relative.encode("utf-8")).hexdigest()
            for relative in successor_authority.QUALIFIED_FILE_PATHS
        },
        "frozen_numerical_entries": [
            {"relative_path": relative, "sha256": digest}
            for relative, digest in successor_authority.DIAG2_FROZEN_NUMERICAL_ENTRIES
        ],
    }


def _qualification_record(
    payload: dict[str, successor_authority.JsonValue],
    output_root: Path,
    authority_sha256: str,
) -> dict[str, successor_authority.JsonValue]:
    return {
        "schema_version": successor_authority.QUALIFICATION_SCHEMA_VERSION,
        "plan_sha256": successor_authority.PLAN_SHA256,
        "authority_sha256": authority_sha256,
        "output_root": str(output_root.absolute()),
        "controlling_cpu": {
            "command": successor_authority.CONTROLLING_CPU_COMMAND,
            "passed": successor_authority.CONTROLLING_CPU_PASSED,
            "duration_seconds": 1.25,
        },
        "static_checks": {
            "ruff_check": True,
            "ruff_format_check": True,
            "compileall": True,
            "git_diff_check": True,
        },
        "qualified_files": payload["qualified_files"],
        "frozen_numerical_entries": payload["frozen_numerical_entries"],
        "native_reference_manifest_sha256": (
            successor_authority.NATIVE_REFERENCE_MANIFEST_SHA256
        ),
        "no_gpu_used_for_qualification": True,
        "independent_reviews": [
            {"reviewer": "implementation", "session": "session-a", "verdict": "GO"},
            {"reviewer": "atomicity", "session": "session-b", "verdict": "GO"},
        ],
        "authorization": {
            "preflight_launches": 1,
            "maximum_cold_launches": 1,
            "warm_allowed": False,
            "retry_allowed": False,
        },
    }


def _diag3_authority_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    repository = tmp_path / "diag3-repository"
    output_root = tmp_path / "campaigns" / "diag3-output"
    output_root.parent.mkdir(parents=True)
    qualified: dict[str, str] = {}
    for relative in successor_authority.QUALIFIED_FILE_PATHS:
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = f"isolated DIAG3 qualified bytes: {relative}\n".encode()
        path.write_bytes(data)
        qualified[relative] = hashlib.sha256(data).hexdigest()

    frozen_relative = "src/isolated_diag3_frozen.py"
    frozen_path = repository / frozen_relative
    frozen_path.parent.mkdir(parents=True)
    frozen_data = b"isolated DIAG3 frozen numerical bytes\n"
    frozen_path.write_bytes(frozen_data)
    frozen = ((frozen_relative, hashlib.sha256(frozen_data).hexdigest()),)
    monkeypatch.setattr(
        successor_authority,
        "DIAG2_FROZEN_NUMERICAL_ENTRIES",
        frozen,
    )

    reference_root = tmp_path / "native-reference"
    reference_root.mkdir()
    reference_manifest = reference_root / "artifact-manifest.json"
    reference_manifest.write_bytes(b"isolated native-reference manifest\n")
    monkeypatch.setattr(
        successor_authority,
        "NATIVE_REFERENCE_MANIFEST_SHA256",
        hashlib.sha256(reference_manifest.read_bytes()).hexdigest(),
    )
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_bytes(b"isolated interpreter\n")

    consumed_root = tmp_path / "consumed-r1"
    consumed_root.mkdir()
    consumed_hashes: dict[str, str] = {}
    for relative in ("diagnostic.json", "artifact-manifest.json"):
        path = consumed_root / relative
        data = f"isolated consumed R1 bytes: {relative}\n".encode()
        path.write_bytes(data)
        consumed_hashes[relative] = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(successor_authority, "R1_ROOT", consumed_root)
    monkeypatch.setattr(successor_authority, "R1_ARTIFACT_SHA256", consumed_hashes)
    monkeypatch.setattr(
        successor_authority,
        "load_and_validate_diag2_artifact",
        lambda _root: SimpleNamespace(
            verdict="DIAGNOSTIC_INCOMPLETE",
            next_route="NOT_PRODUCED",
            failure=SimpleNamespace(
                stage=SimpleNamespace(value="COLD_CRASH"),
                reason=SimpleNamespace(value="CHILD_EXIT_NONZERO"),
            ),
        ),
    )

    plan_prefix = b"isolated DIAG3 plan prefix\n"
    monkeypatch.setattr(
        successor_authority,
        "PLAN_SHA256",
        hashlib.sha256(plan_prefix).hexdigest(),
    )
    payload: dict[str, successor_authority.JsonValue] = {
        "schema_version": successor_authority.SCHEMA_VERSION,
        "route": successor_authority.ROUTE,
        "plan_sha256": successor_authority.PLAN_SHA256,
        "scientific_evidence_schema": successor_authority.DIAG2_SCHEMA_VERSION,
        "execution_policy": {
            "parent_platform": "cpu",
            "child_platform": "cuda",
            "jax_enable_x64": True,
            "compilation_cache_enabled": False,
            "child_preallocate": True,
            "command_buffer_enabled": False,
            "required_xla_flag": "--xla_gpu_enable_command_buffer=",
        },
        "launch": {
            "output_root": str(output_root),
            "reference_root": str(reference_root),
            "input_root": str(input_root),
            "interpreter": str(interpreter),
            "gpu_uuid": successor_authority.GPU_UUID,
            "preflight_launches": 1,
            "maximum_cold_launches": 1,
            "warm_allowed": False,
            "retry_allowed": False,
        },
        "consumed_r1": {
            "root": str(consumed_root),
            "diagnostic_sha256": consumed_hashes["diagnostic.json"],
            "manifest_sha256": consumed_hashes["artifact-manifest.json"],
        },
        "native_reference_manifest_sha256": (
            successor_authority.NATIVE_REFERENCE_MANIFEST_SHA256
        ),
        "frozen_numerical_entries": [
            {"relative_path": relative, "sha256": digest} for relative, digest in frozen
        ],
        "qualified_files": qualified,
    }
    authority_path = repository / successor_authority.AUTHORITY_RELATIVE_PATH
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_bytes = runner.canonical_json_bytes(payload)
    authority_path.write_bytes(authority_bytes)
    authority_sha256 = hashlib.sha256(authority_bytes).hexdigest()
    record = _qualification_record(payload, output_root, authority_sha256)
    plan_path = repository / successor_authority.PLAN_RELATIVE_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(
        plan_prefix + b"## Qualification Record\n" + runner.canonical_json_bytes(record)
    )
    arguments = {
        "repository_root": repository,
        "output_root": output_root,
        "reference_root": reference_root,
        "input_root": input_root,
        "interpreter": interpreter,
    }
    return SimpleNamespace(
        repository=repository,
        authority_path=authority_path,
        authority=payload,
        output_root=output_root,
        arguments=arguments,
    )


def _validate_qualification_test_record(
    record: dict[str, successor_authority.JsonValue],
    payload: dict[str, successor_authority.JsonValue],
    output_root: Path,
    authority_sha256: str,
) -> None:
    successor_authority._validate_qualification_record(
        b"## Qualification Record\n" + runner.canonical_json_bytes(record),
        payload,
        authority_sha256=authority_sha256,
        output_root=output_root,
    )


def _outcome(
    sample: SampleName,
    *,
    complete: bool = True,
    quality: bool = True,
) -> runner.SupervisedSample:
    return runner.SupervisedSample(
        sample=sample,
        terminal_status=(
            runner.ChildTerminalStatus.COMPLETE
            if complete
            else runner.ChildTerminalStatus.TIMEOUT
        ),
        child_pid=100 + list(runner.SAMPLE_ORDER).index(sample),
        child_start_time_ticks=1_000 + list(runner.SAMPLE_ORDER).index(sample),
        process_seconds=1.0,
        producer={"native_equivalent_quality": quality},
        memory={"sample_count": 1},
        failure_reasons=(),
    )


def _publish_fake_policy_authority(
    campaign_root: Path,
) -> runner.ArtifactRef:
    path = campaign_root / "policy-authority.json"
    runner._publish_canonical_json(
        path,
        {"schema_version": f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy"},
    )
    return runner._artifact_ref(
        path,
        campaign_root,
        f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy",
    )


def test_schedule_runs_only_cold_when_cold_audit_does_not_pass() -> None:
    called: list[SampleName] = []

    def execute(sample: SampleName) -> runner.SupervisedSample:
        called.append(sample)
        return _outcome(sample, quality=False)

    outcomes = runner.run_sample_schedule(execute, lambda _cold: False)

    assert called == [SampleName.COLD]
    assert tuple(item.sample for item in outcomes) == (SampleName.COLD,)


def test_schedule_runs_all_unreplaced_warms_after_valid_cold() -> None:
    called: list[SampleName] = []

    def execute(sample: SampleName) -> runner.SupervisedSample:
        called.append(sample)
        # A failed first warm must not suppress warm-2 or warm-3.
        return _outcome(sample, complete=sample is not SampleName.WARM_1)

    outcomes = runner.run_sample_schedule(execute, lambda _cold: True)

    assert called == list(runner.SAMPLE_ORDER)
    assert tuple(item.sample for item in outcomes) == runner.SAMPLE_ORDER
    assert (
        len({(item.child_pid, item.child_start_time_ticks) for item in outcomes}) == 4
    )


def test_schedule_never_trusts_worker_quality_summary() -> None:
    called: list[SampleName] = []

    def execute(sample: SampleName) -> runner.SupervisedSample:
        called.append(sample)
        return _outcome(sample, quality=True)

    outcomes = runner.run_sample_schedule(execute, lambda _cold: False)

    assert called == [SampleName.COLD]
    assert len(outcomes) == 1


def test_schedule_uses_recomputed_cold_receipt_not_worker_summary() -> None:
    called: list[SampleName] = []

    def execute(sample: SampleName) -> runner.SupervisedSample:
        called.append(sample)
        return _outcome(sample, quality=False)

    outcomes = runner.run_sample_schedule(execute, lambda _cold: True)

    assert called == list(runner.SAMPLE_ORDER)
    assert len(outcomes) == 4


def test_diagnostic_schedule_is_exactly_preflight_then_cold() -> None:
    called: list[str] = []

    outcomes = runner.run_diagnostic_schedule(
        lambda: called.append("preflight") or _outcome(SampleName.COLD),
        lambda _preflight: True,
        lambda: called.append("cold") or _outcome(SampleName.COLD),
    )

    assert called == ["preflight", "cold"]
    assert len(outcomes) == 2


def test_diagnostic_schedule_stops_after_failed_preflight() -> None:
    called: list[str] = []

    outcomes = runner.run_diagnostic_schedule(
        lambda: called.append("preflight") or _outcome(SampleName.COLD),
        lambda _preflight: False,
        lambda: called.append("cold") or _outcome(SampleName.COLD),
    )

    assert called == ["preflight"]
    assert len(outcomes) == 1


@pytest.mark.parametrize(
    ("terminal_status", "producer", "expected_reason"),
    (
        (runner.ChildTerminalStatus.TIMEOUT, {}, "CHILD_TIMEOUT"),
        (runner.ChildTerminalStatus.CRASH, {}, "CHILD_EXIT_NONZERO"),
        (
            runner.ChildTerminalStatus.PROTOCOL_FAILURE,
            {},
            "PRODUCER_DECODE_FAILED",
        ),
        (
            runner.ChildTerminalStatus.MONITOR_FAILURE,
            {},
            "MONITOR_BINDING_FAILED",
        ),
    ),
)
def test_diag2_supervision_represents_missing_producer_as_absent(
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: runner.ChildTerminalStatus,
    producer: dict[str, object],
    expected_reason: str,
) -> None:
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        terminal_status,
        123,
        0 if terminal_status is runner.ChildTerminalStatus.MONITOR_FAILURE else 456,
        1.0,
        producer,
        (
            None
            if terminal_status is runner.ChildTerminalStatus.MONITOR_FAILURE
            else {"peak_memory_fraction": 0.1}
        ),
        (expected_reason,),
        stdout=b"",
        stderr=b"raw",
        memory_samples=(),
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "absent producer must not be parsed"
        ),
    )

    assert supervised.producer is None
    assert supervised.producer_absence_reason is not None
    assert supervised.producer_absence_reason.value == expected_reason
    if terminal_status is runner.ChildTerminalStatus.MONITOR_FAILURE:
        assert supervised.monitor_failure_kind is runner.MonitorFailureKind.BINDING


def test_diag2_malformed_producer_stdout_is_raw_only_and_never_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_stdout = b'{"execution_status":"SUCCESS","unexpected":true}\n'
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        runner.ChildTerminalStatus.PROTOCOL_FAILURE,
        123,
        456,
        1.0,
        {},
        {"peak_memory_fraction": 0.1},
        ("WORKER_PROTOCOL:ValueError:deadbeef",),
        stdout=raw_stdout,
        stderr=b"",
        memory_samples=(),
        process_diagnostics={"returncode": 0},
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.PREFLIGHT,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "malformed stdout must never reach the typed producer parser"
        ),
    )

    assert supervised.producer is None
    assert supervised.producer_absence_reason is (
        runner.AbsenceReason.PRODUCER_DECODE_FAILED
    )
    assert supervised.stdout == raw_stdout


def test_diag2_supervision_retains_schema_valid_compile_failure_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = {"execution_status": "COMPILE_OOM"}
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        runner.ChildTerminalStatus.COMPILE_FAILURE,
        123,
        456,
        1.0,
        producer,
        {"peak_memory_fraction": 0.1},
        ("CHILD_COMPILE_OOM",),
        stdout=b"{}",
        stderr=b"oom",
        memory_samples=(),
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda value, **_kwargs: value,
    )

    assert supervised.producer == producer
    assert supervised.producer_absence_reason is None


def test_diag3_supervision_retains_typed_trace_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = {"execution_status": "TRACE_NORMALIZATION_FAILED"}
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        runner.ChildTerminalStatus.COMPLETE,
        123,
        456,
        1.0,
        producer,
        {"peak_memory_fraction": 0.1},
        ("TRACE_NORMALIZATION_FAILED:" + "1" * 64,),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
        process_diagnostics={"returncode": 0},
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda value, **_kwargs: value,
    )

    assert supervised.producer == producer
    assert supervised.producer_absence_reason is None
    assert supervised.selected_failure_reason is (
        runner.FailureReasonCodeV2.TRACE_NORMALIZATION_FAILED
    )


def test_diag2_monitor_finalization_precedes_child_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        runner.ChildTerminalStatus.CRASH,
        123,
        456,
        1.0,
        {},
        None,
        ("CHILD_EXIT_NONZERO", "MONITOR_FINALIZATION_FAILED"),
        stdout=b"",
        stderr=b"crash",
        memory_samples=(),
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "crashed child has no producer"
        ),
    )

    assert supervised.producer is None
    assert supervised.selected_failure_reason is (
        runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
    )
    assert supervised.monitor_failure_kind is runner.MonitorFailureKind.FINALIZATION
    assert supervised.terminal_status is runner.ChildTerminalStatus.CRASH


def test_diag2_monitor_finalization_retains_valid_exit_zero_producer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    producer = {"execution_status": "COMPLETE"}
    outcome = runner.SupervisedSample(
        SampleName.COLD,
        runner.ChildTerminalStatus.MONITOR_FAILURE,
        123,
        456,
        1.0,
        producer,
        None,
        ("MONITOR_FINALIZATION_FAILED",),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
        process_diagnostics={"returncode": 0},
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda value, **_kwargs: value,
    )

    assert supervised.producer == producer
    assert supervised.producer_absence_reason is None
    assert supervised.terminal_status is runner.ChildTerminalStatus.COMPLETE
    assert supervised.monitor_failure_kind is runner.MonitorFailureKind.FINALIZATION
    assert supervised.selected_failure_reason is (
        runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
    )
    assert (
        0
        < supervised.process_started_monotonic_ns
        <= supervised.process_stopped_monotonic_ns
    )


def test_diag2_monitor_start_failure_kills_reaps_and_retains_binding_streams(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Child:
        pid = 123
        returncode = 0
        killed = False
        reaped = False

        def kill(self) -> None:
            self.killed = True

        def communicate(self) -> tuple[bytes, bytes]:
            assert self.killed
            self.reaped = True
            return b"partial stdout", b"partial stderr"

    class Monitor:
        def start(self) -> None:
            raise RuntimeError("monitor thread failed to start")

    child = Child()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(
        runner, "BoundProcessGpuMemoryMonitor", lambda **_kwargs: Monitor()
    )
    invocation = runner.SnapshotChildInvocation(("python", "child.py"), tmp_path, {})

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        invocation,
        mode=runner.DiagnosticChildMode.PREFLIGHT,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "binding failure cannot publish a producer"
        ),
    )
    (tmp_path / "preflight").mkdir()
    references = runner._publish_diag2_supervision(
        tmp_path,
        tmp_path / "preflight",
        supervised,
        producer_schema=runner.PREFLIGHT_SCHEMA_VERSION,
    )

    assert child.killed and child.reaped
    assert supervised.monitor_failure_kind is runner.MonitorFailureKind.BINDING
    assert supervised.child_start_time_ticks == 0
    assert supervised.stdout == b"partial stdout"
    assert supervised.stderr == b"partial stderr"
    assert references["terminal"] is not None
    assert references["process"] is not None
    assert references["producer"] is None
    assert references["memory"] is None
    assert references["memory_samples"] is None


def test_diag2_process_evidence_retains_parent_monotonic_interval(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    directory = root / "cold"
    directory.mkdir(parents=True)
    outcome = runner.DiagnosticSupervisedSampleV2(
        SampleName.COLD,
        True,
        runner.ChildTerminalStatus.COMPLETE,
        123,
        456,
        0.25,
        {"execution_status": "COMPLETE"},
        None,
        None,
        None,
        (),
        ("python", "child"),
        b"{}",
        b"",
        (),
        process_diagnostics={"returncode": 0},
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
    )

    references = runner._publish_diag2_supervision(
        root,
        directory,
        outcome,
        producer_schema="producer-v1",
    )

    assert references["process"] is not None
    process = runner.load_canonical_json_bytes(
        (directory / "process.json").read_bytes()
    )
    assert process["schema_version"] == runner.DIAG2_PROCESS_SCHEMA_VERSION
    assert process["process_started_monotonic_ns"] == 100
    assert process["process_stopped_monotonic_ns"] == 200
    assert process["monitor_failure_kind"] == "NONE"
    terminal = runner.load_canonical_json_bytes(
        (directory / "terminal.json").read_bytes()
    )
    assert terminal["monitor_failure_kind"] == "NONE"


def test_diag2_supervision_v2_refs_resolve_and_classify_actual_crash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    directory = root / "preflight"
    directory.mkdir(parents=True)
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.CRASH,
        child_pid=1470196,
        child_start_time_ticks=47646434,
        process_seconds=2.0,
        producer=None,
        producer_absence_reason=runner.AbsenceReason.CHILD_EXIT_NONZERO,
        selected_failure_reason=runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO,
        memory={"schema_version": runner.MEMORY_SCHEMA_VERSION},
        raw_failure_reasons=("CHILD_EXIT_1:" + "a" * 64,),
        observed_child_argv=("python", "child"),
        stdout=b"",
        stderr=b"traceback",
        memory_samples=(runner.RawGpuMemorySample(1, 526),),
        process_diagnostics={"returncode": 1},
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
    )

    published = runner._publish_diag2_supervision(
        root,
        directory,
        outcome,
        producer_schema=runner.PREFLIGHT_SCHEMA_VERSION,
    )
    refs = {name: None for name in runner.DIAG2_EVIDENCE_SLOT_NAMES}
    for suffix, reference in published.items():
        refs[f"preflight_{suffix}"] = reference

    terminal = published["terminal"]
    process = published["process"]
    assert terminal is not None
    assert process is not None
    assert terminal.schema_version == (runner.DIAG2_CHILD_TERMINAL_SCHEMA_VERSION)
    assert process.schema_version == runner.DIAG2_PROCESS_SCHEMA_VERSION
    assert (
        runner.classify_diag3_subordinate_child_outcome(
            root,
            artifact_refs=refs,
            mode="preflight",
        )
        is runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO
    )


def test_diag2_supervision_requires_parent_created_child_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "preflight"
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=False,
        terminal_status=runner.ChildTerminalStatus.CRASH,
        child_pid=0,
        child_start_time_ticks=0,
        process_seconds=0.0,
        producer=None,
        producer_absence_reason=runner.AbsenceReason.CHILD_LAUNCH_FAILED,
        selected_failure_reason=runner.FailureReasonCodeV2.CHILD_LAUNCH_FAILED,
        memory=None,
        raw_failure_reasons=("launch failed",),
        observed_child_argv=None,
        stdout=b"",
        stderr=b"",
        memory_samples=(),
    )

    with pytest.raises(ValueError, match="must exist before supervision"):
        runner._publish_diag2_supervision(
            tmp_path,
            directory,
            outcome,
            producer_schema=runner.PREFLIGHT_SCHEMA_VERSION,
        )

    assert not directory.exists()


@pytest.mark.parametrize("producer_present", (False, True))
def test_diag5_child_supervision_publishes_only_a_contiguous_failure_prefix(
    tmp_path: Path,
    producer_present: bool,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    directory = staging / "preflight"
    directory.mkdir(parents=True)
    runner._publish_bytes(directory / "runtime-evidence.json", b"runtime")
    runner._publish_bytes(directory / "policy.json", b"policy")
    publication = runner.Diag2Publication(
        staging,
        tmp_path / "campaign",
        "claim",
    )
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS
    }
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=(
            runner.ChildTerminalStatus.COMPLETE
            if producer_present
            else runner.ChildTerminalStatus.PROTOCOL_FAILURE
        ),
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=0.25,
        producer={"execution_status": "SUCCESS"} if producer_present else None,
        producer_absence_reason=(
            None if producer_present else runner.AbsenceReason.PRODUCER_DECODE_FAILED
        ),
        selected_failure_reason=(
            runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
            if producer_present
            else runner.FailureReasonCodeV2.PRODUCER_DECODE_FAILED
        ),
        memory=None,
        raw_failure_reasons=("failure",),
        observed_child_argv=("child",),
        stdout=b"stdout",
        stderr=b"stderr",
        memory_samples=(),
    )

    failure = runner._publish_diag5_child_supervision(
        publication,
        refs,
        directory,
        outcome,
        mode=runner.DiagnosticChildMode.PREFLIGHT,
    )

    assert (failure is None) is producer_present
    present = tuple(name for name, reference in refs.items() if reference is not None)
    assert present == (
        ("preflight_producer", "preflight_terminal", "preflight_process")
        if producer_present
        else ()
    )
    assert (directory / "runtime-evidence.json").exists()
    assert (directory / "policy.json").exists()
    assert directory.exists()


def test_diag5_child_supervision_failure_preserves_raw_subtree_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    directory = staging / "cold"
    directory.mkdir(parents=True)
    publication = runner.Diag2Publication(staging, tmp_path / "campaign", "claim")
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS
    }
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=0.25,
        producer={"execution_status": "COMPLETE"},
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory=None,
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"stdout",
        stderr=b"",
        memory_samples=(),
    )

    def fail_after_partial_write(*_args: object, **_kwargs: object) -> object:
        (directory / "terminal.json").write_bytes(b"partial")
        raise OSError("supervision publication")

    monkeypatch.setattr(runner, "_publish_diag2_supervision", fail_after_partial_write)

    failure = runner._publish_diag5_child_supervision(
        publication,
        refs,
        directory,
        outcome,
        mode=runner.DiagnosticChildMode.COLD,
    )

    assert failure is not None
    assert failure.stage is runner.FailureStageV5.COLD
    assert failure.reason is runner.FailureReasonCodeV5.COLD_PROTOCOL_INVALID
    assert all(reference is None for reference in refs.values())
    assert directory.is_dir()
    assert (directory / "terminal.json").read_bytes() == b"partial"


@pytest.mark.parametrize("mode", tuple(runner.DiagnosticChildMode))
@pytest.mark.parametrize(
    ("reason", "producer_present", "memory_present"),
    (
        (runner.FailureReasonCodeV2.CHILD_TIMEOUT, False, True),
        (runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED, False, False),
        (runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO, False, True),
        (runner.FailureReasonCodeV2.CHILD_COMPILE_FAILED, True, True),
        (runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID, False, True),
    ),
)
def test_diag5_launched_supervision_reasons_publish_exact_parent_closure(
    tmp_path: Path,
    mode: runner.DiagnosticChildMode,
    reason: runner.FailureReasonCodeV2,
    producer_present: bool,
    memory_present: bool,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    directory = staging / mode.value
    directory.mkdir(parents=True)
    publication = runner.Diag2Publication(staging, tmp_path / "campaign", "claim")
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS
    }
    raw_stdout = b'{"invalid":true}\n'
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.PROTOCOL_FAILURE,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=0.25,
        producer={"execution_status": "COMPILE_FAILURE"} if producer_present else None,
        producer_absence_reason=(
            None if producer_present else runner.AbsenceReason.PRODUCER_SCHEMA_INVALID
        ),
        selected_failure_reason=reason,
        memory={"peak_memory_fraction": 0.1} if memory_present else None,
        raw_failure_reasons=(reason.value,),
        observed_child_argv=("child",),
        stdout=raw_stdout,
        stderr=b"stderr",
        memory_samples=(runner.RawGpuMemorySample(1, 2),) if memory_present else (),
        process_diagnostics={"returncode": 1},
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
        monitor_failure_kind=(
            runner.MonitorFailureKind.FINALIZATION
            if reason is runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
            else runner.MonitorFailureKind.NONE
        ),
    )

    failure = runner._publish_diag5_child_supervision(
        publication,
        refs,
        directory,
        outcome,
        mode=mode,
    )

    assert failure is None
    prefix = mode.value
    assert tuple(name for name, ref in refs.items() if ref is not None) == (
        f"{prefix}_producer",
        f"{prefix}_terminal",
        f"{prefix}_process",
    )
    producer = runner.load_canonical_json_bytes(
        (directory / "producer.json").read_bytes()
    )
    if producer_present:
        assert producer == outcome.producer
        assert not (directory / "invalid-producer.bin").exists()
    else:
        assert producer["document_origin"] == "PARENT_SUPERVISOR"
        assert producer["execution_status"] == "SUPERVISION_FAILURE"
        runner.validate_diag5_supervisor_failure_producer_payload(
            producer,
            mode=prefix,
        )
        if reason is runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID:
            assert (directory / "invalid-producer.bin").read_bytes() == raw_stdout
        else:
            assert not (directory / "invalid-producer.bin").exists()
    assert (directory / "stdout.bin").read_bytes() == raw_stdout
    assert (directory / "stderr.bin").read_bytes() == b"stderr"


def test_diag5_cold_success_defers_auxiliary_slots_until_source_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    directory = staging / "cold"
    directory.mkdir(parents=True)
    publication = runner.Diag2Publication(staging, tmp_path / "campaign", "claim")
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS
    }
    schemas = {
        "producer": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
        "terminal": runner.DIAG5_CHILD_TERMINAL_SCHEMA_VERSION,
        "process": runner.DIAG5_PROCESS_SCHEMA_VERSION,
        "memory": runner.DIAG5_MEMORY_SCHEMA_VERSION,
        "memory_samples": runner.DIAG5_MEMORY_SAMPLES_SCHEMA_VERSION,
    }
    filenames = {
        "producer": "producer.json",
        "terminal": "terminal.json",
        "process": "process.json",
        "memory": "gpu-memory.json",
        "memory_samples": "gpu-memory-samples.json",
    }
    child_refs: dict[str, runner.ArtifactRef] = {}
    for suffix, filename in filenames.items():
        path = directory / filename
        path.write_bytes(b"{}\n")
        child_refs[suffix] = runner._artifact_ref(path, staging, schemas[suffix])
    (directory / "runtime-evidence.json").write_bytes(b"{}\n")
    (directory / "policy.json").write_bytes(
        runner.canonical_json_bytes(
            {"schema_version": "single-stage-native-equivalent-quality-policy-v1"}
        )
    )
    monkeypatch.setattr(
        runner,
        "_publish_diag2_supervision",
        lambda *_args, **_kwargs: child_refs,
    )
    monkeypatch.setattr(
        runner, "validate_diag5_policy_evidence_payload", lambda _value: object()
    )
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=0.25,
        producer={"execution_status": "COMPLETE"},
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory={"peak_memory_fraction": 0.1},
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"",
        stderr=b"",
        memory_samples=(runner.RawGpuMemorySample(1, 2),),
        process_diagnostics={"returncode": 0},
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
    )

    failure = runner._publish_diag5_child_supervision(
        publication,
        refs,
        directory,
        outcome,
        mode=runner.DiagnosticChildMode.COLD,
        defer_success_auxiliary_slots=True,
    )

    assert failure is None
    assert tuple(name for name, ref in refs.items() if ref is not None) == (
        "cold_producer",
        "cold_terminal",
        "cold_process",
    )
    refs.update(
        runner._diag5_child_success_auxiliary_references(
            publication,
            mode=runner.DiagnosticChildMode.COLD,
        )
    )
    assert (
        tuple(name for name, ref in refs.items() if ref is not None)
        == tuple(runner.DIAG5_EVIDENCE_SLOT_PATHS)[13:20]
    )


@pytest.mark.parametrize(
    ("kind", "status", "start_ticks"),
    (
        (
            runner.MonitorFailureKind.BINDING,
            runner.ChildTerminalStatus.MONITOR_FAILURE,
            0,
        ),
        (
            runner.MonitorFailureKind.FINALIZATION,
            runner.ChildTerminalStatus.COMPLETE,
            456,
        ),
    ),
)
def test_diag2_process_and_terminal_share_typed_monitor_failure_kind(
    tmp_path: Path,
    kind: runner.MonitorFailureKind,
    status: runner.ChildTerminalStatus,
    start_ticks: int,
) -> None:
    root = tmp_path / "artifact"
    directory = root / "preflight"
    directory.mkdir(parents=True)
    outcome = runner.DiagnosticSupervisedSampleV2(
        SampleName.COLD,
        True,
        status,
        123,
        start_ticks,
        0.25,
        None,
        (
            runner.AbsenceReason.MONITOR_BINDING_FAILED
            if kind is runner.MonitorFailureKind.BINDING
            else runner.AbsenceReason.MONITOR_FINALIZATION_FAILED
        ),
        (
            runner.FailureReasonCodeV2.MONITOR_BINDING_FAILED
            if kind is runner.MonitorFailureKind.BINDING
            else runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
        ),
        None,
        (),
        None if start_ticks == 0 else ("python", "child"),
        b"",
        b"",
        (),
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
        monitor_failure_kind=kind,
    )

    runner._publish_diag2_supervision(
        root,
        directory,
        outcome,
        producer_schema="producer-v1",
    )

    process = runner.load_canonical_json_bytes(
        (directory / "process.json").read_bytes()
    )
    terminal = runner.load_canonical_json_bytes(
        (directory / "terminal.json").read_bytes()
    )
    assert process["monitor_failure_kind"] == kind.value
    assert terminal["monitor_failure_kind"] == kind.value
    assert terminal["terminal_status"] == status.value


def test_diag2_launch_failure_returns_typed_absence_without_legacy_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "supervise_sample",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("launch failed")),
    )

    supervised = runner.supervise_diag2_sample(
        SampleName.COLD,
        runner.SnapshotChildInvocation(("python",), Path.cwd(), {}),
        mode=runner.DiagnosticChildMode.PREFLIGHT,
        gpu_uuid="GPU-test",
        physical_memory_bytes=1,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "launch failure has no producer"
        ),
    )

    assert supervised.launched is False
    assert supervised.producer is None
    assert (
        supervised.producer_absence_reason is runner.AbsenceReason.CHILD_LAUNCH_FAILED
    )
    assert supervised.selected_failure_reason is (
        runner.FailureReasonCodeV2.CHILD_LAUNCH_FAILED
    )


def test_diag2_parent_import_forces_cpu_before_jax_import() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(repository)!r}); "
        "import json, os; "
        "import benchmarks.run_single_stage_native_equivalent_quality_campaign; "
        "from benchmarks import process_gpu_monitor as gm; "
        "from unittest.mock import patch; "
        "import jax; payload={key: os.environ.get(key) for key in ("
        "'JAX_PLATFORMS','JAX_PLATFORM_NAME','JAX_COMPILATION_CACHE_DIR',"
        "'JAX_ENABLE_COMPILATION_CACHE','XLA_PYTHON_CLIENT_PREALLOCATE')}; "
        "pid=os.getpid(); sha='a'*64; "
        "fake=lambda argv,**kwargs: gm.SupervisorGpuQuery(argv,sha,True,False,0,"
        "b'GPU-target, 32768\\n' if argv==gm.SUPERVISOR_GPU_INVENTORY_QUERY "
        "else str(pid+1).encode()+b', GPU-target, 1\\n',b''); "
        "patched=patch.object(gm,'_run_supervisor_query',side_effect=fake); "
        "patched.start(); observation=gm.capture_supervisor_gpu_zero("
        "gpu_uuid='GPU-target',visible_device='GPU-target',supervisor_pid=pid,"
        "supervisor_start_ticks=1,query_executable_sha256=sha); patched.stop(); "
        "payload['backend']=jax.default_backend(); payload['parent_pid']=pid; "
        "payload['gpu_zero_gate']=observation.gate_passes; "
        "payload['matching_rows']=len(observation.matching_rows); "
        "print(json.dumps(payload))"
    )
    environment = dict(os.environ)
    environment.update(
        {
            "JAX_PLATFORMS": "cuda",
            "JAX_PLATFORM_NAME": "gpu",
            "JAX_COMPILATION_CACHE_DIR": "/tmp/forbidden-cache",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        }
    )

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            script,
            "--diagnostic-successor-authority=authority.json",
        ),
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = runner.json.loads(completed.stdout.splitlines()[-1])
    parent_pid = payload.pop("parent_pid")
    assert isinstance(parent_pid, int) and parent_pid > 0
    assert payload == {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": None,
        "JAX_COMPILATION_CACHE_DIR": None,
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "backend": "cpu",
        "gpu_zero_gate": True,
        "matching_rows": 0,
    }


def test_diag3_successor_authority_validates_fresh_one_shot_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag3_authority_fixture(tmp_path, monkeypatch)
    validated = successor_authority.validate_successor_authority(
        fixture.authority_path,
        **fixture.arguments,
    )

    assert validated == fixture.authority
    assert validated["route"] == successor_authority.ROUTE
    assert validated["execution_policy"] == {
        "parent_platform": "cpu",
        "child_platform": "cuda",
        "jax_enable_x64": True,
        "compilation_cache_enabled": False,
        "child_preallocate": True,
        "command_buffer_enabled": False,
        "required_xla_flag": "--xla_gpu_enable_command_buffer=",
    }


@pytest.mark.parametrize("occupied_path", ["final", "staging"])
def test_diag3_successor_authority_rejects_consumed_output_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    occupied_path: str,
) -> None:
    fixture = _diag3_authority_fixture(tmp_path, monkeypatch)
    occupied = (
        fixture.output_root
        if occupied_path == "final"
        else fixture.output_root.with_name(f"{fixture.output_root.name}.partial-test")
    )
    occupied.mkdir()

    with pytest.raises(
        FileExistsError,
        match="output root or staging sibling already exists",
    ):
        successor_authority.validate_successor_authority(
            fixture.authority_path,
            **fixture.arguments,
        )


def test_diag4_authority_addition_preserves_diag1_through_diag3_api() -> None:
    assert tuple(successor_authority.SuccessorAuthorityClaim.__dataclass_fields__) == (
        "payload",
        "authority_sha256",
        "plan_sha256",
    )
    assert tuple(
        inspect.signature(successor_authority.validate_successor_authority).parameters
    ) == (
        "authority_path",
        "repository_root",
        "output_root",
        "reference_root",
        "input_root",
        "interpreter",
    )
    assert tuple(
        inspect.signature(successor_authority.claim_successor_authority).parameters
    ) == (
        "authority_path",
        "repository_root",
        "output_root",
        "reference_root",
        "input_root",
        "interpreter",
    )


def test_diag3_snapshot_closure_is_frozen_against_later_diag2_growth() -> None:
    qualified = {
        relative: hashlib.sha256(relative.encode()).hexdigest()
        for relative in successor_authority.QUALIFIED_FILE_PATHS
    }
    authority_sha256 = "a" * 64
    plan_sha256 = "b" * 64
    expected = {
        **qualified,
        successor_authority.AUTHORITY_RELATIVE_PATH: authority_sha256,
        successor_authority.PLAN_RELATIVE_PATH: plan_sha256,
    }
    assert len(expected) == 16
    assert successor_authority.DIAG3_SOURCE_DELTA_ALLOWLIST == frozenset(expected)
    assert (
        diagnostic_receipt.DIAG2_SOURCE_DELTA_ALLOWLIST
        - successor_authority.DIAG3_SOURCE_DELTA_ALLOWLIST
    ) == {
        "benchmarks/qualify_single_stage_native_equivalent_quality_gntr3_cpu.py",
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag4_iterative_retraction_plan.md",
        # The projected-route modules landed after DIAG1 sealed its baseline.
        # They are new files, not edits to a reviewed numerical source, so the
        # DIAG2 filter excludes them by path and the frozen filtered count and
        # digest still describe the DIAG1 tree.  DIAG3's own closure is
        # unaffected: it never saw them.
        "benchmarks/regenerate_execution_source_manifest.py",
        "benchmarks/rehearse_single_stage_projected_route_cpu.py",
        "benchmarks/run_single_stage_projected_route_gpu_root.py",
        # The standalone package validator landed after the root sealed. It reads
        # a published certificate and never participates in a certified run, so
        # it is a new file by the same rule: filtered out by path, leaving the
        # DIAG1 count and digest describing the DIAG1 tree.
        "benchmarks/validate_projected_route_package.py",
        # The shipped projected-route example landed in the same freeze as the
        # GPU launcher, before the root opened, so the certified bytes and the
        # shipped bytes are one tree.  It is a new file too.
        "examples/jax/3_Advanced/single_stage_boozer_vacuum_projected_route.py",
        "src/simsopt_jax/geo/optimizers/dense_tangent_curvature.py",
        "src/simsopt_jax/geo/optimizers/lagrangian_newton_cg.py",
        "src/simsopt_jax/geo/optimizers/projected_lbfgs.py",
        "src/simsopt_jax/geo/optimizers/quasi_newton_metric.py",
        "src/simsopt_jax/geo/optimizers/tangent_gauss_newton.py",
    }
    snapshot = SnapshotPublication(
        root=Path("/snapshot"),
        manifest_path=Path("/snapshot/source-manifest.json"),
        manifest_sha256="c" * 64,
        entries=tuple(
            SnapshotEntry("test", relative, 0, digest)
            for relative, digest in sorted(expected.items())
        ),
        worktree=WorktreeIdentity("d" * 40, "e" * 64, "f" * 64, "/repository"),
    )
    claim = SimpleNamespace(
        payload={"qualified_files": qualified},
        authority_sha256=authority_sha256,
        plan_sha256=plan_sha256,
    )
    successor_authority.validate_successor_snapshot(snapshot, claim)


def test_diag3_successor_authority_rejects_a_different_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag3_authority_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="launch output_root differs"):
        successor_authority.validate_successor_authority(
            fixture.authority_path,
            **{
                **fixture.arguments,
                "output_root": fixture.output_root.with_name("wrong-root"),
            },
        )


def test_diag3_qualification_record_accepts_exact_typed_evidence(
    tmp_path: Path,
) -> None:
    payload = _qualification_payload()
    authority_sha256 = "a" * 64
    record = _qualification_record(payload, tmp_path, authority_sha256)

    _validate_qualification_test_record(record, payload, tmp_path, authority_sha256)


@pytest.mark.parametrize("passed", [849, 851, 850.0, True])
def test_diag3_qualification_record_requires_exact_integer_pass_count(
    tmp_path: Path,
    passed: successor_authority.JsonValue,
) -> None:
    payload = _qualification_payload()
    authority_sha256 = "a" * 64
    record = _qualification_record(payload, tmp_path, authority_sha256)
    controlling = record["controlling_cpu"]
    assert isinstance(controlling, dict)
    controlling["passed"] = passed

    with pytest.raises((TypeError, ValueError)):
        _validate_qualification_test_record(record, payload, tmp_path, authority_sha256)


@pytest.mark.parametrize("duration_seconds", [1, -0.25, True])
def test_diag3_qualification_record_requires_finite_positive_float_duration(
    tmp_path: Path,
    duration_seconds: successor_authority.JsonValue,
) -> None:
    payload = _qualification_payload()
    authority_sha256 = "a" * 64
    record = _qualification_record(payload, tmp_path, authority_sha256)
    controlling = record["controlling_cpu"]
    assert isinstance(controlling, dict)
    controlling["duration_seconds"] = duration_seconds

    with pytest.raises(TypeError, match="finite positive JSON float"):
        _validate_qualification_test_record(record, payload, tmp_path, authority_sha256)


def test_diag3_qualification_record_rejects_nonfinite_duration(
    tmp_path: Path,
) -> None:
    payload = _qualification_payload()
    authority_sha256 = "a" * 64
    record = _qualification_record(payload, tmp_path, authority_sha256)
    record_bytes = runner.canonical_json_bytes(record).replace(
        b'"duration_seconds":1.25', b'"duration_seconds":1e309'
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        successor_authority._validate_qualification_record(
            b"## Qualification Record\n" + record_bytes,
            payload,
            authority_sha256=authority_sha256,
            output_root=tmp_path,
        )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("static_checks", "ruff_check"),
        ("authorization", "warm_allowed"),
        ("authorization", "preflight_launches"),
    ],
)
def test_diag3_qualification_record_rejects_boolean_integer_aliases(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    payload = _qualification_payload()
    authority_sha256 = "a" * 64
    record = _qualification_record(payload, tmp_path, authority_sha256)
    nested = record[section]
    assert isinstance(nested, dict)
    nested[field] = 1 if field != "preflight_launches" else True

    with pytest.raises(TypeError):
        _validate_qualification_test_record(record, payload, tmp_path, authority_sha256)


def test_diag3_successor_authority_claim_is_process_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag3_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_successor_authority(  # noqa: SIM117
        fixture.authority_path, **fixture.arguments
    ):
        with pytest.raises(RuntimeError, match="already claimed"):
            with successor_authority.claim_successor_authority(
                fixture.authority_path, **fixture.arguments
            ):
                raise AssertionError("a concurrent authority claim was admitted")


def test_diag3_claim_blocks_replacement_inode_claim_in_another_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bytes = b'{"authority":"original"}\n'
    repository = tmp_path / "repository"
    authority_path = _claim_test_repository(repository, original_bytes)
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    output_root = output_parent / "successor"
    monkeypatch.setattr(
        successor_authority,
        "_validate_successor_authority_bytes",
        _claim_test_authority_validator,
    )
    monkeypatch.setattr(
        successor_authority,
        "_validate_qualification_record",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        successor_authority,
        "PLAN_SHA256",
        hashlib.sha256(b"claim-test-plan\n").hexdigest(),
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_claim_process_expecting_replacement,
        args=(
            authority_path,
            repository,
            output_root,
            ready,
            release,
            "authority inode is not bound",
        ),
    )
    holder.start()
    assert ready.wait(timeout=10.0)

    replacement = authority_path.with_suffix(".replacement")
    replacement.write_bytes(b'{"authority":"replacement"}\n')
    os.replace(replacement, authority_path)
    contender = context.Process(
        target=_blocked_claim_process,
        args=(authority_path, repository, output_root),
    )
    contender.start()
    contender.join(timeout=10.0)
    release.set()
    holder.join(timeout=10.0)

    assert contender.exitcode == 0
    assert holder.exitcode == 0


def test_diag3_claim_blocks_same_output_from_a_different_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_bytes = b'{"authority":"same-output"}\n'
    first_repository = tmp_path / "repository-a"
    second_repository = tmp_path / "repository-b"
    first_authority = _claim_test_repository(first_repository, authority_bytes)
    second_authority = _claim_test_repository(second_repository, authority_bytes)
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    output_root = output_parent / "successor"
    monkeypatch.setattr(
        successor_authority,
        "_validate_successor_authority_bytes",
        _claim_test_authority_validator,
    )
    monkeypatch.setattr(
        successor_authority,
        "_validate_qualification_record",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        successor_authority,
        "PLAN_SHA256",
        hashlib.sha256(b"claim-test-plan\n").hexdigest(),
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_claim_process,
        args=(
            first_authority,
            first_repository,
            output_root,
            ready,
            release,
            hashlib.sha256(authority_bytes).hexdigest(),
        ),
    )
    holder.start()
    assert ready.wait(timeout=10.0)

    contender = context.Process(
        target=_blocked_claim_process,
        args=(second_authority, second_repository, output_root),
    )
    contender.start()
    contender.join(timeout=10.0)
    release.set()
    holder.join(timeout=10.0)

    assert contender.exitcode == 0
    assert holder.exitcode == 0


def test_diag3_claim_blocks_replaced_output_parent_and_detects_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_bytes = b'{"authority":"output-parent-replacement"}\n'
    first_repository = tmp_path / "repository-a"
    second_repository = tmp_path / "repository-b"
    first_authority = _claim_test_repository(first_repository, authority_bytes)
    second_authority = _claim_test_repository(second_repository, authority_bytes)
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    output_root = output_parent / "successor"
    monkeypatch.setattr(
        successor_authority,
        "_validate_successor_authority_bytes",
        _claim_test_authority_validator,
    )
    monkeypatch.setattr(
        successor_authority,
        "_validate_qualification_record",
        lambda *_arguments, **_keywords: None,
    )
    monkeypatch.setattr(
        successor_authority,
        "PLAN_SHA256",
        hashlib.sha256(b"claim-test-plan\n").hexdigest(),
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_claim_process_expecting_replacement,
        args=(
            first_authority,
            first_repository,
            output_root,
            ready,
            release,
            "directory inode is not bound",
        ),
    )
    holder.start()
    assert ready.wait(timeout=10.0)

    displaced_output_parent = tmp_path / "campaigns-displaced"
    os.replace(output_parent, displaced_output_parent)
    output_parent.mkdir()
    contender = context.Process(
        target=_blocked_claim_process,
        args=(second_authority, second_repository, output_root),
    )
    contender.start()
    contender.join(timeout=10.0)
    release.set()
    holder.join(timeout=10.0)

    assert contender.exitcode == 0
    assert holder.exitcode == 0


def test_diag3_successor_authority_binds_the_executed_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag3_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_successor_authority(
        fixture.authority_path, **fixture.arguments
    ) as claim:
        qualified = claim.payload["qualified_files"]
        assert isinstance(qualified, dict)
        expected = {
            **{str(path): str(digest) for path, digest in qualified.items()},
            successor_authority.AUTHORITY_RELATIVE_PATH: claim.authority_sha256,
            successor_authority.PLAN_RELATIVE_PATH: claim.plan_sha256,
        }
        entries = tuple(
            SnapshotEntry("test", relative, 0, digest)
            for relative, digest in sorted(expected.items())
        )
        snapshot = SnapshotPublication(
            root=tmp_path,
            manifest_path=tmp_path / "source-manifest.json",
            manifest_sha256="a" * 64,
            entries=entries,
            worktree=WorktreeIdentity(
                "b" * 40,
                "c" * 64,
                "d" * 64,
                str(fixture.repository),
            ),
        )
        successor_authority.validate_successor_snapshot(snapshot, claim)

        changed = replace(entries[0], sha256="f" * 64)
        with pytest.raises(ValueError, match="snapshot differs from authority"):
            successor_authority.validate_successor_snapshot(
                replace(snapshot, entries=(changed, *entries[1:])), claim
            )


def test_diag4_authority_consumes_once_and_binds_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        assert dict(claim.expected_numerical_identity) == fixture.identity
        assert dict(claim.expected_frozen_numerical_entries) == fixture.frozen
        with pytest.raises(TypeError):
            claim.expected_frozen_numerical_entries[fixture.frozen_relative] = "f" * 64
        expected_entries: list[SnapshotEntry] = []
        for relative, entry in sorted(fixture.execution_entries.items()):
            role = (
                "test"
                if relative.startswith("tests/")
                else "benchmark"
                if relative.startswith("benchmarks/")
                else "execution_source"
            )
            assert isinstance(entry, dict)
            expected_entries.append(
                SnapshotEntry(
                    role,
                    relative,
                    int(entry["size_bytes"]),
                    str(entry["sha256"]),
                )
            )
        manifest_bytes = fixture.execution_manifest_path.read_bytes()
        expected_entries.extend(
            (
                SnapshotEntry(
                    "execution_source_manifest",
                    successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
                    len(manifest_bytes),
                    hashlib.sha256(manifest_bytes).hexdigest(),
                ),
                SnapshotEntry(
                    "native_extension",
                    f"native/{fixture.native_extension.name}",
                    fixture.native_extension.stat().st_size,
                    hashlib.sha256(fixture.native_extension.read_bytes()).hexdigest(),
                ),
            )
        )
        snapshot = SnapshotPublication(
            root=tmp_path,
            manifest_path=tmp_path / "source-manifest.json",
            manifest_sha256="a" * 64,
            entries=tuple(
                sorted(expected_entries, key=lambda entry: entry.relative_path)
            ),
            worktree=WorktreeIdentity(
                "b" * 40, "c" * 64, "d" * 64, str(fixture.repository)
            ),
        )
        successor_authority.validate_diag4_successor_snapshot(snapshot, claim)
        changed_entry = replace(snapshot.entries[0], sha256="f" * 64)
        with pytest.raises(ValueError, match="GPU source snapshot differs"):
            successor_authority.validate_diag4_successor_snapshot(
                replace(snapshot, entries=(changed_entry, *snapshot.entries[1:])),
                claim,
            )
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-bound"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        consumed = successor_authority.consume_diag4_successor_authority(claim)
        assert consumed.path.is_file()
        assert consumed.path.stat().st_mode & 0o777 == 0o444
        assert consumed.path.read_bytes() == runner.canonical_json_bytes(
            dict(consumed.payload)
        )
        with pytest.raises(RuntimeError, match="already consumed"):
            successor_authority.consume_diag4_successor_authority(claim)
        os.replace(staging, fixture.output_root)
        successor_authority.revalidate_diag4_successor_authority(
            claim, require_output_absent=False
        )


def test_diag4_authority_real_consumed_lifetime_through_final_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-lifetime"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        successor_authority.consume_diag4_successor_authority(claim)
        terminal = staging / "terminal.json"
        terminal_payload = {
            "schema_version": "diag4-lifetime-test-v1",
            "launched_children": [],
            "terminal": True,
        }
        terminal.write_bytes(runner.canonical_json_bytes(terminal_payload))
        terminal_descriptor = os.open(terminal, os.O_RDONLY)
        try:
            os.fsync(terminal_descriptor)
        finally:
            os.close(terminal_descriptor)
        terminal.chmod(0o444)
        staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
        staging.chmod(0o555)
        os.replace(staging, fixture.output_root)
        parent_descriptor = os.open(
            fixture.output_root.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        reloaded = runner.load_canonical_json_bytes(
            (fixture.output_root / "terminal.json").read_bytes()
        )
        assert reloaded == terminal_payload
        successor_authority.revalidate_diag4_successor_authority(
            claim, require_output_absent=False
        )


@pytest.mark.parametrize("mutation", ["qualification", "identity"])
def test_diag4_authority_rejects_stale_qualification_or_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    if mutation == "qualification":
        plan_path = fixture.repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH
        plan_path.write_bytes(plan_path.read_bytes() + b"\n")
    else:
        payload = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
        assert isinstance(payload, dict)
        identity = payload["numerical_identity"]
        assert isinstance(identity, dict)
        identity["problem_sha256"] = "f" * 64
        fixture.authority_path.write_bytes(runner.canonical_json_bytes(payload))

    with pytest.raises(ValueError):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


@pytest.mark.parametrize(
    "identity_field", sorted(successor_authority.DIAG4_IDENTITY_FIELDS)
)
def test_diag4_authority_rejects_each_independent_identity_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_field: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    identity = authority["numerical_identity"]
    assert isinstance(identity, dict)
    identity[identity_field] = "f" * 64
    fixture.authority_path.write_bytes(runner.canonical_json_bytes(authority))

    with pytest.raises(ValueError):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("execution_source_manifest_sha256", "f" * 64),
        ("execution_source_entries_sha256", "f" * 64),
        ("native_extension_path", "/tmp/not-the-loaded-extension.so"),
        ("native_extension_sha256", "f" * 64),
        ("native_extension_size_bytes", 1),
    ],
)
def test_diag4_authority_rejects_execution_source_and_native_identity_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    changed: successor_authority.JsonValue,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    authority[field] = changed
    fixture.authority_path.write_bytes(runner.canonical_json_bytes(authority))
    with pytest.raises((FileNotFoundError, TypeError, ValueError)):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_authority_rejects_coherently_resealed_stale_aggregate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    changed_problem_sha256 = "e" * 64

    def mutate_record(record: dict[str, successor_authority.JsonValue]) -> None:
        identity = record["numerical_identity"]
        assert isinstance(identity, dict)
        identity["problem_sha256"] = changed_problem_sha256

    _diag4_reseal_qualification(fixture, mutate_record)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    identity = authority["numerical_identity"]
    assert isinstance(identity, dict)
    identity["problem_sha256"] = changed_problem_sha256
    fixture.authority_path.write_bytes(runner.canonical_json_bytes(authority))

    with pytest.raises(
        ValueError,
        match="scientific result differs|aggregate numerical identity differs",
    ):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


@pytest.mark.parametrize(
    ("mapping_name", "mutation"),
    [
        ("qualified", "missing"),
        ("qualified", "extra"),
        ("frozen", "missing"),
        ("frozen", "extra"),
    ],
)
def test_diag4_authority_requires_exact_source_map_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mapping_name: str,
    mutation: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    mapping = dict(getattr(fixture, mapping_name))
    if mutation == "missing":
        mapping.pop(min(mapping))
    else:
        mapping["unexpected.py"] = "f" * 64
    validator = (
        successor_authority._diag4_qualified_files
        if mapping_name == "qualified"
        else successor_authority._diag4_frozen_numerical_entries
    )
    with pytest.raises(ValueError, match="membership differs"):
        validator(mapping)


def test_diag4_execution_source_manifest_binds_no_ignore_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    manifest_bytes = fixture.execution_manifest_path.read_bytes()
    entries, sizes, entries_sha256 = (
        successor_authority._diag4_execution_source_entries(
            manifest_bytes,
            repository=fixture.repository,
            qualified=fixture.qualified,
            frozen=fixture.frozen,
            locked_leaf_bytes=None,
        )
    )
    assert "src/simsopt/_version.py" in entries
    assert len(entries) == successor_authority.DIAG4_EXECUTION_SOURCE_ENTRY_COUNT
    assert sizes["src/simsopt/_version.py"] > 0
    assert entries_sha256 == fixture.execution_entries_sha256


@pytest.mark.parametrize("mutation", ["missing", "extra", "self", "size", "aggregate"])
def test_diag4_execution_source_manifest_rejects_identity_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    manifest = runner.load_canonical_json_bytes(
        fixture.execution_manifest_path.read_bytes()
    )
    assert isinstance(manifest, dict)
    entries = manifest["entries"]
    assert isinstance(entries, dict)
    if mutation == "missing":
        entries.pop(min(entries))
    elif mutation == "extra":
        extra_path = fixture.repository / "src" / "unexpected.py"
        extra_path.write_bytes(b"unexpected = True\n")
        entries["src/unexpected.py"] = {
            "sha256": hashlib.sha256(extra_path.read_bytes()).hexdigest(),
            "size_bytes": extra_path.stat().st_size,
        }
    elif mutation == "self":
        entries[successor_authority.DIAG4_EXECUTION_SOURCE_MANIFEST_PATH] = {
            "sha256": "f" * 64,
            "size_bytes": 1,
        }
    elif mutation == "size":
        entry = entries[min(entries)]
        assert isinstance(entry, dict)
        entry["size_bytes"] = int(entry["size_bytes"]) + 1
    if mutation == "aggregate":
        manifest["entries_sha256"] = "f" * 64
    else:
        manifest["entries_sha256"] = hashlib.sha256(
            runner.canonical_json_bytes(entries)
        ).hexdigest()
    with pytest.raises(ValueError):
        successor_authority._diag4_execution_source_entries(
            runner.canonical_json_bytes(manifest),
            repository=fixture.repository,
            qualified=fixture.qualified,
            frozen=fixture.frozen,
            locked_leaf_bytes=None,
        )


def test_diag4_execution_source_manifest_rejects_new_unlisted_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    extra = fixture.repository / "src" / "new_unlisted.py"
    extra.write_bytes(b"NEW = True\n")
    with pytest.raises(ValueError, match="membership differs"):
        successor_authority._diag4_execution_source_entries(
            fixture.execution_manifest_path.read_bytes(),
            repository=fixture.repository,
            qualified=fixture.qualified,
            frozen=fixture.frozen,
            locked_leaf_bytes=None,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "controlling_command",
        "controlling_count",
        "controlling_duration",
        "static_boolean",
        "static_command",
        "execution_manifest_hash",
        "execution_entries_hash",
        "native_path",
        "native_hash",
        "native_size",
        "review_role",
        "reviewer_identity",
        "review_session",
        "review_source_hash",
        "review_frozen_hash",
        "review_execution_manifest_hash",
        "review_execution_entries_hash",
        "review_count",
    ],
)
def test_diag4_qualification_rejects_command_and_review_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)

    def mutate(record: dict[str, successor_authority.JsonValue]) -> None:
        controlling = record["controlling_cpu"]
        static_checks = record["static_checks"]
        static_commands = record["static_commands"]
        reviews = record["independent_reviews"]
        assert isinstance(controlling, dict)
        assert isinstance(static_checks, dict)
        assert isinstance(static_commands, dict)
        assert isinstance(reviews, list)
        assert isinstance(reviews[0], dict)
        if mutation == "controlling_command":
            controlling["command"] = "pytest"
        elif mutation == "controlling_count":
            controlling["run_count"] = 2
        elif mutation == "controlling_duration":
            controlling["duration_seconds"] = 0.0
        elif mutation == "static_boolean":
            static_checks["ruff_check"] = False
        elif mutation == "static_command":
            command = static_commands["ruff_check"]
            assert isinstance(command, dict)
            command["command"] = "ruff check"
        elif mutation == "execution_manifest_hash":
            record["execution_source_manifest_sha256"] = "f" * 64
        elif mutation == "execution_entries_hash":
            record["execution_source_entries_sha256"] = "f" * 64
        elif mutation == "native_path":
            record["native_extension_path"] = "/tmp/other-extension.so"
        elif mutation == "native_hash":
            record["native_extension_sha256"] = "f" * 64
        elif mutation == "native_size":
            record["native_extension_size_bytes"] = 1
        elif mutation == "review_role":
            reviews[0]["role"] = reviews[1]["role"]
        elif mutation == "reviewer_identity":
            reviews[0]["reviewer"] = reviews[1]["reviewer"]
        elif mutation == "review_session":
            reviews[0]["session"] = reviews[1]["session"]
        elif mutation == "review_source_hash":
            reviews[0]["reviewed_qualified_files_sha256"] = "f" * 64
        elif mutation == "review_frozen_hash":
            reviews[0]["reviewed_frozen_numerical_entries_sha256"] = "f" * 64
        elif mutation == "review_execution_manifest_hash":
            reviews[0]["reviewed_execution_source_manifest_sha256"] = "f" * 64
        elif mutation == "review_execution_entries_hash":
            reviews[0]["reviewed_execution_source_entries_sha256"] = "f" * 64
        else:
            reviews.pop()

    _diag4_reseal_qualification(fixture, mutate)
    with pytest.raises((TypeError, ValueError)):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_controlling_command_is_exact_final_ssot_literal() -> None:
    repository = Path(__file__).resolve().parents[2]
    plan = (repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    section = plan.split("The controlling pytest command is exactly:", 1)[1]
    literal = section.split("```text\n", 1)[1].split("\n```", 1)[0]
    assert successor_authority.DIAG4_CONTROLLING_CPU_COMMAND == literal
    assert literal.count("--basetemp") == 1
    assert (
        "--basetemp /home/jungdaesuh/simsopt-campaigns/"
        "neq-gntr3-diag4-pytest-qualification-20260811T223700Z"
    ) in literal


@pytest.mark.parametrize(
    ("name", "heading"),
    [
        ("ruff_check", "The Ruff check command is exactly:"),
        ("ruff_format_check", "The Ruff format command is exactly:"),
        ("compileall", "The compile command is exactly:"),
        ("git_diff_check", "The whitespace command is exactly:"),
    ],
)
def test_diag4_static_commands_are_exact_final_ssot_literals(
    name: str,
    heading: str,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    plan = (repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH).read_text(
        encoding="utf-8"
    )
    section = plan.split(heading, 1)[1]
    literal = section.split("```text\n", 1)[1].split("\n```", 1)[0]
    assert successor_authority.DIAG4_STATIC_COMMANDS[name] == literal


@pytest.mark.parametrize(
    ("evidence_name", "field", "changed"),
    [
        ("historical_cpu20", "use", "CURRENT_BYTE_QUALIFICATION"),
        ("historical_cpu20", "promotion_eligible", True),
        ("historical_cpu20", "run_count", 2),
        ("historical_cpu20", "duration_seconds", 1.0),
        ("decisive_cpu_qualification", "scientific_outcome", "NO_HIT"),
        ("decisive_cpu_qualification", "qualification_passed", False),
        ("decisive_cpu_qualification", "run_count", 2),
        ("decisive_cpu_qualification", "artifact_manifest_sha256", "f" * 64),
        ("decisive_cpu_qualification", "scientific_evidence_sha256", "f" * 64),
    ],
)
def test_diag4_qualification_separates_historical_and_decisive_cpu_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evidence_name: str,
    field: str,
    changed: successor_authority.JsonValue,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    evidence = authority[evidence_name]
    assert isinstance(evidence, dict)
    evidence[field] = changed
    validator = (
        successor_authority._diag4_historical_cpu20
        if evidence_name == "historical_cpu20"
        else successor_authority._diag4_decisive_cpu_qualification
    )
    with pytest.raises((TypeError, ValueError)):
        if evidence_name == "historical_cpu20":
            validator(evidence, locked_leaf_bytes=None)
        else:
            validator(
                evidence,
                locked_leaf_bytes=None,
                expected_numerical_identity=fixture.identity,
                expected_source_entries=successor_authority._diag4_expected_cpu_source_entries(
                    {
                        relative: str(entry["sha256"])
                        for relative, entry in fixture.execution_entries.items()
                    },
                    hashlib.sha256(
                        fixture.execution_manifest_path.read_bytes()
                    ).hexdigest(),
                ),
                expected_execution_source_manifest_sha256=hashlib.sha256(
                    fixture.execution_manifest_path.read_bytes()
                ).hexdigest(),
                expected_execution_source_entries_sha256=(
                    fixture.execution_entries_sha256
                ),
                expected_native_extension_path=fixture.native_extension.resolve(),
                expected_native_extension_sha256=hashlib.sha256(
                    fixture.native_extension.read_bytes()
                ).hexdigest(),
                expected_native_extension_size_bytes=(
                    fixture.native_extension.stat().st_size
                ),
            )


def test_diag4_qualification_rejects_unmanifested_cpu_tree_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    fixture.cpu_qualification_root.chmod(0o755)
    extra = fixture.cpu_qualification_root / "unmanifested.json"
    extra.write_bytes(b"{}\n")
    extra.chmod(0o444)
    fixture.cpu_qualification_root.chmod(0o555)
    with pytest.raises(ValueError, match="tree is not closed"):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_qualification_joins_cpu_source_manifest_to_authority_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    authority = runner.load_canonical_json_bytes(fixture.authority_path.read_bytes())
    assert isinstance(authority, dict)
    evidence = authority["decisive_cpu_qualification"]
    assert isinstance(evidence, dict)
    expected = successor_authority._diag4_expected_cpu_source_entries(
        {
            relative: str(entry["sha256"])
            for relative, entry in fixture.execution_entries.items()
        },
        hashlib.sha256(fixture.execution_manifest_path.read_bytes()).hexdigest(),
    )
    expected[fixture.qualified_relative] = "f" * 64
    with pytest.raises(ValueError, match="source authority differs"):
        successor_authority._diag4_decisive_cpu_qualification(
            evidence,
            locked_leaf_bytes=None,
            expected_numerical_identity=fixture.identity,
            expected_source_entries=expected,
            expected_execution_source_manifest_sha256=hashlib.sha256(
                fixture.execution_manifest_path.read_bytes()
            ).hexdigest(),
            expected_execution_source_entries_sha256=(fixture.execution_entries_sha256),
            expected_native_extension_path=fixture.native_extension.resolve(),
            expected_native_extension_sha256=hashlib.sha256(
                fixture.native_extension.read_bytes()
            ).hexdigest(),
            expected_native_extension_size_bytes=fixture.native_extension.stat().st_size,
        )


def test_diag4_qualification_requires_public_scientific_reconstruction_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        successor_authority,
        "validate_native_equivalent_scientific_evidence",
        lambda **_kwargs: SimpleNamespace(
            outcome=diagnostic_receipt.ScientificOutcome.NO_HIT
        ),
    )
    with pytest.raises(ValueError, match="reconstruction is not QUALITY_HIT"):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "qualified_manifest",
        "qualified_source",
        "native_manifest",
        "native_entry",
        "plan_prefix",
        "qualification_static",
        "cardinality_preflight",
        "cardinality_cold",
        "cardinality_warm",
        "cardinality_retry",
    ],
)
def test_diag4_authority_rejects_bound_evidence_and_cardinality_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    if mutation == "qualified_manifest":
        authority = runner.load_canonical_json_bytes(
            fixture.authority_path.read_bytes()
        )
        assert isinstance(authority, dict)
        qualified = authority["qualified_files"]
        assert isinstance(qualified, dict)
        qualified[fixture.qualified_relative] = "f" * 64
        authority["qualified_files_sha256"] = hashlib.sha256(
            runner.canonical_json_bytes(qualified)
        ).hexdigest()
        fixture.authority_path.write_bytes(runner.canonical_json_bytes(authority))
    elif mutation == "qualified_source":
        fixture.qualified_path.write_bytes(b"changed qualified source\n")
    elif mutation == "native_manifest":
        fixture.native_manifest.write_bytes(
            fixture.native_manifest.read_bytes() + b"\n"
        )
    elif mutation == "native_entry":
        fixture.native_entry_path.write_bytes(b"changed native entry\n")
    elif mutation == "plan_prefix":
        plan_path = fixture.repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH
        plan_path.write_bytes(b"changed-prefix\n" + plan_path.read_bytes())
    else:

        def mutate(record: dict[str, successor_authority.JsonValue]) -> None:
            if mutation == "qualification_static":
                checks = record["static_checks"]
                assert isinstance(checks, dict)
                checks["ruff_check"] = False
                return
            authorization = record["authorization"]
            assert isinstance(authorization, dict)
            field = {
                "cardinality_preflight": "preflight_launches",
                "cardinality_cold": "maximum_cold_launches",
                "cardinality_warm": "warm_allowed",
                "cardinality_retry": "retry_allowed",
            }[mutation]
            authorization[field] = (
                2 if field in {"preflight_launches", "maximum_cold_launches"} else True
            )

        _diag4_reseal_qualification(fixture, mutate)

    with pytest.raises((TypeError, ValueError)):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_authority_revalidates_consumed_diag3_manifest_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    consumed_path = fixture.consumed_root / min(
        successor_authority.DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS
    )
    consumed_path.write_bytes(b"mutated\n")

    with pytest.raises(ValueError, match="consumed DIAG3 entry differs"):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_authority_rejects_output_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    fixture.output_root.mkdir()

    with pytest.raises(FileExistsError, match="output root"):
        successor_authority.validate_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        )


def test_diag4_claim_blocks_same_output_from_distinct_authority_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "campaigns" / "shared-output"
    consumed_root = tmp_path / "consumed-diag3"
    first = _diag4_authority_fixture(
        tmp_path,
        monkeypatch,
        repository_name="diag4-repository-a",
        output_root=output_root,
        consumed_root=consumed_root,
    )
    second = _diag4_authority_fixture(
        tmp_path,
        monkeypatch,
        repository_name="diag4-repository-b",
        output_root=output_root,
        consumed_root=consumed_root,
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_diag4_claim_process,
        args=(first, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10.0)
    contender = context.Process(target=_diag4_blocked_claim_process, args=(second,))
    contender.start()
    contender.join(timeout=10.0)
    release.set()
    holder.join(timeout=10.0)

    assert contender.exitcode == 0
    assert holder.exitcode == 0


def test_diag4_claim_detects_authority_directory_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    documentation = fixture.authority_path.parent

    with pytest.raises(ValueError, match="directory inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            os.replace(documentation, fixture.repository / "docs-displaced")
            documentation.mkdir()


def test_diag4_claim_detects_authority_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="authority inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            replacement = fixture.authority_path.with_suffix(".replacement")
            replacement.write_bytes(fixture.authority_path.read_bytes())
            os.replace(replacement, fixture.authority_path)


def test_diag4_claim_detects_output_parent_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    output_parent = fixture.output_root.parent

    with pytest.raises(ValueError, match="directory inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            os.replace(output_parent, tmp_path / "campaigns-displaced")
            output_parent.mkdir()


def test_diag4_claim_revalidates_frozen_numerical_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="bound leaf bytes differ"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            fixture.frozen_path.write_bytes(b"mutated frozen numerics\n")


@pytest.mark.parametrize(
    "leaf_kind",
    [
        "qualified",
        "frozen",
        "execution_manifest",
        "execution_entry",
        "live_native_extension",
        "native_manifest",
        "native_entry",
        "consumed_diag3",
        "cpu20_harness",
        "cpu20_result",
        "cpu_manifest",
        "cpu_scientific_evidence",
        "interpreter",
        "plan",
    ],
)
def test_diag4_claim_rejects_same_byte_bound_leaf_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_kind: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    paths = {
        "qualified": fixture.qualified_path,
        "frozen": fixture.frozen_path,
        "execution_manifest": fixture.execution_manifest_path,
        "execution_entry": fixture.repository / "src" / "simsopt" / "_version.py",
        "live_native_extension": fixture.native_extension,
        "native_manifest": fixture.native_manifest,
        "native_entry": fixture.native_entry_path,
        "consumed_diag3": fixture.consumed_root
        / min(successor_authority.DIAG4_REQUIRED_CONSUMED_DIAG3_PATHS),
        "cpu20_harness": fixture.cpu20_harness,
        "cpu20_result": fixture.cpu20_result,
        "cpu_manifest": fixture.cpu_manifest,
        "cpu_scientific_evidence": fixture.cpu_qualification_root
        / "scientific-evidence.json",
        "interpreter": fixture.interpreter,
        "plan": fixture.repository / successor_authority.DIAG4_PLAN_RELATIVE_PATH,
    }
    target = paths[leaf_kind]

    with pytest.raises(ValueError, match="inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            if leaf_kind.startswith("cpu_"):
                target.parent.chmod(0o755)
            replacement = target.with_name(f"{target.name}.replacement")
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
            if leaf_kind.startswith("cpu_"):
                target.parent.chmod(0o555)


def test_diag4_claim_rejects_open_to_fstatat_leaf_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_open = successor_authority.os.open
    replacement = fixture.qualified_path.with_suffix(".replacement")
    replacement.write_bytes(fixture.qualified_path.read_bytes())
    qualified_parent_identity = fixture.qualified_path.parent.stat()
    replaced = False

    def replace_after_open(
        path: str | bytes,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        parent_identity = os.fstat(dir_fd) if dir_fd is not None else None
        if (
            path == fixture.qualified_path.name
            and parent_identity is not None
            and (parent_identity.st_dev, parent_identity.st_ino)
            == (qualified_parent_identity.st_dev, qualified_parent_identity.st_ino)
            and not replaced
        ):
            replaced = True
            os.replace(replacement, fixture.qualified_path)
        return descriptor

    monkeypatch.setattr(successor_authority.os, "open", replace_after_open)
    with pytest.raises(ValueError, match="inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            raise AssertionError("leaf replacement race entered claim")


@pytest.mark.parametrize("target_kind", ["file", "root"])
def test_diag4_claim_rejects_cpu_qualification_mode_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_kind: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    target = (
        fixture.cpu_manifest
        if target_kind == "file"
        else fixture.cpu_qualification_root
    )
    with pytest.raises(ValueError, match="mode|bound leaf bytes differ"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ):
            target.chmod(0o644 if target_kind == "file" else 0o755)


def test_diag4_consumed_claim_rejects_competing_partial_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)

    with pytest.raises(FileExistsError, match="competing final or staging"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ) as claim:
            staging = fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-bound"
            )
            staging.mkdir()
            successor_authority.bind_diag4_staging_root(claim, staging)
            successor_authority.consume_diag4_successor_authority(claim)
            fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-competitor"
            ).mkdir()


def test_diag4_claim_rejects_bound_staging_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    staging = fixture.output_root.with_name(f"{fixture.output_root.name}.partial-bound")

    with pytest.raises(ValueError, match="staging directory inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ) as claim:
            staging.mkdir()
            successor_authority.bind_diag4_staging_root(claim, staging)
            os.replace(staging, tmp_path / "staging-displaced")
            staging.mkdir()


def test_diag4_claim_finalizes_unconsumed_prelaunch_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-prelaunch"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        os.replace(staging, fixture.output_root)
        successor_authority.finalize_diag4_prelaunch_failure(claim)
        successor_authority.revalidate_diag4_successor_authority(
            claim, require_output_absent=True
        )
        assert not successor_authority.diag4_consumption_marker_path(
            fixture.output_root
        ).exists()
        with pytest.raises(RuntimeError, match="already finalized"):
            successor_authority.finalize_diag4_prelaunch_failure(claim)


def test_diag4_prelaunch_failure_finalization_requires_bound_renamed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        with pytest.raises(RuntimeError, match="staging root must be bound"):
            successor_authority.finalize_diag4_prelaunch_failure(claim)
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-prelaunch"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        with pytest.raises(FileExistsError, match="output state differs"):
            successor_authority.finalize_diag4_prelaunch_failure(claim)


def test_diag4_prelaunch_failure_finalization_rejects_consumed_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-consumed"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        successor_authority.consume_diag4_successor_authority(claim)
        with pytest.raises(RuntimeError, match="consumed authority"):
            successor_authority.finalize_diag4_prelaunch_failure(claim)


@pytest.mark.parametrize("competitor", ["marker", "partial"])
def test_diag4_prelaunch_failure_finalization_rejects_competing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    competitor: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-prelaunch"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        os.replace(staging, fixture.output_root)
        if competitor == "marker":
            extra = successor_authority.diag4_consumption_marker_path(
                fixture.output_root
            )
            extra.write_bytes(b"unexpected marker\n")
        else:
            extra = fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-competitor"
            )
            extra.mkdir()
        with pytest.raises(FileExistsError, match="output state differs"):
            successor_authority.finalize_diag4_prelaunch_failure(claim)
        if extra.is_dir():
            extra.rmdir()
        else:
            extra.unlink()
        os.replace(fixture.output_root, staging)


def test_diag4_finalized_prelaunch_failure_detects_final_inode_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="staging directory inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ) as claim:
            staging = fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-prelaunch"
            )
            staging.mkdir()
            successor_authority.bind_diag4_staging_root(claim, staging)
            os.replace(staging, fixture.output_root)
            successor_authority.finalize_diag4_prelaunch_failure(claim)
            os.replace(fixture.output_root, tmp_path / "final-displaced")
            fixture.output_root.mkdir()


def test_diag4_finalized_prelaunch_failure_holds_claim_across_processes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "campaigns" / "shared-final"
    consumed_root = tmp_path / "consumed-diag3"
    first = _diag4_authority_fixture(
        tmp_path,
        monkeypatch,
        repository_name="diag4-finalized-a",
        output_root=output_root,
        consumed_root=consumed_root,
    )
    second = _diag4_authority_fixture(
        tmp_path,
        monkeypatch,
        repository_name="diag4-finalized-b",
        output_root=output_root,
        consumed_root=consumed_root,
    )
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_diag4_finalized_claim_process,
        args=(first, ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10.0)
    contender = context.Process(target=_diag4_blocked_claim_process, args=(second,))
    contender.start()
    contender.join(timeout=10.0)
    release.set()
    holder.join(timeout=10.0)

    assert contender.exitcode == 0
    assert holder.exitcode == 0


@pytest.mark.parametrize("fault", ["write", "chmod", "file_fsync"])
def test_diag4_consumption_prepublication_fault_remains_unconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-fault"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        if fault == "write":
            monkeypatch.setattr(
                successor_authority.os,
                "write",
                lambda *_args: (_ for _ in ()).throw(OSError("write fault")),
            )
        elif fault == "chmod":
            monkeypatch.setattr(
                successor_authority.os,
                "fchmod",
                lambda *_args: (_ for _ in ()).throw(OSError("chmod fault")),
            )
        else:
            monkeypatch.setattr(
                successor_authority.os,
                "fsync",
                lambda *_args: (_ for _ in ()).throw(OSError("file fsync fault")),
            )
        with pytest.raises(OSError, match="fault"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.STAGING_BOUND
        )
        assert not successor_authority.diag4_consumption_marker_path(
            fixture.output_root
        ).exists()
        assert not tuple(
            fixture.output_root.parent.glob(f".*{fixture.output_root.name}*.pending-*")
        )


def test_diag4_consumption_parent_fsync_fault_is_typed_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_fsync = successor_authority.os.fsync
    calls = 0

    def fault_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("parent fsync fault")
        real_fsync(descriptor)

    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-fsync"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        monkeypatch.setattr(successor_authority.os, "fsync", fault_parent_fsync)
        with pytest.raises(OSError, match="parent fsync fault"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        )
        assert successor_authority.diag4_consumption_marker_path(
            fixture.output_root
        ).is_file()
        os.replace(staging, fixture.output_root)


def test_diag4_consumption_post_marker_binding_fault_stays_typed_consumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_revalidate = successor_authority.revalidate_diag4_successor_authority
    calls = 0

    def fault_second_revalidation(
        claim: successor_authority.Diag4SuccessorAuthorityClaim,
        *,
        require_output_absent: bool,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("post-marker binding fault")
        real_revalidate(claim, require_output_absent=require_output_absent)

    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-binding"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        monkeypatch.setattr(
            successor_authority,
            "revalidate_diag4_successor_authority",
            fault_second_revalidation,
        )
        with pytest.raises(ValueError, match="post-marker binding fault"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.CONSUMED
        )
        os.replace(staging, fixture.output_root)


def test_diag4_consumption_pending_replacement_cannot_change_marker_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_link = successor_authority.os.link
    context = multiprocessing.get_context("fork")
    replace_now = context.Event()
    replaced = context.Event()
    marker = successor_authority.diag4_consumption_marker_path(fixture.output_root)
    pending = marker.with_name(f"{marker.name}.pending-{os.getpid()}")
    displaced = tmp_path / "verified-pending-inode"
    replacer = context.Process(
        target=_diag4_replace_pending_process,
        args=(pending, displaced, replace_now, replaced),
    )

    def replace_before_link(*args: object, **kwargs: object) -> None:
        replace_now.set()
        assert replaced.wait(timeout=10.0)
        real_link(*args, **kwargs)

    replacer.start()
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-replacement"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        monkeypatch.setattr(successor_authority.os, "link", replace_before_link)
        with pytest.raises(ValueError, match="pending marker inode is not bound"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        )
        assert marker.read_bytes() == runner.canonical_json_bytes(
            successor_authority._diag4_marker_payload(claim)
        )
        assert (marker.stat().st_dev, marker.stat().st_ino) == (
            displaced.stat().st_dev,
            displaced.stat().st_ino,
        )
        pending.unlink()
        displaced.unlink()
        os.replace(staging, fixture.output_root)
    replacer.join(timeout=10.0)
    assert replacer.exitcode == 0


def test_diag4_consumption_target_collision_is_typed_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_link = successor_authority.os.link
    marker = successor_authority.diag4_consumption_marker_path(fixture.output_root)

    def collide_then_link(*args: object, **kwargs: object) -> None:
        marker.write_bytes(b"foreign marker\n")
        real_link(*args, **kwargs)

    with pytest.raises(  # noqa: SIM117
        successor_authority.Diag4ConsumptionMarkerInvalidError,
        match="consumption marker",
    ):
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ) as claim:
            staging = fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-collision"
            )
            staging.mkdir()
            successor_authority.bind_diag4_staging_root(claim, staging)
            monkeypatch.setattr(successor_authority.os, "link", collide_then_link)
            with pytest.raises(FileExistsError):
                successor_authority.consume_diag4_successor_authority(claim)
            assert successor_authority.diag4_authority_lifecycle(claim) is (
                successor_authority.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
            )


def test_diag4_consumption_link_fault_cleans_pending_and_remains_unconsumed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-link-fault"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        monkeypatch.setattr(
            successor_authority.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link fault")),
        )
        with pytest.raises(OSError, match="link fault"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.STAGING_BOUND
        )
        assert not tuple(
            fixture.output_root.parent.glob(f".*{fixture.output_root.name}*.pending-*")
        )


def test_diag4_consumption_cleanup_fault_is_typed_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    real_unlink = successor_authority._unlink_diag4_pending_marker
    calls = 0

    def fail_first_cleanup(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("cleanup fault")
        real_unlink(*args, **kwargs)

    with successor_authority.claim_diag4_successor_authority(
        fixture.authority_path, **_diag4_claim_arguments(fixture)
    ) as claim:
        staging = fixture.output_root.with_name(
            f"{fixture.output_root.name}.partial-cleanup-fault"
        )
        staging.mkdir()
        successor_authority.bind_diag4_staging_root(claim, staging)
        monkeypatch.setattr(
            successor_authority,
            "_unlink_diag4_pending_marker",
            fail_first_cleanup,
        )
        with pytest.raises(OSError, match="cleanup fault"):
            successor_authority.consume_diag4_successor_authority(claim)
        assert successor_authority.diag4_authority_lifecycle(claim) is (
            successor_authority.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN
        )
        assert not tuple(
            fixture.output_root.parent.glob(f".*{fixture.output_root.name}*.pending-*")
        )
        os.replace(staging, fixture.output_root)


def test_diag4_claim_rejects_same_byte_consumption_marker_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _diag4_authority_fixture(tmp_path, monkeypatch)
    marker = successor_authority.diag4_consumption_marker_path(fixture.output_root)

    with pytest.raises(  # noqa: SIM117
        successor_authority.Diag4ConsumptionMarkerInvalidError,
        match="consumption marker",
    ):
        with successor_authority.claim_diag4_successor_authority(
            fixture.authority_path, **_diag4_claim_arguments(fixture)
        ) as claim:
            staging = fixture.output_root.with_name(
                f"{fixture.output_root.name}.partial-marker-replacement"
            )
            staging.mkdir()
            successor_authority.bind_diag4_staging_root(claim, staging)
            successor_authority.consume_diag4_successor_authority(claim)
            replacement = marker.with_name(f"{marker.name}.replacement")
            replacement.write_bytes(marker.read_bytes())
            replacement.chmod(0o444)
            os.replace(replacement, marker)
            os.replace(staging, fixture.output_root)


@pytest.mark.skipif(
    runner.shutil.which("nvidia-smi") is None,
    reason="live read-only NVIDIA management query is unavailable",
)
def test_diag2_fresh_cpu_parent_real_gpu_query_excludes_exact_pid() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(repository)!r}); "
        "import json, os; "
        "import benchmarks.run_single_stage_native_equivalent_quality_campaign as r; "
        "import jax; observation=r._capture_diag2_supervisor_zero("
        "{'CUDA_VISIBLE_DEVICES':r.GPU_UUID},"
        "query_executable_sha256=r.supervisor_query_executable_sha256()); "
        "print(json.dumps({'backend':jax.default_backend(),"
        "'supervisor_pid':os.getpid(),'gate':observation.gate_passes,"
        "'matching_pids':[row.pid for row in observation.matching_rows]}))"
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            "-c",
            script,
            "--diagnostic-successor-authority=authority.json",
        ),
        cwd=repository,
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )

    payload = runner.json.loads(completed.stdout.splitlines()[-1])
    assert payload["backend"] == "cpu"
    assert payload["supervisor_pid"] > 0
    assert payload["matching_pids"] == []
    assert payload["gate"] is True


def test_legacy_parent_import_preserves_caller_environment() -> None:
    repository = Path(__file__).resolve().parents[2]
    script = (
        f"import sys; sys.path.insert(0, {str(repository)!r}); "
        "import json, os; "
        "import benchmarks.run_single_stage_native_equivalent_quality_campaign; "
        "print(json.dumps({key: os.environ.get(key) for key in ("
        "'JAX_PLATFORMS','JAX_PLATFORM_NAME','JAX_COMPILATION_CACHE_DIR',"
        "'JAX_ENABLE_COMPILATION_CACHE','XLA_PYTHON_CLIENT_PREALLOCATE')}))"
    )
    expected = {
        "JAX_PLATFORMS": "cpu",
        "JAX_PLATFORM_NAME": "cpu",
        "JAX_COMPILATION_CACHE_DIR": "/tmp/legacy-cache",
        "JAX_ENABLE_COMPILATION_CACHE": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
    }
    environment = {**os.environ, **expected}

    completed = subprocess.run(
        (sys.executable, "-I", "-c", script, "--preflight-only"),
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert runner.json.loads(completed.stdout.splitlines()[-1]) == expected


def test_diag2_numpy_constraint_scale_reproduces_frozen_fp64_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native_raw_equalities = np.zeros(255, dtype="<f8")
    native_raw_equalities[254] = -float.fromhex(runner._DIAG2_VOLUME_TARGET_HEX)
    monkeypatch.setattr(
        runner.jax,
        "device_get",
        lambda *_args: pytest.fail("DIAG2 parent policy must not call JAX"),
    )

    volume_target, scale = runner._diag2_constraint_scale(
        0.0,
        native_raw_equalities,
    )

    assert volume_target.hex() == runner._DIAG2_VOLUME_TARGET_HEX
    assert float(scale[0]).hex() == runner._DIAG2_BOOZER_SCALE_HEX
    assert float(scale[254]).hex() == runner._DIAG2_VOLUME_SCALE_HEX
    assert runner._sha256(scale.tobytes(order="C")) == runner._DIAG2_SCALE_SHA256
    assert scale.dtype == np.dtype("<f8")
    assert scale.flags.c_contiguous


@pytest.mark.parametrize("invalid_volume", (True, 1, np.float64(1.0), float("inf")))
def test_diag2_numpy_policy_rejects_noncanonical_reference_volume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_volume: object,
) -> None:
    (tmp_path / runner.REFERENCE_FILENAME).write_bytes(b"{}")
    monkeypatch.setattr(
        runner,
        "validate_native_equivalent_reference",
        lambda _root: SimpleNamespace(usable=True),
    )
    monkeypatch.setattr(
        runner,
        "load_canonical_json_bytes",
        lambda _data: {"evidence": {"observables": {"volume": invalid_volume}}},
    )

    with pytest.raises(TypeError, match="finite float"):
        runner._derive_diag2_policy_authority(tmp_path)


def test_diag2_staging_seals_and_atomically_publishes_without_replace(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    final = output_parent / "diag2"
    publication = runner._prepare_diag2_publication(
        final,
        repository_root=repository,
    )
    payload = publication.staging_root / "nested/evidence.bin"
    payload.parent.mkdir()
    payload.write_bytes(b"evidence")

    runner._seal_and_sync_diag2_staging(publication.staging_root)
    runner._atomic_publish_diag2(publication)

    assert not publication.staging_root.exists()
    assert (final / "nested/evidence.bin").read_bytes() == b"evidence"
    assert (final / "nested/evidence.bin").stat().st_mode & 0o777 == 0o444
    assert (final / "nested").stat().st_mode & 0o777 == 0o555
    assert final.stat().st_mode & 0o777 == 0o555


def test_diag2_source_pre_failure_seals_and_reloads_v2_final_artifact(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    publication = runner._prepare_diag2_publication(
        output_parent / "diag2",
        repository_root=repository,
    )
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG2_EVIDENCE_SLOT_NAMES
    }
    failure = runner.StructuredFailureV2(
        runner.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        runner.FailureReasonCodeV2.SOURCE_PRE,
        "a" * 64,
    )

    receipt = runner._publish_diag2_terminal_and_receipt(
        publication,
        refs,
        failure=failure,
        launched_children=(),
        policy_authority_produced=False,
        preflight_authorized=False,
        cold_authorized=False,
    )

    assert receipt.verdict == "DIAGNOSTIC_INCOMPLETE"
    assert not publication.staging_root.exists()
    assert publication.final_root.is_dir()
    assert runner.load_and_validate_diag3_artifact(publication.final_root) == receipt
    with pytest.raises(ValueError, match="DIAG2 receipt schema differs"):
        diagnostic_receipt.load_and_validate_diag2_artifact(publication.final_root)


def test_diag2_source_post_setup_failure_seals_before_later_authorities(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    publication = runner._prepare_diag2_publication(
        output_parent / "diag2",
        repository_root=repository,
    )
    source_root = publication.staging_root / "source-snapshot"
    source_root.mkdir()
    (source_root / "source-manifest.json").write_bytes(b"{corrupt")
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG2_EVIDENCE_SLOT_NAMES
    }
    failure = runner.StructuredFailureV2(
        runner.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
        runner.FailureReasonCodeV2.SOURCE_POST,
        "a" * 64,
    )

    receipt = runner._publish_diag2_terminal_and_receipt(
        publication,
        refs,
        failure=failure,
        launched_children=(),
        policy_authority_produced=False,
        preflight_authorized=False,
        cold_authorized=False,
    )

    assert receipt.failure is not None
    assert receipt.failure.reason is runner.FailureReasonCodeV2.SOURCE_POST
    assert not publication.staging_root.exists()
    assert runner.load_and_validate_diag3_artifact(publication.final_root) == receipt


def test_diag2_publication_rejects_existing_final_and_symlink_parent(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    existing = output_parent / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="absent"):
        runner._prepare_diag2_publication(existing, repository_root=repository)
    partial = output_parent / "diag2.partial-existing"
    partial.mkdir()
    with pytest.raises(FileExistsError, match="staging siblings must be absent"):
        runner._prepare_diag2_publication(
            output_parent / "diag2", repository_root=repository
        )
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "alias-parent"
    alias.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        runner._prepare_diag2_publication(alias / "diag2", repository_root=repository)


def test_diag2_atomic_publish_never_replaces_destination(tmp_path: Path) -> None:
    parent = tmp_path / "campaigns"
    parent.mkdir()
    nonce = "0" * 32
    staging = parent / f"diag2.partial-{nonce}"
    staging.mkdir()
    (staging / "evidence.bin").write_bytes(b"new")
    final = parent / "diag2"
    final.mkdir()
    (final / "evidence.bin").write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        runner._atomic_publish_diag2(runner.Diag2Publication(staging, final, nonce))

    assert staging.is_dir()
    assert (final / "evidence.bin").read_bytes() == b"existing"


def test_diag3_numerical_bundle_commits_or_quarantines_by_one_rename(
    tmp_path: Path,
) -> None:
    cold = tmp_path / "cold"
    cold.mkdir()
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "history.json").write_bytes(b"history")

    runner._materialize_cold_numerical_bundle(publication)

    assert not publication.pending_root.exists()
    assert (publication.committed_root / "history.json").read_bytes() == b"history"
    assert not (cold / "history.json").exists()

    second = tmp_path / "second-cold"
    second.mkdir()
    second_publication = runner._cold_numerical_bundle_publication(second)
    second_publication.pending_root.mkdir()
    (second_publication.pending_root / "partial.bin").write_bytes(b"partial")

    runner._quarantine_cold_numerical_bundle(second_publication)

    assert not second_publication.pending_root.exists()
    assert (
        second_publication.uncommitted_root / "partial.bin"
    ).read_bytes() == b"partial"


def test_diag3_numerical_commit_never_replaces_destination(tmp_path: Path) -> None:
    cold = tmp_path / "cold"
    cold.mkdir()
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "new.bin").write_bytes(b"new")
    publication.committed_root.mkdir()
    (publication.committed_root / "old.bin").write_bytes(b"old")

    with pytest.raises(FileExistsError):
        runner._materialize_cold_numerical_bundle(publication)

    assert (publication.pending_root / "new.bin").read_bytes() == b"new"
    assert (publication.committed_root / "old.bin").read_bytes() == b"old"


def test_diag3_invalid_pending_bundle_is_quarantined_and_untyped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cold = tmp_path / "cold"
    cold.mkdir()
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "partial.bin").write_bytes(b"partial")
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=1,
        child_start_time_ticks=2,
        process_seconds=1.0,
        producer={"execution_status": "COMPLETE"},
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory={"peak_memory_fraction": 0.1},
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
    )
    monkeypatch.setattr(
        runner,
        "_validate_pending_cold_numerical_bundle",
        lambda *_args: (_ for _ in ()).throw(ValueError("incomplete bundle")),
    )

    resolved = runner._resolve_cold_numerical_bundle(cold, outcome)

    assert resolved.producer is None
    assert resolved.selected_failure_reason is (
        runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID
    )
    assert not publication.pending_root.exists()
    assert (publication.uncommitted_root / "partial.bin").read_bytes() == b"partial"


def test_diag4_parent_commits_complete_bundle_and_quarantines_postsolve_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete_cold = tmp_path / "complete" / "cold"
    complete_cold.mkdir(parents=True)
    complete_publication = runner._cold_numerical_bundle_publication(complete_cold)
    complete_publication.pending_root.mkdir()
    (complete_publication.pending_root / "history.json").write_bytes(b"complete")
    complete = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=1,
        child_start_time_ticks=2,
        process_seconds=1.0,
        producer={
            "schema_version": runner.DIAG4_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        },
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory={"peak_memory_fraction": 0.1},
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
    )
    monkeypatch.setattr(
        runner, "_validate_pending_diag4_numerical_bundle", lambda *_args: None
    )

    committed = runner._resolve_diag4_cold_numerical_bundle(complete_cold, complete)

    assert committed is complete
    assert not complete_publication.pending_root.exists()
    assert (complete_publication.committed_root / "history.json").read_bytes() == (
        b"complete"
    )

    failed_cold = tmp_path / "failed" / "cold"
    failed_cold.mkdir(parents=True)
    failed_publication = runner._cold_numerical_bundle_publication(failed_cold)
    failed_publication.pending_root.mkdir()
    (failed_publication.pending_root / "terminal-numerical.json").write_bytes(
        b"postsolve"
    )
    failed = replace(
        complete,
        terminal_status=runner.ChildTerminalStatus.CRASH,
        producer=None,
        producer_absence_reason=runner.AbsenceReason.CHILD_EXIT_NONZERO,
        selected_failure_reason=runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO,
    )

    retained = runner._resolve_diag4_cold_numerical_bundle(failed_cold, failed)

    assert retained is failed
    assert not failed_publication.pending_root.exists()
    assert (
        failed_publication.uncommitted_root / "terminal-numerical.json"
    ).read_bytes() == b"postsolve"


def _diag4_structural_pending_fixture(
    cold: Path,
) -> tuple[
    runner.ColdNumericalBundlePublication,
    dict[str, object],
]:
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir(parents=True)
    runtime_path = cold / "runtime-evidence.json"
    policy_path = cold / "policy.json"
    runner._publish_canonical_json(runtime_path, {"runtime": "test"})
    runner._publish_canonical_json(policy_path, {"policy": "test"})
    arrays: dict[str, object] = {}
    for name in runner.DIAGNOSTIC_ARRAY_SPECS:
        path = publication.pending_root / "arrays" / f"{name}.npy"
        runner._publish_bytes(path, f"array:{name}".encode())
        arrays[name] = {
            "artifact": runner._artifact_ref_payload(
                runner._artifact_ref_at(
                    path,
                    cold.parent,
                    publication.committed_root / "arrays" / f"{name}.npy",
                    "test-array-v1",
                )
            )
        }
    documents = {
        "history_evidence": ("history.json", {"history": "test"}, "history-v1"),
        "solve_timing_evidence": (
            "solve-timing.json",
            {"timing": "test"},
            runner.DIAG4_SOLVE_TIMING_SCHEMA_VERSION,
        ),
        "safeguard_telemetry_evidence": (
            "safeguard-telemetry.json",
            {"telemetry": "test"},
            runner.DIAG4_SAFEGUARD_TELEMETRY_SCHEMA_VERSION,
        ),
        "terminal_numerical_evidence": (
            "terminal-numerical.json",
            {"arrays": arrays},
            f"{runner.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal",
        ),
    }
    references: dict[str, runner.ArtifactRef] = {}
    for field, (relative, payload, schema) in documents.items():
        path = publication.pending_root / relative
        runner._publish_canonical_json(path, payload)
        references[field] = runner._artifact_ref_at(
            path,
            cold.parent,
            publication.committed_root / relative,
            schema,
        )
    identity = {
        "base_neq_gntr1_policy_sha256": "1" * 64,
        "problem_sha256": "2" * 64,
        "optimizer_options_sha256": "3" * 64,
        "scaling_sha256": "4" * 64,
        "bootstrap_state_sha256": "5" * 64,
        "initial_physical_state_sha256": "6" * 64,
        "identity_sha256": "7" * 64,
    }
    producer: dict[str, object] = {
        "schema_version": runner.DIAG4_COLD_RESULT_SCHEMA_VERSION,
        "numerical_bundle_schema_version": (
            runner.DIAG4_NUMERICAL_BUNDLE_SCHEMA_VERSION
        ),
        "route": runner.DIAG4_ROUTE,
        "numerical_route": runner.DIAG4_NUMERICAL_ROUTE,
        "numerical_result_schema_version": (
            runner.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION
        ),
        "plan_sha256": runner.DIAG4_PLAN_SHA256,
        "execution_status": "COMPLETE",
        "runtime": {},
        "runtime_evidence": runner._artifact_ref_payload(
            runner._artifact_ref(runtime_path, cold.parent, "runtime-v1")
        ),
        "policy_evidence": runner._artifact_ref_payload(
            runner._artifact_ref(policy_path, cold.parent, "policy-v1")
        ),
        **identity,
        "source_manifest_sha256": "8" * 64,
        **{
            field: runner._artifact_ref_payload(reference)
            for field, reference in references.items()
        },
        "profiler_enabled": False,
        "profiler_start_calls": 0,
        "profiler_stop_calls": 0,
        "trace_normalization_calls": 0,
        "endpoint_audit_called": True,
        "campaign_authorized": False,
        "failure_reasons": [],
    }
    return publication, producer


def _diag4_complete_supervised(
    producer: Mapping[str, object],
) -> runner.DiagnosticSupervisedSampleV2:
    return runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=1,
        child_start_time_ticks=2,
        process_seconds=1.0,
        producer=dict(producer),
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory={"peak_memory_fraction": 0.1},
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
    )


@pytest.mark.parametrize(
    ("mode", "expected_stage", "expected_reason"),
    [
        (
            runner.DiagnosticChildMode.PREFLIGHT,
            runner.FailureStageV4.PREFLIGHT,
            runner.FailureReasonCodeV4.PREFLIGHT_PROTOCOL_INVALID,
        ),
        (
            runner.DiagnosticChildMode.COLD,
            runner.FailureStageV4.COLD,
            runner.FailureReasonCodeV4.COLD_PROTOCOL_INVALID,
        ),
    ],
)
@pytest.mark.parametrize("physical", [False, True])
def test_diag4_child_supervision_publication_fault_converges_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: runner.DiagnosticChildMode,
    expected_stage: runner.FailureStageV4,
    expected_reason: runner.FailureReasonCodeV4,
    physical: bool,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    publication = runner._prepare_diag2_publication(
        tmp_path / f"diag4-{mode.value}",
        repository_root=repository,
    )
    directory = publication.staging_root / mode.value
    directory.mkdir()
    refs = {name: None for name in runner.DIAG4_EVIDENCE_SLOT_NAMES}
    error: OSError | ValueError = (
        OSError("physical publication fault")
        if physical
        else ValueError("protocol publication fault")
    )
    monkeypatch.setattr(
        runner,
        "_publish_diag2_supervision",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )

    if physical:
        with pytest.raises(runner.Diag4HardPublicationError) as caught:
            runner._publish_diag4_child_supervision(
                publication,
                refs,
                directory,
                _diag4_complete_supervised({}),
                mode=mode,
            )
        assert caught.value.reason is expected_reason
        assert caught.value.root == publication.staging_root
    else:
        failure = runner._publish_diag4_child_supervision(
            publication,
            refs,
            directory,
            _diag4_complete_supervised({}),
            mode=mode,
        )
        assert failure is not None
        assert failure.stage is expected_stage
        assert failure.reason is expected_reason
    assert publication.staging_root.is_dir()
    assert not publication.final_root.exists()


@pytest.mark.parametrize("fault", ("symlink", "hardlink", "special"))
def test_diag4_real_pending_validation_rejects_link_and_special_faults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    cold = tmp_path / "campaign" / "cold"
    cold.mkdir(parents=True)
    publication, producer = _diag4_structural_pending_fixture(cold)
    monkeypatch.setattr(
        runner, "validate_diag4_numerical_documents", lambda **_kwargs: None
    )

    runner._validate_pending_diag4_numerical_bundle(publication, producer)
    target = publication.pending_root / "arrays" / "accepted_mask.npy"
    if fault == "symlink":
        original = target.read_bytes()
        target.unlink()
        backing = cold / "symlink-backing.bin"
        backing.write_bytes(original)
        target.symlink_to(backing)
    elif fault == "hardlink":
        original = target.read_bytes()
        target.unlink()
        backing = cold / "hardlink-backing.bin"
        backing.write_bytes(original)
        os.link(backing, target)
    else:
        os.mkfifo(publication.pending_root / "unexpected.fifo")

    with pytest.raises(ValueError, match="missing or extra|linked or special"):
        runner._validate_pending_diag4_numerical_bundle(publication, producer)


@pytest.mark.parametrize(
    ("fault", "expected_reason"),
    (
        ("timing", runner.FailureReasonCodeV4.TIMING_INVALID),
        ("rename", runner.FailureReasonCodeV4.COMMIT_RENAME_FAILED),
        ("postrename", runner.FailureReasonCodeV4.COMMITTED_DEEP_LOAD_FAILED),
    ),
)
def test_diag4_numerical_commit_faults_converge_to_exact_v4_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
    expected_reason: runner.FailureReasonCodeV4,
) -> None:
    cold = tmp_path / fault / "cold"
    cold.mkdir(parents=True)
    publication, producer = _diag4_structural_pending_fixture(cold)
    outcome = _diag4_complete_supervised(producer)
    validation_count = 0

    def validate(
        _bundle: runner.ColdNumericalBundlePublication,
        _producer: Mapping[str, object],
    ) -> None:
        nonlocal validation_count
        validation_count += 1
        if fault == "timing":
            raise runner.Diag4NumericalDocumentError(
                runner.FailureReasonCodeV4.TIMING_INVALID,
                "timing mutation",
            )
        if fault == "postrename" and validation_count == 2:
            raise ValueError("committed mutation")

    monkeypatch.setattr(runner, "_validate_pending_diag4_numerical_bundle", validate)
    if fault == "rename":
        monkeypatch.setattr(
            runner,
            "_materialize_cold_numerical_bundle",
            lambda _publication: (_ for _ in ()).throw(OSError("rename fault")),
        )

    resolved, failure = runner._resolve_diag4_cold_numerical_bundle_v4(cold, outcome)

    assert failure is not None
    assert failure.stage is runner.FailureStageV4.NUMERICAL_COMMIT
    assert failure.reason is expected_reason
    assert resolved.producer is None
    if fault == "timing":
        assert not publication.pending_root.exists()
        assert publication.uncommitted_root.is_dir()
    elif fault == "rename":
        assert publication.pending_root.is_dir()
        assert not publication.committed_root.exists()
    else:
        assert not publication.pending_root.exists()
        assert publication.committed_root.is_dir()


def test_diag4_terminal_is_structurally_joined_before_commit_and_deep_loaded_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = tmp_path / "terminal-join" / "cold"
    cold.mkdir(parents=True)
    publication, producer = _diag4_structural_pending_fixture(cold)
    artifact_roots: list[Path | None] = []
    terminal_documents: list[object] = []

    def validate_documents(**documents: object) -> None:
        artifact_root = documents["artifact_root"]
        assert artifact_root is None or isinstance(artifact_root, Path)
        artifact_roots.append(artifact_root)
        terminal_documents.append(documents["terminal_numerical"])

    monkeypatch.setattr(
        runner,
        "validate_diag4_numerical_documents",
        validate_documents,
    )

    _resolved, failure = runner._resolve_diag4_cold_numerical_bundle_v4(
        cold,
        _diag4_complete_supervised(producer),
    )

    assert failure is None
    assert artifact_roots == [None, cold.parent]
    assert len(terminal_documents) == 2
    assert terminal_documents[0] == terminal_documents[1]
    assert isinstance(terminal_documents[0], dict)
    assert frozenset(terminal_documents[0]) == {"arrays"}
    assert not publication.pending_root.exists()
    assert publication.committed_root.is_dir()


def test_diag4_invalid_pending_with_failed_quarantine_selects_quarantine_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = tmp_path / "invalid-quarantine" / "cold"
    cold.mkdir(parents=True)
    _publication, producer = _diag4_structural_pending_fixture(cold)
    outcome = _diag4_complete_supervised(producer)
    monkeypatch.setattr(
        runner,
        "_validate_pending_diag4_numerical_bundle",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid pending tree")),
    )
    monkeypatch.setattr(
        runner,
        "_quarantine_cold_numerical_bundle",
        lambda _bundle: (_ for _ in ()).throw(OSError("quarantine failed")),
    )

    resolved, failure = runner._resolve_diag4_cold_numerical_bundle_v4(cold, outcome)

    assert failure is not None
    assert failure.stage is runner.FailureStageV4.NUMERICAL_COMMIT
    assert failure.reason is runner.FailureReasonCodeV4.QUARANTINE_FAILED
    assert resolved.producer is None


def test_diag4_missing_pending_and_quarantine_collision_have_exact_v4_reasons(
    tmp_path: Path,
) -> None:
    missing_cold = tmp_path / "missing" / "cold"
    missing_cold.mkdir(parents=True)
    missing_outcome = _diag4_complete_supervised(
        {
            "schema_version": runner.DIAG4_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        }
    )
    _, missing_failure = runner._resolve_diag4_cold_numerical_bundle_v4(
        missing_cold, missing_outcome
    )
    assert missing_failure is not None
    assert missing_failure.reason is runner.FailureReasonCodeV4.PENDING_RESULT_ABSENT

    collision_cold = tmp_path / "collision" / "cold"
    collision_cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(collision_cold)
    publication.pending_root.mkdir()
    publication.uncommitted_root.mkdir()
    invalid_outcome = replace(
        missing_outcome,
        producer=None,
        producer_absence_reason=runner.AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
    )
    _, collision_failure = runner._resolve_diag4_cold_numerical_bundle_v4(
        collision_cold, invalid_outcome
    )
    assert collision_failure is not None
    assert collision_failure.reason is runner.FailureReasonCodeV4.QUARANTINE_FAILED


def test_diag5_missing_required_pending_selects_v5_absent_reason(
    tmp_path: Path,
) -> None:
    cold = tmp_path / "missing-diag5" / "cold"
    cold.mkdir(parents=True)
    outcome = _diag4_complete_supervised(
        {
            "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        }
    )

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    assert resolution.publication_allowed
    assert resolution.terminal_failure is not None
    assert (
        resolution.terminal_failure.reason
        is runner.FailureReasonCodeV5.PENDING_RESULT_ABSENT
    )
    assert resolution.pending_disposition_failure is None
    assert resolution.outcome.producer is outcome.producer


@pytest.mark.parametrize(
    "reason",
    (
        runner.FailureReasonCodeV5.TIMING_INVALID,
        runner.FailureReasonCodeV5.SAFEGUARD_TELEMETRY_INVALID,
        runner.FailureReasonCodeV5.NUMERICAL_IDENTITY_MISMATCH,
        runner.FailureReasonCodeV5.PENDING_RESULT_INVALID,
    ),
)
def test_diag5_numerical_rejection_converges_to_pending_result_invalid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reason: runner.FailureReasonCodeV5,
) -> None:
    cold = tmp_path / reason.value.lower() / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "opaque.bin").write_bytes(b"pending")
    outcome = _diag4_complete_supervised(
        {
            "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        }
    )
    detail = "typed numerical rejection"
    monkeypatch.setattr(
        runner,
        "_validate_pending_diag5_numerical_bundle",
        lambda *_args: (_ for _ in ()).throw(
            runner.Diag5NumericalDocumentError(reason, detail)
        ),
    )

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    assert resolution.publication_allowed
    assert resolution.terminal_failure == runner.StructuredFailureV5(
        runner.FailureStageV5.NUMERICAL_COMMIT,
        runner.FailureReasonCodeV5.PENDING_RESULT_INVALID,
        runner._sha256(f"{reason.value}:{runner._sha256(detail.encode())}".encode()),
    )
    assert resolution.outcome.producer is outcome.producer
    assert publication.uncommitted_root.is_dir()
    assert not publication.pending_root.exists()


def test_diag5_reserved_source_revalidation_details_are_exact() -> None:
    assert runner._diag5_post_child_source_failure(
        runner.DiagnosticChildMode.PREFLIGHT,
        None,
    ).detail_sha256 == (
        "b0201988e5421a54500000ee56d2a836585f49b62a7a8d689d0c7f516316222e"
    )
    assert runner._diag5_post_child_source_failure(
        runner.DiagnosticChildMode.COLD,
        None,
    ).detail_sha256 == (
        "320b43d84c82b9be812cdf389da4c89f74e548748922d8356a35d51a09192fa4"
    )


@pytest.mark.parametrize(
    "mode",
    (runner.DiagnosticChildMode.PREFLIGHT, runner.DiagnosticChildMode.COLD),
)
def test_diag5_post_child_source_revalidation_preserves_selected_failure(
    mode: runner.DiagnosticChildMode,
) -> None:
    selected = runner._diag5_failure(
        (
            runner.FailureStageV5.PREFLIGHT
            if mode is runner.DiagnosticChildMode.PREFLIGHT
            else runner.FailureStageV5.COLD
        ),
        (
            runner.FailureReasonCodeV5.PREFLIGHT_TIMEOUT
            if mode is runner.DiagnosticChildMode.PREFLIGHT
            else runner.FailureReasonCodeV5.COLD_TIMEOUT
        ),
        "selected",
    )

    assert runner._diag5_post_child_source_failure(mode, selected) is selected


@pytest.mark.parametrize(
    ("mode", "expected_names"),
    (
        (
            runner.DiagnosticChildMode.PREFLIGHT,
            ("preflight_producer", "preflight_terminal", "preflight_process"),
        ),
        (
            runner.DiagnosticChildMode.COLD,
            ("cold_producer", "cold_terminal", "cold_process"),
        ),
    ),
)
def test_diag5_post_child_authority_failure_has_committed_raw_prefix(
    mode: runner.DiagnosticChildMode,
    expected_names: tuple[str, ...],
) -> None:
    refs = {name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS}
    for name in expected_names:
        refs[name] = runner.ArtifactRef(
            runner.DIAG5_EVIDENCE_SLOT_PATHS[name],
            "a" * 64,
            1,
            "test-v1",
        )
    selected = runner._diag5_post_child_source_failure(mode, None)

    assert (
        tuple(name for name, ref in refs.items() if ref is not None) == expected_names
    )
    assert selected.reason is (
        runner.FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
        if mode is runner.DiagnosticChildMode.PREFLIGHT
        else runner.FailureReasonCodeV5.COLD_PROTOCOL_INVALID
    )


@pytest.mark.parametrize(
    ("category", "expected_reason"),
    (
        (
            successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD,
            runner.Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
        ),
        (
            successor_authority.Diag5FinalizerFailureCategory.REVALIDATION,
            runner.Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_REVALIDATION_FAILED,
        ),
        (
            successor_authority.Diag5FinalizerFailureCategory.FINALIZATION,
            runner.Diag5PhysicalPublicationReason.POST_FINAL_AUTHORITY_FINALIZATION_FAILED,
        ),
    ),
)
def test_diag5_finalizer_category_maps_to_exact_physical_reason(
    category: successor_authority.Diag5FinalizerFailureCategory,
    expected_reason: runner.Diag5PhysicalPublicationReason,
) -> None:
    cause = OSError("finalizer phase")
    error = successor_authority.Diag5FinalizerError(category, cause)

    assert runner._diag5_finalizer_publication_reason(error.category) is expected_reason
    assert error.cause is cause


def test_diag5_finalizer_source_variants_preserve_exact_typed_inputs() -> None:
    snapshot = cast(runner.SnapshotPublication, object())
    published = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        snapshot,
    )
    outcome = runner._diag5_failure(
        runner.FailureStageV5.SETUP,
        runner.FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
        "copy",
    )
    terminal = runner.ArtifactRef(
        "supervisor-terminal.json",
        "a" * 64,
        1,
        runner.DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    receipt = runner.ArtifactRef(
        "diagnostic.json",
        "b" * 64,
        1,
        runner.DIAG5_SCHEMA_VERSION,
    )
    pre_source = successor_authority.PreSourceFailure(
        successor_authority.Diag5FinalizerSourceKind.PRE_SOURCE_FAILURE,
        outcome,
        terminal,
        receipt,
    )

    assert published.snapshot is snapshot
    assert pre_source.outcome is outcome
    assert pre_source.supervisor_terminal is terminal
    assert pre_source.diagnostic_receipt is receipt


def test_diag5_malformed_finalizer_source_is_deep_load_failure() -> None:
    assert (
        runner._diag5_finalizer_publication_reason(
            successor_authority.Diag5FinalizerError(
                successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD,
                ValueError("pre-source terminal evidence is absent"),
            ).category
        )
        is runner.Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
    )


def test_diag5_broken_pending_symlink_is_not_misclassified_absent(
    tmp_path: Path,
) -> None:
    cold = tmp_path / "broken-diag5" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.symlink_to(cold / "absent-target")
    outcome = _diag4_complete_supervised(
        {
            "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        }
    )

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    assert not resolution.publication_allowed
    assert resolution.terminal_failure is not None
    assert (
        resolution.terminal_failure.reason
        is runner.FailureReasonCodeV5.QUARANTINE_FAILED
    )
    assert resolution.pending_disposition_failure is resolution.terminal_failure
    assert not publication.pending_root.exists()
    assert publication.uncommitted_root.is_symlink()


def test_diag5_outer_cold_failure_quarantines_and_deep_loads_opaque_bytes(
    tmp_path: Path,
) -> None:
    cold = tmp_path / "outer-failure-diag5" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "opaque.bin").write_bytes(b"opaque")
    outcome = replace(
        _diag4_complete_supervised(
            {
                "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
                "execution_status": "COMPLETE",
            }
        ),
        producer=None,
        producer_absence_reason=runner.AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
    )

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    assert resolution.publication_allowed
    assert resolution.terminal_failure is None
    assert resolution.pending_disposition_failure is None
    assert not publication.pending_root.exists()
    assert (publication.uncommitted_root / "opaque.bin").read_bytes() == b"opaque"


@pytest.mark.parametrize("marker_fails", (False, True))
def test_diag5_empty_pending_quarantine_uses_exact_sibling_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_fails: bool,
) -> None:
    cold = tmp_path / "empty-quarantine" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    outcome = replace(
        _diag4_complete_supervised(
            {
                "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
                "execution_status": "COMPLETE",
            }
        ),
        producer=None,
        producer_absence_reason=runner.AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
    )
    original_publish = runner._publish_canonical_json

    def publish(path: Path, payload: Mapping[str, object]) -> None:
        if marker_fails and path.name.endswith(".empty.json"):
            raise OSError("marker publication")
        original_publish(path, payload)

    monkeypatch.setattr(runner, "_publish_canonical_json", publish)

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    marker = cold / "uncommitted-numerical-result.empty.json"
    assert publication.uncommitted_root.is_dir()
    assert not tuple(publication.uncommitted_root.iterdir())
    if marker_fails:
        assert not resolution.publication_allowed
        assert resolution.pending_disposition_failure is not None
        assert (
            resolution.pending_disposition_failure.reason
            is runner.FailureReasonCodeV5.QUARANTINE_FAILED
        )
        assert not marker.exists()
    else:
        assert resolution.publication_allowed
        payload = runner.load_canonical_json_bytes(marker.read_bytes())
        assert payload == {
            "schema_version": "single-stage-neq-gntr3-empty-quarantine-v1",
            "route": runner.DIAG5_ROUTE,
            "quarantine_relative_path": "cold/uncommitted-numerical-result",
            "selected_failure_reason": "COLD_PRODUCER_INVALID",
        }
        assert marker.stat().st_nlink == 1
        assert publication.uncommitted_root.stat().st_mode & 0o777 == 0o555


def test_diag5_invalid_producer_custody_never_replaces_existing_opaque_bytes(
    tmp_path: Path,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    directory = staging / "preflight"
    directory.mkdir(parents=True)
    invalid = directory / "invalid-producer.bin"
    invalid.write_bytes(b"existing")
    publication = runner.Diag2Publication(staging, tmp_path / "campaign", "claim")
    refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS
    }
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.PROTOCOL_FAILURE,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=0.25,
        producer=None,
        producer_absence_reason=runner.AbsenceReason.PRODUCER_SCHEMA_INVALID,
        selected_failure_reason=runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
        memory=None,
        raw_failure_reasons=("invalid",),
        observed_child_argv=("child",),
        stdout=b"new-invalid",
        stderr=b"",
        memory_samples=(),
        process_diagnostics={"returncode": 0},
        process_started_monotonic_ns=100,
        process_stopped_monotonic_ns=200,
    )

    failure = runner._publish_diag5_child_supervision(
        publication,
        refs,
        directory,
        outcome,
        mode=runner.DiagnosticChildMode.PREFLIGHT,
    )

    assert failure is not None
    assert failure.reason is runner.FailureReasonCodeV5.PREFLIGHT_PROTOCOL_INVALID
    assert invalid.read_bytes() == b"existing"
    assert (directory / "producer.json").read_bytes() == b"new-invalid"
    assert all(reference is None for reference in refs.values())


def test_diag5_invalid_producer_custody_retains_descriptor_bound_bytes(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "preflight"
    directory.mkdir()
    payload = b"invalid producer bytes\x00"

    runner._retain_invalid_diag5_producer_bytes(directory, payload)

    retained = directory / "invalid-producer.bin"
    assert retained.read_bytes() == payload
    assert retained.stat(follow_symlinks=False).st_mode & 0o777 == 0o444
    assert retained.stat(follow_symlinks=False).st_nlink == 1
    assert not (directory / "producer.json").exists()


def test_diag5_invalid_producer_custody_detects_rename_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "preflight"
    directory.mkdir()
    payload = b"invalid producer bytes"
    original_rename = runner._rename_noreplace_at

    def substitute(
        source_directory_descriptor: int,
        source_name: bytes,
        destination_directory_descriptor: int,
        destination_name: bytes,
    ) -> None:
        original_rename(
            source_directory_descriptor,
            source_name,
            destination_directory_descriptor,
            destination_name,
        )
        os.rename(
            destination_name,
            b"invalid-producer.stolen",
            src_dir_fd=destination_directory_descriptor,
            dst_dir_fd=destination_directory_descriptor,
        )
        replacement = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
            dir_fd=destination_directory_descriptor,
        )
        try:
            os.write(replacement, b"replacement")
        finally:
            os.close(replacement)

    monkeypatch.setattr(runner, "_rename_noreplace_at", substitute)

    with pytest.raises(RuntimeError, match="descriptor binding differs"):
        runner._retain_invalid_diag5_producer_bytes(directory, payload)

    assert (directory / "invalid-producer.stolen").read_bytes() == payload
    assert (directory / "invalid-producer.bin").read_bytes() == b"replacement"


def test_diag5_invalid_producer_custody_detects_fstat_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "preflight"
    directory.mkdir()
    original_fstat = runner.os.fstat
    regular_observations = 0

    def drifting_fstat(descriptor: int) -> os.stat_result | SimpleNamespace:
        nonlocal regular_observations
        observed = original_fstat(descriptor)
        if stat.S_ISREG(observed.st_mode):
            regular_observations += 1
            if regular_observations == 2:
                return SimpleNamespace(
                    st_dev=observed.st_dev,
                    st_ino=observed.st_ino + 1,
                    st_size=observed.st_size,
                    st_mode=observed.st_mode,
                    st_nlink=observed.st_nlink,
                )
        return observed

    monkeypatch.setattr(runner.os, "fstat", drifting_fstat)

    with pytest.raises(RuntimeError, match="descriptor binding differs"):
        runner._retain_invalid_diag5_producer_bytes(directory, b"invalid")

    assert (directory / "invalid-producer.bin").read_bytes() == b"invalid"


def test_diag5_invalid_producer_custody_preserves_bytes_on_parent_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "preflight"
    directory.mkdir()
    original_fsync = runner.os.fsync
    calls = 0

    def fail_parent_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("parent fsync")
        original_fsync(descriptor)

    monkeypatch.setattr(runner.os, "fsync", fail_parent_fsync)

    with pytest.raises(OSError, match="parent fsync"):
        runner._retain_invalid_diag5_producer_bytes(directory, b"invalid")

    assert (directory / "invalid-producer.bin").read_bytes() == b"invalid"
    assert not (directory / "producer.json").exists()


def test_diag5_invalid_producer_custody_preserves_source_on_rename_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "preflight"
    directory.mkdir()

    def fail_rename(*_args: object) -> Never:
        raise OSError("rename failed")

    monkeypatch.setattr(runner, "_rename_noreplace_at", fail_rename)

    with pytest.raises(OSError, match="rename failed"):
        runner._retain_invalid_diag5_producer_bytes(directory, b"invalid")

    assert (directory / "producer.json").read_bytes() == b"invalid"
    assert not (directory / "invalid-producer.bin").exists()


def test_diag5_commit_collision_blocks_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = tmp_path / "collision-diag5" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    publication.committed_root.mkdir()
    outcome = _diag4_complete_supervised(
        {
            "schema_version": runner.DIAG5_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        }
    )
    monkeypatch.setattr(
        runner,
        "_validate_pending_diag5_numerical_bundle",
        lambda *_args: None,
    )

    resolution = runner._resolve_diag5_cold_numerical_bundle_v5(cold, outcome)

    assert not resolution.publication_allowed
    assert resolution.pending_disposition_failure is not None
    assert (
        resolution.pending_disposition_failure.reason
        is runner.FailureReasonCodeV5.COMMIT_COLLISION
    )
    assert publication.pending_root.is_dir()


@pytest.mark.parametrize(
    ("lifecycle", "expected_reason"),
    (
        (
            runner.Diag5AuthorityLifecycle.UNCONSUMED,
            runner.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_FAILED,
        ),
        (
            runner.Diag5AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
            runner.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN,
        ),
        (
            runner.Diag5AuthorityLifecycle.CONSUMED,
            runner.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN,
        ),
    ),
)
def test_diag5_consumption_failure_uses_authority_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: runner.Diag5AuthorityLifecycle,
    expected_reason: runner.FailureReasonCodeV5,
) -> None:
    claim = cast(runner.Diag5SuccessorAuthorityClaim, object())
    monkeypatch.setattr(
        runner,
        "diag5_authority_lifecycle",
        lambda _claim: lifecycle,
    )

    failure = runner._diag5_consumption_failure(claim, OSError("consume fault"))

    assert failure.stage is runner.FailureStageV5.BEFORE_PREFLIGHT
    assert failure.reason is expected_reason


def test_diag5_unreadable_consumption_lifecycle_is_uncertain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = cast(runner.Diag5SuccessorAuthorityClaim, object())
    monkeypatch.setattr(
        runner,
        "diag5_authority_lifecycle",
        lambda _claim: (_ for _ in ()).throw(RuntimeError("unreadable lifecycle")),
    )

    failure = runner._diag5_consumption_failure(claim, OSError("consume fault"))

    assert failure.stage is runner.FailureStageV5.BEFORE_PREFLIGHT
    assert failure.reason is runner.FailureReasonCodeV5.AUTHORITY_CONSUMPTION_UNCERTAIN


def test_diag5_physical_publication_wrapper_is_exact() -> None:
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_gpu_output_root=Path("/campaign/final"),
            expected_gpu_rollback_root=Path("/campaign/final.partial-rollback"),
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=runner.GPU_UUID,
        ),
    )
    payload = runner._diag5_physical_publication_failure_payload(
        successor_claim=claim,
        original_reason=runner.Diag5PhysicalPublicationReason.FINAL_FSYNC_FAILED,
        observation=runner.Diag5PhysicalPublicationObservation(
            rollback_cause=runner.Diag5RollbackCause.NONE,
            rollback_state=runner.Diag5RollbackState.SUCCEEDED,
            final_path_state=runner.Diag5PhysicalPathState.ABSENT,
            rollback_path_state=runner.Diag5PhysicalPathState.VISIBLE_VALIDATED,
            evidence_namespace_state_at_seal=(
                runner.Diag5EvidenceNamespaceState.PENDING_BOUND
            ),
        ),
        sealed_artifact_manifest_sha256="b" * 64,
    )

    assert tuple(sorted(payload)) == tuple(
        sorted(
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
        )
    )
    assert payload["schema_version"] == (
        "single-stage-neq-gntr3-diag5-physical-publication-failure-v1"
    )
    assert payload["rollback_state"] == "SUCCEEDED"
    assert payload["rollback_cause"] == "NONE"


def test_diag5_physical_publication_wrapper_rejects_mismatched_state() -> None:
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_gpu_output_root=Path("/campaign/final"),
            expected_gpu_rollback_root=Path("/campaign/final.partial-rollback"),
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=runner.GPU_UUID,
        ),
    )

    with pytest.raises(ValueError, match="state and cause"):
        runner._diag5_physical_publication_failure_payload(
            successor_claim=claim,
            original_reason=(
                runner.Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
            ),
            observation=runner.Diag5PhysicalPublicationObservation(
                rollback_cause=runner.Diag5RollbackCause.ROLLBACK_RENAME_FAILED,
                rollback_state=runner.Diag5RollbackState.SUCCEEDED,
                final_path_state=runner.Diag5PhysicalPathState.VISIBLE_VALIDATED,
                rollback_path_state=runner.Diag5PhysicalPathState.ABSENT,
                evidence_namespace_state_at_seal=(
                    runner.Diag5EvidenceNamespaceState.PENDING_BOUND
                ),
            ),
            sealed_artifact_manifest_sha256="b" * 64,
        )


def test_diag5_post_final_failure_uses_authority_rollback_and_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_gpu_output_root=Path("/campaign/final"),
            expected_gpu_rollback_root=Path("/campaign/final.partial-rollback"),
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=runner.GPU_UUID,
            predecessor_postmortem=runner.ArtifactRef(
                "control/predecessor-postmortem.json",
                "c" * 64,
                1,
                "postmortem-v1",
            ),
            cpu_native_binding=SimpleNamespace(
                path=Path("/cpu/simsoptpp.so"),
                sha256="d" * 64,
                size_bytes=1,
                link_count=1,
                device=1,
                inode=2,
            ),
            gpu_native_binding=SimpleNamespace(
                path=Path("/gpu/simsoptpp.so"),
                sha256="d" * 64,
                size_bytes=1,
                link_count=1,
                device=3,
                inode=4,
            ),
        ),
    )
    reservation = cast(runner.Diag5PhysicalEvidenceReservation, object())
    deep_load_calls: list[Path] = []
    published_payloads: list[Mapping[str, object]] = []

    def rollback(
        _claim: object,
        _reservation: object,
        *,
        deep_load: Callable[[Path], object],
    ) -> object:
        rollback_root = Path("/campaign/final.partial-rollback")
        deep_load_calls.append(rollback_root)
        deep_load(rollback_root)
        return SimpleNamespace(
            rollback_cause=SimpleNamespace(value="NONE"),
            rollback_state=SimpleNamespace(value="SUCCEEDED"),
            final_path_state=SimpleNamespace(value="ABSENT"),
            rollback_path_state=SimpleNamespace(value="VISIBLE_VALIDATED"),
            evidence_namespace_state_at_seal=SimpleNamespace(value="PENDING_BOUND"),
        )

    monkeypatch.setattr(runner, "rollback_diag5_bound_final", rollback)
    monkeypatch.setattr(
        runner,
        "load_and_validate_diag5_rollback",
        lambda root, **_kwargs: root,
    )
    monkeypatch.setattr(
        runner,
        "publish_diag5_physical_failure_evidence",
        lambda _claim, _reservation, payload: (
            published_payloads.append(payload),
            Path("/campaign/failure.json"),
        )[1],
    )

    with pytest.raises(runner.Diag5PhysicalPublicationError) as captured:
        runner._raise_diag5_post_final_failure(
            successor_claim=claim,
            reservation=reservation,
            reason=runner.Diag5PhysicalPublicationReason.FINAL_FSYNC_FAILED,
            sealed_artifact_manifest_sha256="e" * 64,
            cause=OSError("fsync"),
            expected_source_snapshot=cast(runner.SnapshotIdentity, object()),
            physical_memory_bytes=1,
        )

    assert (
        captured.value.reason
        is runner.Diag5PhysicalPublicationReason.FINAL_FSYNC_FAILED
    )
    assert captured.value.evidence_path == Path("/campaign/failure.json")
    assert deep_load_calls == [Path("/campaign/final.partial-rollback")]
    assert len(published_payloads) == 1
    assert published_payloads[0]["original_reason"] == "FINAL_FSYNC_FAILED"


@pytest.mark.parametrize("reason", tuple(runner.Diag5PhysicalPublicationReason))
@pytest.mark.parametrize(
    ("cause", "state", "final_state", "rollback_state"),
    (
        ("NONE", "SUCCEEDED", "ABSENT", "VISIBLE_VALIDATED"),
        ("ROLLBACK_COLLISION", "FAILED", "VISIBLE_INVALID", "VISIBLE_INVALID"),
        ("ROLLBACK_RENAME_FAILED", "FAILED", "VISIBLE_INVALID", "ABSENT"),
        ("ROLLBACK_PARENT_FSYNC_FAILED", "FAILED", "ABSENT", "VISIBLE_INVALID"),
        ("ROLLBACK_DEEP_LOAD_FAILED", "FAILED", "ABSENT", "VISIBLE_INVALID"),
        (
            "ROLLBACK_VISIBILITY_AMBIGUOUS",
            "AMBIGUOUS",
            "VISIBILITY_AMBIGUOUS",
            "VISIBILITY_AMBIGUOUS",
        ),
    ),
)
@pytest.mark.parametrize(
    "namespace_state",
    ("PENDING_BOUND", "PENDING_UNLINKED", "PENDING_AMBIGUOUS"),
)
def test_diag5_physical_wrapper_covers_reason_and_rollback_state_matrix(
    reason: runner.Diag5PhysicalPublicationReason,
    cause: str,
    state: str,
    final_state: str,
    rollback_state: str,
    namespace_state: str,
) -> None:
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_gpu_output_root=Path("/campaign/final"),
            expected_gpu_rollback_root=Path("/campaign/final.partial-rollback"),
        ),
    )
    payload = runner._diag5_physical_publication_failure_payload(
        successor_claim=claim,
        original_reason=reason,
        observation=runner.Diag5PhysicalPublicationObservation(
            rollback_cause=runner.Diag5RollbackCause(cause),
            rollback_state=runner.Diag5RollbackState(state),
            final_path_state=runner.Diag5PhysicalPathState(final_state),
            rollback_path_state=runner.Diag5PhysicalPathState(rollback_state),
            evidence_namespace_state_at_seal=(
                runner.Diag5EvidenceNamespaceState(namespace_state)
            ),
        ),
        sealed_artifact_manifest_sha256="b" * 64,
    )

    assert payload["original_reason"] == reason.value
    assert payload["rollback_cause"] == cause
    assert payload["rollback_state"] == state
    assert len(payload) == 12


def test_diag5_wrapper_publication_fault_is_typed_without_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_gpu_output_root=Path("/campaign/final"),
            expected_gpu_rollback_root=Path("/campaign/final.partial-rollback"),
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=runner.GPU_UUID,
            predecessor_postmortem=runner.ArtifactRef(
                "control/predecessor-postmortem.json", "c" * 64, 1, "postmortem-v1"
            ),
            cpu_native_binding=SimpleNamespace(
                path=Path("/cpu.so"),
                sha256="d" * 64,
                size_bytes=1,
                link_count=1,
                device=1,
                inode=2,
            ),
            gpu_native_binding=SimpleNamespace(
                path=Path("/gpu.so"),
                sha256="d" * 64,
                size_bytes=1,
                link_count=1,
                device=3,
                inode=4,
            ),
        ),
    )
    reservation = cast(runner.Diag5PhysicalEvidenceReservation, object())
    publish_calls = 0
    monkeypatch.setattr(
        runner,
        "rollback_diag5_bound_final",
        lambda *_args, **_kwargs: SimpleNamespace(
            rollback_cause=SimpleNamespace(value="NONE"),
            rollback_state=SimpleNamespace(value="SUCCEEDED"),
            final_path_state=SimpleNamespace(value="ABSENT"),
            rollback_path_state=SimpleNamespace(value="VISIBLE_VALIDATED"),
            evidence_namespace_state_at_seal=SimpleNamespace(value="PENDING_BOUND"),
        ),
    )

    def fail_publish(*_args: object) -> None:
        nonlocal publish_calls
        publish_calls += 1
        raise OSError("wrapper fsync")

    monkeypatch.setattr(runner, "publish_diag5_physical_failure_evidence", fail_publish)

    with pytest.raises(runner.Diag5PhysicalPublicationError) as captured:
        runner._raise_diag5_post_final_failure(
            successor_claim=claim,
            reservation=reservation,
            reason=runner.Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
            sealed_artifact_manifest_sha256="e" * 64,
            cause=ValueError("deep load"),
            expected_source_snapshot=cast(runner.SnapshotIdentity, object()),
            physical_memory_bytes=1,
        )

    assert publish_calls == 1
    assert captured.value.evidence_path is None
    assert (
        captured.value.reason
        is runner.Diag5PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
    )


@pytest.mark.parametrize(
    ("boundary", "expected_stage", "expected_reason"),
    (
        (
            "receipt",
            runner.FailureStageV5.RECEIPT,
            runner.FailureReasonCodeV5.RECEIPT_SCHEMA_INVALID,
        ),
        (
            "manifest",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.MANIFEST_INVALID,
        ),
        (
            "writable",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.MODE_OR_LINK_INVALID,
        ),
        (
            "seal",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.MODE_OR_LINK_INVALID,
        ),
        (
            "deep-load",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.STAGING_DEEP_LOAD_FAILED,
        ),
        (
            "source",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.STAGING_DEEP_LOAD_FAILED,
        ),
        (
            "reservation",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.FINAL_RENAME_FAILED,
        ),
        (
            "collision",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.FINAL_COLLISION,
        ),
        (
            "rename",
            runner.FailureStageV5.PUBLICATION,
            runner.FailureReasonCodeV5.FINAL_RENAME_FAILED,
        ),
    ),
)
def test_diag5_pre_final_boundaries_converge_to_typed_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_stage: runner.FailureStageV5,
    expected_reason: runner.FailureReasonCodeV5,
) -> None:
    staging = tmp_path / "campaign.partial-claim"
    staging.mkdir()
    publication = runner.Diag2Publication(staging, tmp_path / "campaign", "claim")
    binding = SimpleNamespace(
        path=Path("/simsoptpp.so"),
        sha256="d" * 64,
        size_bytes=1,
        link_count=1,
        device=1,
        inode=2,
    )
    claim = cast(
        runner.Diag5SuccessorAuthorityClaim,
        SimpleNamespace(
            authority_sha256="a" * 64,
            expected_frozen_numerical_entries={},
            expected_gpu_uuid=runner.GPU_UUID,
            predecessor_postmortem=runner.ArtifactRef(
                "control/predecessor-postmortem.json", "c" * 64, 1, "postmortem-v1"
            ),
            cpu_native_binding=binding,
            gpu_native_binding=binding,
        ),
    )
    outcome = runner._diag5_failure(
        runner.FailureStageV5.SETUP,
        runner.FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
        "prior",
    )
    refs = {name: None for name in runner.DIAG5_EVIDENCE_SLOT_PATHS}
    cancel_calls = 0

    def cancel(_claim: object, _reservation: object) -> None:
        nonlocal cancel_calls
        cancel_calls += 1

    monkeypatch.setattr(
        runner,
        "build_diag5_supervisor_terminal_payload",
        lambda **_kwargs: {"terminal": True},
    )
    monkeypatch.setattr(runner, "derive_diag5_evidence_slots", lambda **_kwargs: {})
    monkeypatch.setattr(
        runner, "build_diag5_diagnostic_receipt", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        runner, "diag5_diagnostic_receipt_bytes", lambda _value: b"{}\n"
    )
    monkeypatch.setattr(runner, "diag5_artifact_manifest_payload", lambda _root: {})
    monkeypatch.setattr(
        runner, "validate_diag5_writable_staging", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        runner, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    monkeypatch.setattr(runner, "_seal_and_sync_diag2_staging", lambda _root: None)
    monkeypatch.setattr(
        runner, "load_and_validate_diag5_staging", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        runner, "validate_diag5_successor_snapshot", lambda *_args: None
    )
    monkeypatch.setattr(
        runner, "prepare_diag5_physical_failure_evidence", lambda _claim: object()
    )
    monkeypatch.setattr(
        runner, "publish_diag5_bound_staging", lambda *_args: Path("/final")
    )
    monkeypatch.setattr(runner, "cancel_diag5_physical_failure_evidence", cancel)
    target = {
        "receipt": "build_diag5_diagnostic_receipt",
        "manifest": "diag5_artifact_manifest_payload",
        "writable": "validate_diag5_writable_staging",
        "seal": "_seal_and_sync_diag2_staging",
        "deep-load": "load_and_validate_diag5_staging",
        "source": "validate_diag5_successor_snapshot",
        "reservation": "prepare_diag5_physical_failure_evidence",
        "collision": "publish_diag5_bound_staging",
        "rename": "publish_diag5_bound_staging",
    }[boundary]
    injected: BaseException = (
        FileExistsError("collision") if boundary == "collision" else OSError(boundary)
    )
    monkeypatch.setattr(
        runner,
        target,
        lambda *_args, **_kwargs: (_ for _ in ()).throw(injected),
    )

    with pytest.raises(runner.Diag5PreFinalPublicationError) as captured:
        runner._publish_diag5_terminal_and_receipt(
            publication,
            refs,
            outcome=outcome,
            launched_children=(),
            successor_claim=claim,
            source=(
                cast(runner.SnapshotPublication, object())
                if boundary == "source"
                else None
            ),
            expected_source_snapshot=cast(runner.SnapshotIdentity, object()),
            physical_memory_bytes=1,
        )

    assert captured.value.terminal_outcome is outcome
    assert captured.value.publication_failure.stage is expected_stage
    assert captured.value.publication_failure.reason is expected_reason
    assert captured.value.staging_root == staging
    assert not publication.final_root.exists()
    assert cancel_calls == (1 if boundary in {"collision", "rename"} else 0)


@pytest.mark.parametrize("cancel_fails", (False, True))
def test_diag5_failed_final_rename_cancels_reserved_evidence_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cancel_fails: bool,
) -> None:
    publication = runner.Diag2Publication(
        tmp_path / "campaign.partial-claim",
        tmp_path / "campaign",
        "claim",
    )
    publication.staging_root.mkdir()
    outcome = runner._diag5_failure(
        runner.FailureStageV5.SETUP,
        runner.FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
        "prior",
    )
    claim = cast(runner.Diag5SuccessorAuthorityClaim, object())
    reservation = cast(runner.Diag5PhysicalEvidenceReservation, object())
    cancel_calls = 0

    cancellation = successor_authority.Diag5PhysicalCancellationObservation(
        successor_authority.Diag5PhysicalCancellationCause.NONE,
        successor_authority.Diag5PhysicalCancellationState.CANCELLED,
        successor_authority.Diag5EvidenceNamespaceState.PENDING_UNLINKED,
        successor_authority.Diag5PhysicalPathState.VISIBLE_VALIDATED,
        successor_authority.Diag5PhysicalPathState.ABSENT,
        successor_authority.Diag5PhysicalPathState.ABSENT,
    )
    spent_cancellation = replace(
        cancellation,
        cause=successor_authority.Diag5PhysicalCancellationCause.CANCEL_PARENT_FSYNC_FAILED,
        state=successor_authority.Diag5PhysicalCancellationState.SPENT,
        evidence_namespace_state=successor_authority.Diag5EvidenceNamespaceState.PENDING_UNLINKED,
    )

    def cancel(
        _claim: object, _reservation: object
    ) -> successor_authority.Diag5PhysicalCancellationObservation:
        nonlocal cancel_calls
        cancel_calls += 1
        if cancel_fails:
            raise successor_authority.Diag5PhysicalCancellationError(
                spent_cancellation, OSError("cancel fsync")
            )
        return cancellation

    monkeypatch.setattr(
        runner,
        "cancel_diag5_physical_failure_evidence",
        cancel,
    )
    original = FileExistsError("final collision")

    with pytest.raises(runner.Diag5PreFinalPublicationError) as captured:
        runner._raise_diag5_pre_final_after_reservation(
            publication=publication,
            terminal_outcome=outcome,
            successor_claim=claim,
            reservation=reservation,
            reason=runner.FailureReasonCodeV5.FINAL_COLLISION,
            cause=original,
        )

    assert cancel_calls == 1
    assert captured.value.cause is original
    assert (
        captured.value.publication_failure.reason
        is runner.FailureReasonCodeV5.FINAL_COLLISION
    )
    assert captured.value.cancellation_observation == (
        spent_cancellation if cancel_fails else cancellation
    )
    assert captured.value.cancellation_observation.staging_path_state is (
        successor_authority.Diag5PhysicalPathState.VISIBLE_VALIDATED
    )
    assert captured.value.cancellation_observation.final_path_state is (
        successor_authority.Diag5PhysicalPathState.ABSENT
    )
    assert captured.value.cancellation_observation.rollback_path_state is (
        successor_authority.Diag5PhysicalPathState.ABSENT
    )
    assert (captured.value.cleanup_cause is not None) is cancel_fails


def test_diag4_atomic_rename_collision_and_fsync_fault_remain_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision_source = tmp_path / "collision-source"
    collision_destination = tmp_path / "collision-destination"
    collision_source.mkdir()
    collision_destination.mkdir()
    (collision_source / "source.bin").write_bytes(b"source")
    (collision_destination / "destination.bin").write_bytes(b"destination")

    with pytest.raises(FileExistsError):
        runner._rename_noreplace_and_fsync_parent(
            collision_source, collision_destination
        )
    assert (collision_source / "source.bin").read_bytes() == b"source"
    assert (collision_destination / "destination.bin").read_bytes() == b"destination"

    fsync_source = tmp_path / "fsync-source"
    fsync_destination = tmp_path / "fsync-destination"
    fsync_source.mkdir()
    (fsync_source / "result.bin").write_bytes(b"result")
    monkeypatch.setattr(
        runner.os,
        "fsync",
        lambda _descriptor: (_ for _ in ()).throw(OSError("fsync fault")),
    )

    with pytest.raises(OSError, match="fsync fault"):
        runner._rename_noreplace_and_fsync_parent(fsync_source, fsync_destination)
    assert not fsync_source.exists()
    assert (fsync_destination / "result.bin").read_bytes() == b"result"


def test_diag4_final_rollback_rejects_rebound_final_inode(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    publication = runner._prepare_diag2_publication(
        tmp_path / "diag4-final",
        repository_root=repository,
    )
    runner._rename_noreplace(publication.staging_root, publication.final_root)
    displaced = tmp_path / "displaced-original"
    os.replace(publication.final_root, displaced)
    publication.final_root.mkdir()

    with pytest.raises(RuntimeError, match="final inode differs"):
        runner._rollback_diag4_final(publication)

    assert publication.final_root.is_dir()
    assert displaced.is_dir()
    assert not publication.staging_root.exists()


@pytest.mark.parametrize(
    "lifecycle",
    [
        runner.Diag4AuthorityLifecycle.STAGING_BOUND,
        runner.Diag4AuthorityLifecycle.CONSUMED,
        runner.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
    ],
)
@pytest.mark.parametrize(
    "final_fault",
    [
        None,
        "group-prefix",
        "receipt-schema",
        "manifest",
        "mode",
        "staging-load",
        "collision",
        "rename",
        "terminal-write",
        "receipt-write",
        "manifest-write",
        "seal",
        "fsync",
        "deep-load",
        "rollback-rename",
        "rollback-fsync",
        "authority",
    ],
)
def test_diag4_final_publisher_seals_reloads_and_atomically_installs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: runner.Diag4AuthorityLifecycle,
    final_fault: str | None,
) -> None:
    final_root = tmp_path / "diag4-final"
    publication = runner._prepare_diag2_publication(
        final_root,
        repository_root=Path(__file__).resolve().parents[2],
    )
    artifact_refs = {name: None for name in runner.DIAG4_EVIDENCE_SLOT_NAMES}
    outcome = runner._diag4_failure(
        runner.FailureStageV4.AUTHORITY,
        runner.FailureReasonCodeV4.AUTHORITY_INVALID,
        "test",
    )
    launched_children: tuple[str, ...] = ()
    if final_fault is not None:
        outcome = runner._diag4_failure(
            runner.FailureStageV4.SCIENTIFIC,
            runner.FailureReasonCodeV4.NO_HIT,
            "scientific no-hit",
        )
        launched_children = ("preflight", "cold")
    receipt = object()
    successor_claim = object()
    calls: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        runner,
        "derive_diag4_evidence_slots",
        lambda **_kwargs: {"slots": object()},
    )
    monkeypatch.setattr(
        runner, "build_diag4_diagnostic_receipt", lambda **_kwargs: receipt
    )
    monkeypatch.setattr(
        runner, "diag4_diagnostic_receipt_bytes", lambda _receipt: b"{}\n"
    )
    monkeypatch.setattr(
        runner,
        "diag4_artifact_manifest_payload",
        lambda _root: {"schema_version": "manifest-v1", "entries": []},
    )
    monkeypatch.setattr(
        runner,
        "validate_diag4_writable_staging",
        lambda root: calls.append(("writable", root)) or receipt,
    )
    monkeypatch.setattr(
        runner,
        "load_and_validate_diag4_staging",
        lambda root: calls.append(("staging", root)) or receipt,
    )
    monkeypatch.setattr(
        runner,
        "load_and_validate_diag4_artifact",
        lambda root: (
            (_ for _ in ()).throw(ValueError("final deep-load fault"))
            if final_fault in {"deep-load", "rollback-rename", "rollback-fsync"}
            else calls.append(("final", root)) or receipt
        ),
    )
    if final_fault == "fsync":
        original_fsync_parent = runner._fsync_parent
        monkeypatch.setattr(
            runner,
            "_fsync_parent",
            lambda path: (
                (_ for _ in ()).throw(OSError("final fsync fault"))
                if path == publication.final_root
                else original_fsync_parent(path)
            ),
        )
    elif final_fault == "rollback-fsync":
        original_fsync_parent = runner._fsync_parent

        def fsync_parent_with_rollback_fault(path: Path) -> None:
            if path == publication.staging_root:
                raise OSError("rollback fsync fault")
            original_fsync_parent(path)

        monkeypatch.setattr(
            runner,
            "_fsync_parent",
            fsync_parent_with_rollback_fault,
        )

    def finalize_authority(claim: object) -> None:
        assert claim is successor_claim
        calls.append(("finalize", publication.final_root))
        if final_fault == "authority":
            raise RuntimeError("authority finalization fault")

    monkeypatch.setattr(
        runner,
        "finalize_diag4_prelaunch_failure",
        finalize_authority,
    )

    def revalidate_authority(claim: object, *, require_output_absent: bool) -> None:
        assert claim is successor_claim
        assert not require_output_absent
        calls.append(("revalidate", publication.final_root))
        if final_fault == "authority":
            raise RuntimeError("authority revalidation fault")

    monkeypatch.setattr(
        runner,
        "revalidate_diag4_successor_authority",
        revalidate_authority,
    )
    monkeypatch.setattr(
        runner,
        "diag4_authority_lifecycle",
        lambda claim: lifecycle if claim is successor_claim else None,
    )
    if final_fault == "group-prefix":
        monkeypatch.setattr(
            runner,
            "derive_diag4_evidence_slots",
            lambda **_kwargs: (_ for _ in ()).throw(ValueError("prefix fault")),
        )
    elif final_fault == "receipt-schema":
        monkeypatch.setattr(
            runner,
            "build_diag4_diagnostic_receipt",
            lambda **_kwargs: (_ for _ in ()).throw(ValueError("receipt fault")),
        )
    elif final_fault == "manifest":
        monkeypatch.setattr(
            runner,
            "diag4_artifact_manifest_payload",
            lambda _root: (_ for _ in ()).throw(ValueError("manifest fault")),
        )
    elif final_fault == "mode":
        monkeypatch.setattr(
            runner,
            "validate_diag4_writable_staging",
            lambda _root: (_ for _ in ()).throw(ValueError("mode fault")),
        )
    elif final_fault == "staging-load":
        monkeypatch.setattr(
            runner,
            "load_and_validate_diag4_staging",
            lambda _root: (_ for _ in ()).throw(ValueError("staging fault")),
        )
    elif final_fault in {"collision", "rename"}:
        monkeypatch.setattr(
            runner,
            "_rename_noreplace",
            lambda _source, _destination: (_ for _ in ()).throw(
                FileExistsError("collision fault")
                if final_fault == "collision"
                else OSError("rename fault")
            ),
        )
    elif final_fault == "rollback-rename":
        original_rename_noreplace = runner._rename_noreplace
        rename_calls = 0

        def rename_with_rollback_fault(source: Path, destination: Path) -> None:
            nonlocal rename_calls
            rename_calls += 1
            if rename_calls == 2:
                raise OSError("rollback rename fault")
            original_rename_noreplace(source, destination)

        monkeypatch.setattr(
            runner,
            "_rename_noreplace",
            rename_with_rollback_fault,
        )
    elif final_fault == "terminal-write":
        monkeypatch.setattr(
            runner,
            "_publish_canonical_json",
            lambda _path, _payload: (_ for _ in ()).throw(
                OSError("terminal write fault")
            ),
        )
    elif final_fault == "receipt-write":
        monkeypatch.setattr(
            runner,
            "_publish_bytes",
            lambda _path, _payload: (_ for _ in ()).throw(
                OSError("receipt write fault")
            ),
        )
    elif final_fault == "manifest-write":
        original_publish_canonical_json = runner._publish_canonical_json

        def publish_canonical_json_with_manifest_fault(
            path: Path, payload: object
        ) -> None:
            if path.name == runner.DIAG2_MANIFEST_FILENAME:
                raise OSError("manifest write fault")
            original_publish_canonical_json(path, payload)

        monkeypatch.setattr(
            runner,
            "_publish_canonical_json",
            publish_canonical_json_with_manifest_fault,
        )
    elif final_fault == "seal":
        monkeypatch.setattr(
            runner,
            "_seal_and_sync_diag2_staging",
            lambda _root: (_ for _ in ()).throw(OSError("seal fault")),
        )

    if final_fault is not None:
        hard_reason = {
            "terminal-write": runner.FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            "receipt-write": runner.FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            "manifest-write": runner.FailureReasonCodeV4.MANIFEST_INVALID,
            "seal": runner.FailureReasonCodeV4.MODE_OR_LINK_INVALID,
            "fsync": runner.Diag4PhysicalPublicationReason.FINAL_FSYNC_FAILED,
            "deep-load": runner.Diag4PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED,
            "rollback-rename": (
                runner.Diag4PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
            ),
            "rollback-fsync": (
                runner.Diag4PhysicalPublicationReason.FINAL_DEEP_LOAD_FAILED
            ),
            "authority": (
                runner.Diag4PhysicalPublicationReason.POST_FINAL_AUTHORITY_FINALIZATION_FAILED
                if lifecycle is runner.Diag4AuthorityLifecycle.STAGING_BOUND
                else runner.Diag4PhysicalPublicationReason.POST_FINAL_AUTHORITY_REVALIDATION_FAILED
            ),
        }.get(final_fault)
        if hard_reason is not None:
            with pytest.raises(runner.Diag4HardPublicationError) as caught:
                runner._publish_diag4_terminal_and_receipt(
                    publication,
                    artifact_refs,
                    outcome=outcome,
                    launched_children=launched_children,
                    successor_claim=successor_claim,
                )
            assert caught.value.reason is hard_reason
            rollback_failed = final_fault in {"rollback-rename", "rollback-fsync"}
            assert isinstance(
                caught.value,
                (
                    runner.Diag4RollbackHardPublicationError
                    if rollback_failed
                    else runner.Diag4HardPublicationError
                ),
            )
            if rollback_failed:
                rollback_error = caught.value
                assert isinstance(
                    rollback_error, runner.Diag4RollbackHardPublicationError
                )
                assert rollback_error.authority_lifecycle is lifecycle
                assert isinstance(rollback_error.cause, ValueError)
                assert isinstance(rollback_error.rollback_cause, OSError)
                assert rollback_error.staging_exists is (
                    final_fault == "rollback-fsync"
                )
                assert rollback_error.final_exists is (final_fault == "rollback-rename")
                assert rollback_error.root == (
                    publication.final_root
                    if final_fault == "rollback-rename"
                    else publication.staging_root
                )
            else:
                assert caught.value.root == publication.staging_root
                assert publication.staging_root.is_dir()
                assert not publication.final_root.exists()
            authority_calls = [
                call_name
                for call_name, _path in calls
                if call_name in {"finalize", "revalidate"}
            ]
            assert authority_calls == (
                [
                    "finalize"
                    if lifecycle is runner.Diag4AuthorityLifecycle.STAGING_BOUND
                    else "revalidate"
                ]
                if final_fault == "authority"
                else []
            )
            terminal_path = caught.value.root / "supervisor-terminal.json"
            if final_fault == "terminal-write":
                assert not terminal_path.exists()
            else:
                terminal = runner.load_canonical_json_bytes(terminal_path.read_bytes())
                assert (
                    terminal["terminal_outcome"]["reason"]["code"]
                    == runner.FailureReasonCodeV4.NO_HIT.value
                )
            return
        partial = runner._publish_diag4_terminal_and_receipt(
            publication,
            artifact_refs,
            outcome=outcome,
            launched_children=launched_children,
            successor_claim=successor_claim,
        )
        assert isinstance(partial, runner.Diag4VisiblePartial)
        expected_reason = {
            "group-prefix": runner.FailureReasonCodeV4.GROUP_PREFIX_INVALID,
            "receipt-schema": runner.FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
            "manifest": runner.FailureReasonCodeV4.MANIFEST_INVALID,
            "mode": runner.FailureReasonCodeV4.MODE_OR_LINK_INVALID,
            "staging-load": runner.FailureReasonCodeV4.STAGING_DEEP_LOAD_FAILED,
            "collision": runner.FailureReasonCodeV4.FINAL_COLLISION,
            "rename": runner.FailureReasonCodeV4.FINAL_RENAME_FAILED,
        }[final_fault]
        assert partial.outcome.reason is expected_reason
        assert publication.staging_root.is_dir()
        assert not publication.final_root.exists()
        terminal = runner.load_canonical_json_bytes(
            (publication.staging_root / "supervisor-terminal.json").read_bytes()
        )
        assert (
            terminal["terminal_outcome"]["reason"]["code"]
            == partial.outcome.reason.value
        )
        return

    published = runner._publish_diag4_terminal_and_receipt(
        publication,
        artifact_refs,
        outcome=outcome,
        launched_children=launched_children,
        successor_claim=successor_claim,
    )

    assert published is receipt
    assert calls == [
        ("writable", publication.staging_root),
        ("staging", publication.staging_root),
        ("final", publication.final_root),
        (
            (
                "finalize"
                if lifecycle is runner.Diag4AuthorityLifecycle.STAGING_BOUND
                else "revalidate"
            ),
            publication.final_root,
        ),
    ]
    assert not publication.staging_root.exists()
    assert publication.final_root.is_dir()
    assert (publication.final_root.stat().st_mode & 0o777) == 0o555
    assert (
        publication.final_root / "supervisor-terminal.json"
    ).stat().st_mode & 0o777 == 0o444


@pytest.mark.parametrize(
    "reason",
    [
        runner.FailureReasonCodeV4.EVIDENCE_VECTOR_INVALID,
        runner.FailureReasonCodeV4.GROUP_PREFIX_INVALID,
        runner.FailureReasonCodeV4.SCIENTIFIC_RECONSTRUCTION_FAILED,
        runner.FailureReasonCodeV4.RECEIPT_SCHEMA_INVALID,
    ],
)
def test_diag4_run_receipt_failure_factory_accepts_only_exact_reasons(
    reason: runner.FailureReasonCodeV4,
) -> None:
    failure = runner._diag4_receipt_failure(reason, ValueError("receipt fault"))

    assert failure.stage is runner.FailureStageV4.RECEIPT
    assert failure.reason is reason
    assert len(failure.detail_sha256) == 64


def test_diag4_run_receipt_failure_factory_rejects_nonreceipt_reason() -> None:
    with pytest.raises(ValueError, match="receipt failure reason differs"):
        runner._diag4_receipt_failure(
            runner.FailureReasonCodeV4.MANIFEST_INVALID,
            ValueError("publication fault"),
        )


@pytest.mark.parametrize(
    ("lifecycle", "expected"),
    [
        (
            runner.Diag4AuthorityLifecycle.STAGING_BOUND,
            "AUTHORITY_CONSUMPTION_FAILED",
        ),
        (
            runner.Diag4AuthorityLifecycle.CONSUMPTION_UNCERTAIN,
            "AUTHORITY_CONSUMPTION_UNCERTAIN",
        ),
        (
            runner.Diag4AuthorityLifecycle.CONSUMED,
            "AUTHORITY_CONSUMPTION_UNCERTAIN",
        ),
    ],
)
def test_diag4_consume_exception_reason_uses_authority_owned_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: runner.Diag4AuthorityLifecycle,
    expected: str,
) -> None:
    claim = object()
    monkeypatch.setattr(
        runner,
        "diag4_authority_lifecycle",
        lambda observed: lifecycle if observed is claim else None,
    )

    assert runner._diag4_authority_consumption_failure_reason(claim).value == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            runner.Diag4ConsumptionMarkerInvalidError("marker replaced"),
            runner.FailureReasonCodeV4.CONSUMPTION_MARKER_INVALID,
        ),
        (
            ValueError("qualified identity changed"),
            runner.FailureReasonCodeV4.IDENTITY_REVALIDATION_FAILED,
        ),
    ],
)
def test_diag4_before_cold_authority_failure_reason_preserves_typed_precedence(
    error: ValueError,
    expected: runner.FailureReasonCodeV4,
) -> None:
    assert runner._diag4_before_cold_authority_failure_reason(error) is expected


def test_run_diag4_consumes_immediately_before_only_preflight_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "campaigns"
    output_parent.mkdir()
    final_root = output_parent / "diag4"
    reference_root = tmp_path / "reference"
    reference_root.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_bytes(b"python")
    native_extension = tmp_path / "simsoptpp.so"
    native_extension.write_bytes(b"native")
    claim = SimpleNamespace(
        expected_numerical_identity={"identity": "a" * 64},
        expected_frozen_numerical_entries={"src/frozen.py": "b" * 64},
    )
    events: list[str] = []

    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setattr(runner, "read_linux_process_identity", lambda _pid: object())
    monkeypatch.setattr(runner, "supervisor_query_executable_sha256", lambda: "c" * 64)
    monkeypatch.setattr(
        runner,
        "_validate_parent_execution_policy",
        lambda **_kwargs: (
            repository,
            native_extension,
            reference_root,
            input_root,
            interpreter,
            "GPU-test",
            1024,
        ),
    )
    monkeypatch.setattr(
        runner,
        "revalidate_diag4_successor_authority",
        lambda _claim, *, require_output_absent: events.append(
            f"revalidate:{require_output_absent}"
        ),
    )
    monkeypatch.setattr(
        runner,
        "bind_diag4_staging_root",
        lambda _claim, _root: events.append("bind"),
    )

    def prepare_snapshot(staging_root: Path, **_kwargs: object) -> SimpleNamespace:
        manifest = staging_root / "source-manifest.json"
        runner._publish_canonical_json(manifest, {"source": "test"})
        reference = runner._artifact_ref(manifest, staging_root, "source-v1")
        return SimpleNamespace(
            source_identity=lambda _root: SimpleNamespace(snapshot_manifest=reference)
        )

    monkeypatch.setattr(runner, "_prepare_diag4_snapshot", prepare_snapshot)
    monkeypatch.setattr(
        runner,
        "validate_diag4_successor_snapshot",
        lambda _source, _claim: events.append("snapshot"),
    )
    monkeypatch.setattr(
        runner,
        "build_diag4_frozen_numerical_subset_payload",
        lambda _entries: {"schema_version": runner.DIAG2_FROZEN_SUBSET_SCHEMA_VERSION},
    )
    monkeypatch.setattr(
        runner,
        "validate_diag4_frozen_numerical_subset_payload",
        lambda *_args, **_kwargs: None,
    )

    def copy_reference(_source: Path, staging_root: Path) -> Path:
        copied = staging_root / "reference"
        copied.mkdir()
        runner._publish_canonical_json(
            copied / runner.REFERENCE_FILENAME, {"reference": "test"}
        )
        return copied

    monkeypatch.setattr(runner, "copy_validated_reference", copy_reference)

    def publish_policy(
        staging_root: Path,
        _reference_root: Path,
        _reference: runner.ArtifactRef,
    ) -> tuple[runner.ArtifactRef, dict[str, float]]:
        path = staging_root / "policy-authority.json"
        runner._publish_canonical_json(
            path,
            {"schema_version": runner.DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION},
        )
        return (
            runner._artifact_ref(
                path, staging_root, runner.DIAG2_POLICY_AUTHORITY_SCHEMA_VERSION
            ),
            {},
        )

    monkeypatch.setattr(runner, "_publish_diag2_policy_authority", publish_policy)
    monkeypatch.setattr(
        runner,
        "validate_diag2_policy_authority_payload",
        lambda *_args, **_kwargs: None,
    )
    zero = SimpleNamespace(matching_rows=(), gate_passes=True)
    monkeypatch.setattr(
        runner,
        "_capture_diag2_supervisor_zero",
        lambda *_args, **_kwargs: zero,
    )

    def publish_zero(
        staging_root: Path, _zero: object, *, stage: str
    ) -> runner.ArtifactRef:
        path = staging_root / f"supervisor-{stage.lower()}.json"
        runner._publish_canonical_json(
            path,
            {
                "schema_version": runner.DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION,
                "stage": stage,
            },
        )
        return runner._artifact_ref(
            path, staging_root, runner.DIAG2_SUPERVISOR_ZERO_SCHEMA_VERSION
        )

    monkeypatch.setattr(runner, "_publish_diag2_supervisor_zero", publish_zero)
    monkeypatch.setattr(
        runner, "validate_diag2_supervisor_zero_payload", lambda *_args, **_kwargs: None
    )
    invocation = runner.SnapshotChildInvocation(("child",), repository, {})
    monkeypatch.setattr(
        runner, "build_child_invocation", lambda *_args, **_kwargs: invocation
    )
    monkeypatch.setattr(
        runner,
        "consume_diag4_successor_authority",
        lambda _claim: events.append("consume"),
    )

    def supervise(
        *_args: object, **_kwargs: object
    ) -> runner.DiagnosticSupervisedSampleV2:
        events.append("supervise:preflight")
        return runner.DiagnosticSupervisedSampleV2(
            sample=SampleName.COLD,
            launched=False,
            terminal_status=runner.ChildTerminalStatus.CRASH,
            child_pid=0,
            child_start_time_ticks=0,
            process_seconds=0.0,
            producer=None,
            producer_absence_reason=runner.AbsenceReason.CHILD_LAUNCH_FAILED,
            selected_failure_reason=runner.FailureReasonCodeV2.CHILD_LAUNCH_FAILED,
            memory=None,
            raw_failure_reasons=("launch",),
            observed_child_argv=None,
            stdout=b"",
            stderr=b"",
            memory_samples=(),
        )

    monkeypatch.setattr(runner, "supervise_diag2_sample", supervise)
    captured: dict[str, object] = {}

    def publish_final(
        _publication: runner.Diag2Publication,
        _refs: dict[str, runner.ArtifactRef | None],
        **arguments: object,
    ) -> SimpleNamespace:
        captured.update(arguments)
        return SimpleNamespace(
            verdict="INCOMPLETE",
            next_route="NOT_PRODUCED",
            speed_comparison="NOT_PRODUCED",
        )

    monkeypatch.setattr(runner, "_publish_diag4_terminal_and_receipt", publish_final)

    summary = runner.run_diag4(
        final_root,
        reference_root=reference_root,
        input_root=input_root,
        interpreter=interpreter,
        environment={},
        successor_claim=claim,
        repo_root=repository,
    )

    outcome = captured["outcome"]
    assert isinstance(outcome, runner.StructuredFailureV4)
    assert outcome.reason is runner.FailureReasonCodeV4.PREFLIGHT_LAUNCH_FAILED
    assert events[-2:] == ["consume", "supervise:preflight"]
    assert events.count("supervise:preflight") == 1
    assert captured["successor_claim"] is claim
    assert captured["launched_children"] == ()
    assert summary["children"] == []


def test_diag4_publication_fault_does_not_mask_an_earlier_setup_failure(
    tmp_path: Path,
) -> None:
    publication = runner._prepare_diag2_publication(
        tmp_path / "diag4-prior-failure",
        repository_root=Path(__file__).resolve().parents[2],
    )
    refs = {name: None for name in runner.DIAG4_EVIDENCE_SLOT_NAMES}
    setup = runner._diag4_failure(
        runner.FailureStageV4.SETUP,
        runner.FailureReasonCodeV4.SOURCE_PUBLICATION_FAILED,
        "source fault",
    )

    partial = runner._diag4_publication_partial(
        publication,
        refs,
        launched_children=(),
        prior_outcome=setup,
        reason=runner.FailureReasonCodeV4.MANIFEST_INVALID,
        error=ValueError("manifest fault"),
    )

    assert partial.outcome == setup
    terminal = runner.load_canonical_json_bytes(
        (publication.staging_root / "supervisor-terminal.json").read_bytes()
    )
    assert terminal["terminal_outcome"]["stage"] == runner.FailureStageV4.SETUP.value
    assert (
        terminal["terminal_outcome"]["reason"]["code"]
        == runner.FailureReasonCodeV4.SOURCE_PUBLICATION_FAILED.value
    )


def test_diag4_authority_is_absent_from_the_live_legacy_snapshot() -> None:
    repository = Path(__file__).resolve().parents[2]
    native_extension = Path(runner.simsoptpp.__file__).resolve(strict=True)

    legacy = runner._enumerated_source_roots(repository, native_extension)

    legacy_paths = {root.relative_path for root in legacy}
    assert successor_authority.DIAG4_AUTHORITY_RELATIVE_PATH not in legacy_paths


def test_diag4_quarantine_collision_never_overwrites_retained_bytes(
    tmp_path: Path,
) -> None:
    cold = tmp_path / "campaign" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    publication.uncommitted_root.mkdir()
    (publication.pending_root / "new.bin").write_bytes(b"new")
    (publication.uncommitted_root / "old.bin").write_bytes(b"old")

    with pytest.raises(FileExistsError):
        runner._quarantine_cold_numerical_bundle(publication)
    assert (publication.pending_root / "new.bin").read_bytes() == b"new"
    assert (publication.uncommitted_root / "old.bin").read_bytes() == b"old"


def test_diag4_committed_bundle_must_deep_load_after_atomic_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cold = tmp_path / "campaign" / "cold"
    cold.mkdir(parents=True)
    publication = runner._cold_numerical_bundle_publication(cold)
    publication.pending_root.mkdir()
    (publication.pending_root / "history.json").write_bytes(b"complete")
    outcome = runner.DiagnosticSupervisedSampleV2(
        sample=SampleName.COLD,
        launched=True,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=1,
        child_start_time_ticks=2,
        process_seconds=1.0,
        producer={
            "schema_version": runner.DIAG4_COLD_RESULT_SCHEMA_VERSION,
            "execution_status": "COMPLETE",
        },
        producer_absence_reason=None,
        selected_failure_reason=None,
        memory={"peak_memory_fraction": 0.1},
        raw_failure_reasons=(),
        observed_child_argv=("child",),
        stdout=b"{}",
        stderr=b"",
        memory_samples=(),
    )
    validation_roots: list[Path] = []

    def validate(
        bundle: runner.ColdNumericalBundlePublication,
        _producer: Mapping[str, object],
    ) -> None:
        validation_roots.append(bundle.pending_root)
        if bundle.pending_root == bundle.committed_root:
            raise ValueError("committed mutation")

    monkeypatch.setattr(runner, "_validate_pending_diag4_numerical_bundle", validate)

    resolved = runner._resolve_diag4_cold_numerical_bundle(cold, outcome)

    assert validation_roots == [
        publication.pending_root,
        publication.committed_root,
    ]
    assert resolved.producer is None
    assert resolved.selected_failure_reason is (
        runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID
    )
    assert resolved.raw_failure_reasons[0].startswith(
        "NUMERICAL_COMMIT:COMMITTED_DEEP_LOAD_FAILED:"
    )
    assert not publication.pending_root.exists()
    assert (publication.committed_root / "history.json").read_bytes() == b"complete"


def test_diag2_gpu_zero_capture_rejects_supervisor_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    identities = iter(
        (
            SimpleNamespace(pid=os.getpid(), start_ticks=10),
            SimpleNamespace(pid=os.getpid(), start_ticks=11),
        )
    )
    monkeypatch.setattr(
        runner, "read_linux_process_identity", lambda _pid: next(identities)
    )
    monkeypatch.setattr(
        runner,
        "capture_supervisor_gpu_zero",
        lambda **_kwargs: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="identity changed"):
        runner._capture_diag2_supervisor_zero(
            {"CUDA_VISIBLE_DEVICES": runner.GPU_UUID},
            query_executable_sha256="a" * 64,
        )


def test_diag2_supervisor_zero_publishes_both_raw_queries_before_gate(
    tmp_path: Path,
) -> None:
    staging = tmp_path / f"diag2.partial-{'0' * 32}"
    staging.mkdir()
    inventory = process_gpu_monitor.SupervisorGpuQuery(
        argv=process_gpu_monitor.SUPERVISOR_GPU_INVENTORY_QUERY,
        query_executable_sha256="a" * 64,
        launched=True,
        timed_out=False,
        returncode=0,
        stdout=b"GPU-7951f78e-c05d-e01c-303f-d644f4341fe1, 32768\n",
        stderr=b"",
    )
    compute = process_gpu_monitor.SupervisorGpuQuery(
        argv=process_gpu_monitor.SUPERVISOR_COMPUTE_APPS_QUERY,
        query_executable_sha256="a" * 64,
        launched=True,
        timed_out=False,
        returncode=0,
        stdout=b"99, GPU-7951f78e-c05d-e01c-303f-d644f4341fe1, 128\n",
        stderr=b"",
    )
    observation = process_gpu_monitor.SupervisorGpuZeroObservation(
        captured_at_monotonic_ns=1,
        captured_at_unix_ns=2,
        supervisor_pid=123,
        supervisor_start_ticks=456,
        gpu_uuid=runner.GPU_UUID,
        visible_device=runner.GPU_UUID,
        gpu_inventory_query=inventory,
        compute_apps_query=compute,
        inventory_rows=(
            process_gpu_monitor.SupervisorGpuInventoryRow(runner.GPU_UUID, 32768),
        ),
        compute_rows=(
            process_gpu_monitor.SupervisorComputeAppRow(99, runner.GPU_UUID, 128),
        ),
        matching_rows=(),
        parse_valid=True,
    )

    reference = runner._publish_diag2_supervisor_zero(
        staging,
        observation,
        stage="BEFORE_PREFLIGHT",
    )

    assert reference.relative_path == "supervisor/before-preflight.json"
    assert (staging / "supervisor/before-preflight-gpu-inventory.stdout.bin").is_file()
    assert (staging / "supervisor/before-preflight-compute-apps.stdout.bin").is_file()


def test_diag2_supervisor_publication_fault_leaves_visible_partial_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "campaigns"
    parent.mkdir()
    repository = tmp_path / "repository"
    repository.mkdir()
    publication = runner._prepare_diag2_publication(
        parent / "diag2",
        repository_root=repository,
    )
    observation = SimpleNamespace(
        gpu_inventory_query=SimpleNamespace(stdout=b"", stderr=b""),
        compute_apps_query=SimpleNamespace(stdout=b"", stderr=b""),
    )
    monkeypatch.setattr(
        runner,
        "_publish_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk fault")),
    )

    with pytest.raises(OSError, match="disk fault"):
        runner._publish_diag2_supervisor_zero(
            publication.staging_root,
            observation,
            stage="BEFORE_PREFLIGHT",
        )

    assert publication.staging_root.is_dir()
    assert not publication.final_root.exists()


def test_diag4_supervisor_zero_publication_fault_is_typed_hard_partial(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    publication = runner._prepare_diag2_publication(
        tmp_path / "diag4-zero",
        repository_root=repository,
    )
    refs = {name: None for name in runner.DIAG4_EVIDENCE_SLOT_NAMES}
    monkeypatch.setattr(
        runner,
        "_capture_diag2_supervisor_zero",
        lambda *_args, **_kwargs: SimpleNamespace(gate_passes=True),
    )
    monkeypatch.setattr(
        runner,
        "_publish_diag2_supervisor_zero",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk fault")),
    )

    with pytest.raises(runner.Diag4HardPublicationError) as caught:
        runner._diag4_supervisor_zero_gate(
            publication,
            refs,
            environment={},
            query_executable_sha256="a" * 64,
            stage=runner.FailureStageV4.BEFORE_PREFLIGHT,
        )

    assert (
        caught.value.reason
        is runner.FailureReasonCodeV4.SUPERVISOR_GPU_OBSERVATION_INVALID
    )
    assert caught.value.root == publication.staging_root
    assert publication.staging_root.is_dir()
    assert not publication.final_root.exists()


def _diag2_ref(relative_path: str, schema: str = "test-v1") -> runner.ArtifactRef:
    return runner.ArtifactRef(relative_path, "0" * 64, 0, schema)


def _diag2_integrated_mocks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    preflight_gate_error: runner.Diag2PreflightGateError | None = None,
    setup_gate_error: runner.Diag2SetupGateError | None = None,
    post_preflight_setup_gate_error: runner.Diag2SetupGateError | None = None,
    cold_setup_gate_error: runner.Diag2SetupGateError | None = None,
    preflight_failure_reason: runner.FailureReasonCodeV2 | None = None,
    zero_failure_stage: str | None = None,
    cold_offending_slot: str | None = None,
    descriptor_failure_path: str | None = None,
    source_capture_failure_call: int | None = None,
    expected_failure_stage: runner.FailureStageV2 | None = None,
) -> list[str]:
    events: list[str] = []
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("JAX_COMPILATION_CACHE_DIR", raising=False)
    monkeypatch.setenv("JAX_ENABLE_COMPILATION_CACHE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    repository = tmp_path / "repo"
    repository.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / runner.REFERENCE_FILENAME).write_bytes(b"{}")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("")
    interpreter.chmod(0o755)
    staging = tmp_path / f"result.partial-{'a' * 32}"
    staging.mkdir()
    campaign_reference = staging / "native-reference"
    campaign_reference.mkdir()
    (campaign_reference / runner.REFERENCE_FILENAME).write_bytes(b"{}")
    publication = runner.Diag2Publication(staging, tmp_path / "result", "a" * 32)
    source_manifest = _diag2_ref("source-snapshot/source-manifest.json")
    source = SimpleNamespace(
        source_identity=lambda _root: SimpleNamespace(snapshot_manifest=source_manifest)
    )
    source_identity = runner.SourceIdentityEvidence(
        git_head="0" * 40,
        tracked_diff_sha256="1" * 64,
        untracked_bytes_manifest_sha256="2" * 64,
        source_manifest_sha256="3" * 64,
        source_manifest_size_bytes=1,
    )
    invocation = runner.SnapshotChildInvocation(
        (str(interpreter), "child"), repository, {}
    )
    preflight_producer = (
        {"execution_status": "COMPILE_FAILURE"}
        if preflight_failure_reason is runner.FailureReasonCodeV2.CHILD_COMPILE_FAILED
        else (
            {"execution_status": "SUCCESS"}
            if preflight_failure_reason is None
            else None
        )
    )
    preflight_absence = (
        None
        if preflight_producer is not None
        else (
            runner.AbsenceReason.MONITOR_FINALIZATION_FAILED
            if preflight_failure_reason
            is runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
            else (
                runner.AbsenceReason.PRODUCER_SCHEMA_INVALID
                if preflight_failure_reason
                is runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID
                else (
                    runner.AbsenceReason.PRODUCER_DECODE_FAILED
                    if preflight_failure_reason
                    is runner.FailureReasonCodeV2.PRODUCER_DECODE_FAILED
                    else runner.AbsenceReason.CHILD_EXIT_NONZERO
                )
            )
        )
    )
    preflight_status = {
        runner.FailureReasonCodeV2.CHILD_COMPILE_FAILED: (
            runner.ChildTerminalStatus.COMPILE_FAILURE
        ),
        runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO: runner.ChildTerminalStatus.CRASH,
        runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED: (
            runner.ChildTerminalStatus.MONITOR_FAILURE
        ),
        runner.FailureReasonCodeV2.PRODUCER_DECODE_FAILED: (
            runner.ChildTerminalStatus.PROTOCOL_FAILURE
        ),
        runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID: (
            runner.ChildTerminalStatus.PROTOCOL_FAILURE
        ),
    }.get(preflight_failure_reason, runner.ChildTerminalStatus.COMPLETE)
    preflight = runner.DiagnosticSupervisedSampleV2(
        SampleName.COLD,
        True,
        preflight_status,
        101,
        201,
        1.0,
        preflight_producer,
        preflight_absence,
        preflight_failure_reason,
        (
            None
            if preflight_failure_reason
            is runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
            else {"peak_memory_fraction": 0.1}
        ),
        (),
        ("child",),
        b"preflight",
        b"",
        (),
    )
    cold_producer = {
        "execution_status": "COMPLETE",
        "history_evidence": {
            "relative_path": "cold/history.json",
            "sha256": "4" * 64,
            "size_bytes": 1,
            "schema_version": "history-v1",
        },
        "terminal_numerical_evidence": {
            "relative_path": "cold/terminal-numerical.json",
            "sha256": "5" * 64,
            "size_bytes": 1,
            "schema_version": "terminal-v1",
        },
        "raw_trace_evidence": {
            "relative_path": "cold/raw-trace/plugins/profile/run/a.trace.json.gz",
            "sha256": "6" * 64,
            "size_bytes": 1,
            "schema_version": "trace-v1",
        },
        "trace_intervals_evidence": {
            "relative_path": "cold/trace-intervals.json",
            "sha256": "7" * 64,
            "size_bytes": 1,
            "schema_version": "interval-v1",
        },
    }
    cold = runner.DiagnosticSupervisedSampleV2(
        SampleName.COLD,
        True,
        runner.ChildTerminalStatus.COMPLETE,
        102,
        202,
        2.0,
        cold_producer,
        None,
        None,
        {"peak_memory_fraction": 0.1},
        (),
        ("child",),
        b"cold",
        b"",
        (),
    )
    outcomes = iter((preflight, cold))

    monkeypatch.setattr(runner, "read_linux_process_identity", lambda _pid: object())
    monkeypatch.setattr(
        runner,
        "_validate_parent_execution_policy",
        lambda **_kwargs: (
            repository,
            tmp_path / "native.so",
            reference,
            inputs,
            interpreter,
            runner.GPU_UUID,
            32 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        runner, "_prepare_diag2_publication", lambda *_a, **_k: publication
    )
    monkeypatch.setattr(runner, "_prepare_diag2_snapshot", lambda *_a, **_k: source)
    monkeypatch.setattr(
        runner,
        "copy_validated_reference",
        lambda *_a, **_k: campaign_reference,
    )
    monkeypatch.setattr(
        runner,
        "_publish_diag2_policy_authority",
        lambda *_a, **_k: (
            _diag2_ref("policy-authority.json"),
            SimpleNamespace(),
        ),
    )
    setup_calls = 0

    def setup_gate(*_args: object, **_kwargs: object) -> bool:
        nonlocal setup_calls
        setup_calls += 1
        error = {
            1: setup_gate_error,
            2: post_preflight_setup_gate_error,
            3: cold_setup_gate_error,
        }.get(setup_calls)
        if error is not None:
            raise error
        return True

    monkeypatch.setattr(runner, "validate_diag2_setup_authorities", setup_gate)
    monkeypatch.setattr(
        runner,
        "supervisor_query_executable_sha256",
        lambda: "f" * 64,
    )
    monkeypatch.setattr(
        runner,
        "_capture_diag2_supervisor_zero",
        lambda _environment, **_kwargs: SimpleNamespace(matching_rows=()),
    )

    def publish_zero(
        *_args: object, stage: str, **_kwargs: object
    ) -> runner.ArtifactRef:
        events.append(stage)
        slug = "before-preflight" if stage == "BEFORE_PREFLIGHT" else "before-cold"
        return _diag2_ref(f"supervisor/{slug}.json")

    monkeypatch.setattr(runner, "_publish_diag2_supervisor_zero", publish_zero)
    monkeypatch.setattr(
        runner,
        "validate_diag2_supervisor_zero_payload",
        lambda *_a, expected_stage, **_k: (
            (_ for _ in ()).throw(ValueError("query failed"))
            if expected_stage == zero_failure_stage
            else {}
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_canonical_json_bytes",
        lambda _data: {},
    )
    monkeypatch.setattr(
        runner.ArtifactRef,
        "resolve_and_validate",
        lambda _self, _root: interpreter,
    )
    monkeypatch.setattr(runner, "build_child_invocation", lambda *_a, **_k: invocation)
    source_capture_calls = 0

    def capture_source(*_args: object, **_kwargs: object) -> object:
        nonlocal source_capture_calls
        source_capture_calls += 1
        if source_capture_calls == source_capture_failure_call:
            raise ValueError("source capture failed")
        return source_identity

    monkeypatch.setattr(runner, "_capture_source_identity_evidence", capture_source)

    def supervise(
        *_args: object, mode: runner.DiagnosticChildMode, **_kwargs: object
    ) -> runner.DiagnosticSupervisedSampleV2:
        assert (staging / mode.value).is_dir()
        events.append(mode.value)
        return next(outcomes)

    monkeypatch.setattr(runner, "supervise_diag2_sample", supervise)

    def publish_supervision(
        _root: Path,
        directory: Path,
        supervised: runner.DiagnosticSupervisedSampleV2,
        **_kwargs: object,
    ) -> dict[str, runner.ArtifactRef | None]:
        prefix = directory.name
        return {
            "producer": (
                _diag2_ref(f"{prefix}/producer.json")
                if supervised.producer is not None
                else None
            ),
            "terminal": _diag2_ref(f"{prefix}/terminal.json"),
            "process": _diag2_ref(f"{prefix}/process.json"),
            "memory": (
                _diag2_ref(f"{prefix}/gpu-memory.json")
                if supervised.memory is not None
                else None
            ),
            "memory_samples": (
                _diag2_ref(f"{prefix}/gpu-memory-samples.json")
                if supervised.memory is not None
                else None
            ),
        }

    monkeypatch.setattr(runner, "_publish_diag2_supervision", publish_supervision)
    subordinate_outcomes = {
        "preflight": preflight_failure_reason,
        "cold": None,
    }
    monkeypatch.setattr(
        runner,
        "classify_diag3_subordinate_child_outcome",
        lambda _root, *, mode, **_kwargs: subordinate_outcomes[mode],
    )
    monkeypatch.setattr(
        runner,
        "_diag2_existing_reference",
        lambda _root, relative, _schema: _diag2_ref(relative),
    )

    def gate(*_args: object, **_kwargs: object) -> bool:
        events.append("preflight-gate")
        if preflight_gate_error is not None:
            raise preflight_gate_error
        return True

    monkeypatch.setattr(runner, "validate_diag2_preflight_gate", gate)
    monkeypatch.setattr(
        runner,
        "_publish_diag2_execution",
        lambda *_a, **_k: _diag2_ref("execution.json"),
    )
    if descriptor_failure_path is not None:
        original_artifact_from_payload = runner._artifact_from_payload

        def artifact_from_payload(
            payload: dict[str, runner.JsonValue],
        ) -> runner.ArtifactRef:
            if payload.get("relative_path") == descriptor_failure_path:
                raise TypeError("malformed descriptor")
            return original_artifact_from_payload(payload)

        monkeypatch.setattr(runner, "_artifact_from_payload", artifact_from_payload)
    cold_prefix = (
        "cold_runtime",
        "cold_policy",
        "cold_history",
        "cold_terminal_numerical",
        "cold_raw_trace",
        "cold_trace_intervals",
    )

    def classify(
        _root: Path,
        *,
        artifact_refs: dict[str, runner.ArtifactRef | None],
    ) -> object:
        if cold_offending_slot is not None:
            prefix = cold_prefix[: cold_prefix.index(cold_offending_slot)]
            protocol = cold_offending_slot in {"cold_runtime", "cold_policy"}
            return SimpleNamespace(
                typed_slots=prefix,
                failure=runner.StructuredFailureV2(
                    (
                        runner.FailureStageV2.COLD_PROTOCOL_FAILURE
                        if protocol
                        else runner.FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE
                    ),
                    (
                        runner.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID
                        if cold_offending_slot == "cold_runtime"
                        else (
                            runner.FailureReasonCodeV2.POLICY_SCHEMA_INVALID
                            if cold_offending_slot == "cold_policy"
                            else runner.FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID
                        )
                    ),
                    "f" * 64,
                ),
                offending_slot=cold_offending_slot,
            )
        return SimpleNamespace(
            typed_slots=(
                (*cold_prefix, "execution")
                if artifact_refs["execution"] is not None
                else cold_prefix
            ),
            failure=(
                None
                if artifact_refs["execution"] is not None
                else runner.StructuredFailureV2(
                    runner.FailureStageV2.NUMERICAL_EVIDENCE_INCOMPLETE,
                    runner.FailureReasonCodeV2.NUMERICAL_SCHEMA_INVALID,
                    "f" * 64,
                )
            ),
            offending_slot=(
                None if artifact_refs["execution"] is not None else "execution"
            ),
        )

    monkeypatch.setattr(
        runner,
        "classify_diag3_cold_evidence",
        classify,
    )

    def finish(
        _publication: runner.Diag2Publication,
        _refs: dict[str, runner.ArtifactRef | None],
        *,
        failure: runner.StructuredFailureV2 | None,
        **_kwargs: object,
    ) -> object:
        if expected_failure_stage is not None:
            assert failure is not None
            assert failure.stage is expected_failure_stage
        if cold_offending_slot is not None:
            ordered = (*cold_prefix, "execution")
            first = ordered.index(cold_offending_slot)
            assert all(_refs[name] is not None for name in ordered[:first])
            assert all(_refs[name] is None for name in ordered[first:])
        events.append(
            "finish:" + ("complete" if failure is None else failure.reason.value)
        )
        return SimpleNamespace(
            verdict="DIAGNOSTIC_COMPLETE_NO_HIT"
            if failure is None
            else "DIAGNOSTIC_INCOMPLETE",
            next_route="RETRY_MODEL_REUSE" if failure is None else "NOT_PRODUCED",
        )

    monkeypatch.setattr(runner, "_publish_diag2_terminal_and_receipt", finish)
    runner.run_diag2(
        publication.final_root,
        reference_root=reference,
        input_root=inputs,
        interpreter=interpreter,
        environment={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID},
        repo_root=repository,
    )
    return events


def test_diag2_integrated_sequence_is_preflight_zero_cold_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = _diag2_integrated_mocks(monkeypatch, tmp_path)

    assert events == [
        "BEFORE_PREFLIGHT",
        "preflight",
        "preflight-gate",
        "BEFORE_COLD",
        "cold",
        "finish:complete",
    ]


@pytest.mark.parametrize(
    ("reason", "expected_finish"),
    (
        (runner.FailureReasonCodeV2.SOURCE_PRE, "SOURCE_PRE"),
        (
            runner.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
            "FROZEN_SUBSET_INVALID",
        ),
        (runner.FailureReasonCodeV2.REFERENCE_INVALID, "REFERENCE_INVALID"),
        (
            runner.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
            "POLICY_DERIVATION_INVALID",
        ),
    ),
)
def test_diag2_setup_authority_failure_seals_before_gpu_query_or_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: runner.FailureReasonCodeV2,
    expected_finish: str,
) -> None:
    error = runner.Diag2SetupGateError(reason, "policy_authority", "f" * 64)

    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        setup_gate_error=error,
    )

    assert events == [f"finish:{expected_finish}"]


@pytest.mark.parametrize(
    ("capture_call", "stage", "expected_events"),
    (
        (
            1,
            runner.FailureStageV2.SOURCE_PUBLICATION_FAILURE,
            ["finish:SOURCE_POST"],
        ),
        (
            2,
            runner.FailureStageV2.PREFLIGHT_SOURCE_FAILURE,
            ["BEFORE_PREFLIGHT", "preflight", "finish:SOURCE_POST"],
        ),
    ),
)
def test_diag2_prechild_source_recapture_uses_lawful_source_post_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capture_call: int,
    stage: runner.FailureStageV2,
    expected_events: list[str],
) -> None:
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        source_capture_failure_call=capture_call,
        expected_failure_stage=stage,
    )

    assert events == expected_events


def test_diag2_strict_preflight_gate_failure_never_launches_cold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    error = runner.Diag2PreflightGateError(
        runner.FailureReasonCodeV2.RUNTIME_SCHEMA_INVALID,
        "preflight_runtime",
        "bad runtime",
    )
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        preflight_gate_error=error,
    )

    assert events == [
        "BEFORE_PREFLIGHT",
        "preflight",
        "preflight-gate",
        "finish:RUNTIME_SCHEMA_INVALID",
    ]


@pytest.mark.parametrize(
    "reason",
    (
        runner.FailureReasonCodeV2.PRODUCER_DECODE_FAILED,
        runner.FailureReasonCodeV2.PRODUCER_SCHEMA_INVALID,
    ),
)
def test_diag2_missing_or_invalid_preflight_producer_never_reaches_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: runner.FailureReasonCodeV2,
) -> None:
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        preflight_failure_reason=reason,
    )

    assert events == [
        "BEFORE_PREFLIGHT",
        "preflight",
        f"finish:{reason.value}",
    ]


@pytest.mark.parametrize(
    "reason",
    (
        runner.FailureReasonCodeV2.SOURCE_POST,
        runner.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        runner.FailureReasonCodeV2.REFERENCE_INVALID,
        runner.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
    ),
)
def test_diag2_post_launch_setup_drift_seals_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: runner.FailureReasonCodeV2,
) -> None:
    error = runner.Diag2PreflightGateError(reason, "source_manifest", "f" * 64)

    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        preflight_gate_error=error,
    )

    assert events[-1] == f"finish:{reason.value}"
    assert "BEFORE_COLD" not in events


@pytest.mark.parametrize(
    "child_reason",
    (
        runner.FailureReasonCodeV2.CHILD_COMPILE_FAILED,
        runner.FailureReasonCodeV2.CHILD_EXIT_NONZERO,
        runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED,
    ),
)
def test_diag2_post_preflight_setup_drift_precedes_child_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    child_reason: runner.FailureReasonCodeV2,
) -> None:
    setup_error = runner.Diag2SetupGateError(
        runner.FailureReasonCodeV2.SOURCE_PRE,
        "source_manifest",
        "f" * 64,
    )

    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        post_preflight_setup_gate_error=setup_error,
        preflight_failure_reason=child_reason,
    )

    assert events == [
        "BEFORE_PREFLIGHT",
        "preflight",
        "finish:SOURCE_POST",
    ]


@pytest.mark.parametrize(
    "reason",
    (
        runner.FailureReasonCodeV2.SOURCE_PRE,
        runner.FailureReasonCodeV2.FROZEN_SUBSET_INVALID,
        runner.FailureReasonCodeV2.REFERENCE_INVALID,
        runner.FailureReasonCodeV2.POLICY_DERIVATION_INVALID,
    ),
)
def test_diag2_post_cold_setup_drift_seals_after_complete_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    reason: runner.FailureReasonCodeV2,
) -> None:
    error = runner.Diag2SetupGateError(reason, "policy_authority", "f" * 64)

    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        cold_setup_gate_error=error,
    )

    expected = (
        runner.FailureReasonCodeV2.SOURCE_POST.value
        if reason is runner.FailureReasonCodeV2.SOURCE_PRE
        else reason.value
    )
    assert events[-1] == f"finish:{expected}"
    assert events.count("cold") == 1


@pytest.mark.parametrize(
    ("stage", "expected_events"),
    (
        (
            "BEFORE_PREFLIGHT",
            ["BEFORE_PREFLIGHT", "finish:GPU_QUERY_FAILED"],
        ),
        (
            "BEFORE_COLD",
            [
                "BEFORE_PREFLIGHT",
                "preflight",
                "preflight-gate",
                "BEFORE_COLD",
                "finish:GPU_QUERY_FAILED",
            ],
        ),
    ),
)
def test_diag2_each_gpu_zero_gate_failure_stops_at_its_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stage: str,
    expected_events: list[str],
) -> None:
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        zero_failure_stage=stage,
    )

    assert events == expected_events


@pytest.mark.parametrize(
    ("offending_slot", "reason"),
    (
        ("cold_runtime", "RUNTIME_SCHEMA_INVALID"),
        ("cold_policy", "POLICY_SCHEMA_INVALID"),
        ("cold_history", "NUMERICAL_SCHEMA_INVALID"),
        ("cold_raw_trace", "NUMERICAL_SCHEMA_INVALID"),
    ),
)
def test_diag2_cold_classifier_stops_at_first_untyped_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    offending_slot: str,
    reason: str,
) -> None:
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        cold_offending_slot=offending_slot,
    )

    assert events[-1] == f"finish:{reason}"
    assert events.count("cold") == 1


def test_diag2_descriptor_failure_preserves_valid_numerical_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = _diag2_integrated_mocks(
        monkeypatch,
        tmp_path,
        cold_offending_slot="cold_raw_trace",
        descriptor_failure_path="cold/raw-trace/plugins/profile/run/a.trace.json.gz",
    )

    assert events[-1] == "finish:NUMERICAL_SCHEMA_INVALID"


def test_timed_loop_synchronizes_before_timer_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = SimpleNamespace(
        run_solver_loop=lambda: events.append("launch") or object(),
    )
    monkeypatch.setattr(
        runner.jax,
        "block_until_ready",
        lambda value: events.append("synchronize") or value,
    )
    monkeypatch.setattr(
        runner.jax,
        "transfer_guard",
        lambda _mode: _RecordingContext(events),
    )

    result, started, stopped = runner.execute_timed_loop(prepared)

    assert result is not None
    assert stopped >= started
    assert events == ["guard-enter", "launch", "synchronize", "guard-exit"]


def test_diagnostic_timed_loop_annotation_wraps_only_launch_and_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    prepared = SimpleNamespace(
        run_solver_loop=lambda: events.append("launch") or object(),
    )

    class Annotation:
        def __init__(self, name: str) -> None:
            assert name == runner.TRACE_LOOP_ENVELOPE_NAME

        def __enter__(self) -> None:
            events.append("annotation-enter")

        def __exit__(self, *_args: object) -> None:
            events.append("annotation-exit")

    monkeypatch.setattr(
        runner.jax,
        "block_until_ready",
        lambda value: events.append("synchronize") or value,
    )
    monkeypatch.setattr(
        runner.jax,
        "transfer_guard",
        lambda _mode: _RecordingContext(events),
    )
    monkeypatch.setattr(runner.jax.profiler, "TraceAnnotation", Annotation)

    runner.execute_timed_loop(
        prepared,
        trace_annotation=runner.TRACE_LOOP_ENVELOPE_NAME,
    )

    assert events == [
        "guard-enter",
        "annotation-enter",
        "launch",
        "synchronize",
        "annotation-exit",
        "guard-exit",
    ]


def test_compiled_callback_audit_counts_forbidden_host_primitives() -> None:
    clean = SimpleNamespace(_run_loop=SimpleNamespace(as_text=lambda: "stablehlo.add"))
    dirty = SimpleNamespace(
        _run_loop=SimpleNamespace(
            as_text=lambda: "stablehlo.custom_call @xla_python_cpu_callback"
        )
    )

    assert runner._compiled_python_callback_count(clean) == 0
    assert runner._compiled_python_callback_count(dirty) == 1


def test_diag4_worker_selects_authoritative_gntr3_preparer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = SimpleNamespace()
    observed: list[object] = []

    def prepare(
        _reference: Path,
        _inputs: Path,
        route_preparer: object,
    ) -> object:
        observed.append(route_preparer)
        return sentinel

    monkeypatch.setattr(runner, "_prepare_worker_for_route", prepare)

    assert runner._prepare_diag4_worker(Path("reference"), Path("input")) is sentinel
    assert observed == [runner.prepare_neq_gntr3]


def test_diag4_prepared_route_rejects_gntr2_identity_as_gntr3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePreparedNeqGntr3:
        pass

    prepared = FakePreparedNeqGntr3()
    prepared.identity = SimpleNamespace(
        schema_version="single-stage-fullspace-neq-gntr2-result-v1",
        route="NEQ-GNTR2",
    )
    prepared.options = runner.NEQ_GNTR3_OPTIONS
    monkeypatch.setattr(runner, "PreparedNeqGntr3", FakePreparedNeqGntr3)

    with pytest.raises(ValueError, match="route or schema identity"):
        runner._validate_diag4_prepared_route(prepared)


@pytest.mark.parametrize("safeguard", [False, 1])
def test_diag4_prepared_route_rejects_disabled_or_nonboolean_safeguard(
    monkeypatch: pytest.MonkeyPatch,
    safeguard: object,
) -> None:
    class FakePreparedNeqGntr3:
        pass

    prepared = FakePreparedNeqGntr3()
    prepared.identity = SimpleNamespace(
        schema_version=runner.NEQ_GNTR3_SCHEMA_VERSION,
        route=runner.NEQ_GNTR3_ROUTE,
    )
    prepared.options = SimpleNamespace(enable_step_bound_safeguard=safeguard)
    monkeypatch.setattr(runner, "PreparedNeqGntr3", FakePreparedNeqGntr3)

    with pytest.raises(ValueError, match="safeguard options"):
        runner._validate_diag4_prepared_route(prepared)


def test_diag4_safeguard_history_transfer_requires_exact_shapes_and_dtypes() -> None:
    integer_vector = runner._diag4_integer_history_vector(
        np.zeros(300, dtype=np.int32), context="count"
    )
    float_matrix = runner._diag4_float_history_matrix(
        np.zeros((300, 3), dtype=np.float64), context="float"
    )
    float_vector = runner._diag4_float_history_vector(
        np.zeros(300, dtype=np.float64), context="float vector"
    )
    integer_matrix = runner._diag4_integer_history_matrix(
        np.zeros((300, 3), dtype=np.int32), context="integer"
    )
    np.testing.assert_array_equal(integer_vector, np.zeros(300, dtype=np.int32))
    np.testing.assert_array_equal(float_matrix, np.zeros((300, 3), dtype=np.float64))
    np.testing.assert_array_equal(float_vector, np.zeros(300, dtype=np.float64))
    np.testing.assert_array_equal(integer_matrix, np.zeros((300, 3), dtype=np.int32))
    assert not integer_vector.flags.writeable
    assert not float_matrix.flags.writeable
    assert not float_vector.flags.writeable
    assert not integer_matrix.flags.writeable

    with pytest.raises(TypeError, match="int32"):
        runner._diag4_integer_history_vector(
            np.zeros(300, dtype=np.int64), context="count"
        )
    with pytest.raises(TypeError, match="FP64"):
        runner._diag4_float_history_matrix(
            np.zeros((300, 3), dtype=np.float32), context="float"
        )
    with pytest.raises(TypeError, match="FP64"):
        runner._diag4_float_history_vector(
            np.zeros(300, dtype=np.float32), context="float vector"
        )
    with pytest.raises(TypeError, match="int32"):
        runner._diag4_integer_history_matrix(
            np.zeros((300, 2), dtype=np.int32), context="integer"
        )


def test_no_hit_receipt_preserves_observed_accepted_step_count() -> None:
    producer = {
        "candidate_reached": False,
        "candidate": {"accepted_step_count": 203},
    }

    assert runner._noncandidate_accepted_step_count(producer) == 203
    assert runner._noncandidate_accepted_step_count({}) == 0


def test_preflight_child_compiles_without_solver_finalizer_or_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_calls: list[str] = []
    route = SimpleNamespace(
        initial_optimizer_coordinates=object(),
        policy=SimpleNamespace(
            policy_sha256="a" * 64,
            state_size=716,
            equality_size=255,
            objective_residual_size=2110,
        ),
        _run_loop=SimpleNamespace(as_text=lambda: "stablehlo.add"),
        run_solver_loop=lambda: forbidden_calls.append("solver"),
        finalize_result=lambda _value: forbidden_calls.append("finalizer"),
    )
    worker = SimpleNamespace(route=route)
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(runner.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "preflight/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(runner, "_prepare_worker", lambda *_args: worker)
    monkeypatch.setattr(
        runner,
        "produce_native_equivalent_endpoint_audit",
        lambda *_args: forbidden_calls.append("audit"),
    )

    payload = runner.run_snapshot_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "SUCCESS"
    assert payload["campaign_authorized"] is False
    assert payload["solver_dispatched"] is False
    assert payload["finalizer_called"] is False
    assert payload["endpoint_audit_called"] is False
    assert forbidden_calls == []


def test_preflight_compile_failure_never_dispatches_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "preflight/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_prepare_worker",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("out of memory")),
    )

    payload = runner.run_snapshot_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "COMPILE_OOM"
    assert payload["campaign_authorized"] is False


def test_diagnostic_preflight_never_dispatches_any_compiled_executable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    forbidden_calls: list[str] = []
    executable = SimpleNamespace(
        as_text=lambda: "stablehlo.add",
        __call__=lambda *_args: forbidden_calls.append("dispatch"),
    )
    route = SimpleNamespace(
        initial_optimizer_coordinates=object(),
        policy=SimpleNamespace(
            policy_sha256="a" * 64,
            native_raw_equalities=np.zeros(255, dtype=np.float64),
            constraint_inverse_scale=np.ones(255, dtype=np.float64),
            state_size=716,
            equality_size=255,
            objective_residual_size=2110,
        ),
        _run_loop=executable,
        _finalize=executable,
        _map_ledger=executable,
        run_solver_loop=lambda: forbidden_calls.append("solver"),
        finalize_result=lambda _value: forbidden_calls.append("finalizer"),
    )
    worker = SimpleNamespace(
        worker=SimpleNamespace(route=route),
        accepted_quality=SimpleNamespace(_run_quality=executable),
        terminal=SimpleNamespace(_run_endpoint=executable),
    )
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(runner.jax, "block_until_ready", lambda value: value)
    campaign = tmp_path / "campaign"
    (campaign / "preflight").mkdir(parents=True)
    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setenv(runner._DIAG2_CHILD_ENV, "1")
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "preflight/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(runner, "_prepare_diagnostic_worker", lambda *_args: worker)

    payload = runner.run_snapshot_diagnostic_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "SUCCESS"
    assert payload["policy_evidence"]["relative_path"] == "preflight/policy.json"
    assert payload["solver_dispatched"] is False
    assert payload["finalizer_called"] is False
    assert payload["endpoint_audit_called"] is False
    assert forbidden_calls == []


def test_diag2_preflight_compile_failure_is_typed_without_text_oom_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_reference = runner.ArtifactRef(
        "preflight/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
    )
    monkeypatch.setenv(runner._DIAG2_CHILD_ENV, "1")
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(
        runner, "_publish_child_runtime_evidence", lambda _sample: runtime_reference
    )
    monkeypatch.setattr(
        runner,
        "_worker_runtime_payload",
        lambda: {
            "backend": "gpu",
            "device": "gpu",
            "device_uuid": runner.GPU_UUID,
            "jax": "test",
            "jax_enable_x64": True,
            "jaxlib": "test",
            "python": "test",
        },
    )
    monkeypatch.setattr(
        runner,
        "_prepare_diagnostic_worker",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("out of memory")),
    )

    payload = runner.run_snapshot_diagnostic_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "COMPILE_FAILURE"
    assert runner.validate_diag3_producer_payload(payload, mode="preflight") == payload


def test_diag5_preflight_compile_failure_uses_exact_v5_union(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime_reference = runner.ArtifactRef(
        "preflight/runtime-evidence.json",
        "a" * 64,
        1,
        runner.RUNTIME_EVIDENCE_V2_SCHEMA_VERSION,
    )
    monkeypatch.setenv(runner._DIAG5_CHILD_ENV, "1")
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(
        runner, "_publish_child_runtime_evidence", lambda _sample: runtime_reference
    )
    monkeypatch.setattr(
        runner,
        "_worker_runtime_payload",
        lambda: {
            "backend": "gpu",
            "device": "gpu",
            "device_uuid": runner.GPU_UUID,
            "jax": "test",
            "jax_enable_x64": True,
            "jaxlib": "test",
            "python": "test",
        },
    )
    monkeypatch.setattr(
        runner,
        "_prepare_diagnostic_worker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("resource_exhausted")
        ),
    )

    payload = runner.run_snapshot_diagnostic_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["schema_version"] == runner.DIAG5_PREFLIGHT_SCHEMA_VERSION
    assert payload["execution_status"] == "COMPILE_OOM"
    assert payload["numerical_route"] == runner.NEQ_GNTR3_ROUTE
    assert payload["profiler_start_calls"] == 0
    assert runner.validate_diag5_producer_payload(payload, mode="preflight") == payload


def test_diag4_preflight_compiles_exact_identity_without_profiler_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "preflight").mkdir(parents=True)
    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setenv(runner._SNAPSHOT_MANIFEST_ENV, "9" * 64)
    monkeypatch.setenv(runner._DIAG4_CHILD_ENV, "1")

    class FakePreparedNeqGntr3:
        pass

    prepared = FakePreparedNeqGntr3()
    prepared.identity = SimpleNamespace(
        schema_version=runner.NEQ_GNTR3_SCHEMA_VERSION,
        route=runner.NEQ_GNTR3_ROUTE,
        base_neq_gntr1_policy_sha256="1" * 64,
        problem_sha256="2" * 64,
        optimizer_options_sha256="3" * 64,
        scaling_sha256="4" * 64,
        bootstrap_state_sha256="5" * 64,
        initial_physical_state_sha256="6" * 64,
        identity_sha256="7" * 64,
    )
    prepared.initial_optimizer_coordinates = object()
    prepared.options = runner.NEQ_GNTR3_OPTIONS
    prepared.policy = SimpleNamespace(
        policy_sha256="1" * 64,
        native_raw_equalities=np.zeros(255),
        constraint_inverse_scale=np.ones(255),
        state_size=716,
        equality_size=255,
        objective_residual_size=2110,
    )
    worker = SimpleNamespace(worker=SimpleNamespace(route=prepared))
    preparation_modes: list[bool] = []
    timestamps = iter((1, 2, 3))

    monkeypatch.setattr(runner, "PreparedNeqGntr3", FakePreparedNeqGntr3)
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: next(timestamps))
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(runner.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(runner.jax, "device_get", lambda value: value)
    monkeypatch.setattr(
        runner.jax.profiler,
        "start_trace",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 preflight must not profile"),
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "stop_trace",
        lambda: pytest.fail("DIAG4 preflight must not profile"),
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "trace",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 preflight must not profile"),
        raising=False,
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "TraceAnnotation",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 preflight must not annotate"),
    )

    def prepare(*_args: object, diag4: bool = False) -> object:
        preparation_modes.append(diag4)
        return worker

    monkeypatch.setattr(runner, "_prepare_diagnostic_worker", prepare)
    monkeypatch.setattr(runner, "_compiled_diagnostic_callback_count", lambda _w: 0)
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "preflight/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_publish_diagnostic_policy",
        lambda *_args, **_kwargs: runner.ArtifactRef(
            "preflight/policy.json", "b" * 64, 1, "policy-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_worker_runtime_payload",
        lambda: {
            "backend": "gpu",
            "device": "gpu",
            "device_uuid": runner.GPU_UUID,
            "jax": "test",
            "jax_enable_x64": True,
            "jaxlib": "test",
            "python": "test",
        },
    )

    payload = runner.run_snapshot_diagnostic_preflight_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert preparation_modes == [True]
    assert payload["schema_version"] == runner.DIAG4_PREFLIGHT_SCHEMA_VERSION
    assert payload["numerical_route"] == runner.NEQ_GNTR3_ROUTE
    assert payload["numerical_result_schema_version"] == runner.NEQ_GNTR3_SCHEMA_VERSION
    assert payload["mode"] == "TRACE_FREE_COMPILE_ONLY"
    assert payload["optimizer_options_sha256"] == "3" * 64
    assert payload["bootstrap_state_sha256"] == "5" * 64
    assert payload["initial_physical_state_sha256"] == "6" * 64
    assert payload["source_manifest_sha256"] == "9" * 64
    assert payload["profiler_start_calls"] == 0
    assert payload["profiler_stop_calls"] == 0
    assert payload["trace_normalization_calls"] == 0


def test_diag4_child_implementations_have_no_profiler_or_xplane_call_site() -> None:
    forbidden_call_sites = (
        "jax.profiler.",
        "normalize_chrome_trace(",
        "trace_session(",
        "TraceAnnotation(",
        "xplane",
        "raw-trace",
    )
    for implementation in (
        runner._run_snapshot_diag4_child,
        runner.run_snapshot_diagnostic_preflight_child,
    ):
        source = inspect.getsource(implementation)
        assert all(token not in source for token in forbidden_call_sites)


def test_diag2_cold_compile_failure_is_typed_before_profiler_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    (campaign / "cold").mkdir(parents=True)
    runtime_reference = runner.ArtifactRef(
        "cold/runtime-evidence.json", "b" * 64, 1, "runtime-v1"
    )
    monkeypatch.setenv(runner._DIAG2_CHILD_ENV, "1")
    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(
        runner, "_publish_child_runtime_evidence", lambda _sample: runtime_reference
    )
    monkeypatch.setattr(
        runner,
        "_worker_runtime_payload",
        lambda: {
            "backend": "gpu",
            "device": "gpu",
            "device_uuid": runner.GPU_UUID,
            "jax": "test",
            "jax_enable_x64": True,
            "jaxlib": "test",
            "python": "test",
        },
    )
    monkeypatch.setattr(
        runner,
        "_prepare_diagnostic_worker",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("compile failed")),
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "start_trace",
        lambda *_args, **_kwargs: pytest.fail("compile failure must not profile"),
    )

    payload = runner.run_snapshot_diagnostic_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "COMPILE_FAILURE"
    assert runner.validate_diag3_producer_payload(payload, mode="cold") == payload


def test_diag4_cold_is_trace_free_and_stages_timing_and_safeguard_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    campaign = tmp_path / "campaign"
    cold = campaign / "cold"
    cold.mkdir(parents=True)
    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setenv(runner._SNAPSHOT_MANIFEST_ENV, "9" * 64)
    monkeypatch.setenv(runner._DIAG4_PROCESS_STARTED_ENV, "1")

    class FakePreparedNeqGntr3:
        pass

    identity = SimpleNamespace(
        schema_version=runner.NEQ_GNTR3_SCHEMA_VERSION,
        route=runner.NEQ_GNTR3_ROUTE,
        base_neq_gntr1_policy_sha256="1" * 64,
        problem_sha256="2" * 64,
        optimizer_options_sha256="3" * 64,
        scaling_sha256="4" * 64,
        bootstrap_state_sha256="5" * 64,
        initial_physical_state_sha256="6" * 64,
        identity_sha256="7" * 64,
    )
    base = SimpleNamespace(
        optimizer_result=SimpleNamespace(
            optimizer_coordinates=object(), multipliers=object()
        )
    )
    prepared = FakePreparedNeqGntr3()
    prepared.identity = identity
    prepared.options = runner.NEQ_GNTR3_OPTIONS
    prepared.initial_optimizer_coordinates = object()
    prepared.policy = SimpleNamespace(
        policy_sha256="1" * 64,
        native_raw_equalities=object(),
        constraint_inverse_scale=object(),
    )
    prepared.problem = SimpleNamespace(
        config=SimpleNamespace(
            non_qs_weight=object(),
            residual_weight=object(),
            iota_weight=object(),
            major_radius_weight=object(),
            length_weight=object(),
        )
    )
    prepared.scaling = SimpleNamespace(
        bootstrap_anchor=object(), variable_scale=object()
    )
    prepared.finalize_result = lambda _loop: events.append("finalizer") or base
    loop_result = SimpleNamespace(
        accepted_optimizer_coordinates=object(), accepted_state_mask=object()
    )
    worker = SimpleNamespace(
        worker=SimpleNamespace(route=prepared),
        accepted_quality=SimpleNamespace(
            run=lambda *_args: events.append("replay") or object()
        ),
        terminal=SimpleNamespace(
            run_evidence=lambda *_args: (
                events.append("endpoint") or SimpleNamespace(raw_endpoint=object())
            )
        ),
    )
    host_loop = SimpleNamespace(
        history=SimpleNamespace(
            nonlinear_corrections=np.zeros(300, dtype=np.int32),
            maximum_individual_correction_step_ratio=np.full(300, np.nan),
            correction_path_step_ratio=np.full(300, np.nan),
            steihaug_solve_calls=np.zeros(300, dtype=np.int32),
            subtrial_count=np.zeros(300, dtype=np.int32),
            selected_subtrial_index=np.full(300, -1, dtype=np.int32),
            subtrial_trust_radius=np.full((300, 3), np.nan),
            subtrial_outcome=np.zeros((300, 3), dtype=np.int32),
            subtrial_actual_reduction=np.full((300, 3), np.nan),
            subtrial_predicted_reduction=np.full((300, 3), np.nan),
            subtrial_maximum_individual_correction_step_ratio=np.full((300, 3), np.nan),
            subtrial_correction_path_step_ratio=np.full((300, 3), np.nan),
            subtrial_corrected_radius_ratio=np.full((300, 3), np.nan),
            subtrial_steihaug_iterations=np.zeros((300, 3), dtype=np.int32),
            subtrial_steihaug_hvp_evaluations=np.zeros((300, 3), dtype=np.int32),
            subtrial_steihaug_solve_calls=np.zeros((300, 3), dtype=np.int32),
            subtrial_total_hvp_evaluations=np.zeros((300, 3), dtype=np.int32),
            subtrial_nonlinear_corrections=np.zeros((300, 3), dtype=np.int32),
            subtrial_joint_evaluations=np.zeros((300, 3), dtype=np.int32),
            subtrial_joint_linearizations=np.zeros((300, 3), dtype=np.int32),
            subtrial_joint_value_evaluations=np.zeros((300, 3), dtype=np.int32),
            subtrial_objective_residual_linearizations=np.zeros(
                (300, 3), dtype=np.int32
            ),
            subtrial_gram_factorizations=np.zeros((300, 3), dtype=np.int32),
            subtrial_gram_solves=np.zeros((300, 3), dtype=np.int32),
        )
    )
    host_diagnostic = SimpleNamespace(
        base_result=SimpleNamespace(loop_result=host_loop)
    )
    history = {
        "rows": [{"outcome": "INACTIVE"} for _ in range(300)],
        "attempts": 0,
        "accepted_steps": 0,
        "retryable_rejections": 0,
        "status": "ATTEMPT_LIMIT",
        "quality_latch": False,
    }
    timestamp = iter((2, 5, 6, 7, 8))

    monkeypatch.setattr(runner, "PreparedNeqGntr3", FakePreparedNeqGntr3)
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: next(timestamp))
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(
        runner.jax,
        "block_until_ready",
        lambda value: events.append("block") or value,
    )

    def device_get(value: object) -> object:
        if isinstance(value, tuple) and len(value) == 8:
            events.append("d2h")
            return (
                host_diagnostic,
                object(),
                object(),
                np.zeros(255),
                np.ones(255),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                np.zeros(716),
                np.ones(716),
            )
        return value

    monkeypatch.setattr(runner.jax, "device_get", device_get)
    monkeypatch.setattr(
        runner.jax.profiler,
        "start_trace",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 must not start a profiler"),
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "stop_trace",
        lambda: pytest.fail("DIAG4 must not stop a profiler"),
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "trace",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 must not profile"),
        raising=False,
    )
    monkeypatch.setattr(
        runner.jax.profiler,
        "TraceAnnotation",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 must not annotate"),
    )
    monkeypatch.setattr(
        runner,
        "normalize_chrome_trace",
        lambda *_args, **_kwargs: pytest.fail("DIAG4 must not normalize a trace"),
    )
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "cold/runtime-evidence.json", "a" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "read_linux_process_identity",
        lambda _pid: SimpleNamespace(pid=42, start_ticks=99),
    )
    monkeypatch.setattr(
        runner, "_prepare_diagnostic_worker", lambda *_args, **_kwargs: worker
    )
    monkeypatch.setattr(runner, "_compiled_diagnostic_callback_count", lambda _w: 0)
    monkeypatch.setattr(
        runner,
        "execute_timed_loop",
        lambda _route: events.append("loop") or (loop_result, 3, 4),
    )
    monkeypatch.setattr(
        runner,
        "build_native_equivalent_terminal_diagnostic",
        lambda *_args: object(),
    )
    monkeypatch.setattr(runner, "_diagnostic_history_payload", lambda _loop: history)
    monkeypatch.setattr(
        runner,
        "_publish_diagnostic_terminal",
        lambda *_args, **_kwargs: (
            runner.ArtifactRef(
                "cold/numerical-result/terminal-numerical.json",
                "b" * 64,
                1,
                f"{runner.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal",
            ),
            {},
        ),
    )
    monkeypatch.setattr(
        runner,
        "_publish_diagnostic_policy",
        lambda *_args, **_kwargs: runner.ArtifactRef(
            "cold/policy.json", "c" * 64, 1, "policy-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_worker_runtime_payload",
        lambda: {
            "backend": "gpu",
            "device": "gpu",
            "device_uuid": runner.GPU_UUID,
            "jax": "test",
            "jax_enable_x64": True,
            "jaxlib": "test",
            "python": "test",
        },
    )

    payload = runner._run_snapshot_diag4_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["schema_version"] == runner.DIAG4_COLD_RESULT_SCHEMA_VERSION
    assert payload["numerical_route"] == runner.NEQ_GNTR3_ROUTE
    assert payload["numerical_result_schema_version"] == runner.NEQ_GNTR3_SCHEMA_VERSION
    assert payload["profiler_enabled"] is False
    assert payload["profiler_start_calls"] == 0
    assert payload["profiler_stop_calls"] == 0
    assert payload["trace_normalization_calls"] == 0
    assert payload["endpoint_audit_called"] is True
    assert events == [
        "block",
        "loop",
        "finalizer",
        "block",
        "replay",
        "block",
        "endpoint",
        "block",
        "block",
        "d2h",
    ]
    pending = cold / runner._DIAG3_NUMERICAL_PENDING_NAME
    timing = runner.load_canonical_json_bytes(
        (pending / "solve-timing.json").read_bytes()
    )
    telemetry = runner.load_canonical_json_bytes(
        (pending / "safeguard-telemetry.json").read_bytes()
    )
    assert timing["synchronized_solve_seconds"] == 1.0e-9
    assert timing["numerical_route"] == runner.NEQ_GNTR3_ROUTE
    assert timing["numerical_result_schema_version"] == runner.NEQ_GNTR3_SCHEMA_VERSION
    assert telemetry["nonlinear_corrections"]["values"] == [0] * 300
    assert telemetry["steihaug_solve_calls"]["values"] == [0] * 300
    assert telemetry["numerical_route"] == runner.NEQ_GNTR3_ROUTE
    assert (
        telemetry["numerical_result_schema_version"] == runner.NEQ_GNTR3_SCHEMA_VERSION
    )
    assert telemetry["subtrial_count"]["values"] == [0] * 300
    assert telemetry["selected_subtrial_index"]["values"] == [-1] * 300
    assert telemetry["subtrial_outcome"]["values"] == [[0] * 3 for _ in range(300)]


@pytest.mark.parametrize(
    "trace_normalization_outcome",
    ["complete", "expected-failure", "unexpected-failure"],
)
def test_diagnostic_cold_stops_profiler_before_finalizer_and_one_d2h(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    trace_normalization_outcome: str,
) -> None:
    events: list[str] = []
    campaign = tmp_path / "campaign"
    cold = campaign / "cold"
    cold.mkdir(parents=True)
    route = SimpleNamespace(
        policy=SimpleNamespace(
            policy_sha256="a" * 64,
            native_raw_equalities=object(),
            constraint_inverse_scale=object(),
        ),
        problem=SimpleNamespace(
            config=SimpleNamespace(
                non_qs_weight=object(),
                residual_weight=object(),
                iota_weight=object(),
                major_radius_weight=object(),
                length_weight=object(),
            )
        ),
        scaling=SimpleNamespace(bootstrap_anchor=object(), variable_scale=object()),
        initial_optimizer_coordinates=object(),
        finalize_result=lambda _loop: events.append("finalizer") or base,
    )
    accepted_quality = SimpleNamespace(
        run=lambda *_args: events.append("replay") or object()
    )
    terminal_evidence = SimpleNamespace(raw_endpoint=object())
    terminal = SimpleNamespace(
        run_evidence=lambda *_args: events.append("terminal") or terminal_evidence
    )
    worker = SimpleNamespace(
        worker=SimpleNamespace(route=route),
        accepted_quality=accepted_quality,
        terminal=terminal,
    )
    base = SimpleNamespace(
        optimizer_result=SimpleNamespace(
            optimizer_coordinates=object(), multipliers=object()
        )
    )
    diagnostic = SimpleNamespace(base_result=SimpleNamespace(loop_result=object()))
    loop_result = SimpleNamespace(
        accepted_optimizer_coordinates=object(),
        accepted_state_mask=object(),
    )
    trace_path: list[Path] = []

    def start_trace(path: str, **_kwargs: object) -> None:
        events.append("profiler-start")
        target = Path(path) / "plugins/profile/run.trace.json.gz"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"trace")
        trace_path.append(target)

    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setenv(runner._DIAG2_CHILD_ENV, "1")
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(runner.jax.profiler, "start_trace", start_trace)
    monkeypatch.setattr(
        runner.jax.profiler, "stop_trace", lambda: events.append("profiler-stop")
    )
    monkeypatch.setattr(
        runner.jax, "block_until_ready", lambda value: events.append("block") or value
    )
    monkeypatch.setattr(
        runner.jax,
        "device_get",
        lambda _value: (
            events.append("d2h")
            or (
                diagnostic,
                object(),
                terminal_evidence,
                object(),
                object(),
                (1.0, 1.0, 1.0, 1.0, 1.0),
                object(),
                object(),
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "cold/runtime-evidence.json", "b" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(runner, "_prepare_diagnostic_worker", lambda *_args: worker)
    monkeypatch.setattr(runner, "_compiled_diagnostic_callback_count", lambda _w: 0)
    monkeypatch.setattr(
        runner,
        "execute_timed_loop",
        lambda _prepared, **_kwargs: (
            events.append("loop")
            or (
                loop_result,
                runner.time.perf_counter_ns(),
                runner.time.perf_counter_ns(),
            )
        ),
    )
    monkeypatch.setattr(
        runner,
        "build_native_equivalent_terminal_diagnostic",
        lambda *_args: events.append("builder") or object(),
    )
    monkeypatch.setattr(
        runner, "_diagnostic_history_payload", lambda _loop: {"history": True}
    )
    monkeypatch.setattr(
        runner,
        "_publish_diagnostic_terminal",
        lambda *_args, **_kwargs: (
            runner.ArtifactRef(
                "cold/numerical-result/terminal-numerical.json",
                "c" * 64,
                1,
                "terminal-v1",
            ),
            {},
        ),
    )

    def normalize_trace(*_args: object, **_kwargs: object) -> dict[str, bool]:
        if trace_normalization_outcome == "expected-failure":
            raise ValueError("raw Chrome trace contains no in-envelope intervals")
        if trace_normalization_outcome == "unexpected-failure":
            raise RuntimeError("unexpected normalization implementation failure")
        return {"trace": True}

    monkeypatch.setattr(runner, "normalize_chrome_trace", normalize_trace)
    monkeypatch.setattr(
        runner,
        "policy_evidence_payload",
        lambda **_kwargs: {
            "schema_version": f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy"
        },
    )

    if trace_normalization_outcome == "unexpected-failure":
        with pytest.raises(
            RuntimeError, match="unexpected normalization implementation failure"
        ):
            runner.run_snapshot_diagnostic_child(
                reference_root=Path("reference"), input_root=Path("input")
            )
        assert (cold / runner._DIAG3_NUMERICAL_PENDING_NAME).is_dir()
        return
    payload = runner.run_snapshot_diagnostic_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert trace_path
    assert events == [
        "block",
        "profiler-start",
        "loop",
        "profiler-stop",
        "finalizer",
        "block",
        "replay",
        "block",
        "terminal",
        "block",
        "builder",
        "d2h",
    ]
    assert payload["transfer_audit"]["final_d2h_transfers"] == 1
    assert payload["endpoint_audit_called"] is False
    assert payload["schema_version"] == runner.DIAG3_COLD_RESULT_SCHEMA_VERSION
    assert payload["execution_status"] == (
        "TRACE_NORMALIZATION_FAILED"
        if trace_normalization_outcome == "expected-failure"
        else "COMPLETE"
    )
    assert payload["history_evidence"]["relative_path"] == (
        "cold/numerical-result/history.json"
    )
    assert (cold / runner._DIAG3_NUMERICAL_PENDING_NAME).is_dir()
    timestamps = payload["timestamps_ns"]
    assert isinstance(timestamps, dict)
    ordered = tuple(timestamps.values())
    assert ordered == tuple(sorted(ordered))


def test_diagnostic_cold_timeout_stops_profiler_without_finalizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    campaign = tmp_path / "campaign"
    (campaign / "cold").mkdir(parents=True)
    route = SimpleNamespace(
        policy=SimpleNamespace(policy_sha256="a" * 64),
        initial_optimizer_coordinates=object(),
        finalize_result=lambda _loop: events.append("finalizer"),
    )
    worker = SimpleNamespace(worker=SimpleNamespace(route=route))
    monkeypatch.setenv(runner._CAMPAIGN_ROOT_ENV, str(campaign))
    monkeypatch.setattr(runner.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(runner.jax, "devices", lambda: ["gpu"])
    monkeypatch.setattr(runner.jax, "block_until_ready", lambda value: value)
    monkeypatch.setattr(
        runner.jax.profiler,
        "start_trace",
        lambda *_args, **_kwargs: events.append("start"),
    )
    monkeypatch.setattr(
        runner.jax.profiler, "stop_trace", lambda: events.append("stop")
    )
    monkeypatch.setattr(
        runner,
        "_publish_child_runtime_evidence",
        lambda _sample: runner.ArtifactRef(
            "cold/runtime-evidence.json", "b" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(runner, "_prepare_diagnostic_worker", lambda *_args: worker)
    monkeypatch.setattr(runner, "_compiled_diagnostic_callback_count", lambda _w: 0)
    monkeypatch.setattr(
        runner,
        "execute_timed_loop",
        lambda _route, **_kwargs: (_ for _ in ()).throw(
            runner.SolveTimeoutError("bounded")
        ),
    )

    payload = runner.run_snapshot_diagnostic_child(
        reference_root=Path("reference"), input_root=Path("input")
    )

    assert payload["execution_status"] == "TIMEOUT"
    assert payload["transfer_audit"]["final_d2h_transfers"] == 0
    assert events == ["start", "stop"]


def test_diagnostic_terminal_publishes_complete_replay_and_raw_contract(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    cold = campaign / "cold"
    cold.mkdir(parents=True)
    z716 = np.zeros(716, dtype=np.float64)
    z255 = np.zeros(255, dtype=np.float64)
    z257x716 = np.zeros((257, 716), dtype=np.float64)
    z257x255 = np.zeros((257, 255), dtype=np.float64)
    certificate = SimpleNamespace(
        **{name: np.asarray(0.0) for name in runner.FINAL_CERTIFICATE_FIELDS}
    )
    transpose = SimpleNamespace(
        state_probe=z716,
        equality_probe=z255,
        jvp_action=z255,
        vjp_action=z716,
        primal_dot=np.asarray(0.0),
        transpose_dot=np.asarray(0.0),
        denominator=np.asarray(1.0),
        defect=np.asarray(0.0),
    )
    endpoint = SimpleNamespace(
        physical_state=z716,
        raw_equalities=z255,
        scaled_equalities=z255,
        objective_gradient=z716,
        transpose_certificate=transpose,
        evaluation=SimpleNamespace(
            weighted_total=np.asarray(0.0),
            raw_terms=SimpleNamespace(
                non_qs=np.asarray(0.0),
                residual=np.asarray(0.0),
                iota=np.asarray(0.0),
                major_radius=np.asarray(0.0),
                length=np.asarray(0.0),
            ),
            observables=SimpleNamespace(
                iota=np.asarray(0.1),
                G=np.asarray(0.2),
                volume=np.asarray(0.3),
                major_radius=np.asarray(0.4),
                total_length=np.asarray(0.5),
                non_qs_ratio=np.asarray(0.6),
                boozer_residual_scalar=np.asarray(0.7),
                boozer_residual_rms=np.asarray(0.8),
            ),
        ),
    )
    optimizer = SimpleNamespace(
        optimizer_coordinates=z716,
        multipliers=z255,
        constraint_jacobian=np.zeros((255, 716), dtype=np.float64),
        scaled_stationarity_inf=np.asarray(0.0),
        final_certificate=certificate,
    )
    base = SimpleNamespace(
        optimizer_result=optimizer,
        endpoint=endpoint,
        loop_result=SimpleNamespace(accepted_optimizer_coordinates=z257x716),
        accepted_physical_coordinates=z257x716,
        accepted_state_mask=np.ones(257, dtype=np.bool_),
    )
    diagnostic = SimpleNamespace(
        base_result=base,
        raw_kkt_status=np.asarray(0, dtype=np.int32),
    )
    replay = SimpleNamespace(
        objectives=np.zeros(257, dtype=np.float64),
        raw_equalities=z257x255,
        scaled_equalities=z257x255,
        accepted_state_mask=np.ones(257, dtype=np.bool_),
        coordinates_finite=np.ones(257, dtype=np.bool_),
        objective_finite=np.ones(257, dtype=np.bool_),
        raw_equalities_finite=np.ones(257, dtype=np.bool_),
        scaled_equalities_finite=np.ones(257, dtype=np.bool_),
        objective_satisfied=np.ones(257, dtype=np.bool_),
        component_bounds_satisfied=np.ones(257, dtype=np.bool_),
        scaled_feasibility_satisfied=np.ones(257, dtype=np.bool_),
        quality_satisfied=np.ones(257, dtype=np.bool_),
    )
    terminal = SimpleNamespace(
        raw_endpoint=SimpleNamespace(
            raw_stationarity_residual=z716,
            raw_kkt_stationarity_infinity_norm=np.asarray(0.0),
        ),
        objective_residual_vector=np.zeros(2110, dtype=np.float64),
        reconstructed_objective=np.asarray(0.0),
        reconstructed_objective_gradient=z716,
        authoritative_objective=np.asarray(0.0),
        authoritative_objective_gradient=z716,
        value_scaled_defect=np.asarray(0.0),
        gradient_scaled_defect=np.asarray(0.0),
    )

    terminal_ref, arrays = runner._publish_diagnostic_terminal(
        cold,
        campaign,
        diagnostic,
        replay,
        terminal,
        z255,
        np.ones(255, dtype=np.float64),
        z716,
        np.ones(716, dtype=np.float64),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        terminal_seconds=1.0,
    )

    assert frozenset(arrays) == frozenset(runner.DIAGNOSTIC_ARRAY_SPECS)
    assert terminal_ref.relative_path == "cold/terminal-numerical.json"
    payload = runner.load_canonical_json_bytes(
        (cold / "terminal-numerical.json").read_bytes()
    )
    assert isinstance(payload, dict)
    assert frozenset(payload["arrays"]) == frozenset(runner.DIAGNOSTIC_ARRAY_SPECS)

    diag4_cold = campaign / "diag4-cold"
    diag4_cold.mkdir()
    numerical_identity = runner.NativeEquivalentNumericalIdentity(
        numerical_route=runner.DIAG4_NUMERICAL_ROUTE,
        numerical_result_schema_version=runner.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION,
        problem_sha256="1" * 64,
        optimizer_options_sha256="2" * 64,
        base_neq_gntr1_policy_sha256="3" * 64,
        scaling_sha256="4" * 64,
        bootstrap_state_sha256="5" * 64,
        initial_physical_state_sha256="6" * 64,
        identity_sha256="7" * 64,
    )
    diag4_ref, _diag4_arrays = runner._publish_diagnostic_terminal(
        diag4_cold,
        campaign,
        diagnostic,
        replay,
        terminal,
        z255,
        np.ones(255, dtype=np.float64),
        z716,
        np.ones(716, dtype=np.float64),
        (1.0, 1.0, 1.0, 1.0, 1.0),
        terminal_seconds=1.0,
        numerical_identity=numerical_identity,
    )
    diag4_payload = runner.load_canonical_json_bytes(
        (diag4_cold / "terminal-numerical.json").read_bytes()
    )
    assert diag4_ref.schema_version == (
        f"{runner.DIAG4_NUMERICAL_RESULT_SCHEMA_VERSION}-terminal"
    )
    assert diag4_payload["numerical_route"] == runner.DIAG4_NUMERICAL_ROUTE
    assert frozenset(diag4_payload["terminal_observables"]) == frozenset(
        {
            "iota",
            "G",
            "volume",
            "major_radius",
            "total_length",
            "non_qs_ratio",
            "boozer_residual_value",
            "boozer_residual_rms",
        }
    )
    assert (
        diag4_payload["terminal_observables"] == diag4_payload["endpoint_observables"]
    )


class _RecordingContext:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __enter__(self) -> None:
        self.events.append("guard-enter")

    def __exit__(self, *_args: object) -> None:
        self.events.append("guard-exit")


def test_child_invocation_is_snapshot_isolated_and_sample_specific(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.WARM_2,
        environment={
            "CUDA_VISIBLE_DEVICES": runner.GPU_UUID,
            "JAX_PLATFORMS": "cpu",
            "JAX_PLATFORM_NAME": "cpu",
        },
    )

    assert invocation.cwd == snapshot
    assert invocation.argv[:3] == (str(interpreter), "-I", str(runner_path))
    assert invocation.argv[invocation.argv.index("--sample") + 1] == "warm-2"
    assert invocation.environment[runner._SNAPSHOT_MANIFEST_ENV] == "a" * 64
    assert invocation.environment[runner._CAMPAIGN_ROOT_ENV] == str(campaign)
    assert invocation.environment["JAX_PLATFORMS"] == "cuda"
    assert "JAX_PLATFORM_NAME" not in invocation.environment


def test_preflight_invocation_selects_internal_compile_only_child(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.COLD,
        environment={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID},
        preflight_only=True,
    )

    assert "--snapshot-child" in invocation.argv
    assert "--preflight-child" in invocation.argv
    assert "--preflight-only" not in invocation.argv
    assert invocation.environment["JAX_PLATFORMS"] == "cuda"


def test_diagnostic_invocation_freezes_cache_profiler_and_allocator_environment(
    tmp_path: Path,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.COLD,
        environment={
            "CUDA_VISIBLE_DEVICES": runner.GPU_UUID,
            "JAX_COMPILATION_CACHE_DIR": "/tmp/untrusted-cache",
            "JAX_ENABLE_COMPILATION_CACHE": "true",
            runner.TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: "1",
            "XLA_FLAGS": "--unrelated_xla_flag=true",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        diagnostic_mode=runner.DiagnosticChildMode.PREFLIGHT,
    )

    assert "JAX_COMPILATION_CACHE_DIR" not in invocation.environment
    assert invocation.environment["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert invocation.environment[runner.TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT] == str(
        runner.TRACE_VIEWER_MAX_EVENTS
    )
    assert invocation.environment["XLA_FLAGS"] == "--unrelated_xla_flag=true"
    assert invocation.environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


@pytest.mark.parametrize(
    ("caller_flags", "expected_flags"),
    [
        ("", "--xla_gpu_enable_command_buffer="),
        (
            "--unrelated_xla_flag=true",
            "--unrelated_xla_flag=true --xla_gpu_enable_command_buffer=",
        ),
        (
            "--unrelated_xla_flag=true --xla_gpu_enable_command_buffer=kernel",
            "--unrelated_xla_flag=true --xla_gpu_enable_command_buffer=",
        ),
    ],
)
@pytest.mark.parametrize(
    "mode",
    [runner.DiagnosticChildMode.PREFLIGHT, runner.DiagnosticChildMode.COLD],
)
def test_diag2_invocation_disables_command_buffers_without_losing_other_xla_flags(
    tmp_path: Path,
    caller_flags: str,
    expected_flags: str,
    mode: runner.DiagnosticChildMode,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.COLD,
        environment={
            "CUDA_VISIBLE_DEVICES": runner.GPU_UUID,
            "XLA_FLAGS": caller_flags,
        },
        diagnostic_mode=mode,
        diag2=True,
    )

    assert invocation.environment["XLA_FLAGS"] == expected_flags
    assert invocation.environment[runner._DIAG2_CHILD_ENV] == "1"
    assert invocation.environment["JAX_ENABLE_X64"] == "true"


@pytest.mark.parametrize(
    "mode",
    [runner.DiagnosticChildMode.PREFLIGHT, runner.DiagnosticChildMode.COLD],
)
def test_diag4_invocation_is_trace_free_and_uses_distinct_child_identity(
    tmp_path: Path,
    mode: runner.DiagnosticChildMode,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.COLD,
        environment={
            "CUDA_VISIBLE_DEVICES": runner.GPU_UUID,
            "XLA_FLAGS": "--unrelated_xla_flag=true",
            runner.TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT: "123",
        },
        diagnostic_mode=mode,
        diag4=True,
    )

    assert invocation.environment[runner._DIAG4_CHILD_ENV] == "1"
    assert runner._DIAG2_CHILD_ENV not in invocation.environment
    assert runner.TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT not in invocation.environment
    assert invocation.environment["XLA_FLAGS"] == (
        "--unrelated_xla_flag=true --xla_gpu_enable_command_buffer="
    )
    assert invocation.environment["JAX_ENABLE_COMPILATION_CACHE"] == "false"
    assert invocation.environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"


@pytest.mark.parametrize(
    "mode",
    [runner.DiagnosticChildMode.PREFLIGHT, runner.DiagnosticChildMode.COLD],
)
def test_diag5_invocation_binds_installed_native_identity(
    tmp_path: Path,
    mode: runner.DiagnosticChildMode,
) -> None:
    snapshot = tmp_path / "snapshot"
    runner_path = snapshot / runner._ENTRYPOINT
    runner_path.parent.mkdir(parents=True)
    runner_path.write_text("pass\n")
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "inputs"
    input_root.mkdir()
    native = tmp_path / "installed" / "simsoptpp.so"
    native.parent.mkdir()
    native.write_bytes(b"installed-native")
    digest = hashlib.sha256(native.read_bytes()).hexdigest()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)

    invocation = runner.build_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=interpreter,
        reference_root=reference,
        input_root=input_root,
        sample=SampleName.COLD,
        environment={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID},
        diagnostic_mode=mode,
        diag5=True,
        expected_native_extension_path=native.resolve(),
        expected_native_extension_sha256=digest,
        expected_native_extension_size_bytes=native.stat().st_size,
        expected_native_extension_link_count=native.stat().st_nlink,
    )

    assert invocation.environment[runner._DIAG5_CHILD_ENV] == "1"
    assert runner._DIAG4_CHILD_ENV not in invocation.environment
    assert invocation.environment[runner._DIAG5_NATIVE_PATH_ENV] == str(
        native.resolve()
    )
    assert invocation.environment[runner._DIAG5_NATIVE_SHA256_ENV] == digest
    assert invocation.environment[runner._DIAG5_NATIVE_SIZE_ENV] == str(
        native.stat().st_size
    )
    assert invocation.environment[runner._DIAG5_NATIVE_LINK_COUNT_ENV] == str(
        native.stat().st_nlink
    )
    assert runner.TRACE_VIEWER_MAX_EVENTS_ENVIRONMENT not in invocation.environment


def test_diag4_supervisor_injects_parent_process_start_for_child_timing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "child.py"),
        cwd=Path("snapshot"),
        environment={runner._DIAG4_CHILD_ENV: "1"},
    )
    observed: list[str] = []
    timestamps = iter((101, 909))
    monkeypatch.setattr(runner.time, "perf_counter_ns", lambda: next(timestamps))

    def supervise(
        _sample: SampleName,
        child_invocation: runner.SnapshotChildInvocation,
        **_kwargs: object,
    ) -> runner.SupervisedSample:
        observed.append(child_invocation.environment[runner._DIAG4_PROCESS_STARTED_ENV])
        return runner.SupervisedSample(
            sample=SampleName.COLD,
            terminal_status=runner.ChildTerminalStatus.COMPLETE,
            child_pid=7,
            child_start_time_ticks=8,
            process_seconds=1.0,
            producer={"execution_status": "COMPLETE"},
            memory={"peak_memory_fraction": 0.1},
            failure_reasons=(),
            observed_child_argv=child_invocation.argv,
            stdout=b"{}",
            stderr=b"",
            memory_samples=(),
            process_diagnostics={"returncode": 0},
        )

    monkeypatch.setattr(runner, "supervise_sample", supervise)

    outcome = runner.supervise_diag2_sample(
        SampleName.COLD,
        invocation,
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1,
        validate_producer=lambda value, **_kwargs: value,
    )

    assert observed == ["101"]
    assert outcome.process_started_monotonic_ns == 101
    assert outcome.process_stopped_monotonic_ns == 909


@pytest.mark.parametrize(
    "policy_mutation",
    (
        {"JAX_PLATFORMS": "cpu"},
        {"JAX_PLATFORM_NAME": "gpu"},
        {"JAX_COMPILATION_CACHE_DIR": "/tmp/forbidden-cache"},
        {"JAX_ENABLE_COMPILATION_CACHE": "true"},
        {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        {"JAX_ENABLE_X64": "false"},
        {"XLA_FLAGS": "--xla_gpu_enable_command_buffer=kernel"},
    ),
)
def test_diag2_snapshot_child_rejects_noncanonical_policy_before_jax(
    tmp_path: Path,
    policy_mutation: dict[str, str],
) -> None:
    environment = {
        **os.environ,
        runner._DIAG2_CHILD_ENV: "1",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG,
    }
    environment.pop("JAX_PLATFORM_NAME", None)
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    environment.update(policy_mutation)

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            "--snapshot-child",
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "DIAG2 child pre-import policy is not canonical" in completed.stderr
    assert "import jax" not in completed.stderr


def test_diag2_snapshot_child_accepts_frozen_command_buffer_policy_before_jax(
    tmp_path: Path,
) -> None:
    environment = {
        **os.environ,
        runner._DIAG2_CHILD_ENV: "1",
        "JAX_PLATFORMS": "cuda",
        "JAX_ENABLE_COMPILATION_CACHE": "false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "JAX_ENABLE_X64": "true",
        "XLA_FLAGS": XLA_GPU_COMMAND_BUFFER_DISABLE_FLAG,
    }
    environment.pop("JAX_PLATFORM_NAME", None)
    environment.pop("JAX_COMPILATION_CACHE_DIR", None)

    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            "--snapshot-child",
            "--help",
        ),
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--snapshot-child" in completed.stdout


def test_symlinked_venv_launcher_survives_policy_and_isolated_jax_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    launcher = repository / ".venv-qn-gpu/bin/python"
    assert launcher.is_symlink()
    reference = tmp_path / "reference"
    reference.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    monkeypatch.setattr(
        runner, "_physical_gpu_identity", lambda _environment: (runner.GPU_UUID, 1)
    )

    policy = runner._validate_parent_execution_policy(
        repo_root=repository,
        reference_root=reference,
        input_root=inputs,
        interpreter=launcher,
        environment={"CUDA_VISIBLE_DEVICES": runner.GPU_UUID},
    )

    assert policy[4] == launcher
    completed = subprocess.run(
        (str(policy[4]), "-I", "-c", "import jax; print(jax.__file__)"),
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        timeout=30.0,
    )
    assert str(repository / ".venv-qn-gpu") in completed.stdout


def test_real_diagnostic_snapshot_has_all_roles_and_imports_in_isolation(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    publication = runner.prepare_execution_snapshot(
        tmp_path / "diagnostic",
        repo_root=repository,
        native_extension_path=Path(runner.simsoptpp.__file__).resolve(strict=True),
    )
    roles = {entry.role for entry in publication.entries}

    assert roles == {
        "execution_source",
        "configuration",
        "benchmark",
        "test",
        "native_extension",
    }
    assert any(
        entry.relative_path
        == "docs/single_stage_jax_gpu_native_equivalent_quality_no_hit_diagnostic_implementation_plan.md"
        for entry in publication.entries
    )
    bound_paths = {entry.relative_path for entry in publication.entries}
    assert (
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diagnostic_receipt.py"
        in bound_paths
    )
    assert (
        "tests/benchmarks/test_single_stage_native_equivalent_quality_diag2_contract.py"
        in bound_paths
    )
    assert (
        "docs/single_stage_jax_gpu_native_equivalent_quality_diag2_implementation_plan.md"
        in bound_paths
    )
    assert "tests/geo/test_projected_gauss_newton_trust_region.py" in bound_paths
    completed = subprocess.run(
        (
            str(repository / ".venv-qn-gpu/bin/python"),
            "-I",
            str(publication.root / runner._ENTRYPOINT),
            "--help",
        ),
        cwd=publication.root,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert "--diagnostic-only" in completed.stdout


def test_cli_exposes_preflight_only_help() -> None:
    help_text = runner._parser().format_help()

    assert "--preflight-only" in help_text
    assert "without solving" in help_text


def test_cli_exposes_mutually_exclusive_diagnostic_mode() -> None:
    parser = runner._parser()
    help_text = parser.format_help()

    assert "--diagnostic-only" in help_text
    assert "--diagnostic-successor-authority" in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--reference",
                "reference",
                "--input-root",
                "input",
                "--preflight-only",
                "--diagnostic-only",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--reference",
                "reference",
                "--input-root",
                "input",
                "--diagnostic-only",
                "--diagnostic-successor-authority",
                "authority.json",
            ]
        )


@pytest.mark.parametrize(
    "supervisor_mode",
    (
        "--preflight-only",
        "--diagnostic-only",
        "--diagnostic-successor-authority",
    ),
)
def test_snapshot_child_rejects_every_supervisor_mode_composition(
    supervisor_mode: str,
    tmp_path: Path,
) -> None:
    arguments = [
        "--reference",
        str(tmp_path / "reference"),
        "--input-root",
        str(tmp_path / "inputs"),
        "--snapshot-child",
        "--sample",
        runner.SampleName.COLD.value,
        "--diagnostic-child",
        runner.DiagnosticChildMode.COLD.value,
        supervisor_mode,
    ]
    if supervisor_mode == "--diagnostic-successor-authority":
        arguments.append(str(tmp_path / "authority.json"))

    with pytest.raises(
        ValueError, match="snapshot child cannot combine a supervisor mode"
    ):
        runner.main(arguments)


@pytest.mark.parametrize(
    "supervisor_arguments",
    (
        ("--diagnostic-only",),
        ("--diagnostic-successor-authority", "authority.json"),
    ),
)
def test_snapshot_child_rejects_supervisor_mode_before_jax_import(
    supervisor_arguments: tuple[str, ...],
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            "--snapshot-child",
            *supervisor_arguments,
        ),
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "invalid-if-imported"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "snapshot child cannot combine a supervisor mode" in completed.stderr


def test_snapshot_child_rejects_equals_successor_mode_before_jax_import(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            "--snapshot-child",
            "--diagnostic-successor-authority=authority.json",
        ),
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "invalid-if-imported"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "snapshot child cannot combine a supervisor mode" in completed.stderr


@pytest.mark.parametrize(
    "abbreviated_option",
    (
        "--diagnostic-successor-authorit=authority.json",
        "--snap",
    ),
)
def test_protected_option_abbreviation_rejects_before_jax_import(
    abbreviated_option: str,
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            abbreviated_option,
        ),
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "invalid-if-imported"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "abbreviated protected option is forbidden" in completed.stderr


def test_legacy_diagnostic_rejects_before_jax_import(tmp_path: Path) -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(Path(runner.__file__).resolve()),
            "--diagnostic-only",
        ),
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "invalid-if-imported"},
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "legacy --diagnostic-only is not authorized" in completed.stderr


def test_diag3_cli_validates_authority_before_running_diag2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    authority = tmp_path / "authority.json"
    reference = tmp_path / "reference"
    inputs = tmp_path / "inputs"
    interpreter = tmp_path / "python"
    output = tmp_path / "output"

    class Claim:
        def __enter__(self) -> dict[str, object]:
            calls.append("claim")
            return {}

        def __exit__(
            self,
            _exception_type: object,
            _exception: object,
            _traceback: object,
        ) -> None:
            calls.append("release")

    monkeypatch.setattr(runner, "claim_successor_authority", lambda *_a, **_k: Claim())
    monkeypatch.setattr(
        runner,
        "run_diag2",
        lambda *_args, **_kwargs: calls.append("run") or {"schema_version": "summary"},
    )

    exit_code = runner.main(
        (
            "--output",
            str(output),
            "--reference",
            str(reference),
            "--input-root",
            str(inputs),
            "--interpreter",
            str(interpreter),
            "--diagnostic-successor-authority",
            str(authority),
        )
    )

    assert exit_code == 0
    assert calls == ["claim", "run", "release"]
    assert runner.json.loads(capsys.readouterr().out) == {"schema_version": "summary"}


def test_diag3_legacy_diagnostic_cli_cannot_bypass_successor_authority(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="not authorized"):
        runner.main(
            (
                "--output",
                str(output),
                "--reference",
                str(tmp_path / "reference"),
                "--input-root",
                str(tmp_path / "inputs"),
                "--diagnostic-only",
            )
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("mode_args", "expected_caller"),
    (((), "campaign"), (("--preflight-only",), "preflight")),
)
def test_diag2_addition_preserves_legacy_public_cli_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mode_args: tuple[str, ...],
    expected_caller: str,
) -> None:
    calls: list[str] = []
    outcome = _outcome(SampleName.COLD)
    monkeypatch.setattr(
        runner,
        "run_campaign",
        lambda *_args, **_kwargs: calls.append("campaign") or (outcome,),
    )
    monkeypatch.setattr(
        runner,
        "run_preflight",
        lambda *_args, **_kwargs: calls.append("preflight") or outcome,
    )
    arguments = (
        "--output",
        str(tmp_path / "output"),
        "--reference",
        str(tmp_path / "reference"),
        "--input-root",
        str(tmp_path / "inputs"),
        *mode_args,
    )

    assert runner.main(arguments) == 0
    assert calls == [expected_caller]
    assert capsys.readouterr().out


def test_diagnostic_invalid_gpu_policy_leaves_output_absent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"
    reference = tmp_path / "reference"
    inputs = tmp_path / "inputs"
    reference.mkdir()
    inputs.mkdir()
    monkeypatch.setattr(
        runner,
        "_physical_gpu_identity",
        lambda _environment: (_ for _ in ()).throw(ValueError("wrong UUID")),
    )

    with pytest.raises(ValueError, match="wrong UUID"):
        runner.run_diagnostic(
            output,
            reference_root=reference,
            input_root=inputs,
            interpreter=Path(sys.executable),
            environment={},
            repo_root=Path(__file__).resolve().parents[2],
        )

    assert not output.exists()


def test_diagnostic_failed_preflight_never_creates_cold_or_campaign_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"
    reference = tmp_path / "reference"
    inputs = tmp_path / "inputs"
    reference.mkdir()
    inputs.mkdir()
    source_identity = runner.SourceIdentityEvidence(
        git_head="a" * 40,
        tracked_diff_sha256="b" * 64,
        untracked_bytes_manifest_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
        source_manifest_size_bytes=1,
    )

    def prepare(path: Path, **_kwargs: object) -> object:
        path.mkdir()
        snapshot = path / runner.SOURCE_SNAPSHOT_DIRECTORY
        snapshot.mkdir()
        manifest = snapshot / "source-manifest.json"
        manifest.write_bytes(b"x")
        return SimpleNamespace(
            root=snapshot,
            manifest_sha256="d" * 64,
            source_identity=lambda _root: SimpleNamespace(
                snapshot_manifest=runner.ArtifactRef(
                    runner.SOURCE_MANIFEST_ARTIFACT, "d" * 64, 1, "snapshot-v1"
                )
            ),
        )

    def copy_reference(_source: Path, root: Path) -> Path:
        destination = root / "native-reference"
        destination.mkdir()
        (destination / runner.REFERENCE_FILENAME).write_bytes(b"{}")
        return destination

    outcome = runner.SupervisedSample(
        sample=SampleName.COLD,
        terminal_status=runner.ChildTerminalStatus.COMPLETE,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=1.0,
        producer={
            "execution_status": "SUCCESS",
            "runtime_evidence": {
                "relative_path": "preflight/runtime-evidence.json",
                "sha256": "e" * 64,
                "size_bytes": 1,
                "schema_version": "runtime-v1",
            },
        },
        memory={"peak_memory_fraction": 0.1},
        failure_reasons=(),
        observed_child_argv=("python", "runner"),
        stdout=b"{}",
        stderr=b"compile failed",
        memory_samples=(),
    )
    sealed: list[object] = []
    gate_calls: list[frozenset[str]] = []
    monkeypatch.setattr(runner, "prepare_execution_snapshot", prepare)
    monkeypatch.setattr(runner, "copy_validated_reference", copy_reference)
    monkeypatch.setattr(
        runner,
        "_publish_parent_policy_authority",
        lambda _reference, root: _publish_fake_policy_authority(root),
    )
    monkeypatch.setattr(runner, "_physical_gpu_identity", lambda _env: ("gpu", 1))
    monkeypatch.setattr(
        runner, "_capture_source_identity_evidence", lambda *_args: source_identity
    )
    monkeypatch.setattr(
        runner,
        "build_child_invocation",
        lambda *_args, **_kwargs: runner.SnapshotChildInvocation(
            ("python", "runner"), tmp_path, {}
        ),
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)
    monkeypatch.setattr(
        runner,
        "_published_runtime_reference",
        lambda *_args: runner.ArtifactRef(
            "preflight/runtime-evidence.json", "e" * 64, 1, "runtime-v1"
        ),
    )
    monkeypatch.setattr(
        runner,
        "_published_diagnostic_policy_reference",
        lambda *_args: runner.ArtifactRef(
            "preflight/policy.json",
            "f" * 64,
            1,
            f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy",
        ),
    )

    def reject_wrong_policy(_root: Path, **kwargs: object) -> bool:
        evidence_refs = kwargs["evidence_refs"]
        assert isinstance(evidence_refs, dict)
        gate_calls.append(frozenset(evidence_refs))
        raise ValueError("preflight policy SHA differs from recomputed raw policy")

    monkeypatch.setattr(
        runner,
        "validate_diagnostic_preflight_gate",
        reject_wrong_policy,
    )
    monkeypatch.setattr(
        runner,
        "_seal_diagnostic_receipt",
        lambda _root, receipt: sealed.append(receipt),
    )
    incomplete = SimpleNamespace(verdict="DIAGNOSTIC_INCOMPLETE")
    monkeypatch.setattr(
        runner,
        "build_incomplete_diagnostic_receipt",
        lambda *, artifact_root, evidence_refs: (
            incomplete
            if artifact_root == output and evidence_refs
            else pytest.fail("incomplete receipt received the wrong authority")
        ),
    )

    summary = runner.run_diagnostic(
        output,
        reference_root=reference,
        input_root=inputs,
        interpreter=Path(sys.executable),
        environment={},
        repo_root=Path(__file__).resolve().parents[2],
    )

    assert summary["children"] == ["preflight"]
    assert gate_calls == [runner.PREFLIGHT_EVIDENCE_REF_KEYS]
    assert sealed
    assert not (output / "cold").exists()
    assert not (output / runner.CAMPAIGN_RECEIPT_FILENAME).exists()


def test_all_reference_semantic_failure_derives_and_seals_incomplete_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    evidence_refs = {
        name: runner.ArtifactRef(f"{name}.json", "a" * 64, 1, "test-v1")
        for name in runner.DIAGNOSTIC_EVIDENCE_REF_KEYS
    }
    incomplete = SimpleNamespace(verdict="DIAGNOSTIC_INCOMPLETE")
    calls: list[str] = []
    monkeypatch.setattr(
        runner,
        "build_diagnostic_receipt",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("semantic mismatch")),
    )

    def derive(*, artifact_root: Path, evidence_refs: object) -> object:
        assert artifact_root == tmp_path
        assert evidence_refs == evidence_refs_authority
        calls.append("derive")
        return incomplete

    evidence_refs_authority = evidence_refs
    monkeypatch.setattr(runner, "build_incomplete_diagnostic_receipt", derive)
    monkeypatch.setattr(
        runner,
        "_seal_diagnostic_receipt",
        lambda root, receipt: calls.append(
            "seal" if root == tmp_path and receipt is incomplete else "wrong"
        ),
    )

    receipt = runner._build_and_seal_diagnostic_receipt(tmp_path, evidence_refs)

    assert receipt is incomplete
    assert calls == ["derive", "seal"]


def test_integrated_diagnostic_runs_strict_preflight_then_one_cold_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "diagnostic"
    reference = tmp_path / "reference"
    inputs = tmp_path / "inputs"
    reference.mkdir()
    inputs.mkdir()
    source_identity = runner.SourceIdentityEvidence(
        git_head="a" * 40,
        tracked_diff_sha256="b" * 64,
        untracked_bytes_manifest_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
        source_manifest_size_bytes=1,
    )
    order: list[str] = []

    def prepare(path: Path, **_kwargs: object) -> object:
        path.mkdir()
        snapshot = path / runner.SOURCE_SNAPSHOT_DIRECTORY
        snapshot.mkdir()
        (snapshot / "source-manifest.json").write_bytes(b"x")
        return SimpleNamespace(
            root=snapshot,
            manifest_sha256="d" * 64,
            source_identity=lambda _root: SimpleNamespace(
                snapshot_manifest=runner.ArtifactRef(
                    runner.SOURCE_MANIFEST_ARTIFACT, "d" * 64, 1, "snapshot-v1"
                )
            ),
        )

    def copy_reference(_source: Path, root: Path) -> Path:
        destination = root / "native-reference"
        destination.mkdir()
        (destination / runner.REFERENCE_FILENAME).write_bytes(b"{}")
        return destination

    def invocation(*_args: object, **kwargs: object) -> runner.SnapshotChildInvocation:
        mode = kwargs["diagnostic_mode"]
        assert isinstance(mode, runner.DiagnosticChildMode)
        return runner.SnapshotChildInvocation(
            ("python", "runner", mode.value),
            tmp_path,
            {runner._DIAGNOSTIC_CHILD_ENV: mode.value},
        )

    def artifact(path: Path, schema: str) -> runner.ArtifactRef:
        return runner._artifact_ref(path, output, schema)

    def supervise(
        _sample: SampleName,
        child: runner.SnapshotChildInvocation,
        **_kwargs: object,
    ) -> runner.SupervisedSample:
        mode = runner.DiagnosticChildMode(
            child.environment[runner._DIAGNOSTIC_CHILD_ENV]
        )
        order.append(mode.value)
        runtime_path = output / mode.value / "runtime-evidence.json"
        runner._publish_canonical_json(
            runtime_path,
            {
                "schema_version": runner.RUNTIME_EVIDENCE_SCHEMA_VERSION,
                "runtime_identity": {"effective_environment_sha256": "e" * 64},
            },
        )
        runtime_ref = artifact(runtime_path, runner.RUNTIME_EVIDENCE_SCHEMA_VERSION)
        producer: dict[str, object] = {
            "execution_status": "SUCCESS"
            if mode is runner.DiagnosticChildMode.PREFLIGHT
            else "COMPLETE",
            "runtime_evidence": runner._artifact_ref_payload(runtime_ref),
            "warm_samples": ["warm-1", "warm-2", "warm-3"],
            "campaign.json": {"promotion_authorized": True},
        }
        if mode is runner.DiagnosticChildMode.PREFLIGHT:
            policy_path = output / "preflight" / "policy.json"
            runner._publish_canonical_json(
                policy_path,
                {"schema_version": f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy"},
            )
            producer["policy_evidence"] = runner._artifact_ref_payload(
                artifact(
                    policy_path,
                    f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-policy",
                )
            )
        if mode is runner.DiagnosticChildMode.COLD:
            for field, filename, schema in (
                ("history_evidence", "history.json", "history-v1"),
                (
                    "terminal_numerical_evidence",
                    "terminal-numerical.json",
                    "terminal-v1",
                ),
                ("raw_trace_evidence", "raw.trace.json.gz", "trace-v1"),
                ("trace_intervals_evidence", "trace-intervals.json", "interval-v1"),
                ("policy_evidence", "policy.json", "policy-v1"),
            ):
                evidence_path = output / "cold" / filename
                evidence_path.write_bytes(b"x")
                producer[field] = runner._artifact_ref_payload(
                    artifact(evidence_path, schema)
                )
            producer.update(
                {
                    "runtime": {"backend": "gpu", "jax_enable_x64": True},
                    "policy_sha256": "f" * 64,
                    "phase_schema_sha256": runner.GNTR_DIAGNOSTIC_PHASE_SCHEMA_SHA256,
                    "transfer_audit": {
                        "hot_h2d_transfers": 0,
                        "hot_d2h_transfers": 0,
                        "python_callbacks": 0,
                        "final_d2h_transfers": 1,
                    },
                    "timestamps_ns": {"solve_started": 1, "solve_stopped": 2},
                }
            )
        return runner.SupervisedSample(
            SampleName.COLD,
            runner.ChildTerminalStatus.COMPLETE,
            123,
            456,
            1.0,
            producer,
            {
                "peak_memory_bytes": 1,
                "peak_memory_fraction": 0.1,
                "schema_version": runner.MEMORY_SCHEMA_VERSION,
            },
            (),
            observed_child_argv=child.argv,
            stdout=b"{}",
            stderr=b"",
            memory_samples=(),
        )

    def publish_supervision(
        _root: Path,
        directory: Path,
        _outcome: runner.SupervisedSample,
        **_kwargs: object,
    ) -> dict[str, runner.ArtifactRef]:
        result: dict[str, runner.ArtifactRef] = {}
        for name in (
            "producer",
            "child_terminal",
            "process",
            "memory",
            "memory_samples",
        ):
            path = directory / f"{name}.json"
            runner._publish_canonical_json(path, {"schema_version": f"{name}-v1"})
            result[name] = artifact(path, f"{name}-v1")
        return result

    complete_receipt = SimpleNamespace(verdict="DIAGNOSTIC_COMPLETE")
    sealed: list[object] = []
    monkeypatch.setattr(runner, "prepare_execution_snapshot", prepare)
    monkeypatch.setattr(runner, "copy_validated_reference", copy_reference)
    monkeypatch.setattr(
        runner,
        "_publish_parent_policy_authority",
        lambda _reference, root: _publish_fake_policy_authority(root),
    )
    monkeypatch.setattr(runner, "_physical_gpu_identity", lambda _env: ("gpu", 1))
    monkeypatch.setattr(runner, "build_child_invocation", invocation)
    monkeypatch.setattr(runner, "supervise_sample", supervise)
    monkeypatch.setattr(
        runner, "_capture_source_identity_evidence", lambda *_args: source_identity
    )
    monkeypatch.setattr(runner, "_publish_diagnostic_supervision", publish_supervision)

    def strict_gate(_root: Path, **kwargs: object) -> bool:
        assert frozenset(kwargs["evidence_refs"]) == runner.PREFLIGHT_EVIDENCE_REF_KEYS
        assert order == ["preflight"]
        order.append("strict-gate")
        return True

    monkeypatch.setattr(runner, "validate_diagnostic_preflight_gate", strict_gate)
    monkeypatch.setattr(
        runner,
        "execution_evidence_payload",
        lambda **_kwargs: {
            "schema_version": f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-execution"
        },
    )
    monkeypatch.setattr(
        runner, "build_diagnostic_receipt", lambda **_kwargs: complete_receipt
    )
    monkeypatch.setattr(
        runner,
        "_seal_diagnostic_receipt",
        lambda _root, receipt: sealed.append(receipt),
    )
    monkeypatch.setattr(
        runner,
        "diagnostic_receipt_payload",
        lambda _receipt: {"verdict": "DIAGNOSTIC_COMPLETE"},
    )

    summary = runner.run_diagnostic(
        output,
        reference_root=reference,
        input_root=inputs,
        interpreter=Path(sys.executable),
        environment={},
        repo_root=Path(__file__).resolve().parents[2],
    )

    assert order == ["preflight", "strict-gate", "cold"]
    assert summary["children"] == ["preflight", "cold"]
    assert sealed == [complete_receipt]
    assert not any(path.name == "campaign.json" for path in output.rglob("*"))
    assert not any("warm" in path.parts for path in output.rglob("*"))


def test_diagnostic_manifest_assigns_frozen_roles_and_trace_pair(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "diagnostic"
    paths = {
        "diagnostic.json": b"receipt",
        "execution.json": b"execution",
        "source-snapshot/source-manifest.json": b"source",
        "native-reference/reference.json": b"reference",
        "cold/arrays/physical_state.npy": b"array",
        "cold/raw-trace/plugins/profile/run/host.trace.json.gz": b"chrome",
        "cold/raw-trace/plugins/profile/run/host.xplane.pb": b"xplane",
    }
    for relative, payload in paths.items():
        path = campaign / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    expected_roles = {
        "cold/arrays/physical_state.npy": "terminal_array",
        "cold/raw-trace/plugins/profile/run/host.trace.json.gz": "raw_trace_chrome",
        "cold/raw-trace/plugins/profile/run/host.xplane.pb": "raw_trace_xplane",
        "diagnostic.json": "diagnostic_receipt",
        "execution.json": "execution_evidence",
        "native-reference/reference.json": "native_reference",
        "source-snapshot/source-manifest.json": "source_snapshot",
    }
    monkeypatch.setattr(
        runner,
        "diagnostic_artifact_manifest_payload",
        lambda _root: {
            "schema_version": "manifest-v1",
            "entries": [
                {
                    "relative_path": relative,
                    "role": role,
                    "sha256": "a" * 64,
                    "size_bytes": len(paths[relative]),
                }
                for relative, role in sorted(expected_roles.items())
            ],
        },
    )

    runner._publish_diagnostic_artifact_manifest(campaign)

    manifest = runner.load_canonical_json_bytes(
        (campaign / runner.DIAGNOSTIC_MANIFEST_FILENAME).read_bytes()
    )
    assert isinstance(manifest, dict)
    roles = {entry["relative_path"]: entry["role"] for entry in manifest["entries"]}
    assert roles == expected_roles


@pytest.mark.parametrize(
    "relative_path",
    ("campaign.json", "warm-1/producer.json", "cold/unknown.json"),
)
def test_diagnostic_manifest_rejects_unknown_campaign_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    relative_path: str,
) -> None:
    campaign = tmp_path / "diagnostic"
    campaign.mkdir()
    (campaign / "diagnostic.json").write_bytes(b"receipt")
    injected = campaign / relative_path
    injected.parent.mkdir(parents=True, exist_ok=True)
    injected.write_bytes(b"injected")

    monkeypatch.setattr(
        runner,
        "diagnostic_artifact_manifest_payload",
        lambda _root: {
            "schema_version": "manifest-v1",
            "entries": [
                {
                    "relative_path": "diagnostic.json",
                    "role": "diagnostic_receipt",
                    "sha256": "a" * 64,
                    "size_bytes": 7,
                }
            ],
        },
    )

    with pytest.raises(ValueError, match="undeclared or missing role"):
        runner._publish_diagnostic_artifact_manifest(campaign)

    assert not (campaign / runner.DIAGNOSTIC_MANIFEST_FILENAME).exists()


def test_campaign_seal_normalizes_jax_trace_pair_file_modes(tmp_path: Path) -> None:
    campaign = tmp_path / "diagnostic"
    trace = campaign / "cold/raw-trace/plugins/profile/run"
    trace.mkdir(parents=True)
    chrome = trace / "host.trace.json.gz"
    xplane = trace / "host.xplane.pb"
    chrome.write_bytes(b"chrome")
    xplane.write_bytes(b"xplane")
    chrome.chmod(0o600)
    xplane.chmod(0o640)

    runner._seal_campaign_tree(campaign)

    assert chrome.stat().st_mode & 0o777 == 0o444
    assert xplane.stat().st_mode & 0o777 == 0o444
    assert trace.stat().st_mode & 0o777 == 0o555
    assert campaign.stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    "retained_names",
    (
        ("host.trace.json.gz",),
        ("host.xplane.pb",),
        ("host.trace.json.gz", "host.xplane.pb"),
    ),
)
def test_incomplete_diagnostic_seals_retained_partial_trace_evidence(
    tmp_path: Path,
    retained_names: tuple[str, ...],
) -> None:
    campaign = tmp_path / "diagnostic"
    profile = campaign / "cold/raw-trace/plugins/profile/run"
    profile.mkdir(parents=True)
    retained = tuple(profile / name for name in retained_names)
    for path in retained:
        path.write_bytes(path.name.encode())
        path.chmod(0o600)
    receipt = runner.build_incomplete_diagnostic_receipt(
        artifact_root=campaign,
        evidence_refs={name: None for name in runner.DIAGNOSTIC_EVIDENCE_REF_KEYS},
    )

    runner._seal_diagnostic_receipt(campaign, receipt)

    validated = runner.load_and_validate_diagnostic_artifact(campaign)
    assert runner.diagnostic_receipt_payload(validated)["verdict"] == (
        "DIAGNOSTIC_INCOMPLETE"
    )
    assert all(path.stat().st_mode & 0o777 == 0o444 for path in retained)
    assert profile.stat().st_mode & 0o777 == 0o555
    assert campaign.stat().st_mode & 0o777 == 0o555


@pytest.mark.parametrize(
    ("terminal_status", "worker_status", "expected_disposition"),
    (
        (runner.ChildTerminalStatus.COMPLETE, "SUCCESS", "SUCCESS"),
        (
            runner.ChildTerminalStatus.COMPILE_FAILURE,
            "COMPILE_FAILURE",
            "COMPILE_FAILURE",
        ),
        (
            runner.ChildTerminalStatus.COMPILE_FAILURE,
            "COMPILE_OOM",
            "COMPILE_OOM",
        ),
        (runner.ChildTerminalStatus.TIMEOUT, "SUCCESS", "COMPILE_TIMEOUT"),
        (runner.ChildTerminalStatus.CRASH, "SUCCESS", "CRASH"),
        (
            runner.ChildTerminalStatus.MONITOR_FAILURE,
            "SUCCESS",
            "MONITOR_FAILURE",
        ),
    ),
)
def test_preflight_supervisor_seals_truthful_non_authorizing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    terminal_status: runner.ChildTerminalStatus,
    worker_status: str,
    expected_disposition: str,
) -> None:
    campaign = tmp_path / "campaign"
    reference = tmp_path / "reference"
    reference.mkdir()
    input_root = tmp_path / "input"
    input_root.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    source_identity = runner.SourceIdentityEvidence(
        git_head="a" * 40,
        tracked_diff_sha256="b" * 64,
        untracked_bytes_manifest_sha256="c" * 64,
        source_manifest_sha256="d" * 64,
        source_manifest_size_bytes=1,
    )

    def prepare(output: Path, **_kwargs: object) -> object:
        output.mkdir()
        snapshot = output / runner.SOURCE_SNAPSHOT_DIRECTORY
        snapshot.mkdir()
        (snapshot / "source-manifest.json").write_bytes(b"x")
        return SimpleNamespace(root=snapshot, manifest_sha256="d" * 64)

    def copy_reference(_source: Path, output: Path) -> Path:
        destination = output / "native-reference"
        destination.mkdir()
        return destination

    outcome = runner.SupervisedSample(
        sample=SampleName.COLD,
        terminal_status=terminal_status,
        child_pid=123,
        child_start_time_ticks=456,
        process_seconds=2.0,
        producer={
            "schema_version": runner.PREFLIGHT_SCHEMA_VERSION,
            "execution_status": worker_status,
            "campaign_authorized": False,
        },
        memory={"peak_memory_fraction": 0.5},
        failure_reasons=(
            ()
            if terminal_status is runner.ChildTerminalStatus.COMPLETE
            else (expected_disposition,)
        ),
    )
    monkeypatch.setattr(runner, "prepare_execution_snapshot", prepare)
    monkeypatch.setattr(runner, "copy_validated_reference", copy_reference)
    monkeypatch.setattr(runner, "_physical_gpu_identity", lambda _env: ("gpu", 1))
    monkeypatch.setattr(
        runner,
        "_capture_source_identity_evidence",
        lambda *_args: source_identity,
    )
    monkeypatch.setattr(
        runner,
        "build_child_invocation",
        lambda *_args, **_kwargs: runner.SnapshotChildInvocation(
            ("python", "runner"), tmp_path, {}
        ),
    )
    monkeypatch.setattr(runner, "supervise_sample", lambda *_args, **_kwargs: outcome)

    result = runner.run_preflight(
        campaign,
        reference_root=reference,
        input_root=input_root,
        interpreter=interpreter,
        environment={},
        repo_root=tmp_path,
    )

    artifact = runner.load_canonical_json_bytes(
        (campaign / "preflight.json").read_bytes()
    )
    assert result.terminal_status is terminal_status
    assert artifact["terminal_disposition"] == expected_disposition
    assert artifact["campaign_authorized"] is False
    assert artifact["solver_dispatched"] is False
    assert not (campaign / runner.CAMPAIGN_RECEIPT_FILENAME).exists()
    assert (campaign / runner.CAMPAIGN_ARTIFACT_MANIFEST_FILENAME).is_file()
    assert campaign.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("operation", (runner.run_preflight, runner.run_campaign))
def test_invalid_gpu_uuid_policy_leaves_no_output_tree(
    operation: Callable[..., object],
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    reference.mkdir()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/sh\n")
    interpreter.chmod(0o755)
    output = tmp_path / "must-not-exist"

    with pytest.raises(ValueError, match="frozen physical RTX 5090 UUID"):
        operation(
            output,
            reference_root=reference,
            input_root=inputs,
            interpreter=interpreter,
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            repo_root=tmp_path,
        )

    assert not output.exists()


def test_source_identity_capture_rehashes_every_snapshot_entry(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    source = tmp_path / "source"
    roles = (
        ("execution_source", "src/package.py"),
        ("configuration", "inputs/problem.json"),
        ("benchmark", "benchmarks/runner.py"),
        ("test", "tests/test_runner.py"),
        ("native_extension", "src/simsoptpp.so"),
    )
    roots: list[runner.SourceRoot] = []
    for role, relative in roles:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        roots.append(runner.SourceRoot(role, path, relative))
    publication = runner.publish_immutable_snapshot(
        campaign / runner.SOURCE_SNAPSHOT_DIRECTORY,
        tuple(roots),
        worktree=WorktreeIdentity(
            git_head="a" * 40,
            tracked_diff_sha256="b" * 64,
            untracked_bytes_manifest_sha256="c" * 64,
            repo_root=str(tmp_path),
        ),
    )

    before = runner._capture_source_identity_evidence(publication, campaign)
    entry = publication.root / "src/package.py"
    entry.chmod(0o644)
    entry.write_bytes(b"changed but manifest is untouched")
    entry.chmod(0o444)

    assert before.source_manifest_sha256 == publication.manifest_sha256
    with pytest.raises(ValueError, match="differs from manifest"):
        runner._capture_source_identity_evidence(publication, campaign)


def test_genuinely_corrupt_snapshot_seals_as_opaque_incomplete_evidence(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    source = tmp_path / "source"
    roles = (
        ("execution_source", "src/package.py"),
        ("configuration", "inputs/problem.json"),
        ("benchmark", "benchmarks/runner.py"),
        ("test", "tests/test_runner.py"),
        ("native_extension", "src/simsoptpp.so"),
    )
    roots: list[runner.SourceRoot] = []
    for role, relative in roles:
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        roots.append(runner.SourceRoot(role, path, relative))
    publication = runner.publish_immutable_snapshot(
        campaign / runner.SOURCE_SNAPSHOT_DIRECTORY,
        tuple(roots),
        worktree=WorktreeIdentity(
            git_head="a" * 40,
            tracked_diff_sha256="b" * 64,
            untracked_bytes_manifest_sha256="c" * 64,
            repo_root=str(tmp_path),
        ),
    )
    corrupt = publication.root / "src/package.py"
    corrupt.chmod(0o644)
    corrupt.write_bytes(b"genuine post-publication corruption")
    corrupt.chmod(0o444)
    with pytest.raises(ValueError, match="differs from manifest"):
        runner._capture_source_identity_evidence(publication, campaign)
    terminal_path = campaign / "cold/terminal.json"
    runner._publish_canonical_json(
        terminal_path,
        {
            "schema_version": f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-child-terminal",
            "terminal_status": "PROTOCOL_FAILURE",
            "failure_reasons": ["SOURCE_PRE:ValueError:" + "0" * 64],
        },
    )
    evidence_refs: dict[str, runner.ArtifactRef | None] = {
        name: None for name in runner.DIAGNOSTIC_EVIDENCE_REF_KEYS
    }
    evidence_refs["child_terminal"] = runner._artifact_ref(
        terminal_path,
        campaign,
        f"{runner.DIAGNOSTIC_SCHEMA_VERSION}-child-terminal",
    )
    receipt = runner.build_incomplete_diagnostic_receipt(
        artifact_root=campaign,
        evidence_refs=evidence_refs,
    )

    runner._seal_diagnostic_receipt(campaign, receipt)

    validated = runner.load_and_validate_diagnostic_artifact(campaign)
    assert runner.diagnostic_receipt_payload(validated)["verdict"] == (
        "DIAGNOSTIC_INCOMPLETE"
    )
    manifest = runner.load_canonical_json_bytes(
        (campaign / runner.DIAGNOSTIC_MANIFEST_FILENAME).read_bytes()
    )
    assert isinstance(manifest, dict)
    source_roles = {
        entry["role"]
        for entry in manifest["entries"]
        if str(entry["relative_path"]).startswith("source-snapshot/")
    }
    assert source_roles == {"source_snapshot_opaque_failure"}
    assert corrupt.stat().st_mode & 0o777 == 0o444
    assert campaign.stat().st_mode & 0o777 == 0o555


def test_prepare_execution_snapshot_covers_every_required_role(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[2]
    native_extension = Path(runner.simsoptpp.__file__).resolve(strict=True)
    campaign = tmp_path / "campaign"

    publication = runner.prepare_execution_snapshot(
        campaign,
        repo_root=repository,
        native_extension_path=native_extension,
    )
    loaded = runner.load_snapshot(publication.root)
    roles = {entry.role for entry in loaded.entries}
    test_paths = {
        entry.relative_path for entry in loaded.entries if entry.role == "test"
    }

    assert roles == {
        "execution_source",
        "configuration",
        "benchmark",
        "test",
        "native_extension",
    }
    assert {
        "tests/benchmarks/test_run_single_stage_native_equivalent_quality_campaign.py",
        "tests/benchmarks/test_single_stage_native_equivalent_endpoint_audit.py",
        "tests/benchmarks/test_single_stage_native_equivalent_quality_receipt.py",
        "tests/benchmarks/test_single_stage_native_equivalent_reference.py",
        "tests/geo/test_fullspace_native_equivalent_quality.py",
    }.issubset(test_paths)
    completed = subprocess.run(
        (
            sys.executable,
            "-I",
            str(publication.root / runner._ENTRYPOINT),
            "--snapshot-child",
            "--help",
        ),
        cwd=publication.root,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        check=True,
        capture_output=True,
        text=True,
        timeout=60.0,
    )
    assert "--preflight-only" in completed.stdout


def test_campaign_artifact_manifest_closes_exact_file_set(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    nested = campaign / "samples" / "cold"
    nested.mkdir(parents=True)
    (campaign / "campaign.json").write_bytes(b"campaign")
    (nested / "producer.json").write_bytes(b"producer")

    runner._publish_campaign_artifact_manifest(campaign)

    payload = runner.load_canonical_json_bytes(
        (campaign / runner.CAMPAIGN_ARTIFACT_MANIFEST_FILENAME).read_bytes()
    )
    assert payload["schema_version"] == runner.CAMPAIGN_ARTIFACT_MANIFEST_SCHEMA
    assert [entry["relative_path"] for entry in payload["entries"]] == [
        "campaign.json",
        "samples/cold/producer.json",
    ]


def test_supervisor_retains_process_timeout_as_terminal_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = _TimeoutProcess()
    monitor = _Monitor()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: monitor,
    )
    monkeypatch.setattr(
        runner,
        "bound_gpu_memory_payload",
        lambda *_args, **_kwargs: {"sample_count": 1},
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"),
        cwd=tmp_path,
        environment={},
    )

    outcome = runner.supervise_sample(
        SampleName.COLD,
        invocation,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
        timeout_seconds=0.01,
    )

    assert process.killed
    assert outcome.terminal_status is runner.ChildTerminalStatus.TIMEOUT
    assert outcome.failure_reasons == ("PROCESS_TIMEOUT_900_SECONDS",)
    assert outcome.memory == {"sample_count": 1}


class _TimeoutProcess:
    pid = 77
    returncode = -9

    def __init__(self) -> None:
        self.killed = False
        self.calls = 0

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        self.calls += 1
        if self.calls == 1:
            raise subprocess.TimeoutExpired(("worker",), timeout)
        return b"", b"timeout"

    def kill(self) -> None:
        self.killed = True

    def poll(self) -> int | None:
        return self.returncode if self.killed else None


class _CommunicationFailureProcess:
    pid = 78
    returncode = -9

    def __init__(
        self,
        *,
        kill_fails: bool = False,
        communication_failures: int = 1,
        wait_fails: bool = False,
    ) -> None:
        self.kill_fails = kill_fails
        self.communication_failures = communication_failures
        self.wait_fails = wait_fails
        self.killed = False
        self.reaped = False
        self.calls = 0
        self.wait_calls = 0
        self.stdout = SimpleNamespace(read=lambda: b"fallback stdout")
        self.stderr = SimpleNamespace(read=lambda: b"fallback stderr")

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        del timeout
        self.calls += 1
        if self.calls <= self.communication_failures:
            raise OSError("communicate fault")
        return b"retained stdout", b"retained stderr"

    def kill(self) -> None:
        if self.kill_fails:
            raise OSError("kill fault")
        self.killed = True

    def poll(self) -> int | None:
        return None if not self.killed else self.returncode

    def wait(self) -> int:
        assert self.killed
        self.wait_calls += 1
        if self.wait_fails:
            raise OSError("wait fault")
        self.reaped = True
        return self.returncode


class _Monitor:
    identity = SimpleNamespace(pid=77, start_ticks=123, argv=("python", "worker.py"))

    def start(self) -> None:
        pass

    def finish(self) -> object:
        return object()


class _FailingMonitor(_Monitor):
    def finish(self) -> object:
        raise RuntimeError("no GPU samples")


@pytest.mark.parametrize(
    ("kill_fails", "communication_failures"),
    [(False, 1), (True, 1), (False, 2)],
)
def test_supervisor_reaps_launched_child_after_communication_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kill_fails: bool,
    communication_failures: int,
) -> None:
    process = _CommunicationFailureProcess(
        kill_fails=kill_fails,
        communication_failures=communication_failures,
    )
    fallback_kills: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: _Monitor(),
    )
    monkeypatch.setattr(
        runner.os,
        "kill",
        lambda pid, sig: (
            fallback_kills.append((pid, sig)) or setattr(process, "killed", True)
        ),
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"), cwd=tmp_path, environment={}
    )

    outcome = runner.supervise_sample(
        SampleName.COLD,
        invocation,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
        retain_raw_evidence=True,
    )

    assert process.killed
    assert process.calls == communication_failures + 1
    assert fallback_kills == ([(process.pid, signal.SIGKILL)] if kill_fails else [])
    assert outcome.terminal_status is runner.ChildTerminalStatus.MONITOR_FAILURE
    assert outcome.child_start_time_ticks == 123
    assert outcome.stdout == b"retained stdout"
    assert outcome.stderr == b"retained stderr"
    assert outcome.failure_reasons[0].startswith("SUPERVISION_IO:OSError:")


@pytest.mark.parametrize("wait_fails", [False, True])
def test_supervisor_permanent_stream_failure_waits_reaps_and_retains_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wait_fails: bool,
) -> None:
    process = _CommunicationFailureProcess(
        communication_failures=sys.maxsize,
        wait_fails=wait_fails,
    )
    waitpid_calls: list[tuple[int, int]] = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: _Monitor(),
    )
    monkeypatch.setattr(
        runner.os,
        "waitpid",
        lambda pid, options: (
            waitpid_calls.append((pid, options))
            or setattr(process, "reaped", True)
            or (pid, signal.SIGKILL)
        ),
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"), cwd=tmp_path, environment={}
    )

    outcome = runner.supervise_diag2_sample(
        SampleName.COLD,
        invocation,
        mode=runner.DiagnosticChildMode.COLD,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
        validate_producer=lambda *_args, **_kwargs: pytest.fail(
            "permanent stream failure cannot publish a producer"
        ),
    )

    assert process.killed and process.reaped
    assert process.calls == 3
    assert process.wait_calls == 1
    assert waitpid_calls == ([(process.pid, 0)] if wait_fails else [])
    assert outcome.launched
    assert outcome.child_pid == process.pid
    assert outcome.child_start_time_ticks == 123
    assert outcome.monitor_failure_kind is runner.MonitorFailureKind.FINALIZATION
    assert (
        outcome.selected_failure_reason
        is runner.FailureReasonCodeV2.MONITOR_FINALIZATION_FAILED
    )
    assert outcome.stdout == b"fallback stdout"
    assert outcome.stderr == b"fallback stderr"
    assert outcome.process_diagnostics is not None
    assert outcome.process_diagnostics["returncode"] == -signal.SIGKILL


def test_diagnostic_supervision_retains_raw_process_and_memory_evidence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stdout = runner.canonical_json_bytes({"execution_status": "SUCCESS"})
    stderr = b"diagnostic stderr"
    process = SimpleNamespace(
        pid=99,
        returncode=0,
        communicate=lambda timeout: (stdout, stderr),
    )
    sample = SimpleNamespace(sampled_at_unix_ns=1234, used_memory_mib=17)
    monitor = _Monitor()
    monitor.finish = lambda: SimpleNamespace(samples=(sample,))
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: monitor,
    )
    monkeypatch.setattr(
        runner,
        "bound_gpu_memory_payload",
        lambda *_args, **_kwargs: {"sample_count": 1},
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"), cwd=tmp_path, environment={}
    )

    outcome = runner.supervise_sample(
        SampleName.COLD,
        invocation,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
        retain_raw_evidence=True,
    )

    assert outcome.terminal_status is runner.ChildTerminalStatus.COMPLETE
    assert outcome.observed_child_argv == ("python", "worker.py")
    assert outcome.stdout == stdout
    assert outcome.stderr == stderr
    assert outcome.memory_samples == (
        runner.RawGpuMemorySample(sampled_at_unix_ns=1234, used_memory_mib=17),
    )
    assert outcome.process_diagnostics is not None
    assert outcome.process_diagnostics["returncode"] == 0


def test_supervisor_rejects_noncanonical_worker_protocol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(
        pid=88,
        returncode=0,
        communicate=lambda timeout: (b'{"b": 1, "a": 2}\n', b""),
    )
    monitor = _Monitor()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: monitor,
    )
    monkeypatch.setattr(
        runner,
        "bound_gpu_memory_payload",
        lambda *_args, **_kwargs: {"sample_count": 1},
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"),
        cwd=tmp_path,
        environment={},
    )

    outcome = runner.supervise_sample(
        SampleName.WARM_1,
        invocation,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
    )

    assert outcome.terminal_status is runner.ChildTerminalStatus.PROTOCOL_FAILURE
    assert outcome.producer == {}
    assert outcome.failure_reasons[0].startswith("WORKER_PROTOCOL:")


def test_monitor_failure_does_not_overwrite_established_child_crash(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    process = SimpleNamespace(
        pid=99,
        returncode=17,
        communicate=lambda timeout: (b"", b"child crashed"),
    )
    monitor = _FailingMonitor()
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        runner,
        "BoundProcessGpuMemoryMonitor",
        lambda **_kwargs: monitor,
    )
    invocation = runner.SnapshotChildInvocation(
        argv=("python", "worker.py"), cwd=tmp_path, environment={}
    )

    outcome = runner.supervise_sample(
        SampleName.COLD,
        invocation,
        gpu_uuid=runner.GPU_UUID,
        physical_memory_bytes=1024,
    )

    assert outcome.terminal_status is runner.ChildTerminalStatus.CRASH
    assert outcome.failure_reasons[0].startswith("CHILD_EXIT_17:")
    assert outcome.failure_reasons[1].startswith("MONITOR:RuntimeError:")
    assert outcome.process_diagnostics is not None
    assert outcome.process_diagnostics["stderr_tail"] == "child crashed"
    assert outcome.process_diagnostics["monitor_error_type"] == "RuntimeError"
    assert outcome.process_diagnostics["monitor_error_message"] == "no GPU samples"


def _diag5_predecessor_postmortem() -> dict[str, object]:
    hashes = {
        "reviewed_qualified_files_sha256": "e1938b81503c696bd5dc796045cdd8164e14453420b48fb38fb0f89b35ddbcc8",
        "reviewed_frozen_numerical_entries_sha256": "57a3bf08fad41871812322b516f994a8e66abe2104c0e8ed0055688e3209f7e0",
        "reviewed_execution_source_manifest_sha256": "386698c597b363e9ce463c8a9bb47628447f04e34611d83d9bd7b7c786439604",
        "reviewed_execution_source_entries_sha256": "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2",
        "reviewed_plan_full_sha256": "5c27a90047291774955858f1b86502bfeb0aec900c733f53d8a29c0dbe41a770",
        "reviewed_plan_prefix_sha256": "987dd67227431a90dd851d4d8ab78f639f9964c57de5ac093fc15f5aac504e5c",
    }
    reviews = [
        {
            "reviewer": reviewer,
            "role": role,
            "session": session,
            "verdict": "RETRACTED",
            **hashes,
        }
        for reviewer, role, session in (
            (
                "codex-numerical-controller-current-manifest",
                "numerical-controller",
                "numerical-controller-20260811T220006-manifest386698c5",
            ),
            (
                "codex-receipt-schema-a55a4fac",
                "receipt-schema",
                "5c87cc42-3234-4b9f-bcd8-3eee3e0ea01d",
            ),
            (
                "/root/ftr_runner_receipt",
                "source-snapshot",
                "source-snapshot-final-20260811-ftr01",
            ),
            (
                "codex-atomic-lifecycle-current-manifest",
                "atomic-lifecycle",
                "/root/diag_runner_map/ssot_atomic_review@2026-08-12T02:01:09Z",
            ),
        )
    ]
    return {
        "schema_version": successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
        "session_reference": "74963",
        "original_stdout_retained": False,
        "original_stderr_retained": False,
        "original_process_receipt": "NOT_PRODUCED",
        "reconstruction": {
            "command_text": successor_authority.DIAG4_CPU_QUALIFICATION_COMMAND,
            "partial_root": str(successor_authority.DIAG5_FAILED_DIAG4_PARTIAL_ROOT),
            "failed_stage": "NATIVE_EXTENSION_RUNTIME_BINDING",
            "exception_class": "QualificationError",
            "exception_message": "native extension runtime binding differs",
            "qualifier_sha256": successor_authority.DIAG5_FAILED_DIAG4_QUALIFIER_SHA256,
            "execution_manifest_sha256": successor_authority.DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_SHA256,
            "execution_entries_sha256": "7b921fed75c8a0154833ee4acf16a82922ce11b4d93dc52154dc54cc71d248b2",
            "execution_source_entry_count": 603,
            "copied_tree_entry_count": 604,
            "predecessor_full_tree_sha256": "c04cbbb79650990ab38e497bd48d6d7ab9cc2714941c58e3ce91e4147997436a",
            "copied_qualifier_predicate": "observed.st_nlink != 1",
            "native_binding": {
                "path": str(successor_authority.DIAG5_FAILED_DIAG4_NATIVE_PATH),
                "loader": "_ScikitBuildLoaderWrapper",
                "sha256": "41b2ca791a720f325ffa9b382b31d29bade73f6516693805d41adc0de6f6ed4b",
                "size_bytes": 2883776,
                "device": 66306,
                "inode": 50480769,
                "link_count": 2,
            },
            "final_root_absent": True,
            "scientific_paths_absent": True,
            "prior_reviews_retracted": reviews,
            "retracted_reviews_sha256": hashlib.sha256(
                runner.canonical_json_bytes(reviews)
            ).hexdigest(),
        },
    }


def _diag5_predecessor_evidence(
    receipt_path: Path,
) -> successor_authority.Diag5PredecessorFailureEvidence:
    manifest_path = (
        successor_authority.DIAG5_FAILED_DIAG4_PARTIAL_ROOT
        / successor_authority.DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_RELATIVE_PATH
    )
    manifest = runner.load_canonical_json_bytes(manifest_path.read_bytes())
    return successor_authority.Diag5PredecessorFailureEvidence(
        partial_root=successor_authority.DIAG5_FAILED_DIAG4_PARTIAL_ROOT,
        failed_stage="NATIVE_EXTENSION_RUNTIME_BINDING",
        exception_class="QualificationError",
        exception_message="native extension runtime binding differs",
        qualifier_sha256=successor_authority.DIAG5_FAILED_DIAG4_QUALIFIER_SHA256,
        execution_manifest_sha256=successor_authority.DIAG5_FAILED_DIAG4_EXECUTION_MANIFEST_SHA256,
        execution_entries_sha256=manifest["entries_sha256"],
        execution_source_entry_count=603,
        copied_tree_entry_count=604,
        predecessor_full_tree_sha256=hashlib.sha256(
            runner.canonical_json_bytes(
                successor_authority._diag5_predecessor_tree_entries(
                    successor_authority.DIAG5_FAILED_DIAG4_PARTIAL_ROOT
                    / "execution-source"
                )
            )
        ).hexdigest(),
        postmortem_path=receipt_path,
        postmortem_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )


def test_diag5_live_native_hardlinks_are_valid_cross_runtime_identity(
    tmp_path: Path,
) -> None:
    cpu = tmp_path / "cpu.so"
    gpu = tmp_path / "gpu.so"
    cpu.write_bytes(b"native-extension")
    os.link(cpu, gpu)

    cpu_binding = successor_authority.observe_diag5_native_extension_binding(cpu)
    gpu_binding = successor_authority.observe_diag5_native_extension_binding(gpu)
    successor_authority.validate_diag5_cross_runtime_native_bindings(
        cpu_binding, gpu_binding
    )

    assert cpu_binding.link_count == gpu_binding.link_count == 2
    assert cpu_binding.path != gpu_binding.path


def test_diag5_native_binary_identity_rejects_different_bytes(tmp_path: Path) -> None:
    cpu = tmp_path / "cpu.so"
    gpu = tmp_path / "gpu.so"
    cpu.write_bytes(b"cpu")
    gpu.write_bytes(b"gpu")

    with pytest.raises(ValueError, match="binary identity differs"):
        successor_authority.validate_diag5_cross_runtime_native_bindings(
            successor_authority.observe_diag5_native_extension_binding(cpu),
            successor_authority.observe_diag5_native_extension_binding(gpu),
        )


def test_diag5_sealed_native_copy_requires_read_only_unique_inode(
    tmp_path: Path,
) -> None:
    live = tmp_path / "live.so"
    sealed = tmp_path / "sealed.so"
    live.write_bytes(b"native")
    binding = successor_authority.observe_diag5_native_extension_binding(live)
    sealed.write_bytes(live.read_bytes())
    sealed.chmod(0o444)

    successor_authority.validate_diag5_sealed_native_copy(sealed, binding)
    alias = tmp_path / "sealed-alias.so"
    os.link(sealed, alias)
    with pytest.raises(ValueError, match="sealed native snapshot copy differs"):
        successor_authority.validate_diag5_sealed_native_copy(sealed, binding)


def test_diag5_predecessor_partial_and_postmortem_are_bound(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    receipt_path = (
        repository / successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH
    )
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(
        runner.canonical_json_bytes(_diag5_predecessor_postmortem())
    )
    receipt_path.chmod(0o444)

    successor_authority.validate_diag5_predecessor_failure(
        _diag5_predecessor_evidence(receipt_path), repository_root=repository
    )

    changed = _diag5_predecessor_postmortem()
    reconstruction = changed["reconstruction"]
    assert isinstance(reconstruction, dict)
    reconstruction["exception_message"] = "different"
    receipt_path.chmod(0o644)
    receipt_path.write_bytes(runner.canonical_json_bytes(changed))
    receipt_path.chmod(0o444)
    with pytest.raises(ValueError, match="reconstruction differs"):
        successor_authority.validate_diag5_predecessor_failure(
            _diag5_predecessor_evidence(receipt_path), repository_root=repository
        )


def test_diag5_native_claim_retains_and_revalidates_hardlink_topology(
    tmp_path: Path,
) -> None:
    native = (tmp_path / "native.so").resolve()
    native.write_bytes(b"native")
    alias = tmp_path / "alias.so"
    os.link(native, alias)

    with pytest.raises(ValueError, match="binding drifted"):  # noqa: SIM117
        with successor_authority.claim_diag5_native_extension_binding(native) as claim:
            assert claim.binding.link_count == 2
            alias.unlink()


def test_diag5_native_claim_rejects_same_byte_path_replacement(tmp_path: Path) -> None:
    native = (tmp_path / "native.so").resolve()
    displaced = tmp_path / "displaced.so"
    native.write_bytes(b"native")

    with pytest.raises(ValueError, match="inode is not bound"):  # noqa: SIM117
        with successor_authority.claim_diag5_native_extension_binding(native):
            native.rename(displaced)
            native.write_bytes(b"native")


def test_diag5_predecessor_rejects_arbitrary_review_ledger(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    postmortem_path = (
        repository / successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH
    )
    postmortem_path.parent.mkdir(parents=True)
    postmortem = _diag5_predecessor_postmortem()
    reconstruction = postmortem["reconstruction"]
    assert isinstance(reconstruction, dict)
    reviews = reconstruction["prior_reviews_retracted"]
    assert isinstance(reviews, list)
    reviews[0] = {**reviews[0], "reviewer": "invented"}
    reconstruction["retracted_reviews_sha256"] = hashlib.sha256(
        runner.canonical_json_bytes(reviews)
    ).hexdigest()
    postmortem_path.write_bytes(runner.canonical_json_bytes(postmortem))
    postmortem_path.chmod(0o444)

    with pytest.raises(ValueError, match="review retraction differs"):
        successor_authority.validate_diag5_predecessor_failure(
            _diag5_predecessor_evidence(postmortem_path),
            repository_root=repository,
        )


def test_diag5_predecessor_rejects_wrong_postmortem_path(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    postmortem_path = repository / "docs" / "wrong.json"
    postmortem_path.parent.mkdir(parents=True)
    postmortem_path.write_bytes(
        runner.canonical_json_bytes(_diag5_predecessor_postmortem())
    )
    postmortem_path.chmod(0o444)

    with pytest.raises(ValueError, match="postmortem path differs"):
        successor_authority.validate_diag5_predecessor_failure(
            _diag5_predecessor_evidence(postmortem_path),
            repository_root=repository,
        )


def test_diag5_live_postmortem_control_validates() -> None:
    repository = Path(__file__).resolve().parents[2]
    postmortem_path = (
        repository / successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH
    )

    successor_authority.validate_diag5_predecessor_failure(
        _diag5_predecessor_evidence(postmortem_path),
        repository_root=repository,
    )


def test_diag5_postmortem_artifact_deep_load_is_exact(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    source = repository / successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_RELATIVE_PATH
    destination = (
        tmp_path
        / successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(source.read_bytes())
    destination.chmod(0o444)
    reference = runner.ArtifactRef(
        successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH,
        successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_SHA256,
        destination.stat().st_size,
        successor_authority.DIAG5_PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
    )

    document = successor_authority.validate_diag5_predecessor_postmortem_artifact(
        tmp_path, reference
    )

    assert document["original_process_receipt"] == "NOT_PRODUCED"


def test_diag5_bound_staging_inode_revalidates_after_final_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    rollback = tmp_path / "output.partial-rollback"
    native = tmp_path / "native.so"
    native.write_bytes(b"native")
    binding = successor_authority.observe_diag5_native_extension_binding(native)
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    native_descriptor = os.open(native, os.O_RDONLY)
    native_leaf = successor_authority._Diag4LockedLeaf(
        native,
        native_descriptor,
        binding.sha256,
        binding.size_bytes,
        native.stat().st_mode,
    )
    native_claim = successor_authority.Diag5NativeExtensionClaim(
        binding, native_leaf, {tmp_path: parent_descriptor}
    )
    cpu_binding = {
        "cpu_native_extension_device": binding.device,
        "cpu_native_extension_inode": binding.inode,
        "cpu_native_extension_link_count": binding.link_count,
        "cpu_native_extension_path": str(binding.path),
        "native_extension_sha256": binding.sha256,
        "native_extension_size_bytes": binding.size_bytes,
    }
    gpu_binding = {
        "gpu_native_extension_device": binding.device,
        "gpu_native_extension_inode": binding.inode,
        "gpu_native_extension_link_count": binding.link_count,
        "gpu_native_extension_path": str(binding.path),
        "native_extension_sha256": binding.sha256,
        "native_extension_size_bytes": binding.size_bytes,
    }
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=native_claim,
        gpu_native_claim=native_claim,
    )
    claim = successor_authority.Diag5SuccessorAuthorityClaim(
        payload={"cpu_native_binding": cpu_binding, "gpu_native_binding": gpu_binding},
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_uuid="gpu",
        expected_numerical_identity={},
        expected_frozen_numerical_entries={},
        expected_gpu_output_root=output,
        expected_gpu_staging_root=staging,
        expected_gpu_rollback_root=rollback,
        expected_cpu_qualification_root=tmp_path,
        expected_cpu_source_snapshot_entries={},
        expected_gpu_source_snapshot_identity=cast(
            successor_authority.SnapshotIdentity, object()
        ),
        expected_native_copy_relative_path="native/x.so",
        expected_copied_native_sha256=binding.sha256,
        expected_copied_native_size_bytes=binding.size_bytes,
        cpu_native_binding=binding,
        gpu_native_binding=binding,
        predecessor_postmortem=runner.ArtifactRef("x", "d" * 64, 1, "x-v1"),
        expected_interpreter={},
        expected_native_reference={},
        expected_input_bundle={},
        _lease=lease,
    )
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_OUTPUT_ROOT", output)
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_STAGING_ROOT", staging)
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_ROLLBACK_ROOT", rollback)
    monkeypatch.setattr(
        successor_authority,
        "_validate_diag5_authority_payload",
        lambda *_args, **_kwargs: (
            binding,
            binding,
            claim.predecessor_postmortem,
            claim.predecessor_postmortem,
            {},
            claim.expected_gpu_source_snapshot_identity,
        ),
    )
    staging.mkdir()
    try:
        successor_authority.bind_diag5_staging_root(claim, staging)
        staging.rename(output)
        successor_authority.revalidate_diag5_successor_authority(claim)
    finally:
        if lease.staging_descriptor is not None:
            os.close(lease.staging_descriptor)
        os.close(native_descriptor)
        os.close(parent_descriptor)
        os.close(ancestor_descriptor)


def test_diag5_consumption_marker_is_durable_and_one_shot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )

    consumed = successor_authority.consume_diag5_successor_authority(claim)

    assert consumed.path.is_file()
    assert successor_authority.diag5_authority_lifecycle(claim).value == "CONSUMED"
    with pytest.raises(RuntimeError, match="cannot be consumed"):
        successor_authority.consume_diag5_successor_authority(claim)
    lease.active = False
    if lease.consumption_marker_descriptor is not None:
        os.close(lease.consumption_marker_descriptor)
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


@pytest.mark.parametrize(
    "mutation",
    [
        "entry",
        "worktree",
        "cached_manifest",
        "leaf_bytes",
        "mode",
        "nlink",
        "manifest",
        "root",
    ],
)
def test_diag5_gpu_snapshot_is_bound_to_held_cpu_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    native = tmp_path / "x.so"
    native.write_bytes(b"native")
    native_descriptor = os.open(native, os.O_RDONLY | os.O_NOFOLLOW)
    directory_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    digest = hashlib.sha256(native.read_bytes()).hexdigest()
    metadata = os.fstat(native_descriptor)
    leaf = successor_authority._Diag4LockedLeaf(
        native,
        native_descriptor,
        digest,
        metadata.st_size,
        metadata.st_mode & 0o777,
    )
    binding = successor_authority.Diag5NativeExtensionBinding(
        native,
        digest,
        metadata.st_size,
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_nlink,
    )
    payloads = {
        "bench.py": b"b",
        "exec.py": b"e",
        "manifest.json": b"m",
        "native/x.so": b"native",
        "test.py": b"t",
    }
    entries = (
        SnapshotEntry("benchmark", "bench.py", 1, hashlib.sha256(b"b").hexdigest()),
        SnapshotEntry(
            "execution_source", "exec.py", 1, hashlib.sha256(b"e").hexdigest()
        ),
        SnapshotEntry(
            "execution_source_manifest",
            "manifest.json",
            1,
            hashlib.sha256(b"m").hexdigest(),
        ),
        SnapshotEntry("native_extension", "native/x.so", metadata.st_size, digest),
        SnapshotEntry("test", "test.py", 1, hashlib.sha256(b"t").hexdigest()),
    )
    worktree = WorktreeIdentity("5" * 40, "6" * 64, "7" * 64, str(tmp_path))
    expected_identity = successor_authority.build_snapshot_identity(entries, worktree)
    mutated_payload = b"B"
    mutated_entries = (
        replace(entries[0], sha256=hashlib.sha256(mutated_payload).hexdigest()),
        *entries[1:],
    )
    mutated_worktree = replace(worktree, tracked_diff_sha256="9" * 64)
    candidate_entries = mutated_entries if mutation == "entry" else entries
    candidate_worktree = mutated_worktree if mutation == "worktree" else worktree
    candidate_identity = successor_authority.build_snapshot_identity(
        candidate_entries, candidate_worktree
    )
    staging = tmp_path / "staging"
    source_root = staging / "source-snapshot"
    source_root.mkdir(parents=True)
    candidate_payloads = dict(payloads)
    if mutation == "entry":
        candidate_payloads["bench.py"] = mutated_payload
    for relative, payload in candidate_payloads.items():
        destination = source_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        destination.chmod(0o444)
    manifest_payload = {
        "entries": [entry.to_payload() for entry in candidate_entries],
        "schema_version": successor_authority.SOURCE_MANIFEST_SCHEMA_VERSION,
        "worktree": candidate_worktree.to_payload(),
    }
    manifest_path = source_root / "source-manifest.json"
    manifest_path.write_bytes(
        successor_authority.canonical_json_bytes(manifest_payload)
    )
    manifest_path.chmod(0o444)
    for directory in (source_root / "native", source_root):
        directory.chmod(0o555)
    snapshot = SnapshotPublication(
        source_root,
        manifest_path,
        "a" * 64
        if mutation == "cached_manifest"
        else candidate_identity.manifest_sha256,
        candidate_entries,
        candidate_worktree,
    )
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    observed = {
        entry.relative_path: (entry.sha256, entry.size_bytes, entry.role)
        for entry in candidate_entries
    }
    claim = SimpleNamespace(
        expected_cpu_source_snapshot_entries=observed,
        expected_gpu_source_snapshot_identity=expected_identity,
        expected_copied_native_sha256=digest,
        expected_copied_native_size_bytes=metadata.st_size,
        _lease=SimpleNamespace(
            active=True,
            staging_descriptor=staging_descriptor,
            gpu_snapshot_root_descriptor=None,
            gpu_native_claim=successor_authority.Diag5NativeExtensionClaim(
                binding, leaf, {tmp_path: directory_descriptor}
            ),
        ),
    )
    monkeypatch.setattr(successor_authority, "DIAG5_EXECUTION_SOURCE_ENTRY_COUNT", 3)
    monkeypatch.setattr(
        successor_authority, "DIAG5_NATIVE_COPY_RELATIVE_PATH", "native/x.so"
    )
    try:
        if mutation in {"leaf_bytes", "mode", "nlink", "manifest", "root"}:
            successor_authority.validate_diag5_successor_snapshot(snapshot, claim)
            if mutation == "leaf_bytes":
                (source_root / "bench.py").chmod(0o644)
                (source_root / "bench.py").write_bytes(b"B")
                (source_root / "bench.py").chmod(0o444)
            elif mutation == "mode":
                (source_root / "bench.py").chmod(0o644)
            elif mutation == "nlink":
                os.link(source_root / "bench.py", tmp_path / "linked-bench.py")
            elif mutation == "manifest":
                manifest_path.chmod(0o644)
                manifest_path.write_bytes(b"{}\n")
                manifest_path.chmod(0o444)
            else:
                source_root.chmod(0o755)
                old_root = staging / "old-source-snapshot"
                source_root.rename(old_root)
                shutil.copytree(old_root, source_root, copy_function=shutil.copy2)
        with pytest.raises(ValueError, match="snapshot|source|root"):
            successor_authority.validate_diag5_successor_snapshot(snapshot, claim)
    finally:
        if claim._lease.gpu_snapshot_root_descriptor is not None:
            os.close(claim._lease.gpu_snapshot_root_descriptor)
        os.close(native_descriptor)
        os.close(directory_descriptor)
        os.close(staging_descriptor)


def test_diag5_immutable_leaf_uses_shared_nonblocking_lock(tmp_path: Path) -> None:
    leaf_path = tmp_path / "leaf.json"
    leaf_path.write_bytes(b"{}\n")
    directory_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    directories = {tmp_path: directory_descriptor}
    first = successor_authority._open_diag5_shared_locked_leaf(
        leaf_path, directories, "first"
    )
    second = successor_authority._open_diag5_shared_locked_leaf(
        leaf_path, directories, "second"
    )
    writer = os.open(leaf_path, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        os.close(writer)
        os.close(second.descriptor)
        os.close(first.descriptor)
        os.close(directory_descriptor)


def test_diag5_consumption_cleanup_ambiguity_spends_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    monkeypatch.setattr(
        successor_authority,
        "_unlink_diag4_pending_marker",
        lambda *_args: (_ for _ in ()).throw(ValueError("replacement")),
    )
    with pytest.raises(ValueError, match="replacement"):
        successor_authority.consume_diag5_successor_authority(claim)
    assert successor_authority.diag5_authority_lifecycle(claim).value == (
        "CONSUMPTION_UNCERTAIN"
    )
    with pytest.raises(RuntimeError, match="cannot be consumed"):
        successor_authority.consume_diag5_successor_authority(claim)
    lease.active = False
    if lease.consumption_marker_descriptor is not None:
        os.close(lease.consumption_marker_descriptor)
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


def test_diag5_consumption_pending_collision_spends_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    pending = tmp_path / (
        f".{output.name}.diag5-authority-consumed.json.pending-{os.getpid()}"
    )
    pending.write_bytes(b"competitor")
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    with pytest.raises(FileExistsError):
        successor_authority.consume_diag5_successor_authority(claim)
    assert successor_authority.diag5_authority_lifecycle(claim).value == (
        "CONSUMPTION_UNCERTAIN"
    )
    with pytest.raises(RuntimeError, match="cannot be consumed"):
        successor_authority.consume_diag5_successor_authority(claim)
    lease.active = False
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


@pytest.mark.parametrize("proof_fails", [False, True])
def test_diag5_consumption_create_failure_requires_stable_absence_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, proof_fails: bool
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    real_open = successor_authority.os.open

    def fail_pending_open(path: object, *args: object, **kwargs: object) -> int:
        if str(path).startswith(f".{output.name}.diag5-authority-consumed"):
            raise OSError("create failed")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(successor_authority.os, "open", fail_pending_open)
    if proof_fails:
        monkeypatch.setattr(
            successor_authority,
            "_assert_diag4_locked_directory_binding",
            lambda *_args: (_ for _ in ()).throw(ValueError("parent drift")),
        )
    with pytest.raises((OSError, ValueError)):
        successor_authority.consume_diag5_successor_authority(claim)
    expected = "CONSUMPTION_UNCERTAIN" if proof_fails else "UNCONSUMED"
    assert successor_authority.diag5_authority_lifecycle(claim).value == expected
    lease.active = False
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


@pytest.mark.parametrize("fault", ["write", "chmod", "file_fsync", "link"])
def test_diag5_prepublication_fault_is_retryable_only_after_cleanup_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    if fault == "write":
        monkeypatch.setattr(
            successor_authority.os,
            "write",
            lambda *_args: (_ for _ in ()).throw(OSError("write")),
        )
    elif fault == "chmod":
        monkeypatch.setattr(
            successor_authority.os,
            "fchmod",
            lambda *_args: (_ for _ in ()).throw(OSError("chmod")),
        )
    elif fault == "file_fsync":
        real_fsync = successor_authority.os.fsync
        calls = 0

        def fail_first_fsync(descriptor: int) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("file fsync")
            real_fsync(descriptor)

        monkeypatch.setattr(successor_authority.os, "fsync", fail_first_fsync)
    else:
        monkeypatch.setattr(
            successor_authority.os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("link")),
        )
    with pytest.raises(OSError):
        successor_authority.consume_diag5_successor_authority(claim)
    assert successor_authority.diag5_authority_lifecycle(claim).value == "UNCONSUMED"
    lease.active = False
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


def test_diag5_postpublication_validation_fault_spends_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    monkeypatch.setattr(
        successor_authority,
        "_assert_diag4_published_marker_binding",
        lambda *_args: (_ for _ in ()).throw(ValueError("published replacement")),
    )
    with pytest.raises(ValueError, match="published replacement"):
        successor_authority.consume_diag5_successor_authority(claim)
    assert (
        successor_authority.diag5_authority_lifecycle(claim).value
        == "CONSUMPTION_UNCERTAIN"
    )
    lease.active = False
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


def test_diag5_prepublication_fault_plus_adjudication_fault_spends_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        plan_prefix_sha256="b" * 64,
        completed_plan_sha256="c" * 64,
        expected_gpu_output_root=output,
        _lease=lease,
    )
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    monkeypatch.setattr(
        successor_authority.os,
        "write",
        lambda *_args: (_ for _ in ()).throw(OSError("write")),
    )
    original_rebind = successor_authority._assert_diag4_locked_directory_binding
    calls = 0

    def fail_adjudication(*args: object) -> None:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("adjudication")
        original_rebind(*args)

    monkeypatch.setattr(
        successor_authority, "_assert_diag4_locked_directory_binding", fail_adjudication
    )
    with pytest.raises(ValueError, match="adjudication"):
        successor_authority.consume_diag5_successor_authority(claim)
    assert (
        successor_authority.diag5_authority_lifecycle(claim).value
        == "CONSUMPTION_UNCERTAIN"
    )
    with pytest.raises(RuntimeError, match="cannot be consumed"):
        successor_authority.consume_diag5_successor_authority(claim)
    lease.active = False
    os.close(staging_descriptor)
    os.close(parent_descriptor)
    os.close(ancestor_descriptor)


def _diag5_physical_authority_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[SimpleNamespace, successor_authority._Diag5AuthorityLease, Path, Path, Path]:
    output = tmp_path / "output"
    staging = tmp_path / "output.partial-claim"
    rollback = tmp_path / "output.partial-rollback"
    physical = tmp_path / ".output.diag5-physical-publication-failure.json"
    staging.mkdir()
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    ancestor_descriptor = os.open(tmp_path.parent, os.O_RDONLY | os.O_DIRECTORY)
    staging_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    lease = successor_authority._Diag5AuthorityLease(
        repository=tmp_path,
        output_root=output,
        authority_path=tmp_path / "authority.json",
        authority_bytes=b"{}\n",
        locked_leaves={},
        directory_descriptors={
            tmp_path.parent: ancestor_descriptor,
            tmp_path: parent_descriptor,
        },
        cpu_native_claim=SimpleNamespace(_leaf=None),
        gpu_native_claim=SimpleNamespace(_leaf=None),
        staging_descriptor=staging_descriptor,
        lifecycle=successor_authority.Diag5AuthorityLifecycle.CONSUMED,
    )
    claim = SimpleNamespace(
        authority_sha256="a" * 64,
        expected_gpu_output_root=output,
        expected_gpu_rollback_root=rollback,
        _lease=lease,
    )
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_OUTPUT_ROOT", output)
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_STAGING_ROOT", staging)
    monkeypatch.setattr(successor_authority, "DIAG5_GPU_ROLLBACK_ROOT", rollback)
    monkeypatch.setattr(successor_authority, "DIAG5_PHYSICAL_FAILURE_PATH", physical)
    monkeypatch.setattr(
        successor_authority, "revalidate_diag5_successor_authority", lambda _claim: None
    )
    monkeypatch.setattr(
        successor_authority,
        "validate_diag5_successor_snapshot",
        lambda _snapshot, _claim: None,
    )
    return claim, lease, output, rollback, physical


def _close_diag5_physical_authority_fixture(
    lease: successor_authority._Diag5AuthorityLease,
) -> None:
    lease.active = False
    if lease.physical_evidence is not None:
        os.close(lease.physical_evidence.descriptor)
    if lease.staging_descriptor is not None:
        os.close(lease.staging_descriptor)
    os.close(lease.directory_descriptors[lease.output_root.parent])
    os.close(lease.directory_descriptors[lease.output_root.parent.parent])


def test_diag5_physical_reservation_clean_finalization_is_descriptor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, output, _rollback, physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    try:
        snapshot = cast(SnapshotPublication, object())
        source_input = successor_authority.PublishedSnapshot(
            successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
            snapshot,
        )
        events: list[str] = []
        original_unlink = successor_authority._unlink_diag4_pending_marker
        monkeypatch.setattr(
            successor_authority,
            "revalidate_diag5_successor_authority",
            lambda _claim: events.append("authority"),
        )
        monkeypatch.setattr(
            successor_authority,
            "_validate_diag5_finalizer_source",
            lambda _claim, observed, *, physical_memory_bytes: (
                events.append("source")
                if observed is source_input and physical_memory_bytes == 1
                else pytest.fail("wrong snapshot")
            ),
        )

        def unlink_after_validation(
            descriptor: int, pending_name: str, directory: int
        ) -> None:
            events.append("unlink")
            original_unlink(descriptor, pending_name, directory)

        monkeypatch.setattr(
            successor_authority,
            "_unlink_diag4_pending_marker",
            unlink_after_validation,
        )
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )
        successor_authority.fsync_diag5_output_parent(claim)
        successor_authority.revalidate_diag5_published_output(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )
        events.clear()
        successor_authority.finalize_diag5_physical_evidence_success(
            claim, reservation, source_input, physical_memory_bytes=1
        )

        assert events == ["authority", "source", "authority", "unlink"]
        assert output.is_dir()
        assert not physical.exists()
        assert not list(tmp_path.glob(f"{physical.name}.pending-*"))
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_physical_reservation_cannot_be_finalized_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    try:
        snapshot = cast(SnapshotPublication, object())
        source_input = successor_authority.PublishedSnapshot(
            successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
            snapshot,
        )
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)

        with pytest.raises(RuntimeError, match="cannot be finalized"):
            successor_authority.finalize_diag5_physical_evidence_success(
                claim, reservation, source_input, physical_memory_bytes=1
            )

        assert (
            successor_authority._diag5_evidence_namespace_state(claim, reservation)
            is successor_authority.Diag5EvidenceNamespaceState.PENDING_BOUND
        )
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )
    finally:
        _close_diag5_physical_authority_fixture(lease)


@pytest.mark.parametrize("physical_memory_bytes", [0, -1, True, 1.0])
def test_diag5_physical_finalizer_rejects_invalid_physical_memory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    physical_memory_bytes: object,
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    source_input = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        cast(SnapshotPublication, object()),
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        with pytest.raises(successor_authority.Diag5FinalizerError) as caught:
            successor_authority.finalize_diag5_physical_evidence_success(
                claim,
                reservation,
                source_input,
                physical_memory_bytes=cast(int, physical_memory_bytes),
            )
        assert (
            caught.value.category
            is successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD
        )
        assert "positive integer" in str(caught.value.cause)
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_physical_finalizer_classifies_memory_mismatch_as_deep_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    source_input = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        cast(SnapshotPublication, object()),
    )
    monkeypatch.setattr(
        successor_authority,
        "_validate_diag5_finalizer_source",
        lambda *_args, physical_memory_bytes: (
            None
            if physical_memory_bytes == 99
            else (_ for _ in ()).throw(ValueError("physical memory mismatch"))
        ),
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )

        with pytest.raises(successor_authority.Diag5FinalizerError) as caught:
            successor_authority.finalize_diag5_physical_evidence_success(
                claim, reservation, source_input, physical_memory_bytes=100
            )

        assert (
            caught.value.category
            is successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD
        )
        assert str(caught.value.cause) == "physical memory mismatch"
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_finalizer_malformed_receipt_precedes_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    source_input = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        cast(SnapshotPublication, object()),
    )
    snapshot_called = False

    def snapshot_drift(*_args: object) -> None:
        nonlocal snapshot_called
        snapshot_called = True
        raise ValueError("snapshot drift")

    monkeypatch.setattr(
        successor_authority,
        "_load_diag5_finalizer_receipt",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("malformed receipt")
        ),
    )
    monkeypatch.setattr(
        successor_authority, "validate_diag5_successor_snapshot", snapshot_drift
    )
    try:
        with pytest.raises(successor_authority.Diag5FinalizerError) as caught:
            successor_authority._diag5_finalizer_deep_load(
                claim, source_input, physical_memory_bytes=1
            )

        assert (
            caught.value.category
            is successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD
        )
        assert str(caught.value.cause) == "malformed receipt"
        assert snapshot_called is False
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_pre_source_finalizer_validates_exact_outcome_and_zero_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    outcome = diagnostic_receipt.StructuredFailureV5(
        diagnostic_receipt.FailureStageV5.SETUP,
        diagnostic_receipt.FailureReasonCodeV5.SOURCE_PUBLICATION_FAILED,
        "a" * 64,
    )
    terminal_ref = diagnostic_receipt.ArtifactRef(
        "supervisor-terminal.json",
        "b" * 64,
        10,
        diagnostic_receipt.DIAG5_SUPERVISOR_TERMINAL_SCHEMA_VERSION,
    )
    receipt_ref = diagnostic_receipt.ArtifactRef(
        "diagnostic.json",
        "c" * 64,
        11,
        diagnostic_receipt.DIAG5_SCHEMA_VERSION,
    )
    slots = {
        name: (
            diagnostic_receipt.EvidenceSlotV5.present(terminal_ref)
            if name == "supervisor_terminal"
            else diagnostic_receipt.EvidenceSlotV5.absent(
                outcome.reason if name == "source_manifest" else None
            )
        )
        for name in diagnostic_receipt.DIAG5_EVIDENCE_SLOT_PATHS
    }
    source_input = successor_authority.PreSourceFailure(
        successor_authority.Diag5FinalizerSourceKind.PRE_SOURCE_FAILURE,
        outcome,
        terminal_ref,
        receipt_ref,
    )
    validated_refs: list[diagnostic_receipt.ArtifactRef] = []
    monkeypatch.setattr(
        successor_authority,
        "_load_diag5_finalizer_receipt",
        lambda _claim, *, physical_memory_bytes: (
            SimpleNamespace(evidence_slots=tuple(slots.items()), failure=outcome)
            if physical_memory_bytes == 123
            else pytest.fail("wrong physical memory")
        ),
    )
    monkeypatch.setattr(
        successor_authority,
        "_diag5_validate_finalizer_artifact_ref",
        lambda _claim, reference, **_kwargs: validated_refs.append(reference),
    )
    try:
        successor_authority._validate_diag5_finalizer_source(
            claim, source_input, physical_memory_bytes=123
        )
        assert validated_refs == [terminal_ref, receipt_ref]

        slots["source_manifest"] = diagnostic_receipt.EvidenceSlotV5.present(
            terminal_ref
        )
        with pytest.raises(ValueError, match="evidence vector differs"):
            successor_authority._validate_diag5_finalizer_source(
                claim, source_input, physical_memory_bytes=123
            )
    finally:
        _close_diag5_physical_authority_fixture(lease)


@pytest.mark.parametrize(
    ("target", "category"),
    [
        ("deep_load", successor_authority.Diag5FinalizerFailureCategory.DEEP_LOAD),
        (
            "finalization",
            successor_authority.Diag5FinalizerFailureCategory.FINALIZATION,
        ),
    ],
)
def test_diag5_physical_finalizer_classifies_internal_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    category: successor_authority.Diag5FinalizerFailureCategory,
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    source_input = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        cast(SnapshotPublication, object()),
    )
    if target == "deep_load":
        monkeypatch.setattr(
            successor_authority,
            "_validate_diag5_finalizer_source",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                ValueError("injected deep-load failure")
            ),
        )
    else:
        monkeypatch.setattr(
            successor_authority,
            "_validate_diag5_finalizer_source",
            lambda *_args, **_kwargs: None,
        )
        monkeypatch.setattr(
            successor_authority,
            "_unlink_diag4_pending_marker",
            lambda *_args: (_ for _ in ()).throw(
                OSError("injected finalization failure")
            ),
        )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )

        with pytest.raises(successor_authority.Diag5FinalizerError) as caught:
            successor_authority.finalize_diag5_physical_evidence_success(
                claim, reservation, source_input, physical_memory_bytes=1
            )

        assert caught.value.category is category
        assert lease.physical_evidence is not None
        assert lease.physical_evidence.active is True
    finally:
        _close_diag5_physical_authority_fixture(lease)


@pytest.mark.parametrize("failure", ["snapshot", "authority"])
def test_diag5_physical_finalization_validation_failure_preserves_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    claim, lease, _output, _rollback, physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    snapshot = cast(SnapshotPublication, object())
    source_input = successor_authority.PublishedSnapshot(
        successor_authority.Diag5FinalizerSourceKind.PUBLISHED_SNAPSHOT,
        snapshot,
    )
    events: list[str] = []
    authority_calls = 0

    def validate_authority(_claim: object) -> None:
        nonlocal authority_calls
        authority_calls += 1
        events.append("authority")
        if failure == "authority" and authority_calls == 2:
            raise ValueError("injected authority drift")

    def validate_snapshot(observed: SnapshotPublication, _claim: object) -> None:
        assert observed is snapshot
        events.append("snapshot")
        if failure == "snapshot":
            cause = ValueError("injected snapshot drift")
            raise successor_authority.Diag5FinalizerError(
                successor_authority.Diag5FinalizerFailureCategory.REVALIDATION,
                cause,
            ) from cause

    monkeypatch.setattr(
        successor_authority,
        "revalidate_diag5_successor_authority",
        validate_authority,
    )
    monkeypatch.setattr(
        successor_authority,
        "_validate_diag5_finalizer_source",
        lambda _claim, observed, *, physical_memory_bytes: (
            validate_snapshot(observed.snapshot, _claim)
            if physical_memory_bytes == 1
            else pytest.fail("wrong physical memory")
        ),
    )
    monkeypatch.setattr(
        successor_authority,
        "_unlink_diag4_pending_marker",
        lambda *_args: pytest.fail("pending reservation unlinked before validation"),
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )

        with pytest.raises(successor_authority.Diag5FinalizerError) as caught:
            successor_authority.finalize_diag5_physical_evidence_success(
                claim, reservation, source_input, physical_memory_bytes=1
            )

        assert (
            caught.value.category
            is successor_authority.Diag5FinalizerFailureCategory.REVALIDATION
        )
        assert str(caught.value.cause) == f"injected {failure} drift"
        expected_events = (
            ["authority", "snapshot"]
            if failure == "snapshot"
            else ["authority", "snapshot", "authority"]
        )
        assert events == expected_events
        assert lease.physical_evidence is not None
        assert lease.physical_evidence.active is True
        assert not physical.exists()
        assert len(list(tmp_path.glob(f"{physical.name}.pending-*"))) == 1
        assert (
            successor_authority._diag5_evidence_namespace_state(claim, reservation)
            is successor_authority.Diag5EvidenceNamespaceState.PENDING_BOUND
        )
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_physical_reservation_can_cancel_after_prefinal_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, output, _rollback, physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        output.mkdir()
        with pytest.raises(FileExistsError):
            successor_authority.publish_diag5_bound_staging(
                claim, successor_authority.Diag5PublishedOutputKind.FINAL
            )
        output.rmdir()

        cancelled = successor_authority.cancel_diag5_physical_failure_evidence(
            claim, reservation
        )

        assert lease.published_output is None
        assert lease.rollback_attempted is False
        assert not physical.exists()
        assert not list(tmp_path.glob(f"{physical.name}.pending-*"))
        with pytest.raises(RuntimeError, match="cannot be cancelled"):
            successor_authority.cancel_diag5_physical_failure_evidence(
                claim, reservation
            )

        assert cancelled.state.value == "CANCELLED"
        assert cancelled.cause.value == "NONE"
        assert cancelled.evidence_namespace_state.value == "PENDING_UNLINKED"
        assert cancelled.staging_path_state.value == "VISIBLE_VALIDATED"
        assert cancelled.final_path_state.value == "ABSENT"
        assert cancelled.rollback_path_state.value == "ABSENT"
    finally:
        _close_diag5_physical_authority_fixture(lease)


@pytest.mark.parametrize("fault", ["unlink", "fsync", "rebound"])
def test_diag5_physical_reservation_cancel_failure_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        if fault == "unlink":
            monkeypatch.setattr(
                successor_authority,
                "_unlink_diag4_pending_marker",
                lambda *_args: (_ for _ in ()).throw(OSError("unlink")),
            )
        elif fault == "fsync":
            monkeypatch.setattr(
                successor_authority.os,
                "fsync",
                lambda *_args: (_ for _ in ()).throw(OSError("fsync")),
            )
        else:
            original_state = successor_authority._diag5_evidence_namespace_state
            calls = 0

            def fail_rebound(*args: object) -> object:
                nonlocal calls
                calls += 1
                if calls == 2:
                    return successor_authority.Diag5EvidenceNamespaceState.PENDING_AMBIGUOUS
                return original_state(*args)

            monkeypatch.setattr(
                successor_authority,
                "_diag5_evidence_namespace_state",
                fail_rebound,
            )
        with pytest.raises(
            successor_authority.Diag5PhysicalCancellationError
        ) as captured:
            successor_authority.cancel_diag5_physical_failure_evidence(
                claim, reservation
            )
        assert captured.value.observation.state.value == "SPENT"
        expected_cause = {
            "unlink": "CANCEL_UNLINK_FAILED",
            "fsync": "CANCEL_PARENT_FSYNC_FAILED",
            "rebound": "CANCEL_VISIBILITY_AMBIGUOUS",
        }[fault]
        assert captured.value.observation.cause.value == expected_cause
        with pytest.raises(RuntimeError, match="cannot be cancelled"):
            successor_authority.cancel_diag5_physical_failure_evidence(
                claim, reservation
            )
    finally:
        _close_diag5_physical_authority_fixture(lease)


@pytest.mark.parametrize("fault", ["pending_missing", "staging_missing", "revalidate"])
def test_diag5_physical_reservation_cancel_entry_or_revalidation_is_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    claim, lease, _output, _rollback, _physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        if fault == "pending_missing":
            os.unlink(
                reservation._lease.pending_name,
                dir_fd=lease.directory_descriptors[tmp_path],
            )
        elif fault == "staging_missing":
            os.rmdir(successor_authority.DIAG5_GPU_STAGING_ROOT)
        else:
            monkeypatch.setattr(
                successor_authority,
                "revalidate_diag5_successor_authority",
                lambda _claim: (_ for _ in ()).throw(ValueError("revalidate")),
            )

        with pytest.raises(
            successor_authority.Diag5PhysicalCancellationError
        ) as captured:
            successor_authority.cancel_diag5_physical_failure_evidence(
                claim, reservation
            )

        assert captured.value.observation.state.value == "SPENT"
        expected_cause = (
            "CANCEL_REVALIDATION_FAILED"
            if fault == "revalidate"
            else "CANCEL_VISIBILITY_AMBIGUOUS"
        )
        assert captured.value.observation.cause.value == expected_cause
        with pytest.raises(RuntimeError, match="cannot be cancelled"):
            successor_authority.cancel_diag5_physical_failure_evidence(
                claim, reservation
            )
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_physical_failure_rolls_back_once_and_seals_reserved_inode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claim, lease, output, rollback, physical = _diag5_physical_authority_fixture(
        tmp_path, monkeypatch
    )
    deep_loads: list[Path] = []
    try:
        reservation = successor_authority.prepare_diag5_physical_failure_evidence(claim)
        successor_authority.publish_diag5_bound_staging(
            claim, successor_authority.Diag5PublishedOutputKind.FINAL
        )
        observation = successor_authority.rollback_diag5_bound_final(
            claim,
            reservation,
            deep_load=lambda path: deep_loads.append(path),
        )
        payload = {
            "schema_version": successor_authority.DIAG5_PHYSICAL_FAILURE_SCHEMA_VERSION,
            "route": successor_authority.DIAG5_ROUTE,
            "authority_sha256": claim.authority_sha256,
            "original_reason": "FINAL_FSYNC_FAILED",
            "rollback_cause": observation.rollback_cause.value,
            "rollback_state": observation.rollback_state.value,
            "final_path": str(output),
            "final_path_state": observation.final_path_state.value,
            "rollback_path": str(rollback),
            "rollback_path_state": observation.rollback_path_state.value,
            "evidence_namespace_state_at_seal": (
                observation.evidence_namespace_state_at_seal.value
            ),
            "sealed_artifact_manifest_sha256": "b" * 64,
        }

        published = successor_authority.publish_diag5_physical_failure_evidence(
            claim, reservation, payload
        )

        assert deep_loads == [rollback]
        assert observation.rollback_state.value == "SUCCEEDED"
        assert not output.exists()
        assert rollback.is_dir()
        assert published == physical
        assert (
            successor_authority.load_canonical_json_bytes(physical.read_bytes())
            == payload
        )
        with pytest.raises(RuntimeError, match="cannot run"):
            successor_authority.rollback_diag5_bound_final(
                claim, reservation, deep_load=lambda _path: None
            )
    finally:
        _close_diag5_physical_authority_fixture(lease)


def test_diag5_publication_uses_only_fixed_claim_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output_parent = tmp_path / "outputs"
    output_parent.mkdir()
    output = output_parent / "campaign"
    staging = Path(f"{output}.partial-claim")
    rollback = Path(f"{output}.partial-rollback")
    claim = SimpleNamespace(
        expected_gpu_output_root=output,
        expected_gpu_staging_root=staging,
        expected_gpu_rollback_root=rollback,
    )

    publication = runner._prepare_diag5_publication(
        output,
        repository_root=repository,
        successor_claim=claim,
    )

    assert publication.staging_root == staging
    assert publication.final_root == output
    assert publication.nonce == "claim"
    assert staging.is_dir()
    assert not rollback.exists()

    staging.rmdir()
    competing = Path(f"{output}.partial-other")
    competing.mkdir()
    with pytest.raises(FileExistsError, match="namespace is not absent"):
        runner._prepare_diag5_publication(
            output,
            repository_root=repository,
            successor_claim=claim,
        )


def test_diag5_bootstrap_uses_final_ssot_plan_hashes() -> None:
    assert runner._DIAG5_BOOTSTRAP_PLAN_SHA256 == (
        "24300e9742bcbb14b3fc3e2cceab37dedc310410290a13370a11d21ef749ec7a"
    )
    assert runner._DIAG5_BOOTSTRAP_COMPLETED_PLAN_SHA256 == (
        "ce244ac37bb437ea022a4b73e62bf49f8d4d5cf88b610ad2146e895f3471ce1c"
    )


def test_diag5_bootstrap_native_binding_retains_shared_lock(
    tmp_path: Path,
) -> None:
    native = tmp_path / "simsoptpp.so"
    native.write_bytes(b"native-v2")
    metadata = native.stat(follow_symlinks=False)
    bindings: list[runner._Diag5BootstrapNativeBinding] = []
    payload = {
        "gpu_native_extension_path": str(native.resolve()),
        "native_extension_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "native_extension_size_bytes": metadata.st_size,
        "gpu_native_extension_link_count": metadata.st_nlink,
        "gpu_native_extension_device": metadata.st_dev,
        "gpu_native_extension_inode": metadata.st_ino,
    }

    runner._diag5_bootstrap_bind_native_extension(payload, bindings)
    writer = os.open(native, os.O_RDONLY)
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(writer, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner._diag5_bootstrap_revalidate_native_bindings(bindings)
    finally:
        os.close(writer)
        for binding in bindings:
            fcntl.flock(binding.descriptor, fcntl.LOCK_UN)
            os.close(binding.descriptor)


def test_diag5_release_revalidates_and_unlocks_all_bootstrap_leaves(
    tmp_path: Path,
) -> None:
    regular = tmp_path / "authority.json"
    regular.write_bytes(b"{}\n")
    native = tmp_path / "simsoptpp.so"
    native.write_bytes(b"native-v2")
    regular_bindings: list[runner._Diag4BootstrapBinding] = []
    native_bindings: list[runner._Diag5BootstrapNativeBinding] = []
    runner._diag5_bootstrap_regular_bytes(regular, "test authority", regular_bindings)
    metadata = native.stat(follow_symlinks=False)
    runner._diag5_bootstrap_bind_native_extension(
        {
            "gpu_native_extension_path": str(native.resolve()),
            "native_extension_sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
            "native_extension_size_bytes": metadata.st_size,
            "gpu_native_extension_link_count": metadata.st_nlink,
            "gpu_native_extension_device": metadata.st_dev,
            "gpu_native_extension_inode": metadata.st_ino,
        },
        native_bindings,
    )
    runner._diag5_retained_bootstrap_bindings.extend(regular_bindings)
    runner._diag5_retained_native_bindings.extend(native_bindings)

    runner._release_diag5_bootstrap_bindings()

    assert not runner._diag5_retained_bootstrap_bindings
    assert not runner._diag5_retained_native_bindings
    for path in (regular, native):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def test_terminal_draft_remains_explicitly_nonpromoting() -> None:
    outcome = _outcome(SampleName.COLD)
    artifact = runner.ArtifactRef("x.json", "a" * 64, 1, "x-v1")

    payload = runner.nonpromoting_sample_draft_payload(
        outcome,
        producer_reference=artifact,
        runtime_reference=artifact,
        source_manifest_reference=artifact,
    )

    assert payload["promotion_eligible"] is False
    assert payload["failure_reasons"] == ["ENDPOINT_AUDIT_RECEIPT_BINDING_NOT_PRODUCED"]
