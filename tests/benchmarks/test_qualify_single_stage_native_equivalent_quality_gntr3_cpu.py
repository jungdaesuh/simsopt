from __future__ import annotations

import copy
import hashlib
import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol

import benchmarks.qualify_single_stage_native_equivalent_quality_gntr3_cpu as qualifier
import benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt as receipt_module
import numpy as np
import pytest
from benchmarks.qualify_single_stage_native_equivalent_quality_gntr3_cpu import (
    QUALIFICATION_SCHEMA_VERSION,
    CpuRuntimeIdentity,
    ProducedEvidence,
    QualificationError,
    run_qualification,
)
from benchmarks.single_stage_fullspace_snapshot import (
    DIAG5_CPU_SNAPSHOT_ROLES,
    ArtifactRef,
    JsonValue,
    WorktreeIdentity,
    canonical_json_bytes,
    load_snapshot,
    publish_immutable_snapshot,
)
from benchmarks.single_stage_native_equivalent_quality_diagnostic_receipt import (
    NativeEquivalentNumericalIdentity,
    ScientificOutcome,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG5_BLANK_PLAN_SHA256,
    DIAG5_BLANK_PLAN_SIZE_BYTES,
    DIAG5_NATIVE_COPY_RELATIVE_PATH,
    DIAG5_PLAN_RELATIVE_PATH,
    DIAG5_PLAN_SHA256,
)
from simsopt_jax.solve.fullspace_native_equivalent_quality import (
    NEQ_GNTR3_OPTIONS,
    NEQ_GNTR3_ROUTE,
    NEQ_GNTR3_SCHEMA_VERSION,
)

_HASH = "a" * 64


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }


def _identity() -> NativeEquivalentNumericalIdentity:
    return NativeEquivalentNumericalIdentity(
        numerical_route=NEQ_GNTR3_ROUTE,
        numerical_result_schema_version=NEQ_GNTR3_SCHEMA_VERSION,
        problem_sha256=_HASH,
        optimizer_options_sha256=_HASH,
        base_neq_gntr1_policy_sha256=_HASH,
        scaling_sha256=_HASH,
        bootstrap_state_sha256=_HASH,
        initial_physical_state_sha256=_HASH,
        identity_sha256=_HASH,
    )


def _execution_manifest(entries: Mapping[str, bytes]) -> bytes:
    entry_payload = {
        relative: {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        for relative, payload in entries.items()
    }
    return canonical_json_bytes(
        {
            "entries": entry_payload,
            "entries_sha256": hashlib.sha256(
                canonical_json_bytes(entry_payload)
            ).hexdigest(),
            "schema_version": qualifier.EXECUTION_SOURCE_AUTHORITY_SCHEMA_VERSION,
        }
    )


def _small_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> qualifier.ExecutionSourceBindings:
    worktree = tmp_path / "worktree"
    worktree.mkdir(parents=True)
    plan_prefix_sha256 = hashlib.sha256(b"prefix\n").hexdigest()
    payloads = {
        "benchmarks/single_stage_native_equivalent_quality_successor_authority.py": (
            b"DIAG5_QUALIFIED_FILE_PATHS = frozenset(())\n"
            b"DIAG5_FROZEN_NUMERICAL_PATHS = frozenset(())\n"
            + f'DIAG5_PLAN_SHA256: str = "{plan_prefix_sha256}"\n'.encode()
        ),
        "benchmarks/single_stage_native_equivalent_quality_diagnostic_receipt.py": (
            b"VALUE = 1\n"
        ),
        "src/pkg/module.py": b"VALUE = 1\n",
        "tests/test_module.py": b"def test_value():\n    assert True\n",
        qualifier.PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH: canonical_json_bytes(
            {
                "reconstruction": {},
                "schema_version": qualifier.PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
            }
        ),
    }
    for relative, payload in payloads.items():
        path = worktree / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    manifest_payload = _execution_manifest(payloads)
    manifest_path = worktree / qualifier.EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
    plan_payload = b"prefix\n## Qualification Record\n"
    plan_path = worktree / qualifier._PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_bytes(plan_payload)
    worktree_descriptor = os.open(
        worktree,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    worktree_stat = os.fstat(worktree_descriptor)
    bindings: list[qualifier.ExecutionSourceFileBinding] = []
    for relative, payload in payloads.items():
        descriptor, device, inode = qualifier._bound_relative_regular_descriptor(
            worktree_descriptor,
            relative,
            size_bytes=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        bindings.append(
            qualifier.ExecutionSourceFileBinding(
                qualifier.ExecutionSourceEntry(
                    relative,
                    hashlib.sha256(payload).hexdigest(),
                    len(payload),
                ),
                worktree / relative,
                descriptor,
                device,
                inode,
            )
        )
    manifest_descriptor, manifest_device, manifest_inode = (
        qualifier._bound_relative_regular_descriptor(
            worktree_descriptor,
            qualifier.EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
            size_bytes=len(manifest_payload),
            sha256=hashlib.sha256(manifest_payload).hexdigest(),
        )
    )
    plan_descriptor, plan_device, plan_inode = (
        qualifier._bound_relative_regular_descriptor(
            worktree_descriptor,
            qualifier._PREQUALIFICATION_PLAN_SOURCE_RELATIVE_PATH,
            size_bytes=len(plan_payload),
            sha256=hashlib.sha256(plan_payload).hexdigest(),
        )
    )
    monkeypatch.setattr(
        qualifier,
        "_validate_execution_source_membership",
        lambda worktree_root, entries, authority_payload: None,
    )
    return qualifier.ExecutionSourceBindings(
        worktree_root=worktree,
        worktree_descriptor=worktree_descriptor,
        worktree_device=worktree_stat.st_dev,
        worktree_inode=worktree_stat.st_ino,
        execution_root=None,
        execution_descriptor=-1,
        execution_device=-1,
        execution_inode=-1,
        authority_sha256=hashlib.sha256(manifest_payload).hexdigest(),
        entries_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    relative: {
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "size_bytes": len(payload),
                    }
                    for relative, payload in payloads.items()
                }
            )
        ).hexdigest(),
        manifest_live_path=manifest_path,
        manifest_live_descriptor=manifest_descriptor,
        manifest_live_device=manifest_device,
        manifest_live_inode=manifest_inode,
        manifest_size_bytes=len(manifest_payload),
        manifest_copied_path=None,
        manifest_copied_descriptor=-1,
        manifest_copied_device=-1,
        manifest_copied_inode=-1,
        plan_source_path=plan_path,
        plan_descriptor=plan_descriptor,
        plan_device=plan_device,
        plan_inode=plan_inode,
        plan_size_bytes=len(plan_payload),
        plan_sha256=hashlib.sha256(plan_payload).hexdigest(),
        plan_prefix_sha256=plan_prefix_sha256,
        entries=tuple(
            sorted(bindings, key=lambda binding: binding.entry.relative_path)
        ),
    )


class _FakeProducer:
    def __init__(self, outcome: ScientificOutcome) -> None:
        self.outcome = outcome
        self.produce_calls = 0
        self.validate_roots: list[Path] = []

    def produce(
        self,
        staging_root: Path,
        runtime_identity: CpuRuntimeIdentity,
    ) -> ProducedEvidence:
        self.produce_calls += 1
        qualifier._publish_json(
            staging_root / "evidence.json",
            {"backend": runtime_identity.backend, "complete": True},
        )
        predecessor_path = (
            staging_root / qualifier.PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH
        )
        qualifier._publish_bytes(
            predecessor_path,
            (
                qualifier.REPOSITORY_ROOT
                / qualifier.PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH
            ).read_bytes(),
        )
        return ProducedEvidence(
            scientific_outcome=self.outcome,
            numerical_identity=_identity(),
            timings_ns=(
                ("process_started_monotonic_ns", 1),
                ("compile_started_monotonic_ns", 2),
                ("compile_completed_monotonic_ns", 3),
                ("state_ready_monotonic_ns", 4),
                ("solve_started_monotonic_ns", 5),
                ("solve_stopped_monotonic_ns", 2_000_000_005),
                ("finalizer_completed_monotonic_ns", 2_000_000_006),
                ("quality_replay_completed_monotonic_ns", 2_000_000_007),
                ("terminal_completed_monotonic_ns", 2_000_000_008),
                ("endpoint_audit_completed_monotonic_ns", 2_000_000_009),
                ("serialization_started_monotonic_ns", 2_000_000_010),
                ("serialization_completed_monotonic_ns", 2_000_000_011),
            ),
            callback_count=0,
            execution_source_manifest_sha256=_HASH,
            execution_source_entries_sha256=_HASH,
            prequalification_plan_control=tuple(
                {
                    "plan_prefix_sha256": DIAG5_PLAN_SHA256,
                    "schema_version": (
                        qualifier.PREQUALIFICATION_PLAN_CONTROL_SCHEMA_VERSION
                    ),
                    "sha256": DIAG5_BLANK_PLAN_SHA256,
                    "size_bytes": DIAG5_BLANK_PLAN_SIZE_BYTES,
                    "snapshot_relative_path": (
                        qualifier.PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH
                    ),
                    "source_relative_path": DIAG5_PLAN_RELATIVE_PATH,
                }.items()
            ),
            source_manifest_sha256=_HASH,
            source_manifest_entries=(),
            native_extension_path=runtime_identity.native_extension_path,
            native_extension_sha256=runtime_identity.native_extension_sha256,
            native_extension_size_bytes=(runtime_identity.native_extension_size_bytes),
            native_extension_link_count=runtime_identity.native_extension_link_count,
            native_extension_device=Path(runtime_identity.native_extension_path)
            .stat(follow_symlinks=False)
            .st_dev,
            native_extension_inode=Path(runtime_identity.native_extension_path)
            .stat(follow_symlinks=False)
            .st_ino,
            predecessor_postmortem=qualifier._artifact_ref(
                predecessor_path,
                staging_root,
                qualifier.PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
            ),
            native_reference_artifact_sha256=_HASH,
            input_fingerprint="input-fingerprint",
            configuration_fingerprint="configuration-fingerprint",
            policy_sha256=_HASH,
        )

    def validate(
        self,
        artifact_root: Path,
        qualification: Mapping[str, JsonValue],
    ) -> ScientificOutcome:
        self.validate_roots.append(artifact_root)
        assert (artifact_root / "evidence.json").read_bytes() == canonical_json_bytes(
            {"backend": "cpu", "complete": True}
        )
        assert qualification["scientific_outcome"] == self.outcome.value
        return self.outcome


class _InvalidEvidenceProducer(_FakeProducer):
    def validate(
        self,
        artifact_root: Path,
        qualification: Mapping[str, JsonValue],
    ) -> ScientificOutcome:
        del artifact_root, qualification
        raise QualificationError("injected evidence failure")


class _Waitable(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...


class _SpawnBlockingProducer(_FakeProducer):
    def __init__(self, ledger: Path, release: _Waitable) -> None:
        super().__init__(ScientificOutcome.NO_HIT)
        self.ledger = ledger
        self.release = release

    def produce(
        self,
        staging_root: Path,
        runtime_identity: CpuRuntimeIdentity,
    ) -> ProducedEvidence:
        descriptor = os.open(
            self.ledger,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC,
            0o600,
        )
        try:
            os.write(descriptor, f"{os.getpid()}\n".encode())
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if not self.release.wait(5.0):
            raise RuntimeError("race release timed out")
        return super().produce(staging_root, runtime_identity)


def _spawn_qualification_racer(
    output_root: str,
    ledger: str,
    release: _Waitable,
) -> None:
    """One racing child, running the runtime the environment it is judged against declares.

    A ``spawn`` child re-imports THIS module but not ``tests/conftest.py``, and
    the parent's FP64 is an in-process ``jax.config.update`` that conftest
    performs -- so the child inherited it only when the operator happened to
    export ``JAX_ENABLE_X64``, and under the box's default environment both
    children died in ``observe_cpu_runtime`` with "qualification requires JAX
    FP64" before either could reach the exclusion this test measures.  The
    result was a suite whose 60th test passed or failed on an environment
    variable rather than on the code.

    The qualifier judges the RUNTIME against the environment it is handed, so
    the child configures the runtime that environment declares.  Nothing is
    widened: ``observe_cpu_runtime`` still refuses a child that is not FP64 CPU.
    """

    environment = _environment()
    qualifier.jax.config.update(
        "jax_enable_x64", environment["JAX_ENABLE_X64"] == "true"
    )
    try:
        run_qualification(
            Path(output_root),
            producer=_SpawnBlockingProducer(Path(ledger), release),
            environment=environment,
        )
    except FileExistsError:
        raise SystemExit(20) from None


@pytest.mark.parametrize("existing_kind", ["final", "partial"])
def test_preexisting_final_or_partial_blocks_before_execution(
    tmp_path: Path,
    existing_kind: str,
) -> None:
    output_root = tmp_path / "qualification"
    existing = (
        output_root
        if existing_kind == "final"
        else tmp_path / "qualification.partial-existing"
    )
    existing.mkdir()
    inode = existing.stat().st_ino
    producer = _FakeProducer(ScientificOutcome.QUALITY_HIT)

    with pytest.raises(FileExistsError):
        run_qualification(
            output_root,
            producer=producer,
            environment=_environment(),
        )

    assert producer.produce_calls == 0
    assert existing.stat().st_ino == inode


def test_linux_rename_noreplace_preserves_existing_destination(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    destination.mkdir()
    destination_inode = destination.stat().st_ino
    (destination / "marker").write_text("retained", encoding="utf-8")

    with pytest.raises(FileExistsError):
        qualifier._rename_noreplace(source, destination)

    assert source.is_dir()
    assert destination.stat().st_ino == destination_inode
    assert (destination / "marker").read_text(encoding="utf-8") == "retained"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("JAX_PLATFORMS", "cuda"),
        ("JAX_ENABLE_X64", "false"),
        ("XLA_PYTHON_CLIENT_PREALLOCATE", "true"),
        ("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "0"),
    ],
)
def test_cpu_policy_failure_precedes_output_and_execution(
    tmp_path: Path,
    name: str,
    value: str,
) -> None:
    environment = _environment()
    environment[name] = value
    output_root = tmp_path / "qualification"
    producer = _FakeProducer(ScientificOutcome.QUALITY_HIT)

    with pytest.raises(QualificationError, match=name):
        run_qualification(
            output_root,
            producer=producer,
            environment=environment,
        )

    assert producer.produce_calls == 0
    assert not output_root.exists()
    assert not tuple(tmp_path.glob("qualification.partial-*"))


def test_cli_rejects_every_nonexact_output_root_before_delegation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def unexpected_run(
        output_root: Path,
        *,
        producer: qualifier.CpuQualificationProducer,
        environment: Mapping[str, str],
    ) -> dict[str, JsonValue]:
        nonlocal called
        del output_root, producer, environment
        called = True
        return {}

    monkeypatch.setattr(qualifier, "run_qualification", unexpected_run)
    with pytest.raises(QualificationError, match="output root must be exactly"):
        qualifier.main(["--output-root", str(tmp_path / "wrong")])
    assert not called


def test_cli_exact_root_delegates_once(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: list[Path] = []

    def fake_run(
        output_root: Path,
        *,
        producer: qualifier.CpuQualificationProducer,
        environment: Mapping[str, str],
    ) -> dict[str, JsonValue]:
        del producer, environment
        observed.append(output_root)
        return {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "qualification_passed": False,
        }

    monkeypatch.setattr(qualifier, "run_qualification", fake_run)
    assert qualifier.main(["--output-root", str(qualifier.EXPECTED_OUTPUT_ROOT)]) == 0
    assert observed == [qualifier.EXPECTED_OUTPUT_ROOT]


def test_source_snapshot_membership_uses_typed_execution_authority() -> None:
    relative_paths = (
        "benchmarks/process_gpu_monitor.py",
        "src/simsopt_jax/runtime/trace_annotations.py",
        "tests/benchmarks/test_process_gpu_monitor.py",
    )
    execution_sources = SimpleNamespace(
        entries=tuple(
            SimpleNamespace(entry=qualifier.ExecutionSourceEntry(relative, _HASH, 1))
            for relative in relative_paths
        ),
        plan_source_path=qualifier.REPOSITORY_ROOT / DIAG5_PLAN_RELATIVE_PATH,
    )
    roots = qualifier._source_roots(
        qualifier.REPOSITORY_ROOT,
        execution_sources,
    )
    repository_paths = {
        root.relative_path for root in roots if root.role != "native_extension"
    }

    assert repository_paths == set(relative_paths) | {
        qualifier.EXECUTION_SOURCE_AUTHORITY_RELATIVE_PATH,
        qualifier.PREQUALIFICATION_PLAN_SNAPSHOT_RELATIVE_PATH,
    }
    assert {root.role for root in roots} == set(qualifier.DIAG5_CPU_SNAPSHOT_ROLES)
    assert len([root for root in roots if root.role == "native_extension"]) == 1


def test_only_predecessor_postmortem_document_maps_to_execution_source() -> None:
    assert (
        qualifier._execution_source_role(
            qualifier.PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH
        )
        == "execution_source"
    )
    assert qualifier._execution_source_role("docs/other.json") == "configuration"


def test_execution_source_authority_parser_is_strict_and_canonical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = {"src/pkg/module.py": b"VALUE = 1\n"}
    payload = _execution_manifest(entries)

    parsed, entries_sha256 = qualifier._parse_execution_source_authority(payload)

    assert parsed == (
        qualifier.ExecutionSourceEntry(
            "src/pkg/module.py",
            hashlib.sha256(entries["src/pkg/module.py"]).hexdigest(),
            len(entries["src/pkg/module.py"]),
        ),
    )
    assert (
        entries_sha256
        == hashlib.sha256(
            canonical_json_bytes(
                {
                    "src/pkg/module.py": {
                        "sha256": parsed[0].sha256,
                        "size_bytes": parsed[0].size_bytes,
                    }
                }
            )
        ).hexdigest()
    )
    with pytest.raises(QualificationError, match="not canonical"):
        qualifier._parse_execution_source_authority(payload.rstrip(b"\n"))
    duplicate = payload.replace(
        b'{"entries":',
        b'{"entries":{},"entries":',
        1,
    )
    with pytest.raises(QualificationError, match="duplicate keys"):
        qualifier._parse_execution_source_authority(duplicate)


def test_live_python_membership_revalidates_addition_and_removal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = {
        "benchmarks/entry.py": b"VALUE = 1\n",
        "src/simsopt/_version.py": b"VERSION = 'test'\n",
        "examples/entry.py": b"VALUE = 1\n",
    }
    for relative, payload in payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    entries = tuple(
        qualifier.ExecutionSourceEntry(
            relative,
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )
        for relative, payload in sorted(payloads.items())
    )
    authority_payload = (
        b"DIAG5_QUALIFIED_FILE_PATHS: Final = frozenset(())\n"
        b"DIAG5_FROZEN_NUMERICAL_PATHS: Final = frozenset(())\n"
        b"DIAG5_EXECUTION_SOURCE_ENTRY_COUNT: Final = 3\n"
    )
    monkeypatch.setattr(
        qualifier,
        "_BROAD_EXECUTION_SOURCE_COUNTS",
        (("benchmarks", 1), ("src", 1), ("examples", 1)),
    )
    root_descriptor = os.open(
        tmp_path,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        qualifier._validate_execution_source_membership(
            root_descriptor,
            entries,
            authority_payload,
        )
        added = tmp_path / "benchmarks/added.py"
        added.write_bytes(b"ADDED = True\n")
        with pytest.raises(QualificationError, match="benchmarks membership differs"):
            qualifier._validate_execution_source_membership(
                root_descriptor,
                entries,
                authority_payload,
            )
        added.unlink()
        (tmp_path / "examples/entry.py").unlink()
        with pytest.raises(QualificationError, match="examples membership differs"):
            qualifier._validate_execution_source_membership(
                root_descriptor,
                entries,
                authority_payload,
            )
    finally:
        os.close(root_descriptor)


def test_execution_source_copy_is_exact_and_survives_root_relocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()
    copied = qualifier._bootstrap_copy_execution_source(authority, staging)
    final = tmp_path / "qualification"
    staging.rename(final)

    copied.validate(
        copied_required=True,
        copied_root=final / qualifier._EXECUTION_SOURCE_DIRECTORY,
    )
    extra = final / qualifier._EXECUTION_SOURCE_DIRECTORY / "src/extra.py"
    extra.parent.chmod(0o755)
    extra.write_bytes(b"EXTRA = True\n")
    try:
        with pytest.raises(QualificationError, match="membership differs"):
            copied.validate(
                copied_required=True,
                copied_root=final / qualifier._EXECUTION_SOURCE_DIRECTORY,
            )
    finally:
        copied.close()


def test_execution_source_copy_failure_closes_live_and_copied_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    authority_descriptors = {
        authority.worktree_descriptor,
        authority.manifest_live_descriptor,
        authority.plan_descriptor,
        *(binding.live_descriptor for binding in authority.entries),
    }
    copied_descriptors: set[int] = set()
    original_bound = qualifier._bound_relative_regular_descriptor

    def recording_bound(
        root_descriptor: int,
        relative_path: str,
        *,
        size_bytes: int,
        sha256: str,
    ) -> tuple[int, int, int]:
        result = original_bound(
            root_descriptor,
            relative_path,
            size_bytes=size_bytes,
            sha256=sha256,
        )
        copied_descriptors.add(result[0])
        return result

    def fail_chmod(path: Path, mode: int) -> None:
        del path, mode
        raise OSError("injected copied-tree seal failure")

    monkeypatch.setattr(
        qualifier,
        "_bound_relative_regular_descriptor",
        recording_bound,
    )
    monkeypatch.setattr(Path, "chmod", fail_chmod)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()

    with pytest.raises(OSError, match="copied-tree seal"):
        qualifier._bootstrap_copy_execution_source(authority, staging)

    for descriptor in authority_descriptors | copied_descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_root_down_open_rejects_intermediate_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "module.py").write_bytes(b"VALUE = 1\n")
    (root / "src").symlink_to(outside, target_is_directory=True)
    descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    try:
        with pytest.raises(OSError):
            qualifier._open_relative_regular(
                descriptor,
                "src/module.py",
                flags=os.O_RDONLY,
            )
    finally:
        os.close(descriptor)


def test_inherited_descriptor_authority_reconstructs_and_revalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()
    copied = qualifier._bootstrap_copy_execution_source(authority, staging)
    assert copied.execution_root is not None
    monkeypatch.setenv(
        qualifier._EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT,
        qualifier._execution_source_descriptor_payload(copied).decode("utf-8"),
    )

    inherited = qualifier._inherited_execution_source_bindings(
        copied.worktree_root,
        copied.execution_root,
    )

    inherited.validate(copied_required=True)
    assert inherited.entries_sha256 == copied.entries_sha256
    inherited.close()


def test_inherited_descriptor_failure_closes_every_owned_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()
    copied = qualifier._bootstrap_copy_execution_source(authority, staging)
    assert copied.execution_root is not None
    payload = qualifier._execution_source_descriptor_payload(copied).decode("utf-8")
    descriptors = qualifier._inherited_descriptor_numbers(payload)
    monkeypatch.setenv(qualifier._EXECUTION_SOURCE_DESCRIPTOR_ENVIRONMENT, payload)
    monkeypatch.setattr(
        qualifier,
        "_bootstrap_constant",
        lambda tree, name: (_ for _ in ()).throw(
            QualificationError("injected inherited construction failure")
        ),
    )

    with pytest.raises(QualificationError, match="injected inherited"):
        qualifier._inherited_execution_source_bindings(
            copied.worktree_root,
            copied.execution_root,
        )

    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_retained_plan_mutation_invalidates_copied_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()
    copied = qualifier._bootstrap_copy_execution_source(authority, staging)
    copied.plan_source_path.write_bytes(b"changed\n## Qualification Record\n")
    try:
        with pytest.raises(
            QualificationError, match="retained execution-source changed"
        ):
            copied.validate(copied_required=True)
    finally:
        copied.close()


def test_predecessor_postmortem_copy_is_descriptor_bound_and_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        qualifier,
        "validate_diag5_predecessor_failure",
        lambda evidence, *, repository_root: None,
    )
    monkeypatch.setattr(
        qualifier,
        "validate_diag5_predecessor_postmortem_artifact",
        lambda artifact_root, reference: {},
    )
    authority = _small_execution_authority(tmp_path, monkeypatch)
    staging = tmp_path / "qualification.partial-claim"
    staging.mkdir()
    try:
        reference = qualifier._publish_predecessor_postmortem(authority, staging)
        assert reference.relative_path == (
            qualifier.PREDECESSOR_POSTMORTEM_ARTIFACT_RELATIVE_PATH
        )
        assert (
            reference.schema_version == qualifier.PREDECESSOR_POSTMORTEM_SCHEMA_VERSION
        )
        assert qualifier._load_json_artifact(staging, reference) == {
            "reconstruction": {},
            "schema_version": qualifier.PREDECESSOR_POSTMORTEM_SCHEMA_VERSION,
        }
        source = authority.worktree_root / (
            qualifier.PREDECESSOR_POSTMORTEM_SOURCE_RELATIVE_PATH
        )
        source.write_bytes(b"changed")
        with pytest.raises(
            QualificationError, match="retained execution-source changed"
        ):
            authority.validate(copied_required=False)
    finally:
        authority.close()


def test_native_runtime_identity_mutation_is_rejected_before_snapshot() -> None:
    runtime = qualifier.observe_cpu_runtime(_environment())

    with pytest.raises(QualificationError, match="native extension runtime"):
        qualifier._capture_native_extension_binding(
            replace(runtime, native_extension_sha256="b" * 64)
        )


def test_live_native_hardlink_is_bound_and_copied_to_unique_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native = tmp_path / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = tmp_path / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    assert native.name != Path(DIAG5_NATIVE_COPY_RELATIVE_PATH).name
    assert binding.relative_path == DIAG5_NATIVE_COPY_RELATIVE_PATH
    authority = _small_execution_authority(tmp_path / "authority", monkeypatch)
    monkeypatch.setattr(qualifier.simsoptpp, "__file__", str(native))
    try:
        qualifier.ImportedSourceBindings((binding,)).validate()
        snapshot = publish_immutable_snapshot(
            tmp_path / "snapshot",
            qualifier._source_roots(authority.worktree_root, authority),
            worktree=WorktreeIdentity(
                "a" * 40,
                _HASH,
                _HASH,
                str(authority.worktree_root),
            ),
            required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
        )
        copied = snapshot.root / binding.relative_path
        assert observed.st_nlink == 2
        assert copied.stat(follow_symlinks=False).st_nlink == 1
        qualifier._validate_execution_source_snapshot(snapshot, authority, binding)
    finally:
        binding_descriptor = binding.descriptor
        qualifier.ImportedSourceBindings((binding,)).close()
        with pytest.raises(OSError):
            os.fstat(binding_descriptor)
        authority.close()


def test_live_native_binding_rejects_alias_byte_mutation(tmp_path: Path) -> None:
    native = tmp_path / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = tmp_path / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    alias.write_bytes(b"NATIVE-extension")
    try:
        with pytest.raises(QualificationError, match="imported source binding changed"):
            qualifier.ImportedSourceBindings((binding,)).validate()
    finally:
        qualifier.ImportedSourceBindings((binding,)).close()


def test_live_native_binding_rejects_link_topology_drift(tmp_path: Path) -> None:
    native = tmp_path / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = tmp_path / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    alias.unlink()
    try:
        with pytest.raises(QualificationError, match="imported source binding changed"):
            qualifier.ImportedSourceBindings((binding,)).validate()
    finally:
        qualifier.ImportedSourceBindings((binding,)).close()


def test_live_native_binding_rejects_identical_byte_path_replacement(
    tmp_path: Path,
) -> None:
    native = tmp_path / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = tmp_path / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    replacement = tmp_path / "replacement.so"
    replacement.write_bytes(native.read_bytes())
    os.replace(replacement, native)
    try:
        with pytest.raises(QualificationError, match="native extension path binding"):
            qualifier.ImportedSourceBindings((binding,)).validate()
    finally:
        qualifier.ImportedSourceBindings((binding,)).close()


def test_live_native_binding_rejects_intermediate_symlink_replacement(
    tmp_path: Path,
) -> None:
    live_directory = tmp_path / "live"
    live_directory.mkdir()
    native = live_directory / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = live_directory / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    detached = tmp_path / "detached"
    live_directory.rename(detached)
    live_directory.symlink_to(detached, target_is_directory=True)
    try:
        with pytest.raises(
            QualificationError, match="native extension directory binding differs"
        ):
            qualifier.ImportedSourceBindings((binding,)).validate()
    finally:
        qualifier.ImportedSourceBindings((binding,)).close()


def test_cpu_six_role_snapshot_round_trip_uses_exact_execution_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    snapshot_root = tmp_path / "snapshot"
    snapshot = publish_immutable_snapshot(
        snapshot_root,
        qualifier._source_roots(authority.worktree_root, authority),
        worktree=WorktreeIdentity(
            "a" * 40,
            _HASH,
            _HASH,
            str(authority.worktree_root),
        ),
        required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
    )
    loaded = load_snapshot(
        snapshot_root,
        required_roles=DIAG5_CPU_SNAPSHOT_ROLES,
    )
    runtime = qualifier.observe_cpu_runtime(_environment())
    native_binding = qualifier._capture_native_extension_binding(runtime)

    assert loaded == snapshot
    qualifier._validate_execution_source_snapshot(
        loaded,
        authority,
        native_binding,
    )
    qualifier.ImportedSourceBindings((native_binding,)).close()
    authority.close()


def test_real_producer_refuses_unbootstrapped_imported_execution(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        QualificationError,
        match="must execute from bootstrapped source",
    ):
        qualifier.ProductionCpuQualificationProducer().produce(
            tmp_path,
            qualifier.observe_cpu_runtime(_environment()),
        )


def test_production_options_are_exact_full_budget_safeguarded_max_two() -> None:
    assert NEQ_GNTR3_OPTIONS.maximum_accepted_steps == 256
    assert NEQ_GNTR3_OPTIONS.maximum_attempts == 300
    assert NEQ_GNTR3_OPTIONS.maximum_nonlinear_corrections == 2
    assert NEQ_GNTR3_OPTIONS.enable_step_bound_safeguard is True


def test_direct_bootstrap_claims_before_invalid_source_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "qualification"
    monkeypatch.setattr(qualifier, "_EXPECTED_OUTPUT_ROOT_TEXT", str(output_root))
    monkeypatch.setattr(
        qualifier.sys,
        "argv",
        ["qualifier.py", "--output-root", str(output_root)],
    )
    for name, value in qualifier._REQUIRED_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        qualifier,
        "_load_execution_source_authority",
        lambda repository: (_ for _ in ()).throw(
            QualificationError("injected invalid execution authority")
        ),
    )

    with pytest.raises(QualificationError, match="invalid execution authority"):
        qualifier._direct_bootstrap()

    assert (tmp_path / "qualification.partial-claim").is_dir()
    assert not output_root.exists()


def test_execution_source_shared_locks_span_binding_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority = _small_execution_authority(tmp_path, monkeypatch)
    paths = (
        authority.manifest_live_path,
        authority.plan_source_path,
        *(binding.live_path for binding in authority.entries),
    )
    contenders = tuple(os.open(path, os.O_RDONLY | os.O_CLOEXEC) for path in paths)
    try:
        for descriptor in contenders:
            with pytest.raises(BlockingIOError):
                qualifier.fcntl.flock(
                    descriptor,
                    qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
                )
        authority.close()
        for descriptor in contenders:
            qualifier.fcntl.flock(
                descriptor,
                qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
            )
    finally:
        for descriptor in contenders:
            os.close(descriptor)


def test_execution_source_exclusive_lock_blocks_admission(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "source.py"
    source.write_bytes(b"VALUE = 1\n")
    root_descriptor = os.open(
        root,
        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
    )
    exclusive = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
    qualifier.fcntl.flock(
        exclusive,
        qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
    )
    try:
        with pytest.raises(QualificationError, match="exclusively locked"):
            qualifier._bound_relative_regular_descriptor(
                root_descriptor,
                "source.py",
                size_bytes=source.stat().st_size,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            )
    finally:
        os.close(exclusive)
        os.close(root_descriptor)


def _retained_test_tree(tmp_path: Path) -> tuple[Path, qualifier.RetainedRegularTree]:
    source = tmp_path / "source-tree"
    nested = source / "nested"
    nested.mkdir(parents=True)
    source.chmod(0o755)
    nested.chmod(0o750)
    (source / "root.json").write_bytes(b'{"root":true}\n')
    (source / "root.json").chmod(0o444)
    (nested / "array.npy").write_bytes(b"array-bytes")
    (nested / "array.npy").chmod(0o640)
    return source, qualifier._admit_regular_tree(source)


def test_retained_input_tree_copies_only_locked_descriptors_and_holds_locks(
    tmp_path: Path,
) -> None:
    source, tree = _retained_test_tree(tmp_path)
    contenders = tuple(
        os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        for path in (source / "root.json", source / "nested/array.npy")
    )
    destination = tmp_path / "copy"
    try:
        for descriptor in contenders:
            with pytest.raises(BlockingIOError):
                qualifier.fcntl.flock(
                    descriptor,
                    qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
                )
        qualifier._copy_retained_regular_tree(tree, destination)
        assert (destination / "root.json").read_bytes() == b'{"root":true}\n'
        assert (destination / "nested/array.npy").read_bytes() == b"array-bytes"
        assert stat.S_IMODE(destination.stat().st_mode) == 0o555
        assert stat.S_IMODE((destination / "nested").stat().st_mode) == 0o555
        assert stat.S_IMODE((destination / "root.json").stat().st_mode) == 0o444
        assert stat.S_IMODE((destination / "nested/array.npy").stat().st_mode) == 0o444
        tree.close()
        for descriptor in contenders:
            qualifier.fcntl.flock(
                descriptor,
                qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
            )
    finally:
        for descriptor in contenders:
            os.close(descriptor)


@pytest.mark.parametrize("mutation", ["add", "remove"])
def test_retained_input_tree_rejects_membership_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, tree = _retained_test_tree(tmp_path)
    try:
        if mutation == "add":
            (source / "extra.json").write_bytes(b"{}\n")
        else:
            (source / "root.json").unlink()
        with pytest.raises(QualificationError, match="membership differs"):
            tree.validate()
    finally:
        tree.close()


@pytest.mark.parametrize("mutation", ["replace", "overwrite"])
def test_retained_input_tree_rejects_leaf_identity_or_byte_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, tree = _retained_test_tree(tmp_path)
    leaf = source / "root.json"
    try:
        if mutation == "replace":
            replacement = source / "replacement"
            replacement.write_bytes(leaf.read_bytes())
            replacement.chmod(0o444)
            os.replace(replacement, leaf)
        else:
            leaf.chmod(0o644)
            leaf.write_bytes(b'{"root":fals}\n')
            leaf.chmod(0o444)
        with pytest.raises(
            QualificationError,
            match="retained execution-source changed|topology differs",
        ):
            tree.validate()
    finally:
        tree.close()


def test_retained_input_tree_rejects_intermediate_symlink_replacement(
    tmp_path: Path,
) -> None:
    source, tree = _retained_test_tree(tmp_path)
    nested = source / "nested"
    detached = source / "detached"
    nested.rename(detached)
    nested.symlink_to(detached.name, target_is_directory=True)
    try:
        with pytest.raises(QualificationError, match="invalid leaf"):
            tree.validate()
    finally:
        tree.close()


def test_live_native_shared_lock_spans_binding_lifetime(tmp_path: Path) -> None:
    native = tmp_path / "simsoptpp.test.so"
    native.write_bytes(b"native-extension")
    alias = tmp_path / "simsoptpp.test.alias.so"
    os.link(native, alias)
    observed = native.stat(follow_symlinks=False)
    runtime = replace(
        qualifier.observe_cpu_runtime(_environment()),
        native_extension_path=str(native.resolve(strict=True)),
        native_extension_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        native_extension_size_bytes=observed.st_size,
        native_extension_link_count=observed.st_nlink,
    )
    binding = qualifier._capture_native_extension_binding(runtime)
    contender = os.open(native, os.O_RDONLY | os.O_CLOEXEC)
    try:
        with pytest.raises(BlockingIOError):
            qualifier.fcntl.flock(
                contender,
                qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
            )
        qualifier.ImportedSourceBindings((binding,)).close()
        qualifier.fcntl.flock(
            contender,
            qualifier.fcntl.LOCK_EX | qualifier.fcntl.LOCK_NB,
        )
    finally:
        os.close(contender)


def test_retained_import_descriptor_rejects_identical_byte_path_replacement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bound.py"
    source.write_bytes(b"VALUE = 1\n")
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    observed = os.fstat(descriptor)
    binding = qualifier.ImportedSourceBindings(
        (
            qualifier.ImportedSourceBinding(
                "bound.py",
                source,
                descriptor,
                observed.st_dev,
                observed.st_ino,
                observed.st_size,
                qualifier._sha256_descriptor(descriptor),
                observed.st_nlink,
            ),
        )
    )
    binding.validate()
    replacement = tmp_path / "replacement.py"
    replacement.write_bytes(source.read_bytes())
    os.replace(replacement, source)
    try:
        with pytest.raises(QualificationError, match="imported source binding changed"):
            binding.validate()
    finally:
        binding.close()


def test_real_telemetry_seam_emits_all_twenty_four_envelopes_and_outer_solves() -> None:
    attempts = receipt_module.MAXIMUM_ATTEMPTS
    subtrial_shape = (attempts, 3)
    inactive = receipt_module.AttemptOutcome.INACTIVE.value
    accepted = receipt_module.AttemptOutcome.ACCEPTED.value
    history: dict[str, JsonValue] = {
        "accepted_steps": 1,
        "attempts": 1,
        "quality_latch": False,
        "retryable_rejections": 0,
        "rows": [
            {"outcome": accepted},
            *({"outcome": inactive} for _ in range(attempts - 1)),
        ],
        "status": receipt_module.LoopStatus.ATTEMPT_LIMIT.value,
    }
    integer_vectors = {
        "nonlinear_corrections": np.zeros(attempts, dtype="<i4"),
        "steihaug_solve_calls": np.zeros(attempts, dtype="<i4"),
        "subtrial_count": np.zeros(attempts, dtype="<i4"),
        "selected_subtrial_index": np.full(attempts, -1, dtype="<i4"),
    }
    integer_vectors["nonlinear_corrections"][0] = 2
    integer_vectors["steihaug_solve_calls"][0] = 1
    integer_vectors["subtrial_count"][0] = 1
    integer_vectors["selected_subtrial_index"][0] = 0
    maximum_ratio = np.full(attempts, np.nan, dtype="<f8")
    path_ratio = np.full(attempts, np.nan, dtype="<f8")
    maximum_ratio[0] = 5.0e-4
    path_ratio[0] = 8.0e-4
    float_matrices = {
        name: np.full(subtrial_shape, np.nan, dtype="<f8")
        for name in receipt_module.DIAG4_SUBTRIAL_FLOAT_MATRIX_FIELDS
    }
    float_matrices["subtrial_trust_radius"][0, 0] = 2.0**-10
    float_matrices["subtrial_actual_reduction"][0, 0] = 1.0
    float_matrices["subtrial_predicted_reduction"][0, 0] = 1.0
    float_matrices["subtrial_maximum_individual_correction_step_ratio"][0, 0] = (
        maximum_ratio[0]
    )
    float_matrices["subtrial_correction_path_step_ratio"][0, 0] = path_ratio[0]
    float_matrices["subtrial_corrected_radius_ratio"][0, 0] = 0.9
    integer_matrices = {
        name: np.zeros(subtrial_shape, dtype="<i4")
        for name in receipt_module.DIAG4_SUBTRIAL_INTEGER_WORK_FIELDS
    }
    integer_matrices["subtrial_steihaug_solve_calls"][0, 0] = 1
    integer_matrices["subtrial_total_hvp_evaluations"][0, 0] = 4
    integer_matrices["subtrial_nonlinear_corrections"][0, 0] = 2
    integer_matrices["subtrial_joint_evaluations"][0, 0] = 5
    integer_matrices["subtrial_joint_linearizations"][0, 0] = 3
    integer_matrices["subtrial_joint_value_evaluations"][0, 0] = 2
    integer_matrices["subtrial_objective_residual_linearizations"][0, 0] = 1
    integer_matrices["subtrial_gram_factorizations"][0, 0] = 3
    integer_matrices["subtrial_gram_solves"][0, 0] = 7
    subtrial_outcome = np.zeros(subtrial_shape, dtype="<i4")
    subtrial_outcome[0, 0] = tuple(receipt_module.AttemptOutcome).index(
        receipt_module.AttemptOutcome.ACCEPTED
    )
    loop_history = SimpleNamespace(
        **integer_vectors,
        maximum_individual_correction_step_ratio=maximum_ratio,
        correction_path_step_ratio=path_ratio,
        subtrial_outcome=subtrial_outcome,
        **float_matrices,
        **integer_matrices,
    )
    payload = qualifier._safeguard_telemetry_from_loop_history(
        history=history,
        history_reference=ArtifactRef("history.json", _HASH, 1, "history-v1"),
        numerical_identity=_identity(),
        loop_history=loop_history,
    )

    envelope_fields = {
        "nonlinear_corrections",
        "maximum_individual_correction_step_ratio",
        "correction_path_step_ratio",
        "steihaug_solve_calls",
        "subtrial_count",
        "selected_subtrial_index",
        *receipt_module.DIAG4_SUBTRIAL_MATRIX_FIELDS,
    }
    assert len(envelope_fields) == 24
    assert envelope_fields <= set(payload)
    assert payload["steihaug_solve_calls"]["dtype"] == "<i4"
    assert payload["steihaug_solve_calls"]["shape"] == [attempts]
    receipt_module.validate_safeguard_telemetry_payload(payload)
    mutated = copy.deepcopy(payload)
    mutated["steihaug_solve_calls"]["values"][0] = 0
    with pytest.raises(ValueError, match="envelope hash differs"):
        receipt_module.validate_safeguard_telemetry_payload(mutated)
