"""Publish one provenance-bound feasible bootstrap from an immutable snapshot."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final, TypeAlias

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_ROOT = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_SOURCE_ROOT))

from benchmarks.run_single_stage_fullspace_gpu import (
    SNAPSHOT_DIRECTORY,
    SnapshotChildInvocation,
    build_snapshot_child_invocation,
    prepare_execution_snapshot,
    publish_child_runtime_provenance,
)
from benchmarks.single_stage_fullspace_bootstrap import (
    SCHEMA_VERSION as BOOTSTRAP_SCHEMA_VERSION,
)
from benchmarks.single_stage_fullspace_bootstrap import (
    publish_bootstrap_artifact,
    validate_bootstrap_artifact,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    ArtifactRef,
    JsonValue,
    SnapshotPublication,
    activate_snapshot_source_imports,
    canonical_json_bytes,
    load_canonical_json_bytes,
    load_snapshot,
    validate_runtime_evidence,
)

if os.environ.get("SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256") is not None:
    activate_snapshot_source_imports(_SOURCE_ROOT)

from simsopt_jax_adapters.geo.single_stage_fullspace import (
    SingleStageFullSpaceBootstrap,
    build_single_stage_fullspace_bootstrap,
)
from simsopt_jax_adapters.geo.single_stage_fullspace_parity import (
    SameStateParityReport,
    build_same_state_parity_report,
    same_state_field_tolerances,
)

SCHEMA_VERSION: Final = "single-stage-fullspace-bootstrap-completion-v2"
ENTRYPOINT_RELATIVE_PATH: Final = "benchmarks/run_single_stage_fullspace_bootstrap.py"
_SNAPSHOT_MANIFEST_ENV: Final = "SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256"
_CAMPAIGN_ROOT_ENV: Final = "SIMSOPT_FULLSPACE_CAMPAIGN_ROOT"
_RUNTIME_RELATIVE_PATH: Final = "evidence/runtime-evidence.json"
_BOOTSTRAP_RELATIVE_PATH: Final = "artifacts/fullspace-bootstrap.json"
_COMPLETION_RELATIVE_PATH: Final = "completion.json"
BootstrapFactory: TypeAlias = Callable[[], SingleStageFullSpaceBootstrap]
ParityReportBuilder: TypeAlias = Callable[[object, object], SameStateParityReport]
ChildRunner: TypeAlias = Callable[
    [SnapshotChildInvocation], subprocess.CompletedProcess[bytes]
]


def _artifact_payload(reference: ArtifactRef) -> dict[str, JsonValue]:
    return {
        "relative_path": reference.relative_path,
        "schema_version": reference.schema_version,
        "sha256": reference.sha256,
        "size_bytes": reference.size_bytes,
    }


def _artifact_reference(value: object, context: str) -> ArtifactRef:
    if not isinstance(value, dict) or frozenset(value) != frozenset(
        ("relative_path", "schema_version", "sha256", "size_bytes")
    ):
        raise ValueError(f"{context} reference is not canonical")
    relative_path = value["relative_path"]
    schema_version = value["schema_version"]
    sha256 = value["sha256"]
    size_bytes = value["size_bytes"]
    if (
        not isinstance(relative_path, str)
        or not isinstance(schema_version, str)
        or not isinstance(sha256, str)
        or isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
    ):
        raise TypeError(f"{context} reference fields have invalid types")
    return ArtifactRef(relative_path, sha256, size_bytes, schema_version)


def _completion_payload(
    publication: SnapshotPublication,
    *,
    campaign_root: Path,
    runtime_evidence: ArtifactRef,
    bootstrap_artifact: ArtifactRef,
    same_state_parity: SameStateParityReport,
) -> dict[str, JsonValue]:
    source_manifest = publication.source_identity(campaign_root).snapshot_manifest
    return {
        "bootstrap_artifact": _artifact_payload(bootstrap_artifact),
        "runtime_evidence": _artifact_payload(runtime_evidence),
        "same_state_parity": asdict(same_state_parity),
        "schema_version": SCHEMA_VERSION,
        "source_manifest": _artifact_payload(source_manifest),
    }


def validate_completion_receipt(
    payload: bytes,
    *,
    campaign_root: Path,
    snapshot_root: Path,
) -> dict[str, JsonValue]:
    """Validate canonical completion bytes and every referenced sealed artifact."""

    document = load_canonical_json_bytes(payload)
    if not isinstance(document, dict) or frozenset(document) != frozenset(
        (
            "bootstrap_artifact",
            "runtime_evidence",
            "same_state_parity",
            "schema_version",
            "source_manifest",
        )
    ):
        raise ValueError("bootstrap completion receipt keys do not match schema")
    if document["schema_version"] != SCHEMA_VERSION:
        raise ValueError("bootstrap completion receipt schema is invalid")

    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(snapshot_root)
    expected_manifest = publication.source_identity(campaign).snapshot_manifest
    source_manifest = _artifact_reference(
        document["source_manifest"], "source manifest"
    )
    if source_manifest != expected_manifest:
        raise ValueError("completion source manifest differs from the snapshot")
    source_manifest.resolve_and_validate(campaign)

    runtime_reference = _artifact_reference(
        document["runtime_evidence"], "runtime evidence"
    )
    if runtime_reference.schema_version != RUNTIME_EVIDENCE_SCHEMA_VERSION:
        raise ValueError("completion runtime evidence schema is invalid")
    runtime_path = runtime_reference.resolve_and_validate(campaign)
    if runtime_path.stat().st_mode & 0o222:
        raise ValueError("completion runtime evidence must be read-only")
    runtime = validate_runtime_evidence(
        runtime_path,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    if runtime.source_identity.snapshot_manifest != expected_manifest:
        raise ValueError("completion runtime evidence uses another source manifest")

    bootstrap_reference = _artifact_reference(
        document["bootstrap_artifact"], "bootstrap artifact"
    )
    if bootstrap_reference.schema_version != BOOTSTRAP_SCHEMA_VERSION:
        raise ValueError("completion bootstrap artifact schema is invalid")
    bootstrap_path = bootstrap_reference.resolve_and_validate(campaign)
    bootstrap = validate_bootstrap_artifact(
        bootstrap_path,
        campaign_root=campaign,
        snapshot_root=publication.root,
    )
    if bootstrap["runtime_evidence"] != _artifact_payload(runtime_reference):
        raise ValueError("completion bootstrap and runtime references differ")
    parity = document["same_state_parity"]
    if not isinstance(parity, dict) or frozenset(parity) != frozenset(
        ("state_little_endian_sha256", "comparisons", "passed")
    ):
        raise ValueError("same-state parity payload keys do not match schema")
    if parity["passed"] is not True or (
        parity["state_little_endian_sha256"]
        != bootstrap["state"]["little_endian_sha256"]
    ):
        raise ValueError("same-state parity did not pass at the bootstrap state")
    comparisons = parity["comparisons"]
    expected_tolerances = same_state_field_tolerances()
    if not isinstance(comparisons, list) or len(comparisons) != len(
        expected_tolerances
    ):
        raise ValueError("same-state parity comparison count differs")
    seen_fields: set[str] = set()
    for comparison in comparisons:
        if not isinstance(comparison, dict) or frozenset(comparison) != frozenset(
            (
                "field",
                "tolerance",
                "max_absolute_error",
                "max_tolerance_ratio",
                "passed",
            )
        ):
            raise ValueError("same-state parity comparison keys do not match schema")
        field = comparison["field"]
        if not isinstance(field, str) or field not in expected_tolerances:
            raise ValueError("same-state parity contains an unknown field")
        if field in seen_fields:
            raise ValueError("same-state parity contains a duplicate field")
        seen_fields.add(field)
        if comparison["tolerance"] != asdict(expected_tolerances[field]):
            raise ValueError("same-state parity tolerance differs from the ledger")
        absolute_error = comparison["max_absolute_error"]
        tolerance_ratio = comparison["max_tolerance_ratio"]
        if (
            isinstance(absolute_error, bool)
            or not isinstance(absolute_error, (int, float))
            or not math.isfinite(float(absolute_error))
            or float(absolute_error) < 0.0
            or isinstance(tolerance_ratio, bool)
            or not isinstance(tolerance_ratio, (int, float))
            or not math.isfinite(float(tolerance_ratio))
            or float(tolerance_ratio) < 0.0
            or float(tolerance_ratio) > 1.0
            or comparison["passed"] is not True
        ):
            raise ValueError("same-state parity comparison did not pass")
    if seen_fields != set(expected_tolerances):
        raise ValueError("same-state parity fields differ from the frozen contract")
    return document


def execute_snapshot_child(
    *,
    campaign_root: Path,
    process_argv: Sequence[str],
    environment: Mapping[str, str],
    bootstrap_factory: BootstrapFactory = build_single_stage_fullspace_bootstrap,
    parity_report_builder: ParityReportBuilder = build_same_state_parity_report,
) -> bytes:
    """Create and validate runtime evidence plus the one authoritative bootstrap."""

    campaign = campaign_root.resolve(strict=True)
    publication = load_snapshot(campaign / SNAPSHOT_DIRECTORY)
    if (
        environment.get(_CAMPAIGN_ROOT_ENV) != str(campaign)
        or environment.get(_SNAPSHOT_MANIFEST_ENV) != publication.manifest_sha256
    ):
        raise ValueError("snapshot child is not bound to this campaign and manifest")
    if Path.cwd().resolve(strict=True) != publication.root:
        raise ValueError("snapshot child must execute from the immutable snapshot root")

    _runtime, runtime_reference = publish_child_runtime_provenance(
        publication,
        campaign_root=campaign,
        process_argv=process_argv,
        environment=environment,
    )
    runtime_path = runtime_reference.resolve_and_validate(campaign)
    runtime_path.chmod(0o444)

    artifact_directory = campaign / "artifacts"
    artifact_directory.mkdir()
    bootstrap = bootstrap_factory()
    bootstrap_reference = publish_bootstrap_artifact(
        campaign / _BOOTSTRAP_RELATIVE_PATH,
        campaign_root=campaign,
        snapshot_root=publication.root,
        runtime_evidence=runtime_reference,
        bootstrap_factory=lambda: bootstrap,
    )
    parity_report = parity_report_builder(bootstrap.z0, bootstrap.problem)
    receipt = canonical_json_bytes(
        _completion_payload(
            publication,
            campaign_root=campaign,
            runtime_evidence=runtime_reference,
            bootstrap_artifact=bootstrap_reference,
            same_state_parity=parity_report,
        )
    )
    validate_completion_receipt(
        receipt,
        campaign_root=campaign,
        snapshot_root=publication.root,
    )
    completion_path = campaign / _COMPLETION_RELATIVE_PATH
    with completion_path.open("xb") as stream:
        stream.write(receipt)
        stream.flush()
        os.fsync(stream.fileno())
    completion_path.chmod(0o444)
    return receipt


def _run_child(
    invocation: SnapshotChildInvocation,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
    )


def run_bootstrap_campaign(
    campaign_root: Path,
    *,
    native_extension_path: Path,
    interpreter: Path,
    environment: Mapping[str, str],
    child_runner: ChildRunner = _run_child,
) -> bytes:
    """Snapshot current sources, run one isolated child, and verify its receipt."""

    publication = prepare_execution_snapshot(
        campaign_root,
        native_extension_path=native_extension_path,
    )
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign_root,
        interpreter=interpreter,
        request_argv=("--snapshot-child",),
        environment=environment,
        entrypoint_relative_path=ENTRYPOINT_RELATIVE_PATH,
    )
    completed = child_runner(invocation)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(
            completed.returncode,
            completed.args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    validate_completion_receipt(
        completed.stdout,
        campaign_root=campaign_root,
        snapshot_root=publication.root,
    )
    completion_path = campaign_root / _COMPLETION_RELATIVE_PATH
    if (
        completion_path.read_bytes() != completed.stdout
        or completion_path.stat().st_mode & 0o222
    ):
        raise ValueError("persisted completion receipt differs or is writable")
    return completed.stdout


def _reject_duplicate_options(argv: Sequence[str]) -> None:
    options = [token.partition("=")[0] for token in argv if token.startswith("--")]
    if len(options) != len(set(options)):
        raise ValueError("duplicate command-line option")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--native-extension", type=Path)
    parser.add_argument("--snapshot-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = tuple(sys.argv[1:] if argv is None else argv)
    _reject_duplicate_options(raw_argv)
    options = _parser().parse_args(raw_argv)
    if options.snapshot_child:
        if options.campaign_root is not None or options.native_extension is not None:
            raise ValueError("snapshot child accepts no live-source paths")
        campaign_value = os.environ.get(_CAMPAIGN_ROOT_ENV)
        if campaign_value is None:
            raise ValueError("snapshot child campaign binding is absent")
        payload = execute_snapshot_child(
            campaign_root=Path(campaign_value),
            process_argv=(str(Path(__file__).resolve()), *raw_argv),
            environment=os.environ,
        )
    else:
        if options.campaign_root is None or options.native_extension is None:
            raise ValueError("parent requires --campaign-root and --native-extension")
        payload = run_bootstrap_campaign(
            options.campaign_root,
            native_extension_path=options.native_extension,
            interpreter=Path(sys.executable),
            environment=os.environ,
        )
    sys.stdout.buffer.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "ENTRYPOINT_RELATIVE_PATH",
    "SCHEMA_VERSION",
    "execute_snapshot_child",
    "main",
    "run_bootstrap_campaign",
    "validate_completion_receipt",
)
