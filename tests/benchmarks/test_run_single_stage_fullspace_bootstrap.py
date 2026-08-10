from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks import run_single_stage_fullspace_bootstrap as runner
from benchmarks.run_single_stage_fullspace_gpu import SnapshotChildInvocation
from benchmarks.single_stage_fullspace_bootstrap import (
    SCHEMA_VERSION as BOOTSTRAP_SCHEMA_VERSION,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    canonical_json_bytes,
)
from simsopt_jax_adapters.geo.single_stage_fullspace_parity import (
    SameStateFieldComparison,
    SameStateParityReport,
    same_state_field_tolerances,
)


def _write_reference(
    campaign: Path, relative_path: str, schema_version: str, payload: bytes
) -> ArtifactRef:
    path = campaign / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    import hashlib

    return ArtifactRef(
        relative_path,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        schema_version,
    )


def _passing_parity_report(state_sha256: str = "a" * 64) -> SameStateParityReport:
    return SameStateParityReport(
        state_little_endian_sha256=state_sha256,
        comparisons=tuple(
            SameStateFieldComparison(field, tolerance, 0.0, 0.0, True)
            for field, tolerance in same_state_field_tolerances().items()
        ),
        passed=True,
    )


def test_parent_uses_bootstrap_snapshot_entrypoint_and_validates_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    publication = SimpleNamespace(root=snapshot)
    invocation = SnapshotChildInvocation(
        argv=(sys.executable, "-I", str(snapshot / runner.ENTRYPOINT_RELATIVE_PATH)),
        cwd=snapshot,
        environment={"BOUND": "yes"},
    )
    receipt = canonical_json_bytes({"schema_version": runner.SCHEMA_VERSION})
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner, "prepare_execution_snapshot", lambda *_args, **_kwargs: publication
    )

    def fake_invocation(*_args: object, **kwargs: object) -> SnapshotChildInvocation:
        captured.update(kwargs)
        return invocation

    monkeypatch.setattr(runner, "build_snapshot_child_invocation", fake_invocation)
    monkeypatch.setattr(
        runner,
        "validate_completion_receipt",
        lambda payload, **_kwargs: {"payload": payload.decode()},
    )

    def child_runner(
        child: SnapshotChildInvocation,
    ) -> subprocess.CompletedProcess[bytes]:
        (campaign / "completion.json").write_bytes(receipt)
        (campaign / "completion.json").chmod(0o444)
        return subprocess.CompletedProcess(child.argv, 0, stdout=receipt, stderr=b"")

    result = runner.run_bootstrap_campaign(
        campaign,
        native_extension_path=tmp_path / "simsoptpp.so",
        interpreter=Path(sys.executable),
        environment={"PATH": "/usr/bin"},
        child_runner=child_runner,
    )

    assert result == receipt
    assert captured["entrypoint_relative_path"] == runner.ENTRYPOINT_RELATIVE_PATH
    assert captured["request_argv"] == ("--snapshot-child",)


def test_parent_propagates_child_failure_without_accepting_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    publication = SimpleNamespace(root=snapshot)
    invocation = SnapshotChildInvocation(
        argv=(sys.executable, "-I", "runner.py"),
        cwd=snapshot,
        environment={},
    )
    validated = False

    monkeypatch.setattr(
        runner, "prepare_execution_snapshot", lambda *_args, **_kwargs: publication
    )
    monkeypatch.setattr(
        runner, "build_snapshot_child_invocation", lambda *_args, **_kwargs: invocation
    )

    def forbidden_validation(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal validated
        validated = True
        return {}

    monkeypatch.setattr(runner, "validate_completion_receipt", forbidden_validation)

    with pytest.raises(subprocess.CalledProcessError) as error:
        runner.run_bootstrap_campaign(
            campaign,
            native_extension_path=tmp_path / "simsoptpp.so",
            interpreter=Path(sys.executable),
            environment={},
            child_runner=lambda child: subprocess.CompletedProcess(
                child.argv, 7, stdout=b"partial", stderr=b"failed"
            ),
        )

    assert error.value.returncode == 7
    assert not validated


def test_snapshot_child_publishes_runtime_then_bootstrap_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    manifest_ref = _write_reference(
        campaign,
        "source-snapshot/source-manifest.json",
        SOURCE_MANIFEST_SCHEMA_VERSION,
        canonical_json_bytes({"schema_version": SOURCE_MANIFEST_SCHEMA_VERSION}),
    )
    publication = SimpleNamespace(
        root=snapshot,
        manifest_sha256=manifest_ref.sha256,
        source_identity=lambda _campaign: SimpleNamespace(
            snapshot_manifest=manifest_ref
        ),
    )
    runtime_ref = _write_reference(
        campaign,
        "evidence/runtime-evidence.json",
        RUNTIME_EVIDENCE_SCHEMA_VERSION,
        canonical_json_bytes({"schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION}),
    )
    factory_calls = 0
    validation_calls = 0

    monkeypatch.chdir(snapshot)
    monkeypatch.setattr(runner, "load_snapshot", lambda _path: publication)
    monkeypatch.setattr(
        runner,
        "publish_child_runtime_provenance",
        lambda *_args, **_kwargs: (SimpleNamespace(), runtime_ref),
    )

    bootstrap = SimpleNamespace(z0=object(), problem=object())

    def factory() -> object:
        nonlocal factory_calls
        factory_calls += 1
        return bootstrap

    def publish_bootstrap(path: Path, **kwargs: object) -> ArtifactRef:
        produced = kwargs["bootstrap_factory"]
        assert callable(produced)
        produced()
        path.write_bytes(b"bootstrap\n")
        path.chmod(0o444)
        return _write_reference(
            campaign,
            "artifacts/reference-copy.json",
            BOOTSTRAP_SCHEMA_VERSION,
            b"bootstrap\n",
        )

    def validate_receipt(payload: bytes, **_kwargs: object) -> dict[str, object]:
        nonlocal validation_calls
        validation_calls += 1
        assert payload.endswith(b"\n")
        return {}

    monkeypatch.setattr(runner, "publish_bootstrap_artifact", publish_bootstrap)
    monkeypatch.setattr(runner, "validate_completion_receipt", validate_receipt)
    parity_report = _passing_parity_report()

    environment = {
        "SIMSOPT_FULLSPACE_CAMPAIGN_ROOT": str(campaign),
        "SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256": manifest_ref.sha256,
    }
    receipt = runner.execute_snapshot_child(
        campaign_root=campaign,
        process_argv=(
            str(snapshot / runner.ENTRYPOINT_RELATIVE_PATH),
            "--snapshot-child",
        ),
        environment=environment,
        bootstrap_factory=factory,
        parity_report_builder=lambda _z0, _problem: parity_report,
    )

    assert factory_calls == 1
    assert validation_calls == 1
    assert runtime_ref.resolve_and_validate(campaign).stat().st_mode & 0o222 == 0
    assert (
        b'"schema_version":"single-stage-fullspace-bootstrap-completion-v2"' in receipt
    )
    assert (campaign / "completion.json").read_bytes() == receipt
    assert (campaign / "completion.json").stat().st_mode & 0o222 == 0


def test_completion_validator_rejects_forged_passing_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    manifest_ref = _write_reference(
        campaign,
        "source-snapshot/source-manifest.json",
        SOURCE_MANIFEST_SCHEMA_VERSION,
        canonical_json_bytes({"schema_version": SOURCE_MANIFEST_SCHEMA_VERSION}),
    )
    runtime_ref = _write_reference(
        campaign,
        "evidence/runtime-evidence.json",
        RUNTIME_EVIDENCE_SCHEMA_VERSION,
        canonical_json_bytes({"schema_version": RUNTIME_EVIDENCE_SCHEMA_VERSION}),
    )
    (campaign / runtime_ref.relative_path).chmod(0o444)
    bootstrap_ref = _write_reference(
        campaign,
        "artifacts/fullspace-bootstrap.json",
        BOOTSTRAP_SCHEMA_VERSION,
        canonical_json_bytes({"schema_version": BOOTSTRAP_SCHEMA_VERSION}),
    )
    publication = SimpleNamespace(
        root=snapshot,
        source_identity=lambda _campaign: SimpleNamespace(
            snapshot_manifest=manifest_ref
        ),
    )
    runtime = SimpleNamespace(
        source_identity=SimpleNamespace(snapshot_manifest=manifest_ref)
    )
    monkeypatch.setattr(runner, "load_snapshot", lambda _path: publication)
    monkeypatch.setattr(
        runner, "validate_runtime_evidence", lambda *_args, **_kwargs: runtime
    )
    monkeypatch.setattr(
        runner,
        "validate_bootstrap_artifact",
        lambda *_args, **_kwargs: {
            "runtime_evidence": runner._artifact_payload(runtime_ref),
            "state": {"little_endian_sha256": "a" * 64},
        },
    )
    payload = runner._completion_payload(
        publication,
        campaign_root=campaign,
        runtime_evidence=runtime_ref,
        bootstrap_artifact=bootstrap_ref,
        same_state_parity=_passing_parity_report(),
    )
    parity = payload["same_state_parity"]
    assert isinstance(parity, dict)
    comparisons = parity["comparisons"]
    assert isinstance(comparisons, tuple)
    first = comparisons[0]
    assert isinstance(first, dict)
    first["max_tolerance_ratio"] = 2.0

    with pytest.raises(ValueError, match="comparison did not pass"):
        runner.validate_completion_receipt(
            canonical_json_bytes(payload),
            campaign_root=campaign,
            snapshot_root=snapshot,
        )


def test_snapshot_child_rejects_wrong_binding_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "source-snapshot"
    snapshot.mkdir()
    publication = SimpleNamespace(root=snapshot, manifest_sha256="a" * 64)
    called = False

    monkeypatch.chdir(snapshot)
    monkeypatch.setattr(runner, "load_snapshot", lambda _path: publication)

    def forbidden(*_args: object, **_kwargs: object) -> tuple[object, ArtifactRef]:
        nonlocal called
        called = True
        raise AssertionError("publication must not start")

    monkeypatch.setattr(runner, "publish_child_runtime_provenance", forbidden)

    with pytest.raises(ValueError, match="not bound"):
        runner.execute_snapshot_child(
            campaign_root=campaign,
            process_argv=("runner.py",),
            environment={
                "SIMSOPT_FULLSPACE_CAMPAIGN_ROOT": str(campaign),
                "SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256": "b" * 64,
            },
        )

    assert not called


def test_parent_refuses_existing_campaign_before_child(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    native = tmp_path / "simsoptpp.so"
    native.write_bytes(b"native")

    with pytest.raises(FileExistsError, match="existing output"):
        runner.run_bootstrap_campaign(
            campaign,
            native_extension_path=native,
            interpreter=Path(sys.executable),
            environment=os.environ,
        )
