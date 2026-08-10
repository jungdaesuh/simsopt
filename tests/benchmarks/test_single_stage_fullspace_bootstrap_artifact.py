from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import single_stage_fullspace_bootstrap as artifact_module
from benchmarks.single_stage_fullspace_bootstrap import (
    SCHEMA_VERSION,
    BootstrapArtifactError,
    publish_bootstrap_artifact,
    validate_bootstrap_artifact,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    ImportBinding,
    RuntimeEvidence,
    RuntimeIdentity,
    RuntimeObservation,
    SourceIdentity,
    canonical_json_bytes,
)
from simsopt_jax_adapters.geo.single_stage_fullspace import (
    Float64Fingerprint,
    SingleStageFullSpaceBootstrap,
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reference(root: Path, relative_path: str, schema_version: str) -> ArtifactRef:
    payload = canonical_json_bytes({"schema_version": schema_version})
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactRef(relative_path, _sha(payload), len(payload), schema_version)


def _evidence(campaign: Path) -> tuple[ArtifactRef, RuntimeEvidence]:
    runtime_ref = _reference(
        campaign, "runtime/runtime-evidence.json", RUNTIME_EVIDENCE_SCHEMA_VERSION
    )
    source = SourceIdentity(
        snapshot_manifest=ArtifactRef(
            "snapshot/source-manifest.json",
            "a" * 64,
            123,
            SOURCE_MANIFEST_SCHEMA_VERSION,
        ),
        git_head="b" * 40,
        tracked_diff_sha256="c" * 64,
        untracked_bytes_manifest_sha256="d" * 64,
        repo_root="/source/repo",
    )
    runtime = RuntimeIdentity(
        argv=("python", "bootstrap.py"),
        cwd="/campaign/snapshot",
        python_executable="/venv/bin/python",
        python_version="3.13.0",
        jax_version="0.6.0",
        jaxlib_version="0.6.0",
        simsopt_module_path="/campaign/snapshot/src/simsopt/__init__.py",
        simsopt_jax_module_path="/campaign/snapshot/src/simsopt_jax/__init__.py",
        native_extension_path="/campaign/snapshot/lib/simsoptpp.so",
        backend="gpu",
        device_uuid="GPU-fixture",
        driver_version="fixture-driver",
        effective_environment_sha256="e" * 64,
    )
    binding = ImportBinding("runner", "benchmarks/bootstrap.py", 1, "f" * 64)
    observation = RuntimeObservation(
        runtime_identity=runtime,
        entrypoint_binding=binding,
        import_bindings=(binding,),
        effective_environment=(),
        device_name="fixture GPU",
        platform_version="fixture platform",
    )
    return runtime_ref, RuntimeEvidence(source, observation, "a" * 64)


def _fingerprint(name: str, value: float) -> Float64Fingerprint:
    scalar = np.asarray(value, dtype="<f8")
    return Float64Fingerprint(
        name=name,
        value=value,
        hexadecimal=value.hex(),
        little_endian_sha256=_sha(scalar.tobytes()),
    )


def _bootstrap() -> SingleStageFullSpaceBootstrap:
    problem = SimpleNamespace(
        layout=SimpleNamespace(
            coil_dof_count=461,
            surface_dof_count=253,
            total_dof_count=716,
        ),
        exact_mask_indices=np.arange(254, dtype=np.int32),
    )
    targets = tuple(
        _fingerprint(name, value)
        for name, value in zip(
            (
                "volume_target",
                "iota_target",
                "major_radius_target",
                "length_target",
            ),
            (0.1, -0.4, 1.0, 10.0),
            strict=True,
        )
    )
    return SingleStageFullSpaceBootstrap(
        problem=problem,
        z0=np.linspace(-1.0, 1.0, 716, dtype=np.float64),
        targets=targets,
        initial_boozer_residual_norm=1.0e-14,
        first_base_current=_fingerprint("first_base_current", 1.0e5),
    )


@pytest.fixture
def published(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, RuntimeEvidence]:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "snapshot"
    snapshot.mkdir()
    runtime_ref, evidence = _evidence(campaign)
    monkeypatch.setattr(
        artifact_module,
        "validate_runtime_evidence",
        lambda *_args, **_kwargs: evidence,
    )
    path = campaign / "bootstrap/bootstrap.json"
    path.parent.mkdir()
    reference = publish_bootstrap_artifact(
        path,
        campaign_root=campaign,
        snapshot_root=snapshot,
        runtime_evidence=runtime_ref,
        bootstrap_factory=_bootstrap,
    )
    assert reference.relative_path == "bootstrap/bootstrap.json"
    assert reference.schema_version == SCHEMA_VERSION
    return campaign, path, evidence


def test_bootstrap_artifact_round_trip_is_canonical_complete_and_read_only(
    published: tuple[Path, Path, RuntimeEvidence],
) -> None:
    campaign, path, evidence = published

    document = validate_bootstrap_artifact(
        path, campaign_root=campaign, snapshot_root=campaign / "snapshot"
    )

    assert path.stat().st_mode & 0o222 == 0
    assert canonical_json_bytes(document) == path.read_bytes()
    assert document["schema_version"] == SCHEMA_VERSION
    assert document["source_identity"]["git_head"] == evidence.source_identity.git_head
    assert document["runtime_identity"]["device_uuid"] == (
        evidence.observation.runtime_identity.device_uuid
    )
    assert document["layout"]["total_dof_count"] == 716
    assert len(document["state"]["values"]) == 716
    assert len(document["exact_mask"]["values"]) == 254


def test_bootstrap_artifact_refuses_overwrite(
    published: tuple[Path, Path, RuntimeEvidence],
) -> None:
    campaign, path, _evidence_value = published
    runtime_ref = ArtifactRef(
        "runtime/runtime-evidence.json",
        _sha((campaign / "runtime/runtime-evidence.json").read_bytes()),
        (campaign / "runtime/runtime-evidence.json").stat().st_size,
        RUNTIME_EVIDENCE_SCHEMA_VERSION,
    )

    with pytest.raises(FileExistsError):
        publish_bootstrap_artifact(
            path,
            campaign_root=campaign,
            snapshot_root=campaign / "snapshot",
            runtime_evidence=runtime_ref,
            bootstrap_factory=_bootstrap,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    (
        (
            lambda document: document["state"]["values"].__setitem__(0, 99.0),
            "state fingerprint differs",
        ),
        (
            lambda document: document["targets"]["targets"][0].__setitem__(
                "hexadecimal", "0x0.0p+0"
            ),
            "scalar identity differs",
        ),
        (
            lambda document: document["runtime_identity"].__setitem__("backend", "cpu"),
            "runtime identity differs",
        ),
        (
            lambda document: document["exact_mask"]["values"].__setitem__(1, 0),
            "mask fingerprint or indices differ",
        ),
    ),
)
def test_bootstrap_validator_rejects_semantic_tampering(
    published: tuple[Path, Path, RuntimeEvidence],
    mutate: object,
    message: str,
) -> None:
    campaign, path, _evidence_value = published
    document = artifact_module.load_canonical_json_bytes(path.read_bytes())
    assert isinstance(document, dict)
    assert callable(mutate)
    mutate(document)
    path.chmod(0o644)
    path.write_bytes(canonical_json_bytes(document))
    path.chmod(0o444)

    with pytest.raises(BootstrapArtifactError, match=message):
        validate_bootstrap_artifact(
            path, campaign_root=campaign, snapshot_root=campaign / "snapshot"
        )


def test_bootstrap_validator_rejects_noncanonical_bytes(
    published: tuple[Path, Path, RuntimeEvidence],
) -> None:
    campaign, path, _evidence_value = published
    path.chmod(0o644)
    path.write_text("{}", encoding="utf-8")
    path.chmod(0o444)

    with pytest.raises(ValueError, match="canonical"):
        validate_bootstrap_artifact(
            path, campaign_root=campaign, snapshot_root=campaign / "snapshot"
        )


def test_bootstrap_validator_rejects_writable_artifact(
    published: tuple[Path, Path, RuntimeEvidence],
) -> None:
    campaign, path, _evidence_value = published
    path.chmod(0o644)

    with pytest.raises(BootstrapArtifactError, match="read-only"):
        validate_bootstrap_artifact(
            path, campaign_root=campaign, snapshot_root=campaign / "snapshot"
        )


def test_bootstrap_validator_rejects_symlink(
    published: tuple[Path, Path, RuntimeEvidence],
) -> None:
    campaign, path, _evidence_value = published
    link = campaign / "bootstrap-link.json"
    link.symlink_to(path)

    with pytest.raises(BootstrapArtifactError, match="symlink"):
        validate_bootstrap_artifact(
            link, campaign_root=campaign, snapshot_root=campaign / "snapshot"
        )


def test_bootstrap_producer_rejects_wrong_runtime_schema_before_solving(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    snapshot = campaign / "snapshot"
    snapshot.mkdir()
    wrong_ref = _reference(campaign, "runtime.json", "wrong-runtime-schema")
    called = False

    def forbidden_factory() -> SingleStageFullSpaceBootstrap:
        nonlocal called
        called = True
        return _bootstrap()

    with pytest.raises(BootstrapArtifactError, match="reference schema"):
        publish_bootstrap_artifact(
            campaign / "bootstrap.json",
            campaign_root=campaign,
            snapshot_root=snapshot,
            runtime_evidence=wrong_ref,
            bootstrap_factory=forbidden_factory,
        )

    assert not called
