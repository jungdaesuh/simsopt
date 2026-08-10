from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from benchmarks.run_single_stage_fullspace_gpu import (
    build_snapshot_child_invocation,
    executed_run_receipt_payload,
    explicit_source_roots,
    prepare_execution_snapshot,
    publish_child_runtime_provenance,
)
from benchmarks.single_stage_fullspace_snapshot import (
    RUNTIME_EVIDENCE_SCHEMA_VERSION,
    SOURCE_MANIFEST_FILENAME,
    SOURCE_MANIFEST_SCHEMA_VERSION,
    ArtifactRef,
    ImportBinding,
    RuntimeIdentity,
    RuntimeObservation,
    SnapshotValidationError,
    SourceRoot,
    WorktreeIdentity,
    build_runtime_evidence,
    canonical_json_bytes,
    capture_worktree_identity,
    effective_environment,
    load_canonical_json_bytes,
    load_snapshot,
    publish_immutable_snapshot,
    publish_runtime_evidence,
    select_physical_gpu_identity,
    validate_runtime_evidence,
)
from benchmarks.validate_single_stage_fullspace_campaign import validate_campaign
from simsopt_jax.solve.fullspace import FullSpaceRoute

_ZERO_SHA256 = "0" * 64
_ONE_SHA256 = "1" * 64
_GIT_HEAD = "a" * 40


def _write(path: Path, payload: bytes = b"VALUE = 1\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _source_roots(source: Path) -> tuple[SourceRoot, ...]:
    return (
        SourceRoot(
            "execution_source",
            _write(source / "simsopt" / "__init__.py"),
            "src/simsopt/__init__.py",
        ),
        SourceRoot(
            "execution_source",
            _write(source / "simsopt_jax" / "__init__.py"),
            "src/simsopt_jax/__init__.py",
        ),
        SourceRoot(
            "execution_source",
            _write(source / "simsopt_jax_adapters" / "__init__.py"),
            "src/simsopt_jax_adapters/__init__.py",
        ),
        SourceRoot(
            "configuration",
            _write(source / "problem.json", b"{}\n"),
            "inputs/problem.json",
        ),
        SourceRoot(
            "benchmark",
            _write(source / "runner.py"),
            "benchmarks/run_single_stage_fullspace_gpu.py",
        ),
        SourceRoot(
            "test",
            _write(source / "test_contract.py"),
            "tests/test_contract.py",
        ),
        SourceRoot(
            "native_extension",
            _write(source / "simsoptpp.so", b"native-extension\n"),
            "src/simsoptpp.so",
        ),
    )


def _worktree(repo_root: Path) -> WorktreeIdentity:
    return WorktreeIdentity(
        git_head=_GIT_HEAD,
        tracked_diff_sha256=_ZERO_SHA256,
        untracked_bytes_manifest_sha256=_ONE_SHA256,
        repo_root=str(repo_root.resolve()),
    )


def _snapshot(tmp_path: Path):
    campaign = tmp_path / "campaign"
    campaign.mkdir(parents=True)
    publication = publish_immutable_snapshot(
        campaign / "snapshot",
        _source_roots(tmp_path / "source"),
        worktree=_worktree(tmp_path),
    )
    return campaign, publication


def _observation(publication) -> RuntimeObservation:
    entry_by_path = {entry.relative_path: entry for entry in publication.entries}
    module_paths = {
        "simsopt": "src/simsopt/__init__.py",
        "simsopt_jax": "src/simsopt_jax/__init__.py",
        "simsopt_jax_adapters": "src/simsopt_jax_adapters/__init__.py",
        "simsoptpp": "src/simsoptpp.so",
    }
    bindings = tuple(
        ImportBinding(
            module=module,
            relative_path=relative_path,
            size_bytes=entry_by_path[relative_path].size_bytes,
            sha256=entry_by_path[relative_path].sha256,
        )
        for module, relative_path in module_paths.items()
    )
    entrypoint_entry = entry_by_path["benchmarks/run_single_stage_fullspace_gpu.py"]
    entrypoint_binding = ImportBinding(
        module="__entrypoint__",
        relative_path=entrypoint_entry.relative_path,
        size_bytes=entrypoint_entry.size_bytes,
        sha256=entrypoint_entry.sha256,
    )
    environment = effective_environment(
        {
            "CUDA_VISIBLE_DEVICES": "GPU-test",
            "JAX_ENABLE_X64": "true",
            "PATH": "/usr/bin",
        }
    )
    environment_digest = hashlib.sha256(
        (
            json.dumps(
                dict(environment),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    ).hexdigest()
    return RuntimeObservation(
        runtime_identity=RuntimeIdentity(
            argv=("benchmarks/run_single_stage_fullspace_gpu.py", "--phase=first-eval"),
            cwd=str(publication.root),
            python_executable=sys.executable,
            python_version=sys.version,
            jax_version="0.test",
            jaxlib_version="0.test",
            simsopt_module_path=str(publication.root / module_paths["simsopt"]),
            simsopt_jax_module_path=str(publication.root / module_paths["simsopt_jax"]),
            native_extension_path=str(publication.root / module_paths["simsoptpp"]),
            backend="gpu",
            device_uuid="GPU-test",
            driver_version="999.1",
            effective_environment_sha256=environment_digest,
        ),
        entrypoint_binding=entrypoint_binding,
        import_bindings=bindings,
        effective_environment=environment,
        device_name="Test GPU",
        platform_version="CUDA test",
    )


def _artifact_ref(campaign: Path, path: Path, schema_version: str) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "relative_path": path.relative_to(campaign).as_posix(),
        "schema_version": schema_version,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _campaign_with_executed_run(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    from benchmarks.single_stage_fullspace_receipt import (
        SCHEMA_VERSION,
        contract_sha256,
    )

    campaign, publication = _snapshot(tmp_path)
    source = publication.source_identity(campaign)
    observation = _observation(publication)
    runtime_evidence = build_runtime_evidence(
        publication.root,
        source_identity=source,
        observation=observation,
    )
    runtime_path = campaign / "evidence" / "runtime-evidence.json"
    runtime_path.parent.mkdir()
    runtime_ref = publish_runtime_evidence(
        runtime_path,
        runtime_evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    run_path = campaign / "runs" / "CFS-P0.json"
    run_path.parent.mkdir()
    run_document: dict[str, object] = {
        "contract_sha256": contract_sha256(),
        "endpoint_certificate": {"certified": True},
        "request": {
            "device": "rtx5090",
            "phase": "first-eval",
            "route": FullSpaceRoute.CFS_P0,
            "sample": None,
            "steps": None,
        },
        "runtime_evidence": asdict(runtime_ref),
        "runtime_identity": asdict(observation.runtime_identity),
        "schema_version": SCHEMA_VERSION,
        "source_identity": asdict(source),
        "terminal_status": "SUCCESS",
        "timing": {"solve_seconds": 1.0},
        "trajectory_equivalence_required": False,
        "transfer_audit": {"hot_d2h": 0},
    }
    run_path.write_bytes(canonical_json_bytes(run_document))
    gate_path = campaign / "gates" / "downstream.json"
    gate_path.parent.mkdir()
    gate_path.write_bytes(
        canonical_json_bytes({"schema_version": SCHEMA_VERSION, "passed": False})
    )
    gate_ref = _artifact_ref(campaign, gate_path, SCHEMA_VERSION)
    campaign_document: dict[str, object] = {
        "baseline_classification": "HISTORICAL_ENGINEERING_ONLY",
        "contract_sha256": contract_sha256(),
        "disposition": "BOUNDED_NEGATIVE",
        "route_outcomes": [
            {
                "disposition": "EXECUTED",
                "gate_evidence": [],
                "receipt": _artifact_ref(campaign, run_path, SCHEMA_VERSION),
                "route": FullSpaceRoute.CFS_P0,
                "terminal_status": "SUCCESS",
                "upstream_gate": None,
            },
            *(
                {
                    "disposition": "NOT_SELECTED_BY_GATE",
                    "gate_evidence": [
                        {"artifact": gate_ref, "gate_id": "CFS-P0-CORRECTNESS"}
                    ],
                    "receipt": None,
                    "route": route,
                    "terminal_status": None,
                    "upstream_gate": "CFS-P0-CORRECTNESS",
                }
                for route in (
                    FullSpaceRoute.CFS_AL1,
                    FullSpaceRoute.CFS_AL2,
                    FullSpaceRoute.CFS_AL1_B,
                )
            ),
        ],
        "schema_version": SCHEMA_VERSION,
    }
    (campaign / "campaign.json").write_bytes(canonical_json_bytes(campaign_document))
    return campaign, run_path, campaign_document


def _rewrite_run_and_campaign(
    campaign: Path,
    run_path: Path,
    run_document: dict[str, object],
    campaign_document: dict[str, object],
) -> None:
    from benchmarks.single_stage_fullspace_receipt import SCHEMA_VERSION

    run_path.write_bytes(canonical_json_bytes(run_document))
    outcomes = campaign_document["route_outcomes"]
    assert isinstance(outcomes, list)
    first_outcome = outcomes[0]
    assert isinstance(first_outcome, dict)
    first_outcome["receipt"] = _artifact_ref(campaign, run_path, SCHEMA_VERSION)
    (campaign / "campaign.json").write_bytes(canonical_json_bytes(campaign_document))


def test_snapshot_publication_is_canonical_exact_and_read_only(tmp_path: Path) -> None:
    campaign, publication = _snapshot(tmp_path)

    loaded = load_snapshot(publication.root)
    document = json.loads(publication.manifest_path.read_bytes())

    assert document["schema_version"] == SOURCE_MANIFEST_SCHEMA_VERSION
    assert loaded.entries == publication.entries
    assert loaded.manifest_sha256 == publication.manifest_sha256
    assert publication.source_identity(campaign).snapshot_manifest.relative_path == (
        "snapshot/source-manifest.json"
    )
    assert stat.S_IMODE(publication.root.stat().st_mode) == 0o555
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o444
        for path in publication.root.rglob("*")
        if path.is_file()
    )


def test_snapshot_refuses_existing_destination_without_overwrite(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "snapshot"
    destination.mkdir()
    marker = _write(destination / "owned.txt", b"owned\n")

    with pytest.raises(FileExistsError):
        publish_immutable_snapshot(
            destination,
            _source_roots(tmp_path / "source"),
            worktree=_worktree(tmp_path),
        )

    assert marker.read_bytes() == b"owned\n"


@pytest.mark.parametrize("secret_name", [".env", ".env.local", ".env.production"])
def test_snapshot_rejects_environment_secret_variants(
    tmp_path: Path, secret_name: str
) -> None:
    roots = list(_source_roots(tmp_path / "source"))
    roots[3] = SourceRoot(
        "configuration",
        _write(tmp_path / "source" / secret_name, b"SECRET=value\n"),
        secret_name,
    )

    with pytest.raises(SnapshotValidationError, match=r"\.env variant"):
        publish_immutable_snapshot(
            tmp_path / "snapshot", roots, worktree=_worktree(tmp_path)
        )


def test_snapshot_allows_env_example_but_rejects_symlink(tmp_path: Path) -> None:
    roots = list(_source_roots(tmp_path / "source"))
    roots[3] = SourceRoot(
        "configuration",
        _write(tmp_path / "source" / ".env.example", b"SETTING=example\n"),
        ".env.example",
    )
    publication = publish_immutable_snapshot(
        tmp_path / "allowed", roots, worktree=_worktree(tmp_path)
    )
    assert (publication.root / ".env.example").is_file()

    target = _write(tmp_path / "target.json", b"{}\n")
    link = tmp_path / "source-link.json"
    link.symlink_to(target)
    roots[3] = SourceRoot("configuration", link, "inputs/problem.json")
    with pytest.raises(SnapshotValidationError, match="symlink"):
        publish_immutable_snapshot(
            tmp_path / "rejected", roots, worktree=_worktree(tmp_path)
        )


def test_snapshot_loader_rejects_changed_unmanifested_and_noncanonical_bytes(
    tmp_path: Path,
) -> None:
    _campaign, publication = _snapshot(tmp_path)
    changed = publication.root / "src/simsopt/__init__.py"
    publication.root.chmod(0o755)
    changed.parent.parent.chmod(0o755)
    changed.parent.chmod(0o755)
    changed.chmod(0o644)
    changed.write_bytes(b"changed\n")
    with pytest.raises(SnapshotValidationError, match="differs"):
        load_snapshot(publication.root)

    _campaign, publication = _snapshot(tmp_path / "extra-case")
    publication.root.chmod(0o755)
    _write(publication.root / "extra.py")
    with pytest.raises(SnapshotValidationError, match="unmanifested"):
        load_snapshot(publication.root)

    _campaign, publication = _snapshot(tmp_path / "json-case")
    publication.root.chmod(0o755)
    manifest = publication.root / SOURCE_MANIFEST_FILENAME
    manifest.chmod(0o644)
    document = json.loads(manifest.read_bytes())
    manifest.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(SnapshotValidationError, match="canonical"):
        load_snapshot(publication.root)


def test_snapshot_loader_rejects_duplicate_json_key(tmp_path: Path) -> None:
    _campaign, publication = _snapshot(tmp_path)
    publication.root.chmod(0o755)
    manifest = publication.root / SOURCE_MANIFEST_FILENAME
    manifest.chmod(0o644)
    manifest.write_text(
        '{"schema_version":"a","schema_version":"b"}\n', encoding="utf-8"
    )

    with pytest.raises(SnapshotValidationError, match="duplicate key"):
        load_snapshot(publication.root)


def test_snapshot_loader_rejects_symlinked_or_writable_directory(
    tmp_path: Path,
) -> None:
    _campaign, publication = _snapshot(tmp_path)
    publication.root.chmod(0o755)
    target = tmp_path / "outside"
    target.mkdir()
    (publication.root / "linked-directory").symlink_to(target, target_is_directory=True)
    with pytest.raises(SnapshotValidationError, match="symlink"):
        load_snapshot(publication.root)

    _campaign, publication = _snapshot(tmp_path / "writable-case")
    publication.root.chmod(0o755)
    (publication.root / "src").chmod(0o755)
    publication.root.chmod(0o555)
    with pytest.raises(SnapshotValidationError, match="writable directory"):
        load_snapshot(publication.root)


def test_runtime_evidence_round_trip_binds_manifest_imports_and_environment(
    tmp_path: Path,
) -> None:
    campaign, publication = _snapshot(tmp_path)
    source_identity = publication.source_identity(campaign)
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=source_identity,
        observation=_observation(publication),
    )
    output = campaign / "run" / "runtime-evidence.json"
    output.parent.mkdir()

    artifact = publish_runtime_evidence(
        output,
        evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    loaded = validate_runtime_evidence(
        output, snapshot_root=publication.root, campaign_root=campaign
    )

    assert artifact.relative_path == "run/runtime-evidence.json"
    assert artifact.schema_version == RUNTIME_EVIDENCE_SCHEMA_VERSION
    assert loaded == evidence


def test_runtime_evidence_publication_is_exclusive(tmp_path: Path) -> None:
    campaign, publication = _snapshot(tmp_path)
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=publication.source_identity(campaign),
        observation=_observation(publication),
    )
    output = _write(campaign / "runtime-evidence.json", b"owned\n")

    with pytest.raises(FileExistsError):
        publish_runtime_evidence(
            output,
            evidence,
            snapshot_root=publication.root,
            campaign_root=campaign,
        )

    assert output.read_bytes() == b"owned\n"


def test_runtime_evidence_survives_campaign_relocation(tmp_path: Path) -> None:
    campaign, publication = _snapshot(tmp_path / "original")
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=publication.source_identity(campaign),
        observation=_observation(publication),
    )
    output = campaign / "runtime-evidence.json"
    publish_runtime_evidence(
        output,
        evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    relocated = tmp_path / "relocated"
    shutil.copytree(campaign, relocated)

    loaded = validate_runtime_evidence(
        relocated / "runtime-evidence.json",
        snapshot_root=relocated / "snapshot",
        campaign_root=relocated,
    )

    assert loaded == evidence


def test_campaign_validator_requires_and_binds_runtime_evidence(
    tmp_path: Path,
) -> None:
    campaign, _run_path, _campaign_document = _campaign_with_executed_run(tmp_path)

    validate_campaign(campaign)


@pytest.mark.parametrize(
    ("identity_name", "field", "replacement", "message"),
    (
        (
            "runtime_identity",
            "driver_version",
            "tampered-driver",
            "runtime identity differs",
        ),
        (
            "source_identity",
            "git_head",
            "b" * 40,
            "source identity differs",
        ),
    ),
)
def test_campaign_validator_rejects_embedded_identity_divergence(
    tmp_path: Path,
    identity_name: str,
    field: str,
    replacement: str,
    message: str,
) -> None:
    campaign, run_path, campaign_document = _campaign_with_executed_run(tmp_path)
    run_document = json.loads(run_path.read_bytes())
    identity = run_document[identity_name]
    assert isinstance(identity, dict)
    identity[field] = replacement
    _rewrite_run_and_campaign(campaign, run_path, run_document, campaign_document)

    with pytest.raises(ValueError, match=message):
        validate_campaign(campaign)


def test_campaign_validator_rejects_missing_runtime_evidence_reference(
    tmp_path: Path,
) -> None:
    campaign, run_path, campaign_document = _campaign_with_executed_run(tmp_path)
    run_document = json.loads(run_path.read_bytes())
    del run_document["runtime_evidence"]
    _rewrite_run_and_campaign(campaign, run_path, run_document, campaign_document)

    with pytest.raises(ValueError, match="keys do not match schema"):
        validate_campaign(campaign)


def test_campaign_validator_rejects_runtime_evidence_digest_tamper(
    tmp_path: Path,
) -> None:
    campaign, run_path, campaign_document = _campaign_with_executed_run(tmp_path)
    run_document = json.loads(run_path.read_bytes())
    runtime_ref = run_document["runtime_evidence"]
    assert isinstance(runtime_ref, dict)
    runtime_ref["sha256"] = _ZERO_SHA256
    _rewrite_run_and_campaign(campaign, run_path, run_document, campaign_document)

    with pytest.raises(ValueError, match="digest mismatch"):
        validate_campaign(campaign)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("contract_sha256", _ZERO_SHA256, "contract digest"),
        ("trajectory_equivalence_required", True, "trajectory equivalence"),
        ("endpoint_certificate", {"certified": False}, "endpoint certification"),
    ),
)
def test_campaign_validator_rejects_contract_or_endpoint_substitution(
    tmp_path: Path, field: str, replacement: object, message: str
) -> None:
    campaign, run_path, campaign_document = _campaign_with_executed_run(tmp_path)
    run_document = json.loads(run_path.read_bytes())
    run_document[field] = replacement
    _rewrite_run_and_campaign(campaign, run_path, run_document, campaign_document)

    with pytest.raises(ValueError, match=message):
        validate_campaign(campaign)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("backend", "physical GPU"),
        ("environment", "environment digest"),
        ("binding", "import binding"),
        ("entrypoint", "entrypoint binding"),
        ("module_path", "runtime path"),
        ("manifest", "manifest reference"),
    ],
)
def test_runtime_binding_mutations_fail_closed(
    tmp_path: Path, mutation: str, message: str
) -> None:
    campaign, publication = _snapshot(tmp_path)
    source = publication.source_identity(campaign)
    observation = _observation(publication)
    if mutation == "backend":
        observation = replace(
            observation,
            runtime_identity=replace(observation.runtime_identity, backend="cpu"),
        )
    elif mutation == "environment":
        observation = replace(
            observation,
            runtime_identity=replace(
                observation.runtime_identity,
                effective_environment_sha256=_ZERO_SHA256,
            ),
        )
    elif mutation == "binding":
        binding = replace(observation.import_bindings[0], sha256=_ZERO_SHA256)
        observation = replace(
            observation,
            import_bindings=(binding, *observation.import_bindings[1:]),
        )
    elif mutation == "module_path":
        observation = replace(
            observation,
            runtime_identity=replace(
                observation.runtime_identity,
                simsopt_module_path=str(publication.root / "wrong.py"),
            ),
        )
    elif mutation == "entrypoint":
        observation = replace(
            observation,
            entrypoint_binding=replace(
                observation.entrypoint_binding, sha256=_ZERO_SHA256
            ),
        )
    else:
        source = replace(
            source,
            snapshot_manifest=replace(source.snapshot_manifest, sha256=_ZERO_SHA256),
        )

    with pytest.raises(SnapshotValidationError, match=message):
        build_runtime_evidence(
            publication.root,
            source_identity=source,
            observation=observation,
        )


def test_runtime_artifact_mutation_and_noncanonical_json_fail_closed(
    tmp_path: Path,
) -> None:
    campaign, publication = _snapshot(tmp_path)
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=publication.source_identity(campaign),
        observation=_observation(publication),
    )
    output = campaign / "runtime-evidence.json"
    publish_runtime_evidence(
        output,
        evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    document = json.loads(output.read_bytes())
    document["runtime_identity"]["backend"] = "cpu"
    output.write_bytes(
        (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
    )

    with pytest.raises(SnapshotValidationError, match="physical GPU"):
        validate_runtime_evidence(
            output, snapshot_root=publication.root, campaign_root=campaign
        )

    output.write_text(json.dumps(document, indent=2), encoding="utf-8")
    with pytest.raises(SnapshotValidationError, match="canonical"):
        validate_runtime_evidence(
            output, snapshot_root=publication.root, campaign_root=campaign
        )


def test_capture_worktree_hashes_exact_untracked_bytes_without_env_secrets(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "test@example.com"),
        check=True,
    )
    subprocess.run(("git", "-C", str(repo), "config", "user.name", "Test"), check=True)
    _write(repo / "tracked.py")
    subprocess.run(("git", "-C", str(repo), "add", "tracked.py"), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)
    untracked = _write(repo / "new.py", b"first\n")
    secret = _write(repo / ".env", b"SECRET=first\n")
    example = _write(repo / ".env.example", b"SETTING=first\n")

    first = capture_worktree_identity(repo)
    secret.write_bytes(b"SECRET=second\n")
    secret_only = capture_worktree_identity(repo)
    example.write_bytes(b"SETTING=second\n")
    example_changed = capture_worktree_identity(repo)
    untracked.write_bytes(b"second\n")
    changed = capture_worktree_identity(repo)

    assert first == secret_only
    assert example_changed.untracked_bytes_manifest_sha256 != (
        first.untracked_bytes_manifest_sha256
    )
    assert changed.untracked_bytes_manifest_sha256 != (
        first.untracked_bytes_manifest_sha256
    )
    assert changed.repo_root == str(repo.resolve())


def test_effective_environment_excludes_secrets_and_has_fixed_order() -> None:
    environment = effective_environment(
        {
            "PATH": "/bin",
            "JAX_ENABLE_X64": "true",
            "AWS_SECRET_ACCESS_KEY": "must-not-appear",
        }
    )

    assert tuple(key for key, _value in environment) == tuple(
        sorted(key for key, _value in environment)
    )
    assert dict(environment)["PATH"] == "/bin"
    assert "AWS_SECRET_ACCESS_KEY" not in dict(environment)


@pytest.mark.parametrize(
    ("visible", "expected_uuid"),
    (("0", "GPU-first"), ("1", "GPU-second"), ("GPU-second", "GPU-second")),
)
def test_physical_gpu_selection_binds_index_or_uuid(
    visible: str, expected_uuid: str
) -> None:
    stdout = "0, GPU-first, NVIDIA RTX 5090, 590.1\n1, GPU-second, NVIDIA A100, 590.1\n"

    uuid, name, driver = select_physical_gpu_identity(stdout, visible)

    assert uuid == expected_uuid
    assert name in ("NVIDIA RTX 5090", "NVIDIA A100")
    assert driver == "590.1"


@pytest.mark.parametrize(
    ("stdout", "visible", "message"),
    (
        (
            "0, GPU-first, NVIDIA RTX 5090, 590.1\n1, GPU-second, NVIDIA A100, 590.1\n",
            "",
            "exactly one",
        ),
        ("0, GPU-first, NVIDIA RTX 5090, 590.1\n", "0,1", "one visible"),
        ("malformed\n", "0", "malformed"),
        ("0, not-stable, GPU, 590.1\n", "0", "stable UUID"),
    ),
)
def test_physical_gpu_selection_rejects_ambiguous_or_malformed_identity(
    stdout: str, visible: str, message: str
) -> None:
    with pytest.raises(SnapshotValidationError, match=message):
        select_physical_gpu_identity(stdout, visible)


def test_runtime_evidence_rejects_symlink_artifact(tmp_path: Path) -> None:
    campaign, publication = _snapshot(tmp_path)
    target = _write(campaign / "target.json", b"{}\n")
    link = campaign / "runtime-evidence.json"
    link.symlink_to(target)

    with pytest.raises(SnapshotValidationError, match="symlink"):
        validate_runtime_evidence(
            link, snapshot_root=publication.root, campaign_root=campaign
        )


def test_snapshot_manifest_filename_is_stable() -> None:
    assert SOURCE_MANIFEST_FILENAME == "source-manifest.json"
    assert os.path.basename(SOURCE_MANIFEST_FILENAME) == SOURCE_MANIFEST_FILENAME


def test_runner_source_selection_is_explicit_complete_and_ignores_bytecode(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    native = _write(tmp_path / "simsoptpp.test.so", b"native\n")

    roots = explicit_source_roots(repo_root, native)
    destinations = tuple(root.relative_path for root in roots)

    assert len(destinations) == len(set(destinations))
    assert all(root.source_path.is_file() for root in roots)
    assert not any(
        "__pycache__" in path or path.endswith(".pyc") for path in destinations
    )
    assert "src/simsopt_jax/objectives/single_stage_fullspace.py" in destinations
    assert "src/simsopt_jax/solve/fullspace.py" in destinations
    assert "benchmarks/run_single_stage_fullspace_gpu.py" in destinations
    assert "docs/single_stage_jax_gpu_coupled_fullspace_phase0_budget.json" in (
        destinations
    )
    assert roots[-1].role == "native_extension"
    assert roots[-1].relative_path == "src/simsoptpp.test.so"


def test_runner_rejects_campaign_inside_source_repository() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(
        ValueError,
        match="campaign output must be outside the source repository",
    ):
        prepare_execution_snapshot(
            repo_root / "artifacts" / "forbidden-fullspace-campaign",
            native_extension_path=repo_root / "unused-native-extension.so",
        )


def test_runner_publishes_real_repository_selection_into_fresh_campaign(
    tmp_path: Path,
) -> None:
    native = _write(tmp_path / "native" / "simsoptpp.test.so", b"native\n")
    campaign = tmp_path / "campaign"

    publication = prepare_execution_snapshot(campaign, native_extension_path=native)
    loaded = load_snapshot(publication.root)

    assert publication.root == campaign / "source-snapshot"
    assert loaded.manifest_sha256 == publication.manifest_sha256
    assert not any(
        "__pycache__" in entry.relative_path or entry.relative_path.endswith(".pyc")
        for entry in loaded.entries
    )
    assert publication.source_identity(campaign).repo_root == str(
        Path(__file__).resolve().parents[2]
    )
    preflight_output = tmp_path / "unused-output"
    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=Path(sys.executable),
        request_argv=(
            "--phase=first-eval",
            "--route=CFS-P0",
            "--device=rtx5090",
            f"--output={preflight_output}",
            "--preflight-only",
        ),
        environment=os.environ,
    )
    completed = subprocess.run(
        invocation.argv,
        cwd=invocation.cwd,
        env=invocation.environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    preflight = load_canonical_json_bytes(completed.stdout.encode())
    assert isinstance(preflight, dict)
    assert preflight["request"]["route"] == "CFS-P0"
    assert not preflight_output.exists()


def test_snapshot_child_invocation_uses_pinned_tree_and_isolated_python(
    tmp_path: Path,
) -> None:
    campaign, publication = _snapshot(tmp_path)

    invocation = build_snapshot_child_invocation(
        publication,
        campaign_root=campaign,
        interpreter=Path(sys.executable),
        request_argv=("--phase=first-eval", "--route=CFS-P0"),
        environment={"PATH": "/usr/bin"},
    )

    assert invocation.argv[:3] == (
        os.path.abspath(sys.executable),
        "-I",
        str(publication.root / "benchmarks/run_single_stage_fullspace_gpu.py"),
    )
    assert invocation.cwd == publication.root
    assert invocation.environment["SIMSOPT_FULLSPACE_SNAPSHOT_MANIFEST_SHA256"] == (
        publication.manifest_sha256
    )
    assert invocation.environment["SIMSOPT_FULLSPACE_CAMPAIGN_ROOT"] == str(campaign)


def test_executed_receipt_embeds_exact_runtime_evidence_identity(
    tmp_path: Path,
) -> None:
    from benchmarks.single_stage_fullspace_receipt import (
        DeviceLane,
        RunPhase,
        RunRequest,
    )

    campaign, publication = _snapshot(tmp_path)
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=publication.source_identity(campaign),
        observation=_observation(publication),
    )
    evidence_dir = campaign / "evidence"
    evidence_dir.mkdir()
    evidence_ref = publish_runtime_evidence(
        evidence_dir / "runtime-evidence.json",
        evidence,
        snapshot_root=publication.root,
        campaign_root=campaign,
    )
    request = RunRequest(
        phase=RunPhase.FIRST_EVAL,
        route=FullSpaceRoute.CFS_P0,
        device=DeviceLane.RTX5090,
        steps=None,
        sample=None,
    )

    payload = executed_run_receipt_payload(
        request,
        runtime_evidence=evidence,
        runtime_evidence_ref=evidence_ref,
        terminal_status="SUCCESS",
        timing={"solve_seconds": 1.0},
        transfer_audit={"hot_d2h": 0},
        endpoint_certificate={"certified": True},
    )

    assert payload["runtime_evidence"] == asdict(evidence_ref)
    assert payload["source_identity"] == asdict(evidence.source_identity)
    assert payload["runtime_identity"] == asdict(evidence.observation.runtime_identity)
    assert payload["trajectory_equivalence_required"] is False


def test_executed_receipt_rejects_unbound_evidence_or_uncertified_endpoint(
    tmp_path: Path,
) -> None:
    from benchmarks.single_stage_fullspace_receipt import (
        DeviceLane,
        RunPhase,
        RunRequest,
    )

    campaign, publication = _snapshot(tmp_path)
    evidence = build_runtime_evidence(
        publication.root,
        source_identity=publication.source_identity(campaign),
        observation=_observation(publication),
    )
    request = RunRequest(
        phase=RunPhase.FIRST_EVAL,
        route=FullSpaceRoute.CFS_P0,
        device=DeviceLane.RTX5090,
        steps=None,
        sample=None,
    )
    wrong_ref = ArtifactRef(
        relative_path="evidence/runtime-evidence.json",
        sha256=_ZERO_SHA256,
        size_bytes=0,
        schema_version=RUNTIME_EVIDENCE_SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="differs from evidence bytes"):
        executed_run_receipt_payload(
            request,
            runtime_evidence=evidence,
            runtime_evidence_ref=wrong_ref,
            terminal_status="SUCCESS",
            timing={},
            transfer_audit={},
            endpoint_certificate={"certified": True},
        )

    actual_payload = canonical_json_bytes(evidence.to_payload())
    matching_ref = replace(
        wrong_ref,
        sha256=hashlib.sha256(actual_payload).hexdigest(),
        size_bytes=len(actual_payload),
    )
    with pytest.raises(ValueError, match="endpoint certification"):
        executed_run_receipt_payload(
            request,
            runtime_evidence=evidence,
            runtime_evidence_ref=matching_ref,
            terminal_status="SUCCESS",
            timing={},
            transfer_audit={},
            endpoint_certificate={"certified": False},
        )


def test_child_runtime_publication_rejects_missing_launch_binding_before_gpu(
    tmp_path: Path,
) -> None:
    campaign, publication = _snapshot(tmp_path)

    with pytest.raises(ValueError, match="launch binding"):
        publish_child_runtime_provenance(
            publication,
            campaign_root=campaign,
            process_argv=("benchmarks/run_single_stage_fullspace_gpu.py",),
            environment={"PATH": "/usr/bin"},
        )
