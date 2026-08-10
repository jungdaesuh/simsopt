from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from benchmarks.run_single_stage_fullspace_gpu import (
    PhaseGateError,
    main,
    parse_request,
)
from benchmarks.single_stage_fullspace_receipt import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_V2,
    ArtifactRef,
    CompleteSample,
    DeviceLane,
    RunPhase,
    RunRequest,
    canonical_json_bytes,
    contract_payload,
    contract_payload_v1,
    contract_payload_v2,
    contract_sha256,
    contract_sha256_v1,
    contract_sha256_v2,
    load_canonical_json_bytes,
    run_request_from_payload,
    run_request_payload,
    run_request_payload_v2,
    run_request_v2_from_payload,
)
from simsopt_jax.solve.fullspace import FullSpaceRoute


def _request(
    *,
    phase: RunPhase,
    route: FullSpaceRoute,
    steps: int | None = None,
    sample: CompleteSample | None = None,
) -> RunRequest:
    return RunRequest(
        phase=phase,
        route=route,
        device=DeviceLane.RTX5090,
        steps=steps,
        sample=sample,
    )


@pytest.mark.parametrize(
    "run_request",
    (
        _request(phase=RunPhase.FIRST_EVAL, route=FullSpaceRoute.CFS_P0),
        _request(phase=RunPhase.CANARY, route=FullSpaceRoute.CFS_P0, steps=10),
        _request(phase=RunPhase.CANARY, route=FullSpaceRoute.CFS_AL1, steps=100),
        _request(
            phase=RunPhase.COMPLETE,
            route=FullSpaceRoute.CFS_AL1,
            sample=CompleteSample.COLD,
        ),
        _request(
            phase=RunPhase.COMPLETE,
            route=FullSpaceRoute.CFS_AL1_B,
            sample=CompleteSample.WARM_3,
        ),
    ),
)
def test_run_request_accepts_only_defined_phase_shapes(
    run_request: RunRequest,
) -> None:
    run_request.validate()


@pytest.mark.parametrize(
    "run_request",
    (
        _request(phase=RunPhase.FIRST_EVAL, route=FullSpaceRoute.CFS_AL1),
        _request(phase=RunPhase.FIRST_EVAL, route=FullSpaceRoute.CFS_P0, steps=10),
        _request(
            phase=RunPhase.FIRST_EVAL,
            route=FullSpaceRoute.CFS_P0,
            sample=CompleteSample.COLD,
        ),
        _request(phase=RunPhase.CANARY, route=FullSpaceRoute.CFS_P0),
        _request(phase=RunPhase.CANARY, route=FullSpaceRoute.CFS_P0, steps=11),
        _request(
            phase=RunPhase.CANARY,
            route=FullSpaceRoute.CFS_P0,
            steps=10,
            sample=CompleteSample.COLD,
        ),
        _request(
            phase=RunPhase.COMPLETE,
            route=FullSpaceRoute.CFS_P0,
            sample=CompleteSample.COLD,
        ),
        _request(phase=RunPhase.COMPLETE, route=FullSpaceRoute.CFS_AL1),
        _request(
            phase=RunPhase.COMPLETE,
            route=FullSpaceRoute.CFS_AL1,
            steps=100,
            sample=CompleteSample.COLD,
        ),
    ),
)
def test_run_request_rejects_cross_field_mismatches(
    run_request: RunRequest,
) -> None:
    with pytest.raises(ValueError):
        run_request.validate()


@pytest.mark.parametrize(
    "payload",
    (
        b'{"b":1,"a":2}\n',
        b'{"a":1}\n\n',
        b'{"a": 1}\n',
        b'{"a":1}',
        b'{"a":1,"a":1}\n',
    ),
)
def test_canonical_json_loader_rejects_noncanonical_json(payload: bytes) -> None:
    with pytest.raises(ValueError):
        load_canonical_json_bytes(payload)


def test_canonical_json_round_trip_is_stable() -> None:
    payload = contract_payload()
    encoded = canonical_json_bytes(payload)

    assert encoded.endswith(b"\n")
    decoded = load_canonical_json_bytes(encoded)
    assert isinstance(decoded, dict)
    assert decoded["schema_version"] == SCHEMA_VERSION
    assert canonical_json_bytes(decoded) == encoded


def test_legacy_v1_contract_bytes_and_aliases_remain_frozen() -> None:
    contract = contract_payload_v1()
    route_encoded = canonical_json_bytes(contract["routes"])
    encoded = canonical_json_bytes(contract)

    assert len(route_encoded) == 5599
    assert hashlib.sha256(route_encoded).hexdigest() == (
        "1cac4bd571dac722ae188693b26ab6cc86d2c5ca64f274f2a5b962a625a7b01b"
    )
    assert len(encoded) == 10722
    assert hashlib.sha256(encoded).hexdigest() == (
        "e680e6a2f6ff0afb9bdcc18e15bf90953b77e7c92baaed1745a4d1008700e4f9"
    )
    assert contract_payload() == contract_payload_v1()
    assert contract_sha256() == contract_sha256_v1()


def test_campaign_v2_contract_is_additive_and_schema_distinct() -> None:
    payload = contract_payload_v2()

    assert payload["schema_version"] == SCHEMA_VERSION_V2
    routes = payload["routes"]
    assert isinstance(routes, dict)
    assert routes["legacy_v1"] == contract_payload_v1()["routes"]
    sqp_routes = routes["sqp_routes"]
    assert isinstance(sqp_routes, list)
    assert isinstance(sqp_routes[0], dict)
    assert sqp_routes[0]["route"] == "CFS-SQP1"
    assert contract_sha256_v2() != contract_sha256_v1()


def test_v1_request_rejects_sqp_without_changing_legacy_requests() -> None:
    legacy = _request(
        phase=RunPhase.COMPLETE,
        route=FullSpaceRoute.CFS_AL2,
        sample=CompleteSample.COLD,
    )
    assert run_request_from_payload(run_request_payload(legacy)["request"]) == legacy
    sqp = _request(
        phase=RunPhase.COMPLETE,
        route=FullSpaceRoute.CFS_SQP1,
        sample=CompleteSample.COLD,
    )

    with pytest.raises(ValueError, match="campaign-v1"):
        run_request_payload(sqp)
    with pytest.raises(ValueError, match="campaign-v1"):
        run_request_from_payload(
            {
                "device": "rtx5090",
                "phase": "complete",
                "route": "CFS-SQP1",
                "sample": "cold",
                "steps": None,
            }
        )


@pytest.mark.parametrize(
    ("phase", "steps", "sample"),
    (
        (RunPhase.CANARY, 1, None),
        (RunPhase.CANARY, 10, None),
        (RunPhase.COMPLETE, None, CompleteSample.COLD),
        (RunPhase.COMPLETE, None, CompleteSample.WARM_3),
    ),
)
def test_v2_request_accepts_only_frozen_sqp_shapes(
    phase: RunPhase, steps: int | None, sample: CompleteSample | None
) -> None:
    request = _request(
        phase=phase,
        route=FullSpaceRoute.CFS_SQP1,
        steps=steps,
        sample=sample,
    )
    envelope = run_request_payload_v2(request)

    assert envelope["schema_version"] == SCHEMA_VERSION_V2
    assert run_request_v2_from_payload(envelope["request"]) == request


@pytest.mark.parametrize(
    ("phase", "steps", "sample"),
    (
        (RunPhase.CANARY, 100, None),
        (RunPhase.CANARY, 10, CompleteSample.COLD),
        (RunPhase.COMPLETE, 10, CompleteSample.COLD),
        (RunPhase.COMPLETE, None, None),
    ),
)
def test_v2_request_rejects_noncontractual_sqp_shapes(
    phase: RunPhase, steps: int | None, sample: CompleteSample | None
) -> None:
    request = _request(
        phase=phase,
        route=FullSpaceRoute.CFS_SQP1,
        steps=steps,
        sample=sample,
    )

    with pytest.raises(ValueError):
        run_request_payload_v2(request)


def test_v2_request_accepts_sqp_derivative_gate_shape() -> None:
    request = _request(
        phase=RunPhase.FIRST_EVAL,
        route=FullSpaceRoute.CFS_SQP1,
        steps=None,
        sample=None,
    )

    assert (
        run_request_v2_from_payload(run_request_payload_v2(request)["request"])
        == request
    )


def _write_artifact(root: Path, relative_path: str) -> ArtifactRef:
    payload = canonical_json_bytes(
        {"schema_version": SCHEMA_VERSION, "value": "evidence"}
    )
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return ArtifactRef(
        relative_path=relative_path,
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        schema_version=SCHEMA_VERSION,
    )


def test_artifact_reference_survives_root_relocation(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    reference = _write_artifact(first_root, "receipts/evidence.json")
    second_root = tmp_path / "relocated"
    (second_root / "receipts").mkdir(parents=True)
    (second_root / "receipts/evidence.json").write_bytes(
        (first_root / "receipts/evidence.json").read_bytes()
    )

    assert reference.resolve_and_validate(second_root) == (
        second_root / "receipts/evidence.json"
    )


@pytest.mark.parametrize(
    "relative_path",
    ("../escape.json", "/absolute.json", "a/../../escape.json", "a//b.json"),
)
def test_artifact_reference_rejects_noncanonical_or_escaping_paths(
    tmp_path: Path, relative_path: str
) -> None:
    reference = ArtifactRef(relative_path, "0" * 64, 0, SCHEMA_VERSION)

    with pytest.raises(ValueError):
        reference.resolve_and_validate(tmp_path)


def test_artifact_reference_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_bytes(canonical_json_bytes({"schema_version": SCHEMA_VERSION}))
    link = tmp_path / "link.json"
    link.symlink_to(target)
    reference = ArtifactRef(
        "link.json",
        hashlib.sha256(target.read_bytes()).hexdigest(),
        target.stat().st_size,
        SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="symlink"):
        reference.resolve_and_validate(tmp_path)


def test_artifact_reference_rejects_symlinked_parent_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    reference = _write_artifact(outside, "evidence.json")
    root = tmp_path / "root"
    root.mkdir()
    (root / "receipts").symlink_to(outside, target_is_directory=True)
    escaped_reference = ArtifactRef(
        "receipts/evidence.json",
        reference.sha256,
        reference.size_bytes,
        reference.schema_version,
    )

    with pytest.raises(ValueError, match="symlink"):
        escaped_reference.resolve_and_validate(root)


def test_artifact_reference_rejects_content_tamper(tmp_path: Path) -> None:
    reference = _write_artifact(tmp_path, "evidence.json")
    (tmp_path / "evidence.json").write_bytes(
        canonical_json_bytes({"schema_version": SCHEMA_VERSION, "value": "tampered"})
    )

    with pytest.raises(ValueError, match="(size|digest) mismatch"):
        reference.resolve_and_validate(tmp_path)


def test_artifact_reference_rejects_schema_mismatch(tmp_path: Path) -> None:
    payload = canonical_json_bytes({"schema_version": "wrong-schema"})
    path = tmp_path / "evidence.json"
    path.write_bytes(payload)
    reference = ArtifactRef(
        "evidence.json",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        reference.resolve_and_validate(tmp_path)


def test_artifact_reference_rejects_noncanonical_content(tmp_path: Path) -> None:
    payload = json.dumps({"schema_version": SCHEMA_VERSION}, indent=2).encode()
    path = tmp_path / "evidence.json"
    path.write_bytes(payload)
    reference = ArtifactRef(
        "evidence.json",
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        SCHEMA_VERSION,
    )

    with pytest.raises(ValueError, match="not canonical"):
        reference.resolve_and_validate(tmp_path)


def _first_eval_argv(output: Path) -> list[str]:
    return [
        "--phase=first-eval",
        "--route=CFS-P0",
        "--device=rtx5090",
        f"--output={output}",
    ]


def test_cli_rejects_abbreviated_option(tmp_path: Path) -> None:
    argv = _first_eval_argv(tmp_path / "out.json")
    argv[0] = "--pha=first-eval"

    with pytest.raises(SystemExit):
        parse_request(argv)


def test_cli_rejects_duplicate_option_even_across_argument_forms(
    tmp_path: Path,
) -> None:
    argv = _first_eval_argv(tmp_path / "out.json")
    argv.extend(("--phase", "first-eval"))

    with pytest.raises(ValueError, match="duplicate option: --phase"):
        parse_request(argv)


def test_cli_rejects_existing_output_without_modifying_it(tmp_path: Path) -> None:
    output = tmp_path / "out.json"
    original = b"user-owned\n"
    output.write_bytes(original)

    with pytest.raises(FileExistsError):
        parse_request(_first_eval_argv(output))
    assert output.read_bytes() == original


def test_cli_enforces_cross_field_contract(tmp_path: Path) -> None:
    argv = _first_eval_argv(tmp_path / "out.json")
    argv[1] = "--route=CFS-AL1"

    with pytest.raises(ValueError, match="first-eval requires CFS-P0"):
        parse_request(argv)


def test_preflight_emits_canonical_request_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "out.json"

    assert main([*_first_eval_argv(output), "--preflight-only"]) == 0
    captured = capsys.readouterr()
    decoded = load_canonical_json_bytes(captured.out.encode())
    assert isinstance(decoded, dict)
    assert decoded["schema_version"] == SCHEMA_VERSION
    request = decoded["request"]
    assert isinstance(request, dict)
    assert request["phase"] == "first-eval"
    assert request["route"] == "CFS-P0"
    assert not output.exists()


def test_execution_remains_fail_closed_beyond_cfs_p0_canary(tmp_path: Path) -> None:
    with pytest.raises(PhaseGateError, match="only the 10/100-step CFS-P0 canary"):
        main(
            [
                "--phase=canary",
                "--route=CFS-AL1",
                "--device=rtx5090",
                "--steps=100",
                f"--output={tmp_path / 'out'}",
            ]
        )
