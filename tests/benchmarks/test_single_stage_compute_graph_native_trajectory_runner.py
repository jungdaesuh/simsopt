from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import benchmarks.single_stage_compute_graph_native_trajectory_runner as runner
import pytest
from benchmarks.single_stage_compute_graph_c0_runner import CommandResult
from benchmarks.single_stage_compute_graph_isolated_launch import SnapshotModuleLaunch
from benchmarks.single_stage_compute_graph_native_reference import (
    NativeReferenceBinding,
)


def _digest(character: str) -> str:
    return character * 64


def _launch(tmp_path: Path) -> runner.NativeTrajectoryLaunch:
    snapshot = tmp_path / "snapshot"
    input_root = snapshot / "phase0-specimen" / "input"
    input_root.mkdir(parents=True)
    candidate = snapshot / "phase0-specimen" / "candidate.npy"
    candidate.write_bytes(b"candidate")
    graph = snapshot / "graph.json"
    graph.write_text("{}\n", encoding="utf-8")
    native = snapshot / "src" / "simsoptpp.so"
    native.parent.mkdir(parents=True)
    native.write_bytes(b"native")
    publication = tmp_path / "publication.json"
    publication.write_text("{}\n", encoding="utf-8")
    attestation = tmp_path / "imports.json"
    attestation.write_text("{}\n", encoding="utf-8")
    binding = NativeReferenceBinding(
        input_bundle_sha256=_digest("1"),
        input_fingerprint=_digest("2"),
        configuration_fingerprint=_digest("3"),
        specimen_sha256=_digest("4"),
        source_sha256=_digest("5"),
        runtime_identity_sha256=_digest("6"),
        interpreter_path=str(Path(sys.executable).absolute()),
        native_simsoptpp_path=str(native.resolve()),
        native_simsoptpp_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        runtime_contract={
            "runtime": {},
            "static_environment": {"PATH": "/usr/bin"},
            "route_environment": {},
            "policies": {},
            "expected_runtime_identity_sha256": _digest("6"),
        },
    )
    return runner.NativeTrajectoryLaunch(
        snapshot_root=snapshot,
        snapshot_publication_path=publication,
        import_attestation_path=attestation,
        snapshot_manifest_sha256=_digest("8"),
        snapshot_publication_sha256=hashlib.sha256(
            publication.read_bytes()
        ).hexdigest(),
        import_attestation_sha256=hashlib.sha256(attestation.read_bytes()).hexdigest(),
        input_root=input_root,
        candidate_path=candidate,
        solver_graph_path=graph,
        output_path=tmp_path / "artifacts" / "raw.json",
        receipt_path=tmp_path / "artifacts" / "receipt.json",
        parameter_sha256=_digest("7"),
        binding=binding,
    )


def test_native_runner_delegates_to_snapshot_launch_ssot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _launch(tmp_path)
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runner,
        "load_snapshot_manifest",
        lambda _root: ((), launch.snapshot_manifest_sha256),
    )

    def build(interpreter, snapshot_root, module, module_args, base_environment):
        captured.update(
            interpreter=interpreter,
            snapshot_root=snapshot_root,
            module=module,
            module_args=module_args,
            base_environment=base_environment,
        )
        return SnapshotModuleLaunch(
            argv=(str(interpreter), "-P", "-s", module),
            cwd=snapshot_root.resolve(),
            environment={"PYTHONPATH": str(snapshot_root.resolve())},
        )

    monkeypatch.setattr(runner, "build_snapshot_module_launch", build)
    child = runner._snapshot_launch(launch)

    assert captured["snapshot_root"] == launch.snapshot_root
    assert captured["module"] == runner.PRODUCER_MODULE
    assert captured["base_environment"]["PATH"] == "/usr/bin"
    assert "PYTHONPATH" not in captured["base_environment"]
    assert "--solver-graph" in captured["module_args"]
    assert child.argv[1:3] == ("-P", "-s")
    assert child.cwd == launch.snapshot_root.resolve()


def test_native_runner_rejects_ambient_candidate_path(tmp_path: Path) -> None:
    launch = _launch(tmp_path)
    outside = tmp_path / "ambient-candidate.npy"
    outside.write_bytes(b"ambient")
    launch = runner.NativeTrajectoryLaunch(
        snapshot_root=launch.snapshot_root,
        snapshot_publication_path=launch.snapshot_publication_path,
        import_attestation_path=launch.import_attestation_path,
        snapshot_manifest_sha256=launch.snapshot_manifest_sha256,
        snapshot_publication_sha256=launch.snapshot_publication_sha256,
        import_attestation_sha256=launch.import_attestation_sha256,
        input_root=launch.input_root,
        candidate_path=outside,
        solver_graph_path=launch.solver_graph_path,
        output_path=launch.output_path,
        receipt_path=launch.receipt_path,
        parameter_sha256=launch.parameter_sha256,
        binding=launch.binding,
    )

    with pytest.raises(runner.NativeTrajectoryRunnerError, match="snapshot"):
        runner._snapshot_launch(launch)


def test_native_runner_uses_bounded_injectable_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch = _launch(tmp_path)
    child = SnapshotModuleLaunch(
        argv=(launch.binding.interpreter_path, "-P", "-s", "child"),
        cwd=launch.snapshot_root.resolve(),
        environment={"PYTHONPATH": str(launch.snapshot_root.resolve())},
    )
    monkeypatch.setattr(runner, "_snapshot_launch", lambda _launch: child)

    def timed_out(argv, environment, cwd, timeout_seconds):
        assert argv == child.argv
        assert environment == child.environment
        assert cwd == child.cwd
        assert timeout_seconds == 900.0
        return CommandResult(
            returncode=124,
            stdout="",
            stderr="",
            elapsed_ns=900_000_000_000,
            timed_out=True,
        )

    with pytest.raises(runner.NativeTrajectoryRunnerError, match="silently"):
        runner.launch_native_trajectory(
            launch,
            artifact_root=tmp_path,
            executor=timed_out,
        )


def _completion() -> dict[str, object]:
    return {
        "returncode": 0,
        "timed_out": False,
        "elapsed_ns": 1,
        "stdout": "",
        "stderr": "",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("returncode", 1),
        ("returncode", False),
        ("timed_out", True),
        ("timed_out", 0),
        ("elapsed_ns", 0),
        ("elapsed_ns", True),
        ("stdout", "unexpected"),
        ("stderr", "unexpected"),
    ],
)
def test_native_completion_tamper_is_rejected(field: str, value: object) -> None:
    completion = _completion()
    completion[field] = value
    with pytest.raises(runner.NativeTrajectoryRunnerError, match="completion"):
        runner._validated_completion(completion)


def test_native_completion_rejects_extra_fields() -> None:
    completion = _completion()
    completion["trusted"] = True
    with pytest.raises(runner.NativeTrajectoryRunnerError, match="fields"):
        runner._validated_completion(completion)
