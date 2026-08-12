from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
from benchmarks import single_stage_compute_graph_canary_runner as _runner
from benchmarks import single_stage_compute_graph_canary_spec_builder as _builder
from benchmarks.single_stage_compute_graph_c0_runner import _write_exclusive_json
from benchmarks.single_stage_compute_graph_canary_runner import CanarySpec
from benchmarks.single_stage_compute_graph_isolated_launch import SnapshotModuleLaunch
from benchmarks.single_stage_compute_graph_snapshot import (
    RoleRoot,
    canonical_json_bytes,
    publish_immutable_snapshot,
)


def _write(path: Path, value: bytes = b"source\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return path


def _snapshot(tmp_path: Path, *, include_evaluator: bool = True) -> Path:
    source = tmp_path / "snapshot-input"
    roots = [
        RoleRoot(
            "execution_source",
            _write(source / "native_boozerqa.py"),
            "examples/jax/parity/cases/native_boozerqa.py",
        ),
        RoleRoot(
            "execution_source",
            _write(source / "linear_solve.py"),
            "src/simsopt_jax/geo/optimizers/linear_solve.py",
        ),
        RoleRoot(
            "execution_source",
            _write(source / "optimizer.py"),
            "src/simsopt_jax/geo/optimizers/optimizer.py",
        ),
        RoleRoot(
            "execution_source",
            _write(source / "surface_objectives_traceable.py"),
            "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py",
        ),
        RoleRoot(
            "configuration",
            _write(source / "configuration.json", b"{}\n"),
            "phase0-specimen/input_bundle/input_bundle.json",
        ),
        RoleRoot(
            "test",
            _write(source / "test_canary.py"),
            "tests/benchmarks/test_canary.py",
        ),
        RoleRoot(
            "native_extension",
            _write(source / "simsoptpp.so"),
            "src/simsoptpp.so",
        ),
    ]
    benchmark_source = (
        _write(source / "single_stage_compute_graph_canary_evaluator.py")
        if include_evaluator
        else _write(source / "other_benchmark.py")
    )
    roots.append(
        RoleRoot(
            "benchmark",
            benchmark_source,
            (
                "benchmarks/single_stage_compute_graph_canary_evaluator.py"
                if include_evaluator
                else "benchmarks/other_benchmark.py"
            ),
        )
    )
    snapshot_root = tmp_path / "snapshot"
    publish_immutable_snapshot(snapshot_root, tuple(roots))
    return snapshot_root


def _canonical(path: Path, document: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_exclusive_json(path, document)
    return path


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[_builder.CanarySpecBuildInputs, dict[str, object], CanarySpec]:
    snapshot_root = _snapshot(tmp_path)
    c0_output = tmp_path / "c0-output"
    qualification = {"outcome": "qualified", "checks": ["bound"]}
    runtime_provenance = {
        "runtime": {"jax": "test"},
        "environment": {"JAX_ENABLE_X64": "1"},
        "policies": {"dense_batch_width": 8},
        "allocation": {"gpu_uuid": "GPU-test"},
    }
    samples = [
        {
            "sample_index": index,
            "wall_ns": 100 + index,
            "peak_process_tree_rss_bytes": 1000 + index,
            "sampled_process_gpu_memory_peak_bytes": 2000 + index,
        }
        for index in range(10)
    ]
    warm_measurement = {"samples": samples, "p50_ns": 104.5, "p95_ns": 109.0}
    first_evaluation = {"objective": 1.0, "gradient": [0.0] * 461}
    gate = {"first_evaluation_gate": first_evaluation}
    warm = {"warm_measurement": warm_measurement}
    c0_spec_path = _canonical(
        tmp_path / "c0-spec.json",
        {"output_root": str(c0_output)},
    )
    _canonical(c0_output / "gate-checkpoint.json", gate)
    _canonical(c0_output / "warm-checkpoint.json", warm)
    qualification_path = _canonical(tmp_path / "qualification.json", qualification)
    provenance_path = _canonical(tmp_path / "provenance.json", runtime_provenance)
    graph_path = tmp_path / "graph.json"
    _builder.write_variant_solver_graph(graph_path, snapshot_root, "C1")
    receipt = {
        "specimen_sha256": "3" * 64,
        "lanes": [
            {
                "lane_id": "rtx5090",
                "qualification": qualification,
                "measurement": {
                    "variant": "C0",
                    "specimen_sha256": "3" * 64,
                    "provenance": runtime_provenance,
                    "first_evaluation_gate": first_evaluation,
                    "warm_measurement": warm_measurement,
                },
            }
        ],
    }
    receipt_path = c0_output / "phase0-receipt.json"
    receipt_path.write_bytes(canonical_json_bytes(receipt))
    runtime_contract = json.dumps(
        {
            "runtime": runtime_provenance["runtime"],
            "static_environment": runtime_provenance["environment"],
            "route_environment": {},
            "policies": runtime_provenance["policies"],
            "expected_runtime_identity_sha256": "7" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    candidate = _write(tmp_path / "candidate.npy", b"candidate")
    native = _canonical(tmp_path / "native.json", {"objective": 1.0})
    spec = CanarySpec(
        variant="C1",
        solver_graph_sha256="1" * 64,
        source_state_sha256="2" * 64,
        specimen_sha256="3" * 64,
        candidate_file_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
        parameter_sha256="4" * 64,
        device_identity_sha256="5" * 64,
        gpu_uuid="GPU-test",
        c0_gate_checkpoint_sha256="6" * 64,
        c0_warm_checkpoint_sha256="6" * 64,
        native_reference_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
        runtime_identity_sha256="7" * 64,
        input_root=tmp_path,
        candidate_path=candidate,
        native_reference_path=native,
        snapshot_root=snapshot_root,
        interpreter_path=Path(sys.executable),
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "canary-output",
        c0_p50_ns=104.5,
        c0_p95_ns=109.0,
        c0_peak_rss_bytes=1009,
        c0_peak_gpu_memory_bytes=2009,
        runtime_contract_json=runtime_contract,
    )
    monkeypatch.setattr(_builder, "validate_spec", lambda document: spec)
    monkeypatch.setattr(
        _builder,
        "load_phase0_receipt",
        lambda path: (receipt, object()),
    )
    monkeypatch.setattr(
        _runner,
        "child_launches",
        lambda checked, environment: (
            SnapshotModuleLaunch(
                argv=(
                    str(checked.interpreter_path),
                    _runner.EVALUATOR_MODULE,
                ),
                cwd=checked.snapshot_root,
                environment=environment,
            ),
        ),
    )
    inputs = _builder.CanarySpecBuildInputs(
        variant="C1",
        c0_spec_path=c0_spec_path,
        c0_receipt_path=receipt_path,
        snapshot_publication_path=tmp_path / "publication.json",
        import_attestation_path=tmp_path / "attestation.json",
        qualification_path=qualification_path,
        device_probe_path=tmp_path / "probe.json",
        runtime_provenance_path=provenance_path,
        variant_solver_graph_path=graph_path,
        cache_directory=tmp_path / "cache",
        output_root=tmp_path / "canary-output",
        destination=tmp_path / "canary-spec.json",
    )
    return inputs, receipt, spec


@pytest.mark.parametrize("variant", ("C1", "C2"))
def test_variant_solver_graph_is_derived_from_exact_manifested_files(
    tmp_path: Path,
    variant: _builder.CanaryVariant,
) -> None:
    snapshot_root = _snapshot(tmp_path)

    document = _builder.variant_solver_graph_document(snapshot_root, variant)

    assert document["variant"] == variant
    assert document["selection"] == {
        "evaluator_module": _runner.EVALUATOR_MODULE,
        "runtime_owner": (
            "examples.jax.parity.cases.native_boozerqa._prepare_jax_variant_runtime"
        ),
        "runtime_selector_argument": {"exact_newton_variant": variant},
        "production_value_and_gradient_route": (
            "fresh_incumbent_controller.value_and_grad"
        ),
    }
    bindings = cast(list[dict[str, object]], document["manifested_implementation"])
    assert [binding["relative_path"] for binding in bindings] == list(
        _builder._IMPLEMENTATION_PATHS
    )
    assert all(len(cast(str, binding["sha256"])) == 64 for binding in bindings)


def test_variant_solver_graph_rejects_unmanifested_evaluator(tmp_path: Path) -> None:
    snapshot_root = _snapshot(tmp_path, include_evaluator=False)

    with pytest.raises(
        _builder.CanarySpecBuilderError,
        match="lacks required variant implementation files",
    ):
        _builder.variant_solver_graph_document(snapshot_root, "C1")


def test_graph_cli_writes_canonical_exclusive_identity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    snapshot_root = _snapshot(tmp_path)
    output = tmp_path / "graph.json"

    result = _builder.main(
        (
            "graph",
            "--variant",
            "C2",
            "--snapshot-root",
            str(snapshot_root),
            "--output",
            str(output),
        )
    )

    assert result == 0
    document = json.loads(output.read_bytes())
    assert output.read_bytes() == canonical_json_bytes(document)
    assert document["variant"] == "C2"
    assert (
        capsys.readouterr().out.strip()
        == hashlib.sha256(output.read_bytes()).hexdigest()
    )


def test_builder_writes_only_runner_schema_after_all_bindings_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _receipt, _spec = _fixture(tmp_path, monkeypatch)

    document, digest = _builder.build_canary_spec(inputs)

    assert set(document) == {
        "schema_id",
        "variant",
        "c0_spec_path",
        "snapshot_publication_path",
        "import_attestation_path",
        "qualification_path",
        "device_probe_path",
        "runtime_provenance_path",
        "variant_solver_graph_path",
        "cache_directory",
        "output_root",
    }
    assert inputs.destination.read_bytes() == canonical_json_bytes(document)
    assert digest == hashlib.sha256(inputs.destination.read_bytes()).hexdigest()
    assert "c0_receipt_path" not in document
    assert "solver_graph_sha256" not in document


def test_builder_rejects_opaque_solver_graph_sha_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _receipt, _spec = _fixture(tmp_path, monkeypatch)
    inputs.variant_solver_graph_path.chmod(0o644)
    inputs.variant_solver_graph_path.write_bytes(
        canonical_json_bytes(
            {
                "variant": "C1",
                "solver_graph_sha256": "a" * 64,
            }
        )
    )

    with pytest.raises(
        _builder.CanarySpecBuilderError,
        match="differs from manifested implementation bytes",
    ):
        _builder.build_canary_spec(inputs)


def test_builder_rejects_receipt_metric_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _receipt, spec = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _builder,
        "validate_spec",
        lambda document: replace(spec, c0_p50_ns=1.0),
    )

    with pytest.raises(
        _builder.CanarySpecBuilderError,
        match="comparison metrics differ",
    ):
        _builder.build_canary_spec(inputs)


def test_builder_rejects_runtime_contract_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs, _receipt, spec = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        _builder,
        "validate_spec",
        lambda document: replace(
            spec,
            runtime_contract_json='{"runtime":"opaque"}',
        ),
    )

    with pytest.raises(
        _builder.CanarySpecBuilderError,
        match="runtime contract differs",
    ):
        _builder.build_canary_spec(inputs)
