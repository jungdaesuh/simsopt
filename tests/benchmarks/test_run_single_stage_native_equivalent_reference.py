from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import benchmarks.run_single_stage_native_equivalent_reference as runner_module
import numpy as np
import pytest
from benchmarks.run_single_stage_native_equivalent_reference import (
    ReferenceRunRequest,
    main,
    run_native_equivalent_reference,
)
from benchmarks.single_stage_native_equivalent_reference import (
    USABLE,
    HistoricalAuthorityPaths,
    ReferenceValidationResult,
    RuntimeProvenance,
    SourcePath,
)
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    HistoricalNativeParameters,
    NativeEndpointError,
    NativeReferenceEvidence,
)


@dataclass(frozen=True)
class _FakeRuntime:
    bootstrap_state: np.ndarray
    fixed_first_base_current: float = 7.0

    def reconstruct_native_reference(
        self,
        historical: HistoricalNativeParameters,
    ) -> NativeReferenceEvidence:
        del historical
        raise NativeEndpointError("runner unit test does not reconstruct")


@dataclass
class _FakeAdapter:
    runtime: _FakeRuntime
    observed_input_root: Path | None = None

    def build_runtime(self, input_root: Path) -> _FakeRuntime:
        self.observed_input_root = input_root
        return self.runtime


@dataclass(frozen=True)
class _RuntimeTargetMismatchAdapter:
    def build_runtime(self, input_root: Path) -> _FakeRuntime:
        del input_root
        raise NativeEndpointError("runtime target mismatch")


def _provenance() -> RuntimeProvenance:
    return RuntimeProvenance(
        argv=("python",),
        cwd="/work",
        python_executable="/python",
        python_version="3.11",
        platform="test",
        numpy_version=np.__version__,
        jax_version="jax",
        jaxlib_version="jaxlib",
        simsopt_path="/simsopt.py",
        simsopt_jax_path="/simsopt_jax.py",
        adapter_path="/adapter.py",
        simsopt_sha256="1" * 64,
        simsopt_jax_sha256="2" * 64,
        adapter_sha256="3" * 64,
        native_extension_path="/simsoptpp.so",
        native_extension_sha256="c" * 64,
        python_executable_sha256="d" * 64,
        effective_environment_sha256="e" * 64,
        git_head="a" * 40,
        tracked_diff_sha256="b" * 64,
        repository_dirty=True,
    )


def test_runner_uses_typed_adapter_seam_and_forwards_explicit_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FakeRuntime(np.zeros(716, dtype=np.float64))
    adapter = _FakeAdapter(runtime)
    request = ReferenceRunRequest(
        output_root=tmp_path / "output",
        receipt=tmp_path / "lane_result.json",
        trajectory=tmp_path / "trajectory.jsonl",
        input_bundle=tmp_path / "inputs" / "input_bundle.json",
        repo_root=tmp_path,
    )
    sources = (SourcePath("source.py", tmp_path / "source.py"),)
    expected = ReferenceValidationResult(
        disposition=USABLE,
        usable=True,
        failure_reasons=(),
        artifact_sha256="c" * 64,
    )
    observed: dict[str, object] = {}

    def fake_produce(
        *,
        output_root: Path,
        historical_paths: HistoricalAuthorityPaths,
        runtime: object,
        runtime_provenance: RuntimeProvenance,
        source_paths: tuple[SourcePath, ...],
    ) -> ReferenceValidationResult:
        observed.update(
            output_root=output_root,
            historical_paths=historical_paths,
            runtime=runtime,
            runtime_provenance=runtime_provenance,
            source_paths=source_paths,
        )
        return expected

    monkeypatch.setattr(
        runner_module,
        "produce_native_equivalent_reference",
        fake_produce,
    )
    result = run_native_equivalent_reference(
        request,
        adapter=adapter,
        runtime_provenance=_provenance(),
        execution_sources=sources,
    )
    assert result is expected
    assert adapter.observed_input_root == request.input_bundle.parent
    assert observed["runtime"] is runtime
    assert observed["output_root"] == request.output_root
    assert observed["historical_paths"] == HistoricalAuthorityPaths(
        request.receipt,
        request.trajectory,
        request.input_bundle,
    )
    assert observed["source_paths"] == sources


def test_runtime_target_mismatch_is_explicit_and_produces_no_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ReferenceRunRequest(
        output_root=tmp_path / "output",
        receipt=tmp_path / "lane_result.json",
        trajectory=tmp_path / "trajectory.jsonl",
        input_bundle=tmp_path / "inputs" / "input_bundle.json",
        repo_root=tmp_path,
    )
    producer_called = False

    def unexpected_producer(**kwargs: object) -> ReferenceValidationResult:
        nonlocal producer_called
        del kwargs
        producer_called = True
        raise AssertionError("producer must not run after runtime build failure")

    monkeypatch.setattr(
        runner_module,
        "produce_native_equivalent_reference",
        unexpected_producer,
    )
    with pytest.raises(NativeEndpointError, match="runtime target mismatch"):
        run_native_equivalent_reference(
            request,
            adapter=_RuntimeTargetMismatchAdapter(),
            runtime_provenance=_provenance(),
            execution_sources=(),
        )
    assert not producer_called
    assert not request.output_root.exists()


def test_validate_only_cli_reports_validator_disposition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    validation = ReferenceValidationResult(
        disposition=USABLE,
        usable=True,
        failure_reasons=(),
        artifact_sha256="d" * 64,
    )
    monkeypatch.setattr(
        runner_module,
        "validate_native_equivalent_reference",
        lambda path: validation if path == artifact.resolve() else None,
    )
    assert main(["--output", str(artifact), "--validate-only"]) == 0
    assert capsys.readouterr().out == (
        f"disposition=USABLE usable=true reference_sha256={'d' * 64}\n"
    )
