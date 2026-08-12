from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import benchmarks.single_stage_native_equivalent_reference as reference_module
import numpy as np
import pytest
from benchmarks.single_stage_native_equivalent_reference import (
    REFERENCE_NOT_PRODUCED,
    USABLE,
    HistoricalAuthorityPaths,
    ReferenceArtifactError,
    ReferenceValidationResult,
    RuntimeProvenance,
    SourcePath,
    load_canonical_json_bytes,
    produce_native_equivalent_reference,
    validate_native_equivalent_reference,
)
from examples.jax.parity.artifacts import canonical_json_bytes as parity_json_bytes
from examples.jax.parity.artifacts import write_array, write_bytes_exclusive
from examples.jax.parity.input_bundle import create_input_bundle
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    HistoricalNativeParameters,
    NativeContinuationStep,
    NativeDyadicPathEvidence,
    NativeEndpointError,
    NativeExplicitStateEvaluation,
    NativeObjectiveTerms,
    NativeReferenceEvidence,
    NativeStateObservables,
)

_COIL_SIZE = 461
_SURFACE_SIZE = 253
_ROOT_SIZE = 255
_STATE_SIZE = 716
_EQUALITY_SIZE = 255
_OBJECTIVE = 4.4822246533126125e-08


def _raw_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _HistoricalFixture:
    paths: HistoricalAuthorityPaths
    bootstrap_coils: np.ndarray
    final_parameters: np.ndarray


@dataclass
class _FakeRuntime:
    bootstrap_state: np.ndarray
    fixed_first_base_current: float
    final_parameters: np.ndarray
    fail_reconstruction: bool = False
    typed_unusable: bool = False

    def reconstruct_native_reference(
        self,
        historical: HistoricalNativeParameters,
    ) -> NativeReferenceEvidence:
        if self.fail_reconstruction:
            raise NativeEndpointError("synthetic continuation failure")
        assert np.array_equal(historical.parameters, self.final_parameters)
        coarse = self._path(256, historical.parameters)
        refined = self._path(512, historical.parameters)
        terminal_root = coarse.roots[-1]
        state = np.concatenate((historical.parameters, terminal_root))
        terms = NativeObjectiveTerms(
            non_qs=_OBJECTIVE,
            residual=0.0,
            iota=0.0,
            major_radius=0.0,
            length=0.0,
        )
        observables = NativeStateObservables(
            iota=-0.4,
            G=7.0,
            volume=2.0,
            major_radius=1.5,
            total_length=6.0,
            non_qs_ratio=_OBJECTIVE,
            boozer_residual_value=0.0,
            boozer_residual_rms=0.0,
            fixed_first_base_current=self.fixed_first_base_current,
        )
        endpoint = NativeExplicitStateEvaluation(
            state=state,
            state_little_endian_sha256=_raw_sha256(state),
            objective_terms=terms,
            objective=_OBJECTIVE,
            observables=observables,
            masked_boozer_equalities=np.zeros(254, dtype=np.float64),
            volume_equality=0.0,
            raw_equalities=np.zeros(255, dtype=np.float64),
            all_finite=True,
        )
        return NativeReferenceEvidence(
            schema_version="single-stage-native-endpoint-v1",
            ssot_sha256=reference_module.SSOT_SHA256,
            historical_input=historical,
            state=state,
            endpoint=endpoint,
            coarse_path=coarse,
            refined_path=refined,
            common_knot_root_infinity_difference=0.0,
            sealed_observables_match=True,
            fixed_first_base_current=self.fixed_first_base_current,
            usable=not self.typed_unusable,
        )

    def _path(
        self,
        segment_count: int,
        final_parameters: np.ndarray,
    ) -> NativeDyadicPathEvidence:
        fractions = np.arange(segment_count + 1, dtype=np.float64) / segment_count
        raw_equalities = np.zeros(_EQUALITY_SIZE, dtype=np.float64)
        roots = (
            fractions[:, None]
            * np.linspace(
                -1.0e-3,
                1.0e-3,
                _ROOT_SIZE,
                dtype=np.float64,
            )[None, :]
        )
        steps = tuple(
            NativeContinuationStep(
                segment_count=segment_count,
                index=index,
                predecessor_index=None if index == 0 else index - 1,
                coil_little_endian_sha256=_raw_sha256(
                    self.bootstrap_state[:_COIL_SIZE]
                    + fraction * (final_parameters - self.bootstrap_state[:_COIL_SIZE])
                ),
                seed_root_little_endian_sha256=_raw_sha256(
                    roots[0 if index == 0 else index - 1]
                ),
                root_little_endian_sha256=_raw_sha256(roots[index]),
                newton_iterations=0 if index == 0 else 1,
                raw_equalities=raw_equalities,
                raw_equalities_little_endian_sha256=_raw_sha256(raw_equalities),
                residual_l2=0.0,
                residual_infinity_norm=0.0,
                scaled_boozer_infinity_norm=0.0,
            )
            for index, fraction in enumerate(fractions)
        )
        return NativeDyadicPathEvidence(
            segment_count=segment_count,
            roots=roots,
            steps=steps,
        )


def _runtime_provenance(
    tmp_path: Path,
    sources: tuple[SourcePath, ...],
) -> RuntimeProvenance:
    binding_paths = {
        name: tmp_path / "runtime" / name
        for name in ("python", "simsopt.py", "simsopt_jax.py", "simsoptpp.so")
    }
    for name, path in binding_paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"runtime:{name}\n".encode())
    adapter_path = next(
        source.path
        for source in sources
        if source.logical_path
        == "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py"
    )
    return RuntimeProvenance(
        argv=("python", "producer.py"),
        cwd="/test",
        python_executable=str(binding_paths["python"]),
        python_version="3.11",
        platform="test-platform",
        numpy_version=np.__version__,
        jax_version="test-jax",
        jaxlib_version="test-jaxlib",
        simsopt_path=str(binding_paths["simsopt.py"]),
        simsopt_jax_path=str(binding_paths["simsopt_jax.py"]),
        adapter_path=str(adapter_path),
        simsopt_sha256=_file_sha256(binding_paths["simsopt.py"]),
        simsopt_jax_sha256=_file_sha256(binding_paths["simsopt_jax.py"]),
        adapter_sha256=_file_sha256(adapter_path),
        native_extension_path=str(binding_paths["simsoptpp.so"]),
        native_extension_sha256=_file_sha256(binding_paths["simsoptpp.so"]),
        python_executable_sha256=_file_sha256(binding_paths["python"]),
        effective_environment_sha256="e" * 64,
        git_head="a" * 40,
        tracked_diff_sha256="b" * 64,
        repository_dirty=True,
    )


def _reference_dict(path: Path) -> dict[str, object]:
    value = load_canonical_json_bytes(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _write_receipt_array(
    receipt_root: Path,
    relative_path: str,
    values: np.ndarray,
) -> dict[str, object]:
    reference = write_array(receipt_root, relative_path, values)
    return {
        "dtype": reference.dtype,
        "order": reference.order,
        "path": reference.path,
        "sha256": reference.sha256,
        "shape": list(reference.shape),
    }


@pytest.fixture
def historical_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> _HistoricalFixture:
    campaign_root = tmp_path / "campaign"
    input_root = campaign_root / "inputs"
    receipt_root = campaign_root / "receipt"
    bootstrap_coils = np.linspace(-1.0, 1.0, _COIL_SIZE, dtype=np.float64)
    bootstrap_surface = np.linspace(-0.5, 0.5, _SURFACE_SIZE, dtype=np.float64)
    final_parameters = bootstrap_coils + 0.25
    create_input_bundle(
        input_root,
        case_id="native-single-stage-boozer-vacuum-optimization",
        random_seed=1,
        arrays={
            "axis_dofs": np.zeros(21, dtype=np.float64),
            "coil_dofs": bootstrap_coils,
            "surface_dofs": bootstrap_surface,
        },
        configuration={"inner_tolerance": 1.0e-13},
        scale="native_default",
    )
    final_path = "values/final.npy"
    bootstrap_coil_path = "values/bootstrap-coils.npy"
    bootstrap_surface_path = "values/bootstrap-surface.npy"
    values: dict[str, object] = {
        "final:parameters": _write_receipt_array(
            receipt_root, final_path, final_parameters
        ),
        "construction:coil_dofs": _write_receipt_array(
            receipt_root, bootstrap_coil_path, bootstrap_coils
        ),
        "construction:surface_dofs": _write_receipt_array(
            receipt_root, bootstrap_surface_path, bootstrap_surface
        ),
    }
    observables = {
        "final:objective": _OBJECTIVE,
        "final:iota": -0.4,
        "final:volume": 2.0,
        "final:non_qs_ratio": _OBJECTIVE,
        "final:boozer_residual": 0.0,
        "final:boozer_residual_rms": 0.0,
        "final:major_radius_penalty": 0.0,
        "final:length_penalty": 0.0,
    }
    for index, (key, value) in enumerate(observables.items()):
        values[key] = _write_receipt_array(
            receipt_root,
            f"values/observable-{index}.npy",
            np.asarray((value,), dtype=np.float64),
        )
    receipt = {
        "backend_mode": "native_cpu",
        "lane": "native-cpu",
        "nit": 1000,
        "normalized_status": "budget_exhausted",
        "precision": "fp64",
        "raw_status": "constraints_satisfied=True",
        "scale": "native_default",
        "success": False,
        "values": values,
    }
    receipt_path = receipt_root / "lane_result.json"
    write_bytes_exclusive(receipt_root, "lane_result.json", parity_json_bytes(receipt))
    trajectory_rows = [
        json.dumps(
            {
                "iteration": index,
                "objective": _OBJECTIVE + (1000 - index) * 1.0e-12,
                "wall_seconds_from_start": float(index),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for index in range(1, 1000)
    ]
    trajectory_rows.append(
        '{"iteration":1000,"objective":4.4822246533126125e-08,'
        '"wall_seconds_from_start":287.30421751597896}'
    )
    trajectory_path = receipt_root / "trajectory.jsonl"
    trajectory_path.write_text("\n".join(trajectory_rows) + "\n", encoding="utf-8")
    input_bundle_path = input_root / "input_bundle.json"
    monkeypatch.setattr(
        reference_module, "HISTORICAL_RECEIPT_SHA256", _file_sha256(receipt_path)
    )
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_TRAJECTORY_SHA256",
        _file_sha256(trajectory_path),
    )
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_INPUT_BUNDLE_SHA256",
        _file_sha256(input_bundle_path),
    )
    monkeypatch.setattr(reference_module, "HISTORICAL_FINAL_PARAMETER_PATH", final_path)
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_FINAL_PARAMETER_SHA256",
        _file_sha256(receipt_root / final_path),
    )
    monkeypatch.setattr(
        reference_module, "HISTORICAL_BOOTSTRAP_COIL_PATH", bootstrap_coil_path
    )
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_BOOTSTRAP_COIL_SHA256",
        _file_sha256(receipt_root / bootstrap_coil_path),
    )
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_BOOTSTRAP_SURFACE_PATH",
        bootstrap_surface_path,
    )
    monkeypatch.setattr(
        reference_module,
        "HISTORICAL_BOOTSTRAP_SURFACE_SHA256",
        _file_sha256(receipt_root / bootstrap_surface_path),
    )
    return _HistoricalFixture(
        paths=HistoricalAuthorityPaths(
            receipt=receipt_path,
            trajectory=trajectory_path,
            input_bundle=input_bundle_path,
        ),
        bootstrap_coils=bootstrap_coils,
        final_parameters=final_parameters,
    )


def _sources(tmp_path: Path) -> tuple[SourcePath, ...]:
    sources: list[SourcePath] = []
    for logical_path in reference_module.REQUIRED_SOURCE_LOGICAL_PATHS:
        path = tmp_path / "sources" / logical_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SOURCE = {logical_path!r}\n", encoding="utf-8")
        sources.append(SourcePath(logical_path, path))
    return tuple(sources)


def _produce(
    tmp_path: Path,
    historical: _HistoricalFixture,
    *,
    fail_reconstruction: bool = False,
    typed_unusable: bool = False,
) -> Path:
    output = tmp_path / "sealed-reference"
    sources = _sources(tmp_path)
    runtime = _FakeRuntime(
        bootstrap_state=np.concatenate(
            (historical.bootstrap_coils, np.zeros(_ROOT_SIZE, dtype=np.float64))
        ),
        fixed_first_base_current=7.0,
        final_parameters=historical.final_parameters,
        fail_reconstruction=fail_reconstruction,
        typed_unusable=typed_unusable,
    )
    result = produce_native_equivalent_reference(
        output_root=output,
        historical_paths=historical.paths,
        runtime=runtime,
        runtime_provenance=_runtime_provenance(tmp_path, sources),
        source_paths=sources,
    )
    assert result.disposition == (
        REFERENCE_NOT_PRODUCED if fail_reconstruction or typed_unusable else USABLE
    )
    return output


def test_producer_seals_self_contained_usable_reference_and_refuses_overwrite(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    output = _produce(tmp_path, historical_fixture)
    validation = validate_native_equivalent_reference(output)
    assert validation.usable
    assert validation.failure_reasons == ()
    assert not output.stat().st_mode & 0o222
    document = _reference_dict(output / "reference.json")
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    assert (
        evidence["constraints_satisfied_boolean_used_as_numerical_reference"] is False
    )
    with pytest.raises(FileExistsError, match="already exists"):
        produce_native_equivalent_reference(
            output_root=output,
            historical_paths=historical_fixture.paths,
            runtime=_FakeRuntime(
                np.concatenate(
                    (
                        historical_fixture.bootstrap_coils,
                        np.zeros(_ROOT_SIZE, dtype=np.float64),
                    )
                ),
                7.0,
                historical_fixture.final_parameters,
            ),
            runtime_provenance=_runtime_provenance(tmp_path, _sources(tmp_path)),
            source_paths=(),
        )


def test_reconstruction_failure_is_sealed_as_reference_not_produced(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    output = _produce(tmp_path, historical_fixture, fail_reconstruction=True)
    validation = validate_native_equivalent_reference(output)
    assert validation.disposition == REFERENCE_NOT_PRODUCED
    assert validation.failure_reasons == ("REFERENCE_EVIDENCE_MISSING",)


def test_typed_unusable_evidence_is_retained_and_sealed_not_produced(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    output = _produce(tmp_path, historical_fixture, typed_unusable=True)
    validation = validate_native_equivalent_reference(output)
    assert validation.disposition == REFERENCE_NOT_PRODUCED
    assert validation.failure_reasons == ("ADAPTER_TYPED_UNUSABLE",)
    document = _reference_dict(output / "reference.json")
    assert document["evidence"] is not None
    assert document["diagnostics"] is not None
    assert document["reconstruction_failure"] == (
        "NativeReferenceEvidence.usable=false"
    )


def _writable_copy(source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    for path in [destination, *destination.rglob("*")]:
        if path.is_dir():
            path.chmod(0o755)
        else:
            path.chmod(0o644)
    return destination


def _refresh_artifact_manifest(root: Path) -> None:
    manifest_path = root / reference_module.ARTIFACT_MANIFEST_FILENAME
    entries: list[dict[str, object]] = []
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate != manifest_path
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        payload = path.read_bytes()
        entries.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    manifest_path.write_bytes(
        reference_module.canonical_json_bytes(
            {
                "entries": entries,
                "schema_version": reference_module.ARTIFACT_MANIFEST_SCHEMA_VERSION,
            }
        )
    )


def _seal_test_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    root.chmod(0o555)


@pytest.mark.parametrize(
    "mutation",
    ("writable", "noncanonical", "authority-bytes", "array-symlink", "summary"),
)
def test_validator_rejects_integrity_and_summary_mutations(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
    mutation: str,
) -> None:
    original = _produce(tmp_path, historical_fixture)
    mutated = _writable_copy(original, tmp_path / f"mutated-{mutation}")
    reference_path = mutated / "reference.json"
    document = _reference_dict(reference_path)
    if mutation == "writable":
        pass
    elif mutation == "noncanonical":
        reference_path.write_text("{}\n", encoding="utf-8")
    elif mutation == "authority-bytes":
        authority_ref = document["authority_manifest"]
        assert isinstance(authority_ref, dict)
        authority_path = mutated / str(authority_ref["relative_path"])
        authority_path.write_bytes(authority_path.read_bytes() + b"x")
    elif mutation == "array-symlink":
        evidence = document["evidence"]
        assert isinstance(evidence, dict)
        arrays = evidence["arrays"]
        assert isinstance(arrays, dict)
        state_ref = arrays["state"]
        assert isinstance(state_ref, dict)
        state_path = mutated / str(state_ref["relative_path"])
        state_path.unlink()
        state_path.symlink_to(reference_path)
    else:
        document["summary_usable"] = False
        reference_path.write_bytes(reference_module.canonical_json_bytes(document))
    if mutation != "writable":
        reference_path.chmod(0o444)
        for path in mutated.rglob("*"):
            if path.is_file() and not path.is_symlink():
                path.chmod(0o444)
            elif path.is_dir():
                path.chmod(0o555)
        mutated.chmod(0o555)
    with pytest.raises(ReferenceArtifactError):
        validate_native_equivalent_reference(mutated)


def test_validator_recomputes_comparisons_instead_of_trusting_passed_boolean(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    original = _produce(tmp_path, historical_fixture)
    mutated = _writable_copy(original, tmp_path / "mutated-derived-boolean")
    reference_path = mutated / "reference.json"
    document = _reference_dict(reference_path)
    evidence = document["evidence"]
    assert isinstance(evidence, dict)
    comparisons = evidence["comparisons"]
    assert isinstance(comparisons, list)
    first = comparisons[0]
    assert isinstance(first, dict)
    first["passed"] = False
    reference_path.write_bytes(reference_module.canonical_json_bytes(document))
    _refresh_artifact_manifest(mutated)
    _seal_test_tree(mutated)
    assert validate_native_equivalent_reference(mutated).usable


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("coil_little_endian_sha256", "0" * 64),
        ("seed_root_little_endian_sha256", "1" * 64),
        ("newton_iterations", 21),
    ),
)
def test_validator_recomputes_continuation_step_contract(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
    field: str,
    value: object,
) -> None:
    original = _produce(tmp_path, historical_fixture)
    mutated = _writable_copy(original, tmp_path / f"mutated-{field}")
    reference_path = mutated / reference_module.REFERENCE_FILENAME
    document = _reference_dict(reference_path)
    diagnostics_reference = document["diagnostics"]
    assert isinstance(diagnostics_reference, dict)
    diagnostics_path = mutated / str(diagnostics_reference["relative_path"])
    diagnostics = _reference_dict(diagnostics_path)
    coarse_steps = diagnostics["coarse_steps"]
    assert isinstance(coarse_steps, list)
    step = coarse_steps[1]
    assert isinstance(step, dict)
    step[field] = value
    diagnostics_payload = reference_module.canonical_json_bytes(diagnostics)
    diagnostics_path.write_bytes(diagnostics_payload)
    diagnostics_reference["sha256"] = hashlib.sha256(diagnostics_payload).hexdigest()
    diagnostics_reference["size_bytes"] = len(diagnostics_payload)
    reference_path.write_bytes(reference_module.canonical_json_bytes(document))
    _refresh_artifact_manifest(mutated)
    _seal_test_tree(mutated)
    with pytest.raises(ReferenceArtifactError, match="disposition differs"):
        validate_native_equivalent_reference(mutated)


def test_validator_rejects_unreferenced_extra_file(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    original = _produce(tmp_path, historical_fixture)
    mutated = _writable_copy(original, tmp_path / "mutated-extra-file")
    (mutated / "undeclared.bin").write_bytes(b"undeclared")
    _seal_test_tree(mutated)
    with pytest.raises(ReferenceArtifactError, match="unreferenced"):
        validate_native_equivalent_reference(mutated)


def test_producer_requires_exact_source_set(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    sources = _sources(tmp_path)
    runtime = _FakeRuntime(
        np.concatenate((historical_fixture.bootstrap_coils, np.zeros(_ROOT_SIZE))),
        7.0,
        historical_fixture.final_parameters,
    )
    with pytest.raises(ReferenceArtifactError, match="source provenance set"):
        produce_native_equivalent_reference(
            output_root=tmp_path / "bad-source-output",
            historical_paths=historical_fixture.paths,
            runtime=runtime,
            runtime_provenance=_runtime_provenance(tmp_path, sources),
            source_paths=sources[:-1],
        )


def test_atomic_publication_has_exactly_one_concurrent_winner(
    tmp_path: Path,
    historical_fixture: _HistoricalFixture,
) -> None:
    sources = _sources(tmp_path)
    provenance = _runtime_provenance(tmp_path, sources)
    output = tmp_path / "concurrent-output"

    def produce() -> ReferenceValidationResult:
        return produce_native_equivalent_reference(
            output_root=output,
            historical_paths=historical_fixture.paths,
            runtime=_FakeRuntime(
                np.concatenate(
                    (historical_fixture.bootstrap_coils, np.zeros(_ROOT_SIZE))
                ),
                7.0,
                historical_fixture.final_parameters,
            ),
            runtime_provenance=provenance,
            source_paths=sources,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(produce) for _ in range(2)]
    outcomes = [future.exception() for future in futures]
    assert sum(outcome is None for outcome in outcomes) == 1
    assert sum(isinstance(outcome, FileExistsError) for outcome in outcomes) == 1
    assert validate_native_equivalent_reference(output).usable
