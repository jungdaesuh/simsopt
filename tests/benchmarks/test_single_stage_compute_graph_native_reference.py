from __future__ import annotations

import hashlib
import inspect
import os
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.single_stage_compute_graph_c0_runner import _runtime_identity
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_route_environment,
    normalize_static_timing_environment,
)
from benchmarks.single_stage_compute_graph_native_reference import (
    SCHEMA_ID,
    NativeReferenceBinding,
    NativeReferenceError,
    _canonical_candidate,
    _validate_runtime_binding,
    build_native_reference_document,
)
from examples.jax.parity.cases import native_boozerqa
from examples.jax.parity.cases.native_boozerqa import (
    NativeBaselineAnchor,
    NativeCandidateEvaluation,
    _validate_reconstructed_bundle_arrays,
)


def _sha256(parameters: np.ndarray) -> str:
    canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _binding(tmp_path: Path) -> NativeReferenceBinding:
    interpreter = tmp_path / "python"
    interpreter.write_bytes(b"python")
    interpreter.chmod(0o755)
    simsoptpp = tmp_path / "simsoptpp.so"
    simsoptpp.write_bytes(b"native-extension")
    policies = {
        "dense_batch_width": 8,
        "point_chunk_size": None,
        "coil_chunk_size": None,
        "quadrature_block_sizes": [128, 122],
    }
    provenance = {
        "interpreter_path": str(interpreter.resolve()),
        "runtime": {},
        "environment": normalize_static_timing_environment(os.environ),
        "policies": policies,
    }
    runtime_identity = _runtime_identity(provenance)
    return NativeReferenceBinding(
        input_bundle_sha256="1" * 64,
        input_fingerprint="2" * 64,
        configuration_fingerprint="3" * 64,
        specimen_sha256="4" * 64,
        source_sha256="5" * 64,
        runtime_identity_sha256=runtime_identity,
        interpreter_path=str(interpreter.resolve()),
        native_simsoptpp_path=str(simsoptpp.resolve()),
        native_simsoptpp_sha256=hashlib.sha256(simsoptpp.read_bytes()).hexdigest(),
        runtime_contract={
            "runtime": {},
            "static_environment": normalize_static_timing_environment(os.environ),
            "route_environment": normalize_route_environment(os.environ),
            "policies": policies,
            "expected_runtime_identity_sha256": runtime_identity,
        },
    )


@dataclass(frozen=True, slots=True)
class _Prepared:
    initial_parameters: np.ndarray
    baseline_anchor: NativeBaselineAnchor

    def evaluate_candidate(self, parameters: np.ndarray) -> NativeCandidateEvaluation:
        raise AssertionError("the serializer must not reevaluate the candidate")


def _prepared() -> _Prepared:
    baseline = np.zeros(461, dtype=np.float64)
    baseline.setflags(write=False)
    return _Prepared(
        initial_parameters=baseline,
        baseline_anchor=NativeBaselineAnchor(
            parameter_sha256=_sha256(baseline),
            surface_sha256="1" * 64,
            iota=-0.4,
            G=1.2,
            iota_target=-0.4,
            volume_target=0.03,
            major_radius_target=1.0,
            total_length_target=18.0,
            inner_solver_success=True,
        ),
    )


def _evaluation() -> NativeCandidateEvaluation:
    return NativeCandidateEvaluation(
        objective=1.25,
        gradient=np.linspace(-1.0, 1.0, 461, dtype=np.float64),
        inner_solver_success=True,
        solver_residual_l2=2.0e-14,
        solver_residual_inf=1.0e-14,
    )


def test_native_reference_serializes_exact_candidate_and_anchor(tmp_path: Path) -> None:
    candidate = np.linspace(0.1, 0.2, 461, dtype=np.float64)
    parameter_sha256 = _sha256(candidate)

    document = build_native_reference_document(
        candidate,
        parameter_sha256,
        _prepared(),
        _evaluation(),
        _evaluation(),
        elapsed_ns=123,
        initial_elapsed_ns=456,
        binding=_binding(tmp_path),
    )

    assert document["schema_id"] == SCHEMA_ID
    assert document["parameter_sha256"] == parameter_sha256
    assert document["identity"] == {
        "input_bundle_sha256": "1" * 64,
        "input_fingerprint": "2" * 64,
        "configuration_fingerprint": "3" * 64,
        "specimen_sha256": "4" * 64,
        "source_sha256": "5" * 64,
        "runtime_identity_sha256": _binding(tmp_path).runtime_identity_sha256,
        "interpreter_path": _binding(tmp_path).interpreter_path,
        "native_simsoptpp_path": _binding(tmp_path).native_simsoptpp_path,
        "native_simsoptpp_sha256": _binding(tmp_path).native_simsoptpp_sha256,
    }
    assert document["objective"] == 1.25
    assert len(document["gradient"]) == 461
    assert document["residual_certificates"] == {
        "solver_residual_l2": 2.0e-14,
        "solver_residual_inf": 1.0e-14,
    }
    assert document["initial_evaluation"] == {
        "parameter_sha256": _prepared().baseline_anchor.parameter_sha256,
        "objective_dtype": "float64",
        "objective": 1.25,
        "gradient_dtype": "float64",
        "gradient": _evaluation().gradient.tolist(),
        "inner_newton_success": True,
        "residual_certificates": {
            "solver_residual_l2": 2.0e-14,
            "solver_residual_inf": 1.0e-14,
        },
        "elapsed_ns": 456,
    }
    assert document["baseline_anchor"] == {
        "parameter_sha256": _prepared().baseline_anchor.parameter_sha256,
        "surface_sha256": "1" * 64,
        "iota": -0.4,
        "G": 1.2,
        "inner_solver_success": True,
        "targets": {
            "iota": -0.4,
            "volume": 0.03,
            "major_radius": 1.0,
            "total_length": 18.0,
        },
    }


def test_native_reference_rejects_failed_or_malformed_evaluation(
    tmp_path: Path,
) -> None:
    candidate = np.ones(461, dtype=np.float64)
    failed = NativeCandidateEvaluation(
        objective=1.0,
        gradient=np.ones(461, dtype=np.float64),
        inner_solver_success=False,
        solver_residual_l2=1.0,
        solver_residual_inf=1.0,
    )

    with pytest.raises(NativeReferenceError, match="inner Newton"):
        build_native_reference_document(
            candidate,
            _sha256(candidate),
            _prepared(),
            failed,
            _evaluation(),
            elapsed_ns=1,
            initial_elapsed_ns=1,
            binding=_binding(tmp_path),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("objective", np.float32(1.0), "objective"),
        ("gradient", np.ones(460, dtype=np.float64), "gradient"),
        ("inner_solver_success", 1, "success must be boolean"),
        ("solver_residual_l2", True, "residual certificates"),
        ("solver_residual_inf", -1.0, "residual certificates"),
    ),
)
def test_native_reference_rejects_malformed_initial_evaluation(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    candidate = np.ones(461, dtype=np.float64)
    malformed = replace(_evaluation(), **{field: value})

    with pytest.raises(NativeReferenceError, match=message):
        build_native_reference_document(
            candidate,
            _sha256(candidate),
            _prepared(),
            _evaluation(),
            malformed,
            elapsed_ns=1,
            initial_elapsed_ns=1,
            binding=_binding(tmp_path),
        )


def test_native_reference_rejects_nonpositive_initial_elapsed_time(
    tmp_path: Path,
) -> None:
    candidate = np.ones(461, dtype=np.float64)

    with pytest.raises(NativeReferenceError, match="elapsed_ns must be positive"):
        build_native_reference_document(
            candidate,
            _sha256(candidate),
            _prepared(),
            _evaluation(),
            _evaluation(),
            elapsed_ns=1,
            initial_elapsed_ns=0,
            binding=_binding(tmp_path),
        )


def test_native_reference_binding_rejects_relative_runtime_path(tmp_path: Path) -> None:
    binding = _binding(tmp_path)

    with pytest.raises(NativeReferenceError, match="absolute executable"):
        NativeReferenceBinding(
            input_bundle_sha256=binding.input_bundle_sha256,
            input_fingerprint=binding.input_fingerprint,
            configuration_fingerprint=binding.configuration_fingerprint,
            specimen_sha256=binding.specimen_sha256,
            source_sha256=binding.source_sha256,
            runtime_identity_sha256=binding.runtime_identity_sha256,
            interpreter_path="relative/python",
            native_simsoptpp_path=binding.native_simsoptpp_path,
            native_simsoptpp_sha256=binding.native_simsoptpp_sha256,
            runtime_contract=binding.runtime_contract,
        )


def test_runtime_binding_validates_observed_interpreter_and_simsoptpp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    binding = _binding(tmp_path)
    monkeypatch.setattr(sys, "executable", binding.interpreter_path)
    monkeypatch.setitem(
        sys.modules,
        "simsoptpp",
        SimpleNamespace(__file__=binding.native_simsoptpp_path),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_native_reference.observe_effective_numerical_policies",
        lambda quadrature_nodes: dict(binding.runtime_contract["policies"]),
    )

    _validate_runtime_binding(binding)

    Path(binding.native_simsoptpp_path).write_bytes(b"drifted")
    with pytest.raises(NativeReferenceError, match="SHA-256"):
        _validate_runtime_binding(binding)


def test_candidate_loader_requires_exact_dtype_shape_and_hash(tmp_path: Path) -> None:
    candidate = np.arange(461, dtype=np.float64)
    candidate_path = tmp_path / "candidate.npy"
    np.save(candidate_path, candidate)

    loaded = _canonical_candidate(candidate_path, _sha256(candidate))

    assert np.array_equal(loaded, candidate)
    assert not loaded.flags.writeable
    with pytest.raises(NativeReferenceError, match="SHA-256"):
        _canonical_candidate(candidate_path, "0" * 64)


def test_reconstructed_bundle_validation_rejects_each_drifted_array() -> None:
    arrays = {
        "axis_dofs": np.arange(3, dtype=np.float64),
        "coil_dofs": np.arange(4, dtype=np.float64),
        "surface_dofs": np.arange(5, dtype=np.float64),
    }
    _validate_reconstructed_bundle_arrays(
        arrays,
        axis_dofs=arrays["axis_dofs"],
        coil_dofs=arrays["coil_dofs"],
        surface_dofs=arrays["surface_dofs"],
    )

    for name in arrays:
        drifted = {key: value.copy() for key, value in arrays.items()}
        drifted[name][0] += 1.0
        with pytest.raises(ValueError, match=f"reconstructed {name}"):
            _validate_reconstructed_bundle_arrays(
                drifted,
                axis_dofs=arrays["axis_dofs"],
                coil_dofs=arrays["coil_dofs"],
                surface_dofs=arrays["surface_dofs"],
            )


def test_reconstructed_bundle_validation_allows_surface_fit_roundoff_only() -> None:
    arrays = {
        "axis_dofs": np.arange(3, dtype=np.float64),
        "coil_dofs": np.arange(4, dtype=np.float64),
        "surface_dofs": np.linspace(-1.5, 1.5, 253, dtype=np.float64),
    }
    reconstructed_surface = arrays["surface_dofs"].copy()
    reconstructed_surface[1::2] += np.finfo(np.float64).eps

    _validate_reconstructed_bundle_arrays(
        arrays,
        axis_dofs=arrays["axis_dofs"],
        coil_dofs=arrays["coil_dofs"],
        surface_dofs=reconstructed_surface,
    )

    reconstructed_axis = arrays["axis_dofs"].copy()
    reconstructed_axis[0] += np.finfo(np.float64).eps
    with pytest.raises(ValueError, match="reconstructed axis_dofs"):
        _validate_reconstructed_bundle_arrays(
            arrays,
            axis_dofs=reconstructed_axis,
            coil_dofs=arrays["coil_dofs"],
            surface_dofs=reconstructed_surface,
        )


def test_native_optimization_consumes_the_canonical_prepared_runtime() -> None:
    source = inspect.getsource(native_boozerqa._native)

    assert "_prepare_native_variant_runtime(bundle, arrays, spec)" in source
    assert "BoozerSurface(" not in source
    assert "NonQuasiSymmetricRatio(" not in source
