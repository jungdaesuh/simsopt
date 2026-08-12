"""Native CPU reference child for the frozen Phase 0 changed-state candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol

import numpy as np
from examples.jax.parity.cases.native_boozerqa import (
    NativeBaselineAnchor,
    NativeCandidateEvaluation,
    _prepare_native_variant_runtime,
)
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import read_input_bundle

from benchmarks.single_stage_compute_graph_c0_runner import _runtime_identity
from benchmarks.single_stage_compute_graph_isolated_launch import (
    normalize_route_environment,
    normalize_static_timing_environment,
    observe_effective_numerical_policies,
)

SCHEMA_ID: Final = "single-stage-compute-graph-native-reference-v3"
EXPECTED_PARAMETER_COUNT: Final = 461


class NativeReferenceError(RuntimeError):
    """The native reference cannot be bound to the frozen specimen."""


class _PreparedNativeReference(Protocol):
    initial_parameters: np.ndarray
    baseline_anchor: NativeBaselineAnchor

    def evaluate_candidate(
        self, parameters: np.ndarray
    ) -> NativeCandidateEvaluation: ...


@dataclass(frozen=True, slots=True)
class NativeReferenceBinding:
    input_bundle_sha256: str
    input_fingerprint: str
    configuration_fingerprint: str
    specimen_sha256: str
    source_sha256: str
    runtime_identity_sha256: str
    interpreter_path: str
    native_simsoptpp_path: str
    native_simsoptpp_sha256: str
    runtime_contract: Mapping[str, object]

    def __post_init__(self) -> None:
        for name in (
            "input_fingerprint",
            "input_bundle_sha256",
            "configuration_fingerprint",
            "specimen_sha256",
            "source_sha256",
            "runtime_identity_sha256",
            "native_simsoptpp_sha256",
        ):
            _sha256(getattr(self, name), name)
        interpreter = Path(self.interpreter_path)
        if (
            not interpreter.is_absolute()
            or Path(os.path.abspath(interpreter)) != interpreter
            or not interpreter.is_file()
            or not os.access(interpreter, os.X_OK)
        ):
            raise NativeReferenceError(
                "interpreter_path must be an absolute executable path"
            )
        native_extension = Path(self.native_simsoptpp_path)
        if (
            not native_extension.is_absolute()
            or native_extension.resolve() != native_extension
        ):
            raise NativeReferenceError(
                "native_simsoptpp_path must be an absolute resolved path"
            )


def _parameter_sha256(parameters: np.ndarray) -> str:
    canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: str, context: str) -> str:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise NativeReferenceError(f"{context} must be a lowercase SHA-256 digest")
    return value


def _canonical_candidate(path: Path, expected_sha256: str) -> np.ndarray:
    loaded = np.load(path, allow_pickle=False)
    if loaded.dtype != np.dtype(np.float64) or loaded.shape != (
        EXPECTED_PARAMETER_COUNT,
    ):
        raise NativeReferenceError(
            "candidate must have exact shape (461,) and dtype float64"
        )
    candidate = np.ascontiguousarray(loaded, dtype=np.dtype("<f8"))
    if not bool(np.all(np.isfinite(candidate))):
        raise NativeReferenceError("candidate must contain only finite values")
    if _parameter_sha256(candidate) != expected_sha256:
        raise NativeReferenceError("candidate SHA-256 does not match the specimen")
    candidate.setflags(write=False)
    return candidate


def _is_fp64_scalar(value: object) -> bool:
    return (
        not isinstance(value, (bool, np.bool_))
        and isinstance(value, (float, np.floating))
        and np.asarray(value).dtype == np.dtype(np.float64)
        and math.isfinite(float(value))
    )


def _validate_evaluation(
    evaluation: NativeCandidateEvaluation, *, context: str
) -> None:
    if not _is_fp64_scalar(evaluation.objective):
        raise NativeReferenceError("native objective must be finite")
    if evaluation.gradient.dtype != np.dtype(
        np.float64
    ) or evaluation.gradient.shape != (EXPECTED_PARAMETER_COUNT,):
        raise NativeReferenceError(
            "native gradient must have exact shape (461,) and dtype float64"
        )
    if not bool(np.all(np.isfinite(evaluation.gradient))):
        raise NativeReferenceError("native gradient must contain only finite values")
    if not isinstance(evaluation.inner_solver_success, (bool, np.bool_)):
        raise NativeReferenceError(
            f"native {context} inner Newton success must be boolean"
        )
    if not bool(evaluation.inner_solver_success):
        raise NativeReferenceError(f"native {context} inner Newton solve failed")
    if (
        not _is_fp64_scalar(evaluation.solver_residual_l2)
        or float(evaluation.solver_residual_l2) < 0.0
        or not _is_fp64_scalar(evaluation.solver_residual_inf)
        or float(evaluation.solver_residual_inf) < 0.0
    ):
        raise NativeReferenceError(
            f"native {context} solver residual certificates are invalid"
        )


def _serialized_evaluation(
    evaluation: NativeCandidateEvaluation,
    *,
    parameter_sha256: str,
    elapsed_ns: int,
) -> dict[str, object]:
    if elapsed_ns < 1:
        raise NativeReferenceError("native evaluation elapsed_ns must be positive")
    return {
        "parameter_sha256": _sha256(parameter_sha256, "evaluation parameter_sha256"),
        "objective_dtype": "float64",
        "objective": float(evaluation.objective),
        "gradient_dtype": "float64",
        "gradient": evaluation.gradient.tolist(),
        "inner_newton_success": bool(evaluation.inner_solver_success),
        "residual_certificates": {
            "solver_residual_l2": float(evaluation.solver_residual_l2),
            "solver_residual_inf": float(evaluation.solver_residual_inf),
        },
        "elapsed_ns": elapsed_ns,
    }


def _validate_runtime_binding(binding: NativeReferenceBinding) -> None:
    observed_interpreter = Path(sys.executable).absolute()
    if observed_interpreter != Path(binding.interpreter_path):
        raise NativeReferenceError("native interpreter path does not match binding")
    simsoptpp_module = sys.modules.get("simsoptpp")
    module_file = getattr(simsoptpp_module, "__file__", None)
    if not isinstance(module_file, str):
        raise NativeReferenceError("native simsoptpp module is not loaded")
    observed_simsoptpp = Path(module_file).resolve()
    if observed_simsoptpp != Path(binding.native_simsoptpp_path):
        raise NativeReferenceError("native simsoptpp path does not match binding")
    if _sha256_path(observed_simsoptpp) != binding.native_simsoptpp_sha256:
        raise NativeReferenceError("native simsoptpp SHA-256 does not match binding")
    contract = binding.runtime_contract
    if frozenset(contract) != frozenset(
        {
            "runtime",
            "static_environment",
            "route_environment",
            "policies",
            "expected_runtime_identity_sha256",
        }
    ):
        raise NativeReferenceError("runtime contract fields are invalid")
    expected_identity = _sha256(
        str(contract["expected_runtime_identity_sha256"]),
        "expected_runtime_identity_sha256",
    )
    provenance = {
        "interpreter_path": binding.interpreter_path,
        "runtime": contract["runtime"],
        "environment": contract["static_environment"],
        "policies": contract["policies"],
    }
    if _runtime_identity(provenance) != expected_identity:
        raise NativeReferenceError("runtime contract identity is inconsistent")
    if expected_identity != binding.runtime_identity_sha256:
        raise NativeReferenceError("runtime contract differs from native binding")
    expected_environment = contract["static_environment"]
    if not isinstance(expected_environment, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in expected_environment.items()
    ):
        raise NativeReferenceError("runtime contract environment is invalid")
    if normalize_static_timing_environment(os.environ) != expected_environment:
        raise NativeReferenceError("static runtime environment differs from contract")
    route_environment = contract["route_environment"]
    if (
        not isinstance(route_environment, dict)
        or normalize_route_environment(os.environ) != route_environment
    ):
        raise NativeReferenceError("route runtime environment differs from contract")
    policies = contract["policies"]
    if not isinstance(policies, dict):
        raise NativeReferenceError("runtime contract policies are invalid")
    blocks = policies.get("quadrature_block_sizes")
    if not isinstance(blocks, list) or not all(
        isinstance(value, int) for value in blocks
    ):
        raise NativeReferenceError("runtime quadrature policy is invalid")
    if observe_effective_numerical_policies(sum(blocks)) != policies:
        raise NativeReferenceError("observed runtime policies differ from specimen")
    runtime = contract["runtime"]
    if not isinstance(runtime, dict):
        raise NativeReferenceError("runtime contract runtime is invalid")
    if runtime:
        import jax

        devices = jax.devices()
        observed_runtime = {
            "python_version": sys.version,
            "jax_version": jax.__version__,
            "jaxlib_version": jax.lib.__version__,
            "jax_backend": jax.default_backend(),
            "fp64_x64_enabled": bool(jax.config.jax_enable_x64),
            "cuda_runtime": str(
                getattr(devices[0].client, "platform_version", "unknown")
            ),
        }
        for key, observed_value in observed_runtime.items():
            if runtime.get(key) != observed_value:
                raise NativeReferenceError(f"observed runtime differs for {key}")


def build_native_reference_document(
    candidate: np.ndarray,
    parameter_sha256: str,
    prepared: _PreparedNativeReference,
    evaluation: NativeCandidateEvaluation,
    initial_evaluation: NativeCandidateEvaluation,
    *,
    elapsed_ns: int,
    initial_elapsed_ns: int,
    binding: NativeReferenceBinding,
) -> dict[str, object]:
    """Validate and serialize changed-state and initial-point native results."""

    if elapsed_ns < 1:
        raise NativeReferenceError("elapsed_ns must be positive")
    if candidate.shape != (EXPECTED_PARAMETER_COUNT,) or candidate.dtype != np.dtype(
        np.float64
    ):
        raise NativeReferenceError("candidate must be an exact FP64 461-vector")
    if not bool(np.all(np.isfinite(candidate))):
        raise NativeReferenceError("candidate must contain only finite values")
    if _parameter_sha256(candidate) != parameter_sha256:
        raise NativeReferenceError("candidate does not match parameter_sha256")
    if prepared.initial_parameters.dtype != np.dtype(
        np.float64
    ) or prepared.initial_parameters.shape != (EXPECTED_PARAMETER_COUNT,):
        raise NativeReferenceError("native baseline must be an exact FP64 461-vector")
    if not bool(np.all(np.isfinite(prepared.initial_parameters))):
        raise NativeReferenceError("native baseline must contain only finite values")
    if np.array_equal(candidate, prepared.initial_parameters):
        raise NativeReferenceError("candidate must differ from the native baseline")

    _validate_evaluation(evaluation, context="changed-state")
    _validate_evaluation(initial_evaluation, context="initial-point")
    anchor = prepared.baseline_anchor
    initial_parameter_sha256 = _parameter_sha256(prepared.initial_parameters)
    if _sha256(anchor.parameter_sha256, "baseline parameter_sha256") != (
        initial_parameter_sha256
    ):
        raise NativeReferenceError("native baseline parameter SHA-256 is inconsistent")
    _sha256(anchor.surface_sha256, "baseline surface_sha256")
    if not anchor.inner_solver_success:
        raise NativeReferenceError("native baseline inner Newton solve failed")
    for name, value in (
        ("iota", anchor.iota),
        ("G", anchor.G),
        ("iota_target", anchor.iota_target),
        ("volume_target", anchor.volume_target),
        ("major_radius_target", anchor.major_radius_target),
        ("total_length_target", anchor.total_length_target),
    ):
        if not math.isfinite(value):
            raise NativeReferenceError(f"native baseline {name} must be finite")

    return {
        "schema_id": SCHEMA_ID,
        "identity": {
            "input_bundle_sha256": binding.input_bundle_sha256,
            "input_fingerprint": binding.input_fingerprint,
            "configuration_fingerprint": binding.configuration_fingerprint,
            "specimen_sha256": binding.specimen_sha256,
            "source_sha256": binding.source_sha256,
            "runtime_identity_sha256": binding.runtime_identity_sha256,
            "interpreter_path": binding.interpreter_path,
            "native_simsoptpp_path": binding.native_simsoptpp_path,
            "native_simsoptpp_sha256": binding.native_simsoptpp_sha256,
        },
        "parameter_sha256": parameter_sha256,
        "objective_dtype": "float64",
        "objective": evaluation.objective,
        "gradient_dtype": "float64",
        "gradient": evaluation.gradient.tolist(),
        "inner_newton_success": evaluation.inner_solver_success,
        "residual_certificates": {
            "solver_residual_l2": evaluation.solver_residual_l2,
            "solver_residual_inf": evaluation.solver_residual_inf,
        },
        "elapsed_ns": elapsed_ns,
        "initial_evaluation": _serialized_evaluation(
            initial_evaluation,
            parameter_sha256=initial_parameter_sha256,
            elapsed_ns=initial_elapsed_ns,
        ),
        "baseline_anchor": {
            "parameter_sha256": anchor.parameter_sha256,
            "surface_sha256": anchor.surface_sha256,
            "iota": anchor.iota,
            "G": anchor.G,
            "inner_solver_success": anchor.inner_solver_success,
            "targets": {
                "iota": anchor.iota_target,
                "volume": anchor.volume_target,
                "major_radius": anchor.major_radius_target,
                "total_length": anchor.total_length_target,
            },
        },
    }


def evaluate_native_reference(
    input_root: Path,
    candidate: np.ndarray,
    parameter_sha256: str,
    binding: NativeReferenceBinding,
) -> Mapping[str, object]:
    """Prepare the canonical native runtime and evaluate the frozen candidate."""

    bundle, arrays = read_input_bundle(input_root)
    if bundle.case_id != SPEC.case_id or bundle.scale != "native_default":
        raise NativeReferenceError(
            "input bundle must be the native-default single-stage specimen"
        )
    if bundle.input_fingerprint != binding.input_fingerprint:
        raise NativeReferenceError("input fingerprint does not match binding")
    if _sha256_path(input_root / "input_bundle.json") != binding.input_bundle_sha256:
        raise NativeReferenceError("input bundle bytes do not match binding")
    if bundle.configuration_fingerprint != binding.configuration_fingerprint:
        raise NativeReferenceError("configuration fingerprint does not match binding")
    _validate_runtime_binding(binding)
    started_ns = time.monotonic_ns()
    prepared = _prepare_native_variant_runtime(bundle, arrays, SPEC)
    evaluation = prepared.evaluate_candidate(candidate)
    elapsed_ns = time.monotonic_ns() - started_ns
    initial_started_ns = time.monotonic_ns()
    initial_prepared = _prepare_native_variant_runtime(bundle, arrays, SPEC)
    if (
        initial_prepared.initial_parameters.dtype != np.dtype(np.float64)
        or initial_prepared.initial_parameters.shape != (EXPECTED_PARAMETER_COUNT,)
        or not bool(np.all(np.isfinite(initial_prepared.initial_parameters)))
        or not np.array_equal(
            initial_prepared.initial_parameters, prepared.initial_parameters
        )
    ):
        raise NativeReferenceError(
            "independent native runtimes expose different initial parameters"
        )
    initial_evaluation = initial_prepared.evaluate_candidate(
        initial_prepared.initial_parameters
    )
    initial_elapsed_ns = time.monotonic_ns() - initial_started_ns
    document = build_native_reference_document(
        candidate,
        parameter_sha256,
        prepared,
        evaluation,
        initial_evaluation,
        elapsed_ns=elapsed_ns,
        initial_elapsed_ns=initial_elapsed_ns,
        binding=binding,
    )
    return document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--parameter-sha256", required=True)
    parser.add_argument("--input-fingerprint", required=True)
    parser.add_argument("--input-bundle-sha256", required=True)
    parser.add_argument("--configuration-fingerprint", required=True)
    parser.add_argument("--specimen-sha256", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--runtime-identity-sha256", required=True)
    parser.add_argument("--interpreter-path", required=True)
    parser.add_argument("--native-simsoptpp-path", required=True)
    parser.add_argument("--native-simsoptpp-sha256", required=True)
    parser.add_argument("--runtime-contract-json", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        parameter_sha256 = _sha256(args.parameter_sha256, "parameter_sha256")
        candidate = _canonical_candidate(args.candidate, parameter_sha256)
        runtime_contract = json.loads(args.runtime_contract_json)
        if not isinstance(runtime_contract, dict):
            raise NativeReferenceError("runtime contract must be a JSON object")
        binding = NativeReferenceBinding(
            input_bundle_sha256=args.input_bundle_sha256,
            input_fingerprint=args.input_fingerprint,
            configuration_fingerprint=args.configuration_fingerprint,
            specimen_sha256=args.specimen_sha256,
            source_sha256=args.source_sha256,
            runtime_identity_sha256=args.runtime_identity_sha256,
            interpreter_path=args.interpreter_path,
            native_simsoptpp_path=args.native_simsoptpp_path,
            native_simsoptpp_sha256=args.native_simsoptpp_sha256,
            runtime_contract=runtime_contract,
        )
        document = evaluate_native_reference(
            args.input_root,
            candidate,
            parameter_sha256,
            binding,
        )
        sys.stdout.write(json.dumps(document, sort_keys=True, separators=(",", ":")))
        sys.stdout.write("\n")
    except (OSError, ValueError, RuntimeError) as error:
        sys.stderr.write(f"native reference failed: {error}\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
