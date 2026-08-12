"""Sealed native-equivalent reference artifacts for the NEQ-GNTR1 route.

The producer imports exact historical bytes, delegates physical reconstruction
to the stable native endpoint adapter, and atomically seals a self-contained
artifact.  The validator derives usability from retained bytes and never trusts
producer summary booleans.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal, Protocol, TypeAlias

import numpy as np
from examples.jax.parity.input_bundle import read_input_bundle
from simsopt_jax_adapters.geo.single_stage_native_endpoint import (
    BRANCH_ROOT_TOLERANCE,
    COARSE_SEGMENT_COUNT,
    REFINED_SEGMENT_COUNT,
    SEALED_OBSERVABLE_ATOL,
    SEALED_OBSERVABLE_RTOL,
    SSOT_SHA256,
    HistoricalNativeObservablePaths,
    HistoricalNativeParameterMetadata,
    HistoricalNativeParameters,
    NativeEndpointError,
    NativeReferenceEvidence,
    load_historical_native_parameters,
)

SCHEMA_VERSION: Final = "single-stage-native-equivalent-reference-v1"
AUTHORITY_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-native-equivalent-authority-manifest-v1"
)
SOURCE_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-native-equivalent-source-manifest-v1"
)
RUNTIME_SCHEMA_VERSION: Final = "single-stage-native-equivalent-runtime-v1"
DIAGNOSTICS_SCHEMA_VERSION: Final = "single-stage-native-equivalent-diagnostics-v1"
REFERENCE_FILENAME: Final = "reference.json"
ARTIFACT_MANIFEST_FILENAME: Final = "artifact-manifest.json"
ARTIFACT_MANIFEST_SCHEMA_VERSION: Final = (
    "single-stage-native-equivalent-artifact-manifest-v1"
)
USABLE: Final = "USABLE"
REFERENCE_NOT_PRODUCED: Final = "REFERENCE_NOT_PRODUCED"

STATE_SIZE: Final = 716
COIL_SIZE: Final = 461
ROOT_SIZE: Final = 255
EQUALITY_SIZE: Final = 255
COARSE_ROOT_SHAPE: Final = (COARSE_SEGMENT_COUNT + 1, ROOT_SIZE)
REFINED_ROOT_SHAPE: Final = (REFINED_SEGMENT_COUNT + 1, ROOT_SIZE)
FP64_DTYPE: Final = "<f8"
COMMON_KNOT_TOLERANCE: Final = BRANCH_ROOT_TOLERANCE
OBSERVABLE_RTOL: Final = SEALED_OBSERVABLE_RTOL
OBSERVABLE_ATOL: Final = SEALED_OBSERVABLE_ATOL
EXACT_NEWTON_TOLERANCE: Final = 1.0e-13
EXACT_NEWTON_MAXIMUM_ITERATIONS: Final = 20

REQUIRED_SOURCE_LOGICAL_PATHS: Final = (
    "benchmarks/single_stage_native_equivalent_reference.py",
    "benchmarks/run_single_stage_native_equivalent_reference.py",
    "docs/single_stage_jax_gpu_native_equivalent_quality_speed_implementation_plan.md",
    "examples/jax/parity/cases/native_boozerqa.py",
    "examples/jax/parity/cases/native_single_stage_boozer_vacuum.py",
    "examples/jax/parity/input_bundle.py",
    "src/simsopt_jax/objectives/single_stage_fullspace.py",
    "src/simsopt_jax_adapters/geo/single_stage_fullspace.py",
    "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py",
)

HISTORICAL_RECEIPT_SHA256: Final = (
    "8118529751f184f60f0c4d26f338cd1832aae579004d62866fb2a2f6617e9fe4"
)
HISTORICAL_TRAJECTORY_SHA256: Final = (
    "fa81b533b7bd8127b021bc2aa206c01914f91a3ef2e34eee6e0636e2031fed8f"
)
HISTORICAL_INPUT_BUNDLE_SHA256: Final = (
    "9583586c7f2d3798b88eae1475a283b213d1579bd0378d47c89d73d99314b1b7"
)
HISTORICAL_FINAL_PARAMETER_PATH: Final = (
    "values/7c91dc5ad435a89b1f0d2e3b71c8daae68dd50075404abc014f426a2ab193732.npy"
)
HISTORICAL_FINAL_PARAMETER_SHA256: Final = (
    "6ee73fd90f1f4586c7c366e9bb006d1e11735b761a723b3bad813c7a961576fb"
)
HISTORICAL_BOOTSTRAP_COIL_PATH: Final = (
    "values/e926a987718899d49ba20205288be6ec749e4d173e048e4ee9d3ebd57c29934a.npy"
)
HISTORICAL_BOOTSTRAP_COIL_SHA256: Final = (
    "90d798050df3acff5803d7719202e42eae8b86f0c945584b4fd9142a52aaee44"
)
HISTORICAL_BOOTSTRAP_SURFACE_PATH: Final = (
    "values/fe8c1c45f1bc60d81fcc3f54486846c0f3dfcc4a4ab67973a5e692b542269697.npy"
)
HISTORICAL_BOOTSTRAP_SURFACE_SHA256: Final = (
    "6faf75fa46d6ce485c36af04bca77ed9d96d468807e9d67a2aa0dc497ff5522f"
)

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)
Disposition: TypeAlias = Literal["USABLE", "REFERENCE_NOT_PRODUCED"]


class ReferenceArtifactError(ValueError):
    """The reference artifact violates a frozen integrity or semantic gate."""


@dataclass(frozen=True, slots=True)
class HistoricalAuthorityPaths:
    """Exact external historical files selected by the caller."""

    receipt: Path
    trajectory: Path
    input_bundle: Path


@dataclass(frozen=True, slots=True)
class SourcePath:
    """One current execution-bearing source file copied into the artifact."""

    logical_path: str
    path: Path


@dataclass(frozen=True, slots=True)
class RuntimeProvenance:
    """Current process/runtime identity captured before reconstruction."""

    argv: tuple[str, ...]
    cwd: str
    python_executable: str
    python_version: str
    platform: str
    numpy_version: str
    jax_version: str
    jaxlib_version: str
    simsopt_path: str
    simsopt_jax_path: str
    adapter_path: str
    simsopt_sha256: str
    simsopt_jax_sha256: str
    adapter_sha256: str
    native_extension_path: str
    native_extension_sha256: str
    python_executable_sha256: str
    effective_environment_sha256: str
    git_head: str
    tracked_diff_sha256: str
    repository_dirty: bool


@dataclass(frozen=True, slots=True)
class FileReference:
    relative_path: str
    sha256: str
    size_bytes: int

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ArrayReference:
    relative_path: str
    file_sha256: str
    content_sha256: str
    size_bytes: int
    dtype: str
    shape: tuple[int, ...]
    order: str

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "content_sha256": self.content_sha256,
            "dtype": self.dtype,
            "file_sha256": self.file_sha256,
            "order": self.order,
            "relative_path": self.relative_path,
            "shape": list(self.shape),
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ReferenceValidationResult:
    disposition: Disposition
    usable: bool
    failure_reasons: tuple[str, ...]
    artifact_sha256: str


class NativeReferenceRuntime(Protocol):
    bootstrap_state: np.ndarray
    fixed_first_base_current: float

    def reconstruct_native_reference(
        self,
        historical: HistoricalNativeParameters,
    ) -> NativeReferenceEvidence: ...


_OBSERVABLE_KEYS: Final = {
    "objective": "final:objective",
    "iota": "final:iota",
    "volume": "final:volume",
    "non_qs": "final:non_qs_ratio",
    "boozer_residual_value": "final:boozer_residual",
    "boozer_residual_rms": "final:boozer_residual_rms",
    "major_radius_penalty": "final:major_radius_penalty",
    "length_penalty": "final:length_penalty",
}


def canonical_json_bytes(value: JsonValue) -> bytes:
    """Encode the protocol's unique strict JSON representation."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def load_canonical_json_bytes(payload: bytes) -> JsonValue:
    """Decode strict canonical JSON while rejecting duplicate keys."""

    def object_from_pairs(
        pairs: list[tuple[str, JsonValue]],
    ) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceArtifactError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        decoded: JsonValue = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_from_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReferenceArtifactError("artifact is not UTF-8 JSON") from error
    if canonical_json_bytes(decoded) != payload:
        raise ReferenceArtifactError("JSON artifact is not canonical")
    return decoded


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_content_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(FP64_DTYPE))
    return _sha256(array.tobytes(order="C"))


def _mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ReferenceArtifactError(f"{name} must be a string-keyed object")
    return value


def _sequence(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ReferenceArtifactError(f"{name} must be an array")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReferenceArtifactError(f"{name} must be a nonempty string")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReferenceArtifactError(f"{name} must be an integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReferenceArtifactError(f"{name} must be a number")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ReferenceArtifactError(f"{name} must be finite")
    return scalar


def _safe_relative_path(value: str, name: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ReferenceArtifactError(f"{name} is not a canonical relative path")
    return path


def _reject_symlink_components(path: Path, name: str) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ReferenceArtifactError(f"{name} contains a symlink: {current}")


def _read_external_file(path: Path, expected_sha256: str, name: str) -> bytes:
    _reject_symlink_components(path, name)
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ReferenceArtifactError(f"{name} is not a regular file")
    payload = resolved.read_bytes()
    if _sha256(payload) != expected_sha256:
        raise ReferenceArtifactError(f"{name} SHA-256 mismatch")
    return payload


def _load_exact_npy(
    payload: bytes,
    *,
    dtype: str,
    shape: tuple[int, ...],
    name: str,
) -> np.ndarray:
    from io import BytesIO

    stream = BytesIO(payload)
    try:
        array = np.load(stream, allow_pickle=False)
    except (ValueError, EOFError) as error:
        raise ReferenceArtifactError(f"{name} is not a valid NPY array") from error
    if stream.read(1) != b"":
        raise ReferenceArtifactError(f"{name} has trailing bytes")
    if array.dtype.str != dtype or array.shape != shape:
        raise ReferenceArtifactError(f"{name} dtype or shape mismatch")
    if not array.flags.c_contiguous:
        raise ReferenceArtifactError(f"{name} must use C order")
    if not np.all(np.isfinite(array)):
        raise ReferenceArtifactError(f"{name} contains nonfinite values")
    return np.array(array, dtype=np.dtype(dtype), copy=True, order="C")


def _write_exclusive(path: Path, payload: bytes) -> FileReference:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o644,
    )
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return FileReference(
        relative_path="",
        sha256=_sha256(payload),
        size_bytes=len(payload),
    )


def _write_relative_file(
    root: Path,
    relative_path: str,
    payload: bytes,
) -> FileReference:
    relative = _safe_relative_path(relative_path, "output path")
    target = root.joinpath(*relative.parts)
    written = _write_exclusive(target, payload)
    return FileReference(
        relative_path=relative.as_posix(),
        sha256=written.sha256,
        size_bytes=written.size_bytes,
    )


def _npy_bytes(values: np.ndarray) -> bytes:
    from io import BytesIO

    array = np.asarray(values)
    if array.dtype != np.dtype(np.float64) or not np.all(np.isfinite(array)):
        raise ReferenceArtifactError("reference arrays must be finite float64")
    canonical = np.ascontiguousarray(array, dtype=np.dtype(FP64_DTYPE))
    stream = BytesIO()
    np.lib.format.write_array(stream, canonical, version=(2, 0), allow_pickle=False)
    return stream.getvalue()


def _write_array(root: Path, values: np.ndarray) -> ArrayReference:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(FP64_DTYPE))
    content_sha256 = _array_content_sha256(canonical)
    relative_path = f"arrays/{content_sha256}.npy"
    payload = _npy_bytes(canonical)
    file_reference = _write_relative_file(root, relative_path, payload)
    return ArrayReference(
        relative_path=relative_path,
        file_sha256=file_reference.sha256,
        content_sha256=content_sha256,
        size_bytes=file_reference.size_bytes,
        dtype=FP64_DTYPE,
        shape=canonical.shape,
        order="C",
    )


def _artifact_ref_payload(reference: FileReference) -> dict[str, JsonValue]:
    return reference.to_payload()


def _array_ref_payload(reference: ArrayReference) -> dict[str, JsonValue]:
    return reference.to_payload()


def _receipt_array_reference(
    receipt: Mapping[str, object],
    key: str,
) -> dict[str, object]:
    values = _mapping(receipt.get("values"), "historical receipt values")
    return _mapping(values.get(key), f"historical receipt {key}")


def _receipt_array(
    receipt_directory: Path,
    receipt: Mapping[str, object],
    key: str,
    *,
    expected_path: str | None = None,
    expected_sha256: str | None = None,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, bytes, str, str]:
    reference = _receipt_array_reference(receipt, key)
    relative_path = _string(reference.get("path"), f"{key} path")
    sha256 = _string(reference.get("sha256"), f"{key} SHA-256")
    dtype = _string(reference.get("dtype"), f"{key} dtype")
    order = _string(reference.get("order"), f"{key} order")
    shape_values = _sequence(reference.get("shape"), f"{key} shape")
    shape = tuple(_integer(item, f"{key} shape") for item in shape_values)
    if expected_path is not None and relative_path != expected_path:
        raise ReferenceArtifactError(f"{key} path differs from the frozen path")
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ReferenceArtifactError(f"{key} digest differs from the frozen digest")
    if dtype != FP64_DTYPE or order != "C" or shape != expected_shape:
        raise ReferenceArtifactError(f"{key} array contract mismatch")
    relative = _safe_relative_path(relative_path, f"{key} path")
    source_path = receipt_directory.joinpath(*relative.parts)
    payload = _read_external_file(source_path, sha256, key)
    array = _load_exact_npy(
        payload,
        dtype=FP64_DTYPE,
        shape=expected_shape,
        name=key,
    )
    return array, payload, relative_path, sha256


def _copy_named_bytes(
    root: Path,
    category: str,
    label: str,
    payload: bytes,
    suffix: str,
) -> FileReference:
    digest = _sha256(payload)
    relative_path = f"{category}/{label}-{digest}{suffix}"
    return _write_relative_file(root, relative_path, payload)


def _prepare_historical_authority(
    output_root: Path,
    paths: HistoricalAuthorityPaths,
) -> tuple[object, dict[str, JsonValue], np.ndarray]:
    receipt_payload = _read_external_file(
        paths.receipt,
        HISTORICAL_RECEIPT_SHA256,
        "historical receipt",
    )
    trajectory_payload = _read_external_file(
        paths.trajectory,
        HISTORICAL_TRAJECTORY_SHA256,
        "historical trajectory",
    )
    input_payload = _read_external_file(
        paths.input_bundle,
        HISTORICAL_INPUT_BUNDLE_SHA256,
        "historical input bundle",
    )
    receipt_value = load_canonical_json_bytes(receipt_payload)
    receipt = _mapping(receipt_value, "historical receipt")
    if (
        receipt.get("lane") != "native-cpu"
        or receipt.get("backend_mode") != "native_cpu"
        or receipt.get("precision") != "fp64"
        or receipt.get("scale") != "native_default"
        or receipt.get("nit") != 1000
        or receipt.get("normalized_status") != "budget_exhausted"
        or receipt.get("success") is not False
    ):
        raise ReferenceArtifactError("historical receipt identity/status mismatch")

    final_parameters, final_payload, final_path, _ = _receipt_array(
        paths.receipt.parent,
        receipt,
        "final:parameters",
        expected_path=HISTORICAL_FINAL_PARAMETER_PATH,
        expected_sha256=HISTORICAL_FINAL_PARAMETER_SHA256,
        expected_shape=(COIL_SIZE,),
    )
    bootstrap_coils, bootstrap_coil_payload, bootstrap_coil_path, _ = _receipt_array(
        paths.receipt.parent,
        receipt,
        "construction:coil_dofs",
        expected_path=HISTORICAL_BOOTSTRAP_COIL_PATH,
        expected_sha256=HISTORICAL_BOOTSTRAP_COIL_SHA256,
        expected_shape=(COIL_SIZE,),
    )
    _, bootstrap_surface_payload, bootstrap_surface_path, _ = _receipt_array(
        paths.receipt.parent,
        receipt,
        "construction:surface_dofs",
        expected_path=HISTORICAL_BOOTSTRAP_SURFACE_PATH,
        expected_sha256=HISTORICAL_BOOTSTRAP_SURFACE_SHA256,
        expected_shape=(253,),
    )

    trajectory_lines = trajectory_payload.decode("utf-8").splitlines()
    if len(trajectory_lines) != 1000:
        raise ReferenceArtifactError("historical trajectory must contain 1000 rows")
    final_trajectory = _mapping(
        json.loads(trajectory_lines[-1]),
        "historical terminal trajectory row",
    )
    if (
        final_trajectory.get("iteration") != 1000
        or _finite(final_trajectory.get("objective"), "historical trajectory objective")
        != 4.4822246533126125e-08
        or _finite(
            final_trajectory.get("wall_seconds_from_start"),
            "historical trajectory timestamp",
        )
        != 287.30421751597896
    ):
        raise ReferenceArtifactError("historical terminal trajectory row mismatch")

    input_document = _mapping(
        load_canonical_json_bytes(input_payload),
        "historical input bundle",
    )
    if (
        input_document.get("case_id")
        != "native-single-stage-boozer-vacuum-optimization"
        or input_document.get("scale") != "native_default"
    ):
        raise ReferenceArtifactError("historical input bundle identity mismatch")

    copied: dict[str, JsonValue] = {
        "receipt": _artifact_ref_payload(
            _copy_named_bytes(
                output_root, "authority", "lane-result", receipt_payload, ".json"
            )
        ),
        "trajectory": _artifact_ref_payload(
            _copy_named_bytes(
                output_root, "authority", "trajectory", trajectory_payload, ".jsonl"
            )
        ),
        "input_bundle": _artifact_ref_payload(
            _copy_named_bytes(
                output_root, "authority", "input-bundle", input_payload, ".json"
            )
        ),
        "final_parameters": _artifact_ref_payload(
            _copy_named_bytes(
                output_root, "authority", "final-parameters", final_payload, ".npy"
            )
        ),
        "bootstrap_coils": _artifact_ref_payload(
            _copy_named_bytes(
                output_root,
                "authority",
                "bootstrap-coils",
                bootstrap_coil_payload,
                ".npy",
            )
        ),
        "bootstrap_surface": _artifact_ref_payload(
            _copy_named_bytes(
                output_root,
                "authority",
                "bootstrap-surface",
                bootstrap_surface_payload,
                ".npy",
            )
        ),
    }

    input_root = paths.input_bundle.parent
    bundle, bundle_arrays = read_input_bundle(input_root)
    if not np.array_equal(bundle_arrays["coil_dofs"], bootstrap_coils):
        raise ReferenceArtifactError("receipt and input-bundle bootstrap coils differ")
    input_array_copies: dict[str, JsonValue] = {}
    for name, reference in sorted(bundle.arrays.items()):
        relative = _safe_relative_path(reference.path, f"input array {name}")
        source = input_root.joinpath(*relative.parts)
        payload = _read_external_file(source, reference.sha256, f"input array {name}")
        input_array_copies[name] = _artifact_ref_payload(
            _copy_named_bytes(
                output_root,
                "authority",
                f"input-{name}",
                payload,
                ".npy",
            )
        )
    copied["input_arrays"] = input_array_copies

    observable_values: dict[str, float] = {}
    observable_copies: dict[str, JsonValue] = {}
    for name, receipt_key in _OBSERVABLE_KEYS.items():
        array, payload, _, _ = _receipt_array(
            paths.receipt.parent,
            receipt,
            receipt_key,
            expected_shape=(1,),
        )
        observable_values[name] = float(array[0])
        observable_copies[name] = _artifact_ref_payload(
            _copy_named_bytes(
                output_root,
                "authority",
                name,
                payload,
                ".npy",
            )
        )
    copied["observable_arrays"] = observable_copies

    endpoint_document: dict[str, JsonValue] = {
        "endpoint": {
            "boozer_residual_rms": observable_values["boozer_residual_rms"],
            "boozer_residual_value": observable_values["boozer_residual_value"],
            "iota": observable_values["iota"],
            "length_penalty": observable_values["length_penalty"],
            "major_radius_penalty": observable_values["major_radius_penalty"],
            "non_qs": observable_values["non_qs"],
            "objective": observable_values["objective"],
            "parameters": final_parameters.tolist(),
            "volume": observable_values["volume"],
        }
    }
    endpoint_payload = canonical_json_bytes(endpoint_document)
    endpoint_copy = _write_relative_file(
        output_root,
        "authority/historical-endpoint-authority.json",
        endpoint_payload,
    )
    copied["historical_endpoint_authority"] = _artifact_ref_payload(endpoint_copy)
    metadata = HistoricalNativeParameterMetadata(
        source_sha256=endpoint_copy.sha256,
        parameter_path=("endpoint", "parameters"),
        parameter_little_endian_sha256=_array_content_sha256(final_parameters),
        parameter_dtype=FP64_DTYPE,
        parameter_shape=(COIL_SIZE,),
        observable_paths=HistoricalNativeObservablePaths(
            objective=("endpoint", "objective"),
            iota=("endpoint", "iota"),
            volume=("endpoint", "volume"),
            non_qs=("endpoint", "non_qs"),
            boozer_residual_value=("endpoint", "boozer_residual_value"),
            boozer_residual_rms=("endpoint", "boozer_residual_rms"),
            major_radius_penalty=("endpoint", "major_radius_penalty"),
            length_penalty=("endpoint", "length_penalty"),
        ),
    )
    historical = load_historical_native_parameters(endpoint_payload, metadata)
    authority_manifest: dict[str, JsonValue] = {
        "copied_files": copied,
        "final_parameter_original_path": final_path,
        "bootstrap_coil_original_path": bootstrap_coil_path,
        "bootstrap_surface_original_path": bootstrap_surface_path,
        "historical_constraints_satisfied": receipt.get("raw_status"),
        "constraints_satisfied_boolean_used_as_numerical_reference": False,
        "input_fingerprint": bundle.input_fingerprint,
        "configuration_fingerprint": bundle.configuration_fingerprint,
        "schema_version": AUTHORITY_MANIFEST_SCHEMA_VERSION,
    }
    return historical, authority_manifest, bootstrap_coils


def _copy_sources(
    output_root: Path,
    source_paths: Sequence[SourcePath],
) -> dict[str, JsonValue]:
    if (
        tuple(source.logical_path for source in source_paths)
        != REQUIRED_SOURCE_LOGICAL_PATHS
    ):
        raise ReferenceArtifactError(
            "source provenance set or order differs from contract"
        )
    entries: list[JsonValue] = []
    logical_paths: set[str] = set()
    for index, source in enumerate(source_paths):
        if source.logical_path in logical_paths:
            raise ReferenceArtifactError("source logical paths must be unique")
        logical_paths.add(source.logical_path)
        _safe_relative_path(source.logical_path, "source logical path")
        _reject_symlink_components(source.path, "source path")
        resolved = source.path.resolve(strict=True)
        if not resolved.is_file():
            raise ReferenceArtifactError("source provenance path is not a file")
        payload = resolved.read_bytes()
        copied = _copy_named_bytes(
            output_root,
            "source",
            f"{index:03d}",
            payload,
            resolved.suffix or ".bin",
        )
        entries.append(
            {
                "copied": copied.to_payload(),
                "logical_path": source.logical_path,
                "source_sha256": copied.sha256,
                "source_size_bytes": copied.size_bytes,
            }
        )
    return {
        "entries": entries,
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
    }


def _copy_runtime_bindings(
    output_root: Path,
    provenance: RuntimeProvenance,
) -> dict[str, JsonValue]:
    bindings = (
        (
            "python_executable",
            provenance.python_executable,
            provenance.python_executable_sha256,
        ),
        ("simsopt", provenance.simsopt_path, provenance.simsopt_sha256),
        ("simsopt_jax", provenance.simsopt_jax_path, provenance.simsopt_jax_sha256),
        ("adapter", provenance.adapter_path, provenance.adapter_sha256),
        (
            "native_extension",
            provenance.native_extension_path,
            provenance.native_extension_sha256,
        ),
    )
    copied: dict[str, JsonValue] = {}
    for index, (name, raw_path, declared_sha256) in enumerate(bindings):
        path = Path(raw_path)
        if not path.is_absolute():
            raise ReferenceArtifactError(
                f"runtime binding path is not absolute: {name}"
            )
        _reject_symlink_components(path, f"runtime binding {name}")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ReferenceArtifactError(f"runtime binding is not a file: {name}")
        payload = resolved.read_bytes()
        if _sha256(payload) != declared_sha256:
            raise ReferenceArtifactError(f"runtime binding digest differs: {name}")
        reference = _copy_named_bytes(
            output_root,
            "runtime-binding",
            f"{index:03d}-{name}",
            payload,
            resolved.suffix or ".bin",
        )
        copied[name] = reference.to_payload()
    return copied


def _runtime_payload(
    provenance: RuntimeProvenance,
    bindings: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    payload = asdict(provenance)
    return {
        "argv": list(provenance.argv),
        "cwd": provenance.cwd,
        "python_executable": provenance.python_executable,
        "python_version": provenance.python_version,
        "platform": provenance.platform,
        "numpy_version": provenance.numpy_version,
        "jax_version": provenance.jax_version,
        "jaxlib_version": provenance.jaxlib_version,
        "simsopt_path": provenance.simsopt_path,
        "simsopt_jax_path": provenance.simsopt_jax_path,
        "adapter_path": provenance.adapter_path,
        "simsopt_sha256": provenance.simsopt_sha256,
        "simsopt_jax_sha256": provenance.simsopt_jax_sha256,
        "adapter_sha256": provenance.adapter_sha256,
        "native_extension_path": provenance.native_extension_path,
        "native_extension_sha256": provenance.native_extension_sha256,
        "python_executable_sha256": provenance.python_executable_sha256,
        "effective_environment_sha256": provenance.effective_environment_sha256,
        "git_head": provenance.git_head,
        "tracked_diff_sha256": provenance.tracked_diff_sha256,
        "repository_dirty": bool(payload["repository_dirty"]),
        "bindings": bindings,
        "schema_version": RUNTIME_SCHEMA_VERSION,
    }


def _step_payload(step: object) -> dict[str, JsonValue]:
    values = asdict(step)
    return {
        "coil_little_endian_sha256": str(values["coil_little_endian_sha256"]),
        "index": int(values["index"]),
        "newton_iterations": int(values["newton_iterations"]),
        "predecessor_index": (
            None
            if values["predecessor_index"] is None
            else int(values["predecessor_index"])
        ),
        "residual_infinity_norm": float(values["residual_infinity_norm"]),
        "residual_l2": float(values["residual_l2"]),
        "root_little_endian_sha256": str(values["root_little_endian_sha256"]),
        "scaled_boozer_infinity_norm": float(values["scaled_boozer_infinity_norm"]),
        "seed_root_little_endian_sha256": str(values["seed_root_little_endian_sha256"]),
        "segment_count": int(values["segment_count"]),
    }


def _observable_comparisons(
    evidence: NativeReferenceEvidence,
) -> list[JsonValue]:
    observed = {
        "objective": evidence.endpoint.objective,
        "iota": evidence.endpoint.observables.iota,
        "volume": evidence.endpoint.observables.volume,
        "non_qs": evidence.endpoint.objective_terms.non_qs,
        "boozer_residual_value": evidence.endpoint.objective_terms.residual,
        "boozer_residual_rms": evidence.endpoint.observables.boozer_residual_rms,
        "major_radius_penalty": evidence.endpoint.objective_terms.major_radius,
        "length_penalty": evidence.endpoint.objective_terms.length,
    }
    sealed = asdict(evidence.historical_input.sealed_observables)
    comparisons: list[JsonValue] = []
    for name in _OBSERVABLE_KEYS:
        actual = float(observed[name])
        expected = float(sealed[name])
        difference = abs(actual - expected)
        tolerance = OBSERVABLE_ATOL + OBSERVABLE_RTOL * abs(expected)
        comparisons.append(
            {
                "absolute_difference": difference,
                "atol": OBSERVABLE_ATOL,
                "name": name,
                "observed": actual,
                "passed": difference <= tolerance,
                "reference": expected,
                "rtol": OBSERVABLE_RTOL,
                "tolerance": tolerance,
            }
        )
    return comparisons


def _evidence_payload(
    output_root: Path,
    evidence: NativeReferenceEvidence,
) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    if not np.array_equal(
        evidence.state[:COIL_SIZE], evidence.historical_input.parameters
    ):
        raise ReferenceArtifactError("reference state does not contain final coils")
    state_reference = _write_array(output_root, evidence.state)
    equality_reference = _write_array(output_root, evidence.endpoint.raw_equalities)
    coarse_reference = _write_array(output_root, evidence.coarse_path.roots)
    refined_reference = _write_array(output_root, evidence.refined_path.roots)
    diagnostics: dict[str, JsonValue] = {
        "coarse_steps": [_step_payload(step) for step in evidence.coarse_path.steps],
        "refined_steps": [_step_payload(step) for step in evidence.refined_path.steps],
        "schema_version": DIAGNOSTICS_SCHEMA_VERSION,
    }
    comparisons = _observable_comparisons(evidence)
    objective_terms = asdict(evidence.endpoint.objective_terms)
    observables = asdict(evidence.endpoint.observables)
    payload: dict[str, JsonValue] = {
        "arrays": {
            "coarse_roots": coarse_reference.to_payload(),
            "raw_equalities": equality_reference.to_payload(),
            "refined_roots": refined_reference.to_payload(),
            "state": state_reference.to_payload(),
        },
        "common_knot_root_infinity_difference": evidence.common_knot_root_infinity_difference,
        "comparisons": comparisons,
        "constraints_satisfied_boolean_used_as_numerical_reference": False,
        "endpoint_all_finite": evidence.endpoint.all_finite,
        "equality_order": [
            "masked_boozer_residual[254]",
            "signed_volume_minus_target[1]",
        ],
        "fixed_first_base_current": evidence.fixed_first_base_current,
        "historical_parameter_little_endian_sha256": (
            evidence.historical_input.parameter_little_endian_sha256
        ),
        "layout_order": [
            "coil_dofs[461]",
            "surface_dofs[253]",
            "iota[1]",
            "G[1]",
        ],
        "objective": evidence.endpoint.objective,
        "objective_terms": {
            key: float(value) for key, value in objective_terms.items()
        },
        "observables": {key: float(value) for key, value in observables.items()},
        "sealed_observables_match": evidence.sealed_observables_match,
        "state_little_endian_sha256": evidence.endpoint.state_little_endian_sha256,
        "usable": evidence.usable,
    }
    return payload, diagnostics


def _seal_tree(root: Path) -> None:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    directories = sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for path in files:
        path.chmod(0o444)
    for path in directories:
        path.chmod(0o555)
    root.chmod(0o555)


def _write_artifact_manifest(root: Path) -> FileReference:
    entries: list[JsonValue] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative_path = path.relative_to(root).as_posix()
        if relative_path == ARTIFACT_MANIFEST_FILENAME:
            raise ReferenceArtifactError("artifact manifest already exists")
        payload = path.read_bytes()
        entries.append(
            {
                "relative_path": relative_path,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    manifest: dict[str, JsonValue] = {
        "entries": entries,
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
    }
    return _write_relative_file(
        root,
        ARTIFACT_MANIFEST_FILENAME,
        canonical_json_bytes(manifest),
    )


def _atomic_publish_no_replace(source: Path, destination: Path) -> None:
    renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(
            error_number,
            os.strerror(error_number),
            destination,
        )
    raise OSError(error_number, os.strerror(error_number), destination)


def produce_native_equivalent_reference(
    *,
    output_root: Path,
    historical_paths: HistoricalAuthorityPaths,
    runtime: NativeReferenceRuntime,
    runtime_provenance: RuntimeProvenance,
    source_paths: Sequence[SourcePath],
) -> ReferenceValidationResult:
    """Atomically produce and validate one immutable reference artifact."""

    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"reference output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_root.name}.partial-",
        dir=output_root.parent,
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        historical, authority_manifest, bootstrap_coils = _prepare_historical_authority(
            temporary_root, historical_paths
        )
        if not np.array_equal(runtime.bootstrap_state[:COIL_SIZE], bootstrap_coils):
            raise ReferenceArtifactError(
                "runtime bootstrap coils differ from historical authority"
            )
        authority_reference = _write_relative_file(
            temporary_root,
            "manifests/authority.json",
            canonical_json_bytes(authority_manifest),
        )
        source_manifest = _copy_sources(temporary_root, source_paths)
        source_reference = _write_relative_file(
            temporary_root,
            "manifests/source.json",
            canonical_json_bytes(source_manifest),
        )
        runtime_bindings = _copy_runtime_bindings(
            temporary_root,
            runtime_provenance,
        )
        runtime_reference = _write_relative_file(
            temporary_root,
            "manifests/runtime.json",
            canonical_json_bytes(
                _runtime_payload(runtime_provenance, runtime_bindings)
            ),
        )

        reconstruction_failure: str | None = None
        try:
            evidence = runtime.reconstruct_native_reference(historical)
        except NativeEndpointError as error:
            evidence = None
            reconstruction_failure = f"{type(error).__name__}: {error}"

        evidence_payload: dict[str, JsonValue] | None
        diagnostics_reference: FileReference | None
        if evidence is None:
            evidence_payload = None
            diagnostics_reference = None
            disposition: Disposition = REFERENCE_NOT_PRODUCED
        else:
            evidence_payload, diagnostics = _evidence_payload(
                temporary_root,
                evidence,
            )
            diagnostics_reference = _write_relative_file(
                temporary_root,
                "manifests/diagnostics.json",
                canonical_json_bytes(diagnostics),
            )
            if evidence.usable:
                disposition = USABLE
            else:
                disposition = REFERENCE_NOT_PRODUCED
                reconstruction_failure = "NativeReferenceEvidence.usable=false"

        document: dict[str, JsonValue] = {
            "authority_manifest": authority_reference.to_payload(),
            "diagnostics": (
                None
                if diagnostics_reference is None
                else diagnostics_reference.to_payload()
            ),
            "disposition": disposition,
            "evidence": evidence_payload,
            "policy": {
                "common_knot_tolerance": COMMON_KNOT_TOLERANCE,
                "coarse_segment_count": COARSE_SEGMENT_COUNT,
                "equality_size": EQUALITY_SIZE,
                "exact_newton_maximum_iterations": EXACT_NEWTON_MAXIMUM_ITERATIONS,
                "exact_newton_tolerance": EXACT_NEWTON_TOLERANCE,
                "observable_atol": OBSERVABLE_ATOL,
                "observable_rtol": OBSERVABLE_RTOL,
                "refined_segment_count": REFINED_SEGMENT_COUNT,
                "ssot_sha256": SSOT_SHA256,
                "state_size": STATE_SIZE,
            },
            "reconstruction_failure": reconstruction_failure,
            "runtime_provenance": runtime_reference.to_payload(),
            "schema_version": SCHEMA_VERSION,
            "source_manifest": source_reference.to_payload(),
            "summary_usable": disposition == USABLE,
        }
        _write_relative_file(
            temporary_root,
            REFERENCE_FILENAME,
            canonical_json_bytes(document),
        )
        artifact_manifest = _write_artifact_manifest(temporary_root)
        _seal_tree(temporary_root)
        validation = validate_native_equivalent_reference(temporary_root)
        _atomic_publish_no_replace(temporary_root, output_root)
        return ReferenceValidationResult(
            disposition=validation.disposition,
            usable=validation.usable,
            failure_reasons=validation.failure_reasons,
            artifact_sha256=artifact_manifest.sha256,
        )


def _artifact_file(root: Path, reference_value: object, name: str) -> bytes:
    reference = _mapping(reference_value, name)
    relative_path = _string(reference.get("relative_path"), f"{name} path")
    expected_sha256 = _string(reference.get("sha256"), f"{name} SHA-256")
    expected_size = _integer(reference.get("size_bytes"), f"{name} size")
    relative = _safe_relative_path(relative_path, f"{name} path")
    path = root.joinpath(*relative.parts)
    _reject_symlink_components(path, name)
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root.resolve(strict=True)) or not resolved.is_file():
        raise ReferenceArtifactError(f"{name} escapes the artifact root")
    payload = resolved.read_bytes()
    if len(payload) != expected_size or _sha256(payload) != expected_sha256:
        raise ReferenceArtifactError(f"{name} integrity mismatch")
    return payload


def _array_from_reference(
    root: Path,
    reference_value: object,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    reference = _mapping(reference_value, name)
    relative_path = _string(reference.get("relative_path"), f"{name} path")
    expected_file_sha256 = _string(reference.get("file_sha256"), f"{name} file SHA-256")
    expected_content_sha256 = _string(
        reference.get("content_sha256"), f"{name} content SHA-256"
    )
    size = _integer(reference.get("size_bytes"), f"{name} size")
    dtype = _string(reference.get("dtype"), f"{name} dtype")
    order = _string(reference.get("order"), f"{name} order")
    shape = tuple(
        _integer(item, f"{name} shape")
        for item in _sequence(reference.get("shape"), f"{name} shape")
    )
    if dtype != FP64_DTYPE or order != "C" or shape != expected_shape:
        raise ReferenceArtifactError(f"{name} array contract mismatch")
    relative = _safe_relative_path(relative_path, f"{name} path")
    if relative_path != f"arrays/{expected_content_sha256}.npy":
        raise ReferenceArtifactError(f"{name} is not content-addressed")
    path = root.joinpath(*relative.parts)
    _reject_symlink_components(path, name)
    payload = path.resolve(strict=True).read_bytes()
    if len(payload) != size or _sha256(payload) != expected_file_sha256:
        raise ReferenceArtifactError(f"{name} file integrity mismatch")
    array = _load_exact_npy(payload, dtype=dtype, shape=shape, name=name)
    if _array_content_sha256(array) != expected_content_sha256:
        raise ReferenceArtifactError(f"{name} content integrity mismatch")
    return array


def _validate_read_only_tree(root: Path) -> None:
    _reject_symlink_components(root, "artifact root")
    resolved = root.resolve(strict=True)
    if not resolved.is_dir() or stat.S_IMODE(resolved.stat().st_mode) & 0o222:
        raise ReferenceArtifactError("artifact root must be a read-only directory")
    for path in resolved.rglob("*"):
        if path.is_symlink():
            raise ReferenceArtifactError("artifact tree must not contain symlinks")
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o222:
            raise ReferenceArtifactError(f"artifact path is writable: {path}")
        if not path.is_file() and not path.is_dir():
            raise ReferenceArtifactError(f"artifact path is not regular: {path}")


def _validate_artifact_manifest(root: Path) -> str:
    manifest_path = root / ARTIFACT_MANIFEST_FILENAME
    _reject_symlink_components(manifest_path, "artifact manifest")
    payload = manifest_path.resolve(strict=True).read_bytes()
    manifest = _mapping(
        load_canonical_json_bytes(payload),
        "artifact manifest",
    )
    if manifest.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        raise ReferenceArtifactError("artifact manifest schema mismatch")
    expected_paths: set[str] = {ARTIFACT_MANIFEST_FILENAME}
    for entry_value in _sequence(manifest.get("entries"), "artifact entries"):
        entry = _mapping(entry_value, "artifact entry")
        relative_path = _string(entry.get("relative_path"), "artifact entry path")
        relative = _safe_relative_path(relative_path, "artifact entry path")
        if relative_path in expected_paths:
            raise ReferenceArtifactError("artifact manifest contains duplicate path")
        expected_paths.add(relative_path)
        path = root.joinpath(*relative.parts)
        _reject_symlink_components(path, "artifact entry")
        entry_payload = path.resolve(strict=True).read_bytes()
        if entry.get("size_bytes") != len(entry_payload) or entry.get(
            "sha256"
        ) != _sha256(entry_payload):
            raise ReferenceArtifactError("artifact manifest entry mismatch")
    observed_paths = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if observed_paths != expected_paths:
        raise ReferenceArtifactError("artifact contains unreferenced or missing files")
    return _sha256(payload)


def _validate_copied_manifest_files(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    copied = _mapping(manifest.get("copied_files"), "copied authority files")
    for name, value in copied.items():
        if name in ("input_arrays", "observable_arrays"):
            nested = _mapping(value, f"copied authority {name}")
            for nested_name, nested_reference in nested.items():
                _artifact_file(root, nested_reference, f"{name}.{nested_name}")
        else:
            _artifact_file(root, value, f"copied authority {name}")


def _copied_reference(
    copied: Mapping[str, object],
    name: str,
) -> dict[str, object]:
    return _mapping(copied.get(name), f"copied authority {name}")


def _validate_historical_authority_semantics(
    root: Path,
    manifest: Mapping[str, object],
) -> None:
    copied = _mapping(manifest.get("copied_files"), "copied authority files")
    required_digests = {
        "receipt": HISTORICAL_RECEIPT_SHA256,
        "trajectory": HISTORICAL_TRAJECTORY_SHA256,
        "input_bundle": HISTORICAL_INPUT_BUNDLE_SHA256,
        "final_parameters": HISTORICAL_FINAL_PARAMETER_SHA256,
        "bootstrap_coils": HISTORICAL_BOOTSTRAP_COIL_SHA256,
        "bootstrap_surface": HISTORICAL_BOOTSTRAP_SURFACE_SHA256,
    }
    for name, expected_sha256 in required_digests.items():
        reference = _copied_reference(copied, name)
        if reference.get("sha256") != expected_sha256:
            raise ReferenceArtifactError(f"frozen authority digest differs: {name}")
    if (
        manifest.get("final_parameter_original_path") != HISTORICAL_FINAL_PARAMETER_PATH
        or manifest.get("bootstrap_coil_original_path")
        != HISTORICAL_BOOTSTRAP_COIL_PATH
        or manifest.get("bootstrap_surface_original_path")
        != HISTORICAL_BOOTSTRAP_SURFACE_PATH
    ):
        raise ReferenceArtifactError("frozen historical array paths differ")

    receipt = _mapping(
        load_canonical_json_bytes(
            _artifact_file(root, copied.get("receipt"), "copied receipt")
        ),
        "copied receipt",
    )
    if (
        receipt.get("nit") != 1000
        or receipt.get("normalized_status") != "budget_exhausted"
        or receipt.get("success") is not False
    ):
        raise ReferenceArtifactError("copied historical receipt status differs")
    load_canonical_json_bytes(
        _artifact_file(root, copied.get("input_bundle"), "copied input bundle")
    )
    trajectory = _artifact_file(
        root,
        copied.get("trajectory"),
        "copied trajectory",
    ).decode("utf-8")
    rows = trajectory.splitlines()
    if len(rows) != 1000:
        raise ReferenceArtifactError("copied historical trajectory row count differs")
    terminal = _mapping(json.loads(rows[-1]), "copied terminal trajectory row")
    if (
        terminal.get("iteration") != 1000
        or terminal.get("objective") != 4.4822246533126125e-08
        or terminal.get("wall_seconds_from_start") != 287.30421751597896
    ):
        raise ReferenceArtifactError("copied terminal trajectory row differs")

    final_parameters = _load_exact_npy(
        _artifact_file(
            root,
            copied.get("final_parameters"),
            "copied final parameters",
        ),
        dtype=FP64_DTYPE,
        shape=(COIL_SIZE,),
        name="copied final parameters",
    )
    observable_references = _mapping(
        copied.get("observable_arrays"),
        "copied observable arrays",
    )
    observable_values = {
        name: float(
            _load_exact_npy(
                _artifact_file(
                    root,
                    observable_references.get(name),
                    f"copied observable {name}",
                ),
                dtype=FP64_DTYPE,
                shape=(1,),
                name=f"copied observable {name}",
            )[0]
        )
        for name in _OBSERVABLE_KEYS
    }
    endpoint_authority = _mapping(
        load_canonical_json_bytes(
            _artifact_file(
                root,
                copied.get("historical_endpoint_authority"),
                "historical endpoint authority",
            )
        ),
        "historical endpoint authority",
    )
    endpoint = _mapping(endpoint_authority.get("endpoint"), "authority endpoint")
    parameters = _sequence(endpoint.get("parameters"), "authority parameters")
    if parameters != final_parameters.tolist():
        raise ReferenceArtifactError("authority parameters differ from copied NPY")
    for name, expected in observable_values.items():
        if endpoint.get(name) != expected:
            raise ReferenceArtifactError(
                f"authority observable differs from copied NPY: {name}"
            )


def _diagnostic_reasons(
    diagnostics: Mapping[str, object],
    coarse_roots: np.ndarray,
    refined_roots: np.ndarray,
    initial_parameters: np.ndarray,
    final_parameters: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    if diagnostics.get("schema_version") != DIAGNOSTICS_SCHEMA_VERSION:
        reasons.append("DIAGNOSTICS_SCHEMA")
    for label, raw_steps, roots, segment_count in (
        (
            "coarse",
            diagnostics.get("coarse_steps"),
            coarse_roots,
            COARSE_SEGMENT_COUNT,
        ),
        (
            "refined",
            diagnostics.get("refined_steps"),
            refined_roots,
            REFINED_SEGMENT_COUNT,
        ),
    ):
        steps = _sequence(raw_steps, f"{label} steps")
        if len(steps) != segment_count + 1:
            reasons.append(f"{label.upper()}_STEP_COUNT")
            continue
        for index, raw_step in enumerate(steps):
            step = _mapping(raw_step, f"{label} step {index}")
            if index == 0:
                expected_parameters = initial_parameters
                expected_seed = roots[0]
            elif index == segment_count:
                expected_parameters = final_parameters
                expected_seed = roots[index - 1]
            else:
                fraction = np.float64(index) / np.float64(segment_count)
                expected_parameters = initial_parameters + fraction * (
                    final_parameters - initial_parameters
                )
                expected_seed = roots[index - 1]
            if (
                step.get("segment_count") != segment_count
                or step.get("index") != index
                or step.get("predecessor_index") != (None if index == 0 else index - 1)
                or step.get("coil_little_endian_sha256")
                != _array_content_sha256(expected_parameters)
                or step.get("seed_root_little_endian_sha256")
                != _array_content_sha256(expected_seed)
                or step.get("root_little_endian_sha256")
                != _array_content_sha256(roots[index])
            ):
                reasons.append(f"{label.upper()}_STEP_IDENTITY")
                break
            if (
                _integer(step.get("newton_iterations"), "Newton iterations") < 0
                or _integer(step.get("newton_iterations"), "Newton iterations")
                > EXACT_NEWTON_MAXIMUM_ITERATIONS
                or _finite(step.get("residual_l2"), "residual l2") < 0.0
                or _finite(step.get("residual_infinity_norm"), "residual infinity")
                < 0.0
                or _finite(
                    step.get("scaled_boozer_infinity_norm"),
                    "scaled Boozer infinity",
                )
                > 1.0e-10
            ):
                reasons.append(f"{label.upper()}_STEP_DIAGNOSTICS")
                break
    return reasons


def _is_lower_hex(value: object, length: int) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_runtime_provenance(
    root: Path,
    runtime: Mapping[str, object],
    source: Mapping[str, object],
) -> None:
    required_strings = (
        "cwd",
        "python_executable",
        "python_version",
        "platform",
        "numpy_version",
        "jax_version",
        "jaxlib_version",
        "simsopt_path",
        "simsopt_jax_path",
        "adapter_path",
        "native_extension_path",
    )
    for name in required_strings:
        _string(runtime.get(name), f"runtime {name}")
    argv = _sequence(runtime.get("argv"), "runtime argv")
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ReferenceArtifactError("runtime argv must contain nonempty strings")
    for name in (
        "tracked_diff_sha256",
        "native_extension_sha256",
        "python_executable_sha256",
        "simsopt_sha256",
        "simsopt_jax_sha256",
        "adapter_sha256",
        "effective_environment_sha256",
    ):
        if not _is_lower_hex(runtime.get(name), 64):
            raise ReferenceArtifactError(f"runtime {name} must be SHA-256")
    if not _is_lower_hex(runtime.get("git_head"), 40):
        raise ReferenceArtifactError("runtime git_head must be a Git object ID")
    if not isinstance(runtime.get("repository_dirty"), bool):
        raise ReferenceArtifactError("runtime repository_dirty must be Boolean")
    bindings = _mapping(runtime.get("bindings"), "runtime bindings")
    expected_binding_digests = {
        "python_executable": runtime.get("python_executable_sha256"),
        "simsopt": runtime.get("simsopt_sha256"),
        "simsopt_jax": runtime.get("simsopt_jax_sha256"),
        "adapter": runtime.get("adapter_sha256"),
        "native_extension": runtime.get("native_extension_sha256"),
    }
    if set(bindings) != set(expected_binding_digests):
        raise ReferenceArtifactError("runtime binding set differs")
    for name, expected_digest in expected_binding_digests.items():
        binding_payload = _artifact_file(root, bindings.get(name), f"runtime {name}")
        if _sha256(binding_payload) != expected_digest:
            raise ReferenceArtifactError(f"runtime binding mismatch: {name}")
    entries = _sequence(source.get("entries"), "source entries")
    logical_entries = {
        _string(
            _mapping(value, "source entry").get("logical_path"), "logical path"
        ): _mapping(value, "source entry")
        for value in entries
    }
    adapter_entry = logical_entries[
        "src/simsopt_jax_adapters/geo/single_stage_native_endpoint.py"
    ]
    if adapter_entry.get("source_sha256") != runtime.get("adapter_sha256"):
        raise ReferenceArtifactError("adapter import is not bound to copied source")


def _derive_usable_reasons(
    root: Path,
    document: Mapping[str, object],
) -> list[str]:
    reasons: list[str] = []
    policy = _mapping(document.get("policy"), "reference policy")
    expected_policy = {
        "common_knot_tolerance": COMMON_KNOT_TOLERANCE,
        "coarse_segment_count": COARSE_SEGMENT_COUNT,
        "equality_size": EQUALITY_SIZE,
        "exact_newton_maximum_iterations": EXACT_NEWTON_MAXIMUM_ITERATIONS,
        "exact_newton_tolerance": EXACT_NEWTON_TOLERANCE,
        "observable_atol": OBSERVABLE_ATOL,
        "observable_rtol": OBSERVABLE_RTOL,
        "refined_segment_count": REFINED_SEGMENT_COUNT,
        "ssot_sha256": SSOT_SHA256,
        "state_size": STATE_SIZE,
    }
    if policy != expected_policy:
        reasons.append("POLICY_IDENTITY")
    evidence_value = document.get("evidence")
    if evidence_value is None:
        return [*reasons, "REFERENCE_EVIDENCE_MISSING"]
    evidence = _mapping(evidence_value, "reference evidence")
    arrays = _mapping(evidence.get("arrays"), "reference arrays")
    state = _array_from_reference(root, arrays.get("state"), "state", (STATE_SIZE,))
    equalities = _array_from_reference(
        root,
        arrays.get("raw_equalities"),
        "raw equalities",
        (EQUALITY_SIZE,),
    )
    coarse = _array_from_reference(
        root,
        arrays.get("coarse_roots"),
        "coarse roots",
        COARSE_ROOT_SHAPE,
    )
    refined = _array_from_reference(
        root,
        arrays.get("refined_roots"),
        "refined roots",
        REFINED_ROOT_SHAPE,
    )
    if not np.array_equal(state[COIL_SIZE:], coarse[-1]):
        reasons.append("TERMINAL_ROOT_STATE")
    common_difference = float(np.max(np.abs(coarse - refined[::2])))
    if common_difference > COMMON_KNOT_TOLERANCE or not math.isclose(
        common_difference,
        _finite(
            evidence.get("common_knot_root_infinity_difference"),
            "reported common-knot difference",
        ),
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        reasons.append("COMMON_KNOT_REFINEMENT")
    if evidence.get("layout_order") != [
        "coil_dofs[461]",
        "surface_dofs[253]",
        "iota[1]",
        "G[1]",
    ]:
        reasons.append("STATE_LAYOUT_ORDER")
    if evidence.get("equality_order") != [
        "masked_boozer_residual[254]",
        "signed_volume_minus_target[1]",
    ]:
        reasons.append("EQUALITY_ORDER")
    if (
        evidence.get("constraints_satisfied_boolean_used_as_numerical_reference")
        is not False
    ):
        reasons.append("BOOLEAN_FEASIBILITY_SUBSTITUTION")
    if evidence.get("usable") is not True:
        reasons.append("ADAPTER_TYPED_UNUSABLE")
    if evidence.get("endpoint_all_finite") is not True:
        reasons.append("ENDPOINT_NONFINITE")
    if evidence.get("sealed_observables_match") is not True:
        reasons.append("SEALED_OBSERVABLE_MISMATCH")
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(equalities)):
        reasons.append("NONFINITE_REFERENCE")
    if evidence.get("state_little_endian_sha256") != _array_content_sha256(state):
        reasons.append("STATE_IDENTITY")
    objective_terms = _mapping(evidence.get("objective_terms"), "objective terms")
    expected_term_names = {"non_qs", "residual", "iota", "major_radius", "length"}
    if set(objective_terms) != expected_term_names:
        reasons.append("OBJECTIVE_TERM_LEDGER")
    else:
        term_sum = sum(
            _finite(objective_terms[name], name) for name in expected_term_names
        )
        if not math.isclose(
            term_sum,
            _finite(evidence.get("objective"), "objective"),
            rel_tol=0.0,
            abs_tol=1.0e-18,
        ):
            reasons.append("OBJECTIVE_RECONSTRUCTION")
    observables = _mapping(evidence.get("observables"), "observables")
    expected_observables = {
        "iota",
        "G",
        "volume",
        "major_radius",
        "total_length",
        "non_qs_ratio",
        "boozer_residual_value",
        "boozer_residual_rms",
        "fixed_first_base_current",
    }
    if set(observables) != expected_observables:
        reasons.append("OBSERVABLE_LEDGER")
    elif _finite(
        evidence.get("fixed_first_base_current"),
        "fixed first current",
    ) != _finite(observables["fixed_first_base_current"], "observable current"):
        reasons.append("FIXED_CURRENT")
    comparisons = _sequence(evidence.get("comparisons"), "observable comparisons")
    if len(comparisons) != len(_OBSERVABLE_KEYS):
        reasons.append("OBSERVABLE_COMPARISON_COUNT")
    else:
        names: set[str] = set()
        for comparison_value in comparisons:
            comparison = _mapping(comparison_value, "observable comparison")
            name = _string(comparison.get("name"), "comparison name")
            names.add(name)
            observed = _finite(comparison.get("observed"), f"{name} observed")
            reference = _finite(comparison.get("reference"), f"{name} reference")
            difference = abs(observed - reference)
            tolerance = OBSERVABLE_ATOL + OBSERVABLE_RTOL * abs(reference)
            if (
                name not in _OBSERVABLE_KEYS
                or comparison.get("rtol") != OBSERVABLE_RTOL
                or comparison.get("atol") != OBSERVABLE_ATOL
                or not math.isclose(
                    _finite(comparison.get("absolute_difference"), "difference"),
                    difference,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or not math.isclose(
                    _finite(comparison.get("tolerance"), "tolerance"),
                    tolerance,
                    rel_tol=0.0,
                    abs_tol=0.0,
                )
                or difference > tolerance
            ):
                reasons.append("SEALED_OBSERVABLE_PARITY")
                break
        if names != set(_OBSERVABLE_KEYS):
            reasons.append("OBSERVABLE_COMPARISON_NAMES")
    authority_payload = _artifact_file(
        root,
        document.get("authority_manifest"),
        "authority manifest",
    )
    authority = _mapping(load_canonical_json_bytes(authority_payload), "authority")
    copied = _mapping(authority.get("copied_files"), "copied authority files")
    initial_parameters = _load_exact_npy(
        _artifact_file(root, copied.get("bootstrap_coils"), "bootstrap coils"),
        dtype=FP64_DTYPE,
        shape=(COIL_SIZE,),
        name="bootstrap coils",
    )
    diagnostics_payload = _artifact_file(
        root,
        document.get("diagnostics"),
        "diagnostics",
    )
    diagnostics = _mapping(
        load_canonical_json_bytes(diagnostics_payload),
        "diagnostics",
    )
    reasons.extend(
        _diagnostic_reasons(
            diagnostics,
            coarse,
            refined,
            initial_parameters,
            state[:COIL_SIZE],
        )
    )
    return reasons


def validate_native_equivalent_reference(
    artifact_root: Path,
) -> ReferenceValidationResult:
    """Independently validate and semantically adjudicate a sealed artifact."""

    _validate_read_only_tree(artifact_root)
    artifact_sha256 = _validate_artifact_manifest(artifact_root)
    reference_path = artifact_root / REFERENCE_FILENAME
    reference_payload = reference_path.read_bytes()
    document = _mapping(
        load_canonical_json_bytes(reference_payload),
        "reference document",
    )
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ReferenceArtifactError("reference schema mismatch")
    authority_payload = _artifact_file(
        artifact_root,
        document.get("authority_manifest"),
        "authority manifest",
    )
    authority = _mapping(
        load_canonical_json_bytes(authority_payload),
        "authority manifest",
    )
    if authority.get("schema_version") != AUTHORITY_MANIFEST_SCHEMA_VERSION:
        raise ReferenceArtifactError("authority manifest schema mismatch")
    if (
        authority.get("constraints_satisfied_boolean_used_as_numerical_reference")
        is not False
    ):
        raise ReferenceArtifactError("historical Boolean substituted for feasibility")
    _validate_copied_manifest_files(artifact_root, authority)
    _validate_historical_authority_semantics(artifact_root, authority)
    source_payload = _artifact_file(
        artifact_root,
        document.get("source_manifest"),
        "source manifest",
    )
    source = _mapping(load_canonical_json_bytes(source_payload), "source manifest")
    if source.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise ReferenceArtifactError("source manifest schema mismatch")
    source_entries = _sequence(source.get("entries"), "source entries")
    if (
        tuple(
            _string(_mapping(value, "source entry").get("logical_path"), "logical path")
            for value in source_entries
        )
        != REQUIRED_SOURCE_LOGICAL_PATHS
    ):
        raise ReferenceArtifactError("source manifest set or order differs")
    for entry_value in source_entries:
        entry = _mapping(entry_value, "source entry")
        copied_payload = _artifact_file(
            artifact_root,
            entry.get("copied"),
            "copied source",
        )
        if entry.get("source_sha256") != _sha256(copied_payload) or entry.get(
            "source_size_bytes"
        ) != len(copied_payload):
            raise ReferenceArtifactError("source manifest entry mismatch")
    runtime_payload = _artifact_file(
        artifact_root,
        document.get("runtime_provenance"),
        "runtime provenance",
    )
    runtime = _mapping(
        load_canonical_json_bytes(runtime_payload),
        "runtime provenance",
    )
    if runtime.get("schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ReferenceArtifactError("runtime provenance schema mismatch")
    _validate_runtime_provenance(artifact_root, runtime, source)

    reasons = _derive_usable_reasons(artifact_root, document)
    usable = not reasons
    disposition: Disposition = USABLE if usable else REFERENCE_NOT_PRODUCED
    if document.get("disposition") != disposition:
        raise ReferenceArtifactError("producer disposition differs from derived gate")
    if document.get("summary_usable") is not usable:
        raise ReferenceArtifactError(
            "producer usability summary differs from derived gate"
        )
    if usable and document.get("reconstruction_failure") is not None:
        raise ReferenceArtifactError("usable reference reports reconstruction failure")
    return ReferenceValidationResult(
        disposition=disposition,
        usable=usable,
        failure_reasons=tuple(reasons),
        artifact_sha256=artifact_sha256,
    )


__all__ = (
    "AUTHORITY_MANIFEST_SCHEMA_VERSION",
    "COMMON_KNOT_TOLERANCE",
    "DIAGNOSTICS_SCHEMA_VERSION",
    "HISTORICAL_FINAL_PARAMETER_PATH",
    "HISTORICAL_FINAL_PARAMETER_SHA256",
    "HISTORICAL_INPUT_BUNDLE_SHA256",
    "HISTORICAL_RECEIPT_SHA256",
    "HISTORICAL_TRAJECTORY_SHA256",
    "REFERENCE_FILENAME",
    "REFERENCE_NOT_PRODUCED",
    "RUNTIME_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA_VERSION",
    "USABLE",
    "HistoricalAuthorityPaths",
    "NativeReferenceRuntime",
    "ReferenceArtifactError",
    "ReferenceValidationResult",
    "RuntimeProvenance",
    "SourcePath",
    "canonical_json_bytes",
    "load_canonical_json_bytes",
    "produce_native_equivalent_reference",
    "validate_native_equivalent_reference",
)
