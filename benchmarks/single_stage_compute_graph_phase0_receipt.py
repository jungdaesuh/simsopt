"""Fail-closed Phase 0 receipt for single-stage GPU compute-graph work.

The schema keeps the RTX 5090 and A100 lanes structurally separate, binds every
measurement to one frozen changed-state specimen and immutable source manifest,
and recomputes first-evaluation, profile-coverage, gap-budget, and pivot gates.
It is a compute-graph engineering receipt, not an r5 campaign verdict.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Literal

from benchmarks.validation_ladder_contract import PARITY_LADDER_TOLERANCES

PHASE0_SCHEMA_ID: Final = "single-stage-jax-gpu-compute-graph-phase0-v5"
FORMAL_COMPLETE_PATH_FACTOR: Final = 0.9
HLO_MODULE_SET_IDENTITY_SOURCE: Final = "jax-profiler-kernel-hlo-module-set-sha256"
SAMPLED_PROCESS_GPU_MEMORY_SOURCE: Final = "nvidia-smi-sampled-process-peak-bytes"
HLO_MODULE_SET_IDENTITY_PREFIX: Final = "hlo-module-set-sha256:"
LANE_AGGREGATION_POLICY: Final = "separate_no_pooling"
RTX_LANE_ID: Final = "rtx5090"
A100_LANE_ID: Final = "a100"
REQUIRED_WARM_SAMPLES: Final = 10
FIRST_EVALUATION_LIMIT_NS: Final = 900_000_000_000
STATE_DIMENSION: Final = 255
COIL_DOF_COUNT: Final = 461
MINIMUM_ATTRIBUTION_COVERAGE: Final = 0.90
_GPU_RUNTIME_PARITY: Final = PARITY_LADDER_TOLERANCES["gpu_runtime"]
PHASE0_OBJECTIVE_ATOL: Final = float(_GPU_RUNTIME_PARITY["same_state_forward_atol"])
PHASE0_OBJECTIVE_RTOL: Final = float(_GPU_RUNTIME_PARITY["same_state_forward_rtol"])
PHASE0_GRADIENT_ATOL: Final = float(_GPU_RUNTIME_PARITY["same_state_gradient_atol"])
PHASE0_GRADIENT_RTOL: Final = float(_GPU_RUNTIME_PARITY["same_state_gradient_rtol"])

LaneId = Literal["rtx5090", "a100"]
QualificationOutcome = Literal["qualified", "blocked"]

_SHA256_HEX_LENGTH: Final = 64
_LOWER_HEX: Final = frozenset("0123456789abcdef")
_QUALIFIED_CHECKS: Final = frozenset(
    {
        "source_snapshot",
        "import_bindings",
        "native_extension",
        "device_identity",
        "runtime_backend",
        "fp64_policy",
        "cpu_affinity",
        "strict_transfer_smoke",
    }
)
_A100_ADDITIONAL_CHECKS: Final = frozenset(
    {
        "slurm_allocation",
        "cuda_12_6_compatibility",
        "dependency_overlay",
        "resolved_cuda_libraries",
    }
)
_REQUIRED_IMPORT_BINDINGS: Final = frozenset(
    {"simsopt", "simsopt_jax", "simsopt_jax_adapters", "simsoptpp"}
)
_REQUIRED_POLICIES: Final = frozenset(
    {
        "dense_batch_width",
        "point_chunk_size",
        "coil_chunk_size",
        "quadrature_block_sizes",
    }
)
_ALLOWED_PROFILE_PHASES: Final = frozenset(
    {
        "newton.warm_start",
        "newton.solver_control",
        "newton.residual_jvp",
        "newton.jacobian_construction",
        "newton.dense_materialization",
        "newton.linear_solve",
        "newton.lu_factor",
        "newton.refinement",
        "adjoint.dense_matrix",
        "adjoint.lu_factor",
        "adjoint.lu_solve",
        "adjoint.refinement",
        "adjoint.outer_vjp_rhs",
        "adjoint.implicit_coil_vjp",
        "biotsavart.forward",
        "biotsavart.vjp",
    }
)


class Phase0ReceiptError(ValueError):
    """The Phase 0 receipt is incomplete, inconsistent, or misleading."""


@dataclass(frozen=True)
class LaneAudit:
    """Validated per-device outcome without pooling any timing samples."""

    lane_id: LaneId
    outcome: QualificationOutcome
    device_uuid: str | None
    warm_p50_ns: float | None
    pivot_fired: bool | None


@dataclass(frozen=True)
class Phase0ReceiptAudit:
    """Immutable summary of a fully validated Phase 0 receipt."""

    artifact_id: str
    specimen_sha256: str
    rtx: LaneAudit
    a100: LaneAudit


def canonical_json_bytes(document: object) -> bytes:
    """Return the sole canonical JSON encoding for Phase 0 evidence."""

    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(document: object) -> str:
    """Hash the canonical JSON encoding of a schema subtree."""

    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def _load_canonical_artifact_json(
    raw_bytes: bytes, context: str
) -> Mapping[str, object]:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Phase0ReceiptError(f"{context} contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Phase0ReceiptError(
                    f"{context} contains non-finite JSON constant {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Phase0ReceiptError(
            f"{context} is not valid UTF-8 JSON: {error}"
        ) from error
    checked = _mapping(document, context)
    if canonical_json_bytes(checked) != raw_bytes:
        raise Phase0ReceiptError(f"{context} is not canonical JSON")
    return checked


def canonical_hlo_module_set_identity(modules: Sequence[str]) -> str:
    """Bind the sorted unique profiler HLO module-name set, not executable bytes."""

    normalized = tuple(sorted(set(modules)))
    if not normalized or any(not module for module in normalized):
        raise ValueError("HLO module-name set must be non-empty")
    payload = {
        "identity_source": HLO_MODULE_SET_IDENTITY_SOURCE,
        "hlo_modules": list(normalized),
    }
    return HLO_MODULE_SET_IDENTITY_PREFIX + canonical_sha256(payload)


def _hlo_module_set_identity(value: object, context: str) -> str:
    identity = _string(value, context)
    if not identity.startswith(HLO_MODULE_SET_IDENTITY_PREFIX):
        raise Phase0ReceiptError(f"{context} has an unsupported identity prefix")
    _sha256(identity.removeprefix(HLO_MODULE_SET_IDENTITY_PREFIX), context)
    return identity


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(k, str) for k in value):
        raise Phase0ReceiptError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise Phase0ReceiptError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise Phase0ReceiptError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase0ReceiptError(f"{context} must be a non-empty string")
    return value


def _xla_flags_string(value: object, context: str) -> str:
    if not isinstance(value, str):
        raise Phase0ReceiptError(f"{context} must be a string")
    return value


def _boolean(value: object, context: str) -> bool:
    if not isinstance(value, bool):
        raise Phase0ReceiptError(f"{context} must be boolean")
    return value


def _integer(value: object, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise Phase0ReceiptError(f"{context} must be an integer of at least {minimum}")
    return value


def _number(value: object, context: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Phase0ReceiptError(f"{context} must be numeric")
    checked = float(value)
    if not math.isfinite(checked) or checked < minimum:
        raise Phase0ReceiptError(f"{context} must be finite and at least {minimum}")
    return checked


def _fraction(value: object, context: str) -> float:
    checked = _number(value, context)
    if checked > 1.0:
        raise Phase0ReceiptError(f"{context} must not exceed 1")
    return checked


def _sha256(value: object, context: str) -> str:
    checked = _string(value, context)
    if len(checked) != _SHA256_HEX_LENGTH or any(c not in _LOWER_HEX for c in checked):
        raise Phase0ReceiptError(
            f"{context} must be a lowercase hexadecimal SHA-256 digest"
        )
    return checked


def _git_object_id(value: object, context: str) -> str:
    checked = _string(value, context)
    if len(checked) not in (40, 64) or any(c not in _LOWER_HEX for c in checked):
        raise Phase0ReceiptError(
            f"{context} must be a lowercase SHA-1 or SHA-256 Git object ID"
        )
    return checked


def _literal(value: object, expected: frozenset[str], context: str) -> str:
    checked = _string(value, context)
    if checked not in expected:
        raise Phase0ReceiptError(
            f"{context} must be one of {sorted(expected)}, got {checked!r}"
        )
    return checked


def _optional_string(value: object, context: str) -> str | None:
    return None if value is None else _string(value, context)


def _require_close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6):
        raise Phase0ReceiptError(
            f"{context} is {actual!r}, expected recomputed value {expected!r}"
        )


def _safe_relative_path(value: object, context: str) -> str:
    checked = _string(value, context)
    path = PurePosixPath(checked)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "." in path.parts
        or str(path) != checked
    ):
        raise Phase0ReceiptError(f"{context} must be a canonical safe relative path")
    return checked


def _validate_manifest(value: object, context: str) -> str:
    document = _mapping(value, context)
    _exact_keys(document, frozenset({"schema_id", "entries"}), context)
    _literal(
        document["schema_id"],
        frozenset({"single-stage-compute-graph-source-manifest-v1"}),
        f"{context}.schema_id",
    )
    entries = _sequence(document["entries"], f"{context}.entries")
    if not entries:
        raise Phase0ReceiptError(f"{context}.entries must be non-empty")
    paths: list[str] = []
    for index, raw_entry in enumerate(entries):
        entry_context = f"{context}.entries[{index}]"
        entry = _mapping(raw_entry, entry_context)
        _exact_keys(
            entry,
            frozenset({"role", "relative_path", "size_bytes", "sha256"}),
            entry_context,
        )
        _string(entry["role"], f"{entry_context}.role")
        paths.append(
            _safe_relative_path(
                entry["relative_path"], f"{entry_context}.relative_path"
            )
        )
        _integer(entry["size_bytes"], f"{entry_context}.size_bytes")
        _sha256(entry["sha256"], f"{entry_context}.sha256")
    if len(paths) != len(set(paths)):
        raise Phase0ReceiptError(f"{context} contains duplicate paths")
    return canonical_sha256(document)


def _validate_specimen(value: object, context: str) -> str:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "specimen_id",
                "input_bundle_sha256",
                "parameter_sha256",
                "state_dimension",
                "coil_dof_count",
                "grids",
                "weights",
                "tolerances",
                "solver_graph_id",
                "solver_graph_sha256",
            }
        ),
        context,
    )
    _string(document["specimen_id"], f"{context}.specimen_id")
    _sha256(document["input_bundle_sha256"], f"{context}.input_bundle_sha256")
    _sha256(document["parameter_sha256"], f"{context}.parameter_sha256")
    if (
        _integer(document["state_dimension"], f"{context}.state_dimension")
        != STATE_DIMENSION
    ):
        raise Phase0ReceiptError(f"{context}.state_dimension must be {STATE_DIMENSION}")
    if (
        _integer(document["coil_dof_count"], f"{context}.coil_dof_count")
        != COIL_DOF_COUNT
    ):
        raise Phase0ReceiptError(f"{context}.coil_dof_count must be {COIL_DOF_COUNT}")
    for field in ("grids", "weights", "tolerances"):
        values = _mapping(document[field], f"{context}.{field}")
        if not values:
            raise Phase0ReceiptError(f"{context}.{field} must be non-empty")
        for key, raw_value in values.items():
            _string(key, f"{context}.{field} key")
            if field == "grids":
                _integer(raw_value, f"{context}.{field}.{key}", minimum=1)
            else:
                _number(raw_value, f"{context}.{field}.{key}")
    _string(document["solver_graph_id"], f"{context}.solver_graph_id")
    _sha256(document["solver_graph_sha256"], f"{context}.solver_graph_sha256")
    return canonical_sha256(document)


def _validate_identity_map(value: object, context: str) -> None:
    document = _mapping(value, context)
    if not document:
        raise Phase0ReceiptError(f"{context} must be non-empty")
    for key, raw_value in document.items():
        _string(key, f"{context} key")
        if isinstance(raw_value, bool):
            continue
        if isinstance(raw_value, int) and not isinstance(raw_value, bool):
            continue
        if raw_value is None:
            continue
        if isinstance(raw_value, str) and raw_value:
            continue
        if isinstance(raw_value, Sequence) and not isinstance(
            raw_value, (str, bytes, bytearray)
        ):
            for item_index, item in enumerate(raw_value):
                if not isinstance(item, (str, int)) or isinstance(item, bool):
                    raise Phase0ReceiptError(
                        f"{context}.{key}[{item_index}] must be string or integer"
                    )
            continue
        raise Phase0ReceiptError(f"{context}.{key} has unsupported identity value")


def _validate_qualification(
    value: object, lane_id: LaneId, context: str
) -> QualificationOutcome:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset({"outcome", "attempted_identity", "checks", "blocker"}),
        context,
    )
    outcome = _literal(
        document["outcome"], frozenset({"qualified", "blocked"}), f"{context}.outcome"
    )
    _validate_identity_map(
        document["attempted_identity"], f"{context}.attempted_identity"
    )
    checks = _sequence(document["checks"], f"{context}.checks")
    required = _QUALIFIED_CHECKS | (
        _A100_ADDITIONAL_CHECKS if lane_id == A100_LANE_ID else frozenset()
    )
    observed: dict[str, bool] = {}
    for index, raw_check in enumerate(checks):
        check_context = f"{context}.checks[{index}]"
        check = _mapping(raw_check, check_context)
        _exact_keys(check, frozenset({"check_id", "passed", "evidence"}), check_context)
        check_id = _string(check["check_id"], f"{check_context}.check_id")
        if check_id in observed:
            raise Phase0ReceiptError(f"{context} contains duplicate check {check_id!r}")
        observed[check_id] = _boolean(check["passed"], f"{check_context}.passed")
        _string(check["evidence"], f"{check_context}.evidence")
    if frozenset(observed) != required:
        raise Phase0ReceiptError(
            f"{context} qualification checks must be exactly {sorted(required)}"
        )
    blocker_value = document["blocker"]
    if outcome == "qualified":
        if not all(observed.values()) or blocker_value is not None:
            raise Phase0ReceiptError(
                f"{context} qualified outcome requires every check to pass and no blocker"
            )
        return "qualified"
    blocker = _mapping(blocker_value, f"{context}.blocker")
    _exact_keys(
        blocker,
        frozenset({"code", "check_id", "reason", "evidence_sha256"}),
        f"{context}.blocker",
    )
    _string(blocker["code"], f"{context}.blocker.code")
    blocker_check = _string(blocker["check_id"], f"{context}.blocker.check_id")
    _string(blocker["reason"], f"{context}.blocker.reason")
    _sha256(blocker["evidence_sha256"], f"{context}.blocker.evidence_sha256")
    failed_checks = tuple(
        check_id for check_id, passed in observed.items() if not passed
    )
    if not failed_checks or blocker_check != failed_checks[0]:
        raise Phase0ReceiptError(
            f"{context}.blocker must identify the first failed qualification check"
        )
    return "blocked"


def _validate_provenance(
    value: object, lane_id: LaneId, context: str
) -> tuple[str, str, str]:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "repository_commit",
                "source_state_sha256",
                "git_status_short",
                "tracked_diff_sha256",
                "untracked_manifest_sha256",
                "immutable_root",
                "immutable_tree_sha256",
                "source_manifest",
                "source_manifest_sha256",
                "interpreter_path",
                "runtime",
                "allocation",
                "import_bindings",
                "package_overlay",
                "environment",
                "policies",
                "compilation_cache_directory",
            }
        ),
        context,
    )
    _git_object_id(document["repository_commit"], f"{context}.repository_commit")
    source_state = _sha256(
        document["source_state_sha256"], f"{context}.source_state_sha256"
    )
    status = _sequence(document["git_status_short"], f"{context}.git_status_short")
    for index, line in enumerate(status):
        _string(line, f"{context}.git_status_short[{index}]")
    _sha256(document["tracked_diff_sha256"], f"{context}.tracked_diff_sha256")
    _sha256(
        document["untracked_manifest_sha256"], f"{context}.untracked_manifest_sha256"
    )
    _string(document["immutable_root"], f"{context}.immutable_root")
    _sha256(document["immutable_tree_sha256"], f"{context}.immutable_tree_sha256")
    manifest_sha = _validate_manifest(
        document["source_manifest"], f"{context}.source_manifest"
    )
    if (
        _sha256(document["source_manifest_sha256"], f"{context}.source_manifest_sha256")
        != manifest_sha
    ):
        raise Phase0ReceiptError(
            f"{context}.source_manifest_sha256 does not match manifest bytes"
        )
    _string(document["interpreter_path"], f"{context}.interpreter_path")
    runtime = _mapping(document["runtime"], f"{context}.runtime")
    _exact_keys(
        runtime,
        frozenset(
            {
                "python_version",
                "jax_version",
                "jaxlib_version",
                "cuda_runtime",
                "cuda_driver",
                "jax_backend",
                "fp64_x64_enabled",
                "resolved_cuda_libraries",
            }
        ),
        f"{context}.runtime",
    )
    for field in (
        "python_version",
        "jax_version",
        "jaxlib_version",
        "cuda_runtime",
        "cuda_driver",
    ):
        _string(runtime[field], f"{context}.runtime.{field}")
    if (
        _literal(
            runtime["jax_backend"], frozenset({"gpu"}), f"{context}.runtime.jax_backend"
        )
        != "gpu"
    ):
        raise AssertionError("unreachable")
    if not _boolean(runtime["fp64_x64_enabled"], f"{context}.runtime.fp64_x64_enabled"):
        raise Phase0ReceiptError(f"{context}.runtime.fp64_x64_enabled must be true")
    libraries = _sequence(
        runtime["resolved_cuda_libraries"], f"{context}.runtime.resolved_cuda_libraries"
    )
    if not libraries:
        raise Phase0ReceiptError(
            f"{context}.runtime.resolved_cuda_libraries must be non-empty"
        )
    for index, library in enumerate(libraries):
        _string(library, f"{context}.runtime.resolved_cuda_libraries[{index}]")
    allocation = _mapping(document["allocation"], f"{context}.allocation")
    _exact_keys(
        allocation,
        frozenset(
            {
                "hostname",
                "scheduler",
                "allocation_id",
                "job_id",
                "gpu_name",
                "gpu_uuid",
                "gpu_memory_bytes",
                "cpu_affinity",
                "cuda_compatibility_version",
                "cuda_compatibility_path",
            }
        ),
        f"{context}.allocation",
    )
    for field in (
        "hostname",
        "scheduler",
        "allocation_id",
        "job_id",
        "gpu_name",
        "gpu_uuid",
        "cpu_affinity",
        "cuda_compatibility_version",
        "cuda_compatibility_path",
    ):
        _string(allocation[field], f"{context}.allocation.{field}")
    device_uuid = _string(allocation["gpu_uuid"], f"{context}.allocation.gpu_uuid")
    _integer(
        allocation["gpu_memory_bytes"],
        f"{context}.allocation.gpu_memory_bytes",
        minimum=1,
    )
    if lane_id == A100_LANE_ID and allocation["cuda_compatibility_version"] != "12.6":
        raise Phase0ReceiptError("A100 measured lane requires CUDA 12.6 compatibility")
    bindings = _mapping(document["import_bindings"], f"{context}.import_bindings")
    if frozenset(bindings) != _REQUIRED_IMPORT_BINDINGS:
        raise Phase0ReceiptError(
            f"{context}.import_bindings has incomplete package identity"
        )
    for package, raw_binding in bindings.items():
        binding = _mapping(raw_binding, f"{context}.import_bindings.{package}")
        _exact_keys(
            binding,
            frozenset({"path", "sha256"}),
            f"{context}.import_bindings.{package}",
        )
        _string(binding["path"], f"{context}.import_bindings.{package}.path")
        _sha256(binding["sha256"], f"{context}.import_bindings.{package}.sha256")
    overlay = _mapping(document["package_overlay"], f"{context}.package_overlay")
    if "lineax" not in overlay:
        raise Phase0ReceiptError(f"{context}.package_overlay must include lineax")
    for package, version in overlay.items():
        _string(package, f"{context}.package_overlay key")
        _string(version, f"{context}.package_overlay.{package}")
    _validate_identity_map(document["environment"], f"{context}.environment")
    policies = _mapping(document["policies"], f"{context}.policies")
    if frozenset(policies) != _REQUIRED_POLICIES:
        raise Phase0ReceiptError(
            f"{context}.policies must bind every chunk and dense-batch policy"
        )
    _integer(
        policies["dense_batch_width"],
        f"{context}.policies.dense_batch_width",
        minimum=1,
    )
    for field in ("point_chunk_size", "coil_chunk_size"):
        if policies[field] is not None:
            _integer(policies[field], f"{context}.policies.{field}", minimum=1)
    blocks = _sequence(
        policies["quadrature_block_sizes"], f"{context}.policies.quadrature_block_sizes"
    )
    if tuple(blocks) != (128, 122):
        raise Phase0ReceiptError(
            f"{context}.policies.quadrature_block_sizes must be [128, 122]"
        )
    cache_directory = _string(
        document["compilation_cache_directory"],
        f"{context}.compilation_cache_directory",
    )
    return source_state, device_uuid, cache_directory


def _validate_first_evaluation(
    value: object, specimen: Mapping[str, object], context: str
) -> None:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "variant",
                "wall_time_limit_ns",
                "elapsed_ns",
                "completed",
                "objective_dtype",
                "objective",
                "gradient_dtype",
                "gradient",
                "native_objective",
                "native_gradient",
                "objective_atol",
                "objective_rtol",
                "gradient_atol",
                "gradient_rtol",
                "inner_newton_success",
                "adjoint_success",
                "residual_certificates",
            }
        ),
        context,
    )
    _literal(document["variant"], frozenset({"C0"}), f"{context}.variant")
    if (
        _integer(
            document["wall_time_limit_ns"], f"{context}.wall_time_limit_ns", minimum=1
        )
        != FIRST_EVALUATION_LIMIT_NS
    ):
        raise Phase0ReceiptError(
            f"{context}.wall_time_limit_ns must be the 900-second gate"
        )
    elapsed = _integer(document["elapsed_ns"], f"{context}.elapsed_ns", minimum=1)
    if elapsed > FIRST_EVALUATION_LIMIT_NS or not _boolean(
        document["completed"], f"{context}.completed"
    ):
        raise Phase0ReceiptError(
            f"{context} did not complete inside the first-evaluation gate"
        )
    for field in ("objective_dtype", "gradient_dtype"):
        _literal(document[field], frozenset({"float64"}), f"{context}.{field}")
    objective = _number(
        document["objective"], f"{context}.objective", minimum=-math.inf
    )
    native_objective = _number(
        document["native_objective"], f"{context}.native_objective", minimum=-math.inf
    )
    gradient = tuple(
        _number(item, f"{context}.gradient[{index}]", minimum=-math.inf)
        for index, item in enumerate(
            _sequence(document["gradient"], f"{context}.gradient")
        )
    )
    native_gradient = tuple(
        _number(item, f"{context}.native_gradient[{index}]", minimum=-math.inf)
        for index, item in enumerate(
            _sequence(document["native_gradient"], f"{context}.native_gradient")
        )
    )
    expected_count = _integer(specimen["coil_dof_count"], "specimen.coil_dof_count")
    if len(gradient) != expected_count or len(native_gradient) != expected_count:
        raise Phase0ReceiptError(
            f"{context} gradients must contain exactly {expected_count} entries"
        )
    objective_atol = _number(document["objective_atol"], f"{context}.objective_atol")
    objective_rtol = _number(document["objective_rtol"], f"{context}.objective_rtol")
    gradient_atol = _number(document["gradient_atol"], f"{context}.gradient_atol")
    gradient_rtol = _number(document["gradient_rtol"], f"{context}.gradient_rtol")
    if abs(objective - native_objective) > objective_atol + objective_rtol * abs(
        native_objective
    ):
        raise Phase0ReceiptError(f"{context} objective parity failed")
    for index, (actual, expected) in enumerate(
        zip(gradient, native_gradient, strict=True)
    ):
        if abs(actual - expected) > gradient_atol + gradient_rtol * abs(expected):
            raise Phase0ReceiptError(
                f"{context} gradient parity failed at index {index}"
            )
    if not _boolean(
        document["inner_newton_success"], f"{context}.inner_newton_success"
    ):
        raise Phase0ReceiptError(f"{context} inner Newton solve failed")
    if not _boolean(document["adjoint_success"], f"{context}.adjoint_success"):
        raise Phase0ReceiptError(f"{context} adjoint solve failed")
    residuals = _mapping(
        document["residual_certificates"], f"{context}.residual_certificates"
    )
    if not residuals:
        raise Phase0ReceiptError(f"{context}.residual_certificates must be non-empty")
    for name, residual in residuals.items():
        _string(name, f"{context}.residual_certificates key")
        _number(residual, f"{context}.residual_certificates.{name}")


def _validate_warm_measurement(value: object, context: str) -> float:
    document = _mapping(value, context)
    _exact_keys(document, frozenset({"samples", "p50_ns", "p95_ns"}), context)
    samples = _sequence(document["samples"], f"{context}.samples")
    if len(samples) < REQUIRED_WARM_SAMPLES:
        raise Phase0ReceiptError(
            f"{context} requires at least {REQUIRED_WARM_SAMPLES} warm samples"
        )
    wall_times: list[int] = []
    for index, raw_sample in enumerate(samples):
        sample_context = f"{context}.samples[{index}]"
        sample = _mapping(raw_sample, sample_context)
        _exact_keys(
            sample,
            frozenset(
                {
                    "sample_index",
                    "wall_ns",
                    "peak_process_tree_rss_bytes",
                    "sampled_process_gpu_memory_peak_bytes",
                    "sampled_process_gpu_memory_source",
                    "profiled",
                }
            ),
            sample_context,
        )
        if _integer(sample["sample_index"], f"{sample_context}.sample_index") != index:
            raise Phase0ReceiptError(f"{context} sample indices must be contiguous")
        wall_times.append(
            _integer(sample["wall_ns"], f"{sample_context}.wall_ns", minimum=1)
        )
        _integer(
            sample["peak_process_tree_rss_bytes"],
            f"{sample_context}.peak_process_tree_rss_bytes",
            minimum=1,
        )
        _integer(
            sample["sampled_process_gpu_memory_peak_bytes"],
            f"{sample_context}.sampled_process_gpu_memory_peak_bytes",
            minimum=1,
        )
        _literal(
            sample["sampled_process_gpu_memory_source"],
            frozenset({SAMPLED_PROCESS_GPU_MEMORY_SOURCE}),
            f"{sample_context}.sampled_process_gpu_memory_source",
        )
        if _boolean(sample["profiled"], f"{sample_context}.profiled"):
            raise Phase0ReceiptError(
                f"{sample_context} promotion timing must be unprofiled"
            )
    p50 = _number(document["p50_ns"], f"{context}.p50_ns", minimum=1)
    p95 = _number(document["p95_ns"], f"{context}.p95_ns", minimum=1)
    _require_close(p50, float(statistics.median(wall_times)), f"{context}.p50_ns")
    ordered = sorted(wall_times)
    expected_p95 = float(ordered[math.ceil(0.95 * len(ordered)) - 1])
    _require_close(p95, expected_p95, f"{context}.p95_ns")
    return p50


def _interval_union_ns(intervals: Sequence[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    merged: list[tuple[int, int]] = []
    start, end = ordered[0]
    for next_start, next_end in ordered[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            merged.append((start, end))
            start, end = next_start, next_end
    merged.append((start, end))
    return sum(end_ns - start_ns for start_ns, end_ns in merged)


def _validate_profile(
    value: object, context: str
) -> tuple[float, float, str, dict[str, float]]:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "evaluation_envelope_ns",
                "device_active_ns",
                "phase_interval_unions",
                "attributed_union_ns",
                "unattributed_ns",
                "attribution_coverage",
                "pjrt_execute_count",
                "kernel_launch_count",
                "kernel_duration_ns",
                "inter_launch_gap_ns",
                "hlo_module_set_identity",
                "hlo_module_set_identity_source",
            }
        ),
        context,
    )
    envelope = _integer(
        document["evaluation_envelope_ns"],
        f"{context}.evaluation_envelope_ns",
        minimum=1,
    )
    device_active = _integer(
        document["device_active_ns"], f"{context}.device_active_ns", minimum=1
    )
    if device_active > envelope:
        raise Phase0ReceiptError(
            f"{context}.device_active_ns exceeds evaluation envelope"
        )
    phase_rows = _sequence(
        document["phase_interval_unions"], f"{context}.phase_interval_unions"
    )
    all_intervals: list[tuple[int, int]] = []
    phase_ids: list[str] = []
    phase_wall_shares: dict[str, float] = {}
    for row_index, raw_row in enumerate(phase_rows):
        row_context = f"{context}.phase_interval_unions[{row_index}]"
        row = _mapping(raw_row, row_context)
        _exact_keys(row, frozenset({"phase_id", "intervals"}), row_context)
        phase_id = _literal(
            row["phase_id"], _ALLOWED_PROFILE_PHASES, f"{row_context}.phase_id"
        )
        phase_ids.append(phase_id)
        intervals = _sequence(row["intervals"], f"{row_context}.intervals")
        phase_intervals: list[tuple[int, int]] = []
        for interval_index, raw_interval in enumerate(intervals):
            interval = _sequence(
                raw_interval, f"{row_context}.intervals[{interval_index}]"
            )
            if len(interval) != 2:
                raise Phase0ReceiptError(
                    f"{row_context}.intervals[{interval_index}] must be [start_ns, end_ns]"
                )
            start = _integer(
                interval[0], f"{row_context}.intervals[{interval_index}][0]"
            )
            end = _integer(
                interval[1], f"{row_context}.intervals[{interval_index}][1]", minimum=1
            )
            if end <= start:
                raise Phase0ReceiptError(
                    f"{row_context}.intervals[{interval_index}] must have positive duration"
                )
            if start < 0 or end > envelope:
                raise Phase0ReceiptError(
                    f"{row_context}.intervals[{interval_index}] exceeds evaluation envelope"
                )
            all_intervals.append((start, end))
            phase_intervals.append((start, end))
        phase_wall_shares[phase_id] = _interval_union_ns(phase_intervals) / envelope
    if not phase_ids or len(phase_ids) != len(set(phase_ids)):
        raise Phase0ReceiptError(f"{context} phase IDs must be non-empty and unique")
    attributed = _integer(
        document["attributed_union_ns"], f"{context}.attributed_union_ns"
    )
    recomputed_attributed = _interval_union_ns(all_intervals)
    if attributed != recomputed_attributed or attributed > device_active:
        raise Phase0ReceiptError(
            f"{context}.attributed_union_ns does not match interval union"
        )
    summed_phase_ns = sum(share * envelope for share in phase_wall_shares.values())
    if not math.isclose(summed_phase_ns, attributed, rel_tol=0.0, abs_tol=1e-9):
        raise Phase0ReceiptError(
            f"{context} phase interval unions must be mutually exclusive"
        )
    unattributed = _integer(document["unattributed_ns"], f"{context}.unattributed_ns")
    if unattributed != device_active - attributed:
        raise Phase0ReceiptError(
            f"{context}.unattributed_ns does not reconcile device time"
        )
    coverage = _fraction(
        document["attribution_coverage"], f"{context}.attribution_coverage"
    )
    _require_close(
        coverage, attributed / device_active, f"{context}.attribution_coverage"
    )
    if coverage < MINIMUM_ATTRIBUTION_COVERAGE:
        raise Phase0ReceiptError(f"{context} has less than 90% phase attribution")
    _integer(document["pjrt_execute_count"], f"{context}.pjrt_execute_count", minimum=1)
    _integer(
        document["kernel_launch_count"], f"{context}.kernel_launch_count", minimum=1
    )
    durations = _sequence(
        document["kernel_duration_ns"], f"{context}.kernel_duration_ns"
    )
    if not durations:
        raise Phase0ReceiptError(f"{context}.kernel_duration_ns must be non-empty")
    for index, duration in enumerate(durations):
        _integer(duration, f"{context}.kernel_duration_ns[{index}]", minimum=1)
    gap_ns = _integer(document["inter_launch_gap_ns"], f"{context}.inter_launch_gap_ns")
    _literal(
        document["hlo_module_set_identity_source"],
        frozenset({HLO_MODULE_SET_IDENTITY_SOURCE}),
        f"{context}.hlo_module_set_identity_source",
    )
    identity = _hlo_module_set_identity(
        document["hlo_module_set_identity"],
        f"{context}.hlo_module_set_identity",
    )
    return coverage, gap_ns / envelope, identity, phase_wall_shares


def _validate_command_buffer(value: object, context: str) -> None:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "resolved_configuration",
                "observed_capture_participation",
                "graph_launched_device_ns",
                "uncaptured_device_ns",
                "captured_launch_count",
                "uncaptured_launch_count",
                "ab_control",
            }
        ),
        context,
    )
    _xla_flags_string(
        document["resolved_configuration"], f"{context}.resolved_configuration"
    )
    observed = _boolean(
        document["observed_capture_participation"],
        f"{context}.observed_capture_participation",
    )
    graph_ns = _integer(
        document["graph_launched_device_ns"], f"{context}.graph_launched_device_ns"
    )
    captured = _integer(
        document["captured_launch_count"], f"{context}.captured_launch_count"
    )
    _integer(document["uncaptured_device_ns"], f"{context}.uncaptured_device_ns")
    _integer(document["uncaptured_launch_count"], f"{context}.uncaptured_launch_count")
    if observed != (graph_ns > 0 and captured > 0):
        raise Phase0ReceiptError(
            f"{context} observed capture flag contradicts trace evidence"
        )
    control_value = document["ab_control"]
    if control_value is None:
        return
    control = _mapping(control_value, f"{context}.ab_control")
    _exact_keys(
        control,
        frozenset(
            {
                "control_id",
                "resolved_configuration",
                "sample_wall_ns",
                "included_in_promotion_timing",
            }
        ),
        f"{context}.ab_control",
    )
    _string(control["control_id"], f"{context}.ab_control.control_id")
    _string(
        control["resolved_configuration"],
        f"{context}.ab_control.resolved_configuration",
    )
    samples = _sequence(
        control["sample_wall_ns"], f"{context}.ab_control.sample_wall_ns"
    )
    if not samples:
        raise Phase0ReceiptError(
            f"{context}.ab_control.sample_wall_ns must be non-empty"
        )
    for index, sample in enumerate(samples):
        _integer(sample, f"{context}.ab_control.sample_wall_ns[{index}]", minimum=1)
    if _boolean(
        control["included_in_promotion_timing"],
        f"{context}.ab_control.included_in_promotion_timing",
    ):
        raise Phase0ReceiptError(
            f"{context}.ab_control cannot contribute to promotion timing"
        )


def _validate_attribution_control(
    value: object,
    *,
    lane_id: LaneId,
    specimen: Mapping[str, object],
    specimen_sha256: str,
    source_state_sha256: str,
    device_uuid: str,
    warm_p50_ns: float,
    context: str,
) -> dict[str, float]:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "schema_id",
                "state",
                "promotion_eligible",
                "blockers",
                "production_binding",
                "direct_default_measurement",
                "attribution_replay",
                "equivalence",
                "stability",
                "selected_attribution",
            }
        ),
        context,
    )
    _literal(
        document["schema_id"],
        frozenset({"single-stage-compute-graph-attribution-evidence-v4"}),
        f"{context}.schema_id",
    )
    state = _string(document["state"], f"{context}.state")
    if state == "NON_PROMOTING":
        raise Phase0ReceiptError(f"{context} is explicitly non-promoting")
    if state != "PRODUCED":
        raise Phase0ReceiptError(f"{context}.state must be PRODUCED")
    if not _boolean(document["promotion_eligible"], f"{context}.promotion_eligible"):
        raise Phase0ReceiptError(f"{context} is not promotion eligible")
    blockers = _sequence(document["blockers"], f"{context}.blockers")
    if blockers:
        raise Phase0ReceiptError(f"{context}.blockers must be empty")

    binding = _mapping(document["production_binding"], f"{context}.production_binding")
    _exact_keys(
        binding,
        frozenset(
            {
                "candidate_sha256",
                "specimen_sha256",
                "input_bundle_sha256",
                "source_sha256",
                "production_runtime_identity_sha256",
                "lane_id",
                "gpu_uuid",
                "gate_checkpoint_sha256",
                "warm_checkpoint_sha256",
                "warm_p50_ns",
            }
        ),
        f"{context}.production_binding",
    )
    expected_binding = {
        "candidate_sha256": specimen["parameter_sha256"],
        "specimen_sha256": specimen_sha256,
        "input_bundle_sha256": specimen["input_bundle_sha256"],
        "source_sha256": source_state_sha256,
        "lane_id": lane_id,
        "gpu_uuid": device_uuid,
        "warm_p50_ns": warm_p50_ns,
    }
    for field, expected in expected_binding.items():
        if binding[field] != expected:
            raise Phase0ReceiptError(
                f"{context}.production_binding.{field} differs from measurement"
            )
    _sha256(
        binding["production_runtime_identity_sha256"],
        f"{context}.production_binding.production_runtime_identity_sha256",
    )
    _sha256(
        binding["gate_checkpoint_sha256"],
        f"{context}.production_binding.gate_checkpoint_sha256",
    )
    _sha256(
        binding["warm_checkpoint_sha256"],
        f"{context}.production_binding.warm_checkpoint_sha256",
    )

    direct = _mapping(
        document["direct_default_measurement"],
        f"{context}.direct_default_measurement",
    )
    replay = _mapping(document["attribution_replay"], f"{context}.attribution_replay")
    parsed_attempts: dict[str, list[Mapping[str, object]]] = {}
    for name, evidence, authoritative, expected_mode in (
        ("direct_default_measurement", direct, True, "default_control"),
        (
            "attribution_replay",
            replay,
            False,
            "command_buffer_disabled",
        ),
    ):
        _exact_keys(
            evidence,
            frozenset({"authoritative_for_timing", "attempts"}),
            f"{context}.{name}",
        )
        if (
            _boolean(
                evidence["authoritative_for_timing"],
                f"{context}.{name}.authoritative_for_timing",
            )
            is not authoritative
        ):
            raise Phase0ReceiptError(f"{context}.{name} timing authority is invalid")
        attempts = _sequence(evidence["attempts"], f"{context}.{name}.attempts")
        if len(attempts) != 3:
            raise Phase0ReceiptError(f"{context}.{name} must contain three attempts")
        parsed: list[Mapping[str, object]] = []
        for index, raw_attempt in enumerate(attempts):
            attempt_context = f"{context}.{name}.attempts[{index}]"
            attempt = _mapping(raw_attempt, attempt_context)
            _exact_keys(
                attempt,
                frozenset(
                    {
                        "mode",
                        "attempt_index",
                        "runtime_identity_sha256",
                        "xla_flag_tokens",
                        "compilation_cache_root",
                        "artifact_root",
                        "raw_trace_path",
                        "raw_trace_sha256",
                        "child_observation_path",
                        "child_observation_sha256",
                        "hlo_anchor_path",
                        "hlo_anchor_sha256",
                        "profile_derivation_version",
                        "objective",
                        "gradient",
                        "solve_certificate",
                        "solve_certificate_categorical_identity_sha256",
                        "module_topology_identity_sha256",
                        "evaluation_envelope_ns",
                        "device_active_ns",
                        "device_active_share",
                        "phase_device_ns",
                        "attribution_coverage",
                    }
                ),
                attempt_context,
            )
            _literal(
                attempt["mode"], frozenset({expected_mode}), f"{attempt_context}.mode"
            )
            if (
                _integer(attempt["attempt_index"], f"{attempt_context}.attempt_index")
                != index
            ):
                raise Phase0ReceiptError(
                    f"{attempt_context}.attempt_index is not canonical"
                )
            for field in (
                "runtime_identity_sha256",
                "solve_certificate_categorical_identity_sha256",
                "module_topology_identity_sha256",
                "raw_trace_sha256",
                "child_observation_sha256",
                "hlo_anchor_sha256",
            ):
                _sha256(attempt[field], f"{attempt_context}.{field}")
            tokens = _sequence(
                attempt["xla_flag_tokens"], f"{attempt_context}.xla_flag_tokens"
            )
            if any(
                not isinstance(token, str) or not token or token.strip() != token
                for token in tokens
            ):
                raise Phase0ReceiptError(
                    f"{attempt_context}.xla_flag_tokens are invalid"
                )
            for field in ("compilation_cache_root", "artifact_root"):
                raw_path = _string(attempt[field], f"{attempt_context}.{field}")
                checked_path = PurePosixPath(raw_path)
                if not checked_path.is_absolute() or ".." in checked_path.parts:
                    raise Phase0ReceiptError(
                        f"{attempt_context}.{field} must be a normalized absolute path"
                    )
            for field in (
                "raw_trace_path",
                "child_observation_path",
                "hlo_anchor_path",
            ):
                raw_path = _string(attempt[field], f"{attempt_context}.{field}")
                checked_path = PurePosixPath(raw_path)
                if checked_path.is_absolute() or ".." in checked_path.parts:
                    raise Phase0ReceiptError(
                        f"{attempt_context}.{field} must be a safe relative path"
                    )
            _literal(
                attempt["profile_derivation_version"],
                frozenset({"compute-graph-profile-attribution-v1"}),
                f"{attempt_context}.profile_derivation_version",
            )
            objective = attempt["objective"]
            if (
                isinstance(objective, bool)
                or not isinstance(objective, (int, float))
                or not math.isfinite(objective)
            ):
                raise Phase0ReceiptError(f"{attempt_context}.objective must be finite")
            gradient = _sequence(attempt["gradient"], f"{attempt_context}.gradient")
            if not gradient or any(
                isinstance(component, bool)
                or not isinstance(component, (int, float))
                or not math.isfinite(component)
                for component in gradient
            ):
                raise Phase0ReceiptError(
                    f"{attempt_context}.gradient must be non-empty and finite"
                )
            _mapping(
                attempt["solve_certificate"],
                f"{attempt_context}.solve_certificate",
            )
            envelope = _integer(
                attempt["evaluation_envelope_ns"],
                f"{attempt_context}.evaluation_envelope_ns",
                minimum=1,
            )
            active = _integer(
                attempt["device_active_ns"],
                f"{attempt_context}.device_active_ns",
                minimum=1,
            )
            if active > envelope:
                raise Phase0ReceiptError(
                    f"{attempt_context} device time exceeds envelope"
                )
            _require_close(
                _fraction(
                    attempt["device_active_share"],
                    f"{attempt_context}.device_active_share",
                ),
                active / envelope,
                f"{attempt_context}.device_active_share",
            )
            phase_rows = _sequence(
                attempt["phase_device_ns"], f"{attempt_context}.phase_device_ns"
            )
            phase_ids: list[str] = []
            attributed_ns = 0
            for phase_index, raw_phase in enumerate(phase_rows):
                phase_context = f"{attempt_context}.phase_device_ns[{phase_index}]"
                phase = _mapping(raw_phase, phase_context)
                _exact_keys(
                    phase, frozenset({"phase_id", "duration_ns"}), phase_context
                )
                phase_ids.append(
                    _literal(
                        phase["phase_id"],
                        _ALLOWED_PROFILE_PHASES,
                        f"{phase_context}.phase_id",
                    )
                )
                attributed_ns += _integer(
                    phase["duration_ns"],
                    f"{phase_context}.duration_ns",
                    minimum=1,
                )
            if not phase_ids or len(phase_ids) != len(set(phase_ids)):
                raise Phase0ReceiptError(
                    f"{attempt_context}.phase_device_ns IDs must be unique"
                )
            if attributed_ns > active:
                raise Phase0ReceiptError(
                    f"{attempt_context}.phase_device_ns exceeds device time"
                )
            _require_close(
                _fraction(
                    attempt["attribution_coverage"],
                    f"{attempt_context}.attribution_coverage",
                ),
                attributed_ns / active,
                f"{attempt_context}.attribution_coverage",
            )
            parsed.append(attempt)
        parsed_attempts[name] = parsed

    all_attempts = (
        parsed_attempts["direct_default_measurement"]
        + parsed_attempts["attribution_replay"]
    )
    from benchmarks.single_stage_compute_graph_attribution_control import (
        AttributionControlError,
        require_promoting_attribution_evidence,
    )

    try:
        require_promoting_attribution_evidence(document)
    except AttributionControlError as error:
        raise Phase0ReceiptError(
            f"{context} attribution evidence validation failed: {error}"
        ) from error
    if (
        len({attempt["module_topology_identity_sha256"] for attempt in all_attempts})
        != 1
    ):
        raise Phase0ReceiptError(f"{context} module topology identities differ")
    default_runtime_ids = {
        attempt["runtime_identity_sha256"]
        for attempt in parsed_attempts["direct_default_measurement"]
    }
    disabled_runtime_ids = {
        attempt["runtime_identity_sha256"]
        for attempt in parsed_attempts["attribution_replay"]
    }
    if len(default_runtime_ids) != 1 or len(disabled_runtime_ids) != 1:
        raise Phase0ReceiptError(f"{context} runtime identity is unstable")
    if default_runtime_ids != {binding["production_runtime_identity_sha256"]}:
        raise Phase0ReceiptError(
            f"{context} default runtime identity differs from production"
        )
    if default_runtime_ids & disabled_runtime_ids:
        raise Phase0ReceiptError(f"{context} disabled runtime identity is not distinct")
    default_token_sequences = {
        tuple(_sequence(attempt["xla_flag_tokens"], "default XLA tokens"))
        for attempt in parsed_attempts["direct_default_measurement"]
    }
    if len(default_token_sequences) != 1:
        raise Phase0ReceiptError(f"{context} default XLA tokens are unstable")
    default_tokens = next(iter(default_token_sequences))
    if any(
        str(token).startswith("--xla_gpu_enable_command_buffer")
        for token in default_tokens
    ):
        raise Phase0ReceiptError(f"{context} default replay overrides command buffers")
    for attempt in parsed_attempts["attribution_replay"]:
        tokens = tuple(_sequence(attempt["xla_flag_tokens"], "disabled XLA tokens"))
        if tokens != (*default_tokens, "--xla_gpu_enable_command_buffer="):
            raise Phase0ReceiptError(
                f"{context} disabled XLA tokens are not the exact default extension"
            )
    cache_roots = [attempt["compilation_cache_root"] for attempt in all_attempts]
    artifact_roots = [attempt["artifact_root"] for attempt in all_attempts]
    if len(set(cache_roots)) != 6 or len(set(artifact_roots)) != 6:
        raise Phase0ReceiptError(f"{context} attempt roots are not isolated")

    stability = _mapping(document["stability"], f"{context}.stability")
    _exact_keys(
        stability,
        frozenset(
            {
                "required_attempts_per_mode",
                "max_phase_total_variation_distance",
                "observed_default_phase_total_variation_distance",
                "observed_disabled_phase_total_variation_distance",
                "max_default_phase_envelope_total_variation_distance",
                "observed_default_phase_envelope_total_variation_distance",
                "max_default_device_active_share_spread",
                "observed_default_device_active_share_spread",
                "minimum_attribution_coverage",
                "observed_minimum_default_attribution_coverage",
                "observed_minimum_disabled_attribution_coverage",
                "direct_default_route",
                "disabled_transfer_fallback",
            }
        ),
        f"{context}.stability",
    )
    if (
        _integer(
            stability["required_attempts_per_mode"],
            f"{context}.stability.required_attempts_per_mode",
        )
        != 3
    ):
        raise Phase0ReceiptError(f"{context}.stability requires three attempts")
    for field in (
        "max_phase_total_variation_distance",
        "observed_default_phase_total_variation_distance",
        "observed_disabled_phase_total_variation_distance",
        "max_default_phase_envelope_total_variation_distance",
        "observed_default_phase_envelope_total_variation_distance",
        "max_default_device_active_share_spread",
        "observed_default_device_active_share_spread",
        "minimum_attribution_coverage",
        "observed_minimum_default_attribution_coverage",
        "observed_minimum_disabled_attribution_coverage",
    ):
        _fraction(stability[field], f"{context}.stability.{field}")
    for route_field in ("direct_default_route", "disabled_transfer_fallback"):
        route_context = f"{context}.stability.{route_field}"
        route = _mapping(stability[route_field], route_context)
        _exact_keys(route, frozenset({"eligible", "blockers"}), route_context)
        _boolean(route["eligible"], f"{route_context}.eligible")
        blockers = _sequence(route["blockers"], f"{route_context}.blockers")
        if any(not isinstance(blocker, str) or not blocker for blocker in blockers):
            raise Phase0ReceiptError(f"{route_context}.blockers are invalid")

    selection = _mapping(
        document["selected_attribution"], f"{context}.selected_attribution"
    )
    route = _literal(
        selection.get("route"),
        frozenset({"direct_default", "disabled_transfer_fallback"}),
        f"{context}.selected_attribution.route",
    )
    common_keys = {
        "route",
        "method",
        "phase_shares",
        "unattributed_default_envelope_share",
    }
    if route == "direct_default":
        _exact_keys(
            selection, frozenset(common_keys), f"{context}.selected_attribution"
        )
        _literal(
            selection["method"],
            frozenset({"direct-default-median-phase-envelope-share"}),
            f"{context}.selected_attribution.method",
        )
    else:
        _exact_keys(
            selection,
            frozenset({*common_keys, "default_device_active_share"}),
            f"{context}.selected_attribution",
        )
        _literal(
            selection["method"],
            frozenset({"disabled-device-fraction-times-default-device-active-share"}),
            f"{context}.selected_attribution.method",
        )
    rows = _sequence(
        selection["phase_shares"], f"{context}.selected_attribution.phase_shares"
    )
    if not rows:
        raise Phase0ReceiptError(
            f"{context}.selected_attribution.phase_shares must be non-empty"
        )
    phase_shares: dict[str, float] = {}
    for index, raw_row in enumerate(rows):
        row_context = f"{context}.selected_attribution.phase_shares[{index}]"
        row = _mapping(raw_row, row_context)
        expected_row_keys = {"phase_id", "selected_default_envelope_share"}
        if route == "disabled_transfer_fallback":
            expected_row_keys.add("disabled_device_fraction")
        _exact_keys(row, frozenset(expected_row_keys), row_context)
        phase_id = _literal(
            row["phase_id"], _ALLOWED_PROFILE_PHASES, f"{row_context}.phase_id"
        )
        if phase_id in phase_shares:
            raise Phase0ReceiptError(f"{row_context}.phase_id is duplicated")
        selected_share = _fraction(
            row["selected_default_envelope_share"],
            f"{row_context}.selected_default_envelope_share",
        )
        if route == "direct_default":
            attempts_for_phase = parsed_attempts["direct_default_measurement"]
            expected_share = statistics.median(
                _integer(
                    next(
                        phase["duration_ns"]
                        for phase in _sequence(
                            attempt["phase_device_ns"], "direct phase rows"
                        )
                        if _mapping(phase, "direct phase row")["phase_id"] == phase_id
                    ),
                    f"{row_context}.direct_duration_ns",
                )
                / _integer(
                    attempt["evaluation_envelope_ns"],
                    f"{row_context}.evaluation_envelope_ns",
                    minimum=1,
                )
                for attempt in attempts_for_phase
            )
        else:
            default_device_active_share = _fraction(
                selection["default_device_active_share"],
                f"{context}.selected_attribution.default_device_active_share",
            )
            expected_default_active_share = statistics.median(
                _integer(attempt["device_active_ns"], "default device-active duration")
                / _integer(
                    attempt["evaluation_envelope_ns"], "default envelope duration"
                )
                for attempt in parsed_attempts["direct_default_measurement"]
            )
            _require_close(
                default_device_active_share,
                expected_default_active_share,
                f"{context}.selected_attribution.default_device_active_share",
            )
            disabled_device_fraction = _fraction(
                row["disabled_device_fraction"],
                f"{row_context}.disabled_device_fraction",
            )
            expected_disabled_fraction = statistics.median(
                _integer(
                    next(
                        phase["duration_ns"]
                        for phase in _sequence(
                            attempt["phase_device_ns"], "disabled phase rows"
                        )
                        if _mapping(phase, "disabled phase row")["phase_id"] == phase_id
                    ),
                    f"{row_context}.disabled_duration_ns",
                )
                / _integer(
                    attempt["device_active_ns"],
                    f"{row_context}.disabled_device_active_ns",
                    minimum=1,
                )
                for attempt in parsed_attempts["attribution_replay"]
            )
            _require_close(
                disabled_device_fraction,
                expected_disabled_fraction,
                f"{row_context}.disabled_device_fraction",
            )
            expected_share = expected_disabled_fraction * expected_default_active_share
        _require_close(
            selected_share,
            expected_share,
            f"{row_context}.selected_default_envelope_share",
        )
        phase_shares[phase_id] = selected_share
    unattributed = _fraction(
        selection["unattributed_default_envelope_share"],
        f"{context}.selected_attribution.unattributed_default_envelope_share",
    )
    _require_close(
        sum(phase_shares.values()) + unattributed,
        1.0,
        f"{context}.selected_attribution phase shares",
    )
    return phase_shares


def _validate_newton_telemetry(
    value: object,
    context: str,
    *,
    artifact_root: Path | None = None,
    expected_identity: Mapping[str, object] | None = None,
) -> None:
    document = _mapping(value, context)
    artifact_binding_fields = (
        frozenset({"raw_evidence_relative_path", "raw_evidence_file_sha256"})
        if expected_identity is not None
        else frozenset()
    )
    _exact_keys(
        document,
        frozenset(
            {
                "telemetry_schema_id",
                "route_id",
                "measurement_method",
                "host_callback_used",
                "raw_evidence_sha256",
                "residual_evaluations",
                "linear_operator_applications",
                "observed_wall_ns",
                "unobserved_wall_ns",
                "observer_effect_ratio",
                "collected_outside_timed_samples",
            }
        )
        | artifact_binding_fields,
        context,
    )
    _literal(
        document["telemetry_schema_id"],
        frozenset({"single-stage-compute-graph-newton-telemetry-v2"}),
        f"{context}.telemetry_schema_id",
    )
    _literal(
        document["route_id"],
        frozenset({"production-exact-newton"}),
        f"{context}.route_id",
    )
    _literal(
        document["measurement_method"],
        frozenset({"device_resident_fixed_shape_exact_newton_counts"}),
        f"{context}.measurement_method",
    )
    if _boolean(document["host_callback_used"], f"{context}.host_callback_used"):
        raise Phase0ReceiptError(f"{context} must not use a host callback")
    _sha256(document["raw_evidence_sha256"], f"{context}.raw_evidence_sha256")
    residual_evaluations = _integer(
        document["residual_evaluations"], f"{context}.residual_evaluations", minimum=1
    )
    linear_operator_applications = _integer(
        document["linear_operator_applications"],
        f"{context}.linear_operator_applications",
        minimum=1,
    )
    if residual_evaluations < linear_operator_applications + 2:
        raise Phase0ReceiptError(
            f"{context} residual evaluations omit initial or candidate work"
        )
    observed = _integer(
        document["observed_wall_ns"], f"{context}.observed_wall_ns", minimum=1
    )
    unobserved = _integer(
        document["unobserved_wall_ns"], f"{context}.unobserved_wall_ns", minimum=1
    )
    ratio = _number(
        document["observer_effect_ratio"], f"{context}.observer_effect_ratio"
    )
    _require_close(ratio, observed / unobserved, f"{context}.observer_effect_ratio")
    if not _boolean(
        document["collected_outside_timed_samples"],
        f"{context}.collected_outside_timed_samples",
    ):
        raise Phase0ReceiptError(f"{context} telemetry must be outside timed samples")
    if expected_identity is None:
        return
    relative_path = _safe_relative_path(
        document["raw_evidence_relative_path"],
        f"{context}.raw_evidence_relative_path",
    )
    file_sha256 = _sha256(
        document["raw_evidence_file_sha256"],
        f"{context}.raw_evidence_file_sha256",
    )
    if artifact_root is None:
        return
    evidence_path = (artifact_root / relative_path).resolve()
    resolved_root = artifact_root.resolve()
    if not evidence_path.is_relative_to(resolved_root):
        raise Phase0ReceiptError(
            f"{context}.raw_evidence_relative_path escapes receipt artifact root"
        )
    try:
        raw_bytes = evidence_path.read_bytes()
    except FileNotFoundError as error:
        raise Phase0ReceiptError(
            f"{context} raw telemetry artifact is missing"
        ) from error
    if hashlib.sha256(raw_bytes).hexdigest() != file_sha256:
        raise Phase0ReceiptError(
            f"{context} raw telemetry artifact byte digest differs"
        )
    raw_document = _load_canonical_artifact_json(
        raw_bytes, f"{context} raw telemetry artifact"
    )
    from benchmarks.single_stage_compute_graph_newton_telemetry import (
        NewtonTelemetryError,
        TelemetryIdentity,
        validate_newton_telemetry_evidence,
    )

    try:
        expected_lane = expected_identity["lane_id"]
        lane_id: LaneId = RTX_LANE_ID if expected_lane == RTX_LANE_ID else A100_LANE_ID
        raw_fields = validate_newton_telemetry_evidence(
            raw_document,
            TelemetryIdentity(
                candidate_sha256=str(expected_identity["candidate_sha256"]),
                specimen_sha256=str(expected_identity["specimen_sha256"]),
                input_bundle_sha256=str(expected_identity["input_bundle_sha256"]),
                source_sha256=str(expected_identity["source_sha256"]),
                runtime_identity_sha256=str(
                    expected_identity["production_runtime_identity_sha256"]
                ),
                lane_id=lane_id,
                gpu_uuid=str(expected_identity["gpu_uuid"]),
                gate_checkpoint_sha256=str(expected_identity["gate_checkpoint_sha256"]),
                warm_checkpoint_sha256=str(expected_identity["warm_checkpoint_sha256"]),
                warm_p50_ns=float(expected_identity["warm_p50_ns"]),
            ),
        )
    except NewtonTelemetryError as error:
        raise Phase0ReceiptError(
            f"{context} raw telemetry artifact is invalid: {error}"
        ) from error
    expected_fields = {
        **dict(raw_fields),
        "raw_evidence_relative_path": relative_path,
        "raw_evidence_file_sha256": file_sha256,
    }
    if dict(document) != expected_fields:
        raise Phase0ReceiptError(
            f"{context} fields differ from bound raw telemetry artifact"
        )


def _validate_cold_compile(value: object, context: str) -> str:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "wall_ns",
                "peak_process_tree_rss_bytes",
                "sampled_process_gpu_memory_peak_bytes",
                "sampled_process_gpu_memory_source",
                "hlo_module_set_identity",
                "hlo_module_set_identity_source",
            }
        ),
        context,
    )
    for field in (
        "wall_ns",
        "peak_process_tree_rss_bytes",
        "sampled_process_gpu_memory_peak_bytes",
    ):
        _integer(document[field], f"{context}.{field}", minimum=1)
    _literal(
        document["sampled_process_gpu_memory_source"],
        frozenset({SAMPLED_PROCESS_GPU_MEMORY_SOURCE}),
        f"{context}.sampled_process_gpu_memory_source",
    )
    _literal(
        document["hlo_module_set_identity_source"],
        frozenset({HLO_MODULE_SET_IDENTITY_SOURCE}),
        f"{context}.hlo_module_set_identity_source",
    )
    return _hlo_module_set_identity(
        document["hlo_module_set_identity"], f"{context}.hlo_module_set_identity"
    )


def _validate_gap_budget(
    value: object,
    *,
    warm_p50_ns: float,
    attribution_coverage: float,
    phase_wall_shares: Mapping[str, float],
    context: str,
) -> bool:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "candidate_value_and_gradient_reference_timings_ns",
                "matched_complete_path_reference_timings_ns",
                "c0_complete_path_value_and_gradient_evaluation_count",
                "c0_complete_path_value_and_gradient_evaluation_count_semantics",
                "formal_target_factor",
                "formal_target_ns",
                "projection_method",
                "candidate_phases",
                "unattributed_share",
                "unattributed_conservative_reduction",
                "unattributed_optimistic_reduction",
                "candidate_value_and_gradient_conservative_projected_ns",
                "candidate_value_and_gradient_optimistic_projected_ns",
                "conservative_complete_path_projected_ns",
                "optimistic_complete_path_projected_ns",
                "faithful_levers",
                "all_faithful_levers_bounded",
                "target_reachable_optimistically",
                "reduction_evidence_kind",
                "claim_ceiling",
                "empirical_canary_bindings",
                "pivot_fired",
            }
        ),
        context,
    )
    candidate_references = _mapping(
        document["candidate_value_and_gradient_reference_timings_ns"],
        f"{context}.candidate_value_and_gradient_reference_timings_ns",
    )
    _exact_keys(
        candidate_references,
        frozenset({"c0_warm_p50"}),
        f"{context}.candidate_value_and_gradient_reference_timings_ns",
    )
    candidate_c0 = _number(
        candidate_references["c0_warm_p50"],
        f"{context}.candidate_value_and_gradient_reference_timings_ns.c0_warm_p50",
        minimum=1,
    )
    _require_close(
        candidate_c0,
        warm_p50_ns,
        f"{context}.candidate_value_and_gradient_reference_timings_ns.c0_warm_p50",
    )
    complete_references = _mapping(
        document["matched_complete_path_reference_timings_ns"],
        f"{context}.matched_complete_path_reference_timings_ns",
    )
    _exact_keys(
        complete_references,
        frozenset({"native", "c0", "optax"}),
        f"{context}.matched_complete_path_reference_timings_ns",
    )
    native = _number(
        complete_references["native"],
        f"{context}.matched_complete_path_reference_timings_ns.native",
        minimum=1,
    )
    complete_c0 = _number(
        complete_references["c0"],
        f"{context}.matched_complete_path_reference_timings_ns.c0",
        minimum=1,
    )
    optax = _number(
        complete_references["optax"],
        f"{context}.matched_complete_path_reference_timings_ns.optax",
        minimum=1,
    )
    if complete_c0 <= candidate_c0:
        raise Phase0ReceiptError(
            f"{context} matched C0 complete path must exceed its candidate V&G envelope"
        )
    evaluation_count = _integer(
        document["c0_complete_path_value_and_gradient_evaluation_count"],
        f"{context}.c0_complete_path_value_and_gradient_evaluation_count",
        minimum=1,
    )
    _literal(
        document["c0_complete_path_value_and_gradient_evaluation_count_semantics"],
        frozenset(
            {
                "scipy_optimize_result_nfev_for_combined_objective_and_gradient_callable_within_complete_path_boundary"
            }
        ),
        f"{context}.c0_complete_path_value_and_gradient_evaluation_count_semantics",
    )
    if candidate_c0 * evaluation_count > complete_c0:
        raise Phase0ReceiptError(
            f"{context} C0 V&G evaluation envelope exceeds matched complete path"
        )
    factor = _number(
        document["formal_target_factor"],
        f"{context}.formal_target_factor",
        minimum=0,
    )
    _require_close(
        factor, FORMAL_COMPLETE_PATH_FACTOR, f"{context}.formal_target_factor"
    )
    formal_target = _number(
        document["formal_target_ns"], f"{context}.formal_target_ns", minimum=1
    )
    _require_close(
        formal_target,
        FORMAL_COMPLETE_PATH_FACTOR * min(native, optax),
        f"{context}.formal_target_ns",
    )
    _literal(
        document["projection_method"],
        frozenset(
            {
                "candidate_value_and_gradient_savings_subtracted_from_matched_c0_complete_path"
            }
        ),
        f"{context}.projection_method",
    )
    phases = _sequence(document["candidate_phases"], f"{context}.candidate_phases")
    if not phases:
        raise Phase0ReceiptError(f"{context}.candidate_phases must be non-empty")
    phase_ids: list[str] = []
    conservative_saving = 0.0
    optimistic_saving = 0.0
    measured_share = 0.0
    for index, raw_phase in enumerate(phases):
        phase_context = f"{context}.candidate_phases[{index}]"
        phase = _mapping(raw_phase, phase_context)
        _exact_keys(
            phase,
            frozenset(
                {
                    "phase_id",
                    "measured_share",
                    "conservative_reduction",
                    "optimistic_reduction",
                    "overlap_disposition",
                }
            ),
            phase_context,
        )
        phase_id = _literal(
            phase["phase_id"], _ALLOWED_PROFILE_PHASES, f"{phase_context}.phase_id"
        )
        phase_ids.append(phase_id)
        share = _fraction(phase["measured_share"], f"{phase_context}.measured_share")
        expected_share = phase_wall_shares.get(phase_id)
        if expected_share is None:
            raise Phase0ReceiptError(
                f"{phase_context}.phase_id is absent from profile intervals"
            )
        _require_close(share, expected_share, f"{phase_context}.measured_share")
        conservative = _fraction(
            phase["conservative_reduction"], f"{phase_context}.conservative_reduction"
        )
        optimistic = _fraction(
            phase["optimistic_reduction"], f"{phase_context}.optimistic_reduction"
        )
        if conservative > optimistic:
            raise Phase0ReceiptError(
                f"{phase_context} conservative reduction exceeds optimistic reduction"
            )
        _literal(
            phase["overlap_disposition"],
            frozenset({"disjoint", "excluded_overlap"}),
            f"{phase_context}.overlap_disposition",
        )
        if phase["overlap_disposition"] == "excluded_overlap" and share != 0.0:
            raise Phase0ReceiptError(
                f"{phase_context} excluded overlap must have zero measured share"
            )
        measured_share += share
        conservative_saving += share * conservative
        optimistic_saving += share * optimistic
    if len(phase_ids) != len(set(phase_ids)):
        raise Phase0ReceiptError(
            f"{context}.candidate_phases contains duplicate phases"
        )
    if frozenset(phase_ids) != frozenset(phase_wall_shares):
        raise Phase0ReceiptError(
            f"{context}.candidate_phases must cover every profiled phase"
        )
    unattributed_share = _fraction(
        document["unattributed_share"], f"{context}.unattributed_share"
    )
    _require_close(
        unattributed_share, 1.0 - measured_share, f"{context}.unattributed_share"
    )
    _require_close(
        measured_share + unattributed_share,
        1.0,
        f"{context} phase shares plus unattributed share",
    )
    unattributed_conservative = _fraction(
        document["unattributed_conservative_reduction"],
        f"{context}.unattributed_conservative_reduction",
    )
    unattributed_optimistic = _fraction(
        document["unattributed_optimistic_reduction"],
        f"{context}.unattributed_optimistic_reduction",
    )
    if unattributed_conservative > unattributed_optimistic:
        raise Phase0ReceiptError(
            f"{context} unattributed conservative bound exceeds optimistic bound"
        )
    conservative_saving += unattributed_share * unattributed_conservative
    optimistic_saving += unattributed_share * unattributed_optimistic
    if optimistic_saving >= 1.0 or conservative_saving >= 1.0:
        raise Phase0ReceiptError(f"{context} projected savings must remain below one")
    candidate_conservative_projected = _number(
        document["candidate_value_and_gradient_conservative_projected_ns"],
        f"{context}.candidate_value_and_gradient_conservative_projected_ns",
        minimum=1,
    )
    candidate_optimistic_projected = _number(
        document["candidate_value_and_gradient_optimistic_projected_ns"],
        f"{context}.candidate_value_and_gradient_optimistic_projected_ns",
        minimum=1,
    )
    _require_close(
        candidate_conservative_projected,
        candidate_c0 * (1.0 - conservative_saving),
        f"{context}.candidate_value_and_gradient_conservative_projected_ns",
    )
    _require_close(
        candidate_optimistic_projected,
        candidate_c0 * (1.0 - optimistic_saving),
        f"{context}.candidate_value_and_gradient_optimistic_projected_ns",
    )
    conservative_complete_projected = _number(
        document["conservative_complete_path_projected_ns"],
        f"{context}.conservative_complete_path_projected_ns",
        minimum=1,
    )
    optimistic_complete_projected = _number(
        document["optimistic_complete_path_projected_ns"],
        f"{context}.optimistic_complete_path_projected_ns",
        minimum=1,
    )
    _require_close(
        conservative_complete_projected,
        complete_c0
        - evaluation_count * (candidate_c0 - candidate_conservative_projected),
        f"{context}.conservative_complete_path_projected_ns",
    )
    _require_close(
        optimistic_complete_projected,
        complete_c0
        - evaluation_count * (candidate_c0 - candidate_optimistic_projected),
        f"{context}.optimistic_complete_path_projected_ns",
    )
    levers = _sequence(document["faithful_levers"], f"{context}.faithful_levers")
    if not levers:
        raise Phase0ReceiptError(f"{context}.faithful_levers must be non-empty")
    lever_ids: list[str] = []
    computed_all_bounded = True
    for index, raw_lever in enumerate(levers):
        lever_context = f"{context}.faithful_levers[{index}]"
        lever = _mapping(raw_lever, lever_context)
        _exact_keys(
            lever,
            frozenset({"lever_id", "disposition", "evidence_sha256"}),
            lever_context,
        )
        lever_ids.append(_string(lever["lever_id"], f"{lever_context}.lever_id"))
        disposition = _literal(
            lever["disposition"],
            frozenset({"measured", "bounded", "stopped", "unbounded"}),
            f"{lever_context}.disposition",
        )
        computed_all_bounded = computed_all_bounded and disposition != "unbounded"
        _sha256(lever["evidence_sha256"], f"{lever_context}.evidence_sha256")
    if len(lever_ids) != len(set(lever_ids)):
        raise Phase0ReceiptError(f"{context}.faithful_levers contains duplicates")
    all_bounded = _boolean(
        document["all_faithful_levers_bounded"],
        f"{context}.all_faithful_levers_bounded",
    )
    if all_bounded != computed_all_bounded:
        raise Phase0ReceiptError(
            f"{context}.all_faithful_levers_bounded is not recomputable"
        )
    reachable = _boolean(
        document["target_reachable_optimistically"],
        f"{context}.target_reachable_optimistically",
    )
    if reachable != (optimistic_complete_projected <= formal_target):
        raise Phase0ReceiptError(
            f"{context}.target_reachable_optimistically is inconsistent"
        )
    _literal(
        document["reduction_evidence_kind"],
        frozenset({"theoretical_policy_assumptions"}),
        f"{context}.reduction_evidence_kind",
    )
    _literal(
        document["claim_ceiling"],
        frozenset({"DIAGNOSTIC_ONLY_NO_PIVOT_AUTHORITY"}),
        f"{context}.claim_ceiling",
    )
    canary_bindings = _sequence(
        document["empirical_canary_bindings"],
        f"{context}.empirical_canary_bindings",
    )
    if canary_bindings:
        raise Phase0ReceiptError(
            f"{context}.empirical_canary_bindings must be empty in Phase 0"
        )
    pivot = _boolean(document["pivot_fired"], f"{context}.pivot_fired")
    if pivot:
        raise Phase0ReceiptError(
            f"{context}.pivot_fired cannot be asserted from diagnostic Phase 0 assumptions"
        )
    return pivot


def _validate_measurement(
    value: object,
    *,
    lane_id: LaneId,
    specimen: Mapping[str, object],
    specimen_sha256: str,
    artifact_root: Path | None,
    context: str,
) -> tuple[str, str, str, float, bool]:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "variant",
                "specimen_sha256",
                "provenance",
                "first_evaluation_gate",
                "cold_compile",
                "warm_measurement",
                "profile",
                "attribution_control",
                "command_buffer",
                "newton_telemetry",
                "gap_budget",
            }
        ),
        context,
    )
    _literal(document["variant"], frozenset({"C0"}), f"{context}.variant")
    if (
        _sha256(document["specimen_sha256"], f"{context}.specimen_sha256")
        != specimen_sha256
    ):
        raise Phase0ReceiptError(
            f"{context}.specimen_sha256 differs from frozen specimen"
        )
    source_state, device_uuid, cache_directory = _validate_provenance(
        document["provenance"], lane_id, f"{context}.provenance"
    )
    _validate_first_evaluation(
        document["first_evaluation_gate"], specimen, f"{context}.first_evaluation_gate"
    )
    cold_identity = _validate_cold_compile(
        document["cold_compile"], f"{context}.cold_compile"
    )
    warm_p50 = _validate_warm_measurement(
        document["warm_measurement"], f"{context}.warm_measurement"
    )
    coverage, _gap_share, profile_identity, _direct_phase_wall_shares = (
        _validate_profile(document["profile"], f"{context}.profile")
    )
    if profile_identity != cold_identity:
        raise Phase0ReceiptError(
            f"{context} profile HLO module-set identity differs from cold identity"
        )
    _validate_command_buffer(document["command_buffer"], f"{context}.command_buffer")
    transferred_phase_shares = _validate_attribution_control(
        document["attribution_control"],
        lane_id=lane_id,
        specimen=specimen,
        specimen_sha256=specimen_sha256,
        source_state_sha256=source_state,
        device_uuid=device_uuid,
        warm_p50_ns=warm_p50,
        context=f"{context}.attribution_control",
    )
    production_binding = _mapping(
        _mapping(document["attribution_control"], f"{context}.attribution_control")[
            "production_binding"
        ],
        f"{context}.attribution_control.production_binding",
    )
    _validate_newton_telemetry(
        document["newton_telemetry"],
        f"{context}.newton_telemetry",
        artifact_root=artifact_root,
        expected_identity=production_binding,
    )
    pivot = _validate_gap_budget(
        document["gap_budget"],
        warm_p50_ns=warm_p50,
        attribution_coverage=coverage,
        phase_wall_shares=transferred_phase_shares,
        context=f"{context}.gap_budget",
    )
    return source_state, device_uuid, cache_directory, warm_p50, pivot


def _lane_id(value: object, context: str) -> LaneId:
    checked = _literal(value, frozenset({RTX_LANE_ID, A100_LANE_ID}), context)
    return RTX_LANE_ID if checked == RTX_LANE_ID else A100_LANE_ID


def _validate_phase0_receipt(
    document: object, *, artifact_root: Path | None
) -> Phase0ReceiptAudit:
    """Validate all receipt bytes and recompute every Phase 0 decision gate."""

    root = _mapping(document, "receipt")
    _exact_keys(
        root,
        frozenset(
            {
                "schema_id",
                "artifact_id",
                "evidence_kind",
                "lane_aggregation_policy",
                "specimen",
                "specimen_sha256",
                "lanes",
            }
        ),
        "receipt",
    )
    _literal(root["schema_id"], frozenset({PHASE0_SCHEMA_ID}), "receipt.schema_id")
    artifact_id = _string(root["artifact_id"], "receipt.artifact_id")
    _literal(
        root["evidence_kind"],
        frozenset({"compute_graph_engineering_phase0_noncampaign"}),
        "receipt.evidence_kind",
    )
    _literal(
        root["lane_aggregation_policy"],
        frozenset({LANE_AGGREGATION_POLICY}),
        "receipt.lane_aggregation_policy",
    )
    specimen = _mapping(root["specimen"], "receipt.specimen")
    specimen_sha = _validate_specimen(specimen, "receipt.specimen")
    if _sha256(root["specimen_sha256"], "receipt.specimen_sha256") != specimen_sha:
        raise Phase0ReceiptError(
            "receipt.specimen_sha256 does not match specimen bytes"
        )
    lanes = _sequence(root["lanes"], "receipt.lanes")
    if len(lanes) != 2:
        raise Phase0ReceiptError("receipt.lanes must contain exactly RTX 5090 and A100")
    audits: dict[LaneId, LaneAudit] = {}
    measured_bindings: list[tuple[str, str, str]] = []
    for index, raw_lane in enumerate(lanes):
        lane_context = f"receipt.lanes[{index}]"
        lane = _mapping(raw_lane, lane_context)
        _exact_keys(
            lane,
            frozenset({"lane_id", "device_class", "qualification", "measurement"}),
            lane_context,
        )
        lane_id = _lane_id(lane["lane_id"], f"{lane_context}.lane_id")
        if lane_id in audits:
            raise Phase0ReceiptError(f"receipt.lanes contains duplicate {lane_id!r}")
        expected_device_class = (
            "NVIDIA GeForce RTX 5090" if lane_id == RTX_LANE_ID else "NVIDIA A100"
        )
        _literal(
            lane["device_class"],
            frozenset({expected_device_class}),
            f"{lane_context}.device_class",
        )
        outcome = _validate_qualification(
            lane["qualification"], lane_id, f"{lane_context}.qualification"
        )
        measurement = lane["measurement"]
        if outcome == "blocked":
            if lane_id == RTX_LANE_ID:
                raise Phase0ReceiptError("RTX 5090 Phase 0 lane may not be blocked")
            if measurement is not None:
                raise Phase0ReceiptError(
                    "blocked A100 lane must contain no timing measurement"
                )
            audits[lane_id] = LaneAudit(lane_id, "blocked", None, None, None)
            continue
        if measurement is None:
            raise Phase0ReceiptError(
                f"qualified {lane_id} lane requires a C0 measurement"
            )
        source_state, device_uuid, cache_directory, p50, pivot = _validate_measurement(
            measurement,
            lane_id=lane_id,
            specimen=specimen,
            specimen_sha256=specimen_sha,
            artifact_root=artifact_root,
            context=f"{lane_context}.measurement",
        )
        measured_bindings.append((source_state, device_uuid, cache_directory))
        audits[lane_id] = LaneAudit(lane_id, "qualified", device_uuid, p50, pivot)
    if frozenset(audits) != frozenset({RTX_LANE_ID, A100_LANE_ID}):
        raise Phase0ReceiptError(
            "receipt.lanes must identify RTX 5090 and A100 exactly once"
        )
    if len(measured_bindings) == 2:
        source_states = {binding[0] for binding in measured_bindings}
        device_uuids = {binding[1] for binding in measured_bindings}
        cache_directories = {binding[2] for binding in measured_bindings}
        if len(source_states) != 1:
            raise Phase0ReceiptError(
                "RTX 5090 and A100 measurements use different source states"
            )
        if len(device_uuids) != 2:
            raise Phase0ReceiptError(
                "RTX 5090 and A100 measurements must use distinct devices"
            )
        if len(cache_directories) != 2:
            raise Phase0ReceiptError(
                "RTX 5090 and A100 compilation caches must remain separate"
            )
    return Phase0ReceiptAudit(
        artifact_id=artifact_id,
        specimen_sha256=specimen_sha,
        rtx=audits[RTX_LANE_ID],
        a100=audits[A100_LANE_ID],
    )


def validate_phase0_receipt(document: object) -> Phase0ReceiptAudit:
    """Validate the receipt schema and recompute every internal Phase 0 gate."""

    return _validate_phase0_receipt(document, artifact_root=None)


def write_phase0_receipt(path: Path, document: object) -> Phase0ReceiptAudit:
    """Validate and exclusively publish one canonical receipt file."""

    audit = _validate_phase0_receipt(document, artifact_root=path.parent)
    with path.open("xb") as stream:
        stream.write(canonical_json_bytes(document))
    return audit


def load_phase0_receipt(path: Path) -> tuple[Mapping[str, object], Phase0ReceiptAudit]:
    """Load JSON without duplicate keys and validate it before returning."""

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise Phase0ReceiptError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                Phase0ReceiptError(f"non-finite JSON constant {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise Phase0ReceiptError(f"receipt is not valid JSON: {error}") from error
    checked = _mapping(document, "receipt")
    return checked, _validate_phase0_receipt(checked, artifact_root=path.parent)


__all__ = (
    "A100_LANE_ID",
    "COIL_DOF_COUNT",
    "FIRST_EVALUATION_LIMIT_NS",
    "FORMAL_COMPLETE_PATH_FACTOR",
    "HLO_MODULE_SET_IDENTITY_PREFIX",
    "HLO_MODULE_SET_IDENTITY_SOURCE",
    "LANE_AGGREGATION_POLICY",
    "PHASE0_GRADIENT_ATOL",
    "PHASE0_GRADIENT_RTOL",
    "PHASE0_OBJECTIVE_ATOL",
    "PHASE0_OBJECTIVE_RTOL",
    "PHASE0_SCHEMA_ID",
    "REQUIRED_WARM_SAMPLES",
    "RTX_LANE_ID",
    "SAMPLED_PROCESS_GPU_MEMORY_SOURCE",
    "STATE_DIMENSION",
    "LaneAudit",
    "Phase0ReceiptAudit",
    "Phase0ReceiptError",
    "canonical_hlo_module_set_identity",
    "canonical_json_bytes",
    "canonical_sha256",
    "load_phase0_receipt",
    "validate_phase0_receipt",
    "write_phase0_receipt",
)
