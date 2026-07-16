"""Run and adjudicate one native-CPU/JAX-CPU/JAX-CUDA optimization triplet.

The three lanes execute as isolated child processes from one immutable seed/config.
This module owns provenance, resource accounting, and fail-closed parity policy;
the child drivers own only their backend-specific optimization implementation.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
NATIVE_DRIVER = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_native.py"
)
JAX_DRIVER = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)

OBJECTIVE_CONTRACT_ID = "banana-single-stage-common-v1"
ORDERED_TERMS = (
    "non_quasisymmetric_ratio",
    "boozer_residual",
    "iota",
    "length_max",
    "coil_coil_distance",
    "coil_surface_distance",
    "curvature",
)
TERM_WEIGHTS: dict[str, float] = {
    "non_quasisymmetric_ratio": 1.0,
    "boozer_residual": 1.0e3,
    "iota": 1.0e4,
    "length_max": 5.0e-2,
    "coil_coil_distance": 1.0e6,
    "coil_surface_distance": 1.0e2,
    "curvature": 1.0e-2,
}
INACTIVE_TERM_REQUIREMENTS: dict[str, float] = {"coil_surface_distance": 0.0}
INPUT_NAMES = ("surface", "biotsavart", "boozer_state")
RECORDED_ENV_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_ENABLE_X64",
    "JAX_PLATFORMS",
    "SIMSOPT_JAX_PLATFORM",
    "SIMSOPT_JAX_BACKEND",
    "SIMSOPT_JAX_CUDA_LIBRARY_MODE",
    "SIMSOPT_MIXED_PRECISION",
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_FLAGS",
    "OMP_NUM_THREADS",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
)
LANES = ("native_cpu", "jax_cpu", "jax_gpu")
EXECUTION_POLICY = "concurrent-different-nodes"
BARRIER_PROTOCOL = "shared-ready-files-v1"
BARRIER_TIMEOUT_SECONDS = 300.0
BARRIER_POLL_SECONDS = 0.1
RUN_MANIFEST_NAME = "run_manifest.json"
RUN_MANIFEST_DIGEST_NAME = "run_manifest.sha256"
LEGACY_LANE_ALIASES = {"cpu": "native_cpu", "jax": "jax_gpu"}


class ContractError(ValueError):
    """Raised when a child artifact cannot establish the comparison contract."""


@dataclass(frozen=True)
class TermMetrics:
    name: str
    raw: float
    weight: float
    weighted: float


@dataclass(frozen=True)
class StateMetrics:
    dofs: tuple[float, ...]
    dof_count: int
    dofs_sha256: str
    objective: float
    gradient: tuple[float, ...]
    gradient_count: int
    gradient_sha256: str
    gradient_norm: float
    iota: float
    G: float
    volume: float
    terms: tuple[TermMetrics, ...]


@dataclass(frozen=True)
class ObjectiveContract:
    contract_id: str
    ordered_terms: tuple[str, ...]
    weights: dict[str, float]
    optimizer_method: str
    constraint_method: str
    dtype: str
    mixed_precision: bool
    adjoint_acceptance_policy: str
    inactive_term_requirements: dict[str, float]
    dof_names: tuple[str, ...]
    dof_count: int
    dof_names_sha256: str


@dataclass(frozen=True)
class ParsedLaneResult:
    comparison_schema_version: int
    backend: str
    precision: str
    constraint_method: str
    mixed_precision: bool
    contract: ObjectiveContract
    input_sha256: dict[str, str]
    run_config_sha256: str
    initial_state: StateMetrics
    final_state: StateMetrics
    optimizer_method: str
    optimizer_success: bool
    optimizer_iterations: int
    optimizer_evaluations: int
    optimizer_rejected_evaluations: int


@dataclass(frozen=True)
class ComparisonTolerances:
    initial_objective_rtol: float = 1.0e-8
    initial_objective_atol: float = 1.0e-10
    initial_gradient_rtol: float = 1.0e-6
    initial_gradient_atol: float = 1.0e-8
    initial_iota_atol: float = 1.0e-10
    initial_G_rtol: float = 1.0e-10
    initial_G_atol: float = 1.0e-10
    initial_volume_rtol: float = 1.0e-10
    initial_volume_atol: float = 1.0e-12
    initial_term_rtol: float = 1.0e-6
    initial_term_atol: float = 1.0e-10
    final_objective_rtol: float = 1.0e-3
    final_objective_atol: float = 1.0e-10
    final_dofs_rtol: float = 1.0e-3
    final_dofs_atol: float = 1.0e-6
    final_gradient_rtol: float = 1.0e-3
    final_gradient_atol: float = 1.0e-8
    final_iota_atol: float = 1.0e-4
    final_G_rtol: float = 1.0e-5
    final_G_atol: float = 1.0e-8
    final_volume_rtol: float = 1.0e-4
    final_volume_atol: float = 1.0e-8
    final_term_rtol: float = 1.0e-3
    final_term_atol: float = 1.0e-10


@dataclass(frozen=True)
class SourceIdentity:
    commit_sha: str
    tree_sha: str
    status_porcelain: str


@dataclass(frozen=True)
class LaneExecution:
    lane: str
    command: tuple[str, ...]
    environment: dict[str, str]
    returncode: int
    started_at_utc: str
    ended_at_utc: str
    wall_seconds: float
    host_max_rss_kib: int
    run_dir: str
    results_json: str
    results_sha256: str
    stdout_log: str
    stderr_log: str
    resource_log: str
    run_manifest_sha256: str
    assigned_node: str
    actual_node: str
    slurm_job_id: str
    slurm_step_id: str
    slurm_step_nodelist: str
    barrier_peer_observed_at_utc: dict[str, str]
    barrier_peer_slurm_job_ids: dict[str, str]


@dataclass(frozen=True)
class StepIdentity:
    assigned_node: str
    actual_node: str
    slurm_job_id: str
    slurm_step_id: str
    slurm_step_nodelist: str
    slurm_step_num_nodes: str
    slurm_node_id: str
    slurm_process_id: str


@dataclass(frozen=True)
class PreparedLaneExecution:
    """Validated immutable inputs required to execute one prepared lane."""

    command: tuple[str, ...]
    run_manifest_sha256: str
    assigned_node: str
    allocation_slurm_job_id: str | None


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    """Return the canonical SHA-256 identity used by all benchmark contracts."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_float64_sequence(values: Sequence[float]) -> str:
    """Return the child-contract hash for one flat little-endian FP64 vector."""
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<d", float(value)))
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ContractError(f"{path} must contain a JSON object")
    return value


def _require_mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, dict) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"{key} must be a JSON object")
    return value


def _require_string(parent: Mapping[str, object], key: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ContractError(f"{key} must be a non-empty string")
    return value


def _require_bool(parent: Mapping[str, object], key: str) -> bool:
    value = parent.get(key)
    if not isinstance(value, bool):
        raise ContractError(f"{key} must be a boolean")
    return value


def _require_int(parent: Mapping[str, object], key: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{key} must be an integer")
    if value < 0:
        raise ContractError(f"{key} must be non-negative")
    return value


def _require_positive_int(parent: Mapping[str, object], key: str) -> int:
    value = _require_int(parent, key)
    if value <= 0:
        raise ContractError(f"{key} must be positive")
    return value


def _require_finite_float(parent: Mapping[str, object], key: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{key} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{key} must be finite")
    return result


def _require_string_tuple(parent: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a non-empty JSON string array")
    if not all(isinstance(item, str) and item for item in value):
        raise ContractError(f"{key} must contain only non-empty strings")
    return tuple(value)


def _require_float_tuple(parent: Mapping[str, object], key: str) -> tuple[float, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ContractError(f"{key} must be a non-empty JSON number array")
    parsed: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ContractError(f"{key}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ContractError(f"{key}[{index}] must be finite")
        parsed.append(number)
    return tuple(parsed)


def _require_sha256(parent: Mapping[str, object], key: str) -> str:
    value = _require_string(parent, key)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ContractError(f"{key} must be a lowercase SHA-256 digest")
    return value


def _parse_weights(contract: Mapping[str, object]) -> dict[str, float]:
    raw_weights = _require_mapping(contract, "weights")
    weights: dict[str, float] = {}
    for name, raw_value in raw_weights.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ContractError(f"objective_contract.weights[{name!r}] must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise ContractError(f"objective_contract.weights[{name!r}] must be finite")
        weights[name] = value
    return weights


def _parse_inactive_term_requirements(
    contract: Mapping[str, object],
) -> dict[str, float]:
    raw_requirements = _require_mapping(contract, "inactive_term_requirements")
    requirements: dict[str, float] = {}
    for name, raw_value in raw_requirements.items():
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ContractError(
                f"objective_contract.inactive_term_requirements[{name!r}] must be numeric"
            )
        value = float(raw_value)
        if not math.isfinite(value):
            raise ContractError(
                f"objective_contract.inactive_term_requirements[{name!r}] must be finite"
            )
        requirements[name] = value
    return requirements


def _parse_terms(
    state: Mapping[str, object],
    *,
    state_key: str,
    ordered_terms: tuple[str, ...],
    weights: Mapping[str, float],
) -> tuple[TermMetrics, ...]:
    raw_terms = _require_mapping(state, "terms")
    if set(raw_terms) != set(ordered_terms) or len(raw_terms) != len(ordered_terms):
        raise ContractError(
            f"{state_key}.terms must contain exactly objective_contract.ordered_terms"
        )
    terms: list[TermMetrics] = []
    for name in ordered_terms:
        raw_term = _require_mapping(raw_terms, name)
        raw = _require_finite_float(raw_term, "raw")
        weight = _require_finite_float(raw_term, "weight")
        weighted = _require_finite_float(raw_term, "weighted")
        expected_weight = weights[name]
        if weight != expected_weight:
            raise ContractError(
                f"{state_key}.terms[{name!r}].weight does not match the contract"
            )
        expected_weighted = weight * raw
        if not math.isclose(
            weighted,
            expected_weighted,
            rel_tol=1.0e-13,
            abs_tol=1.0e-15,
        ):
            raise ContractError(
                f"{state_key}.terms[{name!r}].weighted is not weight * raw"
            )
        terms.append(TermMetrics(name, raw, weight, weighted))
    return tuple(terms)


def _parse_state(
    payload: Mapping[str, object],
    key: str,
    contract: ObjectiveContract,
) -> StateMetrics:
    state = _require_mapping(payload, key)
    dofs = _require_float_tuple(state, "dofs")
    dof_count = _require_int(state, "dof_count")
    dofs_sha256 = _require_sha256(state, "dofs_sha256")
    if dof_count != len(dofs) or dof_count != contract.dof_count:
        raise ContractError(
            f"{key}.dof_count must match its DOF vector and objective contract"
        )
    if dofs_sha256 != sha256_float64_sequence(dofs):
        raise ContractError(f"{key}.dofs_sha256 does not identify {key}.dofs")

    gradient = _require_float_tuple(state, "gradient")
    gradient_count = _require_int(state, "gradient_count")
    gradient_sha256 = _require_sha256(state, "gradient_sha256")
    if gradient_count != len(gradient) or gradient_count != contract.dof_count:
        raise ContractError(
            f"{key}.gradient_count must match its gradient and objective contract"
        )
    if gradient_sha256 != sha256_float64_sequence(gradient):
        raise ContractError(f"{key}.gradient_sha256 does not identify {key}.gradient")

    gradient_norm = _require_finite_float(state, "gradient_norm")
    recomputed_gradient_norm = math.hypot(*gradient)
    if not math.isclose(
        gradient_norm,
        recomputed_gradient_norm,
        rel_tol=1.0e-12,
        abs_tol=1.0e-14,
    ):
        raise ContractError(f"{key}.gradient_norm does not match {key}.gradient")

    objective = _require_finite_float(state, "objective")
    terms = _parse_terms(
        state,
        state_key=key,
        ordered_terms=contract.ordered_terms,
        weights=contract.weights,
    )
    recomputed_objective = sum(term.weighted for term in terms)
    if not math.isclose(
        objective,
        recomputed_objective,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ContractError(f"{key}.objective does not match its weighted terms")

    return StateMetrics(
        dofs=dofs,
        dof_count=dof_count,
        dofs_sha256=dofs_sha256,
        objective=objective,
        gradient=gradient,
        gradient_count=gradient_count,
        gradient_sha256=gradient_sha256,
        gradient_norm=gradient_norm,
        iota=_require_finite_float(state, "iota"),
        G=_require_finite_float(state, "G"),
        volume=_require_finite_float(state, "volume"),
        terms=terms,
    )


def parse_lane_result(payload: Mapping[str, object]) -> ParsedLaneResult:
    """Parse and validate the backend-neutral portion of one child result."""
    raw_contract = _require_mapping(payload, "objective_contract")
    dof_names = _require_string_tuple(raw_contract, "dof_names")
    contract = ObjectiveContract(
        contract_id=_require_string(raw_contract, "id"),
        ordered_terms=_require_string_tuple(raw_contract, "ordered_terms"),
        weights=_parse_weights(raw_contract),
        optimizer_method=_require_string(raw_contract, "optimizer_method"),
        constraint_method=_require_string(raw_contract, "constraint_method"),
        dtype=_require_string(raw_contract, "dtype"),
        mixed_precision=_require_bool(raw_contract, "mixed_precision"),
        adjoint_acceptance_policy=_require_string(
            raw_contract, "adjoint_acceptance_policy"
        ),
        inactive_term_requirements=_parse_inactive_term_requirements(raw_contract),
        dof_names=dof_names,
        dof_count=_require_int(raw_contract, "dof_count"),
        dof_names_sha256=_require_sha256(raw_contract, "dof_names_sha256"),
    )
    expected_dof_names_sha256 = sha256_json(list(dof_names))
    if contract.dof_count != len(dof_names):
        raise ContractError(
            "objective_contract.dof_count does not match objective_contract.dof_names"
        )
    if contract.dof_names_sha256 != expected_dof_names_sha256:
        raise ContractError(
            "objective_contract.dof_names_sha256 does not identify dof_names"
        )
    if set(contract.weights) != set(contract.ordered_terms):
        raise ContractError(
            "objective_contract.weights must contain exactly ordered_terms"
        )
    if not set(contract.inactive_term_requirements).issubset(contract.ordered_terms):
        raise ContractError(
            "objective_contract.inactive_term_requirements must name ordered terms"
        )

    raw_inputs = _require_mapping(payload, "input_sha256")
    input_sha256 = {name: _require_sha256(raw_inputs, name) for name in INPUT_NAMES}
    optimizer = _require_mapping(payload, "optimizer")
    return ParsedLaneResult(
        comparison_schema_version=_require_int(payload, "comparison_schema_version"),
        backend=_require_string(payload, "backend"),
        precision=_require_string(payload, "precision"),
        constraint_method=_require_string(payload, "constraint_method"),
        mixed_precision=_require_bool(payload, "mixed_precision"),
        contract=contract,
        input_sha256=input_sha256,
        run_config_sha256=_require_sha256(payload, "run_config_sha256"),
        initial_state=_parse_state(payload, "initial_state", contract),
        final_state=_parse_state(payload, "final_state", contract),
        optimizer_method=_require_string(optimizer, "method"),
        optimizer_success=_require_bool(optimizer, "success"),
        optimizer_iterations=_require_int(optimizer, "nit"),
        optimizer_evaluations=_require_int(optimizer, "nfev"),
        optimizer_rejected_evaluations=_require_int(optimizer, "rejected_evaluations"),
    )


def _numeric_check(
    *,
    name: str,
    cpu: float,
    jax: float,
    rtol: float,
    atol: float,
) -> dict[str, object]:
    absolute_difference = abs(cpu - jax)
    limit = atol + rtol * max(abs(cpu), abs(jax))
    return {
        "name": name,
        "cpu": cpu,
        "jax": jax,
        "absolute_difference": absolute_difference,
        "limit": limit,
        "rtol": rtol,
        "atol": atol,
        "passed": absolute_difference <= limit,
    }


def _exact_check(name: str, cpu: object, jax: object) -> dict[str, object]:
    return {
        "name": name,
        "cpu": cpu,
        "jax": jax,
        "passed": cpu == jax,
    }


def _vector_numeric_check(
    *,
    name: str,
    cpu: tuple[float, ...],
    jax: tuple[float, ...],
    rtol: float,
    atol: float,
) -> dict[str, object]:
    component_checks = [
        abs(cpu_value - jax_value) <= atol + rtol * max(abs(cpu_value), abs(jax_value))
        for cpu_value, jax_value in zip(cpu, jax, strict=False)
    ]
    return {
        "name": name,
        "cpu_count": len(cpu),
        "jax_count": len(jax),
        "rtol": rtol,
        "atol": atol,
        "passed": len(cpu) == len(jax) and all(component_checks),
    }


def _progress_check(
    *,
    name: str,
    initial: float,
    final: float,
) -> dict[str, object]:
    return {
        "name": name,
        "initial": initial,
        "final": final,
        "passed": final < initial,
    }


def _term_parity_checks(
    *,
    state_name: str,
    cpu: StateMetrics,
    jax: StateMetrics,
    rtol: float,
    atol: float,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    for cpu_term, jax_term in zip(cpu.terms, jax.terms, strict=True):
        checks.extend(
            (
                _exact_check(
                    f"{state_name}.terms.{cpu_term.name}.name",
                    cpu_term.name,
                    jax_term.name,
                ),
                _numeric_check(
                    name=f"{state_name}.terms.{cpu_term.name}.raw",
                    cpu=cpu_term.raw,
                    jax=jax_term.raw,
                    rtol=rtol,
                    atol=atol,
                ),
                _numeric_check(
                    name=f"{state_name}.terms.{cpu_term.name}.weighted",
                    cpu=cpu_term.weighted,
                    jax=jax_term.weighted,
                    rtol=rtol,
                    atol=atol,
                ),
            )
        )
    return checks


def _inactive_term_checks(
    *,
    lane: str,
    state_name: str,
    state: StateMetrics,
) -> list[dict[str, object]]:
    terms = {term.name: term for term in state.terms}
    return [
        _exact_check(
            f"{state_name}.terms.{name}.inactive.{lane}",
            terms[name].raw,
            required_value,
        )
        for name, required_value in INACTIVE_TERM_REQUIREMENTS.items()
    ]


def compare_lane_results(
    cpu: ParsedLaneResult,
    jax: ParsedLaneResult,
    *,
    expected_input_sha256: Mapping[str, str],
    expected_run_config_sha256: str,
    tolerances: ComparisonTolerances,
    mode: str = "production",
    cpu_lane_name: str = "cpu",
    jax_lane_name: str = "jax",
    expected_cpu_backend: str = "native-simsopt-cpu",
    expected_jax_backend: str = "jax-cuda",
    expected_cpu_adjoint_policy: str = "native-plu-finite-gradient",
    expected_jax_adjoint_policy: str = "checked-residual-and-condition",
) -> dict[str, object]:
    """Apply invariant gates and mode-dependent optimization-outcome policy."""
    if mode not in {"production", "diagnostic"}:
        raise ValueError(f"Unsupported comparison mode {mode!r}")

    required_checks = [
        _exact_check("comparison_schema_version.cpu", cpu.comparison_schema_version, 1),
        _exact_check("comparison_schema_version.jax", jax.comparison_schema_version, 1),
        _exact_check("backend.cpu", cpu.backend, expected_cpu_backend),
        _exact_check("backend.jax", jax.backend, expected_jax_backend),
        _exact_check("precision.cpu", cpu.precision, "float64"),
        _exact_check("precision.jax", jax.precision, "float64"),
        _exact_check("constraint_method.cpu", cpu.constraint_method, "soft-penalty"),
        _exact_check("constraint_method.jax", jax.constraint_method, "soft-penalty"),
        _exact_check("mixed_precision.cpu", cpu.mixed_precision, False),
        _exact_check("mixed_precision.jax", jax.mixed_precision, False),
        _exact_check(
            "contract.id.cpu", cpu.contract.contract_id, OBJECTIVE_CONTRACT_ID
        ),
        _exact_check(
            "contract.id.jax", jax.contract.contract_id, OBJECTIVE_CONTRACT_ID
        ),
        _exact_check(
            "contract.ordered_terms.cpu", cpu.contract.ordered_terms, ORDERED_TERMS
        ),
        _exact_check(
            "contract.ordered_terms.jax", jax.contract.ordered_terms, ORDERED_TERMS
        ),
        _exact_check("contract.weights.cpu", cpu.contract.weights, TERM_WEIGHTS),
        _exact_check("contract.weights.jax", jax.contract.weights, TERM_WEIGHTS),
        _exact_check(
            "contract.inactive_term_requirements.cpu",
            cpu.contract.inactive_term_requirements,
            INACTIVE_TERM_REQUIREMENTS,
        ),
        _exact_check(
            "contract.inactive_term_requirements.jax",
            jax.contract.inactive_term_requirements,
            INACTIVE_TERM_REQUIREMENTS,
        ),
        _exact_check(
            "contract.dof_names", cpu.contract.dof_names, jax.contract.dof_names
        ),
        _exact_check(
            "contract.dof_count", cpu.contract.dof_count, jax.contract.dof_count
        ),
        _exact_check(
            "contract.dof_names_sha256",
            cpu.contract.dof_names_sha256,
            jax.contract.dof_names_sha256,
        ),
        _exact_check(
            "contract.optimizer_method.cpu", cpu.contract.optimizer_method, "L-BFGS-B"
        ),
        _exact_check(
            "contract.optimizer_method.jax", jax.contract.optimizer_method, "L-BFGS-B"
        ),
        _exact_check("optimizer.method.cpu", cpu.optimizer_method, "L-BFGS-B"),
        _exact_check("optimizer.method.jax", jax.optimizer_method, "L-BFGS-B"),
        _exact_check(
            "contract.constraint_method.cpu",
            cpu.contract.constraint_method,
            "soft-penalty",
        ),
        _exact_check(
            "contract.constraint_method.jax",
            jax.contract.constraint_method,
            "soft-penalty",
        ),
        _exact_check("contract.dtype.cpu", cpu.contract.dtype, "float64"),
        _exact_check("contract.dtype.jax", jax.contract.dtype, "float64"),
        _exact_check(
            "contract.mixed_precision.cpu", cpu.contract.mixed_precision, False
        ),
        _exact_check(
            "contract.mixed_precision.jax", jax.contract.mixed_precision, False
        ),
        _exact_check(
            "contract.adjoint_acceptance_policy.cpu",
            cpu.contract.adjoint_acceptance_policy,
            expected_cpu_adjoint_policy,
        ),
        _exact_check(
            "contract.adjoint_acceptance_policy.jax",
            jax.contract.adjoint_acceptance_policy,
            expected_jax_adjoint_policy,
        ),
        _exact_check("inputs.cpu", cpu.input_sha256, dict(expected_input_sha256)),
        _exact_check("inputs.jax", jax.input_sha256, dict(expected_input_sha256)),
        _exact_check(
            "run_config.cpu", cpu.run_config_sha256, expected_run_config_sha256
        ),
        _exact_check(
            "run_config.jax", jax.run_config_sha256, expected_run_config_sha256
        ),
        _exact_check(
            "initial_state.dofs_sha256",
            cpu.initial_state.dofs_sha256,
            jax.initial_state.dofs_sha256,
        ),
        _exact_check(
            "initial_state.dof_count",
            cpu.initial_state.dof_count,
            jax.initial_state.dof_count,
        ),
        _numeric_check(
            name="initial_state.objective",
            cpu=cpu.initial_state.objective,
            jax=jax.initial_state.objective,
            rtol=tolerances.initial_objective_rtol,
            atol=tolerances.initial_objective_atol,
        ),
        _numeric_check(
            name="initial_state.gradient_norm",
            cpu=cpu.initial_state.gradient_norm,
            jax=jax.initial_state.gradient_norm,
            rtol=tolerances.initial_gradient_rtol,
            atol=tolerances.initial_gradient_atol,
        ),
        _vector_numeric_check(
            name="initial_state.gradient",
            cpu=cpu.initial_state.gradient,
            jax=jax.initial_state.gradient,
            rtol=tolerances.initial_gradient_rtol,
            atol=tolerances.initial_gradient_atol,
        ),
        _numeric_check(
            name="initial_state.iota",
            cpu=cpu.initial_state.iota,
            jax=jax.initial_state.iota,
            rtol=0.0,
            atol=tolerances.initial_iota_atol,
        ),
        _numeric_check(
            name="initial_state.G",
            cpu=cpu.initial_state.G,
            jax=jax.initial_state.G,
            rtol=tolerances.initial_G_rtol,
            atol=tolerances.initial_G_atol,
        ),
        _numeric_check(
            name="initial_state.volume",
            cpu=cpu.initial_state.volume,
            jax=jax.initial_state.volume,
            rtol=tolerances.initial_volume_rtol,
            atol=tolerances.initial_volume_atol,
        ),
    ]
    if (
        cpu.contract.ordered_terms == ORDERED_TERMS
        and jax.contract.ordered_terms == ORDERED_TERMS
    ):
        required_checks.extend(
            _term_parity_checks(
                state_name="initial_state",
                cpu=cpu.initial_state,
                jax=jax.initial_state,
                rtol=tolerances.initial_term_rtol,
                atol=tolerances.initial_term_atol,
            )
        )

    outcome_checks = [
        _exact_check("optimizer.success.cpu", cpu.optimizer_success, True),
        _exact_check("optimizer.success.jax", jax.optimizer_success, True),
        _exact_check(
            "optimizer.rejected_evaluations.cpu",
            cpu.optimizer_rejected_evaluations,
            0,
        ),
        _exact_check(
            "optimizer.rejected_evaluations.jax",
            jax.optimizer_rejected_evaluations,
            0,
        ),
        _progress_check(
            name="final_state.objective_progress.cpu",
            initial=cpu.initial_state.objective,
            final=cpu.final_state.objective,
        ),
        _progress_check(
            name="final_state.objective_progress.jax",
            initial=jax.initial_state.objective,
            final=jax.final_state.objective,
        ),
        _progress_check(
            name="final_state.gradient_progress.cpu",
            initial=cpu.initial_state.gradient_norm,
            final=cpu.final_state.gradient_norm,
        ),
        _progress_check(
            name="final_state.gradient_progress.jax",
            initial=jax.initial_state.gradient_norm,
            final=jax.final_state.gradient_norm,
        ),
        _numeric_check(
            name="final_state.objective",
            cpu=cpu.final_state.objective,
            jax=jax.final_state.objective,
            rtol=tolerances.final_objective_rtol,
            atol=tolerances.final_objective_atol,
        ),
        _vector_numeric_check(
            name="final_state.dofs",
            cpu=cpu.final_state.dofs,
            jax=jax.final_state.dofs,
            rtol=tolerances.final_dofs_rtol,
            atol=tolerances.final_dofs_atol,
        ),
        _numeric_check(
            name="final_state.gradient_norm",
            cpu=cpu.final_state.gradient_norm,
            jax=jax.final_state.gradient_norm,
            rtol=tolerances.final_gradient_rtol,
            atol=tolerances.final_gradient_atol,
        ),
        _vector_numeric_check(
            name="final_state.gradient",
            cpu=cpu.final_state.gradient,
            jax=jax.final_state.gradient,
            rtol=tolerances.final_gradient_rtol,
            atol=tolerances.final_gradient_atol,
        ),
        _numeric_check(
            name="final_state.iota",
            cpu=cpu.final_state.iota,
            jax=jax.final_state.iota,
            rtol=0.0,
            atol=tolerances.final_iota_atol,
        ),
        _numeric_check(
            name="final_state.G",
            cpu=cpu.final_state.G,
            jax=jax.final_state.G,
            rtol=tolerances.final_G_rtol,
            atol=tolerances.final_G_atol,
        ),
        _numeric_check(
            name="final_state.volume",
            cpu=cpu.final_state.volume,
            jax=jax.final_state.volume,
            rtol=tolerances.final_volume_rtol,
            atol=tolerances.final_volume_atol,
        ),
    ]
    if (
        cpu.contract.ordered_terms == ORDERED_TERMS
        and jax.contract.ordered_terms == ORDERED_TERMS
    ):
        outcome_checks.extend(
            _term_parity_checks(
                state_name="final_state",
                cpu=cpu.final_state,
                jax=jax.final_state,
                rtol=tolerances.final_term_rtol,
                atol=tolerances.final_term_atol,
            )
        )
        for lane, state_name, state in (
            ("cpu", "initial_state", cpu.initial_state),
            ("cpu", "final_state", cpu.final_state),
            ("jax", "initial_state", jax.initial_state),
            ("jax", "final_state", jax.final_state),
        ):
            outcome_checks.extend(
                _inactive_term_checks(
                    lane=lane,
                    state_name=state_name,
                    state=state,
                )
            )

    for check in required_checks + outcome_checks:
        name = str(check["name"])
        if name.endswith(".cpu"):
            check["name"] = f"{name[:-4]}.{cpu_lane_name}"
        elif name.endswith(".jax"):
            check["name"] = f"{name[:-4]}.{jax_lane_name}"
    for check in required_checks:
        check["category"] = "integrity"
    for check in outcome_checks:
        check["category"] = "outcome"
    required_failures = [
        str(check["name"]) for check in required_checks if not bool(check["passed"])
    ]
    outcome_failures = [
        str(check["name"]) for check in outcome_checks if not bool(check["passed"])
    ]
    failures = required_failures + (outcome_failures if mode == "production" else [])
    advisory_failures = outcome_failures if mode == "diagnostic" else []
    claim_ready = (
        mode == "production" and not required_failures and not outcome_failures
    )
    return {
        "mode": mode,
        "passed": not failures,
        "claim_ready": claim_ready,
        "failures": failures,
        "required_failures": required_failures,
        "outcome_failures": outcome_failures,
        "advisory_failures": advisory_failures,
        "checks": required_checks + outcome_checks,
        "tolerances": asdict(tolerances),
    }


def parse_gnu_time_verbose(text: str) -> int:
    """Return GNU time's maximum resident set size in KiB."""
    prefix = "Maximum resident set size (kbytes):"
    values = [
        line.split(":", 1)[1].strip()
        for line in text.splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(values) != 1:
        raise ContractError(
            "GNU time output must contain exactly one maximum-RSS record"
        )
    try:
        max_rss_kib = int(values[0])
    except ValueError as error:
        raise ContractError("GNU time maximum RSS must be an integer") from error
    if max_rss_kib <= 0:
        raise ContractError("GNU time maximum RSS must be positive")
    return max_rss_kib


def _git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def source_relevant_git_status(status_porcelain: str) -> str:
    """Exclude only Slurm's top-level scheduler logs from source cleanliness."""
    scheduler_log = re.compile(r"^\?\? slurm-[0-9]+\.(?:out|err)$")
    return "\n".join(
        line
        for line in status_porcelain.splitlines()
        if not scheduler_log.fullmatch(line)
    )


def source_identity(repo_root: Path) -> SourceIdentity:
    raw_status = _git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )
    status = source_relevant_git_status(raw_status)
    if status:
        raise RuntimeError(
            "Full-loop performance evidence requires a clean source tree; "
            f"git status reported:\n{status}"
        )
    return SourceIdentity(
        commit_sha=_git(repo_root, "rev-parse", "HEAD"),
        tree_sha=_git(repo_root, "rev-parse", "HEAD^{tree}"),
        status_porcelain=status,
    )


def _selected_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        name: environment[name] for name in RECORDED_ENV_NAMES if name in environment
    }


def _canonical_lane(lane: str) -> str:
    canonical = LEGACY_LANE_ALIASES.get(lane, lane)
    if canonical not in LANES:
        raise ValueError(f"Unsupported lane {lane!r}")
    return canonical


def lane_environment(
    lane: str, parent_environment: Mapping[str, str]
) -> dict[str, str]:
    canonical_lane = _canonical_lane(lane)
    environment = dict(parent_environment)
    environment.update(
        {
            "JAX_ENABLE_X64": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "SIMSOPT_MIXED_PRECISION": "0",
        }
    )
    if canonical_lane in {"native_cpu", "jax_cpu"}:
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "SIMSOPT_JAX_PLATFORM": "cpu",
                "SIMSOPT_JAX_BACKEND": "cpu",
            }
        )
        return environment
    visible_devices = environment.get("CUDA_VISIBLE_DEVICES", "")
    if (
        not visible_devices
        or "," in visible_devices
        or any(character.isspace() for character in visible_devices)
    ):
        raise RuntimeError(
            "JAX lane requires exactly one Slurm-assigned CUDA_VISIBLE_DEVICES selector"
        )
    environment.update(
        {
            "JAX_PLATFORMS": "cuda",
            "SIMSOPT_JAX_PLATFORM": "cuda",
            "SIMSOPT_JAX_BACKEND": "cuda",
            "SIMSOPT_JAX_CUDA_LIBRARY_MODE": "bundled",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


def _shared_configuration(
    args: argparse.Namespace,
    input_sha256: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "environment_lock_sha256": args.environment_lock_sha256,
        "objective_contract_id": OBJECTIVE_CONTRACT_ID,
        "objective_profile": "common-seven-term",
        "ordered_terms": list(ORDERED_TERMS),
        "term_weights": TERM_WEIGHTS,
        "inactive_term_requirements": INACTIVE_TERM_REQUIREMENTS,
        "comparison_mode": args.mode,
        "input_sha256": dict(input_sha256),
        "surface": {
            "vmec_s": args.vmec_s,
            "surface_scale": args.surface_scale,
            "mpol": args.mpol,
            "ntor": args.ntor,
            "nphi": args.nphi,
            "ntheta": args.ntheta,
            "constraint_weight": args.constraint_weight,
        },
        "physics": {
            "iota_target": args.iota_target,
        },
        "outer_optimizer": {
            "method": "L-BFGS-B",
            "maxiter": args.maxiter,
            "maxcor": args.outer_maxcor,
            "maxls": args.outer_maxls,
            "ftol": args.outer_ftol,
            "gtol": args.outer_gtol,
        },
        "boozer_solver": {
            "bfgs_tol": args.boozer_bfgs_tol,
            "bfgs_maxiter": args.boozer_bfgs_maxiter,
            "newton_tol": args.boozer_newton_tol,
            "newton_maxiter": args.boozer_newton_maxiter,
            "limited_memory": False,
        },
        "constraint_method": "soft-penalty",
        "dtype": "float64",
        "mixed_precision": False,
    }


def build_lane_command(
    args: argparse.Namespace,
    *,
    lane: str,
    run_dir: Path,
    run_config_sha256: str,
) -> tuple[str, ...]:
    canonical_lane = _canonical_lane(lane)
    surface_normalization_arguments: tuple[str, ...] = (
        "--vmec-s",
        str(args.vmec_s),
    )
    if args.surface_scale is not None:
        surface_normalization_arguments += (
            "--surface-scale",
            str(args.surface_scale),
        )
    if canonical_lane == "native_cpu":
        driver = NATIVE_DRIVER
        lane_arguments: Sequence[str] = ()
        output_arguments: Sequence[str] = (
            "--output-root",
            str(run_dir),
            "--overwrite",
        )
        exclusion_arguments: Sequence[str] = ()
    elif canonical_lane in {"jax_cpu", "jax_gpu"}:
        driver = JAX_DRIVER
        lane_arguments = (
            ("--backend", "cpu", "--platform", "cpu")
            if canonical_lane == "jax_cpu"
            else ("--backend", "jax", "--platform", "cuda")
        )
        output_arguments = (
            "--run-dir",
            str(run_dir),
            "--overwrite",
            "--skip-postprocess",
        )
        exclusion_arguments = (
            "--weight-poloidal-extent",
            "0.0",
            "--weight-ellipse-width",
            "0.0",
            "--weight-global-curvature-radius",
            "0.0",
            "--weight-currents",
            "0.0",
            "--include-boozer-residual",
            "--no-current-penalties",
            "--no-width",
        )
    command = (
        str(args.python),
        str(driver),
        *lane_arguments,
        *output_arguments,
        "--objective-profile",
        "common-seven-term",
        "--surface-path",
        str(args.surface_path),
        "--biotsavart-file",
        str(args.biotsavart_file),
        "--boozer-state-path",
        str(args.boozer_state_path),
        "--run-config-sha256",
        run_config_sha256,
        *surface_normalization_arguments,
        "--mpol",
        str(args.mpol),
        "--ntor",
        str(args.ntor),
        "--nphi",
        str(args.nphi),
        "--ntheta",
        str(args.ntheta),
        "--constraint-weight",
        str(args.constraint_weight),
        "--iota-target",
        str(args.iota_target),
        "--maxiter",
        str(args.maxiter),
        "--outer-maxcor",
        str(args.outer_maxcor),
        "--outer-maxls",
        str(args.outer_maxls),
        "--outer-ftol",
        str(args.outer_ftol),
        "--outer-gtol",
        str(args.outer_gtol),
        "--boozer-bfgs-tol",
        str(args.boozer_bfgs_tol),
        "--boozer-bfgs-maxiter",
        str(args.boozer_bfgs_maxiter),
        "--boozer-newton-tol",
        str(args.boozer_newton_tol),
        "--boozer-newton-maxiter",
        str(args.boozer_newton_maxiter),
        "--weight-non-quasisymmetric-ratio",
        str(TERM_WEIGHTS["non_quasisymmetric_ratio"]),
        "--weight-boozer-residual",
        str(TERM_WEIGHTS["boozer_residual"]),
        "--weight-iota",
        str(TERM_WEIGHTS["iota"]),
        "--weight-coil-length",
        str(TERM_WEIGHTS["length_max"]),
        "--weight-coil-coil-distance",
        str(TERM_WEIGHTS["coil_coil_distance"]),
        "--weight-coil-surface-distance",
        str(TERM_WEIGHTS["coil_surface_distance"]),
        "--weight-coil-curvature",
        str(TERM_WEIGHTS["curvature"]),
        *exclusion_arguments,
    )
    return command


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short_hostname(value: str, *, label: str) -> str:
    stripped = value.strip()
    if not stripped or any(character.isspace() for character in stripped):
        raise ContractError(f"{label} must be one non-empty hostname")
    short = stripped.split(".", 1)[0]
    if not short or any(character in short for character in ",[]"):
        raise ContractError(f"{label} must identify exactly one host: {value!r}")
    return short


def _raw_step_identity(assigned_node: str) -> StepIdentity:
    return StepIdentity(
        assigned_node=assigned_node,
        actual_node=socket.gethostname().split(".", 1)[0],
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        slurm_step_id=os.environ.get("SLURM_STEP_ID", ""),
        slurm_step_nodelist=os.environ.get("SLURM_STEP_NODELIST", ""),
        slurm_step_num_nodes=os.environ.get("SLURM_STEP_NUM_NODES", ""),
        slurm_node_id=os.environ.get("SLURM_NODEID", ""),
        slurm_process_id=os.environ.get("SLURM_PROCID", ""),
    )


def _validate_step_identity(identity: StepIdentity) -> None:
    actual_node = _short_hostname(identity.actual_node, label="actual hostname")
    assigned_node = _short_hostname(identity.assigned_node, label="assigned node")
    if actual_node != assigned_node:
        raise ContractError(
            "Lane step ran on the wrong node: "
            f"assigned={assigned_node}, actual={actual_node}"
        )
    if not identity.slurm_job_id:
        raise ContractError("run-lane requires SLURM_JOB_ID from an active allocation")
    if identity.slurm_step_id in {"", "batch", "extern"}:
        raise ContractError("run-lane requires a dedicated Slurm srun step")
    if identity.slurm_step_num_nodes != "1":
        raise ContractError(
            "run-lane requires a one-node Slurm step; "
            f"SLURM_STEP_NUM_NODES={identity.slurm_step_num_nodes!r}"
        )
    step_node = _short_hostname(
        identity.slurm_step_nodelist,
        label="SLURM_STEP_NODELIST",
    )
    if step_node != assigned_node:
        raise ContractError(
            "Slurm step node does not match the manifest assignment: "
            f"assigned={assigned_node}, step_node={step_node}"
        )


def _step_identity_payload(identity: StepIdentity) -> dict[str, object]:
    return {
        "assigned_node": identity.assigned_node,
        "actual_node": identity.actual_node,
        "slurm_job_id": identity.slurm_job_id,
        "slurm_step_id": identity.slurm_step_id,
        "slurm_step_nodelist": identity.slurm_step_nodelist,
        "slurm_step_num_nodes": identity.slurm_step_num_nodes,
        "slurm_node_id": identity.slurm_node_id,
        "slurm_process_id": identity.slurm_process_id,
    }


def _unbound_step_identity_payload(
    parent_environment: Mapping[str, str],
) -> dict[str, object]:
    """Capture scheduler identity before the immutable lane assignment is loaded."""
    return {
        "assigned_node": None,
        "actual_node": socket.gethostname().split(".", 1)[0],
        "slurm_job_id": parent_environment.get("SLURM_JOB_ID", ""),
        "slurm_step_id": parent_environment.get("SLURM_STEP_ID", ""),
        "slurm_step_nodelist": parent_environment.get("SLURM_STEP_NODELIST", ""),
        "slurm_step_num_nodes": parent_environment.get("SLURM_STEP_NUM_NODES", ""),
        "slurm_node_id": parent_environment.get("SLURM_NODEID", ""),
        "slurm_process_id": parent_environment.get("SLURM_PROCID", ""),
    }


def _write_json_exclusive(path: Path, payload: object) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _enter_lane_barrier(
    *,
    output_root: Path,
    lane: str,
    identity: StepIdentity,
    run_manifest_sha256: str,
) -> tuple[str, dict[str, str], dict[str, str]]:
    """Publish this lane once and wait until both peer steps are live."""
    barrier_root = output_root / "barrier"
    if not barrier_root.is_dir():
        raise ContractError(f"Prepared barrier directory is missing: {barrier_root}")
    ready_at_utc = _utc_now()
    _write_json_exclusive(
        barrier_root / f"{lane}.ready.json",
        {
            "protocol": BARRIER_PROTOCOL,
            "lane": lane,
            "run_manifest_sha256": run_manifest_sha256,
            "ready_at_utc": ready_at_utc,
            **_step_identity_payload(identity),
        },
    )

    peer_observed_at_utc: dict[str, str] = {}
    peer_slurm_job_ids: dict[str, str] = {}
    deadline = time.monotonic() + BARRIER_TIMEOUT_SECONDS
    while len(peer_observed_at_utc) != len(LANES) - 1:
        for peer_lane in LANES:
            if peer_lane == lane or peer_lane in peer_observed_at_utc:
                continue
            peer_path = barrier_root / f"{peer_lane}.ready.json"
            if not peer_path.is_file():
                continue
            peer = _load_json_object(peer_path)
            if _require_string(peer, "protocol") != BARRIER_PROTOCOL:
                raise ContractError(f"Barrier protocol mismatch in {peer_path}")
            if _require_string(peer, "lane") != peer_lane:
                raise ContractError(f"Barrier lane mismatch in {peer_path}")
            if _require_sha256(peer, "run_manifest_sha256") != run_manifest_sha256:
                raise ContractError(f"Barrier manifest mismatch in {peer_path}")
            assigned = _short_hostname(
                _require_string(peer, "assigned_node"),
                label=f"{peer_lane} assigned node",
            )
            actual = _short_hostname(
                _require_string(peer, "actual_node"),
                label=f"{peer_lane} actual node",
            )
            if assigned != actual:
                raise ContractError(f"Barrier peer {peer_lane} attested the wrong node")
            peer_slurm_job_id = _require_string(peer, "slurm_job_id")
            if peer_slurm_job_id != identity.slurm_job_id:
                raise ContractError(
                    f"Barrier peer {peer_lane} belongs to a different Slurm job: "
                    f"expected={identity.slurm_job_id}, actual={peer_slurm_job_id}"
                )
            _require_string(peer, "slurm_step_id")
            peer_observed_at_utc[peer_lane] = _utc_now()
            peer_slurm_job_ids[peer_lane] = peer_slurm_job_id
        if len(peer_observed_at_utc) == len(LANES) - 1:
            break
        if time.monotonic() >= deadline:
            missing = sorted(set(LANES) - {lane} - set(peer_observed_at_utc))
            raise TimeoutError(f"Timed out waiting for barrier peers: {missing}")
        time.sleep(BARRIER_POLL_SECONDS)
    return ready_at_utc, peer_observed_at_utc, peer_slurm_job_ids


def _run_lane(
    *,
    lane: str,
    parent_environment: Mapping[str, str],
    output_root: Path,
) -> LaneExecution:
    """Validate and execute one manifest lane inside a terminal evidence envelope."""
    canonical_lane = _canonical_lane(lane)
    if canonical_lane != lane:
        raise ValueError("Operational run-lane requires a canonical lane name")
    run_dir = output_root / canonical_lane
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    resource_path = run_dir / "gnu_time.txt"
    invocation_path = run_dir / "invocation.json"
    execution_path = run_dir / "execution.json"
    results_path = run_dir / "results.json"
    recorded_environment = _selected_environment(parent_environment)
    runner_started_at_utc = _utc_now()
    invocation: dict[str, object] = {
        "schema_version": 2,
        "lane": canonical_lane,
        "command": [],
        "environment": recorded_environment,
        "run_manifest_sha256": None,
        "runner_started_at_utc": runner_started_at_utc,
        "barrier_protocol": BARRIER_PROTOCOL,
        "barrier_peer_observed_at_utc": {},
        "barrier_peer_slurm_job_ids": {},
        **_unbound_step_identity_payload(parent_environment),
    }
    execution: dict[str, object] = {
        **invocation,
        "status": "starting",
        "returncode": None,
        "started_at_utc": None,
        "ended_at_utc": None,
        "runner_ended_at_utc": None,
        "wall_seconds": None,
        "host_max_rss_kib": None,
        "run_dir": str(run_dir),
        "results_json": str(results_path),
        "results_sha256": None,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resource_log": str(resource_path),
    }
    _write_json(invocation_path, invocation)
    _write_json(execution_path, execution)

    terminal_evidence_written = False
    try:
        prepared = _prepare_lane_execution(output_root, canonical_lane)
        command = prepared.command
        run_manifest_sha256 = prepared.run_manifest_sha256
        identity = _raw_step_identity(prepared.assigned_node)
        invocation.update(
            {
                "command": list(command),
                "run_manifest_sha256": run_manifest_sha256,
                **_step_identity_payload(identity),
            }
        )
        execution.update(
            {
                "command": list(command),
                "run_manifest_sha256": run_manifest_sha256,
                **_step_identity_payload(identity),
            }
        )
        _write_json(invocation_path, invocation)
        _write_json(execution_path, execution)
        _validate_step_identity(identity)
        if (
            prepared.allocation_slurm_job_id is not None
            and identity.slurm_job_id != prepared.allocation_slurm_job_id
        ):
            raise ContractError(
                "Lane step belongs to a different Slurm allocation: "
                f"expected={prepared.allocation_slurm_job_id}, "
                f"actual={identity.slurm_job_id}"
            )
        environment = lane_environment(canonical_lane, parent_environment)
        recorded_environment = _selected_environment(environment)
        invocation["environment"] = recorded_environment
        execution["environment"] = recorded_environment
        _write_json(invocation_path, invocation)
        _write_json(execution_path, execution)
        time_executable = Path("/usr/bin/time")
        if not time_executable.is_file():
            raise RuntimeError("Full-loop comparator requires GNU /usr/bin/time")
        (
            barrier_ready_at_utc,
            peer_observed_at_utc,
            peer_slurm_job_ids,
        ) = _enter_lane_barrier(
            output_root=output_root,
            lane=canonical_lane,
            identity=identity,
            run_manifest_sha256=run_manifest_sha256,
        )
        invocation.update(
            {
                "barrier_ready_at_utc": barrier_ready_at_utc,
                "barrier_peer_observed_at_utc": peer_observed_at_utc,
                "barrier_peer_slurm_job_ids": peer_slurm_job_ids,
            }
        )
        _write_json(invocation_path, invocation)

        timed_command = (
            str(time_executable),
            "--verbose",
            "--output",
            str(resource_path),
            *command,
        )
        started_at_utc = _utc_now()
        started_at = time.monotonic()
        with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                timed_command,
                cwd=REPO_ROOT,
                env=dict(environment),
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
                text=True,
            )
        wall_seconds = time.monotonic() - started_at
        ended_at_utc = _utc_now()
        host_max_rss_kib = parse_gnu_time_verbose(
            resource_path.read_text(encoding="utf-8")
        )
        results_sha256 = sha256_file(results_path) if results_path.is_file() else None
        execution.update(
            {
                "status": "passed"
                if completed.returncode == 0 and results_sha256
                else "failed",
                "returncode": completed.returncode,
                "started_at_utc": started_at_utc,
                "ended_at_utc": ended_at_utc,
                "runner_ended_at_utc": _utc_now(),
                "wall_seconds": wall_seconds,
                "host_max_rss_kib": host_max_rss_kib,
                "results_sha256": results_sha256,
                "barrier_ready_at_utc": barrier_ready_at_utc,
                "barrier_peer_observed_at_utc": peer_observed_at_utc,
                "barrier_peer_slurm_job_ids": peer_slurm_job_ids,
            }
        )
        invocation["runner_ended_at_utc"] = execution["runner_ended_at_utc"]
        _write_json(invocation_path, invocation)
        _write_json(execution_path, execution)
        terminal_evidence_written = True
        if completed.returncode != 0:
            raise RuntimeError(
                f"{canonical_lane} child exited with status {completed.returncode}; "
                f"see {stderr_path}"
            )
        if results_sha256 is None:
            raise ContractError(
                f"{canonical_lane} child did not produce {results_path}"
            )
        return LaneExecution(
            lane=canonical_lane,
            command=command,
            environment=recorded_environment,
            returncode=completed.returncode,
            started_at_utc=started_at_utc,
            ended_at_utc=ended_at_utc,
            wall_seconds=wall_seconds,
            host_max_rss_kib=host_max_rss_kib,
            run_dir=str(run_dir),
            results_json=str(results_path),
            results_sha256=results_sha256,
            stdout_log=str(stdout_path),
            stderr_log=str(stderr_path),
            resource_log=str(resource_path),
            run_manifest_sha256=run_manifest_sha256,
            assigned_node=identity.assigned_node,
            actual_node=identity.actual_node,
            slurm_job_id=identity.slurm_job_id,
            slurm_step_id=identity.slurm_step_id,
            slurm_step_nodelist=identity.slurm_step_nodelist,
            barrier_peer_observed_at_utc=peer_observed_at_utc,
            barrier_peer_slurm_job_ids=peer_slurm_job_ids,
        )
    except Exception as error:
        if not terminal_evidence_written:
            execution.update(
                {
                    "status": "failed",
                    "runner_ended_at_utc": _utc_now(),
                    "failure": {
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                }
            )
            invocation["runner_ended_at_utc"] = execution["runner_ended_at_utc"]
            invocation["failure"] = execution["failure"]
            _write_json(invocation_path, invocation)
            _write_json(execution_path, execution)
        raise


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _lowercase_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise argparse.ArgumentTypeError("value must be a lowercase SHA-256 digest")
    return value


def _parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        add_help=add_help,
        description=(
            "Prepare one full-loop native CPU, JAX CPU, and FP64 JAX CUDA "
            "banana comparison using host SciPy L-BFGS-B and no ALM."
        ),
    )
    parser.add_argument("--surface-path", type=Path, required=True)
    parser.add_argument("--biotsavart-file", type=Path, required=True)
    parser.add_argument("--boozer-state-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--environment-lock-sha256",
        type=_lowercase_sha256,
        required=True,
        help="SHA-256 of the immutable dependency lock used by all three lanes.",
    )
    parser.add_argument(
        "--mode",
        choices=("production", "diagnostic"),
        default="production",
        help=(
            "Production requires every outcome gate; diagnostic keeps outcome "
            "failures advisory and can never be claim-ready."
        ),
    )

    parser.add_argument("--mpol", type=_positive_int, default=10)
    parser.add_argument("--ntor", type=_positive_int, default=10)
    parser.add_argument("--nphi", type=_positive_int, default=255)
    parser.add_argument("--ntheta", type=_positive_int, default=64)
    parser.add_argument("--vmec-s", type=float, default=1.0)
    parser.add_argument("--surface-scale", type=_positive_float, default=None)
    parser.add_argument("--constraint-weight", type=_positive_float, default=1.0)
    parser.add_argument("--iota-target", type=float, required=True)

    parser.add_argument("--maxiter", type=_positive_int, default=150)
    parser.add_argument("--outer-maxcor", type=_positive_int, default=300)
    parser.add_argument("--outer-maxls", type=_positive_int, default=20)
    parser.add_argument("--outer-ftol", type=_positive_float, default=1.0e-15)
    parser.add_argument("--outer-gtol", type=_positive_float, default=1.0e-8)
    parser.add_argument("--boozer-bfgs-tol", type=_positive_float, default=1.0e-10)
    parser.add_argument("--boozer-bfgs-maxiter", type=_positive_int, default=1500)
    parser.add_argument("--boozer-newton-tol", type=_positive_float, default=1.0e-11)
    parser.add_argument("--boozer-newton-maxiter", type=_positive_int, default=40)

    parser.add_argument(
        "--initial-objective-rtol", type=_nonnegative_float, default=1.0e-8
    )
    parser.add_argument(
        "--initial-objective-atol", type=_nonnegative_float, default=1.0e-10
    )
    parser.add_argument(
        "--initial-gradient-rtol", type=_nonnegative_float, default=1.0e-6
    )
    parser.add_argument(
        "--initial-gradient-atol", type=_nonnegative_float, default=1.0e-8
    )
    parser.add_argument("--initial-iota-atol", type=_nonnegative_float, default=1.0e-10)
    parser.add_argument("--initial-G-rtol", type=_nonnegative_float, default=1.0e-10)
    parser.add_argument("--initial-G-atol", type=_nonnegative_float, default=1.0e-10)
    parser.add_argument(
        "--final-objective-rtol", type=_nonnegative_float, default=1.0e-3
    )
    parser.add_argument(
        "--final-objective-atol", type=_nonnegative_float, default=1.0e-10
    )
    parser.add_argument(
        "--final-dofs-rtol",
        type=_nonnegative_float,
        default=ComparisonTolerances.final_dofs_rtol,
        help=(
            "Componentwise endpoint-DOF relative tolerance. The 1e-3 default "
            "tests optimizer-basin agreement without requiring byte-identical "
            "CPU/GPU trajectories."
        ),
    )
    parser.add_argument(
        "--final-dofs-atol",
        type=_nonnegative_float,
        default=ComparisonTolerances.final_dofs_atol,
        help=(
            "Componentwise endpoint-DOF absolute floor. The 1e-6 default keeps "
            "near-zero geometry coefficients within micron-scale agreement."
        ),
    )
    parser.add_argument("--final-iota-atol", type=_nonnegative_float, default=1.0e-4)
    parser.add_argument("--final-G-rtol", type=_nonnegative_float, default=1.0e-5)
    parser.add_argument("--final-G-atol", type=_nonnegative_float, default=1.0e-8)
    return parser


def _cli_parser() -> argparse.ArgumentParser:
    """Return the operational prepare/run-lane/adjudicate command surface."""
    parser = argparse.ArgumentParser(
        description="Run a three-node full-loop comparison in explicit phases."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        parents=[_parser(add_help=False)],
        help="Freeze the shared comparison contract before Slurm lane steps start.",
    )
    prepare.add_argument("--native-cpu-node", required=True)
    prepare.add_argument("--jax-cpu-node", required=True)
    prepare.add_argument("--jax-gpu-node", required=True)

    run_lane = subparsers.add_parser(
        "run-lane",
        help="Execute one manifest-defined lane inside its pinned Slurm step.",
    )
    run_lane.add_argument("--output-root", type=Path, required=True)
    run_lane.add_argument("--lane", choices=LANES, required=True)

    adjudicate = subparsers.add_parser(
        "adjudicate",
        help="Validate all lane evidence and write the final comparison artifact.",
    )
    adjudicate.add_argument("--output-root", type=Path, required=True)
    adjudicate.add_argument(
        "--native-cpu-returncode", type=_nonnegative_int, required=True
    )
    adjudicate.add_argument(
        "--jax-cpu-returncode", type=_nonnegative_int, required=True
    )
    adjudicate.add_argument(
        "--jax-gpu-returncode", type=_nonnegative_int, required=True
    )
    return parser


def _resolved_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def _absolute_existing_file(path: Path, *, label: str) -> Path:
    """Make a file path absolute without collapsing ``..`` or resolving symlinks."""
    absolute = path.expanduser().absolute()
    if not absolute.is_file():
        raise FileNotFoundError(f"{label} does not exist: {absolute}")
    return absolute


def resolve_external_output_root(repo_root: Path, output_root: Path) -> Path:
    """Resolve an output root and reject writes into the source checkout."""
    resolved_repo_root = repo_root.expanduser().resolve(strict=True)
    resolved_output_root = output_root.expanduser().resolve(strict=False)
    if (
        resolved_output_root == resolved_repo_root
        or resolved_repo_root in resolved_output_root.parents
    ):
        raise ValueError(
            "Full-loop performance artifacts must be outside the source checkout: "
            f"repo_root={resolved_repo_root}, output_root={resolved_output_root}"
        )
    return resolved_output_root


def _normalize_args(args: argparse.Namespace) -> None:
    args.surface_path = _resolved_existing_file(args.surface_path, label="surface seed")
    args.biotsavart_file = _resolved_existing_file(
        args.biotsavart_file, label="Biot-Savart seed"
    )
    args.boozer_state_path = _resolved_existing_file(
        args.boozer_state_path, label="Boozer state seed"
    )
    args.python = _absolute_existing_file(args.python, label="Python interpreter")
    args.output_root = resolve_external_output_root(REPO_ROOT, args.output_root)
    if args.output_root.exists():
        raise FileExistsError(f"Refusing to reuse output root: {args.output_root}")
    for driver in (NATIVE_DRIVER, JAX_DRIVER):
        if not driver.is_file():
            raise FileNotFoundError(f"Full-loop child driver is missing: {driver}")


def _tolerances(args: argparse.Namespace) -> ComparisonTolerances:
    return ComparisonTolerances(
        initial_objective_rtol=args.initial_objective_rtol,
        initial_objective_atol=args.initial_objective_atol,
        initial_gradient_rtol=args.initial_gradient_rtol,
        initial_gradient_atol=args.initial_gradient_atol,
        initial_iota_atol=args.initial_iota_atol,
        initial_G_rtol=args.initial_G_rtol,
        initial_G_atol=args.initial_G_atol,
        final_objective_rtol=args.final_objective_rtol,
        final_objective_atol=args.final_objective_atol,
        final_dofs_rtol=args.final_dofs_rtol,
        final_dofs_atol=args.final_dofs_atol,
        final_iota_atol=args.final_iota_atol,
        final_G_rtol=args.final_G_rtol,
        final_G_atol=args.final_G_atol,
    )


def _assigned_nodes(args: argparse.Namespace) -> dict[str, str]:
    nodes = {
        "native_cpu": _short_hostname(
            args.native_cpu_node,
            label="native CPU assigned node",
        ),
        "jax_cpu": _short_hostname(
            args.jax_cpu_node,
            label="JAX CPU assigned node",
        ),
        "jax_gpu": _short_hostname(
            args.jax_gpu_node,
            label="JAX GPU assigned node",
        ),
    }
    if len(set(nodes.values())) != len(LANES):
        raise ContractError(f"All three lane nodes must be distinct: {nodes}")
    return nodes


def _write_immutable_run_manifest(output_root: Path, payload: object) -> str:
    manifest_path = output_root / RUN_MANIFEST_NAME
    digest_path = output_root / RUN_MANIFEST_DIGEST_NAME
    _write_json(manifest_path, payload)
    run_manifest_sha256 = sha256_file(manifest_path)
    with digest_path.open("x", encoding="utf-8") as handle:
        handle.write(f"{run_manifest_sha256}\n")
        handle.flush()
        os.fsync(handle.fileno())
    manifest_path.chmod(0o444)
    digest_path.chmod(0o444)
    return run_manifest_sha256


def _prepare_pair(args: argparse.Namespace) -> int:
    """Freeze one immutable three-lane contract before scheduler steps start."""
    _normalize_args(args)
    assigned_nodes = _assigned_nodes(args)
    source = source_identity(REPO_ROOT)
    input_paths = {
        "surface": args.surface_path,
        "biotsavart": args.biotsavart_file,
        "boozer_state": args.boozer_state_path,
    }
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items()}
    shared_configuration = _shared_configuration(args, input_sha256)
    run_config_sha256 = sha256_json(shared_configuration)
    commands = {
        lane: build_lane_command(
            args,
            lane=lane,
            run_dir=args.output_root / lane,
            run_config_sha256=run_config_sha256,
        )
        for lane in LANES
    }
    execution_topology: dict[str, object] = {
        "policy": EXECUTION_POLICY,
        "assigned_nodes": assigned_nodes,
        "barrier": {
            "protocol": BARRIER_PROTOCOL,
            "participants": list(LANES),
        },
    }
    allocation_slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    if allocation_slurm_job_id:
        execution_topology["slurm_job_id"] = allocation_slurm_job_id
    manifest = {
        "schema_version": 2,
        "prepared_at_utc": _utc_now(),
        "source": asdict(source),
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "input_sha256": input_sha256,
        "environment_lock_sha256": args.environment_lock_sha256,
        "shared_configuration": shared_configuration,
        "run_config_sha256": run_config_sha256,
        "comparison_mode": args.mode,
        "commands": {lane: list(command) for lane, command in commands.items()},
        "tolerances": asdict(_tolerances(args)),
        "execution_topology": execution_topology,
    }
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "barrier").mkdir()
    run_manifest_sha256 = _write_immutable_run_manifest(args.output_root, manifest)
    print(
        json.dumps(
            {
                "run_manifest": str(args.output_root / RUN_MANIFEST_NAME),
                "run_manifest_sha256": run_manifest_sha256,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


def _existing_output_root(output_root: Path) -> Path:
    resolved = resolve_external_output_root(REPO_ROOT, output_root)
    if not resolved.is_dir():
        raise FileNotFoundError(f"Prepared output root does not exist: {resolved}")
    return resolved


def _manifest_tolerances(manifest: Mapping[str, object]) -> ComparisonTolerances:
    values = _require_mapping(manifest, "tolerances")

    def tolerance(name: str) -> float:
        value = _require_finite_float(values, name)
        if value < 0.0:
            raise ContractError(f"tolerances.{name} must be non-negative")
        return value

    return ComparisonTolerances(
        initial_objective_rtol=tolerance("initial_objective_rtol"),
        initial_objective_atol=tolerance("initial_objective_atol"),
        initial_gradient_rtol=tolerance("initial_gradient_rtol"),
        initial_gradient_atol=tolerance("initial_gradient_atol"),
        initial_iota_atol=tolerance("initial_iota_atol"),
        initial_G_rtol=tolerance("initial_G_rtol"),
        initial_G_atol=tolerance("initial_G_atol"),
        initial_volume_rtol=tolerance("initial_volume_rtol"),
        initial_volume_atol=tolerance("initial_volume_atol"),
        initial_term_rtol=tolerance("initial_term_rtol"),
        initial_term_atol=tolerance("initial_term_atol"),
        final_objective_rtol=tolerance("final_objective_rtol"),
        final_objective_atol=tolerance("final_objective_atol"),
        final_dofs_rtol=tolerance("final_dofs_rtol"),
        final_dofs_atol=tolerance("final_dofs_atol"),
        final_gradient_rtol=tolerance("final_gradient_rtol"),
        final_gradient_atol=tolerance("final_gradient_atol"),
        final_iota_atol=tolerance("final_iota_atol"),
        final_G_rtol=tolerance("final_G_rtol"),
        final_G_atol=tolerance("final_G_atol"),
        final_volume_rtol=tolerance("final_volume_rtol"),
        final_volume_atol=tolerance("final_volume_atol"),
        final_term_rtol=tolerance("final_term_rtol"),
        final_term_atol=tolerance("final_term_atol"),
    )


def _manifest_assigned_nodes(manifest: Mapping[str, object]) -> dict[str, str]:
    topology = _require_mapping(manifest, "execution_topology")
    if _require_string(topology, "policy") != EXECUTION_POLICY:
        raise ContractError("Run manifest execution policy is not concurrent")
    barrier = _require_mapping(topology, "barrier")
    if _require_string(barrier, "protocol") != BARRIER_PROTOCOL:
        raise ContractError("Run manifest barrier protocol is unsupported")
    if _require_string_tuple(barrier, "participants") != LANES:
        raise ContractError("Run manifest barrier participants must be all three lanes")
    raw_nodes = _require_mapping(topology, "assigned_nodes")
    if set(raw_nodes) != set(LANES):
        raise ContractError(
            "Run manifest must assign exactly the three canonical lanes"
        )
    nodes = {
        lane: _short_hostname(
            _require_string(raw_nodes, lane),
            label=f"{lane} assigned node",
        )
        for lane in LANES
    }
    if len(set(nodes.values())) != len(LANES):
        raise ContractError("Run manifest lane nodes must be pairwise distinct")
    return nodes


def _manifest_slurm_job_id(manifest: Mapping[str, object]) -> str | None:
    topology = _require_mapping(manifest, "execution_topology")
    value = topology.get("slurm_job_id")
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ContractError("Run manifest Slurm job ID must be a non-empty string")
    return value


def _load_run_manifest(output_root: Path) -> tuple[dict[str, object], str]:
    manifest_path = output_root / RUN_MANIFEST_NAME
    digest_path = output_root / RUN_MANIFEST_DIGEST_NAME
    if not manifest_path.is_file() or not digest_path.is_file():
        raise ContractError("Prepared run manifest or digest is missing")
    if manifest_path.stat().st_mode & 0o222 or digest_path.stat().st_mode & 0o222:
        raise ContractError("Prepared run manifest and digest must be read-only")
    expected_digest = digest_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ContractError("Run manifest digest sidecar is invalid")
    actual_digest = sha256_file(manifest_path)
    if actual_digest != expected_digest:
        raise ContractError("Run manifest changed after prepare")
    manifest = _load_json_object(manifest_path)
    if _require_int(manifest, "schema_version") != 2:
        raise ContractError("Run manifest schema_version must be 2")
    _require_string(manifest, "prepared_at_utc")
    _require_mapping(manifest, "source")
    input_paths = _require_mapping(manifest, "input_paths")
    input_sha256 = _require_mapping(manifest, "input_sha256")
    if set(input_paths) != set(INPUT_NAMES) or set(input_sha256) != set(INPUT_NAMES):
        raise ContractError("Run manifest input identities are incomplete")
    for name in INPUT_NAMES:
        _require_string(input_paths, name)
        _require_sha256(input_sha256, name)
    _require_sha256(manifest, "environment_lock_sha256")
    shared_configuration = _require_mapping(manifest, "shared_configuration")
    run_config_sha256 = _require_sha256(manifest, "run_config_sha256")
    if sha256_json(shared_configuration) != run_config_sha256:
        raise ContractError("Run manifest shared configuration digest is invalid")
    if _require_string(manifest, "comparison_mode") not in {
        "production",
        "diagnostic",
    }:
        raise ContractError("Run manifest comparison mode is unsupported")
    commands = _require_mapping(manifest, "commands")
    if set(commands) != set(LANES):
        raise ContractError("Run manifest commands must cover exactly three lanes")
    for lane in LANES:
        _require_string_tuple(commands, lane)
    _manifest_tolerances(manifest)
    _manifest_assigned_nodes(manifest)
    _manifest_slurm_job_id(manifest)
    return manifest, actual_digest


def _validate_manifest_inputs_and_source(manifest: Mapping[str, object]) -> None:
    recorded_source = _require_mapping(manifest, "source")
    current_source = source_identity(REPO_ROOT)
    expected_source = {
        "commit_sha": _require_string(recorded_source, "commit_sha"),
        "tree_sha": _require_string(recorded_source, "tree_sha"),
        "status_porcelain": recorded_source.get("status_porcelain"),
    }
    if asdict(current_source) != expected_source:
        raise ContractError("Source identity changed after prepare")
    input_paths = _require_mapping(manifest, "input_paths")
    input_sha256 = _require_mapping(manifest, "input_sha256")
    for name in INPUT_NAMES:
        path = Path(_require_string(input_paths, name))
        if not path.is_file():
            raise ContractError(f"Prepared {name} input is missing: {path}")
        if sha256_file(path) != _require_sha256(input_sha256, name):
            raise ContractError(f"Prepared {name} input changed after prepare")


def _prepare_lane_execution(output_root: Path, lane: str) -> PreparedLaneExecution:
    manifest, run_manifest_sha256 = _load_run_manifest(output_root)
    _validate_manifest_inputs_and_source(manifest)
    assigned_nodes = _manifest_assigned_nodes(manifest)
    commands = _require_mapping(manifest, "commands")
    return PreparedLaneExecution(
        command=_require_string_tuple(commands, lane),
        run_manifest_sha256=run_manifest_sha256,
        assigned_node=assigned_nodes[lane],
        allocation_slurm_job_id=_manifest_slurm_job_id(manifest),
    )


def _run_lane_phase(args: argparse.Namespace) -> int:
    output_root = _existing_output_root(args.output_root)
    _run_lane(
        lane=args.lane,
        parent_environment=os.environ,
        output_root=output_root,
    )
    return 0


def _parse_utc_timestamp(value: str, *, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ContractError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ContractError(f"{label} must include a timezone")
    return parsed


def _string_mapping(
    parent: Mapping[str, object],
    key: str,
    *,
    allow_empty_values: bool = False,
) -> dict[str, str]:
    raw = _require_mapping(parent, key)
    result: dict[str, str] = {}
    for name, value in raw.items():
        if not isinstance(value, str) or (not value and not allow_empty_values):
            expected = "a string" if allow_empty_values else "a non-empty string"
            raise ContractError(f"{key}.{name} must be {expected}")
        result[name] = value
    return result


def _load_successful_lane_execution(
    *,
    output_root: Path,
    lane: str,
    assigned_node: str,
    run_manifest_sha256: str,
    expected_command: tuple[str, ...],
) -> LaneExecution:
    run_dir = output_root / lane
    invocation = _load_json_object(run_dir / "invocation.json")
    execution = _load_json_object(run_dir / "execution.json")
    for payload_name, payload in (("invocation", invocation), ("execution", execution)):
        if _require_string(payload, "lane") != lane:
            raise ContractError(f"{payload_name} lane mismatch for {lane}")
        if _require_sha256(payload, "run_manifest_sha256") != run_manifest_sha256:
            raise ContractError(f"{payload_name} manifest mismatch for {lane}")
        if (
            _short_hostname(
                _require_string(payload, "assigned_node"),
                label=f"{lane} assigned node",
            )
            != assigned_node
        ):
            raise ContractError(f"{payload_name} assigned-node mismatch for {lane}")
        if (
            _short_hostname(
                _require_string(payload, "actual_node"),
                label=f"{lane} actual node",
            )
            != assigned_node
        ):
            raise ContractError(f"{payload_name} actual-node mismatch for {lane}")
        if (
            _short_hostname(
                _require_string(payload, "slurm_step_nodelist"),
                label=f"{lane} step nodelist",
            )
            != assigned_node
        ):
            raise ContractError(f"{payload_name} step-node mismatch for {lane}")
        _require_string(payload, "slurm_job_id")
        _require_string(payload, "slurm_step_id")
    slurm_job_id = _require_string(execution, "slurm_job_id")
    if _require_string(invocation, "slurm_job_id") != slurm_job_id:
        raise ContractError(f"{lane} invocation/execution Slurm job IDs differ")
    if _require_string(execution, "status") != "passed":
        raise ContractError(f"{lane} execution did not pass")
    if _require_int(execution, "returncode") != 0:
        raise ContractError(f"{lane} child return code was nonzero")
    command = _require_string_tuple(execution, "command")
    if (
        command != expected_command
        or _require_string_tuple(
            invocation,
            "command",
        )
        != expected_command
    ):
        raise ContractError(f"{lane} command differs from the run manifest")
    environment = _string_mapping(
        execution,
        "environment",
        allow_empty_values=True,
    )
    if (
        _string_mapping(
            invocation,
            "environment",
            allow_empty_values=True,
        )
        != environment
    ):
        raise ContractError(f"{lane} invocation/execution environments differ")
    started_at_utc = _require_string(execution, "started_at_utc")
    ended_at_utc = _require_string(execution, "ended_at_utc")
    started = _parse_utc_timestamp(started_at_utc, label=f"{lane} start")
    ended = _parse_utc_timestamp(ended_at_utc, label=f"{lane} end")
    if ended <= started:
        raise ContractError(f"{lane} execution interval is not positive")
    wall_seconds = _require_finite_float(execution, "wall_seconds")
    if wall_seconds <= 0.0:
        raise ContractError(f"{lane} wall_seconds must be positive")
    host_max_rss_kib = _require_int(execution, "host_max_rss_kib")
    if host_max_rss_kib <= 0:
        raise ContractError(f"{lane} host_max_rss_kib must be positive")
    results_path = Path(_require_string(execution, "results_json"))
    expected_results_path = run_dir / "results.json"
    if results_path != expected_results_path or not results_path.is_file():
        raise ContractError(f"{lane} results path is missing or non-canonical")
    results_sha256 = _require_sha256(execution, "results_sha256")
    if sha256_file(results_path) != results_sha256:
        raise ContractError(f"{lane} results changed after execution")
    for artifact_key in ("stdout_log", "stderr_log", "resource_log"):
        if not Path(_require_string(execution, artifact_key)).is_file():
            raise ContractError(f"{lane} {artifact_key} artifact is missing")
    peer_observed_at_utc = _string_mapping(
        execution,
        "barrier_peer_observed_at_utc",
    )
    expected_peers = set(LANES) - {lane}
    if set(peer_observed_at_utc) != expected_peers:
        raise ContractError(f"{lane} did not observe both barrier peers")
    invocation_peers = _string_mapping(
        invocation,
        "barrier_peer_observed_at_utc",
    )
    if invocation_peers != peer_observed_at_utc:
        raise ContractError(f"{lane} invocation/execution barrier evidence differs")
    for peer_lane, observed_at_utc in peer_observed_at_utc.items():
        _parse_utc_timestamp(
            observed_at_utc,
            label=f"{lane} observation of {peer_lane}",
        )
    peer_slurm_job_ids = _string_mapping(
        execution,
        "barrier_peer_slurm_job_ids",
    )
    if set(peer_slurm_job_ids) != expected_peers:
        raise ContractError(f"{lane} did not bind both barrier peers to a Slurm job")
    if set(peer_slurm_job_ids.values()) != {slurm_job_id}:
        raise ContractError(f"{lane} observed a barrier peer from another Slurm job")
    if _string_mapping(invocation, "barrier_peer_slurm_job_ids") != peer_slurm_job_ids:
        raise ContractError(f"{lane} invocation/execution barrier Slurm job IDs differ")
    return LaneExecution(
        lane=lane,
        command=command,
        environment=environment,
        returncode=0,
        started_at_utc=started_at_utc,
        ended_at_utc=ended_at_utc,
        wall_seconds=wall_seconds,
        host_max_rss_kib=host_max_rss_kib,
        run_dir=str(run_dir),
        results_json=str(results_path),
        results_sha256=results_sha256,
        stdout_log=_require_string(execution, "stdout_log"),
        stderr_log=_require_string(execution, "stderr_log"),
        resource_log=_require_string(execution, "resource_log"),
        run_manifest_sha256=run_manifest_sha256,
        assigned_node=assigned_node,
        actual_node=_require_string(execution, "actual_node"),
        slurm_job_id=slurm_job_id,
        slurm_step_id=_require_string(execution, "slurm_step_id"),
        slurm_step_nodelist=_require_string(execution, "slurm_step_nodelist"),
        barrier_peer_observed_at_utc=peer_observed_at_utc,
        barrier_peer_slurm_job_ids=peer_slurm_job_ids,
    )


def _aggregate_parity(
    comparisons: Mapping[str, Mapping[str, object]],
    *,
    mode: str,
) -> dict[str, object]:
    required_failures: list[str] = []
    outcome_failures: list[str] = []
    advisory_failures: list[str] = []
    for comparison_name, comparison in comparisons.items():
        required_failures.extend(
            f"{comparison_name}.{failure}"
            for failure in _require_string_tuple_allow_empty(
                comparison,
                "required_failures",
            )
        )
        outcome_failures.extend(
            f"{comparison_name}.{failure}"
            for failure in _require_string_tuple_allow_empty(
                comparison,
                "outcome_failures",
            )
        )
        advisory_failures.extend(
            f"{comparison_name}.{failure}"
            for failure in _require_string_tuple_allow_empty(
                comparison,
                "advisory_failures",
            )
        )
    passed = all(
        _require_bool(comparison, "passed") for comparison in comparisons.values()
    )
    claim_ready = all(
        _require_bool(comparison, "claim_ready") for comparison in comparisons.values()
    )
    failures = required_failures + (outcome_failures if mode == "production" else [])
    return {
        "mode": mode,
        "passed": passed,
        "claim_ready": claim_ready,
        "failures": failures,
        "required_failures": required_failures,
        "outcome_failures": outcome_failures,
        "advisory_failures": advisory_failures,
    }


def _require_string_tuple_allow_empty(
    parent: Mapping[str, object],
    key: str,
) -> tuple[str, ...]:
    value = parent.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ContractError(f"{key} must be a JSON string array")
    return tuple(value)


def _validated_gpu_memory_hook(
    output_root: Path,
    execution: LaneExecution,
) -> dict[str, object]:
    """Load claim-grade JAX-GPU process-memory evidence for one pair."""
    hook_path = output_root / "jax_gpu" / "gpu_process_memory.json"
    if not hook_path.is_file():
        raise ContractError(f"JAX-GPU process-memory evidence is missing: {hook_path}")
    hook = _load_json_object(hook_path)
    if _require_int(hook, "schema_version") != 1:
        raise ContractError("JAX-GPU process-memory schema_version must be 1")
    if _require_string(hook, "metric") != "nvidia-smi compute-process used_memory":
        raise ContractError("JAX-GPU process-memory metric is unsupported")
    if _require_string(hook, "unit") != "MiB":
        raise ContractError("JAX-GPU process-memory unit must be MiB")
    if _require_string(hook, "pair") != output_root.name:
        raise ContractError("JAX-GPU process-memory pair identity differs")
    if _short_hostname(
        _require_string(hook, "node"),
        label="JAX-GPU process-memory node",
    ) != _short_hostname(execution.actual_node, label="JAX-GPU execution node"):
        raise ContractError("JAX-GPU process-memory node differs from execution")
    if _require_string(hook, "slurm_step_id") != execution.slurm_step_id:
        raise ContractError("JAX-GPU process-memory Slurm step differs from execution")
    expected_selector = execution.environment.get("CUDA_VISIBLE_DEVICES")
    if (
        not expected_selector
        or _require_string(hook, "cuda_visible_devices") != expected_selector
    ):
        raise ContractError(
            "JAX-GPU process-memory CUDA selector differs from execution"
        )

    inventory = hook.get("gpu_inventory")
    if not isinstance(inventory, list) or not inventory:
        raise ContractError("JAX-GPU process-memory inventory must be non-empty")
    inventory_uuids: set[str] = set()
    for index, item in enumerate(inventory):
        if not isinstance(item, dict) or not all(isinstance(key, str) for key in item):
            raise ContractError(f"gpu_inventory[{index}] must be a JSON object")
        _require_int(item, "index")
        _require_string(item, "name")
        inventory_uuids.add(_require_string(item, "uuid"))
        _require_positive_int(item, "memory_total_mib")
        _require_string(item, "driver_version")

    sample_count = _require_positive_int(hook, "sample_count")
    gpu_uuids = set(_require_string_tuple(hook, "gpu_uuids"))
    if not gpu_uuids.issubset(inventory_uuids):
        raise ContractError("JAX-GPU process-memory samples reference unknown GPUs")
    process_ids = hook.get("process_ids")
    if not isinstance(process_ids, list) or not process_ids:
        raise ContractError("JAX-GPU process-memory process_ids must be non-empty")
    if any(
        isinstance(process_id, bool)
        or not isinstance(process_id, int)
        or process_id <= 0
        for process_id in process_ids
    ):
        raise ContractError(
            "JAX-GPU process-memory process_ids must be positive integers"
        )
    first_sample = _parse_utc_timestamp(
        _require_string(hook, "first_sample_at_utc"),
        label="JAX-GPU first memory sample",
    )
    last_sample = _parse_utc_timestamp(
        _require_string(hook, "last_sample_at_utc"),
        label="JAX-GPU last memory sample",
    )
    if last_sample < first_sample:
        raise ContractError("JAX-GPU process-memory sample interval is reversed")
    _require_positive_int(hook, "maximum_used_memory_mib")

    sampler_queries = _require_mapping(hook, "sampler_queries")
    query_count = _require_positive_int(sampler_queries, "query_count")
    successful_query_count = _require_int(
        sampler_queries,
        "successful_query_count",
    )
    if (
        _require_int(sampler_queries, "failure_count") != 0
        or successful_query_count != query_count
        or _require_bool(sampler_queries, "all_succeeded") is not True
    ):
        raise ContractError(
            "JAX-GPU process-memory sampler queries did not all succeed"
        )
    if sample_count < len(process_ids):
        raise ContractError(
            "JAX-GPU process-memory sample count is internally inconsistent"
        )
    return hook


def _failed_comparison_payload(
    *,
    failures: Sequence[str],
    step_returncodes: Mapping[str, int],
    manifest: Mapping[str, object] | None,
    run_manifest_sha256: str | None,
    lanes: Mapping[str, object],
) -> dict[str, object]:
    mode = (
        _require_string(manifest, "comparison_mode")
        if manifest is not None
        else "unknown"
    )
    return {
        "schema_version": 2,
        "status": "failed",
        "comparison_mode": mode,
        "claim_ready": False,
        "run_manifest_sha256": run_manifest_sha256,
        "step_returncodes": dict(step_returncodes),
        "execution_topology": (
            manifest.get("execution_topology") if manifest is not None else None
        ),
        "lanes": dict(lanes),
        "comparisons": {},
        "parity": {
            "mode": mode,
            "passed": False,
            "claim_ready": False,
            "failures": list(failures),
            "required_failures": list(failures),
            "outcome_failures": [],
            "advisory_failures": [],
        },
        "performance": None,
        "failures": list(failures),
    }


def _adjudicate_prepared_pair(
    *,
    output_root: Path,
    manifest: Mapping[str, object],
    run_manifest_sha256: str,
    step_returncodes: Mapping[str, int],
) -> tuple[dict[str, object], int]:
    _validate_manifest_inputs_and_source(manifest)
    allocation_slurm_job_id = _manifest_slurm_job_id(manifest)
    if (
        allocation_slurm_job_id is not None
        and os.environ.get("SLURM_JOB_ID", "") != allocation_slurm_job_id
    ):
        raise ContractError(
            "Adjudication belongs to a different Slurm allocation: "
            f"expected={allocation_slurm_job_id}, "
            f"actual={os.environ.get('SLURM_JOB_ID', '')}"
        )
    assigned_nodes = _manifest_assigned_nodes(manifest)
    commands = _require_mapping(manifest, "commands")
    failures = [
        f"step_returncode.{lane}" for lane in LANES if step_returncodes[lane] != 0
    ]
    lane_evidence: dict[str, object] = {}
    executions: dict[str, LaneExecution] = {}
    for lane in LANES:
        execution_path = output_root / lane / "execution.json"
        if execution_path.is_file():
            lane_evidence[lane] = _load_json_object(execution_path)
        try:
            executions[lane] = _load_successful_lane_execution(
                output_root=output_root,
                lane=lane,
                assigned_node=assigned_nodes[lane],
                run_manifest_sha256=run_manifest_sha256,
                expected_command=_require_string_tuple(commands, lane),
            )
        except (ContractError, FileNotFoundError) as error:
            failures.append(f"lane_evidence.{lane}: {error}")
    if len(executions) == len(LANES):
        slurm_job_ids = {execution.slurm_job_id for execution in executions.values()}
        if len(slurm_job_ids) != 1:
            failures.append("execution_topology.slurm_job_ids_not_equal")
        allocation_slurm_job_id = _manifest_slurm_job_id(manifest)
        if allocation_slurm_job_id is not None and slurm_job_ids != {
            allocation_slurm_job_id
        }:
            failures.append("execution_topology.slurm_job_id_not_manifest_allocation")
        step_ids = {execution.slurm_step_id for execution in executions.values()}
        if len(step_ids) != len(LANES):
            failures.append("execution_topology.slurm_step_ids_not_distinct")
        starts = [
            _parse_utc_timestamp(execution.started_at_utc, label=f"{lane} start")
            for lane, execution in executions.items()
        ]
        ends = [
            _parse_utc_timestamp(execution.ended_at_utc, label=f"{lane} end")
            for lane, execution in executions.items()
        ]
        if max(starts) >= min(ends):
            failures.append("execution_topology.lane_intervals_do_not_overlap")
    if failures:
        return (
            _failed_comparison_payload(
                failures=failures,
                step_returncodes=step_returncodes,
                manifest=manifest,
                run_manifest_sha256=run_manifest_sha256,
                lanes=lane_evidence,
            ),
            1,
        )

    gpu_memory_hook = _validated_gpu_memory_hook(
        output_root,
        executions["jax_gpu"],
    )
    parsed = {
        lane: parse_lane_result(_load_json_object(Path(executions[lane].results_json)))
        for lane in LANES
    }
    input_sha256_mapping = _require_mapping(manifest, "input_sha256")
    input_sha256 = {
        name: _require_sha256(input_sha256_mapping, name) for name in INPUT_NAMES
    }
    run_config_sha256 = _require_sha256(manifest, "run_config_sha256")
    tolerances = _manifest_tolerances(manifest)
    mode = _require_string(manifest, "comparison_mode")
    comparisons = {
        "native_cpu_vs_jax_cpu": compare_lane_results(
            parsed["native_cpu"],
            parsed["jax_cpu"],
            expected_input_sha256=input_sha256,
            expected_run_config_sha256=run_config_sha256,
            tolerances=tolerances,
            mode=mode,
            cpu_lane_name="native_cpu",
            jax_lane_name="jax_cpu",
            expected_jax_backend="jax-cpu",
        ),
        "native_cpu_vs_jax_gpu": compare_lane_results(
            parsed["native_cpu"],
            parsed["jax_gpu"],
            expected_input_sha256=input_sha256,
            expected_run_config_sha256=run_config_sha256,
            tolerances=tolerances,
            mode=mode,
            cpu_lane_name="native_cpu",
            jax_lane_name="jax_gpu",
        ),
        "jax_cpu_vs_jax_gpu": compare_lane_results(
            parsed["jax_cpu"],
            parsed["jax_gpu"],
            expected_input_sha256=input_sha256,
            expected_run_config_sha256=run_config_sha256,
            tolerances=tolerances,
            mode=mode,
            cpu_lane_name="jax_cpu",
            jax_lane_name="jax_gpu",
            expected_cpu_backend="jax-cpu",
            expected_jax_backend="jax-cuda",
            expected_cpu_adjoint_policy="checked-residual-and-condition",
        ),
    }
    parity = _aggregate_parity(comparisons, mode=mode)
    claim_ready = _require_bool(parity, "claim_ready")
    parity_passed = _require_bool(parity, "passed")
    status = "passed" if claim_ready else "diagnostic" if parity_passed else "failed"
    performance = {
        "wall_seconds": {lane: executions[lane].wall_seconds for lane in LANES},
        "host_max_rss_kib": {lane: executions[lane].host_max_rss_kib for lane in LANES},
        "speedups": {
            "native_cpu_over_jax_cpu": executions["native_cpu"].wall_seconds
            / executions["jax_cpu"].wall_seconds,
            "native_cpu_over_jax_gpu": executions["native_cpu"].wall_seconds
            / executions["jax_gpu"].wall_seconds,
            "jax_cpu_over_jax_gpu": executions["jax_cpu"].wall_seconds
            / executions["jax_gpu"].wall_seconds,
        },
        "gpu_process_memory": gpu_memory_hook,
    }
    comparison = {
        "schema_version": 2,
        "status": status,
        "comparison_mode": mode,
        "claim_ready": claim_ready,
        "source": manifest["source"],
        "input_sha256": input_sha256,
        "environment_lock_sha256": _require_sha256(
            manifest,
            "environment_lock_sha256",
        ),
        "run_config_sha256": run_config_sha256,
        "run_manifest_sha256": run_manifest_sha256,
        "step_returncodes": dict(step_returncodes),
        "execution_topology": manifest["execution_topology"],
        "lanes": {lane: asdict(executions[lane]) for lane in LANES},
        "comparisons": comparisons,
        "parity": parity,
        "performance": performance,
        "failures": parity["failures"],
    }
    return comparison, 0 if parity_passed else 1


def _adjudicate_phase(args: argparse.Namespace) -> int:
    output_root = resolve_external_output_root(REPO_ROOT, args.output_root)
    step_returncodes = {
        "native_cpu": args.native_cpu_returncode,
        "jax_cpu": args.jax_cpu_returncode,
        "jax_gpu": args.jax_gpu_returncode,
    }
    manifest: dict[str, object] | None = None
    run_manifest_sha256: str | None = None
    try:
        manifest, run_manifest_sha256 = _load_run_manifest(output_root)
        comparison, exit_code = _adjudicate_prepared_pair(
            output_root=output_root,
            manifest=manifest,
            run_manifest_sha256=run_manifest_sha256,
            step_returncodes=step_returncodes,
        )
    except Exception as error:
        comparison = _failed_comparison_payload(
            failures=(f"adjudication: {type(error).__name__}: {error}",),
            step_returncodes=step_returncodes,
            manifest=manifest,
            run_manifest_sha256=run_manifest_sha256,
            lanes={},
        )
        exit_code = 1
    comparison_path = output_root / "comparison.json"
    _write_json(comparison_path, comparison)
    print(json.dumps(comparison["performance"], indent=2, sort_keys=True), flush=True)
    if exit_code != 0:
        print(
            f"Full-loop adjudication failed: {comparison['failures']}", file=sys.stderr
        )
    else:
        print(f"Full-loop adjudication complete: {comparison_path}", flush=True)
    return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.command == "prepare":
        return _prepare_pair(args)
    if args.command == "run-lane":
        return _run_lane_phase(args)
    if args.command == "adjudicate":
        return _adjudicate_phase(args)
    raise AssertionError(f"Unsupported command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
