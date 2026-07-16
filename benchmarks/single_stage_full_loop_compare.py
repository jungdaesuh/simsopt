"""Run and adjudicate one native-SIMSOPT-CPU/JAX-CUDA optimization pair.

The two lanes execute as isolated child processes from one immutable seed/config.
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
) -> dict[str, object]:
    """Apply invariant gates and mode-dependent optimization-outcome policy."""
    if mode not in {"production", "diagnostic"}:
        raise ValueError(f"Unsupported comparison mode {mode!r}")

    required_checks = [
        _exact_check("comparison_schema_version.cpu", cpu.comparison_schema_version, 1),
        _exact_check("comparison_schema_version.jax", jax.comparison_schema_version, 1),
        _exact_check("backend.cpu", cpu.backend, "native-simsopt-cpu"),
        _exact_check("backend.jax", jax.backend, "jax-cuda"),
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
            "native-plu-finite-gradient",
        ),
        _exact_check(
            "contract.adjoint_acceptance_policy.jax",
            jax.contract.adjoint_acceptance_policy,
            "checked-residual-and-condition",
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


def lane_environment(
    lane: str, parent_environment: Mapping[str, str]
) -> dict[str, str]:
    environment = dict(parent_environment)
    environment.update(
        {
            "JAX_ENABLE_X64": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
            "SIMSOPT_MIXED_PRECISION": "0",
        }
    )
    if lane == "cpu":
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "",
                "JAX_PLATFORMS": "cpu",
                "SIMSOPT_JAX_PLATFORM": "cpu",
                "SIMSOPT_JAX_BACKEND": "cpu",
            }
        )
        return environment
    if lane != "jax":
        raise ValueError(f"Unsupported lane {lane!r}")
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
    surface_normalization_arguments: tuple[str, ...] = (
        "--vmec-s",
        str(args.vmec_s),
    )
    if args.surface_scale is not None:
        surface_normalization_arguments += (
            "--surface-scale",
            str(args.surface_scale),
        )
    if lane == "cpu":
        driver = NATIVE_DRIVER
        lane_arguments: Sequence[str] = ()
        output_arguments: Sequence[str] = (
            "--output-root",
            str(run_dir),
            "--overwrite",
        )
        exclusion_arguments: Sequence[str] = ()
    elif lane == "jax":
        driver = JAX_DRIVER
        lane_arguments = ("--backend", "jax", "--platform", "cuda")
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
    else:
        raise ValueError(f"Unsupported lane {lane!r}")

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


def _run_lane(
    *,
    lane: str,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    run_dir: Path,
) -> LaneExecution:
    run_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    resource_path = run_dir / "gnu_time.txt"
    invocation_path = run_dir / "invocation.json"
    time_executable = Path("/usr/bin/time")
    if not time_executable.is_file():
        raise RuntimeError("Full-loop comparator requires GNU /usr/bin/time")

    recorded_environment = _selected_environment(environment)
    _write_json(
        invocation_path,
        {
            "lane": lane,
            "command": list(command),
            "environment": recorded_environment,
        },
    )
    timed_command = (
        str(time_executable),
        "--verbose",
        "--output",
        str(resource_path),
        *command,
    )
    started_at_utc = datetime.now(timezone.utc).isoformat()
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
    ended_at_utc = datetime.now(timezone.utc).isoformat()
    host_max_rss_kib = parse_gnu_time_verbose(resource_path.read_text(encoding="utf-8"))
    execution_path = run_dir / "execution.json"
    execution_payload = {
        "lane": lane,
        "returncode": completed.returncode,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "wall_seconds": wall_seconds,
        "host_max_rss_kib": host_max_rss_kib,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
        "resource_log": str(resource_path),
    }
    _write_json(execution_path, execution_payload)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{lane} child exited with status {completed.returncode}; see {stderr_path}"
        )

    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise ContractError(f"{lane} child did not produce {results_path}")
    results_sha256 = sha256_file(results_path)
    return LaneExecution(
        lane=lane,
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
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare full-loop native CPU and FP64 JAX CUDA banana optimization "
            "on one seed using host SciPy L-BFGS-B and no ALM."
        )
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
        help="SHA-256 of the immutable dependency lock used by both lanes.",
    )
    parser.add_argument("--order", choices=("cpu-jax", "jax-cpu"), default="cpu-jax")
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


def _resolved_existing_file(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


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
    args.python = _resolved_existing_file(args.python, label="Python interpreter")
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _normalize_args(args)
    source = source_identity(REPO_ROOT)
    input_paths = {
        "surface": args.surface_path,
        "biotsavart": args.biotsavart_file,
        "boozer_state": args.boozer_state_path,
    }
    input_sha256 = {name: sha256_file(path) for name, path in input_paths.items()}
    shared_configuration = _shared_configuration(args, input_sha256)
    run_config_sha256 = sha256_json(shared_configuration)
    lanes = tuple(args.order.split("-"))
    environments = {lane: lane_environment(lane, os.environ) for lane in lanes}
    commands = {
        lane: build_lane_command(
            args,
            lane=lane,
            run_dir=args.output_root / lane,
            run_config_sha256=run_config_sha256,
        )
        for lane in lanes
    }

    args.output_root.mkdir(parents=True, exist_ok=False)
    _write_json(
        args.output_root / "run_manifest.json",
        {
            "schema_version": 1,
            "source": asdict(source),
            "input_paths": {name: str(path) for name, path in input_paths.items()},
            "input_sha256": input_sha256,
            "environment_lock_sha256": args.environment_lock_sha256,
            "shared_configuration": shared_configuration,
            "run_config_sha256": run_config_sha256,
            "comparison_mode": args.mode,
            "lane_order": list(lanes),
            "commands": {lane: list(command) for lane, command in commands.items()},
            "environments": {
                lane: _selected_environment(environment)
                for lane, environment in environments.items()
            },
        },
    )

    executions: dict[str, LaneExecution] = {}
    for lane in lanes:
        print(f"Starting {lane} full-loop lane", flush=True)
        executions[lane] = _run_lane(
            lane=lane,
            command=commands[lane],
            environment=environments[lane],
            run_dir=args.output_root / lane,
        )

    parsed = {
        lane: parse_lane_result(_load_json_object(Path(execution.results_json)))
        for lane, execution in executions.items()
    }
    parity = compare_lane_results(
        parsed["cpu"],
        parsed["jax"],
        expected_input_sha256=input_sha256,
        expected_run_config_sha256=run_config_sha256,
        tolerances=_tolerances(args),
        mode=args.mode,
    )
    cpu_execution = executions["cpu"]
    jax_execution = executions["jax"]
    performance = {
        "cpu_wall_seconds": cpu_execution.wall_seconds,
        "jax_wall_seconds": jax_execution.wall_seconds,
        "cpu_over_jax_speedup": cpu_execution.wall_seconds / jax_execution.wall_seconds,
        "cpu_host_max_rss_kib": cpu_execution.host_max_rss_kib,
        "jax_host_max_rss_kib": jax_execution.host_max_rss_kib,
        "jax_over_cpu_host_max_rss_ratio": jax_execution.host_max_rss_kib
        / cpu_execution.host_max_rss_kib,
    }
    status = (
        "passed"
        if parity["claim_ready"]
        else "diagnostic"
        if parity["passed"]
        else "failed"
    )
    comparison = {
        "schema_version": 1,
        "status": status,
        "comparison_mode": args.mode,
        "claim_ready": parity["claim_ready"],
        "source": asdict(source),
        "input_sha256": input_sha256,
        "environment_lock_sha256": args.environment_lock_sha256,
        "run_config_sha256": run_config_sha256,
        "lane_order": list(lanes),
        "lanes": {lane: asdict(execution) for lane, execution in executions.items()},
        "parity": parity,
        "performance": performance,
    }
    comparison_path = args.output_root / "comparison.json"
    _write_json(comparison_path, comparison)
    print(json.dumps(performance, indent=2, sort_keys=True), flush=True)
    if not parity["passed"]:
        print(f"Full-loop parity failed: {parity['failures']}", file=sys.stderr)
        return 1
    if not parity["claim_ready"]:
        print(
            "Diagnostic comparison completed with advisory outcome failures: "
            f"{parity['advisory_failures']}",
            flush=True,
        )
        return 0
    print(f"Full-loop comparison passed: {comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
