"""Bounded, correlated trial evidence for Boozer outer optimization."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

import numpy as np
from simsopt_jax.geo.optimizer_host_lbfgs import (
    HostLineSearchTrial,
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
)

if TYPE_CHECKING:
    from benchmarks.fixtures.custom_quasi_newton import Fixture

_SCHEMA_VERSION = 1
_DIAGNOSTIC_ROUTE = "host_more_thuente_objective_probe"
_DEFAULT_PARAMETER_BYTE_CAP = 64 * 1024 * 1024
_SHA256_HEX_LENGTH = 64

TrialPhase = Literal["initial", "line_search", "final_refresh"]
TrialProvider = Literal["native", "custom"]
GradientSource = Literal["candidate", "baseline", "unavailable"]


@dataclass(frozen=True)
class TrialKey:
    """Identity shared by optimizer-owned and objective-owned trial evidence."""

    evaluation_index: int
    parameter_sha256: str


@dataclass(frozen=True)
class ObjectiveTrialEvidence:
    """Objective-owned physical and inner-solve evidence for one parameter set."""

    raw_objective: float | None
    raw_objective_certified: bool
    filtered_objective: float | None
    gradient_inf_norm: float | None
    gradient_finite: bool
    gradient_source: GradientSource
    gradient_source_parameter_sha256: str | None
    predictor_kind: str | None
    predictor_success: bool | None
    primal_success: bool
    adjoint_success: bool | None
    newton_success: bool
    newton_stop_reason_code: int | None
    newton_accepted_iterations: int | None
    newton_attempted_iterations: int | None
    newton_last_linear_solve_success: bool | None
    inner_penalty_residual_l2: float | None
    inner_final_gradient_inf_norm: float | None


@dataclass(frozen=True)
class LineSearchTrialEvidence:
    """Host-line-search evidence; absent for initial and final refresh records."""

    trial_ordinal: int | None
    step_length: float | None
    directional_derivative: float | None
    armijo_margin: float | None
    curvature_margin: float | None


@dataclass(frozen=True)
class JoinedBoozerTrialRecord:
    """One correlation-checked record written by the diagnostic join owner."""

    key: TrialKey
    phase: TrialPhase
    objective: ObjectiveTrialEvidence
    line_search: LineSearchTrialEvidence
    parameter_archive_key: str
    parameter_shape: tuple[int, ...]
    parameter_dtype: Literal["<f8"] = "<f8"


@dataclass(frozen=True)
class BoozerTrialTraceSummary:
    """Validated artifact metadata used by receipt qualification."""

    case: Literal["boozer"]
    provider: TrialProvider
    production_route: str
    diagnostic_route: Literal["host_more_thuente_objective_probe"]
    record_count: int
    max_records: int
    parameter_bytes: int
    parameter_byte_cap: int


@dataclass(frozen=True)
class BoozerHostDiagnosticResult:
    """Controlled host-driver outcome plus its validated trace artifact."""

    provider: TrialProvider
    converged: bool
    status: int
    iterations: int
    evaluations: int
    final_objective: float
    final_gradient_inf_norm: float
    final_parameters: tuple[float, ...]
    trial_trace: Path


def parameter_sha256(parameters: np.ndarray) -> str:
    """Hash canonical contiguous little-endian float64 parameter bytes."""

    canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).reshape(-1)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def run_boozer_host_diagnostic(
    fixture_case: Fixture,
    *,
    provider: TrialProvider,
    manifest_path: Path,
    maxiter: int = 1000,
    maxls: int = 20,
    gtol: float = 1.0e-10,
    parameter_byte_cap: int = _DEFAULT_PARAMETER_BYTE_CAP,
) -> BoozerHostDiagnosticResult:
    """Run native or JAX objective data under one matched host BFGS driver."""

    if fixture_case.name != "boozer" or fixture_case.method != "bfgs":
        raise ValueError("host diagnostic requires the Boozer BFGS fixture")
    evaluator = (
        fixture_case.native_trial_evaluator
        if provider == "native"
        else fixture_case.trial_evaluator
    )
    if evaluator is None:
        raise ValueError(f"Boozer fixture has no {provider} trial evaluator")
    if provider == "native" and fixture_case.native_reset is not None:
        fixture_case.native_reset()
    initial = np.ascontiguousarray(fixture_case.initial, dtype=np.dtype("<f8"))
    max_records = 1 + maxiter * maxls + 1
    worst_case_parameter_bytes = max_records * initial.size * initial.dtype.itemsize
    if worst_case_parameter_bytes > parameter_byte_cap:
        raise ValueError(
            "declared trial byte cap cannot hold the worst-case bounded trace"
        )
    records: list[JoinedBoozerTrialRecord] = []
    parameters: dict[str, np.ndarray] = {}
    pending: (
        tuple[
            TrialKey,
            np.ndarray,
            ObjectiveTrialEvidence,
        ]
        | None
    ) = None
    baseline_hash = parameter_sha256(initial)
    parameters[baseline_hash] = initial.copy()

    def objective_evidence(parameters_host: np.ndarray):
        nonlocal pending
        candidate = np.ascontiguousarray(parameters_host, dtype=np.dtype("<f8"))
        evaluation = evaluator(candidate)
        gradient = np.asarray(evaluation.gradient, dtype=np.float64)
        if gradient.shape != candidate.shape:
            raise ValueError("trial evaluator returned a mismatched gradient shape")
        candidate_hash = parameter_sha256(candidate)
        parameters.setdefault(candidate_hash, candidate.copy())
        if evaluation.gradient_source == "candidate":
            gradient_source_hash = candidate_hash
        elif evaluation.gradient_source == "baseline":
            gradient_source_hash = baseline_hash
        else:
            gradient_source_hash = None
        gradient_finite = bool(np.all(np.isfinite(gradient)))
        gradient_inf_norm = float(np.max(np.abs(gradient))) if gradient_finite else None
        evidence = ObjectiveTrialEvidence(
            raw_objective=evaluation.raw_objective,
            raw_objective_certified=evaluation.raw_objective_certified,
            filtered_objective=evaluation.filtered_objective,
            gradient_inf_norm=gradient_inf_norm,
            gradient_finite=gradient_finite,
            gradient_source=evaluation.gradient_source,
            gradient_source_parameter_sha256=gradient_source_hash,
            predictor_kind=evaluation.predictor_kind,
            predictor_success=evaluation.predictor_success,
            primal_success=evaluation.primal_success,
            adjoint_success=evaluation.adjoint_success,
            newton_success=evaluation.newton_success,
            newton_stop_reason_code=evaluation.newton_stop_reason_code,
            newton_accepted_iterations=evaluation.newton_accepted_iterations,
            newton_attempted_iterations=evaluation.newton_attempted_iterations,
            newton_last_linear_solve_success=(
                evaluation.newton_last_linear_solve_success
            ),
            inner_penalty_residual_l2=evaluation.inner_penalty_residual_l2,
            inner_final_gradient_inf_norm=(evaluation.inner_final_gradient_inf_norm),
        )
        key = TrialKey(len(records), candidate_hash)
        if not records:
            records.append(
                JoinedBoozerTrialRecord(
                    key=key,
                    phase="initial",
                    objective=evidence,
                    line_search=LineSearchTrialEvidence(None, None, None, None, None),
                    parameter_archive_key=candidate_hash,
                    parameter_shape=tuple(candidate.shape),
                )
            )
        else:
            if pending is not None:
                raise RuntimeError("line search evaluated twice before observation")
            pending = (key, candidate, evidence)
        filtered = evaluation.filtered_objective
        return (
            float("nan") if filtered is None else filtered,
            gradient,
        )

    def observe_line_search(trial: HostLineSearchTrial) -> None:
        nonlocal pending
        if pending is None:
            raise RuntimeError("line-search observation has no objective evidence")
        key, candidate, evidence = pending
        records.append(
            JoinedBoozerTrialRecord(
                key=key,
                phase="line_search",
                objective=evidence,
                line_search=LineSearchTrialEvidence(
                    trial_ordinal=trial.trial_ordinal,
                    step_length=_finite_or_none(trial.alpha),
                    directional_derivative=_finite_or_none(
                        trial.directional_derivative
                    ),
                    armijo_margin=_finite_or_none(trial.armijo_margin),
                    curvature_margin=_finite_or_none(trial.curvature_margin),
                ),
                parameter_archive_key=key.parameter_sha256,
                parameter_shape=tuple(candidate.shape),
            )
        )
        pending = None

    def observed_more_thuente(**kwargs):
        return line_search_value_and_grad_more_thuente_host(
            **kwargs,
            trial_observer=observe_line_search,
        )

    result = minimize_bfgs_host_core(
        objective_evidence,
        initial,
        maxiter=maxiter,
        maxls=maxls,
        gtol=gtol,
        line_search_value_and_grad=observed_more_thuente,
    )
    if pending is not None:
        raise RuntimeError("host diagnostic ended with unjoined trial evidence")
    final_parameters = np.ascontiguousarray(result.x_k, dtype=np.dtype("<f8"))
    final_evaluation = evaluator(final_parameters)
    final_hash = parameter_sha256(final_parameters)
    parameters.setdefault(final_hash, final_parameters.copy())
    final_gradient = np.asarray(final_evaluation.gradient, dtype=np.float64)
    final_gradient_finite = bool(np.all(np.isfinite(final_gradient)))
    if final_evaluation.gradient_source == "candidate":
        final_gradient_source_hash = final_hash
    elif final_evaluation.gradient_source == "baseline":
        final_gradient_source_hash = baseline_hash
    else:
        final_gradient_source_hash = None
    final_evidence = ObjectiveTrialEvidence(
        raw_objective=final_evaluation.raw_objective,
        raw_objective_certified=final_evaluation.raw_objective_certified,
        filtered_objective=final_evaluation.filtered_objective,
        gradient_inf_norm=(
            float(np.max(np.abs(final_gradient))) if final_gradient_finite else None
        ),
        gradient_finite=final_gradient_finite,
        gradient_source=final_evaluation.gradient_source,
        gradient_source_parameter_sha256=final_gradient_source_hash,
        predictor_kind=final_evaluation.predictor_kind,
        predictor_success=final_evaluation.predictor_success,
        primal_success=final_evaluation.primal_success,
        adjoint_success=final_evaluation.adjoint_success,
        newton_success=final_evaluation.newton_success,
        newton_stop_reason_code=final_evaluation.newton_stop_reason_code,
        newton_accepted_iterations=final_evaluation.newton_accepted_iterations,
        newton_attempted_iterations=final_evaluation.newton_attempted_iterations,
        newton_last_linear_solve_success=(
            final_evaluation.newton_last_linear_solve_success
        ),
        inner_penalty_residual_l2=final_evaluation.inner_penalty_residual_l2,
        inner_final_gradient_inf_norm=(final_evaluation.inner_final_gradient_inf_norm),
    )
    records.append(
        JoinedBoozerTrialRecord(
            key=TrialKey(len(records), final_hash),
            phase="final_refresh",
            objective=final_evidence,
            line_search=LineSearchTrialEvidence(None, None, None, None, None),
            parameter_archive_key=final_hash,
            parameter_shape=tuple(final_parameters.shape),
        )
    )
    write_boozer_trial_trace(
        manifest_path,
        provider=provider,
        production_route=(
            "scipy_bfgs" if provider == "native" else "custom_bfgs_stepwise"
        ),
        maxiter=maxiter,
        maxls=maxls,
        records=tuple(records),
        parameters_by_sha256=parameters,
        parameter_byte_cap=parameter_byte_cap,
    )
    validate_boozer_trial_trace(
        manifest_path,
        expected_provider=provider,
        expected_production_route=(
            "scipy_bfgs" if provider == "native" else "custom_bfgs_stepwise"
        ),
        expected_maxiter=maxiter,
        expected_evaluations=result.nfev,
    )
    return BoozerHostDiagnosticResult(
        provider=provider,
        converged=result.converged,
        status=result.status,
        iterations=result.k,
        evaluations=result.nfev,
        final_objective=result.f_k,
        final_gradient_inf_norm=float(np.max(np.abs(result.g_k))),
        final_parameters=tuple(float(value) for value in result.x_k),
        trial_trace=manifest_path,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_line(record: JoinedBoozerTrialRecord) -> str:
    return json.dumps(
        asdict(record),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def write_boozer_trial_trace(
    manifest_path: Path,
    *,
    provider: TrialProvider,
    production_route: str,
    maxiter: int,
    maxls: int,
    records: tuple[JoinedBoozerTrialRecord, ...],
    parameters_by_sha256: dict[str, np.ndarray],
    parameter_byte_cap: int = _DEFAULT_PARAMETER_BYTE_CAP,
) -> Path:
    """Write one atomic manifest plus bounded JSONL/NPZ evidence pair."""

    if manifest_path.exists():
        raise FileExistsError(f"trial trace already exists: {manifest_path}")
    if maxiter <= 0 or maxls <= 0 or parameter_byte_cap <= 0:
        raise ValueError("trial bounds must be positive")
    max_records = 1 + maxiter * maxls + 1
    if not records or len(records) > max_records:
        raise ValueError("trial record count exceeds the declared bound")
    expected_indices = tuple(range(len(records)))
    if tuple(record.key.evaluation_index for record in records) != expected_indices:
        raise ValueError("trial evaluation indices must be contiguous from zero")
    if records[0].phase != "initial" or records[-1].phase != "final_refresh":
        raise ValueError("trial trace must begin at initial and end at final_refresh")
    if any(record.phase != "line_search" for record in records[1:-1]):
        raise ValueError("intermediate trial records must be line_search evaluations")

    canonical_parameters: dict[str, np.ndarray] = {}
    for expected_hash, parameters in sorted(parameters_by_sha256.items()):
        canonical = np.ascontiguousarray(parameters, dtype=np.dtype("<f8")).reshape(-1)
        actual_hash = parameter_sha256(canonical)
        if expected_hash != actual_hash:
            raise ValueError("parameter archive key does not match canonical bytes")
        canonical_parameters[expected_hash] = canonical
    parameter_bytes = sum(array.nbytes for array in canonical_parameters.values())
    if parameter_bytes > parameter_byte_cap:
        raise ValueError("trial parameter archive exceeds the declared byte cap")
    for record in records:
        if record.key.parameter_sha256 not in canonical_parameters:
            raise ValueError("trial record references an absent parameter vector")
        if record.parameter_archive_key != record.key.parameter_sha256:
            raise ValueError(
                "trial record archive key differs from its correlation key"
            )
        parameters = canonical_parameters[record.key.parameter_sha256]
        if record.parameter_shape != tuple(parameters.shape):
            raise ValueError("trial record parameter shape does not match the archive")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records_name = f"{manifest_path.stem}.records.jsonl"
    parameters_name = f"{manifest_path.stem}.parameters.npz"
    with tempfile.TemporaryDirectory(
        prefix=f".{manifest_path.stem}.", dir=manifest_path.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        records_path = temporary / records_name
        parameters_path = temporary / parameters_name
        records_path.write_text(
            "".join(f"{_json_line(record)}\n" for record in records),
            encoding="utf-8",
        )
        np.savez(
            parameters_path,
            **{f"p_{key}": value for key, value in canonical_parameters.items()},
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "case": "boozer",
            "provider": provider,
            "production_route": production_route,
            "diagnostic_route": _DIAGNOSTIC_ROUTE,
            "maxiter": maxiter,
            "maxls": maxls,
            "max_records": max_records,
            "record_count": len(records),
            "diagnostic_evaluations": len(records) - 1,
            "parameter_dtype": "<f8",
            "parameter_bytes": parameter_bytes,
            "parameter_byte_cap": parameter_byte_cap,
            "records_path": records_name,
            "records_sha256": _sha256(records_path),
            "parameters_path": parameters_name,
            "parameters_sha256": _sha256(parameters_path),
        }
        temporary_manifest = temporary / manifest_path.name
        temporary_manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        records_path.replace(manifest_path.parent / records_name)
        parameters_path.replace(manifest_path.parent / parameters_name)
        temporary_manifest.replace(manifest_path)
    return manifest_path


def _required_int(payload: dict[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"trial trace {field} must be an integer")
    return value


def _required_string(payload: dict[str, object], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise TypeError(f"trial trace {field} must be a nonempty string")
    return value


def _relative_file(root: Path, relative: object, *, field: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise TypeError(f"trial trace {field} must be a relative path")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"trial trace {field} escapes its artifact directory")
    resolved = root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"trial trace artifact is missing: {resolved}")
    return resolved


def _optional_finite_nonnegative(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"trial record {field} must be a number or null")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"trial record {field} must be finite and nonnegative")
    return number


def _optional_finite(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"trial record {field} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"trial record {field} must be finite")
    return number


def _validate_record(
    payload: dict[str, object],
    *,
    expected_index: int,
    archive: dict[str, np.ndarray],
) -> JoinedBoozerTrialRecord:
    key = payload.get("key")
    objective = payload.get("objective")
    line_search = payload.get("line_search")
    shape = payload.get("parameter_shape")
    if not isinstance(key, dict) or not isinstance(objective, dict):
        raise TypeError("trial record key and objective must be objects")
    if not isinstance(line_search, dict) or not isinstance(shape, list):
        raise TypeError("trial record line_search and shape are invalid")
    key = cast(dict[str, object], key)
    objective = cast(dict[str, object], objective)
    line_search = cast(dict[str, object], line_search)
    evaluation_index = _required_int(key, "evaluation_index")
    if evaluation_index != expected_index:
        raise ValueError("trial evaluation indices are not contiguous")
    parameter_hash = _required_string(key, "parameter_sha256")
    if len(parameter_hash) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in parameter_hash
    ):
        raise ValueError("trial parameter hash is not lowercase SHA-256")
    archive_key = _required_string(payload, "parameter_archive_key")
    if archive_key != parameter_hash or archive_key not in archive:
        raise ValueError("trial correlation key is absent from parameter archive")
    array = archive[archive_key]
    if parameter_sha256(array) != parameter_hash:
        raise ValueError("trial parameter hash does not match archived bytes")
    if payload.get("parameter_dtype") != "<f8":
        raise ValueError("trial parameter dtype is not canonical <f8")
    typed_shape = tuple(shape)
    if any(not isinstance(size, int) or isinstance(size, bool) for size in typed_shape):
        raise TypeError("trial parameter shape must contain integers")
    if typed_shape != tuple(array.shape):
        raise ValueError("trial parameter shape differs from archived array")
    phase = payload.get("phase")
    if phase not in {"initial", "line_search", "final_refresh"}:
        raise ValueError("trial phase is invalid")
    trial_ordinal = line_search.get("trial_ordinal")
    if trial_ordinal is not None and (
        not isinstance(trial_ordinal, int)
        or isinstance(trial_ordinal, bool)
        or trial_ordinal <= 0
    ):
        raise ValueError("trial ordinal must be a positive integer or null")
    step_length = _optional_finite_nonnegative(
        line_search.get("step_length"), field="step_length"
    )
    directional_derivative = _optional_finite(
        line_search.get("directional_derivative"), field="directional_derivative"
    )
    armijo_margin = _optional_finite(
        line_search.get("armijo_margin"), field="armijo_margin"
    )
    curvature_margin = _optional_finite(
        line_search.get("curvature_margin"), field="curvature_margin"
    )
    line_fields = (trial_ordinal, step_length)
    optional_line_fields = (
        directional_derivative,
        armijo_margin,
        curvature_margin,
    )
    if phase == "line_search" and any(value is None for value in line_fields):
        raise ValueError("line-search trial identity is incomplete")
    if phase != "line_search" and any(value is not None for value in line_fields):
        raise ValueError("non-line-search trial carries line-search evidence")
    if phase != "line_search" and any(
        value is not None for value in optional_line_fields
    ):
        raise ValueError("non-line-search trial carries Wolfe evidence")
    raw_objective = _optional_finite(
        objective.get("raw_objective"), field="raw_objective"
    )
    filtered_objective = _optional_finite(
        objective.get("filtered_objective"), field="filtered_objective"
    )
    gradient_inf_norm = _optional_finite_nonnegative(
        objective.get("gradient_inf_norm"), field="gradient_inf_norm"
    )
    for field in (
        "raw_objective_certified",
        "gradient_finite",
        "primal_success",
        "newton_success",
    ):
        if not isinstance(objective.get(field), bool):
            raise TypeError(f"trial record {field} must be a boolean")
    if objective["raw_objective_certified"] is True and raw_objective is None:
        raise ValueError("certified trial has no raw physical objective")
    if objective["gradient_finite"] is True and gradient_inf_norm is None:
        raise ValueError("finite trial gradient has no norm")
    if (
        phase == "line_search"
        and objective["gradient_finite"] is True
        and any(value is None for value in optional_line_fields)
    ):
        raise ValueError("finite line-search trial has incomplete Wolfe evidence")
    gradient_source = objective.get("gradient_source")
    if gradient_source not in {"candidate", "baseline", "unavailable"}:
        raise ValueError("trial gradient source is invalid")
    source_hash = objective.get("gradient_source_parameter_sha256")
    if source_hash is not None and (
        not isinstance(source_hash, str) or source_hash not in archive
    ):
        raise ValueError("trial gradient source hash is absent from archive")
    if gradient_source == "candidate" and source_hash != parameter_hash:
        raise ValueError("candidate gradient is not bound to trial parameters")
    if gradient_source == "unavailable" and source_hash is not None:
        raise ValueError("unavailable gradient cannot carry a source hash")
    predictor_kind = objective.get("predictor_kind")
    if predictor_kind is not None and not isinstance(predictor_kind, str):
        raise TypeError("trial predictor_kind must be a string or null")
    nullable_bools: dict[str, bool | None] = {}
    for field in (
        "predictor_success",
        "adjoint_success",
        "newton_last_linear_solve_success",
    ):
        value = objective.get(field)
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"trial record {field} must be a boolean or null")
        nullable_bools[field] = cast(bool | None, value)
    nullable_ints: dict[str, int | None] = {}
    for field in (
        "newton_stop_reason_code",
        "newton_accepted_iterations",
        "newton_attempted_iterations",
    ):
        value = objective.get(field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value < 0
        ):
            raise ValueError(f"trial record {field} must be nonnegative or null")
        nullable_ints[field] = cast(int | None, value)
    inner_residual = _optional_finite_nonnegative(
        objective.get("inner_penalty_residual_l2"),
        field="inner_penalty_residual_l2",
    )
    inner_gradient = _optional_finite_nonnegative(
        objective.get("inner_final_gradient_inf_norm"),
        field="inner_final_gradient_inf_norm",
    )
    return JoinedBoozerTrialRecord(
        key=TrialKey(evaluation_index, parameter_hash),
        phase=cast(TrialPhase, phase),
        objective=ObjectiveTrialEvidence(
            raw_objective=raw_objective,
            raw_objective_certified=cast(bool, objective["raw_objective_certified"]),
            filtered_objective=filtered_objective,
            gradient_inf_norm=gradient_inf_norm,
            gradient_finite=cast(bool, objective["gradient_finite"]),
            gradient_source=cast(GradientSource, gradient_source),
            gradient_source_parameter_sha256=cast(str | None, source_hash),
            predictor_kind=cast(str | None, predictor_kind),
            predictor_success=nullable_bools["predictor_success"],
            primal_success=cast(bool, objective["primal_success"]),
            adjoint_success=nullable_bools["adjoint_success"],
            newton_success=cast(bool, objective["newton_success"]),
            newton_stop_reason_code=nullable_ints["newton_stop_reason_code"],
            newton_accepted_iterations=nullable_ints["newton_accepted_iterations"],
            newton_attempted_iterations=nullable_ints["newton_attempted_iterations"],
            newton_last_linear_solve_success=nullable_bools[
                "newton_last_linear_solve_success"
            ],
            inner_penalty_residual_l2=inner_residual,
            inner_final_gradient_inf_norm=inner_gradient,
        ),
        line_search=LineSearchTrialEvidence(
            trial_ordinal=cast(int | None, trial_ordinal),
            step_length=step_length,
            directional_derivative=directional_derivative,
            armijo_margin=armijo_margin,
            curvature_margin=curvature_margin,
        ),
        parameter_archive_key=archive_key,
        parameter_shape=cast(tuple[int, ...], typed_shape),
    )


def validate_boozer_trial_trace(
    manifest_path: Path,
    *,
    expected_provider: str,
    expected_production_route: str,
    expected_maxiter: int,
    expected_evaluations: int | None = None,
) -> BoozerTrialTraceSummary:
    """Validate all linked bytes and semantic bounds for one trial trace."""

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("trial trace manifest must be a JSON object")
    payload = cast(dict[str, object], payload)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported Boozer trial trace schema")
    if payload.get("case") != "boozer":
        raise ValueError("trial trace case is not boozer")
    if payload.get("provider") != expected_provider:
        raise ValueError("trial trace provider differs from measurement")
    if payload.get("production_route") != expected_production_route:
        raise ValueError("trial trace production route differs from measurement")
    if payload.get("diagnostic_route") != _DIAGNOSTIC_ROUTE:
        raise ValueError("trial trace diagnostic route is unsupported")
    maxiter = _required_int(payload, "maxiter")
    maxls = _required_int(payload, "maxls")
    max_records = _required_int(payload, "max_records")
    record_count = _required_int(payload, "record_count")
    diagnostic_evaluations = _required_int(payload, "diagnostic_evaluations")
    parameter_bytes = _required_int(payload, "parameter_bytes")
    parameter_byte_cap = _required_int(payload, "parameter_byte_cap")
    if maxiter != expected_maxiter or maxls <= 0:
        raise ValueError("trial trace solver bounds differ from measurement")
    if max_records != 1 + maxiter * maxls + 1:
        raise ValueError("trial trace record bound is not derived from solver bounds")
    if diagnostic_evaluations < 1 or record_count != diagnostic_evaluations + 1:
        raise ValueError("trial trace diagnostic evaluation count is inconsistent")
    if (
        expected_evaluations is not None
        and diagnostic_evaluations != expected_evaluations
    ):
        raise ValueError("trial trace diagnostic evaluations differ from expectation")
    if not 0 < record_count <= max_records:
        raise ValueError("trial trace record count differs from objective evaluations")
    if parameter_bytes < 0 or parameter_byte_cap <= 0:
        raise ValueError("trial trace parameter byte accounting is invalid")
    if parameter_bytes > parameter_byte_cap:
        raise ValueError("trial trace parameter archive exceeds its byte cap")
    if payload.get("parameter_dtype") != "<f8":
        raise ValueError("trial trace parameter dtype is not canonical <f8")
    records_path = _relative_file(
        manifest_path.parent, payload.get("records_path"), field="records_path"
    )
    parameters_path = _relative_file(
        manifest_path.parent,
        payload.get("parameters_path"),
        field="parameters_path",
    )
    if payload.get("records_sha256") != _sha256(records_path):
        raise ValueError("trial trace JSONL checksum mismatch")
    if payload.get("parameters_sha256") != _sha256(parameters_path):
        raise ValueError("trial trace parameter archive checksum mismatch")
    with np.load(parameters_path, allow_pickle=False) as loaded:
        archive: dict[str, np.ndarray] = {}
        for name in loaded.files:
            if not name.startswith("p_"):
                raise ValueError("trial parameter archive key is invalid")
            parameter_hash = name.removeprefix("p_")
            array = np.asarray(loaded[name])
            if array.dtype != np.dtype("<f8") or array.ndim != 1:
                raise ValueError("trial parameter archive array is not flat <f8")
            archive[parameter_hash] = array
    if sum(array.nbytes for array in archive.values()) != parameter_bytes:
        raise ValueError("trial trace parameter byte derivation mismatch")
    raw_lines = records_path.read_text(encoding="utf-8").splitlines()
    if len(raw_lines) != record_count:
        raise ValueError("trial trace JSONL record count mismatch")
    records: list[JoinedBoozerTrialRecord] = []
    for index, line in enumerate(raw_lines):
        raw_record = json.loads(line)
        if not isinstance(raw_record, dict):
            raise TypeError("trial JSONL entries must be objects")
        records.append(
            _validate_record(
                cast(dict[str, object], raw_record),
                expected_index=index,
                archive=archive,
            )
        )
    if records[0].phase != "initial" or records[-1].phase != "final_refresh":
        raise ValueError("trial phases do not bracket the optimization")
    if any(record.phase != "line_search" for record in records[1:-1]):
        raise ValueError("trial intermediate phases are not line-search evaluations")
    return BoozerTrialTraceSummary(
        case="boozer",
        provider=cast(TrialProvider, expected_provider),
        production_route=expected_production_route,
        diagnostic_route=_DIAGNOSTIC_ROUTE,
        record_count=record_count,
        max_records=max_records,
        parameter_bytes=parameter_bytes,
        parameter_byte_cap=parameter_byte_cap,
    )


__all__ = [
    "BoozerHostDiagnosticResult",
    "BoozerTrialTraceSummary",
    "JoinedBoozerTrialRecord",
    "LineSearchTrialEvidence",
    "ObjectiveTrialEvidence",
    "TrialKey",
    "parameter_sha256",
    "run_boozer_host_diagnostic",
    "validate_boozer_trial_trace",
    "write_boozer_trial_trace",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("native", "custom"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--maxls", type=int, default=20)
    parser.add_argument("--gtol", type=float, default=1.0e-10)
    args = parser.parse_args()

    from benchmarks.fixtures.custom_quasi_newton import fixture

    result = run_boozer_host_diagnostic(
        fixture("boozer"),
        provider=cast(TrialProvider, args.provider),
        manifest_path=args.output,
        maxiter=args.maxiter,
        maxls=args.maxls,
        gtol=args.gtol,
    )
    payload = asdict(result)
    payload["trial_trace"] = str(result.trial_trace)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
