"""Write immutable single-stage speed campaign receipt trees.

The public writer accepts a complete typed campaign and creates one fresh
artifact root containing only the protocol's campaign, lane, and trajectory
files.  It validates the schedule before writing so incomplete measurements
cannot become a claim-bearing receipt.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Final, Literal

from simsopt.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_OPTAX_DRIVER_ID,
)

ReceiptLaneId = Literal[
    "native_cpu", "jax_gpu_custom", "jax_gpu_optax", "jax_cpu_custom"
]
SamplePhase = Literal["cold", "warmup", "warm"]
AdjointRoute = Literal["exact_jacobian_dense_fp64_lu"]
DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE: Final[AdjointRoute] = "exact_jacobian_dense_fp64_lu"

RECEIPT_LANE_IDS: tuple[ReceiptLaneId, ...] = (
    "native_cpu",
    "jax_gpu_custom",
    "jax_gpu_optax",
    "jax_cpu_custom",
)
_EXPECTED_SAMPLE_SEQUENCE: tuple[tuple[SamplePhase, int], ...] = (
    ("cold", 0),
    ("warmup", 0),
    *(("warm", index) for index in range(7)),
)
_REQUIRED_NON_NATIVE_PARITY_OBSERVABLES = frozenset(
    (
        "final_objective",
        "final_iota",
        "final_volume",
        "final_non_qs_ratio",
        "final_boozer_residual",
    )
)
_ENDPOINT_TRAJECTORY_REL_TOLERANCE = 1e-6
_CLAIM_BEARING_CAMPAIGN_ID = "single-stage-speed-20260804"
_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_EXPECTED_ENDPOINT_IDENTITIES: dict[ReceiptLaneId, tuple[str, str]] = {
    "native_cpu": (
        "native_cpu",
        "simsopt_scipy_bfgs_with_boozer_newton",
    ),
    "jax_gpu_custom": ("jax_gpu_fast", JAX_FAST_DRIVER_ID),
    "jax_gpu_optax": ("jax_gpu_fast", JAX_OPTAX_DRIVER_ID),
    "jax_cpu_custom": ("jax_cpu_fast", JAX_FAST_DRIVER_ID),
}
_EXPECTED_ADJOINT_ROUTES: dict[ReceiptLaneId, AdjointRoute | None] = {
    "native_cpu": None,
    "jax_gpu_custom": DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
    "jax_gpu_optax": DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
    "jax_cpu_custom": DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
}


@dataclass(frozen=True)
class CampaignMetadata:
    """Identity and execution metadata persisted in ``campaign.json``."""

    campaign_id: str
    git_describe: str
    hostname: str
    device_name: str
    python_version: str
    jax_version: str
    iteration_budget: int
    scale: Literal["native_default"]
    created_utc: str


@dataclass(frozen=True)
class TrajectoryPoint:
    """One optimizer iteration's objective and elapsed wall time in seconds."""

    iteration: int
    objective: float
    wall_seconds_from_start: float


@dataclass(frozen=True)
class SampleMeasurement:
    """One cold, warmup, or warm sample with its complete trajectory."""

    phase: SamplePhase
    sample_index: int
    wall_seconds: float
    trajectory: tuple[TrajectoryPoint, ...]


@dataclass(frozen=True)
class EndpointObservables:
    """Terminal scientific observables and inner-solver validity for one lane."""

    final_objective: float
    final_iota: float
    final_volume: float
    final_non_qs_ratio: float
    final_boozer_residual: float
    inner_solver_success: bool


@dataclass(frozen=True)
class ParityRow:
    """One frozen-tolerance-table comparison against the native lane."""

    observable: str
    native_value: float
    lane_value: float
    tolerance: float


@dataclass(frozen=True)
class EndpointCertificateAudit:
    """Persisted scientific endpoint certificate without changing its verdict."""

    success: bool
    initial_stationary: bool
    terminal_stationary: bool
    constraints_satisfied: bool
    outer_status: int


@dataclass(frozen=True)
class EndpointAudit:
    """Provenance and optimizer outcome required to audit one terminal endpoint."""

    backend_mode: str
    driver: str
    input_fingerprint: str
    configuration_fingerprint: str
    effective_construction_fingerprint: str
    initial_parameters_sha256: str
    final_parameters_sha256: str
    final_gradient_inf_norm: float
    normalized_status: str
    raw_status: str
    nit: int
    nfev: int
    njev: int
    certificate: EndpointCertificateAudit
    adjoint_route: AdjointRoute | None = None


@dataclass(frozen=True)
class LaneEndpoint:
    """Terminal fp64 endpoint and required native-comparison evidence."""

    observables: EndpointObservables
    precision: Literal["fp64"]
    audit: EndpointAudit
    parity_rows: tuple[ParityRow, ...] = ()


@dataclass(frozen=True)
class LaneReceipt:
    """All sample and terminal evidence belonging to one campaign lane."""

    lane_id: ReceiptLaneId
    samples: tuple[SampleMeasurement, ...]
    endpoint: LaneEndpoint


@dataclass(frozen=True)
class CampaignReceipt:
    """A complete campaign receipt whose lane IDs and sample schedule are exact."""

    metadata: CampaignMetadata
    lanes: tuple[LaneReceipt, ...]


def _finite_positive(value: float, field: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{field} must be positive and finite")


def _finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")


def _finite_nonnegative(value: float, field: str) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")


def _nonempty(value: str, field: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field} must be non-empty")


def _sha256(value: str, field: str) -> None:
    if len(value) != _SHA256_LENGTH or any(
        character not in _HEX_DIGITS for character in value
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")


def _validate_metadata(metadata: CampaignMetadata) -> None:
    for field, value in (
        ("campaign_id", metadata.campaign_id),
        ("git_describe", metadata.git_describe),
        ("hostname", metadata.hostname),
        ("device_name", metadata.device_name),
        ("python_version", metadata.python_version),
        ("jax_version", metadata.jax_version),
        ("created_utc", metadata.created_utc),
    ):
        _nonempty(value, field)
    if metadata.iteration_budget <= 0:
        raise ValueError("iteration_budget must be positive")
    if metadata.scale != "native_default":
        raise ValueError("scale must be native_default")
    if "T" not in metadata.created_utc:
        raise ValueError("created_utc must be a UTC ISO timestamp")
    created_at = datetime.fromisoformat(metadata.created_utc)
    if created_at.tzinfo is None or created_at.utcoffset() != timedelta(0):
        raise ValueError("created_utc must be a UTC ISO timestamp")


def _validate_trajectory(
    trajectory: tuple[TrajectoryPoint, ...], *, field: str, iteration_budget: int
) -> None:
    if not trajectory:
        raise ValueError(f"{field} must contain at least one iteration")
    previous_iteration = 0
    previous_wall_seconds = 0.0
    for point in trajectory:
        if isinstance(point.iteration, bool) or not isinstance(point.iteration, int):
            raise TypeError(f"{field} iteration must be an integer")
        if point.iteration != previous_iteration + 1:
            raise ValueError(f"{field} iterations must be contiguous from 1")
        if point.iteration > iteration_budget:
            raise ValueError(f"{field} exceeds iteration budget")
        _finite(point.objective, f"{field} objective")
        _finite_nonnegative(point.wall_seconds_from_start, f"{field} wall time")
        if point.wall_seconds_from_start < previous_wall_seconds:
            raise ValueError(f"{field} wall times must be nondecreasing")
        previous_iteration = point.iteration
        previous_wall_seconds = point.wall_seconds_from_start


def _validate_lane(lane: LaneReceipt, *, iteration_budget: int) -> None:
    sample_sequence = tuple(
        (sample.phase, sample.sample_index) for sample in lane.samples
    )
    if sample_sequence != _EXPECTED_SAMPLE_SEQUENCE:
        raise ValueError(
            f"{lane.lane_id} samples must be ordered 1 cold, 1 warmup, and 7 warm"
        )
    for sample in lane.samples:
        _finite_positive(sample.wall_seconds, f"{lane.lane_id} sample wall_seconds")
        _validate_trajectory(
            sample.trajectory,
            field=(f"{lane.lane_id} {sample.phase} {sample.sample_index} trajectory"),
            iteration_budget=iteration_budget,
        )
        if sample.trajectory[-1].wall_seconds_from_start > sample.wall_seconds:
            raise ValueError(
                f"{lane.lane_id} {sample.phase} {sample.sample_index} trajectory "
                "exceeds sample wall_seconds"
            )
    for field, value in (
        ("final_objective", lane.endpoint.observables.final_objective),
        ("final_iota", lane.endpoint.observables.final_iota),
        ("final_volume", lane.endpoint.observables.final_volume),
        ("final_non_qs_ratio", lane.endpoint.observables.final_non_qs_ratio),
        ("final_boozer_residual", lane.endpoint.observables.final_boozer_residual),
    ):
        _finite(value, field)
    if not lane.endpoint.observables.inner_solver_success:
        raise ValueError(f"{lane.lane_id} inner solver must report success")
    if lane.endpoint.precision != "fp64":
        raise ValueError(f"{lane.lane_id} precision must be fp64")
    if lane.lane_id == "native_cpu":
        if lane.endpoint.parity_rows:
            raise ValueError("native_cpu must not contain parity rows")
    elif not lane.endpoint.parity_rows:
        raise ValueError(f"{lane.lane_id} must contain parity rows")
    if lane.lane_id != "native_cpu":
        observables = tuple(row.observable for row in lane.endpoint.parity_rows)
        if (
            len(observables) != len(_REQUIRED_NON_NATIVE_PARITY_OBSERVABLES)
            or frozenset(observables) != _REQUIRED_NON_NATIVE_PARITY_OBSERVABLES
        ):
            raise ValueError(
                f"{lane.lane_id} parity rows must contain exactly "
                "final objective, iota, volume, non-QS ratio, and Boozer residual"
            )
    for row in lane.endpoint.parity_rows:
        _nonempty(row.observable, "parity observable")
        _finite(row.native_value, "parity native_value")
        _finite(row.lane_value, "parity lane_value")
        _finite_nonnegative(row.tolerance, "parity tolerance")
        if abs(row.lane_value - row.native_value) > row.tolerance:
            raise ValueError(
                f"{lane.lane_id} parity row {row.observable} exceeds tolerance"
            )
    _validate_endpoint_audit(lane.endpoint.audit)
    expected_backend_mode, expected_driver = _EXPECTED_ENDPOINT_IDENTITIES[lane.lane_id]
    if (
        lane.endpoint.audit.backend_mode != expected_backend_mode
        or lane.endpoint.audit.driver != expected_driver
    ):
        raise ValueError(
            f"{lane.lane_id} endpoint backend/driver identity does not match protocol"
        )
    expected_adjoint_route = _EXPECTED_ADJOINT_ROUTES[lane.lane_id]
    if lane.endpoint.audit.adjoint_route != expected_adjoint_route:
        raise ValueError(
            f"{lane.lane_id} endpoint adjoint route does not match protocol"
        )
    if not lane.endpoint.audit.certificate.constraints_satisfied:
        raise ValueError(
            f"{lane.lane_id} endpoint certificate constraints must be satisfied"
        )
    warm_sample = lane.samples[-1]
    if lane.endpoint.audit.nit != len(warm_sample.trajectory):
        raise ValueError(
            f"{lane.lane_id} endpoint nit must match warm sample 6 trajectory"
        )

    warm_final = next(
        sample.trajectory[-1].objective
        for sample in lane.samples
        if sample.phase == "warm" and sample.sample_index == 6
    )
    if not math.isclose(
        lane.endpoint.observables.final_objective,
        warm_final,
        rel_tol=_ENDPOINT_TRAJECTORY_REL_TOLERANCE,
        abs_tol=0.0,
    ):
        raise ValueError(
            f"{lane.lane_id} final_objective must match warm sample 6 final objective"
        )


def _validate_endpoint_audit(audit: EndpointAudit) -> None:
    for field, value in (
        ("backend_mode", audit.backend_mode),
        ("driver", audit.driver),
        ("input_fingerprint", audit.input_fingerprint),
        ("configuration_fingerprint", audit.configuration_fingerprint),
        (
            "effective_construction_fingerprint",
            audit.effective_construction_fingerprint,
        ),
        ("normalized_status", audit.normalized_status),
        ("raw_status", audit.raw_status),
    ):
        _nonempty(value, field)
    _sha256(audit.input_fingerprint, "input_fingerprint")
    _sha256(audit.configuration_fingerprint, "configuration_fingerprint")
    _sha256(
        audit.effective_construction_fingerprint,
        "effective_construction_fingerprint",
    )
    _sha256(audit.initial_parameters_sha256, "initial_parameters_sha256")
    _sha256(audit.final_parameters_sha256, "final_parameters_sha256")
    _finite_nonnegative(audit.final_gradient_inf_norm, "final_gradient_inf_norm")
    for field, value in (
        ("nit", audit.nit),
        ("nfev", audit.nfev),
        ("njev", audit.njev),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a nonnegative integer")
    if isinstance(audit.certificate.outer_status, bool) or not isinstance(
        audit.certificate.outer_status, int
    ):
        raise TypeError("outer_status must be an integer")
    for field, value in (
        ("certificate.success", audit.certificate.success),
        ("certificate.initial_stationary", audit.certificate.initial_stationary),
        ("certificate.terminal_stationary", audit.certificate.terminal_stationary),
        (
            "certificate.constraints_satisfied",
            audit.certificate.constraints_satisfied,
        ),
    ):
        if not isinstance(value, bool):
            raise TypeError(f"{field} must be a boolean")
    if audit.normalized_status not in {"converged", "budget_exhausted"}:
        raise ValueError("normalized_status must be converged or budget_exhausted")
    if audit.certificate.success != (audit.normalized_status == "converged"):
        raise ValueError("endpoint certificate success contradicts normalized_status")


def _endpoint_observable(endpoint: LaneEndpoint, observable: str) -> float:
    values = {
        "final_objective": endpoint.observables.final_objective,
        "final_iota": endpoint.observables.final_iota,
        "final_volume": endpoint.observables.final_volume,
        "final_non_qs_ratio": endpoint.observables.final_non_qs_ratio,
        "final_boozer_residual": endpoint.observables.final_boozer_residual,
    }
    return values[observable]


def _validate_receipt(receipt: CampaignReceipt) -> None:
    _validate_metadata(receipt.metadata)
    lane_ids = tuple(lane.lane_id for lane in receipt.lanes)
    if lane_ids != RECEIPT_LANE_IDS:
        raise ValueError(f"lane IDs must be exactly {RECEIPT_LANE_IDS}")
    for lane in receipt.lanes:
        _validate_lane(lane, iteration_budget=receipt.metadata.iteration_budget)
    native = receipt.lanes[0].endpoint
    for lane in receipt.lanes[1:]:
        audit = lane.endpoint.audit
        for field in (
            "input_fingerprint",
            "configuration_fingerprint",
            "effective_construction_fingerprint",
            "initial_parameters_sha256",
        ):
            if getattr(audit, field) != getattr(native.audit, field):
                raise ValueError(
                    f"{lane.lane_id} endpoint {field} does not match native_cpu"
                )
        for row in lane.endpoint.parity_rows:
            if row.native_value != _endpoint_observable(
                native, row.observable
            ) or row.lane_value != _endpoint_observable(lane.endpoint, row.observable):
                raise ValueError(
                    f"{lane.lane_id} parity row {row.observable} is not endpoint-bound"
                )


def _validate_artifact_root(artifact_root: Path, receipt: CampaignReceipt) -> None:
    if artifact_root.exists():
        raise FileExistsError(f"artifact root already exists: {artifact_root}")
    if (
        receipt.metadata.campaign_id == _CLAIM_BEARING_CAMPAIGN_ID
        and artifact_root.resolve().is_relative_to(Path("/tmp"))
    ):
        raise ValueError(
            "claim-bearing campaign receipts must not be written under /tmp"
        )


def _json_bytes(document: object) -> bytes:
    return (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, document: object) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(document))
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(path.parent)


def _campaign_document(metadata: CampaignMetadata) -> dict[str, object]:
    return {
        "campaign_id": metadata.campaign_id,
        "lanes": list(RECEIPT_LANE_IDS),
        "git_describe": metadata.git_describe,
        "hostname": metadata.hostname,
        "device_name": metadata.device_name,
        "python_version": metadata.python_version,
        "jax_version": metadata.jax_version,
        "iteration_budget": metadata.iteration_budget,
        "scale": metadata.scale,
        "created_utc": metadata.created_utc,
    }


def _measurement_document(samples: tuple[SampleMeasurement, ...]) -> dict[str, object]:
    return {
        "samples": [
            {
                "phase": sample.phase,
                "sample_index": sample.sample_index,
                "wall_seconds": sample.wall_seconds,
            }
            for sample in samples
        ]
    }


def _endpoint_document(endpoint: LaneEndpoint) -> dict[str, object]:
    document: dict[str, object] = {
        "observables": {
            "final_objective": endpoint.observables.final_objective,
            "final_iota": endpoint.observables.final_iota,
            "final_volume": endpoint.observables.final_volume,
            "final_non_qs_ratio": endpoint.observables.final_non_qs_ratio,
            "final_boozer_residual": endpoint.observables.final_boozer_residual,
            "inner_solver_success": endpoint.observables.inner_solver_success,
        },
        "precision": endpoint.precision,
        "audit": {
            "backend_mode": endpoint.audit.backend_mode,
            "driver": endpoint.audit.driver,
            "adjoint_route": endpoint.audit.adjoint_route,
            "input_fingerprint": endpoint.audit.input_fingerprint,
            "configuration_fingerprint": endpoint.audit.configuration_fingerprint,
            "effective_construction_fingerprint": endpoint.audit.effective_construction_fingerprint,
            "initial_parameters_sha256": endpoint.audit.initial_parameters_sha256,
            "final_parameters_sha256": endpoint.audit.final_parameters_sha256,
            "final_gradient_inf_norm": endpoint.audit.final_gradient_inf_norm,
            "normalized_status": endpoint.audit.normalized_status,
            "raw_status": endpoint.audit.raw_status,
            "nit": endpoint.audit.nit,
            "nfev": endpoint.audit.nfev,
            "njev": endpoint.audit.njev,
            "certificate": {
                "success": endpoint.audit.certificate.success,
                "initial_stationary": endpoint.audit.certificate.initial_stationary,
                "terminal_stationary": endpoint.audit.certificate.terminal_stationary,
                "constraints_satisfied": endpoint.audit.certificate.constraints_satisfied,
                "outer_status": endpoint.audit.certificate.outer_status,
            },
        },
    }
    if endpoint.parity_rows:
        document["parity"] = {
            "rows": [
                {
                    "observable": row.observable,
                    "native_value": row.native_value,
                    "lane_value": row.lane_value,
                    "tolerance": row.tolerance,
                }
                for row in endpoint.parity_rows
            ]
        }
    return document


def _write_receipt_tree(artifact_root: Path, receipt: CampaignReceipt) -> None:
    _write_json(artifact_root / "campaign.json", _campaign_document(receipt.metadata))
    lanes_root = artifact_root / "lanes"
    lanes_root.mkdir()
    _fsync_directory(artifact_root)
    for lane in receipt.lanes:
        lane_root = lanes_root / lane.lane_id
        lane_root.mkdir()
        _fsync_directory(lanes_root)
        _write_json(lane_root / "measurement.json", _measurement_document(lane.samples))
        _write_json(lane_root / "endpoint.json", _endpoint_document(lane.endpoint))
        for sample in lane.samples:
            trajectory_path = (
                lane_root / f"trajectory-{sample.phase}-{sample.sample_index}.jsonl"
            )
            with trajectory_path.open("xb") as stream:
                for point in sample.trajectory:
                    stream.write(
                        _json_bytes(
                            {
                                "iteration": point.iteration,
                                "objective": point.objective,
                                "wall_seconds_from_start": point.wall_seconds_from_start,
                            }
                        )
                    )
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(lane_root)


def write_campaign_receipt(artifact_root: Path, receipt: CampaignReceipt) -> Path:
    """Create one fresh, complete receipt tree and return its artifact root.

    ``artifact_root`` must not exist.  A sibling staging directory is renamed
    only after all protocol files are durable, so the final root is never partial.
    """

    _validate_receipt(receipt)
    _validate_artifact_root(artifact_root, receipt)
    artifact_root.parent.mkdir(parents=True, exist_ok=True)
    _fsync_directory(artifact_root.parent)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{artifact_root.name}.staging-", dir=artifact_root.parent
        )
    )
    try:
        _write_receipt_tree(staging_root, receipt)
        staging_root.rename(artifact_root)
        _fsync_directory(artifact_root.parent)
    finally:
        if staging_root.exists():
            shutil.rmtree(staging_root)
    return artifact_root
