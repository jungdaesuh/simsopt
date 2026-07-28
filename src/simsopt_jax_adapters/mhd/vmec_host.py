"""Hash-bound host receipts for VMEC evaluations in hybrid JAX workflows."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def boundary_sha256(boundary_dofs: np.ndarray) -> str:
    """Return the canonical SHA-256 of a float64 boundary vector."""
    values = np.ascontiguousarray(boundary_dofs, dtype="<f8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class VmecHostEvaluation:
    """Immutable evidence tying one VMEC result to its boundary and process."""

    boundary_sha256: str
    output_file: str
    output_sha256: str
    output_mtime_ns: int
    started_ns: int
    vmec_iteration_before: int
    vmec_iteration_after: int
    mpi_world_size: int
    success: bool
    objective: float
    gradient: np.ndarray
    elapsed_seconds: float
    evaluation_count: int

    @classmethod
    def from_output(
        cls,
        *,
        boundary_dofs: np.ndarray,
        output_file: Path,
        vmec_iteration_before: int,
        vmec_iteration_after: int,
        mpi_world_size: int,
        objective: float,
        gradient: np.ndarray,
        elapsed_seconds: float,
        evaluation_count: int,
        started_ns: int,
    ) -> VmecHostEvaluation:
        stat = output_file.stat()
        return cls(
            boundary_sha256=boundary_sha256(boundary_dofs),
            output_file=str(output_file.resolve()),
            output_sha256=_file_sha256(output_file),
            output_mtime_ns=int(stat.st_mtime_ns),
            started_ns=int(started_ns),
            vmec_iteration_before=int(vmec_iteration_before),
            vmec_iteration_after=int(vmec_iteration_after),
            mpi_world_size=int(mpi_world_size),
            success=True,
            objective=float(objective),
            gradient=np.ascontiguousarray(gradient, dtype=np.float64),
            elapsed_seconds=float(elapsed_seconds),
            evaluation_count=int(evaluation_count),
        )


def validate_vmec_host_evaluation(
    evaluation: VmecHostEvaluation,
    *,
    expected_boundary_sha256: str,
    expected_mpi_world_size: int,
    expected_gradient_size: int,
) -> None:
    """Reject stale, mismatched, non-finite, or incomplete VMEC evidence."""
    if evaluation.boundary_sha256 != expected_boundary_sha256:
        raise RuntimeError("VMEC boundary hash does not match the requested boundary.")
    if evaluation.vmec_iteration_after <= evaluation.vmec_iteration_before:
        raise RuntimeError("VMEC iteration did not advance for the requested boundary.")
    if evaluation.mpi_world_size != expected_mpi_world_size:
        raise RuntimeError("VMEC MPI world size does not match the configured layout.")
    if not evaluation.success:
        raise RuntimeError("VMEC solve failed.")
    if not evaluation.output_file or not evaluation.output_sha256:
        raise RuntimeError("VMEC output receipt is missing.")
    if evaluation.output_mtime_ns < evaluation.started_ns:
        raise RuntimeError("VMEC output receipt is stale.")
    if evaluation.gradient.size != expected_gradient_size:
        raise RuntimeError("VMEC gradient size does not match the surface layout.")
    if not np.isfinite(evaluation.objective):
        raise RuntimeError("VMEC objective is non-finite.")
    if not np.all(np.isfinite(evaluation.gradient)):
        raise RuntimeError("VMEC gradient is non-finite.")
    if evaluation.elapsed_seconds < 0.0 or evaluation.evaluation_count < 1:
        raise RuntimeError("VMEC evaluation metadata is invalid.")


__all__ = (
    "VmecHostEvaluation",
    "boundary_sha256",
    "validate_vmec_host_evaluation",
)
