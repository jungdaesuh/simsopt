"""Fail-closed VMEC host-evaluation receipt tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from simsopt_jax_adapters.mhd.vmec_host import (
    VmecHostEvaluation,
    boundary_sha256,
    hybrid_result_is_scientifically_successful,
    validate_vmec_host_evaluation,
    vmec_result_is_receiptable,
)


def _receipt(tmp_path: Path) -> VmecHostEvaluation:
    output = tmp_path / "wout_test.nc"
    output.write_bytes(b"authoritative VMEC output")
    boundary = np.asarray((1.0, 0.2, -0.1), dtype=np.float64)
    return VmecHostEvaluation.from_output(
        boundary_dofs=boundary,
        output_file=output,
        vmec_iteration_before=2,
        vmec_iteration_after=3,
        mpi_world_size=4,
        objective=0.125,
        gradient=np.asarray((0.5, -0.25, 0.75), dtype=np.float64),
        elapsed_seconds=1.5,
        evaluation_count=7,
        started_ns=output.stat().st_mtime_ns,
    )


def test_vmec_host_receipt_accepts_hash_bound_fresh_complete_evaluation(
    tmp_path: Path,
) -> None:
    receipt = _receipt(tmp_path)
    validate_vmec_host_evaluation(
        receipt,
        expected_boundary_sha256=boundary_sha256(
            np.asarray((1.0, 0.2, -0.1), dtype=np.float64)
        ),
        expected_mpi_world_size=4,
        expected_gradient_size=3,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"boundary_sha256": "0" * 64}, "boundary hash"),
        ({"vmec_iteration_after": 2}, "did not advance"),
        ({"mpi_world_size": 1}, "MPI world size"),
        ({"success": False}, "solve failed"),
        ({"output_sha256": ""}, "output receipt"),
        ({"output_mtime_ns": 0}, "stale"),
        ({"gradient": np.asarray((1.0, 2.0))}, "gradient size"),
    ),
)
def test_vmec_host_receipt_rejects_incomplete_or_mismatched_authority(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    receipt = replace(_receipt(tmp_path), **mutation)

    with pytest.raises(RuntimeError, match=message):
        validate_vmec_host_evaluation(
            receipt,
            expected_boundary_sha256=boundary_sha256(
                np.asarray((1.0, 0.2, -0.1), dtype=np.float64)
            ),
            expected_mpi_world_size=4,
            expected_gradient_size=3,
        )


def test_hybrid_scientific_success_rejects_failure_threshold_cap() -> None:
    assert not hybrid_result_is_scientifically_successful(
        initial_objective=100.0,
        final_objective=100.0,
        final_gradient=np.zeros(3, dtype=np.float64),
        failure_threshold=100.0,
        expected_boundary_sha256="a" * 64,
        final_boundary_sha256="a" * 64,
    )


def test_failed_vmec_trial_without_output_is_not_receiptable(tmp_path: Path) -> None:
    missing_output = tmp_path / "missing_wout.nc"

    assert not vmec_result_is_receiptable(
        objective=1.0e12,
        failure_threshold=100.0,
        output_file=missing_output,
    )
    assert hybrid_result_is_scientifically_successful(
        initial_objective=2.0,
        final_objective=1.0,
        final_gradient=np.asarray((0.1, -0.2, 0.3), dtype=np.float64),
        failure_threshold=100.0,
        expected_boundary_sha256="a" * 64,
        final_boundary_sha256="a" * 64,
    )
