"""Contracts for the single-stage initial-gradient agreement artifact."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
from benchmarks.single_stage_initial_gradient_agreement import (
    compare_probe_documents,
)


def _probe(route: str, gradient: list[float]) -> dict[str, object]:
    return {
        "route": route,
        "input_fingerprint": "1" * 64,
        "configuration_fingerprint": "2" * 64,
        "effective_construction_fingerprint": "3" * 64,
        "initial_parameters_sha256": "4" * 64,
        "initial_objective": 1.0,
        "gradient": gradient,
        "adjoint_route": route,
    }


def test_comparator_records_frozen_tolerance_agreement_metrics() -> None:
    parity = _probe("parity", [1.0, -2.0, 0.0])
    direct = _probe("direct", [1.0 + 1.0e-10, -2.0, 1.0e-13])

    agreement = compare_probe_documents(direct, parity)

    assert agreement["passed"] is True
    assert agreement["tolerance_contract"] == "mirror_single_stage_initial_gradient"
    assert agreement["rtol"] == 2.0e-9
    assert agreement["atol"] == 2.0e-12
    assert agreement["gradient_dimension"] == 3
    assert agreement["max_abs_difference"] == pytest.approx(1.0e-10)
    assert float(agreement["max_tolerance_ratio"]) < 1.0


def test_comparator_rejects_identity_drift_and_out_of_tolerance_gradient() -> None:
    parity = _probe("parity", [1.0, 2.0])
    drifted = deepcopy(_probe("direct", [1.0, 2.0]))
    drifted["initial_parameters_sha256"] = "5" * 64

    with pytest.raises(ValueError, match="initial_parameters_sha256"):
        compare_probe_documents(drifted, parity)

    disagreeing = _probe("direct", [1.0, 2.1])
    with pytest.raises(ValueError, match="max tolerance ratio"):
        compare_probe_documents(disagreeing, parity)


def test_comparator_rejects_nonfinite_and_shape_mismatched_gradients() -> None:
    parity = _probe("parity", [1.0, 2.0])

    with pytest.raises(ValueError, match="must be finite"):
        compare_probe_documents(_probe("direct", [1.0, np.nan]), parity)
    with pytest.raises(ValueError, match="matching nonempty FP64 vectors"):
        compare_probe_documents(_probe("direct", [1.0]), parity)
