"""Derived review package export from a retained local parity run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.derived_review_summary import (
    build_derived_review_summary,
    main,
    render_derived_review_summary,
)

_CASE_ID = "synthetic-case"
_LANE_RECEIPT = {
    "lane": "native-cpu",
    "backend_mode": "native",
    "platform": "cpu",
    "precision": "float64",
    "driver": "synthetic",
    "scale": "bounded",
    "success": True,
    "normalized_status": "converged",
    "raw_status": "0",
    "nit": 2,
    "nfev": 3,
    "njev": 1,
    "input_fingerprint": "input-fingerprint",
    "configuration_fingerprint": "configuration-fingerprint",
    "effective_construction_fingerprint": "construction-fingerprint",
    "completed_workflow_stages": ["construct", "solve"],
    "applicability": "applicable",
    "schema_version": 1,
}
_PROVENANCE = {
    "repository_commit": "0" * 40,
    "authoritative": True,
    "simsoptpp_sha256": "1" * 64,
    "simsoptpp_build_commit": "2" * 40,
    "generated_source_bindings": {
        "build/simsoptpp.so": "local-build-receipt-sha256:" + "3" * 64,
        "src/simsopt/_version.py": "setuptools-scm checkout commit",
    },
    "jax_version": "test-jax",
    "python_version": "3.11.15",
    "devices": [{"platform": "cpu", "kind": "cpu"}],
    "jax_effective_transfer_guards": {"transfer_guard": "allow"},
}


def _write_array(path: Path, values: list[float]) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    return {
        "dtype": array.dtype.str,
        "order": "C",
        "path": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "shape": list(array.shape),
    }


@pytest.fixture(name="run_directory")
def _run_directory(tmp_path: Path) -> Path:
    """Build the smallest retained run layout the exporter reads."""
    run = tmp_path / "20260101T000000Z-abcd1234"
    lane_directory = run / _CASE_ID / "native-cpu"
    endpoint = _write_array(lane_directory / "values" / "final_objective.npy", [0.5])
    endpoint["path"] = f"values/{endpoint['path']}"
    receipt = {
        **_LANE_RECEIPT,
        "provenance": _PROVENANCE,
        "values": {"final:objective": endpoint},
    }
    lane_directory.joinpath("lane_result.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8"
    )
    input_directory = run / _CASE_ID / "inputs"
    coil_dofs = _write_array(input_directory / "inputs" / "coil_dofs.npy", [1.0, -2.0])
    coil_dofs["path"] = f"inputs/{coil_dofs['path']}"
    input_directory.joinpath("input_bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "case_id": _CASE_ID,
                "scale": "bounded",
                "random_seed": 0,
                "configuration": {"maxiter": 2},
                "configuration_fingerprint": "configuration-fingerprint",
                "input_fingerprint": "input-fingerprint",
                "arrays": {"coil_dofs": coil_dofs},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run.joinpath("summary.json").write_text(
        json.dumps(
            {
                "run_id": run.name,
                "repository_commit": "0" * 40,
                "scale": "bounded",
                "verdict": "pass",
                "authoritative": True,
                "lanes": ["native-cpu"],
                "cases": [
                    {
                        "case_id": _CASE_ID,
                        "verdict": "pass",
                        "scale_tier": "bounded",
                        "authoritative": True,
                        "comparisons": [{"name": "final:objective", "passed": True}],
                        "executions": [
                            {
                                "lane": "native-cpu",
                                "returncode": 0,
                                "result_directory": f"{_CASE_ID}/native-cpu",
                            }
                        ],
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return run


def test_export_is_byte_deterministic(run_directory: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second" / "review-summary.json"

    assert main(["--run", str(run_directory), "--output", str(first)]) == 0
    assert main(["--run", str(run_directory), "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8") == render_derived_review_summary(
        run_directory
    )


def test_export_embeds_the_original_artifact_hashes(run_directory: Path) -> None:
    report = build_derived_review_summary(run_directory)
    lane = report["lanes"][0]
    receipt_path = run_directory / _CASE_ID / "native-cpu" / "lane_result.json"
    bundle_path = run_directory / _CASE_ID / "inputs" / "input_bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert report["evidence_kind"] == "derived_numerical_review_only"
    assert (
        report["original_summary_sha256"]
        == hashlib.sha256((run_directory / "summary.json").read_bytes()).hexdigest()
    )
    assert (
        lane["original_receipt_sha256"]
        == hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    )
    assert lane["native_build_receipt_sha256"] == "3" * 64
    assert (
        lane["values"]["final:objective"]["original_npy_sha256"]
        == hashlib.sha256(
            (
                run_directory
                / _CASE_ID
                / "native-cpu"
                / "values"
                / "final_objective.npy"
            ).read_bytes()
        ).hexdigest()
    )
    assert lane["values"]["final:objective"]["values"] == [0.5]
    assert (
        report["input_bundle"]["arrays"]["coil_dofs"]["original_npy_sha256"]
        == bundle["arrays"]["coil_dofs"]["sha256"]
    )
    assert report["input_bundle"]["arrays"]["coil_dofs"]["values"] == [1.0, -2.0]


def test_export_rejects_an_array_that_lost_its_recorded_hash(
    run_directory: Path,
) -> None:
    array_path = (
        run_directory / _CASE_ID / "native-cpu" / "values" / "final_objective.npy"
    )
    with array_path.open("wb") as stream:
        np.save(stream, np.asarray([0.75], dtype=np.float64), allow_pickle=False)

    with pytest.raises(ValueError, match="does not match its receipt"):
        build_derived_review_summary(run_directory)
