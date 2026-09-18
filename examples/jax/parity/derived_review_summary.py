"""Export inspectable numerical evidence, never a canonical authority bundle.

The exported package restates one retained local parity run as plain JSON: every
lane receipt field, the endpoint and input array values, and the original
SHA-256 of each artifact it read. It is derivative review evidence whose bytes
are bound by `derived_summary_sha256` in `examples/jax/authority_evidence.json`;
it never establishes authority on its own.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

_LANE_RECEIPT_FIELDS = (
    "lane",
    "backend_mode",
    "platform",
    "precision",
    "driver",
    "scale",
    "success",
    "normalized_status",
    "raw_status",
    "nit",
    "nfev",
    "njev",
    "input_fingerprint",
    "configuration_fingerprint",
    "effective_construction_fingerprint",
    "completed_workflow_stages",
    "applicability",
)
_BUILD_RECEIPT_PREFIX = "local-build-receipt-sha256:"
_SCHEMA = "boozer-parity-review-summary-v1"
_LIMITATIONS = (
    "Not a canonical published bundle or independently portable build attestation.",
    "Original local audit requires its clean source checkout, binary, build "
    "receipt, and full bundle.",
    "Endpoint quality does not establish strict parity or optimizer convergence.",
)


def _read_bound_array(
    directory: Path, descriptor: dict[str, object]
) -> dict[str, object]:
    """Read one hash-bound .npy descriptor into a JSON-serialisable record."""
    array_path = directory / str(descriptor["path"])
    payload = array_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != descriptor["sha256"]:
        raise ValueError(f"retained array does not match its receipt: {array_path}")
    array = np.load(array_path, allow_pickle=False)
    if (
        list(array.shape) != descriptor["shape"]
        or array.dtype.str != descriptor["dtype"]
    ):
        raise ValueError(f"retained array does not match its dtype/shape: {array_path}")
    return {
        "dtype": descriptor["dtype"],
        "shape": descriptor["shape"],
        "original_npy_sha256": descriptor["sha256"],
        "values": array.tolist(),
    }


def _lane_record(lane_directory: Path) -> dict[str, object]:
    """Restate one retained lane receipt with its array values inlined."""
    receipt_bytes = (lane_directory / "lane_result.json").read_bytes()
    receipt = json.loads(receipt_bytes)
    provenance = receipt["provenance"]
    build_bindings = [
        value.removeprefix(_BUILD_RECEIPT_PREFIX)
        for value in provenance["generated_source_bindings"].values()
        if value.startswith(_BUILD_RECEIPT_PREFIX)
    ]
    if len(build_bindings) != 1:
        raise ValueError(
            f"lane receipt lacks exactly one native build binding: {lane_directory}"
        )
    return {
        **{field: receipt[field] for field in _LANE_RECEIPT_FIELDS},
        "original_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "repository_commit": provenance["repository_commit"],
        "original_authoritative_flag": provenance["authoritative"],
        "native_binary_sha256": provenance["simsoptpp_sha256"],
        "native_build_source_commit": provenance["simsoptpp_build_commit"],
        "native_build_receipt_sha256": build_bindings[0],
        "jax_version": provenance["jax_version"],
        "python_version": provenance["python_version"],
        "devices": provenance["devices"],
        "jax_effective_transfer_guards": provenance["jax_effective_transfer_guards"],
        "values": {
            name: _read_bound_array(lane_directory, descriptor)
            for name, descriptor in receipt["values"].items()
        },
    }


def build_derived_review_summary(run_directory: Path) -> dict[str, object]:
    """Build the derived review package for one single-case retained run."""
    summary_bytes = (run_directory / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    if len(summary["cases"]) != 1:
        raise ValueError("derived review summary requires a single-case run")
    case = summary["cases"][0]
    lanes = [
        _lane_record(run_directory / execution["result_directory"])
        for execution in case["executions"]
    ]
    input_directory = run_directory / case["case_id"] / "inputs"
    input_bundle = json.loads((input_directory / "input_bundle.json").read_text())
    input_values = {
        name: _read_bound_array(input_directory, descriptor)
        for name, descriptor in input_bundle["arrays"].items()
    }
    return {
        "schema": _SCHEMA,
        "evidence_kind": "derived_numerical_review_only",
        "limitations": list(_LIMITATIONS),
        "run_id": summary["run_id"],
        "repository_commit": summary["repository_commit"],
        "original_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "scale": summary["scale"],
        "original_verdict": summary["verdict"],
        "original_authoritative_flag": summary["authoritative"],
        "case": {key: value for key, value in case.items() if key != "executions"},
        "input_bundle": {
            **{key: value for key, value in input_bundle.items() if key != "arrays"},
            "arrays": input_values,
        },
        "lanes": lanes,
    }


def render_derived_review_summary(run_directory: Path) -> str:
    """Render the canonical, byte-reproducible text of the derived package."""
    report = build_derived_review_summary(run_directory)
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Write the derived review package for one retained run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        render_derived_review_summary(arguments.run), encoding="utf-8"
    )
    print(arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
