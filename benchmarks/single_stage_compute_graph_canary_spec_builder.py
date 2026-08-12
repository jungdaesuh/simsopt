"""Build a provenance-bound C1/C2 canary runner specification.

The runner intentionally accepts a small path-only document.  This builder is
the sole preflight that resolves those paths through the completed C0 receipt,
the immutable snapshot, and an exact source-bound variant graph before writing
the runner document exclusively.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, cast

from benchmarks.single_stage_compute_graph_c0_runner import (
    _load_canonical_json_object,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_canary_runner import (
    CANARY_SPEC_SCHEMA_ID,
    EVALUATOR_MODULE,
    CanaryRunnerError,
    CanarySpec,
    validate_spec,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import (
    Phase0ReceiptError,
    load_phase0_receipt,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    ManifestEntry,
    SnapshotError,
    canonical_json_bytes,
    load_snapshot_manifest,
)

VARIANT_SOLVER_GRAPH_SCHEMA_ID: Final = (
    "single-stage-compute-graph-variant-solver-graph-v1"
)
CanaryVariant = Literal["C1", "C2"]

_IMPLEMENTATION_PATHS: Final = (
    "benchmarks/single_stage_compute_graph_canary_evaluator.py",
    "examples/jax/parity/cases/native_boozerqa.py",
    "src/simsopt_jax/geo/optimizers/linear_solve.py",
    "src/simsopt_jax/geo/optimizers/optimizer.py",
    "src/simsopt_jax_adapters/geo/surface_objectives_traceable.py",
)
_IMPLEMENTATION_ROLES: Final = {
    _IMPLEMENTATION_PATHS[0]: "benchmark",
    **{path: "execution_source" for path in _IMPLEMENTATION_PATHS[1:]},
}


class CanarySpecBuilderError(RuntimeError):
    """The requested canary spec is not fully bound to current evidence."""


@dataclass(frozen=True, slots=True)
class CanarySpecBuildInputs:
    variant: CanaryVariant
    c0_spec_path: Path
    c0_receipt_path: Path
    snapshot_publication_path: Path
    import_attestation_path: Path
    qualification_path: Path
    device_probe_path: Path
    runtime_provenance_path: Path
    variant_solver_graph_path: Path
    cache_directory: Path
    output_root: Path
    destination: Path


def _manifest_entry_document(entry: ManifestEntry) -> dict[str, object]:
    return {
        "role": entry.role,
        "relative_path": entry.relative_path,
        "size_bytes": entry.size_bytes,
        "sha256": entry.sha256,
    }


def variant_solver_graph_document(
    snapshot_root: Path,
    variant: CanaryVariant,
) -> dict[str, object]:
    """Derive the exact variant graph identity from manifested source bytes."""

    if variant not in ("C1", "C2"):
        raise CanarySpecBuilderError("variant must be C1 or C2")
    try:
        entries, manifest_sha256 = load_snapshot_manifest(snapshot_root)
    except (OSError, SnapshotError, ValueError) as error:
        raise CanarySpecBuilderError(
            f"snapshot manifest is invalid: {error}"
        ) from error
    by_path = {entry.relative_path: entry for entry in entries}
    if len(by_path) != len(entries):
        raise CanarySpecBuilderError("snapshot manifest contains duplicate paths")
    missing = tuple(path for path in _IMPLEMENTATION_PATHS if path not in by_path)
    if missing:
        raise CanarySpecBuilderError(
            f"snapshot lacks required variant implementation files: {missing}"
        )
    role_drift = tuple(
        path
        for path in _IMPLEMENTATION_PATHS
        if by_path[path].role != _IMPLEMENTATION_ROLES[path]
    )
    if role_drift:
        raise CanarySpecBuilderError(
            f"variant implementation manifest roles drifted: {role_drift}"
        )
    return {
        "schema_id": VARIANT_SOLVER_GRAPH_SCHEMA_ID,
        "variant": variant,
        "snapshot_manifest_sha256": manifest_sha256,
        "selection": {
            "evaluator_module": EVALUATOR_MODULE,
            "runtime_owner": (
                "examples.jax.parity.cases.native_boozerqa._prepare_jax_variant_runtime"
            ),
            "runtime_selector_argument": {
                "exact_newton_variant": variant,
            },
            "production_value_and_gradient_route": (
                "fresh_incumbent_controller.value_and_grad"
            ),
        },
        "manifested_implementation": [
            _manifest_entry_document(by_path[path]) for path in _IMPLEMENTATION_PATHS
        ],
    }


def write_variant_solver_graph(
    destination: Path,
    snapshot_root: Path,
    variant: CanaryVariant,
) -> str:
    """Write one canonical, exclusive, byte-derived variant graph identity."""

    return _write_exclusive_json(
        destination,
        variant_solver_graph_document(snapshot_root, variant),
    )


def _target_receipt_lane(
    receipt: Mapping[str, object],
    gpu_uuid: str,
) -> Mapping[str, object]:
    lanes = receipt.get("lanes")
    if not isinstance(lanes, list):
        raise CanarySpecBuilderError("C0 receipt lanes are invalid")
    matches: list[Mapping[str, object]] = []
    for lane in lanes:
        if not isinstance(lane, dict):
            continue
        measurement = lane.get("measurement")
        if not isinstance(measurement, dict):
            continue
        provenance = measurement.get("provenance")
        allocation = (
            provenance.get("allocation") if isinstance(provenance, dict) else None
        )
        if isinstance(allocation, dict) and allocation.get("gpu_uuid") == gpu_uuid:
            matches.append(lane)
    if len(matches) != 1:
        raise CanarySpecBuilderError(
            "C0 receipt does not contain exactly one measured target GPU lane"
        )
    return matches[0]


def _validate_runtime_contract(
    spec: CanarySpec,
    runtime_provenance: Mapping[str, object],
) -> None:
    try:
        contract = json.loads(spec.runtime_contract_json)
    except json.JSONDecodeError as error:
        raise CanarySpecBuilderError(
            "derived runtime contract is invalid JSON"
        ) from error
    expected = {
        "runtime": runtime_provenance.get("runtime"),
        "static_environment": runtime_provenance.get("environment"),
        "route_environment": {},
        "policies": runtime_provenance.get("policies"),
        "expected_runtime_identity_sha256": spec.runtime_identity_sha256,
    }
    if contract != expected or spec.runtime_contract_json != json.dumps(
        expected,
        sort_keys=True,
        separators=(",", ":"),
    ):
        raise CanarySpecBuilderError(
            "derived runtime contract differs from current runtime provenance"
        )


def _validate_receipt_binding(
    receipt: Mapping[str, object],
    spec: CanarySpec,
    *,
    qualification: Mapping[str, object],
    runtime_provenance: Mapping[str, object],
    gate_checkpoint: Mapping[str, object],
    warm_checkpoint: Mapping[str, object],
) -> None:
    if receipt.get("specimen_sha256") != spec.specimen_sha256:
        raise CanarySpecBuilderError("C0 receipt specimen differs from canary spec")
    lane = _target_receipt_lane(receipt, spec.gpu_uuid)
    if lane.get("qualification") != qualification:
        raise CanarySpecBuilderError(
            "C0 receipt qualification differs from current qualification"
        )
    measurement = lane.get("measurement")
    if not isinstance(measurement, dict):
        raise CanarySpecBuilderError("target C0 receipt lane is not measured")
    if (
        measurement.get("variant") != "C0"
        or measurement.get("specimen_sha256") != spec.specimen_sha256
        or measurement.get("provenance") != runtime_provenance
        or measurement.get("first_evaluation_gate")
        != gate_checkpoint.get("first_evaluation_gate")
        or measurement.get("warm_measurement")
        != warm_checkpoint.get("warm_measurement")
    ):
        raise CanarySpecBuilderError(
            "C0 receipt measurement differs from current provenance/checkpoints"
        )
    warm_measurement = measurement["warm_measurement"]
    if not isinstance(warm_measurement, dict):
        raise CanarySpecBuilderError("C0 receipt warm measurement is invalid")
    samples = warm_measurement.get("samples")
    if not isinstance(samples, list) or not samples:
        raise CanarySpecBuilderError("C0 receipt warm samples are missing")
    peak_rss = max(
        int(cast(dict[str, object], sample)["peak_process_tree_rss_bytes"])
        for sample in samples
        if isinstance(sample, dict)
    )
    peak_gpu = max(
        int(cast(dict[str, object], sample)["sampled_process_gpu_memory_peak_bytes"])
        for sample in samples
        if isinstance(sample, dict)
    )
    expected_metrics = (
        float(warm_measurement["p50_ns"]),
        float(warm_measurement["p95_ns"]),
        peak_rss,
        peak_gpu,
    )
    actual_metrics = (
        spec.c0_p50_ns,
        spec.c0_p95_ns,
        spec.c0_peak_rss_bytes,
        spec.c0_peak_gpu_memory_bytes,
    )
    if actual_metrics != expected_metrics:
        raise CanarySpecBuilderError(
            "canary C0 comparison metrics differ from the measured receipt lane"
        )


def _runner_document(inputs: CanarySpecBuildInputs) -> dict[str, object]:
    return {
        "schema_id": CANARY_SPEC_SCHEMA_ID,
        "variant": inputs.variant,
        "c0_spec_path": str(inputs.c0_spec_path.resolve()),
        "snapshot_publication_path": str(inputs.snapshot_publication_path.resolve()),
        "import_attestation_path": str(inputs.import_attestation_path.resolve()),
        "qualification_path": str(inputs.qualification_path.resolve()),
        "device_probe_path": str(inputs.device_probe_path.resolve()),
        "runtime_provenance_path": str(inputs.runtime_provenance_path.resolve()),
        "variant_solver_graph_path": str(inputs.variant_solver_graph_path.resolve()),
        "cache_directory": str(inputs.cache_directory.resolve()),
        "output_root": str(inputs.output_root.resolve()),
    }


def build_canary_spec(inputs: CanarySpecBuildInputs) -> tuple[dict[str, object], str]:
    """Validate every upstream artifact and exclusively write the runner spec."""

    document = _runner_document(inputs)
    try:
        spec = validate_spec(document)
        qualification = _load_canonical_json_object(
            inputs.qualification_path,
            "canary qualification",
        )
        runtime_provenance = _load_canonical_json_object(
            inputs.runtime_provenance_path,
            "canary runtime provenance",
        )
        c0_spec = _load_canonical_json_object(inputs.c0_spec_path, "C0 runner spec")
        c0_output_value = c0_spec.get("output_root")
        if not isinstance(c0_output_value, str) or not c0_output_value:
            raise CanarySpecBuilderError("C0 runner spec output_root is invalid")
        c0_output_root = Path(c0_output_value).resolve()
        expected_receipt_path = c0_output_root / "phase0-receipt.json"
        if inputs.c0_receipt_path.resolve() != expected_receipt_path:
            raise CanarySpecBuilderError(
                "C0 receipt must be the current receipt in the C0 output root"
            )
        gate_path = c0_output_root / "gate-checkpoint.json"
        warm_path = c0_output_root / "warm-checkpoint.json"
        gate_checkpoint = _load_canonical_json_object(gate_path, "C0 gate checkpoint")
        warm_checkpoint = _load_canonical_json_object(
            warm_path,
            "C0 warm checkpoint",
        )
        receipt_bytes = inputs.c0_receipt_path.read_bytes()
        receipt, _audit = load_phase0_receipt(inputs.c0_receipt_path)
        if receipt_bytes != canonical_json_bytes(receipt):
            raise CanarySpecBuilderError("C0 receipt is not canonical JSON")
    except (CanaryRunnerError, Phase0ReceiptError, OSError, ValueError) as error:
        raise CanarySpecBuilderError(
            f"canary evidence validation failed: {error}"
        ) from error
    _validate_runtime_contract(spec, runtime_provenance)
    _validate_receipt_binding(
        receipt,
        spec,
        qualification=qualification,
        runtime_provenance=runtime_provenance,
        gate_checkpoint=gate_checkpoint,
        warm_checkpoint=warm_checkpoint,
    )
    expected_graph = variant_solver_graph_document(spec.snapshot_root, inputs.variant)
    try:
        graph = _load_canonical_json_object(
            inputs.variant_solver_graph_path,
            "variant solver graph",
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise CanarySpecBuilderError(
            f"variant solver graph is invalid: {error}"
        ) from error
    if graph != expected_graph:
        raise CanarySpecBuilderError(
            "variant solver graph differs from manifested implementation bytes"
        )
    # The runner rebuilds this check before each child.  Doing it here proves the
    # exact evaluator module is already present in the current benchmark manifest.
    from benchmarks.single_stage_compute_graph_canary_runner import child_launches

    try:
        launches = child_launches(spec, {})
    except (OSError, RuntimeError, ValueError) as error:
        raise CanarySpecBuilderError(
            f"manifested evaluator launch is invalid: {error}"
        ) from error
    if not launches or EVALUATOR_MODULE not in launches[0].argv:
        raise CanarySpecBuilderError("manifested evaluator module was not selected")
    digest = _write_exclusive_json(inputs.destination, document)
    return document, digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    graph = commands.add_parser(
        "graph",
        help="write the manifested C1/C2 solver-graph identity",
    )
    graph.add_argument("--variant", choices=("C1", "C2"), required=True)
    graph.add_argument("--snapshot-root", type=Path, required=True)
    graph.add_argument("--output", type=Path, required=True)
    spec = commands.add_parser(
        "spec",
        help="validate upstream evidence and write the canary runner spec",
    )
    spec.add_argument("--variant", choices=("C1", "C2"), required=True)
    spec.add_argument("--c0-spec", type=Path, required=True)
    spec.add_argument("--c0-receipt", type=Path, required=True)
    spec.add_argument("--snapshot-publication", type=Path, required=True)
    spec.add_argument("--import-attestation", type=Path, required=True)
    spec.add_argument("--qualification", type=Path, required=True)
    spec.add_argument("--device-probe", type=Path, required=True)
    spec.add_argument("--runtime-provenance", type=Path, required=True)
    spec.add_argument("--variant-solver-graph", type=Path, required=True)
    spec.add_argument("--cache-directory", type=Path, required=True)
    spec.add_argument("--output-root", type=Path, required=True)
    spec.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    variant: CanaryVariant = args.variant
    if args.command == "graph":
        print(write_variant_solver_graph(args.output, args.snapshot_root, variant))
        return 0
    _document, digest = build_canary_spec(
        CanarySpecBuildInputs(
            variant=variant,
            c0_spec_path=args.c0_spec,
            c0_receipt_path=args.c0_receipt,
            snapshot_publication_path=args.snapshot_publication,
            import_attestation_path=args.import_attestation,
            qualification_path=args.qualification,
            device_probe_path=args.device_probe,
            runtime_provenance_path=args.runtime_provenance,
            variant_solver_graph_path=args.variant_solver_graph,
            cache_directory=args.cache_directory,
            output_root=args.output_root,
            destination=args.output,
        )
    )
    print(digest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
