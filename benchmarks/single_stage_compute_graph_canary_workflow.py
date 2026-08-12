"""Run the complete post-canary evidence and promotion-finalization workflow."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from benchmarks.single_stage_compute_graph_c0_runner import (
    _load_canonical_json_object,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_canary_profile_runner import run_profile
from benchmarks.single_stage_compute_graph_canary_runner import CanarySpec
from benchmarks.single_stage_compute_graph_native_trajectory_runner import (
    launch_native_trajectory,
)
from benchmarks.single_stage_compute_graph_promotion_finalizer import (
    PROMOTION_FINALIZER_SPEC_SCHEMA_ID,
    _load_canary_spec,
    _mapping,
    _native_trajectory_launch,
    finalize_promotion,
)
from benchmarks.single_stage_compute_graph_trajectory_oracle import (
    TrajectoryOracleIdentity,
    build_variant_trajectory_oracle,
    write_variant_trajectory_oracle,
)
from benchmarks.single_stage_compute_graph_variant_trajectory_runner import (
    VariantTrajectoryLaunch,
    launch_variant_trajectory,
)

Variant = Literal["C1", "C2"]


class CanaryWorkflowError(RuntimeError):
    """The post-canary workflow inputs or evidence topology are invalid."""


@dataclass(frozen=True, slots=True)
class CanaryWorkflowInputs:
    variant: Variant
    canary_spec_path: Path
    base_canary_artifact_path: Path
    c0_receipt_path: Path
    trajectory_artifact_root: Path
    native_raw_path: Path
    native_trajectory_receipt_path: Path
    c0_raw_path: Path | None
    c0_trajectory_receipt_path: Path | None
    variant_raw_path: Path
    variant_trajectory_receipt_path: Path
    profile_count_path: Path
    trajectory_oracle_path: Path
    profile_output_root: Path
    nsys_binary: Path
    nvtx_library: Path
    nsys_version: str
    finalizer_spec_destination: Path
    promotion_destination: Path


def _variant_paths(inputs: CanaryWorkflowInputs) -> tuple[Path, Path | None]:
    if inputs.variant == "C1":
        if inputs.c0_raw_path is None or inputs.c0_trajectory_receipt_path is None:
            raise CanaryWorkflowError("C1 requires C0 raw and launch-receipt paths")
        return inputs.c0_raw_path.resolve(), inputs.c0_trajectory_receipt_path.resolve()
    if inputs.c0_raw_path is not None or inputs.c0_trajectory_receipt_path is not None:
        raise CanaryWorkflowError("C2 forbids unused C0 raw or launch-receipt paths")
    return inputs.native_raw_path.resolve(), None


def _require_file(path: Path, context: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise CanaryWorkflowError(f"{context} must be an existing file")
    return resolved


def _require_new_path(path: Path, context: str) -> Path:
    resolved = path.resolve()
    if resolved.exists() or not resolved.parent.is_dir():
        raise CanaryWorkflowError(
            f"{context} must not exist and its parent must be an existing directory"
        )
    return resolved


def _require_trajectory_path(path: Path, root: Path, context: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CanaryWorkflowError(f"{context} must be inside trajectory artifact root")
    return resolved


def _runtime_base_environment(spec: CanarySpec) -> dict[str, str]:
    try:
        contract = json.loads(spec.runtime_contract_json)
    except json.JSONDecodeError as error:
        raise CanaryWorkflowError("canary runtime contract is invalid JSON") from error
    document = _mapping(contract, "canary runtime contract")
    static = _mapping(document.get("static_environment"), "static environment")
    if not all(
        isinstance(key, str) and isinstance(value, str) for key, value in static.items()
    ):
        raise CanaryWorkflowError("static environment must contain only strings")
    return {str(key): str(value) for key, value in static.items()}


def _input_bundle_sha256(
    canary_spec_document: Mapping[str, object],
) -> str:
    c0_spec_path = canary_spec_document.get("c0_spec_path")
    if not isinstance(c0_spec_path, str):
        raise CanaryWorkflowError("canary spec lacks C0 spec path")
    c0_spec = _load_canonical_json_object(Path(c0_spec_path), "workflow C0 spec")
    specimen = _mapping(
        _mapping(c0_spec.get("receipt_template"), "C0 receipt template").get(
            "specimen"
        ),
        "C0 specimen",
    )
    value = specimen.get("input_bundle_sha256")
    if not isinstance(value, str):
        raise CanaryWorkflowError("C0 specimen lacks input-bundle SHA")
    return value


def _trajectory_identity(
    spec: CanarySpec, canary_spec_document: Mapping[str, object]
) -> TrajectoryOracleIdentity:
    return TrajectoryOracleIdentity(
        variant=spec.variant,
        parameter_sha256=spec.parameter_sha256,
        specimen_sha256=spec.specimen_sha256,
        input_bundle_sha256=_input_bundle_sha256(canary_spec_document),
        solver_graph_sha256=spec.solver_graph_sha256,
        one_step_reference_source_sha256=spec.source_state_sha256,
        trajectory_reference_source_sha256=spec.source_state_sha256,
        variant_source_sha256=spec.source_state_sha256,
    )


def _finalizer_document(
    inputs: CanaryWorkflowInputs,
    *,
    trajectory_reference_raw_path: Path,
    c0_trajectory_receipt_path: Path | None,
) -> dict[str, object]:
    return {
        "schema_id": PROMOTION_FINALIZER_SPEC_SCHEMA_ID,
        "canary_spec_path": str(inputs.canary_spec_path.resolve()),
        "base_canary_artifact_path": str(inputs.base_canary_artifact_path.resolve()),
        "profile_evidence_path": str(
            (inputs.profile_output_root / "profile-evidence.json").resolve()
        ),
        "trajectory_oracle_path": str(inputs.trajectory_oracle_path.resolve()),
        "trajectory_artifact_root": str(inputs.trajectory_artifact_root.resolve()),
        "one_step_reference_raw_path": str(inputs.native_raw_path.resolve()),
        "trajectory_reference_raw_path": str(trajectory_reference_raw_path),
        "variant_raw_path": str(inputs.variant_raw_path.resolve()),
        "native_trajectory_receipt_path": str(
            inputs.native_trajectory_receipt_path.resolve()
        ),
        "c0_trajectory_receipt_path": (
            None
            if c0_trajectory_receipt_path is None
            else str(c0_trajectory_receipt_path)
        ),
        "variant_trajectory_receipt_path": str(
            inputs.variant_trajectory_receipt_path.resolve()
        ),
        "c0_receipt_path": str(inputs.c0_receipt_path.resolve()),
        "destination": str(inputs.promotion_destination.resolve()),
    }


def build_promotion_finalizer_spec(inputs: CanaryWorkflowInputs) -> dict[str, object]:
    """Validate completed evidence topology and exclusively write its finalizer spec."""

    canary_spec_document, spec = _load_canary_spec(inputs.canary_spec_path.resolve())
    del canary_spec_document
    if spec.variant != inputs.variant:
        raise CanaryWorkflowError("workflow variant differs from validated canary spec")
    trajectory_reference, c0_receipt = _variant_paths(inputs)
    for path, context in (
        (inputs.native_raw_path, "native raw trajectory"),
        (inputs.native_trajectory_receipt_path, "native launch receipt"),
        (trajectory_reference, "trajectory-reference raw trajectory"),
        (inputs.variant_raw_path, "variant raw trajectory"),
        (inputs.variant_trajectory_receipt_path, "variant launch receipt"),
        (inputs.profile_count_path, "profile count evidence"),
        (inputs.trajectory_oracle_path, "trajectory oracle"),
    ):
        _require_trajectory_path(path, inputs.trajectory_artifact_root, context)
    if c0_receipt is not None:
        _require_trajectory_path(
            c0_receipt, inputs.trajectory_artifact_root, "C0 trajectory receipt"
        )
    if (
        inputs.variant == "C2"
        and trajectory_reference != inputs.native_raw_path.resolve()
    ):
        raise CanaryWorkflowError("C2 must reuse one native raw reference path")
    for path, context in (
        (inputs.canary_spec_path, "canary spec"),
        (inputs.base_canary_artifact_path, "base canary artifact"),
        (inputs.c0_receipt_path, "C0 receipt"),
        (inputs.native_raw_path, "native raw trajectory"),
        (inputs.native_trajectory_receipt_path, "native launch receipt"),
        (trajectory_reference, "trajectory-reference raw trajectory"),
        (inputs.variant_raw_path, "variant raw trajectory"),
        (inputs.variant_trajectory_receipt_path, "variant launch receipt"),
        (inputs.profile_count_path, "profile count evidence"),
        (inputs.trajectory_oracle_path, "trajectory oracle"),
        (inputs.profile_output_root / "profile-evidence.json", "profile evidence"),
    ):
        _require_file(path, context)
    if c0_receipt is not None:
        _require_file(c0_receipt, "C0 trajectory launch receipt")
    _require_new_path(inputs.finalizer_spec_destination, "finalizer spec destination")
    _require_new_path(inputs.promotion_destination, "promotion destination")
    document = _finalizer_document(
        inputs,
        trajectory_reference_raw_path=trajectory_reference,
        c0_trajectory_receipt_path=c0_receipt,
    )
    _write_exclusive_json(inputs.finalizer_spec_destination.resolve(), document)
    return document


def run_canary_workflow(inputs: CanaryWorkflowInputs) -> Mapping[str, object]:
    """Produce every post-canary artifact and invoke the sole promotion finalizer."""

    if not isinstance(inputs.nsys_version, str) or not inputs.nsys_version:
        raise CanaryWorkflowError("Nsight version must be a non-empty string")
    canary_spec_document, spec = _load_canary_spec(inputs.canary_spec_path.resolve())
    if spec.variant != inputs.variant:
        raise CanaryWorkflowError("workflow variant differs from validated canary spec")
    trajectory_reference, c0_receipt = _variant_paths(inputs)
    for path, context in (
        (inputs.native_raw_path, "native raw trajectory"),
        (inputs.native_trajectory_receipt_path, "native launch receipt"),
        (trajectory_reference, "trajectory-reference raw trajectory"),
        (inputs.variant_raw_path, "variant raw trajectory"),
        (inputs.variant_trajectory_receipt_path, "variant launch receipt"),
        (inputs.profile_count_path, "profile count evidence"),
        (inputs.trajectory_oracle_path, "trajectory oracle"),
    ):
        _require_trajectory_path(path, inputs.trajectory_artifact_root, context)
    if c0_receipt is not None:
        _require_trajectory_path(
            c0_receipt, inputs.trajectory_artifact_root, "C0 trajectory receipt"
        )
    for path, context in (
        (inputs.canary_spec_path, "canary spec"),
        (inputs.base_canary_artifact_path, "base canary artifact"),
        (inputs.c0_receipt_path, "C0 receipt"),
        (inputs.nsys_binary, "Nsight binary"),
        (inputs.nvtx_library, "NVTX library"),
    ):
        _require_file(path, context)
    for path, context in (
        (inputs.trajectory_artifact_root, "trajectory artifact root"),
        (inputs.profile_output_root, "profile output root"),
        (inputs.finalizer_spec_destination, "finalizer spec destination"),
        (inputs.promotion_destination, "promotion destination"),
    ):
        _require_new_path(path, context)
    inputs.trajectory_artifact_root.resolve().mkdir()

    native_launch = _native_trajectory_launch(
        receipt_path=inputs.native_trajectory_receipt_path.resolve(),
        raw_path=inputs.native_raw_path.resolve(),
        canary_spec_document=canary_spec_document,
        spec=spec,
    )
    launch_native_trajectory(
        native_launch, artifact_root=inputs.trajectory_artifact_root.resolve()
    )
    if inputs.variant == "C1":
        if inputs.c0_raw_path is None or c0_receipt is None:
            raise AssertionError("C1 paths were validated above")
        launch_variant_trajectory(
            VariantTrajectoryLaunch(
                spec=spec,
                spec_path=inputs.canary_spec_path.resolve(),
                lane="C0",
                output_path=inputs.c0_raw_path.resolve(),
                receipt_path=c0_receipt,
            ),
            artifact_root=inputs.trajectory_artifact_root.resolve(),
        )
    launch_variant_trajectory(
        VariantTrajectoryLaunch(
            spec=spec,
            spec_path=inputs.canary_spec_path.resolve(),
            lane=spec.variant,
            output_path=inputs.variant_raw_path.resolve(),
            receipt_path=inputs.variant_trajectory_receipt_path.resolve(),
            profile_count_output_path=inputs.profile_count_path.resolve(),
            canary_artifact_path=inputs.base_canary_artifact_path.resolve(),
        ),
        artifact_root=inputs.trajectory_artifact_root.resolve(),
    )

    oracle = build_variant_trajectory_oracle(
        identity=_trajectory_identity(spec, canary_spec_document),
        artifact_root=inputs.trajectory_artifact_root.resolve(),
        one_step_reference_raw_path=inputs.native_raw_path.resolve(),
        trajectory_reference_raw_path=trajectory_reference,
        variant_raw_path=inputs.variant_raw_path.resolve(),
    )
    write_variant_trajectory_oracle(inputs.trajectory_oracle_path.resolve(), oracle)
    run_profile(
        spec=spec,
        canary_artifact_path=inputs.base_canary_artifact_path.resolve(),
        nsys_binary=inputs.nsys_binary.resolve(),
        nvtx_library=inputs.nvtx_library.resolve(),
        expected_nsys_version=inputs.nsys_version,
        output_root=inputs.profile_output_root.resolve(),
        profile_count_evidence_path=inputs.profile_count_path.resolve(),
        environment=_runtime_base_environment(spec),
    )
    build_promotion_finalizer_spec(inputs)
    return finalize_promotion(
        canary_spec_path=inputs.canary_spec_path.resolve(),
        base_canary_artifact_path=inputs.base_canary_artifact_path.resolve(),
        profile_evidence_path=(
            inputs.profile_output_root / "profile-evidence.json"
        ).resolve(),
        trajectory_oracle_path=inputs.trajectory_oracle_path.resolve(),
        trajectory_artifact_root=inputs.trajectory_artifact_root.resolve(),
        one_step_reference_raw_path=inputs.native_raw_path.resolve(),
        trajectory_reference_raw_path=trajectory_reference,
        variant_raw_path=inputs.variant_raw_path.resolve(),
        native_trajectory_receipt_path=(
            inputs.native_trajectory_receipt_path.resolve()
        ),
        c0_trajectory_receipt_path=c0_receipt,
        variant_trajectory_receipt_path=(
            inputs.variant_trajectory_receipt_path.resolve()
        ),
        c0_receipt_path=inputs.c0_receipt_path.resolve(),
        destination=inputs.promotion_destination.resolve(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("C1", "C2"), required=True)
    parser.add_argument("--canary-spec", type=Path, required=True)
    parser.add_argument("--base-canary-artifact", type=Path, required=True)
    parser.add_argument("--c0-receipt", type=Path, required=True)
    parser.add_argument("--trajectory-artifact-root", type=Path, required=True)
    parser.add_argument("--native-raw", type=Path, required=True)
    parser.add_argument("--native-trajectory-receipt", type=Path, required=True)
    parser.add_argument("--c0-raw", type=Path)
    parser.add_argument("--c0-trajectory-receipt", type=Path)
    parser.add_argument("--variant-raw", type=Path, required=True)
    parser.add_argument("--variant-trajectory-receipt", type=Path, required=True)
    parser.add_argument("--profile-count", type=Path, required=True)
    parser.add_argument("--trajectory-oracle", type=Path, required=True)
    parser.add_argument("--profile-output-root", type=Path, required=True)
    parser.add_argument("--nsys-binary", type=Path, required=True)
    parser.add_argument("--nvtx-library", type=Path, required=True)
    parser.add_argument("--nsys-version", required=True)
    parser.add_argument("--finalizer-spec", type=Path, required=True)
    parser.add_argument("--promotion-destination", type=Path, required=True)
    args = parser.parse_args(argv)
    run_canary_workflow(
        CanaryWorkflowInputs(
            variant=cast(Variant, args.variant),
            canary_spec_path=args.canary_spec,
            base_canary_artifact_path=args.base_canary_artifact,
            c0_receipt_path=args.c0_receipt,
            trajectory_artifact_root=args.trajectory_artifact_root,
            native_raw_path=args.native_raw,
            native_trajectory_receipt_path=args.native_trajectory_receipt,
            c0_raw_path=args.c0_raw,
            c0_trajectory_receipt_path=args.c0_trajectory_receipt,
            variant_raw_path=args.variant_raw,
            variant_trajectory_receipt_path=args.variant_trajectory_receipt,
            profile_count_path=args.profile_count,
            trajectory_oracle_path=args.trajectory_oracle,
            profile_output_root=args.profile_output_root,
            nsys_binary=args.nsys_binary,
            nvtx_library=args.nvtx_library,
            nsys_version=args.nsys_version,
            finalizer_spec_destination=args.finalizer_spec,
            promotion_destination=args.promotion_destination,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
