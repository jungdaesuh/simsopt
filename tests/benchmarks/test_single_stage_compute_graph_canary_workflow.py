from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from benchmarks.single_stage_compute_graph_c0_runner import (
    _load_canonical_json_object,
    _write_exclusive_json,
)
from benchmarks.single_stage_compute_graph_canary_workflow import (
    CanaryWorkflowError,
    CanaryWorkflowInputs,
    build_promotion_finalizer_spec,
    main,
    run_canary_workflow,
)
from benchmarks.single_stage_compute_graph_promotion_finalizer import (
    PROMOTION_FINALIZER_SPEC_SCHEMA_ID,
)


def _file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _inputs(tmp_path: Path, variant: str = "C1") -> CanaryWorkflowInputs:
    trajectory_root = tmp_path / "trajectory"
    c0_raw = trajectory_root / "c0.json" if variant == "C1" else None
    c0_launch = trajectory_root / "c0-receipt.json" if variant == "C1" else None
    return CanaryWorkflowInputs(
        variant=variant,  # type: ignore[arg-type]
        canary_spec_path=tmp_path / "canary-spec.json",
        base_canary_artifact_path=tmp_path / "canary.json",
        c0_receipt_path=tmp_path / "phase0-receipt.json",
        trajectory_artifact_root=trajectory_root,
        native_raw_path=trajectory_root / "native.json",
        native_trajectory_receipt_path=trajectory_root / "native-receipt.json",
        c0_raw_path=c0_raw,
        c0_trajectory_receipt_path=c0_launch,
        variant_raw_path=trajectory_root / f"{variant.lower()}.json",
        variant_trajectory_receipt_path=(
            trajectory_root / f"{variant.lower()}-receipt.json"
        ),
        profile_count_path=trajectory_root / "profile-counts.json",
        trajectory_oracle_path=trajectory_root / "oracle.json",
        profile_output_root=tmp_path / "profile",
        nsys_binary=tmp_path / "nsys",
        nvtx_library=tmp_path / "nvtx.so",
        nsys_version="2026.1",
        finalizer_spec_destination=tmp_path / "finalizer-spec.json",
        promotion_destination=tmp_path / "promotion.json",
    )


def _completed_evidence(inputs: CanaryWorkflowInputs) -> None:
    for path in (
        inputs.canary_spec_path,
        inputs.base_canary_artifact_path,
        inputs.c0_receipt_path,
        inputs.native_raw_path,
        inputs.native_trajectory_receipt_path,
        inputs.variant_raw_path,
        inputs.variant_trajectory_receipt_path,
        inputs.profile_count_path,
        inputs.trajectory_oracle_path,
        inputs.profile_output_root / "profile-evidence.json",
    ):
        _file(path)
    if inputs.c0_raw_path is not None:
        _file(inputs.c0_raw_path)
    if inputs.c0_trajectory_receipt_path is not None:
        _file(inputs.c0_trajectory_receipt_path)


def test_c1_builder_requires_c0_paths_and_writes_canonical_spec(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path)
    _completed_evidence(inputs)
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._load_canary_spec",
        lambda _path: ({}, SimpleNamespace(variant="C1")),
    )

    document = build_promotion_finalizer_spec(inputs)

    assert document["schema_id"] == PROMOTION_FINALIZER_SPEC_SCHEMA_ID
    assert document["trajectory_reference_raw_path"] == str(
        inputs.c0_raw_path.resolve()  # type: ignore[union-attr]
    )
    assert document["c0_trajectory_receipt_path"] == str(
        inputs.c0_trajectory_receipt_path.resolve()  # type: ignore[union-attr]
    )
    assert (
        _load_canonical_json_object(
            inputs.finalizer_spec_destination, "written finalizer spec"
        )
        == document
    )
    with pytest.raises(CanaryWorkflowError, match="must not exist"):
        build_promotion_finalizer_spec(inputs)


def test_builder_enforces_variant_specific_c0_path_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    c1 = _inputs(tmp_path / "c1")
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._load_canary_spec",
        lambda _path: ({}, SimpleNamespace(variant="C1")),
    )
    with pytest.raises(CanaryWorkflowError, match="C1 requires C0"):
        build_promotion_finalizer_spec(
            replace(
                c1,
                c0_raw_path=None,
                c0_trajectory_receipt_path=None,
            )
        )

    c2 = _inputs(tmp_path / "c2", "C2")
    injected = replace(
        c2,
        c0_raw_path=tmp_path / "unused-c0.json",
        c0_trajectory_receipt_path=tmp_path / "unused-c0-receipt.json",
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._load_canary_spec",
        lambda _path: ({}, SimpleNamespace(variant="C2")),
    )
    with pytest.raises(CanaryWorkflowError, match="C2 forbids"):
        build_promotion_finalizer_spec(injected)


def test_c2_builder_binds_one_native_file_to_both_reference_roles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path, "C2")
    _completed_evidence(inputs)
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._load_canary_spec",
        lambda _path: ({}, SimpleNamespace(variant="C2")),
    )

    document = build_promotion_finalizer_spec(inputs)

    assert (
        document["one_step_reference_raw_path"]
        == document["trajectory_reference_raw_path"]
    )
    assert document["c0_trajectory_receipt_path"] is None


def test_workflow_invokes_every_producer_before_finalizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path)
    for path in (
        inputs.canary_spec_path,
        inputs.base_canary_artifact_path,
        inputs.c0_receipt_path,
        inputs.nsys_binary,
        inputs.nvtx_library,
    ):
        _file(path)
    spec = SimpleNamespace(
        variant="C1", runtime_contract_json='{"static_environment":{}}'
    )
    events = []
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._load_canary_spec",
        lambda _path: ({}, spec),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._native_trajectory_launch",
        lambda **_kwargs: SimpleNamespace(
            output_path=inputs.native_raw_path,
            receipt_path=inputs.native_trajectory_receipt_path,
        ),
    )

    def launch_native(launch, *, artifact_root):
        del artifact_root
        events.append("native")
        _file(launch.output_path)
        _file(launch.receipt_path)

    def launch_variant(launch, *, artifact_root):
        del artifact_root
        events.append(launch.lane)
        _file(launch.output_path)
        _file(launch.receipt_path)
        if launch.profile_count_output_path is not None:
            _file(launch.profile_count_output_path)

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.launch_native_trajectory",
        launch_native,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.launch_variant_trajectory",
        launch_variant,
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow._trajectory_identity",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.build_variant_trajectory_oracle",
        lambda **_kwargs: events.append("oracle-build") or {},
    )

    def write_oracle(path, document):
        del document
        events.append("oracle-write")
        _write_exclusive_json(path, {})

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.write_variant_trajectory_oracle",
        write_oracle,
    )

    def profile(**kwargs):
        events.append("profile")
        kwargs["output_root"].mkdir()
        _write_exclusive_json(kwargs["output_root"] / "profile-evidence.json", {})

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.run_profile", profile
    )

    def finalize(**_kwargs):
        events.append("finalize")
        assert inputs.finalizer_spec_destination.is_file()
        return {"status": "MEASURED_PROMOTABLE"}

    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.finalize_promotion",
        finalize,
    )

    artifact = run_canary_workflow(inputs)

    assert artifact["status"] == "MEASURED_PROMOTABLE"
    assert events == [
        "native",
        "C0",
        "C1",
        "oracle-build",
        "oracle-write",
        "profile",
        "finalize",
    ]


def test_cli_constructs_typed_workflow_and_invokes_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inputs = _inputs(tmp_path)
    captured = []
    monkeypatch.setattr(
        "benchmarks.single_stage_compute_graph_canary_workflow.run_canary_workflow",
        captured.append,
    )

    assert (
        main(
            [
                "--variant",
                "C1",
                "--canary-spec",
                str(inputs.canary_spec_path),
                "--base-canary-artifact",
                str(inputs.base_canary_artifact_path),
                "--c0-receipt",
                str(inputs.c0_receipt_path),
                "--trajectory-artifact-root",
                str(inputs.trajectory_artifact_root),
                "--native-raw",
                str(inputs.native_raw_path),
                "--native-trajectory-receipt",
                str(inputs.native_trajectory_receipt_path),
                "--c0-raw",
                str(inputs.c0_raw_path),
                "--c0-trajectory-receipt",
                str(inputs.c0_trajectory_receipt_path),
                "--variant-raw",
                str(inputs.variant_raw_path),
                "--variant-trajectory-receipt",
                str(inputs.variant_trajectory_receipt_path),
                "--profile-count",
                str(inputs.profile_count_path),
                "--trajectory-oracle",
                str(inputs.trajectory_oracle_path),
                "--profile-output-root",
                str(inputs.profile_output_root),
                "--nsys-binary",
                str(inputs.nsys_binary),
                "--nvtx-library",
                str(inputs.nvtx_library),
                "--nsys-version",
                inputs.nsys_version,
                "--finalizer-spec",
                str(inputs.finalizer_spec_destination),
                "--promotion-destination",
                str(inputs.promotion_destination),
            ]
        )
        == 0
    )
    assert captured == [inputs]


def test_workflow_rejects_empty_nsys_version_before_launch(tmp_path: Path) -> None:
    with pytest.raises(CanaryWorkflowError, match="Nsight version"):
        run_canary_workflow(replace(_inputs(tmp_path), nsys_version=""))
