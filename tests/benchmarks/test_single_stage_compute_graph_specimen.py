from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from benchmarks import single_stage_compute_graph_specimen as subject
from examples.jax.parity.cases.native_single_stage_boozer_vacuum import SPEC
from examples.jax.parity.input_bundle import create_input_bundle


def _configuration() -> dict[str, object]:
    return {
        "mpol": 6,
        "ntor": 6,
        "inner_maxiter": 20,
        "inner_tolerance": 1.0e-13,
        "outer_maxiter": 1_000,
        "outer_rtol": 0.0,
        "outer_atol": 1.0e-8,
        "initial_iota": -0.406,
        "surface_distance": 0.10,
        "non_qs_sdim": 20,
        "residual_weight": 1.0,
        "report_residual": True,
        "reduced_coil_order": 3,
        "reduced_axis_order": 3,
        "reduced_points_per_period": 8,
        "nfp": 3,
        "initial_G": 1.0,
    }


def _install_fake_builder(
    monkeypatch: pytest.MonkeyPatch,
    *,
    configuration: dict[str, object] | None = None,
    coil_dofs: np.ndarray | None = None,
) -> list[tuple[Path, str, object]]:
    calls: list[tuple[Path, str, object]] = []
    selected_configuration = configuration or _configuration()
    selected_coils = (
        np.linspace(-2.0, 3.0, 461, dtype=np.float64)
        if coil_dofs is None
        else coil_dofs
    )

    def fake_create(root: Path, scale: str, spec: object) -> object:
        calls.append((root, scale, spec))
        return create_input_bundle(
            root,
            case_id=SPEC.case_id,
            random_seed=1,
            arrays={
                "axis_dofs": np.arange(9, dtype=np.float64),
                "coil_dofs": selected_coils,
                "surface_dofs": np.arange(253, dtype=np.float64),
            },
            configuration=selected_configuration,
            scale="native_default",
        )

    monkeypatch.setattr(subject, "create_variant_input", fake_create)
    return calls


def test_builds_exact_deterministic_changed_state_and_receipt_specimen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _install_fake_builder(monkeypatch)
    first = subject.build_frozen_changed_state_specimen(tmp_path / "first")
    second = subject.build_frozen_changed_state_specimen(tmp_path / "second")

    assert [(scale, spec) for _, scale, spec in calls] == [
        ("native_default", SPEC),
        ("native_default", SPEC),
    ]
    candidate = np.load(first.candidate_path, allow_pickle=False)
    baseline = np.linspace(-2.0, 3.0, 461, dtype=np.float64)
    assert candidate.dtype.str == "<f8"
    assert candidate.shape == (461,)
    assert np.isfinite(candidate).all()
    assert not np.array_equal(candidate, baseline)
    assert np.array_equal(candidate, np.load(second.candidate_path, allow_pickle=False))
    parameter_sha = hashlib.sha256(
        np.ascontiguousarray(candidate, dtype="<f8").tobytes(order="C")
    ).hexdigest()
    assert first.parameter_sha256 == parameter_sha

    document = json.loads(first.document_path.read_text(encoding="utf-8"))
    second_document = json.loads(second.document_path.read_text(encoding="utf-8"))
    assert document == second_document
    assert document["specimen"]["state_dimension"] == 255
    assert document["specimen"]["coil_dof_count"] == 461
    assert document["specimen"]["grids"] == {
        "inner_surface_points": 169,
        "non_qs_surface_points": 1600,
        "physical_coil_contributions": 18,
        "quadrature_nodes": 250,
    }
    assert document["effective_policies"] == {
        "dense_batch_width": 8,
        "point_chunk_size": None,
        "coil_chunk_size": None,
        "quadrature_block_sizes": [128, 122],
    }
    assert document["candidate"]["parameter_sha256"] == parameter_sha
    assert document["candidate"]["differs_from_baseline"] is True
    assert document["specimen_sha256"] == subject.canonical_sha256(document["specimen"])


@pytest.mark.parametrize(
    "coil_dofs",
    [
        np.zeros(460, dtype=np.float64),
        np.zeros(461, dtype=np.float32),
    ],
)
def test_rejects_invalid_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    coil_dofs: np.ndarray,
) -> None:
    _install_fake_builder(monkeypatch, coil_dofs=coil_dofs)
    with pytest.raises(subject.SpecimenError, match="baseline coil_dofs"):
        subject.build_frozen_changed_state_specimen(tmp_path / "specimen")


def test_rejects_grid_or_policy_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configuration = _configuration()
    configuration["mpol"] = 5
    _install_fake_builder(monkeypatch, configuration=configuration)
    with pytest.raises(subject.SpecimenError, match="grid contract drifted"):
        subject.build_frozen_changed_state_specimen(tmp_path / "grid")

    with pytest.raises(subject.SpecimenError, match="quadrature_block_sizes"):
        subject.EffectivePolicies(quadrature_block_sizes=())


def test_requires_fresh_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_builder(monkeypatch)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    with pytest.raises(subject.SpecimenError, match="must not already exist"):
        subject.build_frozen_changed_state_specimen(occupied)
