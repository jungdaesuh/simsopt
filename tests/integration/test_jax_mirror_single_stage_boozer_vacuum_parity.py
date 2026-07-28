"""Parity contract for the VMEC-free Boozer single-stage mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


REPO_ROOT = Path(__file__).resolve().parents[2]
CASE_ID = "native-single-stage-boozer-vacuum-optimization"


def test_single_stage_boozer_vacuum_is_an_executable_parity_case() -> None:
    case = get_case(CASE_ID)
    runtime = load_runtime_contract_pair(
        REPO_ROOT / "examples" / "jax" / "manifest.json",
        REPO_ROOT / "examples" / "jax" / "parity_manifest.json",
        repo_root=REPO_ROOT,
    )
    relationship = next(
        item for item in runtime.parity.relationships if item.jax_example_id == CASE_ID
    )

    assert case.case_id == CASE_ID
    assert relationship.case_id == CASE_ID
    assert relationship.classification == "full"
    assert relationship.blocker is None
    assert not relationship.omitted_scientific_stages
    assert relationship.workflow_stages
    assert relationship.comparison_routes


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
@pytest.mark.native_cpu_reference
def test_single_stage_boozer_vacuum_case_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(CASE_ID)
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    for observable in (
        "construction:surface_dofs",
        "construction:coil_dofs",
        "initial:parameters",
    ):
        np.testing.assert_array_equal(
            jax.values[observable],
            native.values[observable],
        )
    np.testing.assert_allclose(
        jax.values["initial:objective"],
        native.values["initial:objective"],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        jax.values["initial:gradient"],
        native.values["initial:gradient"],
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax.values["final:parameters"],
        native.values["final:parameters"],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    for observable in (
        "final:objective",
        "final:iota",
        "final:volume",
        "final:non_qs_ratio",
        "final:boozer_residual",
        "final:boozer_residual_rms",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=2.0e-8,
            atol=2.0e-12,
        )
