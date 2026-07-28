"""Exact parity for the ``2_Intermediate/wireframe_rcls_basic.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_wireframe_rcls_basic_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-wireframe-rcls-basic")
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
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:constraint_matrix",
        "construction:constraint_target",
        "construction:free_segments",
        "construction:plasma_points",
        "construction:wireframe_nodes",
        "construction:wireframe_segments",
        "construction:wireframe_segment_signs",
        "construction:plasma_unit_normal",
        "construction:plasma_area_weights",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-20,
        )

    for phase in ("initial", "final"):
        for observable in (
            "currents",
            "normal_field_residual",
            "normal_objective",
            "regularization_objective",
            "total_objective",
            "constraint_residual",
        ):
            np.testing.assert_allclose(
                jax.values[f"{phase}:{observable}"],
                native.values[f"{phase}:{observable}"],
                rtol=1.0e-11,
                atol=1.0e-7,
            )

    for observable in (
        "final:magnetic_field",
        "final:normal_field",
        "final:mean_relative_normal_field",
        "final:maximum_current",
        "final:degrees_of_freedom",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-11,
            atol=1.0e-12,
        )

    assert native.values["construction:response_matrix"].shape == (256, 48)
    assert native.values["final:magnetic_field"].shape == (256, 3)
    assert float(native.values["final:mean_relative_normal_field"]) < 0.04
    assert float(native.values["final:normal_objective"]) < float(
        native.values["initial:normal_objective"]
    )
