"""Exact parity for the ``1_Simple/tracing_fieldlines_QA.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_tracing_fieldlines_qa_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-tracing-fieldlines-qa")
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
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:surface_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:surface_field",
        "interpolation:relative_error",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    np.testing.assert_allclose(
        jax.values["final:times"],
        native.values["final:times"],
        rtol=0.0,
        atol=6.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["final:states"],
        native.values["final:states"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["poincare:positions"],
        native.values["poincare:positions"],
        rtol=0.0,
        atol=7.0e-3,
    )

    assert native.values["final:status"].tolist() == [0, 0, -1]
    assert native.values["poincare:counts"].tolist() == [9, 9, 5]
