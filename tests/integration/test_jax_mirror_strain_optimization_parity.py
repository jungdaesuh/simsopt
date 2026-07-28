"""Exact parity for the ``2_Intermediate/strain_optimization.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_strain_optimization_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-strain-optimization")
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
        "construction:quadpoints",
        "construction:gamma",
        "construction:gammadash",
        "construction:gammadashdash",
        "initial:parameters",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    np.testing.assert_allclose(
        jax.values["initial:objective"],
        native.values["initial:objective"],
        rtol=2.0e-7,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax.values["initial:gradient"],
        native.values["initial:gradient"],
        rtol=2.0e-7,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax.values["final:parameters"],
        native.values["final:parameters"],
        rtol=5.0e-3,
        atol=2.0e-4,
    )
    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=2.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        jax.values["final:gradient"],
        native.values["final:gradient"],
        rtol=1.0,
        atol=3.0e-9,
    )
    for observable in (
        "final:torsional_strain",
        "final:binormal_curvature_strain",
        "final:maximum_torsional_strain",
        "final:maximum_binormal_curvature_strain",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=2.0e-5,
            atol=5.0e-8,
        )

    assert float(native.values["final:objective"]) < float(
        native.values["initial:objective"]
    )
    assert float(jax.values["final:objective"]) < float(
        jax.values["initial:objective"]
    )
