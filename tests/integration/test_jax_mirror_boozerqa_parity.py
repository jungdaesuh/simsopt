"""Exact parity for the ``2_Intermediate/boozerQA.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_boozerqa_workflow_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-boozerqa")
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
        "construction:coil_dofs",
        "initial:parameters",
        "initial:objective",
        "initial:gradient",
        "initial:iota",
        "initial:volume",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    for observable in (
        "final:objective",
        "final:non_qs_ratio",
        "final:iota",
        "final:volume",
        "final:major_radius_penalty",
        "final:length_penalty",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-3,
            atol=1.0e-8,
        )

    np.testing.assert_allclose(
        jax.values["final:parameters"],
        native.values["final:parameters"],
        rtol=0.0,
        atol=2.0e-3,
    )
    assert float(native.values["final:objective"]) <= float(
        native.values["initial:objective"]
    )
    assert float(jax.values["final:objective"]) <= float(
        jax.values["initial:objective"]
    )
