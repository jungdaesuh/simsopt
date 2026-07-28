"""Matched native/JAX parity for the exact ``1_Simple/qfm.py`` mirror."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_qfm_sequence_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-qfm")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
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
        "initial:parameters",
        "initial:qfm_value",
        "initial:qfm_gradient",
        "volume:target",
        "volume:initial:label_value",
        "volume:initial:label_gradient",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-9,
            atol=1.0e-11,
        )

    for stage in ("volume", "toroidal_flux", "area"):
        for phase in ("initial", "penalty", "exact"):
            np.testing.assert_allclose(
                jax.values[f"{stage}:{phase}:qfm_value"],
                native.values[f"{stage}:{phase}:qfm_value"],
                rtol=5.0e-5,
                atol=1.0e-7,
            )
        np.testing.assert_allclose(
            jax.values[f"{stage}:exact:parameters"],
            native.values[f"{stage}:exact:parameters"],
            rtol=2.0e-3,
            atol=2.0e-4,
        )
        assert float(jax.values[f"{stage}:exact:label_residual_abs"]) <= 1.0e-8
        assert float(native.values[f"{stage}:exact:label_residual_abs"]) <= 1.0e-8

    for stage in ("toroidal_flux", "area"):
        np.testing.assert_allclose(
            jax.values[f"{stage}:volume_persistence_objective"],
            native.values[f"{stage}:volume_persistence_objective"],
            rtol=2.0e-2,
            atol=1.0e-5,
        )

    assert float(jax.values["area:exact:qfm_value"]) < float(
        jax.values["initial:qfm_value"]
    )
    assert float(native.values["area:exact:qfm_value"]) < float(
        native.values["initial:qfm_value"]
    )
