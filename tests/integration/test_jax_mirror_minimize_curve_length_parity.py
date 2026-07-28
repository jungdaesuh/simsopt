"""Matched native/JAX parity for the exact ``minimize_curve_length.py`` mirror."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_minimize_curve_length_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-minimize-curve-length")
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
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)
    for name in (
        "initial:parameters",
        "initial:length",
        "initial:residual",
        "initial:residual_jacobian",
        "initial:objective_sum_squares",
        "initial:objective_gradient",
        "final:length",
        "final:residual",
        "final:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax.values[name],
            native.values[name],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
    np.testing.assert_allclose(
        jax.values["final:length"],
        6.0 * np.pi,
        rtol=1.0e-9,
        atol=1.0e-9,
    )
