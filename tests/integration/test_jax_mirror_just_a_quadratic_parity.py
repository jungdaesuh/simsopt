"""Matched native/JAX parity for the exact ``just_a_quadratic.py`` mirror."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_quadratic_case_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-just-a-quadratic")
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
    for name in native.values:
        np.testing.assert_allclose(
            jax.values[name],
            native.values[name],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        jax.values["final:parameters"],
        (1.0, 2.0, 3.0),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
