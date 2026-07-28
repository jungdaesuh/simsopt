"""Matched parity for the exact ``stage_two_optimization_minimal.py`` mirror."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_stage_two_minimal_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-minimal")
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

    for observable in (
        "parameters",
        "objective",
        "objective_gradient",
        "squared_flux",
        "length_penalty",
        "maximum_normal_field",
        "total_curve_length",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=1.0e-8,
            atol=1.0e-10,
        )

    for observation in (native, jax):
        length_target = bundle.configuration["length_target"]
        assert isinstance(length_target, (int, float))
        assert (
            observation.values["final:objective"]
            < observation.values["initial:objective"]
        )
        assert np.linalg.norm(
            observation.values["final:objective_gradient"],
            ord=np.inf,
        ) <= 1.0e-4
        assert observation.values["final:total_curve_length"] <= (
            1.1 * length_target
        )

    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-3,
        atol=1.0e-10,
    )
