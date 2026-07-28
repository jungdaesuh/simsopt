"""Exact parity for ``stage_two_optimization_stochastic.py``."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_stochastic_stage_two_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-stochastic")
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
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "parameters",
        "objective",
        "objective_gradient",
        "training_flux",
        "nominal_flux",
        "out_of_sample_flux",
        "total_curve_length",
        "shortest_curve_distance",
        "maximum_curvature",
        "maximum_mean_squared_curvature",
        "maximum_arclength_variation",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=3.0e-8,
            atol=3.0e-10,
        )

    for observation in (native, jax):
        assert (
            observation.values["final:objective"]
            < (observation.values["initial:objective"])
        )
        assert np.all(np.isfinite(observation.values["final:parameters"]))
        assert np.all(np.isfinite(observation.values["final:objective_gradient"]))
        assert np.isfinite(observation.values["final:training_flux"])
        assert np.isfinite(observation.values["final:nominal_flux"])
        assert np.isfinite(observation.values["final:out_of_sample_flux"])

    assert float(jax.values["final:objective"]) <= (
        1.10 * float(native.values["final:objective"]) + 1.0e-9
    )

    for observation in (native, jax):
        taylor_errors = np.abs(observation.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 3.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 3.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
