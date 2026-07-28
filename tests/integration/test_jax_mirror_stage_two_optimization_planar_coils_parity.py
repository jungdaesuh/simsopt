"""Exact parity for ``2_Intermediate/stage_two_optimization_planar_coils.py``."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_planar_stage_two_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-planar-coils")
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
        "squared_flux",
        "geometric_penalty",
        "planarity_penalty",
        "linking_number",
        "canonical_geometry",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    for stage in ("first", "final"):
        for observation in (native, jax):
            assert (
                observation.values[f"{stage}:objective"]
                < (observation.values["initial:objective"])
            )
            assert np.all(
                np.isfinite(observation.values[f"{stage}:objective_gradient"])
            )
            assert float(observation.values[f"{stage}:planarity_penalty"]) <= 1.0e-24
            assert float(observation.values[f"{stage}:linking_number"]) == 0.0
            assert float(observation.values[f"{stage}:squared_flux"]) <= 1.0e-2
            assert float(observation.values[f"{stage}:geometric_penalty"]) <= 1.0e-3
            canonical_geometry = observation.values[
                f"{stage}:canonical_geometry"
            ].reshape((-1, 13))
            assert np.all(np.isfinite(canonical_geometry))
            assert abs(float(np.sum(canonical_geometry[:, 0])) - 10.4) <= 3.0e-2

        assert float(jax.values[f"{stage}:objective"]) <= (
            1.03 * float(native.values[f"{stage}:objective"]) + 1.0e-9
        )

    for observation in (native, jax):
        taylor_errors = np.abs(observation.values["taylor:errors"][:3])
        assert taylor_errors[1] <= 2.0e-2 * taylor_errors[0]
        assert taylor_errors[2] <= 2.0e-2 * taylor_errors[1]
        assert taylor_errors[2] <= 1.0e-4
