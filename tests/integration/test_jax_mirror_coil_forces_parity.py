"""Exact parity for the ``3_Advanced/coil_forces.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_coil_forces_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-coil-forces")
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

    for observable in native.values:
        if observable.startswith("initial:"):
            np.testing.assert_allclose(
                jax.values[observable],
                native.values[observable],
                rtol=2.0e-8,
                atol=2.0e-10,
            )

    for stage in ("first", "final"):
        for observation in (native, jax):
            assert float(observation.values[f"{stage}:objective"]) < float(
                observation.values["initial:objective"]
            )
            assert np.all(
                np.isfinite(observation.values[f"{stage}:objective_gradient"])
            )

    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-2,
        atol=1.0e-9,
    )
    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax.values["taylor:errors"][:3])) <= 1.0e-4
