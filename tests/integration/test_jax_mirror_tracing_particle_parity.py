"""Exact parity for the ``1_Simple/tracing_particle.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_tracing_particle_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-tracing-particle")
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
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:initial_field",
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
        atol=2.0e-6,
    )
    np.testing.assert_allclose(
        jax.values["final:positions"],
        native.values["final:positions"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["final:parallel_speed_fraction"],
        native.values["final:parallel_speed_fraction"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["poincare:positions"],
        native.values["poincare:positions"],
        rtol=0.0,
        atol=2.0e-3,
    )
    assert float(native.values["conservation:energy_relative_error"]) < 1.0e-6
    assert float(jax.values["conservation:energy_relative_error"]) < 1.0e-6
