"""Exact parity for the ``2_Intermediate/wireframe_rcls_with_ports.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_wireframe_rcls_with_ports_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-wireframe-rcls-with-ports")
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

    construction_observables = tuple(
        observable
        for observable in native.values
        if observable.startswith("construction:")
    )
    for observable in construction_observables:
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-20,
        )

    for observable in (
        "initial:currents",
        "initial:normal_field_residual",
        "initial:normal_objective",
        "initial:regularization_objective",
        "initial:total_objective",
        "initial:constraint_residual",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-12,
        )

    # The constrained system is underdetermined. Its two direct solvers may
    # choose slightly different current vectors while agreeing much more
    # tightly on the physical field and minimized objective.
    np.testing.assert_allclose(
        jax.values["final:currents"],
        native.values["final:currents"],
        rtol=5.0e-7,
        atol=1.0e-2,
    )
    for observable in (
        "final:normal_field_residual",
        "final:normal_objective",
        "final:regularization_objective",
        "final:total_objective",
        "final:magnetic_field",
        "final:normal_field",
        "final:mean_relative_normal_field",
        "final:maximum_current",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
    np.testing.assert_allclose(
        jax.values["final:constraint_residual"],
        native.values["final:constraint_residual"],
        rtol=0.0,
        atol=1.0e-8,
    )
    np.testing.assert_array_equal(
        jax.values["final:degrees_of_freedom"],
        native.values["final:degrees_of_freedom"],
    )
    np.testing.assert_array_equal(
        jax.values["final:port_constraints_satisfied"],
        native.values["final:port_constraints_satisfied"],
    )

    constrained = arrays["constrained_segments"]
    final_currents = np.asarray(jax.values["final:currents"]).reshape(-1)
    np.testing.assert_array_equal(final_currents[constrained], 0.0)
    assert constrained.size > 0
    assert bool(jax.values["final:port_constraints_satisfied"])
    assert float(jax.values["final:normal_objective"]) < float(
        jax.values["initial:normal_objective"]
    )
