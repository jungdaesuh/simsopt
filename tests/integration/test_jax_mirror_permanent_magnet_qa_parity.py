"""Exact parity for the ``2_Intermediate/permanent_magnet_QA.py`` mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from benchmarks.validation_ladder_contract import OPTIMIZER_DRIFT_TOLERANCES
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_permanent_magnet_qa_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-permanent-magnet-qa")
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
    assert native.nit == jax.nit == 2
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:response_matrix",
        "construction:target",
        "construction:moment_maxima",
        "construction:dipole_grid_xyz",
        "initial:moments",
        "initial:residual",
        "initial:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-10,
            atol=1.0e-12,
        )

    assert int(native.values["final:nonzero_count"]) > 0
    assert int(jax.values["final:nonzero_count"]) > 0
    final_relative_tolerance = OPTIMIZER_DRIFT_TOLERANCES[
        "tier2_stage2_e2e"
    ]["final_objective_rel_tol_20_iter"]
    assert final_relative_tolerance is not None
    for observable in (
        "final:objective_sum_squares",
        "final:residual_norm",
        "final:moment_l2_norm",
        "final:proxy_moment_l2_norm",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=final_relative_tolerance,
            atol=0.0,
        )

    assert np.all(np.isfinite(jax.values["final:moments"]))
    assert np.all(np.isfinite(jax.values["final:proxy_moments"]))
    assert float(native.values["final:objective_sum_squares"]) < float(
        native.values["initial:objective_sum_squares"]
    )
