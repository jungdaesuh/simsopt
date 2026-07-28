"""Exact parity for the modular and sector-saddle GSCO mirrors."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


@pytest.mark.parametrize(
    "case_id",
    (
        "native-wireframe-gsco-modular",
        "native-wireframe-gsco-sector-saddle",
    ),
)
def test_exact_wireframe_gsco_matches_native_and_jax_cpu(
    case_id: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case(case_id)
    input_root = tmp_path / case_id
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
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    assert int(jax.values["final:iterations"]) > 0
    assert float(jax.values["final:normal_objective"]) < float(
        jax.values["initial:normal_objective"]
    )
