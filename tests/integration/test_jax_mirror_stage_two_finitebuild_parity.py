"""Exact parity for the finite-build Stage-II mirror."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case
from examples.jax.parity.input_bundle import load_input_bundle


def test_exact_finitebuild_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-finitebuild")
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
    published = set(native.values)
    assert published == set(jax.values)
    assert {"initial:minimum_clearance", "final:minimum_clearance"} <= published

    for observable in native.values:
        if observable.startswith("initial:"):
            np.testing.assert_allclose(
                jax.values[observable],
                native.values[observable],
                rtol=2.0e-8,
                atol=2.0e-10,
            )

    for observation in (native, jax):
        assert float(observation.values["final:objective"]) < float(
            observation.values["initial:objective"]
        )
        assert np.all(np.isfinite(observation.values["final:objective_gradient"]))

    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-2,
        atol=1.0e-9,
    )

    # Minimum clearance is an exact geometry quantity that the two lanes
    # compute with independent evaluators: native reduces the simsoptpp
    # ``CurveCurveDistance.shortest_distance()`` over the symmetric base
    # curves, JAX takes the pairwise minimum packed at diagnostics index 3.
    # Measured once at bounded scale on CPU fp64: the initial state agrees
    # bitwise (gap exactly 0.0) and the final state agrees to a relative gap
    # of 3.17e-12 (absolute 3.00e-13), inherited from the 4.08e-12 spread
    # between the two lanes' converged parameters rather than from the
    # reductions.  The tolerance below sits ~160x above that measured gap.
    for prefix in ("initial", "final"):
        observable = f"{prefix}:minimum_clearance"
        assert float(native.values[observable]) > 0.0
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=5.0e-10,
            atol=1.0e-12,
        )

    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax.values["taylor:errors"][:3])) <= 1.0e-4
