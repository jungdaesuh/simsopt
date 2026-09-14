"""Exact parity for ``2_Intermediate/stage_two_optimization.py``."""

from __future__ import annotations

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import (
    native_stage_two_optimization as stage_two_case,
)
from examples.jax.parity.input_bundle import load_input_bundle

#: Largest absolute gradient difference between the two evaluators at one state,
#: measured at both lanes' endpoints over OMP_NUM_THREADS 1/2/4/8/16: 3.772e-17,
#: against a gradient whose inf-norm is ~1e-5.  The gate is ~26x that, which is
#: tight enough that a real evaluator divergence cannot hide under it and loose
#: enough to absorb summation-order noise between the C++ and XLA reductions.
CROSS_EVALUATION_GRADIENT_ATOL = 1.0e-15

#: Largest relative objective difference in the same measurement: 3.826e-15.
CROSS_EVALUATION_OBJECTIVE_RTOL = 1.0e-12


def test_exact_standard_stage_two_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization")
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
        "maximum_normal_field",
        "total_curve_length",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=2.0e-8,
            atol=2.0e-10,
        )

    for stage in ("first", "final"):
        for observation in (native, jax):
            assert observation.values[f"{stage}:objective"] < (
                observation.values["initial:objective"]
            )
            assert np.all(
                np.isfinite(observation.values[f"{stage}:objective_gradient"])
            )

    # Endpoint gates, measured rather than guessed, with atol removed: at this
    # scale |final delta| is ~1e-9 and an atol of 1e-9 passed the assertion on
    # its own, whatever rtol said.  Both lanes now run SciPy L-BFGS-B under one
    # stopping rule and both stop on the iteration cap, so what is left is
    # trajectory divergence from floating-point arithmetic -- and the native
    # lane's trajectory is thread-count sensitive at this scale.  Measured
    # relative deltas over OMP_NUM_THREADS 1/2/4/8/16 on this box:
    #   first:objective  1.174e-07, 1.174e-07, 1.170e-07, 1.175e-07, 1.177e-07
    #   final:objective  1.750e-03, 8.516e-04, 1.418e-04, 3.094e-03, 5.723e-04
    # The gates are the worst of each sweep with roughly an order of magnitude
    # (first) and 3x (final) of margin.  They compare two TRAJECTORIES and are
    # deliberately the loose half of this file; the tight half is
    # test_both_evaluators_agree_at_each_lanes_endpoint, which compares the two
    # evaluators at one state and gates at 1e-15 absolute.
    np.testing.assert_allclose(
        jax.values["first:objective"],
        native.values["first:objective"],
        rtol=1.0e-6,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=1.0e-2,
        atol=0.0,
    )
    assert np.max(np.abs(native.values["taylor:errors"][:3])) <= 1.0e-4
    assert np.max(np.abs(jax.values["taylor:errors"][:3])) <= 1.0e-4


def test_both_evaluators_agree_at_each_lanes_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cross-evaluation: one state, two evaluators, at both lanes' endpoints.

    The endpoint comparison above cannot separate "the two lanes compute
    different physics" from "the two lanes walked different trajectories to the
    same cap".  This one can: each lane's final coordinates are handed to BOTH
    evaluators, so the only thing that can move the numbers is the objective
    program itself.  It is the in-tree form of the certification's own quality
    check, and it needs no rewritten copy of either lane's objective.
    """
    case = get_case("native-stage-two-optimization")
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

    for holder, endpoint in (
        ("jax", np.asarray(jax.values["final:parameters"], dtype=np.float64)),
        ("native", np.asarray(native.values["final:parameters"], dtype=np.float64)),
    ):
        by_native = stage_two_case.evaluate_at("native-cpu", bundle, endpoint)
        by_jax = stage_two_case.evaluate_at("jax-cpu", bundle, endpoint)

        np.testing.assert_allclose(
            by_jax["objective"],
            by_native["objective"],
            rtol=CROSS_EVALUATION_OBJECTIVE_RTOL,
            atol=0.0,
            err_msg=(
                f"the two evaluators disagree about the objective at the {holder} "
                "lane's endpoint, which is a difference in the objective program "
                "and not in either optimizer's trajectory"
            ),
        )
        np.testing.assert_allclose(
            by_jax["objective_gradient"],
            by_native["objective_gradient"],
            rtol=0.0,
            atol=CROSS_EVALUATION_GRADIENT_ATOL,
            err_msg=(
                f"the two evaluators disagree about the gradient at the {holder} "
                "lane's endpoint; the endpoint-objective gate above cannot see "
                "this, which is why this comparison exists"
            ),
        )
        assert np.all(np.isfinite(by_jax["objective_gradient"]))
        assert np.all(np.isfinite(by_native["objective_gradient"]))
