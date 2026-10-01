"""Matched native/JAX parity for the exact ``just_a_quadratic.py`` mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case, native_just_a_quadratic
from examples.jax.parity.input_bundle import load_input_bundle
from scipy.optimize import OptimizeResult
from simsopt.objectives import LeastSquaresProblem

# A/reference-simple/runs/native-just-a-quadratic/captured-natural-omp1/capture.json
OFFICIAL_SOLUTION = (1.0, 1.9999999992096138, 2.9999999998773283)
OFFICIAL_RAW_STATUS = "1 `gtol` termination condition is satisfied."
OFFICIAL_NFEV = 4
OFFICIAL_NJEV = 4


def test_exact_quadratic_case_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-just-a-quadratic")
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
    # One stopping rule for both lanes: the official SciPy defaults.
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.raw_status == jax.raw_status == OFFICIAL_RAW_STATUS
    # Both lanes are pinned to the official counts because this problem's
    # three weighted residuals are affine in its three parameters: TRF's
    # Gauss-Newton step is exact, so the trajectory cannot depend on rounding
    # and 4 evaluations is an invariant of the problem rather than of a build.
    # The curve and surface cases, whose residuals are nonlinear, publish their
    # counters instead of pinning them.
    assert (native.nfev, native.njev) == (OFFICIAL_NFEV, OFFICIAL_NJEV)
    assert (jax.nfev, jax.njev) == (OFFICIAL_NFEV, OFFICIAL_NJEV)
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)
    for name in native.values:
        np.testing.assert_allclose(
            jax.values[name],
            native.values[name],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    # The official run itself stops 7.90e-10 from the exact minimizer, so the
    # bounded mirror is asserted against the official end point.
    np.testing.assert_allclose(
        jax.values["final:parameters"],
        OFFICIAL_SOLUTION,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_native_default_just_a_quadratic_uses_upstream_solver_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-just-a-quadratic")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "native_default")
    _, arrays = load_input_bundle(input_root, bundle)

    calls: list[dict[str, float | int | str]] = []
    original_solve = native_just_a_quadratic.solve_official_least_squares

    def recording_solve(
        problem: LeastSquaresProblem, **kwargs: float | int | str
    ) -> OptimizeResult:
        calls.append(kwargs)
        return original_solve(problem, **kwargs)

    monkeypatch.setattr(
        native_just_a_quadratic,
        "solve_official_least_squares",
        recording_solve,
    )
    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        native = case.execute("native-cpu", bundle, arrays)

    assert calls == [{}]
    assert native.scale == "native_default"
    assert np.isfinite(native.values["final:parameters"]).all()
    assert (
        native.values["final:objective_sum_squares"]
        < native.values["initial:objective_sum_squares"]
    )
    np.testing.assert_allclose(
        native.values["final:residual"],
        np.zeros(3),
        rtol=0.0,
        atol=1.0e-8,
    )
