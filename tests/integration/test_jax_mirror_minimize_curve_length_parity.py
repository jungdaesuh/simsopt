"""Matched native/JAX parity for the exact ``minimize_curve_length.py`` mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import get_case, native_minimize_curve_length
from examples.jax.parity.input_bundle import load_input_bundle
from scipy.optimize import OptimizeResult
from simsopt.objectives import LeastSquaresProblem

# A/reference-simple/runs/native-minimize-curve-length/captured-controlled-omp1/capture.json
OFFICIAL_FINAL_LENGTH = 18.849556246039942
OFFICIAL_RAW_STATUS = "2 `ftol` termination condition is satisfied."
OFFICIAL_NFEV = 62
OFFICIAL_NJEV = 55


def test_exact_minimize_curve_length_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-minimize-curve-length")
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
    assert (native.nfev, native.njev) == (OFFICIAL_NFEV, OFFICIAL_NJEV)
    assert native.scale == jax.scale == "bounded"
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)
    for name in (
        "initial:parameters",
        "initial:length",
        "initial:residual",
        "initial:residual_jacobian",
        "initial:objective_sum_squares",
        "initial:objective_gradient",
        "final:length",
        "final:residual",
        "final:objective_sum_squares",
    ):
        np.testing.assert_allclose(
            jax.values[name],
            native.values[name],
            rtol=1.0e-9,
            atol=1.0e-11,
        )
    # The official run stops on `ftol` 3.25e-07 short of the circle oracle
    # 6*pi, so the circle is a diagnostic and the official end point is the
    # assertion.
    for observation in (native, jax):
        np.testing.assert_allclose(
            observation.values["final:length"],
            OFFICIAL_FINAL_LENGTH,
            rtol=1.0e-9,
            atol=0.0,
        )


def test_native_default_minimize_curve_length_uses_upstream_solver_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-minimize-curve-length")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "native_default")
    _, arrays = load_input_bundle(input_root, bundle)

    calls: list[dict[str, float | int | str]] = []
    original_solve = native_minimize_curve_length.solve_official_least_squares

    def recording_solve(
        problem: LeastSquaresProblem, **kwargs: float | int | str
    ) -> OptimizeResult:
        calls.append(kwargs)
        return original_solve(problem, **kwargs)

    monkeypatch.setattr(
        native_minimize_curve_length,
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
    assert np.isfinite(native.values["final:length"])
    assert np.isfinite(native.values["final:objective_gradient"]).all()
    assert (
        native.values["final:objective_sum_squares"]
        < native.values["initial:objective_sum_squares"]
    )
