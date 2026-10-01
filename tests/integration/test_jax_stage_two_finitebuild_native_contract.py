"""Native solver-policy coverage for the finite-build Stage-II mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
from examples.jax.parity.cases import native_stage_two_optimization_finitebuild
from examples.jax.parity.input_bundle import load_input_bundle
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions


@dataclass(frozen=True)
class _MinimizeCall:
    jac: bool
    method: str
    options: dict[str, object]
    tol: float


def test_native_finitebuild_uses_upstream_history_at_native_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both scales retain the official optimizer policy.

    The bundle is built once at bounded scale and its scale label is changed
    only for the second native call. This keeps the geometry inexpensive while
    exercising the native case's scale policy. The SciPy delegate is spied on,
    so no optimizer iteration or GPU route runs.
    """
    input_root = tmp_path / "inputs"
    bounded_bundle = native_stage_two_optimization_finitebuild.create_input(
        input_root,
        "bounded",
    )
    _, arrays = load_input_bundle(input_root, bounded_bundle)
    minimize_calls: list[_MinimizeCall] = []

    def spy_minimize(
        _objective,
        initial_parameters,
        *,
        jac,
        method,
        options,
        tol,
    ):
        minimize_calls.append(
            _MinimizeCall(
                jac=jac,
                method=method,
                options=dict(options),
                tol=tol,
            )
        )
        return scipy.optimize.OptimizeResult(
            x=np.asarray(initial_parameters, dtype=np.float64).copy(),
            success=False,
            status=0,
            nit=0,
            nfev=0,
            njev=0,
        )

    monkeypatch.setattr(
        native_stage_two_optimization_finitebuild,
        "minimize",
        spy_minimize,
    )

    native_stage_two_optimization_finitebuild._native(bounded_bundle, arrays)
    bounded_call = minimize_calls.pop()

    native_default_bundle = replace(bounded_bundle, scale="native_default")
    native_stage_two_optimization_finitebuild._native(
        native_default_bundle,
        arrays,
    )
    native_default_call = minimize_calls.pop()

    assert native_stage_two_optimization_finitebuild.FINITE_BUILD_LBFGS_HISTORY == 400
    # Upstream names maxiter, maxcor and tol and leaves the evaluation budget
    # alone, so the native lane's maxfun must be SciPy's own default rather
    # than a branch-chosen number that happens to equal it today.
    assert ScipyLBFGSBOptions.maxfun == 15000
    assert bounded_call.jac is True
    assert bounded_call.method == "L-BFGS-B"
    assert bounded_call.options["maxcor"] == 400
    assert bounded_call.options["maxfun"] == ScipyLBFGSBOptions.maxfun
    assert native_default_call.jac is True
    assert native_default_call.method == "L-BFGS-B"
    assert native_default_call.options["maxcor"] == 400
    assert native_default_call.options["maxfun"] == ScipyLBFGSBOptions.maxfun
    assert native_default_call.options["maxiter"] == 3
    assert native_default_call.options["gtol"] == 1.0e-20
    assert native_default_call.options["ftol"] == 1.0e-20
    assert native_default_call.tol == 1.0e-20
    assert not minimize_calls
