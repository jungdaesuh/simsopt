"""The finite-build workflow keeps upstream's optimizer options in both modes.

Two outer optimizers are shipped and each gets its own statement here: the
MIRROR default is ``scipy.optimize.minimize(..., method="L-BFGS-B")`` over the
JAX objective, and the device-resident L-BFGS-B is the opt-in performance
mode.  Both must carry upstream's option table, and selecting a mode must
change the optimizer and nothing else.

Upstream's numbers are not written here: they are read from the one
``minimize(...)`` call in upstream's own script,
``examples/3_Advanced/stage_two_optimization_finitebuild.py``, at the value its
official (not GitHub Actions) run takes, while the options under test are built
by the workflow from its own package constants.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt_jax.examples import fused_lane, scalar_stage
from simsopt_jax.examples.stage_two_finitebuild import (
    FINITE_BUILD_DEVICE_DRIVER,
    FINITE_BUILD_LBFGS_HISTORY,
    FINITE_BUILD_NATIVE_ITERATIONS,
    FINITE_BUILD_OFFICIAL_DRIVER,
    FINITE_BUILD_TOLERANCE,
    solve_finite_build_stage_two,
)
from simsopt_jax.solve.contracts import (
    OptimizerInput,
    OptimizerResult,
    OptionsBase,
    ValueAndGradFn,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions
from simsopt_jax.solve.simsopt.contracts import SimsoptLBFGSBOptions

NATIVE_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "3_Advanced"
    / "stage_two_optimization_finitebuild.py"
)


@dataclass(frozen=True)
class ProviderCall:
    """One ``scipy.optimize.minimize`` call as upstream's script spells it."""

    options: Mapping[object, object]
    tol: object


def _official_value(node: ast.expr, constants: Mapping[str, ast.expr]) -> object:
    """A literal, following module constants to their official-run value.

    ``X if in_github_actions else Y`` is upstream's CI shortcut; the official
    run is the one outside GitHub Actions, so it takes ``Y``.
    """
    if isinstance(node, ast.Name):
        return _official_value(constants[node.id], constants)
    if (
        isinstance(node, ast.IfExp)
        and isinstance(node.test, ast.Name)
        and node.test.id == "in_github_actions"
    ):
        return _official_value(node.orelse, constants)
    return ast.literal_eval(node)


def _native_provider_call(script: Path) -> ProviderCall:
    """The single ``minimize(...)`` call of an upstream example script."""
    module = ast.parse(script.read_text(encoding="utf-8"))
    constants = {
        target.id: statement.value
        for statement in module.body
        if isinstance(statement, ast.Assign)
        for target in statement.targets
        if isinstance(target, ast.Name)
    }
    (call,) = (
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
    )
    keywords = {
        keyword.arg: keyword.value
        for keyword in call.keywords
        if keyword.arg is not None
    }
    options = keywords["options"]
    assert isinstance(options, ast.Dict)
    return ProviderCall(
        options={
            _official_value(key, constants): _official_value(value, constants)
            for key, value in zip(options.keys, options.values, strict=True)
            if key is not None
        },
        tol=_official_value(keywords["tol"], constants),
    )


#: The one L-BFGS-B call the official run of
#: ``examples/3_Advanced/stage_two_optimization_finitebuild.py`` makes.
OFFICIAL_CALL = _native_provider_call(NATIVE_SCRIPT)


def _official_stopping_tolerance(call: ProviderCall) -> float:
    """Upstream names ftol and gtol here, and passes ``tol`` as the same value."""
    ftol = call.options["ftol"]
    assert isinstance(ftol, float)
    assert ftol == call.options["gtol"] and ftol == call.tol
    return ftol


#: Upstream's own table; ``maxfun`` and ``maxls`` stay at SciPy's defaults
#: because upstream does not set them, which is what ``native_matched`` encodes.
OFFICIAL_OPTIONS = ScipyLBFGSBOptions.native_matched(
    maxiter=int(OFFICIAL_CALL.options["maxiter"]),
    maxcor=int(OFFICIAL_CALL.options["maxcor"]),
    tol=_official_stopping_tolerance(OFFICIAL_CALL),
)

#: The opt-in performance mode must carry that table field for field.
OFFICIAL_DEVICE_OPTIONS = SimsoptLBFGSBOptions(
    maxiter=OFFICIAL_OPTIONS.maxiter,
    maxfun=OFFICIAL_OPTIONS.maxfun,
    maxcor=OFFICIAL_OPTIONS.maxcor,
    gtol=OFFICIAL_OPTIONS.gtol,
    ftol=OFFICIAL_OPTIONS.ftol,
    maxls=OFFICIAL_OPTIONS.maxls,
)


def _prepared() -> fused_lane.PreparedFusedLaneSolve:
    return fused_lane.prepare_fused_lane_solve(
        objective_fn=lambda x: jnp.vdot(x, x),
        diagnostics_fn=lambda x: x,
        initial_parameters=jax.device_put(np.ones(3, dtype=np.float64)),
        objective_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
    )


def _record(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    name: str,
) -> list[OptionsBase]:
    """Capture the options actually dispatched, still running the real solve."""
    captured: list[OptionsBase] = []
    original = getattr(module, name)

    def record_options(
        objective: ValueAndGradFn,
        initial: OptimizerInput,
        *,
        driver: Driver,
        options: OptionsBase,
    ) -> OptimizerResult:
        captured.append(options)
        return original(objective, initial, driver=driver, options=options)

    monkeypatch.setattr(module, name, record_options)
    return captured


def test_finite_build_default_constructs_the_official_scipy_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror default is the provider upstream calls, with its options."""
    captured = _record(monkeypatch, scalar_stage, "dispatch_minimize")

    result = solve_finite_build_stage_two(
        _prepared(),
        driver=FINITE_BUILD_OFFICIAL_DRIVER,
        max_steps=FINITE_BUILD_NATIVE_ITERATIONS,
        rtol=FINITE_BUILD_TOLERANCE,
        atol=FINITE_BUILD_TOLERANCE,
    )

    assert captured == [OFFICIAL_OPTIONS]
    assert result.driver is Driver.SCIPY_LBFGSB
    np.testing.assert_allclose(result.x, np.zeros(3), atol=1.0e-12, rtol=0)


def test_finite_build_device_mode_constructs_the_same_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in performance mode carries the identical option table."""
    captured = _record(monkeypatch, fused_lane, "minimize")

    result = solve_finite_build_stage_two(
        _prepared(),
        driver=FINITE_BUILD_DEVICE_DRIVER,
        max_steps=FINITE_BUILD_NATIVE_ITERATIONS,
        rtol=FINITE_BUILD_TOLERANCE,
        atol=FINITE_BUILD_TOLERANCE,
    )

    assert captured == [OFFICIAL_DEVICE_OPTIONS]
    assert result.driver is Driver.SIMSOPT_LBFGSB
    assert result.success
    np.testing.assert_allclose(result.x, np.zeros(3), atol=1.0e-12, rtol=0)


def test_official_provider_refuses_two_different_stopping_tolerances() -> None:
    """``minimize(..., tol=...)`` sets ftol and gtol together; two cannot pass."""
    with pytest.raises(ValueError, match="rtol and atol must be equal"):
        solve_finite_build_stage_two(
            _prepared(),
            driver=FINITE_BUILD_OFFICIAL_DRIVER,
            max_steps=FINITE_BUILD_NATIVE_ITERATIONS,
            rtol=FINITE_BUILD_TOLERANCE,
            atol=1.0e-12,
        )


def test_finite_build_uses_official_stopping_tolerances() -> None:
    """The constants the shipped mirror passes as ``rtol`` and ``atol``."""
    assert FINITE_BUILD_TOLERANCE == 1.0e-20
    assert FINITE_BUILD_LBFGS_HISTORY == 400
    assert FINITE_BUILD_NATIVE_ITERATIONS == 400
