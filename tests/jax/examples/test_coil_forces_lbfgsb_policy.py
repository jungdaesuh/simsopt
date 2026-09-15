"""Coil-forces native and JAX L-BFGS-B option dicts stay policy-identical.

The native script and the JAX mirror must hand SciPy the same L-BFGS-B
options, including ``maxls``.  A restated literal would go stale when either
lane changes, so both sides of every assertion are read from source.
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import asdict
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from simsopt_jax.examples.scalar_stage import solve_scalar_stage
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions
from simsopt_jax.solve.serial import TraceableParametricScalarProblem

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NATIVE_SCRIPT = REPOSITORY_ROOT / "examples" / "3_Advanced" / "coil_forces.py"
JAX_SCRIPT = REPOSITORY_ROOT / "examples" / "jax" / "3_Advanced" / "coil_forces.py"
NATIVE_BUDGET = 400
NATIVE_MAXLS = 32


def _module_constants(module: ast.Module) -> dict[str, object]:
    """Top-level numeric assigns, with ``X = a if cond else b`` taking the else arm."""
    constants: dict[str, object] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = node.value
        if isinstance(value, ast.Constant):
            constants[target.id] = value.value
        elif isinstance(value, ast.IfExp) and isinstance(value.orelse, ast.Constant):
            constants[target.id] = value.orelse.value
    return constants


def _option_value(node: ast.expr, constants: dict[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return constants[node.id]
    raise AssertionError(f"unsupported option value: {ast.dump(node)}")


def _native_minimize_policies() -> list[dict[str, object]]:
    """The L-BFGS-B option dicts both native ``minimize`` calls name, plus ``tol``."""
    source = NATIVE_SCRIPT.read_text(encoding="utf-8")
    module = ast.parse(source)
    constants = _module_constants(module)
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
    ]
    assert len(calls) == 2, (
        f"{NATIVE_SCRIPT.name} is expected to make exactly two minimize calls; "
        f"it makes {len(calls)}"
    )
    policies = []
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        options_node = keywords["options"]
        assert isinstance(options_node, ast.Dict)
        options = {
            key.value: _option_value(value, constants)
            for key, value in zip(options_node.keys, options_node.values)
            if isinstance(key, ast.Constant)
        }
        policies.append(
            {
                "method": keywords["method"].value,
                "maxiter": options["maxiter"],
                "maxcor": options["maxcor"],
                "maxls": options["maxls"],
                "tol": keywords["tol"].value,
            }
        )
    return policies


def _jax_native_constants() -> dict[str, object]:
    module = ast.parse(JAX_SCRIPT.read_text(encoding="utf-8"))
    constants = _module_constants(module)
    return {
        name: constants[name]
        for name in (
            "NATIVE_ITERATIONS",
            "NATIVE_TOLERANCE",
            "NATIVE_HISTORY_SIZE",
            "NATIVE_LINE_SEARCH_MAX",
        )
    }


def _jax_solve_scalar_stage_keywords() -> dict[str, str]:
    """Keyword names ``_run_stage`` hands ``solve_scalar_stage``."""
    module = ast.parse(JAX_SCRIPT.read_text(encoding="utf-8"))
    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Name) or func.id != "solve_scalar_stage":
            continue
        return {
            keyword.arg: keyword.value.id
            for keyword in node.keywords
            if keyword.arg is not None and isinstance(keyword.value, ast.Name)
        }
    raise AssertionError(f"{JAX_SCRIPT.name} does not call solve_scalar_stage")


def test_coil_forces_lanes_share_one_lbfgsb_option_dict_including_maxls() -> None:
    """Native minimize options and the JAX native_matched dict are the same rule."""
    native_policies = _native_minimize_policies()
    assert native_policies[0] == native_policies[1], (
        "the native script's two stages stop under different rules, so there is "
        f"no single native policy to match: {native_policies}"
    )
    native = native_policies[0]
    jax_constants = _jax_native_constants()
    handed = _jax_solve_scalar_stage_keywords()
    signature = inspect.signature(fmin_l_bfgs_b)
    options = ScipyLBFGSBOptions.native_matched(
        maxiter=jax_constants["NATIVE_ITERATIONS"],
        maxcor=jax_constants["NATIVE_HISTORY_SIZE"],
        tol=jax_constants["NATIVE_TOLERANCE"],
        maxls=jax_constants["NATIVE_LINE_SEARCH_MAX"],
    )

    assert native["method"] == "L-BFGS-B"
    assert native["maxiter"] == jax_constants["NATIVE_ITERATIONS"] == NATIVE_BUDGET
    assert native["maxcor"] == jax_constants["NATIVE_HISTORY_SIZE"] == 300
    assert native["maxls"] == jax_constants["NATIVE_LINE_SEARCH_MAX"] == NATIVE_MAXLS
    assert native["tol"] == jax_constants["NATIVE_TOLERANCE"] == 1.0e-15
    assert handed["maxls"] == "NATIVE_LINE_SEARCH_MAX"
    assert handed["maxcor"] == "NATIVE_HISTORY_SIZE"
    assert handed["tol"] == "NATIVE_TOLERANCE"
    assert asdict(options) == {
        "maxiter": native["maxiter"],
        "maxfun": signature.parameters["maxfun"].default,
        "gtol": native["tol"],
        "ftol": native["tol"],
        "maxcor": native["maxcor"],
        "maxls": native["maxls"],
    }


def test_raising_maxls_is_bitwise_idle_when_the_line_search_never_hits_20() -> None:
    """A run that never exhausts SciPy's default maxls is unchanged at 32."""

    def solve(maxls: int):
        problem = TraceableParametricScalarProblem(
            objective_fn=lambda parameters, weight: jnp.sum(weight * parameters**2),
            objective_parameter=jnp.asarray(1.0, dtype=jnp.float64),
            x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        )
        return solve_scalar_stage(
            problem,
            driver=Driver.SCIPY_LBFGSB,
            max_steps=32,
            maxcor=300,
            tol=1.0e-15,
            maxls=maxls,
        )

    limited = solve(20)
    raised = solve(32)

    assert limited.nfev == raised.nfev
    assert limited.nfev < 20
    assert limited.nit == raised.nit
    assert limited.fun == raised.fun
    assert limited.message == raised.message
    np.testing.assert_array_equal(limited.x, raised.x)
    np.testing.assert_array_equal(limited.jac, raised.jac)
