"""Optimizer-policy equality between the standard Stage-II mirror and its twin.

The mirror's SciPy route and ``examples/2_Intermediate/stage_two_optimization.py``
must stop under the same rule, so the native side of every assertion here is
read out of the native script itself with :mod:`ast` rather than restated: a
restated policy is the thing that goes stale when the script changes, and the
whole point of these tests is to notice that it did.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from scipy.optimize import fmin_l_bfgs_b

from simsopt_jax.examples.stage_two_standard import _solve_stage
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.scipy.contracts import ScipyLBFGSBOptions
from simsopt_jax.solve.serial import TraceableParametricScalarProblem

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
NATIVE_SCRIPT = (
    REPOSITORY_ROOT / "examples" / "2_Intermediate" / "stage_two_optimization.py"
)
NATIVE_BUDGET = 400


def _native_iteration_budget(module: ast.Module) -> dict[str, int]:
    """``MAXITER = 50 if in_github_actions else 400``, resolved to its non-CI arm.

    The probe and the mirror both run outside GitHub Actions, and the native
    child's environment is scrubbed of ``CI`` precisely so it takes that arm.
    """
    budgets: dict[str, int] = {}
    for node in module.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != "MAXITER":
            continue
        value = node.value
        budgets[target.id] = (
            value.orelse.value if isinstance(value, ast.IfExp) else value.value
        )
    assert budgets == {"MAXITER": NATIVE_BUDGET}, (
        f"{NATIVE_SCRIPT.name} no longer sets MAXITER to {NATIVE_BUDGET} outside "
        f"GitHub Actions; it resolves to {budgets}"
    )
    return budgets


def _native_minimize_policy() -> dict[str, object]:
    """The stopping rule the native script hands ``scipy.optimize.minimize``.

    ``tol`` is expanded into ``ftol`` and ``gtol`` because that is what
    ``scipy.optimize.minimize`` does with it for L-BFGS-B, and what the solver
    reads is the pair, not the ``tol``.  Options the call does not name are
    SciPy's own defaults, taken from the public ``fmin_l_bfgs_b`` signature.
    """
    module = ast.parse(NATIVE_SCRIPT.read_text(encoding="utf-8"))
    budget = _native_iteration_budget(module)
    calls = [
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
    ]
    assert len(calls) == 2, (
        f"{NATIVE_SCRIPT.name} is expected to make exactly two minimize calls "
        f"(one per length-weight stage); it makes {len(calls)}"
    )
    signature = inspect.signature(fmin_l_bfgs_b)
    policies = []
    for call in calls:
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        options = {
            key.value: (
                value.value if isinstance(value, ast.Constant) else budget[value.id]
            )
            for key, value in zip(
                keywords["options"].keys, keywords["options"].values
            )
        }
        tolerance = keywords["tol"].value
        policies.append(
            {
                "method": keywords["method"].value,
                "maxiter": options["maxiter"],
                "maxcor": options["maxcor"],
                "ftol": options.get("ftol", tolerance),
                "gtol": options.get("gtol", tolerance),
                "maxfun": options.get(
                    "maxfun", signature.parameters["maxfun"].default
                ),
                "maxls": options.get("maxls", signature.parameters["maxls"].default),
            }
        )
    assert policies[0] == policies[1], (
        "the native script's two stages stop under different rules, so there is "
        f"no single native policy to match: {policies}"
    )
    return policies[0]


def _quadratic_stage(budget: int) -> tuple[TraceableParametricScalarProblem, object]:
    """A two-variable stage that exercises the real solve path in milliseconds."""
    problem = TraceableParametricScalarProblem(
        objective_fn=lambda parameters, weight: jnp.sum(weight * parameters**2),
        objective_parameter=jnp.asarray(1.0, dtype=jnp.float64),
        x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
    )
    result = _solve_stage(problem, Driver.SCIPY_LBFGSB, budget, 1.0e-15, 1.0e-15)
    return problem, result


@pytest.mark.parametrize("option", ["maxiter", "maxcor", "ftol", "gtol", "maxfun", "maxls"])
def test_scipy_route_stops_under_the_native_twins_rule(option: str) -> None:
    native = _native_minimize_policy()
    _problem, result = _quadratic_stage(NATIVE_BUDGET)
    actual = getattr(result.options_used, option)

    assert actual == native[option], (
        f"the mirror's SciPy route ran with {option}={actual!r} while its native "
        f"twin runs with {option}={native[option]!r}; the two lanes are not "
        "solving under the same stopping rule and no matched-policy speed claim "
        "can be read off them"
    )


def test_native_matched_maps_one_tol_onto_both_scipy_tolerances() -> None:
    """``tol`` reaches the solver as ``ftol`` AND ``gtol``, never one of them.

    This is the defect class that made a sibling mirror stop at status 2: a
    route that mapped only one of the pair ran seven orders of magnitude looser
    on the other than the native script it was compared against.
    """
    native = _native_minimize_policy()
    options = ScipyLBFGSBOptions.native_matched(
        maxiter=native["maxiter"], maxcor=native["maxcor"], tol=1.0e-15
    )

    assert options.ftol == 1.0e-15
    assert options.gtol == 1.0e-15
    assert options.ftol == options.gtol == native["ftol"] == native["gtol"]


def test_native_matched_names_nothing_the_native_call_does_not() -> None:
    """Unnamed options stay at SciPy's defaults, on both lanes."""
    native = _native_minimize_policy()
    options = ScipyLBFGSBOptions.native_matched(
        maxiter=native["maxiter"], maxcor=native["maxcor"], tol=native["ftol"]
    )
    signature = inspect.signature(fmin_l_bfgs_b)

    assert options.maxiter == native["maxiter"] == NATIVE_BUDGET
    assert options.maxcor == native["maxcor"] == 300
    assert options.maxfun == signature.parameters["maxfun"].default
    assert options.maxls == signature.parameters["maxls"].default
    assert options == ScipyLBFGSBOptions(
        maxiter=NATIVE_BUDGET, maxcor=300, ftol=1.0e-15, gtol=1.0e-15
    )


def test_scipy_route_builds_its_options_through_the_named_constructor() -> None:
    """The stage's options are the constructor's output, not a parallel literal."""
    _problem, result = _quadratic_stage(NATIVE_BUDGET)

    assert result.options_used == ScipyLBFGSBOptions.native_matched(
        maxiter=NATIVE_BUDGET, maxcor=300, tol=1.0e-15
    )


def test_scipy_route_refuses_a_tolerance_pair_no_native_twin_can_express() -> None:
    """One ``tol`` is what the rule has; a split pair is refused, not halved."""
    problem = TraceableParametricScalarProblem(
        objective_fn=lambda parameters, weight: jnp.sum(weight * parameters**2),
        objective_parameter=jnp.asarray(1.0, dtype=jnp.float64),
        x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
    )

    with pytest.raises(ValueError, match="ftol and gtol"):
        _solve_stage(problem, Driver.SCIPY_LBFGSB, 32, 1.0e-15, 1.0e-8)


def test_scipy_route_carries_no_evaluation_cap_of_its_own() -> None:
    """The fused device port's ``maxfun = 20 * maxiter`` must not leak here.

    Before this contract the SciPy route passed ``maxfun=max_steps * 20`` (8000
    at the native budget of 400) while the native twin stopped at SciPy's 15000,
    so this assertion fails against that behaviour.
    """
    _problem, result = _quadratic_stage(NATIVE_BUDGET)

    assert result.options_used.maxfun == ScipyLBFGSBOptions().maxfun
    assert result.options_used.maxfun != NATIVE_BUDGET * 20
    assert (
        result.options_used.maxfun
        == inspect.signature(fmin_l_bfgs_b).parameters["maxfun"].default
    )


def test_scipy_route_reports_the_minimize_call_as_its_wall_clock() -> None:
    """``wallclock_s`` is the per-stage optimizer region a driver may publish."""
    _problem, result = _quadratic_stage(32)

    assert result.wallclock_s > 0.0
    assert result.driver == Driver.SCIPY_LBFGSB


def test_scipy_route_publishes_the_endpoint_back_onto_the_problem() -> None:
    """The stage endpoint becomes ``problem.x`` so the next stage starts there."""
    problem, result = _quadratic_stage(32)

    assert isinstance(problem.x, jax.Array)
    assert float(jnp.max(jnp.abs(problem.x - jnp.asarray(result.x)))) == 0.0
