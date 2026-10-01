"""Matched parity for the exact ``stage_two_optimization_minimal.py`` mirror."""

from __future__ import annotations

import sys
from contextlib import chdir
from pathlib import Path

import numpy as np
import pytest
import scipy.optimize
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import native_stage_two_optimization_minimal
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    CONVERGED_GRADIENT_INF_NORM_BOUND,
    _scale_configuration,
    _terminal_status,
    build_native_evaluator_for_configuration,
)
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.official_reference import load_official_reference
from simsopt_contracts.optimization_endpoint import StatusConvention
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.examples.stage_two_minimal import MINIMAL_STAGE_TWO_NATIVE_ITERATIONS
from simsopt_jax.solve.driver import Driver

# venv site-packages/tests shadows the repo tests package, so the helper
# is imported as a top-level module from the tests/ directory.
_TESTS_ROOT = str(Path(__file__).resolve().parents[1])
if _TESTS_ROOT not in sys.path:
    sys.path.append(_TESTS_ROOT)
from parity_native_cpu import run_native_cpu_child

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Every official number below comes from the tracked fixture of the official
# run of examples/1_Simple/stage_two_optimization_minimal.py at upstream
# 9e027eac38028d57aa23777be52a781aa860e347; none is pasted here as a literal.
OFFICIAL = load_official_reference("native-stage-two-optimization-minimal")
OFFICIAL_CALL = OFFICIAL.provider_calls[0]
OFFICIAL_INITIAL_OBJECTIVE = OFFICIAL.scalar("taylor:objective")
OFFICIAL_FINAL_OBJECTIVE = OFFICIAL.scalar("final:objective")
OFFICIAL_ITERATION_LIMIT = OFFICIAL_CALL.options["maxiter"]
OFFICIAL_ITERATIONS = OFFICIAL_CALL.result["nit"]
OFFICIAL_FUNCTION_EVALUATIONS = OFFICIAL_CALL.result["nfev"]
OFFICIAL_FINAL_GRADIENT_INFINITY_NORM = float(
    np.max(np.abs(OFFICIAL.array("final:objective_gradient")))
)


def _official_objective(scale: ExecutionScale):
    return build_native_evaluator_for_configuration(
        _scale_configuration(scale)
    ).objective


# The official objective was captured at OMP_NUM_THREADS=1 (the fixture's
# ``capture.threads``). SquaredFlux sums its quadrature terms in an OpenMP
# reduction (src/simsoptpp/integral_BdotN.cpp, as upstream), whose bits depend
# on the thread team, so the value is reproduced bit for bit only in a child
# whose OpenMP is pinned before the extension loads.
_NATIVE_DEFAULT_INITIAL_OBJECTIVE_CHILD = """\
from examples.jax.parity.cases.native_stage_two_optimization_minimal import (
    _scale_configuration,
    build_native_evaluator_for_configuration,
)

objective = build_native_evaluator_for_configuration(
    _scale_configuration("native_default")
).objective
print(repr(float(objective.J())))
"""


def test_minimal_currents_carry_the_official_scaled_parametrization() -> None:
    """Upstream builds ``Current(1.0) * 1e5``: the free dof is 1.0, not 1e5.

    L-BFGS-B is not scale invariant, so a lane whose current degree of freedom
    is 1e5 optimizes a differently scaled problem than upstream.
    """
    configuration = _scale_configuration("native_default")

    assert configuration["initial_current_degree_of_freedom"] == 1.0
    assert configuration["current_scale"] == 1.0e5

    objective = _official_objective("native_default")
    current_indices = [
        index
        for index, name in enumerate(objective.dof_names)
        if name.startswith("Current")
    ]

    assert len(current_indices) == 3
    np.testing.assert_array_equal(
        np.asarray(objective.x, dtype=np.float64)[current_indices],
        np.ones(3),
    )
    completed = run_native_cpu_child(
        _NATIVE_DEFAULT_INITIAL_OBJECTIVE_CHILD, repo_root=_REPO_ROOT
    )
    assert completed.returncode == 0, completed.stderr
    assert float(completed.stdout) == OFFICIAL_INITIAL_OBJECTIVE


def test_minimal_official_endpoint_is_admitted_as_a_budget_exit() -> None:
    """The official run stops at its iteration cap and is never ``converged``."""
    terminal = _terminal_status(
        status_convention="scipy-lbfgsb",
        provider_success=False,
        provider_status=1,
        iterations=OFFICIAL_ITERATIONS,
        max_iterations=OFFICIAL_ITERATION_LIMIT,
        initial_values={
            "initial:objective": np.asarray(OFFICIAL_INITIAL_OBJECTIVE),
            "initial:objective_gradient": np.asarray([1.0]),
            "initial:parameters": np.asarray([1.0]),
        },
        final_values={
            "final:objective": np.asarray(OFFICIAL_FINAL_OBJECTIVE),
            "final:objective_gradient": np.asarray(
                [OFFICIAL_FINAL_GRADIENT_INFINITY_NORM]
            ),
            "final:parameters": np.asarray([1.0]),
            "final:total_curve_length": np.asarray(
                OFFICIAL.scalar("final:total_curve_length")
            ),
        },
        length_target=float(_scale_configuration("native_default")["length_target"]),
    )

    assert terminal.normalized_status == "budget_exhausted"
    assert terminal.success is False
    assert OFFICIAL_FUNCTION_EVALUATIONS > OFFICIAL_ITERATIONS


#: The largest end-point gradient among upstream's own one-ulp starts of the
#: official script (9e027eac3, one thread, k = 0..40; k = 27), every one of them
#: an L-BFGS-B stop at the 300-iteration cap.  The pre-registered nine alone
#: reach 1.171e-4 (k = 3).
UPSTREAM_SCATTER_MAX_GRADIENT_INF_NORM = 1.945e-4


@pytest.mark.parametrize(
    ("provider_success", "provider_status", "iterations", "expected_status"),
    [
        (False, 1, 300, "budget_exhausted"),
        (True, 0, 76, "failed"),
    ],
)
def test_minimal_gradient_bound_binds_converged_stops_only(
    provider_success: bool,
    provider_status: int,
    iterations: int,
    expected_status: str,
) -> None:
    """A gradient above the bound fails a converged stop and not a budget stop.

    Upstream's own script stops at its iteration cap from every one-ulp start
    with an end gradient above ``CONVERGED_GRADIENT_INF_NORM_BOUND`` on 6 of 41
    starts, so that bound is no property of a budget stop; a converged stop
    claims stationarity and keeps it.
    """
    assert UPSTREAM_SCATTER_MAX_GRADIENT_INF_NORM > CONVERGED_GRADIENT_INF_NORM_BOUND
    terminal = _terminal_status(
        status_convention="scipy-lbfgsb",
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=300,
        initial_values={
            "initial:objective": np.asarray(1.0),
            "initial:objective_gradient": np.asarray([1.0]),
            "initial:parameters": np.asarray([0.0]),
        },
        final_values={
            "final:objective": np.asarray(0.5),
            "final:objective_gradient": np.asarray(
                [UPSTREAM_SCATTER_MAX_GRADIENT_INF_NORM]
            ),
            "final:parameters": np.asarray([1.0]),
            "final:total_curve_length": np.asarray(18.0),
        },
        length_target=18.0,
    )

    assert terminal.normalized_status == expected_status
    assert terminal.success is False


@pytest.mark.parametrize(
    ("final_objective", "total_curve_length"),
    [(1.5, 18.0), (float("nan"), 18.0), (0.5, 1.1 * 18.0 * (1.0 + 1.0e-12))],
)
def test_minimal_budget_stop_keeps_decrease_finiteness_and_length(
    final_objective: float,
    total_curve_length: float,
) -> None:
    """The clauses upstream's scatter satisfies on every start still bind a budget stop."""
    terminal = _terminal_status(
        status_convention="scipy-lbfgsb",
        provider_success=False,
        provider_status=1,
        iterations=300,
        max_iterations=300,
        initial_values={
            "initial:objective": np.asarray(1.0),
            "initial:objective_gradient": np.asarray([1.0]),
            "initial:parameters": np.asarray([0.0]),
        },
        final_values={
            "final:objective": np.asarray(final_objective),
            "final:objective_gradient": np.asarray([1.0e-6]),
            "final:parameters": np.asarray([1.0]),
            "final:total_curve_length": np.asarray(total_curve_length),
        },
        length_target=18.0,
    )

    assert terminal.normalized_status == "failed"


def test_minimal_case_declares_the_official_endpoint_quality_band() -> None:
    """At ``native_default`` the verdict is the band, on a published observable.

    Upstream's own run ends at its 300-iteration cap, and so does every
    one-ulp perturbation of it, so the end point is a point in upstream's own
    scatter rather than a converged minimum.  The band states that, and it
    states it about ``final:objective``, which both lanes publish; a band
    whose observable no lane published would be unenforceable.
    """
    case = get_case("native-stage-two-optimization-minimal")
    band = case.quality_band("native_default")

    assert band is not None
    assert band.observable == "final:objective"
    assert band.max_value > OFFICIAL_FINAL_OBJECTIVE
    assert case.work_budget_contract is None


@pytest.mark.parametrize("scale", ("native_default", "bounded"))
def test_minimal_solver_policy_keeps_official_tolerance_in_each_scale(
    scale: ExecutionScale,
) -> None:
    configuration = _scale_configuration(scale)

    assert configuration["rtol"] == 1.0e-15
    assert configuration["atol"] == 1.0e-15
    assert configuration["lbfgs_history_size"] == 300
    # The reduced scale shrinks the geometry, not the optimizer policy: the
    # official call is options={'maxiter': 300, 'maxcor': 300}, tol=1e-15
    # (upstream examples/1_Simple/stage_two_optimization_minimal.py:56,137),
    # and its 50-iteration value belongs to the CI switch, not to this scale.
    assert configuration["max_steps"] == OFFICIAL_ITERATION_LIMIT


def test_minimal_native_default_keeps_official_iteration_budget() -> None:
    configuration = _scale_configuration("native_default")

    assert (
        configuration["max_steps"]
        == MINIMAL_STAGE_TWO_NATIVE_ITERATIONS
        == OFFICIAL_ITERATION_LIMIT
    )


@pytest.mark.parametrize(
    (
        "status_convention",
        "provider_success",
        "provider_status",
        "iterations",
        "expected_status",
    ),
    [
        ("scipy-lbfgsb", True, 0, 76, "converged"),
        ("scipy-lbfgsb", False, 1, 300, "budget_exhausted"),
        ("scipy-lbfgsb", False, 2, 76, "failed"),
        ("private-lbfgsb", True, 0, 76, "converged"),
        ("private-lbfgsb", False, 1, 300, "budget_exhausted"),
        ("private-lbfgsb", False, 2, 76, "failed"),
    ],
)
def test_minimal_terminal_status_respects_provider_result(
    status_convention: StatusConvention,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    expected_status: str,
) -> None:
    initial_values = {
        "initial:objective": np.asarray(1.0),
        "initial:objective_gradient": np.asarray([1.0]),
        "initial:parameters": np.asarray([0.0]),
    }
    final_values = {
        "final:objective": np.asarray(0.5),
        "final:objective_gradient": np.asarray([1.0e-6]),
        "final:parameters": np.asarray([1.0]),
        "final:total_curve_length": np.asarray(18.0),
    }
    terminal = _terminal_status(
        status_convention=status_convention,
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=300,
        initial_values=initial_values,
        final_values=final_values,
        length_target=18.0,
    )

    assert terminal.normalized_status == expected_status
    assert terminal.success is (expected_status == "converged")


def test_native_minimal_producer_does_not_promote_provider_stops(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-minimal")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native_directory = tmp_path / "native"
    native_directory.mkdir()
    with chdir(native_directory):
        baseline = case.execute("native-cpu", bundle, arrays)
    # With the official parametrization and the official 300-iteration cap the
    # reduced geometry converges on its own; the promotions below must not be
    # able to turn a provider stop into that verdict.
    assert baseline.normalized_status == "converged"
    assert baseline.success is True
    assert baseline.nit is not None and baseline.nit < bundle.configuration["max_steps"]

    max_steps = bundle.configuration["max_steps"]
    length_target = bundle.configuration["length_target"]
    assert isinstance(max_steps, int)
    assert isinstance(length_target, (int, float))
    for provider_status, iterations, expected_status in (
        (1, max_steps, "budget_exhausted"),
        (2, baseline.nit, "failed"),
    ):
        result = scipy.optimize.OptimizeResult(
            x=baseline.values["final:parameters"],
            success=False,
            status=provider_status,
            nit=iterations,
            nfev=baseline.nfev,
            njev=baseline.njev,
        )

        def return_result(
            *_args: object,
            _result: scipy.optimize.OptimizeResult = result,
            **_kwargs: object,
        ) -> scipy.optimize.OptimizeResult:
            return _result

        monkeypatch.setattr(
            native_stage_two_optimization_minimal,
            "minimize",
            return_result,
        )
        with chdir(native_directory):
            observation = case.execute("native-cpu", bundle, arrays)

        np.testing.assert_array_equal(
            observation.values["final:parameters"],
            baseline.values["final:parameters"],
        )
        assert (
            observation.values["final:objective"]
            < observation.values["initial:objective"]
        )
        assert (
            np.linalg.norm(observation.values["final:objective_gradient"], ord=np.inf)
            <= 1.0e-4
        )
        assert observation.values["final:total_curve_length"] <= (1.1 * length_target)
        assert observation.normalized_status == expected_status
        assert observation.success is False
        assert observation.raw_status == str(provider_status)


def test_exact_stage_two_minimal_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-stage-two-optimization-minimal")
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

    # The mirror solves with the provider upstream calls; a parity lane that ran
    # the library's device L-BFGS-B would not be mirroring upstream's workflow,
    # so the published driver is asserted on both lanes.
    assert native.driver == jax.driver == Driver.SCIPY_LBFGSB.value

    # Both lanes satisfy the shared official 1e-15 stopping tolerance inside the
    # official 300-iteration cap, so both report their own convergence rather
    # than a stop imposed by a branch-chosen budget.
    assert native.success is True
    assert jax.success is True
    assert native.normalized_status == jax.normalized_status == "converged"
    assert native.nit is not None and native.nit < bundle.configuration["max_steps"]
    assert jax.nit is not None and jax.nit < bundle.configuration["max_steps"]
    assert native.scale == jax.scale == "bounded"
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
        "length_penalty",
        "maximum_normal_field",
        "total_curve_length",
    ):
        np.testing.assert_allclose(
            jax.values[f"initial:{observable}"],
            native.values[f"initial:{observable}"],
            rtol=1.0e-8,
            atol=1.0e-10,
        )

    for observation in (native, jax):
        length_target = bundle.configuration["length_target"]
        assert isinstance(length_target, (int, float))
        assert (
            observation.values["final:objective"]
            < observation.values["initial:objective"]
        )
        assert (
            np.linalg.norm(
                observation.values["final:objective_gradient"],
                ord=np.inf,
            )
            <= 1.0e-4
        )
        assert observation.values["final:total_curve_length"] <= (1.1 * length_target)

    np.testing.assert_allclose(
        jax.values["final:objective"],
        native.values["final:objective"],
        rtol=5.0e-3,
        atol=1.0e-10,
    )
