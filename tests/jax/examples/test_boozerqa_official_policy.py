"""Official BoozerQA method selection is independent of execution mode."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from conftest import ast_names_used
from examples.jax.parity.cases import native_boozerqa
from simsopt_contracts.optimization_endpoint import (
    OptimizationEndpointCertificate,
    certify_optimization_endpoint,
)
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS,
    OFFICIAL_QA_NEWTON_TOLERANCE,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_OUTER_DRIVER,
    OFFICIAL_QA_OUTER_MAXITER,
    OFFICIAL_QA_SURFACE_RESOLUTION,
)
from simsopt_jax.examples.single_stage_boozer_vacuum import OUTER_GRADIENT_TOLERANCE
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.geo import boozer_qa_problem


@pytest.mark.parametrize("mode_driver", (Driver.SIMSOPT_BFGS, Driver.SIMSOPT_LBFGSB))
def test_official_method_does_not_follow_fast_mode(
    monkeypatch: pytest.MonkeyPatch, mode_driver: Driver
) -> None:
    monkeypatch.setattr(native_boozerqa, "scalar_example_driver", lambda: mode_driver)
    assert native_boozerqa._outer_driver(native_boozerqa.BOOZER_QA_SPEC) == (
        Driver.SIMSOPT_BFGS
    )


def test_public_script_uses_official_bfgs_and_scale_specific_newton_tolerance() -> None:
    path = (
        Path(__file__).resolve().parents[3] / "examples/jax/2_Intermediate/boozerQA.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    names = [node.func.id for node in calls if isinstance(node.func, ast.Name)]
    assert "scalar_example_driver" not in names
    assert "minimize_lbfgs_host_core" not in names
    bfgs = [
        node
        for node in calls
        if isinstance(node.func, ast.Name) and node.func.id == "minimize_bfgs_host_core"
    ]
    assert len(bfgs) == 1
    gtol = next(keyword.value for keyword in bfgs[0].keywords if keyword.arg == "gtol")
    assert isinstance(gtol, ast.Name) and gtol.id == "OUTER_GRADIENT_TOLERANCE"
    assert OUTER_GRADIENT_TOLERANCE == 1.0e-15

    options = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_boozer_options"
    )
    returned = next(node for node in options.body if isinstance(node, ast.Return))
    assert isinstance(returned.value, ast.Dict)
    newton_tol = next(
        value
        for key, value in zip(returned.value.keys, returned.value.values, strict=True)
        if isinstance(key, ast.Constant) and key.value == "newton_tol"
    )
    assert isinstance(newton_tol, ast.IfExp)
    assert ast.unparse(newton_tol.test) == "scale == 'native_default'"
    # The official tolerance is read from its owner, not restated here; the
    # reduced-scale value is this branch's own and has no upstream counterpart.
    assert ast.unparse(newton_tol.body) == "OFFICIAL_QA_NEWTON_TOLERANCE"
    assert OFFICIAL_QA_NEWTON_TOLERANCE == 1.0e-13
    assert ast.literal_eval(newton_tol.orelse) == 1.0e-10


def test_official_settings_have_one_owner_shared_with_the_parity_case() -> None:
    """The shipped script and the matched case read the same constants."""
    assert native_boozerqa.BOOZER_QA_SPEC.native_resolution == (
        OFFICIAL_QA_SURFACE_RESOLUTION
    )
    assert native_boozerqa.BOOZER_QA_SPEC.native_inner_tolerance == (
        OFFICIAL_QA_NEWTON_TOLERANCE
    )
    assert native_boozerqa.BOOZER_QA_SPEC.native_outer_maxiter == (
        OFFICIAL_QA_OUTER_MAXITER
    )
    assert native_boozerqa.BOOZER_QA_SPEC.native_non_qs_sdim == (
        OFFICIAL_QA_NON_QS_RESOLUTION
    )
    assert native_boozerqa._outer_driver(native_boozerqa.BOOZER_QA_SPEC) == (
        OFFICIAL_QA_OUTER_DRIVER
    )

    path = (
        Path(__file__).resolve().parents[3] / "examples/jax/2_Intermediate/boozerQA.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simsopt_jax.examples.boozer_official"
        for alias in node.names
    }
    assert {
        "OFFICIAL_QA_INITIAL_IOTA",
        "OFFICIAL_QA_LINE_SEARCH_MAXITER",
        "OFFICIAL_QA_NEWTON_MAXITER",
        "OFFICIAL_QA_NEWTON_TOLERANCE",
        "OFFICIAL_QA_NON_QS_RESOLUTION",
        "OFFICIAL_QA_OUTER_MAXITER",
        "OFFICIAL_QA_SURFACE_DISTANCE",
        "OFFICIAL_QA_SURFACE_RESOLUTION",
    } <= imported
    # The outer objective dictionary itself is assembled once, by the owner of
    # the official workflow that both the script and the parity case run
    # (``simsopt_jax_adapters.geo.boozer_qa_problem``), from the same
    # ``boozer_qa_outer_objective_config``.  The script must reach it through
    # that owner and must not build a second copy.
    workflow_owner = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simsopt_jax_adapters.geo.boozer_qa_problem"
        for alias in node.names
    }
    assert "BoozerQAProblem" in workflow_owner
    assert not any(
        "boozer_qa_outer_objective_config" in name for name in ast_names_used(tree)
    )
    owner_source = Path(boozer_qa_problem.__file__).read_text(encoding="utf-8")
    owner_tree = ast.parse(owner_source)
    owner_imported = {
        alias.name
        for node in owner_tree.body
        if isinstance(node, ast.ImportFrom)
        and node.module == "simsopt_jax.examples.boozer_official"
        for alias in node.names
    }
    assert "boozer_qa_outer_objective_config" in owner_imported


def test_public_script_reports_the_official_budget_exit_explicitly() -> None:
    """``solver_success: false`` is published next to its stopping reason.

    Upstream's own run ends on its iteration budget (MAXITER = 1e3, tol = 1e-15;
    official capture final:outer_solver_status 1 after 1000 iterations), so the
    mirror must be able to say "budget exit" without calling it converged.
    """
    path = (
        Path(__file__).resolve().parents[3] / "examples/jax/2_Intermediate/boozerQA.py"
    )
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    observables = next(
        keyword.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ExampleResult"
        for keyword in node.keywords
        if keyword.arg == "observables"
    )
    assert isinstance(observables, ast.Dict)
    published = {
        key.value
        for key in observables.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "outer_solver_success",
        "outer_solver_iteration_budget",
        "outer_stopping_reason",
        "inner_solver_success",
    } <= published
    # One number, one name: the outer solver's status and iteration count are
    # already published as solver_status / solver_iterations, and a
    # budget-bound flag is a function of outer_stopping_reason.
    assert {"solver_status", "solver_iterations"} <= published
    removed = {"outer_solver_status", "outer_solver_iterations", "outer_budget_bound"}
    assert not removed & published
    # The stopping reason comes from the repository's single owner of the
    # emitter status vocabularies, not from a table restated in the script.
    assert "certify_optimization_endpoint" in source
    assert 'OUTER_STATUS_CONVENTION = "host-bfgs"' in source


#: The script's ``OUTER_STATUS_CONVENTION``, asserted against the source above;
#: it is written out here because a numbered example directory is not an
#: importable package.
_HOST_BFGS = "host-bfgs"


def _outer_endpoint(
    *,
    provider_success: bool,
    provider_status: int,
    iterations: int,
    observables_finite: bool = True,
) -> OptimizationEndpointCertificate:
    """Certify one outer endpoint exactly as the shipped script does."""
    return certify_optimization_endpoint(
        status_convention=_HOST_BFGS,
        provider_success=provider_success,
        provider_status=provider_status,
        iterations=iterations,
        max_iterations=OFFICIAL_QA_OUTER_MAXITER,
        initial_gradient_inf_norm=1.0,
        final_gradient_inf_norm=0.5,
        parameters_finite=True,
        observables_finite=observables_finite,
        inner_success=True,
    )


def test_only_the_official_budget_exit_and_convergence_are_admissible() -> None:
    """The endpoint certificate decides whether the run may report success.

    Upstream's BoozerQA ends on its 1000-iteration budget, so a budget exit
    stays admissible and ``solver_success`` false is not a failure.  A failed
    line search or a non-finite endpoint is a different mode, and the delta
    that only reported the stopping reason let those report ``ok``.
    """
    budget = _outer_endpoint(
        provider_success=False,
        provider_status=1,
        iterations=OFFICIAL_QA_OUTER_MAXITER,
    )
    assert budget.stopping_reason == "iteration-limit"
    assert budget.stopping_reason in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    # Upstream's own endpoint is not certified successful; admissible is not
    # the same claim as converged.
    assert budget.success is False

    converged = _outer_endpoint(
        provider_success=True,
        provider_status=0,
        iterations=12,
    )
    assert converged.stopping_reason == "converged"
    assert converged.stopping_reason in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS

    line_search = _outer_endpoint(
        provider_success=False,
        provider_status=2,
        iterations=3,
    )
    assert line_search.stopping_reason == "line-search-failed"
    assert (
        line_search.stopping_reason not in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    )

    nonfinite = _outer_endpoint(
        provider_success=False,
        provider_status=1,
        iterations=3,
        observables_finite=False,
    )
    assert nonfinite.stopping_reason == "nonfinite"
    assert (
        nonfinite.stopping_reason not in OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS
    )


def test_reported_status_is_gated_on_the_outer_endpoint_certificate() -> None:
    """``scientific_success`` consults the certificate, not only its string."""
    path = (
        Path(__file__).resolve().parents[3] / "examples/jax/2_Intermediate/boozerQA.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    admissible = ast.unparse(assignments["outer_endpoint_admissible"])
    assert "outer_certificate.stopping_reason" in admissible
    assert "OFFICIAL_QA_ADMISSIBLE_OUTER_STOPPING_REASONS" in admissible
    decision = {
        node.id
        for node in ast.walk(assignments["scientific_success"])
        if isinstance(node, ast.Name)
    }
    assert "outer_endpoint_admissible" in decision
