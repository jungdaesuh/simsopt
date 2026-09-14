"""Source-owned contract for the VMEC-free Boozer single-stage pair."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment
from simsopt.single_stage_boozer_vacuum import (
    SOUND_BOUNDED_STOPPING_REASONS,
    scientific_success_for_scale,
)
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    BOUNDED_SCALE,
    NATIVE_SCALE,
)

ROOT = Path(__file__).resolve().parents[3]
NATIVE = ROOT / "examples" / "3_Advanced" / "single_stage_boozer_vacuum_optimization.py"
CONTRACT = ROOT / "src" / "simsopt" / "single_stage_boozer_vacuum.py"
JAX = (
    ROOT
    / "examples"
    / "jax"
    / "3_Advanced"
    / "single_stage_boozer_vacuum_optimization.py"
)


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_native_and_jax_vacuum_single_stage_have_distinct_exact_paths() -> None:
    assert NATIVE.is_file()
    assert CONTRACT.is_file()
    assert JAX.is_file()


def test_native_and_jax_share_the_native_default_iteration_budget() -> None:
    contract = _module(CONTRACT)
    assignments = {
        node.target.id: node.value.value
        for node in contract.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, int)
    }
    assert assignments == {"NATIVE_ITERATIONS": 1000}

    for path in (NATIVE, JAX):
        module = _module(path)
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "simsopt.single_stage_boozer_vacuum"
            and any(alias.name == "NATIVE_ITERATIONS" for alias in node.names)
            for node in module.body
        )


def test_native_and_jax_share_the_outer_gradient_tolerance() -> None:
    contract = _module(CONTRACT)
    tolerance_assignments = {
        node.target.id: node.value.value
        for node in contract.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, float)
    }
    assert tolerance_assignments == {"OUTER_GRADIENT_TOLERANCE": 1.0e-15}

    for path in (NATIVE, JAX):
        module = _module(path)
        assert any(
            isinstance(node, ast.ImportFrom)
            and node.module == "simsopt.single_stage_boozer_vacuum"
            and any(alias.name == "OUTER_GRADIENT_TOLERANCE" for alias in node.names)
            for node in module.body
        )


def test_native_scale_is_selected_by_smoke_flag_not_iteration_budget() -> None:
    module = _module(NATIVE)
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    solve_arguments = {
        argument.arg
        for argument in (*solve.args.args, *solve.args.kwonlyargs)
        if argument.arg is not None
    }
    assert {"max_steps", "native_scale"} <= solve_arguments
    assert not any(
        isinstance(node, ast.Compare)
        and {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        >= {"max_steps", "NATIVE_ITERATIONS"}
        for node in ast.walk(solve)
    )

    main = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    solve_call = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "solve"
    )
    native_scale_keyword = next(
        keyword for keyword in solve_call.keywords if keyword.arg == "native_scale"
    )
    assert isinstance(native_scale_keyword.value, ast.UnaryOp)
    assert isinstance(native_scale_keyword.value.op, ast.Not)
    assert isinstance(native_scale_keyword.value.operand, ast.Attribute)
    assert isinstance(native_scale_keyword.value.operand.value, ast.Name)
    assert native_scale_keyword.value.operand.value.id == "options"
    assert native_scale_keyword.value.operand.attr == "smoke"


def test_all_single_stage_outer_optimizer_branches_use_shared_gtol() -> None:
    native = _module(NATIVE)
    native_minimize = next(
        node
        for node in ast.walk(native)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
    )
    native_options = next(
        keyword.value
        for keyword in native_minimize.keywords
        if keyword.arg == "options"
    )
    assert isinstance(native_options, ast.Dict)
    native_gtol = next(
        value
        for key, value in zip(native_options.keys, native_options.values)
        if isinstance(key, ast.Constant) and key.value == "gtol"
    )
    assert isinstance(native_gtol, ast.Name)
    assert native_gtol.id == "OUTER_GRADIENT_TOLERANCE"

    # The mirror runs the same SciPy BFGS, reached through the shared JAX
    # solve dispatch, so its budget and tolerance are the same two names.
    jax = _module(JAX)
    options_call = next(
        node
        for node in ast.walk(jax)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ScipyBFGSOptions"
    )
    jax_options = {
        keyword.arg: keyword.value
        for keyword in options_call.keywords
        if keyword.arg is not None
    }
    assert isinstance(jax_options["gtol"], ast.Name)
    assert jax_options["gtol"].id == "OUTER_GRADIENT_TOLERANCE"
    assert isinstance(jax_options["maxiter"], ast.Name)
    assert jax_options["maxiter"].id == "max_steps"
    native_maxiter = next(
        value
        for key, value in zip(native_options.keys, native_options.values)
        if isinstance(key, ast.Constant) and key.value == "maxiter"
    )
    assert isinstance(native_maxiter, ast.Name)
    assert native_maxiter.id == "max_steps"


def test_mirror_scale_constants_match_the_native_example_literals() -> None:
    """The two mirrors must size the same problem at each scale, or parity is void."""
    native = _module(NATIVE)
    native_solve = next(
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    # ``resolution = 6 if native_scale else 1`` and ``sDIM=20 if native_scale else 4``
    native_choices = [
        (node.body.value, node.orelse.value)
        for node in ast.walk(native_solve)
        if isinstance(node, ast.IfExp)
        and isinstance(node.body, ast.Constant)
        and isinstance(node.orelse, ast.Constant)
    ]
    # ``_configuration_options`` returns ``{}`` at native scale and the reduced
    # ``get_data("ncsx", ...)`` truncations otherwise.  ``ast.walk`` is breadth
    # first, so the two are collected without relying on source order.
    native_configuration_options = next(
        node
        for node in native.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configuration_options"
    )
    native_option_dicts = sorted(
        (
            ast.literal_eval(node.value)
            for node in ast.walk(native_configuration_options)
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        ),
        key=len,
    )
    assert native_option_dicts == [{}, dict(BOUNDED_SCALE.ncsx_options)]

    assert (NATIVE_SCALE.surface_resolution, BOUNDED_SCALE.surface_resolution) in (
        native_choices
    )
    assert (NATIVE_SCALE.non_qs_resolution, BOUNDED_SCALE.non_qs_resolution) in (
        native_choices
    )
    assert dict(NATIVE_SCALE.ncsx_options) == {}


def test_native_vacuum_single_stage_has_no_simsopt_jax_import() -> None:
    module = _module(NATIVE)
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert not any(
        module_name == "simsopt_jax" or module_name.startswith("simsopt_jax.")
        for module_name in imported_modules
    )


def test_jax_vacuum_single_stage_uses_decomposed_public_jax_kernels() -> None:
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    imported_modules = {
        node.module
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.Import)
        for alias in node.names
    }

    assert "SingleStageVacuumProblem" in imported_names
    assert not any("vmec" in name.lower() for name in imported_modules)
    # No VMEC, and no hand-rolled solver plumbing: the example reaches both the
    # problem and the outer driver through public library modules, and it never
    # imports SciPy itself (the host optimizer boundary is the shared dispatch).
    assert not any(
        name == "scipy" or name.startswith("scipy.") for name in imported_modules
    )
    assert "simsopt_jax.solve.dispatch" in imported_modules


def test_jax_vacuum_single_stage_runs_the_native_scipy_bfgs_policy() -> None:
    """The mirror's outer loop is native's algorithm, not a reimplementation."""
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert {"Driver", "ScipyBFGSOptions", "minimize"} <= imported_names
    # No mirror-private optimizer may survive alongside it.
    assert not {
        "scalar_example_driver",
        "minimize_bfgs_host_core",
        "minimize_lbfgs_host_core",
        "lbfgs_status_is_success",
    } & imported_names

    minimize_call = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "minimize"
    )
    driver = next(
        keyword.value for keyword in minimize_call.keywords if keyword.arg == "driver"
    )
    assert isinstance(driver, ast.Attribute)
    assert isinstance(driver.value, ast.Name)
    assert (driver.value.id, driver.attr) == ("Driver", "SCIPY_BFGS")
    # A callback would re-evaluate the objective per iteration and move the
    # stateful warm start, so the native trajectory requires none.
    assert not any(keyword.arg == "callback" for keyword in minimize_call.keywords)


def test_jax_vacuum_single_stage_uses_fail_closed_endpoint_certificate() -> None:
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    constants = {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "certify_optimization_endpoint" in imported_names
    assert {
        "outer_stopping_reason",
        "initial_stationary",
        "terminal_stationary",
    } <= observable_keys
    # Same optimizer, so the same status vocabulary as the native mirror.
    assert "scipy-bfgs" in constants


def test_native_vacuum_single_stage_uses_scipy_bfgs_endpoint_certificate() -> None:
    module = _module(NATIVE)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    constants = {
        node.value
        for node in ast.walk(module)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }

    assert "certify_optimization_endpoint" in imported_names
    assert {
        "outer_stopping_reason",
        "initial_stationary",
        "terminal_stationary",
    } <= observable_keys
    assert "scipy-bfgs" in constants


def test_jax_status_is_the_finite_improvement_gate_not_the_iteration_cap() -> None:
    """The mirror's ``status`` must not fold the optimizer's verdict into itself.

    At ``gtol`` 1.0e-15 the iteration cap is native's normal termination, so the
    cap may not decide ``status``.  The mirror therefore publishes the
    finite-improvement gate as ``status`` and leaves the optimizer verdict to
    its own observables -- which is why it no longer routes ``status`` through
    ``scientific_success_for_scale`` (the native reference still does; see
    ``test_native_mirror_keeps_the_scale_aware_soundness_convention``).
    """
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    assignment = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "scientific_success"
            for target in node.targets
        )
    )
    gate_source = ast.unparse(assignment.value)

    assert "scientific_success_for_scale" not in imported_names
    assert "accepted_step_contract" not in imported_names
    # The gate is finiteness plus improvement, and mentions no optimizer verdict.
    assert "endpoint.value < initial_objective" in gate_source
    assert "inner_success" in gate_source
    assert "parameters_finite" in gate_source
    assert "observables_finite" in gate_source
    for optimizer_verdict in (
        "optimizer_result",
        "endpoint_certificate",
        "stopping_reason",
        "stationary",
        "max_steps",
    ):
        assert optimizer_verdict not in gate_source, optimizer_verdict

    # The verdict is still published, and the certificate is still fail-closed.
    observable_keys = {
        key.value
        for node in ast.walk(module)
        if isinstance(node, ast.Dict)
        for key in node.keys
        if isinstance(key, ast.Constant) and isinstance(key.value, str)
    }
    assert {
        "outer_solver_success",
        "outer_stopping_reason",
        "solver_status",
        "initial_stationary",
        "terminal_stationary",
    } <= observable_keys
    assert "certify_optimization_endpoint" in imported_names


def test_native_mirror_keeps_the_scale_aware_soundness_convention() -> None:
    """The native reference's own gate is untouched by the mirror's change."""
    contract = _module(CONTRACT)
    tuple_assignments = {
        node.target.id: ast.literal_eval(node.value)
        for node in contract.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Tuple)
    }
    assert tuple_assignments["SOUND_BOUNDED_STOPPING_REASONS"] == (
        "converged",
        "iteration-limit",
    )
    assert SOUND_BOUNDED_STOPPING_REASONS == ("converged", "iteration-limit")
    assert any(
        isinstance(node, ast.FunctionDef)
        and node.name == "scientific_success_for_scale"
        for node in contract.body
    )

    # Structural drift guard: the native reference may not define the gate
    # function or bind the stopping-reason constant locally (annotated or
    # plain); it must reach the contract module's single copy by import.
    module = _module(NATIVE)
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "simsopt.single_stage_boozer_vacuum"
        and any(alias.name == "scientific_success_for_scale" for alias in node.names)
        for node in module.body
    )
    assert not any(
        isinstance(node, ast.FunctionDef)
        and node.name == "scientific_success_for_scale"
        for node in ast.walk(module)
    )
    bound_names = {
        node.id
        for node in ast.walk(module)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }
    assert "SOUND_BOUNDED_STOPPING_REASONS" not in bound_names
    assert "scientific_success_for_scale" not in bound_names


def test_scale_gate_certifies_convergence_only_at_native_default() -> None:
    """Only the status mapping is scale-aware; native_default stays fail-closed."""
    gate = scientific_success_for_scale
    sound = {
        "accepted_step": True,
        "inner_success": True,
        "parameters_finite": True,
        "observables_finite": True,
        "monotone_descent": True,
    }
    truncated = SimpleNamespace(success=False, stopping_reason="iteration-limit")
    converged = SimpleNamespace(success=True, stopping_reason="converged")

    # native_default keeps the full certificate: a truncated run is NOT success.
    assert gate(native_scale=True, certificate=converged, **sound) is True
    assert gate(native_scale=True, certificate=truncated, **sound) is False

    # bounded accepts a sound truncated run, and only a sound one.
    assert gate(native_scale=False, certificate=truncated, **sound) is True
    assert gate(native_scale=False, certificate=converged, **sound) is True
    for defect in ("line-search-failed", "nonfinite", "failed", "evaluation-limit"):
        assert (
            gate(
                native_scale=False,
                certificate=SimpleNamespace(success=False, stopping_reason=defect),
                **sound,
            )
            is False
        ), defect
    for unsound in (
        "accepted_step",
        "inner_success",
        "parameters_finite",
        "observables_finite",
        "monotone_descent",
    ):
        assert (
            gate(
                native_scale=False,
                certificate=truncated,
                **{**sound, unsound: False},
            )
            is False
        ), unsound


def test_jax_finalization_is_one_evaluation_of_the_returned_solution() -> None:
    """Native finalizes by evaluating once at the returned coils; so must the mirror."""
    module = _module(JAX)
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    endpoint_calls = [
        call
        for call in ast.walk(solve)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "problem"
        and call.func.attr == "endpoint"
    ]

    assert len(endpoint_calls) == 1
    (endpoint_call,) = endpoint_calls
    assert len(endpoint_call.args) == 1
    assert isinstance(endpoint_call.args[0], ast.Name)
    assert endpoint_call.args[0].id == "solution"

    # Every published endpoint number comes from that one evaluation.
    endpoint_attributes = {
        node.attr
        for node in ast.walk(solve)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "endpoint"
    }
    assert {
        "value",
        "gradient",
        "inner_success",
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual",
        "boozer_residual_rms",
    } <= endpoint_attributes


def test_both_single_stage_examples_report_implicit_physics_state() -> None:
    required = {
        "inner_solver_success",
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual",
        "gradient",
        "solution",
    }
    for path in (NATIVE, JAX):
        module = _module(path)
        observable_keys = {
            key.value
            for node in ast.walk(module)
            if isinstance(node, ast.Dict)
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        assert required <= observable_keys


def _run_json(
    path: Path,
    *,
    environment: dict[str, str],
    output_directory: Path,
    expected_returncode: int = 0,
    scale_arguments: tuple[str, ...] = ("--smoke",),
) -> dict[str, object]:
    environment = dict(environment)
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(ROOT / "src"),
            str(ROOT),
            *(entry for entry in sys.path if entry),
        )
    )
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(path),
            *scale_arguments,
            "--json",
            "--output-dir",
            str(output_directory),
        ),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == expected_returncode, completed.stderr[-2000:]
    result = json.loads(completed.stdout.splitlines()[-1])
    assert isinstance(result, dict)
    return result


def _relative_l2(measured: object, reference: object) -> float:
    measured_array = np.asarray(measured, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    return float(
        np.linalg.norm(measured_array - reference_array)
        / np.linalg.norm(reference_array)
    )


def _report_parity(
    jax_values: dict[str, object],
    native_values: dict[str, object],
) -> None:
    """Print the parity evidence in full precision, for the run log and receipts."""
    print("\nsingle-stage boozer vacuum, bounded scale, CPU parity")
    for name in ("initial_objective", "final_objective"):
        print(f"  {name}: jax={jax_values[name]!r} native={native_values[name]!r}")
    for name in ("initial_gradient", "gradient", "solution"):
        print(
            f"  {name}: rel_l2="
            f"{_relative_l2(jax_values[name], native_values[name]):.17e}"
        )


def _observables(result: dict[str, object]) -> dict[str, object]:
    observables = result["observables"]
    assert isinstance(observables, dict)
    return observables


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
@pytest.mark.native_cpu_reference
def test_public_vacuum_single_stage_matches_native_in_cpu_parity_mode(
    tmp_path: Path,
) -> None:
    native = _run_json(
        NATIVE,
        environment=dict(os.environ),
        output_directory=tmp_path / "native",
        expected_returncode=0,
    )
    _, jax_environment = build_execution_environment(
        "cpu",
        "parity",
        os.environ,
        repo_root=ROOT,
    )
    jax_result = _run_json(
        JAX,
        environment=jax_environment,
        output_directory=tmp_path / "jax",
        expected_returncode=0,
    )

    native_values = _observables(native)
    jax_values = _observables(jax_result)
    _report_parity(jax_values, native_values)
    # Smoke is a bounded diagnostic/parity lane, never convergence evidence.
    # Both examples spend their two-step budget and publish a SOUND truncated
    # run: status ok because nothing is defective, while the termination
    # evidence stays honest -- the stopping reason is still the exhausted
    # budget and neither endpoint claims stationarity.  Convergence remains
    # gated at native_default. Observable parity below is the actual gate.
    assert native["status"] == "ok"
    assert jax_result["status"] == "ok"
    assert native_values["outer_stopping_reason"] == "iteration-limit"
    assert jax_values["outer_stopping_reason"] == "iteration-limit"
    assert native_values["outer_solver_success"] is False
    assert jax_values["outer_solver_success"] is False
    assert native_values["terminal_stationary"] is False
    assert jax_values["terminal_stationary"] is False
    assert native_values["inner_solver_success"] is True
    assert jax_values["inner_solver_success"] is True
    assert native_values["final_objective"] <= native_values["initial_objective"]
    assert jax_values["final_objective"] <= jax_values["initial_objective"]
    assert jax_result["backend_mode"] == "jax_cpu_parity"
    assert jax_result["platform"] == "cpu"
    assert jax_result["precision"] == "fp64"
    # Both mirrors now run the same SciPy BFGS over the same objective with the
    # same failed-solve sentinel, so the whole trajectory matches, not just the
    # endpoint: equal iteration and evaluation counts are the evidence.
    assert jax_values["solver_iterations"] == native_values["solver_iterations"]
    assert jax_values["solver_evaluations"] == native_values["solver_evaluations"]
    assert jax_values["solver_status"] == native_values["solver_status"]
    np.testing.assert_allclose(
        jax_values["initial_objective"],
        native_values["initial_objective"],
        rtol=1.0e-13,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_values["initial_gradient"],
        native_values["initial_gradient"],
        rtol=1.0e-8,
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        jax_values["final_objective"],
        native_values["final_objective"],
        rtol=1.0e-10,
        atol=0.0,
    )
    np.testing.assert_allclose(
        jax_values["solution"],
        native_values["solution"],
        rtol=1.0e-8,
        atol=1.0e-11,
    )
    for observable in ("iota", "volume", "non_qs_ratio"):
        np.testing.assert_allclose(
            jax_values[observable],
            native_values[observable],
            rtol=1.0e-10,
            atol=1.0e-13,
        )
    # The exact solve drives the Boozer residual to numerical zero (rms ~5e-15
    # against a 1e-13 Newton tolerance), so only the absolute agreement of the
    # two rms values carries information; their ratio does not.
    np.testing.assert_allclose(
        jax_values["boozer_residual_rms"],
        native_values["boozer_residual_rms"],
        rtol=0.0,
        atol=1.0e-13,
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
def test_native_scale_budget_exhaustion_is_not_a_scientific_failure(
    tmp_path: Path,
) -> None:
    """Spending the outer budget at gtol 1.0e-15 must not fail the run.

    This is the case the previous scale-aware gate got wrong: at
    ``native_default`` it required ``certificate.success``, so an honest run
    that improved the objective and then hit its iteration cap -- native's
    normal termination, because 1.0e-15 sits below this objective's rounding
    floor -- was published as ``status: failed`` with exit 1.  One outer
    iteration at the native scale reproduces that endpoint in ~30 s.
    """
    _, jax_environment = build_execution_environment(
        "cpu",
        "parity",
        os.environ,
        repo_root=ROOT,
    )
    result = _run_json(
        JAX,
        environment=jax_environment,
        output_directory=tmp_path / "jax",
        expected_returncode=0,
        scale_arguments=("--max-steps", "1"),
    )
    values = _observables(result)

    assert result["scale"] == "native_default"
    assert result["status"] == "ok"
    # The optimizer verdict is published, honest, and separate from that status.
    assert values["outer_stopping_reason"] == "iteration-limit"
    assert values["outer_solver_success"] is False
    assert values["terminal_stationary"] is False
    assert values["solver_iterations"] == 1
    # ... and the status is exactly the finite-improvement claim.
    assert values["inner_solver_success"] is True
    assert values["final_objective"] < values["initial_objective"]
    assert np.all(np.isfinite(np.asarray(values["initial_gradient"], dtype=float)))
    assert np.all(np.isfinite(np.asarray(values["gradient"], dtype=float)))
