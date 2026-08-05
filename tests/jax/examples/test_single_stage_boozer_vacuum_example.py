"""Source-owned contract for the VMEC-free Boozer single-stage pair."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from examples.jax._lane_environment import build_execution_environment

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

    jax = _module(JAX)
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "lbfgs_status_is_success"
        for node in ast.walk(jax)
    )
    for optimizer_name in ("minimize_bfgs_host_core", "minimize_lbfgs_host_core"):
        optimizer_call = next(
            node
            for node in ast.walk(jax)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == optimizer_name
        )
        gtol = next(
            keyword.value
            for keyword in optimizer_call.keywords
            if keyword.arg == "gtol"
        )
        assert isinstance(gtol, ast.Name)
        assert gtol.id == "OUTER_GRADIENT_TOLERANCE"


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

    assert "make_traceable_objective_session" in imported_names
    assert not any("vmec" in name.lower() for name in imported_modules)
    assert not any("scipy" in name.lower() for name in imported_modules)


def test_jax_vacuum_single_stage_uses_fast_parity_solver_policy() -> None:
    module = _module(JAX)
    imported_names = {
        alias.name
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert "scalar_example_driver" in imported_names
    assert "minimize_bfgs_host_core" in imported_names
    assert "minimize_lbfgs_host_core" in imported_names


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

    assert "certify_optimization_endpoint" in imported_names
    assert {
        "outer_stopping_reason",
        "initial_stationary",
        "terminal_stationary",
    } <= observable_keys


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


def test_jax_finalization_uses_one_controller_anchored_evaluation() -> None:
    module = _module(JAX)
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    anchored_calls = [
        call
        for call in ast.walk(solve)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "evaluate_candidate_from_anchor"
    ]

    assert len(anchored_calls) == 1
    anchored_call = anchored_calls[0]
    assert len(anchored_call.args) == 2
    assert isinstance(anchored_call.args[0], ast.Name)
    assert anchored_call.args[0].id == "solution"
    assert isinstance(anchored_call.args[1], ast.Attribute)
    assert isinstance(anchored_call.args[1].value, ast.Name)
    assert anchored_call.args[1].value.id == "incumbent_controller"
    assert anchored_call.args[1].attr == "current_inner_state"

    baseline_finalization_calls = [
        call
        for call in ast.walk(solve)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in {"forward_result", "host_value_and_gradient"}
    ]
    assert baseline_finalization_calls == []

    anchored_attributes = {
        node.attr
        for node in ast.walk(solve)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "final_evaluation"
    }
    assert {
        "forward_result",
        "gradient",
        "candidate_inner_state",
    } <= anchored_attributes


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
            "--smoke",
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
        expected_returncode=1,
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
        expected_returncode=1,
    )

    native_values = _observables(native)
    jax_values = _observables(jax_result)
    # Smoke is a bounded diagnostic/parity lane, never convergence evidence:
    # both examples fail closed on their two-step budget. Observable parity
    # below is the actual gate.
    assert native["status"] == "failed"
    assert jax_result["status"] == "failed"
    assert native_values["outer_stopping_reason"] == "iteration-limit"
    assert jax_values["outer_stopping_reason"] == "iteration-limit"
    assert jax_result["backend_mode"] == "jax_cpu_parity"
    assert jax_result["platform"] == "cpu"
    assert jax_result["precision"] == "fp64"
    np.testing.assert_allclose(
        jax_values["initial_objective"],
        native_values["initial_objective"],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
    np.testing.assert_allclose(
        jax_values["initial_gradient"],
        native_values["initial_gradient"],
        rtol=2.0e-9,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax_values["final_objective"],
        native_values["final_objective"],
        rtol=2.0e-8,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        jax_values["solution"],
        native_values["solution"],
        rtol=2.0e-8,
        atol=2.0e-10,
    )
    for observable in (
        "iota",
        "volume",
        "non_qs_ratio",
        "boozer_residual_rms",
    ):
        np.testing.assert_allclose(
            jax_values[observable],
            native_values[observable],
            rtol=2.0e-8,
            atol=2.0e-12,
        )
