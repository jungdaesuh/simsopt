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
    assert JAX.is_file()


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
    assert {"forward_result", "gradient", "candidate_inner_state"} <= anchored_attributes


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
    # the JAX example fails closed on its two-step budget while native retains
    # the legacy lenient status. Observable parity below is the actual gate.
    assert native["status"] == "ok"
    assert jax_result["status"] == "failed"
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
