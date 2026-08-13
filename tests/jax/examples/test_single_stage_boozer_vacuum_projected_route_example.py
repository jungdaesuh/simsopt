"""Source-owned contract for the shipped projected-route single-stage example.

The example and the certification chain configure the SAME route, and the
configuration is spelled in both places -- the chain freezes it under
``benchmarks/`` where the plan document pins it, and an example that reached
into ``benchmarks/`` for its own numerics would have its layering backwards.
Twins drift, so the first test here is the one that makes this pair fail closed:
the example's options must equal the certified options field for field.

The rest of the file is the example's own contract -- that it reuses the
chain's bootstrap and the route's public kernel entry points rather than
re-deriving the problem, and that it actually runs.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import pytest
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    CERTIFIED_MAXIMUM_ITERATIONS,
    CERTIFIED_ROUTE_OPTIONS,
    CPU_BOOTSTRAP_OBSERVABLES,
    NATIVE_TARGET_OBJECTIVE,
)

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = (
    ROOT
    / "examples"
    / "jax"
    / "3_Advanced"
    / "single_stage_boozer_vacuum_projected_route.py"
)


def _example_module():
    """Import the shipped example the way a reader would run it."""

    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "projected_route_example", EXAMPLE
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _tree() -> ast.Module:
    return ast.parse(EXAMPLE.read_text(encoding="utf-8"))


def test_the_shipped_example_configures_the_certified_route() -> None:
    """Field for field, including the ones neither constructor names.

    The unnamed fields take optimizer defaults, so a default changed elsewhere
    in the repository would redefine one of the two configurations and not the
    other.  ``projector_tangency_tolerance`` is the field this matters most
    for: at zero the carried projector is never refreshed on tangency, which is
    the known, accepted mechanism behind the A100 no-latch arm.
    """

    module = _example_module()
    differing = [
        field.name
        for field in fields(CERTIFIED_ROUTE_OPTIONS)
        if getattr(module.ROUTE_OPTIONS, field.name)
        != getattr(CERTIFIED_ROUTE_OPTIONS, field.name)
    ]
    assert differing == []
    assert module.PROJECTED_NATIVE_ITERATIONS == CERTIFIED_MAXIMUM_ITERATIONS
    assert module.ROUTE_OPTIONS.objective_target == NATIVE_TARGET_OBJECTIVE


def test_the_example_reuses_the_route_and_the_chains_bootstrap() -> None:
    """No duplicated driver logic and no second spelling of the problem."""

    imported = {
        alias.name
        for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {
        "build_projected_lbfgs_kernels",
        "run_projected_lbfgs",
        "build_single_stage_fullspace_bootstrap",
        "evaluate_fullspace",
        "flatten_fullspace_constraints",
        "run_example",
    } <= imported


def test_the_example_imports_no_host_optimizer_and_no_vmec() -> None:
    """The constraints are enforced on device, not priced by a host solver."""

    modules = {
        node.module
        for node in ast.walk(_tree())
        if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {
        alias.name
        for node in ast.walk(_tree())
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any("scipy" in name.lower() for name in modules)
    assert not any("vmec" in name.lower() for name in modules)
    assert not any(name.startswith("benchmarks") for name in modules)


def test_the_example_reads_its_bounded_budget_from_the_cli_not_the_scale() -> None:
    """``--smoke`` shortens the run; it does not switch to a smaller problem.

    The coupled bootstrap has one scale -- the audited native-scale NCSX
    workload -- so a bounded lane here is a bounded BUDGET on the same problem,
    which is exactly what makes it a rehearsal of the certified run.
    """

    solve = next(
        node
        for node in _tree().body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    replaced = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "replace"
    )
    keyword = next(
        item for item in replaced.keywords if item.arg == "maximum_iterations"
    )
    assert isinstance(keyword.value, ast.Name)
    assert keyword.value.id == "max_steps"


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.single_stage
def test_the_example_runs_its_bounded_lane_feasibly(tmp_path: Path) -> None:
    """Launched as shipped: the entry path is executed, not imported.

    The bounded lane cannot converge and does not claim to; what it asserts is
    that the coupled problem is the audited one (its bootstrap objective is the
    campaign's), that every recorded iterate stayed on the constraint manifold,
    and that the objective never went up.
    """

    environment = {
        **os.environ,
        "JAX_PLATFORMS": "cpu",
        "JAX_ENABLE_X64": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT))),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        (
            sys.executable,
            "-B",
            str(EXAMPLE),
            "--smoke",
            "--json",
            "--output-dir",
            str(tmp_path / "output"),
        ),
        capture_output=True,
        check=False,
        cwd=ROOT,
        env=environment,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]

    assert result["status"] == "ok"
    assert observables["route"] == "projected-lagrangian-newton-cg"
    assert observables["iterations_run"] == 2
    assert observables["monotone_descent"] is True
    assert observables["joint_dof_count"] == 716
    assert observables["equality_count"] == 255
    assert "lagrangian_newton_direction" in observables["selected_kernels"]
    tolerance = observables["feasibility_tolerance"]
    assert observables["maximum_feasibility_inf"] <= tolerance
    assert observables["feasibility_inf"] <= tolerance
    reference = CPU_BOOTSTRAP_OBSERVABLES["objective"]
    assert abs(observables["initial_objective"] - reference) / reference <= 1.0e-10
