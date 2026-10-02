"""Public tracing examples must publish adapter-owned terminal statuses.

The three ``1_Simple`` tracing scripts are executed for real at bounded scale and
their published observables are checked. (The scripts live under
``examples/jax/1_Simple``, whose directory name is not a Python identifier, so
they cannot be imported; a subprocess is the only way to run them.)

One contract of the SHIPPED script cannot be reached behaviourally: the
``native_default`` branch passes ``max_steps=None``, the upstream horizon, and
running the script at that scale costs ~37 minutes (tmax 1e-2). That branch is
therefore asserted on the script's own source, which is the only lane that can
see the script start sending a finite budget where upstream has none.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import jax
import numpy as np
import pytest
import simsoptpp
from examples.jax._lane_environment import build_execution_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# Public scripts must reach the adapter through its PUBLIC status API only; the
# underscore-private helpers they used to import are gone.
PUBLIC_PRODUCERS = (
    ("tracing_fieldlines_QA.py", "compute_fieldlines_with_status"),
    ("tracing_fieldlines_NCSX.py", "compute_fieldlines_with_status"),
    ("tracing_particle.py", "trace_particles_with_status"),
)
# Which observable holds one row per traced lane, per script.
LANE_OBSERVABLE = {
    "tracing_fieldlines_QA.py": "initial_states",
    "tracing_fieldlines_NCSX.py": "initial_states",
    "tracing_particle.py": "initial_state",
}


def _source_tree(filename: str) -> tuple[Path, ast.Module]:
    path = REPOSITORY_ROOT / "examples" / "jax" / "1_Simple" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return path, tree


def _example_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu", "fast", os.environ, repo_root=REPOSITORY_ROOT
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(simsoptpp.__file__).resolve().parent),
            str(REPOSITORY_ROOT / "src"),
            str(sysconfig.get_paths()["purelib"]),
            str(Path(jax.__file__).resolve().parents[1]),
        )
    )
    return environment


def _run_example(filename: str) -> dict:
    script = REPOSITORY_ROOT / "examples" / "jax" / "1_Simple" / filename
    completed = subprocess.run(
        (sys.executable, "-S", str(script), "--smoke", "--json"),
        cwd=REPOSITORY_ROOT,
        env=_example_environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize(("filename", "helper"), PUBLIC_PRODUCERS)
def test_public_producer_publishes_the_core_terminal_status(
    filename: str, helper: str
) -> None:
    """The script publishes one core terminal status per traced lane.

    Fails if the observable disappears, stops being per-lane, stops using the
    core's vocabulary, or stops gating the example's own status -- none of which
    the previous AST-shape version of this test could see. For the two
    field-line scripts the bounded run stops one line on the level set and runs
    the others to ``tmax``, so the published values also have to be a MIX: a
    constant, or a status derived from the end time rather than taken from the
    core, cannot produce it.
    """
    del helper
    payload = _run_example(filename)
    observables = payload["observables"]
    statuses = observables["integrator_status"]
    lanes = observables[LANE_OBSERVABLE[filename]]

    assert isinstance(statuses, list)
    assert len(statuses) == len(lanes)
    assert all(isinstance(status, int) for status in statuses)
    # Core vocabulary: 0 reached tmax, -1 - i fired criterion i. A healthy
    # bounded run must not report a budget stop (1) or a step-control failure
    # (2), and the example's own status must follow.
    assert all(status <= 0 for status in statuses)
    assert payload["status"] == "ok"
    assert np.all(np.isfinite(np.asarray(lanes, dtype=np.float64)))
    if filename != "tracing_particle.py":
        assert 0 in statuses
        assert -1 in statuses


def test_public_scripts_import_no_private_adapter_names() -> None:
    for filename, _helper in PUBLIC_PRODUCERS:
        _path, tree = _source_tree(filename)
        imported = [
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("simsopt_jax")
            for alias in node.names
        ]
        assert imported
        assert [name for name in imported if name.startswith("_")] == []


def test_shipped_particle_script_sends_no_trial_budget_at_native_default() -> None:
    """``tracing_particle.py`` keeps upstream's "no trial limit" branch.

    Upstream loops ``do { ... } while(t < tmax && !stop);``
    (``legacy native extension/tracing.cpp``) -- there is no trial budget at all,
    so the shipped scale must forward ``None``. Asserted on the script because
    running it at ``native_default`` is not affordable; the bounded branch is
    covered behaviourally by ``test_public_producer_publishes_the_core_terminal_status``.
    """
    _path, tree = _source_tree("tracing_particle.py")
    solve = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    budget = next(
        node.value
        for node in ast.walk(solve)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "trace_max_steps"
            for target in node.targets
        )
    )
    assert isinstance(budget, ast.IfExp)
    assert isinstance(budget.body, ast.Constant)
    assert budget.body.value is None
    assert isinstance(budget.orelse, ast.Name)
    assert budget.orelse.id == "BOUNDED_MAX_STEPS"
    # ... and the value reaches the adapter under the documented keyword.
    forwarded = [
        keyword
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
        if keyword.arg == "max_steps"
    ]
    assert len(forwarded) == 1
    assert isinstance(forwarded[0].value, ast.Name)
    assert forwarded[0].value.id == "trace_max_steps"


def test_public_scripts_do_not_derive_the_status_or_ask_for_it_twice() -> None:
    """The status is the core's, taken once and not re-derived.

    Two guards on every shipped tracing script: it must not reconstruct a
    terminal status from the end time (``np.isclose(t, tmax)`` was the pattern
    that hid a budget stop as ``0``), and it must not pass ``collect_status=``,
    a private keyword the public status API replaced.
    """
    for filename, _helper in PUBLIC_PRODUCERS:
        _path, tree = _source_tree(filename)
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        assert not [
            keyword
            for node in calls
            for keyword in node.keywords
            if keyword.arg == "collect_status"
        ], f"{filename} calls the adapter with collect_status="
        assert not [
            node
            for node in calls
            if isinstance(node.func, ast.Attribute) and node.func.attr == "isclose"
        ], f"{filename} derives a value with isclose"
