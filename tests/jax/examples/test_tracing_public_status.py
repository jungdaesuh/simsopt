"""Public tracing examples must publish adapter-owned terminal statuses.

The three ``1_Simple`` tracing scripts are executed for real at bounded scale and
their published observables are checked; the trial budget each scale declares is
checked by running the matching parity case with the adapter entry point
replaced by a recorder, which fails if the value is not forwarded. (The scripts
live under ``examples/jax/1_Simple``, whose directory name is not a Python
identifier, so they cannot be imported; a subprocess is the only way to run
them, and the parity case is the importable twin.)

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
from examples.jax.parity.cases import native_tracing_particle as particle_case
from examples.jax.parity.input_bundle import load_input_bundle

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
# Public scripts must reach the adapter through its PUBLIC status API only; the
# underscore-private helpers they used to import are gone (finding T-10).
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
    running it at ``native_default`` is not affordable; the bounded branch and
    the forwarding are covered behaviourally by
    ``test_particle_case_forwards_the_trial_budget_of_its_scale``.
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


def _recording_tracer(record: dict, statuses: np.ndarray):
    def tracer(_field, initial_points, parallel_speeds, **keywords):
        record["max_steps"] = keywords["max_steps"]
        record["tmax"] = keywords["tmax"]
        count = len(parallel_speeds)
        assert len(statuses) == count
        trajectories = [
            np.asarray(
                [
                    [0.0, *initial_points[index], parallel_speeds[index]],
                    [
                        keywords["tmax"],
                        *initial_points[index],
                        parallel_speeds[index],
                    ],
                ],
                dtype=np.float64,
            )
            for index in range(count)
        ]
        phi_hits = [np.zeros((0, 6), dtype=np.float64) for _ in range(count)]
        return trajectories, phi_hits, np.asarray(statuses, dtype=np.int64)

    return tracer


@pytest.mark.parametrize(
    ("scale", "expected_max_steps"), [("bounded", 512), ("native_default", None)]
)
def test_particle_case_forwards_the_trial_budget_of_its_scale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scale: str,
    expected_max_steps: int | None,
) -> None:
    """The declared budget reaches the adapter; ``native_default`` sends none.

    Upstream has no trial limit (``simsoptpp/tracing.cpp`` loops
    ``do { ... } while(t < tmax && !stop);``), so the shipped scale must forward
    ``None``. The recorder makes the assertion behavioural: it fails if the value
    is read but never passed on.
    """
    input_root = tmp_path / "inputs"
    bundle = particle_case.create_input(input_root, scale)
    _, arrays = load_input_bundle(input_root, bundle)
    count = len(arrays["parallel_speeds"])
    record: dict = {}
    monkeypatch.setattr(
        particle_case,
        "trace_particles_with_status",
        _recording_tracer(record, np.zeros(count, dtype=np.int64)),
    )
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")

    observation = particle_case.execute("jax-cpu", bundle, arrays)

    assert record["max_steps"] == expected_max_steps
    assert record["max_steps"] is None or isinstance(record["max_steps"], int)
    assert observation.values["final:status"].tolist() == [0] * count


def test_particle_case_publishes_the_positive_status_the_core_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step-control failure is published as-is and fails the lane.

    ``TRACING_STATUS_STEP_CONTROL_FAILED`` (2) is a terminal core status; the
    producer must neither swallow it nor relabel it as a criterion stop.
    """
    input_root = tmp_path / "inputs"
    bundle = particle_case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    count = len(arrays["parallel_speeds"])
    statuses = np.zeros(count, dtype=np.int64)
    statuses[0] = 2
    record: dict = {}
    monkeypatch.setattr(
        particle_case,
        "trace_particles_with_status",
        _recording_tracer(record, statuses),
    )
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")

    observation = particle_case.execute("jax-cpu", bundle, arrays)

    assert observation.values["final:status"].tolist() == statuses.tolist()
    assert observation.success is False
    assert observation.normalized_status == "failed"
    assert observation.raw_status == "integration_incomplete_or_failed"
