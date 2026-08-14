"""Source-owned contract for the shipped projected-route single-stage example.

The example and the certification chain configure the SAME route, and the
configuration is spelled in both places -- the chain freezes it under
``benchmarks/`` where the plan document pins it, and an example that reached
into ``benchmarks/`` for its own numerics would have its layering backwards.
Twins drift, so the first test here is the one that makes this pair fail closed:
the example's options must equal the certified options field for field.

The rest of the file is the example's own contract -- that it reuses the
chain's bootstrap and the route's public kernel entry points rather than
re-deriving the problem, that it actually runs, and that it tells the truth
about a run that did not reach the target.  ``LINE_SEARCH_COLLAPSE`` is a
stochastic draw, not a defect and not a success; the attempt protocol tested
here is what keeps it from being published as either.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import jax.numpy as jnp
import pytest
from benchmarks.rehearse_single_stage_projected_route_cpu import (
    CERTIFIED_MAXIMUM_ITERATIONS,
    CERTIFIED_ROUTE_OPTIONS,
    CPU_BOOTSTRAP_OBSERVABLES,
    NATIVE_TARGET_OBJECTIVE,
)
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.geo.optimizers.projected_lbfgs import (
    ProjectedLbfgsRun,
    ProjectedLbfgsStatus,
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


def _drawn_run(
    status: ProjectedLbfgsStatus, *, feasibility_inf: float = 1.0e-14
) -> ProjectedLbfgsRun:
    """A terminal run carrying nothing but the disposition under test.

    The protocol judges a draw on its terminal state, its worst feasibility
    and its objective sequence.  With no banked iterations the first two come
    from the run itself, which is what lets a scripted draw stand in for a
    solve that would cost two minutes to reproduce -- and what lets a draw be
    given a feasibility that misses the route's tolerance.
    """

    return ProjectedLbfgsRun(
        status=status,
        coordinates=jnp.zeros((2,)),
        objective=1.0e-6,
        feasibility_inf=feasibility_inf,
        projected_gradient_inf=1.0e-3,
        stored_pairs=0,
        iterations=(),
        compile_seconds=0.0,
        solve_seconds=0.0,
        projector_materializations=1,
        tangency_forced_refreshes=0,
        line_search_forced_refreshes=0,
    )


def _scripted(*runs: ProjectedLbfgsRun | ProjectedLbfgsStatus):
    """Hand out one pre-decided draw per call, and refuse an extra one."""

    remaining = [
        _drawn_run(item) if isinstance(item, ProjectedLbfgsStatus) else item
        for item in runs
    ]

    def draw() -> ProjectedLbfgsRun:
        assert remaining, "the protocol drew more attempts than were scripted"
        return remaining.pop(0)

    return draw


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
    assert observables["protocol_verdict"] == "ok"
    assert observables["attempt_budget"] == 3
    assert observables["attempts_spent"] == 1
    assert observables["attempt_terminal_states"] == ["ITERATION_LIMIT"]
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


def test_a_line_search_collapse_is_not_a_sound_stop() -> None:
    """The contract that flipped.

    The shipped script used to list ``LINE_SEARCH_COLLAPSE`` beside the two
    real stopping conditions, so a run that ran out of step scale short of the
    objective target published ``status: ok``.  It is now the one state the
    attempt budget exists to redraw, and it is sound in neither the tuple nor
    the judgement built on it.
    """

    module = _example_module()

    assert (
        ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE not in module._SOUND_TERMINAL_STATES
    )
    assert module._SOUND_TERMINAL_STATES == (
        ProjectedLbfgsStatus.ITERATION_LIMIT,
        ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED,
    )
    collapsed = module.judge_attempt(
        _drawn_run(ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE), module.ROUTE_OPTIONS
    )
    latched = module.judge_attempt(
        _drawn_run(ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED), module.ROUTE_OPTIONS
    )
    assert collapsed.sound is False
    assert latched.sound is True


def test_a_collapsed_draw_is_redrawn_and_the_first_sound_draw_ends_it() -> None:
    """One collapse costs one extra attempt, not the whole budget."""

    module = _example_module()

    attempts = module.draw_attempts(
        _scripted(
            ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE,
            ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED,
        ),
        module.ROUTE_OPTIONS,
    )

    assert [attempt.status for attempt in attempts] == [
        ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE,
        ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED,
    ]
    assert module.protocol_verdict(attempts) == module.VERDICT_OK
    assert module.published_status(module.VERDICT_OK) == "ok"


def test_every_draw_collapsing_publishes_retry_exhausted_and_exits_nonzero() -> None:
    """The budget is spent, the miss is named, and the process fails.

    The exit code is asserted through the shared entry point rather than
    reasoned about: ``run_example`` is what turns a published status into a
    return code, and a lesson that reported the miss but exited zero would
    still be green in the strict lane.
    """

    module = _example_module()

    attempts = module.draw_attempts(
        _scripted(
            *([ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE] * module.ATTEMPT_BUDGET)
        ),
        module.ROUTE_OPTIONS,
    )
    verdict = module.protocol_verdict(attempts)

    assert len(attempts) == module.ATTEMPT_BUDGET == 3
    assert verdict == module.VERDICT_RETRY_EXHAUSTED
    assert module.published_status(verdict) == "failed"
    assert (
        run_example(
            ["--smoke", "--json"],
            description=None,
            temporary_prefix="simsopt-jax-projected-route-contract-",
            bounded_steps=1,
            native_default_steps=1,
            solve=lambda _directory, _steps, _scale: ExampleResult(
                example_id=module.EXAMPLE_ID,
                observables={"protocol_verdict": verdict},
                status=module.published_status(verdict),
            ),
        )
        == 1
    )


def test_an_unsound_terminal_state_is_reported_without_being_redrawn() -> None:
    """A defect is reproducible, so redrawing it would only spend the budget."""

    module = _example_module()

    attempts = module.draw_attempts(
        _scripted(ProjectedLbfgsStatus.NONFINITE_STATE), module.ROUTE_OPTIONS
    )
    verdict = module.protocol_verdict(attempts)

    assert len(attempts) == 1
    assert verdict == module.VERDICT_UNSOUND
    assert module.published_status(verdict) == "failed"


def test_a_collapse_that_also_left_the_manifold_is_a_defect_not_a_draw() -> None:
    """The one overlap the two dispositions must not share.

    A collapse is a draw only when nothing ELSE about the attempt was wrong.
    Judging the redraw on the terminal state alone would let an attempt that
    left the constraint manifold be replaced by a later sound one and publish
    ``ok``, which is the single way this protocol could report a defect as a
    success -- so the infeasible collapse must stop the budget and name itself.
    """

    module = _example_module()
    tolerance = module.ROUTE_OPTIONS.feasibility_tolerance
    infeasible_collapse = _drawn_run(
        ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE, feasibility_inf=1.0e3 * tolerance
    )

    judged = module.judge_attempt(infeasible_collapse, module.ROUTE_OPTIONS)
    assert judged.sound is False
    assert judged.collapse_draw is False

    attempts = module.draw_attempts(
        _scripted(infeasible_collapse, ProjectedLbfgsStatus.OBJECTIVE_TARGET_REACHED),
        module.ROUTE_OPTIONS,
    )
    verdict = module.protocol_verdict(attempts)

    assert [attempt.status for attempt in attempts] == [
        ProjectedLbfgsStatus.LINE_SEARCH_COLLAPSE
    ]
    assert verdict == module.VERDICT_UNSOUND
    assert module.published_status(verdict) == "failed"


def test_the_module_docstring_states_the_measured_draw_behaviour() -> None:
    """The stochastic draw is documented where a reader of the script meets it.

    A reader who runs this lesson and gets ``retry_exhausted`` needs to know
    that the outcome is a draw with a measured rate, on which hardware it was
    measured, and that the two certified GPUs disagreed -- otherwise the honest
    failure reads as a broken example.
    """

    module = _example_module()
    docstring = module.__doc__ or ""

    assert "LINE_SEARCH_COLLAPSE" in docstring
    assert "A100" in docstring and "5090" in docstring
    assert "one attempt in five" in docstring
    for verdict in (
        module.VERDICT_OK,
        module.VERDICT_RETRY_EXHAUSTED,
        module.VERDICT_UNSOUND,
    ):
        assert verdict in docstring
