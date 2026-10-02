"""The GSCO stop rule has ONE definition, and it labels native SIMSOPT's own runs.

The three executable GSCO examples run as standalone scripts with no repository
root on ``sys.path`` (``tests/integration/test_jax_external_solver_free_examples.py``),
so the derivation lives in the installed package as
``simsopt_jax.examples.solver_terminal_status``. This file pins what the rule
does on the arrays native ``simsopt.solve.wireframe_optimization.gsco_wireframe``
really returns, not on pasted literals:

* the package module really is where the label family is defined;
* the source examples' workflows run natively, at reduced geometry but with the
  examples' own iteration budgets (2000 for the modular and sector-saddle
  examples, 2500 per stage and no outer-step limit for the multistep one), so
  every solve stops on its own rule, as upstream's runs do; the labels are
  derived from those returned histories and counts.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (
    SurfaceRZFourier,
    ToroidalWireframe,
    create_equally_spaced_curves,
)
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices, gsco_wireframe
from simsopt_jax.core.wireframe_workflow import WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY
from simsopt_jax.examples import solver_terminal_status as package

#: The label family every consumer takes from the package module.
LABEL_FAMILY_NAMES = frozenset(
    {
        "GSCO_MAXIMUM_ITERATION_REACHED",
        "GSCO_MINIMUM_OBJECTIVE_REACHED",
        "GSCO_MULTISTEP_NO_ACCEPTED_UPDATE",
        "GSCO_MULTISTEP_STABLE_CURRENTS",
        "GSCO_MULTISTEP_STAGE_BUDGET_REACHED",
        "GSCO_NO_ACCEPTED_UPDATE",
        "GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP",
        "LaneTerminalLabel",
        "gsco_multistep_terminal_label",
        "gsco_terminal_label",
    }
)

#: Raw status the multistep workflow reports when its outer guard ended the run.
OUTER_GUARD_RAW_STATUS = "outer_step_guard_exhausted_without_final_adjustment"

_TEST_FILES = Path(__file__).resolve().parents[2] / "test_files"
_SURFACE_INPUT = _TEST_FILES / "input.LandremanPaul2021_QA"
_WIREFRAME_INPUT = _TEST_FILES / "nescin.LandremanPaul2021_QA"
_MU0 = 4.0 * np.pi * 1.0e-7

#: Reduced geometry of the examples' ``--smoke`` run.
_PLASMA_RESOLUTION = 4
_SINGLE_STAGE_NPHI = 18
_SINGLE_STAGE_NTHETA = 8
_MULTISTEP_NPHI = 24
_MULTISTEP_NTHETA = 8
_MULTISTEP_MINIMUM_COIL_SIZE = 2

#: The source examples' own iteration budgets.
_SINGLE_STAGE_MAX_ITERATIONS = 2_000
_MULTISTEP_MAX_ITERATIONS_PER_STEP = 2_500

_MULTISTEP_TF_COILS = 3
_MULTISTEP_BREAK_WIDTH = 4
_MULTISTEP_INITIAL_CURRENT_FRACTION = 0.2
_MULTISTEP_LAMBDA_S = 1.0e-7

PACKAGE_SOURCE = Path(package.__file__)


def _fields(label: package.LaneTerminalLabel) -> tuple[str, str, bool]:
    return (label.normalized_status, label.raw_status, label.success)


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for element in target.elts:
            names |= _target_names(element)
        return names
    return set()


def _bound_names(module_path: Path) -> set[str]:
    """Every name the source BINDS by definition or assignment, at any depth."""

    names: set[str] = set()
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _target_names(node.target)
    return names


def _plasma() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(
        _SURFACE_INPUT,
        nphi=_PLASMA_RESOLUTION,
        ntheta=_PLASMA_RESOLUTION,
        range="half period",
    )


def _nescoil_wireframe(nphi: int, ntheta: int) -> ToroidalWireframe:
    surface = SurfaceRZFourier.from_nescoil_input(_WIREFRAME_INPUT, "current")
    return ToroidalWireframe(surface, nphi, ntheta)


@dataclass(frozen=True)
class _SingleStageRun:
    workflow: str
    iterations: int
    budget: int
    constraints_satisfied: bool


@dataclass(frozen=True)
class _Stage:
    iterations: int
    budget: int
    loops: np.ndarray
    currents: np.ndarray


@dataclass(frozen=True)
class _MultistepRun:
    stages: tuple[_Stage, ...]
    final_adjustment_run: bool
    constraints_satisfied: bool


def _single_stage_run(workflow: str) -> _SingleStageRun:
    plasma = _plasma()
    poloidal_current = float(-2.0 * np.pi * plasma.get_rc(0, 0) / _MU0)
    wireframe = _nescoil_wireframe(_SINGLE_STAGE_NPHI, _SINGLE_STAGE_NTHETA)
    if workflow == "modular":
        coil_current = poloidal_current / (2 * wireframe.nfp * 6)
        wireframe.add_tfcoil_currents(6, coil_current)
        lambda_s = 1.0e-6
        default_current = abs(coil_current)
    else:
        tf_current = poloidal_current / (2 * wireframe.nfp * 3)
        wireframe.add_tfcoil_currents(3, tf_current)
        wireframe.set_toroidal_breaks(3, 2, allow_pol_current=True)
        lambda_s = 10.0**-6.5
        default_current = abs(0.05 * poloidal_current)
    wireframe.set_poloidal_current(poloidal_current)
    response, target = bnorm_obj_matrices(
        wireframe, plasma, area_weighted=True, verbose=False
    )
    constrained = np.asarray(wireframe.constrained_segments(), dtype=np.int64)
    x, _, iter_history, *_ = gsco_wireframe(
        wireframe,
        response,
        target,
        lambda_s,
        True,
        False,
        default_current,
        1.1 * default_current,
        _SINGLE_STAGE_MAX_ITERATIONS,
        _SINGLE_STAGE_MAX_ITERATIONS,
        no_new_coils=False,
        max_loop_count=0,
        x_init=np.asarray(wireframe.currents, dtype=np.float64).reshape((-1, 1)),
        loop_count_init=np.zeros(
            len(wireframe.get_free_cells(form="logical")), dtype=np.int64
        ),
        verbose=False,
    )
    return _SingleStageRun(
        workflow=workflow,
        iterations=int(np.asarray(iter_history).reshape(-1)[-1]),
        budget=_SINGLE_STAGE_MAX_ITERATIONS,
        constraints_satisfied=bool(np.all(np.asarray(x).reshape(-1)[constrained] == 0)),
    )


def _multistep_wireframe() -> ToroidalWireframe:
    wireframe = _nescoil_wireframe(_MULTISTEP_NPHI, _MULTISTEP_NTHETA)
    wireframe.set_toroidal_breaks(
        _MULTISTEP_TF_COILS, _MULTISTEP_BREAK_WIDTH, allow_pol_current=True
    )
    wireframe.set_poloidal_current(0.0)
    return wireframe


def _coil_sizes(loop_count: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    coil_ids = np.full(loop_count.shape[0], -1, dtype=np.int64)
    coil_sizes = np.zeros(loop_count.shape[0], dtype=np.int64)
    coil_id = -1

    def count_connected(index: int, current_id: int) -> int:
        if loop_count[index] == 0 or coil_ids[index] >= 0:
            return 0
        coil_ids[index] = current_id
        return 1 + sum(
            count_connected(int(neighbor), current_id) for neighbor in neighbors[index]
        )

    for index in range(loop_count.shape[0]):
        if loop_count[index] != 0:
            coil_id += 1
            count = count_connected(index, coil_id)
            coil_sizes[coil_ids == coil_id] = count
    return coil_sizes


def _multistep_run() -> _MultistepRun:
    """The multistep source script's staged loop, natively, until it stabilizes."""
    plasma = _plasma()
    poloidal_current = float(-2.0 * np.pi * plasma.get_rc(0, 0) / _MU0)
    wireframe = _multistep_wireframe()
    curves = create_equally_spaced_curves(
        _MULTISTEP_TF_COILS, plasma.nfp, True, R0=1.0, R1=0.85
    )
    tf_currents = [
        Current(-poloidal_current / (2 * _MULTISTEP_TF_COILS * plasma.nfp))
        for _ in range(_MULTISTEP_TF_COILS)
    ]
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma,
        ext_field=BiotSavart(
            coils_via_symmetries(curves, tf_currents, plasma.nfp, True)
        ),
        area_weighted=True,
        verbose=False,
    )
    loops = np.asarray(wireframe.get_cell_key(), dtype=np.int32)
    neighbors = np.asarray(wireframe.get_cell_neighbors(), dtype=np.int32)
    current_scale = abs(poloidal_current)
    fraction = _MULTISTEP_INITIAL_CURRENT_FRACTION
    final_max_current = 1.1 * _MULTISTEP_INITIAL_CURRENT_FRACTION * current_scale
    x = np.zeros(wireframe.n_segments, dtype=np.float64)
    previous_x = np.zeros_like(x)
    loop_count = np.zeros(loops.shape[0], dtype=np.int64)
    enclosed = np.zeros(wireframe.n_segments, dtype=np.bool_)
    has_previous = False
    final_adjustment_run = False
    stages: list[_Stage] = []

    while len(stages) < WIREFRAME_GSCO_MULTISTEP_STAGE_CAPACITY:
        final_step = has_previous and np.array_equal(previous_x, x)
        stage_wireframe = _multistep_wireframe()
        if not final_step:
            stage_wireframe.set_segments_constrained(np.flatnonzero(enclosed))
        result = gsco_wireframe(
            stage_wireframe,
            response,
            target,
            _MULTISTEP_LAMBDA_S,
            True,
            final_step,
            0.0 if final_step else fraction * current_scale,
            final_max_current if final_step else 1.1 * fraction * current_scale,
            _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            _MULTISTEP_MAX_ITERATIONS_PER_STEP,
            no_new_coils=final_step,
            max_loop_count=1,
            x_init=x,
            loop_count_init=loop_count,
            verbose=False,
        )
        next_x = np.asarray(result[0], dtype=np.float64).reshape(-1)
        next_loop_count = np.asarray(result[1], dtype=np.int64)
        stages.append(
            _Stage(
                # One history entry per accepted update, holding its own index.
                iterations=int(np.asarray(result[2]).reshape(-1)[-1]),
                budget=_MULTISTEP_MAX_ITERATIONS_PER_STEP,
                currents=np.asarray(result[3], dtype=np.float64).reshape(-1),
                loops=np.asarray(result[4], dtype=np.int64).reshape(-1),
            )
        )
        if final_step:
            x = next_x
            final_adjustment_run = True
            break
        sizes = _coil_sizes(next_loop_count, neighbors)
        small = np.logical_and(sizes > 0, sizes < _MULTISTEP_MINIMUM_COIL_SIZE)
        pruned_x = next_x.copy()
        pruned_x[np.unique(loops[small].reshape(-1))] = 0.0
        pruned_loop_count = next_loop_count.copy()
        pruned_loop_count[small] = 0
        previous_x = x
        x = pruned_x
        loop_count = pruned_loop_count
        enclosed = np.zeros(wireframe.n_segments, dtype=np.bool_)
        enclosed[np.unique(loops[loop_count != 0].reshape(-1))] = True
        enclosed[x != 0.0] = False
        fraction *= 0.5
        has_previous = True

    return _MultistepRun(
        stages=tuple(stages),
        final_adjustment_run=final_adjustment_run,
        constraints_satisfied=bool(
            _multistep_wireframe().check_constraints(currents=x)
        ),
    )


def _ends_in_accepted_undo_pair(stage: _Stage) -> bool:
    return bool(
        stage.loops.shape[0] >= 3
        and int(stage.loops[-1]) == int(stage.loops[-2])
        and float(stage.currents[-1]) * float(stage.currents[-2]) < 0.0
    )


@pytest.fixture(scope="module")
def multistep_run() -> _MultistepRun:
    return _multistep_run()


@pytest.fixture(scope="module")
def undo_pair_stages(multistep_run: _MultistepRun) -> tuple[_Stage, ...]:
    """The native stages whose returned history ends in the accepted undo pair."""
    return tuple(
        stage for stage in multistep_run.stages if _ends_in_accepted_undo_pair(stage)
    )


@pytest.fixture(scope="module", params=("modular", "sector-saddle"))
def single_stage_run(request: pytest.FixtureRequest) -> _SingleStageRun:
    return _single_stage_run(request.param)


def test_the_package_module_defines_the_whole_family() -> None:
    assert LABEL_FAMILY_NAMES <= _bound_names(PACKAGE_SOURCE)


def test_the_native_multistep_run_stops_every_stage_itself(
    multistep_run: _MultistepRun, undo_pair_stages: tuple[_Stage, ...]
) -> None:
    # Every stage stops below its budget, and at least one stage ends in the
    # accepted undo pair, so the undo-pair replays below are not vacuous.
    assert len(multistep_run.stages) >= 2
    assert all(stage.iterations < stage.budget for stage in multistep_run.stages)
    assert len(undo_pair_stages) >= 1


def test_a_native_stage_is_labelled_from_its_own_returned_arrays(
    undo_pair_stages: tuple[_Stage, ...],
) -> None:
    for stage in undo_pair_stages:
        assert stage.iterations < stage.budget
        # The history ends in the accepted undo pair: the last recorded loop
        # repeats the previous one with the opposite current, which is
        # ``stop_undone_loop`` with the undo accepted.
        assert int(stage.loops[-1]) == int(stage.loops[-2])
        assert float(stage.currents[-1]) * float(stage.currents[-2]) < 0.0
        assert _fields(
            package.gsco_terminal_label(
                accepted_updates=stage.iterations,
                max_iterations=stage.budget,
                loop_history=stage.loops,
                current_history=stage.currents,
                endpoint_usable=True,
            )
        ) == ("converged", package.GSCO_MINIMUM_OBJECTIVE_REACHED, True)


def test_the_same_stage_without_the_undo_pair_still_stops_itself(
    multistep_run: _MultistepRun,
) -> None:
    for stage in multistep_run.stages:
        # Counterfactual on the returned arrays: the last accepted loop is a
        # DIFFERENT loop, so any undo pair is gone. The stop is still the
        # solver's own.
        no_undo = stage.loops.copy()
        no_undo[-1] = stage.loops[-1] + 1
        assert _fields(
            package.gsco_terminal_label(
                accepted_updates=stage.iterations,
                max_iterations=stage.budget,
                loop_history=no_undo,
                current_history=stage.currents,
                endpoint_usable=True,
            )
        ) == ("converged", package.GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP, True)


def test_the_same_stage_at_its_budget_is_never_converged(
    multistep_run: _MultistepRun,
) -> None:
    for stage in multistep_run.stages:
        no_undo = stage.loops.copy()
        no_undo[-1] = stage.loops[-1] + 1
        assert _fields(
            package.gsco_terminal_label(
                accepted_updates=stage.budget,
                max_iterations=stage.budget,
                loop_history=no_undo,
                current_history=stage.currents,
                endpoint_usable=True,
            )
        ) == ("budget_exhausted", package.GSCO_MAXIMUM_ITERATION_REACHED, False)


def test_a_stage_whose_endpoint_is_unusable_fails(
    undo_pair_stages: tuple[_Stage, ...],
) -> None:
    for stage in undo_pair_stages:
        assert _fields(
            package.gsco_terminal_label(
                accepted_updates=stage.iterations,
                max_iterations=stage.budget,
                loop_history=stage.loops,
                current_history=stage.currents,
                endpoint_usable=False,
            )
        ) == ("failed", package.GSCO_MINIMUM_OBJECTIVE_REACHED, False)


def test_a_stage_that_accepted_no_loop_is_degenerate(
    multistep_run: _MultistepRun,
) -> None:
    for stage in multistep_run.stages:
        assert _fields(
            package.gsco_terminal_label(
                accepted_updates=0,
                max_iterations=stage.budget,
                loop_history=stage.loops,
                current_history=stage.currents,
                endpoint_usable=True,
            )
        ) == ("failed", package.GSCO_NO_ACCEPTED_UPDATE, False)


@pytest.mark.parametrize("undo_pair", (True, False))
def test_the_native_single_stage_run_stopped_itself(
    single_stage_run: _SingleStageRun, undo_pair: bool
) -> None:
    iterations, budget = single_stage_run.iterations, single_stage_run.budget
    # The native run stops on its own rule well below the cap it was given, so
    # whichever of the two early stops ended it, the label is ``converged``;
    # both history shapes are checked.
    assert 0 < iterations < budget
    assert single_stage_run.constraints_satisfied is True
    loops = np.asarray([0, 7, 7] if undo_pair else [0, 7, 9], dtype=np.int64)
    currents = np.asarray([0.0, 1.0, -1.0], dtype=np.float64)
    expected_reason = (
        package.GSCO_MINIMUM_OBJECTIVE_REACHED
        if undo_pair
        else package.GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP
    )
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=iterations,
            max_iterations=budget,
            loop_history=loops,
            current_history=currents,
            endpoint_usable=True,
        )
    ) == ("converged", expected_reason, True)


def test_a_single_stage_run_at_its_cap_is_a_budget_stop(
    single_stage_run: _SingleStageRun,
) -> None:
    budget = single_stage_run.budget
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=budget,
            max_iterations=budget,
            loop_history=np.asarray([0, 7, 9], dtype=np.int64),
            current_history=np.asarray([0.0, 1.0, 1.0], dtype=np.float64),
            endpoint_usable=True,
        )
    ) == ("budget_exhausted", package.GSCO_MAXIMUM_ITERATION_REACHED, False)


def _stage_counts(run: _MultistepRun) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.asarray([stage.iterations for stage in run.stages], dtype=np.int64),
        np.asarray([stage.budget for stage in run.stages], dtype=np.int64),
    )


def test_the_native_multistep_run_is_labelled_converged(
    multistep_run: _MultistepRun,
) -> None:
    iterations, budget = _stage_counts(multistep_run)
    assert bool(np.all(iterations < budget))
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=iterations,
            stage_budget=budget,
            final_adjustment_run=multistep_run.final_adjustment_run,
            endpoint_usable=multistep_run.constraints_satisfied,
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("converged", package.GSCO_MULTISTEP_STABLE_CURRENTS, True)


def test_one_native_stage_raised_to_its_budget_is_never_converged(
    multistep_run: _MultistepRun,
) -> None:
    iterations, budget = _stage_counts(multistep_run)
    at_budget = iterations.copy()
    at_budget[-1] = budget[-1]
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=at_budget,
            stage_budget=budget,
            final_adjustment_run=True,
            endpoint_usable=True,
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("budget_exhausted", package.GSCO_MULTISTEP_STAGE_BUDGET_REACHED, False)


def test_a_native_run_whose_first_stage_accepted_nothing_fails(
    multistep_run: _MultistepRun,
) -> None:
    iterations, budget = _stage_counts(multistep_run)
    degenerate = iterations.copy()
    degenerate[0] = 0
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=degenerate,
            stage_budget=budget,
            final_adjustment_run=True,
            endpoint_usable=True,
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("failed", package.GSCO_MULTISTEP_NO_ACCEPTED_UPDATE, False)


def test_a_native_run_without_its_final_adjustment_fails(
    multistep_run: _MultistepRun,
) -> None:
    iterations, budget = _stage_counts(multistep_run)
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=iterations,
            stage_budget=budget,
            final_adjustment_run=False,
            endpoint_usable=True,
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("failed", OUTER_GUARD_RAW_STATUS, False)
