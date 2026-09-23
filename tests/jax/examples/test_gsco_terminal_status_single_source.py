"""The GSCO stop rule has ONE definition, and it labels upstream's own runs.

The three executable GSCO examples run as standalone scripts with no repository
root on ``sys.path`` (``tests/integration/test_jax_external_solver_free_examples.py``),
so the derivation lives in the installed package as
``simsopt_jax.examples.solver_terminal_status``; the parity harness module
``examples/jax/parity/cases/_fixed_work_status.py`` RE-EXPORTS it rather than
holding a copy. This file keeps that shape and pins what the rule does:

* the parity module binds the package's own objects (identity), and its source
  defines no GSCO rule of its own -- re-introducing a copy fails the ``ast``
  scan even if the copy is byte-identical, which a text comparison would allow;
* the package module really is where the family is defined, so the two scans
  cannot both be satisfied by an empty file;
* the labels are pinned against the OFFICIAL ``9e027eac3`` runs through the
  tracked fixture (``examples/jax/parity/official_reference``, decision 10), not
  against the other copy of the same code and not against pasted literals.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.cases import _fixed_work_status as parity
from examples.jax.parity.official_reference import (
    OfficialReference,
    load_official_reference,
)
from simsopt_jax.examples import solver_terminal_status as package

#: The module the parity harness must take the GSCO family from.
PACKAGE_MODULE = "simsopt_jax.examples.solver_terminal_status"

#: Exactly the names the parity harness re-exports. A name added here without a
#: matching import, or an import added there without a name here, fails.
RE_EXPORTED_NAMES = frozenset(
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

MULTISTEP = load_official_reference("native-wireframe-gsco-multistep")
MODULAR = load_official_reference("native-wireframe-gsco-modular")
SECTOR_SADDLE = load_official_reference("native-wireframe-gsco-sector-saddle")

#: Stages of the official multistep run, in order.
OFFICIAL_STAGES = tuple(range(1, int(MULTISTEP.scalar("steps_counted_by_script")) + 1))

#: Stages whose recorded loop/current history is stored element-exact. A longer
#: history is kept as a digest only (``INLINE_ELEMENT_LIMIT``), and the rule
#: cannot be replayed on a digest.
INLINE_HISTORY_STAGES = tuple(
    stage
    for stage in OFFICIAL_STAGES
    if MULTISTEP.is_inline(f"step{stage}:history:loops")
    and MULTISTEP.is_inline(f"step{stage}:history:currents")
)


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


def _parse(module_path: Path) -> ast.Module:
    return ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))


def _bound_names(module_path: Path) -> set[str]:
    """Every name the source BINDS by definition or assignment, at any depth."""

    names: set[str] = set()
    for node in ast.walk(_parse(module_path)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names |= _target_names(target)
        elif isinstance(node, ast.AnnAssign):
            names |= _target_names(node.target)
    return names


def _defined_names(module_path: Path) -> set[str]:
    return {
        node.name
        for node in ast.walk(_parse(module_path))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }


def _imported_from(module_path: Path, module: str) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_parse(module_path)):
        if isinstance(node, ast.ImportFrom) and node.module == module:
            names |= {alias.asname or alias.name for alias in node.names}
    return names


PARITY_SOURCE = Path(parity.__file__)
PACKAGE_SOURCE = Path(package.__file__)


def test_the_parity_module_imports_exactly_the_label_family() -> None:
    assert _imported_from(PARITY_SOURCE, PACKAGE_MODULE) == set(RE_EXPORTED_NAMES)


@pytest.mark.parametrize("name", sorted(RE_EXPORTED_NAMES))
def test_the_parity_name_is_the_package_object(name: str) -> None:
    assert getattr(parity, name) is getattr(package, name)


def test_the_parity_module_defines_no_gsco_rule_of_its_own() -> None:
    bound = _bound_names(PARITY_SOURCE)
    assert bound & RE_EXPORTED_NAMES == set()
    defined = _defined_names(PARITY_SOURCE)
    assert [name for name in sorted(defined) if name.startswith("gsco_")] == []
    assert "LaneTerminalLabel" not in defined


def test_the_package_module_defines_the_whole_family() -> None:
    assert RE_EXPORTED_NAMES <= _bound_names(PACKAGE_SOURCE)


def test_the_official_multistep_fixture_stores_six_of_seven_stage_histories() -> None:
    # step1 records 2168 iterations, above the fixture's inline element limit, so
    # its history is a digest; the other six are replayed element-exact below.
    assert OFFICIAL_STAGES == (1, 2, 3, 4, 5, 6, 7)
    assert INLINE_HISTORY_STAGES == (2, 3, 4, 5, 6, 7)


def _stage_history(stage: int) -> tuple[int, int, np.ndarray, np.ndarray]:
    iterations = int(MULTISTEP.scalar(f"step{stage}:iterations"))
    budget = int(MULTISTEP.scalar(f"step{stage}:allocated_iteration_budget"))
    loops = np.asarray(MULTISTEP.array(f"step{stage}:history:loops"), dtype=np.int64)
    currents = np.asarray(
        MULTISTEP.array(f"step{stage}:history:currents"), dtype=np.float64
    )
    return iterations, budget, loops, currents


@pytest.mark.parametrize("stage", INLINE_HISTORY_STAGES)
def test_an_official_stage_is_labelled_from_its_own_recorded_arrays(
    stage: int,
) -> None:
    iterations, budget, loops, currents = _stage_history(stage)
    assert iterations < budget
    # Upstream's own history ends in the accepted undo pair: the last recorded
    # loop repeats the previous one with the opposite current, which is
    # ``stop_undone_loop`` with the undo accepted.
    assert int(loops[-1]) == int(loops[-2])
    assert float(currents[-1]) * float(currents[-2]) < 0.0
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=iterations,
            max_iterations=budget,
            loop_history=loops,
            current_history=currents,
            endpoint_usable=True,
        )
    ) == ("converged", package.GSCO_MINIMUM_OBJECTIVE_REACHED, True)


@pytest.mark.parametrize("stage", INLINE_HISTORY_STAGES)
def test_the_same_stage_without_the_undo_pair_still_stops_itself(stage: int) -> None:
    iterations, budget, loops, currents = _stage_history(stage)
    # Counterfactual on upstream's arrays: the last accepted loop is a DIFFERENT
    # loop, so the undo pair is gone. The stop is still the solver's own.
    no_undo = loops.copy()
    no_undo[-1] = loops[-1] + 1
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=iterations,
            max_iterations=budget,
            loop_history=no_undo,
            current_history=currents,
            endpoint_usable=True,
        )
    ) == ("converged", package.GSCO_STOPPED_WITHOUT_ACCEPTING_A_LOOP, True)


@pytest.mark.parametrize("stage", INLINE_HISTORY_STAGES)
def test_the_same_stage_at_its_recorded_budget_is_never_converged(stage: int) -> None:
    _, budget, loops, currents = _stage_history(stage)
    no_undo = loops.copy()
    no_undo[-1] = loops[-1] + 1
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=budget,
            max_iterations=budget,
            loop_history=no_undo,
            current_history=currents,
            endpoint_usable=True,
        )
    ) == ("budget_exhausted", package.GSCO_MAXIMUM_ITERATION_REACHED, False)


@pytest.mark.parametrize("stage", INLINE_HISTORY_STAGES)
def test_a_stage_whose_endpoint_is_unusable_fails(stage: int) -> None:
    iterations, budget, loops, currents = _stage_history(stage)
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=iterations,
            max_iterations=budget,
            loop_history=loops,
            current_history=currents,
            endpoint_usable=False,
        )
    ) == ("failed", package.GSCO_MINIMUM_OBJECTIVE_REACHED, False)


@pytest.mark.parametrize("stage", INLINE_HISTORY_STAGES)
def test_a_stage_that_accepted_no_loop_is_degenerate(stage: int) -> None:
    _, budget, loops, currents = _stage_history(stage)
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=0,
            max_iterations=budget,
            loop_history=loops,
            current_history=currents,
            endpoint_usable=True,
        )
    ) == ("failed", package.GSCO_NO_ACCEPTED_UPDATE, False)


def _official_single_stage_counts(reference: OfficialReference) -> tuple[int, int]:
    return (
        int(reference.scalar("final:iterations")),
        int(reference.scalar("final:allocated_iteration_budget")),
    )


@pytest.mark.parametrize("reference", (MODULAR, SECTOR_SADDLE), ids=lambda r: r.case_id)
@pytest.mark.parametrize("undo_pair", (True, False))
def test_the_official_single_stage_run_stopped_itself(
    reference: OfficialReference, undo_pair: bool
) -> None:
    iterations, budget = _official_single_stage_counts(reference)
    # The official run stops on its own rule well below the cap it was given, so
    # whichever of the two early stops ended it, the label is ``converged``. Its
    # history is above the fixture's inline limit (a digest only), so both shapes
    # the digest admits are checked.
    assert 0 < iterations < budget
    assert bool(reference.scalar("final:constraints_satisfied")) is True
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


@pytest.mark.parametrize("reference", (MODULAR, SECTOR_SADDLE), ids=lambda r: r.case_id)
def test_a_single_stage_run_at_the_official_cap_is_a_budget_stop(
    reference: OfficialReference,
) -> None:
    _, budget = _official_single_stage_counts(reference)
    assert _fields(
        package.gsco_terminal_label(
            accepted_updates=budget,
            max_iterations=budget,
            loop_history=np.asarray([0, 7, 9], dtype=np.int64),
            current_history=np.asarray([0.0, 1.0, 1.0], dtype=np.float64),
            endpoint_usable=True,
        )
    ) == ("budget_exhausted", package.GSCO_MAXIMUM_ITERATION_REACHED, False)


def _official_stage_counts() -> tuple[np.ndarray, np.ndarray]:
    iterations = np.asarray(
        [int(MULTISTEP.scalar(f"step{stage}:iterations")) for stage in OFFICIAL_STAGES],
        dtype=np.int64,
    )
    budget = np.asarray(
        [
            int(MULTISTEP.scalar(f"step{stage}:allocated_iteration_budget"))
            for stage in OFFICIAL_STAGES
        ],
        dtype=np.int64,
    )
    return iterations, budget


def test_the_official_multistep_run_is_labelled_converged() -> None:
    iterations, budget = _official_stage_counts()
    assert bool(np.all(iterations < budget))
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=iterations,
            stage_budget=budget,
            final_adjustment_run=bool(MULTISTEP.scalar("final:adjustment_run")),
            endpoint_usable=bool(MULTISTEP.scalar("final:constraints_satisfied")),
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("converged", package.GSCO_MULTISTEP_STABLE_CURRENTS, True)


def test_one_official_stage_raised_to_its_budget_is_never_converged() -> None:
    iterations, budget = _official_stage_counts()
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


def test_an_official_run_whose_first_stage_accepted_nothing_fails() -> None:
    iterations, budget = _official_stage_counts()
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


def test_an_official_run_without_its_final_adjustment_fails() -> None:
    iterations, budget = _official_stage_counts()
    assert _fields(
        package.gsco_multistep_terminal_label(
            stage_iterations=iterations,
            stage_budget=budget,
            final_adjustment_run=False,
            endpoint_usable=True,
            limit_raw_status=OUTER_GUARD_RAW_STATUS,
        )
    ) == ("failed", OUTER_GUARD_RAW_STATUS, False)
