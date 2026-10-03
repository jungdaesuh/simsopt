from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import jax
import pytest
import simsoptpp
from examples.jax._lane_environment import build_execution_environment
from simsopt_jax.examples.solver_terminal_status import GSCO_MAXIMUM_ITERATION_REACHED

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Observables the two single-stage GSCO examples publish beyond the pre-wave
#: set, now that their completion comes from the solver's own stop condition
#: (``simsopt_jax.examples.solver_terminal_status``) instead of the scientific
#: predicate "final normal error below initial".
GSCO_SINGLE_STAGE_OBSERVABLES = frozenset(
    {
        "initial_normal_error",
        "final_normal_error",
        "maximum_current",
        "iterations",
        "allocated_iteration_budget",
        "constraints_satisfied",
        "terminal_status",
        "terminal_reason",
        "normal_error_decreased",
        "solver_success",
    }
)

#: What those two examples must report at the scale this test runs them
#: (``--smoke``): the measured count reaches the campaign's reduced cap, which
#: is upstream's ``stop_last_iter`` -- a budget stop, never a converged solve --
#: while the DEMOTED diagnostic ``normal_error_decreased`` is true. Pinning the
#: two together is what fails if ``final_error < initial_error`` becomes the
#: success gate again: it would publish ``solver_success`` true here.
GSCO_SINGLE_STAGE_COMPLETION = (
    ("terminal_status", "budget_exhausted"),
    ("terminal_reason", GSCO_MAXIMUM_ITERATION_REACHED),
    ("solver_success", False),
    ("normal_error_decreased", True),
    ("constraints_satisfied", True),
)


@dataclass(frozen=True)
class ExampleContract:
    path: str
    example_id: str
    observables: frozenset[str]
    #: Observable values the example must publish, exactly. A derived policy
    #: whose only witness is the example's own source can be reverted without
    #: any test noticing, so the value is pinned here (fix wave 4).
    required_values: tuple[tuple[str, str | bool], ...] = ()
    #: True when the example publishes both a MEASURED iteration count and the
    #: budget it was given, so ``terminal_status == "budget_exhausted"`` has to
    #: be that count reaching that budget and not a label chosen some other way.
    budget_stop_follows_the_measured_count: bool = False


EXTERNAL_SOLVER_FREE_CONTRACTS = (
    ExampleContract(
        "1_Simple/tracing_fieldlines_NCSX.py",
        "native-tracing-fieldlines-ncsx",
        frozenset(
            {
                "initial_states",
                "final_states",
                "poincare_hits",
                "interpolation_error",
                "integrator_status",
            }
        ),
    ),
    ExampleContract(
        "1_Simple/tracing_fieldlines_QA.py",
        "native-tracing-fieldlines-qa",
        frozenset(
            {
                "initial_states",
                "final_states",
                "poincare_hits",
                "integrator_status",
            }
        ),
    ),
    ExampleContract(
        "1_Simple/tracing_particle.py",
        "native-tracing-particle",
        frozenset(
            {
                "initial_state",
                "final_state",
                "energy_relative_error",
                "integrator_status",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/boozer.py",
        "native-boozer",
        frozenset(
            {
                "initial_residual",
                "final_residual",
                "iota",
                "volume",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/boozerQA.py",
        "native-boozerqa",
        frozenset(
            {
                "initial_residual",
                "final_residual",
                "iota",
                "volume",
                "solver_success",
                "solver_status",
                "solver_iterations",
                "outer_solver_success",
                "outer_solver_iteration_budget",
                "outer_stopping_reason",
                "inner_solver_success",
            }
        ),
        # Upstream's own run ends on its outer iteration budget, so the
        # example reports that budget exit by name and still publishes
        # ``status: ok``; it never relabels it converged.
        (
            ("outer_stopping_reason", "iteration-limit"),
            ("outer_solver_success", False),
            ("solver_success", False),
            ("inner_solver_success", True),
        ),
    ),
    ExampleContract(
        "2_Intermediate/permanent_magnet_MUSE.py",
        "native-permanent-magnet-muse",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "selected_dipoles",
                "moments",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/permanent_magnet_PM4Stell.py",
        "native-permanent-magnet-pm4stell",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "selected_dipoles",
                "moments",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/permanent_magnet_QA.py",
        "native-permanent-magnet-qa",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "selected_dipoles",
                "moments",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/stage_two_optimization.py",
        "native-stage-two-optimization",
        frozenset(
            {
                "initial_parameters",
                "initial_objective",
                "initial_gradient",
                "solution",
                "final_objective",
                "final_gradient",
                "squared_flux",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/stage_two_optimization_planar_coils.py",
        "native-stage-two-optimization-planar-coils",
        frozenset(
            {
                "initial_parameters",
                "initial_objective",
                "solution",
                "final_objective",
                "planarity_penalty",
                "squared_flux",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/stage_two_optimization_stochastic.py",
        "native-stage-two-optimization-stochastic",
        frozenset(
            {
                "sample_fingerprint",
                "initial_objective",
                "solution",
                "final_objective",
                "out_of_sample_objective",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/wireframe_gsco_modular.py",
        "native-wireframe-gsco-modular",
        GSCO_SINGLE_STAGE_OBSERVABLES,
        GSCO_SINGLE_STAGE_COMPLETION,
        True,
    ),
    ExampleContract(
        "2_Intermediate/wireframe_gsco_sector_saddle.py",
        "native-wireframe-gsco-sector-saddle",
        GSCO_SINGLE_STAGE_OBSERVABLES,
        GSCO_SINGLE_STAGE_COMPLETION,
        True,
    ),
    ExampleContract(
        "2_Intermediate/wireframe_rcls_with_ports.py",
        "native-wireframe-rcls-with-ports",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "constraint_residual",
                "port_clearance",
                "maximum_current",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "3_Advanced/coil_forces.py",
        "native-coil-forces",
        frozenset(
            {
                "force_objective",
                "maximum_force",
                "final_gradient",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "3_Advanced/stage_two_optimization_finitebuild.py",
        "native-stage-two-optimization-finitebuild",
        frozenset(
            {
                "initial_objective",
                "solution",
                "final_objective",
                "squared_flux",
                "minimum_clearance",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "3_Advanced/wireframe_gsco_multistep.py",
        "native-wireframe-gsco-multistep",
        frozenset(
            {
                "stage_objectives",
                "final_normal_error",
                "maximum_current",
                "iterations",
                "solver_success",
            }
        ),
    ),
)


def _source_checkout_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu",
        "fast",
        os.environ,
        repo_root=REPO_ROOT,
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        (
            str(Path(simsoptpp.__file__).resolve().parent),
            str(REPO_ROOT / "src"),
            str(sysconfig.get_paths()["purelib"]),
            str(Path(jax.__file__).resolve().parents[1]),
        )
    )
    return environment


def assert_published_observables(
    contract: ExampleContract, observables: dict[str, object]
) -> None:
    """Check one published payload against its contract.

    Separate from the subprocess run so the same check can be replayed on a
    recorded payload (the old-behaviour proof of fix wave 4).
    """

    missing = contract.observables - observables.keys()
    assert missing == frozenset(), (
        f"{contract.example_id} publishes no {sorted(missing)}"
    )
    for key, expected in contract.required_values:
        published = observables[key]
        assert type(published) is type(expected), (
            f"{contract.example_id}:{key} is a {type(published).__name__}, "
            f"not a {type(expected).__name__}"
        )
        assert published == expected, (
            f"{contract.example_id}:{key} = {published!r}, expected {expected!r}"
        )
    if contract.budget_stop_follows_the_measured_count:
        reached_the_cap = (
            observables["iterations"] == observables["allocated_iteration_budget"]
        )
        assert (observables["terminal_status"] == "budget_exhausted") is reached_the_cap


@pytest.mark.parametrize(
    "contract",
    EXTERNAL_SOLVER_FREE_CONTRACTS,
    ids=lambda contract: contract.example_id,
)
def test_external_solver_free_example_executes_its_scientific_contract(
    contract: ExampleContract,
) -> None:
    example = REPO_ROOT / "examples" / "jax" / contract.path
    assert example.is_file(), f"missing exact-name JAX example: {contract.path}"

    completed = subprocess.run(
        (sys.executable, "-S", str(example), "--smoke", "--json"),
        cwd=REPO_ROOT,
        env=_source_checkout_environment(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["example_id"] == contract.example_id
    assert payload["backend_mode"] == "jax_cpu_fast"
    assert payload["platform"] == "cpu"
    assert payload["precision"] == "fp64"
    assert payload["status"] == "ok"
    assert_published_observables(contract, payload["observables"])
