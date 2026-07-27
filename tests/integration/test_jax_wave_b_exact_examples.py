from __future__ import annotations

import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

import jax
import pytest
from examples.jax._lane_environment import build_execution_environment

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ExampleContract:
    path: str
    example_id: str
    observables: frozenset[str]


WAVE_B_CONTRACTS = (
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
            }
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
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "maximum_current",
                "iterations",
                "solver_success",
            }
        ),
    ),
    ExampleContract(
        "2_Intermediate/wireframe_gsco_sector_saddle.py",
        "native-wireframe-gsco-sector-saddle",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "maximum_current",
                "iterations",
                "solver_success",
            }
        ),
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
                "gradient",
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
            str(REPO_ROOT / "src"),
            str(sysconfig.get_paths()["purelib"]),
            str(Path(jax.__file__).resolve().parents[1]),
        )
    )
    return environment


@pytest.mark.parametrize(
    "contract",
    WAVE_B_CONTRACTS,
    ids=lambda contract: contract.example_id,
)
def test_wave_b_exact_example_executes_its_scientific_contract(
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
    assert contract.observables <= payload["observables"].keys()
