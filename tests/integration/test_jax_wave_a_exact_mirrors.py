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
class MirrorContract:
    path: str
    example_id: str
    observables: frozenset[str]


WAVE_A_CONTRACTS = (
    MirrorContract(
        "1_Simple/minimize_curve_length.py",
        "native-minimize-curve-length",
        frozenset(
            {
                "initial_parameters",
                "initial_length",
                "initial_gradient",
                "solution",
                "final_length",
                "final_gradient",
                "circle_oracle",
                "solver_success",
            }
        ),
    ),
    MirrorContract(
        "1_Simple/surf_vol_area.py",
        "native-surf-vol-area",
        frozenset(
            {
                "first_initial_residuals",
                "first_solution",
                "first_final_residuals",
                "second_initial_residuals",
                "second_solution",
                "second_final_residuals",
            }
        ),
    ),
    MirrorContract(
        "1_Simple/stage_two_optimization_minimal.py",
        "native-stage-two-optimization-minimal",
        frozenset(
            {
                "initial_parameters",
                "initial_objective",
                "initial_gradient",
                "solution",
                "final_objective",
                "final_gradient",
                "squared_flux",
                "curve_length_penalty",
                "solver_success",
            }
        ),
    ),
    MirrorContract(
        "1_Simple/qfm.py",
        "native-qfm",
        frozenset(
            {
                "initial_parameters",
                "initial_residuals",
                "initial_jacobian",
                "solution",
                "final_residuals",
                "constraint_residual",
                "solver_success",
            }
        ),
    ),
    MirrorContract(
        "1_Simple/permanent_magnet_simple.py",
        "native-permanent-magnet-simple",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "selected_dipoles",
                "nonzero_fraction",
                "solver_success",
            }
        ),
    ),
    MirrorContract(
        "2_Intermediate/wireframe_rcls_basic.py",
        "native-wireframe-rcls-basic",
        frozenset(
            {
                "initial_normal_error",
                "final_normal_error",
                "regularization_objective",
                "constraint_residual",
                "maximum_current",
                "solver_success",
            }
        ),
    ),
    MirrorContract(
        "2_Intermediate/strain_optimization.py",
        "native-strain-optimization",
        frozenset(
            {
                "initial_parameters",
                "initial_objective",
                "initial_gradient",
                "solution",
                "final_objective",
                "final_gradient",
                "maximum_strain",
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
    WAVE_A_CONTRACTS,
    ids=lambda contract: contract.example_id,
)
def test_wave_a_exact_mirror_executes_its_scientific_contract(
    contract: MirrorContract,
) -> None:
    example = REPO_ROOT / "examples" / "jax" / contract.path
    assert example.is_file(), f"missing exact-name mirror: {contract.path}"

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
