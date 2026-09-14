"""What one native-scale run of the force mirror publishes.

The native script runs its Taylor test through the same ``JF`` it optimizes, so
``dofs = JF.x`` on line 217 is the *last* Taylor evaluation, ``dofs0 - 1e-7*h``,
not the unperturbed coil set. These tests pin that start state and the mirror's
value/gradient agreement there against a native objective built in-process, and
pin the clock and device observables an artifact driver reads off the run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.force import B2Energy, LpCurveForce
from simsopt.field.selffield import regularization_circ
from simsopt.geo import (
    CurveCurveDistance,
    CurveLength,
    CurveSurfaceDistance,
    LpCurveCurvature,
    MeanSquaredCurvature,
    SurfaceRZFourier,
    create_equally_spaced_curves,
)
from simsopt.objectives import QuadraticPenalty, SquaredFlux, Weight
from simsopt_jax.examples import STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES

ROOT = Path(__file__).resolve().parents[3]
MIRROR = ROOT / "examples" / "jax" / "3_Advanced" / "coil_forces.py"
NATIVE_SURFACE = ROOT / "tests" / "test_files" / "input.LandremanPaul2021_QA"
TAYLOR_EPSILONS = (1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7)


def _native_objective():
    """``examples/3_Advanced/coil_forces.py`` lines 104-155, without the VTK output."""
    surface = SurfaceRZFourier.from_vmec_input(
        NATIVE_SURFACE,
        range="half period",
        nphi=32,
        ntheta=32,
    )
    surface.fix_all()
    base_curves = create_equally_spaced_curves(
        3,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=5,
        use_jax_curve=False,
    )
    base_currents = [Current(1e5) for _ in range(3)]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [regularization_circ(0.05) for _ in range(3)],
    )
    field = BiotSavart(coils)
    field.set_points(surface.gamma().reshape((-1, 3)))
    curves = [coil.curve for coil in coils]
    return (
        SquaredFlux(surface, field)
        + Weight(1e-03)
        * QuadraticPenalty(sum(CurveLength(c) for c in base_curves), 17.4, "max")
        + 1000 * CurveCurveDistance(curves, 0.1, num_basecurves=3)
        + 10 * CurveSurfaceDistance(curves, surface, 0.3)
        + 1e-6 * sum(LpCurveCurvature(c, 2, 5.0) for c in base_curves)
        + 1e-6
        * sum(
            QuadraticPenalty(MeanSquaredCurvature(c), 5, "max") for c in base_curves
        )
        + Weight(1e-2) * LpCurveForce(coils[:3], coils, p=4)
        + Weight(1e-4) * B2Energy(coils)
    )


@pytest.fixture(scope="module")
def native_post_taylor_state() -> tuple[np.ndarray, float, np.ndarray]:
    """Native's optimizer start plus its objective and gradient there."""
    objective = _native_objective()
    initial_dofs = np.asarray(objective.x, dtype=np.float64)
    np.random.seed(1)
    direction = np.random.uniform(size=initial_dofs.shape)
    for epsilon in TAYLOR_EPSILONS:
        objective.x = initial_dofs + epsilon * direction
        objective.J()
        objective.x = initial_dofs - epsilon * direction
        objective.J()
    start = np.asarray(objective.x, dtype=np.float64)
    return start, float(objective.J()), np.asarray(objective.dJ(), dtype=np.float64)


@pytest.fixture(scope="module")
def mirror_observables() -> dict[str, object]:
    """Published observables of one native-scale mirror run, truncated to one step."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(ROOT / "src"), str(ROOT), *(entry for entry in sys.path if entry))
    )
    completed = subprocess.run(
        (sys.executable, str(MIRROR), "--json", "--max-steps", "1"),
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.stdout, completed.stderr[-4000:]
    published = json.loads(completed.stdout.splitlines()[-1])
    observables = published["observables"]
    assert isinstance(observables, dict)
    return observables


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.native_cpu_reference
def test_mirror_starts_from_natives_post_taylor_state(
    native_post_taylor_state: tuple[np.ndarray, float, np.ndarray],
    mirror_observables: dict[str, object],
) -> None:
    native_start, _native_objective_value, _native_gradient = native_post_taylor_state
    mirror_start = np.asarray(mirror_observables["start_parameters"], dtype=np.float64)

    assert mirror_start.shape == native_start.shape
    assert np.array_equal(mirror_start, native_start), (
        "mirror start differs from native's post-Taylor start by up to "
        f"{np.max(np.abs(mirror_start - native_start)):.3e}; the mirror must "
        "replicate dofs0 - 1e-7*h, not the unperturbed coil set"
    )
    assert np.max(np.abs(mirror_start - native_start)) == 0.0


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.native_cpu_reference
def test_mirror_reproduces_native_value_and_gradient_at_the_matched_start(
    native_post_taylor_state: tuple[np.ndarray, float, np.ndarray],
    mirror_observables: dict[str, object],
) -> None:
    _native_start, native_value, native_gradient = native_post_taylor_state
    mirror_value = float(mirror_observables["start_objective"])
    mirror_gradient = np.asarray(mirror_observables["start_gradient"], dtype=np.float64)

    relative_value_error = abs(mirror_value - native_value) / abs(native_value)
    gradient_error = float(np.max(np.abs(mirror_gradient - native_gradient)))
    assert relative_value_error < 1.0e-14, (
        f"objective at the matched start differs relatively by "
        f"{relative_value_error:.3e} (mirror {mirror_value!r}, "
        f"native {native_value!r})"
    )
    assert gradient_error < 1.0e-13, (
        f"gradient at the matched start differs by {gradient_error:.3e} in the "
        "largest component; fixed-state evaluator parity has regressed"
    )


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.native_cpu_reference
def test_mirror_publishes_the_minimize_region_clocks_and_device_attestation(
    mirror_observables: dict[str, object],
) -> None:
    """An artifact driver can bracket the mirror's minimize region, not only solve()."""
    required = (
        "execution_device",
        "solver_options",
        "first_stage_minimize_seconds",
        "second_stage_minimize_seconds",
        "two_stage_minimize_seconds",
        "standard_solve_seconds",
    )
    missing = tuple(name for name in required if name not in mirror_observables)
    assert not missing, (
        f"{missing} absent from the mirror's observables; a driver reading "
        "STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES records null for them instead "
        "of failing, so the artifact silently loses its minimize region"
    )
    assert set(required) <= set(STANDARD_STAGE_TWO_OPTIMIZER_OBSERVABLES)

    device = mirror_observables["execution_device"]
    assert isinstance(device, str) and device, (
        "execution_device must name the device the endpoint array was solved on"
    )
    first = float(mirror_observables["first_stage_minimize_seconds"])
    second = float(mirror_observables["second_stage_minimize_seconds"])
    two_stage = float(mirror_observables["two_stage_minimize_seconds"])
    whole_solve = float(mirror_observables["standard_solve_seconds"])

    assert first > 0.0 and second > 0.0
    assert first <= whole_solve, (
        f"first stage minimize clock {first:.6f}s exceeds the whole solve() call "
        f"{whole_solve:.6f}s; the stage clock is not bracketing the minimize call"
    )
    assert second <= whole_solve, (
        f"second stage minimize clock {second:.6f}s exceeds the whole solve() call "
        f"{whole_solve:.6f}s; the stage clock is not bracketing the minimize call"
    )
    assert two_stage >= first + second, (
        f"two_stage_minimize_seconds {two_stage:.6f}s is shorter than its two "
        f"stage clocks ({first:.6f}s + {second:.6f}s); it is the region from "
        "entry of the first stage's minimize call to return of the second's, "
        "so it contains them both plus the inter-stage length-weight swap"
    )
    assert two_stage <= whole_solve
