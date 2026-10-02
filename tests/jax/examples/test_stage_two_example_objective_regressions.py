"""Stage-II example objectives: device-operand weights publish unchanged values.

The bounded problems of ``examples/jax/3_Advanced/stage_two_optimization_finitebuild.py``
and ``coil_forces.py`` are rebuilt here from the same public constructors and
constants (the scripts are entry points, not importable modules; their scalar
constants are read from their source and pinned below), and the shipped coil
forces run is executed as the runner executes it, as a fresh child process.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import itertools
import json
import os
import subprocess
import sys
import sysconfig
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
import simsoptpp
from examples.jax._lane_environment import build_execution_environment
from simsopt.field import (
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
    coils_via_symmetries,
)
from simsopt.geo import (
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    ForceStageTwoConfig,
    make_finite_build_stage_two_objective,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = REPO_ROOT / "examples" / "jax" / "3_Advanced"
TEST_DATA = REPO_ROOT / "tests" / "test_files"
# ``main()`` runs both examples with this budget in bounded mode.
_BOUNDED_STEPS = 3

#: The scripts' own constants, pinned against their source by the first test.
FINITE_BUILD_CONSTANTS = {
    "SOLVE_OBJECTIVE_SCALE": 1.0e-4,
    "NUM_BASE_CURVES": 4,
    "NUM_FILAMENTS_N": 2,
    "NUM_FILAMENTS_B": 3,
    "GAP_SIZE_N": 0.02,
    "GAP_SIZE_B": 0.04,
    "ROTATION_ORDER": 1,
}
COIL_FORCES_CONSTANTS = {
    "FIRST_LENGTH_WEIGHT": 1.0e-3,
    "SECOND_LENGTH_WEIGHT": 1.0e-4,
}
FIRST_LENGTH_WEIGHT = COIL_FORCES_CONSTANTS["FIRST_LENGTH_WEIGHT"]
SECOND_LENGTH_WEIGHT = COIL_FORCES_CONSTANTS["SECOND_LENGTH_WEIGHT"]


def _script_constants(name: str) -> dict[str, object]:
    path = EXAMPLES / f"{name}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    constants: dict[str, object] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        ):
            constants[node.targets[0].id] = node.value.value
    return constants


def _qa_surface() -> SurfaceRZFourier:
    return SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=4,
        ntheta=4,
    )


def _finite_build_problem():
    """``stage_two_optimization_finitebuild._build_problem("bounded")``."""
    constants = FINITE_BUILD_CONSTANTS
    count = int(constants["NUM_BASE_CURVES"])
    filaments_n = int(constants["NUM_FILAMENTS_N"])
    filaments_b = int(constants["NUM_FILAMENTS_B"])
    gap_n = float(constants["GAP_SIZE_N"])
    gap_b = float(constants["GAP_SIZE_B"])
    surface = _qa_surface()
    base_curves = create_equally_spaced_curves(
        count,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.7,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    filament_count = filaments_n * filaments_b
    base_currents = []
    for index in range(count):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (1.0e5 / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(
                curve,
                filaments_n,
                filaments_b,
                gap_n,
                gap_b,
                rotation_order=int(constants["ROTATION_ORDER"]),
            )
            for curve in base_curves
        )
    )
    filament_currents = list(
        itertools.chain.from_iterable(
            [current] * filament_count for current in base_currents
        )
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(
            apply_symmetries_to_curves(base_filaments, surface.nfp, True),
            apply_symmetries_to_currents(filament_currents, surface.nfp, True),
            strict=True,
        )
    ]
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    config = FiniteBuildStageTwoConfig(
        num_base_curves=count,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=filaments_n,
            numfilaments_b=filaments_b,
            gapsize_n=gap_n,
            gapsize_b=gap_b,
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=tuple(float(CurveLength(curve).J()) for curve in base_curves),
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    return field, flux, config


def _force_stage_config() -> StageTwoObjectiveConfig:
    """``coil_forces._stage_config()``."""
    return StageTwoObjectiveConfig(
        num_base_curves=3,
        length_target=17.4,
        length_target_mode="max",
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def _force_terms_config() -> ForceStageTwoConfig:
    """``coil_forces._force_config()``."""
    return ForceStageTwoConfig(
        num_force_coils=3,
        force_weight=1.0e-2,
        vacuum_energy_weight=1.0e-4,
        force_power=4.0,
        force_threshold=0.0,
        downsample=1,
    )


def _force_problem():
    """``coil_forces._build_problem("bounded")``, on the runtime policy's device."""
    surface = _qa_surface()
    base_curves = create_equally_spaced_curves(
        3,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=2,
        numquadpoints=8,
        use_jax_curve=False,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    regularization = 0.05**2 / np.sqrt(np.e)
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [regularization for _ in base_curves],
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    device = get_runtime_jax_device()
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)), device
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)), device
    )
    target_quadpoints = jax.device_put(
        np.stack(
            tuple(
                np.asarray(curve.quadpoints, dtype=np.float64) for curve in base_curves
            )
        ),
        device,
    )
    regularizations = jax.device_put(
        np.full(len(coils), regularization, dtype=np.float64), device
    )
    return (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    )


def _child_environment() -> dict[str, str]:
    _, environment = build_execution_environment(
        "cpu", "fast", os.environ, repo_root=REPO_ROOT
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


def test_rebuilt_problems_use_the_scripts_own_constants() -> None:
    finite_build = _script_constants("stage_two_optimization_finitebuild")
    coil_forces = _script_constants("coil_forces")

    for name, value in FINITE_BUILD_CONSTANTS.items():
        assert finite_build[name] == value, name
    for name, value in COIL_FORCES_CONSTANTS.items():
        assert coil_forces[name] == value, name


def _bitwise_equal(first: jax.Array, second: jax.Array) -> bool:
    return bool(
        np.array_equal(
            np.asarray(first, dtype=np.float64).view(np.uint64),
            np.asarray(second, dtype=np.float64).view(np.uint64),
        )
    )


def _device_scalar(value: float) -> jax.Array:
    return jax.device_put(np.asarray(value, dtype=np.float64))


def _finite_build_state() -> tuple[jax.Array, TraceableParametricScalarProblem, object]:
    field, flux, config = _finite_build_problem()
    objective = make_finite_build_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )
    parameters = jax.device_put(np.asarray(field.x, dtype=np.float64))
    problem = TraceableParametricScalarProblem(
        objective_fn=lambda current, objective_scale: (
            objective_scale * objective(current)
        ),
        objective_parameter=_device_scalar(
            float(FINITE_BUILD_CONSTANTS["SOLVE_OBJECTIVE_SCALE"])
        ),
        x=parameters,
    )
    return parameters, problem, objective


def _force_state():
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = _force_problem()
    stage_config = _force_stage_config()
    force_config = _force_terms_config()

    def build(config) -> object:
        return make_force_stage_two_objective(
            field,
            flux.traceable_objective(),
            surface_gamma,
            surface_normal,
            target_quadpoints,
            regularizations,
            config,
            force_config,
        )

    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    zero_weight_objective = build(stage_config)

    def weighted_objective(current: jax.Array, length_weight: jax.Array) -> jax.Array:
        return zero_weight_objective(current) + length_penalty(current, length_weight)

    parameters = jax.device_put(np.asarray(field.x, dtype=np.float64))
    return field, stage_config, build, weighted_objective, parameters


def test_finitebuild_publishes_the_fresh_unscaled_gradient_bit_for_bit() -> None:
    parameters, problem, objective = _finite_build_state()
    problem.value_and_grad(parameters)
    problem.set_objective_parameter(_device_scalar(1.0))
    published_value, published_gradient = problem.value_and_grad(parameters)

    reference = TraceableScalarProblem(objective, parameters)
    reference_value, reference_gradient = reference.value_and_grad(parameters)

    assert _bitwise_equal(published_value, reference_value)
    assert _bitwise_equal(published_gradient, reference_gradient)


def test_coil_forces_weighted_objective_is_bitwise_static_below_the_length_target() -> (
    None
):
    """The receipt guard: at the shipped start the split objective is bit-exact.

    The example's base curves start under the 17.4 m length target, so the
    penalty is exactly zero and ``objective(x) + 0.0`` reassociates nothing.
    That is the only regime where the refactored formulation can claim bitwise
    equality with the pre-change one, so the equality is asserted here and only
    here; the active regime is the sibling test below.
    """
    field, stage_config, build, weighted_objective, parameters = _force_state()
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (FIRST_LENGTH_WEIGHT, SECOND_LENGTH_WEIGHT):
        static_program = jax.jit(
            jax.value_and_grad(build(replace(stage_config, length_weight=weight)))
        )
        static_value, static_gradient = static_program(parameters)
        value, gradient = weighted_program(parameters, _device_scalar(weight))

        assert float(length_penalty(parameters, _device_scalar(weight))) == 0.0, (
            "the total base-curve length crossed the length target, so this "
            "point no longer certifies the bitwise claim"
        )
        assert _bitwise_equal(value, static_value)
        assert _bitwise_equal(gradient, static_gradient)


def test_coil_forces_weighted_objective_matches_each_static_length_weight() -> None:
    """Where the penalty is active the device weight must price it correctly.

    Below the length target the weight multiplies an exactly-zero excess, so a
    comparison there cannot fail on the weight at all; this loop runs at an
    extended point where each weight moves the objective.  The refactor
    reassociates the penalty sum, so the agreement is at the ~1 ULP level rather
    than bitwise.
    """
    field, stage_config, build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (FIRST_LENGTH_WEIGHT, SECOND_LENGTH_WEIGHT):
        static_program = jax.jit(
            jax.value_and_grad(build(replace(stage_config, length_weight=weight)))
        )
        static_value, static_gradient = static_program(extended)
        value, gradient = weighted_program(extended, _device_scalar(weight))

        assert float(length_penalty(extended, _device_scalar(weight))) > 0.0
        np.testing.assert_allclose(
            np.asarray(value),
            np.asarray(static_value),
            rtol=1.0e-14,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(gradient),
            np.asarray(static_gradient),
            rtol=1.0e-13,
            atol=0.0,
        )


def test_coil_forces_active_length_penalty_separates_the_two_stage_weights() -> None:
    """Where the penalty bites, the stage weight moves the objective it prices.

    The excess over the length target is read back from the penalty at the first
    weight, so the assertion is on the whole weighted objective: switching the
    device weight may only shift it by the length penalty it buys.
    """
    field, stage_config, _build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    first_weight = FIRST_LENGTH_WEIGHT
    second_weight = SECOND_LENGTH_WEIGHT
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(weighted_objective)
    first_value = float(weighted_program(extended, _device_scalar(first_weight)))
    second_value = float(weighted_program(extended, _device_scalar(second_weight)))
    excess_squared = (
        2.0 * float(length_penalty(extended, _device_scalar(first_weight)))
    ) / first_weight

    assert first_value != second_value, (
        "the length weight left the objective unchanged at a point where the "
        f"penalty is active (excess^2={excess_squared:.6e})"
    )
    np.testing.assert_allclose(
        first_value - second_value,
        0.5 * (first_weight - second_weight) * excess_squared,
        rtol=1.0e-9,
        atol=0.0,
    )


# The published values below are regression pins taken from one live bounded
# run on this branch, not independent oracles: they fail when the example's
# numbers move, and say nothing about whether the physics is right.
#
# Re-anchored 2026-09-20 from one bounded device-mode solve on this machine
# (CPU, fp64, OMP_NUM_THREADS=4).
#
# "No behaviour change" HOLDS ONLY AGAINST THE PRE-WAVE-3 SOURCE: the values
# below are bitwise equal to what that source publishes on this machine.
# Against the campaign baseline the device endpoint DID move, and the campaign
# change that moved it is FINITE_BUILD_LBFGS_HISTORY 10 -> 400 in
# src/simsopt_jax/examples/stage_two_finitebuild.py, still passed as
# ``lbfgs_history`` to the fused device solve this test drives (and as
# ``maxcor`` to the official SciPy route).  400 is upstream's own number
# (official 3_Advanced/stage_two_optimization_finitebuild.py line 173,
# ``'maxcor': 400``), so the change is a fidelity correction -- but it IS a
# device-lane behaviour change, and the certified finite-build device speed-up
# was measured at history 10.
#
def test_coil_forces_example_bounded_solve_lands_on_its_endpoint(tmp_path) -> None:
    """Both shipped stages run, the second after swapping the device length weight."""
    completed = subprocess.run(
        (
            sys.executable,
            "-S",
            str(EXAMPLES / "coil_forces.py"),
            "--smoke",
            "--json",
            "--output-dir",
            str(tmp_path),
        ),
        cwd=REPO_ROOT,
        env=_child_environment(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.splitlines()[-1])
    observables = result["observables"]

    assert result["status"] == "ok"
    assert observables["solver_iterations"] == [_BOUNDED_STEPS, _BOUNDED_STEPS]
    # Regression pin retaken 2026-09-13 after the mirror adopted native's
    # post-Taylor start state and SciPy L-BFGS-B at native policy (ftol=gtol=1e-15).
    np.testing.assert_allclose(
        observables["final_objective"],
        0.003424195336484035,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observables["squared_flux"],
        0.0033617810994737273,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        observables["vacuum_energy"],
        0.6148225859961913,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_force_stage_two_length_penalty_rejects_a_static_length_weight() -> None:
    field, stage_config, _build, _objective, _parameters = _force_state()

    with pytest.raises(ValueError, match="length_weight must be zero"):
        make_force_stage_two_length_penalty(
            field,
            replace(stage_config, length_weight=1.0e-3),
        )
