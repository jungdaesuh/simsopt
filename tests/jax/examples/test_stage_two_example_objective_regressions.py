"""Stage-II example objectives: device-operand weights publish unchanged values."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import jax
import numpy as np
import pytest
from simsopt_jax.solve.serial import (
    TraceableParametricScalarProblem,
    TraceableScalarProblem,
)
from simsopt_jax_adapters.objectives import (
    make_finite_build_stage_two_objective,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "jax" / "3_Advanced"
# ``main()`` runs both examples with this budget in bounded mode.
_BOUNDED_STEPS = 3


def _example(name: str) -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        f"jax_example_{name}",
        EXAMPLES / f"{name}.py",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


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
    example = _example("stage_two_optimization_finitebuild")
    field, flux, config = example._build_problem("bounded")
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
        objective_parameter=_device_scalar(example.SOLVE_OBJECTIVE_SCALE),
        x=parameters,
    )
    return parameters, problem, objective


def _force_state():
    example = _example("coil_forces")
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = example._build_problem("bounded")
    stage_config = example._stage_config()
    force_config = example._force_config()

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
    return example, field, stage_config, build, weighted_objective, parameters


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
    example, field, stage_config, build, weighted_objective, parameters = _force_state()
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (example.FIRST_LENGTH_WEIGHT, example.SECOND_LENGTH_WEIGHT):
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
    example, field, stage_config, build, weighted_objective, parameters = _force_state()
    extended = parameters * 2.0
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)
    weighted_program = jax.jit(jax.value_and_grad(weighted_objective, argnums=0))

    for weight in (example.FIRST_LENGTH_WEIGHT, example.SECOND_LENGTH_WEIGHT):
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
    example, field, stage_config, _build, weighted_objective, parameters = (
        _force_state()
    )
    extended = parameters * 2.0
    first_weight = example.FIRST_LENGTH_WEIGHT
    second_weight = example.SECOND_LENGTH_WEIGHT
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
    example = _example("coil_forces")
    result = example.solve(tmp_path, _BOUNDED_STEPS, "bounded")

    assert result.status == "ok"
    assert result.observables["solver_iterations"] == (_BOUNDED_STEPS, _BOUNDED_STEPS)
    # Regression pin retaken 2026-09-13 after the mirror adopted native's
    # post-Taylor start state and SciPy L-BFGS-B at native policy (ftol=gtol=1e-15).
    np.testing.assert_allclose(
        result.observables["final_objective"],
        0.003424195336484035,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["squared_flux"],
        0.0033617810994737273,
        rtol=1.0e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.observables["vacuum_energy"],
        0.6148225859961913,
        rtol=1.0e-12,
        atol=0.0,
    )


def test_force_stage_two_length_penalty_rejects_a_static_length_weight() -> None:
    _example_module, field, stage_config, _build, _objective, _parameters = (
        _force_state()
    )

    with pytest.raises(ValueError, match="length_weight must be zero"):
        make_force_stage_two_length_penalty(
            field,
            replace(stage_config, length_weight=1.0e-3),
        )
