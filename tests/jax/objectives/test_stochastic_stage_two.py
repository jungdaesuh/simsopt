"""Pure-JAX stochastic Stage-II reduction contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.field import Coil, Current
from simsopt.geo import CurveXYZFourier, SurfaceRZFourier

from simsopt_jax.core import FixedSurfaceFluxSpec, make_fixed_surface_flux_spec
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    StochasticCoilPerturbations,
    make_stochastic_stage_two_objective,
    stage_two_coil_geometry,
    stochastic_flux_mean_from_geometry,
)
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import (
    SquaredFluxJAX,
    coil_current_fixed_geometry_flux_jax,
)


def _circle_geometry() -> tuple[jax.Array, jax.Array]:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 16, endpoint=False)
    gamma = jnp.stack(
        (jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)), axis=1
    )[None, :, :]
    gammadash = jnp.stack(
        (
            -2.0 * jnp.pi * jnp.sin(angles),
            2.0 * jnp.pi * jnp.cos(angles),
            jnp.zeros_like(angles),
        ),
        axis=1,
    )[None, :, :]
    return gamma, gammadash


def test_zero_perturbation_mean_matches_nominal_flux_through_scan() -> None:
    gamma, gammadash = _circle_geometry()
    currents = jnp.asarray((1.0e5,), dtype=jnp.float64)
    flux_spec = make_fixed_surface_flux_spec(
        points=jnp.asarray(((2.0, 0.0, 0.0),), dtype=jnp.float64),
        normal=jnp.asarray((((1.0, 0.0, 0.0),),), dtype=jnp.float64),
        target=jnp.zeros((1, 1), dtype=jnp.float64),
        definition="quadratic flux",
    )
    perturbations = StochasticCoilPerturbations(
        gamma=jnp.zeros((3, *gamma.shape), dtype=jnp.float64),
        gammadash=jnp.zeros((3, *gammadash.shape), dtype=jnp.float64),
    )

    stochastic = stochastic_flux_mean_from_geometry(
        gamma,
        gammadash,
        currents,
        flux_spec,
        perturbations,
    )
    nominal = coil_current_fixed_geometry_flux_jax(
        flux_spec.points,
        gamma,
        gammadash,
        currents,
        flux_spec,
    )
    jaxpr = jax.make_jaxpr(
        lambda current_gamma: stochastic_flux_mean_from_geometry(
            current_gamma,
            gammadash,
            currents,
            flux_spec,
            perturbations,
        )
    )(gamma)

    np.testing.assert_allclose(stochastic, nominal, rtol=1.0e-12, atol=1.0e-12)
    assert any(equation.primitive.name == "scan" for equation in jaxpr.jaxpr.eqns)


def test_zero_error_composed_objective_matches_deterministic_stage_two() -> None:
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    curve = CurveXYZFourier(16, 2)
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    field = BiotSavartJAX([Coil(curve, Current(1.0e5))])
    flux = SquaredFluxJAX(surface, field)
    zero_errors = StochasticCoilPerturbations(
        gamma=jnp.zeros((2, 1, 16, 3), dtype=jnp.float64),
        gammadash=jnp.zeros((2, 1, 16, 3), dtype=jnp.float64),
    )
    objective = make_stochastic_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        zero_errors,
        jnp.asarray(surface.gamma()).reshape((-1, 3)),
        jnp.asarray(surface.normal()).reshape((-1, 3)),
        StageTwoObjectiveConfig(num_base_curves=1),
    )
    parameters = jnp.asarray(field.x, dtype=jnp.float64)

    actual_value, actual_gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    expected_value, expected_gradient = jax.jit(
        jax.value_and_grad(flux.traceable_objective())
    )(parameters)

    np.testing.assert_allclose(actual_value, expected_value, rtol=1.0e-12, atol=1e-12)
    np.testing.assert_allclose(
        actual_gradient,
        expected_gradient,
        rtol=1.0e-11,
        atol=1.0e-12,
    )


_SAMPLE_COUNT = 8
_NATIVE_WORKFLOW_TOLERANCES = parity_ladder_tolerances("native_workflow")
_VALUE_RTOL = _NATIVE_WORKFLOW_TOLERANCES["same_state_value_rtol"]
_VALUE_ATOL = _NATIVE_WORKFLOW_TOLERANCES["same_state_value_atol"]
_DERIVATIVE_RTOL = _NATIVE_WORKFLOW_TOLERANCES["same_state_derivative_rtol"]
_DERIVATIVE_ATOL = _NATIVE_WORKFLOW_TOLERANCES["same_state_derivative_atol"]


class _BoundedProblem(NamedTuple):
    """Bounded-scale stochastic problem shared by the sample-tile tests."""

    gamma: jax.Array
    gammadash: jax.Array
    currents: jax.Array
    flux_spec: FixedSurfaceFluxSpec
    perturbations: StochasticCoilPerturbations


class _BuilderInputs(NamedTuple):
    """Composed-objective inputs shared by the builder-level tests."""

    field: BiotSavartJAX
    parameters: jax.Array
    surface_gamma: jax.Array
    surface_normal: jax.Array
    config: StageTwoObjectiveConfig


def _bounded_problem() -> _BoundedProblem:
    gamma, gammadash = _circle_geometry()
    grid = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64)
    flux_spec = make_fixed_surface_flux_spec(
        points=jnp.stack((2.0 + 0.1 * grid, 0.3 * grid, 0.2 * grid - 0.1), axis=1),
        normal=jnp.stack(
            (jnp.ones_like(grid), 0.2 * grid, -0.1 * grid), axis=1
        ).reshape((2, 3, 3)),
        target=jnp.zeros((2, 3), dtype=jnp.float64),
        definition="quadratic flux",
    )
    gamma_key, gammadash_key = jax.random.split(jax.random.key(20260823))
    return _BoundedProblem(
        gamma=gamma,
        gammadash=gammadash,
        currents=jnp.asarray((1.0e5,), dtype=jnp.float64),
        flux_spec=flux_spec,
        perturbations=StochasticCoilPerturbations(
            gamma=2.0e-2
            * jax.random.normal(
                gamma_key, (_SAMPLE_COUNT, *gamma.shape), dtype=jnp.float64
            ),
            gammadash=2.0e-2
            * jax.random.normal(
                gammadash_key, (_SAMPLE_COUNT, *gammadash.shape), dtype=jnp.float64
            ),
        ),
    )


def _builder_inputs() -> _BuilderInputs:
    curve = CurveXYZFourier(16, 2)
    curve.set("xc(0)", 1.0)
    curve.set("xc(1)", 0.3)
    curve.set("ys(1)", 0.3)
    field = BiotSavartJAX([Coil(curve, Current(1.0e5))])
    return _BuilderInputs(
        field=field,
        parameters=jnp.asarray(field.x, dtype=jnp.float64),
        surface_gamma=jnp.zeros((1, 3), dtype=jnp.float64),
        surface_normal=jnp.asarray(((1.0, 0.0, 0.0),), dtype=jnp.float64),
        config=StageTwoObjectiveConfig(num_base_curves=1),
    )


def _flux_mean(problem: _BoundedProblem, sample_tile: int | None) -> jax.Array:
    return stochastic_flux_mean_from_geometry(
        problem.gamma,
        problem.gammadash,
        problem.currents,
        problem.flux_spec,
        problem.perturbations,
        sample_tile=sample_tile,
    )


def _flux_mean_value_and_grad(
    problem: _BoundedProblem,
    sample_tile: int | None,
) -> tuple[jax.Array, jax.Array]:
    return jax.value_and_grad(
        lambda current_gamma: _flux_mean(
            problem._replace(gamma=current_gamma), sample_tile
        )
    )(problem.gamma)


def _max_abs_diff(actual: jax.Array, expected: jax.Array) -> float:
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))


def _scan_lengths(problem: _BoundedProblem, sample_tile: int | None) -> tuple[int, ...]:
    jaxpr = jax.make_jaxpr(
        lambda current_gamma: _flux_mean(
            problem._replace(gamma=current_gamma), sample_tile
        )
    )(problem.gamma)
    return tuple(
        equation.params["length"]
        for equation in jaxpr.jaxpr.eqns
        if equation.primitive.name == "scan"
    )


def test_default_sample_tile_is_the_sequential_scan_oracle() -> None:
    problem = _bounded_problem()

    implicit_default = stochastic_flux_mean_from_geometry(
        problem.gamma,
        problem.gammadash,
        problem.currents,
        problem.flux_spec,
        problem.perturbations,
    )
    explicit_none = _flux_mean(problem, None)

    np.testing.assert_array_equal(
        np.asarray(implicit_default), np.asarray(explicit_none)
    )
    assert _scan_lengths(problem, None) == (_SAMPLE_COUNT,)


def test_default_builder_matches_the_scan_path_bitwise() -> None:
    problem = _bounded_problem()
    inputs = _builder_inputs()
    objective = make_stochastic_stage_two_objective(
        inputs.field,
        problem.flux_spec,
        problem.perturbations,
        inputs.surface_gamma,
        inputs.surface_normal,
        inputs.config,
    )
    gamma, gammadash, _, currents = stage_two_coil_geometry(
        inputs.field.coil_dof_extraction_spec(),
        inputs.parameters,
    )

    builder_value = objective(inputs.parameters)
    scan_value = stochastic_flux_mean_from_geometry(
        gamma,
        gammadash,
        currents,
        problem.flux_spec,
        problem.perturbations,
    )

    np.testing.assert_array_equal(np.asarray(builder_value), np.asarray(scan_value))


def test_unit_sample_tile_matches_the_scan_bitwise() -> None:
    problem = _bounded_problem()

    oracle_value, oracle_gradient = _flux_mean_value_and_grad(problem, None)
    tiled_value, tiled_gradient = _flux_mean_value_and_grad(problem, 1)

    print(
        "[sample_tile parity] tile=1 "
        f"value max_abs_diff={_max_abs_diff(tiled_value, oracle_value):.6e} "
        f"gradient max_abs_diff={_max_abs_diff(tiled_gradient, oracle_gradient):.6e}"
    )
    np.testing.assert_array_equal(np.asarray(tiled_value), np.asarray(oracle_value))
    np.testing.assert_array_equal(
        np.asarray(tiled_gradient), np.asarray(oracle_gradient)
    )


@pytest.mark.parametrize("sample_tile", (1, 2, 4, _SAMPLE_COUNT))
def test_sample_tile_flux_mean_matches_scan_within_native_workflow_bucket(
    sample_tile: int,
) -> None:
    problem = _bounded_problem()

    oracle = _flux_mean(problem, None)
    tiled = _flux_mean(problem, sample_tile)
    max_abs_diff = _max_abs_diff(tiled, oracle)
    scan_lengths = _scan_lengths(problem, sample_tile)

    print(
        f"[sample_tile parity] tile={sample_tile} value max_abs_diff={max_abs_diff:.6e}"
    )
    assert scan_lengths == (_SAMPLE_COUNT // sample_tile,)
    assert np.isfinite(max_abs_diff)
    np.testing.assert_allclose(tiled, oracle, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)


@pytest.mark.parametrize("sample_tile", (1, 2, 4, _SAMPLE_COUNT))
def test_sample_tile_gradient_matches_scan_within_native_workflow_bucket(
    sample_tile: int,
) -> None:
    problem = _bounded_problem()

    oracle_value, oracle_gradient = _flux_mean_value_and_grad(problem, None)
    tiled_value, tiled_gradient = _flux_mean_value_and_grad(problem, sample_tile)
    value_max_abs_diff = _max_abs_diff(tiled_value, oracle_value)
    gradient_max_abs_diff = _max_abs_diff(tiled_gradient, oracle_gradient)

    print(
        f"[sample_tile parity] tile={sample_tile} "
        f"grad value max_abs_diff={value_max_abs_diff:.6e} "
        f"gradient max_abs_diff={gradient_max_abs_diff:.6e}"
    )
    assert np.isfinite(gradient_max_abs_diff)
    np.testing.assert_allclose(
        tiled_value, oracle_value, rtol=_VALUE_RTOL, atol=_VALUE_ATOL
    )
    np.testing.assert_allclose(
        tiled_gradient,
        oracle_gradient,
        rtol=_DERIVATIVE_RTOL,
        atol=_DERIVATIVE_ATOL,
    )


def test_sample_tile_not_dividing_the_sample_count_raises() -> None:
    problem = _bounded_problem()

    with pytest.raises(ValueError, match="positive divisor of the sample count"):
        _flux_mean(problem, 3)


@pytest.mark.parametrize(
    ("sample_tile", "expected_error", "match"),
    (
        (True, TypeError, "sample_tile must be a Python int; got bool"),
        (3, ValueError, "positive divisor of the sample count"),
        (0, ValueError, "sample_tile must be positive"),
        (-1, ValueError, "sample_tile must be positive"),
    ),
)
def test_builder_rejects_a_bad_sample_tile_at_build_time(
    sample_tile: int,
    expected_error: type[Exception],
    match: str,
) -> None:
    problem = _bounded_problem()
    inputs = _builder_inputs()

    with pytest.raises(expected_error, match=match):
        make_stochastic_stage_two_objective(
            inputs.field,
            problem.flux_spec,
            problem.perturbations,
            inputs.surface_gamma,
            inputs.surface_normal,
            inputs.config,
            sample_tile=sample_tile,
        )
