"""Objective factories validate host configs before any traced evaluation."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.field import Coil, Current, coils_via_symmetries
from simsopt.geo import create_multifilament_grid

from simsopt_jax.objectives.stage_two import (
    StageTwoObjectiveConfig,
    make_fused_stage_two_objective,
    make_stage_two_objective,
    make_stochastic_stage_two_objective,
    prepare_stage_two_config,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
)
from simsopt_jax.objectives.stochastic_stage_two import StochasticCoilPerturbations
from simsopt_jax.core.objectives_flux import fixed_surface_flux_integral
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.finite_build_stage_two import (
    FiniteBuildStageTwoConfig,
    _finite_build_geometry,
    _finite_build_penalties,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX
from simsopt_jax_adapters.objectives.force_stage_two import (
    ForceStageTwoConfig,
    _compiled_force_stage_two_metrics,
    force_stage_two_diagnostics,
)
from test_force_stage_two import _stage_two_force_case
from test_stage_two import _geometry


@pytest.fixture(scope="module")
def force_case():
    return _stage_two_force_case()


@pytest.mark.parametrize("factory", ("plain", "fused", "stochastic"))
def test_required_length_target_is_rejected_at_factory(force_case, factory):
    field, parameters, _, _, _, _, surface = force_case
    config = StageTwoObjectiveConfig(num_base_curves=2, individual_length_weight=1.0)
    sg = jax.device_put(surface.gamma().reshape((-1, 3)))
    sn = jax.device_put(surface.normal().reshape((-1, 3)))
    flux = SquaredFluxJAX(surface, field)
    spec = flux.fixed_surface_flux_spec()
    geometry = stage_two_coil_geometry(field.coil_dof_extraction_spec(), parameters)
    perturbations = StochasticCoilPerturbations(
        gamma=jnp.zeros_like(geometry[0])[None],
        gammadash=jnp.zeros_like(geometry[1])[None],
    )
    with pytest.raises(ValueError, match="individual_length_target is required"):
        if factory == "plain":
            make_stage_two_objective(field, lambda p: jnp.sum(p), sg, sn, config)
        elif factory == "fused":
            make_fused_stage_two_objective(field, spec, sg, sn, config)
        else:
            make_stochastic_stage_two_objective(
                field, spec, perturbations, sg, sn, config
            )


@pytest.mark.parametrize(
    "config, message",
    (
        (StageTwoObjectiveConfig(num_base_curves=0), "positive integer"),
        (StageTwoObjectiveConfig(num_base_curves=99), "available coils"),
        (
            StageTwoObjectiveConfig(num_base_curves=1, length_target_mode="bad"),
            "length_target_mode",
        ),
        (
            StageTwoObjectiveConfig(num_base_curves=1, curvature_weight=np.nan),
            "curvature_weight",
        ),
    ),
)
def test_stage_two_invalid_config_is_rejected_at_factory(force_case, config, message):
    field, _, _, _, _, _, surface = force_case
    with pytest.raises(ValueError, match=message):
        make_stage_two_objective(
            field,
            lambda p: jnp.sum(p),
            surface.gamma(),
            surface.normal(),
            config,
        )


@pytest.mark.parametrize(
    "config, message",
    (
        (ForceStageTwoConfig(num_force_coils=0), "nonempty subset"),
        (ForceStageTwoConfig(num_force_coils=99), "nonempty subset"),
        (ForceStageTwoConfig(num_force_coils=2, downsample=0), "downsample"),
        (ForceStageTwoConfig(num_force_coils=2, downsample=1.5), "downsample"),
        (ForceStageTwoConfig(num_force_coils=2, force_power=0.0), "force_power"),
        (
            ForceStageTwoConfig(num_force_coils=2, force_threshold=np.inf),
            "force_threshold",
        ),
    ),
)
def test_force_invalid_config_is_rejected_at_factory(force_case, config, message):
    field, _, qp, regs, _, _, _ = force_case
    with pytest.raises(ValueError, match=message):
        force_stage_two_diagnostics(field, qp, regs, config)


@pytest.mark.parametrize("operand", ("quadpoints", "regularizations"))
def test_force_operand_shapes_are_checked_at_factory(force_case, operand):
    field, _, qp, regs, _, _, _ = force_case
    with pytest.raises(ValueError, match="target_quadpoints|regularizations"):
        force_stage_two_diagnostics(
            field,
            qp[:1] if operand == "quadpoints" else qp,
            regs[:1] if operand == "regularizations" else regs,
            ForceStageTwoConfig(num_force_coils=2),
        )


def test_prepared_stage_config_still_checks_geometry_at_factory(force_case):
    field, _, _, _, _, _, surface = force_case
    config = prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=99))
    with pytest.raises(ValueError, match="available coils"):
        make_stage_two_objective(
            field,
            lambda p: jnp.sum(p),
            surface.gamma().reshape((-1, 3)),
            surface.normal().reshape((-1, 3)),
            config,
        )


def test_stage_surface_shape_is_checked_at_factory(force_case):
    field, _, _, _, _, _, surface = force_case
    with pytest.raises(ValueError, match="matching"):
        make_stage_two_objective(
            field,
            lambda p: jnp.sum(p),
            surface.gamma().reshape((-1, 3)),
            surface.normal(),
            StageTwoObjectiveConfig(num_base_curves=2),
        )


def test_stage_two_numeric_weights_and_targets_vary_without_retracing():
    gamma, gd, gdd = _geometry()
    config = prepare_stage_two_config(
        StageTwoObjectiveConfig(num_base_curves=1, length_weight=2.0, length_target=1.5)
    )
    traces = []

    @jax.jit
    def evaluate(d, operands):
        traces.append(None)
        return stage_two_geometric_penalty(gamma, d, gdd, gamma[0], gamma[0], operands)

    first_value, first_gradient = jax.value_and_grad(evaluate)(gd, config)
    changed = replace(
        config, length_weight=jax.device_put(4.0), length_target=jax.device_put(1.0)
    )
    second_value, second_gradient = jax.value_and_grad(evaluate)(gd, changed)
    np.testing.assert_array_equal(first_value, 0.25)
    np.testing.assert_array_equal(second_value, 2.0)
    np.testing.assert_array_equal(second_gradient, 4.0 * first_gradient)
    assert len(traces) == 1


def test_force_power_and_threshold_are_traced_operands(force_case):
    field, parameters, qp, regs, _, _, _ = force_case
    geometry = stage_two_coil_geometry(field.coil_dof_extraction_spec(), parameters)
    traces = []

    @jax.jit
    def evaluate(operands):
        traces.append(None)
        return _compiled_force_stage_two_metrics(*geometry, qp, regs, operands)[0]

    first = evaluate(jax.device_put(ForceStageTwoConfig(num_force_coils=2)))
    second = evaluate(
        jax.device_put(
            ForceStageTwoConfig(
                num_force_coils=2, force_power=3.5, force_threshold=0.0001
            )
        )
    )
    assert float(first) != float(second)
    assert len(traces) == 1


@pytest.fixture(scope="module")
def finite_case(force_case):
    _, _, _, _, curves, currents, surface = force_case
    filaments = [
        filament
        for curve in curves
        for filament in create_multifilament_grid(
            curve,
            numfilaments_n=2,
            numfilaments_b=1,
            gapsize_n=0.02,
            gapsize_b=0.04,
            rotation_order=1,
        )
    ]
    filament_currents = [current for current in currents for _ in range(2)]
    field = BiotSavartJAX(coils_via_symmetries(filaments, filament_currents, 1, True))
    config = FiniteBuildStageTwoConfig(
        num_base_curves=2,
        filament_offsets=((-0.01, 0.0), (0.01, 0.0)),
        symmetry_copies=2,
        length_targets=(1.0, 1.0),
        length_weight=0.1,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1.0,
    )
    return field, SquaredFluxJAX(surface, field).fixed_surface_flux_spec(), config


@pytest.mark.parametrize(
    "factory",
    (make_finite_build_stage_two_objective, finite_build_stage_two_diagnostics),
)
@pytest.mark.parametrize(
    "change, message",
    (
        ({"num_base_curves": 0}, "num_base_curves"),
        ({"symmetry_copies": 0}, "symmetry_copies"),
        ({"filament_offsets": ()}, "at least one filament"),
        ({"length_targets": (1.0,)}, "one target per base curve"),
        ({"symmetry_copies": 3}, "coil count"),
        ({"filament_offsets": ((0.0, 0.0), (0.01, 0.0))}, "filament order"),
        ({"length_weight": np.inf}, "finite"),
    ),
)
def test_finite_build_invalid_config_is_rejected_at_factory(
    finite_case, factory, change, message
):
    field, flux, config = finite_case
    with pytest.raises(ValueError, match=message):
        factory(field, flux, replace(config, **change))


def test_finite_build_rejects_nonfilament_geometry(force_case, finite_case):
    field, _, _, _, _, _, _ = force_case
    _, flux, config = finite_case
    with pytest.raises(ValueError, match="filament curve specs"):
        make_finite_build_stage_two_objective(field, flux, config)


def test_finite_build_rejects_different_currents_within_a_pack(finite_case):
    field, flux, config = finite_case
    coils = list(field.coils)
    coils[1] = Coil(coils[1].curve, Current(123.0))
    with pytest.raises(ValueError, match="pack current"):
        make_finite_build_stage_two_objective(BiotSavartJAX(coils), flux, config)


def test_finite_build_offsets_weights_and_targets_vary_without_retracing(finite_case):
    field, flux, config = finite_case
    extraction = field.coil_dof_extraction_spec()
    parameters = jax.device_put(np.asarray(field.x))
    traces = []

    @jax.jit
    def evaluate(p, operands):
        traces.append(None)
        coil_set, gamma, gammadash = _finite_build_geometry(extraction, p, operands)
        length, distance, _ = _finite_build_penalties(gamma, gammadash, operands)
        return fixed_surface_flux_integral(coil_set, flux) + length + distance

    first_value, first_gradient = jax.value_and_grad(evaluate)(
        parameters, jax.device_put(config)
    )
    changed = replace(
        config,
        filament_offsets=((-0.015, 0.0), (0.015, 0.0)),
        length_targets=(0.5, 0.5),
        length_weight=0.2,
        curve_curve_minimum_distance=0.3,
        curve_curve_weight=2.0,
    )
    second_value, second_gradient = jax.value_and_grad(evaluate)(
        parameters, jax.device_put(changed)
    )
    assert float(first_value) != float(second_value)
    assert not np.array_equal(first_gradient, second_gradient)
    assert len(traces) == 1
