"""Objective factories validate host configs before any traced evaluation."""

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from dataclasses import fields, replace

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
from simsopt_jax.runtime.host_boundary import (
    host_transfer_audit,
    host_transfer_phase,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.finite_build_stage_two import (
    FiniteBuildStageTwoConfig,
    _finite_build_geometry,
    _finite_build_penalties,
    _prepare_finite_build_config,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX
from simsopt_jax_adapters.objectives.force_stage_two import (
    ForceStageTwoConfig,
    _compiled_force_stage_two_metrics,
    _prepare_force_config,
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


def test_repreparing_replaced_config_reenables_length_penalty(force_case):
    disabled = prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=1))
    settings = StageTwoObjectiveConfig(
        num_base_curves=1, length_weight=2.0, length_target=1.5
    )
    changed = replace(disabled, length_weight=2.0, length_target=1.5)
    refreshed = prepare_stage_two_config(changed)
    fresh = prepare_stage_two_config(settings)
    gamma, gd, gdd = _geometry()
    evaluate = jax.jit(stage_two_geometric_penalty)
    np.testing.assert_array_equal(
        evaluate(gamma, gd, gdd, gamma[0], gamma[0], refreshed), 0.25
    )
    np.testing.assert_array_equal(
        evaluate(gamma, gd, gdd, gamma[0], gamma[0], refreshed),
        evaluate(gamma, gd, gdd, gamma[0], gamma[0], fresh),
    )

    field, parameters, _, _, _, _, surface = force_case
    sg = jax.device_put(surface.gamma().reshape((-1, 3)))
    sn = jax.device_put(surface.normal().reshape((-1, 3)))
    objective = make_stage_two_objective(field, lambda p: jnp.sum(p[:0]), sg, sn, changed)
    equivalent = make_stage_two_objective(field, lambda p: jnp.sum(p[:0]), sg, sn, settings)
    np.testing.assert_array_equal(objective(parameters), equivalent(parameters))
    assert float(objective(parameters)) > 0.0


def test_repreparing_replaced_config_disables_undefined_geometry():
    active = prepare_stage_two_config(
        StageTwoObjectiveConfig(num_base_curves=1, curvature_weight=1.0)
    )
    refreshed = prepare_stage_two_config(replace(active, curvature_weight=0.0))
    gamma, gd, gdd = _geometry()
    value, gradient = jax.value_and_grad(
        lambda d: stage_two_geometric_penalty(
            gamma, d, gdd, gamma[0], gamma[0], refreshed
        )
    )(jnp.zeros_like(gd))
    np.testing.assert_array_equal(value, 0.0)
    np.testing.assert_array_equal(gradient, np.zeros_like(gd))


@pytest.mark.parametrize(
    "change, message",
    (
        ({"num_base_curves": 0}, "positive integer"),
        ({"length_target_mode": "bad"}, "length_target_mode"),
        ({"mean_squared_curvature_target_mode": "bad"}, "mean_squared_curvature_target_mode"),
        ({"length_weight": np.nan}, "length_weight"),
        ({"individual_length_weight": 1.0}, "individual_length_target is required"),
    ),
)
def test_replaced_prepared_config_is_revalidated_at_factory(force_case, change, message):
    field, _, _, _, _, _, surface = force_case
    prepared = prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=1))
    changed = replace(prepared, **change)
    with pytest.raises(ValueError, match=message):
        prepare_stage_two_config(changed)
    with pytest.raises(ValueError, match=message):
        make_stage_two_objective(
            field, lambda p: jnp.sum(p), surface.gamma(), surface.normal(), changed
        )


@pytest.mark.parametrize("prepared", (False, True))
@pytest.mark.parametrize(
    "name",
    tuple(
        field.name for field in fields(StageTwoObjectiveConfig)
        if field.name not in (
            "num_base_curves", "length_target_mode", "mean_squared_curvature_target_mode",
            "length_target", "individual_length_target",
        )
    ),
)
def test_required_numeric_fields_reject_none_at_factory(force_case, prepared, name):
    field, _, _, _, _, _, surface = force_case
    config = StageTwoObjectiveConfig(num_base_curves=1)
    if prepared:
        config = prepare_stage_two_config(config)
    with pytest.raises(ValueError, match=f"{name} must be finite"):
        make_stage_two_objective(
            field, lambda p: jnp.sum(p), surface.gamma(), surface.normal(),
            replace(config, **{name: None}),
        )


def test_optional_length_targets_allow_none():
    config = prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=1))
    assert config.length_target is None
    assert config.individual_length_target is None


@pytest.mark.parametrize("value", (1, np.int32(1), np.int64(1)))
def test_stage_count_accepts_python_and_numpy_integers(value):
    config = prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=value))
    assert config.num_base_curves == 1


@pytest.mark.parametrize("value", (True, np.bool_(True), 1.0))
def test_stage_count_rejects_booleans_and_floats(value):
    with pytest.raises(ValueError, match="positive integer"):
        prepare_stage_two_config(StageTwoObjectiveConfig(num_base_curves=value))


@pytest.mark.parametrize("integer", (int, np.int32, np.int64))
def test_force_counts_accept_python_and_numpy_integers(force_case, integer):
    field, parameters, qp, regs, _, _, _ = force_case
    diagnostics = force_stage_two_diagnostics(
        field, qp, regs,
        ForceStageTwoConfig(num_force_coils=integer(2), downsample=integer(1)),
    )
    assert np.all(np.isfinite(diagnostics(parameters)))


@pytest.mark.parametrize("name", ("num_force_coils", "downsample"))
@pytest.mark.parametrize("value", (True, np.bool_(True), 1.0))
def test_force_counts_reject_booleans_and_floats(force_case, name, value):
    field, _, qp, regs, _, _, _ = force_case
    with pytest.raises(ValueError, match=name):
        force_stage_two_diagnostics(
            field, qp, regs, replace(ForceStageTwoConfig(num_force_coils=2), **{name: value})
        )


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


@pytest.mark.parametrize("integer", (int, np.int32, np.int64))
def test_finite_build_counts_accept_python_and_numpy_integers(finite_case, integer):
    field, flux, config = finite_case
    objective = make_finite_build_stage_two_objective(
        field, flux, replace(config, num_base_curves=integer(2), symmetry_copies=integer(2))
    )
    assert np.isfinite(objective(jax.device_put(np.asarray(field.x))))


@pytest.mark.parametrize("name", ("num_base_curves", "symmetry_copies"))
@pytest.mark.parametrize("value", (True, np.bool_(True), 1.0))
def test_finite_build_counts_reject_booleans_and_floats(finite_case, name, value):
    field, flux, config = finite_case
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        make_finite_build_stage_two_objective(field, flux, replace(config, **{name: value}))


def test_single_pack_objective_has_zero_pair_penalty(finite_case):
    field, flux, config = finite_case
    single_pack = BiotSavartJAX(list(field.coils[:config.filaments_per_base]))
    config = replace(config, num_base_curves=1, symmetry_copies=1, length_targets=(1.0,))
    parameters = jax.device_put(np.asarray(single_pack.x))
    objective = make_finite_build_stage_two_objective(single_pack, flux, config)
    coil_set, gamma, gd = _finite_build_geometry(
        single_pack.coil_dof_extraction_spec(), parameters, jax.device_put(config)
    )
    length, distance, _ = _finite_build_penalties(gamma, gd, jax.device_put(config))
    np.testing.assert_array_equal(distance, 0.0)
    np.testing.assert_array_equal(objective(parameters), fixed_surface_flux_integral(coil_set, flux) + length)
    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    assert np.isfinite(value)
    assert np.all(np.isfinite(gradient))
    with pytest.raises(ValueError, match="at least two coil packs"):
        finite_build_stage_two_diagnostics(single_pack, flux, config)


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


@pytest.mark.parametrize("kind", ("stage", "force", "finite"))
def test_prepared_configs_audit_validation_and_reuse_device_operands(
    force_case, finite_case, kind
):
    if kind == "stage":
        host_config = StageTwoObjectiveConfig(
            num_base_curves=1, length_weight=2.0, length_target=1.5
        )
        prepare = prepare_stage_two_config
        gamma, gd, gdd = _geometry()

        @jax.jit
        def evaluate(config):
            return stage_two_geometric_penalty(gamma, gd, gdd, gamma[0], gamma[0], config)

    elif kind == "force":
        field, parameters, qp, regs, _, _, _ = force_case
        extraction = field.coil_dof_extraction_spec()
        host_config = ForceStageTwoConfig(num_force_coils=2)

        def prepare(config):
            return _prepare_force_config(config, extraction, qp, regs)

        geometry = stage_two_coil_geometry(extraction, parameters)

        @jax.jit
        def evaluate(config):
            return _compiled_force_stage_two_metrics(*geometry, qp, regs, config)[0]

    else:
        field, flux, host_config = finite_case
        extraction = field.coil_dof_extraction_spec()
        parameters = jax.device_put(np.asarray(field.x))

        def prepare(config):
            return _prepare_finite_build_config(config, extraction)

        @jax.jit
        def evaluate(config):
            coil_set, gamma, gd = _finite_build_geometry(extraction, parameters, config)
            length, distance, _ = _finite_build_penalties(gamma, gd, config)
            return fixed_surface_flux_integral(coil_set, flux) + length + distance

    with host_transfer_audit() as audit:
        with host_transfer_phase("preparation"):
            config = prepare(host_config)
        with host_transfer_phase("warmup"):
            expected = evaluate(config)
            expected.block_until_ready()
        with host_transfer_phase("revalidation"):
            refreshed = prepare(config)
        with host_transfer_phase("jit_calls"), jax.transfer_guard("disallow_explicit"):
            for _ in range(3):
                actual = evaluate(refreshed)
                actual.block_until_ready()

    np.testing.assert_array_equal(actual, expected)
    summaries = {summary.phase: summary for summary in audit.summary()}
    assert summaries["preparation"].calls == (1 if kind == "finite" else 0)
    assert summaries["warmup"].calls == 0
    assert summaries["jit_calls"].calls == 0
    assert summaries["jit_calls"].bytes == 0
    leaves = jax.tree.leaves(config)
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    if kind == "finite":
        leaves += [leaf for leaf in jax.tree.leaves(extraction) if isinstance(leaf, jax.Array)]
    assert summaries["revalidation"].calls == (2 if kind == "finite" else 1)
    assert summaries["revalidation"].leaves == len(leaves)
    assert summaries["revalidation"].bytes == sum(leaf.nbytes for leaf in leaves)

    # A host scalar in a traced config is uploaded on every invocation. These
    # controls prove the guard catches H2D traffic the D2H ledger cannot count.
    host_operand = replace(refreshed, **{
        "force_power" if kind == "force" else "length_weight": 0.5
    })
    for _ in range(2):
        with jax.transfer_guard("disallow_explicit"), pytest.raises(jax.errors.JaxRuntimeError):
            evaluate(host_operand).block_until_ready()
