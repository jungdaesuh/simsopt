"""Composable JAX Stage-II objective tests."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.objectives.stage_two import (
    StageTwoObjectiveConfig,
    stage_two_coil_geometry,
    stage_two_geometric_penalty,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def _geometry() -> tuple[jax.Array, jax.Array, jax.Array]:
    gamma = jnp.asarray(
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ),
        dtype=jnp.float64,
    )
    gammadash = jnp.asarray(
        (
            ((0.0, 2.0, 0.0), (-2.0, 0.0, 0.0)),
            ((0.0, -2.0, 0.0), (2.0, 0.0, 0.0)),
        ),
        dtype=jnp.float64,
    )
    return gamma, gammadash, -gamma


def test_stage_two_current_linearization_avoids_host_zero_tangents() -> None:
    surface = SurfaceRZFourier(nfp=2, stellsym=True, mpol=1, ntor=0)
    curves = create_equally_spaced_curves(
        2,
        surface.nfp,
        surface.stellsym,
        R0=1.0,
        R1=0.25,
        order=1,
        numquadpoints=8,
        use_jax_curve=False,
    )
    currents = [Current(1.0e5), Current(1.0e5)]
    currents[0].fix_all()
    field = BiotSavartJAX(
        coils_via_symmetries(curves, currents, surface.nfp, surface.stellsym)
    )
    extraction = field.coil_dof_extraction_spec()
    parameters = jax.device_put(jnp.asarray(field.x, dtype=jnp.float64))
    tangent = jax.device_put(jnp.ones_like(parameters))

    def total_current(current_parameters: jax.Array) -> jax.Array:
        return jnp.sum(stage_two_coil_geometry(extraction, current_parameters)[3])

    with jax.transfer_guard("disallow"):
        value, linearized = jax.linearize(total_current, parameters)
        directional_derivative = linearized(tangent)
        jax.block_until_ready((value, directional_derivative))

    assert bool(jnp.isfinite(value))
    assert bool(jnp.isfinite(directional_derivative))


def _length_only_config(
    *,
    target: float | None,
    target_mode: Literal["max", "identity"] = "max",
) -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=1,
        length_weight=2.0,
        length_target=target,
        length_target_mode=target_mode,
        curve_curve_weight=0.0,
        curve_surface_weight=0.0,
        curvature_weight=0.0,
        mean_squared_curvature_weight=0.0,
    )


def test_stage_two_geometric_penalty_supports_linear_and_target_length_terms() -> None:
    gamma, gammadash, gammadashdash = _geometry()
    surface_gamma = jnp.asarray(((3.0, 0.0, 0.0),), dtype=jnp.float64)
    surface_normal = jnp.asarray(((1.0, 0.0, 0.0),), dtype=jnp.float64)

    linear = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        surface_gamma,
        surface_normal,
        _length_only_config(target=None),
    )
    targeted = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        surface_gamma,
        surface_normal,
        _length_only_config(target=1.5),
    )

    np.testing.assert_allclose(linear, 4.0)
    np.testing.assert_allclose(targeted, 0.25)

    identity = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        surface_gamma,
        surface_normal,
        _length_only_config(target=3.0, target_mode="identity"),
    )
    np.testing.assert_allclose(identity, 1.0)


def test_stage_two_geometric_penalty_supports_per_base_curve_length_targets() -> None:
    gamma, gammadash, gammadashdash = _geometry()
    value = stage_two_geometric_penalty(
        gamma,
        gammadash,
        gammadashdash,
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        StageTwoObjectiveConfig(
            num_base_curves=2,
            length_weight=1.0e-8,
            individual_length_target=1.5,
            individual_length_weight=0.1,
        ),
    )

    expected = 1.0e-8 * 4.0 + 0.1 * (0.5 * 0.5**2 + 0.5 * 0.5**2)
    np.testing.assert_allclose(value, expected)


def test_stage_two_geometric_penalty_is_jittable_and_differentiable() -> None:
    gamma, gammadash, gammadashdash = _geometry()
    surface_gamma = jnp.asarray(((3.0, 0.0, 0.0),), dtype=jnp.float64)
    surface_normal = jnp.asarray(((1.0, 0.0, 0.0),), dtype=jnp.float64)
    config = _length_only_config(target=None)

    def penalty(current_gammadash: jax.Array) -> jax.Array:
        return stage_two_geometric_penalty(
            gamma,
            current_gammadash,
            gammadashdash,
            surface_gamma,
            surface_normal,
            config,
        )

    value, gradient = jax.jit(jax.value_and_grad(penalty))(gammadash)

    np.testing.assert_allclose(value, 4.0)
    assert gradient.shape == gammadash.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_stage_two_geometric_penalty_includes_arclength_variation() -> None:
    gamma, gammadash, gammadashdash = _geometry()
    varying_speed = gammadash.at[0, 0].set(jnp.asarray((0.0, 1.0, 0.0)))
    varying_speed = varying_speed.at[0, 1].set(jnp.asarray((-3.0, 0.0, 0.0)))
    config = StageTwoObjectiveConfig(
        num_base_curves=1,
        arclength_variation_weight=2.0,
    )

    value = stage_two_geometric_penalty(
        gamma,
        varying_speed,
        gammadashdash,
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        config,
    )

    np.testing.assert_allclose(value, 2.0)


def test_stage_two_geometric_penalty_includes_linking_number() -> None:
    angles = jnp.linspace(0.0, 2.0 * jnp.pi, 128, endpoint=False)
    gamma = jnp.stack(
        (
            jnp.stack(
                (jnp.cos(angles), jnp.sin(angles), jnp.zeros_like(angles)), axis=1
            ),
            jnp.stack(
                (1.0 + jnp.cos(angles), jnp.zeros_like(angles), jnp.sin(angles)),
                axis=1,
            ),
        )
    )
    gammadash = jnp.stack(
        (
            jnp.stack(
                (
                    -2.0 * jnp.pi * jnp.sin(angles),
                    2.0 * jnp.pi * jnp.cos(angles),
                    jnp.zeros_like(angles),
                ),
                axis=1,
            ),
            jnp.stack(
                (
                    -2.0 * jnp.pi * jnp.sin(angles),
                    jnp.zeros_like(angles),
                    2.0 * jnp.pi * jnp.cos(angles),
                ),
                axis=1,
            ),
        )
    )
    config = StageTwoObjectiveConfig(
        num_base_curves=1,
        linking_number_weight=3.0,
    )

    value = stage_two_geometric_penalty(
        gamma,
        gammadash,
        jnp.zeros_like(gammadash),
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.zeros((1, 3), dtype=jnp.float64),
        config,
    )

    np.testing.assert_allclose(value, 3.0)
