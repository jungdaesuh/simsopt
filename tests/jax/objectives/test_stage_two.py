"""Composable JAX Stage-II objective tests."""

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np

from simsopt_jax.objectives.stage_two import (
    StageTwoObjectiveConfig,
    stage_two_geometric_penalty,
)


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
