"""Pure-JAX force Stage-II objective contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves

from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.field.force import (
    curve_force_norms_pure,
    lp_force_pure,
)
from simsopt_jax_adapters.objectives import (
    ForceStageTwoConfig,
    force_stage_two_diagnostics,
)


def _circle(radius: float, point_count: int) -> tuple[jax.Array, jax.Array]:
    angle = jnp.linspace(0.0, 2.0 * jnp.pi, point_count, endpoint=False)
    gamma = jnp.stack(
        (radius * jnp.cos(angle), radius * jnp.sin(angle), jnp.zeros_like(angle)),
        axis=1,
    )
    gammadash = jnp.stack(
        (
            -2.0 * jnp.pi * radius * jnp.sin(angle),
            2.0 * jnp.pi * radius * jnp.cos(angle),
            jnp.zeros_like(angle),
        ),
        axis=1,
    )
    return gamma, gammadash


def test_force_stage_two_diagnostics_slice_on_device_under_transfer_guard() -> None:
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
    coils = coils_via_symmetries(
        curves,
        currents,
        surface.nfp,
        surface.stellsym,
    )
    field = BiotSavartJAX(coils)
    parameters = jax.device_put(jnp.asarray(field.x, dtype=jnp.float64))
    target_quadpoints = jax.device_put(
        jnp.stack(
            tuple(jnp.asarray(curve.quadpoints, dtype=jnp.float64) for curve in curves)
        )
    )
    regularizations = jax.device_put(
        jnp.full((len(coils),), 0.05**2 / np.sqrt(np.e), dtype=jnp.float64)
    )
    diagnostics = force_stage_two_diagnostics(
        field,
        target_quadpoints,
        regularizations,
        ForceStageTwoConfig(num_force_coils=len(curves)),
    )

    with jax.transfer_guard("disallow"):
        values = diagnostics(parameters)
        values.block_until_ready()

    assert values.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(values)))


def test_public_force_norms_reconstruct_lp_force_objective() -> None:
    target_gamma, target_gammadash = _circle(1.0, 16)
    source_gamma, source_gammadash = _circle(1.3, 16)
    target_gammadashdash = jnp.gradient(target_gammadash, axis=0) * 16.0
    target_gammas = target_gamma[None, :, :]
    target_gammadashs = target_gammadash[None, :, :]
    source_gammas = source_gamma[None, :, :]
    source_gammadashs = source_gammadash[None, :, :]
    target_currents = jnp.asarray((1.7e4,), dtype=jnp.float64)
    source_currents = jnp.asarray((-1.7e4,), dtype=jnp.float64)
    quadpoints = jnp.linspace(0.0, 1.0, 16, endpoint=False)[None, :]
    regularizations = jnp.asarray((0.05**2 / np.sqrt(np.e),), dtype=jnp.float64)
    power = 4.0

    force_norms = curve_force_norms_pure(
        target_gammas,
        source_gammas,
        target_gammadashs,
        source_gammadashs,
        target_gammadashdash[None, :, :],
        quadpoints,
        target_currents,
        source_currents,
        regularizations,
        downsample=1,
    )
    objective = lp_force_pure(
        target_gammas,
        source_gammas,
        target_gammadashs,
        source_gammadashs,
        target_gammadashdash[None, :, :],
        quadpoints,
        target_currents,
        source_currents,
        regularizations,
        power,
        0.0,
        1,
    )
    speed = jnp.linalg.norm(target_gammadashs, axis=-1)
    reconstructed = jnp.sum(force_norms**power * speed) / (16.0 * power)

    np.testing.assert_allclose(objective, reconstructed, rtol=1.0e-12, atol=1.0e-12)
