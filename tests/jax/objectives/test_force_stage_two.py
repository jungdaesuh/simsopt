"""Pure-JAX force Stage-II objective contracts."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import B2Energy, Current, LpCurveForce, coils_via_symmetries
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


_REGULARIZATION = 0.05**2 / np.sqrt(np.e)


def _stage_two_force_case():
    """Two-base-curve stellarator-symmetric case shared by the diagnostics tests.

    Returns ``(field, parameters, target_quadpoints, regularizations, curves,
    currents, surface)``: the JAX field and its device dof vector for the
    diagnostics under test, plus the base ``curves``, ``currents``, and
    ``surface`` a caller needs to rebuild the same symmetry-expanded coil set
    as native ``RegularizedCoil`` objects for an oracle comparison.
    """
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
    coils = coils_via_symmetries(curves, currents, surface.nfp, surface.stellsym)
    field = BiotSavartJAX(coils)
    parameters = jax.device_put(jnp.asarray(field.x, dtype=jnp.float64))
    target_quadpoints = jax.device_put(
        jnp.stack(
            tuple(jnp.asarray(curve.quadpoints, dtype=jnp.float64) for curve in curves)
        )
    )
    regularizations = jax.device_put(
        jnp.full((len(coils),), _REGULARIZATION, dtype=jnp.float64)
    )
    return (
        field,
        parameters,
        target_quadpoints,
        regularizations,
        curves,
        currents,
        surface,
    )


def test_force_stage_two_diagnostics_slice_on_device_under_transfer_guard() -> None:
    (
        field,
        parameters,
        target_quadpoints,
        regularizations,
        curves,
        _,
        _,
    ) = _stage_two_force_case()
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


def test_force_stage_two_diagnostics_match_native_force_objectives() -> None:
    """The three diagnostics equal native LpCurveForce, max |F|, and B2Energy.

    ``num_force_coils`` splits the symmetry-expanded coil list into targets
    (the leading base-curve slots) and sources (the rest), passed to native
    ``LpCurveForce`` as ``target_coils`` / ``source_coils_coarse``.  Native's
    own docstring notes that coils common to both lists are removed during
    initialization, so this split is not independently checkable through
    ``diagnostics[0]`` alone when, as here, ``target_coils`` is disjoint from
    ``source_coils_coarse`` and no dedup fires; ``diagnostics[1]`` (max |F|
    over the target coils only) does exercise the target/source distinction.

    Native ``LpCurveForce``/``B2Energy`` (``simsopt.field.force``) are
    themselves ``jax.numpy``-based and, as of this test, byte-identical to
    ``hiddenSymmetries/simsopt``'s current ``master``; the diagnostics under
    test are an independent fork re-implementation, so this oracle check is
    fork-reimplementation vs. upstream implementation, not two copies of the
    same code.  The comparison still exercises the compiled C++ leg: native
    ``RegularizedCoil.force()`` — called here to compute
    ``native_max_force_mn_per_m`` — reaches the mutual field through
    ``simsopt.field.biotsavart.BiotSavart`` (a subclass of the compiled
    ``simsoptpp.BiotSavart``), combined with a ``jax.numpy`` self-field term.
    """
    (
        field,
        parameters,
        target_quadpoints,
        regularizations,
        curves,
        currents,
        surface,
    ) = _stage_two_force_case()
    config = ForceStageTwoConfig(num_force_coils=len(curves))
    diagnostics = np.asarray(
        force_stage_two_diagnostics(
            field,
            target_quadpoints,
            regularizations,
            config,
        )(parameters)
    )

    regularized_coils = coils_via_symmetries(
        curves,
        currents,
        surface.nfp,
        surface.stellsym,
        regularizations=[_REGULARIZATION] * len(curves),
    )
    target_coils = regularized_coils[: config.num_force_coils]
    source_coils = regularized_coils[config.num_force_coils :]

    native_force_objective = float(
        LpCurveForce(
            target_coils,
            source_coils,
            p=config.force_power,
            threshold=config.force_threshold,
        ).J()
    )
    native_max_force_mn_per_m = (
        max(
            float(
                np.max(
                    np.linalg.norm(np.asarray(coil.force(regularized_coils)), axis=1)
                )
            )
            for coil in target_coils
        )
        / 1.0e6
    )
    native_energy = float(B2Energy(regularized_coils).J())

    np.testing.assert_allclose(
        diagnostics[0], native_force_objective, rtol=1.0e-12, atol=0.0
    )
    np.testing.assert_allclose(
        diagnostics[1], native_max_force_mn_per_m, rtol=1.0e-8, atol=0.0
    )
    np.testing.assert_allclose(diagnostics[2], native_energy, rtol=1.0e-12, atol=0.0)


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
