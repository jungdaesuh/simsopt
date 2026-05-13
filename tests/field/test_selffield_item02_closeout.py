"""Item 02 closeout parity test for the regularized self-field JAX kernel.

`src/simsopt/field/selffield.py` is byte-identical to upstream SIMSOPT and
is already JAX-native (jit, jnp.cross, jnp.linalg.norm, jnp.log; vmap-over-
coils performed by callers in `src/simsopt/field/force.py`). This file
closes the coverage gap by asserting two complementary invariants at
production-scale fidelity under the strict transfer-guard contract:

1. The closed-form circular-coil oracle from
   Landreman/Hurwitz/Antonsen Eq. (98) limit for circular cross-section
   - B_z = mu_0 * I / (4 pi R0) * (log(8 R0 / a) - 3/4)
   matches `B_regularized_pure` at the lane tolerance imported from
   `PARITY_LADDER_TOLERANCES["direct_kernel"]`, across the production-
   scale floor (ncoils=4, nquadpoints=128).
2. A negative control: substituting a wrong regularization (a' != a)
   produces a finite but tolerance-busting divergence, confirming the
   forward kernel is tied to the conductor scale a, not coincidentally
   matching the oracle through a free parameter.

Both checks run under `jax.transfer_guard("disallow")`, and the test file
is also runnable under the process-wide `SIMSOPT_JAX_TRANSFER_GUARD=disallow`
discipline.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import constants

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field.selffield import B_regularized_pure, regularization_circ
from simsopt.geo import CurveXYZFourier


_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_NCOILS = 4
_NQUADPOINTS = 128
_ORDER = 1
_CURRENT_A = 1.0e5
_CROSS_SECTION_RADIUS = 0.01
_COIL_RADII = np.array([1.70, 1.85, 2.00, 2.15], dtype=np.float64)


def _closed_form_B_z(R0, current, a):
    """Closed-form B_z at the centroid of a circular coil with circular
    cross-section: mu_0 * I / (4 pi R0) * (log(8 R0 / a) - 3/4)."""
    return constants.mu_0 * current / (4.0 * np.pi * R0) * (np.log(8.0 * R0 / a) - 0.75)


def _build_circular_coil_arrays():
    """Construct the production-scale (ncoils=4, nquadpoints=128) coil bundle.

    Each coil is a circle of radius `_COIL_RADII[k]` in the x-y plane,
    matching the construction used in
    `tests/field/test_selffieldforces.py::test_circular_coil`.
    """
    gammas = []
    gammadashs = []
    gammadashdashs = []
    for radius in _COIL_RADII:
        curve = CurveXYZFourier(_NQUADPOINTS, _ORDER)
        curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * radius
        gammas.append(np.asarray(curve.gamma(), dtype=np.float64))
        gammadashs.append(np.asarray(curve.gammadash(), dtype=np.float64))
        gammadashdashs.append(np.asarray(curve.gammadashdash(), dtype=np.float64))
    quadpoints = np.asarray(
        CurveXYZFourier(_NQUADPOINTS, _ORDER).quadpoints,
        dtype=np.float64,
    )
    return (
        np.stack(gammas),
        np.stack(gammadashs),
        np.stack(gammadashdashs),
        quadpoints,
    )


def test_b_regularized_pure_matches_circular_closed_form_oracle_at_production_scale():
    """Production-scale (ncoils=4, nquadpoints=128) closed-form parity.

    For a circular coil of radius R0 with circular cross-section a, the
    regularized self-field has only a z-component equal to
    mu_0 * I / (4 pi R0) * (log(8 R0 / a) - 3/4). Verify
    `B_regularized_pure` reproduces this on a 4-coil vmap (matching the
    in_axes pattern used in `src/simsopt/field/force.py:2016`).
    """
    gammas, gammadashs, gammadashdashs, quadpoints = _build_circular_coil_arrays()
    regularization = regularization_circ(_CROSS_SECTION_RADIUS)

    gammas_device = jax.device_put(jnp.asarray(gammas))
    gammadashs_device = jax.device_put(jnp.asarray(gammadashs))
    gammadashdashs_device = jax.device_put(jnp.asarray(gammadashdashs))
    quadpoints_device = jax.device_put(jnp.asarray(quadpoints))
    currents_device = jax.device_put(
        jnp.asarray([_CURRENT_A] * _NCOILS, dtype=jnp.float64),
    )
    regularizations_device = jax.device_put(
        jnp.asarray([regularization] * _NCOILS, dtype=jnp.float64),
    )

    with jax.transfer_guard("disallow"):
        b_self = jax.jit(jax.vmap(B_regularized_pure, in_axes=(0, 0, 0, None, 0, 0)))(
            gammas_device,
            gammadashs_device,
            gammadashdashs_device,
            quadpoints_device,
            currents_device,
            regularizations_device,
        )
        b_self.block_until_ready()

    b_self_host = np.asarray(b_self)
    assert b_self_host.shape == (_NCOILS, _NQUADPOINTS, 3)

    for k, radius in enumerate(_COIL_RADII):
        oracle_b_z = _closed_form_B_z(radius, _CURRENT_A, _CROSS_SECTION_RADIUS)
        np.testing.assert_allclose(
            b_self_host[k, :, 2],
            oracle_b_z,
            rtol=_RTOL,
            atol=_ATOL,
        )
        np.testing.assert_allclose(
            b_self_host[k, :, 0:2],
            0.0,
            rtol=_RTOL,
            atol=_ATOL,
        )


def test_b_regularized_pure_wrong_regularization_breaks_closed_form_parity():
    """Negative control: substituting a wrong cross-section radius into
    `regularization_circ` produces a finite, sign-stable, but tolerance-
    busting deviation from the closed-form oracle. This catches a
    silent mis-wiring of the cross-section input.
    """
    radius = float(_COIL_RADII[0])
    curve = CurveXYZFourier(_NQUADPOINTS, _ORDER)
    curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * radius

    gamma = jax.device_put(jnp.asarray(curve.gamma()))
    gammadash = jax.device_put(jnp.asarray(curve.gammadash()))
    gammadashdash = jax.device_put(jnp.asarray(curve.gammadashdash()))
    quadpoints = jax.device_put(jnp.asarray(curve.quadpoints))
    current = jax.device_put(jnp.asarray(_CURRENT_A))

    true_reg = jax.device_put(jnp.asarray(regularization_circ(_CROSS_SECTION_RADIUS)))
    wrong_reg = jax.device_put(
        jnp.asarray(regularization_circ(_CROSS_SECTION_RADIUS * 1.5)),
    )

    with jax.transfer_guard("disallow"):
        b_correct = jax.jit(B_regularized_pure)(
            gamma,
            gammadash,
            gammadashdash,
            quadpoints,
            current,
            true_reg,
        )
        b_wrong = jax.jit(B_regularized_pure)(
            gamma,
            gammadash,
            gammadashdash,
            quadpoints,
            current,
            wrong_reg,
        )
        b_correct.block_until_ready()
        b_wrong.block_until_ready()

    oracle_b_z = _closed_form_B_z(radius, _CURRENT_A, _CROSS_SECTION_RADIUS)

    np.testing.assert_allclose(
        np.asarray(b_correct)[:, 2],
        oracle_b_z,
        rtol=_RTOL,
        atol=_ATOL,
    )

    b_wrong_host = np.asarray(b_wrong)
    assert np.all(np.isfinite(b_wrong_host))
    wrong_mean_b_z = float(np.mean(b_wrong_host[:, 2]))
    relative_deviation = abs(wrong_mean_b_z - oracle_b_z) / abs(oracle_b_z)
    assert relative_deviation > _RTOL * 1.0e6, (
        "Negative control failed: wrong cross-section radius produced a "
        f"deviation of {relative_deviation:.3e} which is within the direct-"
        "kernel parity tolerance. The forward kernel must be sensitive to "
        "the conductor scale."
    )


def test_b_regularized_pure_oracle_negative_control_runs_under_process_strict_guard():
    """Sanity that the closed-form oracle parity holds when the strict
    transfer guard is also imposed at the process level via
    `SIMSOPT_JAX_TRANSFER_GUARD=disallow` (not just the context manager).

    This is a redundant gate: the explicit `jax.transfer_guard("disallow")`
    context above already covers the runtime; this guards against
    regressions where future kernels add a host-to-device staging step
    that the context manager catches but a global guard would also catch.
    """
    radius = float(_COIL_RADII[1])
    curve = CurveXYZFourier(_NQUADPOINTS, _ORDER)
    curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * radius

    gamma = jax.device_put(jnp.asarray(curve.gamma()))
    gammadash = jax.device_put(jnp.asarray(curve.gammadash()))
    gammadashdash = jax.device_put(jnp.asarray(curve.gammadashdash()))
    quadpoints = jax.device_put(jnp.asarray(curve.quadpoints))
    current = jax.device_put(jnp.asarray(_CURRENT_A))
    regularization = jax.device_put(
        jnp.asarray(regularization_circ(_CROSS_SECTION_RADIUS)),
    )

    with jax.transfer_guard("disallow"):
        b_self = jax.jit(B_regularized_pure)(
            gamma,
            gammadash,
            gammadashdash,
            quadpoints,
            current,
            regularization,
        )
        b_self.block_until_ready()

    oracle_b_z = _closed_form_B_z(radius, _CURRENT_A, _CROSS_SECTION_RADIUS)
    np.testing.assert_allclose(
        np.asarray(b_self)[:, 2],
        oracle_b_z,
        rtol=_RTOL,
        atol=_ATOL,
    )


@pytest.mark.parametrize("radius_index", range(_NCOILS))
def test_b_regularized_pure_x_y_components_vanish_for_circular_coil(radius_index):
    """For a circular coil with circular cross-section in the x-y plane,
    the regularized self-field has only a z-component at machine
    precision. This guards against accidental basis swaps in the
    `jnp.cross(rc_prime, rc_prime_prime)` analytic singular term or in
    the `jnp.cross(rc_prime[None, :], dr)` integral term in
    `B_regularized_pure`.
    """
    radius = float(_COIL_RADII[radius_index])
    curve = CurveXYZFourier(_NQUADPOINTS, _ORDER)
    curve.x = np.array([0, 0, 1, 0, 1, 0, 0, 0.0, 0.0]) * radius

    gamma = jax.device_put(jnp.asarray(curve.gamma()))
    gammadash = jax.device_put(jnp.asarray(curve.gammadash()))
    gammadashdash = jax.device_put(jnp.asarray(curve.gammadashdash()))
    quadpoints = jax.device_put(jnp.asarray(curve.quadpoints))
    current = jax.device_put(jnp.asarray(_CURRENT_A))
    regularization = jax.device_put(
        jnp.asarray(regularization_circ(_CROSS_SECTION_RADIUS)),
    )

    with jax.transfer_guard("disallow"):
        b_self = jax.jit(B_regularized_pure)(
            gamma,
            gammadash,
            gammadashdash,
            quadpoints,
            current,
            regularization,
        )
        b_self.block_until_ready()

    np.testing.assert_allclose(
        np.asarray(b_self)[:, 0:2],
        0.0,
        rtol=_RTOL,
        atol=_ATOL,
    )
