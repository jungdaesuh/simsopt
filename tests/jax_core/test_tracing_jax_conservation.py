"""JAX-isolated conservation tests for Boozer guiding-centre tracing."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from simsopt.field.boozermagneticfield_jax import BoozerAnalyticJAX
from simsopt.jax_core.tracing import (
    GuidingCenterTracingSpec,
    trace_guiding_center_boozer,
)


_FIELD_PARAMS = {
    "etabar": 0.5,
    "B0": 1.0,
    "N": 4,
    "G0": 1.5,
    "psi0": 0.3,
    "iota0": 0.4,
    "K1": 0.08,
}


def _trace_mode(mode: str):
    field = BoozerAnalyticJAX(**_FIELD_PARAMS)
    stz0 = np.array([0.35, 0.2, 0.4], dtype=np.float64)
    field.set_points(stz0.reshape((1, 3)))
    modB0 = float(np.asarray(field.modB()).reshape(-1)[0])
    G0 = abs(float(np.asarray(field.G()).reshape(-1)[0]))

    mass = 1.0
    charge = 1.0
    speed_total = 0.8
    v_par0 = 0.1
    mu = (speed_total * speed_total - v_par0 * v_par0) / (2.0 * modB0)
    spec = GuidingCenterTracingSpec(
        tmax=5.0,
        dtmax=G0 * 0.5 * np.pi / (modB0 * speed_total),
        rtol=1.0e-10,
        atol=1.0e-12,
        max_steps=20000,
    )
    y0 = jnp.asarray([*stz0, v_par0], dtype=jnp.float64)
    result = trace_guiding_center_boozer(
        spec, y0, field, m=mass, q=charge, mu=mu, mode=mode
    )
    assert int(result.status) == 0
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]
    assert live.shape[0] > 3
    return field, live, mass, mu


@pytest.mark.parametrize("mode", ("vacuum", "no_k", "full"))
def test_boozer_guiding_center_conserves_mu_and_energy(mode):
    """Boozer GC modes conserve moment and energy across a mirror bounce."""

    field, live, mass, mu0 = _trace_mode(mode)
    field.set_points(live[:, 1:4])
    modB = np.asarray(field.modB()).reshape(-1)
    v_par = live[:, 4]
    sign_changes = np.count_nonzero(np.diff(np.signbit(v_par)))

    energy = mass * (0.5 * v_par * v_par + mu0 * modB)
    energy0 = float(energy[0])
    mu_from_energy = energy0 / (mass * modB) - 0.5 * v_par * v_par / modB

    assert sign_changes >= 1
    np.testing.assert_allclose(energy[1:], energy0, rtol=0.0, atol=1.0e-10)
    np.testing.assert_allclose(mu_from_energy[1:], mu0, rtol=0.0, atol=1.0e-10)
