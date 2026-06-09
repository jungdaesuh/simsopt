"""JAX-isolated conservation tests for Boozer guiding-centre tracing."""

from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from simsopt.field.tracing import compute_poloidal_transits, compute_toroidal_transits
from simsopt_jax_adapters.field.boozer_field import BoozerAnalyticJAX
from simsopt_jax.core.tracing import (
    GuidingCenterTracingSpec,
    MaxToroidalFluxStoppingCriterion,
    MinToroidalFluxStoppingCriterion,
    ToroidalTransitStoppingCriterion,
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


def _trace_mode(mode: str, field_params: dict[str, float] | None = None):
    params = _FIELD_PARAMS if field_params is None else field_params
    field = BoozerAnalyticJAX(**params)
    stz0 = np.array([0.35, 0.2, 0.4], dtype=np.float64)
    return _trace_initial_condition(mode, field, stz0, speed_total=0.8, v_par0=0.1)


def _trace_initial_condition(
    mode: str,
    field: BoozerAnalyticJAX,
    stz0: np.ndarray,
    *,
    speed_total: float,
    v_par0: float,
):
    field.set_points(stz0.reshape((1, 3)))
    modB0 = float(np.asarray(field.modB()).reshape(-1)[0])
    G0 = abs(float(np.asarray(field.G()).reshape(-1)[0]))

    mass = 1.0
    charge = 1.0
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
    return field, live, mass, charge, mu


def _canonical_momentum(
    field: BoozerAnalyticJAX,
    live: np.ndarray,
    *,
    mass: float,
    charge: float,
    mode: str,
) -> np.ndarray:
    field.set_points(live[:, 1:4])
    modB = np.asarray(field.modB()).reshape(-1)
    G = np.asarray(field.G()).reshape(-1)
    psip = np.asarray(field.psip()).reshape(-1)
    v_par = live[:, 4]
    if mode == "vacuum":
        return v_par * G / modB - charge * psip / mass
    I = np.asarray(field.I()).reshape(-1)
    psi = field.psi0 * live[:, 1]
    return v_par * (G + I) / modB + charge * (psi - psip) / mass


def _trace_vacuum_boozer_with_criteria(
    *,
    stz0: np.ndarray,
    speed_total: float,
    v_par0: float,
    charge: float,
    tmax: float,
    stopping_criteria: tuple,
):
    field = BoozerAnalyticJAX(1.2, 1.0, 4, 1.1, 0.8, 1.0)
    field.set_points(stz0.reshape((1, 3)))
    modB0 = float(np.asarray(field.modB()).reshape(-1)[0])
    G0 = abs(float(np.asarray(field.G()).reshape(-1)[0]))
    mass = 1.0
    mu = (speed_total * speed_total - v_par0 * v_par0) / (2.0 * modB0)
    spec = GuidingCenterTracingSpec(
        tmax=tmax,
        dtmax=G0 * 0.5 * np.pi / (modB0 * speed_total),
        rtol=1.0e-10,
        atol=1.0e-12,
        max_steps=20000,
        max_phi_hits=16,
    )
    result = trace_guiding_center_boozer(
        spec,
        jnp.asarray([*stz0, v_par0], dtype=jnp.float64),
        field,
        m=mass,
        q=charge,
        mu=mu,
        mode="vacuum",
        stopping_criteria=stopping_criteria,
    )
    live = np.asarray(result.trajectory)[np.asarray(result.mask)]
    assert live.shape[0] > 3
    return result, live


@pytest.mark.parametrize("mode", ("vacuum", "no_k", "full"))
def test_boozer_guiding_center_conserves_mu_and_energy(mode):
    """Boozer GC modes conserve moment and energy across a mirror bounce."""

    field, live, mass, _charge, mu0 = _trace_mode(mode)
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


@pytest.mark.parametrize(
    ("mode", "field_params"),
    (
        ("vacuum", {**_FIELD_PARAMS, "N": 0, "K1": 0.0}),
        (
            "no_k",
            {
                **_FIELD_PARAMS,
                "N": 1,
                "G1": 0.2,
                "I0": 0.5,
                "I1": 0.1,
                "K1": 0.0,
            },
        ),
        (
            "full",
            {
                **_FIELD_PARAMS,
                "N": 1,
                "G1": 0.2,
                "I0": 0.5,
                "I1": 0.1,
                "K1": 0.6,
            },
        ),
    ),
)
def test_boozer_guiding_center_conserves_canonical_momentum(mode, field_params):
    """Mirror the legacy Boozer GC canonical-momentum invariant."""

    field, live, mass, charge, _mu0 = _trace_mode(mode, field_params)
    momentum = _canonical_momentum(
        field,
        live,
        mass=mass,
        charge=charge,
        mode=mode,
    )
    np.testing.assert_allclose(momentum[1:], momentum[0], rtol=0.0, atol=1.0e-10)


def test_boozer_guiding_center_transit_diagnostics_reach_one_turn():
    """Mirror the legacy Boozer poloidal/toroidal transit diagnostic check."""

    result, live = _trace_vacuum_boozer_with_criteria(
        stz0=np.array([0.35, 0.2, 0.4], dtype=np.float64),
        speed_total=1.0,
        v_par0=1.0,
        charge=1.0e3,
        tmax=50.0,
        stopping_criteria=(ToroidalTransitStoppingCriterion(1.0),),
    )

    assert int(result.status) == -1
    assert int(result.phi_hits_count) == 1
    assert int(np.asarray(result.phi_hits)[0, 1]) == -1
    np.testing.assert_allclose(compute_poloidal_transits([live]), [1.0])
    np.testing.assert_allclose(compute_toroidal_transits([live]), [1.0])


@pytest.mark.parametrize(
    ("stz0", "expected_status", "expected_event_index", "boundary_s"),
    (
        (np.array([0.45, 0.2, 0.4], dtype=np.float64), -1, -1, 0.4),
        (np.array([0.45, 2.0, 0.4], dtype=np.float64), -2, -2, 0.6),
    ),
)
def test_boozer_guiding_center_toroidal_flux_stopping_bounds(
    stz0, expected_status, expected_event_index, boundary_s
):
    """Mirror the legacy Boozer toroidal-flux confinement criterion check."""

    min_s = 0.4
    max_s = 0.6
    result, live = _trace_vacuum_boozer_with_criteria(
        stz0=stz0,
        speed_total=1.0,
        v_par0=0.1,
        charge=1.0,
        tmax=5.0,
        stopping_criteria=(
            MinToroidalFluxStoppingCriterion(min_s),
            MaxToroidalFluxStoppingCriterion(max_s),
        ),
    )

    hits = np.asarray(result.phi_hits)
    assert int(result.status) == expected_status
    assert int(result.phi_hits_count) == 1
    assert int(hits[0, 1]) == expected_event_index
    assert np.all(live[:, 1] > min_s)
    assert np.all(live[:, 1] < max_s)
    if expected_event_index == -1:
        assert hits[0, 2] <= boundary_s
    else:
        assert hits[0, 2] >= boundary_s
    assert abs(hits[0, 2] - boundary_s) < 1.0e-2
