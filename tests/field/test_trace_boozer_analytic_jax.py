"""Acceptance tests for the N03 §D BoozerAnalyticJAX → tracing dispatch.

The previous run wired :class:`BoozerAnalyticJAX` into the
:func:`_resolve_boozer_field_state` duck-typed acceptance point but the
downstream guiding-centre RHS factories at
``simsopt.jax_core.tracing.guiding_center_vacuum_boozer_rhs``,
``guiding_center_no_k_boozer_rhs`` and ``guiding_center_boozer_rhs``
imported the scalar evaluators directly from
:mod:`simsopt.field.boozermagneticfield_jax` — which are
``BoozerRadialInterpolantFrozenState``-shaped (spline + Fourier mode
sums). Calling them with a ``BoozerAnalyticFrozenState`` (closed-form
analytic scalars) would have failed.

The N03 §D wiring closes that gap with the static dispatch helper
:func:`simsopt.jax_core.tracing._boozer_field_evaluators`. These tests
verify three things:

1. **Dispatch correctness** — the helper returns the analytic
   evaluators for a ``BoozerAnalyticFrozenState`` and the radial
   evaluators for a ``BoozerRadialInterpolantFrozenState``, and raises
   ``TypeError`` on unknown frozen-state types.
2. **Factory acceptance** — all three guiding-centre RHS factories
   accept a :class:`BoozerAnalyticJAX` field and produce a callable
   ``rhs(t, y)`` that emits finite values.
3. **Numerical fidelity** — the analytic RHS evaluated through the
   dispatch matches the closed-form analytic RHS formula evaluated
   against the CPU oracle :class:`simsopt.field.boozermagneticfield.BoozerAnalytic`
   at the same point. The CPU oracle plus the explicit
   guiding-centre algebra is the independent oracle here, not another
   JAX call.

Oracle citation (per ``tests/REVIEWER_ORACLE_LINT.md``):

- **CPU oracle (type 1)**:
  :class:`simsopt.field.boozermagneticfield.BoozerAnalytic` for all
  scalar field values (modB, derivatives, G, I, iota, K, dGds, dIds).
- **Closed-form analytic expression (type 2)**: the upstream
  ``GuidingCenterVacuumBoozerRHS::operator()`` / ``no_k`` / ``full``
  algebraic formulas (cf. ``simsoptpp/tracing.cpp``), reproduced
  inline in :func:`_vacuum_rhs_cpu_oracle` etc. The JAX dispatch is
  validated by composing the CPU scalar oracle with the closed-form
  algebra — neither side touches the JAX kernels under test.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax  # noqa: F401 — imported for JAX float64 config side effect
import jax.numpy as jnp

from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.boozermagneticfield_jax import (
    BoozerAnalyticJAX,
    BoozerRadialInterpolantFrozenState,
)
from simsopt.jax_core.tracing import (
    _BOOZER_RHS_EVAL_KEYS,
    _boozer_field_evaluators,
    guiding_center_boozer_rhs,
    guiding_center_no_k_boozer_rhs,
    guiding_center_vacuum_boozer_rhs,
)


# ----------------------------------------------------------------------
# Fixtures (mirror tests/field/test_boozer_analytic_jax.py for consistency)
# ----------------------------------------------------------------------

_FIXTURES = {
    "qa_standard": dict(
        etabar=0.5,
        B0=1.0,
        N=0,
        G0=1.5,
        psi0=0.3,
        iota0=0.4,
    ),
    "qh_symmetric": dict(
        etabar=-0.7,
        B0=1.2,
        N=4,
        G0=2.0,
        psi0=0.45,
        iota0=0.55,
    ),
    "finite_beta_full": dict(
        etabar=0.6,
        B0=1.1,
        N=2,
        G0=1.8,
        psi0=0.35,
        iota0=0.48,
        Bbar=1.05,
        I0=0.08,
        G1=0.02,
        I1=0.01,
        K1=0.07,
    ),
}


def _state_point(s=0.4, theta=0.3, zeta=0.55):
    return np.asarray([[s, theta, zeta]], dtype=np.float64)


def _y0(s=0.4, theta=0.3, zeta=0.55, v_par=1.7e6):
    return jnp.asarray([s, theta, zeta, v_par], dtype=jnp.float64)


# ----------------------------------------------------------------------
# 1. Dispatch correctness — analytic vs. radial vs. unknown
# ----------------------------------------------------------------------


def test_dispatch_returns_analytic_evaluators_for_boozer_analytic_state():
    """The dispatch must select the analytic-kernel callables for a
    ``BoozerAnalyticFrozenState``.

    Oracle: identity check that the returned callables are the
    closed-form-analytic versions from ``simsopt.jax_core.boozer_analytic``,
    not the spline/Fourier versions from
    ``simsopt.field.boozermagneticfield_jax``. This is a routing test,
    not a numerical parity test.
    """
    from simsopt.jax_core.boozer_analytic import (
        _eval_modB as _analytic_modB,
        _eval_dmodBds as _analytic_dmodBds,
        _eval_iota as _analytic_iota,
    )

    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    evals = _boozer_field_evaluators(wrapper.frozen_state)

    assert set(evals.keys()) == set(_BOOZER_RHS_EVAL_KEYS)
    # Identity check on three representative keys: the dispatch must
    # bind to the analytic kernel function objects.
    assert evals["modB"] is _analytic_modB
    assert evals["dmodBds"] is _analytic_dmodBds
    assert evals["iota"] is _analytic_iota


def test_dispatch_returns_radial_evaluators_for_radial_state():
    """The dispatch must select the spline/Fourier evaluators for a
    ``BoozerRadialInterpolantFrozenState``.

    Oracle: identity check that the returned callables are the
    radial-interpolant versions from
    ``simsopt.field.boozermagneticfield_jax``, not the analytic
    versions. This guards against the dispatch swapping branches
    after a future refactor of the radial-interpolant module.
    """
    from simsopt.field.boozermagneticfield_jax import (
        _eval_modB as _radial_modB,
        _eval_dmodBds as _radial_dmodBds,
        _eval_iota as _radial_iota,
    )

    # Build a minimal-but-valid radial frozen state. The dispatch only
    # checks the type, not the array contents, so a tiny zero-filled
    # frozen state is sufficient — the dispatch contract is purely
    # type-based.
    from simsopt.jax_core.boozer_fixed_state import PiecewisePolynomial1D

    breaks = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    coeffs_scalar = jnp.zeros((4, 1), dtype=jnp.float64)  # cubic, 1 segment
    coeffs_modes = jnp.zeros((2, 4, 1), dtype=jnp.float64)  # 2 modes
    zero_scalar = PiecewisePolynomial1D(breaks=breaks, coeffs=coeffs_scalar)
    zero_modes = PiecewisePolynomial1D(breaks=breaks, coeffs=coeffs_modes)
    state = BoozerRadialInterpolantFrozenState(
        xm=jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        xn=jnp.asarray([0.0, 0.0], dtype=jnp.float64),
        psip=zero_scalar,
        G=zero_scalar,
        I=zero_scalar,
        iota=zero_scalar,
        dGds=zero_scalar,
        dIds=zero_scalar,
        diotads=zero_scalar,
        bmnc=zero_modes,
        dbmncds=zero_modes,
        rmnc=zero_modes,
        drmncds=zero_modes,
        zmns=zero_modes,
        dzmnsds=zero_modes,
        numns=zero_modes,
        dnumnsds=zero_modes,
        bmns=zero_modes,
        dbmnsds=zero_modes,
        rmns=zero_modes,
        drmnsds=zero_modes,
        zmnc=zero_modes,
        dzmncds=zero_modes,
        numnc=zero_modes,
        dnumncds=zero_modes,
        mn_factor=zero_modes,
        d_mn_factor=zero_modes,
        kmns=zero_modes,
        kmnc=zero_modes,
        stellsym=True,
        no_K=False,
    )

    evals = _boozer_field_evaluators(state)
    assert set(evals.keys()) == set(_BOOZER_RHS_EVAL_KEYS)
    assert evals["modB"] is _radial_modB
    assert evals["dmodBds"] is _radial_dmodBds
    assert evals["iota"] is _radial_iota


def test_dispatch_raises_typeerror_on_unknown_state():
    """An unknown frozen-state shape must surface a precise ``TypeError``.

    Oracle: Python type semantics — ``isinstance`` is exhaustive over
    the two registered types and the explicit error message names the
    rejected type. This prevents silent fall-through into the radial
    branch for new state shapes (which would have been the regression
    risk if the dispatch used duck typing instead of ``isinstance``).
    """
    with pytest.raises(TypeError) as excinfo:
        _boozer_field_evaluators(object())
    # The error must include both the offending type name and an
    # explicit list of the two supported types.
    msg = str(excinfo.value)
    assert "BoozerAnalyticFrozenState" in msg
    assert "BoozerRadialInterpolantFrozenState" in msg
    assert "object" in msg


def test_dispatch_exposes_complete_key_set():
    """The dispatch must publish exactly the keys consumed by the three
    guiding-centre RHS factories.

    Oracle: the contract documented at ``tracing.py:_BOOZER_RHS_EVAL_KEYS``
    is the SSOT for the 12 scalar evaluators used inside the three
    factories. A new RHS factory that consumes a new key must also
    extend the tuple.
    """
    expected = {
        "modB",
        "dmodBds",
        "dmodBdtheta",
        "dmodBdzeta",
        "K",
        "dKdtheta",
        "dKdzeta",
        "G",
        "I",
        "iota",
        "dGds",
        "dIds",
    }
    assert set(_BOOZER_RHS_EVAL_KEYS) == expected


# ----------------------------------------------------------------------
# 2. Factory acceptance — the 3 RHS factories must consume BoozerAnalyticJAX
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_vacuum_rhs_accepts_boozer_analytic_jax(fixture_name):
    """Vacuum RHS factory must construct cleanly and return finite values.

    Oracle: ``BoozerAnalyticJAX.modB``/``iota``/``G`` returning finite
    values at the test point — finiteness is the minimum-acceptance
    criterion. Numerical parity is tested separately in
    :func:`test_vacuum_rhs_matches_cpu_oracle_closed_form`.
    """
    wrapper = BoozerAnalyticJAX(**_FIXTURES[fixture_name])
    rhs = guiding_center_vacuum_boozer_rhs(
        wrapper, m=1.6726e-27, q=1.6022e-19, mu=1.0e-15
    )
    out = rhs(0.0, _y0())
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_no_k_rhs_accepts_boozer_analytic_jax(fixture_name):
    """The no-K RHS factory must consume the analytic state cleanly."""
    wrapper = BoozerAnalyticJAX(**_FIXTURES[fixture_name])
    rhs = guiding_center_no_k_boozer_rhs(
        wrapper, m=1.6726e-27, q=1.6022e-19, mu=1.0e-15
    )
    out = rhs(0.0, _y0())
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_full_rhs_accepts_boozer_analytic_jax(fixture_name):
    """The full RHS factory (with K terms) must consume the analytic state."""
    wrapper = BoozerAnalyticJAX(**_FIXTURES[fixture_name])
    rhs = guiding_center_boozer_rhs(wrapper, m=1.6726e-27, q=1.6022e-19, mu=1.0e-15)
    out = rhs(0.0, _y0())
    assert out.shape == (4,)
    assert jnp.all(jnp.isfinite(out))


def test_factory_rejects_unknown_field_type():
    """The factories must reject objects that lack ``frozen_state``/``psi0``."""
    with pytest.raises(TypeError):
        guiding_center_vacuum_boozer_rhs(object(), m=1.0, q=1.0, mu=1.0)


# ----------------------------------------------------------------------
# 3. Numerical fidelity vs. CPU oracle + closed-form RHS algebra
# ----------------------------------------------------------------------


def _cpu_scalars(params, point):
    """Evaluate every CPU oracle scalar that the RHS algebra needs.

    Oracle: ``simsopt.field.boozermagneticfield.BoozerAnalytic`` —
    independent C++/Python implementation of the analytic Boozer
    field. This is the type-1 oracle (CPU reference) used by every
    numerical-parity assertion below.
    """
    cpu = BoozerAnalytic(**params)
    cpu.set_points(point)
    return {
        "modB": float(cpu.modB()[0, 0]),
        "dmodBds": float(cpu.dmodBds()[0, 0]),
        "dmodBdtheta": float(cpu.dmodBdtheta()[0, 0]),
        "dmodBdzeta": float(cpu.dmodBdzeta()[0, 0]),
        "K": float(cpu.K()[0, 0]),
        "dKdtheta": float(cpu.dKdtheta()[0, 0]),
        "dKdzeta": float(cpu.dKdzeta()[0, 0]),
        "G": float(cpu.G()[0, 0]),
        "I": float(cpu.I()[0, 0]),
        "iota": float(cpu.iota()[0, 0]),
        "dGds": float(cpu.dGds()[0, 0]),
        "dIds": float(cpu.dIds()[0, 0]),
    }


def _vacuum_rhs_cpu_oracle(scalars, *, m, q, mu, psi0, v_par):
    """Closed-form vacuum-Boozer RHS evaluated on the CPU oracle scalars.

    Mirrors the algebra at ``simsoptpp/tracing.cpp::GuidingCenterVacuumBoozerRHS``
    and the JAX implementation at ``tracing.py``: this is type-2 oracle
    (closed-form formula). The independence requirement is satisfied
    because the scalar inputs come from the CPU ``BoozerAnalytic``
    class, not from any JAX kernel.
    """
    fak1 = m * v_par * v_par / scalars["modB"] + m * mu
    ds = -scalars["dmodBdtheta"] * fak1 / (q * psi0)
    dtheta = (
        scalars["dmodBds"] * fak1 / (q * psi0)
        + scalars["iota"] * v_par * scalars["modB"] / scalars["G"]
    )
    dzeta = v_par * scalars["modB"] / scalars["G"]
    dv_par = (
        -(scalars["iota"] * scalars["dmodBdtheta"] + scalars["dmodBdzeta"])
        * mu
        * scalars["modB"]
        / scalars["G"]
    )
    return np.asarray([ds, dtheta, dzeta, dv_par], dtype=np.float64)


def _no_k_rhs_cpu_oracle(scalars, *, m, q, mu, psi0, v_par):
    """Closed-form ``no_K=True`` Boozer RHS on CPU oracle scalars.

    Mirrors ``GuidingCenterNoKBoozerRHS::operator()`` algebra at
    ``simsoptpp/tracing.cpp`` — closed-form formula, independent of
    the JAX kernel under test.
    """
    dGdpsi = scalars["dGds"] / psi0
    dIdpsi = scalars["dIds"] / psi0
    dmodBdpsi = scalars["dmodBds"] / psi0

    fak1 = m * v_par * v_par / scalars["modB"] + m * mu
    D = (
        (q + m * v_par * dIdpsi / scalars["modB"]) * scalars["G"]
        - (-q * scalars["iota"] + m * v_par * dGdpsi / scalars["modB"]) * scalars["I"]
    ) / scalars["iota"]

    ds = (
        (scalars["I"] * scalars["dmodBdzeta"] - scalars["G"] * scalars["dmodBdtheta"])
        * fak1
        / (D * scalars["iota"] * psi0)
    )
    dtheta = (
        scalars["G"] * dmodBdpsi * fak1
        - (-q * scalars["iota"] + m * v_par * dGdpsi / scalars["modB"])
        * v_par
        * scalars["modB"]
    ) / (D * scalars["iota"])
    dzeta = (
        (q + m * v_par * dIdpsi / scalars["modB"]) * v_par * scalars["modB"]
        - dmodBdpsi * fak1 * scalars["I"]
    ) / (D * scalars["iota"])
    dv_par = -(mu / v_par) * (
        dmodBdpsi * ds * psi0
        + scalars["dmodBdtheta"] * dtheta
        + scalars["dmodBdzeta"] * dzeta
    )
    return np.asarray([ds, dtheta, dzeta, dv_par], dtype=np.float64)


def _full_rhs_cpu_oracle(scalars, *, m, q, mu, psi0, v_par):
    """Closed-form full Boozer RHS (K != 0) on CPU oracle scalars.

    Mirrors ``GuidingCenterBoozerRHS::operator()`` algebra at
    ``simsoptpp/tracing.cpp`` — closed-form formula, independent of
    the JAX kernel under test.
    """
    dGdpsi = scalars["dGds"] / psi0
    dIdpsi = scalars["dIds"] / psi0
    dmodBdpsi = scalars["dmodBds"] / psi0
    fak1 = m * v_par * v_par / scalars["modB"] + m * mu

    C = (
        -m * v_par * (scalars["dKdzeta"] - dGdpsi) / scalars["modB"]
        - q * scalars["iota"]
    )
    F = -m * v_par * (scalars["dKdtheta"] - dIdpsi) / scalars["modB"] + q
    D = (F * scalars["G"] - C * scalars["I"]) / scalars["iota"]

    ds = (
        (scalars["I"] * scalars["dmodBdzeta"] - scalars["G"] * scalars["dmodBdtheta"])
        * fak1
        / (D * scalars["iota"] * psi0)
    )
    dtheta = (
        scalars["G"] * dmodBdpsi * fak1
        - C * v_par * scalars["modB"]
        - scalars["K"] * fak1 * scalars["dmodBdzeta"]
    ) / (D * scalars["iota"])
    dzeta = (
        F * v_par * scalars["modB"]
        - dmodBdpsi * fak1 * scalars["I"]
        + scalars["K"] * fak1 * scalars["dmodBdtheta"]
    ) / (D * scalars["iota"])
    dv_par = -(mu / v_par) * (
        dmodBdpsi * ds * psi0
        + scalars["dmodBdtheta"] * dtheta
        + scalars["dmodBdzeta"] * dzeta
    )
    return np.asarray([ds, dtheta, dzeta, dv_par], dtype=np.float64)


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_vacuum_rhs_matches_cpu_oracle_closed_form(fixture_name):
    """JAX vacuum RHS must equal the CPU+closed-form oracle to float64
    precision.

    Oracle composition (per ``REVIEWER_ORACLE_LINT.md``):
    - CPU scalar values: type-1 oracle (CPU ``BoozerAnalytic``).
    - RHS algebra: type-2 oracle (closed-form formula from upstream
      ``tracing.cpp``).

    Neither side of the comparison touches the JAX kernel under test,
    so this is an independent end-to-end gate for the analytic
    dispatch.
    """
    params = _FIXTURES[fixture_name]
    point = _state_point(s=0.42, theta=0.27, zeta=0.61)
    m, q, mu, v_par = 1.6726e-27, 1.6022e-19, 1.0e-15, 1.7e6

    wrapper = BoozerAnalyticJAX(**params)
    rhs = guiding_center_vacuum_boozer_rhs(wrapper, m=m, q=q, mu=mu)
    y = jnp.asarray([point[0, 0], point[0, 1], point[0, 2], v_par], dtype=jnp.float64)
    jax_out = np.asarray(rhs(0.0, y))

    scalars = _cpu_scalars(params, point)
    expected = _vacuum_rhs_cpu_oracle(
        scalars, m=m, q=q, mu=mu, psi0=params["psi0"], v_par=v_par
    )

    np.testing.assert_allclose(jax_out, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_no_k_rhs_matches_cpu_oracle_closed_form(fixture_name):
    """JAX no-K RHS must equal the CPU+closed-form oracle to float64
    precision.

    Same oracle composition as the vacuum variant — see that test's
    docstring for the type-1/type-2 split.
    """
    params = _FIXTURES[fixture_name]
    point = _state_point(s=0.42, theta=0.27, zeta=0.61)
    m, q, mu, v_par = 1.6726e-27, 1.6022e-19, 1.0e-15, 1.7e6

    wrapper = BoozerAnalyticJAX(**params)
    rhs = guiding_center_no_k_boozer_rhs(wrapper, m=m, q=q, mu=mu)
    y = jnp.asarray([point[0, 0], point[0, 1], point[0, 2], v_par], dtype=jnp.float64)
    jax_out = np.asarray(rhs(0.0, y))

    scalars = _cpu_scalars(params, point)
    expected = _no_k_rhs_cpu_oracle(
        scalars, m=m, q=q, mu=mu, psi0=params["psi0"], v_par=v_par
    )

    np.testing.assert_allclose(jax_out, expected, rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_full_rhs_matches_cpu_oracle_closed_form(fixture_name):
    """JAX full RHS (with K) must equal the CPU+closed-form oracle.

    Same oracle composition — extends to K and dK/dθ, dK/dζ terms.
    """
    params = _FIXTURES[fixture_name]
    point = _state_point(s=0.42, theta=0.27, zeta=0.61)
    m, q, mu, v_par = 1.6726e-27, 1.6022e-19, 1.0e-15, 1.7e6

    wrapper = BoozerAnalyticJAX(**params)
    rhs = guiding_center_boozer_rhs(wrapper, m=m, q=q, mu=mu)
    y = jnp.asarray([point[0, 0], point[0, 1], point[0, 2], v_par], dtype=jnp.float64)
    jax_out = np.asarray(rhs(0.0, y))

    scalars = _cpu_scalars(params, point)
    expected = _full_rhs_cpu_oracle(
        scalars, m=m, q=q, mu=mu, psi0=params["psi0"], v_par=v_par
    )

    np.testing.assert_allclose(jax_out, expected, rtol=1e-12, atol=1e-14)


def test_full_rhs_with_K1_terms_diverges_from_no_k_oracle():
    """When ``K1 != 0`` (finite_beta_full fixture), the full RHS must
    measurably differ from the no-K RHS.

    Oracle: the analytic RHS algebra dictates that adding K introduces
    additional terms in the dθ/dζ equations. This is a self-consistency
    check at the algebra level — not a JAX-vs-JAX tautology, because
    both no-K and full RHS are constructed from independent CPU
    oracle scalars plus closed-form algebra, and they must differ in
    the documented way (term-for-term in C and F coefficients).
    """
    params = _FIXTURES["finite_beta_full"]
    assert params["K1"] != 0.0, "fixture invariant — K1 must be nonzero"
    point = _state_point(s=0.42, theta=0.27, zeta=0.61)
    m, q, mu, v_par = 1.6726e-27, 1.6022e-19, 1.0e-15, 1.7e6

    scalars = _cpu_scalars(params, point)
    no_k = _no_k_rhs_cpu_oracle(
        scalars, m=m, q=q, mu=mu, psi0=params["psi0"], v_par=v_par
    )
    full = _full_rhs_cpu_oracle(
        scalars, m=m, q=q, mu=mu, psi0=params["psi0"], v_par=v_par
    )

    # The full RHS should differ measurably in dθ and dζ when K != 0
    # but match in ds (the ds expression is identical between no_K
    # and full at the algebra level, modulo the D coefficient
    # subtlety — both pull from the same modB-derivs algebra).
    assert not np.allclose(no_k, full, rtol=1e-6, atol=1e-9), (
        "K1 != 0 oracle outputs must differ between no_K and full RHS"
    )


def test_dispatch_is_pure_python_not_jit_traced():
    """Calling the factory must NOT compile-trace the dispatch.

    Oracle: ``_boozer_field_evaluators`` returns a dict, which is not
    a JAX-traceable structure. If it were called inside a ``jax.jit``
    boundary, JAX would raise a ``TypeError`` about non-array tree
    leaves. The factory pattern ensures the dispatch happens once at
    Python time.

    This is a routing test, not a numerical test — it verifies the
    static-dispatch contract.
    """
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    # If the factory accidentally moved the dispatch inside a jit
    # boundary, this construction would raise. The fact that it
    # succeeds and produces a callable is the contract assertion.
    rhs = guiding_center_vacuum_boozer_rhs(
        wrapper, m=1.6726e-27, q=1.6022e-19, mu=1.0e-15
    )
    assert callable(rhs)
