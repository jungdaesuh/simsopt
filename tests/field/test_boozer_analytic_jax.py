"""Parity tests for ``BoozerAnalyticJAX`` (item N01).

The CPU oracle is :class:`simsopt.field.boozermagneticfield.BoozerAnalytic`.
We construct matched (CPU, JAX) pairs for three fixtures (QA, QH, finite-β
with all corrections non-zero) and compare every public method at the
direct-kernel parity-ladder tolerance (rtol=1e-12, atol=1e-14).

These tests do NOT require simsoptpp's BoozerRadialInterpolant / VMEC
fixtures — the CPU oracle is itself pure NumPy on top of
``sopp.BoozerMagneticField``'s caching layer.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsoptpp as sopp

from simsopt.field.boozermagneticfield import BoozerAnalytic
from simsopt.field.boozermagneticfield_jax import (
    BoozerAnalyticFrozenState,
    BoozerAnalyticJAX,
    freeze_boozer_analytic_state,
)
from simsopt.jax_core.boozer_analytic import (
    _eval_dGds,
    _eval_dIds,
    _eval_dKdtheta,
    _eval_dKdzeta,
    _eval_diotads,
    _eval_dmodBds,
    _eval_dmodBdtheta,
    _eval_dmodBdzeta,
    _eval_G,
    _eval_I,
    _eval_K,
    _eval_iota,
    _eval_modB,
    _eval_psip,
)


_RTOL = 1e-12
_ATOL = 1e-14


_PARITY_METHODS = (
    "modB",
    "dmodBds",
    "dmodBdtheta",
    "dmodBdzeta",
    "G",
    "dGds",
    "I",
    "dIds",
    "iota",
    "diotads",
    "psip",
    "K",
    "dKdtheta",
    "dKdzeta",
)
_FROZEN_STATE_FIELDS = (
    "etabar",
    "B0",
    "Bbar",
    "N",
    "G0",
    "I0",
    "G1",
    "I1",
    "K1",
    "iota0",
    "psi0",
)
_KERNELS = (
    _eval_modB,
    _eval_dmodBds,
    _eval_dmodBdtheta,
    _eval_dmodBdzeta,
    _eval_G,
    _eval_dGds,
    _eval_I,
    _eval_dIds,
    _eval_iota,
    _eval_diotads,
    _eval_psip,
    _eval_K,
    _eval_dKdtheta,
    _eval_dKdzeta,
)


_FIXTURES = {
    "qa_standard": dict(
        etabar=1.1,
        B0=1.0,
        N=0,
        G0=1.1,
        psi0=0.8,
        iota0=0.4,
        Bbar=1.0,
        I0=0.0,
        G1=0.0,
        I1=0.0,
        K1=0.0,
    ),
    "qh_symmetric": dict(
        etabar=0.7,
        B0=1.3,
        N=1,
        G0=1.4,
        psi0=0.9,
        iota0=0.42,
        Bbar=1.5,
        I0=0.0,
        G1=0.0,
        I1=0.0,
        K1=0.0,
    ),
    "finite_beta_full": dict(
        etabar=1.05,
        B0=1.2,
        N=2,
        G0=1.25,
        psi0=0.85,
        iota0=0.45,
        Bbar=0.95,
        I0=0.05,
        G1=0.08,
        I1=0.04,
        K1=0.18,
    ),
}


def _make_points() -> np.ndarray:
    """Return a (49, 3) array of (s, θ, ζ) sample points.

    s stays above 0.05 to avoid the analytic ``1/sqrt(s)`` singularity in
    ``dmodBds`` near the axis.
    """
    s = np.array([0.05, 0.18, 0.37, 0.52, 0.68, 0.81, 0.94], dtype=np.float64)
    theta = np.array([0.1, 1.05, 2.4, 3.27, 4.18, 5.1, 6.05], dtype=np.float64)
    zeta = np.array(
        [0.02, 0.41, 0.97, 1.55, 2.11, 2.69, 3.07], dtype=np.float64
    )
    grid_s, grid_t = np.meshgrid(s, theta, indexing="xy")
    # Drop one axis to a length-7 zeta sample to get a (7, 7) = 49 grid.
    pts = np.stack(
        [
            grid_s.flatten(),
            grid_t.flatten(),
            np.tile(zeta, len(s)),
        ],
        axis=1,
    )
    return np.ascontiguousarray(pts, dtype=np.float64)


def _pair(fixture_name: str):
    params = _FIXTURES[fixture_name]
    cpu = BoozerAnalytic(**params)
    jax_wrapper = BoozerAnalyticJAX(**params)
    return cpu, jax_wrapper


def _compare_all_methods(cpu, jax_wrapper, points, method_names=_PARITY_METHODS):
    cpu.set_points(points)
    jax_wrapper.set_points(points)
    for name in method_names:
        cpu_val = getattr(cpu, name)()
        jax_val = getattr(jax_wrapper, name)()
        np.testing.assert_allclose(
            jax_val,
            cpu_val,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=name,
        )


# ----------------------------------------------------------------------
# Construction / metadata tests
# ----------------------------------------------------------------------


def test_construction_does_not_inherit_sopp():
    """The JAX wrapper must not inherit from the C++ binding base class."""
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    assert not isinstance(wrapper, sopp.BoozerMagneticField)


def test_construction_args_match_cpu():
    """Same constructor signature on both classes."""
    params = _FIXTURES["finite_beta_full"]
    cpu = BoozerAnalytic(**params)
    jax_wrapper = BoozerAnalyticJAX(**params)
    assert cpu.psi0 == pytest.approx(jax_wrapper.psi0)
    assert int(cpu.N) == int(jax_wrapper.N)


def test_frozen_state_arrays_are_jax_scalars():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["finite_beta_full"])
    state = wrapper.frozen_state
    assert isinstance(state, BoozerAnalyticFrozenState)
    for field_name in _FROZEN_STATE_FIELDS:
        leaf = getattr(state, field_name)
        assert isinstance(leaf, jax.Array), field_name
        assert leaf.shape == (), field_name
        assert leaf.dtype == jnp.float64, field_name


def test_freeze_helper_matches_constructor():
    params = _FIXTURES["finite_beta_full"]
    helper_state = freeze_boozer_analytic_state(**params)
    constructor_state = BoozerAnalyticJAX(**params).frozen_state
    for field_name in _FROZEN_STATE_FIELDS:
        np.testing.assert_allclose(
            np.asarray(getattr(helper_state, field_name)),
            np.asarray(getattr(constructor_state, field_name)),
            rtol=0.0,
            atol=0.0,
        )


# ----------------------------------------------------------------------
# Set-points contract
# ----------------------------------------------------------------------


def test_set_points_returns_self():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    out = wrapper.set_points(np.array([[0.3, 0.5, 0.7]]))
    assert out is wrapper


def test_set_points_round_trip_through_get_points():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qh_symmetric"])
    points = np.array([[0.2, 0.5, 1.0], [0.8, 2.1, 3.5]], dtype=np.float64)
    wrapper.set_points(points)
    np.testing.assert_array_equal(wrapper.get_points(), points)


def test_set_points_rejects_bad_shape():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        wrapper.set_points(np.zeros((4, 2)))


def test_set_points_invalidates_cache():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    pts_a = np.array([[0.3, 0.5, 0.6]], dtype=np.float64)
    pts_b = np.array([[0.7, 1.5, 2.1]], dtype=np.float64)
    wrapper.set_points(pts_a)
    val_a = float(wrapper.modB()[0, 0])
    wrapper.set_points(pts_b)
    val_b = float(wrapper.modB()[0, 0])
    assert val_a != pytest.approx(val_b)


# ----------------------------------------------------------------------
# Parity tests — three fixtures
# ----------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_public_api_matches_cpu(fixture_name):
    cpu, jax_wrapper = _pair(fixture_name)
    _compare_all_methods(cpu, jax_wrapper, _make_points())


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_modB_derivs_bundle_matches_individual_methods(fixture_name):
    cpu, jax_wrapper = _pair(fixture_name)
    points = _make_points()
    cpu.set_points(points)
    jax_wrapper.set_points(points)
    bundle = jax_wrapper.modB_derivs()
    np.testing.assert_allclose(
        bundle,
        cpu.modB_derivs(),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        bundle[:, [0]], cpu.dmodBds(), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        bundle[:, [1]], cpu.dmodBdtheta(), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        bundle[:, [2]], cpu.dmodBdzeta(), rtol=_RTOL, atol=_ATOL
    )


@pytest.mark.parametrize("fixture_name", list(_FIXTURES.keys()))
def test_K_derivs_bundle_matches_individual_methods(fixture_name):
    cpu, jax_wrapper = _pair(fixture_name)
    points = _make_points()
    cpu.set_points(points)
    jax_wrapper.set_points(points)
    bundle = jax_wrapper.K_derivs()
    np.testing.assert_allclose(
        bundle,
        cpu.K_derivs(),
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        bundle[:, [0]], cpu.dKdtheta(), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        bundle[:, [1]], cpu.dKdzeta(), rtol=_RTOL, atol=_ATOL
    )


# ----------------------------------------------------------------------
# Helicity invariants
# ----------------------------------------------------------------------


def test_qa_dmodBdzeta_is_zero():
    """For N=0 (axisymmetric / QA), dmodB/dζ ≡ 0; sibling derivative must be non-trivial.

    Pairing the dζ=0 check with a non-trivial dθ assertion blocks the
    silent-failure mode where every dmodB/d? returns 0 (e.g., a
    broken kernel that always emits `s - s`).
    """
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qa_standard"])
    wrapper.set_points(_make_points())
    np.testing.assert_allclose(
        wrapper.dmodBdzeta(), 0.0, rtol=0.0, atol=_ATOL
    )
    assert np.max(np.abs(wrapper.dmodBdtheta())) > 1e-6, (
        "dmodB/dθ should be non-trivial under QA (etabar != 0); "
        "an all-zero derivative bundle would silently pass the dζ=0 check."
    )


def test_qh_dmodBdtheta_plus_dmodBdzeta_is_zero():
    """For N=1 (QH), dmodB/dθ + dmodB/dζ ≡ 0; both summands must be non-trivial.

    Pairing the cancellation check with non-triviality blocks the
    silent-failure mode where both derivatives are zero everywhere.
    """
    wrapper = BoozerAnalyticJAX(**_FIXTURES["qh_symmetric"])
    wrapper.set_points(_make_points())
    np.testing.assert_allclose(
        wrapper.dmodBdtheta() + wrapper.dmodBdzeta(),
        0.0,
        rtol=0.0,
        atol=_ATOL,
    )
    assert np.max(np.abs(wrapper.dmodBdtheta())) > 1e-6, (
        "dmodB/dθ should be non-trivial under QH (etabar != 0); "
        "an all-zero pair would silently pass the cancellation check."
    )
    assert np.max(np.abs(wrapper.dmodBdzeta())) > 1e-6, (
        "dmodB/dζ should be non-trivial under QH (etabar != 0); "
        "an all-zero pair would silently pass the cancellation check."
    )


# ----------------------------------------------------------------------
# JIT trace + transfer-guard tests
# ----------------------------------------------------------------------


def test_modB_evaluates_under_jit():
    wrapper = BoozerAnalyticJAX(**_FIXTURES["finite_beta_full"])
    points = jnp.asarray(_make_points(), dtype=jnp.float64)

    @jax.jit
    def call_modB(state, pts):
        return _eval_modB(state, pts)

    jitted = call_modB(wrapper.frozen_state, points)
    eager = _eval_modB(wrapper.frozen_state, points)
    np.testing.assert_allclose(
        np.asarray(jitted), np.asarray(eager), rtol=_RTOL, atol=_ATOL
    )


def test_kernels_under_disallow_transfer_guard():
    """All kernels run cleanly under a strict transfer guard."""
    wrapper = BoozerAnalyticJAX(**_FIXTURES["finite_beta_full"])
    points = jnp.asarray(_make_points(), dtype=jnp.float64)
    state = wrapper.frozen_state

    with jax.transfer_guard("disallow"):
        results = [fn(state, points) for fn in _KERNELS]

    for arr in results:
        assert isinstance(arr, jax.Array)
        assert arr.dtype == jnp.float64
        assert arr.shape == (points.shape[0],)


# ----------------------------------------------------------------------
# Rejection path — CPU BoozerAnalytic still rejected by JAX tracing route
# ----------------------------------------------------------------------


def test_generic_cpu_field_is_still_rejected(monkeypatch):
    """``trace_particles_boozer`` JAX route must still reject vanilla CPU fields."""
    from simsopt.field import tracing as tracing_module

    monkeypatch.setattr(tracing_module, "is_jax_backend", lambda: True)

    cpu = BoozerAnalytic(**_FIXTURES["qa_standard"])
    with pytest.raises(NotImplementedError, match=r"BoozerRadialInterpolantJAX"):
        # The rejection is on the *field* argument and happens before any
        # other validation, so it does not matter that the rest of the
        # arguments here would not be a valid trace_particles_boozer call.
        tracing_module.trace_particles_boozer(
            field=cpu,
            stz_inits=np.array([[0.5, 0.0, 0.0]]),
            parallel_speeds=np.array([1e6]),
            tmax=1e-6,
            mass=1.6726219e-27,
            charge=1.602176634e-19,
            Ekin=1e3 * 1.602176634e-19,
            tol=1e-10,
            mode="gc",
        )


# ----------------------------------------------------------------------
# Direct kernel sanity (independent of wrapper boundary)
# ----------------------------------------------------------------------


def test_kernels_return_expected_shape():
    state = freeze_boozer_analytic_state(**_FIXTURES["finite_beta_full"])
    points = jnp.asarray(_make_points(), dtype=jnp.float64)
    for fn in _KERNELS:
        out = fn(state, points)
        assert out.shape == (points.shape[0],), fn.__name__
        assert out.dtype == jnp.float64, fn.__name__
