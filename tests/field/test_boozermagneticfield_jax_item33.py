"""Public-wrapper parity tests for ``BoozerRadialInterpolantJAX`` (item 33).

These tests construct a real ``BoozerRadialInterpolant`` from a checked-in
VMEC ``wout`` fixture, freeze its splines into the JAX wrapper, and
compare every public-API method against the CPU oracle at the
``direct_kernel`` parity-ladder tolerance.

Both stellsym and non-stellsym fixtures are exercised. The ``no_K``
branch is covered as a separate construction.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from simsopt.field.boozermagneticfield_jax import (
    BoozerRadialInterpolantFrozenState,
    BoozerRadialInterpolantJAX,
    freeze_boozer_radial_state,
)
import simsopt.jax_core.boozer_radial_field as radial_field
from simsopt.jax_core.boozer_fixed_state import PiecewisePolynomial1D, ppoly_eval
from simsopt.mhd import boozer as boozer_module
from simsopt.mhd.vmec import Vmec


pytestmark = pytest.mark.skipif(
    boozer_module.booz_xform is None,
    reason="booz_xform python package not found",
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = float(_DIRECT_KERNEL["rtol"])
_ATOL = float(_DIRECT_KERNEL["atol"])

_TEST_FILES = (Path(__file__).parent / ".." / "test_files").resolve()
_WOUT_STELLSYM = str((_TEST_FILES / "wout_n3are_R7.75B5.7_lowres.nc").resolve())
_WOUT_ASYM = str((_TEST_FILES / "wout_10x10.nc").resolve())


_API_METHODS_STELLSYM = (
    "modB",
    "dmodBdtheta",
    "dmodBdzeta",
    "dmodBds",
    "K",
    "dKdtheta",
    "dKdzeta",
    "nu",
    "dnudtheta",
    "dnudzeta",
    "dnuds",
    "R",
    "dRdtheta",
    "dRdzeta",
    "dRds",
    "Z",
    "dZdtheta",
    "dZdzeta",
    "dZds",
    "psip",
    "G",
    "I",
    "iota",
    "dGds",
    "dIds",
    "diotads",
)


def _make_evaluation_points(nfp: int) -> np.ndarray:
    s = np.array([0.05, 0.18, 0.37, 0.52, 0.68, 0.81, 0.94], dtype=np.float64)
    theta = np.array([0.1, 1.05, 2.4, 3.27, 4.18, 5.1, 6.05], dtype=np.float64)
    zeta_max = 2.0 * np.pi / max(nfp, 1)
    zeta = np.linspace(0.02 * zeta_max, 0.96 * zeta_max, 7)
    return np.stack([s, theta, zeta], axis=1)


def _constant_scalar_profile(value: float) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=jnp.array([0.0, 1.0], dtype=jnp.float64),
        coeffs=jnp.array([[value]], dtype=jnp.float64),
    )


def _constant_mode_profile(values: list[float]) -> PiecewisePolynomial1D:
    return PiecewisePolynomial1D(
        breaks=jnp.array([0.0, 1.0], dtype=jnp.float64),
        coeffs=jnp.asarray(values, dtype=jnp.float64)[:, None, None],
    )


def _synthetic_radial_wrapper() -> BoozerRadialInterpolantJAX:
    def zero_modes() -> PiecewisePolynomial1D:
        return _constant_mode_profile([0.0, 0.0])

    state = BoozerRadialInterpolantFrozenState(
        xm=jnp.array([0.0, 1.0], dtype=jnp.float64),
        xn=jnp.array([0.0, 1.0], dtype=jnp.float64),
        psip=_constant_scalar_profile(1.0),
        G=_constant_scalar_profile(2.0),
        I=_constant_scalar_profile(0.2),
        iota=_constant_scalar_profile(0.4),
        dGds=_constant_scalar_profile(0.03),
        dIds=_constant_scalar_profile(0.04),
        diotads=_constant_scalar_profile(0.05),
        bmnc=_constant_mode_profile([1.1, 0.2]),
        dbmncds=_constant_mode_profile([0.01, 0.02]),
        rmnc=_constant_mode_profile([1.4, 0.05]),
        drmncds=_constant_mode_profile([0.02, 0.01]),
        zmns=_constant_mode_profile([0.0, 0.08]),
        dzmnsds=_constant_mode_profile([0.0, 0.01]),
        numns=_constant_mode_profile([0.0, 0.03]),
        dnumnsds=_constant_mode_profile([0.0, 0.004]),
        bmns=zero_modes(),
        dbmnsds=zero_modes(),
        rmns=zero_modes(),
        drmnsds=zero_modes(),
        zmnc=zero_modes(),
        dzmncds=zero_modes(),
        numnc=zero_modes(),
        dnumncds=zero_modes(),
        mn_factor=_constant_mode_profile([1.0, 1.0]),
        d_mn_factor=zero_modes(),
        kmns=_constant_mode_profile([0.0, 0.06]),
        kmnc=zero_modes(),
        stellsym=True,
        no_K=False,
    )
    return BoozerRadialInterpolantJAX.from_frozen_state(state, psi0=1.0, nfp=1)


@pytest.fixture(scope="module")
def stellsym_bri_and_jax():
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(vmec, order=3, mpol=4, ntor=4, rescale=True)
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


@pytest.fixture(scope="module")
def stellsym_no_K_bri_and_jax():
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(
        vmec, order=3, mpol=4, ntor=4, rescale=True, no_K=True
    )
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


@pytest.fixture(scope="module")
def stellsym_enforce_vacuum_bri_and_jax():
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(
        vmec, order=3, mpol=4, ntor=4, rescale=True, enforce_vacuum=True
    )
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


@pytest.fixture(scope="module")
def stellsym_enforce_qs_bri_and_jax():
    vmec = Vmec(_WOUT_STELLSYM)
    bri = BoozerRadialInterpolant(vmec, order=3, mpol=4, ntor=4, rescale=False, N=0)
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


@pytest.fixture(scope="module")
def asym_bri_and_jax():
    vmec = Vmec(_WOUT_ASYM)
    bri = BoozerRadialInterpolant(vmec, order=3, mpol=4, ntor=4, rescale=True)
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


def _compare_all_methods(bri, wrapper, points, method_names):
    bri.set_points(points)
    wrapper.set_points(points)
    for name in method_names:
        cpu_value = getattr(bri, name)()
        jax_value = getattr(wrapper, name)()
        np.testing.assert_allclose(
            jax_value, cpu_value, rtol=_RTOL, atol=_ATOL, err_msg=name
        )


def _toroidal_covariant_component(field, points):
    field.set_points(points)
    R = np.ravel(field.R())
    dRdtheta = np.ravel(field.dRdtheta())
    dRdzeta = np.ravel(field.dRdzeta())
    dZdtheta = np.ravel(field.dZdtheta())
    dZdzeta = np.ravel(field.dZdzeta())
    nu = np.ravel(field.nu())
    dnudtheta = np.ravel(field.dnudtheta())
    dnudzeta = np.ravel(field.dnudzeta())
    modB = np.ravel(field.modB())
    G = np.ravel(field.G())
    I = np.ravel(field.I())
    iota = np.ravel(field.iota())

    phi = points[:, 2] - nu
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    dphidtheta = -dnudtheta
    dphidzeta = 1.0 - dnudzeta
    x_theta = np.stack(
        (
            dRdtheta * cos_phi - R * sin_phi * dphidtheta,
            dRdtheta * sin_phi + R * cos_phi * dphidtheta,
            dZdtheta,
        ),
        axis=1,
    )
    x_zeta = np.stack(
        (
            dRdzeta * cos_phi - R * sin_phi * dphidzeta,
            dRdzeta * sin_phi + R * cos_phi * dphidzeta,
            dZdzeta,
        ),
        axis=1,
    )
    sqrtg = (G + iota * I) / (modB * modB)
    B_contravariant = (x_zeta + iota[:, None] * x_theta) / sqrtg[:, None]
    return np.einsum("ij,ij->i", B_contravariant, x_zeta), G


def test_freeze_boozer_radial_state_returns_jax_pytree(stellsym_bri_and_jax):
    bri, _ = stellsym_bri_and_jax
    state = freeze_boozer_radial_state(bri)
    assert isinstance(state, BoozerRadialInterpolantFrozenState)
    leaves, _ = jax.tree.flatten(state)
    for leaf in leaves:
        assert isinstance(leaf, jax.Array), type(leaf)


def test_wrapper_exposes_metadata_matching_upstream(stellsym_bri_and_jax):
    bri, wrapper = stellsym_bri_and_jax
    assert wrapper.psi0 == pytest.approx(bri.psi0)
    assert wrapper.stellsym is True
    assert wrapper.nfp == int(bri.booz.bx.nfp)
    assert wrapper.no_K is False


def test_mn_factor_extrapolation_below_first_retained_knot(stellsym_bri_and_jax):
    """Frozen state preserves upstream inverse-power mn_factor extrapolation."""

    bri, wrapper = stellsym_bri_and_jax
    first_knot = float(bri.mn_factor_splines[0].get_knots()[0])
    s_probe = np.asarray([0.5 * first_knot], dtype=np.float64)

    expected = np.vstack([spline(s_probe) for spline in bri.mn_factor_splines])
    actual = np.asarray(
        ppoly_eval(wrapper._frozen_state.mn_factor, jnp.asarray(s_probe))
    )

    np.testing.assert_allclose(actual, expected, rtol=_RTOL, atol=_ATOL)


def test_set_points_round_trips_through_get_points(stellsym_bri_and_jax):
    _, wrapper = stellsym_bri_and_jax
    points = np.array([[0.2, 0.5, 1.0], [0.8, 2.1, 3.5]])
    wrapper.set_points(points)
    np.testing.assert_array_equal(wrapper.get_points(), points)


def test_wrapper_restart_payload_round_trips_public_state(stellsym_bri_and_jax):
    bri, wrapper = stellsym_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    wrapper.set_points(points)

    payload = wrapper.as_dict(serial_objs_dict={})
    restored = BoozerRadialInterpolantJAX.from_dict(payload, {}, {})

    assert restored.psi0 == pytest.approx(wrapper.psi0)
    assert restored.nfp == wrapper.nfp
    assert restored.stellsym is wrapper.stellsym
    assert restored.no_K is wrapper.no_K
    np.testing.assert_array_equal(restored.get_points(), wrapper.get_points())

    for name in ("modB", "K", "nu", "R", "Z", "psip", "G", "I", "iota"):
        np.testing.assert_allclose(
            getattr(restored, name)(),
            getattr(wrapper, name)(),
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=name,
        )


def test_set_points_rejects_bad_shape(stellsym_bri_and_jax):
    _, wrapper = stellsym_bri_and_jax
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        wrapper.set_points(np.zeros((4, 2)))


def test_stellsym_public_api_matches_cpu(stellsym_bri_and_jax):
    bri, wrapper = stellsym_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)


def test_stellsym_no_K_public_api_matches_cpu(stellsym_no_K_bri_and_jax):
    bri, wrapper = stellsym_no_K_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)
    # K and its derivatives must be exactly zero.
    np.testing.assert_array_equal(wrapper.K(), np.zeros_like(wrapper.K()))
    np.testing.assert_array_equal(wrapper.dKdtheta(), np.zeros_like(wrapper.dKdtheta()))
    np.testing.assert_array_equal(wrapper.dKdzeta(), np.zeros_like(wrapper.dKdzeta()))


def test_enforce_vacuum_public_api_matches_cpu(stellsym_enforce_vacuum_bri_and_jax):
    bri, wrapper = stellsym_enforce_vacuum_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)
    assert wrapper.no_K is True


def test_enforce_qs_public_api_matches_cpu(stellsym_enforce_qs_bri_and_jax):
    bri, wrapper = stellsym_enforce_qs_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)


def test_modB_derivs_bundle_matches_individual_methods(stellsym_bri_and_jax):
    bri, wrapper = stellsym_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    bri.set_points(points)
    wrapper.set_points(points)
    bundle = wrapper.modB_derivs()
    np.testing.assert_allclose(
        bundle[:, [0]], wrapper.dmodBds(), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        bundle[:, [1]], wrapper.dmodBdtheta(), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        bundle[:, [2]], wrapper.dmodBdzeta(), rtol=_RTOL, atol=_ATOL
    )


def test_set_points_invalidates_cache(stellsym_bri_and_jax):
    _, wrapper = stellsym_bri_and_jax
    points_a = np.array([[0.3, 0.5, 0.6]], dtype=np.float64)
    wrapper.set_points(points_a)
    val_a = float(wrapper.modB()[0, 0])
    points_b = np.array([[0.7, 1.5, 2.1]], dtype=np.float64)
    wrapper.set_points(points_b)
    val_b = float(wrapper.modB()[0, 0])
    assert val_a != val_b


def test_radial_columns_cached_once_per_points_cycle(monkeypatch):
    wrapper = _synthetic_radial_wrapper()
    state = wrapper.frozen_state
    points = _make_evaluation_points(wrapper.nfp)

    column_calls: Counter[int] = Counter()
    scalar_calls: Counter[int] = Counter()
    original_column_at = radial_field._column_at
    original_scalar_at = radial_field._scalar_at

    def counting_column_at(s, profile):
        column_calls[id(profile)] += 1
        return original_column_at(s, profile)

    def counting_scalar_at(s, profile):
        scalar_calls[id(profile)] += 1
        return original_scalar_at(s, profile)

    monkeypatch.setattr(radial_field, "_column_at", counting_column_at)
    monkeypatch.setattr(radial_field, "_scalar_at", counting_scalar_at)

    wrapper.set_points(points)
    wrapper.modB()
    wrapper.dmodBdtheta()
    wrapper.dmodBdzeta()
    wrapper.dmodBds()
    wrapper.G()
    wrapper.I()
    wrapper.iota()

    assert column_calls[id(state.mn_factor)] == 1
    assert column_calls[id(state.bmnc)] == 1
    assert scalar_calls[id(state.G)] == 1
    assert scalar_calls[id(state.I)] == 1
    assert scalar_calls[id(state.iota)] == 1
    assert max(column_calls.values()) == 1
    assert max(scalar_calls.values()) == 1

    cached_column_calls = column_calls.copy()
    cached_scalar_calls = scalar_calls.copy()
    wrapper.modB()
    wrapper.G()
    assert column_calls == cached_column_calls
    assert scalar_calls == cached_scalar_calls

    wrapper.set_points(points + np.array([0.01, 0.0, 0.0]))
    wrapper.modB()
    assert column_calls[id(state.mn_factor)] == 2
    assert scalar_calls[id(state.G)] == 2


def test_post_construction_psi0_mutation_does_not_change_wrapper(stellsym_bri_and_jax):
    """The JAX wrapper is an immutable snapshot of psi0-scaled K splines."""

    bri, wrapper = stellsym_bri_and_jax
    original_psi0 = float(bri.psi0)
    points = _make_evaluation_points(bri.booz.bx.nfp)
    wrapper.set_points(points)
    K_before = np.asarray(wrapper.K())
    try:
        bri.psi0 = original_psi0 * 1.25
        assert wrapper.psi0 == pytest.approx(original_psi0)
        np.testing.assert_array_equal(np.asarray(wrapper.K()), K_before)
    finally:
        bri.psi0 = original_psi0


def test_post_construction_flag_and_mode_mutation_does_not_change_wrapper(
    stellsym_bri_and_jax,
):
    """Frozen state ignores later CPU flag and mode-table mutation."""

    bri, wrapper = stellsym_bri_and_jax
    points = _make_evaluation_points(bri.booz.bx.nfp)
    wrapper.set_points(points)
    modB_before = np.asarray(wrapper.modB())
    K_before = np.asarray(wrapper.K())
    original = {
        "enforce_qs": bri.enforce_qs,
        "enforce_vacuum": bri.enforce_vacuum,
        "N": getattr(bri, "N", None),
        "no_K": bri.no_K,
        "xm_b": bri.xm_b,
        "xn_b": bri.xn_b,
    }
    try:
        bri.enforce_qs = True
        bri.enforce_vacuum = True
        bri.N = 99
        bri.no_K = True
        bri.xm_b = np.asarray(bri.xm_b).copy()
        bri.xn_b = np.asarray(bri.xn_b).copy()
        bri.xm_b[0] = bri.xm_b[0] + 10
        bri.xn_b[0] = bri.xn_b[0] + 10

        assert wrapper.no_K is False
        np.testing.assert_array_equal(np.asarray(wrapper.modB()), modB_before)
        np.testing.assert_array_equal(np.asarray(wrapper.K()), K_before)
    finally:
        bri.enforce_qs = original["enforce_qs"]
        bri.enforce_vacuum = original["enforce_vacuum"]
        if original["N"] is None and hasattr(bri, "N"):
            delattr(bri, "N")
        elif original["N"] is not None:
            bri.N = original["N"]
        bri.no_K = original["no_K"]
        bri.xm_b = original["xm_b"]
        bri.xn_b = original["xn_b"]


def test_asym_public_api_matches_cpu(asym_bri_and_jax):
    bri, wrapper = asym_bri_and_jax
    assert wrapper.stellsym is False
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)


@pytest.mark.parametrize(
    ("fixture_name", "max_relative_residual"),
    [
        ("stellsym_bri_and_jax", 1.5e-2),
        ("asym_bri_and_jax", 3.5e-2),
    ],
)
def test_covariant_toroidal_identity_matches_G(
    request,
    fixture_name,
    max_relative_residual,
):
    """Low-res BOOZXFORM fixtures keep B·∂x/∂ζ close to covariant G(s)."""
    bri, wrapper = request.getfixturevalue(fixture_name)
    points = _make_evaluation_points(bri.booz.bx.nfp)
    cpu_component, cpu_G = _toroidal_covariant_component(bri, points)
    jax_component, jax_G = _toroidal_covariant_component(wrapper, points)

    np.testing.assert_allclose(jax_component, cpu_component, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(jax_G, cpu_G, rtol=_RTOL, atol=_ATOL)
    assert np.max(np.abs((cpu_component - cpu_G) / cpu_G)) <= max_relative_residual


def test_wrapper_is_not_exported_via_field_namespace_lazy_init():
    # Regression target: eager export of ``BoozerRadialInterpolantJAX`` via
    # ``simsopt.field.__all__`` would trigger eager import of the JAX-only
    # module, breaking CPU-only installs that do not have JAX available (see
    # CLAUDE.md "No simsoptpp dependency" pattern — same constraint applies
    # in reverse for JAX-only wrappers consumed from a CPU-only env).
    import simsopt.field as field_pkg

    assert "BoozerRadialInterpolantJAX" not in getattr(field_pkg, "__all__", ())


def test_wrapper_has_no_dofs(stellsym_bri_and_jax):
    # Regression target: ``BoozerRadialInterpolantJAX`` is a frozen evaluator,
    # not an optimizable. Inadvertently exposing DOFs (e.g. via a stray
    # ``Optimizable`` base or default DOF registration) would silently change
    # the decision-vector size of any Stage 2 composite that includes it,
    # corrupting the optimizer state without raising.
    _, wrapper = stellsym_bri_and_jax
    assert wrapper.local_full_x.size == 0
    assert wrapper.full_x.size == 0
