"""Public-wrapper parity tests for ``BoozerRadialInterpolantJAX`` (item 33).

These tests construct a real ``BoozerRadialInterpolant`` from a checked-in
VMEC ``wout`` fixture, freeze its splines into the JAX wrapper, and
compare every public-API method against the CPU oracle at the
``direct_kernel`` parity-ladder tolerance.

Both stellsym and non-stellsym fixtures are exercised. The ``no_K``
branch is covered as a separate construction.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from simsopt.field.boozermagneticfield_jax import (
    BoozerRadialInterpolantFrozenState,
    BoozerRadialInterpolantJAX,
    freeze_boozer_radial_state,
)
from simsopt.mhd.vmec import Vmec


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
def asym_bri_and_jax():
    vmec = Vmec(_WOUT_ASYM)
    bri = BoozerRadialInterpolant(vmec, order=3, mpol=4, ntor=4, rescale=True)
    wrapper = BoozerRadialInterpolantJAX(bri)
    return bri, wrapper


def _compare_all_methods(bri, wrapper, points, method_names):
    bri.set_points(points)
    wrapper.set_points(points)
    failures = []
    for name in method_names:
        cpu_value = getattr(bri, name)()
        jax_value = getattr(wrapper, name)()
        try:
            np.testing.assert_allclose(
                jax_value, cpu_value, rtol=_RTOL, atol=_ATOL, err_msg=name
            )
        except AssertionError as exc:
            failures.append((name, str(exc)))
    if failures:
        report = "\n".join(f"{name}: {msg.splitlines()[0]}" for name, msg in failures)
        raise AssertionError(f"Method parity failures:\n{report}")


def test_freeze_boozer_radial_state_returns_pytree(stellsym_bri_and_jax):
    bri, _ = stellsym_bri_and_jax
    state = freeze_boozer_radial_state(bri)
    assert isinstance(state, BoozerRadialInterpolantFrozenState)
    leaves, _ = jax.tree_util.tree_flatten(state)
    # All leaves must be JAX arrays (not Python or NumPy objects).
    for leaf in leaves:
        assert isinstance(leaf, jax.Array), type(leaf)


def test_wrapper_exposes_metadata_matching_upstream(stellsym_bri_and_jax):
    bri, wrapper = stellsym_bri_and_jax
    assert wrapper.psi0 == pytest.approx(bri.psi0)
    assert wrapper.stellsym is True
    assert wrapper.nfp == int(bri.booz.bx.nfp)
    assert wrapper.no_K is False


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


def test_asym_public_api_matches_cpu(asym_bri_and_jax):
    bri, wrapper = asym_bri_and_jax
    assert wrapper.stellsym is False
    points = _make_evaluation_points(bri.booz.bx.nfp)
    _compare_all_methods(bri, wrapper, points, _API_METHODS_STELLSYM)


def test_wrapper_is_not_exported_via_field_namespace_lazy_init():
    # Item 33 deliberately does not auto-register the wrapper in the
    # ``simsopt.field`` lazy-export map. Users must import the module
    # path explicitly. This test guards against accidental promotion.
    import simsopt.field as field_pkg

    assert "BoozerRadialInterpolantJAX" not in getattr(field_pkg, "__all__", ())


def test_wrapper_has_no_dofs(stellsym_bri_and_jax):
    _, wrapper = stellsym_bri_and_jax
    assert wrapper.local_full_x.size == 0
    assert wrapper.full_x.size == 0
