"""Check native precision when geometry is imported before backend selection."""

from __future__ import annotations

import sys

from import_smoke_cases import prefer_local_simsopt_source_tree

prefer_local_simsopt_source_tree()

from simsopt.field import Coil, Current
from simsopt.geo import CurveHelical, CurveLength, CurveXYZFourier
from simsopt.geo.curveobjectives import curve_length_pure

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.config import set_backend
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX


def _assert_native_and_field_precision(helical, dtype, rtol) -> None:
    """CurveLength of a JAX-backed native curve and a BiotSavartJAX field."""
    helical_length = CurveLength(helical).J()
    assert np.isfinite(helical_length)
    np.testing.assert_allclose(
        helical_length,
        np.mean(np.linalg.norm(helical.gammadash(), axis=1)),
        rtol=rtol,
    )
    coil_curve = CurveXYZFourier(32, 1)
    coil_curve.x = np.array(
        [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    )  # xc(1), ys(1)
    field = BiotSavartJAX([Coil(coil_curve, Current(1e5)), Coil(helical, Current(2e4))])
    field.set_points(np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.05]]))
    magnetic_field = np.asarray(jax.device_get(field.B()))
    assert magnetic_field.dtype == dtype
    assert np.all(np.isfinite(magnetic_field))


case = sys.argv[1]
assert case in (
    "no_selector",
    "native_cpu",
    "explicit_x64_off",
    "curve_before_parity",
)
expected_x64 = case != "explicit_x64_off"
assert jax.config.values["jax_enable_x64"] is expected_x64
assert curve_length_pure(jnp.ones(8)).dtype == (
    jnp.float64 if expected_x64 else jnp.float32
)

if case == "native_cpu":
    set_backend("native_cpu")
    assert jax.config.values["jax_enable_x64"] is True
    _assert_native_and_field_precision(CurveHelical(32, 2), jnp.float64, 1e-14)

if case == "curve_before_parity":
    # A JAX-backed native curve built before the parity backend is configured
    # (``SIMSOPT_BACKEND_MODE`` already names the parity mode) must not capture
    # a precision that configuration later changes.
    helical = CurveHelical(32, 2)
    set_backend("jax", device="cpu", intent="parity")
    assert jax.config.values["jax_enable_x64"] is True
    _assert_native_and_field_precision(helical, jnp.float64, 1e-14)
