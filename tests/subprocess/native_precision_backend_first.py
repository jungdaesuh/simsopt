"""Check FP64 precision when the backend is selected before geometry import."""

from __future__ import annotations

import sys

from import_smoke_cases import prefer_local_simsopt_source_tree

prefer_local_simsopt_source_tree()

from simsopt_jax.config import set_backend

case = sys.argv[1]
if case == "native_before_geo":
    set_backend("native_cpu")
else:
    assert case == "fp64_before_geo"
    set_backend("jax_cpu_parity", precision="fp64")

from simsopt.field import Coil, Current
from simsopt.geo import CurveHelical, CurveLength, CurveXYZFourier
from simsopt.geo.curveobjectives import curve_length_pure
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

import jax
import jax.numpy as jnp
import numpy as np

assert jax.config.values["jax_enable_x64"] is True
assert jnp.zeros(1).dtype == jnp.float64
assert curve_length_pure(jnp.ones(8)).dtype == jnp.float64

# Native JAX-backed geometry built after the import is FP64.
helical = CurveHelical(32, 2)
helical_length = CurveLength(helical).J()
assert np.isfinite(helical_length)
np.testing.assert_allclose(
    helical_length,
    np.mean(np.linalg.norm(helical.gammadash(), axis=1)),
    rtol=1e-14,
)

coil_curve = CurveXYZFourier(32, 1)
coil_curve.x = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # xc(1), ys(1)
field = BiotSavartJAX([Coil(coil_curve, Current(1e5))])
field.set_points(np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.05]]))
magnetic_field = np.asarray(jax.device_get(field.B()))
assert magnetic_field.dtype == jnp.float64
assert np.all(np.isfinite(magnetic_field))
