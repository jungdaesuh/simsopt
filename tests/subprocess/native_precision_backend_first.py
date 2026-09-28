"""Check JAX backend precision when selected before geometry import."""

from __future__ import annotations

import sys

from import_smoke_cases import prefer_local_simsopt_source_tree

prefer_local_simsopt_source_tree()

from simsopt_jax.config import set_backend

case = sys.argv[1]
if case == "smoke_before_geo":
    set_backend("jax_cpu_float32_smoke")
else:
    assert case == "fp64_before_geo"
    set_backend("jax_cpu_parity", precision="fp64")

from simsopt.geo.curveobjectives import curve_length_pure

import jax
import jax.numpy as jnp

expected_x64 = case == "fp64_before_geo"
assert jax.config.jax_enable_x64 is expected_x64
assert jnp.zeros(1).dtype == (jnp.float64 if expected_x64 else jnp.float32)
assert curve_length_pure(jnp.ones(8)).dtype == (
    jnp.float64 if expected_x64 else jnp.float32
)
