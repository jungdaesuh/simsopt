"""Check native precision when geometry is imported before backend selection."""

from __future__ import annotations

import sys

from import_smoke_cases import prefer_local_simsopt_source_tree

prefer_local_simsopt_source_tree()

from simsopt.geo.curveobjectives import curve_length_pure

import jax
import jax.numpy as jnp
from simsopt_jax.config import set_backend

case = sys.argv[1]
assert case in ("no_selector", "native_cpu", "explicit_x64_off", "geo_before_smoke")
expected_x64 = case != "explicit_x64_off"
assert jax.config.jax_enable_x64 is expected_x64
assert curve_length_pure(jnp.ones(8)).dtype == (
    jnp.float64 if expected_x64 else jnp.float32
)

if case == "geo_before_smoke":
    set_backend("jax_cpu_float32_smoke")
    assert jax.config.jax_enable_x64 is False
    assert jnp.zeros(1).dtype == jnp.float32
