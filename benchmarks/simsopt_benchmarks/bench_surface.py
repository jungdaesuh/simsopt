"""
Surface Fourier VJP benchmarks.

Targets audit items A1 (leak in VJP parallel region), A2 (missing-braces
angle-recurrence bug on non-XSIMD), and general surface VJP shape.
"""
from __future__ import annotations

import numpy as np

from ._fixtures import make_surface


class SurfaceVJP:
    """All four dgamma* and dgammadash*_by_dcoeff_vjp entry points."""

    params = [["prod", "small"]]
    param_names = ["size"]

    def setup(self, size):
        self.s = make_surface(size)
        rng = np.random.default_rng(2)
        gamma_shape = self.s.gamma().shape
        self.v = rng.standard_normal(gamma_shape)

    def time_dgamma_by_dcoeff_vjp(self, size):
        self.s.dgamma_by_dcoeff_vjp(self.v)

    def time_dgammadash1_by_dcoeff_vjp(self, size):
        self.s.dgammadash1_by_dcoeff_vjp(self.v)

    def time_dgammadash2_by_dcoeff_vjp(self, size):
        self.s.dgammadash2_by_dcoeff_vjp(self.v)
