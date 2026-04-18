"""
Biot–Savart forward and VJP benchmarks.

Targets audit items B1 (parallelism axis), B2 (VJP inner-loop reduction),
B4 (serial coil reduction tail), B5 (cache redesign), B6 (point-buffer
reuse), B10 (Python VJP waste).
"""
from __future__ import annotations

import numpy as np

from ._fixtures import make_coils, output_fingerprint


class BiotSavartForward:
    """Forward evaluation of B (and optionally dB, ddB)."""

    params = (["prod", "small"], [0, 1, 2])
    param_names = ["size", "derivs"]

    def setup(self, size, derivs):
        self.ctx = make_coils(size)
        self.ctx["bs"].set_points(self.ctx["points"])

    def time_compute(self, size, derivs):
        self.ctx["bs"].compute(derivs)

    def track_fingerprint(self, size, derivs):
        bs = self.ctx["bs"]
        bs.compute(derivs)
        fp = output_fingerprint(np.asarray(bs.B()))
        if derivs >= 1:
            fp += output_fingerprint(np.asarray(bs.dB_by_dX()))
        if derivs >= 2:
            fp += output_fingerprint(np.asarray(bs.d2B_by_dXdX()))
        return fp


class BiotSavartVJP:
    """B_vjp and B_and_dB_vjp — the gradient path for coil optimization."""

    params = [["prod", "small"]]
    param_names = ["size"]

    def setup(self, size):
        self.ctx = make_coils(size)
        self.ctx["bs"].set_points(self.ctx["points"])
        rng = np.random.default_rng(1)
        self.v = rng.standard_normal(self.ctx["points"].shape)
        self.vgrad = rng.standard_normal(
            (self.ctx["points"].shape[0], 3, 3)
        )

    def time_B_vjp(self, size):
        self.ctx["bs"].B_vjp(self.v)

    def time_B_and_dB_vjp(self, size):
        self.ctx["bs"].B_and_dB_vjp(self.v, self.vgrad)
