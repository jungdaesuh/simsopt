"""
Peak-RSS regression guard.

Targets audit item A1 (heap leaks in surface Fourier VJPs). Runs N
consecutive VJP calls and records peak RSS. Used to assert "RSS does
not grow unboundedly over a long optimization run".
"""
from __future__ import annotations

import os
import resource

import numpy as np

from ._fixtures import make_surface


def _peak_rss_mib() -> float:
    """Peak resident set size in MiB.

    getrusage reports KB on Linux and bytes on macOS (per its man page).
    We pick the larger of rusage and os.getpid()/proc to handle both.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: KB. macOS: bytes.
    if ru > 10 * 1024 * 1024:  # heuristic: probably bytes
        return ru / (1024 ** 2)
    return ru / 1024


class LongRunRSS:
    """Run many VJP calls and record peak RSS.

    A1 manifests as peak RSS growing roughly linearly with ``n_iters``.
    Post-fix, RSS plateaus.
    """

    params = [[100, 500]]
    param_names = ["n_iters"]
    timeout = 600

    def setup(self, n_iters):
        self.s = make_surface("small")
        rng = np.random.default_rng(3)
        self.v = rng.standard_normal(self.s.gamma().shape)

    def track_peak_rss_mib(self, n_iters):
        for _ in range(n_iters):
            self.s.dgamma_by_dcoeff_vjp(self.v)
        return _peak_rss_mib()
