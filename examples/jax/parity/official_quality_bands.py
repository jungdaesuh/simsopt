"""Endpoint quality bands of the official path-dependent mirrors, derived from upstream's own end-point scatter.

Rule (pre-registered 2026-09-20 before any sample existed; amended the same day ONLY by dropping the branch-native
value from the sample set, which can only tighten it): for a mirror whose lanes are proven to be the same function and
gradient as upstream at every official state, but whose optimizer path forks at round-off level, the ``native_default``
verdict is an endpoint quality band whose ceiling comes from UPSTREAM evidence only::

    S       = upstream's nine end values under one-ulp start perturbations (tracked sensitivity record, k = 0..8)
    ceiling = max(S) * (1 + (max(S) - min(S)) / min(S))

No branch value, native or JAX, enters the ceiling. The band is a gross-failure guard: with nine samples it cannot
decide differences of a few tenths of a percent, and under a band the arbiter makes lane comparisons informational, so
the mirror claim of a band case rests on its tracked official-state replay test (same function and gradient at
upstream's recorded states). Deterministic algorithms (GPMO, GSCO, RCLS, the tiny least-squares cases) get no band.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

from examples.jax.parity.contracts import QualityBand
from examples.jax.parity.official_reference import (
    load_official_sensitivity,
    official_sensitivity_case_ids,
)


def upstream_scatter_ceiling(samples: Sequence[float]) -> float:
    """``max(S) * (1 + (max(S) - min(S)) / min(S))`` for positive finite upstream end values ``S``."""
    if len(samples) < 2:
        raise ValueError("an upstream-scatter ceiling needs at least two samples")
    if not all(math.isfinite(sample) and sample > 0.0 for sample in samples):
        raise ValueError("upstream end values must be finite and positive")
    low, high = min(samples), max(samples)
    return high * (1.0 + (high - low) / low)


def official_quality_band(case_id: str) -> QualityBand:
    """The ``native_default`` band of one official case, computed from the tracked upstream record."""
    sensitivity = load_official_sensitivity(case_id)
    samples = sensitivity.end_values
    return QualityBand(
        observable=sensitivity.lane_observable,
        max_value=upstream_scatter_ceiling(samples),
        derivation=(
            f"rule v2 of 2026-09-20: upstream's {len(samples)} end values of "
            f"{sensitivity.observable} (lane key {sensitivity.lane_observable}) under one-ulp start perturbations (tracked "
            f"sensitivity record, min {min(samples)!r}, max {max(samples)!r}); "
            "ceiling = max(S) * (1 + (max(S) - min(S)) / min(S)); gross-failure guard"
        ),
        scale="native_default",
    )


OFFICIAL_BAND_CASE_IDS: Final[tuple[str, ...]] = official_sensitivity_case_ids()
