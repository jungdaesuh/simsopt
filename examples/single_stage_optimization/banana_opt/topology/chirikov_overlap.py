"""Chirikov island-overlap parameter ``s`` for a resonance chain ladder.

Two neighbouring island chains overlap — and the KAM surface between them is
destroyed — when their half-widths sum to more than the radial distance between
their resonance centres. Chirikov's overlap parameter quantifies this:

.. math::
   s_{m,m+1} = \\frac{W_m/2 + W_{m+1}/2}{|r_{m+1} - r_m|},

where ``r_m`` is the radial label of the ``p/q`` resonance (a
:class:`~banana_opt.topology.iota_profile.RationalCrossing`) and ``W_m`` is the
*full* radial width of its island. ``s >= 1`` is the classical overlap (chaos)
criterion; the resonance-overlap estimate is known to over-predict the onset, so
the refined "two-thirds rule" uses ``s >= 2/3`` (see
:data:`CHIRIKOV_TWO_THIRDS_THRESHOLD`). The aggregate over a ladder of chains is
either the ``max`` (a sharp diagnostic) or a smooth log-sum-exp surrogate (for
use as an optimisation objective, where a sub-gradient-free maximum would stall).

**Island width source.** In a region of finite magnetic shear the pendulum
approximation gives the full width

.. math::  W = C\\,\\sqrt{|A| / |s'|},  \\qquad C = 4,

with ``A`` the resonant perturbation amplitude and ``s' = d\\iota/d\\text{label}``
the local shear (the signed
:attr:`~banana_opt.topology.iota_profile.RationalCrossing.local_shear`). This
``1/\\sqrt{s'}`` **diverges at a shearless curve** (``s' -> 0``), exactly the
nontwist regime of the banana-coil edge. There the physical island pair does not
blow up — the twin chains reconnect — so when ``|s'|`` falls below a configurable
floor the width is taken from the *measured* O/X radial separation of the twin
periodic-orbit pair (:func:`solve_periodic_orbit`), which stays finite. This is
the nontwist handling the design calls for: never evaluate ``1/\\sqrt{0}``.

The pure core (:func:`chirikov_overlap_pairwise`, :func:`chirikov_overlap_scalar`)
is calibrated against the standard map analytically: with the analytic half-widths
of two unit-spaced resonances summing to the spacing, ``s == 1`` exactly. It is
independent of any field machinery and is what the unit tests pin down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log, sqrt

import numpy as np

__all__ = [
    "CHIRIKOV_OVERLAP_THRESHOLD",
    "CHIRIKOV_TWO_THIRDS_THRESHOLD",
    "PENDULUM_WIDTH_COEFFICIENT",
    "DEFAULT_SHEAR_FLOOR",
    "DEFAULT_LSE_BETA",
    "ChirikovPair",
    "ChirikovOverlap",
    "pendulum_full_width",
    "island_full_width",
    "chirikov_overlap_from_crossings",
    "chirikov_overlap_pairwise",
    "chirikov_overlap_scalar",
]

# Geometric overlap onset: islands touch when half-widths sum to the spacing.
CHIRIKOV_OVERLAP_THRESHOLD = 1.0
# Refined onset accounting for the chaotic layer / higher-order islands that the
# bare two-resonance estimate ignores (Chirikov's "two-thirds rule").
CHIRIKOV_TWO_THIRDS_THRESHOLD = 2.0 / 3.0
# Pendulum separatrix FULL width W = C*sqrt(|A|/|s'|); C = 4 for H = (s'/2) dI^2
# + A cos(theta) (max excursion |dI| = 2 sqrt(|A|/|s'|), full width twice that).
PENDULUM_WIDTH_COEFFICIENT = 4.0
# Below this |local_shear| the pendulum 1/sqrt(shear) is abandoned for the
# measured O/X separation (nontwist / shearless band). Tunable, not hard-coded.
DEFAULT_SHEAR_FLOOR = 1.0e-3
# Sharpness of the log-sum-exp max surrogate; larger -> closer to the hard max.
DEFAULT_LSE_BETA = 50.0


@dataclass(frozen=True, slots=True)
class ChirikovPair:
    """Overlap of one adjacent resonance pair on the radial ladder.

    ``inner_label``/``outer_label`` are the two resonance radial labels
    (``inner_label < outer_label``); ``inner_width``/``outer_width`` their island
    *full* widths; ``overlap`` the dimensionless ``s = (W_in/2 + W_out/2)/spacing``.
    """

    inner_label: float
    outer_label: float
    inner_width: float
    outer_width: float
    spacing: float
    overlap: float


@dataclass(frozen=True, slots=True)
class ChirikovOverlap:
    """Aggregate island-overlap result over a resonance ladder."""

    scalar: float
    aggregate: str
    pairs: tuple[ChirikovPair, ...]

    @property
    def overlapping(self) -> bool:
        """Whether the classical ``s >= 1`` criterion is met anywhere."""
        return self.scalar >= CHIRIKOV_OVERLAP_THRESHOLD


def pendulum_full_width(
    amplitude: float,
    shear: float,
    *,
    coefficient: float = PENDULUM_WIDTH_COEFFICIENT,
) -> float:
    r"""Pendulum-approximation island FULL width ``C*sqrt(|amplitude|/|shear|)``.

    ``amplitude`` is the resonant perturbation strength ``A`` and ``shear`` the
    local ``d\iota/d\text{label}``. Raises if ``shear`` is zero (the caller must
    route the shearless case through a measured width — see
    :func:`island_full_width`); raises on non-finite inputs so a bad upstream
    value can never silently produce ``inf``/``nan`` overlap.
    """
    a = float(amplitude)
    s = float(shear)
    if not (isfinite(a) and isfinite(s)):
        raise ValueError(f"pendulum_full_width needs finite inputs; got A={a}, shear={s}")
    if a < 0.0:
        raise ValueError(f"pendulum_full_width needs a non-negative amplitude; got {a}")
    if s == 0.0:
        raise ZeroDivisionError(
            "pendulum_full_width diverges at zero shear; route the shearless "
            "band through a measured O/X separation (island_full_width)."
        )
    return float(coefficient) * sqrt(a / abs(s))


def island_full_width(
    *,
    shear: float,
    amplitude: float | None = None,
    measured_full_width: float | None = None,
    shear_floor: float = DEFAULT_SHEAR_FLOOR,
    coefficient: float = PENDULUM_WIDTH_COEFFICIENT,
) -> float:
    r"""Island full width, pendulum in the twist region, measured at shearless.

    Routing:

    - ``|shear| < shear_floor`` (the nontwist / shearless band, where
      ``1/\sqrt{shear}`` blows up): the measured O/X twin-pair separation
      ``measured_full_width`` is used; it MUST be supplied and finite there,
      otherwise ``ValueError`` (we never fabricate a width).
    - ``|shear| >= shear_floor``: a supplied finite ``measured_full_width``
      overrides; otherwise the pendulum estimate from ``amplitude`` is used (and
      ``amplitude`` must then be supplied).
    """
    s = float(shear)
    floor = float(shear_floor)
    if not isfinite(s):
        raise ValueError(f"island_full_width needs finite shear; got {s}")
    if floor < 0.0:
        raise ValueError(f"shear_floor must be non-negative; got {floor}")
    measured_given = measured_full_width is not None
    if measured_given and not isfinite(float(measured_full_width)):
        raise ValueError(
            f"measured_full_width must be finite; got {measured_full_width!r}"
        )
    if abs(s) < floor:
        if not measured_given:
            raise ValueError(
                "island_full_width in the shearless band "
                f"(|shear|={abs(s)} < floor={floor}) requires a finite measured "
                "O/X width; the pendulum estimate diverges there."
            )
        return float(measured_full_width)
    if measured_given:
        return float(measured_full_width)
    if amplitude is None:
        raise ValueError(
            "island_full_width in the twist band requires either a measured "
            "width or a resonant amplitude for the pendulum estimate."
        )
    return pendulum_full_width(amplitude, s, coefficient=coefficient)


def chirikov_overlap_from_crossings(
    crossings: Sequence,
    measured_full_widths: Sequence[float],
    *,
    shear_floor: float = DEFAULT_SHEAR_FLOOR,
    aggregate: str = "max",
    lse_beta: float = DEFAULT_LSE_BETA,
) -> ChirikovOverlap:
    r"""Chirikov overlap over a ladder of resonance crossings with measured widths.

    ``crossings`` are
    :class:`~banana_opt.topology.iota_profile.RationalCrossing` objects (each
    supplying ``radial_label`` and signed ``local_shear``); ``measured_full_widths``
    the matching island full widths measured from the residue / periodic-orbit O-X
    separation. Each width is routed through :func:`island_full_width`, which
    enforces that any crossing in the shearless band (``|local_shear| < shear_floor``)
    carries a finite measurement rather than a divergent pendulum estimate. This is
    the field-level entry point for the ``n=5`` chain ladder of the banana-coil edge.
    """
    crossings = list(crossings)
    widths_in = list(measured_full_widths)
    if len(crossings) != len(widths_in):
        raise ValueError(
            "crossings and measured_full_widths must match in length; "
            f"got {len(crossings)} and {len(widths_in)}"
        )
    positions = [float(c.radial_label) for c in crossings]
    widths = [
        island_full_width(
            shear=float(c.local_shear),
            measured_full_width=w,
            shear_floor=shear_floor,
        )
        for c, w in zip(crossings, widths_in)
    ]
    return chirikov_overlap_scalar(
        positions, widths, aggregate=aggregate, lse_beta=lse_beta
    )


def chirikov_overlap_pairwise(
    positions: Sequence[float],
    full_widths: Sequence[float],
) -> tuple[ChirikovPair, ...]:
    r"""Per-adjacent-pair overlap ``s = (W_in/2 + W_out/2)/spacing``.

    ``positions`` are the resonance radial labels and ``full_widths`` the matching
    island full widths (same radial units). They are sorted by position; adjacent
    pairs are formed on the sorted ladder. Coincident resonances (zero spacing)
    raise rather than divide by zero. Fewer than two resonances yields an empty
    tuple (no pair to overlap). All inputs must be finite.
    """
    pos = np.asarray(positions, dtype=float)
    wid = np.asarray(full_widths, dtype=float)
    if pos.shape != wid.shape or pos.ndim != 1:
        raise ValueError(
            f"positions and full_widths must be 1-D and equal length; "
            f"got {pos.shape} and {wid.shape}"
        )
    if pos.size < 2:
        return ()
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(wid)):
        raise ValueError("chirikov_overlap_pairwise requires finite positions/widths")
    if np.any(wid < 0.0):
        raise ValueError("island full widths must be non-negative")
    order = np.argsort(pos, kind="stable")
    pos = pos[order]
    wid = wid[order]
    pairs: list[ChirikovPair] = []
    for i in range(pos.size - 1):
        spacing = float(pos[i + 1] - pos[i])
        if spacing <= 0.0:
            raise ValueError(
                "chirikov_overlap_pairwise requires distinct resonance labels; "
                f"coincident at label {float(pos[i])}"
            )
        half_sum = 0.5 * float(wid[i]) + 0.5 * float(wid[i + 1])
        pairs.append(
            ChirikovPair(
                inner_label=float(pos[i]),
                outer_label=float(pos[i + 1]),
                inner_width=float(wid[i]),
                outer_width=float(wid[i + 1]),
                spacing=spacing,
                overlap=half_sum / spacing,
            )
        )
    return tuple(pairs)


def _log_sum_exp_max(values: np.ndarray, beta: float) -> float:
    """Smooth (log-sum-exp) surrogate of ``max(values)``; numerically stable.

    ``LSE_beta(x) = max(x) + (1/beta) log sum exp(beta (x - max(x))) >= max(x)``,
    converging to the hard max as ``beta -> inf``. Used so the overlap objective
    is differentiable (a bare max has zero sub-gradient off the active pair).
    """
    b = float(beta)
    if b <= 0.0:
        raise ValueError(f"lse_beta must be positive; got {b}")
    m = float(np.max(values))
    return m + log(float(np.sum(np.exp(b * (values - m))))) / b


def chirikov_overlap_scalar(
    positions: Sequence[float],
    full_widths: Sequence[float],
    *,
    aggregate: str = "max",
    lse_beta: float = DEFAULT_LSE_BETA,
) -> ChirikovOverlap:
    """Aggregate Chirikov overlap over a resonance ladder.

    ``aggregate="max"`` returns the sharp worst-pair overlap (diagnostic);
    ``aggregate="softmax"`` returns a smooth log-sum-exp upper bound on the max
    (an optimisation objective). With fewer than two resonances the scalar is
    ``0.0`` (no overlap possible).
    """
    pairs = chirikov_overlap_pairwise(positions, full_widths)
    if len(pairs) == 0:
        return ChirikovOverlap(scalar=0.0, aggregate=aggregate, pairs=pairs)
    overlaps = np.asarray([p.overlap for p in pairs], dtype=float)
    if aggregate == "max":
        scalar = float(np.max(overlaps))
    elif aggregate == "softmax":
        scalar = _log_sum_exp_max(overlaps, lse_beta)
    else:
        raise ValueError(f"aggregate must be 'max' or 'softmax'; got {aggregate!r}")
    return ChirikovOverlap(scalar=scalar, aggregate=aggregate, pairs=pairs)
