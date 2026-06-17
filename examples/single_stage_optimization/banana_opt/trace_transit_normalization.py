"""Transit normalization for field-line trace horizons.

Why this module exists
----------------------
``simsopt.field.compute_fieldlines`` integrates the autonomous ODE

.. math::

    \\frac{d\\mathbf{x}}{dt} = \\mathbf{B}(\\mathbf{x})

so the integration variable ``t`` is *neither* physical time *nor* arc
length.  Arc length advances as :math:`ds/dt = |\\mathbf{B}|`, and the
toroidal angle advances as

.. math::

    \\frac{d\\phi}{dt} = \\frac{B_\\phi(\\mathbf{x})}{R}.

The number of toroidal transits accumulated over a horizon ``tmax`` is
therefore

.. math::

    N(t_{max}) = \\frac{1}{2\\pi} \\int_0^{t_{max}}
        \\frac{B_\\phi(\\mathbf{x}(t))}{R(t)}\\, dt
    \\approx \\frac{B_0\\, t_{max}}{2\\pi R_0}

for a TF-dominated field, where :math:`B_0` is the mean toroidal field
strength on the magnetic axis and :math:`R_0` the axis major radius.

Dimension check: with ``dx/dt = B`` the variable ``t`` carries units of
``m / T``; hence ``B_0 * tmax`` has units ``T * m / T = m`` (an effective
path length) and ``B_0 * tmax / (2 pi R_0)`` is dimensionless.  A fixed
``tmax`` therefore buys a *different* number of transits for every
configuration: transits scale linearly with ``|B|`` and inversely with
``R0``.  Cross-configuration survival comparisons at fixed ``tmax``
(e.g. the EQ current-pattern probe) are only apples-to-apples after
normalizing the horizon by transits, which is what this module provides.

All helpers are pure and side-effect free on their inputs; broken inputs
raise immediately (no fallbacks).
"""

from __future__ import annotations

import math

import numpy as np

TRANSIT_NORMALIZATION_SCHEMA_VERSION = "trace_transit_normalization_v1"

_DEFAULT_AXIS_SAMPLES = 64


def _require_positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return value


def measure_axis_field(
    field,
    surf=None,
    *,
    r_axis_m: float | None = None,
    z_axis_m: float = 0.0,
    n_phi: int = _DEFAULT_AXIS_SAMPLES,
):
    """Measure the mean toroidal field strength on the (approximate) magnetic axis.

    The axis is approximated by the circle ``R = r_axis_m`` at ``Z = z_axis_m``.
    Exactly one of ``surf`` (whose ``major_radius()`` provides ``r_axis_m``) or
    an explicit ``r_axis_m`` must be given.  The field's evaluation points are
    restored afterwards so this measurement has no side effect on subsequent
    tracing.

    Returns a metadata dict with ``b_axis_T`` (mean ``|B_phi|`` on the circle),
    ``b_mod_mean_T`` (mean ``|B|``), ``r_axis_m``, ``z_axis_m``, ``n_phi`` and
    the approximation method label.
    """
    if (surf is None) == (r_axis_m is None):
        raise ValueError("provide exactly one of surf or r_axis_m")
    if n_phi < 4:
        raise ValueError(f"n_phi must be >= 4, got {n_phi}")
    if r_axis_m is None:
        r_axis_m = float(surf.major_radius())
    r_axis_m = _require_positive(r_axis_m, "r_axis_m")
    z_axis_m = float(z_axis_m)

    phi = np.linspace(0.0, 2.0 * np.pi, int(n_phi), endpoint=False)
    points = np.column_stack(
        [
            r_axis_m * np.cos(phi),
            r_axis_m * np.sin(phi),
            np.full(phi.shape, z_axis_m),
        ]
    )

    saved_points = np.array(field.get_points_cart_ref(), copy=True)
    try:
        field.set_points(points)
        b = np.asarray(field.B())
    finally:
        field.set_points(saved_points)

    phi_hat = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    b_phi = np.einsum("ij,ij->i", b, phi_hat)
    b_axis = float(np.mean(np.abs(b_phi)))
    b_mod_mean = float(np.mean(np.linalg.norm(b, axis=1)))
    if not math.isfinite(b_axis) or b_axis <= 0.0:
        raise ValueError(
            f"measured mean |B_phi| on the axis circle is not positive: {b_axis!r}"
        )
    return {
        "schema_version": TRANSIT_NORMALIZATION_SCHEMA_VERSION,
        "method": "circular_axis_approximation",
        "r_axis_m": r_axis_m,
        "z_axis_m": z_axis_m,
        "n_phi": int(n_phi),
        "b_axis_T": b_axis,
        "b_mod_mean_T": b_mod_mean,
    }


def predicted_transits(tmax: float, *, b_axis_T: float, r_axis_m: float) -> float:
    """Toroidal transits bought by ``tmax``: ``N = B0 * tmax / (2 pi R0)``."""
    tmax = _require_positive(tmax, "tmax")
    b_axis_T = _require_positive(b_axis_T, "b_axis_T")
    r_axis_m = _require_positive(r_axis_m, "r_axis_m")
    return b_axis_T * tmax / (2.0 * math.pi * r_axis_m)


def tmax_for_transits(
    n_transits: float, *, b_axis_T: float, r_axis_m: float
) -> float:
    """Horizon ``tmax`` that buys ``n_transits``: ``tmax = 2 pi R0 N / B0``."""
    n_transits = _require_positive(n_transits, "n_transits")
    b_axis_T = _require_positive(b_axis_T, "b_axis_T")
    r_axis_m = _require_positive(r_axis_m, "r_axis_m")
    return 2.0 * math.pi * r_axis_m * n_transits / b_axis_T


def transit_normalization_metadata(field, surf, tmax: float) -> dict:
    """Sidecar-ready metadata: axis measurement plus implied transit count.

    Intended for opt-in reporting next to existing Poincare/topology
    artifacts so fixed-``tmax`` results can be compared across
    configurations without changing any trace contract.
    """
    axis = measure_axis_field(field, surf)
    tmax = _require_positive(tmax, "tmax")
    return {
        **axis,
        "tmax": tmax,
        "implied_transits": predicted_transits(
            tmax,
            b_axis_T=axis["b_axis_T"],
            r_axis_m=axis["r_axis_m"],
        ),
        "transits_per_unit_tmax": predicted_transits(
            1.0,
            b_axis_T=axis["b_axis_T"],
            r_axis_m=axis["r_axis_m"],
        ),
    }
