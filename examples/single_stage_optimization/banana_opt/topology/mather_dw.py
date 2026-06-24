r"""Mather :math:`\Delta W` last-torus-flux certification of a ``p/q`` chain.

This is an **offline / diagnostic** module: a converse-KAM-style certification of
whether a target irrational invariant torus survives, built from the *magnetic
action* of closed (periodic) field-line orbits. It is far too expensive to live
in an optimiser inner loop and is wired into nothing here — it is a stand-alone
certification tool a caller runs on a frozen configuration.

**The action.** The Mather / Aubry action of a closed field-line orbit is the
magnetic action

.. math::  W = \oint \mathbf{A}\cdot d\boldsymbol{\ell},

with :math:`\mathbf{A}` the magnetic vector potential and :math:`d\boldsymbol{\ell}`
the tangent along the orbit. For a **closed** loop this is **gauge invariant**:
under :math:`\mathbf{A}\to\mathbf{A}+\nabla f` it changes by
:math:`\oint\nabla f\cdot d\boldsymbol{\ell}=0`, and by Stokes

.. math::  \oint\mathbf{A}\cdot d\boldsymbol{\ell}
           = \iint(\nabla\times\mathbf{A})\cdot d\mathbf{S}
           = \iint\mathbf{B}\cdot d\mathbf{S},

i.e. the action equals the magnetic flux threaded by the loop. The trapezoidal
midpoint quadrature used by :func:`closed_loop_action` preserves the gauge
invariance *exactly* (to floating-point round-off) on a closed loop: the added
gradient telescopes around the ring regardless of the loop shape, so a gauge
change can never perturb the certificate. That exactness is the analytic anchor
the unit tests pin down.

**The diagnostic.** For the ``p/q`` resonant chain the island/cantorus signature is

.. math::  \Delta W(p/q) = W_X - W_O,

the action of the **X-point minimax** period-``q`` orbit minus the **O-point
minimizing** period-``q`` orbit of the chain (the standard Aubry--Mather
minimax/minimizing pair). Along the noble convergents
:math:`p_i/q_i\to\omega^\ast` of a target irrational rotation number
:math:`\omega^\ast`, :math:`\Delta W(p_i/q_i)\to 0` **iff** the torus at
:math:`\omega^\ast` is an intact KAM surface; it stays **bounded away from zero**
when the torus is broken (an Aubry--Mather cantorus). With the O point taken as
the action *minimizer* the minimax action obeys :math:`W_X\ge W_O`, so
:math:`\Delta W\ge 0` and the certificate keys on the magnitude.

**Nontwist caveat.** At a shearless (nontwist) curve the twist condition fails and
the Aubry--Mather minimax characterisation loses its footing, so a
:class:`MatherCertificate` produced near a shearless band carries the
:attr:`~MatherCertificate.heuristic` flag and must be cross-checked against
converse-KAM rather than trusted on its own.

**Calibration.** The pure core is calibrated against the Chirikov--Taylor standard
map, whose generating function and Aubry--Mather minimax/minimizing orbits have a
known last-torus breakup at :math:`K_c\approx0.971635`; the unit tests verify the
generating-function action against a hand-checked orbit and that
:math:`|\Delta W|` of the golden-mean convergents lifts from
:math:`\sim10^{-9}` (intact) to :math:`\sim10^{-4}` (broken) across ``K_c``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

__all__ = [
    "GOLDEN_MEAN",
    "MatherCertificate",
    "closed_loop_action",
    "golden_convergents",
    "mather_delta_w",
    "mather_delta_w_from_orbits",
    "noble_convergents",
]

# Golden mean omega = (sqrt(5) - 1)/2 = [0; 1, 1, 1, ...], the most-irrational
# rotation number and the canonical "last torus" of the standard map. Its
# continued-fraction convergents are the consecutive-Fibonacci ratios
# F_i / F_{i+1} (1/1, 1/2, 2/3, 3/5, 5/8, ...), the noble chain certified here.
GOLDEN_MEAN = (sqrt(5.0) - 1.0) / 2.0


@dataclass(frozen=True, slots=True)
class MatherCertificate:
    r"""Mather :math:`\Delta W` certificate for a noble chain toward ``omega_star``.

    ``omega_star`` is the target irrational rotation number; ``convergents`` are the
    per-``p/q`` records ``(p, q, delta_w)`` along the noble chain, in increasing
    ``q`` (deepening approximation to ``omega_star``); ``terminal_abs_delta_w`` is
    ``|delta_w|`` of the deepest (largest-``q``) convergent, the quantity the
    certificate keys on. ``torus_intact`` is the KAM verdict (terminal
    ``|delta_w|`` below ``tolerance``). ``heuristic`` is set when the shearless /
    nontwist caveat applies, in which case the verdict must be cross-checked
    against converse-KAM rather than trusted alone.
    """

    omega_star: float
    convergents: tuple[tuple[int, int, float], ...]
    terminal_abs_delta_w: float
    torus_intact: bool
    heuristic: bool = False

    def __post_init__(self) -> None:
        omega_star = float(self.omega_star)
        terminal = float(self.terminal_abs_delta_w)
        if not isfinite(omega_star):
            raise ValueError(f"MatherCertificate omega_star must be finite; got {omega_star}")
        if not isfinite(terminal) or terminal < 0.0:
            raise ValueError(
                f"MatherCertificate terminal_abs_delta_w must be finite and "
                f"non-negative; got {terminal}"
            )
        records: list[tuple[int, int, float]] = []
        for record in self.convergents:
            p, q, delta_w = record
            p_int = int(p)
            q_int = int(q)
            delta_value = float(delta_w)
            if q_int <= 0:
                raise ValueError(f"MatherCertificate convergent q must be positive; got {q_int}")
            if not isfinite(delta_value):
                raise ValueError(
                    f"MatherCertificate convergent delta_w must be finite; got {delta_value}"
                )
            records.append((p_int, q_int, delta_value))
        object.__setattr__(self, "omega_star", omega_star)
        object.__setattr__(self, "terminal_abs_delta_w", terminal)
        object.__setattr__(self, "convergents", tuple(records))
        object.__setattr__(self, "torus_intact", bool(self.torus_intact))
        object.__setattr__(self, "heuristic", bool(self.heuristic))


def closed_loop_action(points: np.ndarray, vector_potential: np.ndarray) -> float:
    r"""Magnetic action :math:`\oint\mathbf{A}\cdot d\boldsymbol{\ell}` of a closed loop.

    ``points`` is an ``(N, 3)`` array of Cartesian samples of the orbit and
    ``vector_potential`` the matching ``(N, 3)`` samples of :math:`\mathbf{A}` at
    those points. The loop is treated as **closed**: the final segment joins
    sample ``N-1`` back to sample ``0`` (do *not* repeat the first point). Each
    segment contributes the trapezoidal midpoint estimate
    :math:`\tfrac12(\mathbf{A}_i+\mathbf{A}_{i+1})\cdot(\mathbf{p}_{i+1}-\mathbf{p}_i)`,
    summed with the index wrapping ``N-1 -> 0``. On a closed loop this quadrature
    is gauge invariant to round-off (an added :math:`\nabla f` telescopes to zero).

    Raises on non-finite inputs, shape mismatch, or fewer than three samples (a
    closed loop needs at least a triangle); the certificate must never be built on
    a degenerate or ``nan``-poisoned orbit.
    """
    pts = np.asarray(points, dtype=float)
    a_field = np.asarray(vector_potential, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"closed_loop_action points must be (N, 3); got {pts.shape}")
    if a_field.shape != pts.shape:
        raise ValueError(
            f"closed_loop_action vector_potential must match points shape {pts.shape}; "
            f"got {a_field.shape}"
        )
    if pts.shape[0] < 3:
        raise ValueError(
            f"closed_loop_action needs at least 3 samples to close a loop; got {pts.shape[0]}"
        )
    if not np.all(np.isfinite(pts)) or not np.all(np.isfinite(a_field)):
        raise ValueError("closed_loop_action requires finite points and vector_potential")
    next_pts = np.roll(pts, -1, axis=0)
    next_a = np.roll(a_field, -1, axis=0)
    segments = next_pts - pts
    midpoint_a = 0.5 * (a_field + next_a)
    return float(np.sum(np.einsum("ij,ij->i", midpoint_a, segments)))


def mather_delta_w(action_x: float, action_o: float) -> float:
    r"""Island/cantorus diagnostic :math:`\Delta W = W_X - W_O`.

    ``action_x`` is the action of the X-point minimax period-``q`` orbit and
    ``action_o`` that of the O-point minimizing period-``q`` orbit of the same
    ``p/q`` chain. With the O point taken as the action minimizer the minimax
    action obeys :math:`W_X\ge W_O`, so the result is non-negative for a
    consistently labelled pair; the certificate keys on its magnitude. Raises on
    non-finite inputs.
    """
    w_x = float(action_x)
    w_o = float(action_o)
    if not (isfinite(w_x) and isfinite(w_o)):
        raise ValueError(f"mather_delta_w needs finite actions; got W_X={w_x}, W_O={w_o}")
    return w_x - w_o


def noble_convergents(
    continued_fraction_tail: tuple[int, ...],
    depth: int,
) -> tuple[tuple[int, int], ...]:
    r"""Convergents :math:`p_i/q_i` of the continued fraction ``[0; a1, a2, ...]``.

    ``continued_fraction_tail`` supplies the partial quotients ``(a1, a2, ...)``
    after the leading ``0`` (the rotation numbers certified here lie in
    ``(0, 1)``); ``depth`` is how many convergents to return (truncated to the
    available tail). The convergents are built by the standard recurrence
    ``h_i = a_i h_{i-1} + h_{i-2}``, ``k_i = a_i k_{i-1} + k_{i-2}`` seeded with
    ``h_{-1}=1, h_0=0`` and ``k_{-1}=0, k_0=1`` (the ``[0; ...]`` leading term),
    so the ``i``-th convergent is ``h_i/k_i = p_i/q_i`` in lowest terms. All
    partial quotients must be positive integers; ``depth`` must be positive.
    """
    tail = tuple(int(a) for a in continued_fraction_tail)
    requested_depth = int(depth)
    if requested_depth <= 0:
        raise ValueError(f"noble_convergents depth must be positive; got {requested_depth}")
    if len(tail) == 0:
        raise ValueError("noble_convergents needs at least one partial quotient")
    if any(a <= 0 for a in tail):
        raise ValueError(f"noble_convergents partial quotients must be positive; got {tail}")
    usable = min(requested_depth, len(tail))
    # Numerator/denominator recurrence seeded for the leading [0; ...] term:
    # (h_{-1}, h_0) = (1, 0) and (k_{-1}, k_0) = (0, 1) so that h_i/k_i = p_i/q_i.
    h_prev, h_curr = 1, 0
    k_prev, k_curr = 0, 1
    convergents: list[tuple[int, int]] = []
    for index in range(usable):
        a = tail[index]
        h_prev, h_curr = h_curr, a * h_curr + h_prev
        k_prev, k_curr = k_curr, a * k_curr + k_prev
        convergents.append((h_curr, k_curr))
    return tuple(convergents)


def golden_convergents(depth: int) -> tuple[tuple[int, int], ...]:
    r"""Convergents of the golden mean :math:`[0; 1, 1, 1, \ldots]`.

    Returns the first ``depth`` consecutive-Fibonacci ratios
    ``F_i/F_{i+1}`` (``1/1, 1/2, 2/3, 3/5, 5/8, ...``), the noble chain whose
    limit is :data:`GOLDEN_MEAN`. Thin wrapper over :func:`noble_convergents` with
    an all-ones tail of length ``depth``.
    """
    requested_depth = int(depth)
    if requested_depth <= 0:
        raise ValueError(f"golden_convergents depth must be positive; got {requested_depth}")
    return noble_convergents((1,) * requested_depth, requested_depth)


def mather_delta_w_from_orbits(
    field,
    o_trajectory: np.ndarray,
    x_trajectory: np.ndarray,
) -> float:
    r"""Offline :math:`\Delta W = W_X - W_O` from O/X closed orbits in a field.

    ``field`` is a ``BiotSavart``-like object exposing ``set_points`` and ``A`` (so
    the vector potential is sampled with ``field.set_points(pts); A = field.A()``);
    ``o_trajectory`` and ``x_trajectory`` are the O-point minimizing and X-point
    minimax **closed** period-``q`` orbits as Cartesian ``(N, 3)`` point arrays.
    The action of each is :func:`closed_loop_action` of its ``A`` samples, and the
    diagnostic is :func:`mather_delta_w` of the two.

    Note: a :func:`solve_periodic_orbit` result does not expose a Cartesian
    trajectory directly -- its orbit samples are the return-map ``states`` (shape
    ``(N, 2)``, cylindrical ``[R, Z]``) at the toroidal angles ``phi_grid``.
    Convert them to Cartesian ``[R cos phi, R sin phi, Z]`` before passing here
    (these routines expect ``(N, 3)`` Cartesian).

    Orbit *solving* is deliberately left to the caller -- this routine only scores
    supplied trajectories, keeping it dependency-light. It is **certification-only
    / offline**: sampling ``A`` along resolved period-``q`` orbits for every noble
    convergent is far too expensive for an optimiser inner loop. Raises (via
    :func:`closed_loop_action`) on non-finite or degenerate trajectories.
    """
    w_o = _action_in_field(field, o_trajectory)
    w_x = _action_in_field(field, x_trajectory)
    return mather_delta_w(w_x, w_o)


def _action_in_field(field, trajectory: np.ndarray) -> float:
    """Sample ``A`` on ``trajectory`` via ``field`` and return its closed-loop action.

    ``trajectory`` is an ``(N, 3)`` Cartesian closed orbit; the vector potential is
    read with ``field.set_points``/``field.A`` and reshaped to ``(N, 3)`` to match.
    Validation of finiteness/shape is delegated to :func:`closed_loop_action`.
    """
    pts = np.asarray(trajectory, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"orbit trajectory must be (N, 3); got {pts.shape}")
    # ascontiguousarray (not just asarray/reshape): a Cartesian orbit built by transposing
    # a (3, N) stack -- np.array([R cos phi, R sin phi, Z]).T, the (N, 3) layout this
    # module's docstring tells callers to pass -- is non-C-contiguous, and reshape preserves
    # those strides -> the pybind simsoptpp.BiotSavart.set_points overload rejects a
    # non-contiguous (N, 3) array (same trap converse_kam._b_and_dB fixes). (np.column_stack
    # / np.stack(axis=1) are already C-contiguous; this guards the .T-of-stacked layout.)
    field.set_points(np.ascontiguousarray(pts, dtype=np.float64))
    a_field = np.asarray(field.A(), dtype=float).reshape(pts.shape)
    return closed_loop_action(pts, a_field)
