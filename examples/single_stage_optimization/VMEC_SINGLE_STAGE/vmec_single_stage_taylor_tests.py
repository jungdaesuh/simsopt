"""Paper-faithful Taylor / finite-difference gradient-correctness tests.

Implements the three named gradient checks from
``program_cw_poloidal_legacy_vmec_true_single_stage_plan.md`` section
"Gradient contract":

- ``taylor_J2_alone_centered_eq28``     -- paper Eq. (28), centered FD, slope band [1.7, 2.2].
- ``taylor_J_combined_forward_eq29_xcoils``  -- paper Eq. (29) forward FD on coil DOFs, [0.8, 1.2].
- ``taylor_J_combined_centered_eq29_xsurface`` -- paper Eq. (29) centered FD on surface DOFs, [1.7, 2.2].

A fourth function ``assert_dJ1_dxcoils_zero`` checks the load-bearing identity
that the J1 component has zero gradient with respect to coil DOFs. Plan
section "Gradient contract" calls this an asserted invariant: "VMEC is
fixed-boundary; J1 has no functional dependence on coil DOFs, and the FD
harness must not sweep coil DOFs through J1."

The pass-criteria bands ``[1.7, 2.2]`` and ``[0.8, 1.2]`` are placeholders
pending first-campaign calibration against measured VMEC objective noise.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np


# Plan default FD step schedule (placeholder, recorded in artifact_manifest
# runtime_config so calibration sweeps participate in run-identity hashing).
DEFAULT_TAYLOR_SCHEDULE: Sequence[float] = (
    1.0e-1,
    3.0e-2,
    1.0e-2,
    3.0e-3,
    1.0e-3,
    3.0e-4,
    1.0e-4,
    3.0e-5,
)

# Plan default fixed-random-direction seed for reproducibility within a run.
DEFAULT_TAYLOR_RNG_SEED: int = 0

# Slope-band placeholders (plan "Finite-difference step schedule").
SECOND_ORDER_SLOPE_BAND: tuple = (1.7, 2.2)
FIRST_ORDER_SLOPE_BAND: tuple = (0.8, 1.2)

# Minimum number of consecutive Delta_x at which the slope must lie in the
# pass band (plan: "stable middle band of at least four consecutive values").
DEFAULT_STABLE_BAND_WIDTH: int = 4


def _draw_unit_random_direction(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw a fixed-magnitude random direction in ``[0, 1]^n``.

    Paper Eq. 28 spec: ``h`` is drawn once per test invocation in ``[0, 1]^n``,
    not resampled per Delta_x. We do not normalize; the FD slope is scale-
    invariant and the unnormalized draw matches the paper.
    """

    return rng.uniform(low=0.0, high=1.0, size=n)


def _convergence_slopes(deltas: Sequence[float], errors: Sequence[float]) -> np.ndarray:
    """Compute consecutive log-log slopes of ``error(delta)``."""

    log_delta = np.log10(np.asarray(deltas, dtype=float))
    log_err = np.log10(np.asarray(errors, dtype=float))
    # Slopes between consecutive (delta_k, delta_{k+1}).
    return np.diff(log_err) / np.diff(log_delta)


def _stable_band_passes(
    slopes: np.ndarray,
    band: tuple,
    stable_width: int,
) -> bool:
    """Return ``True`` iff ``stable_width`` consecutive slopes lie in ``band``."""

    if slopes.size < stable_width:
        return False
    lo, hi = band
    in_band = (slopes >= lo) & (slopes <= hi)
    # Sliding window of width ``stable_width`` over the boolean mask.
    for start in range(in_band.size - stable_width + 1):
        if in_band[start : start + stable_width].all():
            return True
    return False


def _centered_fd_errors(
    build_J: Callable[[np.ndarray], float],
    build_dJ: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    direction: np.ndarray,
    schedule: Sequence[float],
) -> Sequence[float]:
    """Centered FD error vs analytic gradient along ``direction``.

    Paper Eq. 28: error of the centered-difference approximation to the
    directional derivative. For each ``delta`` in ``schedule``::

        err(delta) = | (J(x0 + delta*h) - J(x0 - delta*h)) / (2*delta) - <g, h> |

    The leading Taylor-error term is ``(delta^2 / 6) * <h, H^3 J . h>`` (where
    ``H^3 J`` is the third-derivative tensor); the slope of ``log10(err)``
    vs ``log10(delta)`` is therefore expected to be ``2`` for smooth ``J``.
    """

    g = build_dJ(x0)
    g_dot_h = float(np.dot(g, direction))
    errors = []
    for delta in schedule:
        plus = build_J(x0 + delta * direction)
        minus = build_J(x0 - delta * direction)
        approx = (plus - minus) / (2.0 * delta)
        errors.append(abs(approx - g_dot_h))
    return errors


def _forward_fd_errors(
    build_J: Callable[[np.ndarray], float],
    build_dJ: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    direction: np.ndarray,
    schedule: Sequence[float],
) -> Sequence[float]:
    """Forward FD error vs analytic gradient along ``direction``.

    Paper Eq. 29 forward variant. For each ``delta`` in ``schedule``::

        err(delta) = | (J(x0 + delta*h) - J(x0)) / delta - <g, h> |

    Leading Taylor error ``(delta / 2) * <h, H^2 J . h>``; expected slope of
    ``log10(err)`` vs ``log10(delta)`` is ``1``.
    """

    J0 = build_J(x0)
    g = build_dJ(x0)
    g_dot_h = float(np.dot(g, direction))
    errors = []
    for delta in schedule:
        plus = build_J(x0 + delta * direction)
        approx = (plus - J0) / delta
        errors.append(abs(approx - g_dot_h))
    return errors


def _taylor_result(
    label: str,
    deltas: Sequence[float],
    errors: Sequence[float],
    band: tuple,
    stable_width: int,
) -> Dict[str, object]:
    slopes = _convergence_slopes(deltas, errors)
    passed = _stable_band_passes(slopes, band, stable_width)
    return {
        "label": label,
        "schedule": list(map(float, deltas)),
        "errors": list(map(float, errors)),
        "slopes": list(map(float, slopes.tolist())),
        "slope_band": (float(band[0]), float(band[1])),
        "stable_band_width": int(stable_width),
        "passed": bool(passed),
    }


def taylor_J2_alone_centered_eq28(
    build_J2: Callable[[np.ndarray], float],
    build_dJ2: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    schedule: Sequence[float] = DEFAULT_TAYLOR_SCHEDULE,
    rng_seed: int = DEFAULT_TAYLOR_RNG_SEED,
    stable_width: int = DEFAULT_STABLE_BAND_WIDTH,
) -> Dict[str, object]:
    """Paper Eq. 28 centered-FD check on ``J2`` alone.

    Expected slope band ``[1.7, 2.2]`` (second-order). The random direction
    ``h`` is drawn once with ``rng_seed`` and reused across every ``delta``.
    """

    rng = np.random.default_rng(rng_seed)
    direction = _draw_unit_random_direction(int(x0.size), rng)
    errors = _centered_fd_errors(build_J2, build_dJ2, x0, direction, schedule)
    return _taylor_result(
        "taylor_J2_alone_centered_eq28",
        schedule,
        errors,
        SECOND_ORDER_SLOPE_BAND,
        stable_width,
    )


def taylor_J_combined_forward_eq29_xcoils(
    build_J: Callable[[np.ndarray], float],
    build_dJ: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    coil_slice: slice,
    schedule: Sequence[float] = DEFAULT_TAYLOR_SCHEDULE,
    rng_seed: int = DEFAULT_TAYLOR_RNG_SEED,
    stable_width: int = DEFAULT_STABLE_BAND_WIDTH,
) -> Dict[str, object]:
    """Paper Eq. 29 forward-FD check on combined ``J`` over coil DOFs.

    Expected slope band ``[0.8, 1.2]`` (first-order). The random direction is
    drawn over the full DOF vector but zeroed outside ``coil_slice`` so the
    test isolates the coil-DOF gradient.
    """

    rng = np.random.default_rng(rng_seed)
    direction = np.zeros_like(x0, dtype=float)
    direction[coil_slice] = _draw_unit_random_direction(
        int(direction[coil_slice].size), rng
    )
    errors = _forward_fd_errors(build_J, build_dJ, x0, direction, schedule)
    return _taylor_result(
        "taylor_J_combined_forward_eq29_xcoils",
        schedule,
        errors,
        FIRST_ORDER_SLOPE_BAND,
        stable_width,
    )


def taylor_J_combined_centered_eq29_xsurface(
    build_J: Callable[[np.ndarray], float],
    build_dJ: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    surface_slice: slice,
    schedule: Sequence[float] = DEFAULT_TAYLOR_SCHEDULE,
    rng_seed: int = DEFAULT_TAYLOR_RNG_SEED,
    stable_width: int = DEFAULT_STABLE_BAND_WIDTH,
) -> Dict[str, object]:
    """Paper Eq. 29 centered-FD check on combined ``J`` over surface DOFs.

    Expected slope band ``[1.7, 2.2]`` (second-order). Direction is zeroed
    outside ``surface_slice`` so the test isolates the surface-DOF gradient.
    """

    rng = np.random.default_rng(rng_seed)
    direction = np.zeros_like(x0, dtype=float)
    direction[surface_slice] = _draw_unit_random_direction(
        int(direction[surface_slice].size), rng
    )
    errors = _centered_fd_errors(build_J, build_dJ, x0, direction, schedule)
    return _taylor_result(
        "taylor_J_combined_centered_eq29_xsurface",
        schedule,
        errors,
        SECOND_ORDER_SLOPE_BAND,
        stable_width,
    )


def assert_dJ1_dxcoils_zero(
    build_J1: Callable[[np.ndarray], float],
    x0: np.ndarray,
    coil_slice: slice,
    eps: float = 1.0e-7,
    fd_zero_floor: float = 1.0e-12,
) -> Dict[str, object]:
    """Assert the identity ``dJ1/dxcoils == 0`` to FD round-off.

    Forward-FDs ``J1`` over each coil DOF and verifies the result is below
    ``fd_zero_floor``. VMEC is fixed-boundary, so J1 has no functional
    dependence on coil DOFs; any nonzero result is a wiring bug (e.g., a
    coil DOF accidentally driving a VMEC re-run).

    The returned dict is shaped like the Taylor-test dicts for uniform
    handling by the autoresearch ledger.
    """

    J0 = float(build_J1(x0))
    fd_grad = np.zeros_like(x0, dtype=float)
    coil_indices = np.arange(x0.size)[coil_slice]
    for idx in coil_indices:
        x_perturb = x0.copy()
        x_perturb[idx] = x_perturb[idx] + eps
        fd_grad[idx] = (float(build_J1(x_perturb)) - J0) / eps
    coil_grad = fd_grad[coil_slice]
    max_abs = float(np.max(np.abs(coil_grad))) if coil_grad.size > 0 else 0.0
    passed = max_abs <= fd_zero_floor
    return {
        "label": "assert_dJ1_dxcoils_zero",
        "eps": float(eps),
        "fd_zero_floor": float(fd_zero_floor),
        "max_abs_dJ1_dxcoils": max_abs,
        "passed": bool(passed),
    }
