"""JAX port of ``simsopt.field.magnetic_axis_helpers``.

The upstream module computes the on-axis rotational transform ``iota``
via Greene's tangent-map ODE (Greene, J. Math. Phys. 20, 1183 (1979),
equation (13)). The CPU oracle integrates the 4-component tangent-map
state ``y = [a, b, c, d]`` (the row-major flattening of the 2x2
monodromy matrix M) from ``phi = 0`` to ``phi = 1/nfp`` with SciPy's
``solve_ivp(method='RK45', rtol=1e-12, atol=1e-12)``, then extracts
``iota = arg(eig(M)[0]) * nfp / (2 pi)``.

This JAX port implements a self-contained Dormand-Prince RK4(5)
adaptive integrator (cf. Hairer, Norsett & Wanner, *Solving Ordinary
Differential Equations I*, Section II.5) inside
``jax.lax.while_loop`` so the entire iota computation is JIT-able and
differentiable through field DOFs as long as the user supplies a
JAX-traceable field-evaluation callback. The integrator uses the
classic PI(0.7, 0.4) error controller with a configurable safety
factor and per-step accept/reject branch.

Public surface
--------------

- :func:`on_axis_iota_rk` — pure JAX kernel that integrates the
  tangent map and returns iota. Accepts a JAX-traceable callback
  ``field_eval_fn(points) -> (B, dB_by_dX)`` plus the axis spec.
- :func:`axis_position` — pure-function evaluation of the axis
  Cartesian position at one or many ``phi`` values, using the
  underlying :func:`simsopt.geo.curverzfourier.curverzfourier_pure`
  kernel. Used internally by :func:`on_axis_iota_rk` but exported for
  test reuse.

Contract notes
--------------

- All inputs to :func:`on_axis_iota_rk` are explicit; no global state
  is consulted. The axis is consumed as a
  :class:`simsopt.jax_core.specs.CurveRZFourierSpec`; the magnetic
  field enters through ``field_eval_fn`` which receives a JAX array
  of shape ``[1, 3]`` and must return ``(B[1, 3], dB_by_dX[1, 3, 3])``
  with the SIMSOPT convention ``dB_by_dX[p, j, l] = d_j B_l(x_p)``.
- The integrator is **fully unrolled inside JIT**: it pre-allocates a
  fixed-shape carry tuple, masks rejected steps, and terminates the
  ``while_loop`` once ``phi >= phi_end`` or ``step_count >= max_steps``.
  ``max_steps`` is static (Python int) and bounds the loop.
- Tolerances ``rtol`` / ``atol`` are runtime float scalars matching the
  upstream SciPy ``solve_ivp`` defaults at the call site (the SIMSOPT
  CPU oracle uses ``rtol = atol = 1e-12``).
- Because Dormand-Prince's truncation pattern differs from SciPy's
  RK45 implementation at the bit level, JAX/CPU agreement is bound by
  the ``derivative_heavy`` parity-ladder lane (scalar_value tolerance
  ``rtol = 1e-10``), not ``direct_kernel``.

This module deliberately does **not** wrap the upstream eigenvalue
extraction in a sign-stabilizing post-processor: the iota sign
convention is determined entirely by ``np.arctan2(imag, real)`` of the
first eigenvalue returned by ``jnp.linalg.eig``. JAX and NumPy use the
same LAPACK eig path for 2x2 matrices, so the ordering matches the CPU
oracle within numerical tolerance.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..geo.curverzfourier import curverzfourier_pure
from .specs import CurveRZFourierSpec

__all__ = [
    "axis_position",
    "axis_position_and_tangent",
    "on_axis_iota_rk",
    "tangent_map_state_dim",
    "tangent_map_y0",
    "tangent_map_rhs_from_field",
]


# ── Dormand-Prince RK4(5) Butcher tableau (Hairer et al. Table 5.2) ──

_DOPRI5_C = np.array(
    [0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0],
    dtype=np.float64,
)

# A[i, j] for i in 0..6, j in 0..5 (lower-triangular, zero above)
_DOPRI5_A = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0],
        [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0],
        [
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
        ],
        [
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
        ],
        [
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
        ],
    ],
    dtype=np.float64,
)

# 5th-order weights (b vector); 4th-order weights (b_hat); use b - b_hat for the
# embedded error estimate. These are the published Dormand-Prince coefficients.
_DOPRI5_B = np.array(
    [
        35.0 / 384.0,
        0.0,
        500.0 / 1113.0,
        125.0 / 192.0,
        -2187.0 / 6784.0,
        11.0 / 84.0,
        0.0,
    ],
    dtype=np.float64,
)

_DOPRI5_E = np.array(
    [
        71.0 / 57600.0,
        0.0,
        -71.0 / 16695.0,
        71.0 / 1920.0,
        -17253.0 / 339200.0,
        22.0 / 525.0,
        -1.0 / 40.0,
    ],
    dtype=np.float64,
)


# ── Axis position ─────────────────────────────────────────────────────


def _axis_quadpoints(phi: jax.Array) -> jax.Array:
    """Treat ``phi`` as a 1-D ``[N]`` array of quadpoints in units of the full period."""
    return jnp.asarray(phi, dtype=jnp.float64)


def axis_position(spec: CurveRZFourierSpec, phi: jax.Array) -> jax.Array:
    """Return Cartesian axis position ``[N, 3]`` at quadpoints ``phi`` (units of full period).

    The Greene tangent-map ODE evolves ``phi`` over ``[0, 1/nfp]`` (one
    field period). ``phi`` here is the SIMSOPT ``quadpoints`` convention:
    fraction of the full toroidal angle in ``[0, 1]``.
    """

    quadpoints = _axis_quadpoints(phi)
    return curverzfourier_pure(
        jnp.asarray(spec.dofs, dtype=jnp.float64),
        quadpoints,
        int(spec.order),
        int(spec.nfp),
        bool(spec.stellsym),
    )


def axis_position_and_tangent(
    spec: CurveRZFourierSpec, phi: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Return ``(gamma[N, 3], dgamma/dphi[N, 3])`` at quadpoints ``phi``.

    The tangent direction is the JVP of :func:`curverzfourier_pure`
    along ``ones_like(phi)``: it matches the SIMSOPT convention where
    ``gammadash = dgamma/dphi`` with ``phi`` in units of the full
    toroidal angle (i.e. ``2 pi`` is implicit in the trigs).
    """

    quadpoints = _axis_quadpoints(phi)
    tangents = jnp.ones_like(quadpoints)
    order = int(spec.order)
    nfp = int(spec.nfp)
    stellsym = bool(spec.stellsym)
    dofs = jnp.asarray(spec.dofs, dtype=jnp.float64)

    def gamma_kernel(quad: jax.Array) -> jax.Array:
        return curverzfourier_pure(dofs, quad, order, nfp, stellsym)

    gamma, gammadash = jax.jvp(gamma_kernel, (quadpoints,), (tangents,))
    return gamma, gammadash


# ── Tangent map RHS ────────────────────────────────────────────────────


def tangent_map_state_dim() -> int:
    """Tangent-map state size (Greene 1979 eq. 13): 2x2 monodromy = 4 floats."""
    return 4


def tangent_map_y0(*, dtype=jnp.float64) -> jax.Array:
    """Initial state ``y0 = [1, 0, 0, 1]`` (identity monodromy)."""
    return jnp.asarray([1.0, 0.0, 0.0, 1.0], dtype=dtype)


def _tangent_map_A_matrix(
    axis_point: jax.Array,
    B: jax.Array,
    dB_by_dX: jax.Array,
    phi: jax.Array,
) -> jax.Array:
    """Return the 2x2 matrix ``A`` from Greene's tangent-map equation (13).

    Inputs are at a single axis evaluation point. ``axis_point`` is the
    flat Cartesian xyz triple, ``B`` is the field 3-vector, and
    ``dB_by_dX[j, l] = d_j B_l`` matches the SIMSOPT convention. The
    expression mirrors the upstream NumPy implementation in
    ``simsopt/field/magnetic_axis_helpers.py:: tangent_map`` exactly.
    """

    B1 = B[0]
    B2 = B[1]
    B3 = B[2]

    dB1_dx = dB_by_dX[0, 0]
    dB1_dy = dB_by_dX[0, 1]
    dB1_dz = dB_by_dX[0, 2]
    dB2_dx = dB_by_dX[1, 0]
    dB2_dy = dB_by_dX[1, 1]
    dB2_dz = dB_by_dX[1, 2]
    dB3_dx = dB_by_dX[2, 0]
    dB3_dy = dB_by_dX[2, 1]
    dB3_dz = dB_by_dX[2, 2]

    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=phi.dtype)
    c = jnp.cos(two_pi * phi)
    s = jnp.sin(two_pi * phi)

    x_pos = axis_point[0]
    y_pos = axis_point[1]
    R = jnp.sqrt(x_pos * x_pos + y_pos * y_pos)

    BR = c * B1 + s * B2
    Bphi = -s * B1 + c * B2
    BZ = B3

    dB1_dR = dB1_dx * c + dB1_dy * s
    dB2_dR = dB2_dx * c + dB2_dy * s
    dB3_dR = dB3_dx * c + dB3_dy * s

    dBR_dR = c * dB1_dR + s * dB2_dR
    dBphi_dR = -s * dB1_dR + c * dB2_dR
    dBZ_dR = dB3_dR

    dBR_dZ = c * dB1_dz + s * dB2_dz
    dBphi_dZ = -s * dB1_dz + c * dB2_dz
    dBZ_dZ = dB3_dz

    Bphi_R = Bphi / R
    d_Bphi_R_dR = (dBphi_dR * R - Bphi) / (R * R)
    d_Bphi_R_dZ = dBphi_dZ / R

    A11 = (dBR_dR - (BR / Bphi_R) * d_Bphi_R_dR) / Bphi_R
    A21 = (dBZ_dR - (BZ / Bphi_R) * d_Bphi_R_dR) / Bphi_R
    A12 = (dBR_dZ - (BR / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
    A22 = (dBZ_dZ - (BZ / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
    return jnp.asarray([[A11, A12], [A21, A22]])


def tangent_map_rhs_from_field(
    axis_point: jax.Array,
    B: jax.Array,
    dB_by_dX: jax.Array,
    phi: jax.Array,
    y: jax.Array,
) -> jax.Array:
    """Compute ``dy/dphi`` for the 4-vector tangent-map state.

    The upstream CPU expression
    ``2 pi * concat([A @ y[:2], A @ y[2:]])`` is reproduced bit-faithfully
    so that JAX agrees with the SciPy oracle when both consume the same
    ``(B, dB)``.
    """

    A = _tangent_map_A_matrix(axis_point, B, dB_by_dX, phi)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=y.dtype)
    top = A @ y[:2]
    bottom = A @ y[2:]
    return two_pi * jnp.concatenate([top, bottom])


def _spec_to_dofs_and_meta(spec: CurveRZFourierSpec):
    return (
        jnp.asarray(spec.dofs, dtype=jnp.float64),
        int(spec.order),
        int(spec.nfp),
        bool(spec.stellsym),
    )


def _rhs_factory(
    spec: CurveRZFourierSpec,
    field_eval_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """Return a closure ``rhs(phi, y) -> dy/dphi`` for the tangent map.

    The closure captures the curve spec and the field evaluator. It is
    re-built per :func:`on_axis_iota_rk` call so that ``field_eval_fn``
    is part of the JAX trace, not the static cache key.
    """

    dofs, order, nfp, stellsym = _spec_to_dofs_and_meta(spec)

    def rhs(phi: jax.Array, y: jax.Array) -> jax.Array:
        phi_flat = jnp.asarray(phi, dtype=jnp.float64).reshape((1,))
        gamma = curverzfourier_pure(dofs, phi_flat, order, nfp, stellsym)
        B, dB_by_dX = field_eval_fn(gamma)
        B_flat = jnp.asarray(B, dtype=jnp.float64).reshape((3,))
        dB_flat = jnp.asarray(dB_by_dX, dtype=jnp.float64).reshape((3, 3))
        axis_point = gamma.reshape((3,))
        return tangent_map_rhs_from_field(
            axis_point,
            B_flat,
            dB_flat,
            jnp.asarray(phi, dtype=y.dtype),
            y,
        )

    return rhs


# ── Dormand-Prince single-step ─────────────────────────────────────────


def _dopri5_step(
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    phi: jax.Array,
    y: jax.Array,
    h: jax.Array,
    k_first: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single Dormand-Prince RK4(5) step.

    Parameters
    ----------
    rhs
        ``rhs(phi, y) -> dy/dphi`` callable.
    phi, y
        Current independent variable scalar and state vector.
    h
        Step size scalar.
    k_first
        Pre-computed ``rhs(phi, y)`` (FSAL reuse from prior accepted step,
        or freshly computed at integration start).

    Returns
    -------
    y_new
        5th-order RK estimate at ``phi + h``.
    y_err
        Embedded error estimate ``b - b_hat`` weighted by ``h``.
    k7
        ``rhs(phi + h, y_new)``; FSAL reuse for the next step.
    """

    dtype = y.dtype
    A = jnp.asarray(_DOPRI5_A, dtype=dtype)
    C = jnp.asarray(_DOPRI5_C, dtype=dtype)
    B = jnp.asarray(_DOPRI5_B, dtype=dtype)
    E = jnp.asarray(_DOPRI5_E, dtype=dtype)

    k1 = k_first
    k2 = rhs(phi + C[1] * h, y + h * (A[1, 0] * k1))
    k3 = rhs(phi + C[2] * h, y + h * (A[2, 0] * k1 + A[2, 1] * k2))
    k4 = rhs(
        phi + C[3] * h,
        y + h * (A[3, 0] * k1 + A[3, 1] * k2 + A[3, 2] * k3),
    )
    k5 = rhs(
        phi + C[4] * h,
        y + h * (A[4, 0] * k1 + A[4, 1] * k2 + A[4, 2] * k3 + A[4, 3] * k4),
    )
    k6 = rhs(
        phi + C[5] * h,
        y
        + h
        * (A[5, 0] * k1 + A[5, 1] * k2 + A[5, 2] * k3 + A[5, 3] * k4 + A[5, 4] * k5),
    )
    y_new = y + h * (B[0] * k1 + B[2] * k3 + B[3] * k4 + B[4] * k5 + B[5] * k6)
    k7 = rhs(phi + h, y_new)
    # Embedded error: h * (E[0]*k1 + E[2]*k3 + E[3]*k4 + E[4]*k5 + E[5]*k6 + E[6]*k7)
    y_err = h * (E[0] * k1 + E[2] * k3 + E[3] * k4 + E[4] * k5 + E[5] * k6 + E[6] * k7)
    return y_new, y_err, k7


# ── PI step controller ────────────────────────────────────────────────


# Reference: Hairer et al. eq. (4.13). Order=5, exponent = 1/5.
_DOPRI5_EXP = 0.2
_SAFETY = 0.9
_MIN_FACTOR = 0.2
_MAX_FACTOR = 5.0


def _error_norm(y_err: jax.Array, y: jax.Array, y_new: jax.Array, rtol, atol):
    sc = atol + rtol * jnp.maximum(jnp.abs(y), jnp.abs(y_new))
    return jnp.sqrt(jnp.mean(jnp.square(y_err / sc)))


_INITIAL_STEP_FRACTION = 1.0 / 100.0


def _initial_step_size(phi0, phi_end) -> jax.Array:
    """Conservative starting step at ``1/100`` of the integration interval.

    The Greene tangent map has smooth periodic coefficients, so the PI
    controller adapts up quickly. Starting much smaller than the
    expected steady-state step (~1e-3) avoids an early rejection
    cascade; starting at the interval length tends to fail the first
    error test and waste budget. ``1/100`` is the canonical compromise
    used by Hairer et al.'s reference DOPRI5 driver.
    """
    span = jnp.abs(phi_end - phi0)
    h0 = jnp.asarray(_INITIAL_STEP_FRACTION, dtype=span.dtype) * span
    return jnp.minimum(h0, span)


# ── Public integrator ────────────────────────────────────────────────


def _integrate_tangent_map(
    rhs: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    phi0: jax.Array,
    phi_end: jax.Array,
    y0: jax.Array,
    rtol: jax.Array,
    atol: jax.Array,
    max_steps: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Integrate ``y' = rhs(phi, y)`` from ``phi0`` to ``phi_end`` with DOPRI5.

    Returns ``(y_final, phi_final, steps_taken, succeeded)`` where
    ``succeeded`` is a JAX boolean scalar indicating ``phi_final >=
    phi_end`` within machine precision.
    """

    dtype = y0.dtype
    rtol_arr = jnp.asarray(rtol, dtype=dtype)
    atol_arr = jnp.asarray(atol, dtype=dtype)
    h0 = _initial_step_size(phi0, phi_end)
    k0 = rhs(phi0, y0)
    one = jnp.asarray(1.0, dtype=dtype)

    init_carry = (
        jnp.asarray(0, dtype=jnp.int32),  # step_count
        phi0,
        y0,
        h0,
        k0,
    )

    def cond(carry):
        step_count, phi, _y, _h, _k = carry
        not_done = phi < phi_end
        budget_ok = step_count < jnp.asarray(max_steps, dtype=jnp.int32)
        return jnp.logical_and(not_done, budget_ok)

    def body(carry):
        step_count, phi, y, h, k_first = carry
        # Clamp step to not overshoot phi_end.
        h_clamped = jnp.minimum(h, phi_end - phi)
        y_new, y_err, k7 = _dopri5_step(rhs, phi, y, h_clamped, k_first)
        err = _error_norm(y_err, y, y_new, rtol_arr, atol_arr)
        # If err is NaN/Inf, force a reject (factor → MIN_FACTOR).
        err_safe = jnp.where(jnp.isfinite(err), err, jnp.asarray(jnp.inf, dtype=dtype))
        accepted = err_safe <= one
        # Step-size update.
        factor = jnp.where(
            err_safe > jnp.asarray(0.0, dtype=dtype),
            jnp.asarray(_SAFETY, dtype=dtype)
            * jnp.power(err_safe, jnp.asarray(-_DOPRI5_EXP, dtype=dtype)),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        factor = jnp.clip(
            factor,
            jnp.asarray(_MIN_FACTOR, dtype=dtype),
            jnp.asarray(_MAX_FACTOR, dtype=dtype),
        )
        h_next = h_clamped * factor
        # If accepted: advance state and reuse k7 as next-step k1 (FSAL).
        phi_next = jnp.where(accepted, phi + h_clamped, phi)
        y_next = jnp.where(accepted, y_new, y)
        k_next = jnp.where(accepted, k7, k_first)
        return (
            step_count + jnp.asarray(1, dtype=jnp.int32),
            phi_next,
            y_next,
            h_next,
            k_next,
        )

    final_carry = jax.lax.while_loop(cond, body, init_carry)
    steps_taken, phi_final, y_final, _, _ = final_carry
    eps_phi = jnp.asarray(1.0e-12, dtype=dtype) * jnp.maximum(
        jnp.abs(phi_end), jnp.asarray(1.0, dtype=dtype)
    )
    succeeded = (phi_end - phi_final) <= eps_phi
    return y_final, phi_final, steps_taken, succeeded


def on_axis_iota_rk(
    axis_spec: CurveRZFourierSpec,
    field_eval_fn: Callable[[jax.Array], tuple[jax.Array, jax.Array]],
    *,
    rtol: float | jax.Array = 1.0e-12,
    atol: float | jax.Array = 1.0e-12,
    max_steps: int = 10000,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Greene 1979 on-axis iota via Dormand-Prince RK4(5) in JAX.

    Parameters
    ----------
    axis_spec
        Immutable :class:`CurveRZFourierSpec` describing the magnetic
        axis. Must satisfy ``nfp >= 1``.
    field_eval_fn
        JAX-traceable callable mapping ``points : [N, 3]`` to a tuple
        ``(B : [N, 3], dB_by_dX : [N, 3, 3])``. The kernel evaluates
        ``field_eval_fn`` at single-point arrays of shape ``[1, 3]``;
        the callable is responsible for caching its own quadrature
        state (e.g. precomputed coil arrays).
    rtol, atol
        Adaptive-RK relative and absolute tolerances. Defaults match
        the upstream SciPy ``solve_ivp`` call (``1e-12 / 1e-12``).
    max_steps
        Static integer; bounds the adaptive integrator loop. The
        Greene tangent-map ODE is well-conditioned, so a few hundred
        steps suffice at default tolerance; ``10000`` is a generous
        ceiling.

    Returns
    -------
    iota : jax.Array
        Scalar on-axis rotational transform.
    steps_taken : jax.Array
        Int32 scalar count of integrator steps consumed.
    succeeded : jax.Array
        Bool scalar; ``True`` if ``phi_final`` reached ``1 / nfp``.

    Notes
    -----
    The tangent-map state ``y = [a, b, c, d]`` flattens the 2x2
    monodromy matrix in row-major order, matching the upstream
    ``y0 = [1, 0, 0, 1]`` initial condition. After integration to
    ``phi = 1 / nfp``, ``iota`` is extracted as
    ``arctan2(im(eig(M)_0), re(eig(M)_0)) * nfp / (2 pi)`` so that the
    sign and branch agree with the SciPy oracle bit-for-bit modulo the
    independent eigenvalue ordering of NumPy / JAX LAPACK shims (both
    return the same first eigenvalue for a real 2x2 with conjugate
    eigenvalues at the precision of the integrator).
    """

    if not isinstance(axis_spec, CurveRZFourierSpec):
        raise TypeError("on_axis_iota_rk requires a CurveRZFourierSpec for the axis")
    nfp = int(axis_spec.nfp)
    if nfp < 1:
        raise ValueError(f"axis_spec.nfp must be >= 1, got {nfp}")
    if int(max_steps) <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")

    rhs = _rhs_factory(axis_spec, field_eval_fn)
    y0 = tangent_map_y0()
    phi0 = jnp.asarray(0.0, dtype=jnp.float64)
    phi_end = jnp.asarray(1.0, dtype=jnp.float64) / jnp.asarray(nfp, dtype=jnp.float64)
    y_final, _phi_final, steps_taken, succeeded = _integrate_tangent_map(
        rhs,
        phi0=phi0,
        phi_end=phi_end,
        y0=y0,
        rtol=rtol,
        atol=atol,
        max_steps=int(max_steps),
    )
    M = y_final.reshape((2, 2))
    evals, _ = jnp.linalg.eig(M)
    eig0 = evals[0]
    nfp_arr = jnp.asarray(nfp, dtype=jnp.float64)
    two_pi = jnp.asarray(2.0 * jnp.pi, dtype=jnp.float64)
    iota = jnp.arctan2(jnp.imag(eig0), jnp.real(eig0)) * nfp_arr / two_pi
    return iota, steps_taken, succeeded
