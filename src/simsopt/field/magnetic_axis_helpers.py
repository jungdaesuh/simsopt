import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from simsopt.geo import CurveRZFourier

__all__ = [
    "MagneticAxisNotLocatedError",
    "compute_on_axis_iota",
    "locate_magnetic_axis_point",
]


class MagneticAxisNotLocatedError(RuntimeError):
    """Raised when the magnetic-axis fixed-point solve cannot locate an acceptable axis."""


def _cylindrical_basis(phi):
    c = math.cos(float(phi))
    s = math.sin(float(phi))
    return (
        np.asarray([c, s, 0.0], dtype=float),
        np.asarray([-s, c, 0.0], dtype=float),
        np.asarray([0.0, 0.0, 1.0], dtype=float),
    )


def _fieldline_rhs_phi(phi, state, magnetic_field, min_bphi_over_b):
    radius = float(state[0])
    z = float(state[1])
    if radius <= 0.0 or not math.isfinite(radius) or not math.isfinite(z):
        raise ValueError("magnetic-axis state requires finite R > 0 and finite Z")

    e_r, e_phi, e_z = _cylindrical_basis(phi)
    point = np.asarray(
        [[radius * math.cos(float(phi)), radius * math.sin(float(phi)), z]],
        dtype=float,
    )
    magnetic_field.set_points(point)
    b_vector = np.asarray(magnetic_field.B(), dtype=float)
    if b_vector.shape != (1, 3):
        raise ValueError(
            f"Magnetic field B() must have shape (1, 3), got {b_vector.shape}"
        )
    b = b_vector[0]
    b_norm = float(np.linalg.norm(b))
    if b_norm <= 0.0 or not math.isfinite(b_norm):
        raise ValueError("Magnetic field norm must be finite and positive")
    b_phi = float(np.dot(b, e_phi))
    if abs(b_phi) / b_norm <= float(min_bphi_over_b):
        raise ValueError("|B_phi|/|B| fell below the magnetic-axis threshold")
    return np.asarray(
        [
            radius * float(np.dot(b, e_r)) / b_phi,
            radius * float(np.dot(b, e_z)) / b_phi,
        ],
        dtype=float,
    )


def _axis_return_residual(
    state,
    *,
    magnetic_field,
    phi0,
    toroidal_span,
    scale,
    rtol,
    atol,
    max_step,
    min_bphi_over_b,
):
    start = np.asarray(state, dtype=float)
    solution = solve_ivp(
        lambda phi, rz: _fieldline_rhs_phi(
            phi,
            rz,
            magnetic_field,
            min_bphi_over_b,
        ),
        (float(phi0), float(phi0) + float(toroidal_span)),
        start,
        method="DOP853",
        rtol=float(rtol),
        atol=float(atol),
        max_step=float(max_step),
        t_eval=(float(phi0) + float(toroidal_span),),
    )
    if not solution.success:
        raise MagneticAxisNotLocatedError(
            f"magnetic-axis return-map integration failed: {solution.message}"
        )
    return (np.asarray(solution.y[:, -1], dtype=float) - start) / float(scale)


def locate_magnetic_axis_point(
    magnetic_field,
    initial_guess,
    *,
    nfp=1,
    phi0=0.0,
    r_bounds=None,
    z_bounds=None,
    residual_tolerance=1.0e-6,
    optimizer_tolerance=1.0e-10,
    rtol=1.0e-9,
    atol=1.0e-11,
    max_step=0.05,
    min_bphi_over_b=1.0e-8,
    max_nfev=80,
):
    """Locate the magnetic-axis fixed point in an R/Z Poincare section.

    The returned point is the root of ``P(R, Z) - (R, Z)`` for the field-line
    return map over one full toroidal turn (2*pi). Callers may seed the solve
    from a boundary centroid, but the returned point is accepted only from the
    field-line fixed-point residual.

    The full-torus turn is used rather than a single field period (2*pi/nfp)
    because the axis is a fixed point of the field-period map only when the field
    has exact discrete nfp symmetry. Optimized coil fields break that symmetry,
    so their axis closes only over the full torus; ``nfp`` is retained as
    device metadata and for input validation, not to shorten the return map.
    """
    guess = np.asarray(initial_guess, dtype=float)
    if guess.shape != (2,):
        raise ValueError("magnetic-axis initial_guess must be [R, Z]")
    if guess[0] <= 0.0 or not np.all(np.isfinite(guess)):
        raise ValueError(
            "magnetic-axis initial_guess requires finite R > 0 and finite Z"
        )
    resolved_nfp = int(nfp)
    if resolved_nfp <= 0:
        raise ValueError("magnetic-axis locator requires nfp > 0")
    accept_tolerance = float(residual_tolerance)
    if accept_tolerance <= 0.0 or not math.isfinite(accept_tolerance):
        raise ValueError("magnetic-axis residual_tolerance must be finite and positive")
    solver_tolerance = float(optimizer_tolerance)
    if solver_tolerance <= 0.0 or not math.isfinite(solver_tolerance):
        raise ValueError(
            "magnetic-axis optimizer_tolerance must be finite and positive"
        )

    lower_r, upper_r = (
        (np.finfo(float).eps, np.inf) if r_bounds is None else tuple(r_bounds)
    )
    lower_z, upper_z = (-np.inf, np.inf) if z_bounds is None else tuple(z_bounds)
    lower = np.asarray([float(lower_r), float(lower_z)], dtype=float)
    upper = np.asarray([float(upper_r), float(upper_z)], dtype=float)
    if not np.all(lower < upper):
        raise ValueError("magnetic-axis bounds must be ordered")

    scale = max(float(abs(guess[0])), 1.0)
    result = least_squares(
        lambda state: _axis_return_residual(
            state,
            magnetic_field=magnetic_field,
            phi0=float(phi0),
            toroidal_span=2.0 * math.pi,
            scale=scale,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            min_bphi_over_b=min_bphi_over_b,
        ),
        guess,
        bounds=(lower, upper),
        xtol=solver_tolerance,
        ftol=solver_tolerance,
        gtol=solver_tolerance,
        max_nfev=int(max_nfev),
    )
    residual_norm = float(np.linalg.norm(result.fun))
    if residual_norm > accept_tolerance:
        raise MagneticAxisNotLocatedError(
            "magnetic-axis fixed-point solve did not converge below "
            f"{accept_tolerance:g}; residual={residual_norm:g}; "
            f"optimizer_success={bool(result.success)}; "
            f"optimizer_message={result.message}"
        )
    return {
        "r": float(result.x[0]),
        "z": float(result.x[1]),
        "phi": float(phi0),
        "nfp": resolved_nfp,
        "source": "magnetic_axis_fieldline_fixed_point",
        "normalized_return_residual": residual_norm,
        "residual_accept_tolerance": accept_tolerance,
        "optimizer_tolerance": solver_tolerance,
        "optimizer_success": bool(result.success),
        "iterations": int(result.nfev),
    }


def compute_on_axis_iota(axis, magnetic_field):
    """
    Computes the rotational transform on the magnetic axis of a device using a method based on
    equation (13) of Greene, Journal of Mathematical Physics 20, 1183 (1979); doi: 10.1063/1.524170.
    This method was shared by Stuart Hudson and Matt Landreman.  NOTE: this function does not check that the provided
    axis is actually a magnetic axis of the input magnetic_field.

    Args:
        axis: a CurveRZFourier corresponding to the magnetic axis of the magnetic field.
        magnetic_field: the magnetic field in which the axis lies.

    Returns:
        iota: the rotational transform on the given axis.
    """
    assert type(axis) is CurveRZFourier

    def tangent_map(phi, x):
        ind = np.array(phi)
        out = np.zeros((1, 3))
        axis.gamma_impl(out, ind)
        magnetic_field.set_points(out)

        out = out.flatten()
        B = magnetic_field.B().flatten()
        dB_by_dX = magnetic_field.dB_by_dX().reshape((3, 3))
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

        c = np.cos(2 * np.pi * phi)
        s = np.sin(2 * np.pi * phi)

        R = np.sqrt(out[0] ** 2 + out[1] ** 2)
        dx_dR = c
        dy_dR = s

        BR = c * B1 + s * B2
        Bphi = -s * B1 + c * B2
        BZ = B3

        dB1_dR = dB1_dx * dx_dR + dB1_dy * dy_dR
        dB2_dR = dB2_dx * dx_dR + dB2_dy * dy_dR
        dB3_dR = dB3_dx * dx_dR + dB3_dy * dy_dR

        dBR_dR = c * dB1_dR + s * dB2_dR
        dBphi_dR = -s * dB1_dR + c * dB2_dR
        dBZ_dR = dB3_dR

        dBR_dZ = c * dB1_dz + s * dB2_dz
        dBphi_dZ = -s * dB1_dz + c * dB2_dz
        dBZ_dZ = dB3_dz

        Bphi_R = Bphi / R
        d_Bphi_R_dR = (dBphi_dR * R - Bphi) / R**2
        d_Bphi_R_dZ = dBphi_dZ / R

        A11 = (dBR_dR - (BR / Bphi_R) * d_Bphi_R_dR) / Bphi_R
        A21 = (dBZ_dR - (BZ / Bphi_R) * d_Bphi_R_dR) / Bphi_R
        A12 = (dBR_dZ - (BR / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
        A22 = (dBZ_dZ - (BZ / Bphi_R) * d_Bphi_R_dZ) / Bphi_R
        A = np.array([[A11, A12], [A21, A22]])
        return 2 * np.pi * np.array([A @ x[:2], A @ x[2:]]).flatten()

    t_span = [0, 1 / axis.nfp]
    t_eval = t_span

    y0 = np.array([1, 0, 0, 1])
    results = solve_ivp(
        tangent_map, t_span, y0, t_eval=t_eval, rtol=1e-12, atol=1e-12, method="RK45"
    )
    M = results.y[:, -1].reshape((2, 2))
    evals, evecs = np.linalg.eig(M)
    iota = np.arctan2(np.imag(evals[0]), np.real(evals[0])) * axis.nfp / (2 * np.pi)
    return iota
