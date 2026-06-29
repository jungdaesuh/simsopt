"""Differentiable edge-delivered-iota proxy (exact field-line-winding identity).

This is a cheap, differentiable surrogate for the expensive field-line-trace
oracle in ``edge_delivered_iota.py``. It exists so a gradient-based Stage 2
optimizer can be *steered* toward delivering external rotational transform to
the fixed HBT-EP plasma edge, without running a multi-turn trace in the hot loop.

Formulation (exact axisymmetric field-line winding identity)
------------------------------------------------------------
On a fixed flux contour ``C`` (built from the EQDSK ``psi`` geometry, NOT from
the banana coils) the rotational transform of a field line satisfies

    iota = 2*pi / oint_C ( B_phi / (R * B_pol) ) dl_pol ,   B_pol = sqrt(B_R^2 + B_Z^2)

because along a field line ``dl_pol / B_pol = R dphi / B_phi``. The contour points
are FIXED, so ``B = B_tokamak(const) + B_banana(coil DOFs)`` and ``iota`` is an
explicit differentiable function of the banana-coil DOFs. The gradient is taken
analytically through ``simsopt`` ``BiotSavart.B_vjp`` -- no finite differences,
no fabricated gradient.

The promotion-facing scalar matches the oracle's convention:
``delta_abs_iota = |iota_hybrid| - |iota_tokamak|`` (the tokamak term is constant).

Scope / honesty
---------------
This estimator assumes a nested flux surface *exists*. At a chaotic edge (no
surface) the estimate is a transform STEERING signal, not a confinement or chaos
guarantee. It is validated against the trace oracle only where the oracle is
itself reliable (tokamak-only 1/q anchor and surviving traces); see the lane
plan and tests.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from banana_opt.edge_delivered_iota import (
    AxisymmetricTokamakField,
    EqdskEquilibrium,
    edge_radial_labels,
)

DEFAULT_PROXY_POLOIDAL_POINTS = 96
DEFAULT_PROXY_PHI_PLANES = 4
# Outboard search reach for the flux-contour root find, in units of the minor
# radius: the ray from the axis is scanned for a psi crossing out to this fraction
# beyond r/a = 1 before bracketing.
PROXY_RAY_REACH_FRACTION = 1.35
PROXY_RAY_SCAN_POINTS = 64


@dataclass(frozen=True, slots=True)
class EdgeIotaProxyContours:
    """Fixed flux contours for the edge proxy, independent of banana-coil DOFs.

    ``points_xyz`` is the stacked Cartesian evaluation set (one row per contour
    point per toroidal plane) handed to ``BiotSavart.set_points``; every other
    per-point array is row-aligned to it so a single ``B_vjp`` recovers the full
    gradient. ``tokamak_cyl_B`` and ``iota_tokamak`` capture the constant tokamak
    field once so the hot path only evaluates the banana coils.
    """

    axis_r_m: float
    axis_z_m: float
    helicity_sign: int
    radial_labels: tuple[float, ...]
    phi_planes: tuple[float, ...]
    points_xyz: np.ndarray  # (Ntot, 3) Cartesian
    point_R: np.ndarray  # (Ntot,)
    point_phi: np.ndarray  # (Ntot,)
    dl_pol: np.ndarray  # (Ntot,) poloidal arc-length weight
    segment_label: np.ndarray  # (Ntot,) index into radial_labels
    segment_plane: np.ndarray  # (Ntot,) index into phi_planes
    tokamak_cyl_B: np.ndarray  # (Ntot, 3) constant tokamak (B_R, B_phi, B_Z)
    iota_tokamak: np.ndarray  # (n_labels,) tokamak-only proxy iota per label

    @property
    def n_labels(self) -> int:
        return len(self.radial_labels)

    @property
    def n_planes(self) -> int:
        return len(self.phi_planes)


def _solve_flux_contour_point(
    tokamak_field: AxisymmetricTokamakField,
    *,
    axis_r_m: float,
    axis_z_m: float,
    angle_rad: float,
    target_psi: float,
    max_reach_m: float,
) -> tuple[float, float]:
    """Ray-cast one flux-contour point: solve psi(ray(t)) = target_psi for t>0."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def psi_minus_target(t: float) -> float:
        return (
            tokamak_field.psi(axis_r_m + t * cos_a, axis_z_m + t * sin_a)
            - target_psi
        )

    scan = np.linspace(1.0e-4, max_reach_m, PROXY_RAY_SCAN_POINTS)
    values = np.array([psi_minus_target(t) for t in scan])
    sign_change = np.where(values[:-1] * values[1:] <= 0.0)[0]
    if sign_change.size == 0:
        raise ValueError(
            "edge-iota proxy: no psi crossing along ray at "
            f"angle {angle_rad:.4f} rad (target_psi={target_psi:.6e}); "
            "contour does not close inside the search reach."
        )
    lo_index = int(sign_change[0])
    t_root = float(brentq(psi_minus_target, scan[lo_index], scan[lo_index + 1]))
    return (axis_r_m + t_root * cos_a, axis_z_m + t_root * sin_a)


def _tokamak_cyl_B_on_points(
    tokamak_field: AxisymmetricTokamakField,
    point_R: np.ndarray,
    point_phi: np.ndarray,
    point_Z: np.ndarray,
) -> np.ndarray:
    """Tokamak (B_R, B_phi, B_Z) at each (R, phi, Z). Shape (N, 3)."""
    out = np.empty((point_R.shape[0], 3), dtype=float)
    for i in range(out.shape[0]):
        out[i] = tokamak_field(
            float(point_R[i]), float(point_phi[i]), float(point_Z[i])
        )
    return out


def _banana_cyl_B_on_contours(
    banana_biot_savart, contours: EdgeIotaProxyContours
) -> np.ndarray:
    """Banana-coil (B_R, B_phi, B_Z) at every contour point. Shape (Ntot, 3)."""
    banana_biot_savart.set_points(contours.points_xyz)
    B_cart = np.asarray(banana_biot_savart.B(), dtype=float)  # (Ntot, 3) xyz
    cos_phi = np.cos(contours.point_phi)
    sin_phi = np.sin(contours.point_phi)
    B_R = B_cart[:, 0] * cos_phi + B_cart[:, 1] * sin_phi
    B_phi = -B_cart[:, 0] * sin_phi + B_cart[:, 1] * cos_phi
    B_Z = B_cart[:, 2]
    return np.column_stack((B_R, B_phi, B_Z))


@dataclass(frozen=True, slots=True)
class _WindingForward:
    """SSOT forward pass of the field-line-winding identity on a contour set.

    Holds the per-label ``iota`` plus the intermediates the analytic gradient
    needs (``plane_integral``, ``inv = 1/(R*B_pol)``, ``B_pol`` and the cylindrical
    field components), so the value path and the gradient path never transcribe
    the identity twice.
    """

    iota: np.ndarray  # (n_labels,)
    plane_integral: np.ndarray  # (n_labels, n_planes)
    inv: np.ndarray  # (Ntot,) = 1/(R*B_pol)
    B_pol: np.ndarray  # (Ntot,)
    B_R: np.ndarray  # (Ntot,)
    B_phi: np.ndarray  # (Ntot,)
    B_Z: np.ndarray  # (Ntot,)


def _winding_forward(
    cyl_B: np.ndarray,
    *,
    point_R: np.ndarray,
    dl_pol: np.ndarray,
    segment_label: np.ndarray,
    segment_plane: np.ndarray,
    n_labels: int,
    n_planes: int,
    helicity_sign: int,
) -> _WindingForward:
    """Per-label proxy iota = mean over planes of s_theta * 2pi / I_plane."""
    B_R = cyl_B[:, 0]
    B_phi = cyl_B[:, 1]
    B_Z = cyl_B[:, 2]
    B_pol = np.hypot(B_R, B_Z)
    inv = 1.0 / (point_R * B_pol)
    integrand = B_phi * inv * dl_pol
    segment_index = segment_label * int(n_planes) + segment_plane
    plane_integral = np.bincount(
        segment_index,
        weights=integrand,
        minlength=int(n_labels) * int(n_planes),
    ).reshape((n_labels, n_planes))
    plane_iota = float(helicity_sign) * 2.0 * math.pi / plane_integral
    return _WindingForward(
        iota=np.mean(plane_iota, axis=1),
        plane_integral=plane_integral,
        inv=inv,
        B_pol=B_pol,
        B_R=B_R,
        B_phi=B_phi,
        B_Z=B_Z,
    )


def build_edge_iota_proxy_contours(
    tokamak_field: AxisymmetricTokamakField,
    *,
    eqdsk: EqdskEquilibrium,
    minor_radius_m: float,
    edge_band: tuple[float, float],
    sample_count: int,
    helicity_sign: int,
    poloidal_points: int = DEFAULT_PROXY_POLOIDAL_POINTS,
    phi_planes: int = DEFAULT_PROXY_PHI_PLANES,
) -> EdgeIotaProxyContours:
    """Build the fixed flux contours and tokamak-only iota anchor for the proxy."""
    if poloidal_points < 8 or phi_planes < 1:
        raise ValueError("edge-iota proxy requires poloidal_points>=8 and phi_planes>=1.")
    axis_r = float(eqdsk.rmaxis)
    axis_z = float(eqdsk.zmaxis)
    labels = edge_radial_labels(edge_band, sample_count)
    angles = np.linspace(0.0, 2.0 * math.pi, poloidal_points, endpoint=False)
    planes = tuple(float(2.0 * math.pi * j / phi_planes) for j in range(phi_planes))
    max_reach = PROXY_RAY_REACH_FRACTION * float(minor_radius_m)

    point_xyz: list[list[float]] = []
    point_R: list[float] = []
    point_phi: list[float] = []
    point_Z: list[float] = []
    dl_pol: list[float] = []
    segment_label: list[int] = []
    segment_plane: list[int] = []

    for label_index, r_over_a in enumerate(labels):
        target_psi = tokamak_field.psi(
            axis_r + float(r_over_a) * float(minor_radius_m), axis_z
        )
        contour = np.array(
            [
                _solve_flux_contour_point(
                    tokamak_field,
                    axis_r_m=axis_r,
                    axis_z_m=axis_z,
                    angle_rad=float(angle),
                    target_psi=float(target_psi),
                    max_reach_m=max_reach,
                )
                for angle in angles
            ]
        )  # (M, 2): (R, Z)
        R = contour[:, 0]
        Z = contour[:, 1]
        # Closed-loop poloidal arc length: symmetric segment weight per vertex.
        seg_fwd = np.hypot(np.roll(R, -1) - R, np.roll(Z, -1) - Z)
        seg_bwd = np.hypot(R - np.roll(R, 1), Z - np.roll(Z, 1))
        weights = 0.5 * (seg_fwd + seg_bwd)
        for plane_index, phi in enumerate(planes):
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            for m in range(len(R)):
                point_xyz.append([R[m] * cos_phi, R[m] * sin_phi, Z[m]])
                point_R.append(float(R[m]))
                point_phi.append(float(phi))
                point_Z.append(float(Z[m]))
                dl_pol.append(float(weights[m]))
                segment_label.append(label_index)
                segment_plane.append(plane_index)

    point_R_arr = np.asarray(point_R, dtype=float)
    point_phi_arr = np.asarray(point_phi, dtype=float)
    point_Z_arr = np.asarray(point_Z, dtype=float)
    dl_pol_arr = np.asarray(dl_pol, dtype=float)
    segment_label_arr = np.asarray(segment_label, dtype=int)
    segment_plane_arr = np.asarray(segment_plane, dtype=int)
    tokamak_cyl_B = _tokamak_cyl_B_on_points(
        tokamak_field, point_R_arr, point_phi_arr, point_Z_arr
    )
    # Tokamak-only anchor iota per label (constant; banana current = 0). Computed
    # from the local arrays so the frozen contour set is constructed exactly once.
    iota_tokamak = _winding_forward(
        tokamak_cyl_B,
        point_R=point_R_arr,
        dl_pol=dl_pol_arr,
        segment_label=segment_label_arr,
        segment_plane=segment_plane_arr,
        n_labels=len(labels),
        n_planes=len(planes),
        helicity_sign=helicity_sign,
    ).iota

    return EdgeIotaProxyContours(
        axis_r_m=axis_r,
        axis_z_m=axis_z,
        helicity_sign=int(helicity_sign),
        radial_labels=tuple(float(x) for x in labels),
        phi_planes=planes,
        points_xyz=np.ascontiguousarray(point_xyz, dtype=float),
        point_R=point_R_arr,
        point_phi=point_phi_arr,
        dl_pol=dl_pol_arr,
        segment_label=segment_label_arr,
        segment_plane=segment_plane_arr,
        tokamak_cyl_B=tokamak_cyl_B,
        iota_tokamak=iota_tokamak,
    )


@dataclass(frozen=True, slots=True)
class EdgeIotaProxyResult:
    """Proxy edge-transform value and its gradient w.r.t. banana DOFs."""

    iota_hybrid: np.ndarray  # (n_labels,)
    iota_tokamak: np.ndarray  # (n_labels,)
    delta_abs: np.ndarray  # (n_labels,) = |iota_hybrid| - |iota_tokamak|
    delta_abs_mean: float
    delta_abs_p10: float
    grad_delta_abs_mean: np.ndarray  # d(delta_abs_mean)/d(banana_bs.x)


def edge_iota_proxy_value_and_grad(
    banana_biot_savart,
    contours: EdgeIotaProxyContours,
    *,
    grad_optimizable=None,
) -> EdgeIotaProxyResult:
    """Evaluate the differentiable edge-iota proxy and its analytic gradient.

    ``banana_biot_savart`` supplies the (coil-DOF-dependent) field; tokamak terms
    are constant and drop out of the gradient. The gradient of the edge-band-mean
    ``delta_abs`` is returned via a single ``B_vjp`` call, projected onto the free
    DOFs of ``grad_optimizable`` -- pass the full Stage 2 ``BiotSavart`` there so
    the banana-slice gradient lands in the full optimizer DOF vector; defaults to
    ``banana_biot_savart`` for standalone use.
    """
    target_optimizable = (
        banana_biot_savart if grad_optimizable is None else grad_optimizable
    )
    banana_cyl_B = _banana_cyl_B_on_contours(banana_biot_savart, contours)
    hybrid_cyl_B = banana_cyl_B + contours.tokamak_cyl_B

    n_labels = contours.n_labels
    n_planes = contours.n_planes
    s = float(contours.helicity_sign)
    fwd = _winding_forward(
        hybrid_cyl_B,
        point_R=contours.point_R,
        dl_pol=contours.dl_pol,
        segment_label=contours.segment_label,
        segment_plane=contours.segment_plane,
        n_labels=n_labels,
        n_planes=n_planes,
        helicity_sign=contours.helicity_sign,
    )
    B_R, B_phi, B_Z = fwd.B_R, fwd.B_phi, fwd.B_Z
    B_pol, inv, plane_integral = fwd.B_pol, fwd.inv, fwd.plane_integral
    iota_hybrid = fwd.iota
    iota_tokamak = contours.iota_tokamak
    delta_abs = np.abs(iota_hybrid) - np.abs(iota_tokamak)

    # ----- analytic gradient of mean_label(|iota_hybrid|) w.r.t. banana DOFs -----
    # delta_abs_mean = (1/n_labels) sum_label (|iota_label| - |iota_tok_label|)
    # iota_label = (1/n_planes) sum_plane s*2pi/I_{label,plane}
    # d iota_label / d I_{label,plane} = -(1/n_planes) s*2pi / I^2
    sign_iota = np.sign(iota_hybrid)
    d_iota_d_plane_integral = -(s * 2.0 * math.pi) / (n_planes * plane_integral**2)
    # d(delta_abs_mean)/d(integrand at a point) routed through its (label, plane):
    point_weight = (
        (sign_iota / n_labels)[contours.segment_label]
        * d_iota_d_plane_integral[contours.segment_label, contours.segment_plane]
    )  # (Ntot,)

    # integrand = B_phi * inv * dl, inv = 1/(R*B_pol), B_pol = hypot(B_R, B_Z)
    dl = contours.dl_pol
    d_int_dBphi = inv * dl
    common = -B_phi * dl * inv * inv * contours.point_R / B_pol
    d_int_dBR = common * B_R
    d_int_dBZ = common * B_Z

    dL_dBR = point_weight * d_int_dBR
    dL_dBphi = point_weight * d_int_dBphi
    dL_dBZ = point_weight * d_int_dBZ

    # rotate cotangent from cylindrical (B_R, B_phi, B_Z) back to Cartesian
    cos_phi = np.cos(contours.point_phi)
    sin_phi = np.sin(contours.point_phi)
    dL_dBx = dL_dBR * cos_phi - dL_dBphi * sin_phi
    dL_dBy = dL_dBR * sin_phi + dL_dBphi * cos_phi
    dL_dBz = dL_dBZ
    cotangent = np.ascontiguousarray(
        np.column_stack((dL_dBx, dL_dBy, dL_dBz)), dtype=float
    )

    grad = banana_biot_savart.B_vjp(cotangent)(target_optimizable)

    return EdgeIotaProxyResult(
        iota_hybrid=iota_hybrid,
        iota_tokamak=np.asarray(iota_tokamak, dtype=float),
        delta_abs=delta_abs,
        delta_abs_mean=float(np.mean(delta_abs)),
        delta_abs_p10=float(np.percentile(delta_abs, 10)),
        grad_delta_abs_mean=np.asarray(grad, dtype=float),
    )
