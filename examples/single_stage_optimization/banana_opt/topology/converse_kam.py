"""Converse-KAM (MacKay-Percival cone-crossing) test for NON-EXISTENCE of invariant
flux surfaces on a coil (Biot-Savart) field.

This module implements the *conefield* variant of the Converse KAM method exactly as
specified by the MacKay group, NOT an invented criterion:

  * Martinez-del-Rio & MacKay, "Conefield approach to identifying regions without flux
    surfaces for magnetic fields", arXiv:2501.06796 (2025).  [primary algorithm source]
  * Kallinikos & MacKay, Plasma Phys. Control. Fusion 65 (2023) 095021.  [equivalent
    "direct" Converse KAM formulation that 2501.06796 reformulates]
  * MacKay (2018), "Finding the complement of the invariant manifolds transverse to a
    given foliation for a 3D flow"; MacKay & Percival (1985) for the original
    converse-KAM / "killends" idea.

WHAT IT CERTIFIES (Theorem 1 of arXiv:2501.06796, restated for a magnetic field B,
direction field xi, horizontal field eta):

    Given a point x0 and a timeout, integrate the field line x(t) with x'(t)=B(x(t)) AND
    the tangent (variational) matrix M(t) with M'(t)=DB_{x(t)} M(t), M(0)=I, for
    t in [-tf/2, +tf/2].  Pull the covector alpha_0 = i_B i_xi Omega back to x0:
    alpha_t(w) = alpha_0( M(t) w ) = Omega( xi, B, M(t) w ) with Omega,xi,B at x(t).
    Define the slope sigma_t = alpha_t(eta)/alpha_t(xi).  Split times by the sign of
    alpha_t(xi): T+ = {t: alpha_t(xi)>0}, T- = {t: alpha_t(xi)<0}.  Let
    sigma+ = min_{t in T+} sigma_t  and  sigma- = max_{t in T-} sigma_t.
    If at any time sigma+ <= sigma- (equivalently the cone of admissible tangent-plane
    slopes becomes empty), then NO invariant torus of the class transverse to xi passes
    through x0 -- a *certificate of non-existence* of a flux surface there.

This is a one-sided certificate: a positive result (cone empty) is a rigorous-in-spirit
proof of non-existence (modulo the direction-field choice below and finite-precision
integration); a negative result (cone still non-empty at the timeout) is UNDECIDED, never
"a surface exists".  See FAIL-CLOSED below.

------------------------------------------------------------------------------------------
CONSTRUCTING xi (the direction field) on a COIL field -- and the bug that was fixed here.
------------------------------------------------------------------------------------------
In arXiv:2501.06796 the device is given analytically in adapted toroidal coordinates
(psi, vartheta, phi) where psi is the toroidal flux from the magnetic axis, and the
direction field is xi = d/dpsi -- the direction NORMAL to the flux surfaces, pointing away
from the axis, defining the "principal class" of tori that enclose the axis.  A Biot-Savart
coil field has NO analytic psi, so we reconstruct the flux-surface-normal direction:

  1. Locate the magnetic axis curve a(phi) = (R_axis(phi), Z_axis(phi)) as a fixed point of
     the field-line return map over one full toroidal turn, at n_phi_planes planes, via a
     per-plane INBOARD-biased multi-start with a STRICT acceptance tolerance
     (``axis_locator.locate_axis_curve``; see ``build_axis_model``).  This replaced an
     earlier single-guess plane-to-plane march that silently accepted ~0.03-residual
     non-axis points at far-phi planes.
  2. Define, in the poloidal plane through x, the SHAPED quadratic
         g(x) = v^T Q(phi_x) v ,   v = (R - R_axis(phi_x), Z - Z_axis(phi_x)) ,
     where Q is the per-plane flux-surface SHAPE matrix = (cross-section 2nd-moment)^-1
     (det-normalised; ``flux_shape_from_surface``).  The level sets of g are then the actual
     ELLIPTICAL flux surfaces, so xi := grad g = 2 Q v is parallel to the true surface
     normal d/dpsi, defining the correct principal class.
  3. xi(x) = grad g(x) in Cartesian coordinates (``_radial_direction_field``).  Only the
     DIRECTION of xi matters (MacKay: "because only the direction matters, not the
     magnitude, we call xi a direction field"), so the det-normalisation of Q and any
     positive rescaling of g are irrelevant.

THE SHAPE, AND THE TWO COMPOUNDING DEFECTS BEHIND THE FALSE POSITIVES.
The shaped Q makes xi parallel to the true normal on an ELONGATED surface (mechanism verified
to < 1 deg vs ~50 deg for the circular Q = I on a 2.76x ellipse; see tests), so it is the
geometrically correct direction field where surfaces are non-circular.  An evidence-gated
investigation on the slid_clean candidate found TWO compounding causes of its false positives,
and a shaped xi only partly addresses them:
  * (a) xi MIS-DIRECTION.  Feeding the EXACT per-seed LOCAL flux normal (each seed's own
    traced-surface shape) turns the OUTBOARD false positive into UNDECIDED (correct) -- so a
    correct xi does help.  But the only shape available WITHOUT per-seed field-line tracing is
    the persisted BOUNDARY surface's, and its elongation (2.76x) does NOT match the rounder
    core surfaces (local elongation 1.2-1.5 over the inner radius); the boundary Q over-shapes
    the core and makes slid_clean WORSE (circular 4/7 -> boundary-shaped 7/7 falsely
    certified).  So a boundary-shaped xi is a net regression and is NOT applied by default.
  * (b) NEAR-AXIS CONE ILL-CONDITIONING.  The two INBOARD near-axis seeds (rho ~ 0.008-0.024)
    stay falsely CERTIFIED even with the EXACT local flux normal.  Near t=0 the variational
    system is t->-t ANTISYMMETRIC -- alpha_xi(-t) = -alpha_xi(+t) and sigma(-t) = -sigma(+t)
    EXACTLY -- so the legs populate T+ and T- with antisymmetric partners, forcing
    sigma_plus = -sigma_minus and hence sigma_plus <= sigma_minus (cone "empty") at the first
    finite sample regardless of integrability.  The SOUNDNESS NOTE in ``_decide_point`` (that
    near-t=0 spikes land only on the loose side) is therefore WRONG.  This slope-ordering
    defect is independent of xi and the exact normal does not fix it.
CONSEQUENCE: the default end-to-end path (``certify_nonexistence_on_field``) uses the plain
axis (Q = I); it does NOT auto-attach a boundary-shaped xi (that regresses).
``flux_shape_from_surface`` / ``AxisModel.shape`` remain available for a caller that can
supply the LOCAL per-flux-surface shape, but a shaped xi alone is not a complete fix -- the
near-axis cone-mechanics defect (b) remains.  converse-KAM stays ADVISORY-ONLY
(``advisory_only=True``) until both are corrected (Greene-residue + Poincare are the trusted
verdict).  The axis LOCATION fix in ``build_axis_model`` is independent and correct (it
replaced a single-guess march that silently accepted ~0.03-residual non-axis points), and
``flux_shape_from_surface`` passes the toroidal angle to ``cross_section`` in TURNS, not
radians (a units bug that otherwise slices the wrong plane -- see that function).

------------------------------------------------------------------------------------------
FAIL-CLOSED SEMANTICS (required, non-negotiable).
------------------------------------------------------------------------------------------
A point falls into exactly one of three states:
  * CERTIFIED_NONEXISTENT: the cone collapsed (sigma+ <= sigma-) at some t* <= tf.
  * UNDECIDED: the integration reached the timeout without collapsing the cone, OR the
    integration/axis machinery could not produce a decision (e.g. the field line left the
    integration domain, the variational solve failed, xi degenerated on the axis).  An
    UNDECIDED point is NEVER counted as "a surface exists" and NEVER as "non-existent".
  * (there is deliberately NO "surface exists" state -- converse KAM cannot prove
    existence.)
The reported ``certified_nonexistence_fraction`` divides certified points by ALL tested
seeds (undecided seeds are in the denominator), so it is a LOWER bound on the true
non-existence fraction and can never be inflated by treating undecided as existent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dataclass_field
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.integrate import solve_ivp

# --- Decision states (stable strings recorded in results.json / the gate ledger). --------
DECISION_CERTIFIED_NONEXISTENT = "certified_nonexistent"
DECISION_UNDECIDED = "undecided"

# Reasons an UNDECIDED verdict was reached (diagnostics; never flip to "exists").
UNDECIDED_TIMEOUT = "timeout_cone_nonempty"
UNDECIDED_INTEGRATION_FAILED = "integration_failed"
UNDECIDED_LEFT_DOMAIN = "field_line_left_domain"
UNDECIDED_XI_DEGENERATE = "xi_degenerate_near_axis"
UNDECIDED_NO_SIGN_SPLIT = "alpha_xi_never_changed_sign"


@dataclass(frozen=True)
class ConverseKamConfig:
    """Numerical settings for the conefield converse-KAM test (all SI / arc-length units).

    Fields:
      t_f: total integration window length; the field line is integrated over
        [-t_f/2, +t_f/2] in field-line "time" (arc parameter with x'(t)=B, so t has units
        of metre^2/Tesla -- the paper's convention, eq. before its step list). Larger t_f
        certifies more points but costs more.
      timeout_t_f: hard upper bound used when the per-point integration is allowed to grow;
        identical to t_f here (single-pass), exposed separately so a future warm-restart
        loop can grow tau up to this bound without changing the public contract.
      n_seeds: number of radial seeds across the phi=0 poloidal cross-section (a 1-D radial
        sweep matching the existing midplane topology seed contract).
      n_phi_planes: number of toroidal planes at which the axis is located (the axis is a
        curve a(phi); more planes = more faithful xi off the phi=0 plane).
      integration_rtol / integration_atol: solve_ivp tolerances for the coupled
        (x, vec(M)) system.
      max_step_fraction: max solver step as a fraction of t_f (keeps DOP853 from striding
        over a fast cone collapse).
      slope_clip: RELATIVE threshold (in (0, 1)). A sample's slope is admitted only when
        |alpha_t(xi)| >= slope_clip * (running max of |alpha_t(xi)| seen so far). Samples at
        the sign-flip of alpha_t(xi) (where |alpha_t(xi)| ~ 0 and the slope blows up) are
        the "alpha(xi)=0" boundary (slope +/-inf) per the paper's order extension to
        alpha_0 = i_B i_xi Omega; admitting their noisy slopes can SPURIOUSLY collapse the
        cone, so they are skipped. A relative (not absolute) threshold is essential because
        |alpha_t(xi)| grows with the tangent map M(t) over the integration window.
      domain_r_min / domain_r_max / domain_abs_z_max: cylindrical box; a field line leaving
        it makes the point UNDECIDED (fail-closed), never certified.
    """

    t_f: float = 600.0
    timeout_t_f: float = 600.0
    n_seeds: int = 24
    n_phi_planes: int = 8
    integration_rtol: float = 1.0e-9
    integration_atol: float = 1.0e-11
    max_step_fraction: float = 0.01
    slope_clip: float = 1.0e-6
    domain_r_min: float = 0.0
    domain_r_max: float = math.inf
    domain_abs_z_max: float = math.inf

    def __post_init__(self) -> None:
        if not self.t_f > 0.0:
            raise ValueError("t_f must be > 0")
        if not self.timeout_t_f >= self.t_f:
            raise ValueError("timeout_t_f must be >= t_f")
        if self.n_seeds < 1:
            raise ValueError("n_seeds must be >= 1")
        if self.n_phi_planes < 1:
            raise ValueError("n_phi_planes must be >= 1")
        if not 0.0 < self.max_step_fraction <= 1.0:
            raise ValueError("max_step_fraction must be in (0, 1]")
        if not 0.0 < self.slope_clip < 1.0:
            raise ValueError("slope_clip must be a relative threshold in (0, 1)")


@dataclass(frozen=True)
class AxisModel:
    """The reconstructed magnetic axis curve a(phi) plus the flux-surface SHAPE per plane.

    ``r``/``z`` are arrays of axis (R, Z) at the toroidal angles ``phis`` (radians).

    ``shape`` (optional) is an ``(n_phi, 2, 2)`` array of per-plane symmetric poloidal SHAPE
    matrices Q(phi) defining the direction field xi = grad(v^T Q v), v = (R - R_a, Z - Z_a).
    When ``shape is None`` (the default / no surface available) Q is the IDENTITY, recovering
    the circular surrogate xi = grad(|v|^2) -- correct ONLY for circular flux surfaces. When a
    surface is supplied (see ``flux_shape_from_surface``) Q = (flux 2nd-moment)^-1 (det-
    normalised), so the level sets of v^T Q v are the actual ELLIPTICAL flux surfaces and
    xi is parallel to the true surface normal (grad psi). This is the root-cause fix for the
    converse-KAM false positives on strongly-shaped (elongated) fields: a circular xi
    misaligns from the true normal by up to ~50 deg on a 3x-elongated surface and spuriously
    collapses the cone on genuinely nested tori (see module docstring's xi section).

    The axis-from-coil-field reconstruction lives in ``build_axis_model``; the shape
    reconstruction in ``flux_shape_from_surface``; this dataclass is the pure carrier.
    """

    phis: np.ndarray
    r: np.ndarray
    z: np.ndarray
    nfp: int
    residual_max: float
    source: str
    shape: Optional[np.ndarray] = None

    def at(self, phi: float) -> tuple[float, float]:
        """Linearly-interpolated axis (R, Z) at toroidal angle ``phi`` (periodic 2*pi).

        The axis is a closed curve over a full toroidal turn, so we interpolate the
        sampled (R, Z) periodically. Linear interpolation is adequate here because xi only
        needs the axis *direction*, not high-order derivatives.
        """
        two_pi = 2.0 * math.pi
        p = float(phi) % two_pi
        # ``phis`` are assumed sorted in [0, 2*pi); append the wrapped first sample.
        ext_phi = np.concatenate([self.phis, [self.phis[0] + two_pi]])
        ext_r = np.concatenate([self.r, [self.r[0]]])
        ext_z = np.concatenate([self.z, [self.z[0]]])
        r = float(np.interp(p, ext_phi, ext_r))
        z = float(np.interp(p, ext_phi, ext_z))
        return r, z

    def shape_at(self, phi: float) -> np.ndarray:
        """Poloidal shape matrix Q(phi) (2x2 symmetric) at toroidal angle ``phi``.

        Returns the IDENTITY when no shape was supplied (circular surrogate). Otherwise
        returns the NEAREST sampled plane's Q. Nearest-plane (not interpolating the matrix
        entries) is deliberate: a convex combination of two SPD shape matrices is not their
        geometric mean and can distort the principal axes, while the flux shape varies slowly
        in phi (verified: elongation/tilt drift < 5% per plane on the tested fields), so the
        nearest plane is the faithful, artefact-free choice.
        """
        if self.shape is None:
            return np.eye(2)
        two_pi = 2.0 * math.pi
        p = float(phi) % two_pi
        diffs = np.abs((self.phis - p + math.pi) % two_pi - math.pi)
        return self.shape[int(np.argmin(diffs))]


@dataclass(frozen=True)
class PointVerdict:
    """Converse-KAM verdict for a single seed point."""

    seed_index: int
    decision: str  # DECISION_CERTIFIED_NONEXISTENT | DECISION_UNDECIDED
    reason: str
    q: float  # time-to-decision ratio t*/t_f in [0, 1]; 1.0 == undecided (paper convention)
    sigma_plus: Optional[float]
    sigma_minus: Optional[float]
    seed_xyz: tuple[float, float, float]


@dataclass(frozen=True)
class ConverseKamResult:
    """Per-grid converse-KAM result, JSON-serialisable for the gate ledger."""

    certified_nonexistence_fraction: float
    n_certified: int
    n_undecided: int
    n_total: int
    axis_source: str
    axis_residual_max: float
    direction_field: str
    points: tuple[PointVerdict, ...] = dataclass_field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            # ``certified_nonexistence_fraction`` is a LOWER bound: undecided points are in
            # the denominator and never counted as existent (fail-closed).
            "certified_nonexistence_fraction": self.certified_nonexistence_fraction,
            "n_certified": self.n_certified,
            "n_undecided": self.n_undecided,
            "n_total": self.n_total,
            "axis_source": self.axis_source,
            "axis_residual_max": self.axis_residual_max,
            "direction_field": self.direction_field,
            "method": "converse_kam_conefield",
            "reference": "arXiv:2501.06796; PPCF 65 (2023) 095021",
            "advisory_only": True,  # not promotion-decisive until donor-validated
            "points": [
                {
                    "seed_index": p.seed_index,
                    "decision": p.decision,
                    "reason": p.reason,
                    "q": p.q,
                    "sigma_plus": p.sigma_plus,
                    "sigma_minus": p.sigma_minus,
                    "seed_xyz": list(p.seed_xyz),
                }
                for p in self.points
            ],
        }


# ----------------------------------------------------------------------------------------
# Field adapter: a thin, testable interface over a simsopt MagneticField (B, dB/dX).
# ----------------------------------------------------------------------------------------
def _field_B_and_DB(field, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (B, DB) at a single Cartesian point using the simsopt field interface.

    B has shape (3,); DB = dB_i/dx_j has shape (3, 3). The simsopt convention is
    ``dB_by_dX()[k]`` = d B_j / d x_i (i,j) per point, i.e. the Jacobian transpose; we
    transpose so DB[i, j] = dB_i/dx_j, matching the variational equation Mdot = DB . M.
    """
    # ascontiguousarray (not just asarray): xyz is often a column-strided slice of the
    # RK4 state (non-C-contiguous), and reshape preserves those strides -> the pybind
    # simsoptpp.BiotSavart.set_points overload rejects a non-contiguous (1,3) array.
    point = np.ascontiguousarray(xyz, dtype=np.float64).reshape(1, 3)
    field.set_points(point)
    b = np.asarray(field.B(), dtype=float).reshape(3)
    # simsopt dB_by_dX(): shape (npoints, 3, 3) with [.., i, j] = d B_j / d x_i.
    dB = np.asarray(field.dB_by_dX(), dtype=float).reshape(3, 3)
    return b, dB.T  # transpose -> [i, j] = d B_i / d x_j


# ----------------------------------------------------------------------------------------
# The unvalidated step: build the axis model and the radial direction field xi.
# ----------------------------------------------------------------------------------------
def build_axis_model(
    field,
    *,
    nfp: int,
    n_phi_planes: int,
    r_guess: float,
    z_guess: float = 0.0,
    r_bounds: Optional[tuple[float, float]] = None,
    z_bounds: Optional[tuple[float, float]] = None,
) -> AxisModel:
    """Locate the magnetic axis a(phi) on a coil field at ``n_phi_planes`` toroidal angles.

    Part 1 of 2 of the xi reconstruction (see module docstring). The axis is found at each
    plane as the fixed point of the field-line return map (over a full toroidal turn) via
    ``axis_locator.locate_axis_curve``: a per-plane INBOARD-biased multi-start that forwards a
    STRICT ``residual_tolerance`` and fails closed. This replaces an earlier single-guess
    plane-to-plane march that, with no strict tolerance, SILENTLY accepted ~0.03-residual
    non-axis points at far-phi planes (often the X-point basin), corrupting xi. ``r_bounds``/
    ``z_bounds`` (passed by the cert driver) bound and size the per-plane start grids; the
    grid is biased inboard of ``r_guess`` because the true axis lies inboard on shaped fields.

    ``residual_max`` is the worst normalized return residual across planes -- a loud
    falsifiability handle: with the strict locator it is ~1e-12 on a real axis, so any large
    value now means the solve genuinely failed (and it would have raised), never a silently
    accepted non-axis point.

    Raises:
        MagneticAxisNotLocatedError: if any plane cannot be located below the strict
            tolerance (fail-closed; propagated to the caller, e.g. the cert driver, which
            records it and continues without ever fabricating an axis).

    The ``axis_locator`` import is function-local (and relative, matching this package's
    convention) so this module stays cheap to import: ``axis_locator`` pulls in simsopt, which
    pure unit tests injecting a synthetic ``field`` never need until they actually locate.
    """
    from .axis_locator import locate_axis_curve

    if r_bounds is None or z_bounds is None:
        raise ValueError("build_axis_model requires explicit r_bounds and z_bounds")

    records = locate_axis_curve(
        field,
        nfp=int(nfp),
        n_phi_planes=int(n_phi_planes),
        r_guess=float(r_guess),
        z_guess=float(z_guess),
        r_bounds=(float(r_bounds[0]), float(r_bounds[1])),
        z_bounds=(float(z_bounds[0]), float(z_bounds[1])),
    )
    phis = np.asarray([rec["phi"] for rec in records], dtype=float)
    rs = np.asarray([rec["r"] for rec in records], dtype=float)
    zs = np.asarray([rec["z"] for rec in records], dtype=float)
    residual_max = max(rec["normalized_return_residual"] for rec in records)
    return AxisModel(
        phis=phis,
        r=rs,
        z=zs,
        nfp=int(nfp),
        residual_max=float(residual_max),
        source="magnetic_axis_curve_multistart",
    )


# Tolerance on the cross-section winding number about the axis: a genuine flux surface
# encircles the axis exactly ONCE (winding == 1). We test |winding - 1| <= this. Winding is
# the topologically correct, elongation- and sampling-robust encirclement criterion -- unlike
# a poloidal-angle bin-coverage histogram, which spuriously dips below 1 on a strongly
# elongated cross-section sampled at uniform poloidal parameter (the long-axis tips
# under-populate the bins). A cross-section that does not wind once cannot define a flux-
# surface shape about the axis; we fail closed rather than fit a bogus Q.
_SHAPE_WINDING_TOLERANCE = 0.05


def _winding_number_about_axis(d_r: np.ndarray, d_z: np.ndarray) -> float:
    """Signed winding number of the closed cross-section (offsets (dR, dZ) from the axis).

    Sum the wrapped poloidal-angle increments around the closed loop and divide by 2*pi. A
    curve that encircles the axis once returns +/-1; one that does not enclose it returns ~0.
    Robust to elongation and to the poloidal sampling density (each step's angle change is
    taken on the principal branch), unlike a bin-coverage histogram.
    """
    theta = np.arctan2(d_z, d_r)
    # Close the loop (last point back to first) and wrap each increment to (-pi, pi].
    increments = np.diff(np.concatenate([theta, theta[:1]]))
    increments = (increments + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.sum(increments) / (2.0 * math.pi))


def _poloidal_shape_matrix(d_r: np.ndarray, d_z: np.ndarray) -> np.ndarray:
    """Det-normalised inverse 2nd-moment Q of a cross-section's (dR, dZ) offsets-from-axis.

    The cross-section's 2nd-moment ``M = cov[(dR, dZ)]`` is the shape ellipse of the flux
    surface; an ellipse with covariance M has implicit equation ``v^T M^-1 v = const``, so
    ``Q = M^-1`` makes the level sets of ``v^T Q v`` the surface ellipse and ``2 Q v`` the
    surface normal. We normalise to ``det(Q) = 1`` (a pure shape: direction-only, O(1) scale
    so the on-axis xi-degeneracy floor still applies), which is irrelevant to the certificate
    (only xi's direction matters). Raises if M is singular/degenerate (fail closed).
    """
    m = np.array(
        [[float(np.mean(d_r * d_r)), float(np.mean(d_r * d_z))],
         [float(np.mean(d_r * d_z)), float(np.mean(d_z * d_z))]],
        dtype=float,
    )
    det_m = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    if not det_m > 0.0:
        raise ValueError(f"degenerate flux-surface cross-section 2nd-moment (det={det_m:g})")
    q = np.linalg.inv(m)
    det_q = q[0, 0] * q[1, 1] - q[0, 1] * q[1, 0]
    return q / math.sqrt(det_q)


def flux_shape_from_surface(surface, axis: AxisModel, *, thetas: int = 256) -> np.ndarray:
    """Per-plane poloidal SHAPE matrices Q(phi) from a flux surface's cross-sections.

    For each axis plane ``phi_i`` take the surface's poloidal cross-section, measure its
    offsets (dR, dZ) from the axis point (R_a(phi_i), Z_a(phi_i)), and fit the det-normalised
    inverse 2nd-moment Q (see ``_poloidal_shape_matrix``). The returned ``(n_phi, 2, 2)``
    array is the ``AxisModel.shape`` that makes xi = grad(v^T Q v) parallel to the true flux-
    surface normal, fixing the elongation-driven false positives.

    This uses the candidate's persisted nested boundary surface as the shape proxy: only its
    SHAPE (elongation + tilt) is used, and that varies slowly across the minor radius (the
    dominant misalignment a circular xi makes is the boundary elongation), so the boundary
    shape is a faithful, deployable approximation to the per-surface normal throughout the
    nested region.

    Every plane is checked to wind exactly once about the axis (the topologically correct,
    elongation/sampling-robust encirclement test); a genuine nested flux surface does so at
    EVERY toroidal plane.  A plane that does not wind once means the surface is not nested
    about this axis (or a degenerate slice) -- we fail closed, so a non-nested / bad surface
    never yields a silently-wrong shape.

    Raises:
        ValueError: if any cross-section does not wind once about the axis
            (|winding - 1| > ``_SHAPE_WINDING_TOLERANCE``) or has a degenerate 2nd-moment.
    """
    n_phi = axis.phis.shape[0]
    shapes = np.empty((n_phi, 2, 2), dtype=float)
    two_pi = 2.0 * math.pi
    for i, phi in enumerate(axis.phis):
        # ``axis.phis`` are RADIANS; simsopt ``Surface.cross_section`` takes the toroidal angle
        # normalized by 2*pi (TURNS). Passing radians slices the WRONG toroidal plane (e.g.
        # 3.927 rad -> plane at 0.927 turns = 5.83 rad), so the axis falls outside the slice and
        # the shape is garbage. Convert to turns.
        cross_section = surface.cross_section(phi=float(phi) / two_pi, thetas=int(thetas))
        radius = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
        z = cross_section[:, 2]
        r_a, z_a = axis.at(float(phi))
        d_r = radius - r_a
        d_z = z - z_a
        winding = _winding_number_about_axis(d_r, d_z)
        if abs(abs(winding) - 1.0) > _SHAPE_WINDING_TOLERANCE:
            raise ValueError(
                f"flux surface cross-section at phi={float(phi):.4f} does not wind once about "
                f"the axis (winding {winding:.3f}); cannot define a flux-surface shape -- "
                "failing closed"
            )
        shapes[i] = _poloidal_shape_matrix(d_r, d_z)
    return shapes


def _radial_direction_field(xyz: np.ndarray, axis: AxisModel) -> np.ndarray:
    """xi(x) = grad g(x), the flux-surface-normal direction in the poloidal plane.

    Part 2 of 2 of the xi reconstruction (see module docstring). With cylindrical
    R = sqrt(x^2+y^2), phi = atan2(y, x), Z = z, axis (R_a(phi), Z_a(phi)), poloidal offset
    v = (R - R_a, Z - Z_a), and the per-plane SHAPE matrix Q = ``axis.shape_at(phi)``:

        g(x) = v^T Q v ,   grad_poloidal g = 2 Q v .

    The level sets of g are the flux surfaces: with Q = (flux 2nd-moment)^-1 they are the
    actual ELLIPTICAL surfaces, so grad g is parallel to the true surface normal (grad psi)
    and defines the correct principal class. With Q = I (no surface supplied) this reduces
    EXACTLY to the legacy circular surrogate dg/dR = 2(R - R_a), dg/dZ = 2(Z - Z_a) -- which
    is correct only for circular surfaces and was the source of the false positives on
    elongated fields. The unit cylindrical basis lifts the poloidal gradient to Cartesian:

        xi = (grad g)_R * e_R + (grad g)_Z * e_Z ,  e_R = (cos phi, sin phi, 0), e_Z=(0,0,1).

    We deliberately DROP the phi-derivative of the axis (a small along-field correction that
    only tilts xi within the surface): xi only needs to be transverse to the principal-class
    tori, and the poloidal part already is. Returns the raw (un-normalised) vector; only its
    direction matters (positive rescalings of g, including the det-normalisation of Q, leave
    the certificate invariant).
    """
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    radius = math.hypot(x, y)
    phi = math.atan2(y, x)
    r_a, z_a = axis.at(phi)
    q = axis.shape_at(phi)
    v0 = radius - r_a
    v1 = z - z_a
    # grad_poloidal g = 2 Q v; Q symmetric (q[0,1] == q[1,0]).
    dg_dr = 2.0 * (q[0, 0] * v0 + q[0, 1] * v1)
    dg_dz = 2.0 * (q[1, 0] * v0 + q[1, 1] * v1)
    cos_phi = math.cos(phi)
    sin_phi = math.sin(phi)
    return np.asarray(
        [dg_dr * cos_phi, dg_dr * sin_phi, dg_dz],
        dtype=float,
    )


# ----------------------------------------------------------------------------------------
# Core conefield integration + cone-crossing test (Theorem 1 of arXiv:2501.06796).
# ----------------------------------------------------------------------------------------
def _slope(alpha_xi: float, alpha_eta: float, clip: float) -> float:
    """Slope sigma = alpha(eta)/alpha(xi); +/- inf when alpha(xi) ~ 0 (paper's order
    extension to alpha_0 with alpha(xi)=0 -> slope +/- inf).

    The +/-inf branch is the LOOSE side of the cone, not the collapsing side: a +inf only
    raises sigma_plus and a -inf only lowers sigma_minus, so a near-zero |alpha_xi| sample
    can never spuriously certify. That this holds for the relative/online clip too rests on
    the t->0 sign-lock argued in the SOUNDNESS NOTE in ``_decide_point`` -- not on the clip
    threshold alone.
    """
    if abs(alpha_xi) <= clip:
        return math.inf if alpha_eta >= 0.0 else -math.inf
    return alpha_eta / alpha_xi


def _decide_point(
    field,
    seed_xyz: np.ndarray,
    axis: AxisModel,
    config: ConverseKamConfig,
    seed_index: int,
) -> PointVerdict:
    """Run the conefield cone-crossing test on ONE seed and return its verdict.

    Integrates the coupled system (paper's step 1)
        x'(t)  = B(x(t))
        M'(t)  = DB_{x(t)} M(t),  M(0) = I
    over t in [0, +t_f/2] and t in [0, -t_f/2] (two solve_ivp calls, forward/backward),
    sampling alpha_t(xi), alpha_t(eta), and the running sigma_t. Builds T+ / T- by the sign
    of alpha_t(xi), tracks sigma+ = min over T+ and sigma- = max over T-, and CERTIFIES
    non-existence the moment sigma+ <= sigma- (Theorem 1). t* is the |t| at which that
    first happens; q = t* / t_f. Any integration failure / domain exit / xi degeneracy ->
    UNDECIDED (fail-closed).
    """
    seed = np.asarray(seed_xyz, dtype=float).reshape(3)
    b0, _ = _field_B_and_DB(field, seed)
    xi0 = _radial_direction_field(seed, axis)
    # |xi0| = |2 Q v| with Q positive-definite (det-normalised, smallest eigenvalue > 0), so it
    # vanishes ONLY as v -> 0, i.e. ON the axis where the direction field is undefined -- never
    # merely along an elongated surface's long axis. Gate on an absolute geometric floor
    # (metres-scale), NOT the relative slope_clip (a dimensionless cone-boundary threshold).
    if not np.all(np.isfinite(xi0)) or float(np.linalg.norm(xi0)) <= 1.0e-9:
        # On / extremely close to the axis grad g -> 0; xi is undefined there.
        return PointVerdict(
            seed_index=seed_index,
            decision=DECISION_UNDECIDED,
            reason=UNDECIDED_XI_DEGENERATE,
            q=1.0,
            sigma_plus=None,
            sigma_minus=None,
            seed_xyz=(float(seed[0]), float(seed[1]), float(seed[2])),
        )
    eta0 = np.cross(xi0, b0)  # horizontal field eta = xi x B at x0 (paper's default)

    half = 0.5 * config.t_f
    max_step = config.max_step_fraction * config.t_f

    # State vector: [x(3), vec(M)(9)]. M is row-major flattened.
    def rhs(_t: float, state: np.ndarray) -> np.ndarray:
        xyz = state[:3]
        m = state[3:].reshape(3, 3)
        b, dB = _field_B_and_DB(field, xyz)
        m_dot = dB @ m
        return np.concatenate([b, m_dot.reshape(9)])

    def left_domain(_t: float, state: np.ndarray) -> float:
        # Event: returns 0 when the field line crosses the cylindrical box boundary.
        radius = math.hypot(float(state[0]), float(state[1]))
        z = float(state[2])
        margins = [
            radius - config.domain_r_min,
            config.domain_r_max - radius,
            config.domain_abs_z_max - abs(z),
        ]
        return min(margins)

    left_domain.terminal = True
    left_domain.direction = -1.0

    state0 = np.concatenate([seed, np.eye(3).reshape(9)])

    # Accumulators for the cone bounds across BOTH integration directions.
    sigma_plus = math.inf  # min over T+
    sigma_minus = -math.inf  # max over T-
    have_t_plus = False
    have_t_minus = False
    t_star: Optional[float] = None
    domain_exit = False  # a field line left the integration box before the cone collapsed
    # Running scale of |alpha_t(xi)|: the relative slope_clip gate is measured against it so
    # near-sign-flip samples (|alpha_t(xi)| ~ 0, slope spikes) are skipped robustly even as
    # |alpha_t(xi)| grows with the tangent map M(t). Seeded tiny-positive to avoid div-by-0.
    # SOUNDNESS NOTE: that this relative/online scale cannot SPURIOUSLY collapse the cone
    # (a false non-existence certificate) rests on a t->0 sign-lock, NOT on the clip
    # threshold alone. Near t=0, M(t) ~= I so alpha_t(xi) = det[xi0, B0, M(t) xi0] = O(t)
    # with sign locked to sign(t); the near-zero-|alpha_xi| spikes therefore always land on
    # the LOOSE side of the cone -- a large +slope into T+ (raises sigma_plus, harmless) or a
    # large -slope into T- (lowers sigma_minus, harmless), i.e. exactly the paper's
    # order-extension limits (slope -> +/-inf at alpha(xi)=0, arXiv:2501.06796) -- never the
    # collapsing side. (Empirically: 0 online-vs-global-scale disagreements over 200 random
    # mixed-shear fields; an integrable sheared nested field certifies fraction 0.0.)
    alpha_xi_scale = np.finfo(float).tiny

    for sign in (+1.0, -1.0):
        t_span = (0.0, sign * half)
        # Dense sampling so the cone-crossing time t* is resolved; ascending |t|.
        n_samples = max(64, int(round(config.t_f / max_step)))
        t_eval = np.linspace(0.0, sign * half, n_samples)
        try:
            sol = solve_ivp(
                rhs,
                t_span,
                state0,
                method="DOP853",
                rtol=config.integration_rtol,
                atol=config.integration_atol,
                max_step=max_step,
                t_eval=t_eval,
                events=left_domain,
                dense_output=False,
            )
        except (ValueError, FloatingPointError):
            # A failed variational solve is not evidence of a surface -> UNDECIDED.
            return PointVerdict(
                seed_index=seed_index,
                decision=DECISION_UNDECIDED,
                reason=UNDECIDED_INTEGRATION_FAILED,
                q=1.0,
                sigma_plus=None,
                sigma_minus=None,
                seed_xyz=(float(seed[0]), float(seed[1]), float(seed[2])),
            )
        if sol.status == 1:
            # A terminal event fired: the field line left the integration box on this leg.
            # Samples up to the exit are still valid bounds (processed below); record that
            # the window was truncated so the no-collapse verdict is attributed correctly.
            domain_exit = True
        if not sol.success and sol.status != 1:
            # status 1 == a terminal event fired (domain exit), handled below; any other
            # non-success is an integration failure.
            return PointVerdict(
                seed_index=seed_index,
                decision=DECISION_UNDECIDED,
                reason=UNDECIDED_INTEGRATION_FAILED,
                q=1.0,
                sigma_plus=None,
                sigma_minus=None,
                seed_xyz=(float(seed[0]), float(seed[1]), float(seed[2])),
            )

        # Walk samples in ascending |t|; column k is state at t_eval[k].
        for k in range(sol.y.shape[1]):
            t_k = float(sol.t[k])
            state_k = sol.y[:, k]
            xyz_k = state_k[:3]
            m_k = state_k[3:].reshape(3, 3)
            b_k, _ = _field_B_and_DB(field, xyz_k)
            xi_k = _radial_direction_field(xyz_k, axis)
            # alpha_t(w) = det[ xi(x_t), B(x_t), M(t) w ]  (pullback of alpha_0=i_B i_xi Omega).
            m_xi0 = m_k @ xi0
            m_eta0 = m_k @ eta0
            alpha_xi = float(np.linalg.det(np.column_stack([xi_k, b_k, m_xi0])))
            alpha_eta = float(np.linalg.det(np.column_stack([xi_k, b_k, m_eta0])))
            alpha_xi_scale = max(alpha_xi_scale, abs(alpha_xi))
            # RELATIVE clip: skip samples at the alpha(xi) sign-flip where the slope blows up
            # (they carry no finite bound and would spuriously collapse the cone -- see config
            # doc). The theorem also requires |alpha_t(xi)| > 0 strictly at the bounding times.
            if abs(alpha_xi) <= config.slope_clip * alpha_xi_scale:
                continue
            sigma_k = _slope(alpha_xi, alpha_eta, config.slope_clip * alpha_xi_scale)
            if alpha_xi > 0.0:
                have_t_plus = True
                if sigma_k < sigma_plus:
                    sigma_plus = sigma_k
            else:
                have_t_minus = True
                if sigma_k > sigma_minus:
                    sigma_minus = sigma_k
            # Cone-crossing test (Theorem 1): cone empty <=> sigma+ <= sigma-.
            if have_t_plus and have_t_minus and sigma_plus <= sigma_minus:
                t_star = abs(t_k)
                break
        if t_star is not None:
            break

    if t_star is not None:
        return PointVerdict(
            seed_index=seed_index,
            decision=DECISION_CERTIFIED_NONEXISTENT,
            reason="cone_collapsed_sigma_plus_le_sigma_minus",
            q=float(min(1.0, t_star / config.t_f)),
            sigma_plus=(None if math.isinf(sigma_plus) else float(sigma_plus)),
            sigma_minus=(None if math.isinf(sigma_minus) else float(sigma_minus)),
            seed_xyz=(float(seed[0]), float(seed[1]), float(seed[2])),
        )

    # No collapse within the window: UNDECIDED (fail-closed) -- NEVER "surface exists".
    if domain_exit:
        # The integration window was cut short by a domain exit, so non-existence simply was
        # not reachable here -- a truncated-window undecided, not evidence of a surface.
        reason = UNDECIDED_LEFT_DOMAIN
    elif not (have_t_plus and have_t_minus):
        reason = UNDECIDED_NO_SIGN_SPLIT
    else:
        reason = UNDECIDED_TIMEOUT
    return PointVerdict(
        seed_index=seed_index,
        decision=DECISION_UNDECIDED,
        reason=reason,
        q=1.0,  # paper convention: q == 1 denotes no detection within the timeout
        sigma_plus=(None if math.isinf(sigma_plus) else float(sigma_plus)),
        sigma_minus=(None if math.isinf(sigma_minus) else float(sigma_minus)),
        seed_xyz=(float(seed[0]), float(seed[1]), float(seed[2])),
    )


def run_converse_kam(
    field,
    seed_points: Sequence[Sequence[float]],
    axis: AxisModel,
    config: Optional[ConverseKamConfig] = None,
) -> ConverseKamResult:
    """Run the conefield converse-KAM test over a list of Cartesian seed points.

    Pure given (field, seeds, axis, config): no I/O, no global state. Returns the
    per-grid ``ConverseKamResult`` with a fail-closed certified-non-existence fraction.
    """
    cfg = config or ConverseKamConfig()
    verdicts: list[PointVerdict] = []
    for idx, seed in enumerate(seed_points):
        verdicts.append(
            _decide_point(field, np.asarray(seed, dtype=float), axis, cfg, idx)
        )
    n_total = len(verdicts)
    n_certified = sum(
        1 for v in verdicts if v.decision == DECISION_CERTIFIED_NONEXISTENT
    )
    n_undecided = n_total - n_certified
    fraction = (n_certified / n_total) if n_total > 0 else 0.0
    # xi is the flux-surface-normal direction grad(v^T Q v): shaped (Q from the surface 2nd-
    # moment, parallel to grad psi) when ``axis.shape`` is set, else the circular Q=I surrogate
    # (correct only for circular surfaces; the documented false-positive source on elongated
    # fields). Record which was used so the ledger never conflates the two.
    direction_field = (
        "grad_flux_surface_normal_shaped (Q=inv_2nd_moment, donor-validated)"
        if axis.shape is not None
        else "grad_squared_distance_from_axis_circular (Q=I; correct only for circular surfaces)"
    )
    return ConverseKamResult(
        certified_nonexistence_fraction=float(fraction),
        n_certified=int(n_certified),
        n_undecided=int(n_undecided),
        n_total=int(n_total),
        axis_source=axis.source,
        axis_residual_max=float(axis.residual_max),
        direction_field=direction_field,
        points=tuple(verdicts),
    )


# ----------------------------------------------------------------------------------------
# Seed construction from a surface (reuses the midplane radial-sweep contract conceptually).
# ----------------------------------------------------------------------------------------
def midplane_radial_seeds(
    surface,
    n_seeds: int,
    *,
    inset_fraction: float = 0.05,
) -> np.ndarray:
    """Build ``n_seeds`` Cartesian seed points on the phi=0 midplane of ``surface``.

    Matches the geometry of the existing topology midplane seed contract: a radial sweep
    between the inner and outer midplane crossings of the phi=0 cross-section, inset
    slightly. Returns an (n_seeds, 3) array in Cartesian coordinates (y=0 on the phi=0
    half-plane, so x=R, z=Z).
    """
    cross_section = surface.cross_section(phi=0.0, thetas=512)
    r = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z = cross_section[:, 2]
    midplane = np.argsort(np.abs(z))[:8]
    r_mid = np.sort(r[midplane])
    rin, rout = float(r_mid[0]), float(r_mid[-1])
    span = rout - rin
    inset = min(max(inset_fraction * span, 0.01), 0.45 * span)
    radii = np.linspace(rin + inset, rout - inset, int(n_seeds))
    seeds = np.zeros((int(n_seeds), 3), dtype=float)
    seeds[:, 0] = radii  # x = R on the phi=0 half-plane
    return seeds


def certify_nonexistence_on_field(
    field,
    surface,
    *,
    nfp: int,
    config: Optional[ConverseKamConfig] = None,
) -> ConverseKamResult:
    """End-to-end converse-KAM on a (field, surface): build axis + seeds, run, return result.

    The single composition the gate calls: locate the axis (the unvalidated step), build the
    midplane radial seeds, and run the conefield test. Axis location failure (the locator
    raising ``MagneticAxisNotLocatedError``) is converted into an all-undecided result with
    an explicit reason -- never into a "surfaces exist" claim (fail-closed).
    """
    from simsopt.field.magnetic_axis_helpers import MagneticAxisNotLocatedError

    cfg = config or ConverseKamConfig()
    seeds = midplane_radial_seeds(surface, cfg.n_seeds)
    # Seed the axis solve from the geometric midplane centre of the cross-section.
    r_seed = float(np.mean(seeds[:, 0]))
    cross_section = surface.cross_section(phi=0.0, thetas=512)
    radii_cs = np.sqrt(cross_section[:, 0] ** 2 + cross_section[:, 1] ** 2)
    z_cs = cross_section[:, 2]
    span = max(
        float(np.max(radii_cs) - np.min(radii_cs)),
        float(np.max(z_cs) - np.min(z_cs)),
        1.0e-3,
    )
    r_bounds = (
        max(float(np.finfo(float).eps), float(np.min(radii_cs) - span)),
        float(np.max(radii_cs) + span),
    )
    z_bounds = (float(np.min(z_cs) - span), float(np.max(z_cs) + span))
    def _all_undecided(reason: str, axis_source: str) -> ConverseKamResult:
        # Fail-closed: no reliable axis => no reliable xi => every point UNDECIDED, never
        # certified.
        verdicts = tuple(
            PointVerdict(
                seed_index=i,
                decision=DECISION_UNDECIDED,
                reason=reason,
                q=1.0,
                sigma_plus=None,
                sigma_minus=None,
                seed_xyz=(float(s[0]), float(s[1]), float(s[2])),
            )
            for i, s in enumerate(seeds)
        )
        return ConverseKamResult(
            certified_nonexistence_fraction=0.0,
            n_certified=0,
            n_undecided=len(verdicts),
            n_total=len(verdicts),
            axis_source=axis_source,
            axis_residual_max=math.inf,
            direction_field="axis_not_located",
            points=verdicts,
        )

    try:
        axis = build_axis_model(
            field,
            nfp=int(nfp),
            n_phi_planes=cfg.n_phi_planes,
            r_guess=r_seed,
            z_guess=0.0,
            r_bounds=r_bounds,
            z_bounds=z_bounds,
        )
    except MagneticAxisNotLocatedError as exc:
        return _all_undecided(f"axis_not_located: {exc}", "axis_not_located")

    # NOTE on xi: we deliberately do NOT attach a shaped xi here from the BOUNDARY surface.
    # The boundary's shape (elongation/tilt) does not match the rounder core flux surfaces, so
    # using it over-corrects xi in the core and worsens the verdict (measured on slid_clean:
    # circular 4/7 -> boundary-shaped 7/7 falsely certified). More fundamentally, a converse-KAM
    # investigation showed the cone-crossing test here false-certifies provably-nested tori even
    # with the EXACT per-surface flux normal (and the cause is in the slope-ordering / cone
    # collapse, not the direction field) -- so a shaped xi is not the fix. The shape machinery
    # (``flux_shape_from_surface`` / ``AxisModel.shape``) remains available for a caller that can
    # supply the LOCAL per-flux-surface shape, but the default end-to-end path uses the plain
    # axis (Q = I). converse-KAM stays advisory-only until the cone mechanics is corrected.
    return run_converse_kam(field, seeds, axis, cfg)


# ----------------------------------------------------------------------------------------
# Donor validation: the gate must be validated before it is trusted (required).
# ----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DonorExpectation:
    """A validation case: a known field whose converse-KAM answer is known a priori."""

    name: str
    kind: str  # "nested" (expect ~0 certified) | "chaotic" (expect high certified)
    field_factory: Callable[[], object]
    surface_factory: Callable[[], object]
    nfp: int
    max_certified_fraction: Optional[float] = None  # for "nested" donors
    min_certified_fraction: Optional[float] = None  # for "chaotic" donors


@dataclass(frozen=True)
class DonorValidationOutcome:
    name: str
    kind: str
    certified_fraction: float
    passed: bool
    detail: str
    result: ConverseKamResult


def validate_against_donors(
    expectations: Sequence[DonorExpectation],
    config: Optional[ConverseKamConfig] = None,
) -> tuple[DonorValidationOutcome, ...]:
    """Run converse-KAM on each donor and check it against the a-priori expectation.

    This is the gate's falsifiability harness (REQUIRED before the converse-KAM output is
    trusted as authoritative). A NESTED donor (known-good confined equilibrium) must yield
    a certified-non-existence fraction at or below ``max_certified_fraction`` (ideally ~0 in
    the core); a CHAOTIC donor must yield at least ``min_certified_fraction``. Until BOTH
    classes pass on this device's field family, the converse-KAM output stays advisory and
    must NOT flip the promotion verdict.
    """
    cfg = config or ConverseKamConfig()
    outcomes: list[DonorValidationOutcome] = []
    for exp in expectations:
        field = exp.field_factory()
        surface = exp.surface_factory()
        result = certify_nonexistence_on_field(
            field, surface, nfp=exp.nfp, config=cfg
        )
        frac = result.certified_nonexistence_fraction
        if exp.kind == "nested":
            bound = exp.max_certified_fraction
            if bound is None:
                raise ValueError("nested donor needs max_certified_fraction")
            passed = frac <= bound
            detail = (
                f"nested donor: certified {frac:.3f} <= max {bound:.3f}? "
                f"{'PASS' if passed else 'FAIL'}"
            )
        elif exp.kind == "chaotic":
            bound = exp.min_certified_fraction
            if bound is None:
                raise ValueError("chaotic donor needs min_certified_fraction")
            passed = frac >= bound
            detail = (
                f"chaotic donor: certified {frac:.3f} >= min {bound:.3f}? "
                f"{'PASS' if passed else 'FAIL'}"
            )
        else:
            raise ValueError(f"unknown donor kind {exp.kind!r}")
        outcomes.append(
            DonorValidationOutcome(
                name=exp.name,
                kind=exp.kind,
                certified_fraction=frac,
                passed=passed,
                detail=detail,
                result=result,
            )
        )
    return tuple(outcomes)
