"""Regression tests for the conefield converse-KAM non-existence test.

``banana_opt/topology/converse_kam.py`` implements the MacKay-Percival conefield
Converse KAM method (arXiv:2501.06796, Theorem 3.1). A prior first-principles
review verified the cone math is FAITHFUL to the paper but found the module had
ZERO automated coverage; these tests lock the conventions the review checked and
the fail-closed contract. Synthetic fields are used so no simsopt magnetic-axis
locator is needed. Each test names one observable behaviour (DAMP, state-asserting).

Conventions under test (paper Theorem 3.1):
  * pullback covector alpha_t(w) = det[xi, B, M(t) w] = (xi x B) . (M(t) w)
  * slope sigma = alpha(eta) / alpha(xi); +/-inf at alpha(xi) -> 0 (order extension)
  * certify non-existence iff sigma_plus = min_{T+} sigma <= sigma_minus = max_{T-} sigma
  * eta = xi x B; fail-closed: every numerical failure -> UNDECIDED, never certified.

The SOUNDNESS test uses an AXISYMMETRIC field whose circular magnetic axis at
(R_a, 0) coincides with the AxisModel the radial direction field xi is built from
(a linear/Cartesian field would have a straight-line axis, geometrically
inconsistent with that circular axis); with the poloidal flow a rotation about the
axis the lines lie on nested invariant tori, so no seed may certify. The LIVENESS
test uses a strongly sheared linear field whose lines do NOT lie on tori transverse
to xi, so the cone collapses and seeds certify -- proving the certifier is not
vacuous. (An outward-spiral axisymmetric field also certifies but stretches the
variational map exponentially, making it too slow for a unit test.)
"""

import math

import numpy as np
import pytest

from dataclasses import replace

from examples.single_stage_optimization.banana_opt.topology.converse_kam import (
    AxisModel,
    ConverseKamConfig,
    DECISION_CERTIFIED_NONEXISTENT,
    DECISION_UNDECIDED,
    UNDECIDED_LEFT_DOMAIN,
    UNDECIDED_NO_SIGN_SPLIT,
    UNDECIDED_XI_DEGENERATE,
    DonorExpectation,
    _field_B_and_DB,
    _radial_direction_field,
    _slope,
    certify_nonexistence_on_field,
    flux_shape_from_surface,
    midplane_radial_seeds,
    run_converse_kam,
    validate_against_donors,
)

# Reason string a CERTIFIED verdict always carries (the only path to a certificate).
_CONE_COLLAPSE_REASON = "cone_collapsed_sigma_plus_le_sigma_minus"


class _LinearField:
    """Synthetic field B(x) = b0 + J (x - x0) with constant Jacobian J[i,j]=dB_i/dx_j.

    Used only where the magnetic axis is irrelevant (convention + fail-closed tests).
    dB_by_dX is returned in the simsopt convention ``[., i, j] = dB_j/dx_i`` (J.T per
    point); _field_B_and_DB transposes it back to dB_i/dx_j, so a round trip recovers J.
    """

    def __init__(self, b0, jac, x0=(0.0, 0.0, 0.0)):
        self._b0 = np.asarray(b0, dtype=float).reshape(3)
        self._j = np.asarray(jac, dtype=float).reshape(3, 3)
        self._x0 = np.asarray(x0, dtype=float).reshape(3)
        self._pts = np.zeros((1, 3), dtype=float)

    def set_points(self, pts):
        self._pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return self

    def B(self):
        # B_i = b0_i + sum_j J[i,j] (x - x0)_j
        return self._b0 + (self._pts - self._x0) @ self._j.T

    def dB_by_dX(self):
        n = self._pts.shape[0]
        # simsopt convention [., i, j] = dB_j/dx_i = J[j, i] = J.T[i, j].
        return np.broadcast_to(self._j.T, (n, 3, 3)).copy()


class _AxisymmetricField:
    """Axisymmetric field with a circular magnetic axis at (R_a, 0), matching AxisModel.

    Poloidal flow about the axis: (b_R, b_Z) = kp*(-Z, R-R_a) is a rotation (nested
    circular flux surfaces) plus an optional outward spiral eps*(R-R_a, Z); toroidal
    b_phi = btor advances phi. dB/dX is a central finite difference of B (converse_kam
    needs only B and dB/dX), so the analytic Jacobian need not be hand-derived. The FD
    is returned directly in the simsopt convention [., i, j] = dB_j/dx_i.
    """

    def __init__(self, r_a=1.0, kp=0.3, btor=1.0, eps=0.0, fd_step=1.0e-6):
        self.r_a = float(r_a)
        self.kp = float(kp)
        self.btor = float(btor)
        self.eps = float(eps)
        self.h = float(fd_step)
        self._pts = np.zeros((1, 3), dtype=float)

    def set_points(self, pts):
        self._pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return self

    def _b_cart(self, pts):
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        radius = np.hypot(x, y)
        radius_safe = np.where(radius > 0.0, radius, 1.0)
        cos_phi = x / radius_safe
        sin_phi = y / radius_safe
        d_r = radius - self.r_a
        b_r = -self.kp * z + self.eps * d_r
        b_z = self.kp * d_r + self.eps * z
        b_phi = self.btor
        b_x = b_r * cos_phi - b_phi * sin_phi
        b_y = b_r * sin_phi + b_phi * cos_phi
        return np.column_stack([b_x, b_y, b_z])

    def B(self):
        return self._b_cart(self._pts)

    def dB_by_dX(self):
        pts = self._pts
        out = np.zeros((pts.shape[0], 3, 3), dtype=float)
        for i in range(3):
            step = np.zeros(3, dtype=float)
            step[i] = self.h
            # [., i, j] = dB_j/dx_i (central difference).
            out[:, i, :] = (self._b_cart(pts + step) - self._b_cart(pts - step)) / (
                2.0 * self.h
            )
        return out


class _PerturbedRotationField:
    """Rigid poloidal rotation about (R_a,0) + strong toroidal carrier, optionally with a
    resonant non-axisymmetric kick that BREAKS integrability. Bounded (no escape to the
    domain wall), so it certifies FAST -- the fast chaotic donor the linear sheared/outward-
    spiral fields cannot supply (a straight axis defeats the locator; a spiral stretches the
    variational map exponentially). dB/dX is a central difference of the closed-form B,
    returned in the simsopt convention [., i, j] = dB_j/dx_i.

      eps == 0 -> nested circular flux surfaces (integrable): cone NEVER collapses.
      eps  > 0 -> resonant (m,n) kick whose amplitude grows with minor radius rho, so OUTER
                  seeds certify non-existence while inner ones stay undecided.
    """

    def __init__(self, eps, r_a=1.0, omega=1.0, btor=2.0, m=2, n=1, fd_step=1.0e-7):
        self.eps = float(eps)
        self.r_a = float(r_a)
        self.omega = float(omega)
        self.btor = float(btor)
        self.m = int(m)
        self.n = int(n)
        self.h = float(fd_step)
        self._pts = np.zeros((1, 3), dtype=float)

    def set_points(self, pts):
        self._pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return self

    def _b_cart(self, pts):
        out = np.empty_like(pts)
        for k, (x, y, z) in enumerate(pts):
            radius = math.hypot(x, y)
            phi = math.atan2(y, x)
            u = radius - self.r_a
            w = z
            b_r = -self.omega * w
            b_z = self.omega * u
            if self.eps != 0.0:
                rho = math.hypot(u, w)
                theta = math.atan2(w, u)
                kick = self.eps * rho * math.cos(self.m * theta - self.n * phi)
                b_r += kick * math.cos(theta)
                b_z += kick * math.sin(theta)
            b_phi = self.btor * self.r_a / radius
            cos_phi, sin_phi = math.cos(phi), math.sin(phi)
            out[k] = [b_r * cos_phi - b_phi * sin_phi, b_r * sin_phi + b_phi * cos_phi, b_z]
        return out

    def B(self):
        return self._b_cart(self._pts)

    def dB_by_dX(self):
        pts = self._pts
        out = np.zeros((pts.shape[0], 3, 3), dtype=float)
        for i in range(3):
            step = np.zeros(3, dtype=float)
            step[i] = self.h
            # [., i, j] = dB_j/dx_i (central difference of the closed-form B).
            out[:, i, :] = (self._b_cart(pts + step) - self._b_cart(pts - step)) / (
                2.0 * self.h
            )
        return out


class _SaddleFixedCircleField:
    """Toroidal carrier + a poloidal SADDLE whose fixed line is the circle R=Rc, Z=0.

    A seed placed exactly on that fixed circle sits at the saddle's fixed point, so by the
    saddle's in-out / up-down symmetry alpha_t(xi) keeps a single sign on BOTH integration
    legs: the T+/T- split required by Theorem 3.1 never forms -> UNDECIDED/NO_SIGN_SPLIT.
    """

    def __init__(self, s=0.6, btor=3.0, r_c=1.2, fd_step=1.0e-7):
        self.s = float(s)
        self.btor = float(btor)
        self.r_c = float(r_c)
        self.h = float(fd_step)
        self._pts = np.zeros((1, 3), dtype=float)

    def set_points(self, pts):
        self._pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return self

    def _b_cart(self, pts):
        out = np.empty_like(pts)
        for k, (x, y, z) in enumerate(pts):
            radius = math.hypot(x, y)
            phi = math.atan2(y, x)
            b_r = self.s * (radius - self.r_c)
            b_z = -self.s * z
            b_phi = self.btor / radius
            cos_phi, sin_phi = math.cos(phi), math.sin(phi)
            out[k] = [b_r * cos_phi - b_phi * sin_phi, b_r * sin_phi + b_phi * cos_phi, b_z]
        return out

    def B(self):
        return self._b_cart(self._pts)

    def dB_by_dX(self):
        pts = self._pts
        out = np.zeros((pts.shape[0], 3, 3), dtype=float)
        for i in range(3):
            step = np.zeros(3, dtype=float)
            step[i] = self.h
            out[:, i, :] = (self._b_cart(pts + step) - self._b_cart(pts - step)) / (
                2.0 * self.h
            )
        return out


class _RingCrossSectionSurface:
    """Minimal surface stub for ``midplane_radial_seeds`` / ``validate_against_donors``.

    ``cross_section(phi, thetas)`` returns an (N,3) Cartesian poloidal cross-section: a
    circle of minor radius ``a`` centred on (R_a, 0) in the phi half-plane. That is the only
    surface attribute the converse-KAM seed/axis machinery touches.
    """

    def __init__(self, a=0.45, r_a=1.0, nfp=1):
        self.a = float(a)
        self.r_a = float(r_a)
        self.nfp = int(nfp)

    def cross_section(self, phi=0.0, thetas=512):
        theta = np.linspace(0.0, 2.0 * math.pi, int(thetas), endpoint=False)
        radius = self.r_a + self.a * np.cos(theta)
        z = self.a * np.sin(theta)
        return np.column_stack(
            [radius * math.cos(phi), radius * math.sin(phi), z]
        )


class _EllipticCrossSectionSurface:
    """Surface stub whose poloidal cross-section is an ELONGATED ellipse about (R_a, 0).

    ``cross_section`` is the ellipse R = R_a + a*cos(theta), Z = b*sin(theta) (semi-axes a in
    R, b in Z; elongation b/a). This is the geometry that breaks the circular xi: its true
    surface normal at angle theta is the direction (2(R-R_a)/a^2, 2 Z/b^2), which the circular
    xi=grad|v|^2 misses by up to ~atan((b/a - a/b)/2) while the shaped xi=grad(v^T M^-1 v)
    matches exactly. Used to lock the root-cause fix (``flux_shape_from_surface``).
    """

    def __init__(self, a=0.04, b=0.11, r_a=1.0, nfp=1):
        self.a = float(a)
        self.b = float(b)
        self.r_a = float(r_a)
        self.nfp = int(nfp)

    def cross_section(self, phi=0.0, thetas=256):
        theta = np.linspace(0.0, 2.0 * math.pi, int(thetas), endpoint=False)
        radius = self.r_a + self.a * np.cos(theta)
        z = self.b * np.sin(theta)
        return np.column_stack([radius * math.cos(phi), radius * math.sin(phi), z])


def _ellipse_true_normal(theta, a, b):
    """Cartesian (phi=0) outward normal of the ellipse (R-R_a)^2/a^2 + Z^2/b^2 = const at the
    boundary point with parameter ``theta``: grad of that quadratic = (2 cos(theta)/a, 2 sin
    (theta)/b) up to scale (e_R=(1,0,0), e_Z=(0,0,1) on the phi=0 half-plane)."""
    return np.array([2.0 * math.cos(theta) / a, 0.0, 2.0 * math.sin(theta) / b])


def _angle_between(u, v):
    """Unsigned angle (degrees) between two 3-vectors, mod direction (anti-parallel == 0)."""
    cu = u / np.linalg.norm(u)
    cv = v / np.linalg.norm(v)
    return math.degrees(math.acos(abs(float(np.clip(np.dot(cu, cv), -1.0, 1.0)))))


def _constant_axis(r_a=1.0, z_a=0.0, n_phi=8):
    """A circular magnetic axis at fixed (R_a, Z_a), constant in phi."""
    phis = np.array([(i / n_phi) * 2.0 * math.pi for i in range(n_phi)], dtype=float)
    return AxisModel(
        phis=phis,
        r=np.full(n_phi, float(r_a)),
        z=np.full(n_phi, float(z_a)),
        nfp=1,
        residual_max=0.0,
        source="test_constant_axis",
    )


# --- Conventions (paper Theorem 3.1) -------------------------------------------------


def test_pullback_covector_equals_cross_product_identity():
    # alpha_t(w) = det[xi, B, w] must equal (xi x B) . w -- the paper's identity
    # alpha_0 = (xi x B)^flat that the determinant form in _decide_point relies on.
    rng = np.random.default_rng(20260620)
    for _ in range(25):
        xi = rng.standard_normal(3)
        b = rng.standard_normal(3)
        w = rng.standard_normal(3)
        det = float(np.linalg.det(np.column_stack([xi, b, w])))
        assert det == pytest.approx(float(np.dot(np.cross(xi, b), w)), abs=1e-12)


def test_slope_is_alpha_eta_over_alpha_xi():
    # sigma = alpha(eta)/alpha(xi) -- numerator eta, denominator xi, no sign/reciprocal.
    assert _slope(2.0, 1.0, 1.0e-9) == pytest.approx(0.5)
    assert _slope(-4.0, 1.0, 1.0e-9) == pytest.approx(-0.25)


def test_slope_extends_to_signed_infinity_at_alpha_xi_zero():
    # The paper extends the order to alpha(xi)=0 with slope -> +/-inf by sign of alpha(eta).
    plus = _slope(1.0e-300, 1.0, 1.0e-9)
    minus = _slope(1.0e-300, -1.0, 1.0e-9)
    assert math.isinf(plus) and plus > 0.0
    assert math.isinf(minus) and minus < 0.0


def test_field_B_and_DB_returns_dBi_dxj_after_transpose():
    # Locks the single most error-prone line: the dB_by_dX transpose. With an
    # ASYMMETRIC Jacobian, a wrong/absent transpose would be detected.
    jac = np.array(
        [[0.1, 0.2, 0.3], [0.0, -0.4, 0.5], [0.6, 0.0, -0.7]], dtype=float
    )
    field = _LinearField(b0=[0.0, 1.0, 0.0], jac=jac, x0=[1.0, 0.0, 0.0])
    _, dB = _field_B_and_DB(field, np.array([1.2, 0.1, 0.05]))
    # _field_B_and_DB returns DB[i,j] = dB_i/dx_j = J.
    np.testing.assert_allclose(dB, jac, atol=1e-12)
    # B at x0 is b0; the toroidal component is recovered.
    b0, _ = _field_B_and_DB(field, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(b0, [0.0, 1.0, 0.0], atol=1e-12)


def test_radial_direction_field_points_outward_and_vanishes_on_axis():
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    # Outboard of the axis (R=1.3, phi=0): xi points in +R == +x on the phi=0 plane.
    xi_out = _radial_direction_field(np.array([1.3, 0.0, 0.0]), axis)
    assert xi_out[0] > 0.0
    assert xi_out[1] == pytest.approx(0.0)
    assert xi_out[2] == pytest.approx(0.0)
    # Above the axis (z=0.2): xi points in +z.
    xi_up = _radial_direction_field(np.array([1.0, 0.0, 0.2]), axis)
    assert xi_up[2] > 0.0
    # On the axis grad g vanishes (xi undefined; the caller treats this as degenerate).
    xi_axis = _radial_direction_field(np.array([1.0, 0.0, 0.0]), axis)
    assert float(np.linalg.norm(xi_axis)) == pytest.approx(0.0)


# --- Fail-closed contract ------------------------------------------------------------


def test_seed_on_axis_is_undecided_xi_degenerate():
    # On the axis xi -> 0; the seed must be UNDECIDED (xi degenerate), never certified.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _LinearField(b0=[0.0, 1.0, 0.0], jac=np.zeros((3, 3)), x0=[1.0, 0.0, 0.0])
    result = run_converse_kam(
        field,
        [[1.0, 0.0, 0.0]],
        axis,
        ConverseKamConfig(t_f=10.0, timeout_t_f=10.0),
    )
    verdict = result.points[0]
    assert verdict.decision == DECISION_UNDECIDED
    assert verdict.reason == UNDECIDED_XI_DEGENERATE
    assert result.certified_nonexistence_fraction == 0.0


def test_field_line_leaving_domain_is_undecided_never_certified():
    # A line that exits the cylindrical box before the cone collapses is UNDECIDED
    # (truncated window), never certified -- the fail-closed truncation path.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _LinearField(b0=[0.0, 1.0, 0.0], jac=np.zeros((3, 3)), x0=[1.0, 0.0, 0.0])
    cfg = ConverseKamConfig(t_f=1000.0, timeout_t_f=1000.0, domain_r_max=1.05)
    result = run_converse_kam(field, [[1.02, 0.0, 0.0]], axis, cfg)
    verdict = result.points[0]
    assert verdict.decision == DECISION_UNDECIDED
    assert verdict.reason == UNDECIDED_LEFT_DOMAIN
    assert result.certified_nonexistence_fraction == 0.0


def test_fraction_counts_undecided_in_denominator_as_lower_bound():
    # certified_nonexistence_fraction = n_certified / n_total with undecided in the
    # denominator -- a lower bound that can never be inflated by undecided seeds.
    axis = _constant_axis()
    field = _LinearField(b0=[0.0, 1.0, 0.0], jac=np.zeros((3, 3)), x0=[1.0, 0.0, 0.0])
    result = run_converse_kam(
        field,
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # both on-axis -> both undecided
        axis,
        ConverseKamConfig(t_f=10.0, timeout_t_f=10.0),
    )
    assert result.n_total == 2
    assert result.n_certified == 0
    assert result.n_undecided == 2
    assert result.certified_nonexistence_fraction == 0.0


# --- Behavioural: soundness (no false cert) and liveness (can certify) ---------------

# Loosened integrator tolerance for the axisymmetric (finite-difference Jacobian)
# behavioural test: the cone-collapse VERDICT is robust to it on this clear-cut field,
# while the default rtol=1e-9 forces many tiny steps through the FD Jacobian.
_BEHAVIOURAL_TOL = dict(
    integration_rtol=1.0e-6, integration_atol=1.0e-8, max_step_fraction=0.05
)


def test_axisymmetric_nested_field_yields_no_false_certificate():
    # SOUNDNESS (the dangerous direction): nested circular flux surfaces about the
    # circular axis -> the field lines lie on invariant tori transverse to
    # xi=grad(dist-from-axis), so NO seed may be certified non-existent (fraction 0).
    # Guards against slope_clip / online-scale artefacts spuriously collapsing the cone.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _AxisymmetricField(r_a=1.0, kp=0.3, btor=1.0, eps=0.0)
    seeds = [[1.0 + 0.05 * k, 0.0, 0.0] for k in range(1, 4)]
    cfg = ConverseKamConfig(t_f=20.0, timeout_t_f=20.0, **_BEHAVIOURAL_TOL)
    result = run_converse_kam(field, seeds, axis, cfg)
    assert result.n_certified == 0
    assert result.certified_nonexistence_fraction == 0.0
    assert all(p.decision == DECISION_UNDECIDED for p in result.points)


def _sheared_no_transverse_torus_field():
    # A uniform poloidal rotation (rate omega) about the straight axis {x=1, z=0},
    # which is offset from the circular xi-axis (R=1): the lines wind around the
    # straight axis and do NOT lie on tori transverse to the radial xi. Analytic
    # (constant) Jacobian, so it is fast (no finite difference).
    omega = 0.15
    jac = np.array(
        [[0.0, 0.0, -omega], [0.0, 0.0, 0.0], [omega, 0.0, 0.0]], dtype=float
    )
    return _LinearField(b0=[0.0, 1.0, 0.0], jac=jac, x0=[1.0, 0.0, 0.0])


def test_sheared_field_with_no_transverse_torus_certifies_nonexistence():
    # LIVENESS: a field whose lines do not lie on tori transverse to xi drives the
    # admissible-slope cone to collapse (sigma_plus <= sigma_minus) -> seeds are
    # CERTIFIED non-existent. Proves the certifier is not vacuous (the soundness test
    # above is not trivially always-undecided) and that a certified verdict is well-formed.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _sheared_no_transverse_torus_field()
    seeds = [[1.0 + 0.05 * k, 0.0, 0.0] for k in range(1, 4)]
    result = run_converse_kam(
        field, seeds, axis, ConverseKamConfig(t_f=40.0, timeout_t_f=40.0)
    )
    assert result.n_certified >= 1
    assert result.certified_nonexistence_fraction > 0.0
    certified = [
        p for p in result.points if p.decision == DECISION_CERTIFIED_NONEXISTENT
    ]
    # A certified verdict reports a finite cone-collapse time ratio q in [0, 1].
    assert all(0.0 <= p.q <= 1.0 for p in certified)


def test_run_converse_kam_is_deterministic():
    # The certificate is a deterministic function of (field, seeds, axis, config):
    # re-running reproduces the verdicts and fraction exactly (no RNG, no global state)
    # -- a property the gate ledger relies on.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _sheared_no_transverse_torus_field()
    seeds = [[1.0 + 0.05 * k, 0.0, 0.0] for k in range(1, 4)]
    cfg = ConverseKamConfig(t_f=40.0, timeout_t_f=40.0)
    first = run_converse_kam(field, seeds, axis, cfg)
    second = run_converse_kam(field, seeds, axis, cfg)
    assert first.certified_nonexistence_fraction == second.certified_nonexistence_fraction
    assert [p.decision for p in first.points] == [p.decision for p in second.points]


# --- Bounded resonant field: integrable vs. surface-breaking (radius-selective) ------

# Short window + fine step + bounded box: fast, faithful, and lets the resonant kick
# break outer surfaces before any line reaches the domain wall.
_RESONANT_CFG = dict(
    t_f=30.0,
    timeout_t_f=30.0,
    max_step_fraction=0.02,
    integration_rtol=1.0e-8,
    integration_atol=1.0e-10,
    domain_r_min=0.1,
    domain_r_max=4.0,
    domain_abs_z_max=4.0,
)


def test_bounded_integrable_field_never_collapses_cone():
    # The bounded integrable (eps=0) rotation has nested circular flux surfaces, so the
    # cone never empties: every seed UNDECIDED, certified fraction exactly 0.0. A second
    # soundness witness on a BOUNDED field (the axisymmetric soundness test uses a
    # different flow), so a clip/online-scale artefact cannot hide behind one field shape.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _PerturbedRotationField(eps=0.0)
    seeds = [[1.0 + dr, 0.0, 0.0] for dr in (0.1, 0.2, 0.3, 0.4)]

    result = run_converse_kam(
        field, seeds, axis, ConverseKamConfig(n_seeds=4, **_RESONANT_CFG)
    )

    assert result.certified_nonexistence_fraction == 0.0
    assert all(p.decision == DECISION_UNDECIDED for p in result.points)


def test_resonant_perturbation_certifies_outer_but_not_inner_seeds():
    # The resonant (m=2,n=1) kick grows with minor radius, so OUTER surfaces break: those
    # seeds CERTIFY non-existence while the innermost seed stays UNDECIDED. The certificate
    # is radius-selective (0 < fraction < 1), not an all-or-nothing collapse.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _PerturbedRotationField(eps=0.5)
    seeds = [[1.0 + dr, 0.0, 0.0] for dr in (0.1, 0.2, 0.3, 0.4)]

    result = run_converse_kam(
        field, seeds, axis, ConverseKamConfig(n_seeds=4, **_RESONANT_CFG)
    )

    assert result.n_certified > 0
    assert 0.0 < result.certified_nonexistence_fraction < 1.0
    inner = next(p for p in result.points if p.seed_index == 0)
    assert inner.decision == DECISION_UNDECIDED
    certified = [p for p in result.points if p.decision == DECISION_CERTIFIED_NONEXISTENT]
    assert all(p.reason == _CONE_COLLAPSE_REASON for p in certified)


# --- Additional fail-closed branches + certify-is-the-only-path ----------------------


def test_single_sign_sweep_is_undecided_no_sign_split():
    # On the saddle's fixed circle alpha_t(xi) keeps one sign on both legs, so the T+/T-
    # split Theorem 3.1 needs never forms. With no two-sided bound the cone cannot be
    # declared empty: UNDECIDED/NO_SIGN_SPLIT, never certified.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _SaddleFixedCircleField(r_c=1.2)

    result = run_converse_kam(
        field,
        [[1.2, 0.0, 0.0]],  # exactly on the saddle's fixed circle
        axis,
        ConverseKamConfig(t_f=15.0, timeout_t_f=15.0, **{
            k: v for k, v in _RESONANT_CFG.items() if k not in ("t_f", "timeout_t_f")
        }),
    )

    verdict = result.points[0]
    assert verdict.decision == DECISION_UNDECIDED
    assert verdict.reason == UNDECIDED_NO_SIGN_SPLIT
    assert result.n_certified == 0


def test_certified_decision_is_reachable_only_via_cone_collapse():
    # The ONLY route to DECISION_CERTIFIED_NONEXISTENT is the cone-collapse reason; every
    # other reason (timeout / left-domain / xi-degenerate / no-sign-split) is UNDECIDED.
    # Drive a mixed grid (broken outer + degenerate on-axis seed) and assert the coupling
    # of decision<->reason both ways, so an undecided seed can never wear the certificate.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _PerturbedRotationField(eps=0.5)
    seeds = [[1.0, 0.0, 0.0]] + [[1.0 + dr, 0.0, 0.0] for dr in (0.2, 0.3, 0.4)]

    result = run_converse_kam(
        field, seeds, axis, ConverseKamConfig(n_seeds=4, **_RESONANT_CFG)
    )

    for p in result.points:
        if p.decision == DECISION_CERTIFIED_NONEXISTENT:
            assert p.reason == _CONE_COLLAPSE_REASON
        else:
            assert p.decision == DECISION_UNDECIDED
            assert p.reason != _CONE_COLLAPSE_REASON
    # The on-axis seed is degenerate (undecided), confirming a non-collapse reason is present.
    assert any(p.reason == UNDECIDED_XI_DEGENERATE for p in result.points)
    assert result.n_certified + result.n_undecided == result.n_total


# --- Scale invariance of the certificate (paper: "only the direction of xi matters") --


def test_positive_rescaling_of_xi_leaves_slope_invariant():
    # The paper's "only the DIRECTION of xi matters": under xi0 -> c*xi0 (c>0) both
    # alpha_xi = det[xi,B,M(c xi0)] and alpha_eta = det[xi,B,M(c eta0)] (eta0=xi0 x B0 also
    # scales by c) scale by c, so sigma = alpha_eta/alpha_xi is invariant. This is the
    # genuine scale invariance of the certificate.
    #
    # NOTE: a global eta -> -eta flip is deliberately NOT asserted invariant. The cone test
    # is the ONE-SIDED predicate min_{T+} sigma <= max_{T-} sigma keyed to the fixed
    # eta = xi x B; negating eta sends sigma -> -sigma and turns it into the complementary
    # predicate max_{T+} sigma >= min_{T-} sigma, which certifies a different seed set
    # (verified numerically). The estimator is sign-convention-bound by construction, so an
    # eta-flip-invariance assertion would be false.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    field = _PerturbedRotationField(eps=0.5)
    seed = np.array([1.3, 0.0, 0.0])
    b0, _ = _field_B_and_DB(field, seed)
    xi0 = _radial_direction_field(seed, axis)
    tangent = np.array([[1.0, 0.2, -0.1], [0.0, 1.3, 0.4], [0.05, -0.2, 0.9]])

    slopes = []
    for c in (1.0, 2.5, 0.3):
        xi0_c = c * xi0
        eta0_c = np.cross(xi0_c, b0)  # == c * (xi0 x b0)
        alpha_xi = float(np.linalg.det(np.column_stack([xi0, b0, tangent @ xi0_c])))
        alpha_eta = float(np.linalg.det(np.column_stack([xi0, b0, tangent @ eta0_c])))
        slopes.append(alpha_eta / alpha_xi)

    assert slopes[1] == pytest.approx(slopes[0], rel=1.0e-12)
    assert slopes[2] == pytest.approx(slopes[0], rel=1.0e-12)


# --- Seed construction + donor validation (validate_against_donors' first caller) ----


def test_midplane_radial_seeds_sweeps_the_phi0_half_plane():
    # midplane_radial_seeds returns an (n,3) Cartesian radial sweep on the phi=0 half-plane
    # (y=0, z=0 so x=R), inset within the surface's inner/outer midplane radii, ascending.
    surface = _RingCrossSectionSurface(a=0.45, r_a=1.0)

    seeds = midplane_radial_seeds(surface, 6)

    assert seeds.shape == (6, 3)
    assert np.allclose(seeds[:, 1], 0.0)
    assert np.allclose(seeds[:, 2], 0.0)
    assert seeds[0, 0] > 1.0 - 0.45
    assert seeds[-1, 0] < 1.0 + 0.45
    assert np.all(np.diff(seeds[:, 0]) > 0.0)


def test_certify_nonexistence_on_field_locates_axis_on_synthetic_field():
    # The single composition the gate calls end-to-end: locate the axis on the synthetic
    # field (the simsopt fixed-point locator, exercised here on a numpy-only field), build
    # midplane seeds, run the cone test. On the integrable donor it locates the axis (tiny
    # residual) and reports fraction 0.0 with the real axis-source string.
    result = certify_nonexistence_on_field(
        _PerturbedRotationField(eps=0.0),
        _RingCrossSectionSurface(a=0.45, r_a=1.0),
        nfp=1,
        config=ConverseKamConfig(n_seeds=6, **_RESONANT_CFG),
    )

    assert result.axis_source == "magnetic_axis_curve_multistart"
    assert result.axis_residual_max < 1.0e-6
    assert result.certified_nonexistence_fraction == 0.0
    assert result.n_total == 6


def test_validate_against_donors_passes_nested_and_chaotic_expectations():
    # The harness's first end-to-end caller. validate_against_donors runs the FULL path
    # (build_axis_model via the simsopt locator -> midplane seeds -> cone test) on a nested
    # donor (expect certified <= max) and a chaotic donor (expect certified >= min). Both
    # must PASS their a-priori expectation, exercising both the "nested" and "chaotic"
    # branches and the certified-fraction comparison.
    cfg = ConverseKamConfig(n_seeds=6, **_RESONANT_CFG)
    expectations = (
        DonorExpectation(
            name="nested_integrable",
            kind="nested",
            field_factory=lambda: _PerturbedRotationField(eps=0.0),
            surface_factory=lambda: _RingCrossSectionSurface(a=0.45, r_a=1.0),
            nfp=1,
            max_certified_fraction=0.0,
        ),
        DonorExpectation(
            name="chaotic_resonant",
            kind="chaotic",
            field_factory=lambda: _PerturbedRotationField(eps=0.5),
            surface_factory=lambda: _RingCrossSectionSurface(a=0.45, r_a=1.0),
            nfp=1,
            min_certified_fraction=0.1,
        ),
    )

    outcomes = validate_against_donors(expectations, cfg)

    by_name = {o.name: o for o in outcomes}
    assert by_name["nested_integrable"].passed is True
    assert by_name["nested_integrable"].certified_fraction == 0.0
    assert by_name["chaotic_resonant"].passed is True
    assert by_name["chaotic_resonant"].certified_fraction >= 0.1


def test_validate_against_donors_fails_a_chaotic_donor_that_does_not_certify():
    # Falsifiability of the harness: handing the NESTED (integrable) field to a CHAOTIC
    # expectation must FAIL the check (certified 0.0 < min), proving the harness does not
    # pass everything -- it is a real gate, and a fail-closed undecided field cannot be
    # rubber-stamped as chaotic.
    cfg = ConverseKamConfig(n_seeds=6, **_RESONANT_CFG)
    expectation = DonorExpectation(
        name="nested_field_wrong_chaotic_label",
        kind="chaotic",
        field_factory=lambda: _PerturbedRotationField(eps=0.0),
        surface_factory=lambda: _RingCrossSectionSurface(a=0.45, r_a=1.0),
        nfp=1,
        min_certified_fraction=0.1,
    )

    (outcome,) = validate_against_donors((expectation,), cfg)

    assert outcome.passed is False
    assert outcome.certified_fraction == 0.0


# --- Shaped (flux-normal) xi vs the circular squared-distance surrogate ----------------------
#
# These tests lock the CORRECTNESS of the shaped direction field xi = grad(v^T Q v),
# Q = (flux 2nd-moment)^-1: on an ELONGATED surface it is parallel to the true flux-surface
# normal, where the circular surrogate xi = grad|v|^2 (Q = I) is off by tens of degrees.
# IMPORTANT SCOPE: an evidence-gated investigation found this is NOT the cause of the
# converse-KAM false positives on the slid_clean candidate -- the cone-crossing test there
# certifies provably-nested tori even with the EXACT per-surface flux normal, because of a
# t->-t antisymmetry in the cone mechanics (see the module docstring). So the shaped xi is the
# geometrically correct direction field (a real, tested capability) but is NOT a fix for the
# false positives, and the end-to-end path does not auto-apply it. These tests therefore lock
# the xi/shape MATH (deterministic, exact), not an end-to-end certified-fraction flip.

_REGRESSION_ELLIPSE = dict(a=0.04, b=0.11)  # elongation b/a = 2.75, matching the real candidate


def test_circular_xi_misaligns_from_true_normal_on_elongated_surface():
    # On a 2.75x-elongated flux surface the circular xi = grad|v|^2 is far from the true
    # surface normal (the geometric deficiency the shaped Q corrects). (Anti-)parallel counts
    # as aligned; the worst case is at the ~45 deg boundary points.
    a, b = _REGRESSION_ELLIPSE["a"], _REGRESSION_ELLIPSE["b"]
    axis = _constant_axis(r_a=1.0, z_a=0.0)  # shape is None -> circular xi
    worst = 0.0
    for theta in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        r = 1.0 + a * math.cos(theta)
        z = b * math.sin(theta)
        xi = _radial_direction_field(np.array([r, 0.0, z]), axis)
        worst = max(worst, _angle_between(xi, _ellipse_true_normal(theta, a, b)))
    # The circular surrogate is wrong by tens of degrees -- this is the defect.
    assert worst > 20.0, f"expected large circular-xi misalignment, got {worst:.1f} deg"


def test_flux_shape_from_surface_makes_xi_parallel_to_true_normal():
    # THE FIX, isolated: with Q = flux_shape_from_surface(elongated surface) the SAME seeds
    # give xi parallel to the true flux-surface normal everywhere (worst error < 1 deg),
    # whereas the circular xi (same axis, shape=None) is off by > 20 deg. This is the exact
    # before/after that the cone test depends on; if a future change regresses xi back toward
    # circular, the shaped worst-angle assertion fails.
    a, b = _REGRESSION_ELLIPSE["a"], _REGRESSION_ELLIPSE["b"]
    surface = _EllipticCrossSectionSurface(a=a, b=b, r_a=1.0)
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    shape = flux_shape_from_surface(surface, axis)
    # Q is det-normalised (pure shape) and weights the SHORT semi-axis (R, a<b) more, so xi
    # points predominantly across the narrow dimension -- the true-normal behaviour.
    assert shape.shape == (axis.phis.shape[0], 2, 2)
    np.testing.assert_allclose(np.linalg.det(shape[0]), 1.0, atol=1e-9)
    axis_shaped = replace(axis, shape=shape)

    worst_circular = 0.0
    worst_shaped = 0.0
    for theta in np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False):
        r = 1.0 + a * math.cos(theta)
        z = b * math.sin(theta)
        true_n = _ellipse_true_normal(theta, a, b)
        worst_circular = max(
            worst_circular, _angle_between(_radial_direction_field(np.array([r, 0.0, z]), axis), true_n)
        )
        worst_shaped = max(
            worst_shaped, _angle_between(_radial_direction_field(np.array([r, 0.0, z]), axis_shaped), true_n)
        )
    assert worst_circular > 20.0  # the bug is present without the shape
    assert worst_shaped < 1.0, f"shaped xi should match the true normal, worst {worst_shaped:.3f} deg"


def test_identity_shape_reproduces_circular_xi_exactly():
    # Backward-compatibility: an AxisModel with shape=None (every legacy/synthetic caller)
    # gives EXACTLY the old circular xi dg/dR=2(R-R_a), dg/dZ=2(Z-Z_a). Guards the whole
    # existing test corpus and the no-surface build_axis_model path from silent drift.
    axis = _constant_axis(r_a=1.0, z_a=0.0)
    assert axis.shape is None
    np.testing.assert_array_equal(axis.shape_at(0.37), np.eye(2))
    # Outboard point (R=1.3, Z=0.2, phi=0): legacy formula gives (0.6, 0, 0.4).
    xi = _radial_direction_field(np.array([1.3, 0.0, 0.2]), axis)
    np.testing.assert_allclose(xi, [0.6, 0.0, 0.4], atol=1e-12)


def test_flux_shape_from_surface_fails_closed_on_non_encircling_cross_section():
    # Fail-closed: a cross-section that does NOT wrap the axis cannot define a flux-surface
    # shape, so flux_shape_from_surface must RAISE rather than fit a bogus Q (which would
    # silently re-enable a wrong xi). Build a surface whose cross-section is a small arc far
    # from the axis ray about (R_a, 0).
    class _ArcSurface:
        nfp = 1

        def cross_section(self, phi=0.0, thetas=256):
            # a short arc near R=1.5, Z~0 (does not encircle the axis at R_a=1.0)
            t = np.linspace(-0.2, 0.2, int(thetas))
            radius = 1.5 + 0.01 * np.cos(t)
            z = 0.01 * np.sin(t)
            return np.column_stack([radius * math.cos(phi), radius * math.sin(phi), z])

    axis = _constant_axis(r_a=1.0, z_a=0.0)
    # The first (every) plane is a non-encircling arc -> fail closed immediately.
    with pytest.raises(ValueError, match="does not wind once about the axis"):
        flux_shape_from_surface(_ArcSurface(), axis)


def test_flux_shape_fails_closed_when_any_single_plane_does_not_wind():
    # A genuine nested flux surface winds once about the axis at EVERY toroidal plane. If even
    # one plane does not wind (a non-nested surface or a degenerate slice), flux_shape_from_
    # surface must RAISE -- it must not silently substitute another plane's shape (the earlier
    # "nearest-valid-plane fill" masked a units bug; with the correct phi units every real plane
    # winds, so a non-winding plane is a true failure). Here plane 2 is a tiny arc off the axis,
    # the rest are good ellipses; the call must fail closed naming that plane.
    n_phi = 6
    phis = np.array([(i / n_phi) * 2.0 * math.pi for i in range(n_phi)], dtype=float)
    axis = AxisModel(
        phis=phis, r=np.full(n_phi, 1.0), z=np.zeros(n_phi), nfp=1,
        residual_max=0.0, source="test",
    )
    a, b, bad = 0.04, 0.11, 2

    class _OneNonWindingPlaneSurface:
        nfp = 1

        def cross_section(self, phi=0.0, thetas=256):
            theta = np.linspace(0.0, 2.0 * math.pi, int(thetas), endpoint=False)
            # cross_section takes phi in TURNS; recover the plane index from the turn value.
            idx = int(round((float(phi) % 1.0) * n_phi)) % n_phi
            if idx == bad:
                radius = 1.0 + 1.0e-4 * np.cos(theta)
                z = 1.0e-4 * np.sin(theta) + 0.5  # shifted off the axis -> winding 0
            else:
                radius = 1.0 + a * np.cos(theta)
                z = b * np.sin(theta)
            return np.column_stack([radius * np.cos(phi), radius * np.sin(phi), z])

    with pytest.raises(ValueError, match="does not wind once about the axis"):
        flux_shape_from_surface(_OneNonWindingPlaneSurface(), axis)


def test_flux_shape_slices_the_correct_toroidal_plane_simsopt_phi_units():
    # REGRESSION for the cross_section UNITS BUG. simsopt's ``Surface.cross_section`` takes the
    # toroidal angle as a FRACTION of 2*pi (turns), NOT radians; ``axis.phis`` are radians.
    # flux_shape_from_surface must convert (phi/2*pi), else it slices the WRONG toroidal plane
    # (e.g. 3.927 rad -> 0.927 turn = 5.83 rad) and builds Q from a mismatched plane. The
    # earlier synthetic-stub tests miss this because their cross_section ignores the units
    # convention; this uses a REAL simsopt surface whose elongation VARIES strongly with the
    # cylindrical angle, so the correct-plane and wrong-plane shapes differ measurably.
    # It FAILS on the radians version and PASSES on phi/(2*pi).
    from simsopt.geo import SurfaceRZFourier

    n_phi = 8
    surface = SurfaceRZFourier(
        nfp=1, stellsym=True, mpol=2, ntor=2,
        quadpoints_phi=np.linspace(0.0, 1.0, 128, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 128, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.15)
    surface.set_zs(1, 0, 0.15)
    # Strong n=1 modulation so the poloidal elongation swings widely between toroidal planes.
    surface.set_zs(1, 1, 0.10)
    surface.set_rc(1, 1, -0.06)
    phis = np.array([(i / n_phi) * 2.0 * math.pi for i in range(n_phi)], dtype=float)
    axis = AxisModel(
        phis=phis, r=np.full(n_phi, 1.0), z=np.zeros(n_phi), nfp=1,
        residual_max=0.0, source="test",
    )

    def elong_of_cross_section(phi_turns):
        cs = surface.cross_section(phi=phi_turns, thetas=300)
        d_r = np.sqrt(cs[:, 0] ** 2 + cs[:, 1] ** 2) - 1.0
        d_z = cs[:, 2]
        m = np.array([[np.mean(d_r * d_r), np.mean(d_r * d_z)],
                      [np.mean(d_r * d_z), np.mean(d_z * d_z)]])
        w = np.linalg.eigvalsh(m)
        return math.sqrt(max(w) / min(w))

    shape = flux_shape_from_surface(surface, axis)
    # A det-normalised Q = M^-1 has eigenvalues lambda, 1/lambda, so the cross-section
    # elongation sqrt(max/min eig of M) equals sqrt(max/min eig of Q).
    for i, phi in enumerate(phis):
        w = np.linalg.eigvalsh(shape[i])
        q_elong = math.sqrt(max(w) / min(w))
        correct = elong_of_cross_section(float(phi) / (2.0 * math.pi))  # turns: the right plane
        wrong = elong_of_cross_section(float(phi))  # radians-as-turns: the bug's wrong plane
        # Q must match the CORRECT plane's shape (units fix), not the wrong plane's.
        assert abs(q_elong - correct) < 1.0e-3, (
            f"plane {i}: Q elong {q_elong:.3f} != correct-plane elong {correct:.3f}"
        )
        if abs(correct - wrong) > 0.05:  # planes where the bug is observable
            assert abs(q_elong - wrong) > 0.04, (
                f"plane {i}: Q elong {q_elong:.3f} matches the WRONG (radians) plane "
                f"{wrong:.3f} -- the cross_section units bug is present"
            )
